"""Whole-body balance for any robot that stands on two (or more) feet: task-space inverse dynamics on the real model.

The two-legged gait code (humanoid.py) needs a robot of a known shape: five or six hinges a leg, position servos, a
trunk it can read an IMU from. Most open humanoids are not that shape: linkages, passive joints, torque motors, weak
servos, a waist, a head, hands with twenty joints. This controller needs none of it. Every control tick it asks the
model what it would take to hold the robot up:

1. Contacts are read from the simulation itself: the points where a body of this robot touches something fixed to the
   world (the floor, a ramp, a step). The feet may stand on anything, at any height.
2. Three wishes are turned into accelerations: the centre of mass stays over the middle of those contact points at
   its standing height, the trunk stays level, every joint stays near its home angle. The first two weigh more than
   the last, so the robot may bend its knees or swing an arm to keep its balance. A touching foot does not move.
3. The accelerations that keep the feet where they are and come closest to the wishes are found by least squares in
   the null space of the contact Jacobian (M, the Jacobians and the bias force come from MuJoCo for the very
   state the robot is in).
4. The contact forces that carry them are found with non-negative least squares over a friction pyramid at every
   contact point: pushing off the floor, never pulling on it, never slipping. A centre of mass that wants to be
   outside the feet gets less than it asks for, and the robot tips as a real one would.
5. The joint torques follow from the dynamics, and each is turned into the control that makes this very actuator
   deliver it: a position servo gets its target moved by torque over stiffness, a torque motor gets torque over gain.

The centre of mass is kept over the support centre with a spring and damper (about 1.6 Hz), which is a capture point
regulator: a push makes the robot shift its weight and bend at the hips and ankles until it is still again, as long
as the push is smaller than what its feet can carry (the stance has no stepping). Leaning, a carried load and an
uneven floor are all in the model the least squares works on, so they need no extra code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors import stance_ik
from robotforge.sim.actuators import is_position_servo, is_torque_motor, is_velocity_servo
from robotforge.sim.geometry import lowest_point

RATE_HZ = 200.0
FALLEN_UP = 0.5  # trunk up-axis z below this means the robot is lying down
# Spring and damper on the centre of mass (rad/s natural frequency, damping ratio), on the trunk's tilt, on the joints.
COM_XY_OMEGA, COM_XY_ZETA = 5.0, 1.2
COM_Z_OMEGA, COM_Z_ZETA = 8.0, 1.2
TRUNK_OMEGA, TRUNK_ZETA = 4.0, 1.2
POSTURE_OMEGA, POSTURE_ZETA = 8.0, 1.2
WEIGHT_COM, WEIGHT_SPIN, WEIGHT_TRUNK, WEIGHT_POSTURE, WEIGHT_FOOT = 10.0, 0.3, 0.3, 0.05, 1.0
FOOT_OMEGA, FOOT_ZETA = 8.0, 1.0  # a foot that lifts at the toe or heel is brought back flat
CONTACT_DAMPING = 20.0  # 1/s: how fast a touching point that moves is brought to rest
CONTACT_LOAD_SHARE = 0.25  # of an even share of the weight that each contact point is made to carry at least
DYNAMICS_WEIGHT = 30.0  # how much more an error in the floating base's dynamics counts than one in a wish
ANGMOM_DAMPING = 5.0  # 1/s: how fast the whole body's angular momentum is asked to fade
DRIFT_STEP = 1e-3  # s
RIDGE = 1e-3  # on the acceleration, so the least squares has one answer
FRICTION_SHARE = 0.7  # of the contact's friction coefficient that the forces use
FORCE_RIDGE = 1e-3  # spreads the load over the contact points instead of piling it on a few
MAX_POINTS_PER_BODY = 4
MAX_PRINT_CORNERS = 6
PRINT_BAND = 0.012  # m above the lowest corner of a foot's print within which a corner still carries weight
CONTACT_MARGIN = 0.004  # m: a contact this far from touching still counts as a support
SETTLE_S = 0.06  # holds the home pose with its servos after the start, while it lands
CANCEL_SHARE = 0.5  # of a joint's inertia that the servo's damper times a control period may add when it is cancelled
STIFF_FROM, STIFF_SPAN = (
    0.25,
    0.35,
)  # a servo is followed as stiff from this much natural frequency x period, fully at the sum
RAMP_S = 1.5  # then eases from the home pose to the balanced stance over this long


@dataclass(frozen=True)
class Drive:
    """What a control value does to one dof: force = gain * ctrl + b0 + b_pos * q + b_vel * qdot, times the gear."""

    actuator: int
    dof: int
    qpos: int
    gear: float
    gain: float
    b0: float
    b_pos: float
    b_vel: float
    force_range: tuple[float, float] | None
    ctrl_range: tuple[float, float] | None


class BalanceError(ValueError):
    """The robot has nothing this controller can work with (no floating base, no actuated joint)."""


def _nnls(a: np.ndarray, b: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Non-negative least squares (Lawson and Hanson's active set method) for the small systems of a contact set."""
    n = a.shape[1]
    x = np.zeros(n)
    passive = np.ones(n, dtype=bool)
    gram, rhs = a.T @ a, a.T @ b
    w = rhs.copy()
    for _ in range(iterations):
        free = np.flatnonzero(passive & (w > 1e-9))
        if free.size == 0:
            break
        j = free[np.argmax(w[free])]
        passive[j] = False
        for _inner in range(n + 1):
            cols = np.flatnonzero(~passive)
            s = np.zeros(n)
            s[cols] = np.linalg.solve(gram[np.ix_(cols, cols)] + 1e-12 * np.eye(cols.size), rhs[cols])
            if s[cols].min() > 0:
                x = s
                break
            negative = cols[s[cols] <= 0]
            alpha = np.min(x[negative] / np.maximum(x[negative] - s[negative], 1e-12))
            x = x + alpha * (s - x)
            passive |= (x <= 1e-12) & ~passive
            x[passive] = 0.0
        w = rhs - gram @ x
    return x


class WholeBodyBalancer:
    """Holds a robot upright on its feet with joint torques worked out from its own dynamics (see the module doc)."""

    rate_hz = RATE_HZ
    max_speed = 0.0

    def __init__(self, model: mujoco.MjModel, robot: str, home_qpos: np.ndarray):
        prefix = f"{robot}/"
        bodies = [b for b in range(1, model.nbody) if (model.body(b).name or "").startswith(prefix)]
        roots = {int(model.body_rootid[b]) for b in bodies}
        free = [
            (b, j)
            for b in roots
            for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free) != 1:
            raise BalanceError(f"robot {robot!r} has no floating trunk, so there is nothing to balance")
        self.robot = robot
        self.trunk, root_joint = free[0]
        self.tree = np.isin(model.body_rootid, [self.trunk])
        self.dofs = np.flatnonzero(self.tree[model.dof_bodyid])
        self.nv = len(self.dofs)
        self.mass = float(model.body_subtreemass[self.trunk])
        self.gravity = -float(model.opt.gravity[2])
        self.drives: list[Drive] = []
        self._find_drives(model)
        if not self.drives:
            raise BalanceError(f"robot {robot!r} has no motor on any joint, so it cannot hold itself up")
        # Which rows of the dynamics a motor can push: the others (the free base, passive joints) only get forces
        # from the contacts.
        driven = {d.dof for d in self.drives}
        self.driven = np.array([i for i, dof in enumerate(self.dofs) if dof in driven])
        self.undriven = np.array([i for i, dof in enumerate(self.dofs) if dof not in driven])
        self.local = {int(dof): i for i, dof in enumerate(self.dofs)}
        self._model_body = model.dof_bodyid.copy()
        self._ankles: set[int] = set()  # the feet and the bodies just above them, once the feet are known
        self.home = home_qpos.copy()
        self.root_dof = int(model.jnt_dofadr[root_joint])
        self.root_qpos = int(model.jnt_qposadr[root_joint])

        scratch = mujoco.MjData(model)
        scratch.qpos[:] = home_qpos
        mujoco.mj_forward(model, scratch)
        self.home_rot = scratch.xmat[self.trunk].reshape(3, 3).copy()
        self.yaw0 = math.atan2(self.home_rot[1, 0], self.home_rot[0, 0])
        floor = min(
            (float(lowest_point(model, scratch, g)[2]) for g in range(model.ngeom) if self._own_geom(model, g)),
            default=0.0,
        )
        self.com_height = float(scratch.subtree_com[self.trunk][2]) - floor
        self._q_hold = {d.actuator: float(home_qpos[d.qpos]) for d in self.drives}
        self.stance = home_qpos.copy()
        self._ramp_from_com: np.ndarray | None = None
        self._feet: dict[int, np.ndarray] = {}  # foot body -> its orientation when it first stood flat
        self._print: dict[int, np.ndarray] = {}  # foot body -> where it first touched, in its own frame
        self._ramp_t0 = 0.0

        self.w_com, self.w_spin, self.w_trunk = WEIGHT_COM, WEIGHT_SPIN, WEIGHT_TRUNK  # how much each wish counts
        self.w_posture, self.w_foot = WEIGHT_POSTURE, WEIGHT_FOOT
        self.com_kp = np.array([COM_XY_OMEGA**2, COM_XY_OMEGA**2, COM_Z_OMEGA**2])
        self.com_kd = np.array([2 * COM_XY_ZETA * COM_XY_OMEGA] * 2 + [2 * COM_Z_ZETA * COM_Z_OMEGA])
        self._mass_matrix = np.zeros((model.nv, model.nv))
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._angmom = np.zeros((3, model.nv))
        self._scratch: mujoco.MjData | None = None
        self._t0: float | None = None
        self.fallen = False
        self.ground_z: float | None = None
        self._support = np.zeros(3)
        self.contacts = 0
        self.last_error = 0.0

    # ---- set-up ------------------------------------------------------------------------

    def _own_geom(self, model: mujoco.MjModel, g: int) -> bool:
        return bool(self.tree[model.geom_bodyid[g]]) and bool(model.geom_contype[g] or model.geom_conaffinity[g])

    def _find_drives(self, model: mujoco.MjModel) -> None:
        for a in range(model.nu):
            if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            joint = int(model.actuator_trnid[a, 0])
            if not self.tree[model.jnt_bodyid[joint]] or int(model.jnt_type[joint]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            if not (is_position_servo(model, a) or is_torque_motor(model, a) or is_velocity_servo(model, a)):
                continue
            if int(model.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE):
                continue
            gear = float(model.actuator_gear[a, 0])
            if gear == 0.0:
                continue
            dof = int(model.jnt_dofadr[joint])
            if any(d.dof == dof for d in self.drives):  # two motors on one joint: the first carries the torque
                continue
            self.drives.append(
                Drive(
                    actuator=a,
                    dof=dof,
                    qpos=int(model.jnt_qposadr[joint]),
                    gear=gear,
                    gain=float(model.actuator_gainprm[a, 0]),
                    b0=float(model.actuator_biasprm[a, 0]),
                    b_pos=float(model.actuator_biasprm[a, 1]),
                    b_vel=float(model.actuator_biasprm[a, 2]),
                    force_range=tuple(model.actuator_forcerange[a]) if model.actuator_forcelimited[a] else None,
                    ctrl_range=tuple(model.actuator_ctrlrange[a]) if model.actuator_ctrllimited[a] else None,
                )
            )

    def _find_stance(self, model: mujoco.MjModel, live: mujoco.MjData, points) -> np.ndarray:
        """The pose the robot stands in moved until its centre of mass is over the middle of its feet (see
        stance_ik): a pose it can hold without anything but its feet holding it up."""
        here: dict[int, list[np.ndarray]] = {}
        for body, pos, _, _ in points:
            here.setdefault(body, []).append(pos)
        feet = [stance_ik.Foot.here(live, body, np.array(pts)) for body, pts in here.items()]
        return stance_ik.balanced_pose(
            model, live.qpos, self.trunk, self.tree, feet, reference=self.home, level=self.home_rot
        )

    # ---- the Session controller interface -----------------------------------------------

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        if speed or turn:
            raise ValueError(
                f"{self.robot} balances on its feet but has no walking gait; it can stand, hold a load and take a push"
            )

    def limit_speed(self, top: float) -> None:  # a stander has no speed to limit
        return None

    @property
    def done(self) -> bool:
        return True

    def status(self) -> dict:
        return {
            "behavior": "fallen" if self.fallen else "stand",
            "robot": self.robot,
            "controller": "whole-body balance",
            "contacts": self.contacts,
            "max_speed": 0.0,
            **({"fallen": True} if self.fallen else {}),
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._t0 is None:
            self._t0 = data.time
        up = float(data.xmat[self.trunk][8])
        self.fallen = self.fallen or up < FALLEN_UP
        points = self._stance_points(data, self._contact_points(model, data))
        self.contacts = len(points)
        if self.fallen or not points or data.time - self._t0 < SETTLE_S:
            self._hold_home(data)
            return
        try:
            torque = self._torques(model, data, points)
        except np.linalg.LinAlgError:
            self._hold_home(data)
            return
        self._apply(data, torque)

    # ---- contacts -----------------------------------------------------------------------

    def _contact_points(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        """(body, point, contact frame, friction) of what this robot stands on. At first the points where its geoms
        touch fixed ones. Once it has stood, the print each foot made then (its corners), carried along with the foot
        for as long as the foot touches anything: the contacts MuJoCo finds under a foot come and go from one step to
        the next, and a force distribution that followed them made the torques jump."""
        live = self._live_contacts(model, data)
        points: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        if not self._print:
            for body, items in live.items():
                for pos, frame, mu in self._spread(items):
                    points.append((body, pos, frame, mu))
            return points
        for body, local in self._print.items():
            items = live.get(body)
            if not items:
                continue
            frame = np.mean([it[1] for it in items], axis=0)
            frame[0:3] /= max(float(np.linalg.norm(frame[0:3])), 1e-9)
            mu = min(it[2] for it in items)
            world = data.xpos[body] + local @ data.xmat[body].reshape(3, 3).T
            low = float(world[:, 2].min())
            points += [(body, p, frame, mu) for p in world if p[2] <= low + PRINT_BAND]
        return points

    def _live_contacts(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> dict[int, list[tuple[np.ndarray, np.ndarray, float]]]:
        """Body -> (point, frame, friction) of every contact of this robot with something fixed to the world."""
        found: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = {}
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > CONTACT_MARGIN:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            mine1, mine2 = bool(self.tree[b1]), bool(self.tree[b2])
            if mine1 == mine2:
                continue
            other = b2 if mine1 else b1
            if model.body_weldid[other] != 0:
                continue
            body = b1 if mine1 else b2
            frame = np.array(c.frame, dtype=float)
            if mine1:  # the frame's normal points from geom1 to geom2; the floor pushes this robot back along it
                frame[0:3] *= -1.0
            found.setdefault(body, []).append((c.pos.copy(), frame, float(c.friction[0])))
        return found

    @staticmethod
    def _spread(items: list[tuple[np.ndarray, np.ndarray, float]]) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """At most MAX_POINTS_PER_BODY of a body's contact points, as far apart as they come."""
        if len(items) <= MAX_POINTS_PER_BODY:
            return items
        chosen = [max(items, key=lambda it: it[0][0] + 0.37 * it[0][1])]
        while len(chosen) < MAX_POINTS_PER_BODY:
            best = max(
                (it for it in items if all(it is not c for c in chosen)),
                key=lambda it: min(float(np.linalg.norm(it[0] - c[0])) for c in chosen),
            )
            chosen.append(best)
        return chosen

    # ---- the balance ---------------------------------------------------------------------

    def _torques(self, model: mujoco.MjModel, data: mujoco.MjData, points) -> np.ndarray:
        idx = self.dofs
        nv = self.nv
        mujoco.mj_fullM(model, data, self._mass_matrix)
        mass = self._mass_matrix[np.ix_(idx, idx)]
        bias = data.qfrc_bias[idx] - data.qfrc_passive[idx]
        qd = data.qvel[idx]

        # The state: contact Jacobians (3 rows a point, world frame), the centre of mass, the angular momentum.
        jc = np.zeros((3 * len(points), nv))
        for k, (body, pos, _, _) in enumerate(points):
            mujoco.mj_jac(model, data, self._jacp, None, pos, body)
            jc[3 * k : 3 * k + 3] = self._jacp[:, idx]
        centre = self._support_centre(data, points)
        self.ground_z = float(np.mean([p[1][2] for p in points]))
        j_com, j_ang = self._momentum_jacobians(model, data)
        com = data.subtree_com[self.trunk].copy()
        v_com = j_com @ qd
        mujoco.mj_subtreeVel(model, data)
        ang = data.subtree_angmom[self.trunk].copy()
        jdot_com, jdot_ang = self._jacobian_drift(model, data, j_com, j_ang)

        if self._ramp_from_com is None:
            self.stance = self._find_stance(model, data, points)
            self._ramp_from_com = com.copy()
            self._ramp_t0 = data.time
            self._feet = {body: data.xmat[body].reshape(3, 3).copy() for body in {b for b, _, _, _ in points}}
            self._print = self._foot_prints(data, points)
            self._ankles = {b for foot in self._print for b in (foot, int(model.body_parentid[foot]))}
        ease = min(max((data.time - self._ramp_t0) / RAMP_S, 0.0), 1.0)
        ease = ease * ease * (3.0 - 2.0 * ease)
        target, v_ref, a_ref = self._com_goal(data, centre, com, ease)

        # What the balance wants, as accelerations: the centre of mass on its way to the middle of the stance, the
        # angular momentum fading, the trunk level, the joints at the stance.
        a_want = a_ref + self.com_kp * (target - com) + self.com_kd * (v_ref - v_com)
        spin = self.mass * self.com_height**2  # kg m^2: turns angular momentum rates into rad/s^2
        mujoco.mj_jacBody(model, data, self._jacp, self._jacr, self.trunk)
        j_rot = self._jacr[:, idx].copy()
        omega = j_rot @ qd
        rot = data.xmat[self.trunk].reshape(3, 3)
        a_rot = (
            TRUNK_OMEGA**2 * stance_ik.rotation_error(self._trunk_target(rot), rot)
            - 2 * TRUNK_ZETA * TRUNK_OMEGA * omega
        )
        jc_pinv = np.linalg.pinv(jc, rcond=1e-6)
        null = np.eye(nv) - jc_pinv @ jc  # accelerations that leave the touching points where they are
        # Touching points that move (a foot rocking on the soft floor) are slowed down, not just left alone.
        qdd_c = -CONTACT_DAMPING * (jc_pinv @ (jc @ qd))
        foot_rows, foot_goal = [], []
        for body, flat in self._feet.items():
            mujoco.mj_jacBody(model, data, self._jacp, self._jacr, body)
            j_foot = self._jacr[:, idx].copy()
            foot_rows.append(j_foot)
            foot_goal.append(
                FOOT_OMEGA**2 * stance_ik.rotation_error(flat, data.xmat[body].reshape(3, 3))
                - 2 * FOOT_ZETA * FOOT_OMEGA * (j_foot @ qd)
            )
        post_rows = np.zeros((len(self.drives), nv))
        post_goal = np.zeros(len(self.drives))
        for n, drive in enumerate(self.drives):
            i = self.local[drive.dof]
            post_rows[n, i] = 1.0
            post_goal[n] = (
                POSTURE_OMEGA**2 * (self._posture(drive.qpos, ease) - data.qpos[drive.qpos])
                - 2 * POSTURE_ZETA * POSTURE_OMEGA * qd[i]
            )

        # The contact forces are non-negative combinations of friction pyramid edges at every contact point: they
        # push, never pull, never slip. Written as columns that turn a weight into a generalized force.
        edges = np.zeros((3 * len(points), 4 * len(points)))
        for k, (_, _, frame, mu) in enumerate(points):
            normal, t1, t2 = frame[0:3], frame[3:6], frame[6:9]
            share = FRICTION_SHARE * max(mu, 0.1)
            for e, tangent in enumerate((t1, -t1, t2, -t2)):
                edges[3 * k : 3 * k + 3, 4 * k + e] = normal + share * tangent
        cols = jc.T @ edges  # (nv, edges)

        # Accelerations (inside the null space of the feet, which do not move) and edge weights together, so that
        # the motion and the forces fit the dynamics of the joints no motor drives (the floating base), and the
        # wishes are met as well as that and the cone allow. For a given set of weights the best accelerations
        # follow in closed form, which leaves a non-negative least squares problem in the weights alone.
        rows = self.undriven
        scale = DYNAMICS_WEIGHT / self.mass
        dyn_a = scale * (mass[rows] @ null)
        dyn_b = scale * cols[rows]
        dyn_c = scale * (mass[rows] @ qdd_c + bias[rows])
        extra = self._extra_tasks(model, data, qd)
        weights = [
            self.w_com,
            self.w_spin,
            self.w_trunk,
            self.w_posture,
            *[self.w_foot] * len(foot_rows),
            *(w for _, _, w in extra),
        ]
        raw = [j_com, j_ang / spin, j_rot, post_rows, *foot_rows, *(r for r, _, _ in extra)]
        goals = [
            a_want - jdot_com,
            (-ANGMOM_DAMPING * ang - jdot_ang) / spin,
            a_rot,
            post_goal,
            *foot_goal,
            *(g for _, g, _ in extra),
        ]
        task_a = np.vstack([w * (r @ null) for w, r in zip(weights, raw, strict=True)])
        task_b = np.concatenate([w * (g - r @ qdd_c) for w, r, g in zip(weights, raw, goals, strict=True)])
        k_mat = dyn_a.T @ dyn_a + task_a.T @ task_a + RIDGE * np.eye(nv)
        k_inv_ab = np.linalg.solve(k_mat, dyn_a.T @ dyn_b)
        k_inv_h = np.linalg.solve(k_mat, -dyn_a.T @ dyn_c + task_a.T @ task_b)
        p_dyn, q_dyn = dyn_a @ k_inv_ab - dyn_b, dyn_a @ k_inv_h + dyn_c
        p_task, q_task = task_a @ k_inv_ab, task_a @ k_inv_h - task_b
        ridge = math.sqrt(FORCE_RIDGE) / (self.mass * self.gravity) * np.eye(cols.shape[1])
        # Every contact point carries a share of the weight at least (CONTACT_MARGIN_SHARE of an even one): a
        # centre of pressure at the very edge of a foot is a foot about to tip, and a real foot tips where a
        # glued one in the model does not.
        floor_load = np.full(cols.shape[1], CONTACT_LOAD_SHARE * self.mass * self.gravity / (4 * len(points)))
        extra = _nnls(
            np.vstack([p_dyn, p_task, ridge]),
            np.concatenate([-q_dyn - p_dyn @ floor_load, -q_task - p_task @ floor_load, -ridge @ floor_load]),
        )
        beta = extra + floor_load
        qdd = qdd_c + null @ (k_inv_h + k_inv_ab @ beta)

        # The torques: what the dynamics ask for beyond what the contacts give.
        f_torque = cols @ beta
        tau = mass @ qdd + bias - f_torque
        self.last_error = float(np.linalg.norm(tau[rows])) if rows.size else 0.0
        self._inertia = np.diag(mass).copy()
        self._qdd = qdd
        self.debug = dict(
            a_want=a_want,
            a_pred=j_com @ qdd + jdot_com,
            v_com=v_com,
            qdd=qdd,
            tau=tau,
            ease=ease,
            com=com,
            target=target,
            ang=ang,
            post=post_goal,
            beta=beta,
        )
        return tau

    @staticmethod
    def _edge_forces(points, beta: np.ndarray) -> np.ndarray:
        """The contact forces (3 a point, world frame) that the friction pyramid weights `beta` stand for."""
        out = np.zeros(3 * len(points))
        for k, (_, _, frame, mu) in enumerate(points):
            normal, t1, t2 = frame[0:3], frame[3:6], frame[6:9]
            m = FRICTION_SHARE * max(mu, 0.1)
            for e, tangent in enumerate((t1, -t1, t2, -t2)):
                out[3 * k : 3 * k + 3] += beta[4 * k + e] * (normal + m * tangent)
        return out

    def _momentum_jacobians(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """(centre of mass Jacobian, angular momentum matrix about it) over this robot's dofs."""
        mujoco.mj_jacSubtreeCom(model, data, self._jacp, self.trunk)
        j_com = self._jacp[:, self.dofs].copy()
        mujoco.mj_angmomMat(model, data, self._angmom, self.trunk)
        return j_com, self._angmom[:, self.dofs].copy()

    def _jacobian_drift(
        self, model: mujoco.MjModel, data: mujoco.MjData, j_com: np.ndarray, j_ang: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """What the two Jacobians change by in a moment along the joint velocities, times those velocities: the part
        of an acceleration that the motion itself brings (zero at rest). A step of a thousandth of a second."""
        qd = data.qvel[self.dofs]
        if float(np.max(np.abs(qd))) < 1e-4:
            return np.zeros(3), np.zeros(3)
        if self._scratch is None:
            self._scratch = mujoco.MjData(model)
        later = self._scratch
        later.qpos[:] = data.qpos
        step = np.zeros(model.nv)
        step[self.dofs] = qd
        mujoco.mj_integratePos(model, later.qpos, step, DRIFT_STEP)
        mujoco.mj_kinematics(model, later)
        mujoco.mj_comPos(model, later)
        j_com2, j_ang2 = self._momentum_jacobians(model, later)
        return (j_com2 - j_com) @ qd / DRIFT_STEP, (j_ang2 - j_ang) @ qd / DRIFT_STEP

    # ---- what a walking subclass changes -------------------------------------------------

    def _stance_points(self, data: mujoco.MjData, points):
        """The contact points the balance counts on: all when standing; a walker leaves out the foot it swings."""
        return points

    def _com_goal(
        self, data: mujoco.MjData, centre: np.ndarray, com: np.ndarray, ease: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, velocity, acceleration) the centre of mass is to have: the middle of the stance, at rest, eased
        there from where it was."""
        target = np.array(
            [
                self._ramp_from_com[0] + ease * (centre[0] - self._ramp_from_com[0]),
                self._ramp_from_com[1] + ease * (centre[1] - self._ramp_from_com[1]),
                self.ground_z + self.com_height,
            ]
        )
        return target, np.zeros(3), np.zeros(3)

    def _extra_tasks(self, model: mujoco.MjModel, data: mujoco.MjData, qd: np.ndarray) -> list:
        """More wishes as (Jacobian rows over this robot's dofs, the accelerations asked, weight): none standing."""
        return []

    def _posture(self, qpos_adr: int, ease: float) -> float:
        """The angle a joint is asked to hold: its home angle, eased toward the one the balanced stance has."""
        return float(self.home[qpos_adr] + ease * (self.stance[qpos_adr] - self.home[qpos_adr]))

    @staticmethod
    def _foot_prints(data: mujoco.MjData, points) -> dict[int, np.ndarray]:
        """Per foot body, the corners of the area it touches, in its own frame."""
        out: dict[int, np.ndarray] = {}
        for body in {b for b, _, _, _ in points}:
            world = np.array([pos for b, pos, _, _ in points if b == body])
            corners = world[stance_ik.hull_indices(world[:, :2])][:MAX_PRINT_CORNERS]
            out[body] = (corners - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
        return out

    def _support_centre(self, data: mujoco.MjData, points) -> np.ndarray:
        """The middle of the stance: the centre of the area the feet cover, a little toward the ankles (see
        stance_ik.middle_of). Once the feet have stood flat, the area is the print they were found with, carried
        along with the feet, so a contact point that flickers on and off as a foot rocks does not move the middle."""
        if not self._print:
            bodies = {body for body, _, _, _ in points}
            centre = stance_ik.hull_centre(np.array([pos for _, pos, _, _ in points]))
        else:
            bodies = set(self._print)
            world = [data.xpos[body] + local @ data.xmat[body].reshape(3, 3).T for body, local in self._print.items()]
            centre = stance_ik.hull_centre(np.vstack(world))
        return stance_ik.toward_ankles(centre, np.mean([data.xpos[b] for b in bodies], axis=0))

    def _trunk_target(self, rot: np.ndarray) -> np.ndarray:
        """The trunk's home orientation, turned to wherever it faces now: tilt is corrected, heading is left alone."""
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        turn = _rz(yaw - self.yaw0)
        return turn @ self.home_rot

    # ---- output --------------------------------------------------------------------------

    def _apply(self, data: mujoco.MjData, tau: np.ndarray) -> None:
        """Set every control so that its actuator gives the torque the balance asked for. A servo is set to aim a
        little off where its joint is, by torque over stiffness. A stiff one (its natural frequency over a control
        period is not small) is also aimed at where the balance wants the joint a period on, or the joint, held
        where it is at every tick, would only creep and the loop with it would rock."""
        period = 1.0 / self.rate_hz
        for drive in self.drives:
            i = self.local[drive.dof]
            force = float(tau[i]) / drive.gear
            if drive.force_range is not None:
                force = float(np.clip(force, *drive.force_range))
            length = drive.gear * float(data.qpos[drive.qpos])
            # The servo's own damper (b_vel) is left on at the ankles, where it keeps a foot from rocking on the soft
            # floor that the rigid contacts the balance works with cannot show. Elsewhere it is cancelled as far as
            # that is safe: it only holds back the motion the balance asks for (the Unitree G1's knee gives 9 N*m per
            # rad/s: a swinging leg lost a third of its torque to it), but cancelling it with the velocity of a
            # control period ago adds a delay to it, and where the damper times the period is more than the joint's
            # inertia (a wrist, a toe, a finger) the loop rings. At most half the inertia's worth is cancelled.
            cancel = 0.0  # the share of the damper's pull that is added back
            if int(self._model_body[drive.dof]) not in self._ankles:
                damper = abs(drive.b_vel) * drive.gear**2
                cancel = 1.0 if damper <= 0 else min(1.0, CANCEL_SHARE * float(self._inertia[i]) / (damper * period))
            speed = cancel * drive.gear * float(data.qvel[drive.dof])
            ctrl = (force - drive.b0 - drive.b_pos * length - drive.b_vel * speed) / drive.gain
            if drive.b_pos < 0.0:  # a position servo
                stiffness = -drive.b_pos * drive.gear**2
                hurry = math.sqrt(stiffness / max(float(self._inertia[i]), 1e-6)) * period
                follow = min(max((hurry - STIFF_FROM) / STIFF_SPAN, 0.0), 1.0)
                if follow:
                    ahead = float(data.qvel[drive.dof]) * period + 0.5 * float(self._qdd[i]) * period**2
                    ctrl += follow * drive.gear * ahead
            if drive.ctrl_range is not None:
                ctrl = float(np.clip(ctrl, *drive.ctrl_range))
            data.ctrl[drive.actuator] = ctrl

    def _hold_home(self, data: mujoco.MjData) -> None:
        """Servos aim at their home angles, motors at nothing: where the robot waits to land, and where it ends
        up when it has fallen."""
        for drive in self.drives:
            if drive.b_pos != 0.0:  # a position servo
                target = self._q_hold[drive.actuator]
                if drive.ctrl_range is not None:
                    target = float(np.clip(target, *drive.ctrl_range))
                data.ctrl[drive.actuator] = target
            else:
                data.ctrl[drive.actuator] = 0.0


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
