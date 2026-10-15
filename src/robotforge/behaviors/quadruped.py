"""Trot gait for four-legged robots (diagonal legs move together) and tripod gait for six-legged ones.

Foot paths are planned open loop, with light feedback from the trunk: its attitude
levels the body and its measured velocity corrects the stride. Foot targets are
expressed in the trunk frame and turned into joint targets with numerical IK, so
the gait only needs to know which joints form each leg. Legs come from our
template's names or, for any other robot (library, imported, built from parts), from
its kinematic tree: four or six chains of three hinges from the floating trunk to a foot.
A tripod gait is the trot's six-legged sibling: front and rear legs of one side step
with the middle leg of the other, so three feet always hold the robot up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint, is_torque_motor
from robotforge.sim.geometry import collides, lowest_point

# Trot phase offset of each leg (diagonal pairs in phase). Also our template's leg names.
TEMPLATE_LEGS = {"front_left": 0.0, "rear_right": 0.0, "front_right": 0.5, "rear_left": 0.5}
# Tripod: two triangles of feet take turns.
TRIPOD_LEGS = {
    "front_left": 0.0,
    "middle_right": 0.0,
    "rear_left": 0.0,
    "front_right": 0.5,
    "middle_left": 0.5,
    "rear_right": 0.5,
}
PLAN_HZ = 100.0  # foot targets and IK; torque-motor PD runs every physics step
TEMPLATE_PERIOD = 0.36  # s per gait cycle for our template dog
# Bigger robots step slower: about 0.36 s per cycle for a Go2-sized leg (hip to foot 0.28 m),
# scaled with leg length. Measured: Go2 walks best at 0.30-0.36 s, Spot (0.44 m legs) at 0.45-0.55 s.
REFERENCE_LEG, REFERENCE_PERIOD = 0.28, 0.36
# Open-loop foot paths are not enough on heavy robots: servos give way under load, so stance feet
# creep forward relative to the trunk. Menagerie's Go2 made 39% of a 0.3 m/s command and slid
# backwards at 10 cm/s when stepping in place. A PI correction on the trunk's measured velocity and
# yaw rate, active only while stepping, makes commands come true (Go2: 0.301 m/s for 0.3).
VELOCITY_KP = 0.5
VELOCITY_KI = 2.0  # 1/s
# The fastest command the gait accepts, as a share of what the longest stride allows. The rest is
# headroom for the velocity correction. Measured top speeds at the full stride: template dog 0.39 of
# 0.52 m/s (it stumbles at full stride), Go2 0.62, Go1 0.66, A1 0.63 of 0.86 m/s. Library robots that
# reach less (Spot: about 0.37) carry their own limit.
SPEED_MARGIN = 0.7
# PD for torque-motor legs: full torque at this joint error (rad), lightly damped on top of the
# joints' own damping. The PD runs explicitly at physics rate, where heavy damping makes light knee
# joints chatter (critical damping sent Go2 bouncing in circles). On Menagerie's Go2 with its raw
# motors these walk 3.2 m in 10 s at 0.3 m/s, straight.
MOTOR_SATURATION = 0.15
MOTOR_DAMPING_RATIO = 0.25

# (name, [hip, thigh, knee] joint ids, foot body id, foot point in that body's frame, phase offset)
LegSpec = tuple[str, list[int], int, np.ndarray, float]


@dataclass
class Leg:
    name: str
    chain: Chain
    actuators: np.ndarray  # actuator ids, same order as chain joints
    phase_offset: float
    hip: np.ndarray  # hip position in the trunk frame
    home_foot: np.ndarray  # foot position in the trunk frame at the home pose
    q: np.ndarray  # last IK solution (warm start)


@dataclass(frozen=True)
class MotorDrive:
    """PD for torque-motor actuators: the gait computes joint targets, this turns them into torques."""

    actuators: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    gear: np.ndarray
    lower: np.ndarray  # control limits (torque / gear)
    upper: np.ndarray


class QuadrupedGait:
    rate_hz = PLAN_HZ

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        legs: list[LegSpec],
        root_joint: int,
        home_qpos: np.ndarray,
        period: float | None = None,
        lift: float | None = None,
        attitude_gain: float = 0.6,
    ):
        self.robot = robot
        self.duty = 0.5
        self.attitude_gain = attitude_gain
        self.root_qpos = model.jnt_qposadr[root_joint]
        self.root_dof = model.jnt_dofadr[root_joint]
        self.trunk = model.jnt_bodyid[root_joint]
        base = home_qpos.copy()
        base[self.root_qpos : self.root_qpos + 7] = [0, 0, 0, 1, 0, 0, 0]  # trunk frame == world frame
        self.ik = IKSolver(model, base)
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}

        self.legs: list[Leg] = []
        for name, joints, body, point, offset in legs:
            missing = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints if j not in joint_to_act]
            if missing:
                raise ValueError(f"robot {robot!r} has no actuator on {', '.join(missing)}, so it cannot walk")
            chain = Chain.from_joints(model, joints, body, point)
            q0 = base[chain.qpos_adr].copy()
            foot = self.ik.point(chain, q0)
            hip = self.ik.data.xpos[model.jnt_bodyid[joints[0]]].copy()
            acts = np.array([joint_to_act[j] for j in joints])
            self.legs.append(Leg(name, chain, acts, offset, hip, foot, q0))
        self._actuators = np.concatenate([leg.actuators for leg in self.legs])
        self._targets = np.concatenate([leg.q for leg in self.legs])
        self.motors = _motor_drive(model, self._actuators)
        if self.motors is not None:
            # Torque motors need their PD loop at physics rate; IK still runs at PLAN_HZ.
            self.rate_hz = 1.0 / model.opt.timestep

        leg_len = float(np.mean([np.linalg.norm(leg.home_foot - leg.hip) for leg in self.legs]))
        if period is None:
            period = float(np.clip(REFERENCE_PERIOD * leg_len / REFERENCE_LEG, 0.3, 0.7))
        self.period = period
        self.lift = lift if lift is not None else 0.25 * leg_len
        # Stride is limited to what the legs can reach comfortably.
        self.max_stride = 0.55 * leg_len
        self._stride_speed = self.max_stride / (self.period * self.duty)  # body speed at the longest stride
        self.max_speed = SPEED_MARGIN * self._stride_speed
        self.command_speed = 0.0
        self.command_turn = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self._last_t: float | None = None
        self._next_plan = -math.inf
        self._phase = 0.0
        self._amplitude = 0.0  # 0 = standing still, 1 = full stepping
        # Trunk velocity (forward, left) in its heading frame and yaw rate, averaged; and the
        # integrated error of the same. Updated by rebinding, never in place, so a shallow copy of
        # the gait (the control layer's preview) cannot disturb the live one.
        self._v_meas = np.zeros(3)
        self._v_int = np.zeros(3)
        self._saturated = False  # the last plan had to shorten a stride

    def limit_speed(self, top: float) -> None:
        """Lower the fastest accepted speed to what this robot is known to reach (m/s)."""
        self.max_speed = min(self.max_speed, top)
        self.command(self.command_speed, self.command_turn)

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        """speed in m/s (negative = backwards), turn in rad/s (positive = left)."""
        self.command_speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.command_turn = float(np.clip(turn, -2.0, 2.0))

    def status(self) -> dict:
        return {
            "behavior": "walk",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        time_went_back = data.time < self._next_plan - 2.0 / PLAN_HZ  # e.g. the data was reset
        if data.time + 1e-9 >= self._next_plan or time_went_back:
            self._plan(data)
            self._next_plan = data.time + 1.0 / PLAN_HZ
        if self.motors is None:
            data.ctrl[self._actuators] = self._targets
        else:
            _apply_pd(self.motors, data, self._targets)

    def _plan(self, data: mujoco.MjData) -> None:
        t = data.time
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        # Smooth command changes so the robot does not stumble on a sudden key press.
        alpha = 1.0 - math.exp(-dt / 0.25)
        self.speed += alpha * (self.command_speed - self.speed)
        self.turn += alpha * (self.command_turn - self.turn)
        moving = abs(self.command_speed) > 1e-3 or abs(self.command_turn) > 1e-3
        self._amplitude += alpha * ((1.0 if moving else 0.0) - self._amplitude)
        self._phase = (self._phase + dt / self.period) % 1.0

        quat = data.xquat[self.trunk]
        roll, pitch = _roll_pitch(quat)
        v_body, turn = self._corrected_velocity(data, dt)
        half_cycle = self.period * self.duty
        saturated = False
        for leg in self.legs:
            # Each foot sweeps backwards during stance at the body's commanded velocity.
            v = np.array([v_body[0] - turn * leg.hip[1], v_body[1] + turn * leg.hip[0]])
            stride = v * half_cycle
            norm = np.linalg.norm(stride)
            if norm > self.max_stride:
                stride *= self.max_stride / norm
                saturated = True
            phase = (self._phase + leg.phase_offset) % 1.0
            target = leg.home_foot.copy()
            if phase < self.duty:
                u = phase / self.duty
                target[:2] += stride * (0.5 - u)
            else:
                u = (phase - self.duty) / (1.0 - self.duty)
                s = 0.5 - 0.5 * math.cos(math.pi * u)  # ease in and out
                target[:2] += stride * (s - 0.5)
                target[2] += self._amplitude * self.lift * math.sin(math.pi * u)
            # Level the trunk: extend legs on the low side.
            target[2] += self.attitude_gain * (-pitch * leg.hip[0] + roll * leg.hip[1])
            leg.q, _ = self.ik.solve(leg.chain, target, leg.q, iterations=4)
        self._targets = np.concatenate([leg.q for leg in self.legs])
        self._saturated = saturated

    def _corrected_velocity(self, data: mujoco.MjData, dt: float) -> tuple[np.ndarray, float]:
        """The body velocity and turn rate to step for: the command plus a PI correction."""
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, data.xquat[self.trunk])
        rot = rot.reshape(3, 3)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        lin = data.qvel[self.root_dof : self.root_dof + 3]  # world frame
        yaw_rate = (rot @ data.qvel[self.root_dof + 3 : self.root_dof + 6])[2]  # angular part is in body frame
        measured = np.array([c * lin[0] + s * lin[1], -s * lin[0] + c * lin[1], yaw_rate])
        beta = 1.0 - math.exp(-dt / self.period)  # average over about one gait cycle
        self._v_meas = self._v_meas + beta * (measured - self._v_meas)
        want = np.array([self.speed, 0.0, self.turn])
        limit = np.array([0.5 * self._stride_speed, 0.5 * self._stride_speed, 1.0])
        if self._amplitude < 0.05:  # standing: nothing to correct, and no windup for the next start
            self._v_int = np.zeros(3)
        elif not self._saturated:  # strides at their limit cannot go faster; integrating then only winds up
            self._v_int = np.clip(self._v_int + VELOCITY_KI * dt * (want - self._v_meas), -limit, limit)
        correction = self._amplitude * (VELOCITY_KP * (want - self._v_meas) + self._v_int)
        corrected = want + np.clip(correction, -limit, limit)
        return corrected[:2], float(corrected[2])


def _roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


# ---- actuators ------------------------------------------------------------------------------


def _motor_drive(model: mujoco.MjModel, actuators: np.ndarray) -> MotorDrive | None:
    """PD gains for torque-motor legs, scaled to each motor's torque limit; None for servo legs."""
    motors = [is_torque_motor(model, int(a)) for a in actuators]
    if not any(motors):
        return None
    if not all(motors):
        raise ValueError("this robot's legs mix position servos and torque motors; the gait drives one kind")
    if not model.actuator_ctrllimited[actuators].all():
        raise ValueError("this robot's leg motors have no torque limits (ctrlrange); the gait needs them")
    joints = model.actuator_trnid[actuators, 0]
    gear = model.actuator_gear[actuators, 0]
    lower, upper = model.actuator_ctrlrange[actuators, 0], model.actuator_ctrlrange[actuators, 1]
    torque_limit = np.maximum(np.abs(lower), np.abs(upper)) * np.abs(gear)
    kp = torque_limit / MOTOR_SATURATION
    dof = model.jnt_dofadr[joints]
    inertia = model.dof_M0[dof]  # each joint's own inertia, armature included
    # Explicit damping is stable only while kd * dt stays well below the inertia it acts on.
    kd = np.minimum(2.0 * MOTOR_DAMPING_RATIO * np.sqrt(kp * inertia), 0.5 * inertia / model.opt.timestep)
    return MotorDrive(actuators, model.jnt_qposadr[joints], dof, kp, kd, gear, lower.copy(), upper.copy())


def _apply_pd(drive: MotorDrive, data: mujoco.MjData, targets: np.ndarray) -> None:
    torque = drive.kp * (targets - data.qpos[drive.qpos_adr]) - drive.kd * data.qvel[drive.dof_adr]
    data.ctrl[drive.actuators] = np.clip(torque / drive.gear, drive.lower, drive.upper)


# ---- finding the legs -----------------------------------------------------------------------


def for_template(
    model: mujoco.MjModel,
    robot: str,
    home_qpos: np.ndarray,
    period: float = TEMPLATE_PERIOD,
    lift: float | None = None,
) -> QuadrupedGait:
    """Build a gait for a robot made from the quadruped template (period: s per cycle, lift: m)."""

    def jid(name: str) -> int:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}/{name}")
        if j < 0:
            raise ValueError(f"robot {robot!r} has no joint {name!r}; is it a quadruped?")
        return j

    legs = []
    for leg, offset in TEMPLATE_LEGS.items():
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/{leg}_foot")
        joints = [jid(f"{leg}_hip"), jid(f"{leg}_thigh"), jid(f"{leg}_knee")]
        legs.append((leg, joints, int(model.site_bodyid[site]), model.site_pos[site].copy(), offset))
    return QuadrupedGait(model, robot, legs, jid("root"), home_qpos, period=period, lift=lift)


def for_robot(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> QuadrupedGait:
    """Build a gait for any four- or six-legged robot: our template by name, others by their kinematic tree."""
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/front_left_foot") >= 0:
        return for_template(model, robot, home_qpos)
    found = discover_legs(model, robot, home_qpos)
    if found is None:
        raise ValueError(
            f"robot {robot!r} does not look like a legged walker "
            "(expected a floating trunk with four or six legs of three hinge joints each)"
        )
    root, legs = found
    return QuadrupedGait(model, robot, legs, root, home_qpos)


def discover_legs(
    model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None
) -> tuple[int, list[LegSpec]] | None:
    """Find the four (or six) legs of a floating robot named `robot`, or None if it has no such legs.

    A leg is a chain of exactly three hinge joints from the trunk to a leaf body whose lowest
    collision geom (the foot) is at least 5 cm below the trunk origin at the home pose. Legs are
    named by where their hips are, assuming the trunk's +x points forward and +y left (the
    convention of MuJoCo Menagerie and URDF); six legs are three per side, front to rear. Robots
    with two-joint or four-joint legs, wheels on the legs, or an extra leaf with three hinges that
    reaches the floor are not recognized.
    """
    prefix = f"{robot}/"
    bodies = [b for b in range(1, model.nbody) if _name(model, mujoco.mjtObj.mjOBJ_BODY, b).startswith(prefix)]
    roots = [b for b in bodies if model.body_parentid[b] == 0]
    if len(roots) != 1 or model.body_jntnum[roots[0]] < 1:
        return None
    trunk = roots[0]
    root_joint = int(model.body_jntadr[trunk])
    if int(model.jnt_type[root_joint]) != mujoco.mjtJoint.mjJNT_FREE:
        return None

    # Pose the robot at home with the trunk at the origin, as the gait's IK will.
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if home_qpos is None else home_qpos
    adr = model.jnt_qposadr[root_joint]
    data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)

    parents = {int(model.body_parentid[b]) for b in bodies}
    candidates = []
    for leaf in (b for b in bodies if b not in parents):
        path = _path_from(model, trunk, leaf)
        joints = [j for b in path for j in _hinges(model, b)]
        if len(joints) != 3:
            continue
        foot = _foot(model, data, path[path.index(model.jnt_bodyid[joints[-1]]) :])
        if foot is None or foot[2][2] > -0.05:  # a leg reaches down below the trunk
            continue
        candidates.append((joints, foot[0], foot[1], data.xpos[model.jnt_bodyid[joints[0]]].copy()))
    if len(candidates) not in (4, 6):
        return None

    center = np.mean([hip for *_, hip in candidates], axis=0)
    sides: dict[str, list] = {"left": [], "right": []}
    for leg in candidates:
        sides["left" if leg[3][1] > center[1] else "right"].append(leg)
    per_side = len(candidates) // 2
    if any(len(side) != per_side for side in sides.values()):
        return None
    rows = ("front", "rear") if per_side == 2 else ("front", "middle", "rear")
    phases = TEMPLATE_LEGS if per_side == 2 else TRIPOD_LEGS
    legs: list[LegSpec] = []
    for side, members in sides.items():
        for row, (joints, body, point, _hip) in zip(rows, sorted(members, key=lambda c: -c[3][0]), strict=True):
            legs.append((f"{row}_{side}", joints, body, point, phases[f"{row}_{side}"]))
    return root_joint, legs


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def _path_from(model: mujoco.MjModel, top: int, body: int) -> list[int]:
    """Bodies from just below `top` down to `body`."""
    path = []
    while body not in (top, 0):
        path.append(body)
        body = int(model.body_parentid[body])
    return path[::-1]


def _hinges(model: mujoco.MjModel, body: int) -> list[int]:
    start, count = model.body_jntadr[body], model.body_jntnum[body]
    return [j for j in range(start, start + count) if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_HINGE]


def _foot(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> tuple[int, np.ndarray, np.ndarray] | None:
    """(body, point in body frame, point in trunk frame) of the lowest collision geom on `bodies`."""
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies and collides(model, g)]
    if not geoms:
        return None
    lowest = min(geoms, key=lambda g: lowest_point(model, data, g)[2])
    body = int(model.geom_bodyid[lowest])
    if int(model.geom_type[lowest]) == mujoco.mjtGeom.mjGEOM_SPHERE:
        world = data.geom_xpos[lowest].copy()  # a ball foot rolls; track its center
    else:
        world = lowest_point(model, data, lowest)
    rot = data.xmat[body].reshape(3, 3)
    return body, rot.T @ (world - data.xpos[body]), world
