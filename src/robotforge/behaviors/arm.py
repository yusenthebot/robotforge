"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down
(leaning a little, high above a grasp or a place, where a short arm cannot point down; see TILTS).
"""

from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint
from robotforge.sim.geometry import highest_point, ray

DOWN = np.array([0.0, 0.0, -1.0])
SPEED = 0.18  # m/s tool speed
APPROACH = 0.07  # m above a grasp or place point
REACH_TOL = 0.008  # m: an IK answer further than this from the target means "cannot reach"
RESTARTS = 24  # extra IK starts, from fixed random joint settings, when the nearest start gets stuck
JOINT_SPEED = 0.6  # rad/s, fastest joint during a move into another elbow or wrist branch
PENETRATION = 0.002  # m: deeper overlaps of the arm with itself or the floor rule an IK answer out
GROUND = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
# Short arms cannot point the gripper straight down everywhere: a PiPER's wrist (joint 5, +-70 degrees) cannot above
# about 13 cm at 30 cm out, nor an SO-100's above about 9 cm, so neither could lift a cube over an 8 cm bin's rim.
# Where the gripper only passes by, high above a grasp or a place, it may lean this far (rad: 20, 35, 50 degrees)
# away from the base or toward it. Grasping and placing still point down.
TILTS = (0.35, 0.6, 0.9)
# An arm riding on a robot carries things over that robot's body: a PiPER on a Go2 swept a cube into the dog's head.
# So the carry clears the robot's shapes under its way: the sector the arm sweeps about its base (its links too, from
# the base out to the tool) from above the grasp to above the place, widened by the object, the jaws around it
# (CARRY_MARGIN, m) and ARC_MARGIN (rad) at either end, looked at with rays every CARRY_STEP (m). A carry that turns
# about half a turn may go round either way (AMBIGUOUS_TURN, rad from half a turn), so then it is the whole disk. Only
# what lies under it counts: the highest shape anywhere on the carrier made a rover with a 25 cm mast behind its arm
# refuse picks in front of it; the tool's ring alone let a UR5e on ANYmal sweep its forearm through the dog's legs.
CARRY_MARGIN = 0.04
ARC_MARGIN = 0.25
CARRY_STEP = 0.01
AMBIGUOUS_TURN = 0.5
# Pointing down leaves the turn of the jaws about the tool's axis free, and the IK left it wherever: beside an SO-100
# its jaws came down 35 degrees across a cube, the fixed jaw on the cube's edge, and pushed it away. A box is grasped
# with the jaws closing square to its faces (on the face nearest to where they close now), where the arm can: within
# SQUARE_TOL (rad) of that.
SQUARE_TOL = 0.05


class OutOfReach(ValueError):
    """Key points of a pick-and-place the tool cannot reach: `bad` names them ("object", "above place", ...).
    `over` is (the body, its top in m) of the robot carrying the arm that the carry had to clear, when only the points
    high above the object and the place were out of reach: that height is then why."""

    def __init__(self, robot: str, bad: list[str], over: tuple[str, float] | None = None):
        self.bad = bad
        self.over = over
        super().__init__(f"{robot} cannot reach the {' and the '.join(bad)} with its gripper pointing down")


@dataclass
class Waypoint:
    position: np.ndarray | None  # tool target in world frame; None = stay
    gripper: float | None  # gripper ctrl value; None = unchanged
    duration: float
    label: str
    joints: np.ndarray | None = None  # a joint-space move to these angles instead of a tool position
    tilt: bool = False  # the gripper may lean there (see TILTS)
    rotation: np.ndarray | None = None  # the tool body's whole orientation there, jaws turned too (see SQUARE_TOL)


class ArmMotion:
    rate_hz = 100.0

    def __init__(self, model: mujoco.MjModel, robot: str, home_qpos: np.ndarray):
        """`robot` names the arm: a robot ("arm") or an arm mounted on one ("rover/arm")."""
        self.robot = robot
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/tcp")
        if site < 0:
            raise ValueError(f"robot {robot!r} has no tool point; is it an arm?")
        self.site = site
        tool_body = int(model.site_bodyid[site])
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}
        # The arm is the powered hinges between the robot's root and the tool: that leaves out wheels,
        # legs or a second arm of the same robot, and works for arms built from parts too.
        joints = _path_hinges(model, tool_body, joint_to_act)
        if not joints:
            raise ValueError(f"{robot!r} has a tool point but no powered joints that move it")
        chain = Chain.from_joints(model, joints, tool_body, model.site_pos[site].copy())
        self.chain = _around_home(model, chain, joints, home_qpos)
        self.joints = joints
        self.actuators = np.array([joint_to_act[j] for j in joints])
        # The tool points along the tcp site's +z (for the templates that is the hand's +z).
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[site])
        self.tool_axis = rot.reshape(3, 3)[:, 2].copy()
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/gripper")
        # A gripper opens at the top of its control range and closes at the bottom, which is not always 0 (a Unitree
        # Z1's is -1.52 rad).
        grip_range = model.actuator_ctrlrange[self.gripper] if self.gripper >= 0 else np.zeros(2)
        self.gripper_open, self.gripper_closed = float(grip_range[1]), float(grip_range[0])
        self.ik = IKSolver(model, home_qpos)
        self.q = home_qpos[self.chain.qpos_adr].copy()
        self.home = self.q.copy()
        # An arm on a moving robot: IK must start from where that robot is now (see sync).
        root = _root_body(model, tool_body)
        free = [j for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])]
        free = [j for j in free if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_adr = int(model.jnt_qposadr[free[0]]) if free else None
        # Paths sweep around the base axis so the tool never cuts through the arm itself.
        self.base_body = int(model.jnt_bodyid[joints[0]])
        # Bodies the arm's joints move, and the robot they belong to: IK answers must not run them into
        # each other, into whatever carries the arm, or into the floor.
        self._moving = {b for b in range(model.nbody) if _descends_from(model, b, self.base_body)}
        self._robot_root = int(model.body_rootid[self.base_body])
        # Shapes that already overlap in the home pose (coarse collision shapes around a joint, common in
        # imported models) are part of the model, not a collision to steer around. Worked out once the
        # moving bodies are known: before that it crashed on any overlap and was then thrown away.
        self._touching_at_home: set[tuple[int, int]] = set()
        self._touching_at_home = self._clashes(self.q)
        self._base_parent = int(model.body_parentid[self.base_body])
        self.ik.point(self.chain, self.q)
        self._read_base(self.ik.data)
        self.closing = _closing(model, self.ik.data, self.gripper, self.chain.body, self.tool_axis)
        self.queue: deque[Waypoint] = deque()
        self._segment: tuple[np.ndarray, Waypoint, float] | None = None  # (start, waypoint, t0)
        self._branch: tuple[np.ndarray, np.ndarray] | None = None  # (q at the start, q at the goal)
        self._axis0 = DOWN  # where the tool pointed when the current move began
        self._axis1 = DOWN  # where it points at the end of the current move
        self._rot0 = np.eye(3)  # the tool body's orientation when the current move began (see Waypoint.rotation)
        self._turn: float | None = None  # how far the current move turns about the base, rad
        self.current_label = "idle"
        self.log: list[str] = []

    def _read_base(self, data: mujoco.MjData) -> None:
        self.base_xy = data.xpos[self.base_body][:2].copy()
        # Heading of the arm's zero yaw in the world (the arm may be placed or carried with a yaw).
        parent_rot = data.xmat[self._base_parent].reshape(3, 3)
        self.base_yaw = float(np.arctan2(parent_rot[1, 0], parent_rot[0, 0]))

    def sync(self, data: mujoco.MjData) -> None:
        """Follow the robot carrying the arm: IK and path planning start from where it is now."""
        if self.free_adr is not None:
            self.ik.base_qpos[self.free_adr : self.free_adr + 7] = data.qpos[self.free_adr : self.free_adr + 7]
        self._read_base(data)

    def base_and_reach(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Where the arm's first joint is, and about how far the arm reaches from it: the spacing of its
        joints along the chain plus the tool offset, as for an arm stretched out straight."""
        points = [*(data.xanchor[j] for j in self.joints), data.site_xpos[self.site]]
        reach = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:], strict=False))
        return np.array(points[0], float), reach

    # ---- planning ------------------------------------------------------------------

    def reach_error(self, target: np.ndarray) -> float:
        return self._solve(target)[1]

    def _solve_tilting(self, target: np.ndarray) -> tuple[np.ndarray, float, bool, np.ndarray]:
        """As _solve, but where the tool cannot point down there it may lean (see TILTS): (q, position error, needed a
        restart, the way the tool points)."""
        q, err, restarted = self._solve(target)
        if err <= REACH_TOL:
            return q, err, restarted, DOWN
        out = np.asarray(target, float)[:2] - self.base_xy
        if np.linalg.norm(out) < 1e-3:
            out = np.array([np.cos(self.base_yaw), np.sin(self.base_yaw)])
        out = np.array([*(out / np.linalg.norm(out)), 0.0])
        for angle in TILTS:
            for side in (1.0, -1.0):  # leaning out, as an arm reaching high does, then in
                axis = np.cos(angle) * DOWN + side * np.sin(angle) * out
                q2, err2, restarted2 = self._solve(target, axis)
                if err2 <= REACH_TOL:
                    return q2, err2, restarted2, axis
        return q, err, restarted, DOWN

    def _solve(
        self, target: np.ndarray, down: np.ndarray = DOWN, rotation: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, bool]:
        """Joint values for the tool at `target` pointing down (or along `down`, or with the tool body turned to
        `rotation`): (q, position error, needed a restart; an orientation not reached counts as out of reach).

        Damped least squares from the current pose can stall on a joint limit or in the wrong elbow
        branch for points the arm reaches easily some other way, so a stuck solve tries again from
        fixed random starts (the base still turned toward the target) and keeps the best answer.
        """
        seed = self._seed(target)
        q, err = self.q, np.inf
        # From the pose the arm is in, then with its base turned toward the target: the first keeps an arm
        # reaching over its shoulder in that pose, the second handles big turns that stall plain DLS.
        aim = {"rotation": rotation} if rotation is not None else {"axis": (self.tool_axis, down)}

        def solve(start: np.ndarray) -> tuple[np.ndarray, float]:
            q2, err2 = self.ik.solve(self.chain, target, start, iterations=60, **aim)
            q2 = self._unwound(q2)
            if rotation is not None and self._turned_from(q2, rotation) > SQUARE_TOL:
                err2 = max(err2, REACH_TOL * 1.01)
            return q2, err2

        for start in (self.q, seed):
            q2, err2 = solve(start)
            if err2 <= REACH_TOL and not self._collides(q2):
                return q2, err2, False
            if err2 < err:
                q, err = q2, err2
        best: tuple[np.ndarray, float, float] | None = None  # (q, error, largest joint move from now)
        rng = np.random.default_rng(0)
        for _ in range(RESTARTS):
            start = rng.uniform(self.chain.lower, self.chain.upper)
            start[0] = seed[0]
            q2, err2 = solve(start)
            if err2 <= REACH_TOL and not self._collides(q2):
                travel = float(np.max(np.abs(q2 - self.q)))
                if best is None or travel < best[2]:  # of the answers that work, the one nearest to now
                    best = (q2, err2, travel)
            elif err2 < err:
                q, err = q2, err2
        if best is not None:
            return best[0], best[1], True
        return q, max(err, REACH_TOL * 1.01), True  # nothing collision-free: as good as out of reach

    def _turned_from(self, q: np.ndarray, rotation: np.ndarray) -> float:
        """How far (rad) the tool body at `q` is turned from `rotation`."""
        self.ik.point(self.chain, q)
        have = self.ik.data.xmat[self.chain.body].reshape(3, 3)
        return float(np.arccos(np.clip((np.trace(have.T @ rotation) - 1) / 2, -1.0, 1.0)))

    def _square(self, faces: np.ndarray | None, at: np.ndarray) -> np.ndarray | None:
        """The tool body's orientation for grasping a box at `at` with its jaws closing square to its faces
        (`faces`: two horizontal unit normals of its sides; see SQUARE_TOL), on the face nearest to where they close
        now; None for an arm whose jaws' closing is not known, or that cannot grasp so there."""
        if faces is None or self.closing is None:
            return None
        self.ik.point(self.chain, self.q)
        now = self.ik.data.xmat[self.chain.body].reshape(3, 3) @ self.closing
        local = np.column_stack([self.tool_axis, self.closing, np.cross(self.tool_axis, self.closing)])
        options = [sign * f for f in faces for sign in (1.0, -1.0)]  # the nearest turn first; the wrist may not go so
        for want in sorted(options, key=lambda f: -float(f[:2] @ now[:2])):
            side = np.array([want[0], want[1], 0.0]) / np.linalg.norm(want[:2])
            rotation = np.column_stack([DOWN, side, np.cross(DOWN, side)]) @ local.T
            if self._solve(at, rotation=rotation)[1] <= REACH_TOL:
                return rotation
        return None

    def _unwound(self, q: np.ndarray) -> np.ndarray:
        """The same pose with each joint that turns a whole turn or more within its limits (a UR5e's, +-360 degrees)
        set the way round nearest to where it is now. On a dog's back, an IK answer one turn away swung a UR5e's base
        through 376 degrees, and the carry about it one and a half turns, until the joint hit its limit and flung the
        cube."""
        out = q.copy()
        for i in np.flatnonzero(self.chain.upper - self.chain.lower >= 2 * np.pi):
            turns = out[i] + 2 * np.pi * np.arange(-2, 3)
            inside = turns[(turns >= self.chain.lower[i] - 1e-9) & (turns <= self.chain.upper[i] + 1e-9)]
            if inside.size:
                out[i] = inside[np.argmin(np.abs(inside - self.q[i]))]
        return out

    def _collides(self, q: np.ndarray) -> bool:
        """True when the arm at `q` runs into itself, its carrier or the floor (in the IK's own copy)."""
        return bool(self._clashes(q) - self._touching_at_home)

    def _clashes(self, q: np.ndarray) -> set[tuple[int, int]]:
        """Geom pairs overlapping at `q` with a moving part of the arm on one side and the robot or the ground
        on the other. Fixed props (bins, tables, walls) do not count: placing into a bin or onto a shelf
        means working right against them, and the moves are not planned around obstacles anyway."""
        m, d = self.ik.model, self.ik.data
        self.ik.point(self.chain, q)  # poses every body
        mujoco.mj_collision(m, d)
        pairs = set()
        for con in d.contact[: d.ncon]:
            if con.dist > -PENETRATION:
                continue
            g1, g2 = int(con.geom1), int(con.geom2)
            if int(m.geom_bodyid[g1]) not in self._moving:
                g1, g2 = g2, g1
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            ground = b2 == 0 and int(m.geom_type[g2]) in GROUND
            if b1 in self._moving and (ground or (b2 != 0 and int(m.body_rootid[b2]) == self._robot_root)):
                pairs.add((min(g1, g2), max(g1, g2)))
        return pairs

    def _seed(self, target: np.ndarray) -> np.ndarray:
        """Start IK with the base already turned toward the target; DLS alone can stall on big turns."""
        q = self.q.copy()
        d = np.asarray(target)[:2] - self.base_xy
        if np.linalg.norm(d) > 1e-3:
            lo, hi = self.chain.lower[0], self.chain.upper[0]
            want = float((np.arctan2(d[1], d[0]) - self.base_yaw + np.pi) % (2 * np.pi) - np.pi)
            # The same heading, one turn either way: take the one inside the limits closest to now.
            inside = [a for a in (want, want - 2 * np.pi, want + 2 * np.pi) if lo - 1e-9 <= a <= hi + 1e-9]
            q[0] = min(inside, key=lambda a: abs(a - q[0])) if inside else np.clip(want, lo, hi)
        return q

    def move_to(self, target, gripper: str | None = None, from_point: np.ndarray | None = None) -> None:
        target = np.asarray(target, float)
        start = from_point if from_point is not None else self.tool_target_end()
        dist = float(np.linalg.norm(target - start)) if start is not None else 0.2
        self.queue.append(Waypoint(target, self._grip(gripper), max(0.5, dist / SPEED), "move"))

    def set_gripper(self, state: str, wait: float = 0.5) -> None:
        self.queue.append(Waypoint(None, self._grip(state), wait, f"gripper {state}"))

    def go_home(self) -> None:
        """Back to the home pose joint by joint, e.g. so every run of a test starts from the same place."""
        travel = float(np.max(np.abs(self.home - self._planned_joints())))
        self.queue.append(Waypoint(None, None, max(0.5, travel / JOINT_SPEED), "home", joints=self.home.copy()))

    def _planned_joints(self) -> np.ndarray:
        for wp in reversed(self.queue):
            if wp.joints is not None:
                return wp.joints
        return self.q

    def pick_and_place(
        self,
        obj_pos: np.ndarray,
        obj_half_height: float,
        place: np.ndarray,
        place_top: float | None = None,
        faces: np.ndarray | None = None,
    ) -> None:
        """Queue a full pick-and-place. Raises OutOfReach if any key point is out of reach.

        `place` is where the object's bottom should end up; `place_top` is the highest point
        around it (e.g. a bin's rim), which the carried object must clear. `faces` are a box's two horizontal
        side normals, to grasp it square to them (see SQUARE_TOL).
        """
        grasp = obj_pos.copy()
        drop = place + [0, 0, obj_half_height + 0.01]
        rim = max(place_top if place_top is not None else place[2], grasp[2] - obj_half_height)
        travel_z = max(grasp[2], drop[2]) + APPROACH
        travel_z = max(travel_z, rim + 2 * obj_half_height + 0.03)
        over = None
        if self.free_adr is not None:  # riding on a robot: clear what of it lies under the carry (see CARRY_MARGIN)
            found = self._carrier_under(grasp, drop, obj_half_height, travel_z - 2 * obj_half_height - 0.03)
            if found is not None:
                top, geom = found
                travel_z = top + 2 * obj_half_height + 0.03
                over = (self.ik.model.body(int(self.ik.model.geom_bodyid[geom])).name, top)
        above = np.array([grasp[0], grasp[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        points = {"object": grasp, "above object": above, "place": drop, "above place": above_drop}
        bad = [
            name
            for name, p in points.items()
            if (self._solve_tilting(p)[1] if name.startswith("above") else self.reach_error(p)) > REACH_TOL
        ]
        if bad:
            high_only = all(name.startswith("above") for name in bad)
            raise OutOfReach(self.robot, bad, over if high_only else None)
        square = self._square(faces, grasp)  # the jaws turn square to the box on the way down to it
        start = self.tool_target_end()
        self.queue.append(Waypoint(above, self._grip("open"), self._time(start, above), "approach", tilt=True))
        self.queue.append(Waypoint(grasp, None, self._time(above, grasp) + 0.3, "descend", rotation=square))
        self.queue.append(Waypoint(None, self._grip("close"), 0.6, "grasp"))
        self.queue.append(Waypoint(above, None, self._time(grasp, above) + 0.3, "lift", tilt=True))
        self.queue.append(Waypoint(above_drop, None, self._time(above, above_drop), "carry", tilt=True))
        self.queue.append(Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower"))
        self.queue.append(Waypoint(None, self._grip("open"), 0.5, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat", tilt=True))

    def _carrier_under(self, a: np.ndarray, b: np.ndarray, half: float, floor: float) -> tuple[float, int] | None:
        """The highest point (m) of the robot carrying the arm (the arm left out) under the way an object `half` high
        is carried from above `a` to above `b` (see CARRY_MARGIN), and the geom there; None when nothing of the robot
        there rises above `floor` (m)."""
        m, d = self.ik.model, self.ik.data
        self.ik.point(self.chain, self.q)  # poses every body, the carrier where it is now (see sync)
        mine = f"{self.robot}/"
        da, db = np.asarray(a, float)[:2] - self.base_xy, np.asarray(b, float)[:2] - self.base_xy
        ra, rb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
        margin = half + CARRY_MARGIN
        r_lo, r_hi = 0.0, max(ra, rb) + margin
        ta = float(np.arctan2(da[1], da[0]))
        turn = self._sweep(ta, float(np.arctan2(db[1], db[0]))) if min(ra, rb) > 1e-3 else 2 * np.pi
        whole = abs(abs(turn) - np.pi) < AMBIGUOUS_TURN or abs(turn) >= 2 * np.pi - 2 * ARC_MARGIN
        tops = {}
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            if int(m.body_rootid[body]) != self._robot_root or (m.body(body).name or "").startswith(mine):
                continue
            if body in self._moving:  # the arm's own, also a gripper mounted on its flange ("dog/gripper/...")
                continue
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            top = float(highest_point(m, d, g)[2])
            reach = float(np.linalg.norm(d.geom_xpos[g][:2] - self.base_xy))
            if top > floor and r_lo - m.geom_rbound[g] <= reach <= r_hi + m.geom_rbound[g]:
                tops[g] = top
        if not tops:
            return None
        start_z = max(tops.values()) + 0.01
        down = np.array([0.0, 0.0, -1.0])
        best: tuple[float, int] | None = None
        for r in np.arange(r_lo, r_hi + CARRY_STEP / 2, CARRY_STEP):
            if whole:
                angles = np.arange(0.0, 2 * np.pi, CARRY_STEP / max(r, CARRY_STEP))
            else:
                spread = ARC_MARGIN + margin / max(r, margin)
                side = 1.0 if turn >= 0 else -1.0
                angles = ta + side * np.arange(-spread, abs(turn) + spread, CARRY_STEP / max(r, CARRY_STEP))
            for angle in angles:
                start = np.array([*(self.base_xy + r * np.array([np.cos(angle), np.sin(angle)])), start_z])
                for g in tops:
                    dist = ray(m, d, g, start, down)
                    if dist >= 0 and start_z - dist > floor and (best is None or start_z - dist > best[0]):
                        best = (start_z - dist, g)
        return best

    def tool_target_end(self) -> np.ndarray:
        """Where the tool will be once the queued moves are done."""
        pending = [*([self._segment[1]] if self._segment is not None else []), *self.queue]
        for wp in reversed(pending):
            if wp.joints is not None:
                return self.ik.point(self.chain, wp.joints)
            if wp.position is not None:
                return wp.position
        return self.ik.point(self.chain, self.q)

    @property
    def done(self) -> bool:
        return not self.queue and self._segment is None

    def status(self) -> dict:
        return {
            "behavior": "arm",
            "robot": self.robot,
            "step": self.current_label,
            "remaining_steps": len(self.queue),
            "done": self.done,
        }

    # ---- control -------------------------------------------------------------------

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.free_adr is not None:
            self.sync(data)
        t = data.time
        if self._segment is None and self.queue:
            wp = self.queue.popleft()
            start = data.site_xpos[self.site].copy()
            self._branch = None
            self._turn = None
            if wp.joints is not None:
                self._branch = (self.q.copy(), wp.joints)
            elif wp.position is not None:
                goal, err, restarted, self._axis1 = (
                    self._solve_tilting(wp.position)
                    if wp.tilt
                    else (*self._solve(wp.position, rotation=wp.rotation), DOWN)
                )
                if err <= REACH_TOL:
                    self._turn = self._path_turn(start, wp.position, float(goal[0] - self.q[0]))
                if restarted and err <= REACH_TOL:
                    # The goal lies in another elbow or wrist branch, which no straight tool path reaches:
                    # move joint by joint instead, as industrial arms do, slowly enough not to fling a load.
                    self._branch = (self.q.copy(), goal)
                    travel = float(np.max(np.abs(goal - self.q)))
                    wp = Waypoint(wp.position, wp.gripper, max(wp.duration, travel / JOINT_SPEED), wp.label)
            self._segment = (start, wp, t)
            # Where the tool points now, in the world, also on a carrier (whose pose the IK follows, see sync): a Z1 on
            # a Go2, snapped from its home pose to pointing down on the first ticks, jolted the dog 25 mm back and it
            # stepped off half a metre.
            self._rot0 = data.xmat[self.chain.body].reshape(3, 3).copy()
            self._axis0 = self._rot0 @ self.tool_axis
            self.current_label = wp.label
            if wp.gripper is not None and self.gripper >= 0:
                data.ctrl[self.gripper] = wp.gripper
        if self._segment is None:
            self.current_label = "idle"
            data.ctrl[self.actuators] = self.q
            return
        start, wp, t0 = self._segment
        u = min(1.0, (t - t0) / wp.duration)
        if wp.position is not None or wp.joints is not None:
            s = u * u * (3 - 2 * u)  # smoothstep
            if self._branch is not None:
                q0, q1 = self._branch
                self.q = q0 + (q1 - q0) * s
            elif wp.rotation is not None:  # the jaws turn square to a box while the tool goes down to it
                target = self._arc_point(start, wp.position, s)
                rotation = _blend(self._rot0, wp.rotation, s)
                self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, rotation=rotation)
            else:
                target = self._arc_point(start, wp.position, s)
                # e.g. out of a home pose, or leaning out over a bin, gradually
                axis = (self.tool_axis, _turn_toward(self._axis0, self._axis1, s))
                self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, axis=axis)
        data.ctrl[self.actuators] = self.q
        if u >= 1.0:
            self._segment = None

    def _arc_point(self, a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
        """Interpolate in cylindrical coordinates (radius, angle, height) about the base axis."""
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        ra, rb = np.linalg.norm(da), np.linalg.norm(db)
        if min(ra, rb) < 1e-3:
            return a + (b - a) * s
        ta = np.arctan2(da[1], da[0])
        turn = self._turn if self._turn is not None else self._sweep(ta, np.arctan2(db[1], db[0]))
        r, t = ra + (rb - ra) * s, ta + turn * s
        return np.array([*(self.base_xy + r * np.array([np.cos(t), np.sin(t)])), a[2] + (b[2] - a[2]) * s])

    def _path_turn(self, a: np.ndarray, b: np.ndarray, joint_turn: float) -> float:
        """How far the path turns about the base: the tool's change in angle, taken around the side the
        base joint really turns. Judging it from tool angles alone mistook a straight lowering near the
        joint's +-180 degree limit for a full circle, which flung the cube."""
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        step = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))  # signed, -pi..pi
        return min((step + k * 2 * np.pi for k in (-1, 0, 1)), key=lambda turn: abs(turn - joint_turn))

    def _sweep(self, from_angle: float, to_angle: float) -> float:
        """Signed turn from one world heading to another that keeps the base joint inside its limits."""
        lo, hi = self.chain.lower[0], self.chain.upper[0]
        start = np.clip(from_angle - self.base_yaw, lo, hi)
        goal = to_angle - self.base_yaw
        options = [
            g
            for g in (goal, goal - 2 * np.pi, goal + 2 * np.pi, goal - 4 * np.pi, goal + 4 * np.pi)
            if lo - 1e-6 <= g <= hi + 1e-6
        ]
        if not options:
            return float((to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi)
        return float(min(options, key=lambda g: abs(g - start)) - start)

    def unreachable(self, points: dict[str, np.ndarray], tol: float = REACH_TOL) -> list[str]:
        """Names of points the tool cannot reach while pointing down."""
        return [name for name, p in points.items() if self.reach_error(p) > tol]

    def _grip(self, state: str | None) -> float | None:
        if state is None or self.gripper < 0:
            return None
        return self.gripper_open if state == "open" else self.gripper_closed

    @staticmethod
    def _time(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.5, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / SPEED)


def _around_home(model: mujoco.MjModel, chain: Chain, joints: list[int], home_qpos: np.ndarray) -> Chain:
    """The chain with each joint that turns without end allowed half a turn either way from its home angle. The IK
    otherwise keeps such joints in -180..180 degrees: a Kinova Gen3 starts with joint 3 at 180, pinned at that edge, and
    its gripper ended 34 cm from where move_arm sent it, the arm jammed against its own base."""
    endless = np.array([not model.jnt_limited[j] for j in joints])
    if not endless.any():
        return chain
    home = home_qpos[chain.qpos_adr]
    lower = np.where(endless, home - np.pi, chain.lower)
    upper = np.where(endless, home + np.pi, chain.upper)
    return dataclasses.replace(chain, lower=lower, upper=upper)


def _path_hinges(model: mujoco.MjModel, body: int, powered: dict[int, int]) -> list[int]:
    """Powered hinge joints from the root of `body`'s robot down to `body`, in that order."""
    joints: list[int] = []
    while body > 0:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        own = [
            j for j in range(start, start + count) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j in powered
        ]
        joints = own + joints
        body = int(model.body_parentid[body])
    return joints


def _closing(
    model: mujoco.MjModel, data: mujoco.MjData, gripper: int, body: int, tool_axis: np.ndarray
) -> np.ndarray | None:
    """The way an arm's jaws close, in the tool body's frame and across its tool axis: along the fingers' slide, or
    across a hinged jaw's axis; from the joint the gripper drives, or the first its tendon pulls (a Panda's, a
    Robotiq's). None when there is no such joint (`data` is posed)."""
    if gripper < 0:
        return None
    target = int(model.actuator_trnid[gripper, 0])
    kind = int(model.actuator_trntype[gripper])
    if kind == int(mujoco.mjtTrn.mjTRN_TENDON):
        start, count = model.tendon_adr[target], model.tendon_num[target]
        wraps = [int(model.wrap_objid[w]) for w in range(start, start + count) if model.wrap_type[w] == 1]
        if not wraps:  # mjWRAP_JOINT is 1: a fixed tendon over joints
            return None
        joint = wraps[0]
    elif kind == int(mujoco.mjtTrn.mjTRN_JOINT):
        joint = target
    else:
        return None
    axis = data.xmat[body].reshape(3, 3).T @ data.xaxis[joint]
    if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
        axis = np.cross(axis, tool_axis)
    elif model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE:
        return None
    across = axis - tool_axis * (axis @ tool_axis)
    norm = float(np.linalg.norm(across))
    return across / norm if norm > 0.3 else None


def _blend(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The orientation a fraction `s` of the way from rotation matrix `a` to `b`."""
    qa, qb = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qa, np.ascontiguousarray(a).ravel())
    mujoco.mju_mat2Quat(qb, np.ascontiguousarray(b).ravel())
    turn = np.zeros(3)
    mujoco.mju_subQuat(turn, qb, qa)  # qa * quat(turn) = qb
    q = qa.copy()
    mujoco.mju_quatIntegrate(q, turn, s)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def _turn_toward(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """The unit vector a fraction `s` of the way from direction `a` to direction `b` (spherical blend)."""
    a = a / np.linalg.norm(a)
    angle = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if angle < 1e-6:
        return b
    if np.pi - angle < 1e-6:  # opposite: turn through any perpendicular direction
        a = a + 1e-3 * np.array([1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        a, angle = a / np.linalg.norm(a), float(np.arccos(np.clip(a @ b / np.linalg.norm(a), -1.0, 1.0)))
    return (np.sin((1 - s) * angle) * a + np.sin(s * angle) * b) / np.sin(angle)


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _root_body(model: mujoco.MjModel, body: int) -> int:
    while model.body_parentid[body] > 0:
        body = int(model.body_parentid[body])
    return body
