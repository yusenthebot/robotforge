"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint, is_position_servo
from robotforge.sim.geometry import collides

DOWN = np.array([0.0, 0.0, -1.0])
SPEED = 0.18  # m/s tool speed
APPROACH = 0.07  # m above a grasp or place point
RISE_MIN = 0.05  # m: a tool this far below the carrying height rises straight up before it moves across
REACH_TOL = 0.008  # m: an IK answer further than this from the target means "cannot reach"
RESTARTS = 24  # extra IK starts, from fixed random joint settings, when the nearest start gets stuck
JOINT_SPEED = 0.6  # rad/s, fastest joint during a move into another elbow or wrist branch
PENETRATION = 0.002  # m: deeper overlaps of the arm with itself or the floor rule an IK answer out
GROUND = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
UPRIGHT = 0.9  # an arm's first hinge turns it toward a target when its axis is this close to vertical (cosine)
FAR_FROM_DOWN = 0.7  # cosine: a tool pointing more than 45 degrees off down moves joint by joint to a goal
CARRY_CHECKS = 7  # points checked along the way across with a load, between the two points above pick and place
CARRY_RAISE = 0.05  # m: each try carries the load this much higher
CARRY_RAISES = 8  # tries before giving up and carrying at the first height (up to 0.4 m higher)
LOAD_CLEARANCE = 0.03  # m kept between a carried load and the robot carrying the arm


class OutOfReach(ValueError):
    """Key points of a pick-and-place the tool cannot reach: `bad` names them ("object", "above place", ...)."""

    def __init__(self, robot: str, bad: list[str]):
        self.bad = bad
        super().__init__(f"{robot} cannot reach the {' and the '.join(bad)} with its gripper pointing down")


class DownReach(NamedTuple):
    """A point against an arm's reach with its gripper pointing down."""

    base: np.ndarray  # the arm's first hinge, raised or lowered toward the point by a lift it rides on
    out: float  # m from the base to the point, across the floor
    reach: float  # m the arm reaches out at the point's height; 0 at a height it cannot get to
    longest: float  # m it reaches out at the height of its base


@dataclass
class Waypoint:
    position: np.ndarray | None  # tool target in world frame; None = stay
    gripper: float | None  # gripper ctrl value; None = unchanged
    duration: float
    label: str
    joints: np.ndarray | None = None  # a joint-space move to these angles instead of a tool position


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
        joint_to_act = _powered(model)
        joints = arm_joints(model, site)
        if not joints:
            raise ValueError(f"{robot!r} has a tool point but no powered joints that move it")
        self.chain = Chain.from_joints(model, joints, tool_body, model.site_pos[site].copy())
        self.joints = joints
        self.actuators = np.array([joint_to_act[j] for j in joints])
        # The tool points along the tcp site's +z (for the templates that is the hand's +z).
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[site])
        self.tool_axis = rot.reshape(3, 3)[:, 2].copy()
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/gripper")
        self.gripper_open = float(model.actuator_ctrlrange[self.gripper][1]) if self.gripper >= 0 else 0.0
        self.ik = IKSolver(model, home_qpos)
        self.q = home_qpos[self.chain.qpos_adr].copy()
        self.home = self.q.copy()
        # Paths sweep around the base axis so the tool never cuts through the arm itself: the arm's first hinge,
        # when it turns about an upright axis (an arm with none, like one hanging from a shoulder, moves in
        # straight lines).
        self.yaw = _yaw_index(model, joints, home_qpos)
        # Slides in the chain: those before the first hinge are a lift that carries the whole arm up and down.
        self._slides = {j for j in joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE}
        first_hinge = next((i for i, j in enumerate(joints) if j not in self._slides), len(joints))
        self._lifts = joints[:first_hinge]
        self.base_body = int(model.jnt_bodyid[joints[self.yaw if self.yaw is not None else 0]])
        # Bodies the arm's joints move, and the robot they belong to: IK answers must not run them into
        # each other, into whatever carries the arm, or into the floor.
        first = int(model.jnt_bodyid[joints[0]])
        self._moving = {b for b in range(model.nbody) if _descends_from(model, b, first)}
        self._robot_root = int(model.body_rootid[first])
        # Shapes that already overlap in the home pose (coarse collision shapes around a joint, common in
        # imported models) are part of the model, not a collision to steer around. Worked out once the
        # moving bodies are known: before that it crashed on any overlap and was then thrown away.
        self._touching_at_home: set[tuple[int, int]] = set()
        self._touching_at_home = self._clashes(self.q)
        self._base_parent = int(model.body_parentid[self.base_body])
        self.ik.point(self.chain, self.q)
        self._read_base(self.ik.data)
        self.queue: deque[Waypoint] = deque()
        self._segment: tuple[np.ndarray, Waypoint, float] | None = None  # (start, waypoint, t0)
        self._branch: tuple[np.ndarray, np.ndarray] | None = None  # (q at the start, q at the goal)
        self._axis0 = DOWN  # where the tool pointed when the current move began
        self._turn: float | None = None  # how far the current move turns about the base, rad
        self.current_label = "idle"
        self.log: list[str] = []

    def _read_base(self, data: mujoco.MjData) -> None:
        self.base_xy = data.xpos[self.base_body][:2].copy()
        # Heading of the arm's zero yaw in the world (the arm may be placed or carried with a yaw).
        parent_rot = data.xmat[self._base_parent].reshape(3, 3)
        self.base_yaw = float(np.arctan2(parent_rot[1, 0], parent_rot[0, 0]))

    def sync(self, data: mujoco.MjData) -> None:
        """Follow whatever carries the arm as it is now: the robot that drives or walks with it, and joints the arm
        rides on but does not move itself (a turntable two arms share, a lift set by hand). IK and path planning
        start from there."""
        self.ik.base_qpos[:] = data.qpos
        self._read_base(data)

    def down_reach(self, data: mujoco.MjData, point: np.ndarray) -> DownReach:
        """How a point compares with the arm's reach with its gripper pointing down (see DownReach). The arm reaches
        with its hinges stretched out to the wrist and the hand hanging below. A lift the arm rides on carries the
        first hinge up or down toward the point rather than adding reach: measured from the foot of the column,
        stretched out straight, a picker claimed 0.98 m for an arm that reaches 0.47 m with its gripper down, and
        points too far away were said to be too near."""
        point = np.asarray(point, float)
        anchors = [data.xanchor[j] for j in self.joints if j not in self._slides]
        arm = sum(float(np.linalg.norm(b - a)) for a, b in zip(anchors, anchors[1:], strict=False))
        wrist = point + np.array([0.0, 0.0, float(np.linalg.norm(data.site_xpos[self.site] - anchors[-1]))])
        base = np.array(anchors[0], float)
        for j in self._lifts:
            axis, now = data.xaxis[j], float(data.qpos[self.ik.model.jnt_qposadr[j]])
            lo, hi = self.ik.model.jnt_range[j]
            base += (float(np.clip(now + (wrist - base) @ axis, lo, hi)) - now) * axis
        here = float(np.sqrt(max(arm**2 - (wrist[2] - base[2]) ** 2, 0.0)))
        return DownReach(base, float(np.linalg.norm(point[:2] - base[:2])), here, arm)

    # ---- planning ------------------------------------------------------------------

    def reach_error(self, target: np.ndarray) -> float:
        return self._solve(target)[1]

    def _solve(self, target: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Joint values for the tool at `target` pointing down: (q, position error, needed a restart).

        Damped least squares from the current pose can stall on a joint limit or in the wrong elbow
        branch for points the arm reaches easily some other way, so a stuck solve tries again from
        fixed random starts (the base still turned toward the target) and keeps the best answer.
        """
        seed = self._seed(target)
        q, err = self.q, np.inf
        # From the pose the arm is in, then with its base turned toward the target: the first keeps an arm
        # reaching over its shoulder in that pose, the second handles big turns that stall plain DLS.
        for start in (self.q, seed):
            q2, err2 = self.ik.solve(self.chain, target, start, iterations=60, axis=(self.tool_axis, DOWN))
            if err2 <= REACH_TOL and not self._collides(q2):
                return q2, err2, False
            if err2 < err:
                q, err = q2, err2
        best: tuple[np.ndarray, float, float] | None = None  # (q, error, largest joint move from now)
        rng = np.random.default_rng(0)
        for _ in range(RESTARTS):
            start = rng.uniform(self.chain.lower, self.chain.upper)
            if self.yaw is not None:
                start[self.yaw] = seed[self.yaw]
            q2, err2 = self.ik.solve(self.chain, target, start, iterations=60, axis=(self.tool_axis, DOWN))
            if err2 <= REACH_TOL and not self._collides(q2):
                travel = float(np.max(np.abs(q2 - self.q)))
                if best is None or travel < best[2]:  # of the answers that work, the one nearest to now
                    best = (q2, err2, travel)
            elif err2 < err:
                q, err = q2, err2
        if best is not None:
            return best[0], best[1], True
        return q, max(err, REACH_TOL * 1.01), True  # nothing collision-free: as good as out of reach

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
        i = self.yaw
        if i is not None and np.linalg.norm(d) > 1e-3:
            lo, hi = self.chain.lower[i], self.chain.upper[i]
            want = float((np.arctan2(d[1], d[0]) - self.base_yaw + np.pi) % (2 * np.pi) - np.pi)
            # The same heading, one turn either way: take the one inside the limits closest to now.
            inside = [a for a in (want, want - 2 * np.pi, want + 2 * np.pi) if lo - 1e-9 <= a <= hi + 1e-9]
            q[i] = min(inside, key=lambda a: abs(a - q[i])) if inside else np.clip(want, lo, hi)
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
        self, obj_pos: np.ndarray, obj_half_height: float, place: np.ndarray, place_top: float | None = None
    ) -> None:
        """Queue a full pick-and-place. Raises OutOfReach if any key point is out of reach.

        `place` is where the object's bottom should end up; `place_top` is the highest point
        around it (e.g. a bin's rim), which the carried object must clear.
        """
        grasp = obj_pos.copy()
        drop = place + [0, 0, obj_half_height + 0.01]
        rim = max(place_top if place_top is not None else place[2], grasp[2] - obj_half_height)
        travel_z = max(grasp[2], drop[2]) + APPROACH
        travel_z = max(travel_z, rim + 2 * obj_half_height + 0.03)
        above = np.array([grasp[0], grasp[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        bad = self.unreachable({"object": grasp, "above object": above, "place": drop, "above place": above_drop})
        if bad:
            raise OutOfReach(self.robot, bad)
        carry_z = self._carry_height(grasp, drop, travel_z, obj_half_height)
        if carry_z is None:
            raise ValueError(
                f"{self.robot} cannot carry it across: on the way around its base from the object to the place, the "
                "load would hit the robot carrying the arm, or the gripper cannot point down there, even carried "
                f"{CARRY_RAISES * CARRY_RAISE:.1f} m higher; put both on the same side of the robot, or turn it"
            )
        above[2] = above_drop[2] = travel_z = carry_z
        start = self.tool_target_end()
        # Starting low (an arm folded over the robot that carries it), go up first and across at the carrying height:
        # swinging around at the folded height ran a picker's hand into the table beside it.
        risen = np.array([start[0], start[1], travel_z])
        if start[2] < travel_z - RISE_MIN and self.reach_error(risen) <= REACH_TOL:
            self.queue.append(Waypoint(risen, self._grip("open"), self._time(start, risen), "rise"))
            start = risen
        elif start[2] > travel_z + RISE_MIN and not self._reaches_around(start, above):
            # Starting high (an arm held up at home) where the tool cannot reach all the way down on a slant, go across
            # high and come straight down over the object: sweeping down on a slant, a scorpion's claw caught on its
            # own front leg on the way and closed 3 cm above the cube. Only then: crossing high, a Panda's upright
            # shoulder lines up its first and third joints, the turn twists into both, and from that twisted grasp
            # the carry went the long way round and flung the cube 1.4 m.
            for z in np.arange(start[2], travel_z + RISE_MIN, -CARRY_RAISE):
                over = np.array([above[0], above[1], z])
                if self._reaches_around(start, over):
                    self.queue.append(Waypoint(over, self._grip("open"), self._time(start, over), "approach"))
                    start = over
                    break
        self.queue.append(Waypoint(above, self._grip("open"), self._time(start, above), "approach"))
        self.queue.append(Waypoint(grasp, None, self._time(above, grasp) + 0.3, "descend"))
        self.queue.append(Waypoint(None, self._grip("close"), 0.6, "grasp"))
        self.queue.append(Waypoint(above, None, self._time(grasp, above) + 0.3, "lift"))
        self.queue.append(Waypoint(above_drop, None, self._time(above, above_drop), "carry"))
        self.queue.append(Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower"))
        self.queue.append(Waypoint(None, self._grip("open"), 0.5, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat"))

    def _carry_height(self, grasp: np.ndarray, drop: np.ndarray, low: float, load_half_height: float) -> float | None:
        """The lowest height from `low` up at which the tool reaches every point of the way across, around the base
        from above the object to above the place, and the load it carries clears the robot the arm rides on. A
        Go2's PiPER moving a floor cube from its left to its right swung it round in front at 10 cm, where its
        gripper cannot point down without hitting the dog's head: the arm stalled, then jumped and flung the cube.
        At 30 cm the gripper got past, but the cube under it struck the head and was knocked out of its fingers.
        None when no height up to CARRY_RAISES tries works."""
        if self.yaw is None:
            return low
        self.ik.point(self.chain, self.q)
        under = self._carrier_boxes()
        for k in range(CARRY_RAISES + 1):
            z = low + k * CARRY_RAISE
            a, b = np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z])
            clear = all(
                p[2] - load_half_height > top + LOAD_CLEARANCE
                for p in self._way_around(a, b)
                for center, half, top in under
                if np.all(np.abs(p[:2] - center) < half + load_half_height + LOAD_CLEARANCE)
            )
            # The first height's ends are checked with the other key points.
            if clear and self._reaches_around(a, b, ends=k > 0):
                return z
        return None

    def _way_around(self, a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
        """Points along the path a move from `a` to `b` takes around the base (ends left out)."""
        steps = np.linspace(0, 1, CARRY_CHECKS + 2)[1:-1]
        if self.yaw is None:
            return [a + (b - a) * s for s in steps]
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        turn = self._sweep(float(np.arctan2(da[1], da[0])), float(np.arctan2(db[1], db[0])))
        return [self._arc_point(a, b, s, turn) for s in steps]

    def _reaches_around(self, a: np.ndarray, b: np.ndarray, ends: bool = True) -> bool:
        """Whether the tool reaches every point of the way from `a` to `b` around the base, pointing down."""
        points = [*([a, b] if ends else []), *self._way_around(a, b)]
        return all(self.reach_error(p) <= REACH_TOL for p in points)

    def _carrier_boxes(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """The robot under the arm as boxes seen from above: (center xy, half size xy, top z) of each solid shape
        of the robot the arm does not move, as posed in the IK's copy."""
        m, d = self.ik.model, self.ik.data
        boxes = []
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            if body in self._moving or int(m.body_rootid[body]) != self._robot_root or not collides(m, g):
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            center = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
            half = np.abs(rot) @ m.geom_aabb[g, 3:]
            boxes.append((center[:2], half[:2], float(center[2] + half[2])))
        return boxes

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
        self.sync(data)
        t = data.time
        if self._segment is None and self.queue:
            wp = self.queue.popleft()
            start = data.site_xpos[self.site].copy()
            self._branch = None
            self._turn = None
            # Where the tool points now. On a robot that drives or walks too: an arm carried folded, gripper forward,
            # that was told it pointed down already snapped its wrist over on the first tick and the kick shoved the
            # robot 0.25 m.
            self._axis0 = data.xmat[self.chain.body].reshape(3, 3) @ self.tool_axis
            if wp.joints is not None:
                self._branch = (self.q.copy(), wp.joints)
            elif wp.position is not None:
                goal, err, restarted = self._solve(wp.position)
                if err <= REACH_TOL and self.yaw is not None:
                    self._turn = self._path_turn(start, wp.position, float(goal[self.yaw] - self.q[self.yaw]))
                turned = float(self._axis0 @ DOWN) < FAR_FROM_DOWN
                if (restarted or turned) and err <= REACH_TOL:
                    # The goal lies in another elbow or wrist branch, which no straight tool path reaches, or the
                    # tool starts far from pointing down (an arm folded for driving): move joint by joint instead,
                    # as industrial arms do, slowly enough not to fling a load. Turning the wrist over along a
                    # straight path, a folded PiPER still had it sideways on the way down and swept the cube away.
                    self._branch = (self.q.copy(), goal)
                    travel = float(np.max(np.abs(goal - self.q)))
                    wp = Waypoint(wp.position, wp.gripper, max(wp.duration, travel / JOINT_SPEED), wp.label)
            self._segment = (start, wp, t)
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
            else:
                target = self._arc_point(start, wp.position, s)
                axis = (self.tool_axis, _turn_toward(self._axis0, DOWN, s))  # e.g. out of a home pose, gradually
                self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, axis=axis)
        data.ctrl[self.actuators] = self.q
        if u >= 1.0:
            self._segment = None

    def _arc_point(self, a: np.ndarray, b: np.ndarray, s: float, turn: float | None = None) -> np.ndarray:
        """Interpolate in cylindrical coordinates (radius, angle, height) about the base axis, turning `turn`
        (by default the current move's turn, or the way the base joint can go)."""
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        ra, rb = np.linalg.norm(da), np.linalg.norm(db)
        if self.yaw is None or min(ra, rb) < 1e-3:
            return a + (b - a) * s
        ta = np.arctan2(da[1], da[0])
        if turn is None:
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
        lo, hi = self.chain.lower[self.yaw], self.chain.upper[self.yaw]
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
        return self.gripper_open if state == "open" else 0.0

    @staticmethod
    def _time(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.5, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / SPEED)


def _powered(model: mujoco.MjModel) -> dict[int, int]:
    """Joint id -> the actuator that drives it."""
    return {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}


def arm_joints(model: mujoco.MjModel, site: int) -> list[int]:
    """The joints that move the arm whose tool point is `site`, from its first joint to the tool.

    The arm is the powered joints between the robot's root and the tool: that leaves out wheels, legs and a second
    arm of the same robot, and works for arms built from parts too. A lift it rides on counts, so it reaches a table
    by raising itself; a waist or a turntable that two arms share belongs to neither, or each arm would turn the other
    one away from its target."""
    powered = _powered(model)
    shared = _shared_joints(model, site, powered)
    return [j for j in _path_joints(model, int(model.site_bodyid[site]), powered) if j not in shared]


def arm_bodies(model: mujoco.MjModel, site: int) -> set[int]:
    """The bodies of the arm whose tool point is `site`: from the body of its first joint down to the tool's, so not
    the robot that carries it, nor a waist it shares with another arm."""
    joints = arm_joints(model, site)
    if not joints:
        return set()
    first = int(model.jnt_bodyid[joints[0]])
    body, bodies = int(model.site_bodyid[site]), set()
    while body > 0:
        bodies.add(body)
        if body == first:
            break
        body = int(model.body_parentid[body])
    return bodies


def _path_joints(model: mujoco.MjModel, body: int, powered: dict[int, int]) -> list[int]:
    """Powered joints from the root of `body`'s robot down to `body`, in that order: hinges, and slides a
    position actuator drives (a lift; a slide on a speed or force command would not hold where IK puts it)."""
    joints: list[int] = []
    while body > 0:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        own = [j for j in range(start, start + count) if j in powered and _steerable(model, j, powered[j])]
        joints = own + joints
        body = int(model.body_parentid[body])
    return joints


def _steerable(model: mujoco.MjModel, joint: int, actuator: int) -> bool:
    kind = int(model.jnt_type[joint])
    if kind == mujoco.mjtJoint.mjJNT_HINGE:
        return True
    return kind == mujoco.mjtJoint.mjJNT_SLIDE and is_position_servo(model, actuator)


def _shared_joints(model: mujoco.MjModel, site: int, powered: dict[int, int]) -> set[int]:
    """Joints that this tool point's arm and another arm of the same robot both ride on."""
    root = int(model.body_rootid[model.site_bodyid[site]])
    mine = set(_path_joints(model, int(model.site_bodyid[site]), powered))
    shared: set[int] = set()
    for other in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, other) or ""
        if other == site or not (name == "tcp" or name.endswith("/tcp")):
            continue
        if int(model.body_rootid[model.site_bodyid[other]]) == root:
            shared |= mine & set(_path_joints(model, int(model.site_bodyid[other]), powered))
    return shared


def _yaw_index(model: mujoco.MjModel, joints: list[int], home_qpos: np.ndarray) -> int | None:
    """Where in the chain the arm's first hinge is, if it turns about an upright axis at the home pose."""
    hinges = [i for i, j in enumerate(joints) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if not hinges:
        return None
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    first = hinges[0]
    return first if abs(float(data.xaxis[joints[first]][2])) > UPRIGHT else None


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
