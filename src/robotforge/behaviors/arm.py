"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, replace
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
# rad a carry may ask the base joint to turn past its limit before the other way round is tried: an arm reaching
# behind its base, where its first joint cannot point (14 degrees each side on a Panda), twists the rest of the way
JOINT_SLACK = 0.35
PATH_SAMPLES = 33  # points a path around the base is measured at: 32 chords of at most 11 degrees in a full turn


class OutOfReach(ValueError):
    """Key points of a pick-and-place the tool cannot reach: `bad` names them ("object", "above place", ...);
    `points` has every key point by name, reached or not, for advice on where to move things."""

    def __init__(self, robot: str, bad: list[str], points: dict[str, np.ndarray] | None = None):
        self.bad = bad
        self.points = points or {}
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
    # rad about the base the plan checked this move with, and goes round the same way (see _plan_move); None = the way
    # the base joint can go when the move starts
    turn: float | None = None
    load: float | None = None  # m, half the height of the load a carry takes across, kept clear of the robot under it


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
        self._reposed: Waypoint | None = None  # the move the arm last changed its pose for (once is enough)
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

    def footprint(self, model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[float, float, float, float]]:
        """Where the part of the robot that the arm's joints do not move stands, seen from above: (x min, y min, x
        max, y max) in the world for each of its collision shapes, the arm's stand or the robot that carries it.
        A bin suggested 0.1 m from the template arm's base landed on its 0.055 m stand, was pushed off beside the
        cube and caught nothing (review)."""
        boxes = []
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            if int(model.body_rootid[body]) != self._robot_root or body in self._moving:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):  # a picture only: nothing pushes on it
                continue
            center, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
            frame = data.geom_xmat[g].reshape(3, 3)
            signs = itertools.product((-1, 1), repeat=3)
            corners = np.array([data.geom_xpos[g] + frame @ (center + np.multiply(s, half)) for s in signs])
            low, high = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
            boxes.append((float(low[0]), float(low[1]), float(high[0]), float(high[1])))
        return boxes

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
        points = {"object": grasp, "above object": above, "place": drop, "above place": above_drop}
        bad = self.unreachable(points)
        if bad:
            raise OutOfReach(self.robot, bad, points)
        across = self._carry_height(grasp, drop, travel_z, obj_half_height)
        if across is None:
            under, fix = "the robot carrying the arm", "put both on the same side of the robot, or turn it"
            if int(self.ik.model.body_jntnum[self._robot_root]) == 0:  # bolted to the floor: no robot to turn
                under, fix = "its own base", "move the object or the place"
            raise ValueError(
                f"{self.robot} cannot carry it across: on the way around its base from the object to the place, the "
                f"load would hit {under}, or the gripper cannot point down there, even carried "
                f"{CARRY_RAISES * CARRY_RAISE:.1f} m higher; {fix}"
            )
        carry_z, carry_turn, carry_joints = across
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
        # With the object in or under the gripper the arm keeps to the paths checked here: straight up and down, and
        # across the way _carry_height went round. Choosing the way across when it started, from where the IK put the
        # base joint, a scorpion's tail went round the other side, right over its body, and flung the cube.
        self.queue.append(Waypoint(grasp, None, self._time(above, grasp) + 0.3, "descend", turn=0.0))
        self.queue.append(Waypoint(None, self._grip("close"), 0.6, "grasp"))
        self.queue.append(Waypoint(above, None, self._time(grasp, above) + 0.3, "lift", turn=0.0))
        carry_time = self._time(above, above_drop)
        if carry_joints is not None:
            carry_time = max(carry_time, float(np.max(np.abs(carry_joints - self._solve(above)[0]))) / JOINT_SPEED)
        self.queue.append(
            Waypoint(above_drop, None, carry_time, "carry", carry_joints, turn=carry_turn, load=obj_half_height)
        )
        self.queue.append(Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower", turn=0.0))
        self.queue.append(Waypoint(None, self._grip("open"), 0.5, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat", turn=0.0))

    def _carry_height(
        self, grasp: np.ndarray, drop: np.ndarray, low: float, load_half_height: float
    ) -> tuple[float, float | None, np.ndarray | None] | None:
        """The lowest height from `low` up at which the tool reaches every point of the way across, around the base
        from above the object to above the place, and the load it carries clears the robot the arm rides on; and the
        way round (rad, None for an arm that moves in straight lines), or the joint angles to carry it to joint by
        joint when no way round works (see _joint_carry). The way the base joint can go from where the IK
        puts it over the object comes first, at any height; then the other way round, if the joint can go that way
        too. Planned from where the tool points instead, a scorpion's tail, which reaches behind it by bending back
        over the top, went round the side it could not turn to. A Go2's PiPER moving a floor cube from its left to
        its right swung it round in front at 10 cm, where its gripper cannot point down without hitting the dog's
        head: the arm stalled, then jumped and flung the cube. At 30 cm the gripper got past, but the cube under it
        struck the head and was knocked out of its fingers. None when no height up to CARRY_RAISES tries works."""
        if self.yaw is None:
            return low, None, None
        da, db = grasp[:2] - self.base_xy, drop[:2] - self.base_xy
        step = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))  # signed, -pi..pi
        joint = float(self._solve(np.array([grasp[0], grasp[1], low]))[0][self.yaw])
        planned = self._inside_limits(step, joint)
        other = planned - np.copysign(2 * np.pi, planned)
        for turn in (planned, *([other] if self._overrun(other, joint) <= JOINT_SLACK else [])):
            for k in range(CARRY_RAISES + 1):
                z = low + k * CARRY_RAISE
                a, b = np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z])
                # The first height's ends are checked with the other key points.
                if self._carries_across(a, b, turn, load_half_height, ends=k > 0):
                    return z, turn, None
        for k in range(CARRY_RAISES + 1):
            z = low + k * CARRY_RAISE
            joints = self._joint_carry(
                np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z]), load_half_height
            )
            if joints is not None:
                return z, None, joints
        return None

    def _joint_carry(self, a: np.ndarray, b: np.ndarray, load_half_height: float) -> np.ndarray | None:
        """The joint angles over the place for a carry from `a` to `b` made joint by joint, when no way round the base
        keeps the gripper pointing down: the pose nearest the one over the object, if on the way there the arm clears
        itself and the robot it rides on, and the load clears that robot and the floor (checked at twice as many
        points as a way round, as a joint move swings wider); None otherwise. A scorpion's tail, which reaches behind
        it by bending back over the top, cannot keep its claw pointing down while it turns from one side to the other
        behind it, but gets there joint by joint."""
        start, err, _ = self._solve(a)
        if err > REACH_TOL:
            return None
        now, self.q = self.q, start  # solved from the pose over the object, to change as little as it can
        try:
            end, err, _ = self._solve(b)
        finally:
            self.q = now
        if err > REACH_TOL:
            return None
        self.ik.point(self.chain, self.q)  # the robot under the arm as it is now
        under = self._carrier_boxes()
        for s in np.linspace(0, 1, 2 * CARRY_CHECKS + 2)[1:-1]:
            q = start + (end - start) * s
            if self._collides(q):
                return None
            p = self.ik.point(self.chain, q)
            if p[2] - load_half_height < LOAD_CLEARANCE or not self._load_clears(p, load_half_height, under):
                return None
        return end

    def _carries_across(
        self, a: np.ndarray, b: np.ndarray, turn: float, load_half_height: float, ends: bool = True
    ) -> bool:
        """Whether the tool reaches every point of the way from `a` to `b` round the base, turning `turn`, with its
        gripper pointing down, and a load of this half height under it clears the robot the arm rides on."""
        self.ik.point(self.chain, self.q)  # the robot under the arm as it is now
        under = self._carrier_boxes()
        clear = all(self._load_clears(p, load_half_height, under) for p in self._way_around(a, b, turn))
        return clear and self._reaches_around(a, b, ends, turn)

    @staticmethod
    def _load_clears(p: np.ndarray, load_half_height: float, under: list[tuple[np.ndarray, np.ndarray, float]]) -> bool:
        """Whether a load of this half height hanging under the tool at `p` clears the boxes of the robot under it."""
        return all(
            p[2] - load_half_height > top + LOAD_CLEARANCE
            for center, half, top in under
            if np.all(np.abs(p[:2] - center) < half + load_half_height + LOAD_CLEARANCE)
        )

    def _planned_turn(self, a: np.ndarray, b: np.ndarray) -> float | None:
        """How far a move from `a` to `b` turns about the base, the way the base joint can go; None for an arm with
        no upright first hinge, which moves in straight lines."""
        if self.yaw is None:
            return None
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        return self._sweep(float(np.arctan2(da[1], da[0])), float(np.arctan2(db[1], db[0])))

    def _way_around(self, a: np.ndarray, b: np.ndarray, turn: float | None = None) -> list[np.ndarray]:
        """Points along the path a move from `a` to `b` takes around the base, turning `turn` (by default the way
        the plan goes; ends left out)."""
        steps = np.linspace(0, 1, CARRY_CHECKS + 2)[1:-1]
        turn = self._planned_turn(a, b) if turn is None else turn
        return [self._arc_point(a, b, s, turn) for s in steps]

    def _reaches_around(self, a: np.ndarray, b: np.ndarray, ends: bool = True, turn: float | None = None) -> bool:
        """Whether the tool reaches every point of the way from `a` to `b` around the base, pointing down."""
        points = [*([a, b] if ends else []), *self._way_around(a, b, turn)]
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
            if wp.position is not None and wp.joints is None:
                wp = self._plan_move(wp, start)
            if wp.joints is not None:
                self._branch = (self.q.copy(), wp.joints)
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

    def _plan_move(self, wp: Waypoint, start: np.ndarray) -> Waypoint:
        """Work out how a move to a tool position goes, as it starts with the tool at `start`: which way round the
        base (self._turn), how long it takes, and whether the arm moves joint by joint or changes its pose first.
        Returns the move to make now."""
        goal, err, restarted = self._solve(wp.position)
        if self.yaw is not None:
            self._turn = self._path_turn(start, wp.position, wp.turn)
            if wp.load is not None:
                self._turn = self._way_across(start, wp, self._turn)
        # Timed by the way it really goes: a carry 310 degrees round a Panda's base, timed as the 0.52 m straight
        # line between its ends, moved the hand at 1.1 m/s and threw the cube.
        wp = replace(wp, duration=max(wp.duration, self._path_length(start, wp.position) / SPEED))
        if err > REACH_TOL:
            return wp  # as near as it gets
        steps = max(1, round(wp.duration * self.rate_hz))
        if wp.turn is None:
            free = not restarted and float(self._axis0 @ DOWN) >= FAR_FROM_DOWN
            if not free or self._follow(self.q, start, wp.position, steps, self._axis0)[1] > REACH_TOL:
                # The goal lies in another elbow or wrist branch, which no straight tool path reaches, the tool
                # starts far from pointing down (an arm folded for driving), or the arm cannot follow the path from
                # here (a scorpion's tail reaching behind it flips over its back, and going round with the tail ran
                # into the limit of its base joint and stopped 30 cm short): move joint by joint instead, as
                # industrial arms do, slowly enough not to fling a load. Turning the wrist over along a straight
                # path, a folded PiPER still had it sideways on the way down and swept the cube away.
                return self._joint_move(wp, goal)
            return wp
        # A planned move keeps to the path the plan checked, if the arm can follow it from here or, up in the air,
        # from another pose with the tool where it is (down by the object, changing pose could knock it). Moving joint
        # by joint instead, a tentacle's hand swung in sideways on the way down and pushed the cube 4 cm, and a Panda
        # base swung its load 250 degrees on the way down and threw it.
        rises = float(wp.position[2]) > float(start[2]) + APPROACH / 2  # off the object or the place
        if rises or wp is self._reposed or self._follow(self.q, start, wp.position, steps)[1] <= REACH_TOL:
            return wp
        reposed = self._repose(wp, start, goal, steps)
        if reposed is not None:
            return reposed
        # A carry the arm cannot follow round from any pose here goes joint by joint, as it did before it had a
        # planned path: following the path anyway, a scorpion's tail lost it halfway and dropped the cube.
        return self._joint_move(wp, goal) if wp.load is not None else wp

    def _joint_move(self, wp: Waypoint, goal: np.ndarray) -> Waypoint:
        travel = float(np.max(np.abs(goal - self.q)))
        return replace(wp, duration=max(wp.duration, travel / JOINT_SPEED), joints=goal)

    def _follow(
        self, q: np.ndarray, a: np.ndarray, b: np.ndarray, steps: int, axis0: np.ndarray = DOWN
    ) -> tuple[np.ndarray, float]:
        """The pose the arm ends in when it follows the current move's path from `a` to `b` (turning self._turn)
        from pose `q`, with the tool turning from `axis0` to pointing down, tick by tick as update() does, and how far
        the tool strays from the path on the way (m)."""
        worst = 0.0
        for k in range(1, steps + 1):
            u = k / steps
            s = u * u * (3 - 2 * u)
            axis = (self.tool_axis, _turn_toward(axis0, DOWN, s))
            q, err = self.ik.solve(self.chain, self._arc_point(a, b, s), q, iterations=6, axis=axis)
            worst = max(worst, err)
        return q, max(worst, float(np.linalg.norm(self.ik.point(self.chain, q) - b)))

    def _repose(self, wp: Waypoint, start: np.ndarray, goal: np.ndarray, steps: int) -> Waypoint | None:
        """A change of pose, joint by joint with the tool where it is, into the pose the arm reaches by backing along
        the path from `goal`, so that it follows the path from there (`wp` is put back in the queue to follow it);
        None when that pose does not work. Straying from the way down, a tentacle whose IK found the grasp 74
        degrees of joint travel away ended 8 cm off and closed on nothing, and a Panda with its first joint at its
        limit slid 8 cm aside."""
        forward = self._turn
        self._turn = None if forward is None else -forward
        here, off = self._follow(goal, wp.position, start, steps)
        self._turn = forward
        if off > REACH_TOL or self._collides(here) or self._follow(here, start, wp.position, steps)[1] > REACH_TOL:
            return None
        self._reposed = wp
        self.queue.appendleft(wp)
        travel = float(np.max(np.abs(here - self.q)))
        return Waypoint(None, None, max(0.5, travel / JOINT_SPEED), wp.label, joints=here)

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

    def _path_turn(self, a: np.ndarray, b: np.ndarray, planned: float | None = None) -> float:
        """How far the path from `a` to `b` turns about the base: the tool's change in angle, taken around the side
        the plan checked (`planned`), or else the shortest way that would keep the base joint inside its limits from
        where it is now if it took the whole turn. Judging it from tool angles alone mistook a straight lowering near
        the joint's +-180 degree limit for a full circle, which flung the cube; judging it from where the IK put the
        base joint, a Panda whose IK twisted its arm round its third joint went 256 degrees the wrong way round and
        stopped 34 cm short of the cube."""
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        step = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))  # signed, -pi..pi
        if planned is not None:
            return min(_whole_turns_from(step), key=lambda turn: abs(turn - planned))
        return self._inside_limits(step, float(self.q[self.yaw]))

    def _inside_limits(self, step: float, joint: float) -> float:
        """Of the turns that carry the tool `step` round the base, give or take whole turns, the shortest that keeps
        the base joint, starting at `joint`, inside its limits if it takes the whole turn; or, when none does, the one
        past them by the least. A point a little beyond the limit, behind the arm, is reached by twisting the other
        joints the rest of the way; taking the short way round through the back instead, a PiPER found its gripper
        could not point down there and refused a pick that it lands the long way round."""
        return min(_whole_turns_from(step), key=lambda turn: (round(self._overrun(turn, joint), 6), abs(turn)))

    def _way_across(self, start: np.ndarray, wp: Waypoint, planned: float) -> float:
        """Which way round a carry goes: the way it was planned, unless the base joint, from where it really is, would
        turn more than JOINT_SLACK past its limit that way while the other way round passes the same checks. Planned
        from where the tool points, the way across assumes the base joint points there too; a tentacle curled 71
        degrees away from its base joint ran that joint into its limit halfway round and dropped the cube from 23 cm
        up, 20 cm from the place, though the other way round was clear."""
        own = self._path_turn(start, wp.position)
        if abs(own - planned) < 1e-6 or self._overrun(planned) <= JOINT_SLACK:
            return planned
        return own if self._carries_across(start, wp.position, own, wp.load) else planned

    def _overrun(self, turn: float, joint: float | None = None) -> float:
        """Radians past its limits the base joint ends after turning `turn` from `joint` (by default from where it is
        now); 0 when it stays inside."""
        lo, hi = self.chain.lower[self.yaw], self.chain.upper[self.yaw]
        joint = float(self.q[self.yaw]) if joint is None else joint
        return max(lo - joint - turn, joint + turn - hi, 0.0)

    def _path_length(self, a: np.ndarray, b: np.ndarray) -> float:
        """Metres the tool travels from `a` to `b` around the base, the way the current move turns."""
        points = [self._arc_point(a, b, s) for s in np.linspace(0, 1, PATH_SAMPLES)]
        return sum(float(np.linalg.norm(q - p)) for p, q in zip(points, points[1:], strict=False))

    def _sweep(self, from_angle: float, to_angle: float) -> float:
        """Signed turn from one world heading to another that keeps the base joint inside its limits, taking the
        joint to point where the tool does (see _inside_limits)."""
        joint = float(np.clip(from_angle - self.base_yaw, self.chain.lower[self.yaw], self.chain.upper[self.yaw]))
        return self._inside_limits(float((to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi), joint)

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


def _whole_turns_from(step: float) -> list[float]:
    """`step` give or take up to two whole turns: the ways round to the same heading."""
    return [step + k * 2 * np.pi for k in (-2, -1, 0, 1, 2)]


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
