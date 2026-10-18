"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down
(leaning a little, high above a grasp or a place, where a short arm cannot point down; see TILTS).
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint, is_position_servo
from robotforge.sim.geometry import collides, highest_point, ray

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
FAR_FROM_DOWN = 0.7  # cosine: a tool pointing more than 45 degrees off where it should moves joint by joint to a goal
CARRY_CHECKS = 7  # points checked along the way across with a load, between the two points above pick and place
CARRY_RAISE = 0.05  # m: each try carries the load this much higher
CARRY_RAISES = 8  # tries before giving up and carrying at the first height (up to 0.4 m higher)
LOAD_CLEARANCE = 0.03  # m kept between a carried load and the robot carrying the arm
# rad a carry may ask the base joint to turn past its limit before the other way round is tried: an arm reaching
# behind its base, where its first joint cannot point (14 degrees each side on a Panda), twists the rest of the way
JOINT_SLACK = 0.35
PATH_SAMPLES = 33  # points a path around the base is measured at: 32 chords of at most 11 degrees in a full turn
FOLLOW_CHECKS = 10  # poses along a move that turns the jaws checked for the arm running into itself or its carrier
REPOSE_STRAY = 0.05  # m the tool may wander while the arm changes pose, joint by joint, with a load in the gripper
# Short arms cannot point the gripper straight down everywhere: a PiPER's wrist (joint 5, +-70 degrees) cannot above
# about 13 cm at 30 cm out, nor an SO-100's above about 9 cm, so neither could lift a cube over an 8 cm bin's rim.
# Where the gripper only passes by, high above a grasp or a place and on the way across between them, it may lean
# this far (rad: 20, 35, 50 degrees) away from the base or toward it. Grasping and placing still point down.
TILTS = (0.35, 0.6, 0.9)
# An arm riding on a robot carries things over that robot's body: a PiPER on a Go2 swept a cube into the dog's head.
# So the load clears the robot's shapes under its way (see _carrier_under): along the tool's way round the base from
# above the grasp to above the place, widened by the object and the jaws around it (CARRY_MARGIN, m), looked at with
# rays every CARRY_STEP (m); and the arm's links keep off the robot all the way (see _follows_clear). Only what lies
# under it counts: the highest shape anywhere on the carrier made a rover with a 25 cm mast behind its arm refuse picks
# in front of it, and a UR5e on ANYmal swept its forearm through the dog's legs where only its tool's way was looked
# at. The load also clears the shapes under the tool all the way (LOAD_CLEARANCE), the arm's own base among them, and
# every robot's: a template arm bolted to the floor carries over its own stand.
CARRY_MARGIN = 0.04
CARRY_STEP = 0.01
CARRY_ARC_STEPS = 8  # points along the way round between two of a carry's checked points, for what lies under it
# Pointing down leaves the turn of the jaws about the tool's axis free, and the IK left it wherever: beside an SO-100
# its jaws came down 35 degrees across a cube, the fixed jaw on the cube's edge, and pushed it away. A box is grasped
# with the jaws closing square to its faces (on the face nearest to where they close now), where the arm can: within
# SQUARE_TOL (rad) of that.
SQUARE_TOL = 0.05
# m the arm keeps between its own links where its jaws turn square to a box: the wrist turns the jaws by bending
# toward the forearm, and lagging its servos a mobile Panda's hand came 1 mm into its forearm, stuck there and closed
# 4 cm beside the cube, from a pose its IK took to be 1 mm clear.
SELF_CLEARANCE = 0.005
SELF_SLACK = 0.0005  # m links already closer than SELF_CLEARANCE where a move begins may close up on the way
# cosine: an arm squares its jaws only with a last joint this close to turning about the tool's axis (a wrist roll,
# as every library arm has). Bending the whole arm to turn them instead, an eight-joint tentacle could not keep up with
# the turn on its way down and its hand came down 5 cm beside the cube.
ROLL = 0.99


class OutOfReach(ValueError):
    """Key points of a pick-and-place the tool cannot reach: `bad` names them ("object", "above place", ...);
    `points` has every key point by name, reached or not, for advice on where to move things. `over` is (the body, its
    top in m) of the robot carrying the arm that the carry has to clear, when only the points high above the object
    and the place are out of reach, at the height that clears it: that height is then why."""

    def __init__(
        self,
        robot: str,
        bad: list[str],
        points: dict[str, np.ndarray] | None = None,
        over: tuple[str, float] | None = None,
    ):
        self.bad = bad
        self.points = points or {}
        self.over = over
        super().__init__(f"{robot} cannot reach the {' and the '.join(bad)} with its gripper pointing down")


class DownReach(NamedTuple):
    """A point against an arm's reach with its gripper pointing down."""

    base: np.ndarray  # the arm's first hinge, raised or lowered toward the point by a lift it rides on
    out: float  # m from the base to the point, across the floor
    reach: float  # m the arm reaches out at the point's height; 0 at a height it cannot get to
    longest: float  # m it reaches out at the height of its base


class _Carry(NamedTuple):
    """How a pick-and-place carries its load across (see ArmMotion._carry_height): the height (None when no way
    works), the way round the base (rad; None for an arm that moves in straight lines or a carry made joint by
    joint), the joint angles a carry made joint by joint goes to, and (the body, its top in m) of the robot carrying
    the arm that every way had to be lifted over, out of the gripper's reach, when that is why none works."""

    z: float | None
    turn: float | None = None
    joints: np.ndarray | None = None
    over: tuple[str, float] | None = None


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
    tilt: bool = False  # the gripper may lean there (see TILTS)
    rotation: np.ndarray | None = None  # the tool body's whole orientation there, jaws turned too (see SQUARE_TOL)
    faces: np.ndarray | None = None  # a box's two side normals: the jaws turn square to them on the way (see _square)


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
        self._own_base = _own_base(model, first, self._robot_root)
        # An arm riding on a robot that moves freely (a dog, a rover) carries things over that robot (see CARRY_MARGIN).
        root_joints = range(
            model.body_jntadr[self._robot_root],
            model.body_jntadr[self._robot_root] + model.body_jntnum[self._robot_root],
        )
        self._rides = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in root_joints)
        # Shapes that already overlap in the home pose (coarse collision shapes around a joint, common in
        # imported models) are part of the model, not a collision to steer around. Worked out once the
        # moving bodies are known: before that it crashed on any overlap and was then thrown away.
        self._touching_at_home: set[tuple[int, int]] = set()
        self._touching_at_home = self._clashes(self.q)
        self._base_parent = int(model.body_parentid[self.base_body])
        self.ik.point(self.chain, self.q)
        self._read_base(self.ik.data)
        self.closing = _closing(model, self.ik.data, self.gripper, self.chain.body, self.tool_axis)
        # Whether the arm's last joint turns the jaws about the tool's axis (a wrist roll), so that squaring them to a
        # box turns that joint rather than bending the arm (see SQUARE_TOL).
        last = self.joints[-1]
        tool = self.ik.data.xmat[self.chain.body].reshape(3, 3) @ self.tool_axis
        self._rolls = bool(
            model.jnt_type[last] == mujoco.mjtJoint.mjJNT_HINGE and abs(float(self.ik.data.xaxis[last] @ tool)) > ROLL
        )
        self._self_pairs: list[tuple[int, int]] | None = None  # see _clear_of_itself, worked out when first needed
        self.queue: deque[Waypoint] = deque()
        self._segment: tuple[np.ndarray, Waypoint, float] | None = None  # (start, waypoint, t0)
        self._branch: tuple[np.ndarray, np.ndarray] | None = None  # (q at the start, q at the goal)
        self._axis0 = DOWN  # where the tool pointed when the current move began
        self._axis1 = DOWN  # where it points at the end of the current move
        self._rot0 = np.eye(3)  # the tool body's orientation when the current move began (see Waypoint.rotation)
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

    def base_and_reach(self, data: mujoco.MjData) -> tuple[np.ndarray, float]:
        """Where the arm's first hinge is, and about how far the arm reaches from it: the spacing of its hinges
        along the chain plus the tool offset, as for an arm stretched out straight. A lift the arm rides on carries
        that hinge up and down rather than adding reach (see down_reach)."""
        points = [*(data.xanchor[j] for j in self.joints if j not in self._slides), data.site_xpos[self.site]]
        reach = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:], strict=False))
        return np.array(points[0], float), reach

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

    def _lean_error(self, target: np.ndarray) -> float:
        """How far the tool stays from `target` where it may lean there (see TILTS)."""
        return self._solve_tilting(target)[1]

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

    def _solve(self, target: np.ndarray, down: np.ndarray = DOWN) -> tuple[np.ndarray, float, bool]:
        """Joint values for the tool at `target` pointing down (or along `down`): (q, position error, needed a
        restart).

        Damped least squares from the current pose can stall on a joint limit or in the wrong elbow
        branch for points the arm reaches easily some other way, so a stuck solve tries again from
        fixed random starts (the base still turned toward the target) and keeps the best answer.
        """
        seed = self._seed(target)
        q, err = self.q, np.inf

        def solve(start: np.ndarray) -> tuple[np.ndarray, float]:
            q2, err2 = self.ik.solve(self.chain, target, start, iterations=60, axis=(self.tool_axis, down))
            return self._unwound(q2), err2

        # From the pose the arm is in, then with its base turned toward the target: the first keeps an arm
        # reaching over its shoulder in that pose, the second handles big turns that stall plain DLS.
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
            if self.yaw is not None:
                start[self.yaw] = seed[self.yaw]
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

    def _clear_of_itself(self, q: np.ndarray, start: np.ndarray) -> bool:
        """Whether the arm at `q` keeps SELF_CLEARANCE between links that a joint of it moves apart, leaving out pairs
        already that close at home. `start` (from _self_gaps) is how far apart they were where the move began: pairs
        closer than SELF_CLEARANCE there may stay so but come no closer (SELF_SLACK). A Panda pointing down over a
        cube with its wrist 3 mm from its forearm turns its jaws square by its wrist alone; held to 5 mm from the
        start, it changed pose instead, twisting its arm to its base joint's limit, carried the cube the long way
        round from there and dropped it."""
        return bool(np.all(self._self_gaps(q) >= np.minimum(start - SELF_SLACK, SELF_CLEARANCE)))

    def _self_gaps(self, q: np.ndarray) -> np.ndarray:
        """How far apart (m, at most SELF_CLEARANCE) the arm's links are at `q`, for each pair _link_pairs gives."""
        if self._self_pairs is None:
            self._self_pairs = self._link_pairs()
        m, d = self.ik.model, self.ik.data
        self.ik.point(self.chain, q)
        fromto = np.zeros(6)
        return np.array([mujoco.mj_geomDistance(m, d, a, b, SELF_CLEARANCE, fromto) for a, b in self._self_pairs])

    def _link_pairs(self) -> list[tuple[int, int]]:
        """Pairs of the arm's solid shapes that its own joints move toward each other: on bodies neither the same nor
        parent and child nor moved by the same last joint of the arm (a hand and its fingers), that can touch, and that
        are farther than SELF_CLEARANCE apart at home."""
        m = self.ik.model
        moved_by = {}  # body -> how many of the arm's joints lie above it
        joint_bodies = [int(m.jnt_bodyid[j]) for j in self.joints]
        for b in self._moving:
            moved_by[b] = sum(_descends_from(m, b, jb) for jb in joint_bodies)
        excluded = {int(sig) for sig in m.exclude_signature}
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self._moving and collides(m, g)]
        pairs = []
        for a, b in itertools.combinations(geoms, 2):
            ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
            if (
                ba == bb
                or int(m.body_parentid[ba]) == bb
                or int(m.body_parentid[bb]) == ba
                or moved_by[ba] == moved_by[bb]
            ):
                continue
            if not (m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]):
                continue
            if (ba << 16) + bb in excluded or (bb << 16) + ba in excluded:  # an <exclude> in the model
                continue
            pairs.append((a, b))
        self.ik.point(self.chain, self.home)
        fromto = np.zeros(6)
        d = self.ik.data
        return [
            (a, b) for a, b in pairs if mujoco.mj_geomDistance(m, d, a, b, SELF_CLEARANCE, fromto) >= SELF_CLEARANCE
        ]

    def _turned_from(self, q: np.ndarray, rotation: np.ndarray) -> float:
        """How far (rad) the tool body at `q` is turned from `rotation`."""
        self.ik.point(self.chain, q)
        have = self.ik.data.xmat[self.chain.body].reshape(3, 3)
        return float(np.arccos(np.clip((np.trace(have.T @ rotation) - 1) / 2, -1.0, 1.0)))

    def _square(self, wp: Waypoint, start: np.ndarray, steps: int) -> np.ndarray | None:
        """The tool body's orientation for the move `wp` down to a box, with the jaws closing square to its faces
        (`wp.faces`: two horizontal unit normals of its sides; see SQUARE_TOL), worked out as the move starts with the
        tool at `start`: of the four ways, the nearest turn from where the jaws close now that the arm follows all the
        way down from the pose it is in, turning only its wrist as far as it can, clear of itself and of what carries
        it. None when no way does (it goes down with the jaws as they are), and for an arm whose jaws' closing is not
        known or that has no wrist roll to turn them (see ROLL). Chosen when the pick was planned, from the pose the
        arm was in then (its home, facing the other way round), a Panda turned its jaws the long way, and a mobile
        Panda's wrist swung its hand into its forearm and closed 4 cm beside the cube."""
        if wp.faces is None or self.closing is None or not self._rolls:
            return None
        now = self._rot0 @ self.closing
        local = np.column_stack([self.tool_axis, self.closing, np.cross(self.tool_axis, self.closing)])
        options = [sign * f for f in wp.faces for sign in (1.0, -1.0)]
        for want in sorted(options, key=lambda f: -float(f[:2] @ now[:2])):  # the nearest turn first
            side = np.array([want[0], want[1], 0.0]) / np.linalg.norm(want[:2])
            rotation = np.column_stack([DOWN, side, np.cross(DOWN, side)]) @ local.T
            end, off = self._follow(self.q, start, wp.position, steps, self._aim(replace(wp, rotation=rotation)))
            if off <= REACH_TOL and self._turned_from(end, rotation) <= SQUARE_TOL:
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
        above = np.array([grasp[0], grasp[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        points = {"object": grasp, "above object": above, "place": drop, "above place": above_drop}
        bad = self._out_of_reach(points)
        if bad:
            raise OutOfReach(self.robot, bad, points)
        across = self._carry_height(grasp, drop, travel_z, obj_half_height)
        if across.z is None and across.over is not None:
            # Every way across passes over a part of the robot carrying the arm, higher than the gripper gets above
            # the object or the place: name it rather than the object and the target.
            high = across.over[1] + 2 * obj_half_height + LOAD_CLEARANCE
            raised = {"above object": [*grasp[:2], high], "above place": [*drop[:2], high]}
            raised = {name: np.array(p, float) for name, p in raised.items()}
            raise OutOfReach(self.robot, self._out_of_reach(raised) or list(raised), {**points, **raised}, across.over)
        if across.z is None:
            under, fix = "the robot carrying the arm", "put both on the same side of the robot, or turn it"
            if int(self.ik.model.body_jntnum[self._robot_root]) == 0:  # bolted to the floor: no robot to turn
                under, fix = "its own base", "move the object or the place"
            raise ValueError(
                f"{self.robot} cannot carry it across: on the way around its base from the object to the place, the "
                f"load would hit {under}, or the gripper cannot point down there, even carried "
                f"{CARRY_RAISES * CARRY_RAISE:.1f} m higher; {fix}"
            )
        above[2] = above_drop[2] = travel_z = across.z
        start = self.tool_target_end()
        # Starting low (an arm folded over the robot that carries it), go up first and across at the carrying height:
        # swinging around at the folded height ran a picker's hand into the table beside it.
        risen = np.array([start[0], start[1], travel_z])
        if start[2] < travel_z - RISE_MIN and self._lean_error(risen) <= REACH_TOL:
            self.queue.append(Waypoint(risen, self._grip("open"), self._time(start, risen), "rise", tilt=True))
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
                    self.queue.append(
                        Waypoint(over, self._grip("open"), self._time(start, over), "approach", tilt=True)
                    )
                    start = over
                    break
        self.queue.append(Waypoint(above, self._grip("open"), self._time(start, above), "approach", tilt=True))
        # With the object in or under the gripper the arm keeps to the paths checked here: straight up and down, and
        # across the way _carry_height went round. Choosing the way across when it started, from where the IK put the
        # base joint, a scorpion's tail went round the other side, right over its body, and flung the cube.
        # On the way down to a box the jaws turn square to it, the way the arm can from where it is then (see _square).
        descend = Waypoint(grasp, None, self._time(above, grasp) + 0.3, "descend", turn=0.0, faces=faces)
        self.queue.append(descend)
        self.queue.append(Waypoint(None, self._grip("close"), 0.6, "grasp"))
        self.queue.append(Waypoint(above, None, self._time(grasp, above) + 0.3, "lift", turn=0.0, tilt=True))
        carry_time = self._time(above, above_drop)
        if across.joints is not None:
            carry_time = max(carry_time, float(np.max(np.abs(across.joints - self._solve(above)[0]))) / JOINT_SPEED)
        self.queue.append(
            Waypoint(
                above_drop,
                None,
                carry_time,
                "carry",
                across.joints,
                turn=across.turn,
                load=obj_half_height,
                tilt=True,
            )
        )
        self.queue.append(Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower", turn=0.0))
        self.queue.append(Waypoint(None, self._grip("open"), 0.5, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat", turn=0.0, tilt=True))

    def _out_of_reach(self, points: dict[str, np.ndarray]) -> list[str]:
        """Names of the key points of a pick-and-place the tool cannot reach (see key_point_error)."""
        return [name for name, p in points.items() if self.key_point_error(name, p) > REACH_TOL]

    def key_point_error(self, name: str, point: np.ndarray) -> float:
        """How far the tool stays from a key point of a pick-and-place, named as in OutOfReach.points: pointing down
        at the object and the place, pointing down or leaning (see TILTS) high above them."""
        return self._lean_error(point) if name.startswith("above") else self.reach_error(point)

    def _carry_height(self, grasp: np.ndarray, drop: np.ndarray, low: float, load_half_height: float) -> _Carry:
        """The lowest height from `low` up at which the tool reaches every point of the way across, around the base
        from above the object to above the place, and the load it carries clears the robot the arm rides on; and the
        way round (rad, None for an arm that moves in straight lines), or the joint angles to carry it to joint by
        joint when no way round works (see _joint_carry). The way the base joint can go from where the IK
        puts it over the object comes first, at any height; then the other way round, if the joint can go that way
        too. Planned from where the tool points instead, a scorpion's tail, which reaches behind it by bending back
        over the top, went round the side it could not turn to. A Go2's PiPER moving a floor cube from its left to
        its right swung it round in front at 10 cm, where its gripper cannot point down without hitting the dog's
        head: the arm stalled, then jumped and flung the cube. At 30 cm the gripper got past, but the cube under it
        struck the head and was knocked out of its fingers. On a robot that moves freely each way starts no lower
        than the part of that robot under the sector it sweeps (see CARRY_MARGIN). No height when no height up to
        CARRY_RAISES tries works."""
        blocked: tuple[str, float] | None = None
        if self.yaw is None:
            z, over = self._clearing(self._straight(grasp, drop, low), low, load_half_height)
            a, b = np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z])
            if z <= low + 1e-9 or (self._lean_error(a) <= REACH_TOL and self._lean_error(b) <= REACH_TOL):
                return _Carry(z)
            return _Carry(None, over=over)
        da, db = grasp[:2] - self.base_xy, drop[:2] - self.base_xy
        step = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))  # signed, -pi..pi
        joint = float(self._solve(np.array([grasp[0], grasp[1], low]))[0][self.yaw])
        planned = self._inside_limits(step, joint)
        other = planned - np.copysign(2 * np.pi, planned)
        ways = (planned, *([other] if self._overrun(other, joint) <= JOINT_SLACK else []))
        lifted = []  # for each way round: whether lifting the load over the carrier put it out of the gripper's reach
        for turn in ways:
            a, b = np.array([grasp[0], grasp[1], low]), np.array([drop[0], drop[1], low])
            start, over = self._clearing([a, *self._way_around(a, b, turn), b], low, load_half_height)
            for k in range(CARRY_RAISES + 1):
                z = start + k * CARRY_RAISE
                a, b = np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z])
                # The first height's ends are checked with the other key points.
                if self._carries_across(a, b, turn, load_half_height, ends=z > low + 1e-9):
                    return _Carry(z, turn)
            ends = (np.array([*grasp[:2], start]), np.array([*drop[:2], start]))
            lifted.append(over is not None and any(self._lean_error(p) > REACH_TOL for p in ends))
            blocked = over if lifted[-1] else blocked
        for k in range(CARRY_RAISES + 1):
            z = low + k * CARRY_RAISE
            joints = self._joint_carry(
                np.array([grasp[0], grasp[1], z]), np.array([drop[0], drop[1], z]), load_half_height
            )
            if joints is not None:
                return _Carry(z, None, joints)
        return _Carry(None, over=blocked if all(lifted) else None)

    def _clearing(
        self, path: list[np.ndarray], low: float, load_half_height: float
    ) -> tuple[float, tuple[str, float] | None]:
        """The lowest height from `low` up at which a load of this half height carried along `path` (tool points in
        order) clears the robot the arm rides on (see _carrier_under), and (the body, its top in m) of the part of
        that robot it had to clear, if any. An arm that rides on nothing starts at `low`."""
        if not self._rides:
            return low, None
        found = self._carrier_under(path, load_half_height, low)
        if found is None:
            return low, None
        need, top, geom = found
        return need, (self.ik.model.body(int(self.ik.model.geom_bodyid[geom])).name, top)

    def _straight(self, a: np.ndarray, b: np.ndarray, z: float) -> list[np.ndarray]:
        """Points along the straight way from above `a` to above `b` at height `z`, ends included."""
        a, b = np.array([a[0], a[1], z]), np.array([b[0], b[1], z])
        return [a + (b - a) * s for s in np.linspace(0, 1, CARRY_CHECKS + 2)]

    def _carrier_under(self, path: list[np.ndarray], half: float, low: float) -> tuple[float, float, int] | None:
        """How high a load `half` high carried along `path` (tool points in order) must go for its bottom to clear by
        LOAD_CLEARANCE the robot carrying the arm (the arm and its own base left out, see _own_base) under it, within
        its half width and CARRY_MARGIN of the way the tool goes about the base, looked at with rays every CARRY_STEP:
        (that height, the top of the part that needs it highest, that part's geom); None when nothing needs it higher
        than `low`. The arm's links are kept off that robot by following the way (see _carries_across): held to clear
        all of the sector between the base and the load by as much as the load, a mobile Panda carried things at 29 cm
        over a 22 cm chassis only its links pass over, and coming down from there beside the chassis, its hand missed
        the cube."""
        m, d = self.ik.model, self.ik.data
        self.ik.point(self.chain, self.q)  # poses every body, the carrier where it is now (see sync)
        points = self._under_load(path, half)
        tops = self._carrier_tops()
        if not tops or not len(points):
            return None
        start_z = max(tops.values()) + 0.01
        best: tuple[float, float, int] | None = None
        for g, top in sorted(tops.items(), key=lambda item: -item[1]):  # highest first: a lower one cannot beat a hit
            if top + 2 * half + LOAD_CLEARANCE <= (low if best is None else best[0]):
                break
            near = points[np.linalg.norm(points - d.geom_xpos[g][:2], axis=1) <= m.geom_rbound[g]]  # rays it can meet
            for xy in near:
                dist = ray(m, d, g, np.array([xy[0], xy[1], start_z]), DOWN)
                need = start_z - dist + 2 * half + LOAD_CLEARANCE
                if dist >= 0 and need > low and (best is None or need > best[0]):
                    best = (need, start_z - dist, g)
                    if start_z - dist >= top - 1e-9:  # its highest point: nothing of it is higher
                        break
        return best

    def _under_load(self, path: list[np.ndarray], half: float) -> np.ndarray:
        """Points (x y) every CARRY_STEP under a load `half` high carried along `path`: within its half width and
        CARRY_MARGIN of the way the tool goes round the base."""
        margin = half + CARRY_MARGIN
        angles, radii = self._polar(path)
        if not len(angles):  # all of it over the base itself
            angles, radii = np.array([0.0, 2 * np.pi]), np.zeros(2)
        track = self.base_xy + radii[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
        rings = []
        for r in np.arange(
            max(float(radii.min()) - margin, 0.0), float(radii.max()) + margin + CARRY_STEP / 2, CARRY_STEP
        ):
            spread = margin / max(r, margin)
            th = np.arange(angles.min() - spread, angles.max() + spread, CARRY_STEP / max(r, CARRY_STEP))
            rings.append(r * np.column_stack([np.cos(th), np.sin(th)]))
        points = self.base_xy + np.concatenate(rings)
        return points[_distance_to_track(points, track) <= margin]

    def _polar(self, path: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """The way the tool goes along `path` about the arm's base: its angles unwound along it (a turn of more than
        half a turn stays one) and its distances from the base, with CARRY_ARC_STEPS between each two of its points;
        points on the base itself left out."""
        polar: list[tuple[float, float]] = []
        for p in path:
            o = np.asarray(p, float)[:2] - self.base_xy
            if np.linalg.norm(o) < 1e-3:
                continue
            t = float(np.arctan2(o[1], o[0]))
            if polar:
                t = polar[-1][0] + (t - polar[-1][0] + np.pi) % (2 * np.pi) - np.pi
            polar.append((t, float(np.linalg.norm(o))))
        if not polar:
            return np.zeros(0), np.zeros(0)
        steps = np.linspace(0, 1, CARRY_ARC_STEPS + 1)[:-1]
        dense = [
            (t0 + (t1 - t0) * s, r0 + (r1 - r0) * s) for (t0, r0), (t1, r1) in itertools.pairwise(polar) for s in steps
        ]
        return tuple(np.array(v) for v in zip(*dense, polar[-1], strict=True))

    def _carrier_tops(self) -> dict[int, float]:
        """The solid shapes of the robot carrying the arm, the arm, its own base (see _own_base) and what is mounted
        on it left out, and the height of each one's highest point (the IK's copy posed)."""
        m, d = self.ik.model, self.ik.data
        mine = f"{self.robot}/"
        tops = {}
        for g in range(m.ngeom):
            body = int(m.geom_bodyid[g])
            if int(m.body_rootid[body]) != self._robot_root or (m.body(body).name or "").startswith(mine):
                continue
            if body in self._moving:  # the arm's own, also a gripper mounted on its flange ("dog/gripper/...")
                continue
            if body in self._own_base:  # what the arm stands on or hangs from: its links turn about it, never over it
                continue
            if collides(m, g):
                tops[g] = float(highest_point(m, d, g)[2])
        return tops

    def _joint_carry(self, a: np.ndarray, b: np.ndarray, load_half_height: float) -> np.ndarray | None:
        """The joint angles over the place for a carry from `a` to `b` made joint by joint, when no way round the base
        keeps the gripper pointing down: the pose nearest the one over the object, if on the way there the arm clears
        itself and the robot it rides on, and the load clears that robot and the floor (checked at twice as many
        points as a way round, as a joint move swings wider, and under the sector it sweeps on a robot that moves
        freely); None otherwise. A scorpion's tail, which reaches behind it by bending back over the top, cannot keep
        its claw pointing down while it turns from one side to the other behind it, but gets there joint by joint."""
        start, err, _, _ = self._solve_tilting(a)
        if err > REACH_TOL:
            return None
        now, self.q = self.q, start  # solved from the pose over the object, to change as little as it can
        try:
            end, err, _, _ = self._solve_tilting(b)
        finally:
            self.q = now
        if err > REACH_TOL:
            return None
        self.ik.point(self.chain, self.q)  # the robot under the arm as it is now
        under = self._carrier_boxes()
        way = []
        for s in np.linspace(0, 1, 2 * CARRY_CHECKS + 2)[1:-1]:
            q = start + (end - start) * s
            if self._collides(q):
                return None
            p = self.ik.point(self.chain, q)
            if p[2] - load_half_height < LOAD_CLEARANCE or not self._load_clears(p, load_half_height, under):
                return None
            way.append(p.copy())
        if self._clearing([a, *way, b], min(p[2] for p in [a, *way, b]), load_half_height)[1] is not None:
            return None
        return end

    def _carries_across(
        self, a: np.ndarray, b: np.ndarray, turn: float, load_half_height: float, ends: bool = True
    ) -> bool:
        """Whether the tool reaches every point of the way from `a` to `b` round the base, turning `turn`, with its
        gripper pointing down (or leaning, see TILTS), and a load of this half height under it clears the robot the
        arm rides on."""
        self.ik.point(self.chain, self.q)  # the robot under the arm as it is now
        under = self._carrier_boxes()
        clear = all(self._load_clears(p, load_half_height, under) for p in self._way_around(a, b, turn))
        return clear and self._reaches_around(a, b, ends, turn) and self._follows_clear(a, b, turn)

    def _follows_clear(self, a: np.ndarray, b: np.ndarray, turn: float) -> bool:
        """Whether an arm riding on a robot keeps off that robot and itself (see _collides) following the way round
        from above the object at `a` to above the place at `b`, turning `turn`, tick by tick from the pose it gets to
        `a` in; always for an arm that rides on nothing. Clearing only what lies under its load, a UR5e on ANYmal
        carrying a cube round in front of the dog at 24 cm struck the dog's body with its wrist and stalled halfway."""
        if not self._rides:
            return True
        start, err, _, axis_a = self._solve_tilting(a)
        if err > REACH_TOL:
            return False
        now, self.q = self.q, start
        try:
            axis_b = self._solve_tilting(b)[3]
        finally:
            self.q = now
        planned, self._turn = self._turn, turn
        try:
            steps = max(1, round(self._path_length(a, b) / SPEED * self.rate_hz))
            aim = lambda s: {"axis": (self.tool_axis, _turn_toward(axis_a, axis_b, s))}  # noqa: E731
            return self._follow(start, a, b, steps, aim, clear=True)[1] < np.inf
        finally:
            self._turn = planned

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
        """Whether the tool reaches every point of the way from `a` to `b` around the base, pointing down where it
        can and leaning where it cannot (see TILTS): these are moves high above a grasp or a place."""
        points = [*([a, b] if ends else []), *self._way_around(a, b, turn)]
        return all(self._lean_error(p) <= REACH_TOL for p in points)

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
            # Where the tool points now, in the world, also on a robot that drives or walks (whose pose the IK follows,
            # see sync): an arm carried folded, gripper forward, that was told it pointed down already snapped its
            # wrist over on the first tick and the kick shoved the robot 0.25 m; a Z1 on a Go2, snapped to pointing
            # down on the first ticks, jolted the dog 25 mm back and it stepped off half a metre.
            self._rot0 = data.xmat[self.chain.body].reshape(3, 3).copy()
            self._axis0 = self._rot0 @ self.tool_axis
            self._axis1 = DOWN
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
                self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, **self._aim(wp)(s))
        data.ctrl[self.actuators] = self.q
        if u >= 1.0:
            self._segment = None
            if self._branch is not None and wp.position is not None:
                self._catch_up(wp)

    def _catch_up(self, wp: Waypoint) -> None:
        """After a joint-by-joint move to a tool position: if the robot carrying the arm moved meanwhile, so that the
        tool is not there, go the rest of the way in a straight line. A Go2 settling under its PiPER sank 2.5 cm while
        the arm unfolded joint by joint to where the IK had put it, and move_arm stopped 2.8 cm short."""
        here = self.ik.point(self.chain, self.q)  # with the carrier where it is now (see sync)
        if float(np.linalg.norm(here - wp.position)) > REACH_TOL:
            rest = Waypoint(wp.position, None, self._time(here, wp.position), wp.label, turn=0.0, tilt=wp.tilt)
            self.queue.appendleft(rest)

    def _aim(self, wp: Waypoint) -> Callable[[float], dict[str, Any]]:
        """How the tool is turned a fraction s of the way along the current move, as IK arguments: the jaws turning
        square to a box on the way down to it, or the tool turning from where it pointed when the move began toward
        where it points at the end (down, or leaning out over a bin; e.g. out of a home pose, gradually)."""
        if wp.rotation is not None:
            rot0, rotation = self._rot0, wp.rotation
            return lambda s: {"rotation": _blend(rot0, rotation, s)}
        axis0, axis1 = self._axis0, self._axis1
        return lambda s: {"axis": (self.tool_axis, _turn_toward(axis0, axis1, s))}

    def _plan_move(self, wp: Waypoint, start: np.ndarray) -> Waypoint:
        """Work out how a move to a tool position goes, as it starts with the tool at `start`: which way round the
        base (self._turn), where the tool points at the end (self._axis1), how long it takes, and whether the arm
        moves joint by joint or changes its pose first. Returns the move to make now."""
        if self.yaw is not None:
            self._turn = self._path_turn(start, wp.position, wp.turn)
            if wp.load is not None:
                self._turn = self._way_across(start, wp, self._turn)
        # Timed by the way it really goes: a carry 310 degrees round a Panda's base, timed as the 0.52 m straight
        # line between its ends, moved the hand at 1.1 m/s and threw the cube.
        wp = replace(wp, duration=max(wp.duration, self._path_length(start, wp.position) / SPEED))
        steps = max(1, round(wp.duration * self.rate_hz))
        if wp.faces is not None:
            wp = replace(wp, faces=None, rotation=self._square(wp, start, steps))
            if wp.rotation is not None:
                return wp  # the arm follows it all the way down from here (see _square)
        if wp.tilt:
            goal, err, restarted, self._axis1 = self._solve_tilting(wp.position)
        else:
            goal, err, restarted = self._solve(wp.position)
        if err > REACH_TOL:
            return wp  # as near as it gets
        aim = self._aim(wp)
        if wp.turn is None:
            free = not restarted and float(self._axis0 @ self._axis1) >= FAR_FROM_DOWN
            if not free or self._follow(self.q, start, wp.position, steps, aim)[1] > REACH_TOL:
                # The goal lies in another elbow or wrist branch, which no straight tool path reaches, the tool
                # starts far from pointing where it should (an arm folded for driving), or the arm cannot follow the
                # path from here (a scorpion's tail reaching behind it flips over its back, and going round with the
                # tail ran into the limit of its base joint and stopped 30 cm short): move joint by joint instead, as
                # industrial arms do, slowly enough not to fling a load. Turning the wrist over along a straight
                # path, a folded PiPER still had it sideways on the way down and swept the cube away.
                return self._joint_move(wp, goal)
            return wp
        # A planned move keeps to the path the plan checked, if the arm can follow it from here or, up in the air,
        # from another pose with the tool where it is (down by the object, changing pose could knock it). Moving joint
        # by joint instead, a tentacle's hand swung in sideways on the way down and pushed the cube 4 cm, and a Panda
        # base swung its load 250 degrees on the way down and threw it.
        rises = float(wp.position[2]) > float(start[2]) + APPROACH / 2  # off the object or the place
        clear = wp.load is not None and self._rides  # a carry keeps off the robot carrying the arm (_follows_clear)
        if rises or wp is self._reposed or self._follow(self.q, start, wp.position, steps, aim, clear)[1] <= REACH_TOL:
            return wp
        if wp.load is not None:
            other = self._other_way_across(start, wp, aim)
            if other is not None:
                return other
        reposed = self._repose(wp, start, goal, steps, aim)
        if reposed is not None:
            return reposed
        # A carry the arm cannot follow round from any pose here goes joint by joint, as it did before it had a
        # planned path: following the path anyway, a scorpion's tail lost it halfway and dropped the cube.
        return self._joint_move(wp, goal) if wp.load is not None else wp

    def _other_way_across(
        self, start: np.ndarray, wp: Waypoint, aim: Callable[[float], dict[str, Any]]
    ) -> Waypoint | None:
        """A carry the arm cannot follow round the way it was planned, from the pose it is in, goes round the other
        way if it can follow that from here and that way passes the plan's checks too (see _carry_height): changing
        pose with the load in the gripper instead, a mobile Panda that had grasped with its jaws turned square swung
        the cube off 60 cm away. None when it cannot."""
        if self.yaw is None or self._turn is None:
            return None
        planned = self._turn
        other = planned - np.copysign(2 * np.pi, planned)
        if self._overrun(other) > JOINT_SLACK:
            return None
        path = [start, *self._way_around(start, wp.position, other), wp.position]
        low = min(float(start[2]), float(wp.position[2]))
        if self._clearing(path, low, wp.load)[1] is not None or not self._carries_across(
            start, wp.position, other, wp.load
        ):
            return None
        self._turn = other
        duration = max(wp.duration, self._path_length(start, wp.position) / SPEED)
        steps = max(1, round(duration * self.rate_hz))
        if self._follow(self.q, start, wp.position, steps, aim, clear=self._rides)[1] > REACH_TOL:
            self._turn = planned
            return None
        return replace(wp, duration=duration, turn=other)

    def _joint_move(self, wp: Waypoint, goal: np.ndarray) -> Waypoint:
        travel = float(np.max(np.abs(goal - self.q)))
        return replace(wp, duration=max(wp.duration, travel / JOINT_SPEED), joints=goal)

    def _follow(
        self,
        q: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        steps: int,
        aim: Callable[[float], dict[str, Any]],
        clear: bool = False,
    ) -> tuple[np.ndarray, float]:
        """The pose the arm ends in when it follows the current move's path from `a` to `b` (turning self._turn)
        from pose `q`, with the tool turned as `aim` says along the way, tick by tick as update() does, and how far
        the tool strays from the path on the way (m); infinitely far when, checked at FOLLOW_CHECKS poses, the arm
        runs into itself or what carries it, with `clear` or where the jaws turn (then also too near itself, see
        _clear_of_itself)."""
        worst = 0.0
        turning = "rotation" in aim(0.0)  # the jaws turn too: the wrist may swing into the arm (see _square)
        gaps = self._self_gaps(q) if turning else None
        every = max(1, steps // FOLLOW_CHECKS) if turning or clear else steps + 1
        for k in range(1, steps + 1):
            u = k / steps
            s = u * u * (3 - 2 * u)
            q, err = self.ik.solve(self.chain, self._arc_point(a, b, s), q, iterations=6, **aim(s))
            worst = max(worst, err)
            checked = k % every == 0 or (k == steps and every <= steps)
            if checked and (self._collides(q) or (turning and not self._clear_of_itself(q, gaps))):
                return q, np.inf
        return q, max(worst, float(np.linalg.norm(self.ik.point(self.chain, q) - b)))

    def _repose(
        self, wp: Waypoint, start: np.ndarray, goal: np.ndarray, steps: int, aim: Callable[[float], dict[str, Any]]
    ) -> Waypoint | None:
        """A change of pose, joint by joint with the tool where it is, into the pose the arm reaches by backing along
        the path from `goal`, so that it follows the path from there (`wp` is put back in the queue to follow it);
        None when that pose does not work. Straying from the way down, a tentacle whose IK found the grasp 74
        degrees of joint travel away ended 8 cm off and closed on nothing, and a Panda with its first joint at its
        limit slid 8 cm aside."""
        forward = self._turn
        self._turn = None if forward is None else -forward
        here, off = self._follow(goal, wp.position, start, steps, lambda s: aim(1.0 - s))
        self._turn = forward
        if off > REACH_TOL or self._collides(here) or self._follow(here, start, wp.position, steps, aim)[1] > REACH_TOL:
            return None
        if wp.load is not None and self._wanders(self.q, here, start):
            return None
        self._reposed = wp
        self.queue.appendleft(wp)
        travel = float(np.max(np.abs(here - self.q)))
        return Waypoint(None, None, max(0.5, travel / JOINT_SPEED), wp.label, joints=here)

    def _wanders(self, q0: np.ndarray, q1: np.ndarray, at: np.ndarray) -> bool:
        """Whether the tool leaves `at` by more than REPOSE_STRAY on the way, joint by joint, from pose `q0` to `q1`:
        two poses with the tool in the same place can lie far apart, and the hand swings out between them."""
        return any(
            float(np.linalg.norm(self.ik.point(self.chain, q0 + (q1 - q0) * s) - at)) > REPOSE_STRAY
            for s in np.linspace(0, 1, FOLLOW_CHECKS + 1)[1:-1]
        )

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
        path = [start, *self._way_around(start, wp.position, own), wp.position]
        low = min(float(start[2]), float(wp.position[2]))
        if self._clearing(path, low, wp.load)[1] is not None:  # that way passes over a part of the carrier
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
        return self.gripper_open if state == "open" else self.gripper_closed

    @staticmethod
    def _time(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.5, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / SPEED)


def _distance_to_track(points: np.ndarray, track: np.ndarray) -> np.ndarray:
    """How far each of `points` (x y) is from the polyline `track` (x y, in order)."""
    if len(track) == 1:
        return np.linalg.norm(points - track[0], axis=1)
    a, ab = track[:-1], np.diff(track, axis=0)
    t = np.einsum("nsk,sk->ns", points[:, None, :] - a[None], ab) / np.maximum(np.einsum("sk,sk->s", ab, ab), 1e-12)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


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


def _own_base(model: mujoco.MjModel, first: int, root: int) -> set[int]:
    """The bodies an arm whose first joint is on body `first` stands on or hangs from that carry nothing else: from
    the parent of `first` up, while each has that one child, short of the robot's main body (a centaur's shoulder
    bracket, a column under a Panda; not the torso both of a centaur's arms hang from). The carry's sweep about the
    base leaves them out: the arm turns about them and never over them, and counting a centaur's shoulder bracket,
    4 cm above its first joint, lifted a carry from 13 to 43 cm and the hand stalled on its way down to the cube."""
    children = np.bincount(model.body_parentid[1:], minlength=model.nbody)
    body, own = int(model.body_parentid[first]), set()
    while body > 0 and body != root and children[body] == 1:
        own.add(body)
        body = int(model.body_parentid[body])
    return own


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False
