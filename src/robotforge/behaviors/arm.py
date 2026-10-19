"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down
(leaning a little, high above a grasp or a place, where a short arm cannot point down; see TILTS).
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.design import end_effectors
from robotforge.sim.actuators import drives_joint, is_position_servo
from robotforge.sim.geometry import collides, highest_point, ray

DOWN = np.array([0.0, 0.0, -1.0])
SPEED = 0.18  # m/s tool speed
APPROACH = 0.07  # m above a grasp or place point
RISE_MIN = 0.05  # m: a tool this far below the carrying height rises straight up before it moves across
REACH_TOL = 0.008  # m: an IK answer further than this from the target means "cannot reach"
RESTARTS = 24  # extra IK starts, from fixed random joint settings, when the nearest start gets stuck
JOINT_SPEED = 0.6  # rad/s, fastest joint during a move into another elbow or wrist branch
PATH_GAP = 0.02  # rad: an arm further than this from where a path starts moves there first
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
HOLD_STEP = 0.02  # m between the heights tried for holding a load over the robot (hold_pose)
HOLD_GRID = 0.04  # m between the spots over the robot tried for it
HOLD_SPREAD = 0.5  # m of height a held load is raised rather than held a meter further off the robot's middle
HOLD_TRIES = 400  # IK checks hold_pose makes at most
HOLD_MARGIN = 0.03  # m the arm may reach past the robot's outline, holding a load
HOLD_BELOW = 0.1  # m below the arm's first hinge that it holds a load at most
HOLD_LIMIT = math.radians(12.0)  # rad a hinge keeps from its limits holding a load (hold_pose)
HOLD_TRAVEL = 0.05  # m of height a pose for the way counts as worth, per rad its joints move from now to get there
TUCK_STARTS = 6  # other IK starts tried for a tucked pose that stays inside the robot's outline
HOLD_RISE = 0.6  # m above the robot's lowest top that a load is held at most
FOLLOW_CHECKS = 10  # poses along a move that turns the jaws checked for the arm running into itself or its carrier
GRIP_TIME = 0.6  # s jaws take to close on a thing (see ArmMotion.grip_time)
HAND_GRIP_TIME = 1.2
REPOSE_STRAY = 0.05  # m the tool may wander while the arm changes pose, joint by joint, with a load in the gripper
# rad of the largest joint's travel between poses checked along a move joint by joint of an arm riding on a robot
# (see _joints_clear)
JOINT_STEP = 0.05
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
# m an arm riding on a robot keeps its links and gripper from that robot all along a carry (see _follows_clear). Kept
# only out of it, a UR5e on ANYmal swung its Robotiq round the dog's back 1 cm past its rear: the servos lagging, the
# gripper caught on the dog, jammed and dropped the cube 60 cm from the bin.
LINK_CLEARANCE = 0.03
# m it keeps from that robot on slow moves close to it: straight down to an object or a place and back up, where the
# gripper has to come near the robot's legs to get to things beside them, and joint by joint. A UR5e on ANYmal reaching
# the floor beside the dog passes its forearm within 3 cm of the dog's side, and held to LINK_CLEARANCE there it
# refused 6 of 14 floor picks that it made without touching the dog; a PiPER on a Go2, held to it unfolding from its
# home pose, unfolded into another pose and came down on the cube's top instead of round it.
NEAR_CLEARANCE = 0.01
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
# Before it closes its jaws or carries a load away, the arm waits until its tool is this near (m) where it is told to
# be, or stops closing in, or SETTLE_MAX (s) has gone by: lagging behind on the way up, a UR5e on ANYmal began the carry
# 4 cm below where it was planned, ran its forearm into the dog's back and lost the cube.
SETTLE = 0.004
SETTLE_MAX = 1.0
SETTLE_EVERY = 0.05  # s between looks at whether the tool still closes in (by at least SETTLE_GAIN, m)
SETTLE_GAIN = 0.0005
PAD_PRESS = 0.001  # m a suction cup or magnet is pressed into the top of what it picks, so the whole face meets it


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
    # The joint angles the plan checked this move from (see _way_down): an arm that cannot follow it from the pose it
    # is in changes into that one first, where it can (see _repose).
    pose: np.ndarray | None = None
    near: float = LINK_CLEARANCE  # m an arm riding on a robot keeps from it on this move (see NEAR_CLEARANCE)
    # joint values to pass through on the way to `joints`, in order (ending with them): a path worked out beforehand
    # (fetch.py reaching in with the hand tilted), followed knot to knot rather than joint by joint
    path: np.ndarray | None = None


class ArmMotion:
    rate_hz = 100.0

    def __init__(self, model: mujoco.MjModel, robot: str, home_qpos: np.ndarray, gravity: bool = False):
        """`robot` names the arm: a robot ("arm") or an arm mounted on one ("rover/arm"). With `gravity` the arm's
        servos also push the torques gravity needs at the pose they are told and lead by what their damping holds them
        back, as an industrial arm's controller does (see _hold)."""
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
        # A suction cup or an electromagnet (end_effectors.py) holds by adhesion: on (at full power) when closed, off
        # when open, and it holds what it picks by the top, under its face, where fingers hold it around the middle.
        self.tool = end_effectors.tool_of(model, self.gripper)
        # s a grasp takes: a dexterous hand (hands.py), its fingers drawn shut by one tendon over a dozen joints,
        # closes round a thing slower than two jaws do; given the jaws' 0.6 s, a LEAP hand lifted off before its thumb
        # had come round, and the cube stayed on the floor.
        self.grip_time = HAND_GRIP_TIME if _is_hand(model, self.gripper) else GRIP_TIME
        if self.tool is not None:
            self.gripper_open, self.gripper_closed = 0.0, float(grip_range[1])
        # An arm with a tool holds its weight up with its servo targets (servo_targets): it carries what it picks by a
        # face at the end of a long reach. Arms with fingers keep the targets, and the moves, they always had.
        self._kp = servo_stiffness(model, self.actuators) if self.tool is not None else None
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
        self._carrier_pairs: tuple[np.ndarray, np.ndarray] | None = None  # see _near_carrier, the same
        # Per joint, for joints on position servos: radians of servo target per N*m it pushes (1 / (kp * gear)), and
        # seconds of its speed by which its damping holds it back (kv / kp); None for an arm whose servos are only told
        # where to be (see _hold).
        gains = model.actuator_gainprm[self.actuators, 0] * model.actuator_gear[self.actuators, 0]
        servos = np.array([is_position_servo(model, a) for a in self.actuators])
        kp = np.where(servos, gains, 1.0)
        # A model that cancels gravity on the arm's links itself needs no more of it.
        cancelled = any(model.body_gravcomp[b] > 0 for b in self._moving)
        self._holding = (
            np.where(servos & (not cancelled), 1.0 / kp, 0.0),
            np.where(servos, -model.actuator_biasprm[self.actuators, 2] / kp, 0.0),
        )
        self._holds_itself = gravity
        self._per_torque: np.ndarray | None = None
        self._lead = np.zeros(len(self.actuators))
        self.carry_weight = gravity
        self._bias = np.zeros(model.nv)
        self._last: tuple[float, np.ndarray] | None = None  # (time, q) the servos were last told
        self._settle: tuple[float, float, float] | None = None  # (since, last look, how far off then) while waiting
        self.queue: deque[Waypoint] = deque()
        # Called once, when the queue has run out and the arm stands idle (or the controller is replaced): what a pick
        # of something heavy firmed for the load (hints.FirmHold) put back.
        self.on_idle: list[Callable[[], None]] = []
        self._segment: tuple[np.ndarray, Waypoint, float] | None = None  # (start, waypoint, t0)
        self._branch: tuple[np.ndarray, np.ndarray] | None = None  # (q at the start, q at the goal)
        self._knots: np.ndarray | None = None  # the joint values a path move passes through, from where it started
        self._axis0 = DOWN  # where the tool pointed when the current move began
        self._axis1 = DOWN  # where it points at the end of the current move
        self._rot0 = np.eye(3)  # the tool body's orientation when the current move began (see Waypoint.rotation)
        self._turn: float | None = None  # how far the current move turns about the base, rad
        self._reposed: Waypoint | None = None  # the move the arm last changed its pose for (once is enough)
        self.current_label = "idle"
        self.log: list[str] = []
        self._warm: np.ndarray | None = None  # the last answer quick_error found (or warm_up)

    @property
    def carry_weight(self) -> bool:
        """Whether the servos hold the arm up against its weight (see _hold): always for an arm made with `gravity`,
        and while fetch.Errand sets it. Unhelped, a mobile Panda reaching in tilted sagged 9 mm and closed its fingers
        on an apple below its middle, and the apple rolled off the pads. An errand holds the weight only, without
        leading the servos by their damping: led on its way, a mobile Panda's hand jostled an apple 1.3 cm along its
        fingers while folding it in, and the apple fell out as the robot slowed by the dining table."""
        return self._per_torque is not None

    @carry_weight.setter
    def carry_weight(self, on: bool) -> None:
        if self._holds_itself:
            self._per_torque, self._lead = self._holding
        elif on:
            self._per_torque, self._lead = self._holding[0], np.zeros(len(self.actuators))
        else:
            self._per_torque, self._lead = None, np.zeros(len(self.actuators))

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

    def farthest(self, data: mujoco.MjData, point: np.ndarray, lean: bool = False) -> float:
        """How far out (m, across the floor from its first hinge, see down_reach) the tool can get at all at the
        height of `point`, pointing down or, with `lean`, leaning as far as TILTS let it: no key point of a
        pick-and-place further out works there (see key_point_error). Advice on where a refused pick works tried every
        2 cm out to a UR5e's whole 1.46 m length from ANYmal's back, and took 11 s, most of it on points it could not
        get anywhere near."""
        point = np.asarray(point, float)
        anchors = [data.xanchor[j] for j in self.joints if j not in self._slides]
        hand = float(np.linalg.norm(data.site_xpos[self.site] - anchors[-1]))
        farthest = 0.0
        for tilt in (0.0, *(TILTS if lean else ())):
            # Leaning, the wrist is lower above the point and the tool reaches out past it.
            wrist_lower = hand * (1.0 - np.cos(tilt))
            at = self.down_reach(data, point - np.array([0.0, 0.0, wrist_lower]))
            farthest = max(farthest, at.reach + hand * np.sin(tilt))
        return farthest

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

    def pose_for(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        """The arm's joint values with its tool at `target` pointing down, and how far off that leaves it (m)."""
        q, err, _ = self._solve(target)
        return q, err

    def quick_error(self, target: np.ndarray) -> float:
        """reach_error from the two starts it tries first (the pose the arm is in, and with its base turned toward the
        target), and from the last answer that worked, without the restarts from random settings: for searches over
        many points, about 20 times cheaper. It may call a point out of reach that the arm reaches in another elbow
        or wrist branch. The last answer carries a search on: a picker folded low over its base reaches nothing on a
        counter from there, and does from its column raised for the point before."""
        best = np.inf
        starts = [self.q, self._seed(target)]
        if self._warm is not None:
            starts.append(self._seed(target, self._warm))
        for start in starts:
            q, err = self.ik.solve(self.chain, target, start, iterations=60, axis=(self.tool_axis, DOWN))
            q = self._unwound(q)
            if err <= REACH_TOL and not self._collides(q):
                self._warm = q
                return err
            best = min(best, max(err, REACH_TOL * 1.01))
        return best

    def warm_up(self, target: np.ndarray) -> bool:
        """Whether the arm reaches `target` (as reach_error, with the restarts), keeping the answer as the start
        quick_error tries last."""
        q, err, _ = self._solve(target)
        if err > REACH_TOL:
            return False
        self._warm = q
        return True

    @contextlib.contextmanager
    def carried_to(self, qpos: np.ndarray) -> Iterator[ArmMotion]:
        """Plan as though whatever carries the arm stood as in `qpos` (a whole qpos: a robot driven elsewhere, turned
        another way), with the arm as it is now: where would it reach from there? Puts it back after."""
        saved, base = self.ik.base_qpos.copy(), (self.base_xy.copy(), self.base_yaw)
        self.ik.base_qpos[:] = qpos
        try:
            self.ik.point(self.chain, self.q)  # poses every body there
            self._read_base(self.ik.data)
            yield self
        finally:
            self.ik.base_qpos[:] = saved
            self.ik.point(self.chain, self.q)
            self.base_xy, self.base_yaw = base

    def _lean_error(self, target: np.ndarray) -> float:
        """How far the tool stays from `target` where it may lean there (see TILTS). Every way it may point is tried
        first without _solve's restarts: a short arm that cannot point down high up ran through all of them before
        leaning, at every point of the way across, 3.7 s of the 3.9 s a Go2's PiPER took to plan a pick."""
        quick = self._solve_tilting(target, restarts=0)[1]
        return quick if quick <= REACH_TOL else self._solve_tilting(target)[1]

    def _solve_tilting(
        self, target: np.ndarray, restarts: int = RESTARTS, accept: Callable[[np.ndarray], bool] | None = None
    ) -> tuple[np.ndarray, float, bool, np.ndarray]:
        """As _solve, but where the tool cannot point down there it may lean (see TILTS): (q, position error, needed a
        restart, the way the tool points)."""
        q, err, restarted = self._solve(target, restarts=restarts, accept=accept)
        if err <= REACH_TOL:
            return q, err, restarted, DOWN
        out = np.asarray(target, float)[:2] - self.base_xy
        if np.linalg.norm(out) < 1e-3:
            out = np.array([np.cos(self.base_yaw), np.sin(self.base_yaw)])
        out = np.array([*(out / np.linalg.norm(out)), 0.0])
        for angle in TILTS:
            for side in (1.0, -1.0):  # leaning out, as an arm reaching high does, then in
                axis = np.cos(angle) * DOWN + side * np.sin(angle) * out
                q2, err2, restarted2 = self._solve(target, axis, restarts, accept)
                if err2 <= REACH_TOL:
                    return q2, err2, restarted2, axis
        return q, err, restarted, DOWN

    def _solve(
        self,
        target: np.ndarray,
        down: np.ndarray = DOWN,
        restarts: int = RESTARTS,
        accept: Callable[[np.ndarray], bool] | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        """Joint values for the tool at `target` pointing down (or along `down`): (q, position error, needed a
        restart).

        Damped least squares from the current pose can stall on a joint limit or in the wrong elbow
        branch for points the arm reaches easily some other way, so a stuck solve tries again from
        `restarts` fixed random starts (the base still turned toward the target) and keeps the best answer.
        An answer that runs the arm into itself, its carrier or the floor does not work, nor one `accept` turns down.
        """
        seed = self._seed(target)
        q, err = self.q, np.inf
        works = lambda q2: not self._collides(q2) and (accept is None or accept(q2))  # noqa: E731

        def solve(start: np.ndarray) -> tuple[np.ndarray, float]:
            q2, err2 = self.ik.solve(self.chain, target, start, iterations=60, axis=(self.tool_axis, down))
            return self._unwound(q2), err2

        # From the pose the arm is in, then with its base turned toward the target: the first keeps an arm
        # reaching over its shoulder in that pose, the second handles big turns that stall plain DLS. (And from the
        # last answer quick_error found, when a search left one: what it found reachable is found again.)
        for start in (self.q, seed, *(() if self._warm is None else (self._seed(target, self._warm),))):
            q2, err2 = solve(start)
            if err2 <= REACH_TOL and works(q2):
                return q2, err2, False
            if err2 < err:
                q, err = q2, err2
        best: tuple[np.ndarray, float, float] | None = None  # (q, error, largest joint move from now)
        rng = np.random.default_rng(0)
        for _ in range(restarts):
            start = rng.uniform(self.chain.lower, self.chain.upper)
            if self.yaw is not None:
                start[self.yaw] = seed[self.yaw]
            q2, err2 = solve(start)
            if err2 <= REACH_TOL and works(q2):
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

    def _near_carrier(self, q: np.ndarray, margin: float = LINK_CLEARANCE) -> set[tuple[int, int]]:
        """The pairs of shapes, one of the arm's (its gripper's too) and one of the robot carrying it (what the arm
        stands on left out, see _own_base), nearer than `margin` (m, at most LINK_CLEARANCE) with the arm at `q`."""
        m, d = self.ik.model, self.ik.data
        if self._carrier_pairs is None:
            mine = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self._moving and collides(m, g)]
            under = [
                g
                for g in range(m.ngeom)
                if int(m.body_rootid[m.geom_bodyid[g]]) == self._robot_root
                and int(m.geom_bodyid[g]) not in self._moving | self._own_base
                and collides(m, g)
            ]
            pairs = np.array(
                [
                    (a, b)
                    for a in mine
                    for b in under
                    if m.geom_contype[a] & m.geom_conaffinity[b] or m.geom_contype[b] & m.geom_conaffinity[a]
                ],
                int,
            ).reshape(-1, 2)
            self._carrier_pairs = (pairs[:, 0], pairs[:, 1])
        arm, carrier = self._carrier_pairs
        if not len(arm):
            return set()
        self.ik.point(self.chain, q)
        apart = np.linalg.norm(d.geom_xpos[arm] - d.geom_xpos[carrier], axis=1)
        maybe = np.flatnonzero(apart - m.geom_rbound[arm] - m.geom_rbound[carrier] < margin)
        fromto = np.zeros(6)
        return {
            (int(arm[i]), int(carrier[i]))
            for i in maybe
            if mujoco.mj_geomDistance(m, d, int(arm[i]), int(carrier[i]), margin, fromto) < margin
        }

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

    def _seed(self, target: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        """Start IK from `q` (the pose the arm is in) with the base already turned toward the target; DLS alone can
        stall on big turns."""
        q = (self.q if q is None else q).copy()
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
        """Queue a move of the tool to `target`. An arm riding on a robot, setting off from rest, goes over that robot
        where the plain way would run into it (see _way_over)."""
        target = np.asarray(target, float)
        start = from_point if from_point is not None else self.tool_target_end()
        if self._rides and from_point is None and self.done:
            over = self._way_over(start, target)
            if over:
                self.queue.extend([replace(over[0], gripper=self._grip(gripper)), *over[1:]])
                return
        dist = float(np.linalg.norm(target - start)) if start is not None else 0.2
        self.queue.append(Waypoint(target, self._grip(gripper), max(0.5, dist / SPEED), "move"))

    def _way_over(self, start: np.ndarray, target: np.ndarray) -> list[Waypoint]:
        """For an arm riding on a robot: nothing when a plain move from `start` to `target` keeps off that robot (see
        _plain_clear); else up, across and down at the lowest height, CARRY_RAISE steps above both, where it keeps off
        it all the way (see _plain_clear, _follows_clear and _way_down). Raises ValueError when no height up to
        CARRY_RAISES steps works. Swung low round the front from its left to its right, a Z1 on Spot caught its gripper
        on the dog's front leg and stopped 67 cm short."""
        if self._plain_clear(start, target):
            return []
        low = max(float(start[2]), float(target[2]))
        for k in range(1, CARRY_RAISES + 1):
            z = low + k * CARRY_RAISE
            a, b = np.array([start[0], start[1], z]), np.array([target[0], target[1], z])
            if self._lean_error(a) > REACH_TOL or self._lean_error(b) > REACH_TOL:
                continue
            down, pose = self._way_down(b, target)
            if not (down and self._plain_clear(start, a, tilt=True)):
                continue
            turn = self._path_turn(a, b) if self.yaw is not None else 0.0
            if not (self._reaches_around(a, b, False, turn) and self._follows_clear(a, b, turn)):
                continue
            return [
                Waypoint(a, None, self._time(start, a), "rise", tilt=True),
                Waypoint(b, None, self._time(a, b), "move over", turn=turn, tilt=True),
                Waypoint(target, None, self._time(b, target) + 0.3, "move", turn=0.0, pose=pose, near=NEAR_CLEARANCE),
            ]
        raise ValueError(
            f"{self.robot} cannot get to {[round(float(v), 3) for v in target]} without running into the robot "
            f"carrying it, even going {CARRY_RAISES * CARRY_RAISE:.1f} m higher on the way; give a point on the same "
            "side of that robot, or turn the robot"
        )

    def _plain_clear(self, start: np.ndarray, target: np.ndarray, tilt: bool = False) -> bool:
        """Whether a plain move from `start` to `target` (leaning there where it may, with `tilt`), the arm at rest in
        the pose it is in, keeps off itself and LINK_CLEARANCE from the robot carrying it as _plan_move makes it: round
        the base in a line where it can follow that, else joint by joint (see _joint_way). True where the arm cannot
        reach `target` at all: that is for the reach check to say. Checked for touching alone, a PiPER on a Go2 moved
        aside brushed the dog's hip on its way."""
        if tilt:
            goal, err, restarted, axis1 = self._solve_tilting(target)
        else:
            (goal, err, restarted), axis1 = self._solve(target), DOWN
        if err > REACH_TOL:
            return True
        self.ik.point(self.chain, self.q)
        axis0 = self.ik.data.xmat[self.chain.body].reshape(3, 3) @ self.tool_axis
        planned, self._turn = self._turn, (self._path_turn(start, target) if self.yaw is not None else None)
        try:
            steps = max(1, round(max(0.5, self._path_length(start, target) / SPEED) * self.rate_hz))
            aim = lambda s: {"axis": (self.tool_axis, _turn_toward(axis0, axis1, s))}  # noqa: E731
            free = not restarted and float(axis0 @ axis1) >= FAR_FROM_DOWN
            keep = {"clear": True, "apart": LINK_CLEARANCE}
            if free and self._follow(self.q, start, target, steps, aim, **keep)[1] <= REACH_TOL:
                return True
        finally:
            self._turn = planned
        if self._joints_clear(self.q, goal):
            return True
        clear = lambda q: self._joints_clear(self.q, q)  # noqa: E731
        other = self._solve_tilting(target, accept=clear)[1] if tilt else self._solve(target, accept=clear)[1]
        return other <= REACH_TOL

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
        below = obj_half_height  # how far the load reaches below the tool point: fingers hold it by the middle
        if self.tool is not None:  # a suction cup or a magnet holds it by its top, pressed on a little
            grasp[2] += obj_half_height - PAD_PRESS
            below = 2 * obj_half_height
        drop = place + [0, 0, below + 0.01]
        rim = max(place_top if place_top is not None else place[2], grasp[2] - below)
        travel_z = max(grasp[2], drop[2]) + APPROACH
        travel_z = max(travel_z, rim + 2 * obj_half_height + 0.03)
        above = np.array([grasp[0], grasp[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        points = {"object": grasp, "above object": above, "place": drop, "above place": above_drop}
        bad = self._out_of_reach(points)
        if bad:
            raise OutOfReach(self.robot, bad, points)
        across = self._across(grasp, drop, travel_z, below, points, "from the object to the place")
        above[2] = above_drop[2] = travel_z = across.z
        taking = self._taking(grasp, above, above_drop, across, below, faces)
        down, pose = self._way_down(above_drop, drop)
        if not down:
            raise ValueError(self._no_way_down("place", travel_z))
        self.queue.extend(taking)
        self._putting(above_drop, drop, pose)

    def _across(
        self,
        a: np.ndarray,
        b: np.ndarray,
        low: float,
        load_half_height: float,
        points: dict[str, np.ndarray],
        way: str,
    ) -> _Carry:
        """How a load is carried from over `a` to over `b` (see _carry_height), the lowest from `low` up; raises
        OutOfReach or ValueError, saying why, when no way works. `points` are the key points so far, `way` says
        which way it is ("from the object to the place")."""
        across = self._carry_height(a, b, low, load_half_height)
        if across.z is None and across.over is not None:
            # Every way across passes over a part of the robot carrying the arm, higher than the gripper gets above
            # the object or the place: name it rather than the object and the target.
            high = across.over[1] + 2 * load_half_height + LOAD_CLEARANCE
            names = [name for name in points if name.startswith("above")][:2] or ["above object", "above place"]
            raised = {names[0]: [*a[:2], high], names[-1]: [*b[:2], high]}
            raised = {name: np.array(p, float) for name, p in raised.items()}
            raise OutOfReach(self.robot, self._out_of_reach(raised) or list(raised), {**points, **raised}, across.over)
        if across.z is None:
            under, fix = "the robot carrying the arm", "put both on the same side of the robot, or turn it"
            if int(self.ik.model.body_jntnum[self._robot_root]) == 0:  # bolted to the floor: no robot to turn
                under, fix = "its own base", "move the object or the place"
            raise ValueError(
                f"{self.robot} cannot carry it across: on the way around its base {way}, the load would hit {under}, "
                f"or the gripper cannot point down there, even carried {CARRY_RAISES * CARRY_RAISE:.1f} m higher; {fix}"
            )
        return across

    def _no_way_down(self, name: str, travel_z: float) -> str:
        return (
            f"{self.robot} cannot go straight down to the {name} from {travel_z:.2f} m, where it carries things, "
            f"without running into the robot carrying it; put the {name} further from that robot"
        )

    def _taking(
        self,
        grasp: np.ndarray,
        above: np.ndarray,
        to: np.ndarray,
        across: _Carry,
        load_half_height: float,
        faces: np.ndarray | None = None,
    ) -> list[Waypoint]:
        """The moves that take hold of the object at `grasp` and carry it across to `to` the way `across` found: from
        where the tool will be, opening, over to `above` it (at the carrying height), down, closing, back up and
        across. `load_half_height` is how far the load reaches below the tool point. Raises ValueError when the arm
        cannot get straight down to it clear of the robot carrying it."""
        start = self.tool_target_end()
        travel_z = float(above[2])
        lead_in = []  # moves before the approach over the object
        # Starting low (an arm folded over the robot that carries it), go up first and across at the carrying height:
        # swinging around at the folded height ran a picker's hand into the table beside it.
        risen = np.array([start[0], start[1], travel_z])
        if start[2] < travel_z - RISE_MIN and self._lean_error(risen) <= REACH_TOL:
            lead_in.append(Waypoint(risen, self._grip("open"), self._time(start, risen), "rise", tilt=True))
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
                    lead_in.append(Waypoint(over, self._grip("open"), self._time(start, over), "approach", tilt=True))
                    start = over
                    break
        # Setting off from rest straight for the object, the pose it gets there in is known now (see _arrival); and
        # over the object it had better be in a pose it follows the carry from.
        arrive = self._arrival(start, above) if self._rides and self.done and not lead_in else None
        carry = self._carry(above, to, across.turn, across.joints, load_half_height)
        then = None
        if self._rides and across.turn is not None and across.joints is None:
            then = lambda q: self._carries_from(q, above, to, across.turn, carry.duration)  # noqa: E731
        down, pose = self._way_down(above, grasp, arrive, then, self.q if arrive is not None else None)
        if not down:
            raise ValueError(self._no_way_down("object", travel_z))
        # Over the object in another pose than the one it would get there in, that it gets into only from where it
        # is now: it goes there joint by joint (see _plan_move).
        direct = None
        if arrive is not None and pose is not None and pose is not arrive and not self._joints_clear(arrive, pose):
            direct = pose
        # With the object in or under the gripper the arm keeps to the paths checked here: straight up and down, and
        # across the way _carry_height went round. Choosing the way across when it started, from where the IK put the
        # base joint, a scorpion's tail went round the other side, right over its body, and flung the cube.
        # On the way down to a box the jaws turn square to it, the way the arm can from where it is then (see _square).
        return [
            *lead_in,
            Waypoint(above, self._grip("open"), self._time(start, above), "approach", tilt=True, pose=direct),
            Waypoint(
                grasp,
                None,
                self._time(above, grasp) + 0.3,
                "descend",
                turn=0.0,
                faces=faces,
                pose=pose,
                near=NEAR_CLEARANCE,
            ),
            Waypoint(None, self._grip("close"), self.grip_time, "grasp"),
            Waypoint(above, None, self._time(grasp, above) + 0.3, "lift", turn=0.0, tilt=True),
            carry,
        ]

    def _carry(
        self, a: np.ndarray, b: np.ndarray, turn: float | None, joints: np.ndarray | None, load_half_height: float
    ) -> Waypoint:
        """The move carrying a load from `a` to `b` the way _carry_height found: round the base turning `turn`, or
        joint by joint to `joints`."""
        carry_time = self._time(a, b)
        if joints is not None:
            carry_time = max(carry_time, float(np.max(np.abs(joints - self._solve(a)[0]))) / JOINT_SPEED)
        return Waypoint(b, None, carry_time, "carry", joints, turn=turn, load=load_half_height, tilt=True)

    def _putting(self, above_drop: np.ndarray, drop: np.ndarray, pose: np.ndarray | None) -> None:
        """Queue lowering the load from `above_drop` to `drop` (from `pose`, see _way_down), letting go and backing
        up."""
        lower = Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower", turn=0.0, pose=pose)
        self.queue.append(replace(lower, near=NEAR_CLEARANCE))
        self.queue.append(Waypoint(None, self._grip("open"), self.grip_time - 0.1, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat", turn=0.0, tilt=True))

    # ---- picking up and putting down on the move (behaviors/fetch.py) --------------------

    def pick_up(
        self, obj_pos: np.ndarray, obj_half_height: float, hold: np.ndarray, faces: np.ndarray | None = None
    ) -> None:
        """Queue picking an object up and carrying it round to over `hold`: a tool point over the robot that carries
        the arm, where the load clears it and sticks out nowhere, so that the robot can drive or walk with it (the
        errand lowers it there: fetch.Errand). Raises OutOfReach for key points the gripper cannot point down at, and
        ValueError when the load cannot be carried round to `hold`, as pick_and_place does.

        Unlike pick_and_place, the gripper points down high above the object too, not leaning there (see TILTS): a
        robot that goes can stand nearer instead. Leaning over a parcel on a rack 0.5 m out, a lift picker came down
        turning its jaws square on the way, and their fingers landed on the parcel's corners."""
        grasp = np.asarray(obj_pos, float).copy()
        hold = np.asarray(hold, float)
        travel_z = max(grasp[2], hold[2]) + APPROACH
        above = np.array([grasp[0], grasp[1], travel_z])
        over = np.array([hold[0], hold[1], travel_z])
        points = {"object": grasp, "above object": above, "above hold": over, "hold": hold}
        bad = self.unreachable(points)
        if bad:
            raise OutOfReach(self.robot, bad, points)
        across = self._across(grasp, hold, travel_z, obj_half_height, points, "from the object to over its robot")
        above[2] = over[2] = across.z
        self.queue.extend(self._taking(grasp, above, over, across, obj_half_height, faces))

    def put_down(self, place: np.ndarray, obj_half_height: float, place_top: float | None = None) -> None:
        """Queue putting down the load the gripper holds where the tool will be (picked up with pick_up): up, across
        to over the place, down, open and back up. `place` and `place_top` as for pick_and_place, and the same
        errors."""
        start = self.tool_target_end()
        drop = np.asarray(place, float) + [0, 0, obj_half_height + 0.01]
        rim = place_top if place_top is not None else float(drop[2] - obj_half_height)
        travel_z = max(max(start[2], drop[2]) + APPROACH, rim + 2 * obj_half_height + 0.03)
        up = np.array([start[0], start[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        points = {"above hold": up, "place": drop, "above place": above_drop}
        bad = self._out_of_reach(points)
        if bad:
            raise OutOfReach(self.robot, bad, points)
        across = self._across(start, drop, travel_z, obj_half_height, points, "from over its robot to the place")
        up[2] = above_drop[2] = across.z
        down, pose = self._way_down(above_drop, drop)
        if not down:
            raise ValueError(self._no_way_down("place", float(across.z)))
        self.queue.append(
            Waypoint(up, None, self._time(start, up) + 0.3, "lift", turn=0.0, load=obj_half_height, tilt=True)
        )
        self.queue.append(self._carry(up, above_drop, across.turn, across.joints, obj_half_height))
        self._putting(above_drop, drop, pose)

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

    def _way_down(
        self,
        top: np.ndarray,
        bottom: np.ndarray,
        arrive: np.ndarray | None = None,
        then: Callable[[np.ndarray], bool] | None = None,
        rest: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray | None]:
        """Whether an arm riding on a robot gets from `top` straight down to `bottom` (and back up the same way)
        clear of itself and NEAR_CLEARANCE from that robot, as _plan_move moves it, and the pose at `top` it does so
        from (see Waypoint.pose): the one it gets to `top` in (`arrive`, when that is known, else as IK puts it), or
        one it changes into up there, backing up from `bottom` or any other that works, that it gets to clear of the
        robot too (see _joints_clear): from `arrive`, or straight from `rest` (the pose the arm sets off from). Of
        those, one from which the move after the way back up works too (`then`, given the pose) comes first. (True,
        None) for an arm that rides on nothing. Leaning over a cube between a centaur's front legs, its hand came down
        into the legs and the pick missed; come over a cube in another pose than the one checked here, a UR5e on ANYmal
        ran its wrist into the dog's hind thigh on the way down, and a mobile Panda, finding no pose to change into,
        stopped short of its base above the cube; changed into one, it lifted the cube in a pose it could not follow
        the carry from, carried it joint by joint and let it slip 3 cm in its jaws."""
        if not self._rides:
            return True, None
        steps = max(1, round((self._time(top, bottom) + 0.3) * self.rate_hz))  # as long as the descend and the lower
        planned, self._turn = self._turn, 0.0
        try:
            high, err, _, tilt = self._solve_tilting(top)
            if err > REACH_TOL:
                return False, None
            down = lambda s: {"axis": (self.tool_axis, _turn_toward(tilt, DOWN, s))}  # noqa: E731
            up = lambda s: {"axis": (self.tool_axis, _turn_toward(DOWN, tilt, s))}  # noqa: E731

            def gets_there(q: np.ndarray) -> bool:
                if arrive is None or q is arrive or self._joints_clear(arrive, q):
                    return True
                return rest is not None and self._joints_clear(rest, q)

            def goes_down(q: np.ndarray) -> bool:
                if not gets_there(q):
                    return False
                return self._follow(q, top, bottom, steps, down, clear=True, apart=NEAR_CLEARANCE)[1] <= REACH_TOL

            def backing_up() -> np.ndarray | None:
                low, err, _ = self._solve(bottom)
                if err > REACH_TOL:
                    return None
                q, off = self._follow(low, bottom, top, steps, up, clear=True, apart=NEAR_CLEARANCE)
                return q if off <= REACH_TOL and gets_there(q) else None

            near = [q for q in ((high,) if arrive is None else (arrive, high)) if goes_down(q)]
            if not near:
                near = [q for q in (backing_up(),) if q is not None]
            for q in near:
                if then is None or then(q):
                    return True, q
            # Another pose over it, as the UR5e's elbow and wrist can turn either way: from 0.15 m over a cube 0.67 m
            # in front of ANYmal the two above both ran its forearm along the dog's front.
            if then is not None:
                found, err, _ = self._solve(top, tilt, accept=lambda q: goes_down(q) and then(q))
                if err <= REACH_TOL:
                    return True, found
            if near:
                return True, near[0]
            found, err, _ = self._solve(top, tilt, accept=goes_down)
            return (True, found) if err <= REACH_TOL else (False, None)
        finally:
            self._turn = planned

    def _carries_from(self, q: np.ndarray, a: np.ndarray, b: np.ndarray, turn: float, duration: float) -> bool:
        """Whether an arm riding on a robot, over the object at `a` in pose `q`, follows the carry round to `b`
        (turning `turn`, over `duration`) as _plan_move checks it when the carry sets off: keeping LINK_CLEARANCE from
        that robot, clear of itself, and on its path."""
        self.ik.point(self.chain, q)
        axis_a = self.ik.data.xmat[self.chain.body].reshape(3, 3) @ self.tool_axis
        now, self.q = self.q, q
        planned, self._turn = self._turn, turn
        try:
            axis_b = self._solve_tilting(b)[3]
            steps = max(1, round(max(duration, self._path_length(a, b) / SPEED) * self.rate_hz))
            aim = lambda s: {"axis": (self.tool_axis, _turn_toward(axis_a, axis_b, s))}  # noqa: E731
            return self._follow(q, a, b, steps, aim, clear=True, apart=LINK_CLEARANCE)[1] <= REACH_TOL
        finally:
            self.q, self._turn = now, planned

    def _arrival(self, start: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        """The pose an arm at rest gets to `target` in, moved there from `start` as _plan_move moves it: following a
        line, or joint by joint (see _joint_way); None where it does not get there. The way down from there is
        checked from that pose (see _way_down)."""
        saved = (self._turn, self._axis0, self._axis1, self._rot0)
        try:
            self.ik.point(self.chain, self.q)
            self._rot0 = self.ik.data.xmat[self.chain.body].reshape(3, 3).copy()
            self._axis0, self._axis1, self._turn = self._rot0 @ self.tool_axis, DOWN, None
            wp = self._plan_move(Waypoint(target, None, self._time(start, target), "approach", tilt=True), start)
            if wp.joints is not None:
                return wp.joints
            steps = max(1, round(wp.duration * self.rate_hz))
            keep = {"clear": self._rides, "apart": LINK_CLEARANCE if self._rides else 0.0}
            end, off = self._follow(self.q, start, target, steps, self._aim(wp), **keep)
            return end if off <= REACH_TOL else None
        finally:
            self._turn, self._axis0, self._axis1, self._rot0 = saved

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
            return self._follow(start, a, b, steps, aim, clear=True, apart=LINK_CLEARANCE)[1] < np.inf
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

    def hold_pose(self, load_half_height: float, load_radius: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Where the tool holds a load (or nothing) while the robot carrying the arm drives or walks, and the arm's
        joint values there: over the robot, the load and the arm inside its outline seen from above, so nothing
        sticks out to catch on furniture, and the load clear of it by LOAD_CLEARANCE; low (a load held high swings)
        and near the middle between the arm's base and the robot's, HOLD_SPREAD m of height counting as much as a
        meter off that middle ahead or behind, or half a meter to a side (where it tips the robot and sticks out), and
        HOLD_TRAVEL per rad the arm's joints move to get there.
        None for an arm that cannot fold over its robot that way. A world point, for the robot as it stands now.

        Every spot over the robot is tried, nearest that middle first: a Panda cannot hold anything low next to its
        own base, and does behind it, over the back of its deck."""
        d = self.ik.data
        self.ik.point(self.chain, self.q)  # the robot under the arm as it is now
        root = np.array(d.xpos[self._robot_root], float)
        rot = d.xmat[self._robot_root].reshape(3, 3)
        heading = float(np.arctan2(rot[1, 0], rot[0, 0]))
        c, s = np.cos(heading), np.sin(heading)
        frame = np.array([[c, s], [-s, c]])  # world xy -> the robot's heading frame
        # Worked out in the robot's heading frame, each shape's box turned with it: boxes square to the world grew
        # with the robot's heading (a Panda's deck 40 percent at 45 degrees), and the robot found another tuck or
        # none, standing at an angle.
        under = self._carrier_boxes(frame, root[:2])
        if not under:
            return None
        corners = [
            center + np.array(sign) * half for center, half, _ in under for sign in itertools.product((-1, 1), repeat=2)
        ]
        outline = (frame, root[:2], np.min(corners, axis=0) - HOLD_MARGIN, np.max(corners, axis=0) + HOLD_MARGIN)
        low, high = np.min(corners, axis=0) + load_radius, np.max(corners, axis=0) - load_radius
        if np.any(low > high):
            return None
        base = frame @ (self.base_xy - root[:2])
        aim = np.clip(base / 2, low, high)  # between the arm's base and the robot's middle
        ceiling = min(top for _, _, top in under) + HOLD_RISE
        # Not far below the arm's shoulder: folded down behind its own base to 0.24 m under its shoulder, a Panda went
        # in and out of that pose only by swinging round, and swung an apple out of its fingers.
        hinge = next((j for j in self.joints if j not in self._slides), self.joints[0])
        shoulder = float(d.xanchor[hinge][2]) - HOLD_BELOW
        # Best first: each spot by the lowest score it could have, the load just clear of what is under it there.
        spots = []
        for x in np.arange(low[0], high[0] + 1e-9, HOLD_GRID):
            for y in np.arange(low[1], high[1] + 1e-9, HOLD_GRID):
                local = np.array([x, y])
                xy = root[:2] + np.array([c * x - s * y, s * x + c * y])
                floor = max(
                    (top for center, half, top in under if np.all(np.abs(local - center) < half + load_radius)),
                    default=-np.inf,
                )
                lowest = max(floor + LOAD_CLEARANCE + load_half_height + 0.005, ceiling - HOLD_RISE, shoulder)
                spread = HOLD_SPREAD * float(np.hypot(x - aim[0], 2 * (y - aim[1])))  # off to a side: twice
                spots.append((lowest + spread, lowest, spread, xy, local))
        spots.sort(key=lambda spot: spot[0])
        # The pose the arm folds into by itself from the one it is in comes first: the moves there keep to it (a
        # Panda folded into another pose found by a restart went back to its own on the way, elbow out). Others, from
        # restarts, only if no spot is held that way.
        for restarts in (0, TUCK_STARTS):
            best: tuple[float, np.ndarray, np.ndarray] | None = None  # (score, point, joint values)
            tries = 0
            for bound, lowest, spread, xy, local in spots:
                if best is not None and bound >= best[0] or tries >= HOLD_TRIES:
                    break  # the spots left can only score worse
                for z in np.arange(lowest, ceiling, HOLD_STEP):
                    if best is not None and z + spread >= best[0]:
                        break
                    p = np.array([*xy, z])
                    if not self._load_clears(np.array([*local, z]), load_half_height, under):
                        continue
                    tries += 1
                    if self.quick_error(p) > REACH_TOL:
                        continue
                    q = self._tucked(p, load_half_height, load_radius, outline, restarts)
                    if q is None:
                        continue
                    travel = HOLD_TRAVEL * float(np.max(np.abs(q - self.q)))  # a lift picker swung 340 degrees
                    if best is None or z + spread + travel < best[0]:
                        best = (float(z) + spread + travel, p, q)
                    break
            if best is not None:
                return best[1], best[2]
        return None

    def _tucked(
        self, p: np.ndarray, half_height: float, radius: float, outline: tuple[np.ndarray, ...], restarts: int
    ) -> np.ndarray | None:
        """Joint values with the tool at `p` pointing down that keep a load held there clear of the arm and the arm
        inside the robot's outline (heading frame, its middle, low and high corners): the answer quick_error found,
        else one from `restarts` other starts. Folded back over its deck from its ready pose, a Panda stuck its elbow
        0.42 m out to the side, and the robot no longer fitted between a kitchen island and the counter."""
        rng = np.random.default_rng(1)
        seed = self._seed(p)
        for k in range(restarts + 1):
            if k == 0:
                q = self._warm
            else:
                start = rng.uniform(self.chain.lower, self.chain.upper)
                if self.yaw is not None:
                    start[self.yaw] = seed[self.yaw]
                q, err = self.ik.solve(self.chain, p, start, iterations=60, axis=(self.tool_axis, DOWN))
                if err > REACH_TOL or self._collides(q):
                    continue
            if (
                not self._load_hits_arm(q, p, half_height, radius, outline[:2])
                and self._inside(q, outline)
                and self._roomy(q)
            ):
                return q
        return None

    def _roomy(self, q: np.ndarray) -> bool:
        """Whether every hinge of the arm at `q` is HOLD_LIMIT or more from its limits. Folded with its elbow and its
        base joint 2 degrees from their stops, a Panda could not lower a load into that pose from above and swung
        round to get there."""
        hinges = np.array([j not in self._slides for j in self.joints])
        lower, upper = self.chain.lower, self.chain.upper
        return bool(np.all(((q - lower > HOLD_LIMIT) & (upper - q > HOLD_LIMIT)) | ~hinges))

    def _inside(self, q: np.ndarray, outline: tuple[np.ndarray, ...]) -> bool:
        """Whether the arm at `q`, seen from above, stays inside the outline (heading frame, middle, low, high)."""
        m = self.ik.model
        frame, middle, low, high = outline
        self.ik.point(self.chain, q)
        boxes = _boxes_in(self.ik, lambda g: int(m.geom_bodyid[g]) in self._moving, frame, middle)
        return all(np.all(center - half >= low) and np.all(center + half <= high) for center, half, _ in boxes)

    def _load_hits_arm(
        self, q: np.ndarray, p: np.ndarray, half_height: float, radius: float, frame: tuple | None = None
    ) -> bool:
        """Whether a load held at `p` with the arm at `q` overlaps the arm's own links (other than the hand holding it),
        seen as boxes: folded back over its robot, an arm can bend round its own load. In a robot's heading frame
        (`frame`: (world xy to it, its middle)) if given, as hold_pose works."""
        m = self.ik.model
        self.ik.point(self.chain, q)
        hand = {b for b in self._moving if _descends_from(m, b, self.chain.body)}
        if frame is not None:
            turn, middle = frame
            p = np.array([*(turn @ (p[:2] - middle)), p[2]])
        low, high = p - [radius, radius, half_height], p + [radius, radius, half_height]
        links = _boxes_in(
            self.ik,
            lambda g: int(m.geom_bodyid[g]) in self._moving and int(m.geom_bodyid[g]) not in hand,
            *(frame if frame is not None else (None, None)),
            tops=False,
        )
        return any(np.all(center + half > low) and np.all(center - half < high) for center, half in links)

    def _carrier_boxes(
        self, frame: np.ndarray | None = None, middle: np.ndarray | None = None
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """The robot under the arm as boxes seen from above: (center xy, half size xy, top z) of each solid shape
        of the robot the arm does not move, as posed in the IK's copy; in a robot's heading frame (`frame`: world xy
        to it, about `middle`) if given, each box turned with its shape there."""
        m = self.ik.model
        return _boxes_in(
            self.ik,
            lambda g: (
                int(m.geom_bodyid[g]) not in self._moving and int(m.body_rootid[m.geom_bodyid[g]]) == self._robot_root
            ),
            frame,
            middle,
        )

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

    def finish(self) -> None:
        """Run what was to be done when the arm is idle, once."""
        callbacks, self.on_idle = self.on_idle, []
        for callback in callbacks:
            callback()

    def release(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """The controller is replaced (see SimSession.set_controller): put back what it changed."""
        self.finish()

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
        if self._segment is None and self.queue and self._settling(data):
            self._command(data)
            return
        if self._segment is None and self.queue:
            wp = self.queue.popleft()
            start = data.site_xpos[self.site].copy()
            self._branch = None
            self._knots = None
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
            if wp.path is not None and float(np.max(np.abs(wp.path[0] - self.q))) > PATH_GAP:
                # A path starts from the pose it was worked out from: get there first, joint by joint.
                self.queue.appendleft(wp)
                travel = float(np.max(np.abs(wp.path[0] - self.q)))
                wp = Waypoint(None, None, max(0.3, travel / JOINT_SPEED), wp.label, wp.path[0], load=wp.load)
            if wp.path is not None:
                self._knots = np.vstack([self.q, wp.path])
            elif wp.joints is not None:
                self._branch = (self.q.copy(), wp.joints)
            self._segment = (start, wp, t)
            self.current_label = wp.label
            if wp.gripper is not None and self.gripper >= 0:
                data.ctrl[self.gripper] = wp.gripper
        if self._segment is None:
            self.current_label = "idle"
            self.finish()
            self._command(data)
            return
        start, wp, t0 = self._segment
        u = min(1.0, (t - t0) / wp.duration)
        if wp.position is not None or wp.joints is not None:
            s = u * u * (3 - 2 * u)  # smoothstep
            if self._knots is not None:
                k = min(s * (len(self._knots) - 1), len(self._knots) - 1 - 1e-9)
                i = int(k)
                self.q = self._knots[i] + (self._knots[i + 1] - self._knots[i]) * (k - i)
            elif self._branch is not None:
                q0, q1 = self._branch
                if wp.position is not None and self._rides:  # see _tracked
                    q1 = self._tracked(wp.position, q1)
                    self._branch = (q0, q1)
                self.q = q0 + (q1 - q0) * s
            else:
                target = self._arc_point(start, wp.position, s)
                self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, **self._aim(wp)(s))
        self._command(data)
        if u >= 1.0:
            self._segment = None
            if self._branch is not None and wp.position is not None:
                self._catch_up(wp)

    def _tracked(self, position: np.ndarray, q1: np.ndarray) -> np.ndarray:
        """The goal of a move joint by joint to a tool position kept on it as the robot carrying the arm moves (see
        sync). Worked out once as it set off, the goal of a UR5e on ANYmal unfolding from its home pose was 6 cm too
        low by the time it got there, the dog having sunk and leant under the arm, and the hand caught the cube on its
        way up from there."""
        q, _ = self.ik.solve(self.chain, position, q1, iterations=3, axis=(self.tool_axis, self._axis1))
        return q

    def _command(self, data: mujoco.MjData) -> None:
        """Tell the arm's servos to be at self.q, moving as fast as it changes (see _hold)."""
        t, qdot = float(data.time), np.zeros_like(self.q)
        if self._last is not None and t > self._last[0]:
            qdot = (self.q - self._last[1]) / (t - self._last[0])
        self._last = (t, self.q.copy())
        if self._kp is not None and self._per_torque is None:  # a cup or a magnet: gravity's share (see __init__)
            data.ctrl[self.actuators] = servo_targets(data, self.chain.dof_adr, self._kp, self.q)
            return
        data.ctrl[self.actuators] = self._hold(self.q, qdot)

    def _hold(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Servo targets for the arm at `q`, moving at `qdot`: `q` itself, and for an arm that holds itself up (see
        __init__) each position servo's target moved on by as much as it takes to push the torque gravity needs at `q`
        (with whatever carries the arm posed as it is now, see sync) and to make up for its damping at that speed.
        That is how industrial arms' controllers hold their pose, and what some Menagerie models leave out: a UR5e's
        shoulder (2000 N*m/rad) sagged 17 mm at 0.73 m out, and riding on ANYmal it closed its jaws 2 cm beside the
        cube and pushed it away; its damping (0.2 s of its speed) kept it 4 cm behind on its way down to a place, and
        its forearm caught on the dog's hind thigh. Pushed through the motors, not cancelled on the links, the arm's
        weight still rests on what carries it."""
        if self._per_torque is None:
            return q
        self.ik.point(self.chain, q)
        self.ik.data.qvel[:] = 0.0  # gravity alone
        mujoco.mj_rne(self.ik.model, self.ik.data, 0, self._bias)
        return q + self._bias[self.chain.dof_adr] * self._per_torque + self._lead * qdot

    def _settling(self, data: mujoco.MjData) -> bool:
        """Whether the arm waits before the next move: one that closes its jaws, carries a load away or goes the way
        the plan checked from a pose (see Waypoint.pose), while the tool is still more than SETTLE from where it is
        told to be and closing in, for up to SETTLE_MAX (see SETTLE). Setting off down 7 mm from where the plan had
        checked the way down, a mobile Panda's IK took another way round its seven joints, and its fifth link hit the
        base."""
        wp, t = self.queue[0], float(data.time)
        closes = wp.position is None and wp.gripper is not None and wp.gripper == self.gripper_closed
        if not (closes or wp.load is not None or wp.pose is not None):
            self._settle = None
            return False
        off = float(np.linalg.norm(self.ik.point(self.chain, self.q) - data.site_xpos[self.site]))
        if self._settle is None:
            self._settle = (t, t, off)
        since, looked, then = self._settle
        if off <= SETTLE or t - since >= SETTLE_MAX:
            self._settle = None
            return False
        if t - looked >= SETTLE_EVERY:
            if then - off < SETTLE_GAIN:  # not closing in any more: waiting longer does not help
                self._settle = None
                return False
            self._settle = (since, t, off)
        return True

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
        reached = err <= REACH_TOL  # else, for a planned move, from here no pose there runs clear of the robot
        if not reached and wp.turn is None:
            return wp  # as near as it gets
        aim = self._aim(wp)
        keep = {"clear": self._rides, "apart": wp.near if self._rides else 0.0}  # on a robot, keep off it
        if wp.turn is None and wp.pose is not None and self._joints_clear(self.q, wp.pose):
            return self._joint_move(wp, wp.pose)  # the pose the plan checked the next move from (see pick_and_place)
        if wp.turn is None:
            free = not restarted and float(self._axis0 @ self._axis1) >= FAR_FROM_DOWN
            if not free or self._follow(self.q, start, wp.position, steps, aim, **keep)[1] > REACH_TOL:
                # The goal lies in another elbow or wrist branch, which no straight tool path reaches, the tool
                # starts far from pointing where it should (an arm folded for driving), or the arm cannot follow the
                # path from here (a scorpion's tail reaching behind it flips over its back, and going round with the
                # tail ran into the limit of its base joint and stopped 30 cm short): move joint by joint instead, as
                # industrial arms do, slowly enough not to fling a load. Turning the wrist over along a straight
                # path, a folded PiPER still had it sideways on the way down and swept the cube away.
                return self._joint_way(wp, goal)
            return wp
        # A planned move keeps to the path the plan checked, if the arm can follow it from here or, up in the air,
        # from another pose with the tool where it is (down by the object, changing pose could knock it). Moving joint
        # by joint instead, a tentacle's hand swung in sideways on the way down and pushed the cube 4 cm, and a Panda
        # base swung its load 250 degrees on the way down and threw it.
        # On a robot the arm keeps LINK_CLEARANCE from it: going down beside a mobile Panda's wheel from high above,
        # its hand swung round into the wheel and missed the cube, and a UR5e on ANYmal, come over the cube in another
        # pose than the plan checked the way down from, ran its wrist into the dog's hind thigh on the way down.
        # Straight up off the object or the place is never checked here; a carry always is, as it may start lower than
        # planned: lagging on the lift, a UR5e on ANYmal set off across 4 cm low, unchecked, and hit the dog's back.
        rises = wp.load is None and float(wp.position[2]) > float(start[2]) + APPROACH / 2
        if rises or wp is self._reposed or self._follow(self.q, start, wp.position, steps, aim, **keep)[1] <= REACH_TOL:
            return wp
        if wp.load is not None:
            other = self._other_way_across(start, wp, aim)
            if other is not None:
                return other
        reposed = self._repose(wp, start, goal if reached else None, steps, aim, **keep)
        if reposed is not None:
            return reposed
        # A carry the arm cannot follow round from any pose here goes joint by joint, as it did before it had a
        # planned path: following the path anyway, a scorpion's tail lost it halfway and dropped the cube.
        if wp.load is not None and reached:
            # Into the pose the way down to the place was checked from, where that has the tool over the place: ended
            # in another, a scorpion's tail changed pose there with the cube in its claw, swung for 6 s and dropped it
            # from 27 cm.
            then = self.queue[0].pose if self.queue else None
            off = np.inf if then is None else float(np.linalg.norm(self.ik.point(self.chain, then) - wp.position))
            if off <= REACH_TOL and self._joints_clear(self.q, then):
                return self._joint_move(wp, then)
            return self._joint_way(wp, goal)
        return self._short_of_carrier(wp, start, steps, aim, **keep) if self._rides else wp

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
        keep = {"clear": self._rides, "apart": LINK_CLEARANCE if self._rides else 0.0}
        if self._follow(self.q, start, wp.position, steps, aim, **keep)[1] > REACH_TOL:
            self._turn = planned
            return None
        return replace(wp, duration=duration, turn=other)

    def _joint_way(self, wp: Waypoint, goal: np.ndarray) -> Waypoint:
        """The move `wp` made joint by joint to `goal`; for an arm riding on a robot that would come nearer that robot
        on the way than NEAR_CLEARANCE (see _joints_clear), to the nearest other pose with the tool at the same place
        (leaning, where it may) that keeps clear, if there is one."""
        if self._joints_clear(self.q, goal):
            return self._joint_move(wp, goal)
        clear = lambda q: self._joints_clear(self.q, q)  # noqa: E731
        if wp.tilt:
            other, err, _, axis = self._solve_tilting(wp.position, accept=clear)
        else:
            (other, err, _), axis = self._solve(wp.position, accept=clear), DOWN
        if err > REACH_TOL:
            return self._joint_move(wp, goal)  # nothing better
        self._axis1 = axis
        return self._joint_move(wp, other)

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
        apart: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """The pose the arm ends in when it follows the current move's path from `a` to `b` (turning self._turn)
        from pose `q`, with the tool turned as `aim` says along the way, tick by tick as update() does, and how far
        the tool strays from the path on the way (m); infinitely far when, checked at FOLLOW_CHECKS poses, the arm
        runs into itself or what carries it, with `clear` or where the jaws turn (then also too near itself, see
        _clear_of_itself), or comes nearer to what carries it than `apart` (m; 0 does not look)."""
        q, off, _ = self._follow_until(q, a, b, steps, aim, clear, apart)
        return q, off

    def _follow_until(
        self,
        q: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        steps: int,
        aim: Callable[[float], dict[str, Any]],
        clear: bool = False,
        apart: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """As _follow, and how far along the path (0 to 1) the last of the checked poses that passed lies."""
        worst, passed = 0.0, 0.0
        turning = "rotation" in aim(0.0)  # the jaws turn too: the wrist may swing into the arm (see _square)
        gaps = self._self_gaps(q) if turning else None
        near = self._near_carrier(q, apart) if apart else set()  # already that near where it starts: may stay
        every = max(1, steps // FOLLOW_CHECKS) if turning or clear or apart else steps + 1
        for k in range(1, steps + 1):
            u = k / steps
            s = u * u * (3 - 2 * u)
            q, err = self.ik.solve(self.chain, self._arc_point(a, b, s), q, iterations=6, **aim(s))
            worst = max(worst, err)
            if not (k % every == 0 or (k == steps and every <= steps)):
                continue
            if self._collides(q) or (turning and not self._clear_of_itself(q, gaps)):
                return q, np.inf, passed
            if apart and self._near_carrier(q, apart) - near:
                return q, np.inf, passed
            passed = s
        return q, max(worst, float(np.linalg.norm(self.ik.point(self.chain, q) - b))), 1.0

    def _short_of_carrier(
        self, wp: Waypoint, start: np.ndarray, steps: int, aim: Callable[[float], dict[str, Any]], **keep: Any
    ) -> Waypoint:
        """The move `wp` of an arm riding on a robot, that no pose here follows clear of that robot, ending before it
        would run into it: the robot has shifted under the arm since the plan checked it. Lowering a cube behind ANYmal,
        whose hind thigh had moved 1.3 cm into the way checked when the pick was planned, a UR5e went on down anyway,
        caught its forearm on the thigh and let go of the cube 7 cm from the place. Only running into it counts here,
        not coming nearer than the move's clearance: stopped that soon, a UR5e closed its jaws above a cube beside the
        dog's front leg that it reached with its wrist 5 mm from the leg."""
        _, off, passed = self._follow_until(self.q, start, wp.position, steps, aim, clear=True)
        if off < np.inf:
            return wp
        return replace(wp, position=self._arc_point(start, wp.position, passed), pose=None)

    def _repose(
        self,
        wp: Waypoint,
        start: np.ndarray,
        goal: np.ndarray | None,
        steps: int,
        aim: Callable[[float], dict[str, Any]],
        **keep: Any,
    ) -> Waypoint | None:
        """A change of pose, joint by joint with the tool where it is, so that the arm follows the path from there
        (`wp` is put back in the queue to follow it), clear of what `keep` says (see _follow): into the pose the plan
        checked the move from (wp.pose), or else the one the arm reaches by backing along the path from `goal`, or else,
        for an arm riding on a robot, the nearest other pose with the tool where it is that works; None when none
        does. Straying from the way down, a tentacle whose IK found the grasp 74 degrees of joint travel away ended 8
        cm off and closed on nothing, and a Panda with its first joint at its limit slid 8 cm aside. Leaning under a
        UR5e reaching out in front of it, ANYmal had moved its front leg into the way down the plan had checked, and
        the arm's wrist caught on the leg."""

        def works(here: np.ndarray) -> bool:
            # From where the tool is in that pose: the plan's pose, with the carrier shifted a little since, had its
            # tool 7.5 mm from a mobile Panda's, too far to follow from the first step.
            if self._follow(here, self.ik.point(self.chain, here), wp.position, steps, aim, **keep)[1] > REACH_TOL:
                return False
            if wp.load is not None and self._wanders(self.q, here, start):
                return False
            return self._joints_clear(self.q, here)  # changing pose, on a robot, keeps clear of it too

        here = next((q for q in self._repose_poses(wp, start, goal, steps, aim) if works(q)), None)
        if here is None and self._rides and "axis" in aim(0.0):
            found, err, _ = self._solve(start, aim(0.0)["axis"][1], accept=works)
            here = found if err <= REACH_TOL else None
        if here is None:
            return None
        self._reposed = wp
        self.queue.appendleft(wp)
        travel = float(np.max(np.abs(here - self.q)))
        return Waypoint(None, None, max(0.5, travel / JOINT_SPEED), wp.label, joints=here)

    def _repose_poses(
        self,
        wp: Waypoint,
        start: np.ndarray,
        goal: np.ndarray | None,
        steps: int,
        aim: Callable[[float], dict[str, Any]],
    ) -> list[np.ndarray]:
        """The poses _repose tries, with the tool at `start` and clear of the robot carrying the arm and the floor."""
        poses = []
        if wp.pose is not None and float(np.linalg.norm(self.ik.point(self.chain, wp.pose) - start)) <= REPOSE_STRAY:
            poses.append(wp.pose)
        if goal is not None:
            forward = self._turn
            self._turn = None if forward is None else -forward
            here, off = self._follow(goal, wp.position, start, steps, lambda s: aim(1.0 - s))
            self._turn = forward
            if off <= REACH_TOL:
                poses.append(here)
        return [q for q in poses if not self._collides(q)]

    def _joints_clear(self, q0: np.ndarray, q1: np.ndarray, margin: float = NEAR_CLEARANCE) -> bool:
        """Whether an arm riding on a robot, moving joint by joint from pose `q0` to `q1`, keeps off that robot, itself
        and the floor (see _collides) and `margin` from that robot (see _near_carrier; parts already that near at
        `q0` may stay so), checked every JOINT_STEP of the largest joint's travel; always for an arm that rides on
        nothing. Unchecked, a UR5e on ANYmal unfolding from its home pose brushed the dog's hind thigh with its wrist,
        and its gripper the dog's back."""
        if not self._rides:
            return True
        near = self._near_carrier(q0, margin)
        checks = max(FOLLOW_CHECKS, int(np.ceil(float(np.max(np.abs(q1 - q0))) / JOINT_STEP)))
        for s in np.linspace(0, 1, checks + 1)[1:]:
            q = q0 + (q1 - q0) * s
            if self._collides(q) or self._near_carrier(q, margin) - near:
                return False
        return True

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


def servo_stiffness(model: mujoco.MjModel, actuators: np.ndarray) -> np.ndarray:
    """Each actuator's position-servo stiffness (0 for one that is not a position servo), for servo_targets."""
    return np.array([float(model.actuator_gainprm[a, 0]) if is_position_servo(model, a) else 0.0 for a in actuators])


def servo_damping(model: mujoco.MjModel, actuators: np.ndarray) -> np.ndarray:
    """Each position servo's damping (its kv, force per joint speed; 0 for one that is not a position servo)."""
    return np.array([-float(model.actuator_biasprm[a, 2]) if is_position_servo(model, a) else 0.0 for a in actuators])


def servo_targets(
    data: mujoco.MjData,
    dofs: np.ndarray,
    kp: np.ndarray,
    q: np.ndarray,
    kv: np.ndarray | None = None,
    qdot: np.ndarray | None = None,
) -> np.ndarray:
    """Servo targets that hold joints at `q`: each aimed past its value by what holds the arm's weight up (gravity's
    share of the bias force over the stiffness). A position servo pushes only in proportion to how far it is kept from
    its target, and these have no integral term the way a real servo's controller has: aimed at the values
    themselves, the template arm hung 1.6 cm low over a switch and 5.6 cm low carrying a plate at full stretch.

    Less what MuJoCo's own gravity compensation already gives (bodies with gravcomp, as a PiPER's): counted twice,
    a PiPER held still stood 3.7 cm too high and pressed no button.

    Given the servos' damping `kv` and the speed `qdot` the targets move at, also aimed past by what that damping takes
    from a joint moving at that speed: a servo damps against the joint's speed, not the error's, so a moving target is
    otherwise followed behind. A PiPER (kv 5 to kp 80, its wrist 1.5 to 10) swinging in to a drawer ran 3 cm behind
    its plan and put a finger on the drawer's top; with it, 6 mm."""
    bias = data.qfrc_bias[dofs] - data.qfrc_gravcomp[dofs]
    if kv is not None and qdot is not None:
        bias = bias + kv * qdot
    return q + np.divide(bias, kp, out=np.zeros_like(bias), where=kp > 0)


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


def _is_hand(model: mujoco.MjModel, gripper: int) -> bool:
    """Whether the gripper is a hand's: a tendon over more than two joints (a Panda's two fingers share one too)."""
    if gripper < 0 or int(model.actuator_trntype[gripper]) != int(mujoco.mjtTrn.mjTRN_TENDON):
        return False
    return int(model.tendon_num[int(model.actuator_trnid[gripper, 0])]) > 2


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


def _boxes_in(
    ik: IKSolver,
    keep: Callable[[int], bool],
    frame: np.ndarray | None = None,
    middle: np.ndarray | None = None,
    tops: bool = True,
) -> list[tuple]:
    """The bounding boxes of the solid shapes `keep` takes, as posed in the IK's copy: in the world, or in a robot's
    heading frame (`frame`: world xy to it, about `middle`), each box turned with its shape there. Seen from above,
    (center xy, half size xy, top z); with `tops` false, (center, half size) in 3D."""
    m, d = ik.model, ik.data
    turn = np.eye(3)
    if frame is not None:
        turn[:2, :2] = frame
    origin = np.zeros(2) if middle is None else np.asarray(middle, float)
    out = []
    for g in range(m.ngeom):
        if not collides(m, g) or not keep(g):
            continue
        rot = turn @ d.geom_xmat[g].reshape(3, 3)
        center = turn @ np.array([*(d.geom_xpos[g][:2] - origin), d.geom_xpos[g][2]]) + rot @ m.geom_aabb[g, :3]
        half = np.abs(rot) @ m.geom_aabb[g, 3:]
        out.append((center[:2], half[:2], float(center[2] + half[2])) if tops else (center, half))
    return out
