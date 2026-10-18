"""Reaching in with the hand tilted, for an arm on a robot that goes (behaviors/fetch.py).

Pointing its gripper straight down, an arm reaches only a little way past the edge of a counter: at a kitchen
island's 0.93 m a Panda on a mobile base points down at most 0.45 m out from its base and a lift picker 0.3 m, while
the apple lies 0.27 m in from the nearest edge and the arm's base stands well back from it. Tilted out toward what it
works on, the same arms reach 0.75 m and 0.45 m (45 degrees), as people do across a table.

The hand turns about the line across, square to the way from the arm's base to the point (TILTS from straight down),
its fingers closing side to side so that neither dips into the counter. It comes in along its own axis (IN), closes,
lifts the load straight up and draws back to where it points down again, over the near side of the counter (its
stage); putting down goes the other way round. Each of those moves is worked out beforehand as a path of joint
values, the tool point on a straight line and the hand turning evenly between its ends (Waypoint.path), and checked
knot by knot: reached, turned as asked and clear of the robot. A pick or a put that does not work out raises
OutOfReach or ValueError, as ArmMotion's own do.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.behaviors.arm import APPROACH, JOINT_SPEED, REACH_TOL, SPEED, ArmMotion, OutOfReach, Waypoint

TILTS = tuple(math.radians(t) for t in (35.0, 60.0, 90.0))  # rad from straight down, tried in turn
IN = 0.06  # m the hand comes in along its own axis, and backs out after letting go
CLEAR = 0.05  # m the hand crosses above the top of what it works on
ROT_TOL = math.radians(4.0)  # how far off the asked turn of the hand an IK answer may be
KNOT_STEP = 0.015  # m between the knots of a path
KNOT_TURN = math.radians(6.0)  # rad of the hand's turn between knots
KNOT_JUMP = 0.35  # rad: a joint that moves more than this between knots flipped over, and the path is refused
TURN_SPEED = 0.8  # rad/s the hand turns at most along a path
STAGE_STEP = 0.03  # m between the stages tried, from the point in toward the arm's base
STAGE_RISE = 0.2  # m higher than asked that a stage may be, for the arm to clear the counter
CONTACT = 0.001  # m: an arm overlapping something else deeper than this touches it
RESTARTS = 16  # random IK starts for the pose a tilted pick or put goes in from, when the arm's own do not do


class Hand:
    """The tilted hand of an arm: how it turns, what it reaches that way, and the moves of a pick or a put."""

    def __init__(self, arm: ArmMotion, load: int = -1):
        self.arm = arm
        self.across = _closing_line(arm)  # in the tool body's frame; None: not known, and it does not tilt
        self.load = load  # the body of what it picks up or carries: the hand may touch that

    @property
    def tilts(self) -> tuple[float, ...]:
        return TILTS if self.across is not None else ()

    def toward(self, point: np.ndarray) -> np.ndarray:
        """The way from the arm's base to `point`, across the floor (a unit vector)."""
        d = np.asarray(point, float)[:2] - self.arm.base_xy
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return np.array([math.cos(self.arm.base_yaw), math.sin(self.arm.base_yaw)])
        return d / n

    def axis(self, tilt: float, toward: np.ndarray) -> np.ndarray:
        """Where the tool points, tilted `tilt` from straight down out along `toward`."""
        return np.array([math.sin(tilt) * toward[0], math.sin(tilt) * toward[1], -math.cos(tilt)])

    def turns(self, tilt: float, toward: np.ndarray) -> list[np.ndarray]:
        """The tool body's rotations for the tilt, its fingers closing level, either way round."""
        a = self.axis(tilt, toward)
        level = np.array([-toward[1], toward[0], 0.0])
        t = self.arm.tool_axis
        local = np.column_stack([t, self.across, np.cross(t, self.across)])
        return [np.column_stack([a, s * level, np.cross(a, s * level)]) @ local.T for s in (1.0, -1.0)]

    def solve(
        self, point: np.ndarray, turns: list[np.ndarray], starts: list[np.ndarray], restarts: int = 0
    ) -> tuple | None:
        """(joint values, rotation) with the tool at `point` turned one of the ways in `turns`, or None: from
        `starts`, then from `restarts` random settings (the base turned toward the point). Reaching over a counter
        whose top is level with its shoulder, a Panda's IK from its own pose put its elbow into the counter; the
        elbow-up pose that clears it came from a restart."""
        arm = self.arm
        rng = np.random.default_rng(0)
        seed = arm._seed(point)
        for turn in turns:
            for k in range(len(starts) + restarts):
                if k < len(starts):
                    start = starts[k]
                else:
                    start = rng.uniform(arm.chain.lower, arm.chain.upper)
                    if arm.yaw is not None:
                        start[arm.yaw] = seed[arm.yaw]
                q, err = arm.ik.solve(arm.chain, point, start, iterations=60, rotation=turn)
                if err <= REACH_TOL and self._off(q, turn) <= ROT_TOL and self._clear(q):
                    return q, turn
        return None

    def reaches(self, point: np.ndarray, tilt: float, lift: float) -> bool:
        """Whether the arm reaches `point` with the hand tilted, from where it comes in and up `lift` above: a quick
        check for searches (from the pose it is in, the last answer and with its base turned toward the point)."""
        arm = self.arm
        turns = self.turns(tilt, self.toward(point))
        starts = [arm.q, arm._seed(point), *(() if arm._warm is None else (arm._seed(point, arm._warm),))]
        found = self.solve(point, turns, starts)
        if found is None:
            return False
        q, turn = found
        arm._warm = q
        come_in = point - self.axis(tilt, self.toward(point)) * IN
        return all(self.solve(p, [turn], [q]) is not None for p in (point + [0.0, 0.0, lift], come_in))

    # ---- the moves ------------------------------------------------------------------------------------

    def pick(self, grasp: np.ndarray, half: float, hold: np.ndarray, tilt: float) -> None:
        """Queue picking up a load at `grasp` (its middle, `half` its half height) with the hand tilted and carrying
        it round to over `hold` over the robot, as ArmMotion.pick_up does pointing down."""
        arm = self.arm
        grasp = np.asarray(grasp, float)
        toward = self.toward(grasp)
        come_in = grasp - self.axis(tilt, toward) * IN
        lifted = grasp + [0.0, 0.0, APPROACH]
        stage, paths = self._way(tilt, toward, max(grasp[2] + half + CLEAR, come_in[2]), come_in, [grasp, lifted])
        back = paths[-1][2]
        across = arm._carry_height(back, hold, back[2], half)
        if across is None:
            raise ValueError(f"{arm.robot} cannot swing the load round from the object to over its robot")
        carry_z, carry_turn, carry_joints = across
        over = np.array([hold[0], hold[1], carry_z])
        arm._go_over(stage, stage[2])
        self._follow(*paths[0], "reach in")
        self._follow(*paths[1], "descend")
        arm.queue.append(Waypoint(None, arm._grip("close"), 0.6, "grasp"))
        self._follow(*paths[2], "lift", half)
        self._follow(*paths[3], "draw back", half)
        start = back
        if carry_z > back[2] + 1e-6:
            start = np.array([back[0], back[1], carry_z])
            arm.queue.append(Waypoint(start, None, arm._time(back, start), "lift", turn=0.0, load=half))
        arm._carry(start, over, carry_turn, carry_joints, half)  # the errand lowers it to `hold` from there

    def put(self, place: np.ndarray, half: float, place_top: float | None, tilt: float) -> None:
        """Queue putting the load the gripper holds over the robot (picked up with pick_up or pick) down at `place`
        (where its bottom goes) with the hand tilted, as ArmMotion.put_down does pointing down."""
        arm = self.arm
        start = arm.tool_target_end()
        drop = np.asarray(place, float) + [0.0, 0.0, half + 0.01]
        rim = place_top if place_top is not None else float(drop[2] - half)
        above = np.array([drop[0], drop[1], max(drop[2] + APPROACH, rim + 2 * half + 0.03)])
        toward = self.toward(drop)
        backed = drop - self.axis(tilt, toward) * IN
        stage, paths = self._way(tilt, toward, above[2], above, [drop, backed])
        up = np.array([start[0], start[1], max(start[2] + APPROACH, stage[2])])
        across = arm._carry_height(up, stage, up[2], half)
        if across is None:
            raise ValueError(f"{arm.robot} cannot swing the load round from over its robot to the place")
        carry_z, carry_turn, carry_joints = across
        up[2] = carry_z
        over_stage = np.array([stage[0], stage[1], carry_z])
        arm.queue.append(Waypoint(up, None, arm._time(start, up) + 0.3, "lift", turn=0.0, load=half))
        arm._carry(up, over_stage, carry_turn, carry_joints, half)
        if carry_z > stage[2] + 1e-6:
            arm.queue.append(Waypoint(stage, None, arm._time(over_stage, stage), "lower", turn=0.0, load=half))
        self._follow(*paths[0], "reach in", half)
        self._follow(*paths[1], "lower", half)
        arm.queue.append(Waypoint(None, arm._grip("open"), 0.5, "release"))
        self._follow(*paths[2], "back out")
        self._follow(*paths[3], "draw back")

    def _way(
        self, tilt: float, toward: np.ndarray, z: float, first: np.ndarray, then: list[np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]]:
        """The stage and the paths of a tilted pick or put: from the stage to `first` (where the hand comes in from,
        tilted), on through `then`, and back to over the stage pointing down again. [(knots, from, to, how far the
        hand turns)]. The fingers close one way round or, if that does not work out, the other. The stage is found
        from the pose at `first`, and the way in worked out backward from there: from a stage pose found by
        itself, a Panda could not turn its hand in to an apple it reached tilted (the pose was another branch).
        Raises OutOfReach."""
        arm = self.arm
        failed: OutOfReach | None = None
        level = self.turns(0.0, toward)  # pointing down, the fingers closing the same way
        for turn, down in zip(self.turns(tilt, toward), level, strict=True):
            starts = [arm.q, arm._seed(first), *(() if arm._warm is None else (arm._seed(first, arm._warm),))]
            found = self.solve(first, [turn], starts, RESTARTS)
            if found is None:
                failed = OutOfReach(arm.robot, ["tilted"], {"from": first})
                continue
            q_first = found[0]
            try:
                stage, inward = self._stage(toward, z, first, q_first, turn, down)
                out = [(inward, stage, first, _angle(down, turn))]
                q, a = q_first, first
                for b in then:
                    knots = self._path(q, a, turn, b, turn)
                    out.append((knots, a, b, 0.0))
                    q, a = knots[-1], b
                back = np.array([stage[0], stage[1], max(stage[2], a[2])])
                out.append((self._path(q, a, turn, back, down), a, back, _angle(turn, down)))
            except OutOfReach as exc:
                failed = exc
                continue
            arm._warm = q_first
            return stage, out
        raise failed if failed is not None else OutOfReach(arm.robot, ["tilted"])

    # ---- helpers --------------------------------------------------------------------------------------

    def _stage(
        self, toward: np.ndarray, z: float, first: np.ndarray, q_first: np.ndarray, turn: np.ndarray, down: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Where the hand points down before it turns and goes in to `first` (reached tilted at `q_first`), and the
        path there from it: on the way from the arm's base to `first`, at height `z` or up to STAGE_RISE higher, as
        far out toward it as the arm reaches pointing down there, the whole arm clear of the counter (pointing down
        0.69 m out at a desk's height, a Panda's forearm was 4 cm into it). The path is worked out from `first` back
        to the stage and returned the right way round: (the stage, the knots from it to `first`)."""
        arm = self.arm
        out = float(np.linalg.norm(np.asarray(first, float)[:2] - arm.base_xy))
        failed: OutOfReach | None = None
        for dz in np.arange(0.0, STAGE_RISE + 1e-9, 0.05):
            for r in np.arange(out, 0.05, -STAGE_STEP):
                p = np.array([*(arm.base_xy + r * toward), z + dz])
                q, err = arm.ik.solve(arm.chain, p, q_first, iterations=80, rotation=down)
                if err > REACH_TOL or self._off(q, down) > ROT_TOL or not self._clear(q):
                    continue
                try:
                    return p, self._path(q_first, first, turn, p, down)[::-1].copy()
                except OutOfReach as exc:
                    failed = exc
        raise failed or OutOfReach(arm.robot, ["stage"], {"stage": np.array([*(arm.base_xy + 0.1 * toward), z])})

    def _path(self, q0: np.ndarray, a: np.ndarray, turn_a: np.ndarray, b: np.ndarray, turn_b: np.ndarray) -> np.ndarray:
        """The knots (joint values) of a straight move of the tool point from `a` to `b` while the tool body turns
        from `turn_a` to `turn_b`, from joint values `q0` at `a`. Raises OutOfReach when a knot is out of reach, the
        hand cannot turn so there or the arm would hit itself or the robot."""
        arm = self.arm
        qa, qb = _quat(turn_a), _quat(turn_b)
        if qa @ qb < 0:
            qb = -qb
        angle = 2 * math.acos(min(1.0, abs(float(qa @ qb))))
        n = max(2, math.ceil(max(float(np.linalg.norm(b - a)) / KNOT_STEP, angle / KNOT_TURN)))
        q = np.asarray(q0, float)
        knots = [q]  # from where it was worked out (the arm gets there first: ArmMotion.update)
        for k in range(1, n + 1):
            s = k / n
            point = a + (b - a) * s
            turn = _matrix(_slerp(qa, qb, s))
            nxt, err = arm.ik.solve(arm.chain, point, q, iterations=40, rotation=turn)
            jump = float(np.max(np.abs(nxt - q)))
            if err > REACH_TOL or self._off(nxt, turn) > ROT_TOL or jump > KNOT_JUMP or not self._clear(nxt):
                raise OutOfReach(arm.robot, ["path tilted"], {"from": a, "to": b})
            q = nxt
            knots.append(q)
        return np.array(knots)

    def _follow(
        self, knots: np.ndarray, a: np.ndarray, b: np.ndarray, angle: float, label: str, load: float | None = None
    ) -> None:
        """Queue following a path worked out by _path, timed by how far the tool goes and turns and the joints go."""
        travel = float(np.sum(np.max(np.abs(np.diff(knots, axis=0)), axis=1)))
        duration = max(0.5, float(np.linalg.norm(b - a)) / SPEED, angle / TURN_SPEED, travel / JOINT_SPEED)
        self.arm.queue.append(Waypoint(None, None, duration, label, knots[-1], load=load, path=knots))

    def _clear(self, q: np.ndarray) -> bool:
        """Whether the arm at `q` touches nothing: itself, the robot, the floor (ArmMotion._collides), and here also
        furniture and other things, but for the load. Pointing down an arm stays above what it works on; reaching
        in low across a desk, a Panda's forearm lay on it and its hand pushed a can into the cube it went for."""
        arm = self.arm
        if arm._collides(q):
            return False
        m, d = arm.ik.model, arm.ik.data  # posed at q by _collides
        for con in d.contact[: d.ncon]:
            if con.dist > -CONTACT:
                continue
            b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
            if (b1 in arm._moving) == (b2 in arm._moving):
                continue
            other = b2 if b1 in arm._moving else b1
            root = int(m.body_rootid[other])
            if root == arm._robot_root or root == self.load:
                continue
            return False
        return True

    def _off(self, q: np.ndarray, turn: np.ndarray) -> float:
        """How far (rad) the tool body at `q` is turned from `turn`."""
        arm = self.arm
        arm.ik.point(arm.chain, q)
        return _angle(turn, arm.ik.data.xmat[arm.chain.body].reshape(3, 3))


def _closing_line(arm: ArmMotion) -> np.ndarray | None:
    """The line the gripper's fingers close along, in the tool body's frame (square to the tool's axis): the axis of
    its first sliding finger. None for a gripper with no sliding fingers."""
    from robotforge.behaviors.fetch import _finger_joints  # fetch imports this module

    model = arm.ik.model
    if arm.gripper < 0:
        return None
    slides = [j for j in _finger_joints(model, arm.gripper) if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_SLIDE]
    if not slides:
        return None
    arm.ik.point(arm.chain, arm.q)
    d = arm.ik.data
    line = d.xmat[arm.chain.body].reshape(3, 3).T @ d.xaxis[slides[0]]
    line = line - (line @ arm.tool_axis) * arm.tool_axis
    n = float(np.linalg.norm(line))
    return line / n if n > 0.5 else None


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """How far (rad) rotation `b` is turned from rotation `a`."""
    return math.acos(max(-1.0, min(1.0, (float(np.trace(a.T @ b)) - 1.0) / 2.0)))


def _quat(m: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(m, float).ravel())
    return q


def _matrix(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _slerp(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    dot = float(np.clip(a @ b, -1.0, 1.0))
    if dot > 0.9995:
        q = a + (b - a) * s
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - s) * theta) * a + math.sin(s * theta) * b) / math.sin(theta)
