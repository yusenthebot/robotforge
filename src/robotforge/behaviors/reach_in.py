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
LOADED_TILT = math.radians(45.0)  # how far off pointing down a gripper with a load may turn on a joint move
RISE = 0.1  # m a hand rises first when a joint move straight from where it is would hit something
HOVER = (0.15, 0.1, 0.06)  # m over where it comes in from that a tilted hand gets to joint by joint (Hand._hover)
VIA_UP = (0.1, 0.25)  # m higher than both ends that a joint move may raise the tool on the way (Hand._vias)
QUICK_RESTARTS = 4  # random IK starts in the quick check of a tilted reach (reaches)
SWEEP_STEP = 0.08  # rad of the furthest joint between the poses a joint move is checked at
LOAD_GAP = 0.01  # m a load moved joint by joint keeps above what is under it


class Hand:
    """The tilted hand of an arm: how it turns, what it reaches that way, and the moves of a pick or a put."""

    def __init__(self, arm: ArmMotion, load: int = -1, faces: tuple | None = None):
        self.arm = arm
        self.across = _closing_line(arm)  # in the tool body's frame; None: not known, and it does not tilt
        self.load = load  # the body of what it picks up or carries: the hand may touch that
        # For a box: (its x axis across the floor, its half sizes along x and y, how wide the gripper opens). The
        # hand comes at it square to a face, its fingers closing across a width they fit round: turned freely, a
        # lift picker's fingers met a 5 cm parcel corner to corner (7.1 cm across, the gripper opens 7.2) and stopped
        # on its top.
        self.faces = faces

    @property
    def tilts(self) -> tuple[float, ...]:
        return TILTS if self.across is not None else ()

    def toward(self, point: np.ndarray) -> np.ndarray:
        """The way the hand comes at `point` across the floor (a unit vector): from the arm's base, or for a box
        (`faces`) square to the face of it nearest that way across which the fingers fit."""
        d = np.asarray(point, float)[:2] - self.arm.base_xy
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-6 else np.array([math.cos(self.arm.base_yaw), math.sin(self.arm.base_yaw)])
        if self.faces is None:
            return d
        ax, half_x, half_y, opening = self.faces
        ay = np.array([-ax[1], ax[0]])
        ways = [v for v, across in ((ax, half_y), (-ax, half_y), (ay, half_x), (-ay, half_x)) if 2 * across < opening]
        return max(ways, key=lambda v: float(v @ d)) if ways else d

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
        found = self.solve(point, turns, starts, QUICK_RESTARTS)
        if found is None:
            return False
        q, turn = found
        arm._warm = q
        come_in = point - self.axis(tilt, self.toward(point)) * IN
        return all(self.solve(p, [turn], [q]) is not None for p in (point + [0.0, 0.0, lift], come_in))

    # ---- the moves ------------------------------------------------------------------------------------

    def pick(
        self,
        grasp: np.ndarray,
        half: float,
        hold: np.ndarray,
        tilt: float,
        hold_q: np.ndarray | None = None,
        radius: float = 0.0,
    ) -> None:
        """Queue picking up a load at `grasp` (its middle, `half` its half height, `radius` round) with the hand tilted
        and carrying it round to over `hold` over the robot, or into the arm's pose `hold_q` there when that is given
        and the way works out, as ArmMotion.pick_up does pointing down."""
        arm = self.arm
        grasp = np.asarray(grasp, float)
        toward = self.toward(grasp)
        come_in = grasp - self.axis(tilt, toward) * IN
        lifted = grasp + [0.0, 0.0, APPROACH]
        if hold_q is not None:
            moves = self._direct(tilt, toward, come_in, [grasp, lifted], hold_q, (half, radius), hold)
            if moves is not None:
                arm.queue.extend(moves)
                return
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

    def _direct(
        self,
        tilt: float,
        toward: np.ndarray,
        first: np.ndarray,
        then: list[np.ndarray],
        hold_q: np.ndarray,
        load: tuple[float, float],
        hold: np.ndarray,
    ) -> list[Waypoint] | None:
        """The moves of a pick that gets to the tilted pose over where the hand comes in (`first`) joint by joint
        from the pose the arm is in, tucked over the robot (straight, or rising first, or by way of another pose:
        _joint_way), comes down to `first` and goes on along the paths through `then`; and back with the load: its
        hand turning down about the line its fingers close along (a joint move turning it every which way let an
        apple roll off a Panda's pads) to where it points down, and from there into `hold_q` joint by joint within
        LOADED_TILT of down, or carried round over `hold` the way the tool points. None when no such way works out.
        Round by a stage pointing down (_way), a Panda took 38 s for one pick; this way 10."""
        arm = self.arm
        half = load[0]
        level = self.turns(0.0, toward)
        for turn, down_turn in zip(self.turns(tilt, toward), level, strict=True):
            starts = [arm.q, arm._seed(first), *(() if arm._warm is None else (arm._seed(first, arm._warm),))]
            found = self.solve(first, [turn], starts, RESTARTS)
            if found is None:
                continue
            q, a, paths = found[0], first, []
            try:
                hover, down = self._hover(found[0], first, turn)
                paths.append((down, hover, first, 0.0))
                for b in then:
                    knots = self._path(q, a, turn, b, turn)
                    paths.append((knots, a, b, 0.0))
                    q, a = knots[-1], b
                z = max(a[2], first[2] + CLEAR)
                back, inward = self._stage(toward, z, a, q, turn, down_turn)
            except OutOfReach:
                continue
            going = self._joint_way(arm.q, down[0], None, "approach")
            if going is None:
                continue
            drawn = inward[::-1].copy()  # from the lifted load back to where the hand points down
            coming = self._joint_way(drawn[-1], hold_q, load, "carry", LOADED_TILT)
            if coming is None:
                across = arm._carry_height(back, hold, back[2], half)
                if across is None:
                    continue
                carry_z, carry_turn, carry_joints = across
                coming = []
                start = back
                if carry_z > back[2] + 1e-6:
                    start = np.array([back[0], back[1], carry_z])
                    coming.append(Waypoint(start, None, arm._time(back, start), "lift", turn=0.0, load=half))
                over = np.array([hold[0], hold[1], carry_z])
                time = arm._time(start, over)
                if carry_joints is not None:
                    coming.append(Waypoint(over, None, time, "carry", carry_joints, turn=carry_turn, load=half))
                else:
                    coming.append(Waypoint(over, None, time, "carry", turn=carry_turn, load=half))
            return [
                Waypoint(None, arm._grip("open"), 0.3, "approach"),
                *going,
                self._follow_move(*paths[0], "reach in"),
                self._follow_move(*paths[1], "descend"),
                Waypoint(None, arm._grip("close"), 0.6, "grasp"),
                self._follow_move(*paths[2], "lift", half),
                self._follow_move(drawn, a, back, _angle(turn, down_turn), "draw back", half),
                *coming,
            ]
        return None

    def _hover(self, q_first: np.ndarray, first: np.ndarray, turn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where over `first` (reached at `q_first`) the hand gets to joint by joint, and the path down from there:
        as high as the arm reaches, up to the first of HOVER. A lift picker reaching in level over a kitchen island
        is at the end of its reach, and keeps its hand level only 6 cm higher. Raises OutOfReach."""
        failed: OutOfReach | None = None
        for up in HOVER:
            hover = first + [0.0, 0.0, up]
            try:
                return hover, self._path(q_first, first, turn, hover, turn)[::-1].copy()
            except OutOfReach as exc:
                failed = exc
        raise failed or OutOfReach(self.arm.robot, ["hover"])

    def _joint_way(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        load: tuple[float, float] | None,
        label: str,
        max_tilt: float | None = None,
        skip: int | None = None,
    ) -> list[Waypoint] | None:
        """Joint moves from `q0` to `q1` that hit nothing (sweeps_clear): straight; or after rising RISE from `q0`
        with the hand turned as it is; or by way of the arm's home pose, its base turned halfway (an arm folded
        down behind its base hits its own robot going straight out to a counter); None if none works. `skip`: the
        body the arm may touch (by default what it picks up or carries)."""
        arm = self.arm
        half = None if load is None else load[0]
        skip = self.load if skip is None else skip

        def move(a: np.ndarray, b: np.ndarray) -> Waypoint:
            return Waypoint(None, None, _joint_time(a, b), label, b.copy(), load=half)

        if sweeps_clear(arm, q0, q1, skip, load, max_tilt):
            return [move(q0, q1)]
        first: list[Waypoint] = []
        p0 = arm.ik.point(arm.chain, q0)
        turn = arm.ik.data.xmat[arm.chain.body].reshape(3, 3).copy()
        try:
            up = self._path(q0, p0, turn, p0 + [0.0, 0.0, RISE], turn)
        except OutOfReach:
            up = None
        if up is not None:
            if sweeps_clear(arm, up[-1], q1, skip, load, max_tilt):
                return [self._follow_move(up, p0, p0 + [0.0, 0.0, RISE], 0.0, "rise", half), move(up[-1], q1)]
            first, q0 = [self._follow_move(up, p0, p0 + [0.0, 0.0, RISE], 0.0, "rise", half)], up[-1]
        for via in self._vias(q0, q1):
            if sweeps_clear(arm, q0, via, skip, load, max_tilt) and sweeps_clear(arm, via, q1, skip, load, max_tilt):
                return [*first, move(q0, via), move(via, q1)]
        return None

    def _vias(self, q0: np.ndarray, q1: np.ndarray) -> list[np.ndarray]:
        """Poses to go by from `q0` to `q1`: the arm's home pose turned halfway, then the pose halfway between the
        two with its tool raised VIA_UP higher than either end (going straight, the hand passed through the
        counter's edge between the robot and a tilted pose over the counter)."""
        arm = self.arm
        home = arm.home.copy()
        if arm.yaw is not None:  # turned halfway from where it starts to where it goes
            home[arm.yaw] = (q0[arm.yaw] + q1[arm.yaw]) / 2
        out = [home]
        mid = (q0 + q1) / 2
        top = max(float(arm.ik.point(arm.chain, q0)[2]), float(arm.ik.point(arm.chain, q1)[2]))
        p = arm.ik.point(arm.chain, mid)
        for up in VIA_UP:
            target = np.array([p[0], p[1], max(p[2], top) + up])
            q, err = arm.ik.solve(arm.chain, target, mid, iterations=60)
            if err <= REACH_TOL:
                out.append(q)
        return out

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
        if out > 1e-6:  # on the line from the arm's base to where it goes in (a box is come at square to a face)
            toward = (np.asarray(first, float)[:2] - arm.base_xy) / out
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
        """Queue following a path worked out by _path."""
        self.arm.queue.append(self._follow_move(knots, a, b, angle, label, load))

    @staticmethod
    def _follow_move(
        knots: np.ndarray, a: np.ndarray, b: np.ndarray, angle: float, label: str, load: float | None = None
    ) -> Waypoint:
        """The move along a path worked out by _path, timed by how far the tool goes and turns and the joints go."""
        travel = float(np.sum(np.max(np.abs(np.diff(knots, axis=0)), axis=1)))
        duration = max(0.5, float(np.linalg.norm(b - a)) / SPEED, angle / TURN_SPEED, travel / JOINT_SPEED)
        return Waypoint(None, None, duration, label, knots[-1], load=load, path=knots)

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
        return not _links_pass_through(arm, (arm._robot_root, self.load))

    def _off(self, q: np.ndarray, turn: np.ndarray) -> float:
        """How far (rad) the tool body at `q` is turned from `turn`."""
        arm = self.arm
        arm.ik.point(arm.chain, q)
        return _angle(turn, arm.ik.data.xmat[arm.chain.body].reshape(3, 3))


def sweeps_clear(
    arm: ArmMotion,
    q0: np.ndarray,
    q1: np.ndarray,
    skip: int = -1,
    load: tuple[float, float] | None = None,
    max_tilt: float | None = None,
) -> bool:
    """Whether the arm moving joint by joint from `q0` to `q1` hits nothing on the way, in the IK's copy of the world:
    the robot it rides on, the floor, furniture, other things (but `skip`, the body of what it picks up or holds).
    With a load (half height, radius) in its gripper (the body `skip`, carried along in the copy with its middle at
    the tool point), the load touches nothing but the gripper either; with `max_tilt` the gripper also keeps within
    that of pointing down."""
    m, d = arm.ik.model, arm.ik.data
    n = max(4, math.ceil(float(np.max(np.abs(q1 - q0))) / SWEEP_STEP))
    carried = load is not None and skip > 0 and int(m.body_jntadr[skip]) >= 0
    adr = int(m.jnt_qposadr[m.body_jntadr[skip]]) if carried else -1
    saved = arm.ik.base_qpos[adr : adr + 3].copy() if carried else None
    hand = {b for b in arm._moving if _descends(m, b, arm.chain.body)}
    try:
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            q = q0 + (q1 - q0) * s
            if carried:
                arm.ik.base_qpos[adr : adr + 3] = arm.ik.point(arm.chain, q)
            p = arm.ik.point(arm.chain, q)
            if max_tilt is not None and load is not None:
                axis = d.xmat[arm.chain.body].reshape(3, 3) @ arm.tool_axis
                if float(axis[2]) > -math.cos(max_tilt):
                    return False
            if load is not None and not carried and p[2] - load[0] < LOAD_GAP:
                return False
            if arm._collides(q):
                return False
            for con in d.contact[: d.ncon]:  # posed at q by _collides
                if con.dist > -CONTACT:
                    continue
                b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
                r1, r2 = int(m.body_rootid[b1]), int(m.body_rootid[b2])
                if carried and skip in (r1, r2):  # the load, where the gripper holds it
                    other = b2 if r1 == skip else b1
                    if other not in hand and int(m.body_rootid[other]) != skip:
                        return False
                    continue
                if (b1 in arm._moving) == (b2 in arm._moving):
                    continue
                other = b2 if b1 in arm._moving else b1
                if int(m.body_rootid[other]) not in (arm._robot_root, skip):
                    return False
    finally:
        if carried:
            arm.ik.base_qpos[adr : adr + 3] = saved
    return True


def _links_pass_through(arm: ArmMotion, ignore: tuple[int, ...]) -> bool:
    """Whether the axis of one of the arm's capsule links, as posed in the IK's copy, passes through something solid
    whose root is not in `ignore`. MuJoCo's capsule-box test misses a capsule that goes right through a thin box: a lift
    picker's upper arm through the 2 cm back panel of a rack read as 3 cm clear, and the real arm pushed on the panel
    instead of reaching the parcel behind it."""
    m, d = arm.ik.model, arm.ik.data
    hit = np.zeros(1, np.int32)
    for g in range(m.ngeom):
        if int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_CAPSULE or int(m.geom_bodyid[g]) not in arm._moving:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2] * float(m.geom_size[g][1])
        start, end = d.geom_xpos[g] - axis, d.geom_xpos[g] + axis
        length = float(np.linalg.norm(end - start))
        way = (end - start) / length
        gone = 0.0
        while gone < length:
            dist = mujoco.mj_ray(m, d, start + way * gone, way, None, 1, -1, hit)
            if dist < 0 or gone + dist > length:
                break
            other = int(hit[0])
            body = int(m.geom_bodyid[other])
            solid = bool(m.geom_contype[other] or m.geom_conaffinity[other])
            if solid and body not in arm._moving and int(m.body_rootid[body]) not in ignore:
                return True
            gone += dist + 1e-4
    return False


def _descends(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _joint_time(q0: np.ndarray, q1: np.ndarray) -> float:
    return max(0.5, float(np.max(np.abs(np.asarray(q1) - np.asarray(q0)))) / JOINT_SPEED)


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
