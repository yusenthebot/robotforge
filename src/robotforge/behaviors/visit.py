"""Going over to work a mechanism: an arm that rides on a robot that drives or walks, told to open a drawer or a door,
press a button, flip a switch or a lever or turn a valve it does not reach from where the robot stands, has the robot
go to a spot from which the arm does the whole job, stand still there, do it (operate.py) and fold the arm back.

Where to stand is looked for as fetch.py looks for where to pick something up (fetch.spots_for): round the middle of
the handle's way, on free floor the robot can get to, where its shapes fit. Each promising spot is rehearsed: the
operation is planned from there in a copy of the world and the spot passed over when the arm would not follow the
handle the whole way, or when the robot would stand in the way of what moves (a drawer pulled out into its legs, a
door swung against it). Where the robot really stops, the operation is planned again; if it falls short from there,
a rover creeps into the spot and tries again, and then, as a robot that walks does at once, looks for another spot
from where it stands (at most REPLANS times). The robot is kept where it stands while the arm works (fetch.Footing).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from robotforge.behaviors import fetch, operate
from robotforge.behaviors.arm import ArmMotion
from robotforge.behaviors.navigate import NoWay
from robotforge.behaviors.reach_in import LOADED_TILT, Hand
from robotforge.design import mechanisms
from robotforge.sim.geometry import collides

if TYPE_CHECKING:
    from robotforge.behaviors.manager import BehaviorManager

log = logging.getLogger(__name__)

TIME_LIMIT = 150.0  # s of sim time an operate that goes over may take: the trip, the operation, the fold
# m kept between the robot (not its arm, whose own way operate checks) and what moves while it works: at the spot it
# stands on, with the mechanism halfway and all the way to where it is going
KEEP_CLEAR = 0.03
WAY_SHARES = (0.5, 1.0)  # of the move, the mechanism's poses checked against the robot's
REPLANS = 2  # spots looked for again from where the robot stopped
STAND_TRIES = 1  # times a rover creeps into its spot again before it looks for another
FULL = 0.97  # share of the move the arm must be planned to follow from a spot for the spot to count
# Spots checked by IK in one search: a quick check of the handle's whole way passes over most, and the spots beside
# a drawer from which a Go2 works it came after 40 nearer the robot's start that do not.
MAX_CHECKS = 160
LIFT = operate.CLEAR_ABOVE  # m above the handle the arm must reach pointing down too, in the quick check of a spot


@dataclass(frozen=True)
class Job:
    """What the arm is to do there: operate's arguments, and the mechanism's joint (qpos address) and goal value."""

    target: str
    shape: str
    action: str
    amount: float | None
    adr: int
    start: float
    goal: float


def plan(manager: BehaviorManager, arm: ArmMotion, target: str, shape: str, action: str, amount: float | None) -> Visit:
    """A visit for `arm` (riding on a robot that moves) to do `action` on the mechanism `target`, or OperateError
    saying why it cannot: the arm cannot fold over its robot to go, or nowhere the robot can stand does the arm reach
    the whole way."""
    session = manager.session
    model, data = session.model, session.data
    robot = arm.robot.split("/", 1)[0]
    job = _job(model, data, target, shape, action, amount)
    arm.sync(data)
    tuck = arm.hold_pose(0.0, 0.0)
    if tuck is None:
        raise operate.OperateError(
            f"{arm.robot} cannot fold back over {robot}, so {robot} cannot go to {target} with it; move {robot} next "
            f"to {target} yourself (move with to=...)"
        )
    _, hold_q = tuck
    try:
        stand = _ClearStand(model, data, robot, arm, hold_q)
    except fetch.FetchError as exc:
        raise operate.OperateError(str(exc)) from exc
    stand.hand.across = None  # operate works with the tool pointing down: no tilted reaching in
    stand.job = job
    stand.walks = manager.kind(robot).value in fetch.WALKERS
    start = np.array(data.xpos[manager.root(robot)][:2], float)
    nav = manager.floor_map(robot, fetch._posed(model, data, arm, hold_q))
    reach = fetch._reachable_cells(nav, start)
    way = _way_points(model, data, job)
    middle = way[1]
    stand.way = (way[0], way[2])
    spot, _ = fetch.spots_for(
        stand,
        nav,
        [middle],
        operate.ABOVE + 0.03 if action == "press" else LIFT,  # a press comes down from over the cap
        start,
        reach,
        _under(model, data, target),
        rehearse=_rehearsal(stand, job),
        max_checks=MAX_CHECKS,
    )
    if spot is None:
        raise operate.OperateError(
            fetch._nowhere(manager, stand, nav, reach, robot, middle, f"{target}'s handle the whole way", -1).replace(
                "put it nearer the edge or lower down", f"move {target} nearer open floor or lower down"
            )
        )
    return Visit(manager, robot, arm, job, hold_q, spot)


def _job(model: mujoco.MjModel, data: mujoco.MjData, target: str, shape: str, action: str, amount: float | None) -> Job:
    info = mechanisms.MECHANISMS[shape]
    if action not in info.actions:
        can = " or ".join(info.actions) or "nothing operate does"
        raise operate.OperateError(f"{target} is a {shape}: it can {can}, not {action}")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{target}/{info.part}")
    if joint < 0:
        raise operate.OperateError(f"{target} has no handle to take hold of; reload the world (open_world)")
    adr = int(model.jnt_qposadr[joint])
    q = float(data.qpos[adr])
    lo, hi = (float(v) for v in model.jnt_range[joint])
    delta, _ = operate._delta(shape, action, amount, q, lo, hi, info.unit)
    return Job(target, shape, action, amount, adr, q, q + delta)


def _way_points(model: mujoco.MjModel, data: mujoco.MjData, job: Job) -> list[np.ndarray]:
    """The handle at the start, halfway and at the end of its way: the robot stands round the middle, so the arm has
    both ends within reach."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{job.target}/handle")
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    out = []
    for share in (0.0, 0.5, 1.0):
        probe.qpos[job.adr] = job.start + share * (job.goal - job.start)
        mujoco.mj_kinematics(model, probe)
        out.append(np.array(probe.site_xpos[site], float))
    return out


def _under(model: mujoco.MjModel, data: mujoco.MjData, target: str) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """The mechanism and what it stands on, as boxes seen from above: the arm's base must not be under them."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == body and collides(model, g):
            lo, hi = fetch.nav_map.world_extent(model, data, g)
            out.append((lo[:2], hi[:2], float(hi[2])))
    return out + fetch._supports(model, data, body)


class _ClearStand(fetch.Stand):
    """A stand that fits only where the robot also keeps clear of what moves (_clear_of_way), checked before the
    arm's reach: of a Go2's first 17 spots in front of a chest the arm reached from, the drawer pulled out ran into
    the dog at every one, and the search gave up at its 40 reach checks before it got to the spots beside it."""

    job: Job | None = None
    way: tuple[np.ndarray, ...] = ()  # the handle at the start, the middle and the end of its way

    def fits(self, xy, heading: float) -> bool:
        return super().fits(xy, heading) and (self.job is None or _clear_of_way(self, xy, heading, self.job))

    def grip(self, xy, heading: float, target: np.ndarray, lift: float, tilts: tuple[float, ...]) -> float | None:
        """Straight down only, at `target` and `lift` above it: an operation comes down onto the handle from as high
        as the arm reaches pointing down (operate.COME_DOWN at most), and carries nothing over the robot, so the
        point over the robot a pick checks (fetch.Stand.grip) does not apply. With it, a Go2's PiPER reached no
        chest of drawers 0.36 m high from anywhere."""
        if 0.0 not in tilts:
            return None
        arm = self.arm
        now, arm.q = arm.q, self.hold_q.copy()
        points = (target, target + [0.0, 0.0, lift], *self.way)  # and both ends of the handle's way
        try:
            with arm.carried_to(self.qpos_at(xy, heading)):
                ok = all(arm.quick_error(p) <= fetch.REACH_TOL for p in points)
        finally:
            arm.q = now
        return 0.0 if ok else None


def _rehearsal(stand: fetch.Stand, job: Job):
    """For spots_for: whether, from (xy, heading), the robot keeps clear of what moves and the arm does it all."""

    def works(xy: np.ndarray, heading: float, point: np.ndarray, tilt: float) -> bool:
        return _full(_plan_from(stand, xy, heading, job))

    return works


def _clear_of_way(stand: fetch.Stand, xy, heading: float, job: Job) -> bool:
    """Whether the robot standing at `xy`, its arm tucked, stays KEEP_CLEAR from the mechanism's moving parts while
    they go to where the job takes them."""
    m, d = stand.model, stand.probe
    moving = _moving_geoms(m, job.target)
    mine = [
        g
        for g in range(m.ngeom)
        if int(m.geom_bodyid[g]) in stand.robot_bodies
        and int(m.geom_bodyid[g]) not in stand.arm._moving
        and collides(m, g)
    ]
    q = stand.qpos_at(xy, heading)
    for share in WAY_SHARES:
        d.qpos[:] = q
        d.qpos[job.adr] = job.start + share * (job.goal - job.start)
        mujoco.mj_kinematics(m, d)
        for g in mine:
            for h in moving:
                gap = np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[h]) - m.geom_rbound[g] - m.geom_rbound[h]
                if gap > KEEP_CLEAR:
                    continue
                if mujoco.mj_geomDistance(m, d, g, h, KEEP_CLEAR, None) < KEEP_CLEAR:
                    return False
    return True


def _moving_geoms(model: mujoco.MjModel, target: str) -> list[int]:
    carcass = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) != carcass
        and operate._below(model, int(model.geom_bodyid[g]), carcass)
        and collides(model, g)
    ]


def _plan_from(stand: fetch.Stand, xy, heading: float, job: Job) -> operate.Operation | None:
    """The operation planned with the robot at (xy, heading) and the arm tucked, in a copy of the world; None if the
    arm cannot take hold from there."""
    arm = stand.arm
    posed = stand.posed(xy, heading)
    now, arm.q = arm.q, stand.hold_q.copy()
    try:
        with arm.carried_to(posed.qpos):
            return _plan(stand.model, posed, arm, job)
    except (operate.OperateError, ValueError):
        return None
    finally:
        arm.q = now


def _plan(model: mujoco.MjModel, data: mujoco.MjData, arm: ArmMotion, job: Job) -> operate.Operation:
    """operate.plan, its IK warmed up on the handle first, the same way in a rehearsal and for real: the arm's last
    reach (ArmMotion.quick_error) seeds its solver, and a lift picker's rehearsed hold on a drawer, seeded by the
    spot search's checks, was not found again where it stopped (0.35 m from the handle, just as planned)."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{job.target}/handle")
    arm._warm = None
    arm.warm_up(np.array(data.site_xpos[site], float))
    return operate.plan(model, data, arm, job.target, job.shape, job.action, job.amount)


def _full(op: operate.Operation | None) -> bool:
    """Whether a planned operation does the whole job (a press or a flip has nothing to fall short of)."""
    if op is None:
        return False
    if not op.legs:  # nothing to do: it is there already
        return True
    planned, can = op.report.get("planned"), op.report.get("can")
    if planned is None or can is None:
        return True
    unit = op.report["unit"]
    can_shown = abs(can) if unit == "m" else abs(math.degrees(can))
    return planned >= FULL * can_shown - 1e-6


@dataclass
class Visit(fetch.Footing):
    """A controller that goes over to work a mechanism: tuck the arm, go to the spot, settle, operate, fold back.
    Runs at fetch.RATE_HZ on top of the robot's walker, its trip and the operation, which do the moving; it sits
    among the controllers where an errand would (fetch.KEY + robot)."""

    manager: BehaviorManager
    robot: str
    arm: ArmMotion
    job: Job
    hold_q: np.ndarray
    spot: fetch.Spot
    rate_hz: float = fetch.RATE_HZ
    step: str = "starting"
    state: str = "working"  # working, done, failed
    why: str | None = None
    note: str | None = None
    op: operate.Operation | None = None
    legs: list[fetch.Leg] = field(default_factory=list)
    started: float = 0.0
    ended: float | None = None
    stood_at: list[float] | None = None
    _leg: fetch.Leg | None = None
    _until: float = 0.0
    _tries: int = 0
    _replans: int = 0
    _creep: tuple[float, fetch.Spot] | None = None
    _stance: tuple[np.ndarray, float] | None = None

    @property
    def done(self) -> bool:
        return self.state != "working"

    @property
    def time_limit(self) -> float:
        return TIME_LIMIT

    def status(self) -> dict[str, Any]:
        return {"behavior": "operate", "robot": self.robot, "target": self.job.target, "step": self.step}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            return
        try:
            self._update(data)
        except Exception as exc:  # an error here must end the visit, not every step of the simulation
            log.exception("%s's visit to %s failed", self.robot, self.job.target)
            self._fail(data, "error", f"it broke off on an internal error: {exc}")

    def _update(self, data: mujoco.MjData) -> None:
        t = float(data.time)
        if self.step == "starting":
            self.started = t
            self._tuck(data, "tucking")
            return
        if not self.manager.pose(self.robot)["upright"]:
            self._fail(data, "fell", "it fell over")
            return
        getattr(self, f"_on_{self.step}")(t, data)

    # -- the steps --

    def _on_tucking(self, t: float, data: mujoco.MjData) -> None:
        if self.arm.done:
            self._go(data)

    def _on_going(self, t: float, data: mujoco.MjData) -> None:
        leg = self._leg
        if leg is None or leg.trip is None or not leg.trip.done:
            return
        leg.result = leg.trip.result()
        short = math.dist(leg.spot.xy, data.xpos[self.manager.root(self.robot)][:2])
        if not leg.trip.arrived and short > fetch.LEG_SLACK:
            self._fail(
                data,
                leg.result.get("why") or "stuck",
                f"on its way to {self.job.target}: {leg.result.get('note') or 'it did not get there'}",
            )
            return
        self.step, self._until = "settling", t + fetch.SETTLE

    def _on_creeping(self, t: float, data: mujoco.MjData) -> None:
        if self._creep_in(t, data):
            self.step, self._until = "settling", t + fetch.SETTLE

    def _on_settling(self, t: float, data: mujoco.MjData) -> None:
        if t >= self._until:
            self._start_work(t, data)

    def _on_working(self, t: float, data: mujoco.MjData) -> None:
        self._hold_still(data)
        if self.op is None or not self.op.done:
            return
        self._stand_free()
        self._tuck(data, "folding")

    def _on_folding(self, t: float, data: mujoco.MjData) -> None:
        if self.arm.done:
            self._finish(data)

    # -- helpers --

    def _tuck(self, data: mujoco.MjData, step: str) -> None:
        """Fold the arm over the robot for the way (hold_q), joint by joint: the arm's own motion takes it back from
        an operation that held it."""
        arm = self.manager._arm_motion(self.arm.robot)  # made afresh where an operation had the arm
        self.arm = arm
        arm.sync(data)
        if float(np.max(np.abs(arm.q - self.hold_q))) > fetch.STOW_SAME:
            moves = Hand(arm)._joint_way(arm.q, self.hold_q, None, "stow", LOADED_TILT, -1)
            if moves is not None:
                arm.queue.extend(moves)
            else:  # no joint move clear of everything: the tool point there, the arm's own way
                arm.move_to(arm.ik.point(arm.chain, self.hold_q))
        self.step = step

    def _go(self, data: mujoco.MjData) -> None:
        spot = self.spot
        try:
            trip = self.manager.navigate(self.robot, list(spot.xy), face=spot.face())
        except NoWay as exc:
            self._fail(data, exc.cause, f"no way to {self.job.target}: {exc}")
            return
        self._leg = fetch.Leg(self.job.target, spot, trip)
        self.legs.append(self._leg)
        self.step = "going"

    def _start_work(self, t: float, data: mujoco.MjData) -> None:
        """Plan the operation from where the robot stands now and set it going, or creep in, or find another spot."""
        session = self.manager.session
        name = self.arm.robot
        arm = ArmMotion(session.model, name, self.manager.home_qpos())
        arm.sync(data)
        arm.q = data.qpos[arm.chain.qpos_adr].copy()
        job = self.job
        try:
            op = _plan(session.model, data, arm, job)
        except operate.OperateError as exc:
            op, why = None, str(exc)
        else:
            why = None
        if not _full(op) and self._again(t, data):
            return
        if op is None:
            self._fail(data, "out_of_reach", f"from where it stopped: {why}")
            return
        self.op = op
        self._stance = self._pose(data)
        self.stood_at = [round(float(v), 3) for v in self._stance[0]] + [round(math.degrees(self._stance[1]), 1)]
        session.set_controller(name, op)
        self.step = "working"

    def _again(self, t: float, data: mujoco.MjData) -> bool:
        """Creep into the spot once more (a rover), or look for another spot from where it stands and go there; True
        if it does either."""
        walks = self.manager.kind(self.robot).value in fetch.WALKERS
        if self._tries < STAND_TRIES and not walks:
            self._tries += 1
            self._creep = (t, self.spot)
            self.step = "creeping"
            return True
        if self._replans >= REPLANS:
            return False
        self._replans += 1
        try:
            visit = plan(
                self.manager,
                self.manager._arm_motion(self.arm.robot),
                self.job.target,
                self.job.shape,
                self.job.action,
                self.job.amount,
            )
        except (operate.OperateError, ValueError, RuntimeError) as exc:
            log.info("%s found no other spot for %s: %s", self.robot, self.job.target, exc)
            return False
        self.spot, self._tries = visit.spot, 0
        self._tuck(data, "tucking")
        return True

    def _fail(self, data: mujoco.MjData, why: str, note: str) -> None:
        self.state, self.why, self.note = "failed", why, note
        self.ended = float(data.time)
        self._leave()

    def _finish(self, data: mujoco.MjData) -> None:
        self.state = "done"
        self.ended = float(data.time)
        self._leave()

    def _leave(self) -> None:
        self._stand_free()
        self.manager.errand_ended(self)

    def interrupt(self, note: str, time: float) -> None:
        if not self.done:
            self.state, self.why, self.note, self.ended = "failed", "interrupted", note, time
        self._stand_free()

    def result(self, model: mujoco.MjModel, data: mujoco.MjData, state: dict[str, Any]) -> dict[str, Any]:
        """What it did: the operation's result (operate.result) and the trip there, where it stood, how long it took."""
        out: dict[str, Any] = {}
        if self.op is not None:
            out.update(operate.result(self.op, model, data, state))
        else:
            out.update({"robot": self.arm.robot, "target": self.job.target, "action": self.job.action})
            out.update({"success": False, "state": state})
        out["went"] = []
        for leg in self.legs:
            r = leg.result or (leg.trip.result() if leg.trip is not None else {})
            out["went"].append(
                {"to": leg.to, **{k: r[k] for k in ("arrived", "path_m", "distance_m", "time_s") if k in r}}
            )
        if self.stood_at is not None:
            out["stood_at"] = self.stood_at  # x, y and heading in degrees
        if self.ended is not None:
            out["visit_s"] = round(self.ended - self.started, 2)
        if self.state != "done":
            out["success"] = False
            out["why"] = self.why or "out_of_time"
            out["step"] = self.step
            if self.note:
                out["note"] = self.note
        return out
