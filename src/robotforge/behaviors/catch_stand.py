"""Catching off a belt with the robot that carries the arm going to where the arm can: base and arm plan together.

An arm on a robot that drives or walks catches what rides by on a conveyor (catch.py) from where the robot stands. When
it does not reach the box's way from there, `stand_for` looks along the box's way for a place beside the belt to stand
at, and `Approach` takes the robot there: a trip to the spot, creeping the last cm in, then the catch is planned again
from where the robot really stopped (a box that rides on while the robot goes is where it will be by then, and the
arm has to unfold on arrival); if it does not reach from there, a spot is looked for once more, as an errand's pick
does (fetch.py). The base then stands still while the arm works, as it does for an errand.

When the box will be at the end of the belt before the robot can be beside it with its arm ready, the robot waits
for it to come to rest there and fetches it as it lies (Approach with no spot), and an approach that arrives too late
for the catch does the same.

A spot is a place on free floor the robot gets to, on either side of the belt, where its shapes touch nothing, from
which the meeting (catch.meeting) is found with the robot posed there and the arm tucked in as it arrives, and the
box's lead on the robot taken into account: the box is where it will be after the drive, so a spot upstream of the
box, which the robot reaches before the box does, ranks before one the box has gone past.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from robotforge.behaviors import catch as catching
from robotforge.behaviors import fetch
from robotforge.behaviors.navigate import NoWay

if TYPE_CHECKING:
    from robotforge.behaviors.arm import ArmMotion
    from robotforge.behaviors.manager import BehaviorManager
    from robotforge.behaviors.navigate import Navigator

log = logging.getLogger(__name__)

KEY = "__catch_go__"  # session controller key of a robot's drive to a place to catch from: KEY + robot
DRIVE_SPEED = (
    0.3  # m/s: how fast a trip is counted to take, with the turns in it (a Go2's cruise is 0.5, a lift picker's less)
)
SPARE_S = 4.0  # s counted for folding the arm, creeping in and settling, over the drive
SLACK_S = 3.0  # s more the box is to be late by at the place it is found for: the drive takes what it takes
BELT_STEP = 0.25  # m between the places along the box's way tried
SIDE_SHARES = (
    0.6,
    0.45,
    0.75,
    0.3,
)  # of how far the arm reaches out at the box's height, the belt's distance from its base
BEARINGS = tuple(math.radians(b) for b in (0, 45, -45, 90, -90))  # the belt at this angle off the robot's heading
MAX_CHECKS = 24  # places checked by the full search for a meeting, nearest first
APPROACH_S = 90.0  # s of sim time the drive and creeping in may take
READY_S = 4.0  # s the arm takes to unfold and come down over the box, at the least, once the robot is there
WAIT_S = 45.0  # s of sim time it waits for the box to come to rest at the end of the belt, at the most
TRIES = 2  # times it looks for a place again from where it stopped
BEHIND = 0.5  # m upstream of the box that places are tried


class NoStand(ValueError):
    """Why there is nowhere beside the belt to catch from. `late`: the box is at the end of the belt, to lie there,
    sooner than the robot could be beside it with its arm ready."""

    def __init__(self, message: str, late: bool = False):
        super().__init__(message)
        self.late = late


@dataclass(frozen=True)
class Stance:
    """Where the robot stands to catch, and what was found there."""

    spot: fetch.Spot
    meet: catching.Meeting
    drive_s: float  # about how long it takes to get there


def stand_for(
    manager: BehaviorManager, arm: ArmMotion, body: int, half: float, riding: np.ndarray, data: mujoco.MjData
) -> Stance:
    """A place for the robot to stand at, from which `arm` catches the box (`body`, riding at `riding` m/s) as it
    comes by; raises NoStand, saying why not."""
    model = manager.session.model
    robot = arm.robot.split("/", 1)[0]
    arm.sync(data)
    tuck = arm.hold_pose(half, catching.load_radius(model, body))
    if tuck is None:
        raise NoStand(f"{arm.robot} cannot fold back over {robot}, so it cannot go anywhere with its arm")
    hold, hold_q = tuck
    root = manager.root(robot)
    rot = data.xmat[root].reshape(3, 3)
    heading = math.atan2(rot[1, 0], rot[0, 0])
    c, s = math.cos(heading), math.sin(heading)
    off = hold - data.xpos[root]
    stand = fetch.Stand(model, data, robot, arm, hold_q, body)
    stand.hold_local = np.array([c * off[0] + s * off[1], -s * off[0] + c * off[1], off[2]])
    stand.walks = manager.kind(robot).value in fetch.WALKERS
    start = np.array(data.xpos[root][:2], float)
    nav = manager.floor_map(robot, fetch._posed(model, data, arm, hold_q))
    stand.ground = nav
    reach = fetch._reachable_cells(nav, start)
    speed = float(np.linalg.norm(riding))
    way = riding / speed
    across = np.array([-way[1], way[0]])
    box = np.array(data.xpos[body], float)
    grasp_z = float(box[2]) + (half if arm.tool is not None else 0.0)
    out = stand.out_at(grasp_z)
    length = min(catching.clear_ahead(model, data, body, way), catching.LOOK_AHEAD)
    stops = length < catching.LOOK_AHEAD  # something fixed ends the box's way: it comes to rest there
    if out <= fetch.MIN_OUT or length <= 0.0:
        raise NoStand(f"{arm.robot} does not reach the height of the belt ({grasp_z:.2f} m) from anywhere")
    candidates = []
    for along in np.arange(-BEHIND, length, BELT_STEP):
        belt = box[:2] + way * along
        for side in (1.0, -1.0):
            for share in SIDE_SHARES:
                base = belt + side * share * out * across
                for bearing in BEARINGS:
                    # The belt at `bearing` off the robot's heading, as the arm sees it (fetch._candidates).
                    facing = math.atan2(-side * across[1], -side * across[0]) - bearing
                    xy = base - stand.arm_base_at((0.0, 0.0), facing)
                    cell = nav.grid.cell(*xy)
                    if not (nav.grid.inside(*cell) and nav.free[cell] and reach[cell]):
                        continue
                    dist = float(np.linalg.norm(xy - start))
                    turn = abs(fetch._wrap(facing - heading))
                    drive = 1.3 * dist / DRIVE_SPEED + turn / 0.6 + SPARE_S
                    cost = drive + 4.0 * abs(share - SIDE_SHARES[0]) + 0.5 * abs(bearing) + max(0.0, along)
                    candidates.append((cost, drive, xy, float(fetch._wrap(facing))))
    candidates.sort(key=lambda c: c[0])
    checked = 0
    for _, drive, xy, facing in candidates:
        if checked >= MAX_CHECKS:
            break
        if not stand.fits(xy, facing):
            continue
        checked += 1
        meet = _meets(stand, arm, data, body, half, riding, xy, facing, drive + SLACK_S)
        if meet is None:
            continue
        return Stance(
            fetch.Spot((float(xy[0]), float(xy[1])), facing, tuple(float(v) for v in meet.point)), meet, drive
        )
    if stops and candidates and min(c[1] for c in candidates) + SLACK_S + READY_S >= length / speed:
        raise NoStand(
            f"the box is at the end of the belt in {length / speed:.0f} s, sooner than {robot} can be beside the belt "
            f"with its arm ready ({min(c[1] for c in candidates) + READY_S:.0f} s at least): it is fetched there as it "
            "lies, or slow the belt down (set_controls)",
            late=True,
        )
    raise NoStand(
        f"{arm.robot} does not reach the box's way from anywhere beside the belt that {robot} can get to "
        f"({checked} places tried)"
    )


def _meets(
    stand: fetch.Stand,
    arm: ArmMotion,
    data: mujoco.MjData,
    body: int,
    half: float,
    riding: np.ndarray,
    xy: np.ndarray,
    heading: float,
    delay: float,
) -> catching.Meeting | None:
    """The meeting with the box from the robot at `xy` and `heading`, its arm tucked, `delay` s from now."""
    now, arm.q = arm.q, stand.hold_q.copy()
    try:
        with arm.carried_to(stand.qpos_at(xy, heading)):
            return catching.meeting(stand.model, data, arm, body, half, riding, delay=delay, sync=False)
    except catching.CatchError:
        return None
    finally:
        arm.q = now


@dataclass
class Approach(fetch.Footing):
    """A controller that takes a robot to where its arm catches a box off a belt and then has it catch (it hands over
    to the catch, `inner`, and is done when that is)."""

    manager: BehaviorManager
    robot: str
    arm: ArmMotion
    obj: str
    body: int
    half: float
    target: str | list[float]
    spot: fetch.Spot | None  # None: the robot waits for the box to come to rest, and fetches it as it lies
    place_region: dict[str, Any] | None  # where the load counts as put, when the target is known
    tries: int = 0
    rate_hz: float = fetch.RATE_HZ
    step: str = "going"
    waited: bool = False  # it gave up the catch, and waits for the box at the end of the belt
    note: str | None = None
    inner: Any = None  # the Catch (or an Approach again, from where it stopped) once it is there
    started: float | None = None
    stood_at: list[float] | None = None
    drive_s: float = 0.0
    _trip: Navigator | None = None
    _creep: tuple[float, fetch.Spot] | None = None
    _stance: tuple[np.ndarray, float] | None = None
    _until: float = 0.0
    failed: bool = field(default=False, init=False)

    @property
    def region(self) -> dict[str, Any] | None:
        """Where the load counts as put (BehaviorManager.placed): what the catch found, else the target's."""
        found = getattr(self.inner, "region", None)
        if found is None and hasattr(self.inner, "place"):  # an errand picking it up where it stopped
            found = self.inner.place.region
        return found if found is not None else self.place_region

    @property
    def done(self) -> bool:
        if self.inner is not None:
            return bool(self.inner.done)
        return self.failed

    @property
    def time_limit(self) -> float:
        return APPROACH_S + catching.CATCH_S + catching.CARRY_S

    def result(self) -> dict[str, Any]:
        done = getattr(self.inner, "result", None)
        out: dict[str, Any] = done() if callable(done) else {"caught": False}
        if self.stood_at is not None:
            out["stood_at"] = self.stood_at
        if self.note and not out.get("note"):
            out["note"] = self.note
        return out

    def status(self) -> dict[str, Any]:
        inner = self.inner.status() if self.inner is not None else {"step": self.step}
        return {"behavior": "catch", "robot": self.arm.robot, **inner, "done": self.done}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done or self.inner is not None:
            return
        try:
            self._update(data)
        except Exception as exc:  # an error here must end the drive, not every step of the simulation
            log.exception("%s's drive to the belt failed", self.robot)
            self._fail(f"it broke off on an internal error: {exc}")

    def _update(self, data: mujoco.MjData) -> None:
        t = float(data.time)
        if self.started is None:
            self.started = t
            if self.spot is None:
                self._wait(t)
            else:
                try:
                    self._trip = self.manager.navigate(self.robot, list(self.spot.xy), face=self.spot.face())
                except NoWay as exc:
                    self._fail(f"no way to a place beside the belt: {exc}")
                    return
        if self.step == "waiting":
            self._on_waiting(t, data)
            return
        if t - self.started > APPROACH_S:
            self.manager.stop(self.robot)
            self._fail("it did not get to the belt in time")
            return
        assert self.spot is not None
        if self.step == "going":
            if self._trip is None or not self._trip.done:
                return
            reached = math.dist(self.spot.xy, self._pose(data)[0])
            if not self._trip.arrived and reached > fetch.LEG_SLACK:
                self._fail(f"on its way to the belt: {self._trip.result().get('note') or 'it did not get there'}")
                return
            self._creep, self.step = (t, self.spot), "creeping"
        elif self.step == "creeping":
            if self._creep_in(t, data):
                self._until, self.step = t + fetch.SETTLE, "settling"
        elif self.step == "settling":
            if t >= self._until:
                self._catch(data)

    def _wait(self, t: float) -> None:
        self.step, self.waited, self._until = "waiting", True, t + WAIT_S

    def _on_waiting(self, t: float, data: mujoco.MjData) -> None:
        if catching.velocity(self.manager.session.model, data, self.body) is not None:
            if t > self._until:
                self._fail(self.note or "the box did not come to rest at the end of the belt")
            return
        self._catch(data)

    def _catch(self, data: mujoco.MjData) -> None:
        """Plan the catch from where the robot stopped, or fetch the box where it lies; if the arm does not reach
        from here, look for another place; if the box is too far gone for that, wait for it at the end of the belt."""
        root = self.manager.root(self.robot)
        rot = data.xmat[root].reshape(3, 3)
        self.stood_at = [round(float(v), 3) for v in (*data.xpos[root][:2], math.atan2(rot[1, 0], rot[0, 0]))]
        riding = catching.velocity(self.manager.session.model, data, self.body)
        try:
            if riding is None:  # it lies still (at the end of the belt): picked up as anything is
                self.inner = self.manager.pick_and_place(self.arm.robot, self.obj, self.target)[0]
            else:
                self.inner = self.manager.catch_from_here(
                    self.arm, self.obj, self.target, self.half, riding, tries=self.tries + 1
                )
            self.manager.end_approach(self)
        except Exception as exc:  # BehaviorError: the arm does not reach from here, and no other place does either
            if riding is not None and not self.waited and self.step != "waiting":
                self.note = str(exc)
                self._wait(float(data.time))  # it gets the box when that lies at the end of the belt
                return
            self._fail(str(exc))

    def _fail(self, note: str) -> None:
        self.note, self.failed, self.step = note, True, "failed"
        self.manager.end_approach(self)

    def interrupt(self, note: str) -> None:
        if not self.done:
            self._fail(note)
