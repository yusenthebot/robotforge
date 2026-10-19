"""Catching something that rides by: an arm picks a box off a running conveyor without stopping the belt.

Where to meet it is worked out from how it moves (its velocity, straight on along the belt, and where something in
its way would stop it): the first point on its way the arm reaches, pointing down, from over it down to it and along
it for as long as the catch takes, that it gets to no sooner than the arm can be waiting over it. The arm goes there
and waits; as the box comes, the hand comes down over the meeting point so that it is there just as the box is, then
follows the box along at its speed (an IK solve every step, the hand square to the box with its fingers either side
of its way), closes on it (or turns its cup on) and lifts it off the belt, slowing to a stop. Then whatever is to
happen next takes the arm back (`then`): putting it down where it goes, or folding it over a robot that will carry it.
A box that stops before it gets there (against the end stop) is picked up where it stops, as anything is.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.arm import DOWN, PAD_PRESS, REACH_TOL, ArmMotion

log = logging.getLogger(__name__)

MOVING = 0.01  # m/s: a thing slower than this lies still and is picked up as anything is
WAIT_ABOVE = 0.08  # m over the grasp the hand waits at
NEAR_ABOVE = 0.02  # m over the grasp the hand is as the box gets to the meeting point
DESCEND_S = 0.8  # s the hand takes from waiting to just over the meeting point
SETTLE_S = 0.25  # s it then takes down onto the box, following it
CLOSE_S = 0.5  # s the fingers close (a cup takes hold faster)
CUP_S = 0.3
LIFT_S = 0.7  # s it takes the box up off the belt, slowing to a stop
LIFT = 0.1  # m it lifts it
SLOW = 0.15  # s: the time constant the hand slows down with once the box is off the belt
ONTO = 0.01  # m short of the meeting point the box may be as the hand goes on down onto it
LAG = 0.04  # s the hand aims ahead of the box, for the servos' lag
JOINT_SPEED = 1.0  # rad/s of the furthest joint, going to wait over the meeting point
SEARCH_STEP = 0.02  # m between the points along its way tried for the meeting
LOOK_AHEAD = 4.0  # m along its way, at most
MARGIN_S = 0.6  # s spare between the arm being there and the box coming
LATE = 3.0  # s after the box should have come that the arm gives up waiting
MAX_PAST = 30  # loose things a look ahead passes through at most
HELD = 0.03  # m: a box whose middle is within its half height and this of the tool point is in the hand


class CatchError(ValueError):
    """Why it cannot be caught where it goes by, saying what to do instead."""


def velocity(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray | None:
    """How fast a loose object moves over the floor (m/s, world x and y), or None when it lies still or is not
    loose."""
    joint = int(model.body_jntadr[body])
    if joint < 0 or int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
        return None
    v = np.array(data.qvel[model.jnt_dofadr[joint] : model.jnt_dofadr[joint] + 2], float)
    return v if float(np.linalg.norm(v)) >= MOVING else None


@dataclass(frozen=True)
class Meeting:
    """Where and when the arm meets the box: the point on its way, the box's velocity there, the grasp height, the
    joint values waiting over it and the hand's turn (None: pointing down, turned any way)."""

    point: np.ndarray  # x, y on the box's way, z of the grasp
    velocity: np.ndarray  # x, y
    q_wait: np.ndarray
    rotation: np.ndarray | None
    arrives: float  # sim time the box gets there
    lifted: np.ndarray  # where the tool is once it has lifted the box off the belt
    q_lifted: np.ndarray


def meeting(
    model: mujoco.MjModel, data: mujoco.MjData, arm: ArmMotion, body: int, half: float, v: np.ndarray
) -> Meeting:
    """The first point on the box's way where the arm can catch it (see the module), or CatchError."""
    arm.sync(data)
    speed = float(np.linalg.norm(v))
    way = v / speed
    start = np.array(data.xpos[body], float)
    grasp_z = float(start[2]) + (half - PAD_PRESS if arm.tool is not None else 0.0)
    stop = _clear_ahead(model, data, body, way)
    rotation = _square(model, data, arm, body, way)
    closing = speed * (SETTLE_S + (CUP_S if arm.tool is not None else CLOSE_S))  # how far it rides, held
    window = closing + speed * SLOW  # and on as the hand slows to a stop, lifting it
    q = arm.q.copy()
    best_late: float | None = None
    # Up to where it stops: one caught as it comes to a stop against the boxes queued ahead of it is held where it
    # stops (the hand follows it), so the catch need not be over before then.
    for s in np.arange(SEARCH_STEP, min(LOOK_AHEAD, stop), SEARCH_STEP):
        p = np.array([*(start[:2] + way * s), grasp_z])
        q_wait = _reach(arm, p + [0.0, 0.0, WAIT_ABOVE], rotation, q)
        if q_wait is None:
            continue
        lifted = np.array([*(start[:2] + way * min(s + window, stop)), grasp_z + LIFT])
        held = np.array([*(start[:2] + way * min(s + closing, stop)), grasp_z])
        ends = (p + [0.0, 0.0, NEAR_ABOVE], p, held, lifted)
        qs: np.ndarray | None = q_wait
        for point in ends:
            qs = _reach(arm, point, rotation, qs)
            if qs is None:
                break
        if qs is None:
            continue
        ready = float(np.max(np.abs(q_wait - arm.q))) / JOINT_SPEED + MARGIN_S + DESCEND_S
        if s / speed >= ready:
            return Meeting(p, v.copy(), q_wait, rotation, float(data.time) + s / speed, lifted, qs)
        best_late = best_late if best_late is not None else s
    if best_late is not None:
        raise CatchError(
            f"{arm.robot} reaches the box's way only {best_late:.2f} m ahead of it, which it gets to sooner than the "
            "arm can be there: slow the belt down (set_controls) or move the arm further along it"
        )
    raise CatchError(
        f"{arm.robot} reaches nowhere along the way the box rides before it stops ({stop:.2f} m ahead): move the arm "
        "or the robot beside the belt further along, or stop the belt and pick it up where it is"
    )


def _clear_ahead(model: mujoco.MjModel, data: mujoco.MjData, body: int, way: np.ndarray) -> float:
    """How far the box's middle goes along `way` before something fixed stops it (the end stop across a belt): a ray
    from its middle that passes through the loose things ahead of it, which ride along too and queue up against the
    stop before it, less their lengths and half its own; LOOK_AHEAD when nothing stops it. Through the boxes ahead as
    though they were not there, the factory arm waited for a box over a point it stopped 5 cm short of, behind two
    boxes waiting at the stop."""
    start = np.array(data.xpos[body], float)
    direction = np.array([way[0], way[1], 0.0])
    hit = np.zeros(1, np.int32)
    gone, exclude, pile = 0.0, body, _length(model, data, body, direction) / 2
    for _ in range(MAX_PAST):
        far = mujoco.mj_ray(model, data, start + direction * gone, direction, None, 1, exclude, hit)
        if far < 0 or gone + far > LOOK_AHEAD:
            return LOOK_AHEAD
        other = int(model.geom_bodyid[hit[0]])
        if other == 0 or int(model.body_weldid[other]) == 0:  # fixed: it stops there
            return max(0.0, gone + float(far) - pile)
        gone += float(far) + 1e-4  # a loose thing: on through it, counting its length
        pile += _length(model, data, other, direction)
        exclude = other
    return max(0.0, gone - pile)


def _length(model: mujoco.MjModel, data: mujoco.MjData, body: int, direction: np.ndarray) -> float:
    """How long a body's shapes are along `direction`."""
    half = 0.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body:
            continue
        rot = data.geom_xmat[g].reshape(3, 3)
        half = max(half, float(np.abs(rot.T @ direction) @ model.geom_aabb[g, 3:]))
    return 2 * half


def _square(model: mujoco.MjModel, data: mujoco.MjData, arm: ArmMotion, body: int, way: np.ndarray):
    """The hand's turn over the box: pointing down, its fingers closing across the box's way (they pass either side
    of it as it comes), square to the box's faces; None for a cup, or a hand whose closing is not known."""
    if arm.tool is not None or arm.closing is None:
        return None
    rot = data.xmat[body].reshape(3, 3)
    across = np.array([-way[1], way[0], 0.0])
    faces = [rot[:, 0], rot[:, 1]]
    n = max(faces, key=lambda f: abs(float(f @ across)))
    n = np.array([n[0], n[1], 0.0])
    n /= max(float(np.linalg.norm(n)), 1e-9)
    t, c = arm.tool_axis / np.linalg.norm(arm.tool_axis), arm.closing
    local = np.column_stack([t, c, np.cross(t, c)])
    world = np.column_stack([DOWN, n, np.cross(DOWN, n)])
    return world @ local.T


def _reach(arm: ArmMotion, point: np.ndarray, rotation: np.ndarray | None, q0: np.ndarray) -> np.ndarray | None:
    """Joint values putting the tool at `point` (pointing down, turned `rotation`), from `q0` and then from the arm's
    own starts; None if it does not reach, or only into itself or what carries it."""
    starts = [q0]
    seeded, err, _ = arm._solve(point)
    if err <= REACH_TOL:
        starts.append(seeded)
    for start in starts:
        if rotation is None:
            q, err = arm.ik.solve(arm.chain, point, start, iterations=60, axis=(arm.tool_axis, DOWN))
        else:
            q, err = arm.ik.solve(arm.chain, point, start, iterations=60, rotation=rotation)
        if err <= REACH_TOL and not arm._collides(q):
            return q
    return None


@dataclass
class Catch:
    """The controller that catches one box (it takes the arm's place among the controllers until it is done, then
    hands the arm to `then`)."""

    arm: ArmMotion
    body: int
    half: float
    meet: Meeting
    then: Callable[[mujoco.MjData, bool], None] | None = None  # (data, caught) once it is done
    after: Any = None  # what `then` set going (putting it down, an errand to carry it): done when that is
    region: dict[str, Any] | None = None  # where the load counts as put (BehaviorManager.placed)
    rate_hz: float = 100.0
    step: str = "going"
    caught: bool = False
    note: str | None = None
    _t0: float | None = None
    _q0: np.ndarray | None = None
    _q: np.ndarray | None = None
    _phase_t: float = 0.0
    _from: np.ndarray | None = None  # where the tool was as a phase began
    _lift_v: np.ndarray = field(default_factory=lambda: np.zeros(2))
    caught_at: list[float] | None = None  # where the box was as the hand closed on it
    _turn: np.ndarray | None = None  # the hand's turn as it waited, kept while it follows

    @property
    def robot(self) -> str:
        return self.arm.robot

    @property
    def done(self) -> bool:
        if self.step == "missed":
            return True
        return self.step == "done" and (self.after is None or bool(self.after.done))

    @property
    def time_limit(self) -> float:
        return 60.0 + (180.0 if self.after is not None else 0.0)

    def result(self) -> dict[str, Any]:
        """How the catch went, and what came after (an errand's trips)."""
        out: dict[str, Any] = {"caught": self.caught}
        if self.caught_at is not None:
            out["caught_at"] = self.caught_at
        if self.note:
            out["note"] = self.note
        if self.after is not None and callable(getattr(self.after, "result", None)):
            out.update(self.after.result())
        return out

    def status(self) -> dict[str, Any]:
        return {"behavior": "catch", "robot": self.arm.robot, "step": self.step, "done": self.done}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            return
        try:
            self._update(model, data)
        except Exception as exc:  # an error here must end the catch, not every step of the simulation
            log.exception("%s's catch failed", self.arm.robot)
            self._end(data, False, f"the catch broke off on an internal error: {exc}")

    def _update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        arm, meet = self.arm, self.meet
        arm.sync(data)
        t = float(data.time)
        if self._t0 is None:
            self._t0, self._q0 = t, arm.q.copy()
            self._grip(data, "open")
        box = np.array(data.xpos[self.body], float)
        way = meet.velocity / float(np.linalg.norm(meet.velocity))
        if self.step == "going":
            span = max(float(np.max(np.abs(meet.q_wait - self._q0))) / JOINT_SPEED, 0.5)
            u = min(1.0, (t - self._t0) / span)
            arm.q = self._q0 + (meet.q_wait - self._q0) * (u * u * (3 - 2 * u))
            if u >= 1.0:
                self.step = "waiting"
        elif self.step == "waiting":
            arm.q = meet.q_wait.copy()
            left = float((meet.point[:2] - box[:2]) @ way)  # how far the box still has to come
            speed = float(np.linalg.norm(self._box_velocity(model, data)))
            if speed < MOVING and left > 0.0:
                self._end(data, False, "the box stopped before it got to the arm; pick it up where it is")
                return
            if left <= float(np.linalg.norm(meet.velocity)) * DESCEND_S:
                self._phase("descending", t, arm)
            elif t > meet.arrives + LATE:
                self._end(data, False, "the box did not come by in time")
                return
        elif self.step == "descending":
            u = self._u(t, DESCEND_S)
            s = u * u * (3 - 2 * u)
            target = meet.point + [0.0, 0.0, NEAR_ABOVE + (WAIT_ABOVE - NEAR_ABOVE) * (1 - s)]
            if not self._track(arm, target):
                return self._end(data, False, "its hand could not follow the way down")
            # On down onto it once it is under the hand: a box that rides a little slower than the belt (it slips)
            # came 3 cm behind, and following it from there the hand jumped back.
            left = float((meet.point[:2] - box[:2]) @ way)
            if u >= 1.0 and left <= ONTO:
                self._phase("settling", t, arm)
            elif u >= 1.0 and t > meet.arrives + LATE:
                return self._end(data, False, "the box did not come by in time")
        elif self.step in ("settling", "closing"):
            u = self._u(t, SETTLE_S) if self.step == "settling" else 1.0
            ahead = box[:2] + meet.velocity * LAG
            target = np.array([*ahead, meet.point[2] + NEAR_ABOVE * (1 - u)])
            if not self._track(arm, target):
                return self._end(data, False, "its hand could not follow the box")
            if self.step == "settling" and u >= 1.0:
                self._grip(data, "close")
                self.caught_at = [round(float(v), 3) for v in box]
                self._phase("closing", t, arm)
            elif self.step == "closing" and t - self._phase_t >= (CUP_S if arm.tool is not None else CLOSE_S):
                self._lift_v = meet.velocity.copy()
                self._phase("lifting", t, arm)
        elif self.step == "lifting":
            u = self._u(t, LIFT_S)
            s = t - self._phase_t
            glide = self._lift_v * SLOW * (1 - math.exp(-s / SLOW))  # on along the belt, slowing to a stop
            target = self._from + np.array([*glide, LIFT * (u * u * (3 - 2 * u))])
            if not self._track(arm, target):
                return self._end(data, self._holds(data), "its hand could not lift the box the whole way")
            if u >= 1.0:
                self._end(data, self._holds(data), None if self._holds(data) else "it closed but did not hold it")
                return
        arm._command(data)

    # -- helpers --

    def _u(self, t: float, span: float) -> float:
        return min(1.0, (t - self._phase_t) / span)

    def _phase(self, step: str, t: float, arm: ArmMotion) -> None:
        self.step, self._phase_t = step, t
        self._from = arm.ik.point(arm.chain, arm.q).copy()

    def _track(self, arm: ArmMotion, target: np.ndarray) -> bool:
        """Solve the arm onto `target` from where it is told to be now; False if it no longer reaches."""
        # Turned as it is waiting, all the way down: left free to turn about the tool's axis, a cup arm's wrist spun
        # 0.2 rad in a step; held to the turn it was planned with, a Panda that had waited 0.1 rad off it swung its
        # shoulder 0.2 rad in a step to put it right.
        if self._turn is None:
            arm.ik.point(arm.chain, self.meet.q_wait)
            self._turn = arm.ik.data.xmat[arm.chain.body].reshape(3, 3).copy()
        q, err = arm.ik.solve(arm.chain, target, arm.q, iterations=30, rotation=self._turn)
        if err > 2 * REACH_TOL or float(np.max(np.abs(q - arm.q))) > 0.2:
            log.info(
                "%s lost the track: error %.4f m, step %.3f rad, to %s",
                arm.robot,
                err,
                float(np.max(np.abs(q - arm.q))),
                np.round(target, 3),
            )
            return False
        arm.q = q
        return True

    def _grip(self, data: mujoco.MjData, state: str) -> None:
        value = self.arm._grip(state)
        if value is not None:
            data.ctrl[self.arm.gripper] = value

    def _box_velocity(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        v = velocity(model, data, self.body)
        return np.zeros(2) if v is None else v

    def _holds(self, data: mujoco.MjData) -> bool:
        tool = data.site_xpos[self.arm.site]
        reach = self.half + HELD + (self.half if self.arm.tool is not None else 0.0)
        return float(np.linalg.norm(data.xpos[self.body] - tool)) <= reach

    def _end(self, data: mujoco.MjData, caught: bool, note: str | None) -> None:
        self.caught, self.note = caught, note
        if not caught:
            self._grip(data, "open")
        # What comes next is set going before the catch says it is done: read from another thread in between (a
        # demo waiting on it), done with nothing after it ended the wait while the box was still in the hand.
        if self.then is not None:
            self.then(data, caught)
        self.step = "done" if caught else "missed"


KEY = "__catch_then__"  # session controller key of what follows a catch: KEY + arm


@dataclass
class Then:
    """Once the arm is done folding the load over its robot, set what carries it going (`start`: an errand, or a
    note why there is none); done when that is."""

    arm: ArmMotion
    start: Callable[[], Any]
    rate_hz: float = 20.0
    hidden: bool = True
    after: Any = None
    note: str | None = None

    @property
    def done(self) -> bool:
        return self.note is not None or (self.after is not None and bool(self.after.done))

    def status(self) -> dict[str, Any]:
        return {"behavior": "catch", "robot": self.arm.robot, "step": "folding" if self.after is None else "carrying"}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.after is not None or self.note is not None or not self.arm.done:
            return
        started = self.start()
        if isinstance(started, str):
            self.note = started
        else:
            self.after = started

    def result(self) -> dict[str, Any]:
        if self.after is not None and callable(getattr(self.after, "result", None)):
            return self.after.result()
        return {"note": self.note} if self.note else {}


def load_radius(model: mujoco.MjModel, body: int) -> float:
    """How far a load's shapes reach from its middle, at most."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    return max((float(model.geom_rbound[g]) for g in geoms), default=0.03)
