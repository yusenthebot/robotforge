"""Control modes: the ways a robot can be run from the viewer's Modes tab.

A mode is either one of the robot's built-in behaviors (keyboard driving with its gait or wheels, a
learned walking policy, or hands off for the joint sliders) or a control-layer algorithm on a task
(MPC standing, MPPI walking, LQR balancing, IK reaching, ...). Which modes a robot gets depends on what
it is: a control mode is offered only when its task really fits the robot.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from robotforge.behaviors import Kind
from robotforge.behaviors.manager import MOBILE, BehaviorError

if TYPE_CHECKING:
    from robotforge.playground import Playground


@dataclass(frozen=True)
class Mode:
    key: str
    task: str | None = None  # control task; None for a built-in behavior
    method: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    drivable: bool = False  # the keyboard drives the robot in this mode
    follows: bool = False  # the robot travels on its own, so the camera should follow it
    then: str | None = None  # after `seconds` of sim time, switch to this mode (e.g. stop walking and stand)
    seconds: float = 0.0


DRIVE = Mode("drive", drivable=True)
LEARNED = Mode("learned", drivable=True)
RHYTHM = Mode("rhythm", drivable=True)  # a walk the robot learned itself (RL II lesson), next to its own gait
MANUAL = Mode("manual")

# In the order the viewer lists them; each is offered only where its task fits the robot.
CONTROL: tuple[Mode, ...] = (
    Mode("hold", "stand", "hold"),
    Mode("stand_mpc", "stand", "mpc"),
    Mode("walk_mppi", "walk", "mppi", {"speed": 0.3}, follows=True, then="stand_mpc", seconds=8.0),
    Mode("balance_lqr", "balance", "lqr"),
    Mode("balance_mpc", "balance", "mpc"),
    Mode("balance_none", "balance", "none"),
    Mode("reach_ik", "reach", "ik"),
    Mode("reach_cartesian", "reach", "cartesian"),
    Mode("reach_mpc", "reach", "mpc"),
    Mode("goto_mpc", "drive_to", "mpc", follows=True),
    Mode("hover_lqr", "hover", "lqr"),
    Mode("hover_mpc", "hover", "mpc"),
)
MODES: dict[str, Mode] = {m.key: m for m in (DRIVE, LEARNED, RHYTHM, MANUAL, *CONTROL)}

# Robots each task is offered for. A task's own check is looser (stand fits anything that moves freely,
# a rover included), so the viewer only lists pairs that make sense and have been tried.
TASK_KINDS: dict[str, frozenset[Kind]] = {
    "stand": frozenset({Kind.QUADRUPED}),
    "walk": frozenset({Kind.QUADRUPED}),
    "balance": frozenset({Kind.BALANCER, Kind.OTHER}),
    "reach": frozenset({Kind.ARM}),
    "drive_to": frozenset({Kind.ROVER}),
    "hover": frozenset({Kind.AIRCRAFT}),
}


RANDOM_POINT = {"reach": "target", "drive_to": "goal"}  # tasks that aim at a random point when given none


class ModeError(ValueError):
    pass


def available(playground: Playground, robot: dict[str, Any], learned: bool) -> list[str]:
    """Mode keys for one robot from `BehaviorManager.robots()`: built-in behaviors first, then algorithms."""
    from robotforge.control.tasks import TaskError, make_task

    try:
        kind = Kind(robot["kind"])
    except ValueError:  # a gripper on its own is told as "gripper": it has sliders, no behavior
        kind = Kind.OTHER
    kept = "learned_walk" in robot  # robots no gait code knows drive with their kept walk
    keys = ["drive"] if (kind in MOBILE or kept) and robot.get("max_speed") else ["manual"]
    if learned:
        keys.append("learned")
    if kept and kind in MOBILE:
        keys.append("rhythm")
    model = playground.session.model
    fits: dict[str, bool] = {}
    for mode in CONTROL:
        if kind not in TASK_KINDS[mode.task]:
            continue
        if mode.task not in fits:
            try:
                make_task(mode.task, model, robot["name"])
                fits[mode.task] = True
            except TaskError:
                fits[mode.task] = False
        if fits[mode.task]:
            keys.append(mode.key)
    return keys


def apply(playground: Playground, robot: str, key: str) -> dict[str, Any]:
    """Switch `robot` to mode `key`. Control modes start from the current state and keep running."""
    mode = MODES.get(key)
    if mode is None:
        raise ModeError(f"unknown mode {key!r}; modes: {', '.join(MODES)}")
    behaviors = playground.behaviors
    session = playground.session
    try:
        if mode in (DRIVE, LEARNED, RHYTHM):
            walker = None if behaviors.kind(robot) not in (Kind.QUADRUPED, Kind.HUMANOID) else "gait"
            if mode is LEARNED:
                walker = "rl"
            elif mode is RHYTHM:
                walker = "rhythm"
            behaviors.move(robot, 0.0, 0.0, walker)
            return {"robot": robot, "mode": key}
        if mode is MANUAL:
            behaviors.kind(robot)  # a clear error for unknown robots
            session.set_controller(robot, None)
            return {"robot": robot, "mode": key}
    except BehaviorError as exc:
        raise ModeError(str(exc)) from exc
    return _attach(playground, robot, mode)


def _attach(playground: Playground, robot: str, mode: Mode) -> dict[str, Any]:
    from robotforge.control.methods import make_controller
    from robotforge.control.tasks import make_task

    session = playground.session
    with session.lock:
        model, snapshot, version = session.model, copy.copy(session.data), session.version
    params = dict(mode.params)
    point = RANDOM_POINT.get(mode.task)
    if point is not None:
        # Tasks draw their random point from a fixed seed; pick a fresh one so each click aims somewhere new.
        probe = make_task(mode.task, model, robot, params)
        probe.reset(snapshot, np.random.default_rng())
        params[point] = [round(float(v), 4) for v in getattr(probe, point)]
    task = make_task(mode.task, model, robot, params)
    task.begin(snapshot)
    controller = make_controller(mode.method, task, model, snapshot, None)
    if mode.then is not None:
        after = MODES[mode.then]
        later_task = make_task(after.task, model, robot, dict(after.params))
        later_task.begin(snapshot)
        later = make_controller(after.method, later_task, model, snapshot, None)
        controller = Then(controller, later, later_task, float(snapshot.time) + mode.seconds)
    with session.lock:
        if session.version != version:
            raise ModeError("the world changed while the controller was being set up; pick the mode again")
        task.begin(session.data)
        if isinstance(controller, Then):
            controller.until = float(session.data.time) + mode.seconds
        session.set_controller(robot, controller)
    playground.modes[robot] = (mode.key, controller)
    described = task.describe()
    marker = described.get("target") or described.get("goal")
    return {"robot": robot, "mode": mode.key, "follows": mode.follows, **({"marker": marker} if marker else {})}


class Then:
    """Run one controller until a moment in sim time, then hand over to another (walk a while, then stand)."""

    def __init__(self, first: Any, second: Any, second_task: Any, until: float) -> None:
        self.first, self.second, self.second_task, self.until = first, second, second_task, until
        self.switched = False

    @property
    def rate_hz(self) -> float:
        return (self.second if self.switched else self.first).rate_hz

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.switched and data.time >= self.until:
            self.switched = True
            self.second_task.begin(data)  # stand where it stopped, not where it started
        (self.second if self.switched else self.first).update(model, data)

    def status(self) -> dict[str, Any]:
        return {**(self.second if self.switched else self.first).status(), "then": self.switched}


def current(playground: Playground) -> dict[str, str]:
    """{robot: mode key} for robots whose controller the viewer can name; robots with none are left out."""
    from robotforge.behaviors.manager import _is_mover
    from robotforge.behaviors.rhythm_walk import RhythmWalker

    out = {}
    for robot, ctrl in list(playground.session.controllers.items()):
        tracked = playground.modes.get(robot)
        if tracked is not None and tracked[1] is ctrl:
            mode = MODES[tracked[0]]
            out[robot] = mode.then if isinstance(ctrl, Then) and ctrl.switched and mode.then else tracked[0]
        elif isinstance(ctrl, RhythmWalker):
            try:
                out[robot] = "rhythm" if playground.behaviors.kind(robot) in MOBILE else "drive"
            except BehaviorError:
                continue
        elif _is_mover(ctrl):
            out[robot] = "learned" if ctrl.status().get("walker") == "rl" else "drive"
    return out
