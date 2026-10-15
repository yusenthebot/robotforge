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


DRIVE = Mode("drive", drivable=True)
LEARNED = Mode("learned", drivable=True)
MANUAL = Mode("manual")

# In the order the viewer lists them; each is offered only where its task fits the robot.
CONTROL: tuple[Mode, ...] = (
    Mode("hold", "stand", "hold"),
    Mode("stand_mpc", "stand", "mpc"),
    Mode("walk_mppi", "walk", "mppi", {"speed": 0.3}, follows=True),
    Mode("balance_lqr", "balance", "lqr"),
    Mode("balance_mpc", "balance", "mpc"),
    Mode("balance_none", "balance", "none"),
    Mode("reach_ik", "reach", "ik"),
    Mode("reach_cartesian", "reach", "cartesian"),
    Mode("reach_mpc", "reach", "mpc"),
    Mode("goto_mpc", "drive_to", "mpc", follows=True),
)
MODES: dict[str, Mode] = {m.key: m for m in (DRIVE, LEARNED, MANUAL, *CONTROL)}

# Robots each task is offered for. A task's own check is looser (stand fits anything that moves freely,
# a rover included), so the viewer only lists pairs that make sense and have been tried.
TASK_KINDS: dict[str, frozenset[Kind]] = {
    "stand": frozenset({Kind.QUADRUPED}),
    "walk": frozenset({Kind.QUADRUPED}),
    "balance": frozenset({Kind.BALANCER, Kind.OTHER}),
    "reach": frozenset({Kind.ARM}),
    "drive_to": frozenset({Kind.ROVER}),
}


RANDOM_POINT = {"reach": "target", "drive_to": "goal"}  # tasks that aim at a random point when given none


class ModeError(ValueError):
    pass


def available(playground: Playground, robot: dict[str, Any], learned: bool) -> list[str]:
    """Mode keys for one robot from `BehaviorManager.robots()`: built-in behaviors first, then algorithms."""
    from robotforge.control.tasks import TaskError, make_task

    kind = Kind(robot["kind"])
    keys = ["drive"] if kind in MOBILE and robot.get("max_speed") else ["manual"]
    if learned:
        keys.append("learned")
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
        if mode is DRIVE or mode is LEARNED:
            walker = None if behaviors.kind(robot) not in (Kind.QUADRUPED, Kind.HUMANOID) else "gait"
            if mode is LEARNED:
                walker = "rl"
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
    with session.lock:
        if session.version != version:
            raise ModeError("the world changed while the controller was being set up; pick the mode again")
        task.begin(session.data)
        session.set_controller(robot, controller)
    playground.modes[robot] = (mode.key, controller)
    described = task.describe()
    marker = described.get("target") or described.get("goal")
    return {"robot": robot, "mode": mode.key, "follows": mode.follows, **({"marker": marker} if marker else {})}


def current(playground: Playground) -> dict[str, str]:
    """{robot: mode key} for robots whose controller the viewer can name; robots with none are left out."""
    from robotforge.behaviors.manager import _is_mover

    out = {}
    for robot, ctrl in list(playground.session.controllers.items()):
        tracked = playground.modes.get(robot)
        if tracked is not None and tracked[1] is ctrl:
            out[robot] = tracked[0]
        elif _is_mover(ctrl):
            out[robot] = "learned" if ctrl.status().get("walker") == "rl" else "drive"
    return out
