"""Built-in tasks: objectives that every algorithm (classic control, MPC, RL) can be pointed at."""

from __future__ import annotations

from typing import Any

import mujoco

from robotforge import wording
from robotforge.control.tasks.balance import Balance
from robotforge.control.tasks.base import Batch, Robot, StateLayout, Task, TaskError, TaskParams
from robotforge.control.tasks.contact import Press, Wipe
from robotforge.control.tasks.drive import DriveTo
from robotforge.control.tasks.hover import Hover
from robotforge.control.tasks.legged import Stand, Walk
from robotforge.control.tasks.reach import Reach
from robotforge.control.tasks.wheel_balance import WheelBalance

TASKS: dict[str, type[Task]] = {t.name: t for t in (Reach, Balance, Stand, Walk, DriveTo, Hover, Press, Wipe)}


def make_task(name: str, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None) -> Task:
    """Bind task `name` to `robot` in `model`. Raises TaskError with advice if it does not fit."""
    try:
        cls = TASKS[name]
    except KeyError:
        raise TaskError(f"unknown task {wording.short(name)}; choose one of: {', '.join(TASKS)}") from None
    if cls is Balance and WheelBalance.fits(model, robot):  # a two-wheeled robot balances on its wheels
        cls = WheelBalance
    return cls(model, robot, params)


def catalog() -> list[dict[str, Any]]:
    """Task names, what they ask for, which robots and algorithms fit, and their parameters."""
    out = []
    for t in TASKS.values():
        schema = t.Params.model_json_schema()
        params = {
            key: {k: v for k, v in prop.items() if k in ("type", "default", "description", "minimum", "maximum")}
            for key, prop in schema.get("properties", {}).items()
        }
        out.append(
            {"name": t.name, "summary": t.summary, "robot": t.robot_hint, "methods": list(t.methods), "params": params}
        )
    return out


__all__ = ["TASKS", "Batch", "Robot", "StateLayout", "Task", "TaskError", "TaskParams", "catalog", "make_task"]
