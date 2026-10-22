"""Built-in tasks: objectives that every algorithm (classic control, MPC, RL) can be pointed at."""

from __future__ import annotations

from typing import Any

import mujoco

from robotforge.control.tasks.balance import Balance
from robotforge.control.tasks.base import Batch, Robot, StateLayout, Task, TaskError, TaskParams
from robotforge.control.tasks.contact import Press, Wipe
from robotforge.control.tasks.drive import DriveTo
from robotforge.control.tasks.hover import Hover
from robotforge.control.tasks.insert import Insert
from robotforge.control.tasks.legged import Stand, Walk
from robotforge.control.tasks.reach import Reach
from robotforge.control.tasks.wheel_balance import WheelBalance as WheelBalance

TASKS: dict[str, type[Task]] = {t.name: t for t in (Reach, Balance, Stand, Walk, DriveTo, Hover, Press, Wipe, Insert)}


def make_task(
    name: str,
    model: mujoco.MjModel,
    robot: str,
    params: dict[str, Any] | None = None,
    *,
    world: Any = None,
) -> Task:
    """Bind a built-in, saved-world or plugin task through the shared registry."""
    from robotforge.control.task_registry import bind_task

    return bind_task(name, model, robot, params, world=world)


def catalog(*, world: Any = None) -> list[dict[str, Any]]:
    """Task metadata and full parameter schemas, without invoking plugin factories."""
    from robotforge.control.task_registry import task_definitions

    return [definition.describe() for definition in task_definitions(world).values()]


__all__ = ["TASKS", "Batch", "Robot", "StateLayout", "Task", "TaskError", "TaskParams", "catalog", "make_task"]
