"""Task-specific configuration for recorded live controller measurements."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from robotforge.experiments.execution import RecordedExecution
from robotforge.experiments.live import task_spec
from robotforge.sim.execution import RunLimits


class ControllerExecution(RecordedExecution):
    def __init__(self, playground: Any, watched: Callable[[], bool], wall_limit: Callable[[], float]):
        super().__init__(playground, watched, wall_limit, owner="run_controller", sim_seconds=60.0)

    def configure(self, task: Any, method: str, params: dict | None, seconds: float) -> None:
        from robotforge.experiments.live import _values

        state = self.call
        wall = seconds * 3 + 5 if self.watched() else state.wall
        # All phases and controller preparation consume this original deadline.
        state.run.limits = RunLimits(seconds, max(wall, 1e-12))
        state.run.deadline = state.run.started + wall
        state.wall = wall
        state.record.update(
            task=task_spec(task),
            algorithm={"domain": "control", "name": method, "params": _values(params or {})},
            seeds={"reset": None, "mode": "current_state"},
            budget={"sim_seconds": seconds, "wall_seconds": wall},
        )
