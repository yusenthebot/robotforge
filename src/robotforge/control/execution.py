"""Owned execution for live controller measurements; other adapters keep their existing clocks."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import tempfile
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from robotforge.experiments import ExperimentHandle, ExperimentStore
from robotforge.experiments.live import experiment_home, task_spec
from robotforge.sim.execution import EndReason, ExecutionError, ExecutionRun, RunLimits

T = TypeVar("T")
log = logging.getLogger(__name__)


@dataclass
class _Call:
    run: ExecutionRun
    record: ExperimentHandle
    wall: float
    robot: str | None = None
    controller: Any = None


class ControllerExecution:
    def __init__(self, playground: Any, watched: Callable[[], bool], wall_limit: Callable[[], float]):
        self.pg = playground
        self.session = playground.session
        self.watched = watched
        self.wall_limit = wall_limit
        self._call: ContextVar[_Call | None] = ContextVar("controller_execution", default=None)

    @property
    def call(self) -> _Call:
        value = self._call.get()
        if value is None:
            raise RuntimeError("controller measurement has no owned execution")
        return value

    def owned(self, invoke: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        @functools.wraps(invoke)
        async def execute(*args: Any, **kwargs: Any) -> str:
            # Configure the task's actual duration after binding, without starting a new wall deadline.
            wall = 185.0 if self.watched() else float(self.wall_limit())
            with self.session.execution.run("run_controller", RunLimits(60.0, max(wall, 1e-12)), self.watched) as run:
                directory = self.pg.store.dir if self.pg.store is not None else None
                record = ExperimentStore(experiment_home(directory)).begin(
                    "execution",
                    environment={"entrypoint": "mcp_run_controller", "generation": run.generation},
                    budget={"sim_seconds": 60.0, "wall_seconds": wall, "includes_preparation": True},
                )
                state = _Call(run, record, wall)
                token = self._call.set(state)
                error: BaseException | None = None
                try:
                    with self.session.lock:
                        run.check_valid()
                        record.snapshot_environment(
                            self.session.model, self.session.data, world=self.pg.current_world(), scope="execution"
                        )
                    output = await invoke(*args, **kwargs)
                except BaseException as exc:
                    error = exc
                    reason = (
                        EndReason.CANCELLED
                        if isinstance(exc, asyncio.CancelledError)
                        else exc.reason
                        if isinstance(exc, ExecutionError)
                        else EndReason.FAILED
                    )
                    run.reason = reason
                    while True:
                        try:
                            await run.drain_workers(cancel_pending=True)
                            break
                        except asyncio.CancelledError:
                            # Repeated cancellation must not detach a still-running preparation thread.
                            continue
                    # A reset or replacement owns its own controller. Never clean it up through stale IDs.
                    with self.session.lock:
                        if (
                            run.current
                            and state.controller is not None
                            and self.session.controllers.get(state.robot) is state.controller
                        ):
                            self.pg.behaviors.stop(state.robot)
                    raise
                finally:
                    try:
                        failure = None
                        try:
                            self._finish(state, error)
                        except Exception as exc:
                            failure = str(exc)
                            log.exception("could not finish controller execution %s", record.record.id)
                            try:
                                record.finish(
                                    "cancelled" if run.reason == EndReason.CANCELLED else "failed",
                                    reason="cancelled" if run.reason == EndReason.CANCELLED else "recording_failed",
                                    error=f"recording_failed: {failure}",
                                )
                            except Exception:
                                log.exception("could not persist controller execution recording failure")
                    finally:
                        self._call.reset(token)
                if failure is not None:
                    # Applied commands still return their normal result; recording failure is explicit.
                    value = json.loads(output)
                    value["execution"]["recording_error"] = failure
                    from robotforge import wording

                    output = wording.dumps(value)
                return output

        @functools.wraps(invoke)
        async def translated(*args: Any, **kwargs: Any) -> str:
            # The SDK wraps unexpected errors before _Server sees them. Translate only after lease cleanup.
            from mcp.server.mcpserver.exceptions import ToolError

            try:
                return await execute(*args, **kwargs)
            except ExecutionError as exc:
                raise ToolError(str(exc)) from exc

        return translated

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

    async def prepare(self, invoke: Callable[..., T], *args: Any) -> T:
        run = self.call.run
        value = await run.call_snapshot_thread(invoke, *args)
        run.check()
        return value

    def attached(self, robot: str, controller: Any, experiment: dict[str, str]) -> None:
        state = self.call
        state.robot, state.controller = robot, controller
        state.run.resources.add(f"robot:{robot.split('/', 1)[0]}")
        state.record.update(origins=[experiment])

    async def advance(self, done: Callable[[], bool], seconds: float) -> dict[str, Any]:
        state = self.call

        def checked() -> bool:
            if self.session.controllers.get(state.robot) is not state.controller:
                raise ExecutionError(EndReason.INVALIDATED, "the robot's controller changed during the run")
            return done()

        try:
            result = await state.run.advance(checked, seconds)
        except ExecutionError as exc:
            if exc.reason == EndReason.INVALIDATED and not state.run.current:
                raise ExecutionError(
                    exc.reason, "the world was reset during the run, which removed the controller; run it again"
                ) from exc
            raise
        values: dict[str, Any] = {"sim_seconds": result.sim_seconds, "shown_live": result.shown_live}
        if result.reason in (EndReason.WALL_LIMIT, EndReason.SIM_LIMIT, EndReason.STEP_LIMIT):
            values["stopped_early"] = (
                f"stopped after {state.wall:.0f} s of computer time; the controller is too slow for this machine "
                "(for MPC use fewer samples or a shorter horizon)"
                if result.reason == EndReason.WALL_LIMIT
                else f"stopped at the execution {result.reason}"
            )
        return values

    def reference(self) -> dict[str, str]:
        return self.call.record.reference()

    def _finish(self, state: _Call, error: BaseException | None) -> None:
        run, record = state.run, state.record
        result = run.result(run.reason)
        values = {
            "reason": result.reason.value,
            "steps": result.steps,
            "sim_seconds": result.sim_seconds,
            "wall_seconds": result.wall_seconds,
            "shown_live": result.shown_live,
            "generation_valid": run.current,
        }
        record.update(metrics={"namespace": "owned_execution", **values}, budget_used=values)
        with self.session.lock:
            if run.current:
                with tempfile.TemporaryDirectory(prefix="robotforge-execution-") as folder:
                    path = Path(folder) / "final.npz"
                    np.savez(
                        path, qpos=self.session.data.qpos, qvel=self.session.data.qvel, ctrl=self.session.data.ctrl
                    )
                    record.artifact(path, "final_physics")
            else:
                values["final_state"] = "unavailable_after_invalidation; see the attachment's final_physics"
        record.update(metrics={"namespace": "owned_execution", **values}, budget_used=values)
        status = (
            "cancelled"
            if result.reason == EndReason.CANCELLED
            else "stopped"
            if result.reason in (EndReason.WALL_LIMIT, EndReason.SIM_LIMIT, EndReason.STEP_LIMIT)
            else "failed"
            if error is not None or result.reason in (EndReason.INVALIDATED, EndReason.UNSTABLE)
            else "completed"
        )
        record.finish(status, reason=result.reason.value, error=str(error) if error is not None else None)
