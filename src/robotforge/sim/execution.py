"""Session-scoped execution: one clock, cumulative budgets and safe invalidation.

Controllers still choose actions and adapters still describe their results. Watching a run only paces its
steps; the viewer never becomes its clock. This module does not know about MCP, scripts or subprocesses.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from robotforge.sim.session import SimSession

CHUNK_SECONDS = 0.02
YIELD_SECONDS = 1 / 60
T = TypeVar("T")


class EndReason(StrEnum):
    COMPLETED = "completed"
    DURATION = "duration"
    SIM_LIMIT = "sim_limit"
    STEP_LIMIT = "step_limit"
    WALL_LIMIT = "wall_limit"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"
    UNSTABLE = "unstable"
    FAILED = "failed"


class ExecutionError(ValueError):
    def __init__(self, reason: EndReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class RunLimits:
    sim_seconds: float | None = None
    wall_seconds: float | None = None
    steps: int | None = None

    def __post_init__(self) -> None:
        for name in ("sim_seconds", "wall_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.steps is not None and (type(self.steps) is not int or self.steps <= 0):
            raise ValueError("steps must be a positive integer")


@dataclass(frozen=True)
class RunResult:
    reason: EndReason
    sim_seconds: float
    wall_seconds: float
    steps: int
    shown_live: bool


class ExecutionKernel:
    def __init__(self, session: SimSession) -> None:
        self.session = session
        self.active: ExecutionRun | None = None
        self.last_result: RunResult | None = None
        self._commands: dict[object, str] = {}
        self._pending: set[asyncio.Task] = set()

    def run(self, owner: str, limits: RunLimits, watched: Callable[[], bool] = lambda: False) -> ExecutionRun:
        """Create a lease. Enter it synchronously before changing controllers or awaiting anything."""
        return ExecutionRun(self, owner, limits, watched)

    @contextlib.contextmanager
    def command(self, name: str) -> Iterator[None]:
        """Admit a legacy command without taking over its viewer-driven clock.

        Its work may run on another thread. Keep admission until that work actually ends so a queued planner
        cannot start mutating a model after a new execution has acquired it.
        """
        token = object()
        with self.session.lock:
            self.require_idle()
            self._commands[token] = name
        try:
            yield
        finally:
            with self.session.lock:
                self._commands.pop(token, None)

    async def call_command(self, name: str, invoke: Callable[[], Awaitable[T]]) -> T:
        admission = self.command(name)
        admission.__enter__()
        try:
            task = asyncio.ensure_future(invoke())
        except BaseException:
            admission.__exit__(None, None, None)
            raise
        self._pending.add(task)

        def finished(task: asyncio.Task) -> None:
            admission.__exit__(None, None, None)
            self._pending.discard(task)
            if not task.cancelled():
                task.exception()  # retrieve failures even if the requesting client already cancelled

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    def require_idle(self) -> None:
        with self.session.lock:
            if self.session.clock_owner is not None or self.active is not None:
                raise ExecutionError(
                    EndReason.INVALIDATED,
                    f"the simulation clock is already owned by {self.session.clock_owner}; "
                    "wait for the running operation to finish",
                )

    def check_step(self, owner: ExecutionRun | None) -> None:
        """Called under the session lock, including each step of an owned batch."""
        if owner is None:
            if self.active is not None or self.session.clock_owner is not None:
                self.require_idle()
        elif self.active is not owner:
            raise ExecutionError(EndReason.INVALIDATED, "this execution no longer owns the simulation clock")
        else:
            owner.check()


class ExecutionRun:
    def __init__(self, kernel: ExecutionKernel, owner: str, limits: RunLimits, watched: Callable[[], bool]) -> None:
        self.kernel = kernel
        self.session = kernel.session
        self.owner = owner
        self.limits = limits
        self.watched = watched
        self.sim_seconds = 0.0
        self.steps = 0
        self.shown_live = False
        self.reason = EndReason.COMPLETED
        self._interrupted: str | None = None
        self._invalidated: str | None = None
        self._entered = False
        self.resources: set[str] = set()
        self._workers: set[asyncio.Task] = set()
        self._workers_stopping = False
        self._closing = False

    def __enter__(self) -> ExecutionRun:
        with self.session.lock:
            self.kernel.require_idle()
            if self.kernel._commands:
                names = ", ".join(sorted(set(self.kernel._commands.values())))
                raise ExecutionError(
                    EndReason.INVALIDATED, f"the simulation is busy with {names}; wait for it to finish"
                )
            if self._entered:
                raise ValueError("an execution lease can only be entered once")
            self._entered = True
            self.generation = self.session.generation
            self._model, self._data = self.session.model, self.session.data
            self._last_time = float(self.session.data.time)
            self.timestep = float(self._model.opt.timestep)
            self.started = time.monotonic()
            self.deadline = self.started + (self.limits.wall_seconds or math.inf)
            self.kernel.active = self
            self.session.clock_owner = self.owner
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, _tb: object) -> None:
        if isinstance(exc, asyncio.CancelledError):
            self.reason = EndReason.CANCELLED
        elif isinstance(exc, ExecutionError):
            self.reason = exc.reason
        elif exc is not None:
            self.reason = EndReason.FAILED
        self._closing = True
        self._workers_stopping = True
        if not self._workers:
            self._release()

    def _release(self) -> None:
        with self.session.lock:
            # A stale finally must never release another operation's lease.
            if self.kernel.active is self:
                self.kernel.last_result = self.result(self.reason)
                self.kernel.active = None
                self.session.clock_owner = None

    @property
    def current(self) -> bool:
        return (
            self.kernel.active is self
            and self.session.clock_owner == self.owner
            and self.session.generation == self.generation
            and self.session.model is self._model
            and self.session.data is self._data
        )

    def check_valid(self) -> None:
        with self.session.lock:
            if not self.current:
                self.reason = EndReason.INVALIDATED
                raise ExecutionError(
                    EndReason.INVALIDATED,
                    "the world was reset or changed, or this execution lost the clock; run it again",
                )
            if self._interrupted is not None:
                self.reason = EndReason.CANCELLED
                raise ExecutionError(EndReason.CANCELLED, self._interrupted)
            if self._invalidated is not None:
                self.reason = EndReason.INVALIDATED
                raise ExecutionError(EndReason.INVALIDATED, self._invalidated)

    def interrupt(self, message: str = "the execution was stopped; run it again to continue") -> None:
        with self.session.lock:
            self._interrupted = message
            self.reason = EndReason.CANCELLED

    def invalidate(self, message: str) -> None:
        """Refuse further work while retaining ownership identity for safe cleanup."""
        with self.session.lock:
            self._invalidated = message
            self.reason = EndReason.INVALIDATED

    def check(self) -> None:
        self.check_valid()
        if time.monotonic() >= self.deadline:
            self.reason = EndReason.WALL_LIMIT
            raise ExecutionError(EndReason.WALL_LIMIT, "execution wall time limit reached")

    def result(self, reason: EndReason) -> RunResult:
        return RunResult(reason, self.sim_seconds, time.monotonic() - self.started, self.steps, self.shown_live)

    async def wait(self, done: Callable[[], bool]) -> None:
        """Wait for outside work without advancing physics, under the same deadline and generation."""
        while True:
            self.check()
            if done():
                return
            await asyncio.sleep(min(YIELD_SECONDS, max(0.0, self.deadline - time.monotonic())))

    async def call_thread(self, fn: Callable[..., T], *args: Any) -> T:
        def guarded() -> T:
            # Planning already uses this lock. Include the validity check in the same critical section so a
            # queued planner cannot bind old IDs to a reset model before it notices cancellation.
            with self.session.lock:
                if self._workers_stopping:
                    raise ExecutionError(EndReason.CANCELLED, "execution ended before the queued planner started")
                self.check()
                return fn(*args)

        return await self._thread(guarded)

    async def call_snapshot_thread(self, fn: Callable[..., T], *args: Any) -> T:
        """Prepare from caller-captured inputs without holding the viewer's pose lock for the whole worker.

        Unlike call_thread planners, this callback must not attach controllers or edit the live session.
        Validate again before using its result; a reset can occur after its brief admission check.
        """

        def guarded() -> T:
            with self.session.lock:
                if self._workers_stopping:
                    raise ExecutionError(EndReason.CANCELLED, "execution ended before snapshot preparation started")
                self.check()
            return fn(*args)

        return await self._thread(guarded)

    async def _thread(self, guarded: Callable[[], T]) -> T:
        worker = asyncio.create_task(asyncio.to_thread(guarded))
        self._workers.add(worker)

        def finished(worker: asyncio.Task) -> None:
            self._workers.discard(worker)
            if not worker.cancelled():
                worker.exception()
            if self._closing and not self._workers:
                self._release()

        worker.add_done_callback(finished)
        return await asyncio.shield(worker)

    async def drain_workers(self, cancel_pending: bool = False) -> None:
        self._workers_stopping |= cancel_pending
        if self._workers:
            await asyncio.shield(asyncio.gather(*tuple(self._workers), return_exceptions=True))

    async def advance(self, done: Callable[[], bool], max_seconds: float) -> RunResult:
        """Run one phase. Limits accumulate across phases; phase results count only this phase's work."""
        if not math.isfinite(max_seconds) or max_seconds < 0:
            raise ValueError("max_seconds must be finite and nonnegative")
        began, start_sim, start_steps = time.monotonic(), self.sim_seconds, self.steps
        phase_steps = max(0, round(max_seconds / self.timestep))
        yielded = began

        def ended(reason: EndReason) -> RunResult:
            if reason in (EndReason.WALL_LIMIT, EndReason.SIM_LIMIT, EndReason.STEP_LIMIT):
                self.reason = reason
            return RunResult(
                reason,
                self.sim_seconds - start_sim,
                time.monotonic() - began,
                self.steps - start_steps,
                self.shown_live,
            )

        while True:
            with self.session.lock:
                tick = time.monotonic()
                # Check before invoking a predicate that may hold old model IDs.
                self.check_valid()
                if float(self.session.data.time) < self._last_time or not (
                    np.isfinite(self.session.data.qpos).all() and np.isfinite(self.session.data.qvel).all()
                ):
                    self.reason = EndReason.UNSTABLE
                    raise ExecutionError(
                        EndReason.UNSTABLE, "the simulation became unstable or its clock rewound; reset the world"
                    )
                if tick >= self.deadline:
                    return ended(EndReason.WALL_LIMIT)
                if done():
                    return ended(EndReason.COMPLETED)
                left = phase_steps - (self.steps - start_steps)
                if left <= 0:
                    return ended(EndReason.DURATION)
                if self.limits.sim_seconds is not None:
                    available = math.floor((self.limits.sim_seconds - self.sim_seconds) / self.timestep + 1e-8)
                    if available <= 0:
                        return ended(EndReason.SIM_LIMIT)
                    left = min(left, available)
                if self.limits.steps is not None:
                    available = self.limits.steps - self.steps
                    if available <= 0:
                        return ended(EndReason.STEP_LIMIT)
                    left = min(left, available)
                count = min(left, max(1, round(CHUNK_SECONDS / self.timestep)))
                before, before_steps = float(self.session.data.time), self.session.steps_taken
                try:
                    self.session.step(count, owner=self)
                except ExecutionError as exc:
                    if exc.reason != EndReason.WALL_LIMIT:
                        raise
                    self.reason = EndReason.WALL_LIMIT
                finally:
                    taken = self.session.steps_taken - before_steps
                    self.steps += taken
                    self.sim_seconds += taken * self.timestep
                self.check_valid()
                if self.reason == EndReason.WALL_LIMIT:
                    return ended(EndReason.WALL_LIMIT)
                after = float(self.session.data.time)
                if after <= before or not (
                    np.isfinite(self.session.data.qpos).all() and np.isfinite(self.session.data.qvel).all()
                ):
                    self.reason = EndReason.UNSTABLE
                    raise ExecutionError(
                        EndReason.UNSTABLE, "the simulation became unstable or its clock rewound; reset the world"
                    )
                self._last_time = after

            # Equal physics batches in both modes. Joining, leaving, pausing or slowing the viewer changes
            # pacing only; it cannot add another clock or prevent a run from finishing.
            if self.watched():
                self.shown_live = True
                delay = taken * self.timestep / max(0.001, self.session.speed) - (time.monotonic() - tick)
                await asyncio.sleep(min(max(0.0, delay), max(0.0, self.deadline - time.monotonic())))
                yielded = time.monotonic()
            elif time.monotonic() - yielded >= YIELD_SECONDS:
                await asyncio.sleep(0)
                yielded = time.monotonic()
