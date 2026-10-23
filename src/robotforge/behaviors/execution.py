"""Ownership of an arm command and the controllers it starts during later phases.

Controller algorithms still decide when to queue, replan and hand off. This ledger observes their provenance;
it does not replace their state machines or change their update order.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any, TypeVar

from robotforge.sim.execution import CHUNK_SECONDS, EndReason, ExecutionError, ExecutionRun, RunLimits

from . import catch as catching
from . import catch_stand, fetch, relay, visit

T = TypeVar("T")
PREFIXES = ("__navigate__", fetch.KEY, catch_stand.KEY, catching.KEY, relay.KEY)


class ArmExecution:
    def __init__(self, playground: Any, run: ExecutionRun) -> None:
        self.pg, self.run, self.session = playground, run, playground.session
        self.expected: dict[str, Any] = {}
        self.owned: dict[int, Any] = {}
        self.roots: list[Any] = []
        self.phases: list[dict[str, Any]] = []
        self._issuing: ContextVar[bool] = ContextVar("arm_command", default=False)
        self._closing = False
        self.session.controller_hooks.append(self._changed)

    def reserve(self, robot: str) -> None:
        """Watch endpoints before planning, including absence, so a foreign attachment cannot be adopted later."""
        with self.session.lock:
            arm = self.pg.behaviors._arm_name(robot)
            for endpoint in {robot, arm, arm.split("/", 1)[0]}:
                self.run.resources.add(f"robot:{endpoint.split('/', 1)[0]}")
                for key in (endpoint, *(prefix + endpoint for prefix in PREFIXES)):
                    self.expected.setdefault(key, self.session.controllers.get(key))

    @contextlib.contextmanager
    def command(self, *, check_deadline: bool = True) -> Iterator[None]:
        # Only surround the synchronous mutation, never an await or a physics batch. Context propagates to a
        # tracked planner thread, whereas another controller updating on that thread has its own provenance.
        with self.session.lock:
            if check_deadline:
                self.run.check()
            else:
                self.run.check_valid()
            token = self._issuing.set(True)
            try:
                yield
            finally:
                self._issuing.reset(token)

    def claim(self, value: Any) -> None:
        if value is None or id(value) in self.owned:
            return
        self.owned[id(value)] = value
        if isinstance(value, (fetch.Errand, visit.Visit, catch_stand.Approach)):
            # A proven descendant may start a trip with a gait already attached. Claim that unchanged gait,
            # never a foreign replacement that failed the watched-slot identity check.
            base = self.session.controllers.get(value.robot)
            if base is not None and self.expected.get(value.robot) is base:
                self.claim(base)
        for name in ("arm", "giver", "after", "inner", "op"):
            child = getattr(value, name, None)
            if child is not None:
                self.claim(child)
        for leg in getattr(value, "legs", ()):
            self.claim(getattr(leg, "trip", None))

    def bind(self, root: Any) -> None:
        self.roots.append(root)
        self.claim(root)
        for key, current in self.session.controllers.items():
            if key in self.expected and id(current) in self.owned:
                self.expected[key] = current

    async def plan(self, invoke: Callable[..., T], *args: Any) -> T:
        def attach() -> T:
            with self.command():
                value = invoke(*args)
                self.bind(value[0] if isinstance(value, tuple) else value)
                return value

        value = await self.run.call_thread(attach)
        self.run.check()
        return value

    def _changed(self, key: str, old: Any, new: Any, reason: str) -> None:
        if self._closing or key not in self.expected:
            return
        if reason in {"model_changed", "reset", "shutdown"}:
            # Observation runs before the old model is discarded. Restore temporary grasp/contact settings
            # now; a reset hook cannot safely use their old actuator/geometry IDs in the replacement model.
            self.cancel()
            self.run.invalidate("the world was reset or changed during the arm command; run it again")
            return
        actor = self.session.updating_controller
        legitimate = id(actor) in self.owned if actor is not None else self._issuing.get()
        if legitimate:
            self.expected[key] = new  # hooks precede the actual slot mutation
            self.claim(new)
        else:
            self.run.invalidate(f"the controller for {key} changed outside the arm command; run it again")

    def check(self) -> None:
        self.run.check_valid()
        self.refresh()
        for key, expected in self.expected.items():
            if self.session.controllers.get(key) is not expected:
                self.run.invalidate(f"the controller for {key} changed outside the arm command; run it again")
                self.run.check_valid()

    def refresh(self) -> None:
        previous = -1
        while previous != len(self.owned):
            previous = len(self.owned)
            for root in tuple(self.owned.values()):
                self.claim_descendants(root)

    def claim_descendants(self, root: Any) -> None:
        # Already-owned roots acquire children later (Relay.after, Approach.inner, Visit.op).
        for name in ("arm", "giver", "after", "inner", "op"):
            self.claim(getattr(root, name, None))
        for leg in getattr(root, "legs", ()):
            self.claim(getattr(leg, "trip", None))

    async def advance(
        self, done: Callable[[], bool], seconds: float, name: str, *, legacy_elapsed: bool = False
    ) -> dict[str, Any]:
        start_time = slice_start = float(self.session.data.time)
        slice_limit = min(0.1, seconds)

        def checked() -> bool:
            nonlocal slice_start, slice_limit
            self.check()
            if done():
                return True
            if legacy_elapsed and float(self.session.data.time) - slice_start >= slice_limit:
                elapsed = float(self.session.data.time) - start_time
                if elapsed >= seconds - 1e-9:
                    return True
                slice_start = float(self.session.data.time)
                slice_limit = min(0.1, seconds - elapsed)
            return False

        began, start_sim, start_steps = time.monotonic(), self.run.sim_seconds, self.run.steps
        try:
            # Preserve run_until's outer 100 ms slices and run_fast's inner 20 ms elapsed-time rule, including
            # its floating-point rounding, inside the same strict cumulative execution budget.
            batch_seconds = max(1, round(CHUNK_SECONDS / self.run.timestep)) * self.run.timestep
            result = await self.run.advance(checked, seconds + (batch_seconds if legacy_elapsed else 0.0))
        except BaseException as exc:
            reason = (
                EndReason.CANCELLED
                if isinstance(exc, asyncio.CancelledError)
                else exc.reason
                if isinstance(exc, ExecutionError)
                else EndReason.FAILED
            )
            self.phases.append(
                {
                    "name": name,
                    "limit_seconds": seconds,
                    "reason": reason.value,
                    "steps": self.run.steps - start_steps,
                    "sim_seconds": self.run.sim_seconds - start_sim,
                    "wall_seconds": time.monotonic() - began,
                }
            )
            raise
        self.phases.append(
            {
                "name": name,
                "limit_seconds": seconds,
                "reason": result.reason.value,
                "steps": result.steps,
                "sim_seconds": result.sim_seconds,
                "wall_seconds": result.wall_seconds,
            }
        )
        values: dict[str, Any] = {"sim_seconds": round(result.sim_seconds, 2), "shown_live": result.shown_live}
        if result.reason in (EndReason.WALL_LIMIT, EndReason.SIM_LIMIT, EndReason.STEP_LIMIT):
            values["stopped_early"] = f"stopped at the execution {result.reason}"
        return values

    async def carrier_stands(self, robot: str) -> dict[str, Any]:
        self.reserve(robot)
        with self.command():
            stop = self.pg.behaviors.stop_carrier(robot)
            if stop is not None:
                self.claim(self.session.controllers.get(stop.robot))
        if stop is None:
            return {}
        waited = await self.advance(stop.still, stop.seconds, "carrier_stands")
        return {"carrier_stopped": f"{stop.robot} stopped first and stood still after {waited['sim_seconds']:.1f} s"}

    def budget(self, recorded: Any, caps: list[float]) -> None:
        """Bind actual phase caps to the original start; preparation never receives a fresh wall deadline."""
        with self.session.lock:
            wall = 3 * sum(caps) + 5 * len(caps) if recorded.watched() else recorded.wall_limit() * len(caps)
            self.run.limits = RunLimits(sum(caps), wall)
            self.run.deadline = self.run.started + wall
            recorded.call.wall = wall
            recorded.call.record.update(
                budget={
                    "sim_seconds": sum(caps),
                    "wall_seconds": wall,
                    "includes_preparation": True,
                    "phase_caps": caps,
                }
            )
            self.run.check()

    def _foreign_base(self, robot: str) -> bool:
        return any(
            (current := self.session.controllers.get(key)) is not None and id(current) not in self.owned
            for key in (robot, "__navigate__" + robot, fetch.KEY + robot, catch_stand.KEY + robot)
        )

    def cancel(self) -> None:
        """Disarm callbacks and restore only this family's resources, preserving any replacement command."""
        with self.session.lock:
            if self._closing or not self.run.current:
                return
            self._closing = True
            self.refresh()
            values = tuple(self.owned.values())
            for value in values:
                if isinstance(value, (relay.Relay, catching.Then)):
                    value.note = "the arm execution ended"
                    value.start = lambda: "the arm execution ended"
                    if isinstance(value, relay.Relay):
                        value.on_done = None
                if isinstance(value, catching.Catch):
                    value.then = None
                    value.step, value.note = "missed", "the arm execution ended"
                if isinstance(value, (fetch.Errand, visit.Visit)):
                    if self._foreign_base(value.robot):
                        value._stance = None  # interrupt must not command a replacement base
                    value.interrupt("the arm execution ended", float(self.session.data.time))
                    self.pg.behaviors.errand_ended(value)
                elif isinstance(value, catch_stand.Approach):
                    value.interrupt("the arm execution ended")
            # End only our navigation, preserving its normal interrupted report and live-trip bookkeeping.
            for key, current in tuple(self.session.controllers.items()):
                if key.startswith("__navigate__") and key in self.expected and id(current) in self.owned:
                    self.pg.behaviors._end_trip(key.removeprefix("__navigate__"))
            # Hidden orchestrators can attach arms later, so remove them before stopping visible endpoints.
            for key, current in tuple(self.session.controllers.items()):
                if key in self.expected and id(current) in self.owned and key.startswith(PREFIXES):
                    self.session.set_controller(key, None)
            for key, current in tuple(self.session.controllers.items()):
                if key in self.expected and id(current) in self.owned and not key.startswith("__"):
                    if "/" not in key and self._foreign_base(key):
                        continue
                    self.pg.behaviors.stop(key)

    def close(self) -> None:
        with self.session.lock:
            if self._changed in self.session.controller_hooks:
                self.session.controller_hooks.remove(self._changed)

    async def __aenter__(self) -> ArmExecution:
        return self

    async def __aexit__(self, typ: Any, exc: BaseException | None, tb: Any) -> None:
        if exc is not None:
            while True:
                try:
                    await self.run.drain_workers(cancel_pending=True)
                    break
                except asyncio.CancelledError:
                    continue
            self.cancel()
        self.close()
