"""MCP tools for the algorithm layer: list what exists, run one algorithm live, compare several.

Registered by robotforge.mcp_server.build_server via `register(...)`, so the server
file only needs one line for the whole control layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from robotforge.control import tasks as task_catalog
from robotforge.control.compare import MONITOR, EpisodeMonitor, compare_methods
from robotforge.control.methods import METHODS, make_controller
from robotforge.control.tasks import TaskError, make_task
from robotforge.playground import Playground

MAX_SECONDS = 60.0
MAX_COMPARE_SECONDS = 30.0
MAX_METHODS = 6

RunUntil = Callable[[Callable[[], bool], float], Awaitable[dict[str, Any]]]


def register(server: MCPServer, playground: Playground, run_until: RunUntil) -> None:
    session = playground.session

    @server.tool()
    def list_algorithms() -> str:
        """List the control tasks (reach, balance, stand, walk, drive_to), which robots and algorithms fit
        each, their parameters, and a one-line explanation of every algorithm (ik, pd, cartesian, lqr, mpc,
        mppi, hold, gait, none). Use it to explain the options before running or comparing them."""
        return _json({"tasks": task_catalog.catalog(), "algorithms": METHODS})

    @server.tool()
    async def run_controller(
        robot: str,
        task: str,
        method: str,
        seconds: float | None = None,
        task_params: dict[str, Any] | None = None,
        method_params: dict[str, Any] | None = None,
        push_at: float | None = None,
    ) -> str:
        """Drive a robot with a control algorithm on a task and report how well it did.

        Examples: robot="arm", task="reach", method="mpc", task_params={"target": [0.3, 0.1, 0.2]};
        robot="pole", task="balance", method="lqr", push_at=2 (shoves it at 2 s to show recovery);
        robot="dog", task="stand", method="mpc"; robot="rover", task="drive_to", task_params={"goal": [1, 0.5]}.
        list_algorithms says which methods fit which task. Plays out in real time when the viewer is open.
        The controller keeps running afterwards (e.g. keeps the pole balanced); call stop(robot) to release.
        Returns reward, success, effort (joules), the task's own numbers and, for MPC, planning speed."""
        with _tool_errors():
            seconds = _seconds(seconds, MAX_SECONDS)
            with session.lock:
                model, snapshot, version = session.model, copy.copy(session.data), session.version
            bound = make_task(task, model, robot, task_params)
            seconds = seconds or bound.default_seconds
            bound.begin(snapshot)
            # Building can take a while (ars trains first), so do it off the event loop the viewer runs on.
            controller = await asyncio.to_thread(make_controller, method, bound, model, snapshot, method_params)
            with session.lock:
                _same_world(session, version)
                bound.begin(session.data)
                monitor = EpisodeMonitor(bound, model, session.data)
                session.set_controller(robot, controller)
                session.set_controller(MONITOR, monitor)
            events: list[str] = []
            try:
                if push_at is not None and 0 < push_at < seconds:
                    first = await run_until(lambda: False, push_at)
                    _same_world(session, version)
                    with session.lock:
                        events.append(f"t={push_at:.1f}s: {bound.push(session.data)}")
                    rest = await run_until(lambda: False, seconds - push_at)
                    total = round(first["sim_seconds"] + rest["sim_seconds"], 2)
                    run = {"sim_seconds": total, "shown_live": rest["shown_live"]}
                else:
                    run = await run_until(lambda: False, seconds)
                _same_world(session, version)
            finally:
                if session.version == version:
                    session.set_controller(MONITOR, None)
            with session.lock:
                return _json(
                    {
                        "robot": robot,
                        "task": bound.describe(),
                        "method": method,
                        "explain": METHODS[method],
                        "reward": round(monitor.reward, 3),
                        "success": bool(bound.success(session.data)) and monitor.failed_at is None,
                        "failed_at": monitor.failed_at,
                        "first_success_at": monitor.first_success,
                        "effort_j": round(monitor.effort, 3),
                        "result": bound.report(session.data),
                        "controller": controller.status(),
                        "events": events,
                        **run,
                        "still_running": f"{method} keeps controlling {robot}; call stop to release it",
                    }
                )

    @server.tool()
    async def compare_controllers(
        robot: str,
        task: str,
        methods: list[str] | None = None,
        seconds: float | None = None,
        task_params: dict[str, Any] | None = None,
        method_params: dict[str, dict[str, Any]] | None = None,
        push_at: float | None = None,
    ) -> str:
        """Run several algorithms on the same robot and task, each from the current state of the world, and
        return a comparison table: reward (higher is better), success, the task's key number, effort in
        joules and computer time per simulated second. Runs off-screen (the live world is not changed).

        methods defaults to every algorithm that fits the task (see list_algorithms); method_params maps a
        method to its settings, e.g. {"mpc": {"samples": 64}}. push_at adds the same shove to every run
        to compare how they recover. Explain the table to the user: what each method does and why it won
        or lost."""
        with _tool_errors():
            seconds = _seconds(seconds, MAX_COMPARE_SECONDS)
            if methods is not None and not 1 <= len(methods) <= MAX_METHODS:
                raise TaskError(f"compare 1 to {MAX_METHODS} methods at a time")
            with session.lock:
                model, start = session.model, copy.copy(session.data)
            comparison = await asyncio.to_thread(
                compare_methods,
                task,
                methods,
                seconds,
                task_params=task_params,
                method_params=method_params,
                push_at=push_at,
                model=model,
                robot=robot,
                data=start,
                seed=None,
            )
            # Best = did not fail, then met the goal, then highest reward.
            best = max(comparison.results, key=lambda r: (not r.failed, r.success, r.reward))
            return _json(
                {
                    **comparison.to_dict(),
                    "best": best.method,
                    "explain": {r.method: METHODS[r.method] for r in comparison.results},
                    "next": f"run_controller(robot={robot!r}, task={task!r}, method={best.method!r}) shows it live",
                }
            )


def _seconds(seconds: float | None, limit: float) -> float | None:
    if seconds is not None and not 0 < seconds <= limit:
        raise TaskError(f"seconds must be in (0, {limit:.0f}]")
    return seconds


def _same_world(session, version: int) -> None:
    if session.version != version:
        raise TaskError("the world was changed during the run, so the controller was removed; run it again")


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Anticipated failures become ToolError so the agent sees the actual message."""
    try:
        yield
    except (TaskError, ValueError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, default=float)
