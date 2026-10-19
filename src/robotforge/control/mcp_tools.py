"""MCP tools for the algorithm layer: list what exists, run one algorithm live, compare several.

Registered by robotforge.mcp_server.build_server via `register(...)`, so the server
file only needs one line for the whole control layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import uuid
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from robotforge import wording
from robotforge.control import tasks as task_catalog
from robotforge.control.compare import MONITOR, EpisodeMonitor, check_push, compare_methods
from robotforge.control.methods import LEARNED, LONG_NAMES, METHODS, explain, make_controller
from robotforge.control.rl.runtime import policy_params
from robotforge.control.tasks import TaskError, make_task
from robotforge.playground import Playground

MAX_SECONDS = 60.0
MAX_COMPARE_SECONDS = 30.0
MAX_METHODS = 6
COMPARE_WALL_SECONDS = 60.0  # computer time for a whole comparison; slower methods are cut short

RunUntil = Callable[[Callable[[], bool], float], Awaitable[dict[str, Any]]]


def register(server: MCPServer, playground: Playground, run_until: RunUntil) -> None:
    session = playground.session

    def world_dir():
        return playground.store.dir if playground.store is not None else None

    @server.tool()
    def list_algorithms() -> str:
        """List the control tasks (reach, balance, stand, walk, drive_to, hover) with the robots, algorithms and
        parameters that fit each, a one-line explanation of every algorithm (ik, pd, cartesian, joint_impedance, cartesian_impedance, osc, lqr,
        mpc, mppi, hold, gait, cascade, none), reinforcement learning (ppo, sac: train_policy first) and the trained policies. Use it to
        explain the options."""
        from robotforge.control.rl.mcp_tools import ALGOS
        from robotforge.control.rl.store import list_policies

        trained = [f"policy:{p['name']} ({p['task']}, {p['state']})" for p in list_policies(world_dir())]
        learning = {**ALGOS, "how": "train_policy(task, robot) in the background, then method='policy:<name>'"}
        return _json(
            {"tasks": task_catalog.catalog(), "algorithms": METHODS, "learning": learning, "trained_policies": trained}
        )

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

        Examples: robot="arm", task="reach", method="mpc", task_params={"target": [0.3, 0.1, 0.2]}; robot="pole",
        task="balance", method="lqr", push_at=2 (shoves it at 2 s to show recovery); robot="dog", task="stand",
        method="mpc"; robot="rover", task="drive_to", task_params={"goal": [1, 0.5]}; robot="drone", task="hover",
        method="lqr", push_at=2 (an aircraft; cascade is its own flight controller). list_algorithms says which
        methods fit which task; method="policy:<name>" runs a policy from train_policy. Plays in real time when the
        viewer is open. The controller stays attached afterwards (e.g. keeps the pole balanced; keeps_running says
        so) until stop(robot). Returns reward, success, effort (joules), the task's own numbers and, for MPC,
        planning speed."""
        with _tool_errors():
            seconds = _seconds(seconds, MAX_SECONDS)
            robot, task, method = _robot(playground, robot), _task(task), _method(method)
            with session.lock:
                model, snapshot, version = session.model, copy.copy(session.data), session.version
            bound = make_task(task, model, robot, task_params)
            seconds = seconds or bound.default_seconds
            check_push(push_at, seconds)
            bound.begin(snapshot)
            method_params = policy_params(method, method_params, world_dir(), bound)
            # Building can take a while (ars trains first), so do it off the event loop the viewer runs on.
            controller = await asyncio.to_thread(make_controller, method, bound, model, snapshot, method_params)
            monitor_key = f"{MONITOR}{robot}/{uuid.uuid4().hex[:8]}"  # one per call: runs may overlap
            with session.lock:
                _same_world(session, version)
                bound.begin(session.data)
                monitor = EpisodeMonitor(bound, model, session.data)
                session.set_controller(robot, controller)
                session.set_controller(monitor_key, monitor)
            events: list[str] = []
            runs: list[dict[str, Any]] = []
            try:
                if push_at is not None:
                    runs.append(await run_until(lambda: False, push_at))
                    _still_running(session, version, monitor_key)
                    if "stopped_early" not in runs[-1]:
                        with session.lock:
                            events.append(f"t={push_at:.1f}s: {bound.push(session.data)}")
                        runs.append(await run_until(lambda: False, seconds - push_at))
                else:
                    runs.append(await run_until(lambda: False, seconds))
                _still_running(session, version, monitor_key)
            finally:
                session.set_controller(monitor_key, None)
            run = {"sim_seconds": round(sum(r["sim_seconds"] for r in runs), 2), "shown_live": runs[-1]["shown_live"]}
            stopped = next((r["stopped_early"] for r in runs if "stopped_early" in r), None)
            with session.lock:
                if session.controllers.get(robot) is not controller:
                    events.append(f"something else took over {robot} during the run (e.g. keyboard driving)")
                result = {
                    "robot": robot,
                    "task": bound.describe(),
                    "method": method,
                    "explain": explain(method),
                    "reward": round(monitor.reward, 3),
                    "success": bool(bound.success(session.data)) and monitor.failed_at is None,
                    "failed_at": monitor.failed_at,
                    "first_success_at": monitor.first_success,
                    "effort_j": round(monitor.effort, 3),
                    "result": bound.report(session.data),
                    "controller": controller.status(),
                    "events": events,
                    **run,
                    "keeps_running": session.controllers.get(robot) is controller,
                    "release_with": f"stop(robot={robot!r})",
                }
                if stopped:
                    result["stopped_early"] = stopped
                return _json(result)

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
        """Run several algorithms on the same robot and task from the world's current state and return a table: reward
        (higher is better), success, the task's key number, effort in joules and computer time per simulated second.
        Runs off-screen: the live world is unchanged and nothing moves in the viewer (run_controller runs one live).

        methods defaults to every algorithm that fits the task (list_algorithms) and may include trained policies,
        e.g. ["lqr", "mpc", "policy:pole_balance"]; method_params maps a method to its settings, e.g. {"mpc":
        {"samples": 64}}. push_at adds the same shove to every run to compare recovery. Explain the table: what each
        method does and why it won or lost."""
        with _tool_errors():
            seconds = _seconds(seconds, MAX_COMPARE_SECONDS)
            robot, task = _robot(playground, robot), _task(task)
            if methods is not None:
                methods = [_method(m) for m in methods]
            if methods is not None and not 1 <= len(methods) <= MAX_METHODS:
                raise TaskError(
                    f"methods must list 1 to {MAX_METHODS} names, got {len(methods)}; leave it out for all that fit"
                )
            with session.lock:
                model, start = session.model, copy.copy(session.data)
            method_params = dict(method_params or {})
            if methods is not None and any(m.startswith("policy:") for m in methods):
                probe = make_task(task, model, robot, task_params)  # check each policy fits before running any
                for m in methods:
                    method_params[m] = policy_params(m, method_params.get(m), world_dir(), probe)
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
                wall_budget=COMPARE_WALL_SECONDS,
            )
            # Best = did not fail, then met the goal, then highest reward.
            best = max(comparison.results, key=lambda r: (not r.failed, r.success, r.reward))
            return _json(
                {
                    **comparison.to_dict(),
                    "best": best.method,
                    "explain": {r.method: explain(r.method) for r in comparison.results},
                    "next": f"run_controller(robot={robot!r}, task={task!r}, method={best.method!r}) shows it live",
                }
            )

    from robotforge.control.rl import mcp_tools as rl_tools  # the learning tools belong to this layer too

    rl_tools.register(server, playground)


def _seconds(seconds: float | None, limit: float) -> float | None:
    if seconds is not None and not 0 < seconds <= limit:
        raise TaskError(wording.seconds_range("seconds", seconds, limit))
    return seconds


def _robot(playground: Playground, robot: str) -> str:
    """The robot's name as the world spells it. A loaded model whose robots have no names (the demo arm) is
    left to the task, which looks the name up itself."""
    names = playground.behaviors.robot_names()
    return playground.behaviors.resolve(robot) if names else robot


def _task(task: str) -> str:
    return wording.match(task, task_catalog.TASKS) or task


def _method(method: str) -> str:
    """ "IK" is ik; a policy is "policy:<name>"."""
    prefix, colon, rest = method.partition(":")
    if colon:
        return f"{prefix.lower()}:{rest}"
    return wording.match(method, [*METHODS, *LEARNED], LONG_NAMES) or method  # "PPO" is ppo


def _same_world(session, version: int) -> None:
    if session.version != version:
        raise TaskError("the world was changed during the run, so the controller was removed; run it again")


def _still_running(session, version: int, monitor_key: str) -> None:
    _same_world(session, version)
    if monitor_key not in session.controllers:
        raise TaskError("the world was reset during the run, which removed the controller; run it again")


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Anticipated failures become ToolError so the agent sees the actual message."""
    try:
        yield
    except (TaskError, ValueError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


def _json(data: object) -> str:
    return wording.dumps(data, default=float)
