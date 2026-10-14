"""MCP tools for reinforcement learning: start training, follow it, stop it, list trained policies.

Training runs in its own process (jobs.py), so these tools return at once and the viewer keeps
running. A trained policy then runs like any other algorithm: run_controller(method="policy:<name>")
and compare_controllers(methods=["lqr", "mpc", "policy:<name>"]).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer

from robotforge.control.env import DEFAULT_ROBOTS
from robotforge.control.mcp_tools import _tool_errors
from robotforge.control.rl import store
from robotforge.control.rl.api import unused_name
from robotforge.control.rl.jobs import JOBS
from robotforge.control.rl.spec import MAX_MINUTES, make_spec
from robotforge.control.tasks import TaskError
from robotforge.playground import Playground

MAX_WAIT_SECONDS = 600.0
PROGRESS_EVERY = 1.0
CURVE_POINTS = 12

ALGOS = {
    "ppo": (
        "PPO (Proximal Policy Optimization): many simulated robots try things at once; after each round the "
        "neural network shifts a little toward the actions that earned more reward, never too far at once."
    ),
    "sac": (
        "SAC (Soft Actor-Critic): one robot tries things and remembers everything it tried; a second network "
        "learns how good each action was, and the policy learns from those memories while staying curious."
    ),
}
MODES = {
    "direct": "the network commands the motors directly and learns from scratch",
    "residual": "the network learns small corrections on top of a built-in controller ({base}) that already works",
}


def register(server: MCPServer, playground: Playground) -> None:
    def world_dir() -> Path | None:
        return playground.store.dir if playground.store is not None else None

    @server.tool()
    async def train_policy(
        task: str,
        robot: str | None = None,
        algo: Literal["ppo", "sac"] | None = None,
        minutes: float | None = None,
        timesteps: int | None = None,
        task_params: dict[str, Any] | None = None,
        name: str | None = None,
        action_mode: Literal["direct", "residual"] | None = None,
        replace: bool = False,
    ) -> str:
        """Teach a robot a task by reinforcement learning (trial and error with a neural network), in the
        background. Returns at once with the policy name and a time estimate; the viewer keeps running.

        Examples: task="balance", robot="pole" (PPO, about a minute); task="walk", robot="dog",
        task_params={"speed": 0.4}, minutes=5 (PPO learning corrections on top of the trot gait);
        task="reach", robot="arm", algo="sac". Defaults per task need no tuning. Give minutes (a time budget)
        or timesteps (simulation steps), or neither for the task's default. The robot is rebuilt from this
        world's description and trained alone on the same floor. Needs the optional rl extra.
        Then: training_status(name) to follow it, and run_controller(method="policy:<name>") or
        compare_controllers(methods=["lqr", "mpc", "policy:<name>"]) to use it."""
        with _tool_errors():
            if minutes is not None and not 0 < minutes <= MAX_MINUTES:
                raise TaskError(f"minutes must be in (0, {MAX_MINUTES:.0f}]")
            robot_spec, floor, notes = _robot_spec(playground, task, robot)
            params = dict(task_params or {})
            if task == "reach" and isinstance(params.get("target"), str):
                params.pop("target")
                notes.append("trains on random targets, so the policy can reach the object wherever it is")
            place = store.policies_home(world_dir())
            chosen = store.check_name(name) if name else unused_name(_default_name(robot_spec["name"], task), place)
            spec = make_spec(
                chosen,
                task,
                robot_spec,
                algo=algo,
                action_mode=action_mode,
                timesteps=timesteps,
                minutes=minutes,
                task_params=params,
                floor=floor,
                world_dir=str(world_dir()) if world_dir() else None,
            )
            # Building the training world once checks the spec; off the event loop the viewer runs on.
            started = await asyncio.to_thread(JOBS.start, spec, place / chosen, replace)
            base = started.get("base_method")
            return _json(
                {
                    "started": chosen,
                    "task": task,
                    "robot": robot_spec["name"],
                    "algo": spec.algo,
                    "explain": ALGOS[spec.algo],
                    "action_mode": spec.action_mode,
                    "mode_means": MODES[spec.action_mode].format(base=base),
                    "timesteps_planned": spec.timesteps if spec.max_minutes is None else None,
                    "minutes_budget": spec.max_minutes,
                    "estimate_minutes": round(started["estimate_seconds"] / 60, 1),
                    "simulations_in_parallel": spec.n_envs,
                    "shoved_while_training": spec.push > 0,
                    "notes": notes,
                    "folder": started["folder"],
                    "next": f"training_status(name={chosen!r}, wait_seconds=60) to follow it",
                    "then": f"run_controller(robot={robot_spec['name']!r}, task={task!r}, method='policy:{chosen}')",
                }
            )

    @server.tool()
    async def training_status(ctx: Context, name: str, wait_seconds: float = 0.0) -> str:
        """How a training run is going: state (starting, running, testing, done, stopped, failed), progress,
        average episode reward (higher is better), success rate, time left, and the final test scores when done.
        With wait_seconds, waits until it finishes or that much time passes, sending progress updates."""
        with _tool_errors():
            if not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
                raise TaskError(f"wait_seconds must be in [0, {MAX_WAIT_SECONDS:.0f}]")
            folder = store.find_policy(name, world_dir())
            deadline = time.monotonic() + wait_seconds
            while True:
                status = JOBS.status(folder)
                if status.get("state") in store.FINAL_STATES or time.monotonic() >= deadline:
                    break
                await ctx.report_progress(100 * float(status.get("progress", 0)), 100, _progress_line(name, status))
                await asyncio.sleep(PROGRESS_EVERY)
            return _json(_status_view(name, folder, status))

    @server.tool()
    async def stop_training(name: str) -> str:
        """Stop a training run early. It keeps the best policy found so far (usable right away) and runs its
        final test; stopping in the first seconds, before anything was learned, leaves nothing to keep."""
        with _tool_errors():
            folder = store.find_policy(name, world_dir())
            status = await asyncio.to_thread(JOBS.stop, folder)
            return _json(_status_view(name, folder, status))

    @server.tool()
    def list_policies() -> str:
        """List trained policies (this world's first, then shared ones): task, robot, algorithm, training state,
        final success rate, and whether it can run. Run one with run_controller(method="policy:<name>")."""
        with _tool_errors():
            out = []
            for item in store.list_policies(world_dir()):
                state = JOBS.status(Path(item["folder"])).get("state", item["state"])
                out.append({**item, "state": state, "method": f"policy:{item['name']}"})
            return _json({"policies": out, "train_more_with": "train_policy"})


def _robot_spec(playground: Playground, task: str, robot: str | None) -> tuple[dict[str, Any], dict, list[str]]:
    """The robot as world.yaml describes it (template, library or file, params, pose) and the floor."""
    if task not in DEFAULT_ROBOTS:
        raise TaskError(f"unknown task {task!r}; choose one of: {', '.join(DEFAULT_ROBOTS)}")
    template, default_name = DEFAULT_ROBOTS[task]
    world = playground.current_world()
    if world is None:
        name = robot or default_name
        note = f"no world is open, so it trains the default {template} robot (named {name!r}); saved for all worlds"
        return {"name": name, "template": template}, {}, [note]
    if robot is None:
        raise TaskError(f"say which robot to train; robots in this world: {', '.join(r.name for r in world.robots)}")
    entry = next((r for r in world.robots if r.name == robot), None)
    if entry is None:
        names = ", ".join(r.name for r in world.robots) or "none"
        raise TaskError(f"no robot named {robot!r}; robots in this world: {names}")
    fields = entry.model_dump(include={"name", "template", "library", "file", "params", "pos", "yaw"})
    return fields, world.floor.model_dump(), []


def _default_name(robot: str, task: str) -> str:
    return store.check_name(f"{robot}_{task}"[:36].rstrip("_"))


def _progress_line(name: str, status: dict[str, Any]) -> str:
    parts = [f"{name}: {status.get('state', '?')}"]
    if "mean_episode_reward" in status:
        parts.append(f"reward {status['mean_episode_reward']:.2f}")
    if "success_rate" in status:
        parts.append(f"success {status['success_rate']:.0%}")
    if "eta_s" in status:
        parts.append(f"about {status['eta_s'] / 60:.1f} min left")
    return ", ".join(parts)


def _status_view(name: str, folder: Path, status: dict[str, Any]) -> dict[str, Any]:
    """The status a person wants to read: progress, scores, what they mean, and what to do next."""
    view: dict[str, Any] = {"name": name, "state": status.get("state", "unknown")}
    keep = (
        "progress",
        "timesteps",
        "timesteps_planned",
        "elapsed_s",
        "eta_s",
        "steps_per_s",
        "mean_episode_reward",
        "mean_episode_seconds",
        "success_rate",
        "best_test",
        "final",
        "stopped_early",
        "error",
        "traceback_tail",
        "note",
    )
    view.update({k: status[k] for k in keep if status.get(k) is not None})
    curve = status.get("curve") or []
    if curve:
        step = max(1, len(curve) // CURVE_POINTS)
        view["learning_curve"] = [{"timesteps": t, "reward": r} for t, r in curve[::step]]
    view["explain"] = (
        "reward is the task score per episode (higher is better); success_rate is the share of recent episodes "
        "that met the goal. Training shoves the robot now and then, so its numbers are a bit lower than a calm run."
    )
    meta = store.read_json(folder / store.POLICY_FILE)
    state = view["state"]
    if state in ("done", "stopped") and (folder / store.WEIGHTS_FILE).exists():
        view["next"] = (
            f"run_controller(robot={meta.get('robot')!r}, task={meta.get('task')!r}, method='policy:{name}') "
            f"or compare_controllers(... methods=[..., 'policy:{name}'])"
        )
    elif state == "failed":
        view["next"] = "read error and traceback_tail; the full log is train.log in the policy folder"
    view["folder"] = str(folder)
    return view


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, default=float)
