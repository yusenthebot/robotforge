"""What train_policy, training_status and list_algorithms need from the training package.

control/rl/mcp_tools.py keeps the tools themselves (their names and signatures are the agent's interface); the
work of reading a request into a run, and of telling an agent how a run is going, lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robotforge import wording
from robotforge.control.rl import store
from robotforge.control.rl.spec import DEFAULTS
from robotforge.control.tasks import TaskError
from robotforge.training import algorithms, api
from robotforge.training.summary import run_view


def new_pipeline(
    playground: Any,
    task: str,
    robot: str | None,
    algo: str | list[str] | None,
    search: dict[str, Any] | None,
    settings: dict[str, Any] | None,
    observation: str | None,
    parallel: int | None,
) -> bool:
    """True when the request needs more than the original PPO/SAC trainer: another algorithm or several, a search,
    sensors in the observation, a task it has no defaults for."""
    if search is not None or settings or parallel is not None or observation is not None:
        return True
    if isinstance(algo, list) or algo not in (None, "ppo", "sac"):
        return True
    if task not in DEFAULTS:
        return True
    world = playground.current_world()
    if world is not None and robot:
        entry = next((r for r in world.robots if r.name == wording.match(robot, [r.name for r in world.robots])), None)
        if entry is not None and (entry.sensors or entry.build):
            return True  # sensors join the observation; a robot built from parts has its own actuators
    return False


def _robot(playground: Any, robot: str | None) -> Any:
    world = playground.current_world()
    if world is None:
        return robot
    names = [r.name for r in world.robots]
    if robot is None:
        if len(names) == 1:
            return names[0]
        raise TaskError(f"say which robot to train; robots in this world: {', '.join(names) or 'none'}")
    found = wording.match(robot, names)
    if found is None:
        raise TaskError(wording.unknown("robot", robot, names))
    return found


def start(
    playground: Any,
    task: str,
    robot: str | None,
    algo: str | list[str] | None,
    minutes: float | None,
    timesteps: int | None,
    task_params: dict[str, Any] | None,
    name: str | None,
    action_mode: str | None,
    shoves: float | None,
    replace: bool,
    observation: str | None,
    search: dict[str, Any] | None,
    settings: dict[str, Any] | None,
    parallel: int | None,
) -> dict[str, Any]:
    """Start a run the way the tool asks for it and describe what started."""
    world = playground.current_world()
    world_dir = playground.store.dir if playground.store is not None else None
    spec, started = api.start(
        task,
        name=name,
        save_dir=store.policies_home(world_dir),
        replace=replace,
        robot=_robot(playground, robot),
        world=world,
        world_dir=world_dir,
        algo=algo,
        search=search,
        settings=settings,
        steps=timesteps,
        minutes=minutes,
        observation=observation or "auto",
        action_mode=action_mode,
        task_params=task_params,
        push=shoves,
        parallel=parallel,
    )
    meta = store.read_json(Path(started["folder"]) / store.POLICY_FILE)
    sources = [{"source": s["source"], "numbers": len(s["labels"])} for s in meta.get("observation_layout", [])]
    notes = []
    if world is None:
        notes.append("no world is open, so it trains a default robot alone on a floor")
    return {
        "started": spec.name,
        "task": task,
        "robot": spec.env["robot"],
        "algorithms": spec.algos,
        "trials": [{"id": t["id"], "algo": t["algo"], "settings": t["params"]} for t in started["trials"]],
        "search": spec.search.model_dump() if spec.search else None,
        "steps_per_trial": spec.steps if spec.minutes is None else None,
        "minutes_budget": spec.minutes,
        "running_at_once": spec.parallel or 1,
        "action_mode": spec.env["action_mode"],
        "observation": {"numbers": started["observation"]["size"], "from": sources},
        "left_out": started["observation"].get("left_out", []),
        "notes": notes,
        "folder": started["folder"],
        "next": f"training_status(name={spec.name!r}, wait_seconds=60); the viewer's Training panel draws it",
        "then": f"run_controller(robot={spec.env['robot']!r}, task={task!r}, method='policy:{spec.name}')",
    }


def status_extra(status: dict[str, Any]) -> dict[str, Any]:
    """The trials of a run for training_status (nothing for a run of the original trainer)."""
    return run_view(status)


def algorithm_catalog() -> dict[str, Any]:
    """Every training algorithm with its settings and ranges, for list_algorithms."""
    return {
        "algorithms": algorithms.catalog(),
        "how": (
            "train_policy(task, robot, algo=...) with one name, or a list to compare them side by side; "
            "search={'trials': 9} tunes the settings; plugins add algorithms (robotforge.extend.algorithm)"
        ),
    }
