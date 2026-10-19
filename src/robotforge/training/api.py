"""The one train call: any robot, any task, any algorithm (or several, or a search), in the background or waited for.

    from robotforge.training.api import train

    policy = train("walk", world=world, robot="dog", algo=["ars", "cem"], steps=150_000)       # two algorithms
    policy = train("reach", robot="arm", algo="ars", search={"method": "halving", "trials": 9})  # a search
    print(policy.meta["final"])                          # the best trial: algorithm, settings, test return

`start` returns at once (the training runs in a process of its own and writes status.json for the viewer and the
agent); `train` waits for it and loads the best policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from robotforge.control.env import DEFAULT_ROBOTS
from robotforge.control.rl import store
from robotforge.control.rl.api import unused_name
from robotforge.control.rl.jobs import JOBS
from robotforge.control.rl.runtime import TrainedPolicy
from robotforge.control.rl.spec import DEFAULTS, MAX_MINUTES
from robotforge.control.tasks import TaskError
from robotforge.control.tasks.base import readable
from robotforge.training import algorithms
from robotforge.training.envs import env_spec
from robotforge.training.runs import MAX_STEPS, RunSpec, default_parallel, make_run_spec, plan_for, start_run
from robotforge.training.search import SearchSpec

# Environment steps per trial that teach each built-in task something with a numpy algorithm on a laptop CPU.
STEPS = {"balance": 60_000, "reach": 150_000, "stand": 100_000, "walk": 200_000, "drive_to": 100_000}
DEFAULT_STEPS = 120_000
RL_MODULES = ("gymnasium", "stable_baselines3", "torch")


def rl_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in RL_MODULES)


def algo_names(algo: str | list[str] | None) -> list[str]:
    """The algorithms asked for, spelled as the registry knows them; numpy's ARS when none is named."""
    names = [algo] if isinstance(algo, str) else list(algo or ["ars"])
    if not names:
        raise TaskError("algo is a name or a list of names; list_algorithms shows them")
    return list(dict.fromkeys(algorithms.get(n).name for n in names))


def search_spec(search: dict[str, Any] | None, seed: int) -> SearchSpec | None:
    if search is None:
        return None
    if not isinstance(search, dict):
        raise TaskError("search is a dict such as {'method': 'halving', 'trials': 9, 'space': {'noise': [0.1, 0.3]}}")
    given = {"method": "halving", "seed": seed, **search}
    try:
        return SearchSpec.model_validate(given)
    except ValidationError as exc:
        raise TaskError(readable(exc, "search settings", list(SearchSpec.model_fields))) from exc


def build_spec(
    task: str,
    *,
    name: str,
    robot: Any = None,
    world: Any = None,
    world_dir: str | Path | None = None,
    algo: str | list[str] | None = None,
    search: dict[str, Any] | None = None,
    settings: dict[str, dict[str, Any]] | None = None,
    steps: int | None = None,
    minutes: float | None = None,
    observation: str = "auto",
    action_mode: str | None = None,
    task_params: dict[str, Any] | None = None,
    push: float | None = None,
    residual_scale: float | None = None,
    control_hz: float | None = None,
    episode_seconds: float | None = None,
    seed: int = 0,
    parallel: int | None = None,
    scope: str | None = None,
    eval_episodes: int = 5,
) -> RunSpec:
    """A complete RunSpec from what was asked plus defaults that work without tuning."""
    names = algo_names(algo)
    sweep = search_spec(search, seed)
    if minutes is not None and not 0 < minutes <= MAX_MINUTES:
        raise TaskError(f"minutes must be in (0, {MAX_MINUTES:.0f}]")
    if minutes is not None and steps is None:
        if sweep is not None:
            raise TaskError("a search needs timesteps (the steps each trial may use) as well as minutes")
        steps = MAX_STEPS  # the clock decides
    d = DEFAULTS.get(task)
    given_env = {
        "robot": robot,
        "world": world,
        "world_dir": world_dir,
        "scope": scope,
        "task_params": task_params,
        "observation": observation,
        "action_mode": action_mode or (d.action_mode if d else "direct"),
        "residual_scale": residual_scale if residual_scale is not None else (d.residual_scale if d else 0.2),
        "control_hz": control_hz,
        "episode_seconds": episode_seconds,
        "push": push if push is not None else (d.push if d else 0.0),
    }
    spec = env_spec(task, **given_env)
    plans = sweep.trials if sweep else len(names)
    return make_run_spec(
        name,
        spec.model_dump(mode="json"),
        algos=names,
        params={n: dict(v) for n, v in (settings or {}).items()},
        steps=steps or STEPS.get(task, DEFAULT_STEPS),
        minutes=minutes,
        search=sweep,
        seed=seed,
        parallel=default_parallel(plans) if parallel is None else parallel,
        eval_episodes=eval_episodes,
    )


def start(
    task: str, *, name: str | None = None, save_dir: str | Path | None = None, replace: bool = False, **kwargs: Any
) -> tuple[RunSpec, dict[str, Any]]:
    """Check the request and start training in the background. Returns (the run's spec, what started)."""
    place = Path(save_dir) if save_dir is not None else store.policies_home(None)
    if name:
        chosen = store.check_name(name)
    else:
        robot = kwargs.get("robot")
        base = (
            robot if isinstance(robot, str) else (robot or {}).get("name") or DEFAULT_ROBOTS.get(task, ("", "robot"))[1]
        )
        chosen = unused_name(store.check_name(f"{base}_{task}"[:36].rstrip("_")), place)
    spec = build_spec(task, name=chosen, **kwargs)
    started = start_run(spec, place / chosen, replace=replace)
    return spec, started


def train(task: str, *, wait_print: bool = True, **kwargs: Any) -> TrainedPolicy:
    """Train and wait; returns the best policy. `start` takes the arguments. wait_print=False keeps it quiet."""
    spec, started = start(task, **kwargs)
    folder = Path(started["folder"])
    plans = len(plan_for(spec))
    if wait_print:
        print(f"training {spec.name}: {plans} trial(s) of {', '.join(spec.algos)}, log: {folder / store.LOG_FILE}")
    try:
        status = JOBS.wait(folder, _print_progress if wait_print else None)
    except KeyboardInterrupt:
        status = JOBS.stop(folder)
    if status.get("state") == "failed":
        raise TaskError(f"training failed: {status.get('error')}\n" + "\n".join(status.get("traceback_tail", [])))
    return TrainedPolicy.load(folder)


def _print_progress(status: dict[str, Any]) -> None:
    lead = status.get("leader")
    parts = [f"{status.get('state', '?'):8s}", f"{100 * status.get('progress', 0):5.1f}%"]
    if "mean_episode_reward" in status:
        parts.append(f"reward {status['mean_episode_reward']:8.2f}")
    if lead:
        parts.append(f"leader {lead}")
    print("  ".join(parts), flush=True)
