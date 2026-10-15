"""The Python way in: train a policy and get it back, ready to run.

    from robotforge.control.rl import train

    policy = train("balance", minutes=2)          # PPO, prints progress, returns when done
    print(policy.meta["final"])                   # success rate and mean return on test episodes

    from robotforge.control import compare_methods
    print(compare_methods("balance", ["lqr", "mpc", f"policy:{policy.name}"], seconds=6, push_at=3,
                          method_params={f"policy:{policy.name}": {"path": str(policy.folder)}}).table)

Training always runs in a separate process, so this works in a plain script without the
`if __name__ == "__main__":` guard that multiprocessing would otherwise demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robotforge import wording
from robotforge.control.env import DEFAULT_ROBOTS
from robotforge.control.rl import store
from robotforge.control.rl.jobs import JOBS
from robotforge.control.rl.runtime import TrainedPolicy
from robotforge.control.rl.spec import DEFAULTS, make_spec
from robotforge.control.tasks import TaskError


def unused_name(base: str, place: Path) -> str:
    name, n = base, 1
    while (place / name).exists():
        n += 1
        name = f"{base}_{n}"
    return name


def train(
    task: str,
    algo: str | None = None,
    *,
    timesteps: int | None = None,
    minutes: float | None = None,
    n_envs: int | None = None,
    seed: int = 0,
    robot_template: str | None = None,
    robot_params: dict[str, Any] | None = None,
    task_params: dict[str, Any] | None = None,
    action_mode: str | None = None,
    name: str | None = None,
    save_dir: str | Path | None = None,
    push: float | None = None,
    residual_scale: float | None = None,
    hyperparams: dict[str, Any] | None = None,
    replace: bool = False,
    quiet: bool = False,
) -> TrainedPolicy:
    """Train `algo` ("ppo" or "sac"; default per task) on `task` with the task's default robot (or
    `robot_template`), wait for it, and return the trained policy. Saved in `save_dir`/<name>
    (default $ROBOTFORGE_HOME/policies/<name>); an existing policy of that name is only overwritten
    with replace=True."""
    if task not in DEFAULT_ROBOTS:
        raise TaskError(f"unknown task {wording.short(task)}; choose one of: {', '.join(DEFAULT_ROBOTS)}")
    template, robot = DEFAULT_ROBOTS[task]
    place = Path(save_dir) if save_dir is not None else store.policies_home(None)
    spec = make_spec(
        store.check_name(name) if name else unused_name(f"{task}_{algo or DEFAULTS[task].algo}", place),
        task,
        {"name": robot, "template": robot_template or template, "params": robot_params or {}},
        algo=algo,
        action_mode=action_mode,
        timesteps=timesteps,
        minutes=minutes,
        n_envs=n_envs,
        seed=seed,
        task_params=task_params,
        push=push,
        residual_scale=residual_scale,
        hyperparams=hyperparams,
    )
    folder = place / spec.name
    started = JOBS.start(spec, folder, replace=replace)
    if not quiet:
        print(
            f"training {spec.algo.upper()} on {task} (about {started['estimate_seconds'] / 60:.1f} min), log: "
            f"{folder / store.LOG_FILE}"
        )
    try:
        status = JOBS.wait(folder, None if quiet else _print_progress)
    except KeyboardInterrupt:
        status = JOBS.stop(folder)
        if status.get("state") not in store.FINAL_STATES:
            status = JOBS.wait(folder)
    if status.get("state") == "failed":
        raise TaskError(f"training failed: {status.get('error')}\n" + "\n".join(status.get("traceback_tail", [])))
    return TrainedPolicy.load(folder)


def load_policy(name: str, world_dir: str | Path | None = None) -> TrainedPolicy:
    """A trained policy by name (open world first, then $ROBOTFORGE_HOME/policies) or by folder path."""
    path = Path(name)
    if (path / store.POLICY_FILE).exists():
        return TrainedPolicy.load(path)
    return TrainedPolicy.load(store.find_policy(name, Path(world_dir) if world_dir else None))


def _print_progress(status: dict[str, Any]) -> None:
    parts = [f"{status.get('state', '?'):8s}", f"{100 * status.get('progress', 0):5.1f}%"]
    if "mean_episode_reward" in status:
        parts.append(f"reward {status['mean_episode_reward']:8.2f}")
    if "success_rate" in status:
        parts.append(f"success {status['success_rate']:4.0%}")
    if "eta_s" in status:
        parts.append(f"eta {status['eta_s']:5.0f}s")
    print("  ".join(parts), flush=True)
