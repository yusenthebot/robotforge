"""Play a policy that training produced: on a model and its data (the live world, a comparison, a test).

A policy that was trained on sensors needs the same observation to run: the same task numbers, and the same
sensors read in the same order. `fit` rebuilds that from the policy's folder (policy.json keeps the environment's
spec) on whatever model it is given, so `policy:<name>`, the viewer's play button and scripts all run it exactly as
it was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from robotforge.control.rl import store
from robotforge.control.rl.runtime import LearnedPolicyController, TrainedPolicy
from robotforge.control.tasks import Task, TaskError
from robotforge.training.envs import EnvSpec, ObservedTask, observed, task_on


def spec_of(meta: dict[str, Any]) -> EnvSpec | None:
    env = meta.get("env")
    return EnvSpec.model_validate(env) if isinstance(env, dict) else None


def fit(meta: dict[str, Any], task: Task, model: mujoco.MjModel, world: Any = None) -> Task:
    """`task` with the observation the policy was trained on (a policy of control/rl has no `env`: unchanged)."""
    spec = spec_of(meta)
    if spec is None or spec.observation == "task" or isinstance(task, ObservedTask):
        return task
    spec = spec.model_copy(update={"robot": task.robot.name})
    return observed(task, spec)


def controller(
    policy: TrainedPolicy,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: str,
    world: Any = None,
    task_params: dict[str, Any] | None = None,
) -> LearnedPolicyController:
    """The controller that runs `policy` on `robot` of `model`, with the task it was trained on (`task_params`
    change its settings, e.g. {"target": "ball"} for a reach policy)."""
    spec = spec_of(policy.meta)
    if spec is None:
        raise TaskError(
            f"policy {policy.name!r} was trained by train_policy's older pipeline; run it with method='policy:...'"
        )
    spec = spec.model_copy(update={"robot": robot, "task_params": {**spec.task_params, **(task_params or {})}})
    task = task_on(spec, model, world)
    if task.robot.name != robot and not task.robot.name.startswith(f"{robot}/"):
        raise TaskError(f"task {spec.task!r} belongs to robot {task.robot.name!r}, not requested robot {robot!r}")
    task.begin(data)
    return policy.controller(task, model, data)  # TrainedPolicy.controller fits the observation (see fit)


def attach(
    session: Any,
    robot: str,
    policy: TrainedPolicy | str,
    world_dir: Path | None = None,
    world: Any = None,
    task_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Let a policy (or the policy folder named `policy`) drive `robot` in the live session until something else
    takes over."""
    if isinstance(policy, str):
        policy = TrainedPolicy.load(store.find_policy(policy, world_dir))
    with session.lock:
        ctl = controller(policy, session.model, session.data, robot, world, task_params)
        session.set_controller(robot, ctl)
    return {"robot": robot, "policy": policy.name, **ctl.status()}
