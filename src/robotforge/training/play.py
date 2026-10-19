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
from robotforge.training.envs import EnvSpec, observed, task_on


def spec_of(meta: dict[str, Any]) -> EnvSpec | None:
    env = meta.get("env")
    return EnvSpec.model_validate(env) if isinstance(env, dict) else None


def fit(meta: dict[str, Any], task: Task, model: mujoco.MjModel, world: Any = None) -> Task:
    """`task` with the observation the policy was trained on (a policy of control/rl has no `env`: unchanged)."""
    spec = spec_of(meta)
    if spec is None or spec.observation == "task":
        return task
    spec = spec.model_copy(update={"robot": task.robot.name})
    return observed(task, spec)


def controller(
    policy: TrainedPolicy, model: mujoco.MjModel, data: mujoco.MjData, robot: str, world: Any = None
) -> LearnedPolicyController:
    """The controller that runs `policy` on `robot` of `model`, with the task it was trained on."""
    spec = spec_of(policy.meta)
    if spec is None:
        raise TaskError(
            f"policy {policy.name!r} was trained by train_policy's older pipeline; run it with method='policy:...'"
        )
    spec = spec.model_copy(update={"robot": robot})
    task = task_on(spec, model, world)
    task.begin(data)
    return policy.controller(task, model, data)  # TrainedPolicy.controller fits the observation (see fit)


def attach(session: Any, robot: str, name: str, world_dir: Path | None = None, world: Any = None) -> dict[str, Any]:
    """Let the policy `name` drive `robot` in the live session until something else takes over."""
    policy = TrainedPolicy.load(store.find_policy(name, world_dir))
    with session.lock:
        ctl = controller(policy, session.model, session.data, robot, world)
        session.set_controller(robot, ctl)
    return {"robot": robot, "policy": name, **ctl.status()}
