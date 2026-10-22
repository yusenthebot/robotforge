"""What to train and how: the training spec, with defaults per task that work without tuning.

The spec is plain data (it is written to train.json and read by the training process), so the
robot is described the way world.yaml describes it (template, library robot or file, with its
parameters and pose) and the training process rebuilds the same robot in a small world of its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from robotforge import wording
from robotforge.control.env import RobotEnv, compile_task_world
from robotforge.control.methods import base_method
from robotforge.control.tasks import TASKS, Task, TaskError, make_task
from robotforge.control.tasks.base import readable


@dataclass(frozen=True)
class TaskDefaults:
    """Settings that learn each task reliably on a laptop CPU, found by experiment (docs/ALGORITHMS.md)."""

    algo: str
    action_mode: str
    timesteps: int
    push: float  # strongest training shove (0 = none); shoves teach the policy to recover
    steps_per_second: float  # rough simulation speed of one process, only for time estimates
    ppo: dict[str, Any]
    sac: dict[str, Any]
    net: tuple[int, ...] = (64, 64)
    # Fast simulations (balance) step quicker in one process than the pipes to worker processes cost.
    inline: bool = False
    # Residual mode: the largest correction, as a share of each motor's half range. Arms need fine
    # corrections (a servo's half range is up to pi rad); legs need room to change the stride.
    residual_scale: float = 0.2


_PPO = {"n_steps": 256, "batch_size": 256, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "learning_rate": 3e-4}
_SAC = {"learning_rate": 3e-4, "batch_size": 256, "gamma": 0.99, "learning_starts": 2000, "buffer_size": 300_000}

DEFAULTS: dict[str, TaskDefaults] = {
    "balance": TaskDefaults("ppo", "direct", 150_000, 1.5, 19_000, {**_PPO}, {**_SAC}, inline=True),
    "reach": TaskDefaults("sac", "direct", 150_000, 0.0, 9_500, {**_PPO}, {**_SAC}, (256, 256), residual_scale=0.05),
    "stand": TaskDefaults("ppo", "residual", 400_000, 2.0, 4_700, {**_PPO, "batch_size": 512}, {**_SAC}, (128, 128)),
    "walk": TaskDefaults("ppo", "residual", 600_000, 1.2, 1_600, {**_PPO, "batch_size": 512}, {**_SAC}, (128, 128)),
    "drive_to": TaskDefaults("ppo", "direct", 300_000, 0.0, 4_300, {**_PPO}, {**_SAC}),
}

MAX_TIMESTEPS = 20_000_000
MAX_MINUTES = 240.0
MAX_ENVS = 32


def default_envs(algo: str) -> int:
    """Leave a couple of cores for the viewer and the agent; SAC learns from one stream of experience."""
    if algo == "sac":
        return 1
    cpus = os.cpu_count() or 2
    return max(1, min(8, cpus - 2))


class RobotSpec(BaseModel):
    """The robot to train, described like a world.yaml entry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    template: str | None = None
    library: str | None = None
    file: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    yaw: float = 0.0


class TrainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    task: str
    task_params: dict[str, Any] = Field(default_factory=dict)
    robot: RobotSpec
    floor: dict[str, Any] = Field(default_factory=dict)
    world_dir: str | None = None  # resolves robot files given relative to a world
    algo: Literal["ppo", "sac"]
    action_mode: Literal["direct", "residual"]
    residual_scale: float = Field(0.2, gt=0, le=1.0)
    timesteps: int = Field(gt=0, le=MAX_TIMESTEPS)
    max_minutes: float | None = Field(None, gt=0, le=MAX_MINUTES)
    n_envs: int = Field(gt=0, le=MAX_ENVS)
    seed: int = 0
    push: float = Field(0.0, ge=0, le=5.0)
    episode_seconds: float | None = Field(None, gt=0, le=60)
    net: list[int] = Field(default_factory=lambda: [64, 64])
    hyperparams: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fits(self) -> TrainSpec:
        if self.task not in TASKS:
            raise ValueError(f"unknown task {wording.short(self.task)}; choose one of: {', '.join(TASKS)}")
        if not self.net or any(not 1 <= n <= 1024 for n in self.net) or len(self.net) > 4:
            raise ValueError("net is 1 to 4 layer sizes between 1 and 1024, e.g. [64, 64]")
        return self


def make_spec(
    name: str,
    task: str,
    robot: dict[str, Any],
    *,
    algo: str | None = None,
    action_mode: str | None = None,
    timesteps: int | None = None,
    minutes: float | None = None,
    n_envs: int | None = None,
    seed: int = 0,
    task_params: dict[str, Any] | None = None,
    floor: dict[str, Any] | None = None,
    world_dir: str | None = None,
    push: float | None = None,
    residual_scale: float | None = None,
    hyperparams: dict[str, Any] | None = None,
    **extra: Any,
) -> TrainSpec:
    """A complete spec from what the user said plus the task's defaults."""
    if task not in DEFAULTS:
        raise TaskError(f"unknown task {wording.short(task)}; choose one of: {', '.join(DEFAULTS)}")
    d = DEFAULTS[task]
    algo = algo or d.algo
    if algo not in ("ppo", "sac"):
        raise TaskError(f"algo is 'ppo' or 'sac', not {algo!r}")
    if minutes is not None and timesteps is None:
        timesteps = MAX_TIMESTEPS  # the clock decides
    try:
        return TrainSpec(
            name=name,
            task=task,
            task_params=task_params or {},
            robot=RobotSpec.model_validate(robot),
            floor=floor or {},
            world_dir=world_dir,
            algo=algo,
            action_mode=action_mode or d.action_mode,
            timesteps=d.timesteps if timesteps is None else timesteps,
            max_minutes=minutes,
            n_envs=n_envs or default_envs(algo),
            seed=seed,
            push=d.push if push is None else push,
            residual_scale=d.residual_scale if residual_scale is None else residual_scale,
            net=list(d.net),
            hyperparams=hyperparams or {},
            **extra,
        )
    except ValidationError as exc:
        raise TaskError(readable(exc, "training settings")) from exc


def algo_settings(spec: TrainSpec) -> dict[str, Any]:
    """The algorithm's keyword arguments: task defaults, then the user's overrides."""
    d = DEFAULTS[spec.task]
    base = dict(d.ppo if spec.algo == "ppo" else d.sac)
    if spec.algo == "ppo":
        # One rollout is n_steps per environment; keep the minibatch a divisor of it.
        rollout = base["n_steps"] * spec.n_envs
        base["batch_size"] = min(base["batch_size"], rollout)
        while rollout % base["batch_size"]:
            base["batch_size"] //= 2
    return {**base, **spec.hyperparams}


def estimate_seconds(spec: TrainSpec) -> float:
    """A rough wall-clock estimate before training starts (the status reports the real pace)."""
    if spec.max_minutes is not None and spec.timesteps >= MAX_TIMESTEPS:
        return 60.0 * spec.max_minutes
    d = DEFAULTS[spec.task]
    rate = d.steps_per_second * (0.6 * spec.n_envs if spec.n_envs > 1 else 0.8)
    if spec.algo == "sac":
        rate = min(rate, 400.0)  # one network update per step
    seconds = spec.timesteps / rate + 10.0  # plus start-up and the final check
    return min(seconds, 60.0 * spec.max_minutes) if spec.max_minutes else seconds


def build_task(spec: TrainSpec) -> Task:
    """The task on the spec's robot, alone on a floor like the world's (the same robot, rebuilt)."""
    from robotforge.design.world import World

    try:
        world = World.model_validate(
            {"name": "training", "floor": spec.floor, "robots": [spec.robot.model_dump(exclude_none=True)]}
        )
    except ValidationError as exc:
        raise TaskError(readable(exc, "robot")) from exc
    model = compile_task_world(spec.task, world, Path(spec.world_dir) if spec.world_dir else None)
    return make_task(spec.task, model, spec.robot.name, spec.task_params)


def build_env(spec: TrainSpec) -> RobotEnv:
    """The environment the policy trains in (each training process builds its own)."""
    return RobotEnv(
        build_task(spec),
        action_mode=spec.action_mode,
        residual_scale=spec.residual_scale,
        episode_seconds=spec.episode_seconds,
    )


def describe_env(spec: TrainSpec) -> dict[str, Any]:
    """Build the training environment once (which also checks the spec fits) and record what the
    runtime needs to run the policy exactly as it was trained."""
    from robotforge.control.rl.runtime import actuator_names

    env = build_env(spec)
    try:
        task = env.task
        return {
            "observation": task.observation_labels(),
            "actuators": actuator_names(task),
            "control_dt": env.control_dt,
            "physics_timestep": float(env.model.opt.timestep),
            "episode_seconds": env.episode_seconds,
            "base_method": base_method(task) if spec.action_mode == "residual" else None,
        }
    finally:
        env.close()
