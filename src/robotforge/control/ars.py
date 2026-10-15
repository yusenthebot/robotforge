"""Augmented Random Search: reinforcement learning with numpy only.

ARS (Mania, Guy and Recht, 2018) trains a linear policy, action = W x normalized
observation, by trying random nudges of W in both directions, keeping the nudges
that earned the most reward, and stepping W toward them. It needs no neural
network library, so it runs anywhere robotforge runs, and on small tasks like
balance it learns in seconds. It is the zero-install way to see reinforcement
learning work on a RobotEnv; deep RL (PPO, SAC) needs a neural network library: see robotforge.control.rl.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.control.actuation import ActionScale
from robotforge.control.env import RobotEnv
from robotforge.control.tasks import Task, TaskError


@dataclass
class LinearPolicy:
    """action = clip(W [obs / scale, 1], -1, 1): a weighted sum of the scaled observations plus a bias.

    Observations are only scaled, not shifted by their running mean: early episodes are mostly
    failures, so their mean is biased, and shifting by it makes the policy push even at rest."""

    weights: np.ndarray  # [n_actions, n_obs + 1]; the last column is the bias
    scale: np.ndarray  # [n_obs]

    @classmethod
    def zeros(cls, n_obs: int, n_actions: int) -> LinearPolicy:
        return cls(np.zeros((n_actions, n_obs + 1)), np.ones(n_obs))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(self.weights, obs)

    def act(self, weights: np.ndarray, obs: np.ndarray) -> np.ndarray:
        x = np.append(np.asarray(obs, float) / self.scale, 1.0)
        return np.clip(weights @ x, -1.0, 1.0).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez(path, weights=self.weights, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> LinearPolicy:
        with np.load(path) as f:
            return cls(f["weights"], f["scale"])


class _RunningStats:
    """Mean and spread of every observation seen so far (Welford), used to normalize inputs."""

    def __init__(self, n: int):
        self.count = 0
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)

    def push(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def scale(self) -> np.ndarray:
        """Typical size of each observation (root mean square), at least 0.01."""
        if self.count < 2:
            return np.ones_like(self.mean)
        rms = np.sqrt(self.m2 / (self.count - 1) + self.mean**2)
        return np.maximum(rms, 1e-2)


@dataclass
class TrainingLog:
    iterations: list[dict[str, float]] = field(default_factory=list)
    wall_seconds: float = 0.0
    env_steps: int = 0


def evaluate(env: RobotEnv, policy: Callable[[np.ndarray], np.ndarray], seed: int) -> tuple[float, int]:
    """One episode with `policy`; returns (return, steps)."""
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return total, steps


def train_ars(
    env: RobotEnv,
    iterations: int = 20,
    directions: int = 8,
    top: int = 4,
    step_size: float = 0.1,
    noise: float = 0.3,
    seed: int = 0,
    callback: Callable[[int, dict[str, float]], None] | None = None,
) -> tuple[LinearPolicy, TrainingLog]:
    """Train a linear policy on `env` with ARS (V2-t: scaled inputs, only the best directions count)."""
    if not 1 <= top <= directions:
        raise TaskError("top must be between 1 and the number of directions")
    rng = np.random.default_rng(seed)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = LinearPolicy.zeros(obs_dim, act_dim)
    stats = _RunningStats(obs_dim)
    log = TrainingLog()
    started = time.perf_counter()

    def rollout(weights: np.ndarray, start: int) -> float:
        obs, _ = env.reset(seed=start)
        total = 0.0
        while True:
            stats.push(np.asarray(obs, float))
            obs, reward, terminated, truncated, _ = env.step(policy.act(weights, obs))
            total += reward
            log.env_steps += 1
            if terminated or truncated:
                return total

    for it in range(iterations):
        deltas = rng.standard_normal((directions, act_dim, obs_dim + 1))
        # Each +/- pair starts from the same random state, so their difference measures the nudge, not luck.
        starts = rng.integers(0, 2**31, size=directions)
        plus = np.array([rollout(policy.weights + noise * d, int(s)) for d, s in zip(deltas, starts, strict=True)])
        minus = np.array([rollout(policy.weights - noise * d, int(s)) for d, s in zip(deltas, starts, strict=True)])
        best = np.argsort(-np.maximum(plus, minus))[:top]
        spread = np.concatenate([plus[best], minus[best]]).std() + 1e-8
        step = np.tensordot(plus[best] - minus[best], deltas[best], axes=1)
        policy.weights = policy.weights + step_size / (top * spread) * step
        policy.scale = stats.scale.copy()
        entry = {
            "iteration": it + 1,
            "mean_return": float(np.mean(np.concatenate([plus, minus]))),
            "best_return": float(max(plus.max(), minus.max())),
        }
        log.iterations.append(entry)
        if callback is not None:
            callback(it + 1, entry)
    log.wall_seconds = time.perf_counter() - started
    return policy, log


class PolicyController:
    """Runs a trained policy on the live world (or in a comparison) like any other controller.

    It reads the same observation and maps actions the same way as RobotEnv in direct mode,
    so a policy trained in the environment behaves the same here."""

    def __init__(
        self,
        task: Task,
        model: mujoco.MjModel,
        policy: Callable[[np.ndarray], np.ndarray],
        info: dict[str, Any] | None = None,
    ):
        self.task, self.policy = task, policy
        self.scale = ActionScale(model, task.actuators)
        self.rate_hz = task.control_hz
        self.info = info or {}

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        action = np.asarray(self.policy(self.task.observation(data)), float)
        data.ctrl[self.task.actuators] = self.scale.to_ctrl(action)

    def status(self) -> dict[str, Any]:
        return {"controller": "policy", **self.info}


def train_controller(task: Task, model: mujoco.MjModel, params: dict[str, Any] | None = None) -> PolicyController:
    """ "ars": learn a linear policy for `task` from scratch (in a private copy of the world), then run it."""
    params = dict(params or {})
    episode = float(params.pop("episode_seconds", task.default_seconds))
    env = RobotEnv(type(task)(model, task.robot.name, task.params.model_dump()), episode_seconds=episode)
    try:
        policy, log = train_ars(env, **params)
    except TypeError as exc:
        allowed = "iterations, directions, top, step_size, noise, seed, episode_seconds"
        raise TaskError(f"ars: bad params {sorted(params)}; use {allowed}") from exc
    last = log.iterations[-1] if log.iterations else {}
    info = {
        "learned": "linear policy (ARS)",
        "iterations": len(log.iterations),
        "train_seconds": round(log.wall_seconds, 2),
        "train_sim_seconds": round(log.env_steps * env.control_dt, 1),
        "final_mean_return": round(last.get("mean_return", float("nan")), 3),
    }
    return PolicyController(task, model, policy, info)
