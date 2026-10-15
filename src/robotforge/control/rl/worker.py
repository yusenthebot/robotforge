"""Training environments, and a lean worker process that steps one of them.

Stable-Baselines3's own SubprocVecEnv workers import SB3, which imports PyTorch: about 1.5 s and
200+ MB per worker, for processes that only run physics. These workers speak the same protocol
but import only robotforge, MuJoCo and gymnasium. Nothing here may import torch.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np

from robotforge.control.env import RobotEnv
from robotforge.control.gym_compat import as_gymnasium
from robotforge.control.rl.spec import TrainSpec, build_env


def make_env(spec_data: dict[str, Any], rank: int = 0) -> Any:
    """A gymnasium env for training (picklable as functools.partial(make_env, spec_data, rank))."""
    spec = TrainSpec.model_validate(spec_data)
    env = as_gymnasium(build_env(spec))
    return Shoves(env, spec.push) if spec.push > 0 else env


class Shoves(gymnasium.Wrapper):
    """Shove the robot at random moments (every 1.5 to 4 s, random strength and side) while it trains.

    A policy that only ever saw calm episodes falls at the first push; one that was shoved while
    learning learns to recover. These are the same shoves the push tests use (task.push)."""

    def __init__(self, env: gymnasium.Env, strength: float):
        super().__init__(env)
        self.strength = float(strength)
        self.inner: RobotEnv = env.unwrapped.inner
        self.next_shove = np.inf

    def _schedule(self) -> None:
        self.next_shove = float(self.inner.data.time) + self.inner.np_random.uniform(1.5, 4.0)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._schedule()
        return out

    def step(self, action):
        inner = self.inner
        if inner.data.time >= self.next_shove:
            rng = inner.np_random
            inner.task.push(inner.data, rng.uniform(0.3, 1.0) * self.strength * rng.choice([-1.0, 1.0]))
            self._schedule()
        return self.env.step(action)


def worker(remote, parent_remote, env_fn) -> None:
    """Serve one environment over a pipe, with Stable-Baselines3's SubprocVecEnv protocol."""
    parent_remote.close()
    env = env_fn()
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                info["terminal_observation"] = obs
                obs, reset_info = env.reset()
            remote.send((obs, reward, done, info, reset_info))
        elif cmd == "reset":
            options = {"options": data[1]} if data[1] else {}
            obs, reset_info = env.reset(seed=data[0], **options)
            remote.send((obs, reset_info))
        elif cmd == "render":
            remote.send(env.render())
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        elif cmd == "env_method":
            remote.send(env.get_wrapper_attr(data[0])(*data[1], **data[2]))
        elif cmd == "get_attr":
            remote.send(env.get_wrapper_attr(data))
        elif cmd == "has_attr":
            try:
                env.get_wrapper_attr(data)
                remote.send(True)
            except AttributeError:
                remote.send(False)
        elif cmd == "set_attr":
            setattr(env, data[0], data[1])
            remote.send(None)
        elif cmd == "is_wrapped":
            remote.send(_is_wrapped(env, data))
        else:
            raise NotImplementedError(f"worker: unknown command {cmd!r}")


def _is_wrapped(env, wrapper_class) -> bool:
    while env is not None and hasattr(env, "env"):
        if isinstance(env, wrapper_class):
            return True
        env = env.env
    return False
