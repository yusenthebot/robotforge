"""check_env: verify that an environment keeps the Gymnasium contract that RL libraries rely on.

Written for RobotEnv (no gymnasium needed), and strict where Stable-Baselines3 is strict:
float32 Box spaces, actions normalized to [-1, 1], observations inside their space with the
right dtype, a 5-tuple from step, plain bool flags, and identical episodes for identical seeds.
"""

from __future__ import annotations

import numpy as np


class EnvCheckError(AssertionError):
    """The environment breaks the Gym-style contract; the message says how."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise EnvCheckError(message)


def check_env(env, steps: int = 60, seed: int = 123) -> None:
    """Raise EnvCheckError on the first contract violation found in `steps` random steps."""
    act, obs_space = env.action_space, env.observation_space
    for name, space in (("action_space", act), ("observation_space", obs_space)):
        for attr in ("low", "high", "shape", "dtype", "sample", "contains", "seed"):
            _require(hasattr(space, attr), f"{name} has no {attr!r}")
        _require(space.dtype == np.float32, f"{name} dtype is {space.dtype}, expected float32")
        _require(space.low.shape == space.shape and space.high.shape == space.shape, f"{name} bounds have wrong shape")
    _require(np.all(np.isfinite(act.low)) and np.all(np.isfinite(act.high)), "action bounds must be finite")
    _require(np.all(act.low == -1) and np.all(act.high == 1), "actions should be normalized to [-1, 1]")
    _require(hasattr(env, "metadata") and "render_modes" in env.metadata, "env.metadata needs 'render_modes'")
    _require(hasattr(env, "render_mode"), "env needs a render_mode attribute (None is fine)")

    def episode(actions: np.ndarray) -> tuple[list[np.ndarray], list[float], list[tuple[bool, bool]]]:
        result = env.reset(seed=seed)
        _require(isinstance(result, tuple) and len(result) == 2, "reset must return (obs, info)")
        obs, info = result
        _check_obs(obs, obs_space, "reset")
        _require(isinstance(info, dict), "reset info must be a dict")
        trace, rewards, flags = [obs], [], []
        for a in actions:
            out = env.step(a)
            five = "step must return (obs, reward, terminated, truncated, info)"
            _require(isinstance(out, tuple) and len(out) == 5, five)
            obs, reward, terminated, truncated, info = out
            _check_obs(obs, obs_space, "step")
            finite = isinstance(reward, float) and np.isfinite(reward)
            _require(finite, f"reward must be a finite float, got {reward!r}")
            _require(type(terminated) is bool and type(truncated) is bool, "terminated and truncated must be bool")
            _require(isinstance(info, dict), "step info must be a dict")
            _require(not (terminated and truncated), "an episode is either terminated or truncated, not both")
            trace.append(obs)
            rewards.append(reward)
            flags.append((terminated, truncated))
            if terminated or truncated:
                break
        return trace, rewards, flags

    act.seed(seed)
    actions = np.array([act.sample() for _ in range(steps)])
    first = episode(actions)
    second = episode(actions)
    _require(len(first[0]) == len(second[0]), "same seed and actions gave episodes of different length")
    for a, b in zip(first[0], second[0], strict=True):
        _require(np.array_equal(a, b), "same seed and actions gave different observations (not deterministic)")
    _require(first[1] == second[1], "same seed and actions gave different rewards (not deterministic)")

    env.reset(seed=seed)
    for bad, why in ((np.zeros(act.shape[0] + 1, np.float32), "wrong shape"), (np.full(act.shape, np.nan), "NaN")):
        try:
            env.step(bad)
        except ValueError:
            continue
        raise EnvCheckError(f"step accepted an action with {why}; it should raise ValueError")
    env.reset(seed=seed)
    _check_obs(env.step(np.full(act.shape, 5.0, np.float32))[0], obs_space, "step with out-of-range action")


def _check_obs(obs, space, where: str) -> None:
    _require(isinstance(obs, np.ndarray), f"{where}: observation must be a numpy array")
    _require(obs.dtype == space.dtype, f"{where}: observation dtype {obs.dtype} != space dtype {space.dtype}")
    _require(obs.shape == space.shape, f"{where}: observation shape {obs.shape} != space shape {space.shape}")
    _require(bool(np.all(np.isfinite(obs))), f"{where}: observation has NaN or infinity")
    _require(space.contains(obs), f"{where}: observation is outside observation_space")
