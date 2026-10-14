"""Use a RobotEnv with Gymnasium-based RL libraries (Stable-Baselines3, CleanRL, ...).

    from robotforge.control import make_env
    from robotforge.control.gym_compat import as_gymnasium

    env = as_gymnasium(make_env("balance"))     # a real gymnasium.Env
    # e.g. stable_baselines3.PPO("MlpPolicy", env).learn(100_000)

gymnasium is optional: it is imported only when this function is called.
"""

from __future__ import annotations

from typing import Any

from robotforge.control.env import RobotEnv

INSTALL_HINT = 'this needs the optional RL packages: pip install "robotforge[rl]" (or: uv sync --extra rl)'


def as_gymnasium(env: RobotEnv) -> Any:
    """Wrap `env` in a gymnasium.Env subclass with gymnasium Box spaces (float32)."""
    try:
        import gymnasium
        from gymnasium import spaces
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc

    class RobotforgeEnv(gymnasium.Env):
        metadata = dict(env.metadata)

        def __init__(self) -> None:
            super().__init__()
            self.inner = env
            self.render_mode = env.render_mode
            self.observation_space = spaces.Box(
                env.observation_space.low, env.observation_space.high, dtype=env.observation_space.dtype
            )
            self.action_space = spaces.Box(env.action_space.low, env.action_space.high, dtype=env.action_space.dtype)

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            return env.reset(seed=seed, options=options)

        def step(self, action):
            return env.step(action)

        def render(self):
            return env.render()

        def close(self) -> None:
            env.close()

    return RobotforgeEnv()
