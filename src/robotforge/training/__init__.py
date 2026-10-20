"""Training any robot on any task with any algorithm: environments, algorithms, searches and the training view.

    from robotforge.training import make_env

    env = make_env("reach", robot={"name": "arm", "template": "arm", "sensors": [{"type": "encoders"}]})

Importing this package loads nothing heavy; the pieces import what they need when used.
"""

from __future__ import annotations

__all__ = ["EnvSpec", "build_env", "describe", "env_spec", "make_env"]


def __getattr__(name: str):
    if name in __all__:
        from robotforge.training import envs

        return getattr(envs, name)
    raise AttributeError(name)
