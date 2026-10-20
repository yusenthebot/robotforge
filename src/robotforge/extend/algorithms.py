"""Plugin algorithms and tasks for training: `@algorithm` and `@task`.

An algorithm is a function `run(ctx)` that spends the step budget of a training context and returns a policy (see
robotforge.training.algorithms for the whole contract and an example). A task is a function that builds a
`Task` for a robot: `build(model, robot, params)`; robotforge.training.custom has a ready-made one that is written
as a reward function. Both appear next to the built-in ones: `list_algorithms`, `train_policy(algo=...)`, and a
hyperparameter search can draw from an algorithm's `Hyper` ranges.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from robotforge.extend.registry import REGISTRY, Item, PluginError


def algorithm(
    name: str,
    *,
    summary: str = "",
    hyper: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    needs: tuple[str, ...] = (),
    explain: str = "",
) -> Callable[[Any], Any]:
    """Register a training algorithm, used as `train_policy(algo=name)`.

    The decorated function `run(ctx)` (or a class with `__init__(self, ctx)` and `run(self)`) spends the context's
    budget and returns the policy: its flat parameter vector (a numpy array, for the shape set with
    `ctx.new_shape(hidden)`), or an object with `__call__(obs)` and `save(path)`.
    hyper: its settings as `Hyper(default, low=, high=, log=, choices=, help=)`; a search draws from the ranges.
    params: plain defaults, `{"sigma": 0.1}`, for settings that need no range.
    needs: modules it imports, so a missing one is explained before training starts."""

    def register(obj: Any) -> Any:
        from robotforge.training.hyper import Hyper, HyperError

        if inspect.isclass(obj):
            if not callable(getattr(obj, "run", None)):
                raise PluginError(f"algorithm {name!r}: the class needs a run(self) method")
        elif not callable(obj):
            raise PluginError(f"algorithm {name!r}: decorate a function run(ctx) or a class with run(self)")
        settings: dict[str, Hyper] = {}
        try:
            for key, value in (params or {}).items():
                settings[key] = value if isinstance(value, Hyper) else Hyper(value)
            for key, value in (hyper or {}).items():
                if not isinstance(value, Hyper):
                    raise PluginError(f"algorithm {name!r}: hyper[{key!r}] must be Hyper(default, low=, high=)")
                settings[key] = value
        except HyperError as exc:
            raise PluginError(f"algorithm {name!r}: {exc}") from exc
        from robotforge.training.algorithms import BUILT_IN

        if name in BUILT_IN:
            raise PluginError(f"algorithm {name!r}: robotforge already has an algorithm of that name; pick another")
        described = {k: h.describe() for k, h in settings.items()}
        REGISTRY.add(
            Item(
                "algorithm",
                name,
                summary or f"the {name} training algorithm (from a plugin)",
                obj,
                meta={
                    "hyper": settings,
                    "needs": tuple(needs),
                    "explain": explain,
                    "describe": {"settings": described, **({"needs": list(needs)} if needs else {})},
                },
            )
        )
        return obj

    return register


def task(name: str, *, summary: str = "", params: dict[str, Any] | None = None) -> Callable[[Any], Any]:
    """Register a task for training, used as `train_policy(task=name)` on any robot.

    The decorated function `build(model, robot, params)` returns a robotforge Task for `robot` in the compiled
    `model` (robotforge.training.custom.RewardTask turns a reward function into one)."""

    def register(fn: Any) -> Any:
        from robotforge.control.tasks import TASKS

        if not callable(fn) or inspect.isclass(fn):
            raise PluginError(f"task {name!r}: decorate a function build(model, robot, params)")
        if name in TASKS:
            raise PluginError(f"task {name!r}: robotforge already has a task of that name; pick another")
        defaults = dict(params or {})
        REGISTRY.add(
            Item(
                "task",
                name,
                summary or f"the {name} task (from a plugin)",
                fn,
                meta={"build": fn, "params": defaults, "describe": {"options": defaults}},
            )
        )
        return fn

    return register
