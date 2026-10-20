"""The algorithm registry: ARS and CEM in numpy, PPO and SAC through the optional rl extra, and plugins' own.

An algorithm is a function `run(ctx)` (see trial.TrainContext) that spends the context's step budget and returns
the policy it found: its flat parameter vector, or a policy object. Its settings are `Hyper`s, which is what a
hyperparameter search draws from. Plugins register algorithms with `robotforge.extend.algorithm`:

    @algorithm("hill_climb", hyper={"sigma": Hyper(0.1, low=0.01, high=1.0, log=True)})
    def hill_climb(ctx):
        theta = ctx.state.get("theta", ctx.new_shape().zeros())
        best = ctx.evaluate(theta)
        while ctx.has_budget():
            candidate = theta + ctx.params["sigma"] * ctx.rng.standard_normal(theta.shape)
            score = ctx.evaluate(candidate)
            if score > best:
                theta, best = candidate, score
            ctx.log(best_return=best)
        ctx.state["theta"] = theta
        return theta
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robotforge import wording
from robotforge.control.tasks import TaskError
from robotforge.training.hyper import Hyper, HyperError

RL_HINT = (
    "needs the optional 'rl' extra (gymnasium, Stable-Baselines3, PyTorch CPU), which is not installed: "
    "uv sync --extra rl (in a robotforge checkout) or pip install 'robotforge[rl]'"
)


@dataclass
class Algorithm:
    name: str
    summary: str
    run: Callable[[Any], Any]
    hyper: dict[str, Hyper] = field(default_factory=dict)
    source: str = "built in"
    needs: tuple[str, ...] = ()  # importable modules it requires
    explain: str = ""

    def params(self, given: dict[str, Any] | None) -> dict[str, Any]:
        """The settings: defaults, then `given`, each checked. An unknown name is refused with the known ones."""
        given = dict(given or {})
        unknown = sorted(set(given) - set(self.hyper))
        if unknown:
            raise TaskError(
                f"{self.name} has no setting {', '.join(map(repr, unknown))}; its settings: "
                f"{', '.join(self.hyper) or 'none'}"
            )
        out = {k: h.default for k, h in self.hyper.items()}
        for key, value in given.items():
            try:
                out[key] = self.hyper[key].check(f"{self.name}.{key}", value)
            except HyperError as exc:
                raise TaskError(str(exc)) from exc
        return out

    def missing(self) -> list[str]:
        return [m for m in self.needs if importlib.util.find_spec(m) is None]

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "summary": self.summary,
            "from": self.source,
            "settings": {k: h.describe() for k, h in self.hyper.items()},
        }
        if self.needs:
            out["needs"] = list(self.needs)
            out["installed"] = not self.missing()
        return out


def need(algo: Algorithm) -> None:
    missing = algo.missing()
    if missing:
        raise TaskError(f"{algo.name} {RL_HINT} (missing: {', '.join(missing)})")


# ---- ARS -------------------------------------------------------------------------------------------------


def run_ars(ctx) -> np.ndarray:
    """Augmented Random Search (Mania, Guy and Recht, 2018): a linear policy nudged along the random directions
    that earned the most, with scaled inputs."""
    p = ctx.params
    shape = ctx.new_shape(())
    theta = np.array(ctx.state.get("theta", shape.zeros()), dtype=np.float64)
    directions = int(p["directions"])
    top = max(1, min(directions, round(float(p["top_fraction"]) * directions)))
    ctx.update_norm(center=False)
    while ctx.has_budget():
        deltas = ctx.rng.standard_normal((directions, shape.n_params))
        # Each +/- pair starts from the same random state, so their difference measures the nudge, not luck.
        starts = ctx.rng.integers(0, 2**31 - 1, directions)
        plus = np.array(
            [
                ctx.rollout(ctx.policy(theta + p["noise"] * d), seed=int(s)).ret
                for d, s in zip(deltas, starts, strict=True)
            ]
        )
        minus = np.array(
            [
                ctx.rollout(ctx.policy(theta - p["noise"] * d), seed=int(s)).ret
                for d, s in zip(deltas, starts, strict=True)
            ]
        )
        best = np.argsort(-np.maximum(plus, minus))[:top]
        spread = np.concatenate([plus[best], minus[best]]).std() + 1e-8
        step = p["step_size"] / (top * spread) * np.tensordot(plus[best] - minus[best], deltas[best], axes=1)
        theta = theta + step
        ctx.update_norm(center=False)
        ctx.state["theta"] = theta
        ctx.log(
            mean_return=float(np.mean(np.concatenate([plus, minus]))),
            best_direction_return=float(max(plus.max(), minus.max())),
            step_size_used=float(np.linalg.norm(step)),
        )
    ctx.state["theta"] = theta
    return theta


# ---- CEM -------------------------------------------------------------------------------------------------


def run_cem(ctx) -> np.ndarray:
    """The cross-entropy method: a Gaussian over the policy's parameters, refit to its best samples each round."""
    p = ctx.params
    hidden = (int(p["hidden"]),) if int(p["hidden"]) > 0 else ()
    shape = ctx.new_shape(hidden)
    n = shape.n_params
    mean = np.array(ctx.state["mean"], dtype=np.float64) if "mean" in ctx.state else (
        shape.random(ctx.rng, 1.0) if hidden else shape.zeros()
    )  # fmt: skip
    std = np.array(ctx.state["std"], dtype=np.float64) if "std" in ctx.state else np.full(n, float(p["init_std"]))
    population = int(p["population"])
    elite = max(2, round(float(p["elite_fraction"]) * population))
    episodes = int(p["episodes"])
    floor = float(p["noise_floor"])
    freeze_at = 0.4 * ctx.budget  # the observation scale settles early; the parameters then stop chasing it
    ctx.update_norm(center=True)
    while ctx.has_budget():
        samples = mean + std * ctx.rng.standard_normal((population, n))
        samples[0] = mean  # the current best guess is always one of the candidates
        starts = [int(s) for s in ctx.rng.integers(0, 2**31 - 1, episodes)]  # every candidate faces the same starts
        returns = np.array([ctx.evaluate(theta, episodes, starts) for theta in samples])
        best = samples[np.argsort(-returns)[:elite]]
        mean = (1 - p["smoothing"]) * mean + p["smoothing"] * best.mean(axis=0)
        std = (1 - p["smoothing"]) * std + p["smoothing"] * best.std(axis=0) + floor
        if ctx.steps_used < freeze_at:
            ctx.update_norm(center=True)
        ctx.state.update(mean=mean, std=std)
        ctx.log(
            mean_return=float(returns.mean()),
            elite_return=float(np.sort(returns)[-elite:].mean()),
            spread=float(std.mean()),
        )
    return mean


# ---- PPO and SAC (optional extra) --------------------------------------------------------------------------


def run_ppo(ctx):
    from robotforge.training import sb3

    return sb3.run(ctx, "ppo")


def run_sac(ctx):
    from robotforge.training import sb3

    return sb3.run(ctx, "sac")


_RL = ("gymnasium", "stable_baselines3", "torch")

BUILT_IN: dict[str, Algorithm] = {
    "ars": Algorithm(
        "ars",
        "Augmented Random Search: a linear policy nudged along the random directions that earned the most. numpy only.",
        run_ars,
        {
            "directions": Hyper(
                8, 4, 32, minimum=1, help="random directions tried per round (each costs two episodes)"
            ),
            "top_fraction": Hyper(0.5, 0.25, 1.0, minimum=0.05, maximum=1.0, help="the share of directions that count"),
            "step_size": Hyper(
                0.1, 0.01, 0.5, log=True, minimum=1e-4, maximum=5.0, help="how far the policy moves per round"
            ),
            "noise": Hyper(
                0.3, 0.05, 0.8, log=True, minimum=1e-4, maximum=5.0, help="how far each direction nudges the parameters"
            ),
        },
        explain="Tries nudges in both directions, keeps the best, steps toward them. Fast and simple; a linear policy.",
    ),
    "cem": Algorithm(
        "cem",
        "Cross-entropy method: sample policies from a Gaussian, refit it to the best. numpy only.",
        run_cem,
        {
            "population": Hyper(24, 8, 40, minimum=4, help="policies sampled per round"),
            "elite_fraction": Hyper(
                0.25, 0.1, 0.5, minimum=0.02, maximum=0.9, help="the share of the population the Gaussian is refit to"
            ),
            "init_std": Hyper(
                0.5, 0.1, 1.5, log=True, minimum=1e-3, maximum=10.0, help="how widely the first round samples"
            ),
            "noise_floor": Hyper(
                0.01,
                0.001,
                0.1,
                log=True,
                minimum=0.0,
                maximum=1.0,
                help="spread added each round so the search never stalls",
            ),
            "smoothing": Hyper(
                0.7, 0.3, 1.0, minimum=0.05, maximum=1.0, help="how much of the new fit replaces the old (1 = all)"
            ),
            "episodes": Hyper(
                1, 1, 3, minimum=1, maximum=10, help="episodes per candidate, on the same starts for every candidate"
            ),
            "hidden": Hyper(
                0, 0, 32, search=False, minimum=0, maximum=256, help="neurons in a hidden layer (0 = a linear policy)"
            ),
        },
        explain="Draws a whole population, keeps the best few and moves the Gaussian onto them.",
    ),
    "ppo": Algorithm(
        "ppo",
        "PPO (Stable-Baselines3): a neural network policy improved in small safe steps from batches of experience.",
        run_ppo,
        {
            "learning_rate": Hyper(
                3e-4, 1e-4, 3e-3, log=True, minimum=1e-6, maximum=1.0, help="size of each network update"
            ),
            "gamma": Hyper(0.99, 0.95, 0.999, minimum=0.5, maximum=0.9999, help="how far ahead rewards count"),
            "n_steps": Hyper(256, choices=(128, 256, 512, 1024), help="steps collected before each update"),
            "batch_size": Hyper(128, choices=(64, 128, 256), help="minibatch size of an update"),
            "n_epochs": Hyper(10, 4, 20, minimum=1, maximum=100, help="passes over a batch per update"),
            "ent_coef": Hyper(0.0, 0.0, 0.02, minimum=0.0, maximum=1.0, help="reward for staying random (exploration)"),
            "clip_range": Hyper(
                0.2, 0.1, 0.4, minimum=0.01, maximum=1.0, help="how far one update may change the policy"
            ),
            "width": Hyper(64, choices=(32, 64, 128), help="neurons per hidden layer (two layers)"),
        },
        needs=_RL,
        explain="Collects a batch of experience, then shifts the network toward better actions without going too far.",
    ),
    "sac": Algorithm(
        "sac",
        "SAC (Stable-Baselines3): an actor and a critic learning from a replay memory of everything tried.",
        run_sac,
        {
            "learning_rate": Hyper(
                3e-4, 1e-4, 3e-3, log=True, minimum=1e-6, maximum=1.0, help="size of each network update"
            ),
            "gamma": Hyper(0.99, 0.95, 0.999, minimum=0.5, maximum=0.9999, help="how far ahead rewards count"),
            "batch_size": Hyper(256, choices=(128, 256, 512), help="minibatch size of an update"),
            "tau": Hyper(
                0.005, 0.001, 0.05, log=True, minimum=1e-5, maximum=1.0, help="how fast the target networks follow"
            ),
            "learning_starts": Hyper(
                1000, 200, 5000, search=False, minimum=0, maximum=1_000_000, help="random steps before learning begins"
            ),
            "width": Hyper(128, choices=(64, 128, 256), help="neurons per hidden layer (two layers)"),
        },
        needs=_RL,
        explain="Learns from a memory of past steps; a second network scores actions, the policy follows it.",
    ),
}


# ---- the registry, plugins included ----------------------------------------------------------------------


def _from_plugin(item: Any) -> Algorithm:
    obj = item.obj
    if inspect.isclass(obj):

        def run(ctx: Any) -> Any:
            return obj(ctx).run()

    else:
        run = obj
    meta = item.meta
    return Algorithm(
        item.name,
        item.summary,
        run,
        dict(meta["hyper"]),
        source=item.plugin,
        needs=tuple(meta.get("needs", ())),
        explain=meta.get("explain", ""),
    )


def all_algorithms() -> dict[str, Algorithm]:
    from robotforge.extend import REGISTRY, ensure_user_plugins

    ensure_user_plugins()
    out = dict(BUILT_IN)
    for name, item in REGISTRY.of("algorithm").items():
        out[name] = _from_plugin(item)
    return out


def get(name: str) -> Algorithm:
    found = all_algorithms()
    key = wording.squash(name) if isinstance(name, str) else name
    if key in found:
        return found[key]
    raise TaskError(wording.unknown("algorithm", name, list(found), where="built in or from plugins"))


def catalog() -> list[dict[str, Any]]:
    return [a.describe() for a in all_algorithms().values()]
