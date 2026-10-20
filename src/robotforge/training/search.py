"""Hyperparameter search: which trials to run, and which to keep.

Random search runs every trial for the full budget. Successive halving ("halving") starts all trials on a small
budget, keeps the best third, continues those on a bigger one, and so on, so most of the compute goes to the
promising settings. A trial is one algorithm with one set of settings and one seed; `algos` with several names
searches over the algorithm too, in turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from robotforge.control.tasks import TaskError
from robotforge.training import algorithms
from robotforge.training.hyper import Hyper, HyperError

MAX_TRIALS = 64
METHODS = ("random", "halving")


class SearchSpec(BaseModel):
    """How to search. space: setting name -> a list of values to pick from, {"low", "high", "log"} to draw between,
    or one fixed value; settings left out of `space` are searched over their own ranges when `all_settings`."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["random", "halving"] = "random"
    trials: int = Field(6, ge=1, le=MAX_TRIALS)
    eta: int = Field(3, ge=2, le=6, description="halving keeps one in eta trials at each rung")
    space: dict[str, Any] = Field(default_factory=dict)
    all_settings: bool = True
    seed: int = 0


@dataclass(frozen=True)
class TrialPlan:
    id: str
    algo: str
    params: dict[str, Any]
    seed: int


def _hyper_for(name: str, spec: Any, algo: algorithms.Algorithm) -> Hyper:
    """The range the search draws `name` from: the user's, else the algorithm's own."""
    base = algo.hyper[name]
    if isinstance(spec, list):
        if not spec:
            raise TaskError(f"search space for {name!r}: the list of values to pick from is empty")
        try:
            return Hyper(spec[0], choices=tuple(spec), help=base.help)
        except HyperError as exc:
            raise TaskError(f"search space for {name!r}: {exc}") from exc
    if isinstance(spec, dict):
        unknown = sorted(set(spec) - {"low", "high", "log"})
        if unknown or "low" not in spec or "high" not in spec:
            raise TaskError(f"search space for {name!r} is a list of values or {{'low', 'high', 'log'}}")
        try:
            return Hyper(
                type(base.default)(spec["low"]),
                low=spec["low"],
                high=spec["high"],
                log=bool(spec.get("log", False)),
                help=base.help,
            )
        except HyperError as exc:
            raise TaskError(f"search space for {name!r}: {exc}") from exc
    return Hyper(spec, choices=(spec,), help=base.help)  # a fixed value


def sample_params(
    algo: algorithms.Algorithm, spec: SearchSpec, fixed: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """One trial's settings: `fixed` ones as given, searched ones drawn, the rest default."""
    out = dict(fixed)
    names = [n for n in algo.hyper if n in spec.space or (spec.all_settings and algo.hyper[n].searchable)]
    for name in names:
        if name in fixed:
            continue
        hyper = _hyper_for(name, spec.space[name], algo) if name in spec.space else algo.hyper[name]
        out[name] = hyper.sample(rng)
    return algo.params(out)


def check_space(space: dict[str, Any], algos: list[algorithms.Algorithm]) -> None:
    known = {n for a in algos for n in a.hyper}
    unknown = sorted(set(space) - known)
    if unknown:
        raise TaskError(
            f"search space names {', '.join(map(repr, unknown))}, which no chosen algorithm has; "
            f"settings: {', '.join(sorted(known))}"
        )


def plan_trials(spec: SearchSpec, names: list[str], fixed: dict[str, dict[str, Any]] | None = None) -> list[TrialPlan]:
    """The trials of a search: algorithms in turn, each with drawn settings and its own seed."""
    algos = [algorithms.get(n) for n in names]
    check_space(spec.space, algos)
    rng = np.random.default_rng(spec.seed)
    fixed = fixed or {}
    plans = []
    for i in range(spec.trials):
        algo = algos[i % len(algos)]
        params = sample_params(algo, spec, fixed.get(algo.name, {}), rng)
        plans.append(TrialPlan(f"t{i + 1:02d}", algo.name, params, int(rng.integers(0, 2**31 - 1))))
    return plans


def single_trial(name: str, params: dict[str, Any] | None, seed: int) -> list[TrialPlan]:
    algo = algorithms.get(name)
    return [TrialPlan("t01", algo.name, algo.params(params), seed)]


def rungs(method: str, trials: int, eta: int) -> list[tuple[float, int]]:
    """(share of the full budget reached by the end of the rung, trials that run it), for each rung."""
    if method != "halving" or trials < 2:
        return [(1.0, trials)]
    count = max(2, int(math.floor(math.log(trials, eta) + 1e-9)) + 1)
    out, alive = [], trials
    for r in range(count):
        out.append((1.0 / eta ** (count - 1 - r), alive))
        alive = max(1, math.ceil(alive / eta))
    return out


def survivors(scores: dict[str, tuple[float, float]], keep: int) -> list[str]:
    """The ids of the `keep` best trials: by mean test return, then success rate (ties by id, so it is stable)."""
    order = sorted(scores, key=lambda i: (-scores[i][0], -scores[i][1], i))
    return order[:keep]
