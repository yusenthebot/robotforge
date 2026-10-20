"""PPO and SAC as training algorithms (need the optional rl extra; skipped without it)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from robotforge.control.rl import TrainedPolicy
from robotforge.training.envs import build_env, env_spec
from robotforge.training.runs import make_run_spec, run_inline
from robotforge.training.search import SearchSpec
from robotforge.training.trial import evaluate_policy

HAVE_RL = all(importlib.util.find_spec(m) is not None for m in ("gymnasium", "stable_baselines3", "torch"))
pytestmark = pytest.mark.skipif(not HAVE_RL, reason="the optional rl extra is not installed (uv sync --extra rl)")


def _env() -> dict:
    return env_spec("balance").model_dump(mode="json")


def test_ppo_and_sac_train_report_losses_and_leave_numpy_policies(tmp_path):
    spec = make_run_spec("nn", _env(), algos=["ppo", "sac"], steps=1_200, parallel=0, eval_episodes=2)
    spec = spec.model_copy(update={"params": {"sac": {"learning_starts": 200}}})
    doc = run_inline(spec, tmp_path / "nn")
    by_algo = {t["algo"]: t for t in doc["trials"]}
    assert {"value_loss", "policy_gradient_loss"} <= set(by_algo["ppo"]["metrics"])
    assert {"actor_loss", "critic_loss"} <= set(by_algo["sac"]["metrics"])
    assert all(t["state"] == "done" and t["returns"] for t in by_algo.values())
    policy = TrainedPolicy.load(tmp_path / "nn")  # plain arrays: it runs without torch
    result = evaluate_policy(build_env(policy.meta["env"]), policy.net, 2)
    assert np.isfinite(result["mean_return"])


def test_a_halving_search_continues_a_ppo_trial_from_its_saved_model(tmp_path):
    spec = make_run_spec(
        "nn_search",
        _env(),
        algos=["ppo"],
        steps=2_000,
        parallel=0,
        eval_episodes=2,
        search=SearchSpec(method="halving", trials=3, eta=3, seed=0, space={"learning_rate": [3e-4]}),
    )
    doc = run_inline(spec, tmp_path / "nn_search")
    done = [t for t in doc["trials"] if t["state"] == "done"]
    pruned = [t for t in doc["trials"] if t["state"] == "pruned"]
    assert len(done) == 1 and len(pruned) == 2
    assert done[0]["steps"] > 2 * max(p["steps"] for p in pruned)  # it went on from where it stopped
    assert (tmp_path / "nn_search" / "trials" / done[0]["id"] / "model.zip").exists()
