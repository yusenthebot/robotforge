"""Training runs: algorithms, hyperparameter search, the training process and the policies they leave."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pytest

from robotforge.control import TaskError
from robotforge.control.rl import JOBS, TrainedPolicy, store
from robotforge.extend import registry
from robotforge.training import algorithms, search
from robotforge.training.envs import env_spec
from robotforge.training.hyper import Hyper, HyperError
from robotforge.training.policy import RunningStats, Shape
from robotforge.training.runs import make_run_spec, plan_for, run_inline, start_run
from robotforge.training.search import SearchSpec


@pytest.fixture
def balance_env() -> dict:
    return env_spec("balance").model_dump(mode="json")


@pytest.fixture(autouse=True)
def no_leftover_training():
    yield
    for folder, proc in list(JOBS.procs.items()):
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        JOBS.procs.pop(folder, None)


def test_importing_the_training_package_loads_nothing_heavy():
    code = (
        "import sys, robotforge.training, robotforge.training.runs\n"
        "print(sorted(m for m in ('torch', 'gymnasium', 'stable_baselines3') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == "[]"


# ---- settings and the policy shape ----


def test_a_hyper_checks_and_draws_values_in_its_range():
    h = Hyper(0.1, low=0.01, high=1.0, log=True, minimum=0.001, maximum=2.0)
    rng = np.random.default_rng(0)
    draws = [h.sample(rng) for _ in range(200)]
    assert all(0.01 <= d <= 1.0 for d in draws) and min(draws) < 0.05 < 0.5 < max(draws)
    assert h.check("x", 1) == 1.0
    with pytest.raises(HyperError, match="between"):
        h.check("x", 5.0)
    assert h.check("x", 1.5) == 1.5  # the search range is where a search looks first, not a limit
    with pytest.raises(HyperError, match="int"):
        Hyper(3, 1, 9).check("n", 2.5)
    with pytest.raises(HyperError, match="one of"):
        Hyper(64, choices=(32, 64)).check("w", 48)
    assert isinstance(Hyper(3, 1, 9).sample(rng), int)


def test_an_algorithm_refuses_a_setting_it_does_not_have_and_lists_its_own():
    ars = algorithms.get("ars")
    assert ars.params({"noise": 0.2})["noise"] == 0.2
    with pytest.raises(TaskError, match="directions"):
        ars.params({"learning_rate": 0.1})
    with pytest.raises(TaskError, match="between"):
        ars.params({"noise": 99.0})
    with pytest.raises(TaskError, match="ars"):
        algorithms.get("arz")


def test_the_flat_parameters_round_trip_through_a_policy():
    shape = Shape(5, 3, (4,))
    rng = np.random.default_rng(0)
    theta = shape.random(rng)
    assert theta.size == shape.n_params == (5 + 1) * 4 + (4 + 1) * 3
    weights, biases = shape.unpack(theta)
    assert np.allclose(shape.pack(weights, biases), theta)
    policy = shape.policy(theta, np.zeros(5), np.ones(5))
    action = policy(rng.normal(size=5))
    assert action.shape == (3,) and np.all(np.abs(action) <= 1.0)
    with pytest.raises(ValueError, match="parameters"):
        shape.unpack(theta[:-1])
    stats = RunningStats(2)
    for x in rng.normal(3.0, 2.0, (200, 2)):
        stats.push(x)
    mean, var = stats.snapshot(center=True)
    assert np.allclose(mean, 3.0, atol=0.5) and np.allclose(var, 4.0, atol=1.5)


# ---- search planning ----


def test_halving_rungs_shrink_trials_and_grow_the_budget():
    assert search.rungs("halving", 9, 3) == [(1 / 9, 9), (1 / 3, 3), (1.0, 1)]
    assert search.rungs("halving", 6, 3) == [(1 / 3, 6), (1.0, 2)]
    assert search.rungs("random", 5, 3) == [(1.0, 5)]
    assert search.rungs("halving", 1, 3) == [(1.0, 1)]


def test_a_search_plan_is_reproducible_and_spreads_over_algorithms():
    spec = SearchSpec(method="random", trials=6, seed=3, space={"noise": {"low": 0.1, "high": 0.5, "log": True}})
    a = search.plan_trials(spec, ["ars", "cem"])
    b = search.plan_trials(spec, ["ars", "cem"])
    assert a == b
    assert [p.algo for p in a] == ["ars", "cem"] * 3
    ars = [p.params for p in a if p.algo == "ars"]
    assert len({p["noise"] for p in ars}) == len(ars) and all(0.1 <= p["noise"] <= 0.5 for p in ars)
    assert len({p["step_size"] for p in ars}) > 1  # settings left out of the space are drawn from their own ranges
    assert all(p.params["hidden"] == 0 for p in a if p.algo == "cem")  # hidden is not searched unless asked


def test_a_search_space_with_a_setting_nobody_has_is_explained():
    with pytest.raises(TaskError, match="no chosen algorithm"):
        search.plan_trials(SearchSpec(space={"banana": [1, 2]}), ["ars"])


def test_survivors_rank_by_test_return_then_success_with_a_stable_tie_break():
    scores = {"t01": (1.0, 0.0), "t02": (5.0, 0.2), "t03": (5.0, 0.6), "t04": (5.0, 0.6)}
    assert search.survivors(scores, 2) == ["t03", "t04"]


# ---- training ----


def test_ars_and_cem_learn_to_balance_and_leave_a_playable_policy(tmp_path, balance_env):
    spec = make_run_spec("pole_two", balance_env, algos=["ars", "cem"], steps=40_000, parallel=0)
    folder = tmp_path / "policies" / "pole_two"
    doc = run_inline(spec, folder)
    assert doc["state"] == "done" and doc["leader"] in {"t01", "t02"}
    assert [t["algo"] for t in doc["trials"]] == ["ars", "cem"]
    for t in doc["trials"]:
        assert t["curve"] and t["returns"] and t["metrics"]  # what the training view draws
        assert t["steps"] >= 40_000 * 0.9
        assert (folder / "trials" / t["id"] / "policy.npz").exists()
    final = doc["final"]
    assert final["mean_return"] > -10  # a policy that does nothing earns about -25 on this task
    policy = TrainedPolicy.load(folder)  # the same files control/rl leaves, so every runner finds it
    assert policy.meta["env"]["task"] == "balance" and policy.meta["final"]["trial"] == final["trial"]
    from robotforge.training.envs import build_env
    from robotforge.training.trial import evaluate_policy

    again = evaluate_policy(build_env(policy.meta["env"]), policy.net, 3)
    assert again["mean_return"] > -10


def test_halving_continues_the_survivors_from_where_they_were(tmp_path, balance_env):
    spec = make_run_spec(
        "halve",
        balance_env,
        algos=["ars"],
        steps=12_000,
        parallel=0,
        search=SearchSpec(method="halving", trials=3, eta=3, seed=1),
    )
    doc = run_inline(spec, tmp_path / "halve")
    states = {t["id"]: t for t in doc["trials"]}
    pruned = [t for t in states.values() if t["state"] == "pruned"]
    finished = [t for t in states.values() if t["state"] == "done"]
    assert len(pruned) == 2 and len(finished) == 1
    assert all(f["rung"] == 1 and f["steps"] > 2 * p["steps"] for f in finished for p in pruned)
    # a continued trial's curve runs on across both rungs
    assert max(x for x, _ in finished[0]["curve"]) > max(x for x, _ in pruned[0]["curve"])
    assert doc["final"]["trial"] in {f["id"] for f in finished}
    assert doc["search"]["method"] == "halving" and len(doc["search"]["rungs"]) == 2


def test_a_stop_request_ends_the_run_with_the_policy_so_far(tmp_path, balance_env):
    spec = make_run_spec("quick", balance_env, algos=["ars"], steps=5_000_000, parallel=0)
    folder = tmp_path / "quick"
    folder.mkdir()
    (folder / store.STOP_FILE).write_text("stop\n", encoding="utf-8")
    doc = run_inline(spec, folder)
    assert doc["state"] == "stopped" and (folder / "policy.npz").exists()


def test_a_plugin_algorithm_trains_like_a_built_in_one(tmp_path, balance_env):
    from robotforge.extend.algorithms import algorithm

    with registry.isolated():

        @algorithm("hill", hyper={"sigma": Hyper(0.2, low=0.05, high=0.5, log=True)}, params={"episodes": 1})
        def hill(ctx):
            theta = ctx.state.get("theta", ctx.new_shape().zeros()).copy()
            best = ctx.evaluate(theta)
            while ctx.has_budget():
                candidate = theta + ctx.params["sigma"] * ctx.rng.standard_normal(theta.shape)
                score = ctx.evaluate(candidate, ctx.params["episodes"])
                if score > best:
                    theta, best = candidate, score
                ctx.log(best_return=best)
            ctx.state["theta"] = theta
            return theta

        assert "hill" in algorithms.all_algorithms()
        spec = make_run_spec(
            "mine", balance_env, algos=["hill"], params={"hill": {"sigma": 0.3}}, steps=6_000, parallel=0
        )
        doc = run_inline(spec, tmp_path / "mine")
        trial = doc["trials"][0]
        assert trial["algo"] == "hill" and trial["settings"] == {"sigma": 0.3, "episodes": 1}
        assert "best_return" in trial["metrics"] and doc["state"] == "done"
        with pytest.raises(TaskError, match="no setting"):
            plan_for(make_run_spec("bad", balance_env, algos=["hill"], params={"hill": {"nope": 1}}, parallel=0))
    assert "hill" not in algorithms.all_algorithms()


def test_a_plugin_that_returns_nonsense_is_explained(tmp_path, balance_env):
    from robotforge.extend.algorithms import algorithm

    with registry.isolated():

        @algorithm("broken")
        def broken(ctx):
            return "a policy"

        spec = make_run_spec("broken", balance_env, algos=["broken"], steps=500, parallel=0)
        with pytest.raises(TaskError, match="parameter vector"):
            run_inline(spec, tmp_path / "broken")


def test_the_training_process_runs_trials_in_parallel_and_keeps_the_status_current(tmp_path, monkeypatch, balance_env):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    folder = tmp_path / "policies" / "proc"
    spec = make_run_spec(
        "proc",
        balance_env,
        algos=["ars", "cem"],
        steps=4_000,
        parallel=2,
        search=SearchSpec(method="random", trials=2, seed=0),
    )
    started = start_run(spec, folder)
    assert started["pid"] and len(started["trials"]) == 2
    seen = []
    status = JOBS.wait(folder, lambda s: seen.append(s.get("state")), timeout=240)
    assert status["state"] == "done", status
    assert "running" in seen or "starting" in seen
    assert {t["state"] for t in status["trials"]} == {"done"} and status["leader"] in {"t01", "t02"}
    assert (folder / "policy.npz").exists() and (folder / "train.log").exists()
    listed = store.list_policies(None)
    assert [p["name"] for p in listed] == ["proc"] and listed[0]["runnable"]
    json.dumps(status)
    with pytest.raises(TaskError, match="already exists"):
        start_run(spec, folder)


def test_stopping_the_training_process_keeps_the_best_policy_so_far(tmp_path, monkeypatch, balance_env):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    folder = tmp_path / "policies" / "long"
    spec = make_run_spec("long", balance_env, algos=["ars"], steps=5_000_000, parallel=1)
    start_run(spec, folder)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:  # wait until a trial has really begun
        trials = JOBS.status(folder).get("trials") or []
        if trials and trials[0]["steps"] > 0:
            break
        time.sleep(0.3)
    else:
        pytest.fail("the trial never started")
    status = JOBS.stop(folder, wait=90)
    status = JOBS.wait(folder, timeout=90)
    assert status["state"] == "stopped", status
    assert (folder / "policy.npz").exists() and status["trials"][0]["state"] == "stopped"
    assert not (folder / store.STOP_FILE).exists()
