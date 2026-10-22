"""Runtime boundaries: optional extras, worker initialization and one shared clock budget."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from robotforge.control.rl import store
from robotforge.training import api, mcp_tools, runs, trial


def test_omitted_algo_uses_numpy_fallback_without_changing_explicit_neural_requests(monkeypatch):
    pg = SimpleNamespace(current_world=lambda: None)
    monkeypatch.setattr(api, "rl_installed", lambda: False)
    assert mcp_tools.new_pipeline(pg, "balance", None, None, None, None, None, None)
    for algo in ("ppo", "sac"):
        assert not mcp_tools.new_pipeline(pg, "balance", None, algo, None, None, None, None)
    monkeypatch.setattr(api, "rl_installed", lambda: True)
    assert not mcp_tools.new_pipeline(pg, "balance", None, None, None, None, None, None)


def test_job_initialization_failure_is_recorded_before_a_job_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(runs.os, "nice", lambda value: None, raising=False)
    store.write_json(tmp_path / runs.RUN_FILE, {"invalid": True})
    with pytest.raises(ValidationError):
        runs.run_job(tmp_path)
    status = store.read_json(tmp_path / store.STATUS_FILE)
    assert status["state"] == "failed" and "ValidationError" in status["error"]
    assert status["traceback_tail"]


def test_plugins_are_initialized_before_planning_and_planning_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(runs.os, "nice", lambda value: None, raising=False)
    spec = runs.make_run_spec("bad", {"world_dir": "world"}, algos=["missing"])
    store.write_json(tmp_path / runs.RUN_FILE, spec.model_dump(mode="json"))
    loaded = []
    monkeypatch.setattr(runs, "_load_plugins", lambda job: loaded.append(job))

    def plan(spec):
        assert loaded == [{"world_dir": "world"}]
        raise RuntimeError("planning failed")

    monkeypatch.setattr(runs, "plan_for", plan)
    monkeypatch.setattr(runs.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    with pytest.raises(RuntimeError, match="planning failed"):
        runs.run_job(tmp_path)
    status = store.read_json(tmp_path / store.STATUS_FILE)
    assert status["state"] == "failed" and "planning failed" in status["error"]


def make_job(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(runs.time, "monotonic", lambda: clock.now)
    spec = runs.make_run_spec(
        "clock", {"task": "balance", "robot": "pole"}, algos=["ars", "cem"], minutes=1, parallel=0
    )
    return runs.Job(spec, tmp_path), clock


def test_queued_trials_get_the_same_absolute_deadline(tmp_path, monkeypatch):
    job, clock = make_job(tmp_path, monkeypatch)
    first = job.job_for(job.trials[0], 100, False)
    clock.now += 45
    second = job.job_for(job.trials[1], 100, False)
    assert first["deadline"] == second["deadline"] == 160.0
    clock.now = 161
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(2,)), action_space=SimpleNamespace(shape=(1,)))
    ctx = trial.TrainContext(env, {}, seed=0, budget=100, deadline=first["deadline"])
    assert not ctx.has_budget()
    # A queued worker must not start expensive resets after its run's deadline either.
    assert ctx.rollout(lambda obs: [0.0]).length == 0


def test_wait_cancels_queued_trials_and_signals_active_workers_at_deadline(tmp_path, monkeypatch):
    job, clock = make_job(tmp_path, monkeypatch)
    active, queued = Future(), Future()
    active.set_running_or_notify_cancel()
    seen = []

    def wait(futures, **kwargs):
        seen.append(set(futures))
        clock.now = 161
        if len(seen) == 2:
            assert queued.cancelled()
            assert job.stop_file.exists()
            active.set_result({"steps": 1, "score": None, "seconds": 1, "stopped": True, "timed_out": True})
            return {active}, set()
        return set(), set(futures)

    monkeypatch.setattr(runs, "wait", wait)
    job._wait({active: job.trials[0], queued: job.trials[1]})
    assert len(seen) == 2 and queued not in seen[1]
    doc = job.document("stopped")
    assert doc["timed_out"] and all(t["state"] == "stopped" for t in doc["trials"])
    assert any("time budget" in note for note in doc["notes"])


def test_rollouts_and_final_evaluation_stop_during_an_episode(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(trial.time, "monotonic", lambda: clock.now)

    def step(action):
        clock.now += 1
        return [0.0], 1.0, False, False, {}

    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(1,)),
        action_space=SimpleNamespace(shape=(1,)),
        reset=lambda **kwargs: ([0.0], {}),
        step=step,
        push=2.0,
    )
    ctx = trial.TrainContext(env, {}, seed=0, budget=100, deadline=3.0)
    episode = ctx.rollout(lambda obs: [0.0])
    assert episode.length == 3 and not ctx.has_budget()
    clock.now = 0.0
    with pytest.raises(trial.Stopped):
        trial.evaluate_policy(env, lambda obs: [0.0], 5, should_stop=lambda: clock.now >= 2.0)
    assert clock.now == 2.0 and env.push == 2.0


def test_zero_seconds_is_an_exhausted_budget(monkeypatch):
    monkeypatch.setattr(trial.time, "monotonic", lambda: 100.0)
    env = SimpleNamespace(observation_space=SimpleNamespace(shape=(1,)), action_space=SimpleNamespace(shape=(1,)))
    assert not trial.TrainContext(env, {}, seed=0, budget=100, max_seconds=0).has_budget()


def test_timeout_preserves_an_unscored_policy_without_inventing_a_score(tmp_path, monkeypatch):
    job, clock = make_job(tmp_path, monkeypatch)
    t = job.trials[0]
    source = job.trial_dir(t)
    source.mkdir(parents=True)
    (source / trial.POLICY_FILE).write_bytes(b"saved policy")
    t.steps, t.state = 9, "stopped"
    clock.now = 161
    assert job.should_stop()
    final = job.finish("stopped")
    assert final["trial"] == t.plan.id and "unmeasured" in final["score_note"]
    assert "mean_return" not in final and "held_out" not in final
    assert (tmp_path / trial.POLICY_FILE).read_bytes() == b"saved policy"


def test_final_held_out_evaluation_uses_the_shared_deadline(tmp_path, monkeypatch):
    job, clock = make_job(tmp_path, monkeypatch)
    closed = []
    monkeypatch.setattr(runs, "build_env", lambda spec: SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(runs.MlpPolicy, "load", lambda path: object())

    def evaluate(*args, should_stop, **kwargs):
        assert not should_stop()
        clock.now = 160
        assert should_stop()
        raise trial.Stopped("deadline")

    monkeypatch.setattr(runs, "evaluate_policy", evaluate)
    assert job._held_out("done") is None
    assert closed and job.timed_out and job.stop_seen


def test_trusted_world_algorithm_survives_job_and_worker_processes(tmp_path, monkeypatch):
    from robotforge.extend import REGISTRY, registry
    from robotforge.training.envs import env_spec

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    world = tmp_path / "world"
    plugins = world / "plugins"
    plugins.mkdir(parents=True)
    (plugins / "zero.py").write_text(
        "from robotforge.extend import algorithm\n"
        "@algorithm('world_zero')\n"
        "def zero(ctx):\n"
        "    theta = ctx.new_shape().zeros()\n"
        "    ctx.rollout(ctx.policy(theta), seed=0)\n"
        "    (ctx.folder / 'plugin_ran').write_text('worker', encoding='utf-8')\n"
        "    return theta\n",
        encoding="utf-8",
    )
    folder = tmp_path / "run"
    folder.mkdir()
    with registry.isolated():
        REGISTRY.trust(world)  # only a temporary test plugin, never a user's world
        env = env_spec("balance", episode_seconds=0.05).model_dump(mode="json")
        env["world_dir"] = str(world)
        spec = runs.make_run_spec(
            "plugin", env, algos=["world_zero"], steps=2, parallel=1, eval_episodes=1, held_out_episodes=0
        )
        store.write_json(folder / runs.RUN_FILE, spec.model_dump(mode="json"))
        store.write_json(folder / store.POLICY_FILE, runs.policy_meta(spec))
    # Both the job process (planning) and its spawned worker must load the trusted plugin themselves.
    result = subprocess.run(
        [sys.executable, "-m", "robotforge.training", "job", str(folder)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    status = store.read_json(folder / store.STATUS_FILE)
    assert status["state"] == "done" and status["final"]["algo"] == "world_zero"
    assert status["final"]["episodes"] == 1
    assert (folder / "trials" / "t01" / "plugin_ran").read_text(encoding="utf-8") == "worker"
    assert (folder / trial.POLICY_FILE).exists()
