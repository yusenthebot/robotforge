"""Real small NumPy jobs retain independent, replayable records across replacement and halving."""

from __future__ import annotations

import time
from concurrent.futures import Future

import pytest

from robotforge.control.rl import store
from robotforge.control.rl.jobs import Jobs
from robotforge.experiments import artifact_path, load_record
from robotforge.training import algorithms, runs, trial
from robotforge.training.envs import env_spec
from robotforge.training.mcp_tools import status_extra
from robotforge.training.search import SearchSpec


def tiny_spec(**kwargs):
    env = env_spec("balance", episode_seconds=0.1).model_dump(mode="json")
    return runs.make_run_spec(
        "tiny",
        env,
        algos=["ars"],
        steps=8,
        parallel=0,
        eval_episodes=1,
        held_out_episodes=0,
        params={"ars": {"directions": 1}},
        **kwargs,
    )


def test_real_training_retains_policy_and_lineage_across_replacement(tmp_path):
    folder = tmp_path / "policies" / "tiny"
    first = runs.run_inline(tiny_spec(), folder)
    meta = store.read_json(folder / store.POLICY_FILE)
    parent = load_record(first["experiment"]["path"])
    child = load_record(meta["policy_experiment"]["path"])
    assert parent.status == child.status == "completed"
    assert child.parent == first["experiment"]
    assert parent.origins == [meta["policy_experiment"]]
    assert child.seeds["ranking"]["base"] == 10_000
    assert parent.seeds["held_out"]["base"] == 500_000
    assert parent.algorithm["run_spec"]["env"] == tiny_spec().env
    assert child.budget["training_steps"] == 8
    assert child.budget_used["training_steps"] == first["timesteps"] >= 8
    assert child.environment["env_spec"] == tiny_spec().env
    roles = {a.role for a in child.artifacts}
    assert "policy" in roles and len(roles) >= 4  # compiled model/state + training output
    artifact = next(a for a in parent.artifacts if a.role == "winning_policy")
    saved = artifact_path(first["experiment"]["path"], artifact)
    before = saved.read_bytes()
    second = runs.run_inline(tiny_spec(seed=42), folder)
    assert second["experiment"]["id"] != first["experiment"]["id"]
    assert saved.read_bytes() == before
    assert load_record(first["experiment"]["path"]).status == "completed"
    view = status_extra(first)
    assert view["experiment"] == first["experiment"]
    assert view["trials"][0]["experiments"] == [meta["policy_experiment"]]
    assert store.list_policies(tmp_path)[0]["name"] == "tiny"


def test_real_halving_records_each_rung_without_overwriting_checkpoint(tmp_path):
    spec = tiny_spec(search=SearchSpec(method="halving", trials=3, eta=3, all_settings=False))
    doc = runs.run_inline(spec, tmp_path / "policies" / "tiny")
    refs = [ref for t in doc["trials"] for ref in t["experiments"]]
    assert len(refs) == len({ref["id"] for ref in refs}) == 4
    survivor = next(t for t in doc["trials"] if len(t["experiments"]) == 2)
    first, second = [load_record(ref["path"]) for ref in survivor["experiments"]]
    assert second.origins == [survivor["experiments"][0]]
    assert first.algorithm["rung"] == 0 and second.algorithm["rung"] == 1
    assert second.budget_used["cumulative_training_steps"] == survivor["steps"]
    assert second.budget_used["training_steps"] + first.budget_used["training_steps"] == survivor["steps"]
    assert all(load_record(ref["path"]).parent == doc["experiment"] for ref in refs)
    resumed = next(a for a in second.artifacts if a.role == "resume_state")
    checkpoint = next(a for a in first.artifacts if a.role == "training_state")
    assert resumed.sha256 == checkpoint.sha256


def prepared_job(tmp_path):
    job = runs.Job(tiny_spec(), tmp_path)
    job.trials[0].rung = 0
    payload = job.job_for(job.trials[0], 8, False)
    return job, payload


def test_worker_missing_dependency_fails_record_before_environment_allocation(tmp_path, monkeypatch):
    job, payload = prepared_job(tmp_path)

    def missing(algo):
        raise RuntimeError("missing optional dependency")

    monkeypatch.setattr(algorithms, "need", missing)
    monkeypatch.setattr(trial, "build_env", lambda value: pytest.fail("must preflight before environment allocation"))
    with pytest.raises(RuntimeError, match="missing optional"):
        trial.run_segment(payload)
    record = load_record(payload["experiment"]["path"])
    assert record.status == "failed" and "missing optional" in record.termination["error"]
    assert record.budget_used["training_steps"] == 0
    assert load_record(job.experiment.path).status == "running"  # worker never writes its parent


def test_algorithm_exception_closes_environment_and_finalizes_run(tmp_path, monkeypatch):
    original = algorithms.get("ars")

    def fail_after_episode(ctx):
        ctx.rollout(ctx.policy(ctx.new_shape().zeros()))
        raise RuntimeError("broken")

    bad = algorithms.Algorithm("ars", "test", fail_after_episode, original.hyper)
    monkeypatch.setattr(algorithms, "get", lambda name: bad)
    closed = []
    original_build = trial.build_env

    def build(spec):
        env = original_build(spec)
        close = env.close

        def tracked_close():
            closed.append(True)
            close()

        env.close = tracked_close
        return env

    monkeypatch.setattr(trial, "build_env", build)
    with pytest.raises(Exception, match="training produced no policy"):
        runs.run_inline(tiny_spec(), tmp_path)
    doc = store.read_json(tmp_path / store.STATUS_FILE)
    assert load_record(doc["experiment"]["path"]).status == "failed"
    child = load_record(doc["trials"][0]["experiments"][0]["path"])
    assert child.status == "failed"
    assert child.budget_used["training_steps"] == doc["timesteps"] > 0
    assert closed == [True]


def test_dead_process_status_finalizes_parent(tmp_path):
    job, _ = prepared_job(tmp_path)
    status = job.document()
    status["updated"] = time.time() - 1000
    store.write_json(tmp_path / store.STATUS_FILE, status)
    fixed = Jobs().status(tmp_path)
    assert fixed["state"] == "failed"
    assert load_record(job.experiment.path).status == "failed"
    assert "stopped without finishing" in load_record(job.experiment.path).termination["error"]


def test_queued_cancel_is_terminal_and_worker_does_not_run(tmp_path):
    job, payload = prepared_job(tmp_path)
    future = Future()
    job.deadline = time.monotonic() - 1
    job._wait({future: job.trials[0]})
    record = load_record(payload["experiment"]["path"])
    assert future.cancelled() and record.status == "cancelled"
    assert record.metrics["started"] is False and record.budget_used["training_steps"] == 0


def test_expired_segment_preserves_unscored_policy_with_timeout(tmp_path):
    job, payload = prepared_job(tmp_path)
    payload["deadline"] = time.monotonic() - 1
    out = trial.run_segment(payload)
    record = load_record(payload["experiment"]["path"])
    assert out["timed_out"] and out["score"] is None
    assert record.status == "stopped" and record.termination["reason"] == "timeout"
    assert record.budget_used["training_steps"] == 0
    assert "policy" in {artifact.role for artifact in record.artifacts}


def test_legacy_bridge_has_explicit_seed_protocol_and_preserves_old_fields(tmp_path):
    from robotforge.control.rl.spec import make_spec

    spec = make_spec("legacy", "balance", {"name": "pole", "template": "cartpole"}, algo="ppo", timesteps=32)
    record = store.legacy_experiment(tmp_path / "policies" / "legacy", spec).record
    assert record.kind == "training_run"
    assert record.seeds["ranking"]["base"] == 10_000
    assert record.seeds["final_evaluation"]["base"] == 20_000
    assert record.budget["training_steps"] == 32
    assert record.environment["train_spec"] == spec.model_dump(mode="json")


def test_real_detached_coordinator_uses_launched_parent_reference(tmp_path, monkeypatch):
    jobs = Jobs()
    monkeypatch.setattr(runs, "JOBS", jobs)
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    folder = tmp_path / "policies" / "tiny"
    launched = runs.start_run(tiny_spec(), folder)
    process = jobs.procs[folder]
    try:
        process.wait(timeout=30)
        status = jobs.status(folder)
        assert process.returncode == 0, (folder / store.LOG_FILE).read_text(encoding="utf-8")
        assert status["state"] == "done"
        assert status["experiment"] == launched["experiment"]
        parent = load_record(launched["experiment"]["path"])
        child = load_record(status["trials"][0]["experiments"][0]["path"])
        assert parent.status == child.status == "completed"
        assert child.parent == launched["experiment"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_launch_error_finalizes_the_allocated_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "JOBS", Jobs())
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("launch denied")

    monkeypatch.setattr(runs.subprocess, "Popen", fail)
    folder = tmp_path / "policies" / "tiny"
    with pytest.raises(Exception, match="launch denied"):
        runs.start_run(tiny_spec(), folder)
    meta = store.read_json(folder / store.POLICY_FILE)
    record = load_record(meta["experiment"]["path"])
    assert record.status == "failed"
    assert "launch denied" in record.termination["error"]


@pytest.mark.parametrize("later_status", ["failed", "cancelled"])
def test_winner_origin_matches_preserved_weights_after_later_rung_fails(tmp_path, later_status):
    job, first = prepared_job(tmp_path)
    out = trial.run_segment(first)
    current = job.trials[0]
    current.steps, current.score = out["steps"], out["score"]
    current.rung = 1
    later = job.job_for(current, 8, True)
    store.open_experiment(later["experiment"]).finish(later_status, reason="did not save new weights")
    current.state = "failed" if later_status == "failed" else "stopped"
    job.finish("done" if later_status == "failed" else "stopped")
    meta = store.read_json(tmp_path / store.POLICY_FILE)
    assert meta["policy_experiment"] == first["experiment"]
    assert job.experiment.record.origins == [first["experiment"]]
    assert job.experiment.record.metrics["policy_origin_status"] == "verified"


def test_unknown_winner_weights_remove_stale_origin(tmp_path):
    job, payload = prepared_job(tmp_path)
    out = trial.run_segment(payload)
    current = job.trials[0]
    current.steps, current.score = out["steps"], out["score"]
    weights = job.trial_dir(current) / trial.POLICY_FILE
    weights.write_bytes(weights.read_bytes() + b"changed after archive")
    meta = store.read_json(tmp_path / store.POLICY_FILE)
    store.write_json(tmp_path / store.POLICY_FILE, {**meta, "policy_experiment": payload["experiment"]})
    job.finish("done")
    meta = store.read_json(tmp_path / store.POLICY_FILE)
    assert "policy_experiment" not in meta
    assert meta["policy_origin_note"].startswith("unknown:")
    assert job.experiment.record.origins == []
    assert job.experiment.record.metrics["policy_origin_status"] == "unknown"


def test_policy_origin_helper_rejects_foreign_spoofed_or_damaged_evidence(tmp_path):
    folder = tmp_path / "local" / "policies" / "tiny"
    job, payload = prepared_job(folder)
    trial.run_segment(payload)
    weights = job.trial_dir(job.trials[0]) / trial.POLICY_FILE
    reference = payload["experiment"]
    assert store.policy_experiment_for(folder, weights, [reference]) == reference
    foreign = store.experiment_store(tmp_path / "foreign" / "policies" / "tiny").begin("training_segment")
    foreign.artifact(weights, "policy")
    assert store.policy_experiment_for(folder, weights, [foreign.reference()]) is None
    assert store.policy_experiment_for(folder, weights, [{**reference, "path": foreign.reference()["path"]}]) is None
    assert store.policy_experiment_for(folder, weights, [{**reference, "kind": "training_run"}]) is None
    handle = store.open_experiment(reference)
    artifact = next(a for a in handle.record.artifacts if a.role == "policy")
    artifact_path(handle.path, artifact).write_bytes(b"damaged archive")
    assert store.policy_experiment_for(folder, weights, [reference]) is None


def test_submit_failure_finalizes_allocated_segment_and_parent(tmp_path, monkeypatch):
    def fail_submit(self, *args, **kwargs):
        raise RuntimeError("executor failed")

    monkeypatch.setattr(runs.ThreadPoolExecutor, "submit", fail_submit)
    with pytest.raises(RuntimeError, match="executor failed"):
        runs.run_inline(tiny_spec(), tmp_path)
    status = store.read_json(tmp_path / store.STATUS_FILE)
    parent = load_record(status["experiment"]["path"])
    child = load_record(status["trials"][0]["experiments"][0]["path"])
    assert parent.status == child.status == "failed"
    assert child.termination["reason"] == "segment submission failed"


def test_coordinator_exception_closes_queued_segments_after_shutdown(tmp_path, monkeypatch):
    events = []

    class Pool:
        def __init__(self, **kwargs):
            self.calls = 0

        def submit(self, fn, payload):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("second submission failed")
            return Future()

        def shutdown(self, **kwargs):
            events.append("shutdown")

    monkeypatch.setattr(runs, "ThreadPoolExecutor", Pool)
    spec = tiny_spec(search=SearchSpec(method="random", trials=2, all_settings=False))
    with pytest.raises(RuntimeError, match="second submission failed"):
        runs.run_inline(spec, tmp_path)
    status = store.read_json(tmp_path / store.STATUS_FILE)
    children = [load_record(t["experiments"][0]["path"]) for t in status["trials"]]
    assert events == ["shutdown"]
    assert [child.status for child in children] == ["cancelled", "failed"]


def test_resume_checkpoint_hash_changes_segment_specification(tmp_path):
    import numpy as np

    records = []
    for index in range(2):
        job, payload = prepared_job(tmp_path / f"run_{index}" / "policies" / "tiny")
        folder = job.trial_dir(job.trials[0])
        folder.mkdir(parents=True)
        # Valid extra state is retained without changing the algorithm's behavior; it is still a distinct input.
        trial.save_arrays(folder / trial.STATE_FILE, {"identity_probe": np.array(index)})
        payload["resume"] = True
        trial.run_segment(payload)
        records.append(load_record(payload["experiment"]["path"]))
    assert records[0].algorithm["resume_state_sha256"] != records[1].algorithm["resume_state_sha256"]
    assert records[0].fingerprint != records[1].fingerprint
    for record in records:
        artifact = next(a for a in record.artifacts if a.role == "resume_state")
        assert artifact.sha256 == record.algorithm["resume_state_sha256"]


@pytest.mark.parametrize(
    "reference_case", ["foreign", "spoofed_path", "spoofed_kind", "spoofed_record_id", "segment", "malformed"]
)
def test_stale_job_cannot_finalize_foreign_or_spoofed_parent(tmp_path, reference_case):
    folder = tmp_path / "local" / "policies" / "tiny"
    folder.mkdir(parents=True)
    local = store.experiment_store(folder).begin("training_run")
    segment = store.experiment_store(folder).begin("training_segment")
    foreign = store.experiment_store(tmp_path / "foreign" / "policies" / "tiny").begin("training_run")
    references = {
        "foreign": foreign.reference(),
        "spoofed_path": {**local.reference(), "path": foreign.reference()["path"]},
        "spoofed_kind": {**local.reference(), "kind": "training_segment"},
        "segment": segment.reference(),
        "spoofed_record_id": local.reference(),
        "malformed": {"id": "../foreign", "path": foreign.reference()["path"]},
    }
    if reference_case == "spoofed_record_id":
        store.write_json(local.path, {**local.record.model_dump(mode="json"), "id": foreign.record.id})
    store.write_json(folder / store.POLICY_FILE, {"experiment": references[reference_case]})
    store.write_json(folder / store.STATUS_FILE, {"state": "running", "updated": time.time() - 1000})
    assert store.training_experiment(folder) is None
    assert Jobs().status(folder)["state"] == "failed"
    assert all(load_record(handle.path).status == "running" for handle in (local, segment, foreign))
