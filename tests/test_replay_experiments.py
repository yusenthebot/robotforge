"""Live records preserve controller identity and close before state is replaced."""

from functools import partial
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robotforge.control.ars import LinearPolicy
from robotforge.control.runtime import start_control
from robotforge.control.tasks import make_task
from robotforge.experiments import ExperimentHandle, load_record, restore_environment
from robotforge.experiments.live import _values, attach_record, history, ledger
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("replay")
    pg.add_robot("arm", template="arm")
    return pg


@pytest.mark.parametrize(
    "end,reason",
    [
        ("stop", "stopped"),
        ("reset", "reset"),
        ("swap", "model_changed"),
        ("replace", "replaced"),
        ("shutdown", "shutdown"),
    ],
)
def test_attachment_closes_before_state_change_and_keeps_original_controller(pg, end, reason):
    out = start_control(pg, "arm", "reach", "ik")
    ctl = pg.session.controllers["arm"]
    assert pg.session.controllers["arm"] is ctl
    assert ledger(pg.session).active[id(ctl)].controller is ctl
    record = load_record(out["experiment"]["path"])
    assert record.status == "running" and record.budget["requested"]["sim_seconds"] is None
    assert record.task["registered_name"] == "reach"
    pg.session.step(25)
    final = pg.session.data.qpos.copy()
    if end == "stop":
        pg.session.set_controller("arm", None)
    elif end == "reset":
        pg.session.reset()
    elif end == "swap":
        pg.session.load_model(mujoco.MjModel.from_xml_string("<mujoco/>"), "test")
    elif end == "replace":
        start_control(pg, "arm", "reach", "pd")
    else:
        pg.session.finish_controller_observations()
    record = load_record(out["experiment"]["path"])
    assert record.status == "stopped" and record.termination["reason"] == reason
    assert record.budget_used["physics_steps"] == 25
    from robotforge.experiments import artifact_path

    path = artifact_path(out["experiment"]["path"], next(a for a in record.artifacts if a.role == "final_physics"))
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["qpos"], final)


def test_policy_and_callback_instances_remain_valid_metadata():
    policy = LinearPolicy.zeros(4, 1)
    assert _values(policy)["runtime_state"] == "not_captured"
    assert _values(partial(float, 1))["callable"].endswith("partial")


def test_snapshot_is_actual_post_release_state(pg):
    class Old:
        def release(self, model, data):
            data.xfrc_applied[:] = 0
            model.dof_damping[:] = 0.123

    pg.session.controllers["arm"] = Old()
    pg.session.data.xfrc_applied[:] = 10
    out = start_control(pg, "arm", "reach", "ik")
    model, data = restore_environment(out["experiment"]["path"])
    np.testing.assert_array_equal(data.xfrc_applied, pg.session.data.xfrc_applied)
    np.testing.assert_array_equal(model.dof_damping, pg.session.model.dof_damping)


def test_unobserved_plugin_tallies_are_unavailable_instead_of_false(pg):
    from robotforge.control.tasks.reach import Reach

    class CountingReach(Reach):
        observed_steps = 0

        def observe(self, data):
            self.observed_steps += 1

        def success(self, data):
            return self.observed_steps >= 5

    task = CountingReach(pg.session.model, "arm")
    ctl = SimpleNamespace(rate_hz=50, update=lambda model, data: None, status=lambda: {})
    out = attach_record(pg.session, "arm", ctl, task, "hold", world_dir=pg.store.dir)
    pg.session.step(10)
    pg.session.set_controller("arm", None)
    result = load_record(out["path"])
    assert task.observed_steps == 0
    assert result.metrics["success_at_end"] is None and result.metrics["task"] is None
    assert result.metrics["task_metric_coverage"].startswith("unavailable")


def test_pre_attachment_failure_preserves_existing_controller(pg, monkeypatch):
    start_control(pg, "arm", "reach", "ik")
    old = pg.session.controllers["arm"]
    monkeypatch.setattr(
        ExperimentHandle, "snapshot_environment", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError, match="disk"):
        start_control(pg, "arm", "reach", "pd")
    assert pg.session.controllers["arm"] is old


def test_zero_step_history_is_bounded_and_separate_executions_have_distinct_ids(pg):
    first = start_control(pg, "arm", "reach", "ik")["experiment"]
    second = start_control(pg, "arm", "reach", "ik")["experiment"]
    assert first["id"] != second["id"]
    a, b = load_record(first["path"]), load_record(second["path"])
    assert a.budget_used["physics_steps"] == 0
    assert a.fingerprint == b.fingerprint
    assert len(history(pg.session, pg.store.dir, limit=1)["experiments"]) == 1
    with pytest.raises(ValueError, match="1 to 100"):
        history(pg.session, pg.store.dir, 101)


def test_closed_live_record_reproduces_on_private_model(pg):
    from robotforge.experiments.playback import reproduce_replay

    out = start_control(pg, "arm", "reach", "pd", {"target": [0.2, 0.1, 0.24]})
    pg.session.step(75)
    pg.session.set_controller("arm", None)
    before = pg.session.data.qpos.copy()
    result = reproduce_replay(out["experiment"]["path"])
    assert result["matches_recorded_final_state"]
    assert result["physics_steps"] == 75 and result["sim_seconds"] > 0
    np.testing.assert_array_equal(pg.session.data.qpos, before)


@pytest.mark.parametrize("recorded_lineage", [False, True])
def test_playback_archives_actual_requested_trial_not_parent_winner(pg, tmp_path, recorded_lineage):
    from robotforge.control.rl import store
    from robotforge.control.rl.runtime import TrainedPolicy
    from robotforge.experiments import artifact_path
    from robotforge.experiments.playback import reproduce_replay
    from robotforge.training.live import _policy_to_play
    from robotforge.training.play import attach
    from robotforge.training.policy import Shape

    task = make_task("reach", pg.session.model, "arm")
    folder = tmp_path / "trial_policy"
    folder.mkdir()
    shape = Shape(len(task.observation_labels()), len(task.actuators))
    for name, scale in (("t01", 0.0), ("t02", 0.1)):
        target = folder / "trials" / name / "policy.npz"
        target.parent.mkdir(parents=True)
        shape.policy(shape.zeros() + scale, np.zeros(shape.obs_dim), np.ones(shape.obs_dim)).save(target)
    meta = {
        "task": "reach",
        "robot": "arm",
        "env": {
            "task": "reach",
            "robot": "arm",
            "world": pg.current_world().model_dump(mode="json"),
            "observation": "task",
            "task_params": {"target": [0.2, 0.1, 0.24]},
        },
        "final": {"trial": "t01"},
    }
    if recorded_lineage:
        from robotforge.experiments import ExperimentStore

        records = ExperimentStore(folder.parent.parent / "experiments")
        parent = records.begin("training_run")
        children = {}
        for name in ("t01", "t02"):
            child = records.begin("training_segment", parent=parent.reference())
            child.artifact(folder / "trials" / name / "policy.npz", "policy")
            child.finish("completed")
            children[name] = child.reference()
        parent.finish("completed")
        meta.update(experiment=parent.reference(), policy_experiment=children["t01"])
        store.write_json(folder / "status.json", {"trials": [{"id": "t02", "experiments": [children["t02"]]}]})
    store.write_json(folder / "policy.json", meta)
    policy, trial = _policy_to_play(folder, "t02")
    assert isinstance(policy, TrainedPolicy) and trial == "t02"
    out = attach(pg.session, "arm", policy, world_dir=pg.store.dir, world=pg.current_world())
    pg.session.step(30)
    pg.session.set_controller("arm", None)
    record = load_record(out["experiment"]["path"])
    assert record.algorithm["trial"] == "t02"
    assert record.algorithm["training_origin"] == ("reported_by_metadata" if recorded_lineage else "unknown_legacy")
    if recorded_lineage:
        assert record.origins == [parent.reference(), children["t02"]]
        assert children["t01"] not in record.origins
    weights = artifact_path(out["experiment"]["path"], next(a for a in record.artifacts if a.role == "policy_weights"))
    assert weights.read_bytes() == (folder / "trials/t02/policy.npz").read_bytes()
    (folder / "trials/t02/policy.npz").unlink()
    assert reproduce_replay(out["experiment"]["path"])["matches_recorded_final_state"]


@pytest.mark.parametrize("finish_fails", [False, True])
def test_post_release_record_failure_returns_attachment_for_cancellation_ownership(pg, monkeypatch, finish_fails):
    class Old:
        def release(self, model, data):
            data.xfrc_applied[:] = 0

    pg.session.controllers["arm"] = Old()
    original = ExperimentHandle.snapshot_environment
    calls = 0

    def snapshot(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-release disk failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ExperimentHandle, "snapshot_environment", snapshot)
    if finish_fails:

        def fail_finish(*args, **kwargs):
            raise OSError("disk still full")

        monkeypatch.setattr(ExperimentHandle, "finish", fail_finish)
    out = start_control(pg, "arm", "reach", "ik")
    assert pg.session.controllers["arm"] is not None
    assert "recording_error" in out["experiment"]
    assert load_record(out["experiment"]["path"]).status == ("running" if finish_fails else "failed")


def test_hidden_physics_controller_is_an_explicit_replay_dependency(pg):
    from robotforge.experiments.playback import reproduce_replay

    class Wind:
        rate_hz = 500

        def update(self, model, data):
            data.qfrc_applied[:] = 0.1

    pg.session.set_controller("__air__arm", Wind())
    out = start_control(pg, "arm", "reach", "pd")
    pg.session.step(3)
    pg.session.set_controller("arm", None)
    record = load_record(out["experiment"]["path"])
    assert record.reproducibility["other_controllers"] == ["__air__arm"]
    with pytest.raises(ValueError, match="other controller"):
        reproduce_replay(out["experiment"]["path"])


def test_replay_refuses_current_global_callback_before_forwarding_physics(pg):
    from robotforge.experiments.playback import reproduce_replay

    out = start_control(pg, "arm", "reach", "pd")
    pg.session.step(3)
    pg.session.set_controller("arm", None)
    called = []
    mujoco.set_mjcb_control(lambda model, data: called.append(True))
    try:
        with pytest.raises(ValueError, match="Python callback control"):
            reproduce_replay(out["experiment"]["path"])
        assert called == []
    finally:
        mujoco.set_mjcb_control(None)
