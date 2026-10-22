"""Real comparison snapshots, exact physics reproduction and bounded lifecycle records."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.control.compare import compare, compare_methods, reproduce_comparison
from robotforge.control.env import task_world
from robotforge.control.tasks import make_task
from robotforge.experiments import load_record, restore_environment
from robotforge.extend import registry


@pytest.fixture(autouse=True)
def local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(tmp_path / "plugins"))
    with registry.isolated():
        yield


@pytest.fixture
def anyio_backend():
    return "asyncio"


def records(comparison):
    parent = load_record(comparison.experiment["path"])
    return parent, [load_record(child["path"]) for child in parent.metrics["children"]]


def physics(result):
    value = result.to_dict()
    for key in ("wall_seconds", "wall_per_sim"):
        value.pop(key)
    return value


def final_state(path):
    from robotforge.experiments.store import verified_artifact

    record = load_record(path)
    ref = next(a for a in record.artifacts if a.role == "final_state")
    with np.load(verified_artifact(Path(path).parent, ref), allow_pickle=False) as values:
        return values["state"].copy()


def test_default_balance_reproduces_bit_exactly_with_push_and_all_method_records(tmp_path):
    normal = compare_methods("balance", ["lqr", "pd", "none"], 0.2, push_at=0.1, push_strength=1.5)
    saved = compare_methods(
        "balance", ["lqr", "pd", "none"], 0.2, push_at=0.1, push_strength=1.5, record_dir=tmp_path / "saved"
    )
    assert "experiment" not in normal.to_dict()
    assert set(saved.to_dict()) == set(normal.to_dict()) | {"experiment"}
    assert [physics(r) for r in normal.results] == [physics(r) for r in saved.results]
    parent, children = records(saved)
    assert parent.status == "completed"
    assert parent.task["registered_name"] == "balance"
    assert parent.seeds == {"reset": 0}
    assert parent.budget["requested"]["sim_seconds_per_method"] == 0.2
    assert parent.algorithm["push"] == {"at_sim_seconds": 0.1, "strength": 1.5}
    assert [child.algorithm["name"] for child in children] == ["lqr", "pd", "none"]
    assert children[0].algorithm["params"]["q_weights"] == [5.0, 50.0, 0.5, 1.0]
    starts = [restore_environment(ref["path"])[1] for ref in parent.metrics["children"]]
    for start in starts[1:]:
        np.testing.assert_array_equal(start.qpos, starts[0].qpos)
    again = reproduce_comparison(saved.experiment["path"], record_dir=tmp_path / "again")
    assert [physics(r) for r in again.results] == [physics(r) for r in saved.results]
    replay_parent, _ = records(again)
    for original, repeated in zip(parent.metrics["children"], replay_parent.metrics["children"], strict=True):
        np.testing.assert_array_equal(final_state(original["path"]), final_state(repeated["path"]))
    child_replay = reproduce_comparison(parent.metrics["children"][1]["path"])
    assert [physics(r) for r in child_replay.results] == [physics(saved.results[1])]


def test_none_seed_retains_custom_live_state_controls_and_external_forces(tmp_path):
    model, robot = task_world("balance")
    data = mujoco.MjData(model)
    data.qpos[:] = [0.13, 0.21]
    data.qvel[:] = [0.2, -0.3]
    data.ctrl[:] = 0.4
    data.qfrc_applied[:] = [0.01, 0.02]
    data.xfrc_applied[1, 0] = 0.03
    data.time = 4.25
    mujoco.mj_forward(model, data)
    original = copy.copy(data)
    saved = compare_methods(
        "balance", ["pd", "none"], 0.1, model=model, robot=robot, data=data, seed=None, record_dir=tmp_path
    )
    parent, _ = records(saved)
    assert parent.seeds == {"reset": None}
    _, restored = restore_environment(saved.experiment["path"])
    for field in ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied"):
        np.testing.assert_array_equal(getattr(restored, field), getattr(original, field))
        np.testing.assert_array_equal(getattr(data, field), getattr(original, field))
    assert data.time == restored.time == original.time
    again = reproduce_comparison(saved.experiment["path"])
    assert [physics(r) for r in saved.results] == [physics(r) for r in again.results]


def test_random_goal_and_method_child_reproduce_without_second_reset(tmp_path):
    saved = compare_methods("reach", ["ik"], 0.08, seed=73, record_dir=tmp_path)
    parent, children = records(saved)
    assert len(children[0].task["resolved_goal"]["target"]) == 3
    replay = reproduce_comparison(parent.metrics["children"][0]["path"])
    assert physics(saved.results[0]) == physics(replay.results[0])


def test_comparison_runs_in_a_fresh_process(tmp_path):
    saved = compare_methods("balance", ["pd"], 0.08, record_dir=tmp_path)
    code = (
        "import json,sys; from robotforge.control.compare import reproduce_comparison; "
        "r=reproduce_comparison(sys.argv[1]).results[0]; "
        "print(json.dumps({'reward':r.reward,'cost':r.cost,'report':r.report}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, saved.experiment["path"]],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    )
    out = json.loads(result.stdout)
    assert out == {"reward": saved.results[0].reward, "cost": saved.results[0].cost, "report": saved.results[0].report}


def test_zero_step_timeout_and_skipped_children_are_finite(tmp_path, monkeypatch):
    import robotforge.control.compare as module

    values = iter([0, 0.5, 2, 3])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(values, 4))
    result = compare_methods("balance", ["pd", "none"], 0.2, wall_budget=1, record_dir=tmp_path)
    assert len(result.results) == 1 and result.skipped == ["none"]
    assert math.isinf(result.results[0].wall_per_sim)
    parent, children = records(result)
    assert parent.status == "stopped"
    assert [child.status for child in children] == ["stopped", "stopped"]
    assert children[0].metrics["wall_per_sim"] is None
    assert "wall_per_sim" in children[0].metrics["unavailable_metrics"]
    assert children[1].termination["reason"] == "not run: out of computer time"
    json.dumps(parent.model_dump(), allow_nan=False)


def test_wholly_skipped_comparison_has_a_valid_table(tmp_path):
    result = compare_methods("balance", ["pd", "none"], 0.1, wall_budget=-1, record_dir=tmp_path)
    assert result.results == []
    assert "method" in result.table
    parent, children = records(result)
    assert parent.status == "stopped"
    assert len(children) == 2


def test_controller_error_finalizes_active_parent_and_unstarted_children(tmp_path, monkeypatch):
    import robotforge.control.compare as module

    real = module.make_controller

    def factory(method, *args, **kwargs):
        if method == "pd":
            raise RuntimeError("controller build failed")
        return real(method, *args, **kwargs)

    monkeypatch.setattr(module, "make_controller", factory)
    with pytest.raises(RuntimeError, match="controller build failed"):
        compare_methods("balance", ["none", "pd", "lqr"], 0.02, record_dir=tmp_path)
    entries = [load_record(p) for p in tmp_path.glob("*/experiment.json")]
    parent = next(r for r in entries if r.kind == "comparison")
    assert parent.status == "failed"
    assert "controller build failed" in parent.termination["error"]
    assert [load_record(c["path"]).status for c in parent.metrics["children"]] == ["completed", "failed", "cancelled"]


def test_low_level_factory_is_recorded_with_explicit_reproduction_limit(tmp_path):
    model, robot = task_world("balance")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = compare(lambda: make_task("balance", model, robot), model, data, ["none"], 0.02, record_dir=tmp_path)
    assert len(result) == 1
    path = next(p for p in tmp_path.glob("*/experiment.json") if load_record(p).kind == "comparison")
    with pytest.raises(ValueError, match="Python task factory"):
        reproduce_comparison(path)


def test_immutable_policy_artifacts_preserve_saved_settings_after_original_is_removed(tmp_path):
    from robotforge.control.rl import MlpPolicy, store
    from robotforge.control.rl.spec import describe_env, make_spec

    folder = store.policies_home(None) / "tiny"
    folder.mkdir(parents=True)
    net = MlpPolicy([np.zeros((1, 4))], [np.zeros(1)], "tanh", "clip", np.zeros(4), np.ones(4))
    net.save(folder / "policy.npz")
    spec = make_spec("tiny", "balance", {"name": "pole", "template": "cartpole"})
    facts = describe_env(spec)
    store.write_json(
        folder / "policy.json",
        {
            "name": "tiny",
            "task": "balance",
            "algo": "ppo",
            "action_mode": "direct",
            **facts,
            "task_params": {"start_angle": 0.25, "randomize": 0.0},
        },
    )
    saved = compare_methods("balance", ["none", "policy:tiny"], 0.08, record_dir=tmp_path / "records")
    parent, children = records(saved)
    assert parent.task["params"]["start_angle"] == 0.25
    assert children[0].task["params"] == children[1].task["params"]
    (folder / "policy.npz").unlink()
    (folder / "policy.json").unlink()
    again = reproduce_comparison(saved.experiment["path"])
    assert [physics(r) for r in saved.results] == [physics(r) for r in again.results]
    policy_ref = parent.algorithm["policies"]["policy:tiny"]["policy.npz"]
    (Path(saved.experiment["path"]).parent / policy_ref["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        reproduce_comparison(saved.experiment["path"])


def test_reproduce_refuses_code_mismatch(tmp_path, monkeypatch):
    import robotforge.experiments as experiments

    saved = compare_methods("balance", ["none"], 0.02, record_dir=tmp_path)
    actual = experiments.provenance()
    monkeypatch.setattr(experiments, "provenance", lambda: {**actual, "active_source_sha256": "different"})
    with pytest.raises(ValueError, match="compatibility mismatch"):
        reproduce_comparison(saved.experiment["path"])


@pytest.mark.anyio
async def test_mcp_automatically_records_live_world_comparison(tmp_path):
    from robotforge.mcp_server import build_server
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("comparison")
    playground.add_robot("pole", template="cartpole")
    playground.session.data.qpos[1] = 0.13
    mujoco.mj_forward(playground.session.model, playground.session.data)
    async with Client(build_server(playground, 0)) as client:
        response = await client.call_tool(
            "compare_controllers",
            {
                "robot": "pole",
                "task": "balance",
                "methods": ["pd"],
                "seconds": 0.04,
            },
        )
    assert not response.is_error, response
    value = json.loads(response.content[0].text)
    assert {"task", "results", "table", "best", "explain", "next", "experiment"} == set(value)
    path = Path(value["experiment"]["path"])
    assert path.parent.parent == (playground.store.dir / "experiments").resolve()
    record = load_record(path)
    assert record.seeds["reset"] is None
    _, start = restore_environment(path)
    assert start.qpos[1] == 0.13
    assert playground.session.data.time == 0.0


def test_recording_does_not_add_resets_or_controller_construction(tmp_path, monkeypatch):
    import robotforge.control.compare as module
    from robotforge.control.tasks.balance import Balance

    counts = {"reset": 0, "controller": 0}
    reset, construct = Balance.reset, module.make_controller

    def resetting(self, *args):
        counts["reset"] += 1
        return reset(self, *args)

    def constructing(*args, **kwargs):
        counts["controller"] += 1
        return construct(*args, **kwargs)

    monkeypatch.setattr(Balance, "reset", resetting)
    monkeypatch.setattr(module, "make_controller", constructing)
    compare_methods("balance", ["pd", "none"], 0.02, record_dir=tmp_path)
    assert counts == {"reset": 2, "controller": 2}


def test_cancelled_comparison_finalizes_all_records(tmp_path, monkeypatch):
    import robotforge.control.compare as module

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "make_controller", cancel)
    with pytest.raises(KeyboardInterrupt):
        compare_methods("balance", ["pd", "none"], 0.02, record_dir=tmp_path)
    entries = [load_record(p) for p in tmp_path.glob("*/experiment.json")]
    assert len(entries) == 3
    assert all(entry.status == "cancelled" for entry in entries)


def test_snapshot_hash_tampering_is_refused(tmp_path):
    saved = compare_methods("balance", ["none"], 0.02, record_dir=tmp_path)
    record = load_record(saved.experiment["path"])
    state_ref = next(a for a in record.artifacts if a.role == "integration_state")
    artifact = Path(saved.experiment["path"]).parent / state_ref.path
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        reproduce_comparison(saved.experiment["path"])


def test_file_backed_plugin_identity_reproduces_and_changed_code_is_refused(tmp_path):
    plugin = tmp_path / "plugins" / "balance.py"
    plugin.parent.mkdir()
    plugin.write_text(
        "from robotforge.extend import task\n"
        "from robotforge.control.tasks.balance import Balance\n"
        "@task('recorded_balance', schema=Balance.Params, methods=('pd',))\n"
        "def build(model, robot, params):\n"
        "    return Balance(model, robot, params)\n",
        encoding="utf-8",
    )
    registry.REGISTRY.user_loaded = False
    model, robot = task_world("balance")
    saved = compare_methods("recorded_balance", ["pd"], 0.04, model=model, robot=robot, record_dir=tmp_path / "records")
    record = load_record(saved.experiment["path"])
    assert record.task["registered_name"] == "recorded_balance"
    assert record.task["implementation_name"] == "balance"
    again = reproduce_comparison(saved.experiment["path"])
    assert physics(saved.results[0]) == physics(again.results[0])
    plugin.write_text(plugin.read_text(encoding="utf-8") + "CHANGED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="compatibility mismatch: plugins"):
        reproduce_comparison(saved.experiment["path"])


def test_reproduction_validates_untrusted_policy_name_before_writing(tmp_path):
    from robotforge.control.tasks import TaskError

    saved = compare_methods("balance", ["none"], 0.002, record_dir=tmp_path / "records")
    path = Path(saved.experiment["path"])
    record = load_record(path)
    target = tmp_path / "must_not_be_created"
    method = f"policy:{target}"
    artifact = record.artifacts[0].model_dump()
    record.algorithm["methods"] = [method]
    record.algorithm["policies"] = {method: {"policy.json": artifact, "policy.npz": artifact}}
    record.fingerprint = record.specification_fingerprint()
    path.write_text(record.model_dump_json(), encoding="utf-8")
    with pytest.raises(TaskError, match="policy names"):
        reproduce_comparison(path)
    assert not target.exists()


def test_recorded_policy_uses_archive_when_checkpoint_changes_before_controller_build(tmp_path, monkeypatch):
    import robotforge.control.compare as module
    from robotforge.control.rl import MlpPolicy, store
    from robotforge.control.rl.spec import describe_env, make_spec

    folder = store.policies_home(None) / "changing"
    folder.mkdir(parents=True)
    net = MlpPolicy([np.zeros((1, 4))], [np.zeros(1)], "tanh", "clip", np.zeros(4), np.ones(4))
    net.save(folder / "policy.npz")
    spec = make_spec("changing", "balance", {"name": "pole", "template": "cartpole"})
    metadata = {
        "name": "changing",
        "task": "balance",
        "algo": "ppo",
        "action_mode": "direct",
        **describe_env(spec),
        "task_params": {"start_angle": 0.25, "randomize": 0.0},
    }
    store.write_json(folder / "policy.json", metadata)
    archived_bytes = {name: (folder / name).read_bytes() for name in ("policy.json", "policy.npz")}
    expected = compare_methods("balance", ["policy:changing"], 0.12)
    original_make = module.make_controller
    bundles = []

    def replace_checkpoint_then_build(method, task, model, data, params):
        replacement = MlpPolicy([np.ones((1, 4))], [np.ones(1)], "tanh", "clip", np.zeros(4), np.ones(4))
        replacement.save(folder / "next.npz")
        (folder / "next.npz").replace(folder / "policy.npz")
        store.write_json(folder / "policy.json", {**metadata, "control_dt": 0.05})
        bundle = Path(params["path"])
        assert bundle != folder and bundle.name == "changing"
        for filename, content in archived_bytes.items():
            assert (bundle / filename).read_bytes() == content
        bundles.append(bundle)
        return original_make(method, task, model, data, params)

    monkeypatch.setattr(module, "make_controller", replace_checkpoint_then_build)
    saved = compare_methods("balance", ["policy:changing"], 0.12, record_dir=tmp_path / "records")
    assert physics(saved.results[0]) == physics(expected.results[0])
    assert saved.results[0].controller["policy"] == "changing"
    replay = reproduce_comparison(saved.experiment["path"], record_dir=tmp_path / "replay")
    assert physics(replay.results[0]) == physics(saved.results[0])
    parent, _ = records(saved)
    repeated, _ = records(replay)
    np.testing.assert_array_equal(
        final_state(parent.metrics["children"][0]["path"]),
        final_state(repeated.metrics["children"][0]["path"]),
    )
    assert len(bundles) == 2 and all(not path.exists() for path in bundles)
