"""Real MuJoCo restore and adversarial storage checks for the common record core."""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

from robotforge.experiments import ExperimentStore, artifact_path, digest, load_record, restore_environment
from robotforge.experiments.schema import ArtifactRef, ExperimentRecord


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("robotforge.experiments.store.provenance", lambda: {"test_source": "same"})
    return ExperimentStore(tmp_path / "experiments")


def test_unique_identity_stable_semantic_fingerprint(store):
    a = store.begin("comparison", seeds={"reset": 7}, budget={"steps": 20}, environment={"path": "/old"})
    b = store.begin("comparison", seeds={"reset": 7}, budget={"steps": 20}, environment={"path": "/new"})
    assert a.record.id != b.record.id
    assert a.record.fingerprint == b.record.fingerprint
    fingerprint = a.record.fingerprint
    a.update(metrics={"score": np.float32(1.5), "trace": np.array([1, 2])}, budget_used={"wall_seconds": 5})
    a.finish("completed", reason="steps_exhausted")
    assert a.record.fingerprint == fingerprint
    assert load_record(a.path).metrics == {"score": 1.5, "trace": [1, 2]}
    c = store.begin("comparison", seeds={"reset": 8}, budget={"steps": 20})
    assert c.record.fingerprint != fingerprint
    with pytest.raises(ValueError, match="already finished"):
        a.update(metrics={"score": 0})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float32(-np.inf), {1: "bad"}, object()])
def test_reject_non_json_and_nonfinite(store, value):
    with pytest.raises(ValueError):
        store.begin("training", algorithm={"value": value})


def test_failed_update_is_atomic(store):
    handle = store.begin("training")
    before = handle.path.read_bytes()
    with pytest.raises(ValueError):
        handle.update(metrics={"loss": float("nan")})
    assert handle.path.read_bytes() == before
    assert handle.record.metrics == {}
    with pytest.raises(ValueError):
        handle.finish("banana")


@pytest.mark.parametrize("path", ["../a", "/a", "C:/a", "C:a", "a\\b", "a/../b", "a//b", "a/./b", "//host/a"])
def test_artifact_paths_cross_platform(path):
    with pytest.raises(ValueError):
        ArtifactRef(path=path, role="weights", sha256="a" * 64, size=0, format="npz")


def test_artifact_retention_dedup_and_corruption(store, tmp_path):
    a, b = store.begin("training"), store.begin("training")
    source = tmp_path / "policy.npz"
    source.write_bytes(b"first checkpoint")
    first = a.artifact(source, "weights")
    repeated = a.artifact(source, "weights")
    other = b.artifact(source, "weights")
    assert first == repeated
    assert len(a.record.artifacts) == 1
    assert (a.folder / first.path).stat().st_ino == (b.folder / other.path).stat().st_ino
    source.write_bytes(b"replacement policy")
    second = a.artifact(source, "weights")
    assert first.sha256 != second.sha256
    assert artifact_path(a.path, first).read_bytes() == b"first checkpoint"
    assert artifact_path(a.path, second).read_bytes() == b"replacement policy"
    (a.folder / first.path).write_bytes(b"tampering")
    with pytest.raises(ValueError, match="integrity"):
        artifact_path(a.path, first)
    with pytest.raises(ValueError, match="corrupt"):
        source.write_bytes(b"first checkpoint")
        a.artifact(source, "weights")


def test_symlink_escape_and_ids(store, tmp_path):
    handle = store.begin("comparison")
    outside = tmp_path / "outside"
    outside.mkdir()
    (handle.folder / "artifacts").symlink_to(outside, target_is_directory=True)
    source = tmp_path / "data.bin"
    source.write_bytes(b"data")
    with pytest.raises(ValueError, match="symlink"):
        handle.artifact(source, "test")
    for value in ("../escape", "C:/path", "nonuuid"):
        with pytest.raises(ValueError, match="UUID"):
            store.open(value)
    assert not list(outside.iterdir())


def test_concurrent_creation_and_resume(store):
    def create(index):
        handle = store.begin("training", seeds={"training": index})
        handle._artifact_bytes(b"shared policy", "weights", "npz")
        return handle.reference()

    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(create, range(12)))
    assert len({ref["id"] for ref in refs}) == 12
    assert len(list((store.root / ".objects").iterdir())) == 1
    for ref in refs:
        resumed = store.resume(ref)
        assert resumed.reference() == ref
        assert load_record(ref["path"]).id == ref["id"]
    assert not list(store.root.rglob("*.tmp"))


def _physics():
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><option timestep="0.003"/>
      <worldbody><body name="moving" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
      <body mocap="true" name="target" pos="1 0 1"><geom type="sphere" size=".03" contype="0" conaffinity="0"/></body>
      </worldbody><actuator><motor joint="unused"/></actuator></mujoco>""".replace(
            "<freejoint/>", '<joint name="slide" type="slide" axis="1 0 0"/>'
        ).replace('joint="unused"', 'joint="slide"')
    )
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.7
    data.qfrc_applied[:] = 0.12
    data.xfrc_applied[1, :3] = [0.2, 0.3, 0.4]
    data.mocap_pos[0] = [2, 3, 4]
    for _ in range(8):
        mujoco.mj_step(model, data)
    return model, data


def test_full_integration_roundtrip_real_mujoco(store):
    model, data = _physics()
    handle = store.begin("comparison", seeds={"reset": 17})
    handle.snapshot_environment(model, data, world={"name": "test"}, scope="live_world", robot="moving")
    restored_model, restored_data = restore_environment(handle.path)
    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
    original = np.empty(mujoco.mj_stateSize(model, state_kind))
    restored = np.empty_like(original)
    mujoco.mj_getState(model, data, original, state_kind)
    mujoco.mj_getState(restored_model, restored_data, restored, state_kind)
    np.testing.assert_array_equal(original, restored)
    np.testing.assert_array_equal(data.ctrl, restored_data.ctrl)
    np.testing.assert_array_equal(data.qfrc_applied, restored_data.qfrc_applied)
    np.testing.assert_array_equal(data.xfrc_applied, restored_data.xfrc_applied)
    np.testing.assert_array_equal(data.mocap_pos, restored_data.mocap_pos)
    for _ in range(25):
        mujoco.mj_step(model, data)
        mujoco.mj_step(restored_model, restored_data)
    np.testing.assert_array_equal(data.qpos, restored_data.qpos)
    np.testing.assert_array_equal(data.qvel, restored_data.qvel)


def test_restore_rejects_wrong_version_and_missing_artifact(store):
    model, data = _physics()
    handle = store.begin("comparison")
    info = handle.snapshot_environment(model, data)
    handle.update(environment={"snapshot": {**info, "mujoco_version": "0.0"}})
    with pytest.raises(ValueError, match="requires MuJoCo"):
        restore_environment(handle.path)
    handle.update(environment={"snapshot": info})
    artifact_path(handle.path, info["state"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        restore_environment(handle.path)


def test_load_rejects_fingerprint_tampering_and_version(store):
    handle = store.begin("comparison")
    record = handle.record.model_dump()
    record["seeds"] = {"reset": 42}
    import json

    handle.path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_record(handle.path)
    record["schema_version"] = 2
    with pytest.raises(ValueError):
        ExperimentRecord.model_validate(record)
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_requested_wall_budget_and_lineage_identity(store):
    a = store.begin(
        "training",
        budget={"wall_seconds": 5},
        origins=[{"id": "00000000-0000-0000-0000-000000000001", "path": "/first", "kind": "training"}],
    )
    b = store.begin(
        "training",
        budget={"wall_seconds": 6},
        origins=[{"id": "00000000-0000-0000-0000-000000000001", "path": "/first", "kind": "training"}],
    )
    c = store.begin(
        "training",
        budget={"wall_seconds": 5},
        origins=[{"id": "00000000-0000-0000-0000-000000000002", "path": "/second", "kind": "training"}],
    )
    assert a.record.fingerprint != b.record.fingerprint
    assert a.record.fingerprint == c.record.fingerprint


def test_provenance_does_not_execute_factories_and_tracks_helpers(tmp_path):
    import importlib
    import sys
    from types import SimpleNamespace

    from robotforge.extend.registry import Item, Loaded, isolated

    module = importlib.import_module("robotforge.experiments.provenance")
    main = tmp_path / "plugin.py"
    helper = tmp_path / "helper.py"
    main.write_text("plugin source", encoding="utf-8")
    helper.write_text("helper version one", encoding="utf-8")
    helper_name = "experiment_test_helper"
    sys.modules[helper_name] = SimpleNamespace(__file__=str(helper))
    calls = []
    try:
        with isolated() as registry:
            item = Item("algorithm", "record_probe", "test", lambda: calls.append(True), path=main)
            registry.items[(item.kind, item.name)] = item
            stat = main.stat()
            registry.files[main] = Loaded(
                main, main.name, "world", (stat.st_mtime_ns, stat.st_size), imported=[helper_name]
            )
            before = module.provenance()["plugins"]
            helper.write_text("helper version two", encoding="utf-8")
            after = module.provenance()["plugins"]
            assert before != after
            assert not calls
            assert after[0]["loaded_file_signature_matches"]
    finally:
        sys.modules.pop(helper_name, None)


@pytest.mark.parametrize("section", ["task", "algorithm"])
@pytest.mark.parametrize("key", ["id", "path", "output", "elapsed", "pid", "world_dir", "timestamp"])
def test_parameter_names_are_never_removed_from_fingerprint(store, section, key):
    a = store.begin("comparison", **{section: {"params": {key: "first"}}})
    b = store.begin("comparison", **{section: {"params": {key: "second"}}})
    assert a.record.fingerprint != b.record.fingerprint
    assert load_record(a.path).fingerprint == a.record.fingerprint


def test_distinct_path_parameters_produce_distinct_real_trajectories_and_fingerprints(store):
    results = []
    for path in ([0.1, 0.2], [0.9, -0.5]):
        model, data = _physics()
        handle = store.begin("comparison", task={"params": {"path": path}})
        for control in handle.record.task["params"]["path"]:
            data.ctrl[:] = control
            for _ in range(10):
                mujoco.mj_step(model, data)
        results.append((handle.record.fingerprint, data.qpos.copy()))
    assert results[0][0] != results[1][0]
    assert not np.array_equal(results[0][1], results[1][1])


def test_reference_shaped_parameters_stay_semantic_but_artifact_metadata_paths_do_not(store):
    first = {"id": "00000000-0000-0000-0000-000000000001", "path": "/first", "kind": "training"}
    second = {"id": "00000000-0000-0000-0000-000000000002", "path": "/second", "kind": "training"}
    a = store.begin("training", task={"params": {"reference": first}})
    b = store.begin("training", task={"params": {"reference": second}})
    assert a.record.fingerprint != b.record.fingerprint
    ref = {"path": "artifacts/one.npy", "role": "weights", "sha256": "a" * 64, "size": 1, "format": "npy"}
    c = store.begin("training", origins=[ref], environment={"env_spec": {"world_dir": "/old", "params": {"path": 1}}})
    d = store.begin(
        "training",
        origins=[{**ref, "path": "artifacts/two.npy"}],
        environment={"env_spec": {"world_dir": "/new", "params": {"path": 1}}},
    )
    assert c.record.fingerprint == d.record.fingerprint
    d.update(origins=[{**ref, "sha256": "b" * 64}])
    assert c.record.fingerprint != d.record.fingerprint


def test_unrecognized_origin_fields_and_corrupted_parameter_fingerprint_remain_visible(store):
    import json

    a = store.begin("training", origins=[{"id": "custom", "path": "first"}], task={"params": {"path": [1, 2]}})
    b = store.begin("training", origins=[{"id": "custom", "path": "second"}], task={"params": {"path": [1, 2]}})
    assert a.record.fingerprint != b.record.fingerprint
    record = a.record.model_dump()
    record["task"]["params"]["path"] = [3, 4]
    a.path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_record(a.path)
