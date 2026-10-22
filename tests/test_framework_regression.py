"""Physical repeatability plus adversarial manifest, evidence and lifecycle checks."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.experiments import load_record
from robotforge.experiments import regression as reg


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    root = tmp_path_factory.mktemp("framework-capture")
    manifest = reg.run_suite(root)
    assert reg.compare_manifests(manifest, manifest)["status"] == "pass"
    assert reg.validate_evidence(manifest, root) == []
    return root, manifest


def test_real_suite_is_repeatable_in_fresh_process_and_baseline_is_readonly(captured, tmp_path):
    root, baseline = captured
    original = (root / "manifest.json").read_bytes()
    script = Path(reg.__file__).resolve().parents[3] / "scripts/framework_regression.py"
    candidate = tmp_path / "fresh"
    result = subprocess.run(
        [sys.executable, str(script), "--out", str(candidate), "--baseline", str(root / "manifest.json")],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / "manifest.json").read_bytes() == original
    after = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    assert reg.compare_manifests(baseline, after)["status"] == "pass"
    assert baseline["cases"][0]["evidence"][0]["id"] != after["cases"][0]["evidence"][0]["id"]
    assert all(load_record(path).status != "running" for path in candidate.rglob("experiment.json"))


def test_real_mcp_case_preserves_measurement_and_static_tool_contract(captured):
    root, manifest = captured
    case = next(case for case in manifest["cases"] if case["case_id"] == "mcp_run_controller")
    assert case["outputs"]["physics_steps"] == 40
    assert case["outputs"]["push_step"] == 20
    assert case["outputs"]["tool_count"] == 60
    child = load_record(root / case["evidence"][0]["path"])
    assert child.status == "completed"
    assert child.algorithm["push"]["physics_step"] == 20
    candidate = copy.deepcopy(manifest)
    changed = next(case for case in candidate["cases"] if case["case_id"] == "mcp_run_controller")
    changed["outputs"]["tool_definitions_sha256"] = "f" * 64
    assert reg.compare_manifests(manifest, candidate)["status"] == "fail"


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "missing",
        "new",
        "duplicate",
        "input",
        "metric",
        "missing_metric",
        "nan",
        "none",
        "provenance",
        "path",
        "failed",
        "baseline_failed",
        "bool_number",
    ],
)
def test_genuine_changes_and_invalid_inputs_fail(captured, mutation):
    _, original = captured
    baseline, candidate = copy.deepcopy(original), copy.deepcopy(original)
    case = candidate["cases"][0]
    if mutation == "schema":
        candidate["schema_version"] = 2
    elif mutation == "missing":
        candidate["cases"].pop()
    elif mutation == "new":
        extra = copy.deepcopy(case)
        extra["case_id"] = "new_case"
        candidate["cases"].append(extra)
    elif mutation == "duplicate":
        candidate["cases"].append(copy.deepcopy(case))
    elif mutation == "input":
        case["input_signature"] = "f" * 64
    elif mutation == "metric":
        case["outputs"]["lqr"]["reward"] += 1
    elif mutation == "missing_metric":
        case["outputs"]["lqr"].pop("reward")
    elif mutation == "nan":
        case["outputs"]["lqr"]["reward"] = float("nan")
    elif mutation == "none":
        case["outputs"]["lqr"]["reward"] = None
    elif mutation == "provenance":
        candidate["source"].pop("mujoco")
    elif mutation == "path":
        case["evidence"][0]["path"] = "../../outside.json"
    elif mutation == "failed":
        case["status"] = "fail"
    elif mutation == "baseline_failed":
        baseline["cases"][0]["status"] = "fail"
    elif mutation == "bool_number":
        case["outputs"]["lqr"]["success"] = 1
    assert reg.compare_manifests(baseline, candidate)["status"] == "fail"


@pytest.mark.parametrize("key", reg.RUNTIME_KEYS)
def test_source_changes_do_not_hide_incompatible_runtime(captured, key):
    _, baseline = captured
    candidate = copy.deepcopy(baseline)
    candidate["source"]["active_source_sha256"] = "f" * 64
    candidate["source"][key] = [{"name": "foreign_plugin"}] if key == "plugins" else "changed"
    report = reg.compare_manifests(baseline, candidate)
    assert report["status"] == "unsupported"
    assert report["source_changed"]


def test_source_only_change_and_case_order_are_informational(captured):
    _, baseline = captured
    candidate = copy.deepcopy(baseline)
    candidate["cases"].reverse()
    candidate["source"]["git_commit"] = "b" * 40
    candidate["source"]["active_source_sha256"] = "f" * 64
    report = reg.compare_manifests(baseline, candidate)
    assert report["status"] == "pass" and report["source_changed"]


@pytest.mark.parametrize(
    "damage", ["missing_artifact", "corrupt_artifact", "missing_record", "corrupt_record", "symlink"]
)
def test_candidate_artifacts_and_records_are_verified(captured, tmp_path, damage):
    original, manifest = captured
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    ref = manifest["cases"][0]["evidence"][0]
    record = root / ref["path"]
    artifact = record.parent / ref["artifacts"][0]["path"]
    if damage == "missing_artifact":
        artifact.unlink()
    elif damage == "corrupt_artifact":
        artifact.write_bytes(b"broken")
    elif damage == "missing_record":
        record.unlink()
    elif damage == "corrupt_record":
        record.write_text("{}", encoding="utf-8")
    else:
        artifact.unlink()
        outside = tmp_path / "foreign"
        outside.write_bytes(b"foreign")
        artifact.symlink_to(outside)
    assert reg.validate_evidence(manifest, root)


def test_array_content_identity_ignores_container_bytes_and_checks_values(tmp_path):
    value = np.array([1.0, 2.0])
    before = reg.array_sha256(value)
    np.savez(tmp_path / "a.npz", value=value)
    np.savez_compressed(tmp_path / "b.npz", value=value)
    assert (tmp_path / "a.npz").read_bytes() != (tmp_path / "b.npz").read_bytes()
    assert reg._arrays(tmp_path / "a.npz") == reg._arrays(tmp_path / "b.npz")
    value[0] += 1
    assert reg.array_sha256(value) != before
    with pytest.raises(ValueError, match="nonfinite"):
        reg.array_sha256(np.array([np.nan]))


def test_failure_finalizes_wrapper_and_allocated_child(tmp_path, monkeypatch):
    def fail(root, state, case_id, handle):
        reg.ExperimentStore(root / "orphans").begin("control")
        raise RuntimeError("injected execution failure")

    monkeypatch.setattr(reg, "_suite_cases", fail)
    manifest = reg.run_suite(tmp_path)
    assert all(case["status"] == "fail" for case in manifest["cases"])
    assert {load_record(path).status for path in tmp_path.rglob("experiment.json")} == {"failed", "cancelled"}
    assert reg.compare_manifests(manifest, manifest)["status"] == "fail"


def test_cancellation_is_terminal_and_environment_is_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", "original-home")

    def cancel(root, state, case_id, handle):
        reg.ExperimentStore(root / "orphans").begin("control")
        raise KeyboardInterrupt()

    monkeypatch.setattr(reg, "_suite_cases", cancel)
    with pytest.raises(KeyboardInterrupt):
        reg.run_suite(tmp_path)
    assert os.environ["ROBOTFORGE_HOME"] == "original-home"
    assert all(load_record(path).status == "cancelled" for path in tmp_path.rglob("experiment.json"))


def test_capture_requires_empty_destination_and_explicit_action(captured, tmp_path):
    root, _ = captured
    original = (root / "manifest.json").read_bytes()
    assert reg.main(["--capture", "--out", str(root)]) == 1
    assert (root / "manifest.json").read_bytes() == original
    with pytest.raises(SystemExit):
        reg.main(["--out", str(tmp_path)])


def test_changed_physical_result_makes_cli_exit_nonzero(captured, tmp_path):
    _, original = captured
    baseline = copy.deepcopy(original)
    baseline["cases"][0]["outputs"]["lqr"]["reward"] += 1
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    assert reg.main(["--out", str(tmp_path / "candidate"), "--baseline", str(path)]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == baseline


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1}])
def test_invalid_baseline_never_becomes_implicit_capture(tmp_path, monkeypatch, value):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(reg, "run_suite", lambda *args: pytest.fail("invalid baseline must fail before running"))
    assert reg.main(["--baseline", str(baseline), "--out", str(tmp_path / "candidate")]) == 1


@pytest.mark.parametrize("field", ["schema_version", "suite"])
def test_schema_identifiers_cannot_be_defaulted(captured, field):
    _, original = captured
    baseline = copy.deepcopy(original)
    baseline.pop(field)
    assert reg.compare_manifests(baseline, original)["status"] == "fail"


def test_active_callback_is_unsupported_without_execution(tmp_path, monkeypatch):
    called = []
    mujoco.set_mjcb_control(lambda model, data: called.append(True))
    monkeypatch.setattr(reg, "_suite_cases", lambda *args: pytest.fail("unsupported runtime must not execute"))
    try:
        assert reg.main(["--capture", "--out", str(tmp_path)]) == 2
        assert called == []
        assert all(load_record(path).status == "stopped" for path in tmp_path.rglob("experiment.json"))
    finally:
        mujoco.set_mjcb_control(None)
