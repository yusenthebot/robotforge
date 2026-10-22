"""Small recorded framework acceptance cases and fail-closed golden comparisons.

Baseline JSON is inert evidence: its paths are never executed, restored or imported.
Array identities include dtype, shape and C-order bytes, not NPZ container timestamps.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .provenance import provenance
from .schema import ArtifactRef, canonical, check_id, digest, plain, relative_artifact
from .store import ExperimentHandle, ExperimentStore, artifact_path, load_record, safe_path

SUITE = "framework-regression-v1"
RUNTIME_KEYS = ("python", "platform", "numpy", "mujoco", "plugins")
CASE_INPUTS = {
    "balance_compare": {
        "task": "balance",
        "methods": ["lqr", "pd"],
        "seed": 7,
        "seconds": 0.2,
        "push_at": 0.1,
        "push_strength": 1.5,
    },
    "reach_control": {"template": "arm", "method": "ik", "target": [0.25, 0.05, 0.25], "physics_steps": 25},
    "reach_replay": {"origin_case": "reach_control", "absolute_tolerance": 1e-8},
    "ars_training": {
        "task": "balance",
        "episode_seconds": 0.1,
        "steps": 8,
        "seed": 7,
        "directions": 1,
        "eval_episodes": 1,
        "held_out_episodes": 0,
        "parallel": 0,
    },
    "policy_lineage": {"origin_case": "ars_training", "replacement_seed": 11},
    "snapshot_restore": {
        "task": "balance",
        "qpos": [0.13, 0.21],
        "qvel": [0.2, -0.3],
        "ctrl": 0.4,
        "force": [0.01, 0.02],
        "body_force": 0.03,
        "time": 4.25,
        "steps": 10,
    },
    "mcp_run_controller": {
        "robot": "pole",
        "template": "cartpole",
        "task": "balance",
        "method": "lqr",
        "seconds": 0.08,
        "push_at": 0.04,
        "task_params": {"start_angle": 0.15, "randomize": 0.0},
        "initial_qpos": [0.0, 0.15],
        "tool_schema_fields": ["name", "title", "description", "inputSchema", "outputSchema", "annotations"],
    },
}


def input_signature(case_id: str) -> str:
    return digest({"input_schema": 1, "case_id": case_id, "inputs": CASE_INPUTS[case_id]})


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    path: str
    kind: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: list[ArtifactRef]

    _id = field_validator("id")(check_id)
    _path = field_validator("path")(relative_artifact)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str
    input_signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pass", "fail", "unsupported"]
    outputs: dict[str, Any]
    evidence: list[Evidence] = Field(min_length=1)
    error: str | None = None

    @field_validator("outputs")
    @classmethod
    def finite_outputs(cls, value: dict) -> dict:
        def available(item: Any) -> None:
            if item is None:
                raise ValueError("unavailable metric")
            if isinstance(item, dict):
                for child in item.values():
                    available(child)
            elif isinstance(item, list):
                for child in item:
                    available(child)

        available(value)
        return plain(value)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1] = 1
    suite: Literal["framework-regression-v1"] = SUITE
    source: dict[str, Any]
    cases: list[Case]


def _checked(value: dict) -> Manifest:
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or "suite" not in value:
        raise ValueError("manifest requires explicit schema_version and suite")
    plain(value)
    manifest = Manifest.model_validate(value)
    ids = [case.case_id for case in manifest.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id")
    if set(ids) != set(CASE_INPUTS):
        raise ValueError(
            f"case inventory differs: missing={sorted(set(CASE_INPUTS) - set(ids))}, "
            f"new={sorted(set(ids) - set(CASE_INPUTS))}"
        )
    for key in (*RUNTIME_KEYS, "active_source_sha256", "git_commit", "git_dirty", "package", "package_version"):
        if key not in manifest.source or manifest.source[key] is None:
            raise ValueError(f"missing provenance: {key}")
        value = manifest.source[key]
        if key == "git_dirty":
            valid = isinstance(value, bool)
        elif key == "plugins":
            valid = isinstance(value, list) and all(isinstance(item, dict) for item in value)
        else:
            valid = isinstance(value, str) and bool(value)
        if not valid:
            raise ValueError(f"invalid provenance: {key}")
    for case in manifest.cases:
        if case.input_signature != input_signature(case.case_id):
            raise ValueError(f"input signature differs: {case.case_id}")
        if case.status == "pass" and not case.outputs:
            raise ValueError(f"unavailable metrics: {case.case_id}")
    return manifest


def compare_manifests(baseline: dict, candidate: dict) -> dict:
    """Compare inert documents. Source changes are informational; runtime changes block parity."""
    try:
        before, after = _checked(baseline), _checked(candidate)
    except (ValueError, TypeError) as exc:
        return {"status": "fail", "issues": [str(exc)], "source_changed": None}
    changed = {
        key: {"baseline": before.source.get(key), "candidate": after.source.get(key)}
        for key in ("git_commit", "git_dirty", "active_source_sha256", "package_version")
        if before.source.get(key) != after.source.get(key)
    }
    runtime = [key for key in RUNTIME_KEYS if before.source[key] != after.source[key]]
    if runtime:
        return {
            "status": "unsupported",
            "issues": [f"incompatible runtime/plugin provenance: {runtime}"],
            "source_changed": changed,
        }
    previous = {case.case_id: case for case in before.cases}
    issues, unsupported = [], []
    for case in after.cases:
        old = previous[case.case_id]
        if old.status != "pass":
            issues.append(f"{case.case_id}: baseline is not passing")
        if case.status == "unsupported":
            unsupported.append(f"{case.case_id}: {case.error}")
        elif case.status != "pass":
            issues.append(f"{case.case_id}: {case.error or case.status}")
        elif canonical(case.outputs) != canonical(old.outputs):
            keys = sorted(set(case.outputs) | set(old.outputs))
            fields = [key for key in keys if canonical(case.outputs.get(key)) != canonical(old.outputs.get(key))]
            issues.append(f"{case.case_id}: changed/unavailable outputs: {', '.join(fields)}")
    return {
        "status": "fail" if issues else "unsupported" if unsupported else "pass",
        "issues": issues + unsupported,
        "source_changed": changed,
    }


def array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError("nonfinite or object array is not regression evidence")
    header = canonical({"dtype": value.dtype.str, "shape": list(value.shape)})
    return hashlib.sha256(header + b"\0" + value.tobytes(order="C")).hexdigest()


def _arrays(path: Path) -> dict[str, str]:
    with np.load(path, allow_pickle=False) as values:
        return {key: array_sha256(values[key]) for key in sorted(values.files)}


def _evidence(path: Path, root: Path) -> Evidence:
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("record escaped regression output directory")
    record = load_record(path)
    for ref in record.artifacts:
        artifact_path(path, ref)
    return Evidence(
        id=record.id,
        path=path.relative_to(root).as_posix(),
        kind=record.kind,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artifacts=record.artifacts,
    )


def validate_evidence(manifest: dict, root: Path) -> list[str]:
    """Validate candidate files under caller-owned root; never call this on baseline references."""
    errors = []
    try:
        parsed = _checked(manifest)
        for case in parsed.cases:
            for expected in case.evidence:
                path = safe_path(root, expected.path)
                actual = _evidence(path, root.resolve())
                if actual != expected:
                    raise ValueError(f"record/artifact manifest differs: {case.case_id}")
                record = load_record(path)
                for key in (*RUNTIME_KEYS, "active_source_sha256"):
                    if record.version.get(key) != parsed.source[key]:
                        raise ValueError(f"record provenance differs: {case.case_id}/{key}")
                if record.status == "running":
                    raise ValueError(f"nonterminal experiment: {case.case_id}")
                if case.status == "pass" and record.status not in ("completed", "stopped"):
                    raise ValueError(f"unsuccessful experiment: {case.case_id}")
            wrapper = load_record(safe_path(root, case.evidence[-1].path))
            if wrapper.kind != "framework_regression" or wrapper.metrics.get("golden") != case.outputs:
                raise ValueError(f"outputs do not match recorded evidence: {case.case_id}")
            if wrapper.algorithm.get("input_signature") != case.input_signature:
                raise ValueError(f"recorded input signature differs: {case.case_id}")
    except (OSError, ValueError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def _role(path: Path, role: str) -> Path:
    record = load_record(path)
    refs = [ref for ref in record.artifacts if ref.role == role]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one {role} artifact")
    return artifact_path(path, refs[0])


def _save_arrays(handle: ExperimentHandle, **arrays: np.ndarray) -> dict:
    hashes = {}
    for name, value in arrays.items():
        hashes[name] = array_sha256(value)
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        handle._artifact_bytes(stream.getvalue(), name, "npy")
    return hashes


def _suite_cases(root: Path, state: dict, case_id: str, handle: ExperimentHandle) -> tuple[dict, list[Path]]:
    from robotforge.control.compare import compare_methods
    from robotforge.control.rl import store
    from robotforge.experiments.snapshot import restore_environment
    from robotforge.training import runs
    from robotforge.training.envs import env_spec

    inputs = CASE_INPUTS[case_id]
    if case_id == "mcp_run_controller":
        import asyncio

        from robotforge.cli import listen_on_free_port
        from robotforge.mcp_server import build_server
        from robotforge.playground import Playground
        from robotforge.sim.session import SimSession

        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world("regression-mcp")
        pg.add_robot(inputs["robot"], template=inputs["template"])
        pg.session.data.qpos[:] = inputs["initial_qpos"]
        mujoco.mj_forward(pg.session.model, pg.session.data)

        async def call(server):
            definitions = [tool.model_dump(mode="json", by_alias=True) for tool in await server.list_tools()]
            definitions = [
                {key: tool.get(key) for key in inputs["tool_schema_fields"]}
                for tool in sorted(definitions, key=lambda tool: tool["name"])
            ]
            handle._artifact_bytes(canonical(definitions), "mcp_tool_definitions", "json")
            response = await server.call_tool(
                "run_controller",
                {key: inputs[key] for key in ("robot", "task", "method", "seconds", "push_at", "task_params")},
            )
            if response.is_error:
                raise ValueError(f"MCP run_controller failed: {response}")
            result = json.loads("".join(block.text for block in response.content if block.type == "text"))
            return result, definitions

        # Own and close the socket; direct tool calls never enter the server/viewer lifespan.
        with listen_on_free_port(0, "127.0.0.1", attempts=1) as listener:
            server = build_server(pg, listener.getsockname()[1], viewer_socket=listener)
            try:
                result, definitions = asyncio.run(call(server))
            finally:
                pg.session.set_controller(inputs["robot"], None)
                pg.session.finish_controller_observations("regression_finished")
                pg.renderer.close()
        path = Path(result["experiment"]["path"])
        record = load_record(path)
        if record.budget_used["physics_steps"] != 40 or record.algorithm["push"]["physics_step"] != 20:
            raise ValueError("MCP execution must run 40 physics steps with push at step 20")
        paths = [path]
        if isinstance(result.get("execution"), dict) and "path" in result["execution"]:
            paths.append(Path(result["execution"]["path"]))
        return {
            "physics_steps": record.budget_used["physics_steps"],
            "push_step": 20,
            "task": record.task,
            "method": record.algorithm["name"],
            "params": record.algorithm["params"],
            "reward": result["reward"],
            "effort_j": result["effort_j"],
            "result": result["result"],
            "final_arrays": _arrays(_role(path, "final_physics")),
            "tool_count": len(definitions),
            "tool_definitions_sha256": digest(definitions),
        }, paths
    if case_id == "balance_compare":
        result = compare_methods(**inputs, record_dir=root / "comparison")
        parent = Path(result.experiment["path"])
        paths = [parent, *(Path(ref["path"]) for ref in load_record(parent).metrics["children"])]
        outputs = {}
        for item, path in zip(result.results, paths[1:], strict=True):
            measured = item.to_dict()
            outputs[item.method] = {
                key: measured[key]
                for key in ("reward", "cost", "effort_j", "sim_seconds", "success", "failed", "report")
            }
            outputs[item.method]["final_state_arrays"] = _arrays(_role(path, "final_state"))
        if len(outputs) != 2:
            raise ValueError("comparison did not finish both methods")
        return outputs, paths
    if case_id == "reach_control":
        from robotforge.control.runtime import start_control
        from robotforge.playground import Playground
        from robotforge.sim.session import SimSession

        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world("regression-reach")
        pg.add_robot("arm", template=inputs["template"])
        try:
            result = start_control(pg, "arm", "reach", inputs["method"], {"target": inputs["target"]})
            state["reach"] = Path(result["experiment"]["path"])
            pg.session.step(inputs["physics_steps"])
        finally:
            pg.session.set_controller("arm", None)
        record = load_record(state["reach"])
        return {
            "physics_steps": record.budget_used["physics_steps"],
            "sim_seconds": record.metrics["sim_seconds"],
            "task": record.metrics["task"],
            "final_arrays": _arrays(_role(state["reach"], "final_physics")),
        }, [state["reach"]]
    if case_id == "reach_replay":
        from .playback import reproduce_replay

        replay = reproduce_replay(state["reach"])
        if not replay["within_absolute_tolerance"]:
            raise ValueError(f"replay exceeded 1e-8: {replay['max_absolute_error']}")
        arrays = _save_arrays(handle, **{key: np.asarray(replay[key]) for key in ("qpos", "qvel", "ctrl")})
        return {
            "physics_steps": replay["physics_steps"],
            "final_arrays": arrays,
            "within_absolute_tolerance": True,
            "max_absolute_error": replay["max_absolute_error"],
        }, [state["reach"]]
    if case_id == "ars_training":
        spec = runs.make_run_spec(
            "tiny",
            env_spec("balance", episode_seconds=0.1).model_dump(mode="json"),
            algos=["ars"],
            steps=8,
            seed=7,
            parallel=0,
            eval_episodes=1,
            held_out_episodes=0,
            params={"ars": {"directions": 1}},
        )
        folder = root / "training" / "policies" / "tiny"
        result = runs.run_inline(spec, folder)
        parent = Path(result["experiment"]["path"])
        meta = store.read_json(folder / store.POLICY_FILE)
        child = Path(meta["policy_experiment"]["path"])
        policy = _role(parent, "winning_policy")
        state.update(training=(spec, folder, parent, child, policy, policy.read_bytes()))
        segment = load_record(child)
        return {
            "policy_arrays": _arrays(policy),
            "training_steps": segment.budget_used["training_steps"],
            "metrics": {"score": segment.metrics["score"]},
        }, [parent, child]
    if case_id == "policy_lineage":
        spec, folder, parent, child, policy, before = state["training"]
        old_parent, old_child = load_record(parent), load_record(child)
        if old_child.parent is None or old_child.parent["id"] != old_parent.id:
            raise ValueError("training child parent mismatch")
        if [ref["id"] for ref in old_parent.origins] != [old_child.id]:
            raise ValueError("winning policy origin mismatch")
        if _role(child, "policy").read_bytes() != before:
            raise ValueError("winning and segment policy bytes differ")
        result = runs.run_inline(spec.model_copy(update={"seed": inputs["replacement_seed"]}), folder)
        if policy.read_bytes() != before:
            raise ValueError("archived policy changed after named-policy replacement")
        replacement = Path(result["experiment"]["path"])
        new_children = [Path(ref["path"]) for trial in result["trials"] for ref in trial["experiments"]]
        return {"immutable_bytes": True, "lineage_verified": True, "policy_arrays": _arrays(policy)}, [
            parent,
            child,
            replacement,
            *new_children,
        ]
    if case_id == "snapshot_restore":
        from robotforge.control.env import task_world

        model, _ = task_world("balance")
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:], data.ctrl[:] = inputs["qpos"], inputs["qvel"], inputs["ctrl"]
        data.qfrc_applied[:] = inputs["force"]
        data.xfrc_applied[1, 0], data.time = inputs["body_force"], inputs["time"]
        mujoco.mj_forward(model, data)
        handle.snapshot_environment(model, data)
        restored_model, restored = restore_environment(handle.path)
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        initial = np.empty(mujoco.mj_stateSize(model, kind))
        recovered = np.empty_like(initial)
        mujoco.mj_getState(model, data, initial, kind)
        mujoco.mj_getState(restored_model, restored, recovered, kind)
        if not np.array_equal(initial, recovered):
            raise ValueError("raw integration snapshot did not restore exactly")
        for _ in range(inputs["steps"]):
            mujoco.mj_step(model, data)
            mujoco.mj_step(restored_model, restored)
        mujoco.mj_getState(model, data, initial, kind)
        mujoco.mj_getState(restored_model, restored, recovered, kind)
        if not np.array_equal(initial, recovered):
            raise ValueError("restored physics diverged")
        return {"exact_restore": True, "final_arrays": _save_arrays(handle, integration=initial)}, []
    raise ValueError(f"unknown regression case: {case_id}")


def run_suite(out: Path) -> dict:
    """Run fixed cases in an empty output directory; reject loaded plugins and physics callbacks."""
    from robotforge.extend import registry

    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("regression output directory must be empty")
    old_env = {key: os.environ.get(key) for key in ("ROBOTFORGE_HOME", "ROBOTFORGE_PLUGINS")}
    os.environ.update(ROBOTFORGE_HOME=str(root / "home"), ROBOTFORGE_PLUGINS="")
    cases, state = [], {}
    try:
        with registry.isolated():
            source = provenance()
            unsupported = []
            if registry.REGISTRY.items or registry.REGISTRY.files or registry.REGISTRY.worlds:
                unsupported.append("loaded plugin state requires a fresh process")
            for name in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias", "contactfilter"):
                getter = getattr(mujoco, f"get_mjcb_{name}", None)
                if getter is not None and getter() is not None:
                    unsupported.append(f"active MuJoCo callback: {name}")
            for case_id in CASE_INPUTS:
                handle = ExperimentStore(root / "cases").begin(
                    "framework_regression",
                    task={"case_id": case_id},
                    algorithm={
                        "suite": SUITE,
                        "input_signature": input_signature(case_id),
                        "inputs": CASE_INPUTS[case_id],
                    },
                )
                outputs, paths, error, status = {}, [], None, "pass"
                try:
                    if unsupported:
                        status, error = "unsupported", "; ".join(unsupported)
                        handle.update(metrics={"golden": {}})
                        handle.finish("stopped", reason=error)
                    else:
                        outputs, paths = _suite_cases(root, state, case_id, handle)
                        Case.finite_outputs(outputs)
                        handle.update(metrics={"golden": outputs})
                        handle.finish("completed", reason="case completed")
                except BaseException as exc:
                    status, error = "fail", f"{type(exc).__name__}: {exc}"
                    outputs = {}
                    if handle.record.status == "running":
                        handle.update(metrics={"golden": outputs})
                        handle.finish(
                            "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                            reason="regression case failed",
                            error=error,
                        )
                    if not isinstance(exc, Exception):
                        raise
                evidence = [_evidence(path, root) for path in [*paths, handle.path]]
                cases.append(
                    Case(
                        case_id=case_id,
                        input_signature=input_signature(case_id),
                        status=status,
                        outputs=outputs,
                        evidence=evidence,
                        error=error,
                    )
                )
            manifest = Manifest(source=source, cases=cases).model_dump(mode="json")
            (root / "manifest.json").write_bytes(canonical(manifest))
            return manifest
    finally:
        # An adapter interrupted between allocation and execution must not leave live records.
        try:
            for path in root.rglob("experiment.json"):
                record = load_record(path)
                if record.status == "running":
                    ExperimentHandle(path.parent, record).finish("cancelled", reason="regression suite cleanup")
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    """Explicit capture or read-only baseline comparison. Exit 2 means unsupported runtime."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", action="store_true", help="explicitly capture a new baseline manifest")
    action.add_argument("--baseline", type=Path, help="compare with an existing inert manifest; never overwrite it")
    args = parser.parse_args(argv)
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
        if not args.capture:
            _checked(baseline)
        candidate = run_suite(args.out)
        if args.capture:
            states = {case["status"] for case in candidate["cases"]}
            report = {
                "status": "fail" if "fail" in states else "unsupported" if "unsupported" in states else "pass",
                "issues": [
                    f"{case['case_id']}: {case['error']}" for case in candidate["cases"] if case["status"] != "pass"
                ],
                "source_changed": {},
            }
        else:
            report = compare_manifests(baseline, candidate)
        errors = validate_evidence(candidate, args.out)
        if errors:
            report.update(status="fail", issues=[*report["issues"], *errors])
        report["mode"] = "capture" if args.capture else "compare"
        (args.out / "report.json").write_bytes(canonical(report))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return {"pass": 0, "fail": 1, "unsupported": 2}[report["status"]]
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "fail", "issues": [str(exc)]}))
        return 1
