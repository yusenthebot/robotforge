"""Re-run a closed, isolated live attachment from verified snapshots and policy bytes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.experiments import artifact_path, load_record, provenance, restore_environment


def reproduce_replay(path: str | Path) -> dict[str, Any]:
    """Return observed final-state parity; unsupported runtime dependencies are refused explicitly.

    This executes on a new private simulation. It never imports a recorded plugin file or
    attaches to a live world, and it does not infer historical training provenance.
    """
    from robotforge.control.methods import make_controller
    from robotforge.control.rl.runtime import MlpPolicy, TrainedPolicy
    from robotforge.control.task_registry import bind_task, resolve_task
    from robotforge.design.world import World
    from robotforge.sim.session import SimSession

    record = load_record(path)
    if record.kind not in ("control", "replay") or record.status == "running":
        raise ValueError("replay reproduction needs a closed live control or replay record")
    if record.parent is not None:
        raise ValueError("comparison children use reproduce_comparison, which restores their randomized task")
    limitations = record.reproducibility
    if limitations.get("other_controllers") or limitations.get("replacement_release"):
        raise ValueError("other controller state or a replacement release is outside replay reproduction")
    steps = record.budget_used.get("physics_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 0 <= steps <= 10_000_000:
        raise ValueError("replay needs a bounded recorded physics-step count")
    snapshot = record.environment["snapshot"]
    world_ref = snapshot.get("world")
    world = World.model_validate_json(artifact_path(path, world_ref).read_text(encoding="utf-8")) if world_ref else None
    name = record.task["registered_name"]
    resolve_task(name, world)  # only the normally installed/trusted registry is consulted
    current = provenance()
    for key in ("active_source_sha256", "mujoco", "numpy", "python", "platform", "plugins"):
        if current.get(key) != record.version.get(key):
            raise ValueError(f"replay runtime differs from its recorded {key}")
    if any(item.get("source_state") != "current_file_not_loaded_bytecode" for item in current.get("plugins", [])):
        raise ValueError("replay cannot reconstruct in-memory plugin implementations")
    for callback in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias", "contactfilter"):
        getter = getattr(mujoco, f"get_mjcb_{callback}", None)
        if getter is not None and getter() is not None:
            raise ValueError(f"replay cannot reproduce with active MuJoCo Python callback {callback}")
    model, data = restore_environment(path)
    start_time = float(data.time)
    robot = snapshot["robot"]
    task = bind_task(name, model, robot, record.task["params"], world=world)
    task.begin(data)
    method = record.algorithm["name"]
    if method.startswith("policy:"):
        roles = {item.role: item for item in record.artifacts}
        if not {"policy_metadata", "policy_weights"} <= roles.keys():
            raise ValueError("replay is missing immutable policy artifacts")
        meta = json.loads(artifact_path(path, roles["policy_metadata"]).read_text(encoding="utf-8"))
        policy = TrainedPolicy(
            Path(record.algorithm["policy"]), meta, MlpPolicy.load(artifact_path(path, roles["policy_weights"]))
        )
        ctl = policy.controller(task, model, data)
    else:
        params = dict(record.algorithm.get("params") or {})

        def missing_runtime(value: Any) -> bool:
            if isinstance(value, dict):
                return value.get("runtime_state") == "not_captured" or any(missing_runtime(v) for v in value.values())
            return isinstance(value, list) and any(missing_runtime(v) for v in value)

        if missing_runtime(params):
            raise ValueError("replay controller parameters contain uncaptured callable runtime state")
        if method == "ars":
            raise ValueError("on-the-spot ARS controller internal policy is not captured by live records")
        ctl = make_controller(method, task, model, data, params)
    session = SimSession(model=model, data=data)
    session.set_controller(robot, ctl)
    push = record.algorithm.get("push") or {}
    push_step = push.get("physics_step")
    if push.get("at") is not None and push_step is None:
        raise ValueError("replay disturbance has no recorded application step")
    for step in range(steps):
        if step == push_step:
            task.push(data, push.get("strength", 1.0))
        session.step()
    final_ref = next((item for item in record.artifacts if item.role == "final_physics"), None)
    if final_ref is None:
        raise ValueError("replay has no captured final state to compare")
    with np.load(artifact_path(path, final_ref), allow_pickle=False) as expected:
        parity = all(np.array_equal(getattr(data, key), expected[key]) for key in ("qpos", "qvel", "ctrl"))
        errors = {
            key: float(np.max(np.abs(getattr(data, key) - expected[key]), initial=0.0))
            for key in ("qpos", "qvel", "ctrl")
        }
    return {
        "origin": {"id": record.id, "kind": record.kind, "path": str(path)},
        "physics_steps": steps,
        "sim_seconds": float(data.time) - start_time,
        "matches_recorded_final_state": parity,
        "max_absolute_error": errors,
        "within_absolute_tolerance": all(value <= 1e-8 for value in errors.values()),
        "absolute_tolerance": 1e-8,
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
    }
