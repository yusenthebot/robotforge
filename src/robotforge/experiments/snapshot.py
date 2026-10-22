"""Compiled MuJoCo snapshots; not portable source archives or Python runtime checkpoints."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import ArtifactRef, canonical, plain
from .store import load_record, record_path, verified_artifact

if TYPE_CHECKING:
    from .store import ExperimentHandle


def snapshot_environment(
    handle: ExperimentHandle,
    model: Any,
    data: Any,
    *,
    world: Any = None,
    scope: str | None = None,
    robot: str | None = None,
) -> dict[str, Any]:
    """Capture the full compiled model and integration state, including controls/external forces."""
    import mujoco
    import numpy as np

    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_kind), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_kind)
    if not np.isfinite(state).all():
        raise ValueError("cannot snapshot a nonfinite integration state")
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=binary)
    model_ref = handle._artifact_bytes(binary.tobytes(), "compiled_model", "mjb")
    stream = io.BytesIO()
    np.save(stream, state, allow_pickle=False)
    state_ref = handle._artifact_bytes(stream.getvalue(), "integration_state", "npy")
    world_ref = None
    if world is not None:
        world_ref = handle._artifact_bytes(canonical(world), "world_specification", "json")
    info = {
        "format": "mujoco_mjb_integration_v1",
        "mujoco_version": mujoco.__version__,
        "scope": scope or "full_compiled_model",
        "robot": robot,
        "model": model_ref.model_dump(),
        "state": state_ref.model_dump(),
        "world": world_ref.model_dump() if world_ref else None,
        "state_spec": int(state_kind),
        "state_size": len(state),
        "model_info": {
            name: int(getattr(model, name))
            for name in ("nq", "nv", "nu", "na", "nbody", "ngeom", "njnt", "nmocap", "nplugin")
        },
        "physics_timestep": float(model.opt.timestep),
        "coverage": {
            "self_contained": ["compiled_model", "integration_state"],
            "world_specification": "reference metadata; may refer to external files" if world is not None else "absent",
            "editable_model_source": "not captured",
            "external_assets": "compiled into MJB where supported by MuJoCo",
            "python_rng_and_controller_state": "not captured; use recorded seed roles and adapter metadata",
            "python_callbacks": "external runtime dependency; not captured",
            "native_plugins": "restore refused" if model.nplugin else "none in compiled model",
            "compatibility": "exact MuJoCo version and compatible platform required",
        },
    }
    handle.update(environment={"snapshot": plain(info)})
    return info


def restore_environment(path: str | Path) -> tuple[Any, Any]:
    """Restore validated bytes without importing world Python or loading pickles."""
    import mujoco
    import numpy as np

    record_file = record_path(path)
    record = load_record(record_file)
    info = record.environment.get("snapshot")
    if not isinstance(info, dict) or info.get("format") != "mujoco_mjb_integration_v1":
        raise ValueError("experiment has no supported compiled environment snapshot")
    if info.get("mujoco_version") != mujoco.__version__:
        raise ValueError(f"snapshot requires MuJoCo {info.get('mujoco_version')}; installed {mujoco.__version__}")
    if info.get("model_info", {}).get("nplugin", 0):
        raise ValueError("native-plugin snapshots require external plugin code and cannot be restored safely here")
    refs = [ArtifactRef.model_validate(info[name]) for name in ("model", "state")]
    for ref in refs:
        if ref not in record.artifacts:
            raise ValueError("snapshot references an artifact missing from the record")
    model_path, state_path = [verified_artifact(record_file.parent, ref) for ref in refs]
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if model.nplugin:
        raise ValueError("native plugins are unsupported in experiment restore")
    data = mujoco.MjData(model)
    state_kind = mujoco.mjtState.mjSTATE_INTEGRATION
    if info.get("state_spec") != int(state_kind):
        raise ValueError("snapshot integration-state definition differs from this MuJoCo build")
    state = np.load(state_path, allow_pickle=False)
    if state.dtype != np.dtype("float64") or state.shape != (mujoco.mj_stateSize(model, state_kind),):
        raise ValueError("snapshot integration state has the wrong type or size")
    if not np.isfinite(state).all():
        raise ValueError("snapshot integration state contains nonfinite values")
    mujoco.mj_setState(model, data, state, state_kind)
    mujoco.mj_forward(model, data)
    # Forward computes derived fields but may change the warm start; retain the exact saved integration state.
    mujoco.mj_setState(model, data, state, state_kind)
    return model, data
