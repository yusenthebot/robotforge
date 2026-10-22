"""Record live attachments without wrapping controllers or owning the simulation clock."""

from __future__ import annotations

import json
import logging
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robotforge.control.task_registry import describe_task, registered_task_name
from robotforge.experiments import ExperimentHandle, ExperimentStore

log = logging.getLogger(__name__)


def _values(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _values(value.tolist())
    if isinstance(value, np.generic):
        return _values(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _values(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_values(child) for child in value]
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return {"callable": f"{module}.{name}", "runtime_state": "not_captured"}
    return value


def _measured(value: Any) -> tuple[Any, list[str]]:
    missing: list[str] = []

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, float) and not math.isfinite(item):
            missing.append(path)
            return None
        if isinstance(item, dict):
            return {key: visit(child, f"{path}.{key}") for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, f"{path}[{i}]") for i, child in enumerate(item)]
        return item

    return visit(_values(value), "metrics"), missing


def experiment_home(world_dir: Path | None = None) -> Path:
    from robotforge.design.store import worlds_home

    return (Path(world_dir) if world_dir is not None else worlds_home().parent) / "experiments"


def task_spec(task: Any) -> dict[str, Any]:
    params = getattr(task, "settings", None)
    return {
        "registered_name": registered_task_name(task),
        "implementation_name": task.name,
        "params": _values(params if isinstance(params, dict) else task.params.model_dump()),
        "description": _values(describe_task(task)),
    }


def _policy(handle: ExperimentHandle, policy: Any) -> None:
    origins = [
        policy.meta[key] for key in ("experiment", "policy_experiment") if isinstance(policy.meta.get(key), dict)
    ]
    with tempfile.TemporaryDirectory(prefix="robotforge-policy-record-") as folder:
        net = Path(folder) / "policy.npz"
        policy.net.save(net)
        weights = handle.artifact(net, "policy_weights")
        meta = Path(folder) / "policy.json"
        meta.write_text(json.dumps(policy.meta, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        metadata = handle.artifact(meta, "policy_metadata")
    algorithm = {
        **handle.record.algorithm,
        "policy": policy.name,
        "policy_sha256": weights.sha256,
        "policy_metadata_sha256": metadata.sha256,
        "training_algorithm": policy.meta.get("algo"),
        "trial": getattr(policy, "trial_id", None),
        "training_origin": "reported_by_metadata" if origins else "unknown_legacy",
    }
    handle.update(algorithm=algorithm, origins=origins)


@dataclass
class _Attachment:
    handle: ExperimentHandle
    task: Any
    controller: Any
    time: float
    steps: int
    wall: float


class LiveExperiments:
    def __init__(self, session: Any):
        self.session = session
        self.active: dict[int, _Attachment] = {}
        self.errors: list[str] = []
        session.controller_hooks.append(self._changed)

    def _changed(self, name: str, old: Any, new: Any, reason: str) -> None:
        if old is not None:
            self.finish(old, reason=reason)

    def finish(
        self, controller: Any, *, status: str = "stopped", reason: str = "stopped", metrics: dict | None = None
    ) -> dict[str, str] | None:
        attached = self.active.pop(id(controller), None)
        if attached is None:
            return None
        handle, task, session = attached.handle, attached.task, self.session
        try:
            from robotforge.control.tasks.base import Task

            observed = getattr(task.observe, "__func__", None) is Task.observe or bool(
                metrics and metrics.get("task_observation") == "episode_monitor"
            )
            measured, missing = _measured(
                {
                    "namespace": "live_attachment",
                    "sim_seconds": float(session.data.time) - attached.time,
                    "task": task.report(session.data) if observed else None,
                    "success_at_end": bool(task.success(session.data)) if observed else None,
                    "task_metric_coverage": "observed" if observed else "unavailable_without_task_observation",
                    **(metrics or {}),
                }
            )
            handle.update(
                metrics={**measured, "nonfinite_values": missing},
                budget_used={
                    "physics_steps": session.steps_taken - attached.steps,
                    "sim_seconds": float(session.data.time) - attached.time,
                    "wall_seconds": time.monotonic() - attached.wall,
                },
            )
            with tempfile.TemporaryDirectory(prefix="robotforge-state-record-") as folder:
                path = Path(folder) / "final.npz"
                np.savez(path, qpos=session.data.qpos, qvel=session.data.qvel, ctrl=session.data.ctrl)
                handle.artifact(path, "final_physics")
            handle.finish(status, reason=reason)
        except Exception as exc:
            self.errors.append(f"{handle.record.id}: {type(exc).__name__}: {exc}")
            self.errors[:] = self.errors[-20:]
            log.exception("could not finish live experiment %s", handle.record.id)
            # A failed observation must not stop the session's controller cleanup.
            try:
                handle.finish("failed", reason="recording_failed", error=str(exc))
            except Exception:
                log.exception("could not persist the live experiment error")
        return handle.reference()


def ledger(session: Any) -> LiveExperiments:
    current = getattr(session, "_live_experiments", None)
    if current is None:
        current = LiveExperiments(session)
        session._live_experiments = current
    return current


def attach_record(
    session: Any,
    robot: str,
    controller: Any,
    task: Any,
    method: str,
    *,
    world: Any = None,
    world_dir: Path | None = None,
    method_params: dict | None = None,
    requested_seconds: float | None = None,
    entrypoint: str = "live",
    overrides: dict | None = None,
) -> dict[str, str]:
    """Snapshot an admitted attachment, then attach the same controller object."""
    book = ledger(session)
    from robotforge.control.method_params import parameter_model
    from robotforge.control.methods import _LQR_WEIGHTS

    fields = parameter_model(method, task).model_fields
    resolved = {
        key: field.get_default(call_default_factory=True) for key, field in fields.items() if not field.is_required()
    }
    if method == "lqr" and not hasattr(task, "lqr_controller"):
        resolved.update(_LQR_WEIGHTS.get(task.name, {}))
    resolved.update(method_params or {})
    handle = ExperimentStore(experiment_home(world_dir)).begin(
        "replay" if method.startswith("policy:") else "control",
        task=task_spec(task),
        algorithm={"domain": "control", "name": method, "params": _values(resolved)},
        seeds={"reset": None, "mode": "current_state"},
        budget={
            "requested": {"sim_seconds": requested_seconds},
            "semantics": "bounded_observation" if requested_seconds is not None else "until_detached",
        },
        environment={"entrypoint": entrypoint, "task_overrides": _values(overrides or {})},
    )
    try:
        handle.snapshot_environment(session.model, session.data, world=world, scope="live", robot=robot)
        policy = getattr(controller, "trained_policy", None)
        if policy is not None:
            _policy(handle, policy)
        handle.update(
            reproducibility={
                **handle.record.reproducibility,
                "level": "captured_inputs",
                "limitations": [
                    "interactive inputs and other controller internal states are not captured",
                    "plugin RNG, external I/O and sensor history may not be reconstructable",
                ],
                "other_controllers": [name for name in session.controllers if name != robot],
            }
        )
        old = session.controllers.get(robot)
        released = old is not None and callable(getattr(old, "release", None))
        session.set_controller(robot, controller)
        if released:
            try:
                # Release callbacks can restore actuator settings and clear applied forces.
                handle.snapshot_environment(session.model, session.data, world=world, scope="live", robot=robot)
                handle.update(reproducibility={"replacement_release": True})
            except Exception as exc:
                # The command has taken effect. Return its recording failure without losing script ownership.
                book.errors.append(f"{handle.record.id}: {exc}")
                log.exception("post-release snapshot failed for experiment %s", handle.record.id)
                try:
                    handle.finish("failed", reason="post_attachment_snapshot_failed", error=str(exc))
                except Exception:
                    log.exception("could not persist post-attachment recording failure %s", handle.record.id)
                return {**handle.reference(), "recording_error": str(exc)}
        book.active[id(controller)] = _Attachment(
            handle, task, controller, float(session.data.time), session.steps_taken, time.monotonic()
        )
    except BaseException as exc:
        handle.finish("failed", reason="attachment_failed", error=str(exc))
        raise
    return handle.reference()


def finish_record(session: Any, controller: Any, **kwargs: Any) -> dict[str, str] | None:
    return ledger(session).finish(controller, **kwargs)


def history(session: Any, world_dir: Path | None = None, limit: int = 20) -> dict[str, Any]:
    """Inspect bounded, recent metadata; malformed records are reported separately."""
    from robotforge.experiments import load_record

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("experiment history limit must be an integer from 1 to 100")
    home = experiment_home(world_dir)
    paths = sorted(home.glob("*/experiment.json"), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    rows, errors = [], []
    for path in paths[:limit]:
        try:
            record = load_record(path)
            rows.append(
                {
                    "id": record.id,
                    "path": str(path),
                    "kind": record.kind,
                    "status": record.status,
                    "task": record.task.get("registered_name"),
                    "algorithm": record.algorithm.get("name"),
                    "created_at": record.created_at,
                    "finished_at": record.finished_at,
                }
            )
        except (OSError, ValueError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    book = getattr(session, "_live_experiments", None)
    return {
        "experiments": rows,
        "active": [item.handle.reference() for item in book.active.values()] if book else [],
        "errors": errors,
        "recording_errors": list(book.errors) if book else [],
    }
