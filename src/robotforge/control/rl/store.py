"""Where trained policies live and what a policy folder contains. Needs no RL libraries.

A policy is a folder, `<world>/policies/<name>/` (or `$ROBOTFORGE_HOME/policies/<name>/`
when no world is open):

    policy.json        what was trained: task, robot, algorithm, observation layout, final scores
    status.json        training progress, rewritten every second or two while it runs
    policy.npz         the trained network as plain numpy arrays (runs without PyTorch)
    model.zip          the Stable-Baselines3 model, to keep training it or use it with SB3 directly
    vecnormalize.pkl   SB3's observation normalizer (its numbers are also inside policy.npz)
    train.log          everything the training process printed
    heartbeat          touched every few seconds while the training process is alive

model.zip and vecnormalize.pkl are Python pickles: loading one runs code from the file, so only
load them from folders you trained yourself. robotforge itself only ever reads policy.npz (plain
arrays, read without pickle) and the JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robotforge import wording
from robotforge.control.tasks import TaskError

if TYPE_CHECKING:
    from robotforge.control.rl.spec import TrainSpec
    from robotforge.experiments import ExperimentHandle, ExperimentStore

POLICY_FILE = "policy.json"
STATUS_FILE = "status.json"
WEIGHTS_FILE = "policy.npz"
MODEL_FILE = "model.zip"
NORM_FILE = "vecnormalize.pkl"
LOG_FILE = "train.log"
SPEC_FILE = "train.json"
STOP_FILE = "stop.request"  # the training process checks for this file and stops cleanly
HEARTBEAT_FILE = "heartbeat"

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
FINAL_STATES = ("done", "stopped", "failed")
WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}


def robotforge_home() -> Path:
    return Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()


def policies_home(world_dir: Path | None) -> Path:
    """Policies of the open world, or the shared folder when no world is open."""
    return (Path(world_dir) if world_dir is not None else robotforge_home()) / "policies"


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or name in WINDOWS_RESERVED:
        raise TaskError(
            "policy names use lowercase letters, digits and _, start with a letter (e.g. pole_ppo), "
            f"got {wording.short(name)}"
        )
    return name


def find_policy(name: str, world_dir: Path | None = None) -> Path:
    """The folder of policy `name`: in the open world first, then the shared folder. Raises with the known names."""
    places = [policies_home(world_dir)] if world_dir is not None else []
    places.append(policies_home(None))
    known = sorted({p["name"] for place in places for p in _summaries(place)})
    name = wording.match(name, known) or check_name(name)  # "Pole_PPO" is pole_ppo
    for place in places:
        folder = place / name
        if (folder / POLICY_FILE).exists():
            return folder
    if not known:
        raise TaskError(f"no policy named {name!r}; none trained yet (train_policy makes one)")
    raise TaskError(wording.unknown("policy", name, known, where="trained", many="policies"))


def read_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    for attempt in range(3):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:  # Windows: the writer is swapping the file in
            if attempt == 2:
                raise TaskError(f"could not read {path.name} in {path.parent}: {exc}") from exc
            time.sleep(0.05)
    return {}


def temp_path(path: Path) -> Path:
    """A private temporary name next to `path` (unique per process and thread)."""
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def replace_file(tmp: Path, path: Path) -> None:
    """Move a finished temporary file into place, so a reader never sees half a file. On Windows the
    move fails while someone has the target open, so retry for a moment (at most about a second)."""
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.03 * (attempt + 1))
    os.replace(tmp, path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    tmp = temp_path(path)
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=_plain), encoding="utf-8")
    replace_file(tmp, path)


def _plain(value: Any) -> Any:
    """numpy numbers and arrays as plain JSON values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot store {type(value).__name__} in JSON")


def summary(folder: Path) -> dict[str, Any]:
    """One line about a policy: what it does, how training went, whether it can run."""
    try:
        meta = read_json(folder / POLICY_FILE)
        status = read_json(folder / STATUS_FILE)
    except TaskError as exc:  # one damaged folder must not hide all the others
        return {"name": folder.name, "state": "unreadable", "error": str(exc), "runnable": False, "folder": str(folder)}
    final = meta.get("final") or {}
    out = {
        "name": folder.name,
        "task": meta.get("task"),
        "robot": meta.get("robot"),
        "algo": meta.get("algo"),
        "action_mode": meta.get("action_mode"),
        "state": status.get("state", "unknown"),
        "timesteps": status.get("timesteps", 0),
        "runnable": (folder / WEIGHTS_FILE).exists(),
        "folder": str(folder),
    }
    for key in ("success_rate", "mean_return"):
        if key in final:
            out[key] = final[key]
    return out


def _summaries(place: Path) -> list[dict[str, Any]]:
    if not place.is_dir():
        return []
    return [summary(f) for f in sorted(place.iterdir()) if (f / POLICY_FILE).exists()]


def list_policies(world_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every policy of the open world, then the shared ones (the open world's win on a name clash)."""
    seen: set[str] = set()
    out = []
    places = [policies_home(world_dir)] if world_dir is not None else []
    places.append(policies_home(None))
    for place in places:
        for item in _summaries(place):
            if item["name"] not in seen:
                seen.add(item["name"])
                out.append({**item, "where": "world" if world_dir is not None and place == places[0] else "shared"})
    return out


# Records live beside policies, so replacing a policy never erases its provenance.
def experiment_store(folder: Path) -> ExperimentStore:
    from robotforge.experiments import ExperimentStore

    return ExperimentStore(Path(folder).resolve().parent.parent / "experiments")


def policy_experiment_for(folder: Path, weights_path: Path, refs: list[dict[str, Any]]) -> dict[str, str] | None:
    """Find a segment that actually archived these weights; invalid evidence means unknown provenance."""
    from robotforge.experiments import artifact_path

    try:
        content = Path(weights_path).read_bytes()
    except OSError:
        return None
    sha, size = hashlib.sha256(content).hexdigest(), len(content)
    ledger = experiment_store(folder)
    for reference in reversed(refs):
        if not isinstance(reference, dict):
            continue
        try:
            # Open by a validated id inside this policy's store, never by a supplied arbitrary path.
            handle = ledger.open(reference["id"])
            if (
                Path(reference["path"]).resolve() != handle.path.resolve()
                or handle.record.id != reference["id"]
                or reference.get("kind") != handle.record.kind
                or handle.record.kind != "training_segment"
            ):
                continue
            for artifact in handle.record.artifacts:
                if artifact.role == "policy" and artifact.sha256 == sha and artifact.size == size:
                    artifact_path(handle.path, artifact)  # verify immutable archived bytes as well as metadata
                    return handle.reference()
        except (KeyError, TypeError, ValueError, OSError):
            # Stale/copied/corrupt references cannot justify attribution; an older valid segment still may.
            continue
    return None


def open_experiment(reference: dict[str, Any]) -> ExperimentHandle:
    from robotforge.experiments import ExperimentStore

    return ExperimentStore(Path(reference["path"]).parent.parent).open(reference["id"])


def training_experiment(folder: Path) -> ExperimentHandle | None:
    """Resume a verified parent inside this policy's store; copied/untrusted references remain unknown."""
    reference = read_json(Path(folder) / POLICY_FILE).get("experiment")
    if not isinstance(reference, dict):
        return None
    try:
        handle = experiment_store(folder).open(reference["id"])
        if (
            Path(reference["path"]).resolve() != handle.path.resolve()
            or handle.record.id != reference["id"]
            or reference.get("kind") != handle.record.kind
            or handle.record.kind != "training_run"
        ):
            return None
        return handle
    except (KeyError, TypeError, ValueError, OSError):
        # A status check may finalize this handle, so its location and identity must both be trusted.
        return None


def finish_experiment(folder: Path, status: dict[str, Any]) -> None:
    """A coordinator or confirmed-dead process monitor finalizes its own parent only."""
    handle = training_experiment(folder)
    if handle is None or handle.record.status != "running":
        return
    state = status.get("state")
    terminal = {"done": "completed", "stopped": "stopped", "failed": "failed"}.get(state)
    if terminal is None:
        return
    reason = "timeout" if status.get("timed_out") else status.get("stopped_early") or state
    handle.update(
        metrics={"phase": state, "final": status.get("final"), "trials": status.get("trials", [])},
        budget_used={
            "training_steps": status.get("timesteps", 0),
            "wall_seconds": status.get("elapsed_s", 0),
            "measurement": "latest coordinator observation; failed workers may have unreported steps"
            if state == "failed"
            else "completed coordinator observation",
        },
    )
    handle.finish(terminal, reason=reason, error=status.get("error"))


def legacy_experiment(folder: Path, spec: TrainSpec) -> ExperimentHandle:
    """The direct SB3 entry point uses the same record protocol without importing torch here."""
    from robotforge.control.rl.spec import algo_settings

    data = spec.model_dump(mode="json")
    handle = experiment_store(folder).begin(
        "training_run",
        environment={"train_spec": data, "scope": "robot", "robot": spec.robot.name},
        task={"registered_name": spec.task, "implementation_name": spec.task, "params": spec.task_params},
        algorithm={
            "domain": "training",
            "name": spec.algo,
            "params": algo_settings(spec),
            "settings_source": "TrainSpec",
        },
        seeds={
            "training": spec.seed,
            "worker_protocol": "training + rank",
            "ranking": {"base": 10_000, "protocol": "base + episode"},
            "final_evaluation": {"base": 20_000, "protocol": "base + episode"},
        },
        budget={
            "training_steps": spec.timesteps,
            "requested_wall_seconds": spec.max_minutes * 60 if spec.max_minutes else None,
            "semantics": "SB3 batch budget may overshoot; evaluation steps excluded; wall includes evaluation",
        },
    )
    handle.update(metrics={"phase": "queued"})
    return handle
