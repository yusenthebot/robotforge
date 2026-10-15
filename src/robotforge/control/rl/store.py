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

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from robotforge import wording
from robotforge.control.tasks import TaskError

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
