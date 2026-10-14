"""Where learned policies live: shipped with robotforge, or installed by the user.

Each policy is a folder with policy.onnx and its card, policy.json. Shipped policies sit in
robotforge/policies/; installed ones in $ROBOTFORGE_HOME/policies/ (default ~/robotforge/policies)
and win over a shipped policy of the same name.
"""

from __future__ import annotations

import difflib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotforge.bridge.spec import NAME_RE, PolicySpec, SpecError, load_spec, save_spec

SHIPPED_DIR = Path(__file__).resolve().parent.parent / "policies"


def user_dir() -> Path:
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / "policies"


@dataclass(frozen=True)
class InstalledPolicy:
    spec: PolicySpec
    directory: Path
    shipped: bool

    def describe(self) -> dict[str, Any]:
        s = self.spec
        out: dict[str, Any] = {
            "name": s.name,
            "title": s.title,
            "robot": s.robot,
            "summary": s.summary,
            "commands": s.commands.model_dump(),
            "policy_hz": round(1.0 / s.policy_dt, 1),
            "shipped": self.shipped,
        }
        if s.trained_with:
            out["trained_with"] = s.trained_with
        return out


def _scan(root: Path, shipped: bool) -> dict[str, InstalledPolicy]:
    found: dict[str, InstalledPolicy] = {}
    if not root.is_dir():
        return found
    for folder in sorted(p for p in root.iterdir() if (p / "policy.json").is_file()):
        try:
            spec = load_spec(folder)
        except SpecError:
            continue  # a broken folder must not hide the others; `load` reports it by name
        found[spec.name] = InstalledPolicy(spec, folder, shipped)
    return found


def all_policies() -> dict[str, InstalledPolicy]:
    return {**_scan(SHIPPED_DIR, True), **_scan(user_dir(), False)}


def load(name: str) -> InstalledPolicy:
    if not NAME_RE.match(name):  # names come from agents and the viewer: never let one be a path
        raise SpecError(f"{name!r} is not a policy name (lowercase letters, digits and _); see list_policies")
    for root, shipped in ((user_dir(), False), (SHIPPED_DIR, True)):
        folder = root / name
        if (folder / "policy.json").is_file():
            return InstalledPolicy(load_spec(folder), folder, shipped)  # raises with the actual problem
    names = sorted(all_policies())
    close = difflib.get_close_matches(name, names, n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise SpecError(f"no policy named {name!r}.{hint} Installed: {', '.join(names) or 'none'}")


def for_robot(library_robot: str | None) -> InstalledPolicy | None:
    """The policy to use for a library robot: an installed one first, then a shipped one."""
    if library_robot is None:
        return None
    matches = [p for p in all_policies().values() if p.spec.robot == library_robot]
    matches.sort(key=lambda p: (p.shipped, p.spec.name))
    return matches[0] if matches else None


def install(source: Path, name: str | None = None) -> InstalledPolicy:
    """Copy a policy folder (policy.json + model) into the user's policies, after validating it."""
    spec = load_spec(source)
    if name is not None and name != spec.name:
        spec = spec.model_copy(update={"name": name})
        PolicySpec.model_validate(spec.model_dump())  # the new name must be valid too
    target = user_dir() / spec.name
    staging = target.with_name(f".{spec.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copy2(source / spec.onnx, staging / spec.onnx)
    save_spec(spec, staging)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)  # one step, so a half-copied policy never looks installed
    return InstalledPolicy(spec, target, False)
