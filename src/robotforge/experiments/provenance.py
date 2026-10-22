"""Describe the active source without importing plugins or constructing their factories."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .schema import digest


@lru_cache(maxsize=8192)
def _file_hash(path: str, mtime_ns: int, ctime_ns: int, size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(path: Path) -> str:
    stat = path.stat()
    return _file_hash(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=3, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def provenance() -> dict[str, Any]:
    """Hash active package files and already registered plugin source (never execute plugins)."""
    package = Path(__file__).resolve().parents[1]
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            files[path.relative_to(package).as_posix()] = _hash(path)
    plugins = []
    module = sys.modules.get("robotforge.extend.registry")
    registry = getattr(module, "REGISTRY", None)
    if registry is not None:
        for (kind, name), item in sorted(registry.items.items()):
            path = item.path
            item_source: dict[str, Any] = {"kind": kind, "name": name, "label": item.plugin}
            if path is not None:
                try:
                    item_source["sha256"] = _hash(Path(path))
                    item_source["source_state"] = "current_file_not_loaded_bytecode"
                except OSError:
                    item_source["source_state"] = "unavailable"
            else:
                item_source["source_state"] = "registered_in_memory_unknown"
            loaded = registry.files.get(path)
            helpers = []
            if loaded is not None:
                for module_name in loaded.imported:
                    helper = sys.modules.get(module_name)
                    helper_file = getattr(helper, "__file__", None)
                    if helper_file:
                        helper_path = Path(helper_file)
                        entry = {"name": helper_path.name}
                        try:
                            entry["sha256"] = _hash(helper_path)
                        except OSError:
                            entry["source_state"] = "unavailable"
                        helpers.append(entry)
                item_source["loaded_file_signature_matches"] = (
                    path is not None
                    and path.exists()
                    and loaded.signature == (path.stat().st_mtime_ns, path.stat().st_size)
                )
            item_source["imported_helpers"] = sorted(
                helpers, key=lambda entry: (entry["name"], entry.get("sha256", ""))
            )
            plugins.append(item_source)
    root = package.parents[1]
    return {
        "package": "robotforge",
        "package_version": _version("robotforge"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mujoco": _version("mujoco"),
        "numpy": _version("numpy"),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain", "--untracked-files=normal")),
        "active_source_sha256": digest(files),
        "plugins": plugins,
        "plugin_dependencies": "external; source hashes describe current files, not captured runtime state",
    }


code_version = provenance
