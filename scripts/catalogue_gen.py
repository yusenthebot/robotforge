"""Regenerate src/robotforge/design/catalogue.json from the robot_descriptions package.

robot_descriptions knows over two hundred open robots (URDF, xacro, MJCF) and the repository and commit each one
lives at. robotforge keeps a plain copy of that list, so the package is not needed at run time and nothing is cloned
at import time. Run it against a throwaway environment, so the project's dependencies stay as they are:

    uv run --no-project --with robot_descriptions python scripts/catalogue_gen.py

The module files are read as text (their `_path.join` calls are evaluated by hand): importing one would clone the
repository it describes.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "design" / "catalogue.json"
# Attribute names that name a model file, best first. Variants (left and right hands, with and without gripper)
# come after the plain name; the plain one is the default.
FORMATS = (("MJCF_PATH", "mjcf"), ("URDF_PATH", "urdf"), ("XACRO_PATH", "xacro"))


def main() -> None:
    spec = importlib.util.find_spec("robot_descriptions")
    if spec is None or spec.origin is None:
        sys.exit("robot_descriptions is not installed; see the docstring")
    package = Path(spec.origin).parent
    from robot_descriptions import DESCRIPTIONS
    from robot_descriptions._repositories import REPOSITORIES

    entries = []
    for module, meta in sorted(DESCRIPTIONS.items()):
        source = package / f"{module}.py"
        if not source.is_file():
            continue
        found = _read_module(source)
        if found is None:
            continue
        repo_name, package_path, models = found
        repo = REPOSITORIES[repo_name]
        model = _choose(models)
        if model is None:
            continue
        path_attr, kind, rel = model
        entries.append(
            {
                "id": re.sub(r"[^a-z0-9]+", "_", module.removesuffix("_description").lower()).strip("_"),
                "module": module,
                "robot": meta.robot,
                "maker": meta.maker,
                "dof": meta.dof,
                "tags": sorted(meta.tags),
                "license": meta.license_spdx,
                "repo": {"name": repo_name, "url": repo.url, "commit": repo.commit},
                "package": package_path,
                "format": kind,
                "model": rel,
                "variants": sorted(f"{k}:{v}" for k, (_, v) in models.items() if k != path_attr),
                "xacro_args": _xacro_args(source),
            }
        )
    OUT.write_text(
        json.dumps({"source": "robot_descriptions 3.3.0", "robots": entries}, indent=0) + "\n", encoding="utf-8"
    )
    print(f"{len(entries)} robots -> {OUT}")


def _read_module(source: Path) -> tuple[str, str, dict[str, tuple[str, str]]] | None:
    """(repository name, package folder, {attribute: (format, file)}), paths relative to the repository."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    repo_name = None
    env: dict[str, str] = {"REPOSITORY_PATH": ""}
    models: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign | ast.Assign):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        name, value = target.id, node.value
        if name == "REPOSITORY_PATH" and isinstance(value, ast.Call):
            first = value.args[0] if value.args else None
            if isinstance(first, ast.Constant):
                repo_name = first.value
        elif isinstance(value, ast.Name) and value.id in env:  # PACKAGE_PATH = REPOSITORY_PATH
            env[name] = env[value.id]
        elif isinstance(value, ast.Call) and _is_join(value):
            parts = [_eval(a, env) for a in value.args]
            if all(p is not None for p in parts):
                env[name] = "/".join(p for p in parts if p)
    if repo_name is None or "PACKAGE_PATH" not in env:
        return None
    for name, rel in env.items():
        for prefix, kind in FORMATS:
            if name == prefix or name.startswith(prefix + "_"):
                models[name] = (kind, rel)
    return repo_name, env["PACKAGE_PATH"], models


def _is_join(call: ast.Call) -> bool:
    return isinstance(call.func, ast.Attribute) and call.func.attr == "join"


def _eval(node: ast.expr, env: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _choose(models: dict[str, tuple[str, str]]) -> tuple[str, str, str] | None:
    """The default model file: the plain MJCF, else the plain URDF, else the plain xacro, else the first variant."""
    for prefix, kind in FORMATS:
        if prefix in models:
            return prefix, kind, models[prefix][1]
    for _prefix, kind in FORMATS:
        variants = sorted(n for n, (k, _) in models.items() if k == kind)
        if variants:
            return variants[0], kind, models[variants[0]][1]
    return None


def _xacro_args(source: Path) -> dict[str, str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign | ast.Assign):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            if isinstance(target, ast.Name) and target.id == "XACRO_ARGS" and node.value is not None:
                try:
                    return {str(k): str(v) for k, v in ast.literal_eval(node.value).items()}
                except (ValueError, SyntaxError):
                    return {}
    return {}


if __name__ == "__main__":
    main()
