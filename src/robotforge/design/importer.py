"""The import doctor: open any URDF or mesh, repair what is wrong, and say what was done.

`import_file` is the single entry point used by `open_model`, `robotforge open` and
`add_robot(file=...)`. It returns a MuJoCo spec that is ready to attach to a world plus a
`DoctorReport` listing everything that was fixed or could not be fixed.

Pipeline for a URDF: expand xacro (if needed), find the mesh files, convert meshes MuJoCo
cannot read, repair joints and inertia, load with MuJoCo's own URDF compiler, then add the
things URDF has no notion of: visual/collision groups, motors, self-collision exclusions and a
floating or fixed base. Expensive steps are cached, so editing a world does not redo them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from robotforge.design.importing import xacro
from robotforge.design.importing.mesh_object import import_mesh_object
from robotforge.design.importing.meshio import (
    TOOL_VERSION,
)
from robotforge.design.importing.options import ImportOptions, ImportResult
from robotforge.design.importing.plan import PLAN_VERSION, RigPlan, build_spec
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MESH_SUFFIXES, MeshResolver
from robotforge.design.importing.rig import make_plan
from robotforge.design.importing.urdf import StagedUrdf, stage_urdf

log = logging.getLogger(__name__)

IMPORTER_VERSION = "1"
URDF_SUFFIXES = frozenset({".urdf", ".xacro"})
IMPORT_SUFFIXES = URDF_SUFFIXES | MESH_SUFFIXES
CACHE_LIMIT = 40
MESH_CACHE_BYTES = 1_000_000_000  # converted meshes are cheap to rebuild, but not worth gigabytes of disk


def is_importable(path: str | Path) -> bool:
    """True for files this module handles (URDF, xacro and plain meshes), judged by extension."""
    return Path(path).suffix.lower() in IMPORT_SUFFIXES


def import_cache_dir() -> Path:
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / "cache" / "import"


def import_file(
    path: str | Path,
    options: dict[str, Any] | ImportOptions | None = None,
    *,
    cache_dir: Path | None = None,
) -> ImportResult:
    """Import a URDF, xacro or mesh file. Raises ValueError with an actionable message on failure."""
    src = Path(path).expanduser()
    opts = ImportOptions.parse(options)
    if not src.is_file():
        raise ValueError(f"file not found: {src}")
    src = src.resolve()
    suffix = src.suffix.lower()
    cache = cache_dir or import_cache_dir()
    try:
        if suffix in URDF_SUFFIXES:
            return _import_urdf(src, opts, cache)
        if suffix in MESH_SUFFIXES:
            return import_mesh_object(src, opts, cache)
    except OSError as exc:  # most likely the cache folder: say how to point it somewhere writable
        raise ValueError(
            f"could not read or write files while importing {src.name} ({exc}); "
            f"robotforge keeps converted meshes in {cache}; set ROBOTFORGE_HOME to a writable folder if that is "
            "the problem"
        ) from exc
    raise ValueError(
        f"{src.name}: robotforge imports .urdf, .xacro and meshes ({', '.join(sorted(MESH_SUFFIXES))}); "
        "MuJoCo .xml files are opened directly"
    )


def compile_standalone(path: str | Path, options: dict[str, Any] | None = None, floor: str = "checker"):
    """A one-robot world around an imported file, for `open_model` and `robotforge open`."""
    from robotforge.design.world import RobotEntry, World, compile_world

    src = Path(path).expanduser().resolve()
    name = _robot_name(src.stem)
    world = World.model_validate({"name": name, "floor": {"style": floor}})
    world.robots.append(RobotEntry(name=name, file=str(src), params=dict(options or {})))
    return compile_world(world, src.parent)


def _robot_name(stem: str) -> str:
    """A valid world name (lowercase, digits, underscore, starts with a letter) from a file name."""
    name = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_") or "robot"
    if not name[0].isalpha():
        name = f"robot_{name}"
    return name[:32]


# ---- URDF ------------------------------------------------------------------------------------


def _import_urdf(src: Path, opts: ImportOptions, cache: Path) -> ImportResult:
    entry = cache / _cache_key(src, opts)
    cached = _load_cached(entry, src)
    if cached is None:
        _stage_and_plan(src, opts, cache, entry)
        cached = _load_cached(entry, src)
        if cached is None:  # pragma: no cover - would mean the cache folder is not writable
            raise ValueError(f"could not write the import cache at {entry}")
    plan, report = cached
    spec = build_spec(entry / "staged.urdf", plan, name=re.sub(r"\W+", "_", src.stem))
    return ImportResult(
        spec=spec,
        report=report,
        kind="robot",
        floating=plan.base == "floating",
        spawn_height=plan.spawn_height,
        home_qpos=dict(plan.home_qpos),
        home_ctrl={a.joint: a.home for a in plan.actuators},
    )


def _cache_key(src: Path, opts: ImportOptions) -> str:
    st = src.stat()
    versions = f"{IMPORTER_VERSION}|{TOOL_VERSION}|{PLAN_VERSION}"
    raw = f"{versions}|{src}|{st.st_size}|{st.st_mtime_ns}|{opts.model_dump_json()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_cached(entry: Path, src: Path) -> tuple[RigPlan, DoctorReport] | None:
    meta = entry / "meta.json"
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        for dep, size, mtime in info["deps"]:
            st = Path(dep).stat()
            if st.st_size != size or st.st_mtime_ns != mtime:
                return None
        if info["missing"]:
            resolver = MeshResolver(src)
            if any(resolver.resolve(uri).path is not None for uri in info["missing"]):
                return None  # a mesh that was missing has appeared since
        plan = RigPlan.from_json(info["plan"])
        report = DoctorReport.from_dict(info["report"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (entry / "staged.urdf").is_file():
        return None
    meta.touch()  # recently used entries survive pruning
    for dep, _, _ in info["deps"]:
        if Path(dep).is_relative_to(entry.parent):
            Path(dep).touch()  # converted meshes in use are the last to be pruned
    return plan, report


def _stage_and_plan(src: Path, opts: ImportOptions, cache: Path, entry: Path) -> None:
    report = DoctorReport(source=str(src))
    text, deps = _read_urdf(src, opts, report)
    try:
        robot = xacro.parse_text(text, src.name)
    except ValueError as exc:
        raise ValueError(f"{exc}. Look for a missing closing tag or stray character") from exc

    resolver = MeshResolver(src)
    staged = stage_urdf(robot, resolver, cache, report, opts.decompose)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "meta.json").unlink(missing_ok=True)  # an entry without meta.json counts as incomplete
    urdf_path = entry / "staged.urdf"
    urdf_path.write_text(staged.text, encoding="utf-8")

    plan = make_plan(staged, urdf_path, report, opts.base, opts.ignore_effort_limits)
    report.stats.update(_stats(staged, plan))

    staged_meshes = {Path(m.get("filename", "")) for m in robot.iter("mesh") if m.get("filename")}
    dep_files = {src, *deps, *staged.mesh_sources, *staged_meshes}
    stamped = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(dep_files, key=str) if p.exists()]
    meta = {"deps": stamped, "missing": staged.unresolved, "plan": plan.to_json(), "report": report.to_dict(full=True)}
    (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    _prune(cache)


def _read_urdf(src: Path, opts: ImportOptions, report: DoctorReport) -> tuple[str, list[Path]]:
    try:
        text = src.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ValueError(f"could not read {src}: {exc}") from exc
    if src.suffix.lower() == ".xacro" or xacro.needs_expansion(text):
        expansion = xacro.expand(src, opts.xacro_args)
        expanded, deps = expansion.text, expansion.files
        if expansion.undefined_macros:
            report.unfixed(
                "xacro",
                "the file calls xacro macros that are never defined in it or in what it includes",
                "left those parts out; if they matter, include the xacro files that define them",
                expansion.undefined_macros,
            )
        report.fixed(
            "xacro",
            "the file is a xacro template (macros and ${...} formulas), not a finished URDF",
            "expanded it with robotforge's built-in xacro reader",
        )
        return expanded, deps
    return text, []


def _stats(staged: StagedUrdf, plan: RigPlan) -> dict[str, Any]:
    movable = [j for j in staged.joints if j.type in ("revolute", "continuous", "prismatic")]
    return {
        "links": len(staged.links),
        "moving_joints": len(movable),
        "motors": len(plan.actuators),
        "mass_kg": round(sum(staged.link_mass.values()), 3),
        "base": plan.base,
        "visual_triangles": staged.visual_triangles,
        "collision_triangles": staged.collision_triangles,
        "stiffness_scale": plan.gain_scale,
    }


def _prune(cache: Path) -> None:
    """Keep the newest CACHE_LIMIT imports, and the converted meshes under MESH_CACHE_BYTES (oldest go first)."""
    entries = [m.parent for m in cache.glob("*/meta.json")]
    for old in sorted(entries, key=lambda d: (d / "meta.json").stat().st_mtime)[:-CACHE_LIMIT]:
        for f in old.iterdir():
            f.unlink(missing_ok=True)
        old.rmdir()
    files = sorted((f for f in (cache / "meshes").glob("*") if f.is_file()), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= MESH_CACHE_BYTES:
            break
        total -= f.stat().st_size
        f.unlink(missing_ok=True)
