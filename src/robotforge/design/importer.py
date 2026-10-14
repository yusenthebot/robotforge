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
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from robotforge.design.importing import collision, rig, xacro
from robotforge.design.importing.meshio import TOOL_VERSION, MeshAsset, MeshError, prepare_mesh, solid_properties
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MESH_SUFFIXES, MeshResolver
from robotforge.design.importing.urdf import StagedUrdf, stage_urdf

log = logging.getLogger(__name__)

IMPORTER_VERSION = "1"
URDF_SUFFIXES = frozenset({".urdf", ".xacro"})
IMPORT_SUFFIXES = URDF_SUFFIXES | MESH_SUFFIXES
CACHE_LIMIT = 40
_MM_LIMIT = 20.0
OBJECT_DENSITY = 500.0  # kg/m^3 for loose mesh objects without a given mass
_OBJECT_COLOR = (0.72, 0.74, 0.78, 1.0)


class ImportOptions(BaseModel):
    """Optional knobs, passed as `params` of a robot entry or as keyword arguments."""

    model_config = ConfigDict(extra="forbid")

    base: Literal["auto", "fixed", "floating"] = "auto"  # URDF: arm-like or free-standing; mesh: fixed or dropped
    decompose: bool = False  # split concave collision meshes into convex pieces (slow, cached)
    scale: float | None = None  # plain meshes: multiply the size (default: guess meters vs millimeters)
    mass: float | None = None  # plain meshes that float: kg (default: from volume at 500 kg/m^3)
    xacro_args: dict[str, str] | None = None  # values for <xacro:arg> names

    @classmethod
    def parse(cls, params: dict[str, Any] | ImportOptions | None) -> ImportOptions:
        if isinstance(params, ImportOptions):
            return params
        try:
            return cls.model_validate(params or {})
        except ValidationError as exc:
            parts = [f"{'.'.join(str(x) for x in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors()]
            allowed = ", ".join(cls.model_fields)
            raise ValueError(f"bad import option ({'; '.join(parts)}). Import options are: {allowed}") from exc


@dataclass
class ImportResult:
    """A spec ready to attach to a world, how to place it, and the doctor's notes."""

    spec: mujoco.MjSpec
    report: DoctorReport
    kind: Literal["robot", "object"]
    floating: bool
    spawn_height: float
    home_qpos: dict[str, float]
    home_ctrl: dict[str, float]


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
    if suffix in URDF_SUFFIXES:
        return _import_urdf(src, opts, cache)
    if suffix in MESH_SUFFIXES:
        return _import_mesh(src, opts, cache)
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
    spec = rig.build_spec(entry / "staged.urdf", plan, name=re.sub(r"\W+", "_", src.stem))
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
    versions = f"{IMPORTER_VERSION}|{TOOL_VERSION}|{rig.PLAN_VERSION}"
    raw = f"{versions}|{src}|{st.st_size}|{st.st_mtime_ns}|{opts.model_dump_json()}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _load_cached(entry: Path, src: Path) -> tuple[rig.RigPlan, DoctorReport] | None:
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
        plan = rig.RigPlan.from_json(info["plan"])
        report = DoctorReport.from_dict(info["report"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (entry / "staged.urdf").is_file():
        return None
    meta.touch()  # recently used entries survive pruning
    return plan, report


def _stage_and_plan(src: Path, opts: ImportOptions, cache: Path, entry: Path) -> None:
    report = DoctorReport(source=str(src))
    text, deps = _read_urdf(src, opts, report)
    try:
        robot = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(
            f"{src.name} is not valid XML ({exc}). Look for a missing closing tag or stray character"
        ) from exc

    resolver = MeshResolver(src)
    staged = stage_urdf(robot, resolver, cache, report, opts.decompose)
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "meta.json").unlink(missing_ok=True)  # an entry without meta.json counts as incomplete
    urdf_path = entry / "staged.urdf"
    urdf_path.write_text(staged.text, encoding="utf-8")

    plan = rig.make_plan(staged, urdf_path, report, opts.base)
    report.stats.update(_stats(staged, plan))

    dep_files = {src, *deps, *staged.mesh_sources}
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
        expanded, deps = xacro.expand(src, opts.xacro_args)
        report.fixed(
            "xacro",
            "the file is a xacro template (macros and ${...} formulas), not a finished URDF",
            "expanded it with robotforge's built-in xacro reader",
        )
        return expanded, deps
    return text, []


def _stats(staged: StagedUrdf, plan: rig.RigPlan) -> dict[str, Any]:
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
    entries = [m.parent for m in cache.glob("*/meta.json")]
    for old in sorted(entries, key=lambda d: (d / "meta.json").stat().st_mtime)[:-CACHE_LIMIT]:
        for f in old.iterdir():
            f.unlink(missing_ok=True)
        old.rmdir()


# ---- plain meshes ----------------------------------------------------------------------------


def _import_mesh(src: Path, opts: ImportOptions, cache: Path) -> ImportResult:
    """A single mesh file as a free-standing object (fixed in place unless base='floating')."""
    report = DoctorReport(source=str(src))
    try:
        asset = prepare_mesh(src, cache / "meshes")
    except MeshError as exc:
        raise ValueError(str(exc)) from exc
    for note in asset.notes:
        report.fixed("mesh", f"{src.name}: the file would not load as it was", note)

    scale = opts.scale
    extent = float(np.max(asset.mesh.extents))
    if scale is None and extent > _MM_LIMIT:
        scale = 0.001
        report.fixed(
            "units",
            f"the mesh is {extent:.0f} units wide, so it is probably in millimeters",
            "scaled it by 0.001; pass scale=1 to keep the original size",
        )
    scale = 1.0 if scale is None else scale
    if scale <= 0:
        raise ValueError("scale must be positive")
    size = asset.mesh.extents * scale
    if float(np.max(size)) < 0.005:
        report.unfixed(
            "units",
            f"the mesh is only {float(np.max(size)) * 1000:.1f} mm wide",
            "left it as is; pass scale=1000 if it was meant to be in meters",
        )

    floating = opts.base == "floating"
    spec = mujoco.MjSpec()
    spec.modelname = re.sub(r"\W+", "_", src.stem)
    body = spec.worldbody.add_body(name="object")
    props = solid_properties(asset.mesh, np.full(3, scale))
    mass = opts.mass if opts.mass is not None else props.volume * OBJECT_DENSITY
    if floating:
        _set_inertial(body, mass, props)
        body.add_freejoint(name="root")
        if opts.mass is None:
            report.info(
                "mass", f"mass is {mass:.3g} kg (volume x {OBJECT_DENSITY:.0f} kg/m^3); pass mass=... to change it"
            )
    elif opts.mass is not None:
        report.info("mass", "mass is ignored because the object is fixed in place")
    _add_mesh_geoms(spec, body, asset, scale, opts, cache, report)

    low = float(asset.bounds[0][2]) * scale
    spawn = -low + (0.003 if floating else 0.0)
    report.info(
        "placement",
        f"the object is {size[0] * 100:.1f} x {size[1] * 100:.1f} x {size[2] * 100:.1f} cm and rests on the floor"
        + ("" if floating else "; it is fixed in place (pass base=floating to let it fall and be pushed)"),
    )
    report.stats.update({"triangles": asset.out_faces, "size_m": [round(float(v), 4) for v in size]})
    return ImportResult(spec, report, "object", floating, spawn, {}, {})


def _set_inertial(body: mujoco.MjsBody, mass: float, props) -> None:
    body.mass = mass
    body.ipos = np.asarray(props.center)
    unit = props.inertia / max(props.volume, 1e-12) * mass
    body.fullinertia = [unit[0, 0], unit[1, 1], unit[2, 2], unit[0, 1], unit[0, 2], unit[1, 2]]
    body.explicitinertial = True


def _add_mesh_geoms(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    asset: MeshAsset,
    scale: float,
    opts: ImportOptions,
    cache: Path,
    report: DoctorReport,
) -> None:
    G = mujoco.mjtGeom
    sc = [scale] * 3
    parts: list[Path] | None = None
    if opts.decompose and asset.out_faces >= collision.DECOMPOSE_MIN_FACES:
        try:
            parts = collision.decompose(asset, cache / "meshes")
            report.fixed(
                "collision",
                "concave shapes would collide as a solid blob",
                f"split the collision shape into {len(parts)} convex pieces (CoACD)",
            )
        except collision.DecomposeError as exc:
            report.unfixed(
                "collision", f"could not split the shape into convex pieces ({exc})", "using one convex outline"
            )
    elif asset.out_faces > 5000:
        report.info(
            "collision",
            f"the mesh has {asset.out_faces:,} triangles; it collides as one convex outline. "
            "Pass decompose=true to keep concave details",
        )

    hull_cap = 64
    collide_with_visual = parts is None and len(asset.pieces) == 1
    for i, piece in enumerate(asset.pieces):
        mesh = spec.add_mesh(name=f"visual{i}", file=str(piece.path), scale=sc)
        if collide_with_visual:
            mesh.maxhullvert = hull_cap
        body.add_geom(
            type=G.mjGEOM_MESH,
            meshname=mesh.name,
            group=rig.VISUAL_GROUP,
            rgba=list(piece.rgba or _OBJECT_COLOR),
            **({} if collide_with_visual else {"contype": 0, "conaffinity": 0}),
        )
    if collide_with_visual:
        return
    hull_paths = parts if parts is not None else [asset.merged]
    for i, path in enumerate(hull_paths):
        mesh = spec.add_mesh(name=f"collision{i}", file=str(path), scale=sc)
        mesh.maxhullvert = hull_cap
        body.add_geom(type=G.mjGEOM_MESH, meshname=mesh.name, group=rig.COLLISION_GROUP, density=0)
