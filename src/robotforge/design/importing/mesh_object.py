"""A single mesh file (STL, OBJ, DAE, GLB, ...) imported as an object: fixed in place or free to fall and be pushed."""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from robotforge.design.importing import collision
from robotforge.design.importing.meshio import MM_LIMIT, MeshAsset, MeshError, prepare_mesh, solid_properties
from robotforge.design.importing.options import ImportOptions, ImportResult
from robotforge.design.importing.plan import COLLISION_GROUP, VISUAL_GROUP
from robotforge.design.importing.report import DoctorReport

OBJECT_DENSITY = 500.0  # kg/m^3 for loose mesh objects without a given mass
_OBJECT_COLOR = (0.72, 0.74, 0.78, 1.0)


def import_mesh_object(src: Path, opts: ImportOptions, cache: Path) -> ImportResult:
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
    if scale is None and extent > MM_LIMIT:
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
            group=VISUAL_GROUP,
            rgba=list(piece.rgba or _OBJECT_COLOR),
            **({} if collide_with_visual else {"contype": 0, "conaffinity": 0}),
        )
    if collide_with_visual:
        return
    hull_paths = parts if parts is not None else [asset.merged]
    for i, path in enumerate(hull_paths):
        mesh = spec.add_mesh(name=f"collision{i}", file=str(path), scale=sc)
        mesh.maxhullvert = hull_cap
        body.add_geom(type=G.mjGEOM_MESH, meshname=mesh.name, group=COLLISION_GROUP, density=0)
