"""Visual and collision shapes of a URDF: find and convert meshes, drop broken shapes, report what was done."""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robotforge.design.importing import collision, inertia
from robotforge.design.importing.inertia import Shape
from robotforge.design.importing.meshio import MM_LIMIT, EmptyMeshError, MeshAsset, MeshError, prepare_mesh
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MeshResolver
from robotforge.design.importing.xmlnum import floats

HEAVY_COLLISION_FACES = 5000


class GeometryContext:
    """State shared while walking every <visual> and <collision> element."""

    def __init__(self, resolver: MeshResolver, work_dir: Path, decompose: bool, colored_materials: set[str]):
        self.resolver = resolver
        self.mesh_dir = work_dir / "meshes"
        self.decompose = decompose
        self.colored_materials = colored_materials
        self.assets: dict[Path, MeshAsset | MeshError] = {}
        self.guessed: list[str] = []
        self.unresolved: list[str] = []
        self.empty: set[str] = set()
        self.unresolved_uris: list[str] = []
        self.unreadable: list[str] = []
        self.unreadable_reasons: set[str] = set()
        self.converted: dict[str, set[str]] = {}
        self.repairs: dict[str, set[str]] = {}
        self.units: list[str] = []
        self.dropped: list[str] = []
        self.promoted = 0
        self.heavy: dict[str, int] = {}
        self.split: dict[str, int] = {}
        self.split_failed: list[str] = []
        self.visual_faces = 0
        self.collision_faces = 0
        self.counter = 0

    def asset(self, source: Path) -> MeshAsset | MeshError:
        if source not in self.assets:
            try:
                self.assets[source] = prepare_mesh(source, self.mesh_dir)
            except MeshError as exc:
                self.assets[source] = exc
        return self.assets[source]


def prepare_geometry(
    robot: ET.Element,
    links: dict[str, ET.Element],
    resolver: MeshResolver,
    work_dir: Path,
    report: DoctorReport,
    decompose: bool,
) -> tuple[GeometryContext, dict[str, list[Shape]], dict[str, list[Shape]]]:
    colored = {m.get("name", "") for m in robot.findall("material") if m.find("color") is not None}
    ctx = GeometryContext(resolver, work_dir, decompose, colored)
    collision_shapes: dict[str, list[Shape]] = {}
    visual_shapes: dict[str, list[Shape]] = {}
    for name, link in links.items():
        collision_shapes[name] = _process_role(link, "collision", name, ctx)
        visual_shapes[name] = _process_role(link, "visual", name, ctx)

    if not any(collision_shapes.values()) and any(visual_shapes.values()):
        # No link has a collision shape: let the visible shapes collide, or the robot would fall through the floor.
        for name, link in links.items():
            for vis in link.findall("visual"):
                col = copy.deepcopy(vis)
                col.tag = "collision"
                for extra in col.findall("material"):
                    col.remove(extra)
                link.append(col)
            collision_shapes[name] = list(visual_shapes[name])
        ctx.promoted = sum(len(v) for v in visual_shapes.values())
    _report_geometry(ctx, report)
    return ctx, collision_shapes, visual_shapes


def _process_role(link: ET.Element, role: str, link_name: str, ctx: GeometryContext) -> list[Shape]:
    shapes: list[Shape] = []
    for el in list(link.findall(role)):
        geom_el = el.find("geometry")
        child = geom_el[0] if geom_el is not None and len(geom_el) else None
        origin = el.find("origin")
        xyz = np.array(floats(origin.get("xyz") if origin is not None else None, 3))
        rpy = floats(origin.get("rpy") if origin is not None else None, 3)
        placed = {"rotation": inertia.rpy_matrix(*rpy), "position": xyz}
        shape: Shape | None = None
        if child is None:
            pass
        elif child.tag == "box":
            size = tuple(abs(v) for v in floats(child.get("size"), 3))
            shape = Shape("box", size, **placed) if min(size) > 1e-9 else None
        elif child.tag == "cylinder":
            r, h = (abs(v) for v in (*floats(child.get("radius"), 1), *floats(child.get("length"), 1)))
            shape = Shape("cylinder", (r, h), **placed) if r > 1e-9 and h > 1e-9 else None
        elif child.tag == "sphere":
            r = abs(floats(child.get("radius"), 1)[0])
            shape = Shape("sphere", (r,), **placed) if r > 1e-9 else None
        elif child.tag == "mesh":
            shape = _process_mesh(link, el, child, role, placed, ctx)
            if shape is not None:
                shapes.append(shape)
            continue
        if shape is None:
            link.remove(el)
            ctx.dropped.append(f"{link_name} ({role}{' ' + child.tag if child is not None else ''})")
            continue
        shapes.append(shape)
    return shapes


def _process_mesh(
    link: ET.Element, el: ET.Element, mesh_el: ET.Element, role: str, placed: dict, ctx: GeometryContext
) -> Shape | None:
    uri = mesh_el.get("filename") or ""
    resolution = ctx.resolver.resolve(uri)
    if resolution.path is None:
        ctx.unresolved_uris.append(uri)
        ctx.unresolved.append(f"{uri or '(empty)'}: {resolution.note}" if resolution.note else uri)
        link.remove(el)
        return None
    if resolution.guessed:
        ctx.guessed.append(resolution.note)
    asset = ctx.asset(resolution.path)
    if isinstance(asset, EmptyMeshError):
        ctx.empty.add(resolution.path.name)
        link.remove(el)
        return None
    if isinstance(asset, MeshError):
        ctx.unreadable.append(f"{resolution.path.name}: {asset}")
        ctx.unreadable_reasons.add(str(asset))
        link.remove(el)
        return None

    scale = np.array(_scale(mesh_el.get("scale")))
    if float(np.max(asset.mesh.extents * np.abs(scale))) > MM_LIMIT:
        scale = scale * 0.001
        ctx.units.append(f"{resolution.path.name} ({float(np.max(asset.mesh.extents)):.0f} units wide)")
    for note in asset.notes:
        bucket = ctx.converted if note.startswith("converted from") else ctx.repairs
        bucket.setdefault(note, set()).add(resolution.path.name)

    scale_text = " ".join(f"{v:.9g}" for v in scale)
    if role == "visual":
        ctx.visual_faces += asset.out_faces
        elements = [_visual_copy(el, piece.path, piece.rgba, scale_text, ctx) for piece in asset.pieces]
    else:
        ctx.collision_faces += asset.out_faces
        paths = [asset.merged]
        if ctx.decompose and asset.out_faces >= collision.DECOMPOSE_MIN_FACES:
            paths = _decompose(asset, ctx) or paths
        elif asset.out_faces > HEAVY_COLLISION_FACES:
            ctx.heavy[resolution.path.name] = asset.out_faces
        elements = []
        for path in paths:
            copy_el = copy.deepcopy(el)
            _point_mesh(copy_el, path, scale_text)
            elements.append(copy_el)
    index = list(link).index(el)
    link.remove(el)
    for offset, new in enumerate(elements):
        link.insert(index + offset, new)
    return Shape("mesh", (), tuple(float(v) for v in scale), asset, **placed)  # type: ignore[arg-type]


def _decompose(asset: MeshAsset, ctx: GeometryContext) -> list[Path] | None:
    try:
        paths = collision.decompose(asset, ctx.mesh_dir)
    except collision.DecomposeError as exc:
        ctx.split_failed.append(f"{asset.source.name}: {exc}")
        return None
    ctx.split[asset.source.name] = len(paths)
    return paths


def _scale(text: str | None) -> tuple[float, float, float]:
    vals = floats(text, 3, default=1.0) if text and len(text.split()) == 3 else None
    if vals is None and text:
        one = floats(text, 1, default=1.0)[0]
        vals = [one, one, one]
    if vals is None or not all(math.isfinite(v) and abs(v) > 1e-9 for v in vals):
        return (1.0, 1.0, 1.0)
    return (vals[0], vals[1], vals[2])


def _point_mesh(el: ET.Element, path: Path, scale_text: str) -> None:
    mesh = el.find("geometry/mesh")
    assert mesh is not None
    mesh.set("filename", str(path))
    mesh.set("scale", scale_text)


def _visual_copy(el: ET.Element, path: Path, rgba, scale_text: str, ctx: GeometryContext) -> ET.Element:
    new = copy.deepcopy(el)
    _point_mesh(new, path, scale_text)
    material = new.find("material")
    has_color = material is not None and (
        material.find("color") is not None or material.get("name") in ctx.colored_materials
    )
    if rgba is not None and not has_color:
        if material is not None:
            new.remove(material)
        ctx.counter += 1
        mat = ET.SubElement(new, "material", {"name": f"rf_color_{ctx.counter}"})
        ET.SubElement(mat, "color", {"rgba": " ".join(f"{c:.4g}" for c in rgba)})
    return new


def _report_geometry(ctx: GeometryContext, report: DoctorReport) -> None:
    if ctx.guessed:
        report.fixed(
            "meshes",
            "some mesh paths pointed at folders that are not on this computer",
            "found the same files by name nearby and used those",
            sorted(set(ctx.guessed)),
        )
    if ctx.unresolved:
        report.unfixed(
            "meshes",
            f"{len(set(ctx.unresolved))} mesh file(s) could not be found",
            "left those shapes out; the rest of the robot still loads",
            sorted(set(ctx.unresolved)),
        )
    if ctx.empty:
        report.info("meshes", "some mesh files are empty placeholders (no triangles); skipped them", sorted(ctx.empty))
    if ctx.unreadable:
        pycollada = any("pycollada" in r for r in ctx.unreadable_reasons)
        action = "left those shapes out; the rest of the robot still loads"
        problem = f"{len(set(ctx.unreadable))} mesh file(s) could not be read"
        if pycollada:
            problem += " (DAE meshes need pycollada; ask the user to approve installing it)"
        report.unfixed("meshes", problem, action, sorted(set(ctx.unreadable)))
    for note, names in sorted(ctx.converted.items()):
        fmt = note.removeprefix("converted from ").removesuffix(" to STL")
        report.fixed(
            "meshes",
            f"{len(names)} mesh file(s) are {fmt} files, which MuJoCo cannot read directly",
            "converted them to STL (stored in robotforge's import cache); colors are kept where the file has them",
            sorted(names),
        )
    for note, names in sorted(ctx.repairs.items()):
        report.fixed("meshes", f"{len(names)} mesh file(s) would not load as they were", note, sorted(names))
    if ctx.units:
        report.fixed(
            "meshes",
            "some meshes are huge, so they are probably in millimeters",
            "scaled them by 0.001",
            ctx.units,
        )
    if ctx.dropped:
        report.fixed(
            "geometry", "some shapes had zero size or an unsupported type", "removed them", sorted(set(ctx.dropped))
        )
    if ctx.promoted:
        report.fixed(
            "collision",
            "the file has no collision shapes, only visible ones",
            "the visible shapes now collide too, so the robot does not fall through the floor",
        )
    if ctx.heavy:
        report.fixed(
            "collision",
            f"{len(ctx.heavy)} collision mesh(es) have thousands of triangles, which slows the physics",
            "limited their collision outline to a simple convex shape; "
            "pass decompose=true to split them into accurate convex pieces instead",
            [f"{n}: {f:,} triangles" for n, f in sorted(ctx.heavy.items())],
        )
    if ctx.split:
        report.fixed(
            "collision",
            "some collision meshes are concave, and MuJoCo would treat them as solid blobs",
            "split them into convex pieces (CoACD)",
            [f"{n}: {k} pieces" for n, k in sorted(ctx.split.items())],
        )
    if ctx.split_failed:
        report.unfixed(
            "collision",
            "could not split some collision meshes into convex pieces",
            "kept them as single shapes",
            ctx.split_failed,
        )
