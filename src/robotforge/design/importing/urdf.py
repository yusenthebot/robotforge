"""Read a URDF, repair what would trip up MuJoCo, and write a clean copy ("staged" URDF).

The staged copy has absolute mesh paths in formats MuJoCo reads, sane joints, valid mass and
inertia on every link, and compiler options that keep visuals and every link as its own body.
MuJoCo then loads it with its own URDF compiler, so we never re-implement kinematics.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotforge.design.importing import collision, inertia
from robotforge.design.importing.inertia import Inertial, Shape
from robotforge.design.importing.meshio import MeshAsset, MeshError, prepare_mesh
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MeshResolver

_UNBOUNDED = 1e9
_MM_LIMIT = 20.0  # a single part bigger than this many meters is almost certainly in millimeters
HEAVY_COLLISION_FACES = 5000
_KEEP_TAGS = {"link", "joint", "material"}
_JOINT_TYPES = {"revolute", "continuous", "prismatic", "fixed", "floating"}


@dataclass
class JointInfo:
    name: str
    type: str  # revolute | continuous | prismatic | fixed | floating
    parent: str
    child: str
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    lower: float | None = None  # None means no limit
    upper: float | None = None
    effort: float | None = None  # N or N m; None when the file gives none
    velocity: float | None = None
    mimic_of: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    damping: float = 0.0
    friction: float = 0.0  # dry friction loss, N m or N


@dataclass
class StagedUrdf:
    text: str
    root: str
    roots: list[str]
    links: list[str]
    joints: list[JointInfo]
    link_shapes: dict[str, list[Shape]]  # collision shapes, or visual shapes where a link has no collision
    link_mass: dict[str, float]
    link_friction: dict[str, float] = field(default_factory=dict)  # Gazebo mu1 hints, link -> sliding friction
    mesh_sources: list[Path] = field(default_factory=list)  # original mesh files that were read
    unresolved: list[str] = field(default_factory=list)  # mesh URIs that were not found
    visual_triangles: int = 0
    collision_triangles: int = 0

    def joint_of(self, link: str) -> JointInfo | None:
        return next((j for j in self.joints if j.child == link), None)


def stage_urdf(
    robot: ET.Element, resolver: MeshResolver, work_dir: Path, report: DoctorReport, decompose: bool
) -> StagedUrdf:
    """Repair `robot` (a parsed <robot> element, modified in place) and return the staged description."""
    if robot.tag != "robot":
        raise ValueError(f"this is not a URDF: the top element is <{robot.tag}>, expected <robot>")
    friction = _gazebo_friction(robot)
    _strip_noise(robot)
    _fix_origins(robot, report)
    links = {el.get("name", ""): el for el in robot.findall("link")}
    if not links:
        raise ValueError("this URDF has no <link> elements, so there is nothing to simulate")
    if "" in links:
        raise ValueError("a <link> in this URDF has no name")
    if len(links) != len(robot.findall("link")):
        raise ValueError("two links in this URDF share a name; every link needs its own name")

    joints = _read_joints(robot, links, report)
    roots = _check_tree(links, joints, report)
    _fix_materials(robot, report)
    ctx, shapes, visual_shapes = _prepare_geometry(robot, links, resolver, work_dir, report, decompose)
    link_shapes = {n: (shapes[n] or visual_shapes[n]) for n in links}
    masses = _fix_inertials(links, joints, roots, link_shapes, report)
    _set_compiler_options(robot)

    return StagedUrdf(
        text=ET.tostring(robot, encoding="unicode"),
        root=_main_root(roots, joints),
        roots=roots,
        links=list(links),
        joints=joints,
        link_shapes=link_shapes,
        link_mass=masses,
        link_friction={k: v for k, v in friction.items() if k in links},
        mesh_sources=[p for p, a in ctx.assets.items() if isinstance(a, MeshAsset)],
        unresolved=sorted(set(ctx.unresolved_uris)),
        visual_triangles=ctx.visual_faces,
        collision_triangles=ctx.collision_faces,
    )


# ---- cleanup ---------------------------------------------------------------------------------


def _strip_noise(robot: ET.Element) -> None:
    """Drop Gazebo, ros_control and MuJoCo-hint blocks; only links, joints and materials matter."""
    for child in list(robot):
        if child.tag not in _KEEP_TAGS:
            robot.remove(child)


def _fix_origins(robot: ET.Element, report: DoctorReport) -> None:
    """Make every <origin> three plain finite numbers per attribute (NaN, missing or short values become 0)."""
    broken = 0
    for origin in robot.iter("origin"):
        for attr in ("xyz", "rpy"):
            text = origin.get(attr)
            if text is None:
                continue
            try:
                raw = [float(p) for p in text.split()]
            except ValueError:
                raw = []
            fixed = _floats(text, 3)
            if len(raw) != 3 or not all(math.isfinite(v) for v in raw):
                broken += 1
            origin.set(attr, " ".join(f"{v:.9g}" for v in fixed))
    if broken:
        report.fixed(
            "geometry", f"{broken} position or rotation value(s) were missing or not numbers", "replaced them with 0"
        )


def _gazebo_friction(robot: ET.Element) -> dict[str, float]:
    """Friction the author asked for in simulator-only tags: Gazebo's <gazebo reference="link"><mu1> and
    PyBullet's <link><contact><lateral_friction value=.../>."""
    out: dict[str, float] = {}

    def keep(link: str | None, text: str | None) -> None:
        try:
            if link and text:
                out[link] = min(max(float(text), 0.01), 5.0)
        except ValueError:
            pass

    for el in robot.findall("gazebo"):
        mu = el.find("mu1")
        keep(el.get("reference"), mu.text if mu is not None else None)
    for link in robot.findall("link"):
        lateral = link.find("contact/lateral_friction")
        keep(link.get("name"), lateral.get("value") if lateral is not None else None)
    return out


def _fix_materials(robot: ET.Element, report: DoctorReport) -> None:
    broken = []
    for color in robot.iter("color"):
        parts = (color.get("rgba") or "").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            vals = []
        if len(vals) == 3:
            vals.append(1.0)
        if len(vals) != 4:
            vals = [0.5, 0.5, 0.5, 1.0]
            broken.append("a color")
        color.set("rgba", " ".join(f"{min(max(v, 0.0), 1.0):.4g}" for v in vals))
    for material in robot.iter("material"):
        for texture in material.findall("texture"):  # MuJoCo's URDF reader ignores textures; keep the file simple
            material.remove(texture)
    if broken:
        report.fixed("materials", "some colors in the file were malformed", "replaced them with gray", broken[:1])


# ---- joints ----------------------------------------------------------------------------------


def _floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    try:
        vals = [float(p) for p in (text or "").split()]
    except ValueError:
        vals = []
    vals = [v if math.isfinite(v) else default for v in vals]
    return (vals + [default] * count)[:count]


def _read_joints(robot: ET.Element, links: dict[str, ET.Element], report: DoctorReport) -> list[JointInfo]:
    joints: list[JointInfo] = []
    seen: set[str] = set()
    renamed, retyped, relimited, axes, missing_refs = [], [], [], [], []
    for el in robot.findall("joint"):
        name = el.get("name") or f"joint_{len(joints)}"
        if name in seen:
            new = f"{name}_{len(joints)}"
            renamed.append(name)
            name = new
        seen.add(name)
        el.set("name", name)
        parent = (el.find("parent").get("link") if el.find("parent") is not None else "") or ""
        child = (el.find("child").get("link") if el.find("child") is not None else "") or ""
        if parent not in links or child not in links:
            missing_refs.append(f"{name}: {parent!r} -> {child!r}")
            robot.remove(el)
            continue
        jtype = el.get("type", "fixed")
        if jtype not in _JOINT_TYPES:
            retyped.append(f"{name} ({jtype})")
            jtype = "fixed"
            el.set("type", "fixed")
        info = JointInfo(name=name, type=jtype, parent=parent, child=child)
        if jtype != "fixed":
            _sanitize_motion(el, info, axes, relimited)
        if jtype == "fixed" or info.type == "fixed":
            info.type = "fixed"
            el.set("type", "fixed")
        mimic = el.find("mimic")
        if mimic is not None and mimic.get("joint"):
            info.mimic_of = mimic.get("joint")
            info.mimic_multiplier = _floats(mimic.get("multiplier"), 1, default=1.0)[0]
            info.mimic_offset = _floats(mimic.get("offset"), 1)[0]
        dyn = el.find("dynamics")
        if dyn is not None:
            info.damping = max(0.0, _floats(dyn.get("damping"), 1)[0])
            info.friction = max(0.0, _floats(dyn.get("friction"), 1)[0])
        joints.append(info)

    if missing_refs:
        raise ValueError(
            "these joints point at links that do not exist, so the robot cannot be assembled: "
            + "; ".join(missing_refs[:5])
        )
    if renamed:
        report.fixed("joints", "some joints shared a name", "gave the repeats unique names", renamed)
    if retyped:
        report.fixed(
            "joints",
            "some joints use a type robotforge cannot simulate (planar or unknown)",
            "made them fixed",
            retyped,
        )
    if axes:
        report.fixed("joints", "some joints had no usable rotation axis", "used the x axis", axes)
    if relimited:
        report.fixed(
            "joints", "some joint limits were missing or inconsistent", "repaired them (details below)", relimited
        )
    for j in joints:  # mimic leaders must exist, otherwise drop the link
        if j.mimic_of and j.mimic_of not in seen:
            j.mimic_of = None
            m = next(el for el in robot.findall("joint") if el.get("name") == j.name).find("mimic")
            if m is not None:
                next(el for el in robot.findall("joint") if el.get("name") == j.name).remove(m)
    return joints


def _sanitize_motion(el: ET.Element, info: JointInfo, axes: list[str], notes: list[str]) -> None:
    axis_el = el.find("axis")
    axis = np.array(_floats(axis_el.get("xyz") if axis_el is not None else "1 0 0", 3))
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0])
        axes.append(info.name)
    axis = axis / np.linalg.norm(axis)
    if axis_el is None:
        axis_el = ET.SubElement(el, "axis")
    axis_el.set("xyz", " ".join(f"{v:.9g}" for v in axis))
    info.axis = (float(axis[0]), float(axis[1]), float(axis[2]))

    limit = el.find("limit")
    if limit is None:
        limit = ET.SubElement(el, "limit")
    effort = _positive(limit.get("effort"))
    velocity = _positive(limit.get("velocity"))
    info.effort, info.velocity = effort, velocity
    has_lo, has_hi = limit.get("lower") is not None, limit.get("upper") is not None

    if info.type == "continuous":
        for attr in ("lower", "upper"):
            limit.attrib.pop(attr, None)
        return
    lo = _floats(limit.get("lower"), 1)[0] if has_lo else None
    hi = _floats(limit.get("upper"), 1)[0] if has_hi else None
    unbounded = (lo is None and hi is None) or (
        lo is not None and hi is not None and lo <= -_UNBOUNDED and hi >= _UNBOUNDED
    )
    if not unbounded and ((lo is not None and not math.isfinite(lo)) or (hi is not None and not math.isfinite(hi))):
        unbounded = True
    if unbounded:
        for attr in ("lower", "upper"):
            limit.attrib.pop(attr, None)
        if info.type == "revolute":
            info.type = "continuous"
            el.set("type", "continuous")
            notes.append(f"{info.name}: revolute joint without limits is now continuous")
        else:
            notes.append(f"{info.name}: sliding joint without limits")
        return
    lo, hi = lo if lo is not None else 0.0, hi if hi is not None else 0.0
    if lo > hi:
        lo, hi = hi, lo
        notes.append(f"{info.name}: lower limit was above the upper limit; swapped them")
    if hi - lo < 1e-9:
        info.type = "fixed"
        el.set("type", "fixed")
        for attr in ("lower", "upper"):
            limit.attrib.pop(attr, None)
        notes.append(f"{info.name}: range is zero wide, so the joint cannot move; made it fixed")
        return
    limit.set("lower", f"{lo:.9g}")
    limit.set("upper", f"{hi:.9g}")
    info.lower, info.upper = lo, hi


def _positive(text: str | None) -> float | None:
    if text is None:
        return None
    val = _floats(text, 1)[0]
    return val if math.isfinite(val) and 0 < val < _UNBOUNDED else None


def _check_tree(links: dict[str, ET.Element], joints: list[JointInfo], report: DoctorReport) -> list[str]:
    parents: dict[str, list[str]] = {}
    for j in joints:
        parents.setdefault(j.child, []).append(j.name)
    multi = {c: names for c, names in parents.items() if len(names) > 1}
    if multi:
        child, names = next(iter(multi.items()))
        raise ValueError(
            f"link '{child}' hangs off {len(names)} joints ({', '.join(names)}), which makes a closed loop. "
            "URDF robots must be trees; remove one of those joints (or model the loop in MJCF)."
        )
    roots = [n for n in links if n not in parents]
    if not roots:
        raise ValueError("every link in this URDF has a parent, so the joints form a cycle; URDF robots must be trees")
    reached: set[str] = set()
    stack = list(roots)
    children: dict[str, list[str]] = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j.child)
    while stack:
        cur = stack.pop()
        if cur in reached:
            continue
        reached.add(cur)
        stack.extend(children.get(cur, []))
    if len(reached) != len(links):
        lost = sorted(set(links) - reached)
        raise ValueError(f"these links form a cycle and are not connected to the rest: {', '.join(lost[:6])}")
    if len(roots) > 1:
        report.unfixed(
            "structure",
            f"the URDF has {len(roots)} separate pieces that are not joined by any joint",
            "the biggest piece is the robot; the others stay fixed in place",
            roots,
        )
    return roots


def _main_root(roots: list[str], joints: list[JointInfo]) -> str:
    if len(roots) == 1:
        return roots[0]
    children: dict[str, list[str]] = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j.child)

    def size(link: str) -> int:
        return 1 + sum(size(c) for c in children.get(link, []))

    return max(roots, key=size)


# ---- geometry --------------------------------------------------------------------------------


class _Geometry:
    """State shared while walking every <visual> and <collision> element."""

    def __init__(self, resolver: MeshResolver, work_dir: Path, decompose: bool, colored_materials: set[str]):
        self.resolver = resolver
        self.mesh_dir = work_dir / "meshes"
        self.decompose = decompose
        self.colored_materials = colored_materials
        self.assets: dict[Path, MeshAsset | MeshError] = {}
        self.guessed: list[str] = []
        self.unresolved: list[str] = []
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


def _prepare_geometry(
    robot: ET.Element,
    links: dict[str, ET.Element],
    resolver: MeshResolver,
    work_dir: Path,
    report: DoctorReport,
    decompose: bool,
) -> tuple[_Geometry, dict[str, list[Shape]], dict[str, list[Shape]]]:
    colored = {m.get("name", "") for m in robot.findall("material") if m.find("color") is not None}
    ctx = _Geometry(resolver, work_dir, decompose, colored)
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


def _process_role(link: ET.Element, role: str, link_name: str, ctx: _Geometry) -> list[Shape]:
    shapes: list[Shape] = []
    for el in list(link.findall(role)):
        geom_el = el.find("geometry")
        child = geom_el[0] if geom_el is not None and len(geom_el) else None
        origin = el.find("origin")
        xyz = np.array(_floats(origin.get("xyz") if origin is not None else None, 3))
        rpy = _floats(origin.get("rpy") if origin is not None else None, 3)
        placed = {"rotation": inertia.rpy_matrix(*rpy), "position": xyz}
        shape: Shape | None = None
        if child is None:
            pass
        elif child.tag == "box":
            size = tuple(abs(v) for v in _floats(child.get("size"), 3))
            shape = Shape("box", size, **placed) if min(size) > 1e-9 else None
        elif child.tag == "cylinder":
            r, h = (abs(v) for v in (*_floats(child.get("radius"), 1), *_floats(child.get("length"), 1)))
            shape = Shape("cylinder", (r, h), **placed) if r > 1e-9 and h > 1e-9 else None
        elif child.tag == "sphere":
            r = abs(_floats(child.get("radius"), 1)[0])
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
    link: ET.Element, el: ET.Element, mesh_el: ET.Element, role: str, placed: dict, ctx: _Geometry
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
    if isinstance(asset, MeshError):
        ctx.unreadable.append(f"{resolution.path.name}: {asset}")
        ctx.unreadable_reasons.add(str(asset))
        link.remove(el)
        return None

    scale = np.array(_scale(mesh_el.get("scale")))
    if float(np.max(asset.mesh.extents * np.abs(scale))) > _MM_LIMIT:
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


def _decompose(asset: MeshAsset, ctx: _Geometry) -> list[Path] | None:
    try:
        paths = collision.decompose(asset, ctx.mesh_dir)
    except collision.DecomposeError as exc:
        ctx.split_failed.append(f"{asset.source.name}: {exc}")
        return None
    ctx.split[asset.source.name] = len(paths)
    return paths


def _scale(text: str | None) -> tuple[float, float, float]:
    vals = _floats(text, 3, default=1.0) if text and len(text.split()) == 3 else None
    if vals is None and text:
        one = _floats(text, 1, default=1.0)[0]
        vals = [one, one, one]
    if vals is None or not all(math.isfinite(v) and abs(v) > 1e-9 for v in vals):
        return (1.0, 1.0, 1.0)
    return (vals[0], vals[1], vals[2])


def _point_mesh(el: ET.Element, path: Path, scale_text: str) -> None:
    mesh = el.find("geometry/mesh")
    assert mesh is not None
    mesh.set("filename", str(path))
    mesh.set("scale", scale_text)


def _visual_copy(el: ET.Element, path: Path, rgba, scale_text: str, ctx: _Geometry) -> ET.Element:
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


def _report_geometry(ctx: _Geometry, report: DoctorReport) -> None:
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


# ---- mass and inertia -----------------------------------------------------------------------


def _parse_inertial(link: ET.Element) -> Inertial | None:
    el = link.find("inertial")
    if el is None:
        return None
    mass_el = el.find("mass")
    mass = _floats(mass_el.get("value"), 1)[0] if mass_el is not None else 0.0
    origin = el.find("origin")
    xyz = np.array(_floats(origin.get("xyz") if origin is not None else None, 3))
    rot = inertia.rpy_matrix(*_floats(origin.get("rpy") if origin is not None else None, 3))
    i = el.find("inertia")
    vals = {k: _floats(i.get(k) if i is not None else None, 1)[0] for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
    tensor = np.array(
        [
            [vals["ixx"], vals["ixy"], vals["ixz"]],
            [vals["ixy"], vals["iyy"], vals["iyz"]],
            [vals["ixz"], vals["iyz"], vals["izz"]],
        ]
    )
    return Inertial(mass, xyz, rot @ tensor @ rot.T)


def _write_inertial(link: ET.Element, value: Inertial) -> None:
    for old in link.findall("inertial"):
        link.remove(old)
    el = ET.Element("inertial")
    ET.SubElement(el, "origin", {"xyz": " ".join(f"{v:.9g}" for v in value.com), "rpy": "0 0 0"})
    ET.SubElement(el, "mass", {"value": f"{value.mass:.9g}"})
    t = (value.tensor + value.tensor.T) / 2
    ET.SubElement(
        el,
        "inertia",
        {
            k: f"{v:.9g}"
            for k, v in zip(
                ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"),
                (t[0, 0], t[0, 1], t[0, 2], t[1, 1], t[1, 2], t[2, 2]),
                strict=True,
            )
        },
    )
    link.insert(0, el)


def _fix_inertials(
    links: dict[str, ET.Element],
    joints: list[JointInfo],
    roots: list[str],
    link_shapes: dict[str, list[Shape]],
    report: DoctorReport,
) -> dict[str, float]:
    parent_joint = {j.child: j for j in joints}
    masses: dict[str, float] = {}
    missing, bad_mass, bad_inertia, nudged, implausible, defaulted = [], [], [], [], [], []
    for name, link in links.items():
        needs = name in roots or parent_joint[name].type != "fixed"
        shapes = link_shapes[name]
        given = _parse_inertial(link)
        result: Inertial | None = given

        if given is None:
            if needs:
                result = inertia.inertial_from_shapes(shapes)
                if result is not None and result.mass >= inertia.TINY_MASS:
                    missing.append(name)
                else:  # no shape, or only a marker-sized one: a placeholder is all the simulator needs
                    result = inertia.default_inertial()
                    if name not in roots:
                        defaulted.append(name)
        else:
            result = _check_given(given, shapes, bad_mass, bad_inertia, nudged, implausible, name, needs, defaulted)

        if result is not None:
            _write_inertial(link, result)
            masses[name] = result.mass
        else:
            for old in link.findall("inertial"):  # a rigid frame with a bogus inertial: let MuJoCo infer it
                link.remove(old)
            masses[name] = 0.0

    if missing:
        report.fixed(
            "inertia", "some links had no mass or inertia", "computed them from the link shapes (1000 kg/m^3)", missing
        )
    if bad_mass:
        report.fixed(
            "inertia",
            "some links had zero, negative or tiny mass",
            "computed mass and inertia from the link shapes",
            bad_mass,
        )
    if bad_inertia:
        report.fixed(
            "inertia",
            "some links had an unusable inertia (zero or not positive)",
            "rebuilt it from the link shapes and kept the mass",
            bad_inertia,
        )
    if nudged:
        report.fixed(
            "inertia",
            "some inertias were physically impossible (one axis heavier than the other two together)",
            "nudged them to the nearest valid values",
            nudged,
        )
    if implausible:
        report.fixed(
            "inertia",
            "some inertias do not match the size of their link (probably wrong units or placeholders)",
            "rebuilt them from the link shapes and kept the mass",
            implausible,
        )
    if defaulted:
        report.fixed(
            "inertia",
            "some moving links have no shape and no mass",
            f"gave them a tiny placeholder mass ({inertia.TINY_MASS} kg)",
            defaulted,
        )
    return masses


def _check_given(
    given: Inertial,
    shapes: list[Shape],
    bad_mass: list[str],
    bad_inertia: list[str],
    nudged: list[str],
    implausible: list[str],
    name: str,
    needs: bool,
    defaulted: list[str],
) -> Inertial | None:
    problem = inertia.inertia_problem(given.mass, given.tensor)
    if not problem:
        text = inertia.inertia_implausible(given.mass, given.tensor, inertia.shapes_radius(shapes, given.com))
        if text:
            fixed = inertia.scaled_to_mass(shapes, given.mass)
            if fixed is not None:
                implausible.append(f"{name}: {text}")
                return Inertial(given.mass, given.com, fixed.tensor)
        return given
    if "mass" in problem:
        derived = inertia.inertial_from_shapes(shapes)
        if derived is not None and derived.mass >= inertia.TINY_MASS:
            bad_mass.append(name)
            return derived
        if needs:
            defaulted.append(name)
            return inertia.default_inertial()
        return None
    if "impossible" in problem:
        nudged.append(name)
        return Inertial(given.mass, given.com, inertia.repair_triangle(given.tensor))
    fixed = inertia.scaled_to_mass(shapes, given.mass)
    if fixed is None:
        fixed = Inertial(given.mass, given.com, np.eye(3) * (given.mass * 1e-3))
    bad_inertia.append(name)
    return Inertial(given.mass, given.com, fixed.tensor)


# ---- compiler options ------------------------------------------------------------------------


def _set_compiler_options(robot: ET.Element) -> None:
    """Keep visuals, keep every link as its own body, and tolerate slightly invalid inertias."""
    block = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        block,
        "compiler",
        {"fusestatic": "false", "discardvisual": "false", "balanceinertia": "true", "strippath": "false"},
    )
