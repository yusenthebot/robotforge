"""Read a URDF, repair what would trip up MuJoCo, and write a clean copy ("staged" URDF).

The staged copy has absolute mesh paths in formats MuJoCo reads, sane joints, valid mass and
inertia on every link, and compiler options that keep visuals and every link as its own body.
MuJoCo then loads it with its own URDF compiler, so we never re-implement kinematics.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotforge.design.importing.inertia import Shape
from robotforge.design.importing.meshio import MeshAsset
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.resolve import MeshResolver
from robotforge.design.importing.urdf_geometry import prepare_geometry
from robotforge.design.importing.urdf_inertia import fix_inertials
from robotforge.design.importing.xmlnum import floats

_UNBOUNDED = 1e9
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
    ctx, shapes, visual_shapes = prepare_geometry(robot, links, resolver, work_dir, report, decompose)
    link_shapes = {n: (shapes[n] or visual_shapes[n]) for n in links}
    masses = fix_inertials(links, joints, roots, link_shapes, report)
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
            fixed = floats(text, 3)
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
            for extra in ("axis", "limit", "dynamics", "mimic"):  # meaningless on a fixed joint; some are malformed
                for old in el.findall(extra):
                    el.remove(old)
        mimic = el.find("mimic")
        if mimic is not None and mimic.get("joint"):
            info.mimic_of = mimic.get("joint")
            info.mimic_multiplier = floats(mimic.get("multiplier"), 1, default=1.0)[0]
            info.mimic_offset = floats(mimic.get("offset"), 1)[0]
        dyn = el.find("dynamics")
        if dyn is not None:
            info.damping = max(0.0, floats(dyn.get("damping"), 1)[0])
            info.friction = max(0.0, floats(dyn.get("friction"), 1)[0])
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
    axis = np.array(floats(axis_el.get("xyz") if axis_el is not None else "1 0 0", 3))
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
    # The motors robotforge adds carry these limits; left in the file, MuJoCo would also clamp the joint's total
    # actuator force to them (and to 0 for placeholder values), whatever the motors are told.
    limit.attrib.pop("effort", None)
    limit.attrib.pop("velocity", None)
    has_lo, has_hi = limit.get("lower") is not None, limit.get("upper") is not None

    if info.type == "continuous":
        for attr in ("lower", "upper"):
            limit.attrib.pop(attr, None)
        return
    lo = floats(limit.get("lower"), 1)[0] if has_lo else None
    hi = floats(limit.get("upper"), 1)[0] if has_hi else None
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
    val = floats(text, 1)[0]
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


# ---- compiler options ------------------------------------------------------------------------


def _set_compiler_options(robot: ET.Element) -> None:
    """Keep visuals, keep every link as its own body, and tolerate slightly invalid inertias."""
    block = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        block,
        "compiler",
        {"fusestatic": "false", "discardvisual": "false", "balanceinertia": "true", "strippath": "false"},
    )
