"""Writing a world or a robot out for other simulators: MJCF (the whole world) and URDF (one robot), with meshes.

MJCF is the world's own spec written as XML, with every mesh turned into an OBJ file in `meshes/` (the XML robotforge
keeps in `build/world.xml` holds meshes as inline numbers, which other tools read badly). MuJoCo centres a mesh on
its centre of mass and turns it to its principal axes when it compiles, and keeps the model in those coordinates;
the OBJ files are written back in the coordinates the XML's geoms expect, so loading the export again gives the
same bodies, joints, masses and the same vertices in the world (`check_mjcf` does that and compares).

URDF has no free joints, loops, ball joints or capsules, so a robot is written as the tree its joints make:
a floating base is the root link, a body with several joints gets a massless link between them, a capsule becomes
a cylinder, and anything lost is listed in `notes`. Geometry and inertia come from the compiled model, so they are
the numbers MuJoCo simulates with.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

MESH_DIR = "meshes"
OBJ_ASSETS = re.compile(r"[^A-Za-z0-9_.-]+")
G, J, OBJ = mujoco.mjtGeom, mujoco.mjtJoint, mujoco.mjtObj


class ExportError(ValueError):
    """Nothing was written; the message says what to change."""


@dataclass
class Exported:
    format: str
    main: Path  # the file to open
    files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self, base: Path) -> dict[str, Any]:
        return {
            "format": self.format,
            "file": str(self.main),
            "files": len(self.files),
            "meshes": sum(1 for f in self.files if f.suffix == ".obj"),
            "notes": self.notes,
        }


def export_world(
    world: Any, entry: Any, base_dir: Path, format: str, folder: Path, own_folder: bool = False
) -> dict[str, Any]:
    """Export a World (its compiled self), or with `entry` (one of its robots) just that robot on the floor, as
    "mjcf" or "urdf" into `folder`; load the result again and say how it compares. own_folder: a folder this export
    owns, emptied of an earlier export first."""
    from robotforge.design.world import compile_world

    if format not in ("mjcf", "urdf"):
        raise ExportError(f"format is mjcf or urdf, got {format!r}")
    if format == "urdf" and entry is None:
        raise ExportError("URDF is one robot at a time: give robot=NAME")
    alone = world.model_copy(update={"robots": [entry], "objects": [], "terrain": [], "water": None, "tasks": []})
    compiled = compile_world(alone if entry is not None else world, base_dir)
    if own_folder and folder.is_dir():
        clear(folder)
    if format == "urdf":
        written = export_urdf(compiled.model, entry.name, folder)
        check = check_urdf(written.main, compiled.model, entry.name)
        open_with = "load it in any URDF tool; in MuJoCo MjModel.from_xml_path(file)"
    else:
        name = entry.name if entry is not None else world.name
        written = export_mjcf(compiled.spec, compiled.model, folder, name)
        check = check_mjcf(written.main, compiled.model)
        open_with = f"python -m mujoco.viewer --mjcf={written.main}"
    return {**written.describe(folder), "folder": str(folder), "check": check, "open_with": open_with}


# ---- MJCF -------------------------------------------------------------------------------------------------------


def export_mjcf(spec: mujoco.MjSpec, model: mujoco.MjModel, out_dir: Path, name: str) -> Exported:
    """The compiled world (`spec` compiled into `model`) as `<out_dir>/<name>.xml` with meshes in `meshes/`."""
    out_dir = Path(out_dir)
    copy = spec.copy()
    assets: dict[str, bytes] = {}
    for mesh in copy.meshes:
        index = mujoco.mj_name2id(model, OBJ.mjOBJ_MESH, mesh.name)
        if index < 0:
            raise ExportError(f"mesh {mesh.name!r} is not in the compiled model")
        file = _asset_name(mesh.name, assets)
        assets[file] = _obj_bytes(_file_vertices(model, index, mesh), _faces(model, index))
        for attribute in ("uservert", "userface", "usernormal", "usertexcoord", "userfacenormal", "userfacetexcoord"):
            setattr(mesh, attribute, [])
        mesh.file = file
        mesh.scale = [1.0, 1.0, 1.0]  # the vertices written are scaled already
    copy.assets = assets
    copy.meshdir = MESH_DIR
    xml = copy.to_xml()
    out_dir.mkdir(parents=True, exist_ok=True)
    main = out_dir / f"{name}.xml"
    written = [main]
    _write(main, xml)
    if assets:
        (out_dir / MESH_DIR).mkdir(exist_ok=True)
    for file, data in assets.items():
        path = out_dir / MESH_DIR / file
        path.write_bytes(data)
        written.append(path)
    return Exported("mjcf", main, written)


def _asset_name(name: str, taken: dict[str, bytes]) -> str:
    stem = OBJ_ASSETS.sub("__", name).strip("_") or "mesh"
    file, n = f"{stem}.obj", 1
    while file in taken:
        n += 1
        file = f"{stem}_{n}.obj"
    return file


def _rotation(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    return mat.reshape(3, 3)


def _compiled_vertices(model: mujoco.MjModel, index: int) -> np.ndarray:
    adr, num = int(model.mesh_vertadr[index]), int(model.mesh_vertnum[index])
    return np.array(model.mesh_vert[adr : adr + num], float)


def _faces(model: mujoco.MjModel, index: int) -> np.ndarray:
    adr, num = int(model.mesh_faceadr[index]), int(model.mesh_facenum[index])
    return np.array(model.mesh_face[adr : adr + num], int)


def _file_vertices(model: mujoco.MjModel, index: int, mesh: Any) -> np.ndarray:
    """The mesh as the XML's geoms expect to find it in its file: undo the centring and turning MuJoCo did, and the
    mesh's own reference frame."""
    verts = _compiled_vertices(model, index)
    in_ref = np.asarray(model.mesh_pos[index], float) + verts @ _rotation(model.mesh_quat[index]).T
    return np.asarray(mesh.refpos, float) + in_ref @ _rotation(mesh.refquat).T


def _obj_bytes(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    lines = [f"v {x:.8g} {y:.8g} {z:.8g}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ---- URDF -------------------------------------------------------------------------------------------------------


def export_urdf(model: mujoco.MjModel, robot: str, out_dir: Path) -> Exported:
    """One robot of the compiled world as `<out_dir>/<robot>.urdf` with its meshes in `meshes/`."""
    out_dir = Path(out_dir)
    prefix = f"{robot}/"
    bodies = [b for b in range(1, model.nbody) if (_name(model, OBJ.mjOBJ_BODY, b)).startswith(prefix)]
    if not bodies:
        raise ExportError(f"the world has no robot named {robot!r}")
    own = set(bodies)
    roots = [b for b in bodies if int(model.body_parentid[b]) not in own]
    if len(roots) != 1:
        raise ExportError(f"{robot} is {len(roots)} separate pieces; a URDF is one tree")
    writer = _Urdf(model, robot, own, roots[0])
    xml = writer.build()
    out_dir.mkdir(parents=True, exist_ok=True)
    main = out_dir / f"{robot}.urdf"
    written = [main]
    _write(main, xml)
    if writer.meshes:
        (out_dir / MESH_DIR).mkdir(exist_ok=True)
    for file, data in writer.meshes.items():
        path = out_dir / MESH_DIR / file
        path.write_bytes(data)
        written.append(path)
    return Exported("urdf", main, written, writer.notes)


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def _pose(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rotation(quat)
    t[:3, 3] = pos
    return t


def _rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw (URDF: fixed axes x, y, z) of a rotation matrix."""
    pitch = float(np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0)))
    if abs(rot[2, 0]) < 1 - 1e-9:
        return float(np.arctan2(rot[2, 1], rot[2, 2])), pitch, float(np.arctan2(rot[1, 0], rot[0, 0]))
    return float(np.arctan2(-rot[1, 2], rot[1, 1])), pitch, 0.0


def _origin(parent: ET.Element, t: np.ndarray) -> None:
    roll, pitch, yaw = _rpy(t[:3, :3])
    ET.SubElement(parent, "origin", xyz=_nums(t[:3, 3]), rpy=_nums((roll, pitch, yaw)))


def _nums(values: Any) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


class _Urdf:
    def __init__(self, model: mujoco.MjModel, robot: str, bodies: set[int], root: int) -> None:
        self.m = model
        self.robot = robot
        self.bodies = bodies
        self.root = root
        self.notes: list[str] = []
        self.meshes: dict[str, bytes] = {}
        self.links: set[str] = set()
        self.xml = ET.Element("robot", name=robot)
        self.materials: dict[tuple, str] = {}
        self.mesh_files: dict[int, str] = {}
        self.children: dict[int, list[int]] = {b: [] for b in bodies}
        for b in sorted(bodies):
            parent = int(model.body_parentid[b])
            if parent in bodies:
                self.children[parent].append(b)
        self.actuated = self._actuators()

    def build(self) -> str:
        mj = ET.SubElement(self.xml, "mujoco")
        ET.SubElement(mj, "compiler", fusestatic="false", discardvisual="false", balanceinertia="true")
        self._body(self.root, None, np.eye(4), np.eye(4))
        self._notes_on_what_is_lost()
        ET.indent(self.xml)
        return '<?xml version="1.0"?>\n' + ET.tostring(self.xml, encoding="unicode") + "\n"

    def _actuators(self) -> dict[int, float]:
        """Joint -> the largest effort an actuator on it gives (0: not known)."""
        out: dict[int, float] = {}
        m = self.m
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            joint = int(m.actuator_trnid[a, 0])
            effort = float(m.actuator_forcerange[a, 1]) if m.actuator_forcelimited[a] else 0.0
            if effort <= 0 and m.actuator_ctrllimited[a]:
                effort = float(max(abs(m.actuator_ctrlrange[a, 0]), abs(m.actuator_ctrlrange[a, 1])))
            out[joint] = max(out.get(joint, 0.0), effort)
        return out

    def _body(self, b: int, parent: str | None, parent_link_t: np.ndarray, parent_body_t: np.ndarray) -> None:
        """Write body `b` and below. Transforms are all relative to the root body's frame: `parent_link_t` is the
        frame the parent link is written in (where its last joint sits), `parent_body_t` the parent body's own."""
        m = self.m
        name = self._link_name(b)
        chain = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b and int(m.jnt_type[j]) != int(J.mjJNT_FREE)]
        floats = len(chain) < sum(1 for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b)
        if floats and parent is not None:
            self.notes.append(f"{name}: a free joint below the root cannot be written; it is fixed to its parent")
        if parent is None:
            body_t = np.eye(4)
            if chain:  # the root has joints of its own (an arm on a rail): they hang from a link standing for the world
                parent, parent_link_t = "world", np.eye(4)
                self._empty_link("world")
                body_t = _pose(m.body_pos[b], m.body_quat[b])
        else:
            body_t = parent_body_t @ _pose(m.body_pos[b], m.body_quat[b])
        steps: list[tuple[int, int]] = []  # (axis of a ball joint or -1, mujoco joint), one per URDF joint
        for j in chain:
            if int(m.jnt_type[j]) == int(J.mjJNT_BALL):
                self.notes.append(f"{self._joint_name(j)}: a ball joint is written as three revolute joints x, y, z")
                steps += [(axis, j) for axis in range(3)]
            else:
                steps.append((-1, j))
        link_t = body_t
        prev, prev_t = parent, parent_link_t
        if not steps:
            self._link(b, name, body_t, body_t)
            if parent is not None:
                self._fixed(parent, name, np.linalg.inv(prev_t) @ body_t)
        for k, (axis, j) in enumerate(steps):
            link_t = body_t @ _pose(m.jnt_pos[j], [1.0, 0.0, 0.0, 0.0])
            link = name if k == len(steps) - 1 else f"{name}__j{k + 1}"
            if link == name:
                self._link(b, link, link_t, body_t)
            else:
                self._empty_link(link)
            self._joint(j, axis, prev, link, np.linalg.inv(prev_t) @ link_t)
            prev, prev_t = link, link_t
        for child in self.children[b]:
            self._body(child, name, link_t, body_t)

    def _link_name(self, b: int) -> str:
        raw = _name(self.m, OBJ.mjOBJ_BODY, b).removeprefix(f"{self.robot}/").replace("/", "__") or f"body{b}"
        name, n = raw, 1
        while name in self.links:
            n += 1
            name = f"{raw}_{n}"
        self.links.add(name)
        return name

    def _joint_name(self, j: int) -> str:
        return _name(self.m, OBJ.mjOBJ_JOINT, j).removeprefix(f"{self.robot}/").replace("/", "__") or f"joint{j}"

    def _empty_link(self, name: str) -> None:
        self.links.add(name)
        link = ET.SubElement(self.xml, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value="1e-6")
        ET.SubElement(inertial, "inertia", ixx="1e-9", ixy="0", ixz="0", iyy="1e-9", iyz="0", izz="1e-9")

    def _link(self, b: int, name: str, link_frame: np.ndarray, body_t: np.ndarray) -> None:
        """A link written in `link_frame`; the body's inertia and geoms are given in the body's frame `body_t`."""
        m = self.m
        link = ET.SubElement(self.xml, "link", name=name)
        to_link = np.linalg.inv(link_frame) @ body_t  # body frame -> link frame
        mass = float(m.body_mass[b])
        inertial = ET.SubElement(link, "inertial")
        _origin(inertial, to_link @ _pose(m.body_ipos[b], m.body_iquat[b]))
        ET.SubElement(inertial, "mass", value=f"{max(mass, 1e-6):.8g}")
        ixx, iyy, izz = (max(float(v), 1e-9) for v in m.body_inertia[b])
        ET.SubElement(
            inertial, "inertia", ixx=f"{ixx:.8g}", ixy="0", ixz="0", iyy=f"{iyy:.8g}", iyz="0", izz=f"{izz:.8g}"
        )
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b:
                continue
            self._geom(link, g, to_link @ _pose(m.geom_pos[g], m.geom_quat[g]))

    def _geom(self, link: ET.Element, g: int, t: np.ndarray) -> None:
        m = self.m
        kind = int(m.geom_type[g])
        shape = self._shape(g, kind)
        if shape is None:
            return
        drawn = int(m.geom_group[g]) != 3
        collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        if drawn:
            visual = ET.SubElement(link, "visual")
            _origin(visual, t)
            self._append_shape(visual, shape)
            self._material(visual, g)
        if collides:
            collision = ET.SubElement(link, "collision")
            _origin(collision, t)
            self._append_shape(collision, shape)

    def _append_shape(self, parent: ET.Element, shape: tuple[str, dict[str, str]]) -> None:
        geometry = ET.SubElement(parent, "geometry")
        ET.SubElement(geometry, shape[0], **shape[1])

    def _shape(self, g: int, kind: int) -> tuple[str, dict[str, str]] | None:
        m = self.m
        size = np.asarray(m.geom_size[g], float)
        if kind == int(G.mjGEOM_BOX):
            return "box", {"size": _nums(size * 2)}
        if kind == int(G.mjGEOM_SPHERE):
            return "sphere", {"radius": f"{size[0]:.8g}"}
        if kind == int(G.mjGEOM_CYLINDER):
            return "cylinder", {"radius": f"{size[0]:.8g}", "length": f"{size[1] * 2:.8g}"}
        if kind == int(G.mjGEOM_CAPSULE):
            self._note_once(
                "capsule", "URDF has no capsules: each is written as a cylinder of the same radius and length"
            )
            return "cylinder", {"radius": f"{size[0]:.8g}", "length": f"{size[1] * 2 + size[0] * 2:.8g}"}
        if kind == int(G.mjGEOM_ELLIPSOID):
            self._note_once("ellipsoid", "URDF has no ellipsoids: each is written as a sphere of the mean radius")
            return "sphere", {"radius": f"{float(size.mean()):.8g}"}
        if kind == int(G.mjGEOM_MESH):
            return "mesh", {"filename": f"{MESH_DIR}/{self._mesh_file(int(m.geom_dataid[g]))}"}
        self._note_once(f"kind{kind}", "a plane or height field on a robot is left out")
        return None

    def _mesh_file(self, index: int) -> str:
        if index not in self.mesh_files:
            name = _name(self.m, OBJ.mjOBJ_MESH, index) or f"mesh{index}"
            file = _asset_name(name, self.meshes)
            self.meshes[file] = _obj_bytes(_compiled_vertices(self.m, index), _faces(self.m, index))
            self.mesh_files[index] = file
        return self.mesh_files[index]

    def _material(self, visual: ET.Element, g: int) -> None:
        rgba = tuple(round(float(v), 4) for v in self.m.geom_rgba[g])
        matid = int(self.m.geom_matid[g])
        if matid >= 0:
            rgba = tuple(round(float(v), 4) for v in self.m.mat_rgba[matid])
        if rgba not in self.materials:
            self.materials[rgba] = f"color{len(self.materials)}"
        material = ET.SubElement(visual, "material", name=self.materials[rgba])
        ET.SubElement(material, "color", rgba=_nums(rgba))

    def _fixed(self, parent: str, child: str, t: np.ndarray) -> None:
        joint = ET.SubElement(self.xml, "joint", name=f"{child}__fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        _origin(joint, t)

    def _joint(self, j: int, axis_index: int, parent: str, child: str, t: np.ndarray) -> None:
        m = self.m
        kind = int(m.jnt_type[j])
        limited = bool(m.jnt_limited[j])
        lo, hi = (float(v) for v in m.jnt_range[j])
        if kind == int(J.mjJNT_BALL):
            direction = np.eye(3)[axis_index]
            name = f"{self._joint_name(j)}_{'xyz'[axis_index]}"
            urdf_kind = "continuous"
            limited = False
        else:
            direction = np.asarray(m.jnt_axis[j], float)
            name = self._joint_name(j)
            if kind == int(J.mjJNT_SLIDE):
                urdf_kind = "prismatic"
                if not limited:
                    lo, hi = -1000.0, 1000.0
                    self._note_once("slide", "a sliding joint without limits is written with limits of 1000 m")
            else:
                urdf_kind = "revolute" if limited else "continuous"
        joint = ET.SubElement(self.xml, "joint", name=name, type=urdf_kind)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        _origin(joint, t)
        ET.SubElement(joint, "axis", xyz=_nums(direction / max(float(np.linalg.norm(direction)), 1e-12)))
        effort = self.actuated.get(j, 0.0)
        attributes = {"effort": f"{effort if effort > 0 else 100.0:.8g}", "velocity": "10"}
        if urdf_kind != "continuous":
            attributes |= {"lower": f"{lo:.8g}", "upper": f"{hi:.8g}"}
        ET.SubElement(joint, "limit", **attributes)
        dof = int(m.jnt_dofadr[j])
        damping, friction = float(m.dof_damping[dof]), float(m.dof_frictionloss[dof])
        if damping or friction:
            ET.SubElement(joint, "dynamics", damping=f"{damping:.8g}", friction=f"{friction:.8g}")

    def _note_once(self, key: str, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def _notes_on_what_is_lost(self) -> None:
        m = self.m
        mine = [
            e
            for e in range(m.neq)
            if _name(m, OBJ.mjOBJ_BODY, int(m.eq_obj1id[e])).startswith(f"{self.robot}/")
            or _name(m, OBJ.mjOBJ_JOINT, int(m.eq_obj1id[e])).startswith(f"{self.robot}/")
        ]
        if mine:
            self.notes.append(f"{len(mine)} equality constraints (closed loops, couplings) cannot be written in URDF")
        if any(int(m.jnt_type[j]) == int(J.mjJNT_FREE) for j in range(m.njnt) if int(m.jnt_bodyid[j]) == self.root):
            self.notes.append("the base floats: it is the URDF's root link with no joint; fix or float it in your tool")


# ---- loading an export back and comparing it ----------------------------------------------------------------------


def check_mjcf(path: Path, model: mujoco.MjModel) -> dict[str, Any]:
    """Load the exported XML in MuJoCo again and compare it with the model it was written from."""
    try:
        again = mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        return {"loads": False, "error": str(exc)}
    out: dict[str, Any] = {"loads": True}
    for label, kind, count in (
        ("bodies", OBJ.mjOBJ_BODY, model.nbody),
        ("joints", OBJ.mjOBJ_JOINT, model.njnt),
        ("actuators", OBJ.mjOBJ_ACTUATOR, model.nu),
        ("geoms", OBJ.mjOBJ_GEOM, model.ngeom),
    ):
        here = {_name(model, kind, i) or f"#{i}" for i in range(count)}
        there = {_name(again, kind, i) or f"#{i}" for i in range(_count(again, kind))}
        out[label] = {
            "count": count,
            "same_names": bool(here == there) if kind != OBJ.mjOBJ_GEOM else count == len(there),
        }
    out["mass_kg"] = {"world": round(float(model.body_mass.sum()), 4), "export": round(float(again.body_mass.sum()), 4)}
    a, b = mujoco.MjData(model), mujoco.MjData(again)
    mujoco.mj_forward(model, a)
    mujoco.mj_forward(again, b)
    same_shape = again.nbody == model.nbody
    out["body_position_error_m"] = float(np.abs(a.xpos - b.xpos).max()) if same_shape else None
    out["vertex_error_m"] = _vertex_error(model, a, again, b)
    if model.nhfield:
        out["height_field_error_m"] = (
            float(np.abs(model.hfield_data - again.hfield_data).max()) if again.nhfield == model.nhfield else None
        )
    ok = all(v["same_names"] for k, v in out.items() if isinstance(v, dict) and "same_names" in v)
    ground = out.get("height_field_error_m", 0.0)
    out["same"] = bool(
        ok and out["body_position_error_m"] is not None and out["body_position_error_m"] < 1e-5 and ground is not None
    )
    return out


def check_urdf(path: Path, model: mujoco.MjModel, robot: str) -> dict[str, Any]:
    """Load the exported URDF in MuJoCo again and compare its bodies and joints with the robot's in `model`."""
    try:
        again = mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        return {"loads": False, "error": str(exc)}
    prefix = f"{robot}/"

    def local(name: str) -> str:
        return name.removeprefix(prefix).replace("/", "__")

    bodies = {
        local(_name(model, OBJ.mjOBJ_BODY, b))
        for b in range(1, model.nbody)
        if _name(model, OBJ.mjOBJ_BODY, b).startswith(prefix)
    }
    joints: dict[str, tuple[float, float, bool]] = {}
    for j in range(model.njnt):
        name = _name(model, OBJ.mjOBJ_JOINT, j)
        kind = int(model.jnt_type[j])
        if not name.startswith(prefix) or kind == int(J.mjJNT_FREE):
            continue
        low, high = (float(v) for v in model.jnt_range[j])
        if kind == int(J.mjJNT_BALL):
            joints.update({f"{local(name)}_{axis}": (0.0, 0.0, False) for axis in "xyz"})
        else:
            joints[local(name)] = (low, high, bool(model.jnt_limited[j]))
    there_bodies = {_name(again, OBJ.mjOBJ_BODY, b) for b in range(1, again.nbody)}
    there_joints = {_name(again, OBJ.mjOBJ_JOINT, j): j for j in range(again.njnt)}
    worst = 0.0
    for name, (low, high, limited) in joints.items():
        j = there_joints.get(name)
        if j is not None and limited and again.jnt_limited[j]:
            worst = max(worst, abs(float(again.jnt_range[j][0]) - low), abs(float(again.jnt_range[j][1]) - high))
    mass = float(
        sum(model.body_mass[b] for b in range(1, model.nbody) if _name(model, OBJ.mjOBJ_BODY, b).startswith(prefix))
    )
    out = {
        "loads": True,
        "bodies": {"count": len(bodies), "same_names": bodies <= there_bodies},
        "joints": {"count": len(joints), "same_names": set(joints) == set(there_joints)},
        "mass_kg": {"robot": round(mass, 4), "export": round(float(again.body_mass.sum()), 4)},
        "joint_range_error": worst,
    }
    out["same"] = bool(
        out["bodies"]["same_names"]
        and out["joints"]["same_names"]
        and abs(mass - float(again.body_mass.sum())) < 1e-3 * max(mass, 1.0)
        and worst < 1e-5
    )
    return out


def _count(model: mujoco.MjModel, kind: mujoco.mjtObj) -> int:
    return {
        OBJ.mjOBJ_BODY: model.nbody,
        OBJ.mjOBJ_JOINT: model.njnt,
        OBJ.mjOBJ_ACTUATOR: model.nu,
        OBJ.mjOBJ_GEOM: model.ngeom,
    }[kind]


def _vertex_error(m1: mujoco.MjModel, d1: mujoco.MjData, m2: mujoco.MjModel, d2: mujoco.MjData) -> float | None:
    """Largest gap between a mesh vertex in the world and the same vertex of the reloaded export."""
    if m1.ngeom != m2.ngeom:
        return None
    worst = 0.0
    for g in range(m1.ngeom):
        if int(m1.geom_type[g]) != int(G.mjGEOM_MESH) or int(m2.geom_type[g]) != int(G.mjGEOM_MESH):
            continue
        first, second = (
            _world_vertices(m1, d1, g),
            _world_vertices(m2, d2, g),
        )
        if first.shape != second.shape:
            return None
        worst = max(worst, float(np.abs(first - second).max()))
    return worst


def _world_vertices(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    verts = _compiled_vertices(model, int(model.geom_dataid[g]))
    return data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T


def clear(path: Path) -> None:
    """Empty an export folder before writing into it again (only what an export writes)."""
    for sub in (path / MESH_DIR,):
        if sub.is_dir():
            shutil.rmtree(sub)
    for leftover in list(path.glob("*.xml")) + list(path.glob("*.urdf")):
        leftover.unlink()
