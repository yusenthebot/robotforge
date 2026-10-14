"""Small synthetic URDF, mesh and Collada files for the import doctor tests (no network, no big files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

INERTIAL = (
    '<inertial><origin xyz="0 0 0"/><mass value="{m}"/>'
    '<inertia ixx="{i}" ixy="0" ixz="0" iyy="{i}" iyz="0" izz="{i}"/></inertial>'
)


def inertial(mass: float = 1.0, i: float = 0.01) -> str:
    return INERTIAL.format(m=mass, i=i)


def box_link(name: str, size=(0.1, 0.1, 0.1), mass: float = 1.0, origin: str = "0 0 0", extra: str = "") -> str:
    s = " ".join(str(v) for v in size)
    geom = f'<geometry><box size="{s}"/></geometry>'
    return (
        f'<link name="{name}">{inertial(mass)}{extra}'
        f'<visual><origin xyz="{origin}"/>{geom}</visual>'
        f'<collision><origin xyz="{origin}"/>{geom}</collision></link>'
    )


def revolute(name: str, parent: str, child: str, xyz: str = "0 0 0.1", axis: str = "0 1 0", limit: str | None = None):
    limit = limit if limit is not None else '<limit lower="-1.5" upper="1.5" effort="20" velocity="2"/>'
    return (
        f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
        f'<origin xyz="{xyz}"/><axis xyz="{axis}"/>{limit}</joint>'
    )


def robot(body: str, name: str = "bot") -> str:
    return f'<?xml version="1.0"?><robot name="{name}">{body}</robot>'


def arm_urdf(name: str = "arm") -> str:
    """Three-link arm on a fixed base, with every ingredient a real URDF has except motors."""
    return robot(
        box_link("base_link", (0.2, 0.2, 0.1), 5.0, "0 0 0.05")
        + box_link("upper", (0.06, 0.06, 0.3), 1.0, "0 0 0.15")
        + box_link("fore", (0.05, 0.05, 0.25), 0.6, "0 0 0.125")
        + revolute("shoulder", "base_link", "upper", "0 0 0.1")
        + revolute("elbow", "upper", "fore", "0 0 0.3"),
        name,
    )


def rover_urdf(caster: bool = True) -> str:
    """A box with two driven wheels (continuous joints) and a skid caster."""
    wheel = (
        '<link name="{n}">' + inertial(0.1, 2e-4) + '<visual><origin rpy="1.5708 0 0"/>'
        '<geometry><cylinder radius="0.05" length="0.03"/></geometry></visual>'
        '<collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.05" length="0.03"/></geometry></collision>'
        "</link>"
    )
    joint = (
        '<joint name="{n}_joint" type="continuous"><parent link="chassis"/><child link="{n}"/>'
        '<origin xyz="0 {y} 0"/><axis xyz="0 1 0"/></joint>'
    )
    return robot(
        box_link("chassis", (0.3, 0.2, 0.08), 2.0, "0 0 0")
        + wheel.format(n="left_wheel")
        + wheel.format(n="right_wheel")
        + joint.format(n="left_wheel", y=0.13)
        + joint.format(n="right_wheel", y=-0.13)
        + (
            '<link name="caster_link">'
            + inertial(0.05, 1e-5)
            + '<collision><geometry><sphere radius="0.02"/></geometry></collision></link>'
            + '<joint name="caster_joint" type="fixed"><parent link="chassis"/><child link="caster_link"/>'
            + '<origin xyz="-0.12 0 -0.03"/></joint>'
            if caster
            else ""
        ),
        "rover",
    )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def box_stl(path: Path, size=(0.1, 0.1, 0.1)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(size).export(path)
    return path


def ascii_stl(path: Path, size=(0.1, 0.1, 0.1)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(trimesh.exchange.stl.export_stl_ascii(trimesh.creation.box(size)).encode())
    return path


def multi_object_obj(path: Path) -> Path:
    """Two tetrahedra in two OBJ objects (MuJoCo would only read the first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "o a\nv 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        "o b\nv 0.3 0 0\nv 0.4 0 0\nv 0.3 0.1 0\nv 0.3 0 0.1\nf 5 7 6\nf 5 6 8\nf 5 8 7\nf 6 7 8\n"
    )
    return path


def flat_quad_stl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    quad = trimesh.Trimesh([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2], [0, 2, 3]], process=False)
    quad.export(path)
    return path


def dae_text(unit: float = 1.0, up: str = "Z_UP", color=(1.0, 0.0, 0.0, 1.0), size=(1.0, 1.0, 1.0), nodes=True) -> str:
    """A one-cube Collada file (12 triangles) with a flat diffuse color."""
    box = trimesh.creation.box(size)
    verts = " ".join(f"{v:.6f}" for v in np.asarray(box.vertices).ravel())
    idx = " ".join(str(i) for i in np.asarray(box.faces).ravel())
    r, g, b, a = color
    scene = (
        '<library_visual_scenes><visual_scene id="S"><node id="n"><matrix>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>'
        '<instance_geometry url="#cube"><bind_material><technique_common>'
        '<instance_material symbol="mat" target="#matid"/></technique_common></bind_material>'
        "</instance_geometry></node></visual_scene></library_visual_scenes>"
        '<scene><instance_visual_scene url="#S"/></scene>'
        if nodes
        else ""
    )
    return f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="u" meter="{unit}"/><up_axis>{up}</up_axis></asset>
<library_effects><effect id="eff"><profile_COMMON><technique sid="c"><phong><diffuse>
<color>{r} {g} {b} {a}</color></diffuse></phong></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="matid"><instance_effect url="#eff"/></material></library_materials>
<library_geometries><geometry id="cube"><mesh>
<source id="pos"><float_array id="posarr" count="{len(box.vertices) * 3}">{verts}</float_array>
<technique_common><accessor source="#posarr" count="{len(box.vertices)}" stride="3">
<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
</accessor></technique_common></source>
<vertices id="vtx"><input semantic="POSITION" source="#pos"/></vertices>
<triangles material="mat" count="{len(box.faces)}"><input semantic="VERTEX" source="#vtx" offset="0"/>
<p>{idx}</p></triangles></mesh></geometry></library_geometries>
{scene}
</COLLADA>"""
