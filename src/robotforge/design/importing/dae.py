"""A small dependency-free reader for Collada (.dae) meshes.

trimesh needs the pycollada package for .dae files, which robotforge does not ship. Robot
descriptions use a narrow, well-behaved slice of Collada (static triangle geometry with flat
material colors), and that slice is easy to read directly. Skinning, animation, textures and
lights are ignored on purpose.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DaeError(ValueError):
    """The file is not a Collada file we can read."""


@dataclass
class DaePart:
    """Triangles sharing one color (rgba is None when the material has no flat color)."""

    vertices: np.ndarray  # (n, 3) float64, meters, Z up
    faces: np.ndarray  # (m, 3) int64
    rgba: tuple[float, float, float, float] | None


@dataclass
class _Prim:
    material: str
    positions: np.ndarray
    faces: np.ndarray


def read_dae(path: Path) -> list[DaePart]:
    """Read a .dae file into triangle parts, one per distinct color."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DaeError(f"not readable as Collada XML ({exc})") from exc
    if _tag(root) != "COLLADA":
        raise DaeError("the file does not start with a COLLADA element")

    unit, up_axis = _asset(root)
    colors = _material_colors(root)
    geometries = {g.get("id", ""): _geometry_primitives(g) for g in root.iter() if _tag(g) == "geometry"}
    nodes = {n.get("id", ""): n for lib in _kids(root, "library_nodes") for n in _kids(lib, "node")}
    controllers = {c.get("id", ""): _skin_source(c) for c in root.iter() if _tag(c) == "controller"}

    instances: list[tuple[str, dict[str, str], np.ndarray]] = []
    scenes = [vs for lib in _kids(root, "library_visual_scenes") for vs in _kids(lib, "visual_scene")]
    for scene in scenes:
        for node in _kids(scene, "node"):
            _walk(node, np.eye(4), nodes, controllers, instances, depth=0)
    if not instances:  # no scene graph: take every geometry as it is
        instances = [(gid, {}, np.eye(4)) for gid in geometries]

    buckets: dict[tuple[float, ...] | None, list[tuple[np.ndarray, np.ndarray]]] = {}
    for gid, bindings, transform in instances:
        for prim in geometries.get(gid, []):
            rgba = colors.get(bindings.get(prim.material, prim.material))
            key = tuple(round(c, 3) for c in rgba) if rgba else None
            verts = prim.positions @ transform[:3, :3].T + transform[:3, 3]
            buckets.setdefault(key, []).append((verts, prim.faces))
    if not buckets:
        raise DaeError("no triangle geometry found")

    parts = []
    for key, chunks in buckets.items():
        offset, verts_all, faces_all = 0, [], []
        for verts, faces in chunks:
            verts_all.append(verts)
            faces_all.append(faces + offset)
            offset += len(verts)
        vertices = np.concatenate(verts_all) * unit
        if up_axis == "Y_UP":
            vertices = np.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], axis=1)
        elif up_axis == "X_UP":
            vertices = np.stack([-vertices[:, 1], vertices[:, 2], vertices[:, 0]], axis=1)
        parts.append(DaePart(vertices, np.concatenate(faces_all), tuple(key) if key else None))  # type: ignore[arg-type]
    return parts


# ---- xml helpers ----------------------------------------------------------------------


def _tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _kids(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _tag(c) == name]


def _kid(el: ET.Element | None, name: str) -> ET.Element | None:
    if el is None:
        return None
    for c in el:
        if _tag(c) == name:
            return c
    return None


def _floats(text: str | None) -> np.ndarray:
    return np.array((text or "").split(), dtype=np.float64)


def _ints(text: str | None) -> np.ndarray:
    return np.array((text or "").split(), dtype=np.int64)


def _ref(url: str | None) -> str:
    return (url or "").lstrip("#")


# ---- asset, materials -------------------------------------------------------------------


def _asset(root: ET.Element) -> tuple[float, str]:
    asset = _kid(root, "asset")
    unit_el = _kid(asset, "unit")
    unit = 1.0
    if unit_el is not None:
        try:
            unit = float(unit_el.get("meter", "1"))
        except ValueError:
            unit = 1.0
    if not unit > 0:
        unit = 1.0
    up = _kid(asset, "up_axis")
    return unit, (up.text or "Z_UP").strip() if up is not None else "Z_UP"


def _material_colors(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    """Material id -> flat diffuse color (alpha included)."""
    effect_colors: dict[str, tuple[float, float, float, float]] = {}
    for effect in root.iter():
        if _tag(effect) != "effect":
            continue
        color = _effect_color(effect)
        if color is not None:
            effect_colors[effect.get("id", "")] = color
    out = {}
    for material in root.iter():
        if _tag(material) != "material":
            continue
        inst = _kid(material, "instance_effect")
        if inst is not None and _ref(inst.get("url")) in effect_colors:
            out[material.get("id", "")] = effect_colors[_ref(inst.get("url"))]
    return out


def _effect_color(effect: ET.Element) -> tuple[float, float, float, float] | None:
    for shader in effect.iter():
        if _tag(shader) not in ("phong", "lambert", "blinn", "constant"):
            continue
        slot = _kid(shader, "emission" if _tag(shader) == "constant" else "diffuse")
        color = _kid(slot, "color")
        if color is None:
            continue
        vals = _floats(color.text)
        if len(vals) >= 3:
            alpha = float(vals[3]) if len(vals) > 3 else 1.0
            return (float(vals[0]), float(vals[1]), float(vals[2]), alpha)
    return None


# ---- geometry ------------------------------------------------------------------------------


def _geometry_primitives(geometry: ET.Element) -> list[_Prim]:
    mesh = _kid(geometry, "mesh")
    if mesh is None:
        return []
    sources: dict[str, np.ndarray] = {}
    for src in _kids(mesh, "source"):
        arr = _kid(src, "float_array")
        if arr is None:
            continue
        data = _floats(arr.text)
        accessor = _kid(_kid(src, "technique_common"), "accessor")
        stride = int(accessor.get("stride", "3")) if accessor is not None else 3
        if stride < 3 or len(data) % stride:
            continue
        sources[src.get("id", "")] = data.reshape(-1, stride)[:, :3]
    vertex_sources = {}
    for vert in _kids(mesh, "vertices"):
        for inp in _kids(vert, "input"):
            if inp.get("semantic") == "POSITION":
                vertex_sources[vert.get("id", "")] = _ref(inp.get("source"))

    prims = []
    for prim in mesh:
        kind = _tag(prim)
        if kind not in ("triangles", "polylist", "polygons", "trifans", "tristrips"):
            continue
        inputs = _kids(prim, "input")
        vertex_input = next((i for i in inputs if i.get("semantic") in ("VERTEX", "POSITION")), None)
        if vertex_input is None:
            continue
        src_id = _ref(vertex_input.get("source"))
        positions = sources.get(vertex_sources.get(src_id, src_id))
        if positions is None:
            continue
        stride = max(int(i.get("offset", "0")) for i in inputs) + 1
        offset = int(vertex_input.get("offset", "0"))
        faces = _triangulate(prim, kind, stride, offset)
        if len(faces) and faces.max() < len(positions) and faces.min() >= 0:
            prims.append(_Prim(prim.get("material", ""), positions, faces))
    return prims


def _triangulate(prim: ET.Element, kind: str, stride: int, offset: int) -> np.ndarray:
    def vertex_ids(p: ET.Element) -> np.ndarray:
        idx = _ints(p.text)
        return idx[: len(idx) // stride * stride].reshape(-1, stride)[:, offset]

    p_el, vcount_el = _kid(prim, "p"), _kid(prim, "vcount")
    if kind == "triangles":
        if p_el is None:
            return np.zeros((0, 3), dtype=np.int64)
        ids = vertex_ids(p_el)
        return ids[: len(ids) // 3 * 3].reshape(-1, 3)
    if kind == "polylist":
        if p_el is None or vcount_el is None:
            return np.zeros((0, 3), dtype=np.int64)
        ids = vertex_ids(p_el)
        counts = _ints(vcount_el.text)
        polys, pos = [], 0
        for n in counts:
            polys.append(ids[pos : pos + n])
            pos += n
        return _fan(polys)
    polys = [vertex_ids(p) for p in _kids(prim, "p")]
    if kind == "polygons":
        return _fan(polys)
    if kind == "trifans":
        return _fan(polys)
    faces = []  # tristrips: alternate the winding so every triangle faces the same way
    for strip in polys:
        for i in range(len(strip) - 2):
            tri = strip[i : i + 3]
            faces.append(tri if i % 2 == 0 else tri[[1, 0, 2]])
    return np.array(faces, dtype=np.int64).reshape(-1, 3)


def _fan(polys: list[np.ndarray]) -> np.ndarray:
    faces = []
    for poly in polys:
        for i in range(1, len(poly) - 1):
            faces.append((poly[0], poly[i], poly[i + 1]))
    return np.array(faces, dtype=np.int64).reshape(-1, 3)


# ---- scene graph ---------------------------------------------------------------------------


def _skin_source(controller: ET.Element) -> str:
    skin = _kid(controller, "skin")
    return _ref(skin.get("source")) if skin is not None else ""


def _node_matrix(node: ET.Element) -> np.ndarray:
    m = np.eye(4)
    for el in node:
        kind = _tag(el)
        vals = _floats(el.text)
        t = np.eye(4)
        if kind == "matrix" and len(vals) == 16:
            t = vals.reshape(4, 4)
        elif kind == "translate" and len(vals) == 3:
            t[:3, 3] = vals
        elif kind == "scale" and len(vals) == 3:
            t[:3, :3] = np.diag(vals)
        elif kind == "rotate" and len(vals) == 4:
            t[:3, :3] = _axis_angle(vals[:3], math.radians(vals[3]))
        else:
            continue
        m = m @ t
    return m


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n == 0:
        return np.eye(3)
    x, y, z = axis / n
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def _walk(
    node: ET.Element,
    parent: np.ndarray,
    library: dict[str, ET.Element],
    controllers: dict[str, str],
    out: list[tuple[str, dict[str, str], np.ndarray]],
    depth: int,
) -> None:
    if depth > 64:
        return
    world = parent @ _node_matrix(node)
    for el in node:
        kind = _tag(el)
        if kind in ("instance_geometry", "instance_controller"):
            target = _ref(el.get("url"))
            if kind == "instance_controller":
                target = controllers.get(target, "")
            bindings = {}
            for im in el.iter():
                if _tag(im) == "instance_material":
                    bindings[im.get("symbol", "")] = _ref(im.get("target"))
            out.append((target, bindings, world))
        elif kind == "instance_node":
            sub = library.get(_ref(el.get("url")))
            if sub is not None:
                _walk(sub, world, library, controllers, out, depth + 1)
        elif kind == "node":
            _walk(el, world, library, controllers, out, depth + 1)
