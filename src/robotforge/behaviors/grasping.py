"""Choosing where and how a two-finger gripper takes hold of a thing, from the thing's shape.

The gripper comes straight down and closes its jaws along one horizontal direction, so a grasp is a place (x, y, z) and
a closing direction. The shape of the thing, a box, a cylinder, a ball, a mesh, or a thing made of several of them,
becomes points on its surface and their convex hull (the shape the fingers can reach); every closing direction and
every height are tried, and each is scored by what makes a grasp hold:

- the jaws open wide enough, with room to spare, and the fingers can come down on either side: nothing above the
  grasp is wider than the opening, and the fingertips clear the table;
- the two contact faces are antipodal: their normals point along the closing direction within the friction cone, so
  the squeeze does not slide the thing out;
- the grasp is near the thing's centre of mass, so it does not tip in the fingers;
- the faces the fingers meet are big and flat, not a corner or an edge.

`plan` returns the best grasps first; manager.pick_and_place asks it for any thing that is not a plain box, can or ball,
and `plan_grasp` shows an agent the choices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

G = mujoco.mjtGeom
YAW_STEP = math.radians(5.0)
HEIGHT_LEVELS = 9  # heights tried between the thing's lowest and highest point
BAND = 0.006  # m: the contact faces are looked for within this height of the grasp height
FACE_DEPTH = 0.002  # m: points this near the extreme along the closing direction make up a contact face
HOLD_DEPTH = 0.004  # m of the thing above the grasp height, at least
FINGER_CLEAR = 0.012  # m the grasp keeps above the lowest point: the fingertips reach this far below the tool point
SPARE = 0.006  # m of opening the jaws keep to spare on either side of a grasp, and above it
FRICTION = 0.6  # the friction of fingertip pads on most things; the friction cone's half angle is atan of it
SPHERE_POINTS = 120
RING_POINTS = 28
MAX_POINTS = 4000


@dataclass(frozen=True)
class Grasp:
    """One way to take hold: the tool point goes to `center`, the jaws close along `closing`."""

    center: np.ndarray  # world, between the contact faces at the grasp height
    closing: np.ndarray  # unit, horizontal
    width: float  # m between the contact faces
    below: float  # m from the center down to the thing's lowest point
    score: float
    grip: float  # 0 to 1: how well the contact normals line up with the closing direction
    offset: float  # m from the centre of mass to the grasp's line, sideways
    note: str

    def as_dict(self) -> dict[str, object]:
        yaw = math.degrees(math.atan2(self.closing[1], self.closing[0])) % 180.0
        return {
            "at": [round(float(v), 3) for v in self.center],
            "closing_deg": round(yaw, 1),
            "width_m": round(self.width, 3),
            "score": round(self.score, 2),
            "grip": round(self.grip, 2),
            "off_center_m": round(self.offset, 3),
            "note": self.note,
        }


# ---- a thing's points -----------------------------------------------------------------------------------------


def _sphere_points() -> np.ndarray:
    """Roughly evenly spread unit vectors (a Fibonacci spiral)."""
    k = np.arange(SPHERE_POINTS) + 0.5
    phi = np.arccos(1 - 2 * k / SPHERE_POINTS)
    theta = math.pi * (1 + 5**0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def geom_points(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> np.ndarray:
    """Points on the surface of geom `g` that the hull of the shape needs, in the world."""
    pos, rot = data.geom_xpos[g], data.geom_xmat[g].reshape(3, 3)
    size, kind = model.geom_size[g], int(model.geom_type[g])
    ring = np.linspace(0, 2 * math.pi, RING_POINTS, endpoint=False)
    if kind == G.mjGEOM_BOX:
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * size
        local = corners
    elif kind == G.mjGEOM_SPHERE:
        local = _sphere_points() * size[0]
    elif kind == G.mjGEOM_ELLIPSOID:
        local = _sphere_points() * size
    elif kind == G.mjGEOM_CYLINDER:
        disc = np.column_stack([size[0] * np.cos(ring), size[0] * np.sin(ring), np.zeros_like(ring)])
        local = np.vstack([disc + [0, 0, size[1]], disc - [0, 0, size[1]]])
    elif kind == G.mjGEOM_CAPSULE:
        caps = _sphere_points() * size[0]
        local = np.vstack([caps + [0, 0, size[1]], caps - [0, 0, size[1]]])
    elif kind == G.mjGEOM_MESH:
        mesh = int(model.geom_dataid[g])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        local = model.mesh_vert[start : start + count]
    else:
        return np.zeros((0, 3))
    return pos + local @ rot.T


def body_points(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    """The world points of every shape that collides on `body` (a thing may be several shapes)."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and _solid(model, g)]
    parts = [geom_points(model, data, g) for g in geoms]
    points = np.vstack(parts) if parts else np.zeros((0, 3))
    if len(points) > MAX_POINTS:
        points = points[np.linspace(0, len(points) - 1, MAX_POINTS).astype(int)]
    return points


def _solid(model: mujoco.MjModel, g: int) -> bool:
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


# ---- the convex hull ------------------------------------------------------------------------------------------


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull of `points`: (vertices, faces), faces as index triples with outward normals. Incremental: a point
    outside the hull removes the faces it sees and closes the hole with faces to it. Raises ValueError for points that
    lie in a plane or a line."""
    pts = np.unique(np.round(np.asarray(points, float), 7), axis=0)
    if len(pts) < 4:
        raise ValueError("too few points for a hull")
    # a first tetrahedron that is not flat
    i0 = 0
    i1 = int(np.argmax(np.linalg.norm(pts - pts[i0], axis=1)))
    line = pts[i1] - pts[i0]
    i2 = int(np.argmax(np.linalg.norm(np.cross(pts - pts[i0], line), axis=1)))
    normal = np.cross(line, pts[i2] - pts[i0])
    if np.linalg.norm(normal) < 1e-9:
        raise ValueError("points on a line")
    i3 = int(np.argmax(np.abs((pts - pts[i0]) @ normal)))
    if abs(float((pts[i3] - pts[i0]) @ normal)) < 1e-9 * max(1.0, np.linalg.norm(normal)):
        raise ValueError("points in a plane")
    centre = pts[[i0, i1, i2, i3]].mean(axis=0)
    faces: list[tuple[int, int, int]] = []

    def add(a: int, b: int, c: int) -> None:
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        faces.append((a, b, c) if n @ (pts[a] - centre) > 0 else (a, c, b))

    for tri in ((i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)):
        add(*tri)
    for p in range(len(pts)):
        if p in (i0, i1, i2, i3):
            continue
        normals = np.array([np.cross(pts[b] - pts[a], pts[c] - pts[a]) for a, b, c in faces])
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1.0
        base = np.array([pts[a] for a, _, _ in faces])
        seen = np.einsum("ij,ij->i", normals / norms[:, None], pts[p] - base) > 1e-9
        if not seen.any():
            continue
        edges: dict[tuple[int, int], int] = {}
        for (a, b, c), gone in zip(faces, seen, strict=True):
            if gone:
                for e in ((a, b), (b, c), (c, a)):
                    edges[e] = edges.get(e, 0) + 1
        horizon = [e for e in edges if (e[1], e[0]) not in edges]
        faces = [f for f, gone in zip(faces, seen, strict=True) if not gone]
        for a, b in horizon:
            add(a, b, p)
    used = sorted({v for f in faces for v in f})
    remap = {v: i for i, v in enumerate(used)}
    return pts[used], np.array([[remap[v] for v in f] for f in faces])


def face_normals(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unit normals, areas) of triangles."""
    cross = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    area = np.linalg.norm(cross, axis=1)
    return cross / np.where(area > 0, area, 1.0)[:, None], area / 2


def hull_volume_centroid(verts: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    """Volume and centroid of a convex hull (a fan of tetrahedra from its middle)."""
    origin = verts.mean(axis=0)
    a, b, c = (verts[faces[:, i]] - origin for i in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    centroids = (a + b + c) / 4.0
    total = float(vol.sum())
    return abs(total), origin + (vol[:, None] * centroids).sum(axis=0) / (total if total else 1.0)


# ---- planning --------------------------------------------------------------------------------------------------


def plan(
    points: np.ndarray,
    opening: float,
    centre_of_mass: np.ndarray | None = None,
    friction: float = FRICTION,
    keep: int = 5,
) -> list[Grasp]:
    """The best `keep` grasps of the thing made of `points` for jaws that open `opening` m, best first (possibly none:
    a thing wider than the opening everywhere, or too flat to get fingers under)."""
    verts, faces = convex_hull(points)
    normals, areas = face_normals(verts, faces)
    _, hull_centre = hull_volume_centroid(verts, faces)
    com = hull_centre if centre_of_mass is None else np.asarray(centre_of_mass, float)
    z_low, z_high = float(verts[:, 2].min()), float(verts[:, 2].max())
    cone = math.cos(math.atan(friction))
    first = z_low + FINGER_CLEAR
    top = z_high - HOLD_DEPTH  # the jaws take hold of at least this much of the thing's height
    levels = [first] if top - first < 0.003 else list(np.linspace(first, top, HEIGHT_LEVELS))
    levels = [z for z in levels if z <= top + 1e-9]
    found: list[Grasp] = []
    for z in levels:
        band = verts[np.abs(verts[:, 2] - z) <= BAND]
        # the hull's own cross-section at z, not only vertices near it: points along the edges that cross the level
        section = _section(verts, faces, z)
        slab = np.vstack([band, section]) if len(section) else band
        if len(slab) < 3:
            continue
        above = verts[verts[:, 2] >= z - BAND]
        for yaw in np.arange(0.0, math.pi, YAW_STEP):
            u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            s = slab @ u
            width = float(s.max() - s.min())
            if width < 0.004 or width > opening - 2 * SPARE:
                continue
            reach = above @ u
            if float(reach.max() - reach.min()) > opening - 2 * SPARE and z < z_high - BAND:
                continue  # the fingers cannot come down on either side: the thing gets wider above the grasp
            a = slab[s <= s.min() + FACE_DEPTH].mean(axis=0)  # the middle of each side the jaws meet
            b = slab[s >= s.max() - FACE_DEPTH].mean(axis=0)
            cos_a, area_a = _contact(verts, faces, normals, areas, a, -u, z)
            cos_b, area_b = _contact(verts, faces, normals, areas, b, u, z)
            grip = min(cos_a, cos_b)
            if grip < cone:
                continue  # the squeeze would slide the thing out
            mid = (a + b) / 2
            across = np.array([-u[1], u[0], 0.0])
            side = abs(float((mid - com) @ across))  # the grasp's line passes this far from the centre of mass sideways
            along = abs(float((mid - com) @ u))
            high = abs(float(mid[2] - com[2]))
            flat = min(1.0, (area_a + area_b) / (2 * 0.0006))  # the contact faces: 6 cm2 or more is as good as it gets
            spare = (opening - width) / opening
            score = (
                2.0 * (grip - cone) / (1 - cone)
                + 2.0 * (1.0 - min(1.0, math.hypot(side, along) / 0.04))
                + 0.6 * (1.0 - min(1.0, high / 0.05))
                + 0.8 * flat
                + 0.5 * min(1.0, spare / 0.4)
            )
            center = np.array([mid[0], mid[1], z])
            found.append(Grasp(center, u, width, z - z_low, score, grip, side, _note(width, opening, grip, flat, side)))
    found.sort(key=lambda g: -g.score)
    out: list[Grasp] = []
    for g in found:  # distinct enough: not the same grasp turned a few degrees
        if all(abs(g.center[2] - o.center[2]) > 0.01 or _turn(g.closing, o.closing) > math.radians(20) for o in out):
            out.append(g)
        if len(out) >= keep:
            break
    return out


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(min(1.0, abs(float(a @ b))))


def _section(verts: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """Points where the hull's edges cross the horizontal plane at height z."""
    out = []
    for f in faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            p, q = verts[f[i]], verts[f[j]]
            if (p[2] - z) * (q[2] - z) < 0:
                t = (z - p[2]) / (q[2] - p[2])
                out.append(p + t * (q - p))
    return np.array(out) if out else np.zeros((0, 3))


def _contact(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    p: np.ndarray,
    toward: np.ndarray,
    z: float,
) -> tuple[float, float]:
    """How well the surface at `p` meets a finger pushing along `toward` (the outward direction of that side): the
    cosine between the best face's outward normal and `toward`, and that face's area (m2). Looked at among the faces
    that touch the grasp height and are near `p`."""
    lo = verts[faces].min(axis=1)[:, 2]
    hi = verts[faces].max(axis=1)[:, 2]
    near_height = (lo <= z + BAND) & (hi >= z - BAND)
    centroids = verts[faces].mean(axis=1)
    near = near_height & (np.linalg.norm(centroids[:, :2] - p[:2], axis=1) < 0.03 + 1e-9)
    if not near.any():
        near = near_height
    if not near.any():
        return 0.0, 0.0
    cos = normals[near] @ toward
    k = int(np.argmax(cos))
    idx = np.nonzero(near)[0][k]
    # a flat side is several triangles with the same normal: add the area of those facing the same way
    same = (normals[near] @ normals[idx] > 0.98) & near[near]
    return float(cos[k]), float(areas[near][same].sum())


def _note(width: float, opening: float, grip: float, flat: float, side: float) -> str:
    bits = [f"{1000 * width:.0f} mm across of {1000 * opening:.0f} mm open"]
    bits.append("flat faces" if flat > 0.6 else "small faces")
    bits.append("squeezes square" if grip > 0.95 else "faces slightly slanted")
    if side > 0.02:
        bits.append(f"{1000 * side:.0f} mm from the centre of mass")
    return ", ".join(bits)
