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

import hashlib
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
LINE_STEP = 0.01  # m between the lines across a thing that the jaws could close along
PALM = 0.035  # m the thing may reach above the grasp height: past that it runs into the hand's palm
HALF_FINGER = 0.012  # m: half the width of a fingertip, and some
FINGER_CLEAR = 0.02  # m the grasp keeps above the lowest point: a fingertip reaches about 1 cm below the tool point
SPARE = 0.006  # m of opening the jaws keep to spare on either side of a grasp, and above it
FRICTION = 0.6  # the friction of fingertip pads on most things; the friction cone's half angle is atan of it
SPHERE_POINTS = 120
RING_POINTS = 28
HULL_DIRECTIONS = 200  # a mesh with more points than twice this is cut down to its extremes along this many directions
CACHE_SIZE = 16  # plans kept, so asking about a grasp and then picking it works the thing out once
CHUNK = 20000  # points looked at at once when cutting a mesh down


@dataclass(frozen=True)
class Grasp:
    """One way to take hold: the tool point goes to `center`, the jaws close along `closing`."""

    center: np.ndarray  # world, between the contact faces at the grasp height
    closing: np.ndarray  # unit, horizontal
    width: float  # m between the contact faces
    below: float  # m from the center down to the thing's lowest point
    above: float  # m from the center up to the thing's highest point
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


def _sphere_points(count: int = SPHERE_POINTS) -> np.ndarray:
    """Roughly evenly spread unit vectors (a Fibonacci spiral)."""
    k = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * k / count)
    theta = math.pi * (1 + 5**0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def extremes(points: np.ndarray) -> np.ndarray:
    """`points` cut down to the ones farthest out along a spread of directions (and the opposite of each): the hull of
    those is the hull of the whole to within a millimetre or two on a hand-sized thing, and takes milliseconds where
    the hull of a dense mesh would take minutes. A cloud of few points is left as it is."""
    if len(points) <= 2 * HULL_DIRECTIONS:
        return points
    base = _sphere_points(HULL_DIRECTIONS)
    dirs = np.vstack([base, -base, np.eye(3), -np.eye(3)])
    best = np.full(len(dirs), -np.inf)
    index = np.zeros(len(dirs), int)
    for start in range(0, len(points), CHUNK):
        proj = points[start : start + CHUNK] @ dirs.T
        top = proj.argmax(axis=0)
        value = proj[top, np.arange(len(dirs))]
        better = value > best
        best[better], index[better] = value[better], top[better] + start
    return points[np.unique(index)]


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


def body_parts(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> list[np.ndarray]:
    """The world points of each shape that collides on `body`, one array per shape (a thing may be several)."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body and _solid(model, g)]
    return [extremes(p) for p in (geom_points(model, data, g) for g in geoms) if len(p)]


def _solid(model: mujoco.MjModel, g: int) -> bool:
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


# ---- the convex hull ------------------------------------------------------------------------------------------


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The convex hull of `points`: (vertices, faces), faces as index triples with outward normals. Incremental: a point
    outside the hull removes the faces it sees and closes the hole with faces to it; the planes of the faces are kept
    as arrays, so a point is tested against every face in one product. Raises ValueError for points that lie in a plane
    or a line."""
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

    def build(tris: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(faces turned to look outward, their unit normals, their offsets) for triangles of point indices."""
        tri = np.array(tris)
        n = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
        flip = np.einsum("ij,ij->i", n, pts[tri[:, 0]] - centre) <= 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        n[flip] = -n[flip]
        length = np.linalg.norm(n, axis=1)
        n = n / np.where(length > 0, length, 1.0)[:, None]
        return tri, n, np.einsum("ij,ij->i", n, pts[tri[:, 0]])

    faces, normals, offsets = build([(i0, i1, i2), (i0, i1, i3), (i0, i2, i3), (i1, i2, i3)])
    # the points far from the middle first: the hull is nearly its final size after a few, the rest mostly fall inside
    order = np.argsort(-np.linalg.norm(pts - centre, axis=1))
    for p in order:
        if p in (i0, i1, i2, i3):
            continue
        seen = normals @ pts[p] - offsets > 1e-9
        if not seen.any():
            continue
        edges = {e for a, b, c in faces[seen].tolist() for e in ((a, b), (b, c), (c, a))}
        horizon = [(a, b) for a, b in edges if (b, a) not in edges]
        added, new_normals, new_offsets = build([(a, b, int(p)) for a, b in horizon])
        faces = np.vstack([faces[~seen], added])
        normals = np.vstack([normals[~seen], new_normals])
        offsets = np.concatenate([offsets[~seen], new_offsets])
    used = np.unique(faces)
    remap = np.zeros(len(pts), int)
    remap[used] = np.arange(len(used))
    return pts[used], remap[faces]


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


def _hulls(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """(all the vertices, all the faces, each shape's own hull, the centre of mass by volume) of a thing made of
    `parts`. Each shape keeps its own hull, so a thing made of several (a post with a plate on it) is not filled in
    between them."""
    own: list[tuple[np.ndarray, np.ndarray]] = []
    loose: list[np.ndarray] = []
    weighted, volumes = np.zeros(3), 0.0
    for part in parts:
        try:
            verts, faces = convex_hull(part)
        except ValueError:  # a flat or thin shape: its points count, its faces do not
            loose.append(part)
            continue
        own.append((verts, faces))
        volume, centre = hull_volume_centroid(verts, faces)
        weighted += volume * centre
        volumes += volume
    if not own:
        raise ValueError("the thing has no volume to take hold of")
    centre = weighted / volumes
    offset, all_faces = 0, []
    for verts, faces in own:
        all_faces.append(faces + offset)
        offset += len(verts)
    verts = np.vstack([v for v, _ in own] + loose)
    return verts, np.vstack(all_faces), own, centre


def plan(
    parts: np.ndarray | list[np.ndarray],
    opening: float,
    centre_of_mass: np.ndarray | None = None,
    friction: float = FRICTION,
    keep: int = 5,
) -> list[Grasp]:
    """The best `keep` grasps of the thing made of `parts` (the points of each shape; one array is one shape) for jaws
    that open `opening` m, best first (possibly none: a thing wider than the opening everywhere, or too flat to get
    fingers under)."""
    parts = [np.asarray(parts, float)] if isinstance(parts, np.ndarray) else [np.asarray(p, float) for p in parts]
    parts = [extremes(p) for p in parts]
    verts, faces, own, hull_centre = _hulls(parts)
    normals, areas = face_normals(verts, faces)
    com = hull_centre if centre_of_mass is None else np.asarray(centre_of_mass, float)
    solids = [p for p in parts if len(p)]
    z_low, z_high = float(verts[:, 2].min()), float(verts[:, 2].max())
    corners = verts[faces]  # per face, once: its lowest and highest point and its middle
    table = (corners[:, :, 2].min(axis=1), corners[:, :, 2].max(axis=1), corners.mean(axis=1))
    cone = math.cos(math.atan(friction))
    first = z_low + FINGER_CLEAR
    top = z_high - HOLD_DEPTH  # the jaws take hold of at least this much of the thing's height
    first = max(first, z_high - PALM)
    levels = [first] if top - first < 0.003 else list(np.linspace(first, top, HEIGHT_LEVELS))
    levels = [z for z in levels if z <= top + 1e-9]
    found: list[Grasp] = []
    for z in levels:
        # the thing's cross-section at this height: one convex polygon per shape that reaches it
        polys = [poly for poly in (_slice(v, f, z) for v, f in own) if len(poly) >= 3]
        if not polys:
            continue
        for yaw in np.arange(0.0, math.pi, YAW_STEP):
            u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            across = np.array([-u[1], u[0], 0.0])
            for line in _lines(polys, across):
                # the part of the cross-section the two fingers meet: the strip HALF_FINGER either side of the line
                # the jaws close along (a T is taken by its stem or its bar, not across both)
                reach = [clip for clip in (_strip(poly, across, line) for poly in polys) if len(clip)]
                if not reach:
                    continue
                points = np.vstack(reach)
                s = points @ u
                width = float(s.max() - s.min())
                if width < 0.004 or width > opening - 2 * SPARE:
                    continue
                a = points[s <= s.min() + FACE_DEPTH].mean(axis=0)  # the middle of each side the jaws meet
                b = points[s >= s.max() - FACE_DEPTH].mean(axis=0)
                if _blocked(solids, (a + b) / 2, u, z, opening - 2 * SPARE):
                    continue  # the fingers cannot come down on either side: the thing gets wider above the grasp
                cos_a, area_a = _contact(*table, normals, areas, a, -u, z)
                cos_b, area_b = _contact(*table, normals, areas, b, u, z)
                grip = min(cos_a, cos_b)
                if grip < cone:
                    continue  # the squeeze would slide the thing out
                mid = (a + b) / 2
                side = abs(float((mid - com) @ across))  # the grasp's line passes this far from the centre of mass
                along = abs(float((mid - com) @ u))
                high = abs(float(mid[2] - com[2]))
                flat = min(1.0, (area_a + area_b) / (2 * 0.0006))  # contact faces: 6 cm2 or more is as good as it gets
                spare = (opening - width) / opening
                score = (
                    2.0 * (grip - cone) / (1 - cone)
                    + 2.0 * (1.0 - min(1.0, math.hypot(side, along) / 0.04))
                    + 0.6 * (1.0 - min(1.0, high / 0.05))
                    + 0.8 * flat
                    + 0.5 * min(1.0, spare / 0.4)
                )
                center = np.array([mid[0], mid[1], z])
                note = _note(width, opening, grip, flat, side)
                found.append(Grasp(center, u, width, z - z_low, z_high - z, score, grip, side, note))
    found.sort(key=lambda g: -g.score)
    out: list[Grasp] = []
    for g in found:  # distinct enough: not the same grasp turned a few degrees
        if all(abs(g.center[2] - o.center[2]) > 0.01 or _turn(g.closing, o.closing) > math.radians(20) for o in out):
            out.append(g)
        if len(out) >= keep:
            break
    return out


def _blocked(solids: list[np.ndarray], mid: np.ndarray, u: np.ndarray, z: float, room: float) -> bool:
    """Whether open jaws (`room` m between the fingers, closing along `u` about the point `mid`) cannot come down to
    the grasp height `z`: some shape of the thing that reaches above it, in the way of the fingers (HALF_FINGER either
    side of the line the jaws close along), is wider along `u` there than they open."""
    across = np.array([-u[1], u[0], 0.0])
    centre = float(mid @ across)
    reach: list[np.ndarray] = []
    for verts in solids:
        if float(verts[:, 2].max()) < z - BAND:
            continue  # all of it is below the grasp
        c = verts @ across
        if float(c.max()) >= centre - HALF_FINGER and float(c.min()) <= centre + HALF_FINGER:
            reach.append(verts @ u)
    if not reach:
        return False
    allong = np.concatenate(reach)
    return float(allong.max() - allong.min()) > room


def _turn(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(min(1.0, abs(float(a @ b))))


def _slice(verts: np.ndarray, faces: np.ndarray, z: float) -> np.ndarray:
    """The convex polygon (n x 3, in order round) where a hull meets the horizontal plane at height z."""
    cut = _section(verts, faces, z)
    if len(cut) < 3:
        return np.zeros((0, 3))
    cut = np.unique(np.round(cut, 7), axis=0)
    centre = cut.mean(axis=0)
    order = np.argsort(np.arctan2(cut[:, 1] - centre[1], cut[:, 0] - centre[0]))
    return cut[order]


def _lines(polys: list[np.ndarray], across: np.ndarray) -> list[float]:
    """Offsets (along `across`) of the lines the jaws could close along: a line through every shape's middle, and
    lines LINE_STEP apart over the whole cross-section."""
    spans = [(float((p @ across).min()), float((p @ across).max())) for p in polys]
    low, high = min(lo for lo, _ in spans), max(hi for _, hi in spans)
    offsets = [(lo + hi) / 2 for lo, hi in spans]
    if high - low > 2 * HALF_FINGER:
        offsets += list(np.arange(low + HALF_FINGER, high - HALF_FINGER + 1e-9, LINE_STEP))
    return sorted(set(round(o, 5) for o in offsets))


def _strip(poly: np.ndarray, across: np.ndarray, centre: float) -> np.ndarray:
    """`poly` (convex, n x 3) cut to the strip HALF_FINGER either side of the line at offset `centre`."""
    out = poly
    for sign, edge in ((1.0, centre - HALF_FINGER), (-1.0, -(centre + HALF_FINGER))):
        d = sign * (out @ across) - edge  # inside where d >= 0
        keep: list[np.ndarray] = []
        for i in range(len(out)):
            j = (i + 1) % len(out)
            if d[i] >= 0:
                keep.append(out[i])
            if (d[i] >= 0) != (d[j] >= 0):
                t = d[i] / (d[i] - d[j])
                keep.append(out[i] + t * (out[j] - out[i]))
        if not keep:
            return np.zeros((0, 3))
        out = np.array(keep)
    return out


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
    low: np.ndarray,
    high: np.ndarray,
    centroids: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    p: np.ndarray,
    toward: np.ndarray,
    z: float,
) -> tuple[float, float]:
    """How well the surface at `p` meets a finger pushing along `toward` (the outward direction of that side): the
    cosine between the best face's outward normal and `toward`, and that face's area (m2). Looked at among the faces
    that touch the grasp height and are near `p`."""
    near_height = (low <= z + BAND) & (high >= z - BAND)
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


_cache: dict[str, list[Grasp]] = {}


def plan_cached(
    parts: list[np.ndarray], opening: float, centre_of_mass: np.ndarray | None = None, keep: int = 5
) -> list[Grasp]:
    """`plan`, remembered by the points, the opening and the centre of mass (to a tenth of a millimetre): the same
    thing in the same place is worked out once. Raises ValueError like `plan`."""
    digest = hashlib.sha1(usedforsecurity=False)
    for part in parts:
        digest.update(np.round(np.asarray(part, float), 4).tobytes())
        digest.update(b"|")
    digest.update(repr((round(float(opening), 5), keep)).encode())
    if centre_of_mass is not None:
        digest.update(np.round(np.asarray(centre_of_mass, float), 4).tobytes())
    key = digest.hexdigest()
    if key not in _cache:
        if len(_cache) >= CACHE_SIZE:
            _cache.pop(next(iter(_cache)))
        _cache[key] = plan(parts, opening, centre_of_mass, keep=keep)
    return _cache[key]
