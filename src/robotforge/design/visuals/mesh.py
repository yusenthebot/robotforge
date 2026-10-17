"""Procedural triangle meshes for visual-only geometry, built with numpy alone (no OpenGL needed).

Every builder returns an immutable `Mesh`: vertices, one normal per vertex and outward
triangles. Hard edges are made by giving each side its own vertices, so both smooth and
crisp edges reach MuJoCo and the browser viewer unchanged. Most shapes are lofts: a stack of
closed rings joined by quads and capped at both ends (boxes with rounded edges, lathed parts,
tapered links, lumpy foliage).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


def _key(value):
    """A hashable cache key: floats rounded to 0.1 micrometer, sequences as tuples."""
    if isinstance(value, float):
        return round(value, 7)
    if isinstance(value, (list, tuple, range)):
        return tuple(_key(v) for v in value)
    return value


def cached(fn):
    """Memoize a pure mesh builder; float noise below 0.1 micrometer still hits the cache.

    Recompiling a world (a robot moved, an object added) then rebuilds no geometry at all.
    The returned meshes are shared, so their arrays are made read-only.
    """
    memo = functools.lru_cache(maxsize=4096)(lambda args, kwargs: _frozen(fn(*args, **dict(kwargs))))

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return memo(_key(args), tuple(sorted((k, _key(v)) for k, v in kwargs.items())))

    wrapper.cache_clear = memo.cache_clear  # type: ignore[attr-defined]
    return wrapper


def _frozen(result):
    meshes = result.values() if isinstance(result, dict) else [result]
    for m in meshes:
        if isinstance(m, Mesh):
            for arr in (m.vertices, m.normals, m.faces, m.keys):
                arr.flags.writeable = False
    return result


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.cross for (..., 3) arrays without its per-call overhead (it dominates small meshes)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


@dataclass(frozen=True, eq=False)
class Mesh:
    vertices: np.ndarray  # (n, 3) float64
    normals: np.ndarray  # (n, 3) float64, unit
    faces: np.ndarray  # (m, 3) int32, counter-clockwise seen from outside
    keys: np.ndarray  # (n,) int64: vertices with the same key share one smooth normal

    @property
    def triangles(self) -> int:
        return int(self.faces.shape[0])

    def moved(
        self,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        rot: np.ndarray | None = None,
        scale: Sequence[float] | float | None = None,
    ) -> Mesh:
        """Scale (in the mesh frame), then rotate by the 3x3 matrix `rot`, then translate."""
        v, n = self.vertices, self.normals
        if scale is not None:
            s = np.broadcast_to(np.asarray(scale, dtype=float), (3,))
            v = v * s
            n = n / s  # normals follow the inverse transpose
            if np.prod(s) < 0:
                faces = self.faces[:, ::-1]
                return Mesh(v, _unit(n), faces, self.keys).moved(pos, rot)
            n = _unit(n)
        if rot is not None:
            v = v @ np.asarray(rot).T
            n = n @ np.asarray(rot).T
        v = v + np.asarray(pos, dtype=float)
        return Mesh(v, n, self.faces, self.keys)

    def faceted(self) -> Mesh:
        """Flat-shaded copy: every triangle gets its own vertices (a low-poly, chiselled look)."""
        v = self.vertices[self.faces].reshape(-1, 3)
        f = np.arange(len(v), dtype=np.int32).reshape(-1, 3)
        keys = np.arange(len(v), dtype=np.int64)
        return Mesh(v, _smooth_normals(v, f, keys), f, keys)

    def displaced(self, offset: np.ndarray) -> Mesh:
        """Move every vertex by `offset` (n, 3) and recompute the smooth normals."""
        v = self.vertices + offset
        return Mesh(v, _smooth_normals(v, self.faces, self.keys), self.faces, self.keys)


def merge(meshes: Iterable[Mesh]) -> Mesh:
    meshes = [m for m in meshes if m.faces.size]
    if not meshes:
        return empty()
    verts, norms, faces, keys = [], [], [], []
    offset, key_offset = 0, 0
    for m in meshes:
        verts.append(m.vertices)
        norms.append(m.normals)
        faces.append(m.faces + offset)
        keys.append(m.keys + key_offset)
        offset += len(m.vertices)
        key_offset += int(m.keys.max()) + 1 if m.keys.size else 0
    return Mesh(np.vstack(verts), np.vstack(norms), np.vstack(faces).astype(np.int32), np.concatenate(keys))


def empty() -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64))


# ---- rotations ---------------------------------------------------------------------------------


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


Z_TO_Y = rot_x(-math.pi / 2)  # turns a shape built along +z so it runs along +y
Z_TO_X = rot_y(math.pi / 2)  # ... along +x


def quat_to_mat(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(q) for q in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def z_to(direction: Sequence[float]) -> np.ndarray:
    """Rotation turning +z onto `direction`."""
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)
    z = np.array([0.0, 0.0, 1.0])
    axis = _cross(z, d)
    s, c = float(np.linalg.norm(axis)), float(z @ d)
    if s < 1e-9:
        return np.eye(3) if c > 0 else rot_x(math.pi)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


# ---- 2D rings ----------------------------------------------------------------------------------


def circle(r: float, n: int) -> np.ndarray:
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def rounded_rect(hx: float, hy: float, r: float, per_corner: int = 4) -> np.ndarray:
    """Counter-clockwise rounded rectangle; always 4 * (per_corner + 1) points, so rings of
    different sizes line up point for point in a loft."""
    r = max(min(r, hx, hy), 1e-5)
    cx, cy = max(hx - r, 0.0), max(hy - r, 0.0)
    pts = []
    for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        a0 = k * math.pi / 2
        for a in np.linspace(a0, a0 + math.pi / 2, per_corner + 1):
            pts.append((sx * cx + r * math.cos(a), sy * cy + r * math.sin(a)))
    return np.array(pts)


def at_z(ring2: np.ndarray, z: float) -> np.ndarray:
    return np.column_stack([ring2, np.full(len(ring2), z)])


# ---- loft --------------------------------------------------------------------------------------


def loft(
    rings: Sequence[np.ndarray],
    sharp: Iterable[int] = (),
    cap_start: bool = True,
    cap_end: bool = True,
    closed: bool = True,
) -> Mesh:
    """Join closed 3D rings (each (n, 3), same n) in order into a surface.

    Rings listed in `sharp` get a crease: the bands on either side do not share normals there.
    Caps are flat fans from each end ring's centroid. Faces are turned outward for closed shapes.
    With `closed=False` the rings are open polylines (no wrap-around band, no caps).
    """
    rings = [np.asarray(r, dtype=float) for r in rings]
    n = len(rings[0])
    sharp = sorted({i for i in sharp if 0 < i < len(rings) - 1})
    bounds = [0, *sharp, len(rings) - 1]
    parts = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        parts.append(_band(rings[a : b + 1], n, closed))
    if closed and cap_start and np.ptp(rings[0], axis=0).max() > 1e-9:
        parts.append(_cap(rings[0], reverse=True))
    if closed and cap_end and np.ptp(rings[-1], axis=0).max() > 1e-9:
        parts.append(_cap(rings[-1], reverse=False))
    mesh = merge(parts)
    if closed and _signed_volume(rings, n) < 0:
        mesh = Mesh(mesh.vertices, mesh.normals, mesh.faces[:, ::-1].copy(), mesh.keys)
    return _finish(mesh)


def _band(rings: list[np.ndarray], n: int, closed: bool) -> Mesh:
    verts = np.vstack(rings)
    m = len(rings)
    i = np.arange(n if closed else n - 1)
    j = (i + 1) % n
    point = np.array([np.ptp(r, axis=0).max() < 1e-12 for r in rings])
    k = np.arange(m - 1)[:, None] * n  # every band between two rings at once (a loop of np.stack was slow)
    a, b, c, d = k + i, k + j, k + n + j, k + n + i
    quads = np.stack([np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)], axis=1)  # (bands, 2, n, 3)
    keep = np.stack([~point[:-1], ~point[1:]], axis=1)  # next to a pole one triangle per quad has no area
    faces = quads[keep].reshape(-1, 3).astype(np.int32)
    return Mesh(verts, np.zeros_like(verts), faces, _position_keys(verts))


def _cap(ring: np.ndarray, reverse: bool) -> Mesh:
    n = len(ring)
    center = ring.mean(axis=0)
    verts = np.vstack([center[None, :], ring])
    i = np.arange(n)
    faces = np.stack([np.zeros(n, dtype=int), 1 + i, 1 + (i + 1) % n], axis=1)
    if reverse:
        faces = faces[:, ::-1]
    return Mesh(verts, np.zeros_like(verts), faces.astype(np.int32), _position_keys(verts))


def _signed_volume(rings: list[np.ndarray], n: int) -> float:
    """Volume enclosed by the side bands plus end caps, with the faces as `loft` builds them."""
    stack = np.stack(rings)
    i = np.arange(n)
    j = (i + 1) % n
    a, b = stack[:-1, i], stack[:-1, j]
    c, d = stack[1:, j], stack[1:, i]
    vol = float(np.sum(a * _cross(b, c)) + np.sum(a * _cross(c, d)))
    for ring, reverse in ((stack[0], True), (stack[-1], False)):
        center = ring.mean(axis=0)
        p, q = (ring[j], ring[i]) if reverse else (ring[i], ring[j])
        vol += float(np.sum(center * _cross(p, q)))
    return vol / 6.0


def _position_keys(verts: np.ndarray) -> np.ndarray:
    """Same key for vertices at the same spot (poles, wrap seams), so they share a normal.

    The keys are np.unique(q, axis=0)'s inverse, found with a lexsort, which is several times faster."""
    q = np.round(verts * 1e7).astype(np.int64)
    if len(q) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    ranked = q[order]
    new = np.ones(len(q), dtype=bool)
    new[1:] = np.any(ranked[1:] != ranked[:-1], axis=1)
    keys = np.empty(len(q), dtype=np.int64)
    keys[order] = np.cumsum(new) - 1
    return keys


def _finish(mesh: Mesh) -> Mesh:
    return Mesh(mesh.vertices, _smooth_normals(mesh.vertices, mesh.faces, mesh.keys), mesh.faces, mesh.keys)


def _smooth_normals(verts: np.ndarray, faces: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Area-weighted face normals summed per smoothing key."""
    if faces.size == 0:
        return np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = _cross(b - a, c - a)  # length = 2 * area
    acc = np.zeros((int(keys.max()) + 1, 3))
    for k in range(3):
        np.add.at(acc, keys[faces[:, k]], fn)
    normals = acc[keys]
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.array([0.0, 0.0, 1.0])
    return np.where(length > _EPS, normals / np.maximum(length, _EPS), fallback)


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(length, _EPS)


# ---- shapes ------------------------------------------------------------------------------------


@cached
def lathe(profile: Sequence[tuple[float, float]], segments: int = 32, sharp: Iterable[int] = ()) -> Mesh:
    """Surface of revolution about z from (radius, z) points, bottom to top.

    End points with radius 0 close the shape with a pole; otherwise the ends get flat caps.
    """
    rings = [at_z(circle(max(r, 0.0), segments), z) for r, z in profile]
    return loft(rings, sharp=sharp)


@cached
def rounded_box(hx: float, hy: float, hz: float, radius: float, segments: int = 3, corner: float | None = None) -> Mesh:
    """Box of half extents (hx, hy, hz) with every edge rounded by `radius`.

    `corner` sets the vertical edges' radius separately (defaults to `radius`).
    """
    rb = max(min(radius, hx, hy, hz), 1e-5)
    rc = rb if corner is None else max(min(corner, hx, hy), 1e-5)
    per = max(2, segments)
    rings = []
    angles = np.linspace(0.0, math.pi / 2, segments + 1)
    for a in angles:  # bottom bevel, from the bottom face up to the side
        inset = rb * (1 - math.sin(a))
        z = -(hz - rb) - rb * math.cos(a)
        rings.append(at_z(rounded_rect(hx - inset, hy - inset, max(rc - inset, 1e-5), per), z))
    for a in angles[::-1]:
        inset = rb * (1 - math.sin(a))
        z = (hz - rb) + rb * math.cos(a)
        rings.append(at_z(rounded_rect(hx - inset, hy - inset, max(rc - inset, 1e-5), per), z))
    return loft(rings)


@cached
def slab(
    hx: float,
    hy: float,
    z0: float,
    z1: float,
    bottom: float,
    top: float,
    corner: float,
    segments: int = 3,
    per_corner: int = 3,
    fence: bool = True,
    bottom_cap: bool = True,
    top_cap: bool = True,
    rims: tuple[float, float] = (0.0, 0.0),
) -> Mesh:
    """A rounded-rectangle block from z0 to z1 with its own edge radius at the bottom and at the top.

    Two slabs stacked with a small gap make a shell with a parting line: a soft outer edge and a
    crisp seam, which one rounded box cannot do. `corner` is the radius of the vertical edges.
    With `fence` every flat face is bordered by a thin strip (see _fenced), so a big face shades
    flat instead of fading into the curve beside it; small parts can do without.

    Without `bottom_cap` or `top_cap` that end is open: the faces at a parting line look into the
    shell, where only a camera inside it could see them. `rims` keeps a flat border that wide round
    the open bottom and top, so a narrow gap beside it still closes.
    """
    h = (z1 - z0) / 2
    rb = max(min(bottom, hx, hy, h), 1e-5)
    rt = max(min(top, hx, hy, h), 1e-5)

    def ring(inset: float, z: float) -> np.ndarray:
        rect = rounded_rect(hx - inset, hy - inset, max(corner - inset, 1e-5), per_corner)
        return at_z(_fenced(rect) if fence else rect, z)

    angles = np.linspace(0.0, math.pi / 2, segments + 1)
    rings = [ring(rb * (1 - math.sin(a)), z0 + rb * (1 - math.cos(a))) for a in angles]
    straight = (z1 - rt) - (z0 + rb)
    if fence and straight > 1e-5:  # fence the flat sides top and bottom as well
        rings += [ring(0.0, z0 + rb + straight * _FENCE), ring(0.0, z1 - rt - straight * _FENCE)]
    rings += [ring(rt * (1 - math.sin(a)), z1 - rt * (1 - math.cos(a))) for a in angles[::-1]]
    inner = 0.8 * min(hx, hy)  # a rim never closes the opening
    creases = []  # a rim is creased off its edge, so the edge shades as it does next to a cap
    if not bottom_cap and rims[0] > 0:
        rings.insert(0, ring(min(rb + rims[0], inner), z0))
        creases.append(1)
    if not top_cap and rims[1] > 0:
        rings.append(ring(min(rt + rims[1], inner), z1))
        creases.append(len(rings) - 2)
    return loft(rings, sharp=creases, cap_start=bottom_cap, cap_end=top_cap)


_FENCE = 0.004  # share of a flat face given to the strip that fences it off from the curve next to it


def _fenced(ring2: np.ndarray) -> np.ndarray:
    """A rounded rectangle with two extra points on each straight side, just inside its ends.

    Smooth normals average the faces around a vertex; without the extra points the vertex where a
    corner meets a flat side leans toward the corner, and that lean is interpolated across the
    whole side: a long flat face shades like a shallow curve. The thin strip takes the blend.
    """
    corners = 4
    per = len(ring2) // corners
    out = []
    for k in range(corners):
        arc = ring2[k * per : (k + 1) * per]
        out.extend(arc)
        a, b = arc[-1], ring2[((k + 1) * per) % len(ring2)]
        out += [a + (b - a) * _FENCE, b + (a - b) * _FENCE]
    return np.array(out)


@cached
def box(hx: float, hy: float, hz: float, chamfer: float = 0.0) -> Mesh:
    """A box with crisp (or slightly chamfered) edges, for planks, slats and seams."""
    if chamfer > 0:
        return rounded_box(hx, hy, hz, chamfer, segments=1, corner=chamfer)
    corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    rings = [at_z(corners, z) for z in (-hz, hz)]
    # Each side face gets its own vertices: crisp edges everywhere.
    sides = []
    for k in range(4):
        p0, p1 = corners[k], corners[(k + 1) % 4]
        quad = np.array([[*p0, -hz], [*p1, -hz], [*p1, hz], [*p0, hz]])
        sides.append(_quad(quad))
    sides.append(_cap(rings[0], reverse=True))
    sides.append(_cap(rings[1], reverse=False))
    return _finish(merge(sides))


def _quad(p: np.ndarray) -> Mesh:
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Mesh(p, np.zeros_like(p), faces, np.arange(4))


@cached
def cylinder(r: float, h: float, segments: int = 24, bevel: float = 0.0, z0: float | None = None, arc: int = 3) -> Mesh:
    """Cylinder along z of radius r, centered (or from z0 to z0 + h), with an optional rounded rim
    (`arc` segments per rounded edge; 1 makes a chamfer)."""
    lo = -h / 2 if z0 is None else z0
    hi = lo + h
    if bevel <= 0:
        return lathe([(0, lo), (r, lo), (r, hi), (0, hi)], segments, sharp=(1, 2))
    b = min(bevel, r * 0.5, h * 0.5)
    prof = [(0.0, lo)]
    for a in np.linspace(0, math.pi / 2, arc + 1):
        prof.append((r - b + b * math.sin(a), lo + b - b * math.cos(a)))
    for a in np.linspace(math.pi / 2, 0, arc + 1):
        prof.append((r - b + b * math.sin(a), hi - b + b * math.cos(a)))
    prof.append((0.0, hi))
    # With a chamfer the flat faces keep crisp edges; rounded rims stay smooth.
    return lathe(prof, segments, sharp=range(1, len(prof) - 1) if arc == 1 else ())


@cached
def sphere(r: float, rings: int = 12, segments: int = 24) -> Mesh:
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, segments)


@cached
def capsule(r: float, length: float, rings: int = 6, segments: int = 20) -> Mesh:
    """Capsule along z from 0 to `length` (the end caps reach r beyond)."""
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi / 2, rings + 1)]
    prof += [(r * math.sin(t), length - r * math.cos(t)) for t in np.linspace(math.pi / 2, math.pi, rings + 1)]
    return lathe(prof, segments)


@cached
def link(
    length: float,
    start: tuple[float, float],
    end: tuple[float, float],
    roundness: float = 0.6,
    rounding: tuple[float, float] = (1.0, 1.0),
    stations: int = 5,
    per_corner: int = 3,
    domes: int = 4,
) -> Mesh:
    """A tapered limb along z from 0 to `length`: rounded-rectangle sections from half extents
    `start` (x, y) to `end`, ends domed by `rounding` times the end section's smaller half extent.
    A thick limb needs more `per_corner` points round its section and more `domes` rings per end."""
    rings = []
    d0 = rounding[0] * min(start)
    d1 = rounding[1] * min(end)
    count = 4 * (per_corner + 1)
    for a in np.linspace(0.0, math.pi / 2, domes + 1):  # bottom dome
        k = math.sin(a)
        hx, hy = start[0] * k, start[1] * k
        z = d0 * (1 - math.cos(a))
        rings.append(
            at_z(rounded_rect(hx, hy, roundness * min(hx, hy), per_corner), z) if k > 1e-9 else _point(count, z)
        )
    for t in np.linspace(0.0, 1.0, stations)[1:-1]:
        z = d0 + t * (length - d0 - d1)
        hx = start[0] + (end[0] - start[0]) * t
        hy = start[1] + (end[1] - start[1]) * t
        rings.append(at_z(rounded_rect(hx, hy, roundness * min(hx, hy), per_corner), z))
    for a in np.linspace(math.pi / 2, 0.0, domes + 1):  # top dome
        k = math.sin(a)
        hx, hy = end[0] * k, end[1] * k
        z = length - d1 * (1 - math.cos(a))
        rings.append(
            at_z(rounded_rect(hx, hy, roundness * min(hx, hy), per_corner), z) if k > 1e-9 else _point(count, z)
        )
    return loft(rings)


def _point(count: int, z: float) -> np.ndarray:
    return np.tile([0.0, 0.0, z], (count, 1))


@cached
def torus(r_major: float, r_minor: float, segments: int = 32, sides: int = 10) -> Mesh:
    """Ring around z."""
    prof = [
        (r_major + r_minor * math.cos(t), r_minor * math.sin(t))
        for t in np.linspace(-math.pi, math.pi, sides, endpoint=False)
    ]
    rings = [at_z(circle(r, segments), z) for r, z in prof]
    return _finish(_closed_tube(rings, segments))


def _closed_tube(rings: list[np.ndarray], n: int) -> Mesh:
    """Rings joined in a loop (last back to first): a torus-like closed surface."""
    m = len(rings)
    verts = np.vstack(rings)
    i = np.arange(n)
    j = (i + 1) % n
    faces = []
    for k in range(m):
        k1 = (k + 1) % m
        a, b, c, d = k * n + i, k * n + j, k1 * n + j, k1 * n + i
        faces.append(np.stack([a, c, b], axis=1))
        faces.append(np.stack([a, d, c], axis=1))
    return _outward(Mesh(verts, np.zeros_like(verts), np.vstack(faces).astype(np.int32), _position_keys(verts)))


def _outward(mesh: Mesh) -> Mesh:
    """Flip all faces of a closed mesh if they point inward."""
    v, f = mesh.vertices, mesh.faces
    vol = float(np.sum(v[f[:, 0]] * _cross(v[f[:, 1]], v[f[:, 2]])))
    if vol < 0:
        return Mesh(v, mesh.normals, f[:, ::-1].copy(), mesh.keys)
    return mesh


@cached
def blob(
    radii: Sequence[float],
    seed: int,
    amplitude: float = 0.12,
    lumps: int = 9,
    sharpness: float = 4.0,
    rings: int = 12,
    segments: int = 24,
    flat_bottom: float | None = None,
) -> Mesh:
    """A lumpy ellipsoid: deterministic smooth bumps from `seed` (foliage, rocks, fruit)."""
    base = sphere(1.0, rings, segments)
    dirs = base.vertices / np.maximum(np.linalg.norm(base.vertices, axis=1, keepdims=True), _EPS)
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(lumps, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    weights = rng.uniform(0.5, 1.0, size=lumps)
    field = (np.maximum(dirs @ centers.T, 0.0) ** sharpness) @ weights
    field = (field - field.mean()) / (np.abs(field - field.mean()).max() or 1.0)
    scale = 1.0 + amplitude * field
    pts = dirs * scale[:, None] * np.asarray(radii, dtype=float)
    if flat_bottom is not None:
        pts[:, 2] = np.maximum(pts[:, 2], flat_bottom)
    return base.displaced(pts - base.vertices)


def tube(path: np.ndarray, radii: Sequence[float], segments: int = 12) -> Mesh:
    """A tube along a 3D polyline with a radius per point (branches, handles, rails)."""
    path = np.asarray(path, dtype=float)
    rings = []
    u = None
    for k, p in enumerate(path):
        t = path[min(k + 1, len(path) - 1)] - path[max(k - 1, 0)]
        t /= np.linalg.norm(t) or 1.0
        if u is None:
            ref = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = _cross(t, ref)
        else:  # carry the previous frame along, so the tube does not twist
            u = u - (u @ t) * t
        u /= np.linalg.norm(u)
        v = _cross(t, u)
        ring2 = circle(radii[k], segments)
        rings.append(p + ring2[:, :1] * u + ring2[:, 1:] * v)
    return loft(rings)


@cached
def revolve(profile: Sequence[tuple[float, float]], segments: int = 32, sharp: Iterable[int] | None = None) -> Mesh:
    """Solid of revolution about z of a closed (radius, z) polygon: rims, rings, cups and bowls with walls.

    Corners listed in `sharp` (all of them by default) get creases; the rest are smoothed.
    """
    pts = list(profile)
    k = len(pts)
    rings = [at_z(circle(max(r, 0.0), segments), z) for r, z in pts]
    creases = set(range(k)) if sharp is None else {i % k for i in sharp}
    if not creases:
        return _finish(_closed_tube(rings, segments))
    start = min(creases)
    order = [(start + i) % k for i in range(k + 1)]
    runs, current = [], [order[0]]
    for idx in order[1:]:
        current.append(idx)
        if idx in creases:
            runs.append(current)
            current = [idx]
    parts = [_band([rings[i] for i in run], segments, True) for run in runs]
    return _finish(_outward(merge(parts)))


def resample_loop(ring2: np.ndarray, step: float) -> np.ndarray:
    """Evenly spaced points (about `step` apart) along a closed 2D polyline."""
    closed = np.vstack([ring2, ring2[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    n = max(8, round(total / step))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


@cached
def dense_rounded_box(hx: float, hy: float, hz: float, radius: float, step: float) -> Mesh:
    """A rounded box with vertices about `step` apart on its sides, ready to be displaced."""
    rb = max(min(radius, hx, hy, hz), 1e-4)
    rings = []
    base = rounded_rect(hx, hy, rb, 6)
    count = len(resample_loop(base, step))

    def ring(inset: float, z: float) -> np.ndarray:  # same point count on every ring, so they loft
        return at_z(_fit(rounded_rect(hx - inset, hy - inset, max(rb - inset, 1e-4), 6), count), z)

    for a in np.linspace(0.0, math.pi / 2, 4):
        rings.append(ring(rb * (1 - math.sin(a)), -(hz - rb) - rb * math.cos(a)))
    levels = max(1, round(2 * (hz - rb) / step))
    for z in np.linspace(-(hz - rb), hz - rb, levels + 1)[1:-1]:
        rings.append(ring(0.0, z))
    for a in np.linspace(math.pi / 2, 0.0, 4):
        rings.append(ring(rb * (1 - math.sin(a)), (hz - rb) + rb * math.cos(a)))
    return loft(rings)


def _fit(loop: np.ndarray, count: int) -> np.ndarray:
    """`count` evenly spaced points along a closed 2D polyline."""
    closed = np.vstack([loop, loop[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, float(s[-1]), count, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


@cached
def noise_displaced(mesh: Mesh, seed: int, amplitude: float, wavelength: float) -> Mesh:
    """Push every vertex along its normal by smooth 3D noise (a sum of random plane waves)."""
    rng = np.random.default_rng(seed)
    p = mesh.vertices
    field = np.zeros(len(p))
    for octave, weight in ((1.0, 1.0), (2.1, 0.5), (4.3, 0.25)):
        for _ in range(4):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            k = 2 * math.pi * octave / wavelength
            field += weight * np.sin(p @ d * k + rng.uniform(0, 2 * math.pi))
    field /= np.abs(field).max() or 1.0
    return mesh.displaced(mesh.normals * (amplitude * field)[:, None])
