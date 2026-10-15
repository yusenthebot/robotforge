"""Procedural triangle meshes for visual-only geometry, built with numpy alone (no OpenGL needed).

Every builder returns an immutable `Mesh`: vertices, one normal per vertex and outward
triangles. Hard edges are made by giving each side its own vertices, so both smooth and
crisp edges reach MuJoCo and the browser viewer unchanged. Most shapes are lofts: a stack of
closed rings joined by quads and capped at both ends (boxes with rounded edges, lathed parts,
tapered links, lumpy foliage).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


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
    axis = np.cross(z, d)
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


def superellipse(a: float, b: float, power: float, n: int) -> np.ndarray:
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / power
    return np.stack([a * np.sign(c) * np.abs(c) ** e, b * np.sign(s) * np.abs(s) ** e], axis=1)


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
    point = [bool(np.ptp(r, axis=0).max() < 1e-12) for r in rings]
    faces = []
    for k in range(m - 1):
        a, b = k * n + i, k * n + j
        c, d = (k + 1) * n + j, (k + 1) * n + i
        if not point[k]:  # next to a pole one triangle per quad is enough, the other has no area
            faces.append(np.stack([a, b, c], axis=1))
        if not point[k + 1]:
            faces.append(np.stack([a, c, d], axis=1))
    faces = np.vstack(faces).astype(np.int32)
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
    i = np.arange(n)
    j = (i + 1) % n
    vol = 0.0
    for r0, r1 in zip(rings[:-1], rings[1:], strict=True):
        a, b, c, d = r0[i], r0[j], r1[j], r1[i]
        vol += float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() + np.einsum("ij,ij->i", a, np.cross(c, d)).sum())
    for ring, reverse in ((rings[0], True), (rings[-1], False)):
        center = np.broadcast_to(ring.mean(axis=0), ring.shape)
        p, q = (ring[j], ring[i]) if reverse else (ring[i], ring[j])
        vol += float(np.einsum("ij,ij->i", center, np.cross(p, q)).sum())
    return vol / 6.0


def _position_keys(verts: np.ndarray) -> np.ndarray:
    """Same key for vertices at the same spot (poles, wrap seams), so they share a normal."""
    q = np.round(verts * 1e7).astype(np.int64)
    _, inverse = np.unique(q, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def _finish(mesh: Mesh) -> Mesh:
    return Mesh(mesh.vertices, _smooth_normals(mesh.vertices, mesh.faces, mesh.keys), mesh.faces, mesh.keys)


def _smooth_normals(verts: np.ndarray, faces: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Area-weighted face normals summed per smoothing key."""
    if faces.size == 0:
        return np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)  # length = 2 * area
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


def lathe(profile: Sequence[tuple[float, float]], segments: int = 32, sharp: Iterable[int] = ()) -> Mesh:
    """Surface of revolution about z from (radius, z) points, bottom to top.

    End points with radius 0 close the shape with a pole; otherwise the ends get flat caps.
    """
    rings = [at_z(circle(max(r, 0.0), segments), z) for r, z in profile]
    return loft(rings, sharp=sharp)


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


def sphere(r: float, rings: int = 12, segments: int = 24) -> Mesh:
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi, rings + 1)]
    return lathe(prof, segments)


def capsule(r: float, length: float, rings: int = 6, segments: int = 20) -> Mesh:
    """Capsule along z from 0 to `length` (the end caps reach r beyond)."""
    prof = [(r * math.sin(t), -r * math.cos(t)) for t in np.linspace(0, math.pi / 2, rings + 1)]
    prof += [(r * math.sin(t), length - r * math.cos(t)) for t in np.linspace(math.pi / 2, math.pi, rings + 1)]
    return lathe(prof, segments)


def link(
    length: float,
    start: tuple[float, float],
    end: tuple[float, float],
    roundness: float = 0.6,
    rounding: tuple[float, float] = (1.0, 1.0),
    stations: int = 5,
    per_corner: int = 3,
) -> Mesh:
    """A tapered limb along z from 0 to `length`: rounded-rectangle sections from half extents
    `start` (x, y) to `end`, ends domed by `rounding` times the end section's smaller half extent."""
    rings = []
    d0 = rounding[0] * min(start)
    d1 = rounding[1] * min(end)
    domes = 4
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
    vol = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum())
    if vol < 0:
        return Mesh(v, mesh.normals, f[:, ::-1].copy(), mesh.keys)
    return mesh


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


def bent_lathe(profile: Sequence[tuple[float, float]], segments: int, bend: Sequence[float]) -> Mesh:
    """A lathe whose rings slide sideways by `bend[i]` (x) per profile point, for crooked trunks."""
    rings = []
    for (r, z), dx in zip(profile, bend, strict=True):
        ring = at_z(circle(max(r, 0.0), segments), z)
        ring[:, 0] += dx
        rings.append(ring)
    return loft(rings)


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
            u = np.cross(t, ref)
        else:  # carry the previous frame along, so the tube does not twist
            u = u - (u @ t) * t
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        ring2 = circle(radii[k], segments)
        rings.append(p + ring2[:, :1] * u + ring2[:, 1:] * v)
    return loft(rings)


def extrude(polygon: np.ndarray, depth: float) -> Mesh:
    """Prism of a convex counter-clockwise 2D polygon (x, y), from z = -depth/2 to depth/2, crisp edges."""
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    parts = []
    for k in range(n):
        p0, p1 = poly[k], poly[(k + 1) % n]
        quad = np.array([[*p0, -depth / 2], [*p1, -depth / 2], [*p1, depth / 2], [*p0, depth / 2]])
        parts.append(_quad(quad))
    parts.append(_cap(at_z(poly, -depth / 2), reverse=True))
    parts.append(_cap(at_z(poly, depth / 2), reverse=False))
    return _finish(merge(parts))


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
