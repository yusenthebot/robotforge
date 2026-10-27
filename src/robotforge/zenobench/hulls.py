"""Convex collision pieces for USD collision meshes.

MuJoCo collides every mesh as its convex hull, while PhysX also cooks triangle meshes (static) and
convex decompositions. A mesh is split into connected parts; a part whose vertices all lie close to
its hull stays one hull, a concave part (a room shell, a shelf, a pot) is split by CoACD. Results are
cached on disk by mesh content, because CoACD takes seconds per part.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CONVEX_TOL = 0.008  # m: deepest vertex inside the hull that still counts as convex
COVER_FRACTION = 0.95  # of the hull's surface the mesh must cover to count as convex
MIN_EXTENT = 0.002  # m: parts smaller than this are dropped
MAX_PIECES = 32
_VERSION = "5"


def cache_dir() -> Path:
    from robotforge.design.menagerie import cache_root

    return cache_root() / "zenobench" / "hulls"


def _key(verts: np.ndarray, faces: np.ndarray, mode: str) -> str:
    h = hashlib.sha1(f"{_VERSION}|{mode}|{CONVEX_TOL}|{MAX_PIECES}".encode())
    h.update(np.ascontiguousarray(verts, np.float32).tobytes())
    h.update(np.ascontiguousarray(faces, np.int32).tobytes())
    return h.hexdigest()[:20]


def hull(points: np.ndarray):
    """scipy's qhull hull, or None for flat or degenerate point sets."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        return ConvexHull(points)
    except (QhullError, ValueError):
        return None


def hull_volume(points: np.ndarray) -> float:
    h = hull(points)
    return float(h.volume) if h is not None else float(np.prod(np.ptp(points, axis=0)))


def inside_depth(verts: np.ndarray, h) -> float:
    """How far the deepest vertex sits inside the hull (0 for a convex shape)."""
    n, d = h.equations[:, :3], h.equations[:, 3]  # n . x + d <= 0 inside
    if len(verts) > 20000:
        verts = verts[np.random.default_rng(0).choice(len(verts), 20000, replace=False)]
    depth = 0.0
    for chunk in np.array_split(verts, max(1, len(verts) // 2000)):
        dist = -(chunk @ n.T + d[None, :])  # >= 0 inside each face plane
        depth = max(depth, float(np.max(np.min(dist, axis=1))))
    return depth


def is_convex(verts: np.ndarray, faces: np.ndarray, h) -> bool:
    """The part's surface is its hull: no vertex inside it, and no hull face missing.

    The second test matters for open surfaces such as a room (floor and walls, no roof): every vertex
    sits on the hull, yet the hull would fill the whole room."""
    if inside_depth(verts, h) > CONVEX_TOL:
        return False
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    return area >= COVER_FRACTION * h.area


def split_parts(verts: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Connected parts of a triangle soup (vertices merged at 0.1 mm) as (vertices, faces)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    keys = np.round(verts / 1e-4).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    v = verts[first]
    f = inverse.reshape(-1)[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    if len(f) == 0:
        return []
    rows = np.r_[f[:, 0], f[:, 1]]
    cols = np.r_[f[:, 1], f[:, 2]]
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(len(v), len(v)))
    _, label = connected_components(graph, directed=False)
    face_label = label[f[:, 0]]
    order = np.argsort(face_label, kind="stable")
    bounds = np.flatnonzero(np.diff(face_label[order])) + 1
    parts = []
    for idx in np.split(order, bounds):
        pf = f[idx]
        used, local = np.unique(pf, return_inverse=True)
        parts.append((v[used], local.reshape(-1, 3)))
    return parts


def _mode(decompose: bool, static: bool) -> str:
    return ("dec" if decompose else "hull") + ("-static" if static else "")


def cached(verts: np.ndarray, faces: np.ndarray, decompose: bool = True, static: bool = False) -> bool:
    mode = _mode(decompose, static)
    return (cache_dir() / f"{_key(np.asarray(verts, np.float64), np.asarray(faces, np.int64), mode)}.npz").exists()


def convex_pieces(
    verts: np.ndarray, faces: np.ndarray, *, decompose: bool = True, static: bool = False
) -> list[np.ndarray]:
    """Vertex arrays of convex pieces that together cover the mesh (one per hull).

    `static` meshes never move, so large concave ones (rooms) may use many exact slabs."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    if len(verts) < 4 or len(faces) == 0:
        return []
    mode = _mode(decompose, static)
    path = cache_dir() / f"{_key(verts, faces, mode)}.npz"
    if path.exists():
        with np.load(path) as z:
            return [z[k].astype(np.float64) for k in sorted(z.files, key=lambda s: int(s[1:]))]
    pieces = _compute(verts, faces, decompose, static) if decompose else [verts]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **{f"p{i}": p.astype(np.float32) for i, p in enumerate(pieces)})
    tmp.replace(path)
    return pieces


def _compute(verts: np.ndarray, faces: np.ndarray, decompose: bool, static: bool) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for pv, pf in split_parts(verts, faces):
        extent = float(np.max(np.ptp(pv, axis=0)))
        if extent < MIN_EXTENT:
            continue
        h = hull(pv)
        if h is None:  # flat part (a wall panel, a tabletop): its points are the piece
            out.append(pv)
            continue
        if not decompose or len(pf) < 8 or is_convex(pv, pf, h):
            out.append(pv[h.vertices])
            continue
        if static and extent > PRISM_MIN_EXTENT and len(pf) <= PRISM_MAX_FACES:
            # a large static surface (a room, a shelf unit): CoACD's error grows with size, slabs are
            # exact; a curved one (a bed) would need too many slabs and goes to a tighter CoACD
            slabs = prisms(pv, pf)
            if len(slabs) <= max(PRISM_BUDGET, PRISM_BUDGET_PER_M2 * _area(pv, pf)):
                out.extend(slabs)
                continue
            out.extend(_coacd(pv, pf, threshold=0.02))
            continue
        out.extend(_coacd(pv, pf))
    return out


PRISM_MIN_EXTENT = 1.5  # m
PRISM_MAX_FACES = 3000
PRISM_DEPTH = 0.03  # m into the solid, behind each face
PRISM_BUDGET = 120  # slabs a part may always use
PRISM_BUDGET_PER_M2 = 8.0  # and more for a large surface (a room shell has tens of m2)


def _area(verts: np.ndarray, faces: np.ndarray) -> float:
    tri = verts[faces]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())


def prisms(verts: np.ndarray, faces: np.ndarray, depth: float = PRISM_DEPTH) -> list[np.ndarray]:
    """Exact static collision for a concave surface. Coplanar triangles are merged into few convex
    polygons (Hertel-Mehlhorn: drop a shared edge whenever the union stays convex); each polygon becomes
    a slab reaching `depth` behind the face (against its normal, into the wall or floor)."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a2 = np.linalg.norm(n, axis=1)
    ok = a2 > 1e-10
    faces, n = faces[ok], n[ok] / a2[ok, None]
    d = np.einsum("ij,ij->i", n, verts[faces[:, 0]])
    key = np.c_[np.round(n / 0.01), np.round(d / 0.002)].astype(np.int64)
    _, label = np.unique(key, axis=0, return_inverse=True)
    out = []
    for g in np.unique(label):
        idx = np.flatnonzero(label == g)
        normal = n[idx].mean(0)
        normal /= np.linalg.norm(normal)
        u = np.cross(normal, (1.0, 0, 0) if abs(normal[0]) < 0.9 else (0, 1.0, 0))
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        for poly in convex_polygons(faces[idx], verts @ u, verts @ w):
            p = verts[poly]
            out.append(np.concatenate([p, p - normal * depth]))
    return out


def convex_polygons(faces: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[list[int]]:
    """Merge triangles of one plane (2D coordinates x, y per vertex) into convex polygons."""
    polys: list[list[int] | None] = []
    for f in faces:
        a, b, c = (int(i) for i in f)
        cross = (x[b] - x[a]) * (y[c] - y[a]) - (y[b] - y[a]) * (x[c] - x[a])
        polys.append([a, b, c] if cross > 0 else [a, c, b])  # counter clockwise
    owner = list(range(len(polys)))

    def find(i: int) -> int:
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    edges: dict[tuple[int, int], list[int]] = {}
    for i, poly in enumerate(polys):
        for k in range(3):
            a, b = poly[k], poly[(k + 1) % 3]
            edges.setdefault((min(a, b), max(a, b)), []).append(i)
    shared = [(e, ids) for e, ids in edges.items() if len(ids) == 2]
    shared.sort(key=lambda item: -((x[item[0][0]] - x[item[0][1]]) ** 2 + (y[item[0][0]] - y[item[0][1]]) ** 2))
    for (a, b), ids in shared:
        i, j = find(ids[0]), find(ids[1])
        if i == j:
            continue
        merged = _splice(polys[i], polys[j], a, b)
        if merged is None or not _convex(merged, x, y):
            continue
        polys[i], polys[j] = merged, None
        owner[j] = i
    return [p for p in polys if p is not None]


def _splice(p: list[int], q: list[int], a: int, b: int) -> list[int] | None:
    """Union of two polygons sharing edge a-b (p runs a->b or b->a, q the other way)."""

    def at(poly, s, t):
        for k in range(len(poly)):
            if poly[k] == s and poly[(k + 1) % len(poly)] == t:
                return k
        return None

    k = at(p, a, b)
    if k is None:
        a, b = b, a
        k = at(p, a, b)
    m = at(q, b, a)
    if k is None or m is None:
        return None
    rot_p = p[k + 1 :] + p[: k + 1]  # starts at b, ends at a
    rot_q = q[m + 1 :] + q[: m + 1]  # starts at a, ends at b
    return rot_p[:-1] + rot_q[:-1]


def _convex(poly: list[int], x: np.ndarray, y: np.ndarray) -> bool:
    n = len(poly)
    for k in range(n):
        o, a, b = poly[k - 1], poly[k], poly[(k + 1) % n]
        cross = (x[a] - x[o]) * (y[b] - y[a]) - (y[a] - y[o]) * (x[b] - x[a])
        if cross < -1e-9:
            return False
    return True


def box_corners(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(0), v.max(0)
    hi = np.maximum(hi, lo + 1e-4)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def _coacd(verts: np.ndarray, faces: np.ndarray, threshold: float = 0.04) -> list[np.ndarray]:
    import coacd

    coacd.set_log_level("error")
    try:
        result = coacd.run_coacd(
            coacd.Mesh(np.asarray(verts, np.float64), np.asarray(faces, np.int32)),
            threshold=threshold,
            max_convex_hull=MAX_PIECES if threshold >= 0.04 else 2 * MAX_PIECES,
            preprocess_mode="auto",
            preprocess_resolution=40,
            mcts_iterations=60,
            seed=0,
        )
    except Exception as exc:  # the native library raises plain RuntimeError
        log.warning("CoACD failed (%s); using the convex hull", exc)
        return [verts]
    return [np.asarray(v) for v, _ in result if len(v) >= 4]
