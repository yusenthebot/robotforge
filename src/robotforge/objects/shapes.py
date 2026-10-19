"""Mesh shapes several generators share: extruded polygons, fluted grips, bent tubes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from robotforge.design.visuals import mesh as M


def prism(points: Sequence[Sequence[float]], z0: float, z1: float, crisp: bool = True) -> M.Mesh:
    """A convex polygon (x, y counter-clockwise) extruded from z0 to z1, each side its own face."""
    pts = np.asarray(points, dtype=float)
    if _area(pts) < 0:
        pts = pts[::-1]
    n = len(pts)
    parts = []
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        quad = np.array([[*a, z0], [*b, z0], [*b, z1], [*a, z1]])
        parts.append(M._quad(quad))
    for z, reverse in ((z0, True), (z1, False)):
        ring = np.column_stack([pts, np.full(n, z)])
        parts.append(M._cap(ring, reverse))
    mesh = M.merge(parts)
    return M._finish(mesh) if crisp else mesh


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2)


def fluted(radius: float, length: float, lobes: int = 6, depth: float = 0.08, segments: int = 48, z0: float = 0.0,
           taper: float = 0.0) -> M.Mesh:  # fmt: skip
    """A cylinder along z from z0 whose surface is scalloped by `lobes` shallow grooves (a screwdriver grip); `taper`
    narrows the far end by that share."""
    a = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    scallop = 1.0 - depth * 0.5 * (1.0 + np.cos(lobes * a))
    rings = []
    for k, (z, scale) in enumerate(
        zip(
            [z0, z0 + 0.004, z0 + length * 0.5, z0 + length - 0.006, z0 + length],
            [0.78, 1.0, 1.0, 1.0 - taper, 0.7 * (1.0 - taper)],
            strict=True,
        )  # fmt: skip
    ):
        flute = scallop if 0 < k < 4 else np.ones_like(a)
        r = radius * scale * flute
        rings.append(np.column_stack([r * np.cos(a), r * np.sin(a), np.full(segments, z)]))
    return M.loft(rings, sharp=(1,))


def lying(mesh: M.Mesh, axis: str = "x") -> M.Mesh:
    """A mesh built along z turned to lie along x (or y)."""
    rot = M.Z_TO_X if axis == "x" else M.Z_TO_Y
    return mesh.moved(rot=rot)


def ngon(radius: float, sides: int, phase: float = 0.0) -> np.ndarray:
    a = phase + np.linspace(0.0, 2 * math.pi, sides, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
