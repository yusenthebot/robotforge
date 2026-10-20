"""Profiles for turned objects (cups, bottles, bowls): lists of (radius, height) points from the bottom up, and the
operations that make a hollow vessel of one."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Profile = list[tuple[float, float]]


def arc(center: tuple[float, float], radius: float, start: float, end: float, steps: int = 6) -> Profile:
    """Points on a circle in the (radius, height) plane, angles in rad from the +r axis counter-clockwise."""
    return [
        (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in np.linspace(start, end, steps)
    ]


def smooth(points: Sequence[tuple[float, float]], per_segment: int = 4) -> Profile:
    """Pass a smooth curve through the points (Catmull-Rom), keeping the first and last as they are."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    padded = np.vstack([pts[0] * 2 - pts[1], pts, pts[-1] * 2 - pts[-2]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        for t in np.linspace(0.0, 1.0, per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(
                0.5
                * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            )
    out.append(pts[-1])
    return [(max(float(r), 0.0), float(z)) for r, z in out]


def offset_inward(outer: Sequence[tuple[float, float]], wall: float) -> Profile:
    """The same line moved `wall` toward the axis, along each point's normal (bottom to top)."""
    pts = np.asarray(outer, dtype=float)
    normals = np.zeros_like(pts)
    for i in range(len(pts) - 1):
        d = pts[i + 1] - pts[i]
        n = np.array([d[1], -d[0]]) / (np.linalg.norm(d) or 1.0)  # outward when going up
        normals[i] += n
        normals[i + 1] += n
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-9)
    inner = pts - normals * wall
    return [(max(float(r), 0.0), float(z)) for r, z in inner]


def hollow(
    outer: Sequence[tuple[float, float]], wall: float, floor: float | None = None, round_rim: int = 4
) -> Profile:
    """A closed profile for lathe(): up the outside, over a rounded rim, down the inside to the floor, with a pole
    on the axis at both ends. `outer` runs from the bottom edge up to the rim; `floor` is the bottom's thickness."""
    floor = wall if floor is None else floor
    outer = list(outer)
    inner = offset_inward(outer, wall)
    rim_o, rim_i = outer[-1], inner[-1]
    middle = ((rim_o[0] + rim_i[0]) / 2, (rim_o[1] + rim_i[1]) / 2)
    radius = max(math.dist(rim_o, rim_i) / 2, 1e-5)
    angle = math.atan2(rim_o[1] - middle[1], rim_o[0] - middle[0])
    # half a circle over the top from the outer rim point to the inner one
    cap = [
        (middle[0] + radius * math.cos(angle + t), middle[1] + radius * math.sin(angle + t))
        for t in np.linspace(0.0, math.pi, round_rim + 2)[1:-1]
    ]
    down = list(reversed(inner))
    # the inside meets the floor `floor` above the bottom
    inside = [(r, max(z, floor)) for r, z in down]
    deduped: Profile = []
    for p in inside:
        if not deduped or math.dist(p, deduped[-1]) > 1e-6:
            deduped.append(p)
    return [(0.0, 0.0), *outer, *cap, *deduped, (0.0, floor)]


def volume_of_revolution(profile: Sequence[tuple[float, float]]) -> float:
    """Volume swept by a closed profile about the axis (Pappus), m^3."""
    pts = np.asarray(profile, dtype=float)
    r, z = pts[:, 0], pts[:, 1]
    r2, z2 = np.roll(r, -1), np.roll(z, -1)
    area = np.sum(r * z2 - r2 * z) / 2.0
    if abs(area) < 1e-15:
        return 0.0
    cx = np.sum((r + r2) * (r * z2 - r2 * z)) / (6.0 * area)
    return abs(float(2.0 * math.pi * cx * area))
