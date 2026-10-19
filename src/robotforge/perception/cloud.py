"""Point clouds: a depth picture or a lidar scan as 3D points in the world, thinned to a number an agent can read.

A depth pixel (u, v) with depth d lies at `lens + ray(u, v) * d`, where the ray goes one meter along the camera's
axis (the pinhole model the renderer uses, so the points agree with the picture). The cloud is thinned by voxel:
all points in a cube of the voxel's size become their mean, and the voxel grows until no more than the wanted
number are left.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MIN_VOXEL = 0.01  # m
GROWTH = 1.3


def depth_points(depth: np.ndarray, cam: Any, stride: int = 1) -> np.ndarray:
    """World points (N x 3) of the pixels of `depth` (H x W, NaN for none) that measured something, every
    `stride`th pixel each way. `cam` is the rendered camera (pos, ray(us, vs))."""
    sub = depth[::stride, ::stride]
    vs, us = np.nonzero(np.isfinite(sub))
    if not len(vs):
        return np.zeros((0, 3))
    vs, us = vs * stride, us * stride
    rays = cam.ray(us + 0.5, vs + 0.5)
    return np.asarray(cam.pos) + rays * depth[vs, us][:, None]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One mean point per cube of side `voxel`."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, count = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(count), 3))
    np.add.at(sums, inverse, points)
    return sums / count[:, None]


def thin(points: np.ndarray, limit: int) -> tuple[np.ndarray, float]:
    """`points` thinned to at most `limit` by the smallest voxel that does it; returns them and the voxel (m)."""
    voxel = MIN_VOXEL
    out = voxel_downsample(points, voxel)
    while len(out) > limit:
        voxel *= GROWTH
        out = voxel_downsample(points, voxel)
    return out, round(voxel, 3)


def summary(points: np.ndarray, limit: int) -> dict[str, Any]:
    """What an agent reads of a cloud: how many points there were, the thinned points and their bounding box."""
    out: dict[str, Any] = {"frame": "world", "total": int(len(points))}
    if len(points) == 0:
        out.update(shown=0, points_m=[])
        return out
    thinned, voxel = thin(points, limit)
    out.update(
        shown=int(len(thinned)),
        voxel_m=voxel,
        box_m=[[round(float(v), 3) for v in thinned.min(axis=0)], [round(float(v), 3) for v in thinned.max(axis=0)]],
        points_m=[[round(float(v), 3) for v in p] for p in thinned],
    )
    return out
