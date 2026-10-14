"""Small geometric queries on a posed model (after mj_kinematics or mj_forward)."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

G = mujoco.mjtGeom


def collides(model: mujoco.MjModel, geom: int) -> bool:
    """True for geoms that take part in contacts (not purely visual)."""
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """The world point of `geom` with the smallest z, exact for the common shapes."""
    pos = data.geom_xpos[geom]
    rot = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    gtype = int(model.geom_type[geom])
    down = np.array([0.0, 0.0, -1.0])
    if gtype == G.mjGEOM_SPHERE:
        return pos + down * size[0]
    if gtype == G.mjGEOM_CAPSULE:
        ends = [pos + rot[:, 2] * size[1], pos - rot[:, 2] * size[1]]
        return min(ends, key=lambda p: p[2]) + down * size[0]
    if gtype == G.mjGEOM_CYLINDER:
        # Lowest point of either end disc: the rim point furthest down.
        axis = rot[:, 2]
        radial = down - axis * (down @ axis)
        n = np.linalg.norm(radial)
        rim = radial / n * size[0] if n > 1e-9 else np.zeros(3)
        ends = [pos + axis * size[1] + rim, pos - axis * size[1] + rim]
        return min(ends, key=lambda p: p[2])
    if gtype == G.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = pos + model.mesh_vert[start : start + count] @ rot.T
        return verts[int(np.argmin(verts[:, 2]))]
    if gtype == G.mjGEOM_BOX:
        half = size
    else:  # ellipsoids, hfields and anything else: their bounding box
        center, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
        pos = pos + rot @ center
    corners = np.array([pos + rot @ (np.array(s) * half) for s in itertools.product((-1, 1), repeat=3)])
    return corners[int(np.argmin(corners[:, 2]))]
