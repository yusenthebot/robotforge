"""Small geometric queries on a posed model (after mj_kinematics or mj_forward)."""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

G = mujoco.mjtGeom


# Geoms in groups from this one up are hidden, as in MuJoCo's own viewer and ours (scene.VISIBLE_GROUP_LIMIT):
# Menagerie puts its collision shapes in group 3.
VISIBLE_GROUPS = 3


def collides(model: mujoco.MjModel, geom: int) -> bool:
    """True for geoms that take part in contacts (not purely visual)."""
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def visible(model: mujoco.MjModel, geom: int) -> bool:
    """True for geoms that are drawn: what a robot looks like (visual meshes, or shapes that are both)."""
    return int(model.geom_group[geom]) < VISIBLE_GROUPS


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """The world point of `geom` with the smallest z, exact for the common shapes."""
    return _furthest(model, data, geom, -1.0)


def highest_point(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """The world point of `geom` with the largest z, exact for the common shapes."""
    return _furthest(model, data, geom, 1.0)


def _furthest(model: mujoco.MjModel, data: mujoco.MjData, geom: int, sign: float) -> np.ndarray:
    """The world point of `geom` furthest down (sign -1) or up (sign 1)."""
    pos = data.geom_xpos[geom]
    rot = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    gtype = int(model.geom_type[geom])
    toward = np.array([0.0, 0.0, sign])

    def height(p: np.ndarray) -> float:
        return sign * float(p[2])

    if gtype == G.mjGEOM_SPHERE:
        return pos + toward * size[0]
    if gtype == G.mjGEOM_CAPSULE:
        ends = [pos + rot[:, 2] * size[1], pos - rot[:, 2] * size[1]]
        return max(ends, key=height) + toward * size[0]
    if gtype == G.mjGEOM_CYLINDER:
        # The furthest point of either end disc: the rim point furthest that way.
        axis = rot[:, 2]
        radial = toward - axis * (toward @ axis)
        n = np.linalg.norm(radial)
        rim = radial / n * size[0] if n > 1e-9 else np.zeros(3)
        ends = [pos + axis * size[1] + rim, pos - axis * size[1] + rim]
        return max(ends, key=height)
    if gtype == G.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = pos + model.mesh_vert[start : start + count] @ rot.T
        return verts[int(np.argmax(sign * verts[:, 2]))]
    if gtype == G.mjGEOM_BOX:
        half = size
    else:  # ellipsoids, hfields and anything else: their bounding box
        center, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
        pos = pos + rot @ center
    corners = np.array([pos + rot @ (np.array(s) * half) for s in itertools.product((-1, 1), repeat=3)])
    return corners[int(np.argmax(sign * corners[:, 2]))]
