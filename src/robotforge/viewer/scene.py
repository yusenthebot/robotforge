"""Describe a compiled MuJoCo model so a thin browser client can draw it.

The description is static per model version: geometry, colors and the body each
geom hangs off. Per-frame traffic is only body poses (see SimSession.body_poses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.design.terrain import STYLES, material_name

# MuJoCo's default geom rgba; when a geom keeps it and has a material, the material color wins.
_DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
# Every this many pixels of a texture count towards its mean color: ANYmal C has 19 textures of 2048 x 2048, and
# reading every pixel added 0.6 s to each world load; the mean moves by less than 1/255 this way.
TEXTURE_SAMPLE = 97
# Geoms in groups >= this are hidden by default (Menagerie puts collision geoms in group 3).
VISIBLE_GROUP_LIMIT = 3

_GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


@dataclass(frozen=True)
class MeshPayload:
    """All mesh triangles as one binary blob of float32 (positions then normals per mesh),
    followed by every heightfield's elevation grid (0..1, row by row from -y to +y)."""

    blob: bytes
    entries: list[dict]  # per mesh: {"id", "offset", "count"} in float32 units, count = vertices
    hfields: list[dict] = field(default_factory=list)  # per hfield: {"id", "offset", "nrow", "ncol", "size"}


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def _geom_rgba(model: mujoco.MjModel, g: int, textured: dict[int, np.ndarray]) -> list[float]:
    rgba = model.geom_rgba[g]
    mat = model.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
        rgba = model.mat_rgba[mat]
        if mat in textured and model.geom_bodyid[g] != 0:  # the ground draws its own textures
            rgba = np.concatenate([rgba[:3] * textured[mat], rgba[3:]])
    return [round(float(c), 4) for c in rgba]


def _texture_colors(model: mujoco.MjModel) -> dict[int, np.ndarray]:
    """The mean color of each material's own picture (2D texture), by material. The viewer draws no textures yet, so a
    textured shell showed plain white: ANYmal C's red covers and black legs, A1's carbon fibre."""
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    colors: dict[int, np.ndarray] = {}
    for mat in range(model.nmat):
        tex = int(model.mat_texid[mat, rgb])
        if tex < 0 or int(model.tex_type[tex]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
            continue
        adr, channels = int(model.tex_adr[tex]), int(model.tex_nchannel[tex])
        size = int(model.tex_width[tex]) * int(model.tex_height[tex]) * channels
        pixels = model.tex_data[adr : adr + size].reshape(-1, channels)[::TEXTURE_SAMPLE, :3]
        colors[mat] = pixels.mean(axis=0) / 255.0
    return colors


def build_mesh_payload(model: mujoco.MjModel) -> MeshPayload:
    """Expand each mesh into non-indexed triangles with MuJoCo's own normals."""
    chunks: list[np.ndarray] = []
    entries: list[dict] = []
    offset = 0
    for mesh_id in range(model.nmesh):
        v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        n0, nn = model.mesh_normaladr[mesh_id], model.mesh_normalnum[mesh_id]
        verts = model.mesh_vert[v0 : v0 + nv]
        faces = model.mesh_face[f0 : f0 + nf]
        positions = verts[faces].reshape(-1, 3)
        if nn > 0:
            normals = model.mesh_normal[n0 : n0 + nn][model.mesh_facenormal[f0 : f0 + nf]]
            normals = normals.reshape(-1, 3)
        else:
            normals = np.zeros_like(positions)
        block = np.concatenate([positions.ravel(), normals.ravel()]).astype(np.float32)
        entries.append({"id": mesh_id, "offset": offset, "count": int(positions.shape[0])})
        chunks.append(block)
        offset += block.size
    hfields = []
    for h in range(model.nhfield):
        nrow, ncol = int(model.hfield_nrow[h]), int(model.hfield_ncol[h])
        adr = int(model.hfield_adr[h])
        grid = model.hfield_data[adr : adr + nrow * ncol].astype(np.float32)
        size = [round(float(v), 6) for v in model.hfield_size[h]]
        hfields.append({"id": h, "offset": offset, "nrow": nrow, "ncol": ncol, "size": size})
        chunks.append(grid)
        offset += grid.size
    blob = np.concatenate(chunks).tobytes() if chunks else b""
    return MeshPayload(blob=blob, entries=entries, hfields=hfields)


def describe(
    model: mujoco.MjModel, version: int, mesh_entries: list[dict], hfield_entries: list[dict] | None = None
) -> dict:
    """JSON-ready scene description for the viewer."""
    textured = _texture_colors(model)
    geoms = []
    for g in range(model.ngeom):
        gtype = _GEOM_TYPE_NAMES.get(int(model.geom_type[g]))
        if gtype is None:
            continue  # SDF and unknown types are not drawn yet
        geoms.append(
            {
                "id": g,
                "name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                "body": int(model.geom_bodyid[g]),
                "type": gtype,
                "size": [round(float(s), 6) for s in model.geom_size[g]],
                "pos": [float(p) for p in model.geom_pos[g]],
                "quat": [float(q) for q in model.geom_quat[g]],
                "rgba": _geom_rgba(model, g, textured),
                "mesh": int(model.geom_dataid[g]) if gtype == "mesh" else -1,
                "hfield": int(model.geom_dataid[g]) if gtype == "hfield" else -1,
                # Ground materials ("ground_grass", ...) tell the viewer which surface texture to draw.
                "material": _name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g]))
                if model.geom_matid[g] >= 0
                else "",
                "group": int(model.geom_group[g]),
            }
        )

    actuators = []
    for a in range(model.nu):
        if model.actuator_ctrllimited[a]:
            lo, hi = (float(x) for x in model.actuator_ctrlrange[a])
        else:
            lo, hi = -1.0, 1.0
        actuators.append(
            {
                "id": a,
                "name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator {a}",
                "range": [lo, hi],
            }
        )

    bodies = [
        {
            "id": b,
            "name": _name(model, mujoco.mjtObj.mjOBJ_BODY, b),
            "parent": int(model.body_parentid[b]),
            "dynamic": bool(model.body_dofnum[b] > 0 or _has_moving_ancestor(model, b)),
        }
        for b in range(model.nbody)
    ]

    return {
        "type": "scene",
        "version": version,
        "timestep": float(model.opt.timestep),
        "nbody": model.nbody,
        "bodies": bodies,
        "geoms": geoms,
        "meshes": mesh_entries,
        "hfields": hfield_entries or [],
        "grounds": _ground_looks(model),
        "actuators": actuators,
        "visibleGroupLimit": VISIBLE_GROUP_LIMIT,
        "stat": {"center": [float(c) for c in model.stat.center], "extent": float(model.stat.extent)},
    }


def _ground_looks(model: mujoco.MjModel) -> dict[str, dict]:
    """Texture pattern and tile size for each ground material in the model, by material name."""
    looks = {}
    for style, look in STYLES.items():
        name = material_name(style)
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name) >= 0:
            looks[name] = {"pattern": look.pattern, "tile": look.tile}
    return looks


def _has_moving_ancestor(model: mujoco.MjModel, body: int) -> bool:
    b = int(model.body_parentid[body])
    while b > 0:
        if model.body_dofnum[b] > 0:
            return True
        b = int(model.body_parentid[b])
    return False
