"""Describe a compiled MuJoCo model so a thin browser client can draw it.

The description is static per model version: geometry, colors and the body each
geom hangs off. Per-frame traffic is only body poses (see SimSession.body_poses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.design.terrain import STYLES, material_name
from robotforge.sim import water as water_model
from robotforge.viewer import textures as model_textures

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
    textures: list[dict] = field(default_factory=list)
    texture_blob: bytes = b""


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def _geom_rgba(model: mujoco.MjModel, g: int, textured: dict[int, np.ndarray], painted: set[int]) -> list[float]:
    """A geom's color: its own, or its material's, times the mean of the material's picture where the viewer does not
    color it vertex by vertex (`painted`: meshes that get their texture's colors, see build_mesh_payload)."""
    rgba = model.geom_rgba[g]
    mat = model.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, _DEFAULT_RGBA):
        rgba = model.mat_rgba[mat]
        mesh = int(model.geom_dataid[g]) if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH else -1
        if mat in textured and model.geom_bodyid[g] != 0 and mesh not in painted:  # the ground draws its own
            rgba = np.concatenate([rgba[:3] * textured[mat], rgba[3:]])
    return [round(float(c), 4) for c in rgba]


def _material_colors(model: mujoco.MjModel) -> dict[int, np.ndarray]:
    """The mean color of each material's own picture (2D texture), by material, for geoms the viewer cannot color per
    vertex (see _texture_colors): drawn plain white, a textured shell hid ANYmal C's red covers and black legs."""
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    colors: dict[int, np.ndarray] = {}
    for mat in range(model.nmat):
        tex = int(model.mat_texid[mat, rgb])
        if tex < 0 or int(model.tex_type[tex]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
            continue
        adr, channels = int(model.tex_adr[tex]), int(model.tex_nchannel[tex])
        if channels not in (3, 4):
            continue  # encoded texture formats are not RGB pixels
        size = int(model.tex_width[tex]) * int(model.tex_height[tex]) * channels
        pixels = model.tex_data[adr : adr + size].reshape(-1, channels)[::TEXTURE_SAMPLE, :3]
        colors[mat] = pixels.mean(axis=0) / 255.0
    return colors


def _mesh_textures(model: mujoco.MjModel) -> dict[int, int]:
    """mesh id -> the 2D color texture of the first geom that draws it with a textured material."""
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    found: dict[int, int] = {}
    for g in range(model.ngeom):
        mat = int(model.geom_matid[g])
        if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH or mat < 0:
            continue
        tex = int(model.mat_texid[mat, role])
        if (
            tex >= 0
            and int(model.tex_type[tex]) == mujoco.mjtTexture.mjTEXTURE_2D
            and int(model.tex_nchannel[tex]) in (3, 4)
        ):
            found.setdefault(int(model.geom_dataid[g]), tex)
    return found


def _mesh_uvs(model: mujoco.MjModel, mesh_id: int) -> np.ndarray | None:
    count = int(model.mesh_texcoordnum[mesh_id])
    if count == 0:
        return None
    f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    indices = model.mesh_facetexcoord[f0 : f0 + nf].ravel()
    if np.any(indices < 0) or np.any(indices >= count):
        return None
    uv = model.mesh_texcoord[model.mesh_texcoordadr[mesh_id] + indices]
    return uv if np.isfinite(uv).all() else None


def _geom_texture(model: mujoco.MjModel, geom: int) -> int:
    mat = int(model.geom_matid[geom])
    if mat < 0:
        return -1
    return int(model.mat_texid[mat, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])


def _texture_repeat(model: mujoco.MjModel, geom: int) -> list[float]:
    mat = int(model.geom_matid[geom])
    repeat = np.where(model.mat_texrepeat[mat] > 0, model.mat_texrepeat[mat], 1.0)
    # Match MuJoCo 3.15's explicit-UV texture matrix, including its size-relative option.
    if model.mat_texuniform[mat]:
        repeat *= np.where(model.geom_size[geom, :2] > 0, model.geom_size[geom, :2], 1.0)
    return [float(v) for v in repeat]


def _texture_colors(model: mujoco.MjModel, mesh_id: int, tex: int) -> np.ndarray | None:
    """Approximate linear RGB per expanded vertex for legacy clients or unsupported native image bindings."""
    uv = _mesh_uvs(model, mesh_id)
    if uv is None:
        return None
    w, h, ch = int(model.tex_width[tex]), int(model.tex_height[tex]), int(model.tex_nchannel[tex])
    if ch not in (3, 4):
        return None
    image = model.tex_data[model.tex_adr[tex] : model.tex_adr[tex] + w * h * ch].reshape(h, w, ch)
    cols = np.clip(np.rint(np.mod(uv[:, 0], 1.0) * (w - 1)), 0, w - 1).astype(int)
    rows = np.clip(np.rint(np.mod(uv[:, 1], 1.0) * (h - 1)), 0, h - 1).astype(int)
    srgb = image[rows, cols, :3].astype(np.float32) / 255.0
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def build_mesh_payload(model: mujoco.MjModel) -> MeshPayload:
    """Expand each mesh into non-indexed triangles with MuJoCo's own normals (and its texture's colors per vertex
    when a geom draws it textured: "colors" is then where they start in the blob)."""
    chunks: list[np.ndarray] = []
    entries: list[dict] = []
    offset = 0
    textures = _mesh_textures(model)
    uvs = {mesh: uv for mesh in textures if (uv := _mesh_uvs(model, mesh)) is not None}
    native = model_textures.build(
        model,
        {
            tex
            for g in range(model.ngeom)
            if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH and int(model.geom_dataid[g]) in uvs
            if (tex := _geom_texture(model, g)) >= 0
        },
    )
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
        colors = _texture_colors(model, mesh_id, textures[mesh_id]) if mesh_id in textures else None
        if colors is not None and len(colors) == positions.shape[0]:
            entries[-1]["colors"] = offset
            chunks.append(colors.ravel())
            offset += colors.size
        if native.entries and mesh_id in uvs:
            uv = uvs[mesh_id].astype(np.float32)
            entries[-1]["uv"] = offset
            chunks.append(uv.ravel())
            offset += uv.size
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
    return MeshPayload(blob=blob, entries=entries, hfields=hfields, textures=native.entries, texture_blob=native.blob)


def describe(
    model: mujoco.MjModel,
    version: int,
    mesh_entries: list[dict],
    hfield_entries: list[dict] | None = None,
    texture_entries: list[dict] | None = None,
) -> dict:
    """JSON-ready scene description for the viewer."""
    textured = _material_colors(model)
    painted = {int(entry["id"]) for entry in mesh_entries if "colors" in entry}  # colored per vertex already
    uv_meshes = {int(entry["id"]) for entry in mesh_entries if "uv" in entry}
    texture_ids = {alias: entry["id"] for entry in texture_entries or [] for alias in entry["aliases"]}
    native_materials = _native_materials(model)
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
                "rgba": _geom_rgba(model, g, textured, painted),
                "mesh": int(model.geom_dataid[g]) if gtype == "mesh" else -1,
                "hfield": int(model.geom_dataid[g]) if gtype == "hfield" else -1,
                # Ground materials ("ground_grass", ...) tell the viewer which surface texture to draw.
                "material": _name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g]))
                if model.geom_matid[g] >= 0
                else "",
                "group": int(model.geom_group[g]),
            }
        )
        material_id = int(model.geom_matid[g])
        if material_id in native_materials:
            geoms[-1]["material_id"] = material_id
        if texture_entries and gtype == "mesh":
            tex = _geom_texture(model, g)
            native_id = texture_ids.get(tex) if int(model.geom_dataid[g]) in uv_meshes else None
            geoms[-1]["vertex_colors"] = int(model.geom_dataid[g]) in painted and tex >= 0 and native_id is None
            if native_id is not None:
                geoms[-1]["texture"] = {
                    "id": native_id,
                    "repeat": _texture_repeat(model, g),
                }

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
        "flexes": _flexes(model),
        "meshes": mesh_entries,
        "hfields": hfield_entries or [],
        **({"textures": texture_entries} if texture_entries else {}),
        "grounds": _ground_looks(model),
        "surfaces": _surfaces(model),
        **({"native_materials": native_materials} if native_materials else {}),
        "water": water.describe() if (water := water_model.decode(model)) is not None else None,
        "actuators": actuators,
        "visibleGroupLimit": VISIBLE_GROUP_LIMIT,
        "stat": {"center": [float(c) for c in model.stat.center], "extent": float(model.stat.extent)},
    }


def _native_materials(model: mujoco.MjModel) -> dict[int, dict[str, float]]:
    """Keep authored PBR separate from the viewer's named presets; -1 means unset in MuJoCo."""
    out = {}
    for material in range(model.nmat):
        values = {}
        for parameter, array in (("roughness", model.mat_roughness), ("metalness", model.mat_metallic)):
            value = float(array[material])
            if np.isfinite(value) and 0 <= value <= 1:
                values[parameter] = value
        if values:
            out[material] = values
    return out


def _flexes(model: mujoco.MjModel) -> list[dict]:
    """Cloths (two-dimensional flexes) as triangle meshes whose corners ride on bodies: each vertex is a body (or a
    point on one, for a pinned edge), so the per-frame body poses already move them and no new traffic is needed.
    Each vertex is `local` in its body's frame; `faces` index the vertices three by three."""
    out = []
    for f in range(model.nflex):
        if int(model.flex_dim[f]) != 2:
            continue  # ropes are capsules on bodies; volumes are not drawn yet
        v0, nv = int(model.flex_vertadr[f]), int(model.flex_vertnum[f])
        e0, ne = int(model.flex_elemdataadr[f]), int(model.flex_elemnum[f])
        faces = model.flex_elem[e0 : e0 + 3 * ne]
        out.append(
            {
                "id": f,
                "bodies": [int(b) for b in model.flex_vertbodyid[v0 : v0 + nv]],
                "local": [round(float(c), 6) for c in model.flex_vert[v0 : v0 + nv].ravel()],
                "faces": [int(i) for i in faces],
                "rgba": [round(float(c), 4) for c in model.flex_rgba[f]],
            }
        )
    return out


def _surfaces(model: mujoco.MjModel) -> dict[str, dict]:
    """The viewer's half of every asset-engine material the model uses (clear coat, texture, ...), by material name.
    Materials the viewer knows by name (rf_shell, rf_metal, ...) are not listed."""
    from robotforge.assets.materials import surface_of

    out = {}
    for m in range(model.nmat):
        name = _name(model, mujoco.mjtObj.mjOBJ_MATERIAL, m)
        found = surface_of(name) if name else None
        if found is not None:
            out[name] = found
    return out


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
