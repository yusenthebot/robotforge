"""Putting an asset into a MuJoCo model: its drawn pieces through `Visuals`, its collision shapes as geoms.

Drawing never changes physics (Visuals makes every mesh massless and contactless). `collide` is the only function
here that adds mass: it gives a body the asset's colliders at a density, so the part weighs what its shape says.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.styles import Style, get_style
from robotforge.design.visuals import HIDDEN_GROUP, VISUAL_GROUP, Visuals

_GEOM = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}


@dataclass(frozen=True)
class Palette:
    """The colors the roles of a piece take. `body` and `accent` are the robot's own; the rest follow the style."""

    body: tuple[float, ...]
    accent: tuple[float, ...]
    structure: tuple[float, ...]
    trim: tuple[float, ...]
    metal: tuple[float, ...] = (0.7, 0.72, 0.75, 1.0)
    rubber: tuple[float, ...] = (0.08, 0.08, 0.09, 1.0)
    glass: tuple[float, ...] = (0.05, 0.07, 0.1, 1.0)
    led: tuple[float, ...] = (0.45, 0.9, 1.0, 1.0)

    @classmethod
    def of(cls, style: str | Style | None = None, body=None, accent=None) -> Palette:
        s = get_style(style)
        return cls(
            body=tuple(body) if body is not None else s.default_body,
            accent=tuple(accent) if accent is not None else s.default_accent,
            structure=s.structure_color,
            trim=s.trim_color,
        )

    def color(self, role: str) -> tuple[float, ...]:
        return getattr(self, role)


def draw(
    vis: Visuals,
    body: mujoco.MjsBody,
    asset: Asset,
    palette: Palette,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    rot: np.ndarray | None = None,
) -> None:
    """Add every piece of `asset` to `vis` on `body`, placed by `pos` and `rot` (a 3x3 rotation)."""
    for piece in asset.pieces:
        vis.add(body, piece.material, piece.color or palette.color(piece.role), piece.mesh, pos, rot)


def collide(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    asset: Asset,
    density: float,
    *,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    rot: np.ndarray | None = None,
    visible: bool = False,
    rgba: Sequence[float] = (0.5, 0.5, 0.5, 1.0),
    prefix: str = "",
) -> list[mujoco.MjsGeom]:
    """Give `body` the asset's collision shapes at `density` kg/m^3. They are hidden from the viewer unless
    `visible` (the asset's drawn pieces are what people see)."""
    placed = asset.moved(pos, rot) if (rot is not None or any(pos)) else asset
    geoms = []
    for k, c in enumerate(placed.colliders):
        geoms.append(_add_collider(spec, body, c, density, visible, rgba, f"{prefix}{asset.name}_{k}"))
    return geoms


def _add_collider(spec, body, c: Collider, density: float, visible: bool, rgba, name: str) -> mujoco.MjsGeom:
    kwargs: dict = {"type": _GEOM[c.kind], "density": density, "rgba": list(rgba)}
    kwargs["group"] = VISUAL_GROUP if visible else HIDDEN_GROUP
    if c.friction is not None:
        kwargs["friction"] = [c.friction, 0.02, 0.002]
    if c.kind == "mesh":
        mesh_name = _unique_mesh_name(spec, f"rf_hull_{name}")
        m = c.mesh
        spec.add_mesh(
            name=mesh_name,
            uservert=m.vertices.ravel(),
            userface=m.faces.ravel(),
            inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
        )
        return body.add_geom(meshname=mesh_name, **kwargs)
    kwargs["pos"] = list(c.pos)
    if c.rot is not None and not np.allclose(c.matrix(), np.eye(3)):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(c.matrix(), dtype=float).ravel())
        kwargs["quat"] = quat.tolist()
    size = {
        "box": list(c.size),
        "cylinder": list(c.size),
        "capsule": list(c.size),
        "sphere": list(c.size),
    }[c.kind]
    kwargs["size"] = size + [0.0] * (3 - len(size))
    return body.add_geom(**kwargs)


def _unique_mesh_name(spec: mujoco.MjSpec, base: str) -> str:
    taken = {m.name for m in spec.meshes}
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    return name
