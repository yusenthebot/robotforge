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
from robotforge.assets.registry import clamped, generate
from robotforge.assets.styles import Style, for_material, get_style
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


class Dress:
    """A style with its palettes: what a template or the builder draws its parts with.

    `color` is the robot's own color. In a two-tone style (cobot, lab) most shells are the style's neutral body color
    (`base`) and `colored` shells take the robot's color; in the others (hobby, rugged) every shell is the robot's
    color. `dark` is a shell in the style's neutral dark. `draw` makes a generator's asset and puts it on a body;
    sizes are pulled into the generator's ranges, since a template can be asked for any size and should still be
    drawn."""

    def __init__(self, style: str | Style | None = None, material: str = "pla", color=None) -> None:
        from robotforge.design import components

        color = None if color is None else components.color(color)  # a name, a hex code or numbers
        self.base_style, self.material, self.color = get_style(style), material, color
        self.style = for_material(self.base_style, material)
        self.colored = Palette.of(self.style, body=color, accent=None)
        self.base = Palette.of(self.style, accent=color) if self.style.two_tone else self.colored
        self.shell = self.base
        self.dark = Palette.of(self.style, body=self.style.structure_color, accent=color)

    def with_material(self, material: str) -> Dress:
        """The same style and color for a part made of another material (aluminium, carbon, rubber)."""
        return self if material == self.material else Dress(self.base_style, material, self.color)

    def attach(self, vis: Visuals) -> Dress:
        """Draw `vis` in this style: its older preset names (shell, dark) become the style's surfaces, and code that
        draws with generators finds the dress as `vis.dress`."""
        vis.remap = {"shell": self.style.body_material, "dark": self.style.structure_material}
        vis.dress = self
        return self

    def make(self, name: str, seed: int = 0, **params) -> Asset:
        return generate(name, style=self.style, seed=seed, **clamped(name, params))

    def draw(
        self,
        vis: Visuals,
        body: mujoco.MjsBody,
        name: str,
        palette: Palette | None = None,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        rot: np.ndarray | None = None,
        seed: int = 0,
        **params,
    ) -> Asset:
        asset = self.make(name, seed, **params)
        draw(vis, body, asset, palette or self.shell, pos, rot)
        return asset

    def limb(
        self,
        vis: Visuals,
        body: mujoco.MjsBody,
        length: float,
        start: tuple[float, float],
        end: tuple[float, float],
        palette: Palette | None = None,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        rot: np.ndarray | None = None,
        **params,
    ) -> Asset:
        """A tapered, domed limb along +z from 0 to `length`, its section's half extents (x, y) going from `start` to
        `end`: what `design.visuals.mesh.link` draws, as a two-part shell with a parting line."""
        (hx0, hy0), (hx1, hy1) = start, end
        return self.draw(
            vis,
            body,
            "link_shell",
            palette,
            pos,
            rot,
            length=length,
            radius=hx0,
            width=hy0 / hx0,
            tip=hx1 / hx0,
            tip_width=hy1 / hx0,
            ends="dome",
            **params,
        )


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
