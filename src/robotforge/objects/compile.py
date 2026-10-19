"""A generated object into a MuJoCo spec: convex collision pieces with their masses and friction, and its meshes.

Collision geoms go to group 3 (the viewer and the renderer skip them) once the visual meshes are on, and stay drawn
when visuals are switched off (`visuals.disabled()`, which tests use to compare physics). The first piece carries
the object's name, so a task or a behavior finds a fixed object by it as it does any other.
"""

from __future__ import annotations

import mujoco

from robotforge.design import visuals
from robotforge.design.visuals import mesh as M
from robotforge.objects import materials
from robotforge.objects.kit import Built

G = mujoco.mjtGeom
KINDS = {
    "box": G.mjGEOM_BOX,
    "cylinder": G.mjGEOM_CYLINDER,
    "capsule": G.mjGEOM_CAPSULE,
    "sphere": G.mjGEOM_SPHERE,
    "ellipsoid": G.mjGEOM_ELLIPSOID,
    "hull": G.mjGEOM_MESH,
}


CYLINDER_SIDES = 24


def _prism(radius: float, half_height: float) -> M.Mesh:
    """A cylinder along z as a convex prism, centred on its own origin."""
    h = max(half_height, 1e-5)
    return M.lathe([(0.0, -h), (radius, -h), (radius, h), (0.0, h)], CYLINDER_SIDES, sharp=(1, 2))


def add_to_spec(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    owner: str,
    built: Built,
    mass: float | None = None,
    movable: bool = True,
) -> None:
    """Add `built` to `body`. `mass` (kg) replaces the mass its materials give; `movable` False leaves out masses."""
    show = visuals.enabled()
    scale = mass / built.mass if mass is not None and built.mass > 0 else 1.0
    color = list(built.visuals[0].rgba) if built.visuals else [0.6, 0.6, 0.6, 1.0]
    hulls = 0
    for i, piece in enumerate(built.pieces):
        name = owner if i == 0 else f"{owner}/{i}"
        kwargs: dict = {
            "name": name,
            "type": KINDS[piece.kind],
            "pos": list(piece.pos),
            "quat": list(piece.quat),
            "rgba": color,
            "friction": list(materials.get(piece.material).coefficients()),
            "condim": 4,
        }
        if piece.kind in ("hull", "cylinder"):
            # A cylinder is a prism here: MuJoCo's own cylinder against the small boxes of a gripper's pads makes
            # contacts that lock (a can stayed jammed between open fingers and was lifted away with them).
            mesh = piece.hull if piece.kind == "hull" else _prism(piece.size[0], piece.size[1])
            if piece.kind == "cylinder":
                kwargs["type"] = G.mjGEOM_MESH
            hulls += 1
            mesh_name = f"{owner}_hull{hulls}"
            spec.add_mesh(name=mesh_name, uservert=mesh.vertices.ravel(), userface=mesh.faces.ravel())
            kwargs["meshname"] = mesh_name
        else:
            kwargs["size"] = list(piece.size) + [0.0] * (3 - len(piece.size))
        if movable:
            kwargs["mass"] = piece.mass * scale
        if not piece.solid:
            kwargs.update(contype=0, conaffinity=0)
        if show:
            kwargs["group"] = visuals.HIDDEN_GROUP
        body.add_geom(**kwargs)
    if not show:
        return
    drawn = visuals.Visuals(spec, owner=owner)
    if not drawn.on:
        return
    for v in built.visuals:
        drawn.add(body, v.preset, v.rgba, v.mesh)
    drawn.finish()
