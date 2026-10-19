"""Visual-only dressing for templates and props: procedural meshes with named material presets.

Physics never sees this layer. Every geom added here has `contype = conaffinity = 0` and
`mass = 0`, so contacts, masses and inertias stay exactly what the collision primitives give.
A primitive that a mesh fully covers moves to group 3, which hides it from the browser viewer
and from MuJoCo's renderer (group does not affect physics).

Materials are named `rf_<preset>` (see PRESETS); the geom rgba is the color, the preset says how
the surface reflects. Robots get their names prefixed when attached to a world, so a robot's
geoms arrive as `<robot>/rf_shell`; props keep the bare `rf_wood`, `rf_foliage`, ...

Rangefinders and lidars cast MuJoCo rays, and a ray hits visual geoms too. Dressing that reaches
past its collision shape (a servo case, a screw head) would change what they read, so `Visuals`
can make its meshes transparent to rays (`rays=False`): MuJoCo's ray skips a geom whose material
is fully transparent, while renderers color it with the geom's own rgba.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.design.visuals.mesh import Mesh, merge

VISUAL_GROUP = 2  # drawn by the viewer and MuJoCo's renderer (Menagerie puts its visual meshes here too)
HIDDEN_GROUP = 3  # collision-only primitives covered by a mesh
# MuJoCo's default geom rgba: a geom left at it is drawn in its material's color instead of its own.
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


@dataclass(frozen=True)
class Preset:
    specular: float
    shininess: float
    reflectance: float = 0.0
    emission: float = 0.0
    metallic: float = 0.0
    roughness: float = 0.6


PRESETS: dict[str, Preset] = {
    "shell": Preset(0.5, 0.75, 0.0, roughness=0.35),  # glossy painted plastic (accent / shell color)
    "dark": Preset(0.22, 0.35, roughness=0.6),  # matte dark plastic, anodized structure
    "metal": Preset(0.85, 0.8, 0.04, metallic=1.0, roughness=0.35),  # brushed aluminium, bolts, axles
    "aluminum": Preset(0.75, 0.7, 0.05, metallic=0.8, roughness=0.4),  # anodised aluminium parts in their color
    "carbon": Preset(0.95, 0.95, 0.06, roughness=0.15),  # clear-coated carbon fibre (the viewer adds the weave)
    "rubber": Preset(0.04, 0.05, roughness=0.95),  # tyres, foot pads
    "glass": Preset(1.0, 0.95, 0.08, roughness=0.05),  # lenses, visors
    "led": Preset(0.3, 0.6, emission=0.85, roughness=0.3),  # small glowing indicator
    "wood": Preset(0.1, 0.2, roughness=0.75),
    "bark": Preset(0.02, 0.05, roughness=1.0),
    "foliage": Preset(0.06, 0.12, roughness=0.9),
    "stone": Preset(0.08, 0.12, roughness=0.85),
    "fabric": Preset(0.02, 0.04, roughness=1.0),
    "ceramic": Preset(0.6, 0.85, 0.03, roughness=0.2),
    "paint": Preset(0.18, 0.3, roughness=0.65),  # painted props and walls
}

_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar("robotforge_visuals", default=True)


def enabled() -> bool:
    return _ENABLED.get()


@contextlib.contextmanager
def disabled() -> Iterator[None]:
    """Build robots and props from their bare collision primitives (tests compare physics this way)."""
    token = _ENABLED.set(False)
    try:
        yield
    finally:
        _ENABLED.reset(token)


def material_name(preset: str) -> str:
    if preset not in PRESETS:
        import robotforge.assets.materials  # noqa: F401  (adds the asset engine's surfaces to PRESETS, once)
    if preset not in PRESETS:
        raise ValueError(f"unknown material preset {preset!r}; use one of {', '.join(PRESETS)}")
    return f"rf_{preset}"


def ensure_material(spec: mujoco.MjSpec, preset: str, known: set[str] | None = None, rays: bool = True) -> str:
    """Add the `rf_<preset>` material to `spec` once; returns its name. With `rays` False the material is
    fully transparent, which only rays look at (geoms using it are colored by their own rgba)."""
    name = material_name(preset)
    if known is not None and name in known:
        return name
    # Scan the list: after spec.attach, MuJoCo 3.15's spec.material() misses materials added later.
    if not any(m.name == name for m in spec.materials):
        p = PRESETS[preset]
        spec.add_material(
            name=name,
            specular=p.specular,
            shininess=p.shininess,
            reflectance=p.reflectance,
            emission=p.emission,
            metallic=p.metallic,
            roughness=p.roughness,
            rgba=[1.0, 1.0, 1.0, 1.0 if rays else 0.0],
        )
    if known is not None:
        known.add(name)
    return name


class Visuals:
    """Collects visual meshes per (body, material, color) and adds one mesh geom per group at `finish`.

    Merging keeps the geom count (and the viewer's draw calls) small: a robot leg is a handful of
    geoms however many little parts it shows.
    """

    def __init__(self, spec: mujoco.MjSpec, owner: str | None = None, rays: bool = True) -> None:
        self.spec = spec
        # Mesh names are rf_<owner>_<preset>: the body name for a robot (its spec is its own), the
        # object's name for a prop (unique in the world, and props on the world body share no body name).
        self.owner = owner
        self.rays = rays  # False: rangefinders and lidars see through these meshes (see the module notes)
        self.on = enabled()
        self._groups: dict[tuple, tuple[mujoco.MjsBody, str, list[float], list[Mesh]]] = {}
        self._materials: set[str] = set()

    def add(
        self,
        body: mujoco.MjsBody,
        preset: str,
        rgba: Sequence[float],
        mesh: Mesh,
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        rot: np.ndarray | None = None,
    ) -> None:
        if not self.on:
            return
        material_name(preset)  # fail early on a typo
        color = [float(c) for c in rgba]
        if len(color) == 3:
            color.append(1.0)
        if not self.rays and tuple(color) == DEFAULT_RGBA:  # would be drawn in the transparent material's color
            color = [0.501, 0.501, 0.501, 1.0]
        key = (body.name, preset, tuple(round(c, 4) for c in color))
        if key not in self._groups:
            self._groups[key] = (body, preset, color, [])
        self._groups[key][3].append(mesh.moved(pos, rot) if (rot is not None or any(pos)) else mesh)

    def hide(self, geom: mujoco.MjsGeom) -> None:
        """Hide a primitive that a visual mesh now covers (physics unchanged)."""
        if self.on:
            geom.group = HIDDEN_GROUP

    def finish(self) -> int:
        """Add the collected meshes to the spec; returns the number of triangles added."""
        triangles = 0
        taken: set[str] = set()
        for body, preset, color, meshes in self._groups.values():
            merged = merge(meshes)
            if merged.triangles == 0:
                continue
            owner = (self.owner or body.name or "world").replace("/", "_")
            base = name = f"rf_{owner}_{preset}"
            k = 1
            while name in taken:
                k += 1
                name = f"{base}{k}"
            taken.add(name)
            self.spec.add_mesh(
                name=name,
                uservert=merged.vertices.ravel(),
                usernormal=merged.normals.ravel(),
                userface=merged.faces.ravel(),
                userfacenormal=merged.faces.ravel(),
                # Visual meshes weigh nothing, but MuJoCo still sizes every mesh; the shell rule
                # accepts open surfaces and overlapping parts.
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
            )
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                material=ensure_material(self.spec, preset, self._materials, self.rays),
                rgba=color,
                contype=0,
                conaffinity=0,
                group=VISUAL_GROUP,
                mass=0.0,
            )
            triangles += merged.triangles
        self._groups.clear()
        return triangles


def shade(rgba: Sequence[float], k: float) -> list[float]:
    return [min(1.0, float(c) * k) for c in rgba[:3]] + [float(rgba[3]) if len(rgba) > 3 else 1.0]


def mix(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    return [float(x) + (float(y) - float(x)) * t for x, y in zip(a[:3], b[:3], strict=True)] + [1.0]
