"""The asset engine: parametric generators for robot parts, with a style, materials and mass from density.

    from robotforge.assets import generate, Palette, draw
    asset = generate("joint_actuator", radius=0.04, length=0.09, style="lab")
    asset.mass_properties(2700.0)       # kg, centre of mass and inertia of its collision shapes
    draw(vis, body, asset, Palette.of("lab"), pos, rot)   # its meshes, through design.visuals.Visuals

A generator is a named recipe with parameter ranges and a seed; the same arguments always make the same asset.
Built-in generators: joint_actuator, link_shell, pedestal, chassis, wheel, foot, sensor_housing, grip_pad and
gripper_body (`catalog()` lists them with their parameters). Plugins add their own with
`robotforge.extend.generator`, materials with `robotforge.extend.material` and styles with `robotforge.extend.style`
(docs/ASSETS.md, docs/EXTENDING.md).
"""

from __future__ import annotations

from typing import Any

from robotforge.assets import registry
from robotforge.assets.asset import Asset, Collider, MassProperties, Piece, hulls_of, mesh_solid
from robotforge.assets.materials import MATERIALS, Material, register_material, surface_of
from robotforge.assets.registry import (
    Args,
    AssetError,
    Generator,
    Param,
    generate,
    generator,
    register_generator,
    unregister_generator,
)
from robotforge.assets.styles import STYLES, Style, for_material, get_style, register_style, unregister_style

__all__ = [
    "MATERIALS",
    "STYLES",
    "Args",
    "Asset",
    "AssetError",
    "Collider",
    "Dress",
    "Generator",
    "MassProperties",
    "Material",
    "Palette",
    "Param",
    "Piece",
    "Style",
    "collide",
    "draw",
    "for_material",
    "generate",
    "generator",
    "catalog",
    "get_style",
    "hulls_of",
    "mesh_solid",
    "register_generator",
    "register_material",
    "register_style",
    "surface_of",
    "unregister_generator",
    "unregister_style",
]


def catalog() -> list[dict[str, Any]]:
    """Every generator, with its summary and its parameters' ranges, for listing (agents read this)."""
    return [g.describe() for g in registry.iter_generators()]


def overview(full: bool = True) -> dict[str, Any]:
    """The generators, the styles and the materials, for list_templates. `full` carries every generator's
    parameters with their ranges (about 11 KB); without it, a generator is a name and a sentence."""
    view: dict[str, Any] = {
        "generators": catalog() if full else {g.name: g.summary for g in registry.iter_generators()},
        "styles": {name: s.summary for name, s in STYLES.items()},
        "materials": {name: m.summary for name, m in MATERIALS.items() if m.summary},
        "how": (
            "add_robot(name, template='asset', params={'generator': NAME, 'args': {...}, 'style': STYLE}) shows "
            "one on a turntable; a style is also a parameter of the arm template, a key of a builder spec "
            "(build={'style': ...}) and a word in describe ('a rugged robot dog', 'a lab-grade arm')"
        ),
    }
    if not full:
        view["more"] = "list_templates(detail='assets') adds every generator's parameters with their ranges"
    return view


def __getattr__(name: str) -> Any:
    # The MuJoCo half is loaded when first used: asking for generators or styles does not need it.
    if name in ("Palette", "Dress", "draw", "collide"):
        from robotforge.assets import bridge

        return getattr(bridge, name)
    raise AttributeError(f"module 'robotforge.assets' has no attribute {name!r}")
