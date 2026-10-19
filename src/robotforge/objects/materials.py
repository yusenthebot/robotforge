"""What things are made of: density for the mass, friction for the grip, a surface look for the viewer.

A generated object gets its mass from the volume of its meshes times the density of what each mesh is made of, and
its friction from the same table, so a steel wrench is heavy and slippery-smooth, a rubber ball is light and grips.
Friction is MuJoCo's three coefficients: sliding, torsional (m) and rolling (m). Where two surfaces meet MuJoCo
takes the larger of the two, so a gripper pad (friction 1 and more) always decides the grip on a hard object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    density: float  # kg/m^3 of the solid (a hollow object is its shell's volume times this)
    friction: float  # sliding coefficient against a typical gripper pad or table
    preset: str  # visuals.PRESETS name: how the surface reflects light
    torsion: float = 0.005  # m: how much a contact patch resists turning about its normal
    rolling: float = 0.0002  # m: how much it resists rolling

    def coefficients(self) -> tuple[float, float, float]:
        return (self.friction, self.torsion, self.rolling)


MATERIALS: dict[str, Material] = {
    m.name: m
    for m in (
        Material("ceramic", 2400.0, 0.5, "ceramic"),
        Material("glass", 2500.0, 0.4, "glass"),
        Material("steel", 7850.0, 0.55, "metal"),
        Material("aluminum", 2700.0, 0.5, "aluminum"),
        Material("plastic", 950.0, 0.5, "shell"),  # PET and HDPE bottles, molded handles
        Material("rubber", 1100.0, 1.1, "rubber", torsion=0.01),
        Material("wood", 650.0, 0.5, "wood"),
        Material("paper", 800.0, 0.45, "paint"),  # cardboard, book covers, labels
        Material("fruit", 880.0, 0.6, "paint", torsion=0.008),
        Material("foam", 150.0, 0.8, "fabric", torsion=0.01),
        Material("water", 1000.0, 0.3, "glass"),
    )
}
WATER = MATERIALS["water"]
DEFAULT = MATERIALS["plastic"]


def get(name: str) -> Material:
    """The material called `name`; ValueError naming the ones there are."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(f"unknown material {name!r}; use one of {', '.join(MATERIALS)}") from None
