"""Surface materials the asset engine adds to the viewer's own presets: how a part reflects, and which procedural
texture it wears.

`design.visuals.PRESETS` names the surfaces a mesh can have (`rf_<name>` in the compiled model); MuJoCo's material
carries only what its own renderer reads (specular, roughness, metallic). The viewer is three.js and can do more:
a clear coat, sheen, and a procedural texture drawn from the mesh's own coordinates (brushed metal, rubber grain,
knurling) with no image file and no UVs. A `Material` holds all of it; registering one adds its MuJoCo half to
`PRESETS` and sends its viewer half in the scene description (`surfaces`), so a plugin's new material reaches the
browser without a change to the viewer.

The textures the viewer knows: none, brushed, weave (carbon twill), speckle (orange peel or micro-blasting),
rubber (fine grain), knurl (diamond cross-hatch). A material naming another texture is refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TEXTURES = ("none", "brushed", "weave", "speckle", "rubber", "knurl")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")


@dataclass(frozen=True)
class Material:
    """One surface. Name it like a preset (`brushed`); a mesh uses it as material `rf_brushed`."""

    name: str
    roughness: float = 0.5
    metalness: float = 0.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.15
    sheen: float = 0.0
    glow: float = 0.0  # emission, as a multiple of the color
    env: float = 1.0  # how strongly it reflects the studio
    texture: str = "none"
    texture_scale: float = 1.0  # 1 is the texture's natural pitch; 2 makes its features twice as coarse
    texture_strength: float = 1.0  # 0 flat, 1 as designed
    summary: str = ""

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise ValueError(f"material name {self.name!r} must be lowercase letters, digits and _, at most 24")
        if self.texture not in TEXTURES:
            raise ValueError(
                f"material {self.name!r}: texture must be one of {', '.join(TEXTURES)}, got {self.texture!r}"
            )
        for field in ("roughness", "metalness", "clearcoat", "clearcoat_roughness", "sheen", "texture_strength"):
            if not 0.0 <= getattr(self, field) <= 1.0:
                raise ValueError(f"material {self.name!r}: {field} is between 0 and 1, got {getattr(self, field)!r}")
        if not 0.1 <= self.texture_scale <= 20.0:
            raise ValueError(f"material {self.name!r}: texture_scale is between 0.1 and 20")

    def surface(self) -> dict:
        """What the viewer needs, as JSON."""
        out: dict = {"roughness": self.roughness, "metalness": self.metalness}
        if self.clearcoat > 0:
            out["clearcoat"] = self.clearcoat
            out["clearcoatRoughness"] = self.clearcoat_roughness
        if self.sheen > 0:
            out["sheen"] = self.sheen
        if self.glow > 0:
            out["glow"] = self.glow
        if self.env != 1.0:
            out["env"] = self.env
        if self.texture != "none":
            out["texture"] = self.texture
            out["textureScale"] = self.texture_scale
            out["textureStrength"] = self.texture_strength
        return out


MATERIALS: dict[str, Material] = {}


def register_material(material: Material, *, replace: bool = False) -> Material:
    """Add a surface: to this table (the viewer's half) and to the visuals presets (MuJoCo's half)."""
    from robotforge.design.visuals import PRESETS, Preset

    if (material.name in MATERIALS or material.name in PRESETS) and not replace:
        raise ValueError(f"material {material.name!r} already exists; pick another name")
    MATERIALS[material.name] = material
    shine = 1.0 - material.roughness
    PRESETS[material.name] = Preset(
        specular=round(0.15 + 0.8 * shine, 3),
        shininess=round(0.2 + 0.7 * shine, 3),
        reflectance=round(0.05 * material.metalness, 3),
        emission=min(1.0, material.glow * 0.4),
        metallic=material.metalness,
        roughness=material.roughness,
    )
    return material


def unregister_material(name: str) -> None:
    from robotforge.design.visuals import PRESETS

    if MATERIALS.pop(name, None) is not None:
        PRESETS.pop(name, None)


def surface_of(material_name: str) -> dict | None:
    """The viewer's half of the material a model calls `rf_<name>` (a robot's prefix `robot/` is ignored); None
    for a material the viewer already knows by name (shell, dark, metal, ...) or one that is not ours."""
    base = material_name[material_name.rfind("/") + 1 :]
    if not base.startswith("rf_"):
        return None
    found = MATERIALS.get(base[3:])
    return found.surface() if found is not None else None


def _builtin() -> None:
    for m in (
        Material(
            "gloss",
            roughness=0.26,
            clearcoat=0.85,
            clearcoat_roughness=0.12,
            texture="speckle",
            texture_scale=0.5,
            texture_strength=0.12,
            summary="glossy painted shell, a clear coat over a faint orange peel (a cobot's white casing)",
        ),
        Material(
            "soft_touch",
            roughness=0.62,
            metalness=0.05,
            sheen=0.25,
            texture="speckle",
            texture_scale=0.6,
            texture_strength=0.3,
            summary="matte soft-touch plastic with a fine grain (joint bands, grips)",
        ),
        Material(
            "brushed",
            roughness=0.36,
            metalness=1.0,
            env=1.25,
            texture="brushed",
            summary="brushed aluminium or stainless: fine streaks along the part's length",
        ),
        Material(
            "anodized",
            roughness=0.34,
            metalness=0.85,
            clearcoat=0.12,
            clearcoat_roughness=0.35,
            env=1.4,
            texture="speckle",
            texture_scale=0.3,
            texture_strength=0.22,
            summary="coloured anodised aluminium, micro-blasted",
        ),
        Material(
            "cast",
            roughness=0.58,
            metalness=0.2,
            texture="speckle",
            texture_scale=1.4,
            texture_strength=0.55,
            summary="painted cast housing with a rough skin (a field robot's body)",
        ),
        Material(
            "powder",
            roughness=0.66,
            texture="speckle",
            texture_scale=1.0,
            texture_strength=0.4,
            summary="powder-coated sheet metal, a fine texture",
        ),
        Material(
            "tread",
            roughness=0.97,
            texture="rubber",
            summary="moulded rubber with a fine grain (tyres, foot pads, bumpers)",
        ),
        Material(
            "knurled",
            roughness=0.45,
            metalness=1.0,
            env=1.2,
            texture="knurl",
            texture_scale=0.6,
            summary="knurled steel: a diamond cross-hatch (grips, thumb nuts, shafts)",
        ),
    ):
        register_material(m)


_builtin()
