"""Styles: one knob that changes how every generated part looks, not what it does.

A style is a set of finishing choices (how soft the edges are, how round a link's section is, whether screws show,
whether there are panel lines, cable glands, ribs or bumpers) plus the materials and neutral colors that go with
them. Generators read the style they are given; the physics (sizes, masses, joints) never depends on it, only the
drawn shell over the same collision shapes.

Four ship with robotforge:

- `cobot`: an industrial collaborative arm. White gloss shells, dark soft-touch joint bands, round sections, few
  screws, a thin light ring.
- `lab`: a research-lab machine. Brushed and anodised aluminium, squarer extruded sections, visible socket-head
  screws, black end caps, labelled and cabled.
- `hobby`: a servo-and-bracket build. Glossy printed plastic in bright colors, chunky rounded shapes, screws in every
  bracket, a servo horn at each joint.
- `rugged`: a field machine. Painted cast housings with a rough skin, thick fillets, ribs, rubber guards and
  bumpers, bolts you can see from across the yard.

A plugin adds its own with `robotforge.extend.style(...)` (docs/ASSETS.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class Style:
    name: str
    summary: str = ""
    # surfaces, by material name (see assets.materials and design.visuals.PRESETS)
    body_material: str = "gloss"  # the shell, in the robot's own color
    accent_material: str = "gloss"
    structure_material: str = "soft_touch"  # joint bands, end caps, covers: the neutral dark parts
    metal_material: str = "metal"  # screws, flanges
    rubber_material: str = "tread"
    # neutral colors (the shell and accent take the robot's color)
    structure_color: Color = (0.11, 0.115, 0.125, 1.0)
    trim_color: Color = (0.66, 0.68, 0.71, 1.0)
    default_body: Color = (0.92, 0.93, 0.94, 1.0)
    default_accent: Color = (0.16, 0.42, 0.9, 1.0)
    # geometry
    fillet: float = 1.0  # scales every edge radius: 0.5 crisp, 1.5 soft
    roundness: float = 0.85  # a link's section: 0 square extrusion, 1 round tube
    taper: float = 0.85  # a link's tip as a share of its root (1 is a straight tube)
    fasteners: float = 1.0  # screws and bolts: 0 none, 1 a few, 2 many
    panel_lines: bool = True  # a parting line round shells and along links
    cable_glands: bool = True  # a connector or gland on joint housings
    ribs: float = 0.0  # cooling fins and stiffening ribs, 0 none
    bumpers: bool = False  # rubber guards on corners, feet and ends
    light_ring: bool = False  # a thin glowing ring at a joint
    wall: float = 1.0  # how thick-walled and chunky shapes read: scales joint housings and collars

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise ValueError(f"style name {self.name!r} must be lowercase letters, digits and _, at most 24")
        for field, low, high in (
            ("fillet", 0.2, 2.5),
            ("roundness", 0.0, 1.0),
            ("taper", 0.4, 1.2),
            ("fasteners", 0.0, 3.0),
            ("ribs", 0.0, 2.0),
            ("wall", 0.6, 1.6),
        ):
            value = getattr(self, field)
            if not low <= value <= high:
                raise ValueError(f"style {self.name!r}: {field} is between {low} and {high}, got {value!r}")

    def tuned(self, **changes: object) -> Style:
        """A copy with some choices changed (a style a robot needs a little differently)."""
        return replace(self, **changes)


STYLES: dict[str, Style] = {}


def register_style(style: Style, *, replace_existing: bool = False) -> Style:
    if style.name in STYLES and not replace_existing:
        raise ValueError(f"style {style.name!r} already exists; pick another name")
    STYLES[style.name] = style
    return style


def unregister_style(name: str) -> None:
    STYLES.pop(name, None)


DEFAULT_STYLE = "cobot"


def get_style(style: str | Style | None) -> Style:
    """A style by name (None is the default); a Style passes through. The message lists what exists."""
    if isinstance(style, Style):
        return style
    name = DEFAULT_STYLE if style is None else str(style).strip().lower()
    found = STYLES.get(name)
    if found is None:
        raise ValueError(f"unknown style {style!r}; the styles are {', '.join(sorted(STYLES))}")
    return found


register_style(
    Style(
        "cobot",
        "industrial collaborative arm: white gloss shells, dark joint bands, round sections, a thin light ring",
        light_ring=True,
    )
)
register_style(
    Style(
        "lab",
        "research-lab machine: brushed and anodised aluminium, squarer sections, visible socket screws, black end caps",
        body_material="brushed",
        accent_material="anodized",
        structure_material="dark",
        structure_color=(0.07, 0.072, 0.078, 1.0),
        default_body=(0.74, 0.76, 0.79, 1.0),
        default_accent=(0.9, 0.38, 0.1, 1.0),
        fillet=0.7,
        roundness=0.45,
        taper=0.92,
        fasteners=2.0,
    )
)
register_style(
    Style(
        "hobby",
        "servo-and-bracket build: glossy printed plastic in bright colors, chunky shapes, screws in every bracket",
        body_material="shell",
        accent_material="shell",
        structure_material="dark",
        structure_color=(0.13, 0.13, 0.14, 1.0),
        default_body=(0.96, 0.62, 0.12, 1.0),
        default_accent=(0.13, 0.13, 0.14, 1.0),
        fillet=1.3,
        roundness=0.7,
        taper=0.8,
        fasteners=1.6,
        panel_lines=False,
        wall=1.15,
    )
)
register_style(
    Style(
        "rugged",
        "field machine: painted cast housings with a rough skin, thick fillets, ribs, rubber guards and bolts",
        body_material="cast",
        accent_material="cast",
        structure_material="tread",
        structure_color=(0.09, 0.09, 0.095, 1.0),
        trim_color=(0.5, 0.52, 0.54, 1.0),
        default_body=(0.93, 0.72, 0.1, 1.0),
        default_accent=(0.12, 0.12, 0.13, 1.0),
        fillet=1.5,
        roundness=0.6,
        taper=0.95,
        fasteners=1.8,
        ribs=1.0,
        bumpers=True,
        wall=1.3,
    )
)

# What a part's material does to a style: bare aluminium is brushed metal in its own color, carbon is a dark twill,
# printed plastic keeps the style's own shell. (The builder's `material` field and the arm's `material` parameter.)
MATERIAL_BODIES: dict[str, tuple[str, Color] | None] = {
    "pla": None,
    "aluminum": ("brushed", (0.76, 0.78, 0.8, 1.0)),
    "carbon": ("carbon", (0.15, 0.155, 0.165, 1.0)),
    "rubber": ("tread", (0.09, 0.09, 0.1, 1.0)),
}


def for_material(style: Style, material: str) -> Style:
    """`style` as it looks when its shell is made of `material`."""
    found = MATERIAL_BODIES.get(material)
    if found is None:
        return style
    body_material, color = found
    return style.tuned(body_material=body_material, accent_material=body_material, default_body=color)
