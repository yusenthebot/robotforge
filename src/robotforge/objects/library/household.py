"""Things from a kitchen drawer and a desk, in the spirit of the YCB set: a pitcher, a spatula, a roll of tape, a pen
and a sponge. Each has a drawn mesh, convex collision, mass from its material and the grasp a gripper would use."""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, profiles, shapes
from robotforge.objects.kit import Choice, Flag, Gen, Range, shade
from robotforge.objects.library.drinkware import _super
from robotforge.objects.materials import get as material_of
from robotforge.objects.registry import generator


def _along_x(mesh: M.Mesh, x: float, z: float, y: float = 0.0) -> M.Mesh:
    """A mesh built along +z from 0, laid along +x from `x`, its axis at height z."""
    return shapes.lying(mesh).moved((x, y, z))


# ---- pitcher ------------------------------------------------------------------------------------------------------


@generator(
    "pitcher",
    size="[diameter, height]",
    default_size=(0.105, 0.19),
    color="#3a6ea5",
    material="plastic",
    kind="container",
    summary="a jug with a handle on its +x side and a pouring lip on -x (take it by the handle; it holds things "
    "dropped in at its mouth)",
    keywords=("pitcher", "jug", "carafe", "creamer", "water jug", "milk jug"),
    params={
        "material": Choice("plastic", "ceramic", doc="what it is made of (a ceramic one is heavy)", vary=False),
        "handle": Choice("round", "tall", doc="the handle's outline"),
        "wall": Range(0.003, 0.0045, 0.0036, "wall thickness, m"),
        "stripe": Flag(False, "a band of contrasting color round the belly"),
    },
    palette=("#3a6ea5", "white", "#c0392b", "#2e8b57", "#e9b44c", "#303030"),
    hints={"plastic": {"material": "plastic"}, "ceramic": {"material": "ceramic"}},
    builtin=True,
)
def pitcher(g: Gen) -> None:
    d, h = g.size
    R, t = d / 2, g.p("wall")
    material = g.p("material")
    rim = 0.84 * R
    floor = 0.008
    belly = profiles.smooth(
        [(0.84 * R, 0.0), (0.97 * R, 0.014), (R, 0.34 * h), (0.96 * R, 0.68 * h), (rim, h)], 3
    )  # the body swells, then narrows to the mouth
    body = profiles.hollow(belly, t, floor, round_rim=4)
    outer = [(0.84 * R, 0.0), (R, 0.32 * h), (0.98 * R, 0.62 * h), (rim, h)]  # collision: four straight stretches
    g.part(M.lathe(body, 48, sharp=(1,)), material, kit.vessel(outer, max(t, 0.004), floor, n=12))
    # the lip on -x: a flat beak at the rim
    beak = shapes.prism([(-rim + 0.014, 0.034), (-rim - 0.034, 0.0), (-rim + 0.014, -0.034)], h - 0.004, h + 0.001)
    g.part(beak, material, [kit.hull(beak)], mass=0.01)
    # the handle on +x, bigger than a mug's: a tube along a superellipse in the xz plane
    a, b, ex = {"round": (0.044, 0.27 * h, 2.2), "tall": (0.042, 0.33 * h, 2.6)}[g.p("handle")]
    zc = 0.52 * h
    wall_x = 0.98 * R - 0.004
    phi = np.linspace(math.pi / 2, -math.pi / 2, 15)
    cx, cz = _super(phi, ex)
    path = np.column_stack([wall_x + a * cx, np.zeros_like(cx), zc + b * cz])
    tube_r = 0.0085
    tube = M.tube(path, [tube_r] * len(path), segments=14).moved(scale=(1.0, 1.25, 1.0))
    solid = [
        kit.capsule_between(tuple(path[i]), tuple(path[i + 1]), tube_r + 0.0005) for i in range(0, len(path) - 1, 2)
    ]
    g.part(tube, material, solid)
    if g.p("stripe"):
        z0, z1 = 0.4 * h, 0.5 * h
        r0, r1 = 0.99 * R, 0.99 * R
        g.decor(
            M.revolve([(r0, z0), (r1, z1), (r1 + 0.0006, z1), (r0 + 0.0006, z0)], 48, sharp=()),
            shade(g.rgba, 0.45),
            "ceramic",
        )
    x_out = wall_x + a
    g.grasp(
        "handle", (x_out, 0.0, zc), (0.0, 1.0), 2 * tube_r + 0.004, 1.0, "pinch the handle's outer bar from above",
        zc - 0.5 * b, zc + 0.9 * b,
    )  # fmt: skip
    g.anchor("mouth", (0.0, 0.0, h))
    g.anchor("handle", (x_out, 0.0, zc))
    g.fact(kind="vessel", mouth_r=rim - t, floor_z=floor, rim_z=h, handle_x=x_out, inner_r=rim - t)


# ---- spatula ------------------------------------------------------------------------------------------------------


@generator(
    "spatula",
    size="[length]",
    default_size=(0.30,),
    color="#2b2f36",
    material="plastic",
    kind="tool",
    summary="a kitchen spatula lying flat, blade at +x and a round handle to take hold of at -x",
    keywords=("spatula", "turner", "flipper", "pancake turner"),
    params={
        "handle": Choice("wood", "plastic", doc="what the handle is made of"),
        "blade_width": Range(0.055, 0.09, 0.072, "width of the blade, m"),
        "slots": Flag(True, "slots cut in the blade"),
        "grip": Range(0.0105, 0.0145, 0.0125, "radius of the handle, m"),
    },
    palette=("#2b2f36", "#c0392b", "#2e6b4f", "#e9b44c", "#3a6ea5"),
    builtin=True,
)
def spatula(g: Gen) -> None:
    length = g.size[0]
    hr = g.p("grip")
    hl, nl = 0.43 * length, 0.09 * length
    bl, bw = length - hl - nl, g.p("blade_width")
    x0 = -length / 2
    hcolor = [0.62, 0.43, 0.26, 1.0] if g.p("handle") == "wood" else shade(g.rgba, 0.8)
    handle = _along_x(M.cylinder(hr, hl, 28, bevel=0.004, z0=0.0), x0, hr)
    g.part(handle, g.p("handle"), [kit.cylinder(hr, hl / 2, (x0 + hl / 2, 0.0, hr), kit.quat_z_to((1, 0, 0)))], hcolor)
    # the neck: a round bar from the handle's height down to the blade
    xa, xb = x0 + hl, x0 + hl + nl
    neck = M.tube(
        np.array([[xa - 0.004, 0.0, hr], [(xa + xb) / 2, 0.0, 0.5 * hr], [xb, 0.0, 0.0035]]),
        [0.0058, 0.0048, 0.0042],
        12,
    )
    g.part(neck, "plastic", [kit.capsule_between((xa, 0, hr * 0.85), (xb, 0, 0.004), 0.005)], g.rgba)
    thick = 0.003
    blade = M.rounded_box(bl / 2, bw / 2, thick / 2, 0.0012, 1, 0.012).moved((xb + bl / 2, 0.0, thick / 2))
    g.part(blade, "plastic", [kit.box(bl / 2, bw / 2, thick / 2, (xb + bl / 2, 0.0, thick / 2))], g.rgba)
    dark = shade(g.rgba, 0.4)
    if g.p("slots"):
        n = 5
        for k in range(n):
            y = (k - (n - 1) / 2) * bw / (n + 0.6)
            slot = M.rounded_box(bl * 0.34, 0.0032, 0.0002, 0.0002, 1, 0.0032).moved(
                (xb + bl * 0.58, y, thick + 0.0001)
            )
            g.decor(slot, dark, "paint")
    ring = M.torus(0.0042, 0.0013, 20, 6).moved((x0 + 0.012, 0.0, hr + 0.0), M.rot_y(math.pi / 2) @ np.eye(3))
    g.decor(ring, shade(hcolor, 0.5), "metal")  # a hanging hole's grommet at the handle's end
    # the grip goes where the mass is: the blade is light, the handle is wood
    mh = math.pi * hr**2 * hl * material_of(g.p("handle")).density
    mb = bl * bw * thick * material_of("plastic").density
    com = (mh * (x0 + hl / 2) + mb * (xb + bl / 2)) / (mh + mb)
    gx = min(max(com, x0 + 0.03), x0 + hl - 0.02)
    g.grasp("handle", (gx, 0.0, hr), (0.0, 1.0), 2 * hr, 1.0, "across the round handle", hr * 0.8, hr * 1.6)
    g.anchor("handle_end", (x0, 0.0, hr))
    g.anchor("blade", (xb + bl / 2, 0.0, thick))
    g.fact(kind="tool", handle_x=gx, tip_x=x0 + length, axis=(1.0, 0.0, 0.0), firm=True)


# ---- tape ---------------------------------------------------------------------------------------------------------


@generator(
    "tape",
    size="[diameter]",
    default_size=(0.066,),
    color="#d9c08a",
    material="paper",
    kind="thing",
    summary="a roll of tape lying flat, a ring around a cardboard core: take it across the ring, jaws on its two sides",
    keywords=("tape", "roll of tape", "masking tape", "duct tape", "adhesive tape", "scotch tape"),
    params={
        "width": Range(0.014, 0.03, 0.022, "width of the tape, m"),
        "style": Choice("masking", "duct", "painter", doc="the tape's finish"),
    },
    palette=("#d9c08a", "#a7adb3", "#3a6ea5", "#c0392b"),
    hints={
        "duct": {"style": "duct"},
        "silver": {"style": "duct"},
        "masking": {"style": "masking"},
        "painter": {"style": "painter"},
        "blue": {"style": "painter"},
    },  # fmt: skip
    builtin=True,
)
def tape(g: Gen) -> None:
    ro, w = g.size[0] / 2, g.p("width")
    ri = 0.0125  # the core's hole
    core_wall = 0.0015
    style = g.p("style")
    color = {"masking": [0.85, 0.75, 0.55, 1.0], "duct": [0.62, 0.65, 0.68, 1.0], "painter": [0.3, 0.55, 0.82, 1.0]}[
        style
    ]
    if style == "masking":
        color = list(g.rgba)  # the seed's color is the masking tape's own
    edge = 0.0025
    ring = M.revolve(
        [(ri + core_wall, 0.0), (ro - edge, 0.0), (ro, edge), (ro, w - edge), (ro - edge, w), (ri + core_wall, w)],
        48,
        sharp=(),
    )
    preset = "metal" if style == "duct" else "paint"
    g.part(ring, "paper", kit.ring_of_boxes(ro, ro, 0.0, w, ro - ri, 10, math.pi / 10), color, preset=preset)
    core = M.revolve(
        [(ri, -0.0004), (ri + core_wall, -0.0004), (ri + core_wall, w + 0.0004), (ri, w + 0.0004)], 40, sharp=()
    )
    g.decor(core, [0.72, 0.58, 0.4, 1.0], "paint")
    # a lighter tab where the tape's end lifts off the roll, on the band between the core and the edge
    rc = (ri + core_wall + ro) / 2
    tab = M.rounded_box(0.0055, 0.004, 0.0003, 0.0003, 1, 0.002).moved((rc, 0.0, w + 0.0003), M.rot_z(0.5) @ np.eye(3))
    g.decor(tab, shade(color, 1.12), "paint")
    g.grasp(
        "body", (0.0, 0.0, w * 0.5), (0.0, 1.0), 2 * ro, 1.0, "across the roll, jaws on its outside", w * 0.3, w * 0.8
    )
    g.anchor("top", (0.0, 0.0, w))
    g.fact(kind="tape", radius=ro, top_z=w)


# ---- pen -------------------------------------------------------------------------------------------------------


@generator(
    "pen",
    size="[length]",
    default_size=(0.14,),
    color="#c0392b",
    material="plastic",
    kind="tool",
    summary="a felt-tip pen lying on its side, a cap with a clip at -x and the tip at +x: take it across the barrel",
    keywords=("pen", "felt pen", "felt-tip", "sharpie", "highlighter", "permanent marker"),
    params={"radius": Range(0.0085, 0.014, 0.0115, "radius of the barrel, m")},
    palette=("#c0392b", "#2e6b4f", "#3a6ea5", "#303030", "#e9b44c"),
    builtin=True,
)
def pen(g: Gen) -> None:
    length = g.size[0]
    r = g.p("radius")
    x0, x1 = -length / 2, length / 2
    cap_r, cap_l = r * 1.06, 0.048
    flat = math.cos(math.pi / 8)  # the barrel is drawn round but collides as an octagon, so it rests on a flat
    zc = cap_r * flat
    barrel_l = length - cap_l - 0.014
    bx = x0 + cap_l
    white = [0.93, 0.93, 0.92, 1.0]

    def octagon(radius: float, long: float, at: float) -> kit.Piece:
        prism = shapes.prism(shapes.ngon(radius, 8, math.pi / 8), 0.0, long)
        return kit.hull(_along_x(prism, at, zc))

    g.part(
        _along_x(M.cylinder(r, barrel_l, 24, bevel=0.0012, z0=0.0), bx, zc),
        "plastic",
        [octagon(r, barrel_l, bx)],
        white,
        mass=0.012,
    )
    sleeve = _along_x(M.cylinder(r * 1.012, barrel_l * 0.5, 24, z0=0.0), bx + barrel_l * 0.25, zc)
    g.decor(sleeve, g.rgba, "paint")  # the colored sleeve
    cap = _along_x(M.cylinder(cap_r, cap_l, 24, bevel=0.0018, z0=0.0), x0, zc)
    g.part(cap, "plastic", [octagon(cap_r, cap_l, x0)], shade(g.rgba, 0.85), mass=0.004)
    clip = M.rounded_box(0.017, 0.0026, 0.0009, 0.0006, 1, 0.0012).moved((x0 + 0.02, 0.0, zc + cap_r + 0.0012))
    g.decor(clip, [0.8, 0.8, 0.82, 1.0], "metal")
    tip_x = bx + barrel_l
    nose_profile = [(0.0, 0.0), (r, 0.0), (r * 0.62, 0.007), (r * 0.3, 0.014), (0.0, 0.014)]
    nose = shapes.lying(M.lathe(nose_profile, 20, sharp=(1,))).moved((tip_x, 0.0, zc))
    g.part(nose, "plastic", [kit.hull(nose)], white, mass=0.002)
    felt = shapes.lying(M.lathe([(0.0, 0.0), (r * 0.3, 0.0), (r * 0.22, 0.0035), (0.0, 0.0035)], 12))
    g.decor(felt.moved((x1 - 0.003, 0.0, zc)), shade(g.rgba, 0.7), "paint")
    g.grasp("body", (bx + barrel_l * 0.5, 0.0, zc), (0.0, 1.0), 2 * r, 1.0, "across the barrel", zc * 0.8, zc * 1.5)
    g.anchor("tip", (x1, 0.0, zc))
    g.fact(kind="tool", handle_x=bx + barrel_l * 0.5, tip_x=x1, axis=(1.0, 0.0, 0.0), roll_free=True, firm=True)


# ---- sponge -------------------------------------------------------------------------------------------------------


@generator(
    "sponge",
    size="[length, width, thickness]",
    default_size=(0.11, 0.065, 0.03),
    color="#e8c43a",
    material="foam",
    kind="thing",
    summary="a kitchen sponge, a yellow block with a green scouring layer on top: take it across its width",
    keywords=("sponge", "scrubber", "scourer"),
    params={"scour": Choice("green", "blue", "white", doc="color of the scouring layer")},
    palette=("#e8c43a", "#e8e0c0", "#e9a0c0", "#a8d8e8"),
    builtin=True,
)
def sponge(g: Gen) -> None:
    length, width, thick = g.size
    hx, hy = length / 2, width / 2
    layer = min(0.008, thick * 0.3)
    scour = {"green": [0.18, 0.5, 0.25, 1.0], "blue": [0.2, 0.4, 0.7, 1.0], "white": [0.9, 0.9, 0.88, 1.0]}[
        g.p("scour")
    ]
    soft = M.rounded_box(hx, hy, (thick - layer) / 2, 0.004, 3, 0.006).moved((0.0, 0.0, (thick - layer) / 2))
    top = M.rounded_box(hx, hy, layer / 2, 0.0035, 3, 0.006).moved((0.0, 0.0, thick - layer / 2))
    g.part(M.merge([soft, top]), "foam", [kit.box(hx, hy, thick / 2, (0.0, 0.0, thick / 2))], g.rgba, preset="fabric")
    g.decor(top.moved((0.0, 0.0, 0.0003), None), scour, "fabric")
    # pores: a few dark dimples on the soft sides, drawn flat
    rng = g.rng("pores")
    for _ in range(14):
        x, z = rng.uniform(-hx * 0.85, hx * 0.85), rng.uniform(0.004, thick - layer - 0.004)
        g.decor(M.sphere(0.0022, 4, 8).moved((x, -hy + 0.0004, z)), shade(g.rgba, 0.6), "fabric")
    g.grasp(
        "body",
        (0.0, 0.0, thick * 0.5),
        (0.0, 1.0),
        width,
        1.0,
        "across its width, jaws on the long sides",
        thick * 0.3,
        thick * 0.8,
    )
    g.anchor("top", (0.0, 0.0, thick))
    g.fact(kind="sponge", top_z=thick)
