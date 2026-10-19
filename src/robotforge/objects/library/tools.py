"""Hand tools lying on a bench: a screwdriver, a hammer and a wrench (take them by the shaft or the handle)."""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, labels, shapes
from robotforge.objects.kit import Choice, Gen, Range, shade
from robotforge.objects.registry import generator

STEEL = [0.74, 0.76, 0.79, 1.0]
DARK_STEEL = [0.22, 0.23, 0.26, 1.0]
WOOD = [0.62, 0.43, 0.26, 1.0]
# (x, y, z) -> (z, x, y): a polygon drawn in (y, z) and extruded along x
Y_Z_TO_X = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _along_x(mesh: M.Mesh, x: float, z: float, y: float = 0.0) -> M.Mesh:
    """A mesh built along +z from 0, laid along +x from `x`, its axis at height z."""
    return shapes.lying(mesh).moved((x, y, z))


@generator(
    "screwdriver",
    size="[length]",
    default_size=(0.21,),
    color="#e0a31b",
    material="plastic",
    kind="tool",
    summary="a screwdriver lying on its side: a fluted handle (+shaft hint on the steel shaft) with a flat or "
    "Phillips tip; take it across the handle or the shaft",
    keywords=("screwdriver", "driver"),
    params={"tip": Choice("flat", "phillips"), "grip": Range(0.012, 0.0155, 0.0135, "grip radius, m")},
    builtin=True,
)
def screwdriver(g: Gen) -> None:
    length = g.size[0]
    hr = g.p("grip")
    hl = 0.46 * length
    x0 = -length / 2
    zc = hr * math.cos(math.pi / 8)  # the handle rests on one of its flats
    handle = _along_x(shapes.fluted(hr, hl, 6, 0.07, 48, taper=0.05), x0, zc)
    flat = shapes.prism(shapes.ngon(hr * 0.985, 8, math.pi / 8), 0.0, hl)
    g.part(handle, "plastic", [kit.hull(_along_x(flat, x0, zc))], g.rgba)
    g.decor(_along_x(M.cylinder(hr * 0.74, 0.003, 24, z0=-0.0004), x0, zc), shade(g.rgba, 0.3), "rubber")
    # ferrule and shaft
    fx = x0 + hl
    g.part(_along_x(M.cylinder(0.0078, 0.016, 20, bevel=0.002, z0=0.0), fx, zc), "steel",
           [kit.cylinder(0.0078, 0.008, (fx + 0.008, 0, zc), kit.quat_z_to((1, 0, 0)))], STEEL)  # fmt: skip
    sx0 = fx + 0.016
    shaft_len = x0 + length - sx0
    g.part(_along_x(M.cylinder(0.0033, shaft_len, 16, z0=0.0), sx0, zc), "steel",
           [kit.cylinder(0.0036, shaft_len / 2, (sx0 + shaft_len / 2, 0, zc), kit.quat_z_to((1, 0, 0)))],
           STEEL)  # fmt: skip
    tip_x = x0 + length
    if g.p("tip") == "flat":
        g.decor(M.rounded_box(0.0035, 0.0034, 0.0006, 0.0003, 1).moved((tip_x - 0.0035, 0, zc)), STEEL, "metal")
    else:
        g.decor(
            M.cylinder(0.0025, 0.004, 12, z0=0.0).moved(rot=M.Z_TO_X).moved((tip_x - 0.004, 0, zc)),
            shade(STEEL, 0.85),
            "metal",
        )
    g.grasp("handle", (x0 + hl * 0.5, 0.0, zc), (0.0, 1.0), 2 * hr, 1.0, "across the fluted handle", zc * 0.8, zc * 1.6)
    g.grasp(
        "shaft",
        (sx0 + 0.035, 0.0, zc),
        (0.0, 1.0),
        0.0072,
        0.7,
        "pinch the steel shaft close to the handle",
        zc * 0.8,
        zc * 1.4,
    )
    g.anchor("handle_end", (x0, 0.0, zc))
    g.anchor("tip", (tip_x, 0.0, zc))
    g.fact(kind="tool", handle_x=x0 + hl * 0.5, tip_x=tip_x, axis=(1.0, 0.0, 0.0))


@generator(
    "hammer",
    size="[length]",
    default_size=(0.27,),
    color="#a5724a",
    material="wood",
    kind="tool",
    summary="a claw hammer lying on its side, head at +x with its face toward -y: a wooden shaft to take hold of "
    "(not the head); light enough for a Panda or a UR5e",
    keywords=("hammer", "mallet"),
    params={"handle": Choice("wood", "rubber", "fiberglass"), "head": Range(0.24, 0.4, 0.3, "head mass, kg")},
    builtin=True,
)
def hammer(g: Gen) -> None:
    length = g.size[0]
    x0, x1 = -length / 2, length / 2
    hx = x1 - 0.019  # the head's middle along the shaft
    handle_color = {"wood": WOOD, "rubber": [0.12, 0.12, 0.14, 1.0], "fiberglass": [0.9, 0.55, 0.12, 1.0]}[
        g.p("handle")
    ]
    hw, ht = 0.0125, 0.009  # half width, half thickness of the shaft
    shaft_len = hx - 0.012 - x0
    shaft = M.rounded_box(shaft_len / 2, hw, ht, 0.0075, 3, 0.0075).moved((x0 + shaft_len / 2, 0, ht))
    g.part(shaft, "wood", [kit.box(shaft_len / 2, hw * 0.92, ht, (x0 + shaft_len / 2, 0, ht))], handle_color)
    g.decor(
        M.rounded_box(0.0035, hw + 0.0006, ht + 0.0006, 0.004, 2, 0.004).moved((x0 + 0.0035, 0, ht)),
        shade(handle_color, 0.45),
        "rubber",
    )
    # the head: a block, a round face toward -y and a curved claw toward +y
    hh = 0.015  # half of its height and its thickness along the shaft
    block = M.rounded_box(hh, 0.032, hh, 0.004, 2).moved((hx, 0.0, hh * 2 * 0.5))
    face = (
        M.cylinder(0.0155, 0.022, 28, bevel=0.002, z0=0.0)
        .moved(rot=M.rot_x(math.pi / 2) @ np.eye(3))
        .moved((hx, -0.032 + 0.0, 0.0155 + 0.0))
    )
    claw_poly = np.array([[0.032, 0.0], [0.07, 0.012], [0.078, 0.02], [0.065, 0.016], [0.032, 0.03]])
    claw = shapes.prism(claw_poly, -hh * 0.75, hh * 0.75).moved(rot=Y_Z_TO_X).moved((hx, 0, 0))
    steel_mass = g.p("head")
    head_pieces = [
        kit.box(hh, 0.032, hh, (hx, 0.0, hh)),
        kit.cylinder(0.0155, 0.011, (hx, -0.043, 0.0155), kit.mat_to_quat(M.rot_x(math.pi / 2))),
        kit.hull(claw),
    ]
    g.part(M.merge([block, face, claw]), "steel", head_pieces, STEEL, mass=steel_mass)
    g.decor(
        M.cylinder(0.011, 0.0008, 24, z0=0.0).moved(rot=M.rot_x(math.pi / 2)).moved((hx, -0.0542, 0.0155)),
        shade(STEEL, 1.12),
        "metal",
    )
    grip_x = hx - 0.012 - 0.07
    g.grasp(
        "shaft",
        (grip_x, 0.0, ht),
        (0.0, 1.0),
        2 * hw,
        1.0,
        "across the shaft, 7 cm behind the head",
        ht * 0.7,
        ht * 1.8,
    )
    g.grasp(
        "shaft",
        (x0 + shaft_len * 0.5, 0.0, ht),
        (0.0, 1.0),
        2 * hw,
        0.75,
        "across the middle of the shaft",
        ht * 0.7,
        ht * 1.8,
    )
    g.anchor("handle_end", (x0, 0.0, ht))
    g.anchor("head", (hx, 0.0, hh))
    g.fact(kind="tool", handle_x=x0 + shaft_len * 0.5, head_x=hx, axis=(1.0, 0.0, 0.0))


@generator(
    "wrench",
    size="[length]",
    default_size=(0.2,),
    color="#9aa3ad",
    material="steel",
    kind="tool",
    summary="a combination wrench lying flat: an open jaw at +x and a ring at -x; take it across the shaft",
    keywords=("wrench", "spanner"),
    params={"finish": Choice("chrome", "black"), "size_mm": Range(8.0, 17.0, 13.0, "the nut it fits, mm")},
    builtin=True,
)
def wrench(g: Gen) -> None:
    length = g.size[0]
    t = 0.011
    x0, x1 = -length / 2, length / 2
    mm = g.p("size_mm") / 1000.0
    color = STEEL if g.p("finish") == "chrome" else DARK_STEEL
    preset = "metal"
    ring_r = mm * 1.1 + 0.002
    open_hw = mm * 1.35 + 0.004
    slot = mm * 0.5 + 0.0005
    jaw_len = open_hw * 1.6
    neck = (0.0092, 0.0112)  # half widths of the shaft: at the ring, at the open end
    rx = x0 + ring_r
    sx0, sx1 = rx + ring_r * 0.7, x1 - jaw_len - 0.002
    polys = {
        "shaft": np.array([[sx0, -neck[0]], [sx1, -neck[1]], [sx1, neck[1]], [sx0, neck[0]]]),
        "back": shapes.rect(x1 - jaw_len - 0.002, -open_hw, x1 - jaw_len * 0.45, open_hw),
        "left": shapes.rect(x1 - jaw_len * 0.45 - 0.0004, slot, x1, open_hw),
        "right": shapes.rect(x1 - jaw_len * 0.45 - 0.0004, -open_hw, x1, -slot),
    }
    pieces, meshes = [], []
    for poly in polys.values():
        m = shapes.prism(poly, 0.0, t)
        meshes.append(m)
        pieces.append(kit.hull(m))
    ring = M.revolve([(mm * 0.62, 0.0), (ring_r, 0.0), (ring_r, t), (mm * 0.62, t)], 6 * 4, sharp=(0, 1, 2, 3)).moved(
        (rx, 0, 0)
    )
    meshes.append(ring)
    pieces.append(kit.cylinder(ring_r, t / 2, (rx, 0, t / 2)))
    g.part(M.merge(meshes), "steel", pieces, color, preset=preset)
    stamp = labels.Sheet((0, 0, t), (1, 0, 0), (0, 1, 0))
    stamp.text(shade(color, 0.5), f"{round(g.p('size_mm')):d}", (sx0 + sx1) / 2, -0.0035, 0.007, "center")
    for rgba, mesh in stamp.patches:
        g.decor(mesh, rgba, "metal")
    g.grasp(
        "shaft",
        ((sx0 + sx1) / 2, 0.0, 0.0085),
        (0.0, 1.0),
        2 * 0.0102,
        1.0,
        "across the shaft, jaws on its edges",
        0.0075,
        0.011,
    )
    g.anchor("ring", (rx, 0.0, t))
    g.anchor("jaw", (x1, 0.0, t / 2))
    g.fact(kind="tool", handle_x=(sx0 + sx1) / 2, axis=(1.0, 0.0, 0.0))
