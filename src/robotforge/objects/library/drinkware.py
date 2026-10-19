"""Things that hold a drink or a meal: mugs, cups, bowls and plates."""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, profiles
from robotforge.objects.kit import Choice, Flag, Gen, Range, shade
from robotforge.objects.registry import generator

TAU = 2 * math.pi


def _super(phi: np.ndarray, exponent: float) -> tuple[np.ndarray, np.ndarray]:
    """Points on a superellipse: exponent 2 is an ellipse, higher is squarer."""
    c, s = np.cos(phi), np.sin(phi)
    k = 2.0 / exponent
    return np.sign(c) * np.abs(c) ** k, np.sign(s) * np.abs(s) ** k


@generator(
    "mug",
    size="[diameter, height]",
    default_size=(0.082, 0.095),
    color="white",
    material="ceramic",
    kind="container",
    summary="a ceramic mug with a handle on its +x side (a handle to grasp, an open top to pour into or from)",
    keywords=("mug", "coffee mug", "teacup", "tea cup", "stein"),
    params={
        "handle": Choice("round", "square", "tall", doc="the handle's outline"),
        "handle_size": Range(0.85, 1.2, 1.0, "how big the handle is, 1 is usual"),
        "taper": Range(0.0, 0.12, 0.04, "how much narrower the foot is than the rim"),
        "wall": Range(0.0035, 0.0055, 0.0045, "wall thickness, m"),
        "stripe": Flag(False, "a band of contrasting color round the middle"),
    },
    palette=("white", "#2d6a9f", "#c0392b", "#2e8b57", "#e9b44c", "#303030", "#d98880", "#7d3c98"),
    builtin=True,
)
def mug(g: Gen) -> None:
    d, h = g.size
    R, t, taper = d / 2, g.p("wall"), g.p("taper")
    Rb = R * (1 - taper)
    floor = 0.007
    outer = [(Rb - 0.004, 0.0), (Rb, 0.004), (R, h)]
    body = profiles.hollow(outer, t, floor, round_rim=4)
    mesh = M.lathe(body, 44, sharp=(1, len(body) - 2))
    # the handle: a tube along a superellipse in the xz plane, joined to the wall at both ends
    style, scale = g.p("handle"), g.p("handle_size")
    a, b, ex = {"round": (0.030, 0.29, 2.0), "square": (0.029, 0.27, 3.4), "tall": (0.019, 0.38, 2.2)}[style]
    a, b = a * scale, b * h * scale
    zc = h * 0.53
    wall_x = Rb + (R - Rb) * zc / h - 0.003
    phi = np.linspace(math.pi / 2, -math.pi / 2, 15)
    cx, cz = _super(phi, ex)
    path = np.column_stack([wall_x + a * cx, np.zeros_like(cx), zc + b * cz])
    tube = M.tube(path, [0.0062] * len(path), segments=14).moved(scale=(1.0, 1.35, 1.0))
    solid = [kit.capsule_between(tuple(path[i]), tuple(path[i + 1]), 0.0068) for i in range(0, len(path) - 1, 2)]
    g.part(mesh, "ceramic", kit.vessel(outer, max(t, 0.004), floor, n=12))
    g.part(tube, "ceramic", solid)
    if g.p("stripe"):
        z0, z1 = h * 0.38, h * 0.52
        r0, r1 = Rb + (R - Rb) * z0 / h, Rb + (R - Rb) * z1 / h
        g.decor(
            M.revolve([(r0, z0), (r1, z1), (r1 + 0.0005, z1), (r0 + 0.0005, z0)], 44, sharp=()),
            shade(g.rgba, 0.45),
            "ceramic",
        )
    x_out = wall_x + a
    g.grasp(
        "handle", (x_out, 0.0, zc), (0.0, 1.0), 0.022, 1.0, "pinch the handle's outer bar from above, jaws across it"
    )
    g.anchor("mouth", (0.0, 0.0, h))
    g.anchor("handle", (x_out, 0.0, zc))
    g.fact(kind="vessel", mouth_r=R - t, floor_z=floor, rim_z=h, handle_x=x_out, inner_r=R - t)


@generator(
    "cup",
    size="[diameter, height]",
    default_size=(0.075, 0.095),
    color="white",
    material="plastic",
    kind="container",
    summary="a tumbler, paper coffee cup or glass: tapered, open at the top, no handle (take it round the body)",
    keywords=("cup", "tumbler", "glass", "paper cup", "beaker", "coffee cup"),
    params={
        "style": Choice("tumbler", "paper", "glass", doc="what it is made of"),
        "lid": Flag(False, "a lid with a sip hole (closes the top)"),
        "taper": Range(0.1, 0.3, 0.2, "how much narrower the foot is than the rim"),
    },
    palette=("white", "#e74c3c", "#3498db", "#f1c40f", "#2ecc71"),
    hints={"coffee": {"style": "paper"}, "plastic": {"style": "tumbler"}},
    builtin=True,
)
def cup(g: Gen) -> None:
    d, h = g.size
    R, style = d / 2, g.p("style")
    taper = g.p("taper") if style != "glass" else 0.06
    Rb = R * (1 - taper)
    wall, floor = {"tumbler": (0.0018, 0.003), "paper": (0.0012, 0.002), "glass": (0.0028, 0.012)}[style]
    material = {"tumbler": "plastic", "paper": "paper", "glass": "glass"}[style]
    rgba = list(g.rgba)
    if style == "glass":
        rgba[3] = 0.42
    elif style == "tumbler":
        rgba[3] = 0.78
    outer = [(Rb - 0.003, 0.0), (Rb, 0.003), (R, h)]
    body = profiles.hollow(outer, wall, floor, round_rim=3)
    g.part(
        M.lathe(body, 40, sharp=(1, len(body) - 2)),
        material,
        kit.vessel(outer, max(wall, 0.003), max(floor, 0.004), n=12),
        rgba,
    )
    # a rolled rim
    g.decor(
        M.torus(R, 0.0016 if style != "glass" else 0.0022, 40, 6).moved((0, 0, h)),
        rgba,
        "glass" if style == "glass" else "shell",
    )
    if style == "paper":
        z0, z1 = h * 0.28, h * 0.72
        r0, r1 = Rb + (R - Rb) * z0 / h, Rb + (R - Rb) * z1 / h
        g.decor(
            M.revolve([(r0, z0), (r1, z1), (r1 + 0.0016, z1), (r0 + 0.0016, z0)], 40, sharp=()),
            [0.58, 0.42, 0.28, 1.0],
            "paint",
        )
    if g.p("lid"):
        lid = M.revolve([(0.0, h), (R + 0.002, h), (R + 0.002, h + 0.004), (0.0, h + 0.006)], 40)
        g.part(lid, "plastic", [kit.cylinder(R, 0.003, (0, 0, h + 0.003))], [0.15, 0.15, 0.17, 1.0])
    g.grasp("body", (0.0, 0.0, h * 0.5), (0.0, 0.0), d + 0.004, 1.0, "round the body, any jaw direction")
    g.anchor("mouth", (0.0, 0.0, h))
    g.fact(kind="vessel", mouth_r=R - wall, floor_z=floor, rim_z=h, inner_r=R - wall, closed=bool(g.p("lid")))


@generator(
    "bowl",
    size="[diameter, height]",
    default_size=(0.15, 0.065),
    color="white",
    material="ceramic",
    kind="container",
    summary="a bowl: wide, open, with a rim to pinch (or to pour into: a bowl takes things dropped in it)",
    keywords=("bowl", "dish", "basin", "salad bowl"),
    params={
        "round": Range(0.4, 1.0, 0.8, "0 flared like a cone, 1 round like half a ball"),
        "wall": Range(0.0035, 0.0065, 0.005, "wall thickness, m"),
    },
    palette=("white", "#e8d8b0", "#6c8ebf", "#c0392b"),
    builtin=True,
)
def bowl(g: Gen) -> None:
    d, h = g.size
    R, t, rnd = d / 2, g.p("wall"), g.p("round")
    rf = 0.36 * R
    theta = np.linspace(0.0, math.pi / 2, 9)
    outer = [(rf + (R - rf) * math.sin(a), h * ((1 - rnd) * math.sin(a) + rnd * (1 - math.cos(a)))) for a in theta]
    outer = [(rf - 0.004, 0.0), *[(r, max(z, 0.004)) for r, z in outer]]
    body = profiles.hollow(outer, t, 0.008, round_rim=4)
    g.part(M.lathe(body, 48, sharp=(1,)), "ceramic", _bowl_pieces(outer, t, h))
    g.grasp("rim", (R - 0.004, 0.0, h - 0.006), (1.0, 0.0), 0.012, 1.0, "pinch the rim: one finger in, one out")
    g.grasp("rim", (-(R - 0.004), 0.0, h - 0.006), (1.0, 0.0), 0.012, 0.9, "the rim on the far side")
    g.anchor("mouth", (0.0, 0.0, h))
    g.fact(kind="vessel", mouth_r=R - t, floor_z=0.008, rim_z=h, inner_r=R - t, rim_r=R)


def _bowl_pieces(outer: list[tuple[float, float]], wall: float, h: float) -> list[kit.Piece]:
    """A bowl's wall in about four stretches (a ring of boxes each) and its foot."""
    pts = outer[1:]
    pick = sorted({0, len(pts) // 4, len(pts) // 2, 3 * len(pts) // 4, len(pts) - 1})
    path = [pts[i] for i in pick]
    pieces = [kit.cylinder(max(pts[0][0] + 0.01, 0.02), 0.005, (0, 0, 0.005))]
    for (ra, za), (rb, zb) in zip(path[:-1], path[1:], strict=False):
        pieces += kit.ring_of_boxes(ra, rb, za, zb, max(wall, 0.005), 14)
    return pieces


@generator(
    "plate",
    size="[diameter, height]",
    default_size=(0.21, 0.022),
    color="white",
    material="ceramic",
    kind="container",
    summary="a plate: flat with a raised rim (pinch the rim, or put things on it)",
    keywords=("plate", "saucer", "platter"),
    params={"rim": Range(0.2, 0.32, 0.26, "rim width as a share of the diameter")},
    palette=("white", "#e8d8b0", "#6c8ebf"),
    builtin=True,
)
def plate(g: Gen) -> None:
    d, h = g.size
    R = d / 2
    rim = g.p("rim")
    r_flat = R * (1 - rim)
    outer = [(r_flat * 0.5, 0.0), (r_flat, 0.0015), (r_flat + (R - r_flat) * 0.45, h * 0.35), (R, h)]
    body = profiles.hollow(outer, 0.0035, 0.0035, round_rim=3)
    pieces = [kit.cylinder(r_flat, 0.0035, (0, 0, 0.0035))]
    pieces += kit.ring_of_boxes(r_flat, R, 0.0015, h, 0.0055, 16)
    g.part(M.lathe(body, 48, sharp=()), "ceramic", pieces)
    g.grasp(
        "rim",
        (R - 0.004, 0.0, h * 0.6),
        (1.0, 0.0),
        0.012,
        1.0,
        "pinch the rim: one finger over the plate, one outside",
    )
    g.grasp("rim", (-(R - 0.004), 0.0, h * 0.6), (1.0, 0.0), 0.012, 0.9, "the rim on the far side")
    g.anchor("top", (0.0, 0.0, 0.0035))
    g.fact(kind="plate", flat_r=r_flat, top_z=0.0035, rim_z=h, rim_r=R)
