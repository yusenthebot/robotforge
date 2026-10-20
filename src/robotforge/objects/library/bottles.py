"""Bottles and cans: round things to take hold of round the body."""

from __future__ import annotations

import math

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, labels, profiles
from robotforge.objects.kit import Choice, Flag, Gen, Range, shade
from robotforge.objects.registry import generator

WORDS = labels.WORDS


@generator(
    "bottle",
    size="[diameter, height]",
    default_size=(0.066, 0.215),
    color="#5aa7d6",
    material="plastic",
    kind="container",
    summary="a bottle with a cap and a printed label, part full of water (take it round the body)",
    keywords=("bottle", "flask", "water bottle", "wine bottle", "soda bottle", "juice bottle"),
    params={
        "style": Choice("water", "soda", "wine", doc="outline: ribbed water bottle, curved soda bottle or wine bottle"),
        "fill": Range(0.0, 1.0, 0.8, "how full of water, 0 empty and 1 full"),
        "label": Flag(True, "a printed label round the middle"),
        "cap": Choice("white", "blue", "black", "red", doc="the cap's color"),
    },
    palette=("#5aa7d6", "#2e8b57", "#8e44ad", "#d35400", "#2c3e50"),
    hints={
        "cola": {"style": "soda"},
        "soda": {"style": "soda"},
        "wine": {"style": "wine"},
        "water": {"style": "water"},
    },
    builtin=True,
)
def bottle(g: Gen) -> None:
    d, h = g.size
    R, style = d / 2, g.p("style")
    shoulder0, shoulder1, neck_r = {
        "water": (0.58, 0.78, 0.36),
        "soda": (0.50, 0.76, 0.32),
        "wine": (0.62, 0.84, 0.32),
    }[style]
    z_body = 0.02 if style != "wine" else 0.012
    body_pts = [(R * 0.8, 0.0), (R * 0.97, 0.006), (R, z_body + 0.006), (R, h * shoulder0)]
    neck_z = h * shoulder1
    sh = profiles.smooth(
        [(R, h * shoulder0), (R * 0.97, h * shoulder0 + 0.35 * (neck_z - h * shoulder0)),
         (R * 0.6, h * shoulder0 + 0.8 * (neck_z - h * shoulder0)), (R * neck_r, neck_z)], 3
    )  # fmt: skip
    neck_top = h * 0.935
    outer = [
        *body_pts,
        *sh[1:],
        (R * neck_r, neck_top),
        (R * neck_r * 1.22, neck_top + 0.001),
        (R * neck_r * 1.22, neck_top + 0.005),
        (R * neck_r, neck_top + 0.006),
    ]
    wall = 0.0006 if style != "wine" else 0.0035
    material = "plastic" if style != "wine" else "glass"
    rgba = list(g.rgba)
    if style == "wine":
        rgba = shade(g.rgba, 0.35)[:3] + [0.85]
    else:
        rgba[3] = 0.72
    shell = profiles.hollow(outer, wall, wall * 2, round_rim=2)
    body_h = h * shoulder0
    pieces = [
        kit.frustum(R * 0.8, R, 0.0, z_body + 0.006),
        kit.cylinder(R, (body_h - z_body - 0.006) / 2, (0, 0, (body_h + z_body + 0.006) / 2)),
        kit.frustum(R, R * neck_r, body_h, neck_z),
        kit.cylinder(R * neck_r * 1.05, (neck_top - neck_z) / 2, (0, 0, (neck_z + neck_top) / 2)),
    ]
    g.part(M.lathe(shell, 40, sharp=(1,)), material, pieces, rgba)
    interior = math.pi * (R - wall) ** 2 * (body_h - z_body) + math.pi * ((R * neck_r) ** 2) * (neck_top - neck_z)
    fill = g.p("fill")
    if fill > 0.01:
        mid = (z_body + body_h) / 2
        g.contents("water", interior * fill * 0.9, (0, 0, mid), R - wall, (body_h - z_body) / 2)
    # the cap
    cap_color = {
        "white": [0.93, 0.93, 0.92, 1],
        "blue": [0.12, 0.3, 0.7, 1],
        "black": [0.08, 0.08, 0.09, 1],
        "red": [0.75, 0.1, 0.1, 1],
    }[g.p("cap")]
    cap_r = R * neck_r * 1.18
    cap = M.cylinder(cap_r, h - neck_top + 0.004, 28, bevel=0.0015, z0=neck_top - 0.004)
    g.part(
        cap,
        "plastic",
        [kit.cylinder(cap_r, (h - neck_top + 0.004) / 2, (0, 0, (h + neck_top - 0.004) / 2))],
        cap_color,
        mass=0.004,
    )
    # the label: a band round the body with a name, a ring and a stripe
    if g.p("label") and style != "wine":
        z0, z1 = h * 0.26, h * 0.5
        wrap = labels.Wrap(R, 0.0)
        wrap.band([0.97, 0.97, 0.95, 1.0], z0, z1)
        word = WORDS[int(g.rng("word").integers(len(WORDS)))]
        ink = shade(g.rgba, 0.45)
        wrap.text(ink, word, 0.0, z0 + (z1 - z0) * 0.5, 0.011, layer=2)
        wrap.band(ink, z0 + 0.003, z0 + 0.0075, layer=1)
        wrap.band(ink, z1 - 0.0075, z1 - 0.003, layer=1)
        for rgba_, mesh in wrap.patches:
            g.decor(mesh, rgba_, "paint")
    elif style == "wine":
        wrap = labels.Wrap(R, 0.0)
        wrap.rect([0.93, 0.9, 0.78, 1.0], -0.028, h * 0.2, 0.028, h * 0.4)
        wrap.text([0.3, 0.12, 0.1, 1.0], "RESERVA", 0.0, h * 0.31, 0.0075)
        wrap.text([0.3, 0.12, 0.1, 1.0], "2019", 0.0, h * 0.24, 0.007)
        for rgba_, mesh in wrap.patches:
            g.decor(mesh, rgba_, "paint")
    g.grasp(
        "body",
        (0.0, 0.0, h * 0.38),
        (0.0, 0.0),
        d,
        1.0,
        "round the body, any jaw direction",
        z_body + 0.03,
        body_h - 0.012,
    )
    g.grasp(
        "neck",
        (0.0, 0.0, neck_z + 0.012),
        (0.0, 0.0),
        2 * R * neck_r,
        0.55,
        "round the neck",
        neck_z + 0.006,
        neck_top - 0.004,
    )
    g.anchor("mouth", (0.0, 0.0, h))
    g.anchor("base", (0.0, 0.0, 0.0))
    g.fact(kind="bottle", radius=R, closed=True, body_h=body_h, neck_r=R * neck_r, filled=round(fill, 2))


@generator(
    "can",
    size="[diameter, height]",
    default_size=(0.066, 0.115),
    color="#c0392b",
    material="aluminum",
    kind="container",
    summary="a drinks can, soup tin or tuna tin with a printed label (take it round the body)",
    keywords=("can", "tin can", "soda can", "soup can", "tuna can", "beverage can"),
    params={
        "style": Choice("soda", "soup", "tuna", doc="soda can with a tab, soup tin with a paper label, squat tuna tin"),
        "fill": Range(0.4, 1.0, 0.9, "how full"),
    },
    palette=("#c0392b", "#2980b9", "#27ae60", "#f39c12", "#8e44ad", "#16a085"),
    hints={
        "tuna": {"style": "tuna", "_size": (0.085, 0.034)},
        "soup": {"style": "soup", "_size": (0.066, 0.102)},
        "cola": {"style": "soda"},
    },
    builtin=True,
)
def can(g: Gen) -> None:
    d, h = g.size
    R, style = d / 2, g.p("style")
    lip = 0.0035
    # the aluminium ends and the printed middle are separate looks over one cylinder to collide with
    body = M.lathe([(0, 0), (R * 0.82, 0), (R * 0.9, 0.003), (R * 0.92, lip), (R * 0.92, lip + 0.004), (R, lip + 0.006),
                    (R, h - lip - 0.006), (R * 0.92, h - lip - 0.004), (R * 0.92, h - lip), (R * 0.9, h - 0.003),
                    (R * 0.82, h), (0, h - 0.002)], 44, sharp=(1, 4, 5, 8))  # fmt: skip
    metal = [0.78, 0.8, 0.82, 1.0]
    area = 2 * math.pi * R * h + 2 * math.pi * R * R
    shell_mass = area * 0.0004 * 2700.0
    g.part(body, "aluminum", [kit.cylinder(R, h / 2, (0, 0, h / 2))], metal, mass=shell_mass)
    fill = g.p("fill")
    g.contents(
        "water", math.pi * (R - 0.0005) ** 2 * (h - 2 * lip) * fill, (0, 0, h / 2), R - 0.0005, (h - 2 * lip) / 2
    )
    z0, z1 = lip + 0.006, h - lip - 0.006
    wrap = labels.Wrap(R, 0.0)
    wrap.band(g.rgba, z0, z1)
    word = WORDS[int(g.rng("word").integers(len(WORDS)))]
    white, dark = [0.97, 0.97, 0.95, 1.0], shade(g.rgba, 0.4)
    if style == "soda":
        wrap.band(white, z0 + (z1 - z0) * 0.30, z0 + (z1 - z0) * 0.34, layer=1)
        wrap.text(white, word, 0.0, z0 + (z1 - z0) * 0.55, 0.012)
        wrap.disc(white, 0.0, z0 + (z1 - z0) * 0.20, 0.007)
    elif style == "soup":
        wrap.band(white, z0 + (z1 - z0) * 0.22, z0 + (z1 - z0) * 0.78, layer=1)
        wrap.text(dark, word, 0.0, z0 + (z1 - z0) * 0.5, 0.011, layer=2)
        wrap.band(dark, z0 + (z1 - z0) * 0.22, z0 + (z1 - z0) * 0.26, layer=2)
        wrap.band(dark, z0 + (z1 - z0) * 0.74, z0 + (z1 - z0) * 0.78, layer=2)
    else:
        wrap.text(white, word, 0.0, z0 + (z1 - z0) * 0.4, 0.008)
    for rgba_, mesh in wrap.patches:
        g.decor(mesh, rgba_, "paint")
    if style == "soda":  # a pull tab and a recessed lid
        g.decor(M.cylinder(R * 0.72, 0.001, 32, z0=h - 0.0006), shade(metal, 0.9), "aluminum")
        g.decor(
            M.rounded_box(0.0075, 0.0045, 0.0007, 0.0006).moved((0.004, 0.0, h + 0.0011)),
            shade(metal, 1.05),
            "aluminum",
        )
    g.grasp("body", (0.0, 0.0, h * 0.5), (0.0, 0.0), d, 1.0, "round the body, any jaw direction", h * 0.3, h - 0.012)
    g.anchor("mouth", (0.0, 0.0, h))
    g.fact(kind="can", radius=R, closed=True)
