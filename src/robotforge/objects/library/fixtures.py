"""Fixtures for the harder tasks: a bottle rack with round sockets and a labelled sorting tray.

They are generators like the objects, fixed by default (add_object puts them in place and nothing knocks them over),
and they have `slots`: named places a thing is put, in the fixture's own frame, which tasks and the arm planner read
(`pick_and_place(robot, bottle, "rack/2")` puts the bottle in the second socket).
"""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, labels
from robotforge.objects.kit import Choice, Gen, Range, shade
from robotforge.objects.registry import generator

TAGS = ("none", "fruit", "cans", "tools", "bottles", "books", "boxes", "mugs")


def _rect_ray(a: float, hx: float, hy: float) -> tuple[float, float]:
    """Where the ray from the middle at angle `a` meets a rectangle of half extents (hx, hy)."""
    c, s = math.cos(a), math.sin(a)
    t = min(hx / abs(c) if abs(c) > 1e-9 else math.inf, hy / abs(s) if abs(s) > 1e-9 else math.inf)
    return t * c, t * s


def _plate_with_hole(cx: float, hx: float, hy: float, r: float, z: float, n: int = 32) -> M.Mesh:
    """A flat rectangle (middle at x = cx; half extents hx, hy) at height z with a round hole of radius r, facing up."""
    angles = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    inner = np.array([[cx + r * math.cos(a), r * math.sin(a), z] for a in angles])
    outer = np.array([[cx + _rect_ray(a, hx, hy)[0], _rect_ray(a, hx, hy)[1], z] for a in angles])
    k = np.arange(n)
    faces = np.concatenate(
        [np.stack([k, (k + 1) % n, n + (k + 1) % n], axis=1), np.stack([k, n + (k + 1) % n, n + k], axis=1)]
    ).astype(np.int32)
    verts = np.vstack([outer, inner])
    return M.Mesh(verts, np.tile([0.0, 0.0, 1.0], (2 * n, 1)), faces, np.arange(2 * n, dtype=np.int64))


@generator(
    "rack",
    size="[length, width, height]",
    default_size=(0.30, 0.11, 0.07),
    color="#b5835a",
    material="wood",
    kind="fixture",
    fixed=True,
    summary="a bottle rack: a wooden block with round sockets to stand bottles in (never moves); "
    'pick_and_place(robot, bottle, "rack") fills the first free socket, "rack/2" the second',
    keywords=("rack", "bottle rack", "holder"),
    params={
        "slots": Range(2.0, 4.0, 3.0, "how many sockets"),
        "bottle": Range(0.03, 0.09, 0.066, "diameter of the bottle it takes, m"),
        "clearance": Range(0.006, 0.014, 0.010, "how much wider a socket is than the bottle, m"),
        "depth": Range(0.03, 0.06, 0.045, "how deep a socket is, m"),
    },
    builtin=True,
)
def rack(g: Gen) -> None:
    length, width, height = g.size
    n = int(round(g.p("slots")))
    r = (g.p("bottle") + g.p("clearance")) / 2
    depth = min(g.p("depth"), height - 0.012)
    cell = length / n
    r = min(r, (min(cell, width) - 0.012) / 2)  # sockets never wider than their cell: more of them, smaller ones
    floor = height - depth
    hx, hy = length / 2, width / 2
    # how it looks: the block, a top with round holes, each socket's wall and floor
    rings = [M.at_z(M.rounded_rect(hx, hy, 0.004, 3), z) for z in (0.0, height)]
    shell = M.loft(rings, cap_start=True, cap_end=False)  # open at the top: the plate over it has the holes
    top = M.merge([_plate_with_hole(-hx + cell * (i + 0.5), cell / 2, hy, r, height + 0.0004) for i in range(n)])
    inside = []
    pieces = [kit.box(hx, hy, floor / 2, (0, 0, floor / 2))]
    outer = min(cell, width) / 2
    for i in range(n):
        cx = -hx + cell * (i + 0.5)
        wall = [(r - 0.0005, floor), (r + 0.002, floor), (r + 0.002, height + 0.0004), (r - 0.0005, height + 0.0004)]
        inside.append(M.revolve(wall, 32, sharp=()).moved((cx, 0, 0)))
        inside.append(M.cylinder(r, 0.001, 32, z0=floor).moved((cx, 0, 0)))
        pieces += [
            p.moved((cx, 0, 0)) for p in kit.ring_of_boxes(outer, outer, floor, height, outer - r, 8, math.pi / 8)
        ]
        g.slot(f"{i + 1}", (cx, 0.0, floor), "socket", radius=r, rim=height, depth=depth)
    g.part(M.merge([shell, top]), "wood", pieces, g.rgba)
    g.decor(M.merge(inside), shade(g.rgba, 0.55), "wood")
    g.fact(kind="rack", sockets=n, socket_r=r, floor_z=floor, rim_z=height)


@generator(
    "tray",
    size="[length, width, height]",
    default_size=(0.20, 0.15, 0.045),
    color="#3b82c4",
    material="plastic",
    kind="fixture",
    fixed=True,
    summary="an open sorting tray with a name printed on its front (fruit, cans, tools, ...), fixed in place; things "
    "are put in it with pick_and_place(robot, thing, tray)",
    keywords=("tray", "sorting tray", "tote"),
    params={"tag": Choice(*TAGS, doc="the name printed on it", vary=False)},
    builtin=True,
)
def tray(g: Gen) -> None:
    length, width, height = g.size
    hx, hy, t, floor = length / 2, width / 2, 0.006, 0.006
    rgba = g.rgba
    parts = [
        (
            M.rounded_box(hx, hy, floor / 2, 0.002, 1, 0.004).moved((0, 0, floor / 2)),
            kit.box(hx, hy, floor / 2, (0, 0, floor / 2)),
        ),
    ]
    for sx in (1, -1):
        wall = M.rounded_box(t / 2, hy, height / 2, 0.002, 1, 0.002).moved((sx * (hx - t / 2), 0, height / 2))
        parts.append((wall, kit.box(t / 2, hy, height / 2, (sx * (hx - t / 2), 0, height / 2))))
    for sy in (1, -1):
        wall = M.rounded_box(hx - t, t / 2, height / 2, 0.002, 1, 0.002).moved((0, sy * (hy - t / 2), height / 2))
        parts.append((wall, kit.box(hx - t, t / 2, height / 2, (0, sy * (hy - t / 2), height / 2))))
    g.part(M.merge([m for m, _ in parts]), "plastic", [p for _, p in parts], rgba)
    if g.p("tag") != "none":
        front = labels.Sheet((0, -hy, 0), (1, 0, 0), (0, 0, 1))
        word = g.p("tag").upper()
        letter = min(height * 0.5, 0.02)
        plate = labels.text_width(word, letter) + 0.012
        front.rect(
            [0.96, 0.96, 0.94, 1.0],
            -plate / 2,
            height / 2 - letter / 2 - 0.004,
            plate / 2,
            height / 2 + letter / 2 + 0.004,
        )
        front.text(shade(rgba, 0.35), word, 0.0, height / 2 - letter / 2, letter, "center")
        for rgba_, mesh in front.patches:
            g.decor(mesh, rgba_, "paint")
    g.slot("inside", (0.0, 0.0, floor), "tray", half=(hx - t, hy - t), rim=height, depth=height - floor)
    g.fact(kind="tray", inner=(hx - t, hy - t), floor_z=floor, rim_z=height, tag=g.p("tag"))
