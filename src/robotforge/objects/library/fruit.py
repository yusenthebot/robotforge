"""Fruit: round or curved things with a skin, a stem and a small flat spot to rest on."""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import kit, profiles
from robotforge.objects.kit import Flag, Gen, Range, mix
from robotforge.objects.registry import generator

BROWN = [0.32, 0.22, 0.12, 1.0]
LEAF = [0.25, 0.5, 0.2, 1.0]


def _scaled(points: list[tuple[float, float]], d: float, height: float) -> list[tuple[float, float]]:
    return [(r * d, z * d * height) for r, z in points]


def _stem(z: float, d: float, lean: float, length: float = 0.016) -> M.Mesh:
    path = np.array([[0.0, 0.0, z - 0.004], [lean * 0.4, 0.0, z + length * 0.5], [lean, 0.0, z + length]])
    return M.tube(path, [0.0022 * d / 0.075, 0.0018 * d / 0.075, 0.0016 * d / 0.075], segments=8)


@generator(
    "apple",
    size="[diameter]",
    default_size=(0.075,),
    color="red",
    material="fruit",
    kind="food",
    summary="an apple with a stem (round: take it in a pinch round the middle, any jaw direction)",
    keywords=("apple",),
    params={
        "squat": Range(0.82, 0.98, 0.9, "height as a share of the diameter"),
        "leaf": Flag(False, "a leaf on the stem"),
    },
    palette=("red", "#8db600", "#e8c53a"),
    builtin=True,
)
def apple(g: Gen) -> None:
    d = g.size[0]
    squat = g.p("squat")
    shape = [(0.0, 0.025), (0.17, 0.0), (0.36, 0.06), (0.47, 0.3), (0.5, 0.52), (0.465, 0.76), (0.34, 0.92),
             (0.15, 0.95), (0.07, 0.88), (0.0, 0.86)]  # fmt: skip
    profile = profiles.smooth(_scaled(shape, d, squat), 3)
    mesh = M.lathe(profile, 36, sharp=(1,))
    convex = [(0.0, 0.0), (0.17 * d, 0.0), (0.38 * d, 0.1 * d * squat), (0.5 * d, 0.5 * d * squat),
              (0.44 * d, 0.8 * d * squat), (0.25 * d, 0.93 * d * squat), (0.0, 0.93 * d * squat)]  # fmt: skip
    g.part(
        mesh,
        "fruit",
        [kit.hull(M.lathe(convex, 24, sharp=(1,)))],
        mix(g.rgba, [0.85, 0.75, 0.2], 0.12 * g.rng("tone").random()),
    )
    lean = 0.004 * (g.rng("lean").random() - 0.5) * 2
    g.decor(_stem(0.8 * d * squat, d, lean), BROWN, "bark")
    if g.p("leaf"):
        leaf = M.blob((0.012, 0.005, 0.0008), 1, amplitude=0.05, rings=6, segments=10).moved(
            (lean + 0.012, 0.0, 0.95 * d * squat)
        )
        g.decor(leaf, LEAF, "foliage")
    h = 0.93 * d * squat
    g.grasp("body", (0.0, 0.0, h * 0.5), (0.0, 0.0), d + 0.004, 1.0, "round the middle, any jaw direction")
    g.anchor("top", (0.0, 0.0, h))
    g.fact(kind="fruit", radius=d / 2)


@generator(
    "orange",
    size="[diameter]",
    default_size=(0.078,),
    color="orange",
    material="fruit",
    kind="food",
    summary="an orange with a dimpled skin (round, any jaw direction)",
    keywords=("orange", "tangerine", "mandarin", "lemon", "lime", "peach"),
    params={"bumps": Range(0.0006, 0.0016, 0.001, "how dimpled the skin is, m")},
    builtin=True,
)
def orange(g: Gen) -> None:
    d = g.size[0]
    r = d / 2
    flat = 0.07 * r
    skin = M.noise_displaced(
        M.sphere(r, 14, 28).moved((0, 0, r - flat * 0.5)), int(g.rng("bumps").integers(1 << 30)), g.p("bumps"), 0.006
    )
    cut = math.acos((r - flat) / r)
    arc = profiles.arc((0.0, r - flat), r, -math.pi / 2 + (math.pi / 2 - cut), math.pi / 2, 12)
    shape = [(0.0, 0.0), (max(r * math.sin(cut), 0.001), 0.0), *arc[1:-1], (0.0, 2 * r - flat)]
    g.part(skin, "fruit", [kit.hull(M.lathe(shape, 24, sharp=(1,)))], g.rgba)
    g.decor(M.cylinder(0.0045, 0.0025, 14, z0=2 * r - flat - 0.0012), LEAF, "foliage")
    h = 2 * r - flat
    g.grasp("body", (0.0, 0.0, h * 0.5), (0.0, 0.0), d + 0.004, 1.0, "round the middle, any jaw direction")
    g.anchor("top", (0.0, 0.0, h))
    g.fact(kind="fruit", radius=r)


@generator(
    "pear",
    size="[diameter]",
    default_size=(0.068,),
    color="#b5c23a",
    material="fruit",
    kind="food",
    summary="a pear: wide at the bottom, narrow at the top, with a stem (take it round the wide part)",
    keywords=("pear",),
    params={"length": Range(1.35, 1.7, 1.5, "height as a multiple of the diameter")},
    palette=("#b5c23a", "#d4b83a"),
    builtin=True,
)
def pear(g: Gen) -> None:
    d = g.size[0]
    k = g.p("length")
    shape = [(0.0, 0.02), (0.16, 0.0), (0.36, 0.06), (0.5, 0.22), (0.5, 0.36), (0.36, 0.58), (0.22, 0.78), (0.17, 0.93),
             (0.12, 1.0), (0.0, 1.0)]  # fmt: skip
    mesh = M.lathe(profiles.smooth(_scaled(shape, d, k), 3), 36, sharp=(1,))
    convex = [(0.0, 0.0), (0.16 * d, 0.0), (0.4 * d, 0.07 * d * k), (0.5 * d, 0.28 * d * k), (0.3 * d, 0.7 * d * k),
              (0.14 * d, 0.98 * d * k), (0.0, 1.0 * d * k)]  # fmt: skip
    g.part(mesh, "fruit", [kit.hull(M.lathe(convex, 24, sharp=(1,)))], g.rgba)
    g.decor(_stem(d * k * 0.99, d, 0.006), BROWN, "bark")
    g.grasp("body", (0.0, 0.0, d * k * 0.22), (0.0, 0.0), d + 0.004, 1.0, "round the wide part, any jaw direction")
    g.anchor("top", (0.0, 0.0, d * k))
    g.fact(kind="fruit", radius=d / 2)


@generator(
    "banana",
    size="[length]",
    default_size=(0.19,),
    color="#f0cf2c",
    material="fruit",
    kind="food",
    summary="a banana lying on its side, curved in the plane of the table (take it across the middle)",
    keywords=("banana", "plantain"),
    params={"curve": Range(0.9, 1.5, 1.2, "how bent: radians of arc"), "ripe": Range(0.0, 1.0, 0.2, "brown spots")},
    builtin=True,
)
def banana(g: Gen) -> None:
    length = g.size[0]
    arc = g.p("curve")
    radius = length / arc
    r_body = 0.0175
    a = np.linspace(-arc / 2, arc / 2, 17)
    path = np.column_stack(
        [radius * np.sin(a), radius * (1 - np.cos(a)) - radius * (1 - math.cos(arc / 2)) / 2, np.full(len(a), r_body)]
    )
    taper = np.interp(np.linspace(0, 1, len(a)), [0, 0.08, 0.2, 0.8, 0.95, 1], [0.35, 0.6, 1.0, 1.0, 0.75, 0.35])
    mesh = M.tube(path, list(r_body * taper), segments=5)
    solid = [kit.capsule_between(tuple(path[i]), tuple(path[i + 2]), r_body * 0.95) for i in range(2, len(a) - 3, 2)]
    g.part(mesh, "fruit", solid, g.rgba)
    g.decor(M.sphere(0.006, 6, 10).moved(tuple(path[-1] + [0.002, 0.0, 0.0])), [0.12, 0.09, 0.05, 1.0], "bark")
    g.decor(M.tube(path[:3], [0.006, 0.005, 0.004], segments=6), [0.4, 0.4, 0.2, 1.0], "bark")
    rng = g.rng("spots")
    if g.p("ripe") > 0.4:
        for _ in range(int(10 * g.p("ripe"))):
            k = int(rng.integers(3, len(a) - 3))
            g.decor(
                M.sphere(0.0022, 5, 8).moved((path[k][0], path[k][1] + rng.uniform(-0.008, 0.008), r_body + 0.013)),
                [0.3, 0.2, 0.08, 1.0],
                "bark",
            )
    mid = path[len(a) // 2]
    tangent = path[len(a) // 2 + 1] - path[len(a) // 2 - 1]
    jaw = (-tangent[1], tangent[0])
    g.grasp(
        "body",
        (float(mid[0]), float(mid[1]), r_body),
        jaw,
        2 * r_body + 0.006,
        1.0,
        "across the middle, jaws across its width",
    )
    g.fact(kind="fruit", radius=r_body)
