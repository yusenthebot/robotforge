"""Furniture and architecture for scenes: walls, shelves, counters, pallets, trees and more.

A prop is one body at its base point (on the ground, at the center of its footprint),
turned by yaw. Fixed props are static bodies; movable ones get a free joint. Each
prop is a handful of boxes, cylinders and spheres, so it collides exactly as drawn
and costs little to simulate. Sizes are in meters; `length` runs along the prop's
own x axis and `depth` along y, with shelves and benches open (or facing) toward -y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.design.visuals import props as looks

G = mujoco.mjtGeom
PANEL = 0.02  # m, board and panel thickness
WOOD = [0.72, 0.58, 0.4, 1.0]
DARK = [0.22, 0.23, 0.25, 1.0]
STONE_TOP = [0.86, 0.85, 0.82, 1.0]
BARK = [0.4, 0.29, 0.2, 1.0]
MATTE = "prop_matte"
MARK = 0.004  # m: markers are this thick, enough to never flicker against the floor


@dataclass(frozen=True)
class ShapeInfo:
    """What a shape's size list means, and its defaults when a field is left out."""

    size_hint: str
    size_lengths: tuple[int, ...]
    size: tuple[float, ...]
    color: str | tuple[float, ...] = "red"
    fixed: bool = False
    pickable: bool = False  # small enough for an arm's pick_and_place
    summary: str = ""


SHAPES: dict[str, ShapeInfo] = {
    "box": ShapeInfo("[length, width, height]", (3,), (0.05, 0.05, 0.05), pickable=True, summary="a block"),
    "sphere": ShapeInfo("[diameter]", (1,), (0.05,), pickable=True, summary="a ball"),
    "cylinder": ShapeInfo("[diameter, height]", (2,), (0.05, 0.08), pickable=True, summary="a can or cup"),
    "table": ShapeInfo("[length, width, height]", (3,), (0.8, 0.5, 0.4), summary="a table (never moves)"),
    "bin": ShapeInfo("[length, width, height]", (3,), (0.2, 0.15, 0.08), summary="an open-top container"),
    "wall": ShapeInfo(
        "[length, thickness, height]", (3,), (2.0, 0.1, 1.0), (0.86, 0.85, 0.82, 1), True, summary="a wall"
    ),
    "pillar": ShapeInfo("[diameter, height]", (2,), (0.3, 2.0), (0.62, 0.62, 0.6, 1), True, summary="a column"),
    "shelf": ShapeInfo(
        "[length, depth, height]",
        (3,),
        (1.0, 0.4, 1.6),
        tuple(WOOD),
        True,
        summary="open shelving, a board about every 40 cm, open toward -y",
    ),
    "door_frame": ShapeInfo(
        "[width, depth, height]",
        (3,),
        (1.0, 0.12, 2.0),
        tuple(WOOD),
        True,
        summary="two posts and a lintel; the opening is 16 cm narrower than the width",
    ),
    "counter": ShapeInfo(
        "[length, depth, height]",
        (3,),
        (1.6, 0.6, 0.9),
        (0.35, 0.42, 0.5, 1),
        True,
        summary="a kitchen counter: cabinet with a stone top (color is the cabinet)",
    ),
    "pallet": ShapeInfo(
        "[length, width, height]", (3,), (1.2, 0.8, 0.144), tuple(WOOD), False, summary="a wooden pallet (20 kg)"
    ),
    "tree": ShapeInfo(
        "[crown diameter, height]", (2,), (1.6, 3.0), (0.27, 0.55, 0.25, 1), True, summary="a tree (color is the crown)"
    ),
    "bench": ShapeInfo(
        "[length, depth, seat height]", (3,), (1.4, 0.45, 0.45), tuple(WOOD), True, summary="a park bench facing -y"
    ),
    "rock": ShapeInfo(
        "[diameter, height]", (2,), (0.6, 0.35), (0.5, 0.49, 0.47, 1), True, summary="a boulder (half buried)"
    ),
    "marker": ShapeInfo(
        "[diameter] or [length, width]",
        (1, 2),
        (0.4,),
        (0.3, 0.8, 0.45, 1),
        True,
        summary="a flat mark on the ground (goal, lane, path); nothing collides with it",
    ),
    "buoy": ShapeInfo(
        "[diameter, height]",
        (2,),
        (0.45, 1.0),
        "red",
        False,
        summary="a floating buoy, moored to the bottom where it is put (in water it floats upright and bobs)",
    ),
    "dock": ShapeInfo(
        "[length, width, deck height]",
        (3,),
        (4.0, 1.2, 1.2),
        tuple(WOOD),
        True,
        summary="a wooden dock: a deck on posts (give a height above the water, the posts reach the bottom)",
    ),
    # Mechanisms and soft things (mechanisms.py): a fixed carcass with one moving part, which operate works.
    "drawer": ShapeInfo(
        "[length, depth, height]",
        (3,),
        (0.3, 0.3, 0.14),
        (0.62, 0.48, 0.34, 1),
        True,
        summary="a chest with a drawer that pulls out toward -y (operate opens it; pick_and_place puts things in it)",
    ),
    "cabinet": ShapeInfo(
        "[length, depth, height]",
        (3,),
        (0.3, 0.24, 0.22),
        (0.86, 0.85, 0.82, 1),
        True,
        summary="a cabinet with a shelf and a door hinged on its left that opens toward -y (operate)",
    ),
    "button": ShapeInfo(
        "[diameter, height]", (2,), (0.04, 0.05), "red", True, summary="a push button that springs back up (operate)"
    ),
    "switch": ShapeInfo(
        "[length, width, height]", (3,), (0.06, 0.04, 0.03), "red", True, summary="a toggle switch, off at the start"
    ),
    "lever": ShapeInfo("[length]", (1,), (0.15,), "red", True, summary="a lever on a box that tips toward -y or +y"),
    "valve": ShapeInfo(
        "[wheel diameter, height]",
        (2,),
        (0.16, 0.22),
        "red",
        True,
        summary="a pipe valve with a handwheel on top, closed at the start (operate opens it: two turns)",
    ),
    "conveyor": ShapeInfo(
        "[length, width, height]",
        (3,),
        (1.2, 0.3, 0.75),
        (0.36, 0.45, 0.56, 1),
        True,
        summary="a conveyor belt running along +x at `speed` m/s (default 0.1); things on it ride along",
    ),
    "rope": ShapeInfo(
        "[length, diameter]",
        (2,),
        (0.6, 0.012),
        (0.85, 0.66, 0.32, 1),
        True,
        summary="a rope: fixed, it hangs from pos (and swings); fixed=false, it lies loose on the ground",
    ),
    "cloth": ShapeInfo(
        "[length, width]",
        (2,),
        (0.3, 0.3),
        (0.3, 0.6, 0.65, 1),
        False,
        summary="a cloth that drapes over what is under it; fixed=true pins its +y edge like a curtain",
    ),
}
PROPS = (
    "wall",
    "pillar",
    "shelf",
    "door_frame",
    "counter",
    "pallet",
    "tree",
    "bench",
    "rock",
    "marker",
    "buoy",
    "dock",
)
DEFAULT_MASS = {"pallet": 20.0}
WATER = 1000.0  # kg/m^3


@dataclass
class Part:
    type: int
    size: list[float]
    pos: list[float]
    rgba: list[float]
    collide: bool = True
    quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    weight: float | None = None  # share of the prop's mass (a buoy's ballast); None: by volume among the rest


def footprint(shape: str, size: list[float]) -> tuple[float, float]:
    """Half extents (x, y) of a shape's footprint in its own frame."""
    if shape == "tree":  # the trunk stands on the ground, the crown can hang over a slope
        return size[0] * 0.06, size[0] * 0.06
    if shape in ("button", "valve"):
        return size[0] / 2 + 0.02, size[0] / 2 + 0.02
    if shape in ("lever", "rope"):
        return 0.04, 0.04
    if shape in ("sphere", "pillar", "rock") or (shape == "marker" and len(size) == 1):
        return size[0] / 2, size[0] / 2
    if shape == "cylinder":
        return size[0] / 2, size[0] / 2
    return size[0] / 2, size[1] / 2


def shelf_levels(size: list[float]) -> list[float]:
    """Heights of a shelf's board tops above its base, bottom board first."""
    height = size[2]
    n = max(2, round(height / 0.4) + 1)
    return [float(PANEL / 2 + i * (height - PANEL) / (n - 1) + PANEL / 2) for i in range(n)]


def build(
    spec: mujoco.MjSpec,
    name: str,
    shape: str,
    size: list[float],
    rgba: list[float],
    base: list[float],
    quat: list[float],
    fixed: bool,
    mass: float | None,
) -> None:
    """Add a prop as one body at `base`, turned by `quat`."""
    parts = _PARTS[shape](size, rgba)
    # Look the name up in the list: after spec.attach, MuJoCo 3.15's spec.material() misses materials added later.
    if not any(m.name == MATTE for m in spec.materials):
        # Wood, plaster and leaves are not glossy; MuJoCo's default specular makes them look like plastic.
        spec.add_material(name=MATTE, specular=0.08, shininess=0.1, reflectance=0.0)
    body = spec.worldbody.add_body(name=name, pos=base, quat=quat)
    movable = not fixed and shape != "marker"
    if movable:
        body.add_freejoint(name=name)
    total = mass if mass is not None else _default_mass(shape, size)
    volumes = [_volume(p) if p.weight is None else 0.0 for p in parts]
    given = sum(p.weight for p in parts if p.weight is not None)
    geoms = []
    for i, part in enumerate(parts):
        kwargs = {}
        if movable and not given:  # the old expression to the last bit, so old worlds replay exactly
            kwargs["mass"] = total * volumes[i] / sum(volumes)
        elif movable:
            share = part.weight if part.weight is not None else (1 - given) * volumes[i] / max(sum(volumes), 1e-12)
            kwargs["mass"] = total * share
        if not part.collide:
            kwargs.update(contype=0, conaffinity=0)
        geom = body.add_geom(
            name=name if i == 0 else f"{name}/{i}",
            type=part.type,
            size=part.size,
            pos=part.pos,
            quat=part.quat,
            rgba=part.rgba,
            material=MATTE,
            **kwargs,
        )
        geoms.append(geom)
    # Meshes over the primitives (visual only); the primitives stay for contacts and mass.
    looks.dress(spec, body, shape, size, rgba, name, geoms)


def _default_mass(shape: str, size: list[float]) -> float:
    if shape == "buoy":  # rides with about half its float under water
        d, h = size
        return 0.55 * WATER * math.pi * (d / 2) ** 2 * BUOY_FLOAT * h
    return DEFAULT_MASS.get(shape, 5.0)


def moor(spec: mujoco.MjSpec, name: str, anchor: list[float], rope: float) -> None:
    """Tie a buoy to the bottom: a rope from `anchor` (a point on the bottom) to the buoy's keel, `rope` m long,
    slack until it is pulled taut, so the buoy bobs and leans but stays where it was put."""
    spec.worldbody.add_site(name=f"{name}/anchor", pos=anchor, size=[0.03, 0, 0], group=3)
    body = next(b for b in spec.worldbody.bodies if b.name == name)
    body.add_site(name=f"{name}/keel", pos=[0, 0, 0], size=[0.02, 0, 0], group=3)
    tendon = spec.add_tendon(name=f"{name}/rope", limited=True, range=[0.0, rope], rgba=[0.15, 0.15, 0.15, 1])
    tendon.width = 0.006
    tendon.wrap_site(f"{name}/anchor")
    tendon.wrap_site(f"{name}/keel")


def _volume(p: Part) -> float:
    if p.type == G.mjGEOM_BOX:
        return 8 * float(np.prod(p.size))
    if p.type == G.mjGEOM_CYLINDER:
        return 2 * np.pi * p.size[0] ** 2 * p.size[1]
    if p.type == G.mjGEOM_ELLIPSOID:
        return 4 / 3 * np.pi * float(np.prod(p.size))
    return 4 / 3 * np.pi * p.size[0] ** 3


def _box(half: list[float], pos: list[float], rgba: list[float], collide: bool = True) -> Part:
    return Part(G.mjGEOM_BOX, half, pos, rgba, collide)


def _wall(size: list[float], rgba: list[float]) -> list[Part]:
    length, thick, height = size
    return [_box([length / 2, thick / 2, height / 2], [0, 0, height / 2], rgba)]


def _pillar(size: list[float], rgba: list[float]) -> list[Part]:
    d, h = size
    return [Part(G.mjGEOM_CYLINDER, [d / 2, h / 2, 0], [0, 0, h / 2], rgba)]


def _shelf(size: list[float], rgba: list[float]) -> list[Part]:
    length, depth, height = size
    t = PANEL
    parts = [
        _box([t / 2, depth / 2, height / 2], [length / 2 - t / 2, 0, height / 2], rgba),
        _box([t / 2, depth / 2, height / 2], [-length / 2 + t / 2, 0, height / 2], rgba),
        _box([length / 2 - t, t / 2, height / 2], [0, depth / 2 - t / 2, height / 2], _shade(rgba, 0.85)),
    ]
    for top in shelf_levels(size):
        parts.append(_box([length / 2 - t, depth / 2 - t / 2, t / 2], [0, -t / 2, top - t / 2], rgba))
    return parts


def _door_frame(size: list[float], rgba: list[float]) -> list[Part]:
    width, depth, height = size
    post = min(0.08, width / 4)
    return [
        _box([width / 2, depth / 2, post / 2], [0, 0, height - post / 2], rgba),
        _box([post / 2, depth / 2, (height - post) / 2], [width / 2 - post / 2, 0, (height - post) / 2], rgba),
        _box([post / 2, depth / 2, (height - post) / 2], [-width / 2 + post / 2, 0, (height - post) / 2], rgba),
    ]


def _counter(size: list[float], rgba: list[float]) -> list[Part]:
    length, depth, height = size
    top, inset = 0.04, 0.03
    body_h = height - top
    return [
        _box([length / 2, depth / 2, top / 2], [0, 0, height - top / 2], STONE_TOP),
        _box([length / 2 - 0.01, depth / 2 - inset / 2, body_h / 2], [0, inset / 2, body_h / 2], rgba),
        # cabinet door seams and a dark toe kick read as a counter, not a block
        _box([0.004, 0.002, body_h / 2 - 0.08], [0, -depth / 2 + inset - 0.002, body_h / 2 + 0.04], DARK, False),
        _box([length / 2 - 0.02, 0.003, 0.04], [0, -depth / 2 + inset - 0.002, 0.04], DARK, False),
    ]


def _pallet(size: list[float], rgba: list[float]) -> list[Part]:
    length, width, height = size
    deck = 0.022
    boards = 5
    board_w = min(0.1, width / (boards * 1.3))
    parts = []
    for i in range(boards):
        y = -width / 2 + board_w / 2 + i * (width - board_w) / (boards - 1)
        parts.append(_box([length / 2, board_w / 2, deck / 2], [0, y, height - deck / 2], rgba))
    runner = min(0.1, width / 6)
    for y in (-width / 2 + runner / 2, 0.0, width / 2 - runner / 2):
        parts.append(
            _box([length / 2, runner / 2, (height - deck) / 2], [0, y, (height - deck) / 2], _shade(rgba, 0.8))
        )
    return parts


# A crown is a cluster of balls (offset x, y, z and radius, as fractions of the crown radius, and a shade),
# which reads as foliage where one big ball reads as a balloon.
CROWN = (
    (0.0, 0.0, 0.05, 0.62, 1.0),
    (0.42, 0.12, -0.12, 0.5, 0.9),
    (-0.4, 0.22, -0.08, 0.5, 0.84),
    (0.02, -0.44, -0.1, 0.5, 0.93),
    (-0.12, 0.02, 0.4, 0.46, 1.06),
    (0.3, -0.25, 0.25, 0.4, 0.97),
    (-0.3, -0.25, 0.2, 0.42, 0.88),
)


def _tree(size: list[float], rgba: list[float]) -> list[Part]:
    crown, height = size
    r = crown / 2
    trunk_r = max(0.04, crown * 0.06)
    center = height - 0.55 * r
    trunk_h = max(0.3, center - 0.2 * r)
    parts = [Part(G.mjGEOM_CYLINDER, [trunk_r, trunk_h / 2, 0], [0, 0, trunk_h / 2], BARK)]
    for dx, dy, dz, k, shade in CROWN:
        parts.append(Part(G.mjGEOM_SPHERE, [r * k, 0, 0], [dx * r, dy * r, center + dz * r], _shade(rgba, shade)))
    return parts


def _bench(size: list[float], rgba: list[float]) -> list[Part]:
    length, depth, seat = size
    slab = 0.03
    leg_x = length / 2 - min(0.15, length / 6)
    parts = [_box([length / 2, depth / 2, slab / 2], [0, 0, seat - slab / 2], rgba)]
    for sx in (1, -1):
        parts.append(_box([0.025, depth / 2 - 0.03, (seat - slab) / 2], [sx * leg_x, 0, (seat - slab) / 2], DARK))
        parts.append(_box([0.025, 0.02, 0.2], [sx * leg_x, depth / 2 - 0.02, seat + 0.2], DARK))
    parts.append(_box([length / 2, 0.015, 0.09], [0, depth / 2 - 0.02, seat + 0.27], rgba))
    return parts


# A boulder is three overlapping squashed balls: offset x, y (fractions of the radius), width and
# height factors, and a shade.
ROCK_LUMPS = ((0.0, 0.0, 1.0, 1.0, 1.0), (0.3, 0.2, 0.7, 0.75, 0.92), (-0.25, -0.22, 0.65, 0.7, 0.86))


def _rock(size: list[float], rgba: list[float]) -> list[Part]:
    """Three overlapping squashed balls, sunk a little into the ground like a boulder."""
    d, h = size
    r = d / 2
    parts = []
    for dx, dy, kxy, kz, shade in ROCK_LUMPS:
        half = [r * kxy, r * kxy * 0.85, h * kz * 0.6]
        parts.append(Part(G.mjGEOM_ELLIPSOID, half, [dx * r, dy * r, h * kz * 0.4], _shade(rgba, shade)))
    return parts


def _marker(size: list[float], rgba: list[float]) -> list[Part]:
    if len(size) == 1:
        return [Part(G.mjGEOM_CYLINDER, [size[0] / 2, MARK / 2, 0], [0, 0, MARK / 2], rgba, collide=False)]
    return [_box([size[0] / 2, size[1] / 2, MARK / 2], [0, 0, MARK / 2], rgba, collide=False)]


BUOY_FLOAT = 0.4  # of a buoy's height: its float; ballast under it, a mast and a top mark over it


def _buoy(size: list[float], rgba: list[float]) -> list[Part]:
    """Ballast at the bottom (heavy, so it floats upright), a float, a mast and a ball as its top mark."""
    d, h = size
    r = d / 2
    ballast = 0.1 * h
    top = ballast + BUOY_FLOAT * h
    mast = 0.35 * h
    white = [0.94, 0.94, 0.92, 1.0]
    return [
        Part(G.mjGEOM_CYLINDER, [0.36 * r, ballast / 2, 0], [0, 0, ballast / 2], DARK, weight=0.45),
        Part(G.mjGEOM_CYLINDER, [r, BUOY_FLOAT * h / 2, 0], [0, 0, ballast + BUOY_FLOAT * h / 2], rgba),
        Part(G.mjGEOM_CYLINDER, [0.07 * r, mast / 2, 0], [0, 0, top + mast / 2], white, weight=0.04),
        Part(G.mjGEOM_SPHERE, [0.3 * r, 0, 0], [0, 0, top + mast + 0.25 * r], rgba, weight=0.02),
    ]


DOCK_DECK = 0.07  # m, deck thickness
DOCK_POST = 0.07  # m, post radius
DOCK_DEPTH = 3.0  # m the posts reach below the dock's base: down to the bottom wherever it is put


def _dock(size: list[float], rgba: list[float]) -> list[Part]:
    """A deck on two rows of posts. Its base is the ground (or the bottom) under it; the posts go on down."""
    length, width, height = size
    n = max(2, round(length / 1.3) + 1)
    parts = [_box([length / 2, width / 2, DOCK_DECK / 2], [0, 0, height - DOCK_DECK / 2], rgba)]
    post_h = height - DOCK_DECK + DOCK_DEPTH
    for i in range(n):
        x = -length / 2 + DOCK_POST + i * (length - 2 * DOCK_POST) / (n - 1)
        for side in (1, -1):
            y = side * (width / 2 - DOCK_POST)
            parts.append(
                Part(
                    G.mjGEOM_CYLINDER,
                    [DOCK_POST, post_h / 2, 0],
                    [x, y, height - DOCK_DECK - post_h / 2],
                    _shade(rgba, 0.7),
                )
            )
    return parts


def _shade(rgba: list[float], k: float) -> list[float]:
    return [min(1.0, c * k) for c in rgba[:3]] + [rgba[3]]


_PARTS = {
    "wall": _wall,
    "pillar": _pillar,
    "shelf": _shelf,
    "door_frame": _door_frame,
    "counter": _counter,
    "pallet": _pallet,
    "tree": _tree,
    "bench": _bench,
    "rock": _rock,
    "marker": _marker,
    "buoy": _buoy,
    "dock": _dock,
}
