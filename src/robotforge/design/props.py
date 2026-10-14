"""Furniture and architecture for scenes: walls, shelves, counters, pallets, trees and more.

A prop is one body at its base point (on the ground, at the center of its footprint),
turned by yaw. Fixed props are static bodies; movable ones get a free joint. Each
prop is a handful of boxes, cylinders and spheres, so it collides exactly as drawn
and costs little to simulate. Sizes are in meters; `length` runs along the prop's
own x axis and `depth` along y, with shelves and benches open (or facing) toward -y.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

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
}
PROPS = ("wall", "pillar", "shelf", "door_frame", "counter", "pallet", "tree", "bench", "rock", "marker")
DEFAULT_MASS = {"pallet": 20.0}


@dataclass
class Part:
    type: int
    size: list[float]
    pos: list[float]
    rgba: list[float]
    collide: bool = True
    quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])


def footprint(shape: str, size: list[float]) -> tuple[float, float]:
    """Half extents (x, y) of a shape's footprint in its own frame."""
    if shape == "tree":  # the trunk stands on the ground, the crown can hang over a slope
        return size[0] * 0.06, size[0] * 0.06
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
    total = mass if mass is not None else DEFAULT_MASS.get(shape, 5.0)
    volumes = [_volume(p) for p in parts]
    for i, part in enumerate(parts):
        kwargs = {}
        if movable:
            kwargs["mass"] = total * volumes[i] / sum(volumes)
        if not part.collide:
            kwargs.update(contype=0, conaffinity=0)
        body.add_geom(
            name=name if i == 0 else f"{name}/{i}",
            type=part.type,
            size=part.size,
            pos=part.pos,
            quat=part.quat,
            rgba=part.rgba,
            material=MATTE,
            **kwargs,
        )


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
_CROWN = (
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
    for dx, dy, dz, k, shade in _CROWN:
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


def _rock(size: list[float], rgba: list[float]) -> list[Part]:
    """Three overlapping squashed balls, sunk a little into the ground like a boulder."""
    d, h = size
    r = d / 2
    lumps = ((0.0, 0.0, 1.0, 1.0, 1.0), (0.3, 0.2, 0.7, 0.75, 0.92), (-0.25, -0.22, 0.65, 0.7, 0.86))
    parts = []
    for dx, dy, kxy, kz, shade in lumps:
        half = [r * kxy, r * kxy * 0.85, h * kz * 0.6]
        parts.append(Part(G.mjGEOM_ELLIPSOID, half, [dx * r, dy * r, h * kz * 0.4], _shade(rgba, shade)))
    return parts


def _marker(size: list[float], rgba: list[float]) -> list[Part]:
    if len(size) == 1:
        return [Part(G.mjGEOM_CYLINDER, [size[0] / 2, MARK / 2, 0], [0, 0, MARK / 2], rgba, collide=False)]
    return [_box([size[0] / 2, size[1] / 2, MARK / 2], [0, 0, MARK / 2], rgba, collide=False)]


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
}
