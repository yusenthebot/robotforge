"""Scenes with water: a harbor on the sea, a lake in a park and an indoor pool (registered in scenes.SCENES).

The floor is the bottom (sand, a lake bed, pool tiles) and land is terrain that rises above the water: concrete
quays in the harbor, a grassy basin round the lake, a tiled deck round the pool. Buoys are moored where they are
put, wooden crates float, a steel block lies on the bottom, and boats and ROVs start on the water (water_slots).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # scenes imports this module
    from robotforge.design.scenes import Content

WOOD = [0.62, 0.48, 0.34, 1]
CRATE = [0.74, 0.58, 0.38, 1]
CONCRETE = [0.66, 0.65, 0.62, 1]
STEEL = [0.5, 0.53, 0.56, 1]
RED, GREEN = [0.85, 0.2, 0.16, 1], [0.15, 0.6, 0.3, 1]
WHITE = [0.94, 0.94, 0.92, 1]
FLOATING_CRATE = 450.0  # kg/m^3, a wooden crate with air in it
STEEL_DENSITY = 7850.0


def _crate(c: Content, name: str, pos: list[float], size: list[float], density: float = FLOATING_CRATE) -> None:
    mass = round(density * size[0] * size[1] * size[2], 2)
    c.add(name, "box", pos, size=size, color=CRATE, mass=mass)


def _steel(c: Content, name: str, pos: list[float], side: float = 0.25) -> None:
    c.add(name, "box", pos, size=[side, side, side], color=STEEL, mass=round(STEEL_DENSITY * side**3, 1))


# ---- harbor -----------------------------------------------------------------------------------------------


def harbor(rng: random.Random, opts: dict[str, float]) -> Content:
    from robotforge.design.scenes import Content, Slot

    level = float(opts["depth"])
    quay = level + 0.4
    c = Content(floor={"style": "sand"})
    c.water = {"level": level, "waves": 0.03, "wavelength": 5.0, "heading": 70.0, "density": 1025.0}
    # A quay along the north and a breakwater down the east side make the harbor basin; the sea is open to the south.
    c.terrain.append(
        {
            "type": "platform",
            "name": "quay",
            "pos": [0.0, 8.0],
            "size": [26.0, 8.0],
            "height": quay,
            "style": "concrete",
        }
    )
    c.terrain.append(
        {
            "type": "platform",
            "name": "breakwater",
            "pos": [11.0, -1.0],
            "size": [4.0, 10.0],
            "height": quay,
            "style": "rock",
        }
    )
    # A wooden dock out from the quay (its base on the sea bed, the deck 35 cm over the water).
    c.add("dock", "dock", [-3.0, 0.9, 0.0], size=[6.2, 1.4, level + 0.35], yaw=90, color=WOOD)
    for i, x in enumerate((-9.0, -6.0, 0.0, 3.0, 6.0)):
        c.add(f"bollard_{i + 1}", "pillar", [x, 4.35], size=[0.28, 0.45], color=[0.2, 0.22, 0.25, 1])
    c.add("lighthouse", "pillar", [11.0, -5.2], size=[0.9, 3.2], color=WHITE)
    c.add("lighthouse_top", "pillar", [11.0, -5.2, quay + 3.2], size=[1.0, 0.6], color=RED, fixed=True)
    # Cargo on the quay, stacked; some crates fell in and float; a steel block lies on the bottom.
    for i in range(6):
        x = -10.0 + 1.0 * (i % 3) + rng.uniform(-0.1, 0.1)
        y = 6.6 + 1.0 * (i // 3)
        _crate(c, f"cargo_{i + 1}", [x, y], [0.8, 0.8, 0.7], density=300.0)
    _crate(c, "cargo_top", [-9.0, 7.1, quay + 0.7 + 0.35], [0.8, 0.8, 0.7], density=300.0)
    for i, (x, y) in enumerate(((2.5, -1.0), (5.0, 1.5), (-6.5, -2.5))):
        _crate(c, f"crate_{i + 1}", [x + rng.uniform(-0.3, 0.3), y + rng.uniform(-0.3, 0.3)], [0.5, 0.4, 0.35])
    _steel(c, "steel_block", [-1.6, -1.5])
    # The way in: green to the east, red to the west, as you come in from the sea.
    for i, (x, y, color) in enumerate(((5.5, -6.0, GREEN), (5.5, -2.5, GREEN), (-0.5, -6.0, RED), (-0.5, -2.5, RED))):
        c.add(f"buoy_{i + 1}", "buoy", [x, y], color=color)
    c.water_slots = [Slot((2.5, -8.0), 90.0), Slot((0.0, -4.2), 0.0), Slot((-6.0, -6.0), 45.0)]
    c.ground_slots = [Slot((-3.0, 6.0), -90.0), Slot((3.0, 6.5), -90.0), Slot((-6.0, 6.0), -90.0)]
    c.starter = [{"name": "boat", "template": "boat"}]
    c.tips = [
        "the boat starts at sea: move('boat', to='dock') brings it in between the buoys to the dock",
        "add_robot('rov', template='rov') puts an ROV on the water; move('rov', to=[0, -3, 0.6]) dives to 0.8 m",
        "the crates float, the steel block lies on the bottom; robots on land stand on the quay",
    ]
    return c


# ---- lake -------------------------------------------------------------------------------------------------

LAKE = (16.0, 11.0)  # m across the hollow
LAKE_LEVEL = 0.6
LAND = 0.9


def lake(rng: random.Random, opts: dict[str, float]) -> Content:
    from robotforge.design.scenes import Content, Slot

    level = LAKE_LEVEL
    c = Content(floor={"style": "sand"})
    c.terrain.append(
        {
            "type": "basin",
            "name": "shore",
            "pos": [0.0, 0.0],
            "size": [32.0, 25.0],
            "lake": list(LAKE),
            "height": LAND,
            "beach": 2.5,
            "seed": rng.randrange(1000),
            "style": "grass",
        }
    )
    a, b = LAKE[0] / 2, LAKE[1] / 2
    c.water = {
        "level": level,
        "waves": 0.012,
        "wavelength": 2.5,
        "heading": 20.0,
        "area": [-a - 4, -b - 4, a + 4, b + 4],
    }
    # A dock out from the west shore; its base on the lake bed, the deck 35 cm over the water.
    c.add("dock", "dock", [-a + 0.3, 0.0, 0.0], size=[4.6, 1.3, level + 0.35], color=WOOD)
    c.add("bench", "bench", [-a - 3.0, 2.2], yaw=-90)
    c.add("picnic_table", "table", [-a - 3.2, -2.4], size=[1.5, 0.8, 0.72], color=WOOD)
    # A slalom of buoys across the lake, a few crates adrift, and a steel block sunk near the dock.
    xs = (-2.5, 0.5, 3.5)
    for i, x in enumerate(xs):
        y = (1.6 if i % 2 == 0 else -1.6) + rng.uniform(-0.3, 0.3)
        c.add(f"buoy_{i + 1}", "buoy", [x, y], color=RED if i % 2 == 0 else GREEN)
    for i in range(3):
        angle = rng.uniform(0, 2 * math.pi)
        x, y = 0.55 * a * math.cos(angle), 0.55 * b * math.sin(angle)
        if all(math.hypot(x - bx, y) > 1.5 for bx in xs):
            _crate(c, f"crate_{i + 1}", [round(x, 2), round(y, 2)], [0.45, 0.35, 0.3])
    _steel(c, "steel_block", [-a + 2.0, 1.2])
    # Trees and rocks on the land round the lake, clear of the shore and the dock.
    spots: list[tuple[float, float]] = []
    tries = 0
    while len(spots) < 12 and tries < 600:
        tries += 1
        x, y = rng.uniform(-14.0, 14.0), rng.uniform(-11.0, 11.0)
        if math.hypot(x / (a + 3.2), y / (b + 3.2)) < 1.0 or abs(y) < 3.5 and x < -a:
            continue
        if any(math.hypot(x - p, y - q) < 2.6 for p, q in spots):
            continue
        spots.append((x, y))
    for i, (x, y) in enumerate(spots):
        crown = rng.uniform(1.4, 2.2)
        c.add(f"tree_{i + 1}", "tree", [round(x, 2), round(y, 2)], size=[round(crown, 2), round(crown * 1.8, 2)])
    for i in range(5):
        angle = rng.uniform(0, 2 * math.pi)
        r = 1.0 + rng.uniform(0.15, 0.3)
        d = rng.uniform(0.5, 0.9)
        c.add(
            f"rock_{i + 1}",
            "rock",
            [round(r * a * math.cos(angle), 2), round(r * b * math.sin(angle), 2)],
            size=[round(d, 2), round(d * 0.6, 2)],
        )
    c.water_slots = [Slot((a - 2.0, -0.8), 180.0), Slot((a - 3.0, 2.0), 180.0), Slot((1.0, 0.0), 180.0)]
    c.ground_slots = [Slot((-a - 3.0, 0.0), 0.0), Slot((0.0, -b - 3.0), 90.0), Slot((0.0, b + 3.0), -90.0)]
    c.starter = [{"name": "boat", "template": "boat"}]
    c.tips = [
        "the boat starts at the east end: move('boat', to='dock') takes it across, round the buoys, to the dock",
        "the buoys are moored; the crates drift; set_floor(water={'level': 0.6, 'waves': 0.04}) makes it rough",
    ]
    return c


# ---- pool -------------------------------------------------------------------------------------------------


def pool(rng: random.Random, opts: dict[str, float]) -> Content:
    from robotforge.design.scenes import Content, Slot

    depth = float(opts["depth"])
    deck = depth + 0.25
    length, width = 12.0, 7.0
    hx, hy = length / 2, width / 2
    c = Content(floor={"style": "tile"})
    c.water = {"level": depth, "waves": 0.003, "wavelength": 1.5, "area": [-hx, -hy, hx, hy]}
    around = 4.0
    for name, pos, size in (
        ("deck_north", [0.0, hy + around / 2], [length + 2 * around, around]),
        ("deck_south", [0.0, -hy - around / 2], [length + 2 * around, around]),
        ("deck_west", [-hx - around / 2, 0.0], [around, width]),
        ("deck_east", [hx + around / 2, 0.0], [around, width]),
    ):
        c.terrain.append({"type": "platform", "name": name, "pos": pos, "size": size, "height": deck, "style": "tile"})
    # Lane lines on the bottom, a ladder's rails at the shallow end, walls round the hall.
    for i, y in enumerate((-2.1, 0.0, 2.1)):
        c.add(f"lane_{i + 1}", "marker", [0.0, y, 0.0], size=[length - 2.0, 0.2], color=[0.12, 0.25, 0.5, 1])
    for side in (-0.3, 0.3):
        c.add(
            f"ladder_{'left' if side > 0 else 'right'}",
            "pillar",
            [-hx + 0.12, side, 0.0],
            size=[0.05, deck + 0.8],
            color=[0.8, 0.82, 0.85, 1],
        )
    hall = length + 2 * around
    for name, pos, size, yaw in (
        ("wall_north", [0.0, hy + around], [hall, 0.2, 3.0], 0.0),
        ("wall_south", [0.0, -hy - around], [hall, 0.2, 3.0], 0.0),
        ("wall_west", [-hall / 2, 0.0], [width + 2 * around, 0.2, 3.0], 90.0),
        ("wall_east", [hall / 2, 0.0], [width + 2 * around, 0.2, 3.0], 90.0),
    ):
        c.add(name, "wall", pos, size=size, yaw=yaw, color=[0.88, 0.9, 0.92, 1])
    _steel(c, "steel_block", [2.5, -1.0], side=0.3)
    _crate(c, "crate", [-2.0, 1.6], [0.4, 0.4, 0.3])
    for i, (x, y, color) in enumerate(((3.5, 2.0, "orange"), (-3.5, -2.2, "yellow"))):
        c.add(f"ball_{i + 1}", "sphere", [x, y], size=[0.3], color=color, mass=0.4)
    c.water_slots = [Slot((-3.0, 0.0), 0.0), Slot((0.0, -1.5), 0.0), Slot((0.0, 1.5), 0.0)]
    c.ground_slots = [Slot((-hx - 2.0, 0.0), 0.0), Slot((0.0, hy + 2.0), -90.0), Slot((0.0, -hy - 2.0), 90.0)]
    c.starter = [{"name": "rov", "template": "rov"}]
    c.tips = [
        f"the pool is {depth:g} m deep; move('rov', to=[2.5, -1.0, 0.6]) dives to the steel block on the bottom",
        "the ROV holds its depth and heading; R and F in the viewer take it up and down",
    ]
    return c
