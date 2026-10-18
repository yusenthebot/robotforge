"""Ready-made scenes: a desk lab, a kitchen, a warehouse, a park, a maze and an obstacle course.

A scene is only world.yaml content (floor, terrain, objects) plus places for robots,
so after loading one an agent can still move, add or remove anything in it. Scenes
are generated from a seed: the same preset and seed always give the same world, and
another seed gives another maze, other tree spots or other crate stacks.
"""

from __future__ import annotations

import difflib
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robotforge.design import library, props

WOOD = [0.62, 0.48, 0.34, 1]
LIGHT_WOOD = [0.78, 0.64, 0.46, 1]
CRATE = [0.74, 0.58, 0.38, 1]
HEDGE = [0.2, 0.42, 0.22, 1]
WHITE = [0.92, 0.92, 0.9, 1]
STEEL = [0.36, 0.45, 0.56, 1]
YELLOW_LINE = [0.95, 0.78, 0.2, 1]
WATER = [0.25, 0.5, 0.75, 1]
PATH = [0.78, 0.7, 0.55, 1]


@dataclass(frozen=True)
class Option:
    default: float
    low: float
    high: float
    description: str
    integer: bool = False


@dataclass
class Slot:
    """Where a robot can go: on the ground (z None) or on a surface at height z (arms on tables)."""

    pos: tuple[float, float]
    yaw: float = 0.0
    z: float | None = None


@dataclass
class Content:
    """What a scene puts in the world."""

    floor: dict[str, Any]
    terrain: list[dict[str, Any]] = field(default_factory=list)
    objects: list[dict[str, Any]] = field(default_factory=list)
    ground_slots: list[Slot] = field(default_factory=list)
    arm_slots: list[Slot] = field(default_factory=list)
    starter: list[dict[str, Any]] = field(default_factory=list)  # robots for an empty world
    tips: list[str] = field(default_factory=list)

    def add(self, name: str, shape: str, pos: list[float], **fields: Any) -> None:
        self.objects.append({"name": name, "shape": shape, "pos": [round(v, 4) for v in pos], **fields})


@dataclass(frozen=True)
class Scene:
    name: str
    title: str
    summary: str
    build: Callable[[random.Random, dict[str, float]], Content]
    options: dict[str, Option] = field(default_factory=dict)


def resting_z(top: float, shape: str, size: list[float]) -> float:
    """The z to give an object so it rests on a surface at height `top` (furniture tops, shelf boards)."""
    if shape == "sphere":
        return top + size[0] / 2
    if shape in ("box", "cylinder"):
        return top + size[-1] / 2
    return top  # tables, bins and props are placed by their base


# ---- desk lab ----------------------------------------------------------------------------------

DESK_TOP = 0.75


def _desk_lab(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "tile"})
    c.add("back_wall", "wall", [0.0, 2.05], size=[5.2, 0.1, 1.2])
    c.add("side_wall", "wall", [-2.55, 0.0], size=[4.2, 0.1, 1.2], yaw=90)
    c.add("desk", "table", [0.0, 0.6], size=[1.4, 0.8, DESK_TOP], color=WOOD)
    # Things to pick up, all within reach of the default arm on the desk's front edge.
    items = [
        ("red_cube", "box", [0.18, 0.55], [0.04, 0.04, 0.04], "red"),
        ("blue_cube", "box", [0.04, 0.63], [0.04, 0.04, 0.04], "blue"),
        ("green_ball", "sphere", [-0.12, 0.6], [0.045], "green"),
        ("yellow_can", "cylinder", [0.3, 0.42], [0.04, 0.06], "yellow"),
    ]
    for name, shape, (x, y), size, color in items:
        c.add(name, shape, [x, y, resting_z(DESK_TOP, shape, size)], size=size, color=color)
    c.add("tray", "bin", [-0.28, 0.48, DESK_TOP], size=[0.16, 0.14, 0.06], color="gray")
    c.add("side_table", "table", [1.6, 1.45], size=[0.9, 0.55, DESK_TOP], color=LIGHT_WOOD)
    for i, color in enumerate(("orange", "purple", "teal")):
        size = [0.08, 0.08, 0.08]
        c.add(f"block_{i + 1}", "box", [1.35 + 0.22 * i, 1.4, resting_z(DESK_TOP, "box", size)], size=size, color=color)
    shelf = [1.2, 0.35, 1.6]
    c.add("shelf", "shelf", [-1.6, 1.8], size=shelf, color=LIGHT_WOOD)
    levels = props.shelf_levels(shelf)
    for i, (x, color) in enumerate(((-1.95, "red"), (-1.6, "blue"), (-1.25, "green"))):
        size = [0.12, 0.12, 0.12]
        c.add(f"box_{i + 1}", "box", [x, 1.78, resting_z(levels[1 + i % 2], "box", size)], size=size, color=color)
    c.arm_slots = [Slot((0.0, 0.32), 90, DESK_TOP), Slot((1.6, 1.25), 90, DESK_TOP)]
    c.ground_slots = [Slot((1.1, -0.9), 90), Slot((-1.1, -0.9), 90), Slot((0.0, -1.5), 90)]
    c.starter = [{"name": "arm", "template": "arm"}]
    c.tips = [
        "pick_and_place('arm', 'red_cube', 'tray') puts the cube in the tray",
        "the side table has three blocks for a second arm (add_robot with template='arm')",
    ]
    return c


# ---- kitchen -----------------------------------------------------------------------------------

COUNTER_TOP = 0.9


def _kitchen(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "wood"})
    c.add("back_wall", "wall", [0.0, 2.05], size=[6.2, 0.1, 2.2])
    c.add("side_wall", "wall", [-3.05, 0.2], size=[3.8, 0.1, 2.2], yaw=90)
    c.add("counter", "counter", [-0.6, 1.7], size=[3.6, 0.6, COUNTER_TOP], color=[0.3, 0.38, 0.47, 1])
    for i, x in enumerate((-1.9, -1.55)):
        c.add(
            f"burner_{i + 1}", "cylinder", [x, 1.72, COUNTER_TOP + 0.004], size=[0.2, 0.008], color="black", fixed=True
        )
    c.add("sink", "bin", [0.5, 1.7, COUNTER_TOP], size=[0.5, 0.38, 0.12], color="gray", fixed=True)
    c.add("fridge", "box", [1.85, 1.62], size=[0.75, 0.7, 1.85], color=WHITE, fixed=True)
    c.add("fridge_handle", "box", [1.55, 1.255, 1.1], size=[0.03, 0.03, 0.5], color="gray", fixed=True)
    shelf = [1.4, 0.3, 0.7]
    c.add("wall_shelf", "shelf", [-0.9, 1.85, 1.45], size=shelf, color=LIGHT_WOOD)
    board = 1.45 + props.shelf_levels(shelf)[0]
    for i, color in enumerate(("red", "yellow", "white", "teal", "orange")):
        size = [0.08, 0.12]
        c.add(
            f"jar_{i + 1}",
            "cylinder",
            [-1.45 + 0.27 * i, 1.83, resting_z(board, "cylinder", size)],
            size=size,
            color=color,
        )
    c.add("island", "counter", [0.0, 0.2], size=[1.4, 0.8, COUNTER_TOP], color=[0.55, 0.42, 0.32, 1])
    island = [
        ("red_cup", "cylinder", [0.15, 0.25], [0.06, 0.08], "red"),
        ("blue_cup", "cylinder", [-0.13, 0.27], [0.06, 0.08], "blue"),
        ("plate", "cylinder", [0.22, 0.17], [0.14, 0.015], "white"),
        ("apple", "sphere", [0.02, 0.33], [0.06], "green"),
        ("cereal", "box", [0.45, 0.45], [0.06, 0.14, 0.2], "yellow"),
    ]
    for name, shape, (x, y), size, color in island:
        c.add(name, shape, [x, y, resting_z(COUNTER_TOP, shape, size)], size=size, color=color)
    c.add("bowl", "bin", [-0.29, 0.1, COUNTER_TOP], size=[0.16, 0.16, 0.06], color="white")
    # A chest of drawers beside the arm, its drawer pulling out toward the arm (operate opens it; the plate fits in it),
    # and a bread cabinet on the counter by the fridge for a robot that comes to it.
    c.add("drawer", "drawer", [0.5, -0.05, COUNTER_TOP], size=[0.3, 0.3, 0.14], yaw=-90, color=WHITE)
    c.add("cabinet", "cabinet", [1.0, 1.75, COUNTER_TOP], size=[0.3, 0.24, 0.3], color=WHITE)
    c.add("dining_table", "table", [-1.9, -0.9], size=[1.2, 0.8, 0.75], color=WOOD)
    c.add("fruit", "sphere", [-1.9, -0.9, resting_z(0.75, "sphere", [0.08])], size=[0.08], color="orange")
    c.arm_slots = [Slot((0.0, -0.02), 90, COUNTER_TOP)]
    c.ground_slots = [Slot((1.3, -1.0), 90), Slot((-0.4, -1.4), 0), Slot((1.3, 0.3), 180)]
    c.starter = [{"name": "arm", "template": "arm"}]
    c.tips = [
        "pick_and_place('arm', 'red_cup', 'bowl') clears the island",
        "operate('arm', 'drawer', 'open') pulls the drawer out; with a suction cup (update_robot end_effector="
        "'suction') the arm can put the plate in it: pick_and_place('arm', 'plate', 'drawer')",
        "the jars on the wall shelf and the sink are out of the arm's reach: try a taller arm",
        "a robot that walks or drives, added without pos, starts on the first free one of robot_places, clear of "
        "the furniture; move(robot, to='island') takes it to the island",
    ]
    return c


# ---- factory -----------------------------------------------------------------------------------

BELT_TOP = 0.6  # m, the conveyor's belt
BELT_WIDTH = 0.12  # m: one box wide
STAND_TOP = 0.55  # m, the arm's stand: boxes on the belt come up to 10 cm above its base
BIN_STAND = 0.5  # m, the bins' stands
FACTORY_BOX = 0.05  # m, the boxes' edge
FACTORY_COLORS = ("red", "green", "blue")
# Where the parts are: the conveyor runs along +x to a stop at its end, where the arm on its stand picks.
CONVEYOR_X, CONVEYOR_LENGTH = -0.6, 2.0
STOP_X = CONVEYOR_X + CONVEYOR_LENGTH / 2 - 0.03  # middle of the stop across the belt's end
PICK_X = STOP_X - 0.015 - FACTORY_BOX / 2  # where a box against the stop has its middle
ARM_AT = (PICK_X, -0.24)  # a box anywhere across the belt at the stop is 0.2 to 0.28 m from the arm


def _factory(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "concrete"})
    c.add("back_wall", "wall", [0.0, 2.0], size=[6.0, 0.15, 2.4], color=[0.72, 0.74, 0.76, 1])
    c.add("lane", "marker", [0.0, -1.0], size=[5.0, 0.08], color=YELLOW_LINE)
    speed = float(opts["speed"])
    # A belt one box wide, so boxes come in single file: on a 30 cm belt the ones that came up while the arm was
    # busy waited side by side at the stop, and the arm could not reach the far one.
    c.add(
        "conveyor",
        "conveyor",
        [CONVEYOR_X, 0.0],
        size=[CONVEYOR_LENGTH, BELT_WIDTH, BELT_TOP],
        speed=speed,
        color=STEEL,
    )
    # A stop across the end of the belt: the belt slides on under what waits against it, ready to be picked.
    c.add(
        "stop",
        "box",
        [STOP_X, 0.0, BELT_TOP + 0.03],
        size=[0.03, BELT_WIDTH + 0.02, 0.06],
        color=YELLOW_LINE,
        fixed=True,
    )
    n = int(opts["boxes"])
    gap = (CONVEYOR_LENGTH - 0.2) / max(n, 1)
    for i in range(n):
        x = PICK_X - 0.12 - i * gap
        size = [FACTORY_BOX] * 3
        c.add(
            f"box_{i + 1}",
            "box",
            [x, rng.uniform(-0.01, 0.01), resting_z(BELT_TOP, "box", size)],
            size=size,
            color=rng.choice(FACTORY_COLORS),
        )
    c.add("stand", "pillar", [ARM_AT[0], ARM_AT[1] - 0.005], size=[0.15, STAND_TOP], color=[0.3, 0.32, 0.35, 1])
    # One bin per color round the arm, each on a stand of its own, where the arm puts things down pointing down.
    for color, (dx, dy) in zip(FACTORY_COLORS, ((0.26, 0.0), (0.0, -0.26), (-0.26, 0.0)), strict=True):
        x, y = ARM_AT[0] + dx, ARM_AT[1] + dy
        c.add(f"{color}_stand", "table", [x, y], size=[0.2, 0.2, BIN_STAND], color=[0.45, 0.47, 0.5, 1])
        c.add(f"{color}_bin", "bin", [x, y, BIN_STAND], size=[0.17, 0.17, 0.08], color=color, fixed=True)
    # A workbench with a drawer of parts.
    c.add("workbench", "table", [1.4, 0.9], size=[1.0, 0.6, 0.75], color=LIGHT_WOOD)
    c.add("parts_drawer", "drawer", [1.4, 0.95, 0.75], size=[0.4, 0.34, 0.14], color=STEEL)
    for i, color in enumerate(("gray", "orange", "gray")):
        size = [0.03, 0.03]
        c.add(
            f"part_{i + 1}",
            "cylinder",
            [1.3 + 0.1 * i, 0.98, resting_z(0.75 + 0.021, "cylinder", size)],
            size=size,
            color=color,
        )
    # Facing its red bin: at rest its hand is over the bin, not over the stop, where it hid the boxes from the camera.
    c.arm_slots = [Slot(ARM_AT, 0, STAND_TOP)]
    c.ground_slots = [Slot((1.4, -0.4), 90), Slot((-1.6, -0.8), 0)]
    c.starter = [{"name": "arm", "template": "arm", "end_effector": "suction"}]
    c.tips = [
        "the conveyor brings colored boxes to the stop at its end; observe(find='all', view='top') says each box's "
        "color and where it is",
        f"pick_and_place('arm', 'box_1', 'red_bin') sorts one (the arm has a suction cup); stop the belt while "
        f"picking with set_controls({{'conveyor/speed': 0}}) and start it again with {speed:g}",
        "the workbench's drawer holds parts: operate opens it",
    ]
    return c


# ---- warehouse ---------------------------------------------------------------------------------


def _warehouse(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "concrete"})
    c.add("back_wall", "wall", [0.0, 5.0], size=[14.0, 0.2, 3.0], color=[0.72, 0.74, 0.76, 1])
    c.add("side_wall", "wall", [-7.0, 1.5], size=[7.2, 0.2, 3.0], yaw=90, color=[0.72, 0.74, 0.76, 1])
    unit = [2.0, 0.6, 2.0]
    levels = props.shelf_levels(unit)
    n = 0
    # Two rows of shelving face each other across an aisle at y = 2.6.
    for row_y, yaw in ((3.9, 0.0), (1.3, 180.0)):
        for x in (-4.0, -2.0, 0.0, 2.0):
            n += 1
            c.add(f"rack_{n}", "shelf", [x, row_y], size=unit, yaw=yaw, color=STEEL)
            for level in levels[:-1]:
                if rng.random() < 0.55:
                    s = rng.choice((0.4, 0.45, 0.5))
                    size = [s, 0.4, round(rng.uniform(0.22, 0.32), 2)]
                    px = x + rng.uniform(-0.55, 0.55)
                    c.add(
                        f"crate_{len(c.objects)}",
                        "box",
                        [px, row_y, resting_z(level, "box", size)],
                        size=size,
                        color=CRATE,
                        mass=4.0,
                    )
    for i, (x, y) in enumerate(((-4.5, -1.6), (-2.6, -1.9), (-0.8, -1.5))):
        c.add(f"pallet_{i + 1}", "pallet", [x, y], yaw=rng.uniform(-8, 8))
        top = 0.144
        for j in range(rng.randint(1, 4)):
            size = [0.5, 0.36, 0.3]
            dx, dy = (-0.3 if j % 2 == 0 else 0.3), (-0.2 if j < 2 else 0.2)
            z = resting_z(top, "box", size)
            c.add(f"crate_{len(c.objects)}", "box", [x + dx, y + dy, z], size=size, color=CRATE, mass=5.0)
    for i, (x, y) in enumerate(((-6.3, 0.6), (4.6, 3.6))):
        c.add(f"column_{i + 1}", "pillar", [x, y], size=[0.4, 3.0])
    c.add("lane_north", "marker", [-1.0, 3.35], size=[9.0, 0.08], color=YELLOW_LINE)
    c.add("lane_south", "marker", [-1.0, 1.85], size=[9.0, 0.08], color=YELLOW_LINE)
    # A loading dock at the east end, reached by a gentle ramp.
    c.terrain.append(
        {
            "type": "ramp",
            "name": "dock",
            "pos": [5.0, -1.4],
            "angle": 8,
            "length": 2.1,
            "landing": 2.0,
            "width": 2.4,
            "direction": "up",
            "style": "concrete",
        }
    )
    c.ground_slots = [Slot((-5.5, 2.6), 0), Slot((-5.5, -0.4), 0), Slot((2.0, -0.4), 0)]
    c.starter = [{"name": "rover", "template": "rover"}]
    c.tips = [
        # A tip that drove straight (move('rover', 0.4)) had an agent split the trip to the dock in two by hand.
        "move('rover', to='dock') finds its own way round the racks and up the ramp onto the dock; "
        "move('rover', 0.4) drives straight down the aisle between the racks",
        "crates are loose: a robot can push them off the pallets",
    ]
    return c


# ---- park --------------------------------------------------------------------------------------


def _park(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "grass"})
    c.terrain.append(
        {
            "type": "hills",
            "name": "hills",
            "pos": [5.5, 4.0],
            "size": [9.0, 8.0],
            "height": 0.6,
            "scale": 2.6,
            "seed": rng.randrange(1000),
        }
    )
    c.add("path", "marker", [0.0, -1.0], size=[16.0, 1.2], color=PATH)
    c.add("pond", "marker", [-4.5, 3.5], size=[3.4], color=WATER)
    c.add("picnic_table", "table", [-1.2, 2.6], size=[1.6, 0.8, 0.72], color=WOOD)
    for i, x in enumerate((-4.0, 2.5)):
        c.add(f"bench_{i + 1}", "bench", [x, 0.0])
    spots: list[tuple[float, float]] = []
    tries = 0
    while len(spots) < 9 and tries < 400:
        tries += 1
        x, y = rng.uniform(-7.5, 9.0), rng.uniform(-6.0, 7.5)
        if abs(y + 1.0) < 1.6 or math.hypot(x, y) < 2.5:  # keep the path and the start clear
            continue
        if math.hypot(x + 4.5, y - 3.5) < 2.6 or math.hypot(x + 1.2, y - 2.6) < 1.8:  # pond, picnic table
            continue
        if any(math.hypot(x - a, y - b) < 2.2 for a, b in spots):
            continue
        spots.append((x, y))
    for i, (x, y) in enumerate(spots):
        crown = rng.uniform(1.2, 2.0)
        c.add(f"tree_{i + 1}", "tree", [x, y], size=[round(crown, 2), round(crown * rng.uniform(1.6, 2.0), 2)])
    for i in range(4):
        x, y = rng.uniform(-6, 6), rng.choice((rng.uniform(-6, -2.8), rng.uniform(1.2, 2.0)))
        d = rng.uniform(0.4, 0.8)
        c.add(f"rock_{i + 1}", "rock", [x, y], size=[round(d, 2), round(d * 0.55, 2)], yaw=rng.uniform(0, 360))
    c.add("ball", "sphere", [1.5, -0.6], size=[0.2], color="red", mass=0.3)
    c.ground_slots = [Slot((0.0, -1.0), 0), Slot((-1.5, -1.0), 0), Slot((1.5, -1.4), 0)]
    c.starter = [{"name": "dog", "template": "quadruped"}]
    c.tips = [
        "move('dog', to='bench_1') finds its own way to a bench round the trees and rocks; move('dog', 0.3) walks "
        "straight along the path, and turning left takes it into the hills",
        "the red ball is loose: nudge it with a robot",
    ]
    return c


# ---- maze --------------------------------------------------------------------------------------


def _maze(rng: random.Random, opts: dict[str, float]) -> Content:
    rows, cols, cell = int(opts["rows"]), int(opts["cols"]), float(opts["cell"])
    height, thick = float(opts["wall_height"]), 0.1
    c = Content(floor={"style": "grass"})
    open_walls = carve_maze(rows, cols, rng)
    x0, y0 = -cols * cell / 2, -rows * cell / 2
    # Horizontal walls (along x) on row lines 0..rows; vertical walls on column lines 0..cols.
    # Runs of neighboring segments become one long wall: fewer bodies, cleaner contacts.
    horizontal = []
    for line in range(rows + 1):
        segs = []
        for k in range(cols):
            if line == 0 and k == 0 or line == rows and k == cols - 1:
                segs.append(False)  # entrance (south-west) and exit (north-east)
            elif 0 < line < rows:
                segs.append((line - 1, k, line, k) not in open_walls)
            else:
                segs.append(True)
        horizontal.append(segs)
    vertical = []
    for line in range(cols + 1):
        segs = []
        for r in range(rows):
            segs.append(not (0 < line < cols and (r, line - 1, r, line) in open_walls))
        vertical.append(segs)
    n = 0
    for line, segs in enumerate(horizontal):
        for start, length in _runs(segs):
            n += 1
            x = x0 + (start + length / 2) * cell
            c.add(f"wall_{n}", "wall", [x, y0 + line * cell], size=[length * cell + thick, thick, height], color=HEDGE)
    for line, segs in enumerate(vertical):
        for start, length in _runs(segs):
            n += 1
            y = y0 + (start + length / 2) * cell
            c.add(
                f"wall_{n}",
                "wall",
                [x0 + line * cell, y],
                size=[length * cell + thick, thick, height],
                yaw=90,
                color=HEDGE,
            )
    goal = (x0 + (cols - 0.5) * cell, y0 + rows * cell + 0.5 * cell)
    c.add("goal", "marker", list(goal), size=[cell * 0.6], color=[0.3, 0.8, 0.45, 1])
    c.add("start", "marker", [x0 + 0.5 * cell, y0 - 0.5 * cell], size=[cell * 0.6], color=[0.95, 0.78, 0.2, 1])
    c.ground_slots = [Slot((x0 + 0.5 * cell, y0 - 0.5 * cell), 90), Slot((x0 + 1.6 * cell, y0 - 0.6 * cell), 90)]
    c.starter = [{"name": "rover", "template": "rover"}]
    middle = (x0 + (cols // 2 + 0.5) * cell, y0 + (rows // 2 + 0.5) * cell)  # a cell's middle, never on a wall
    c.tips = [
        f"the entrance is at the south-west corner, the goal (green mark) beyond the north-east corner at "
        f"[{goal[0]:.2f}, {goal[1]:.2f}]",
        # The tip said to='goal' went "through": the planned trip was 11.1 m, all of it outside the hedges.
        f"move('rover', to='goal') takes the shortest way, round the outside of the hedges; to drive through the "
        f"maze, send it to the middle first, move('rover', to=[{middle[0]:.2f}, {middle[1]:.2f}]), and to='goal' "
        "from there takes it on out of the north-east exit",
        "load_scene('maze', seed=7) makes a different maze; options rows, cols and cell change its size",
    ]
    return c


def carve_maze(rows: int, cols: int, rng: random.Random) -> set[tuple[int, int, int, int]]:
    """Openings between neighboring cells, as (row, col, row, col) with the smaller cell first.

    Depth-first carving makes a perfect maze: every cell is reachable and there is exactly
    one way between any two cells.
    """
    open_walls: set[tuple[int, int, int, int]] = set()
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, k = stack[-1]
        nxt = [(r + dr, k + dk) for dr, dk in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nxt = [(a, b) for a, b in nxt if 0 <= a < rows and 0 <= b < cols and (a, b) not in seen]
        if not nxt:
            stack.pop()
            continue
        a, b = rng.choice(nxt)
        open_walls.add((min(r, a), min(k, b), max(r, a), max(k, b)))
        seen.add((a, b))
        stack.append((a, b))
    return open_walls


def _runs(segs: list[bool]) -> list[tuple[int, int]]:
    """(start, length) of each run of True."""
    out, start = [], None
    for i, on in enumerate([*segs, False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i - start))
            start = None
    return out


# ---- obstacle course ---------------------------------------------------------------------------


def _obstacle_course(rng: random.Random, opts: dict[str, float]) -> Content:
    c = Content(floor={"style": "checker"})
    width = 1.6
    t = c.terrain
    t.append(
        {
            "type": "ramp",
            "name": "ramp",
            "pos": [2.3, 0.0],
            "angle": 10,
            "length": 1.0,
            "landing": 0.6,
            "width": width,
            "style": "wood",
        }
    )
    t.append(
        {
            "type": "stairs",
            "name": "stairs",
            "pos": [5.6, 0.0],
            "steps": 3,
            "step_height": 0.04,
            "step_depth": 0.3,
            "landing": 0.6,
            "width": width,
        }
    )
    t.append(
        {
            "type": "rough",
            "name": "rough",
            "pos": [8.3, 0.0],
            "size": [2.0, width],
            "height": 0.03,
            "seed": rng.randrange(1000),
        }
    )
    for i, x in enumerate((10.0, 10.6)):
        t.append({"type": "curb", "name": f"curb_{i + 1}", "pos": [x, 0.0], "size": [0.12, width], "height": 0.04})
    t.append(
        {
            "type": "stones",
            "name": "stones",
            "pos": [12.2, 0.0],
            "size": [1.6, 1.4],
            "stone": 0.3,
            "gap": 0.04,
            "height": 0.05,
            "jitter": 0.012,
            "seed": rng.randrange(1000),
        }
    )
    t.append(
        {
            "type": "hills",
            "name": "bumps",
            "pos": [15.3, 0.0],
            "size": [3.0, width + 0.4],
            "height": 0.12,
            "scale": 0.8,
            "seed": rng.randrange(1000),
        }
    )
    c.add("start", "marker", [0.0, 0.0], size=[0.5, width], color=[0.3, 0.8, 0.45, 1])
    c.add("finish", "marker", [17.6, 0.0], size=[0.5, width], color=[0.92, 0.3, 0.28, 1])
    for side in (1, -1):
        c.add(
            f"lane_{'left' if side > 0 else 'right'}",
            "marker",
            [8.8, side * (width / 2 + 0.1)],
            size=[18.0, 0.06],
            color=WHITE,
        )
    c.ground_slots = [Slot((0.0, 0.0), 0), Slot((-0.8, 0.0), 0)]
    c.starter = [{"name": "dog", "template": "quadruped"}]
    c.tips = [
        "everything runs along +x from the green start line to the red finish at x = 17.6",
        "in order: a 10 degree ramp, 3 steps of 4 cm, rough ground, two 4 cm curbs, stepping stones, bumps",
    ]
    return c


SCENES: dict[str, Scene] = {
    s.name: s
    for s in [
        Scene(
            "desk_lab",
            "Desk lab",
            "A desk with blocks, a ball and a tray to sort them into, a side table and a "
            "shelf. Starter robot: an arm on the desk.",
            _desk_lab,
        ),
        Scene(
            "kitchen",
            "Kitchen",
            "Counter with a sink, burners and a bread cabinet, a fridge, a wall shelf with jars and an island "
            "with cups, a plate, a bowl, an apple and a chest of drawers. Starter robot: an arm on the island.",
            _kitchen,
        ),
        Scene(
            "factory",
            "Factory",
            "A conveyor bringing colored boxes to a stop, an arm with a suction cup on a stand to sort them into a "
            "red, a green and a blue bin, and a workbench with a drawer of parts. Starter robot: the arm.",
            _factory,
            {
                "boxes": Option(12, 1, 16, "boxes on the belt", integer=True),
                "speed": Option(0.1, 0.0, 0.3, "belt speed, m/s"),
            },
        ),
        Scene(
            "warehouse",
            "Warehouse",
            "Two rows of steel racks with crates, pallets with loose crates, columns, "
            "lane markings and a loading dock up a ramp. Starter robot: a rover.",
            _warehouse,
        ),
        Scene(
            "park",
            "Park",
            "Grass with rolling hills, trees, benches, a path, a pond, rocks and a picnic table. "
            "Starter robot: a robot dog.",
            _park,
        ),
        Scene(
            "maze",
            "Hedge maze",
            "A seeded maze with one way through, from the yellow start mark to the green goal. Starter robot: a rover.",
            _maze,
            {
                "rows": Option(5, 2, 15, "cells from south to north", integer=True),
                "cols": Option(5, 2, 15, "cells from west to east", integer=True),
                "cell": Option(1.0, 0.5, 3.0, "corridor width, m"),
                "wall_height": Option(0.4, 0.1, 2.0, "m"),
            },
        ),
        Scene(
            "obstacle_course",
            "Obstacle course",
            "A straight course along +x: ramp, stairs, rough ground, curbs, "
            "stepping stones and bumps. Starter robot: a robot dog.",
            _obstacle_course,
        ),
    ]
}


def catalog() -> list[dict[str, Any]]:
    out = []
    for s in SCENES.values():
        entry: dict[str, Any] = {"name": s.name, "title": s.title, "summary": s.summary}
        if s.options:
            entry["options"] = {
                k: {"default": o.default, "min": o.low, "max": o.high, "description": o.description}
                for k, o in s.options.items()
            }
        out.append(entry)
    return out


def build(name: str, seed: int = 0, options: dict[str, float] | None = None) -> Content:
    """Generate a scene's content. Raises ValueError with a suggestion for unknown names or options."""
    scene = SCENES.get(name)
    if scene is None:
        close = difflib.get_close_matches(name, list(SCENES), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"unknown scene {name!r}.{hint} Scenes: {', '.join(SCENES)}")
    opts = {k: o.default for k, o in scene.options.items()}
    for key, value in (options or {}).items():
        spec = scene.options.get(key)
        if spec is None:
            known = ", ".join(scene.options) or "none"
            raise ValueError(f"scene {name!r} has no option {key!r}; its options: {known}")
        if not isinstance(value, int | float) or not spec.low <= value <= spec.high:
            raise ValueError(f"{name} option {key} must be a number from {spec.low:g} to {spec.high:g}")
        if spec.integer and value != int(value):
            raise ValueError(f"{name} option {key} must be a whole number")
        opts[key] = value
    return scene.build(random.Random(seed), opts)


# ---- putting a scene into a world -----------------------------------------------------------------


def robot_kind(entry: Any) -> str:
    """'arm' for robots that stand on a surface (fixed base), else 'ground'."""
    if entry.template is not None:
        return "arm" if entry.template == "arm" else "ground"
    if entry.library is not None:
        return "arm" if library.get(entry.library).kind == "arm" else "ground"
    return "ground"


def place(robots: list[Any], content: Content, spawn_height: Callable[[Any], float]) -> list[str]:
    """Move robots onto the scene's slots (arms onto surfaces); returns one note per robot moved.

    `spawn_height(entry)` is how far above its base frame a robot's lowest point sits, which an
    arm on a table needs added to the table height.
    """
    notes = []
    ground, arms = list(content.ground_slots), list(content.arm_slots)
    extra = 0
    for entry in robots:
        kind = robot_kind(entry)
        slots = arms if kind == "arm" and arms else ground
        if slots:
            slot = slots.pop(0)
        else:  # more robots than places: line them up next to the first ground place
            extra += 1
            base = content.ground_slots[0] if content.ground_slots else Slot((0.0, 0.0))
            slot = Slot((base.pos[0], base.pos[1] - 0.8 * extra), base.yaw)
        if slot.z is None:
            entry.pos = [slot.pos[0], slot.pos[1]]
        else:
            entry.pos = [slot.pos[0], slot.pos[1], round(slot.z + spawn_height(entry), 4)]
        entry.yaw = slot.yaw
        where = f"[{slot.pos[0]:g}, {slot.pos[1]:g}]" + (f" on a surface {slot.z:g} m high" if slot.z else "")
        notes.append(f"{entry.name} placed at {where}")
    return notes
