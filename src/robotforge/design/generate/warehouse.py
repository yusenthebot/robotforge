"""Generated warehouses: rows of steel racks with aisles between them, pallets of crates, columns, lane lines, a
packing table with a tote, and a loading dock up a ramp at the east end.

Racks face their aisle; the boards at hand height (0.3 to 1.0 m) carry small parcels a robot with an arm can fetch,
the others big crates. The aisles run along x, 2.4 m wide, so a robot drives the length of the hall between them.
"""

from __future__ import annotations

from robotforge.design import props
from robotforge.design.generate import things
from robotforge.design.generate.draft import HIGH, LOW, Draft
from robotforge.design.generate.space import Footprint
from robotforge.design.generate.words import ITEMS, Wish
from robotforge.design.scenes import resting_z

STEEL = [0.36, 0.45, 0.56, 1]
CRATE = [0.74, 0.58, 0.38, 1]
YELLOW = [0.95, 0.78, 0.2, 1]
WALL = [0.72, 0.74, 0.76, 1]
UNIT = (2.0, 0.6, 2.0)  # one rack: length, depth, height
AISLE = 2.4  # m between two racks that face each other
PACKING = 0.65  # m, the packing table's top
DOCK = {"angle": 8, "length": 2.1, "landing": 2.0, "width": 2.4}


def build(draft: Draft, wish: Wish) -> None:
    rng, c = draft.rng, draft.content
    s = wish.size
    rows = min(max(wish.counts.get("rack") or wish.counts.get("shelf") or rng.choice((2, 3, 3, 4)), 1), 5)
    width = round(16.0 * s, 1)
    depth = round(max(5.5 + rows * 2.1, 10.0 * s), 1)
    x0, x1, y0, y1 = -width / 2, width / 2, -depth / 2, depth / 2
    draft.bounds = (x0 - 0.2, x1 + 0.2, y0 - 0.2, y1 + 0.2)
    c.floor = {"style": "concrete"}
    for name, fp, length in (
        ("north_wall", Footprint(0.0, y1, width / 2 + 0.1, 0.1), width + 0.2),
        ("south_wall", Footprint(0.0, y0, width / 2 + 0.1, 0.1), width + 0.2),
        ("west_wall", Footprint(x0, 0.0, depth / 2, 0.1, 90.0), depth),
    ):
        draft.walls.add(draft.put(name, "wall", fp, [length, 0.2, 2.4], color=WALL))
    # the dock: a ramp up to a landing at the east end (terrain), kept clear of everything else
    dock_x, dock_y = x1 - DOCK["length"] / 2 - DOCK["landing"] - 0.3, y0 + 2.0
    c.terrain.append(
        {
            "type": "ramp",
            "name": "dock",
            "pos": [round(dock_x, 2), round(dock_y, 2)],
            "direction": "up",
            "style": "concrete",
            **DOCK,
        }
    )
    dock_fp = Footprint(dock_x + DOCK["landing"] / 2, dock_y, (DOCK["length"] + DOCK["landing"]) / 2 + 0.4, 1.7)
    draft.keepout.append(dock_fp)
    # racks: rows along x from the north wall down, facing pairs across an aisle, back to back between pairs
    units = max(2, min(5, int((width - 7.0) / UNIT[0])))
    rack_x0 = x0 + 1.4 + UNIT[0] / 2
    y = y1 - 0.15 - UNIT[1] / 2
    aisles: list[float] = []
    n = 0
    for r in range(rows):
        yaw = 0.0 if r % 2 == 0 else 180.0  # open toward -y, then toward +y: the two face the aisle between them
        for k in range(units):
            fp = Footprint(round(rack_x0 + k * UNIT[0], 3), round(y, 3), UNIT[0] / 2, UNIT[1] / 2, yaw)
            n += 1
            name = draft.put(f"rack_{n}", "shelf", fp, list(UNIT), color=STEEL)
            _stock(draft, name, fp, yaw)
        if r % 2 == 0:
            aisles.append(y - UNIT[1] / 2 - AISLE / 2)
            y -= UNIT[1] + AISLE
        else:
            y -= UNIT[1] + 0.1
    lane_x = (rack_x0 - UNIT[0] / 2 + rack_x0 + (units - 0.5) * UNIT[0]) / 2
    lane_len = units * UNIT[0] + 1.0
    for k, ay in enumerate(aisles):
        for side in (1, -1):
            c.add(
                draft.name(f"lane_{k + 1}"),
                "marker",
                [round(lane_x, 2), round(ay + side * (AISLE / 2 - 0.25), 3)],
                size=[round(lane_len, 2), 0.08],
                color=YELLOW,
            )
        draft.keepout.append(Footprint(lane_x, ay, lane_len / 2, AISLE / 2 - 0.3))
    open_top = y + UNIT[1] / 2  # the floor south of the last row is open
    open_area = Footprint((x0 + x1) / 2 - 1.5, (y0 + open_top) / 2, (x1 - x0) / 2 - 2.0, (open_top - y0) / 2 - 0.4)
    # a packing table with a tote on it, where parcels are taken
    table = draft.in_middle(open_area, 1.4, 0.7, gap=0.6)
    if table is not None:
        # low enough that a lift picker (it reaches 0.9 m) puts a parcel down inside the tote over its rim
        name = draft.put("packing_table", "table", table, [1.4, 0.7, PACKING], color=[0.62, 0.48, 0.34, 1])
        top = draft.surface(name, table, PACKING, room=None)
        tote = Footprint(*table.world(0.35, -0.35 + 0.13), 0.16, 0.12, table.yaw)  # near an edge, for an arm
        top.items.append(tote)
        c.add(
            draft.name("tote"),
            "bin",
            [round(tote.x, 3), round(tote.y, 3), PACKING],
            size=[0.32, 0.24, 0.1],
            yaw=table.yaw,
            color="blue",
        )
    # pallets with crates in the open area
    for _ in range(min(max(wish.counts.get("pallet") or rng.randint(2, 4), 1), 8)):
        fp = draft.in_middle(open_area, 1.2, 0.8, gap=0.5)
        if fp is None:
            break
        pallet = draft.put("pallet", "pallet", fp, [1.2, 0.8, 0.144])
        for j in range(rng.randint(1, 4)):
            lx, ly = (-0.3 if j % 2 == 0 else 0.3), (-0.2 if j < 2 else 0.2)
            x, yy = fp.world(lx, ly)
            size = [0.5, 0.36, 0.3]
            c.add(
                draft.name("crate"),
                "box",
                [round(x, 3), round(yy, 3), resting_z(0.144, "box", size)],
                size=size,
                yaw=fp.yaw,
                color=CRATE,
                mass=5.0,
            )
        del pallet
    if "conveyor" in wish.counts:
        belt = draft.in_middle(open_area, 3.0, 0.4, gap=0.6)
        if belt is not None:
            draft.put("conveyor", "conveyor", Footprint(belt.x, belt.y, 1.5, 0.2, 0.0), [3.0, 0.4, 0.75])
    for k in range(min(wish.counts.get("pillar") or 2, 6)):
        spot = draft.in_middle(open_area, 0.4, 0.4, gap=0.8)
        if spot is not None:
            draft.put(f"column_{k + 1}", "pillar", spot, [0.4, 2.4])
    start = draft.slot((x0 + 0.8, aisles[0] if aisles else 0.0))
    if start is None:
        raise ValueError("no free floor for a robot to start on")
    draft.map_floor(start.pos)
    _parcels(draft, wish)
    second = draft.slot((open_area.x, open_area.y))
    draft.content.ground_slots = [start] + ([second] if second else [])


def _stock(draft: Draft, rack: str, fp: Footprint, yaw: float) -> None:
    """A rack's boards: surfaces for parcels at hand height, big crates on the others."""
    levels = props.shelf_levels(list(UNIT))
    for lo, hi in zip(levels, levels[1:], strict=False):
        board = Footprint(*fp.world(0.0, -props.PANEL / 2), UNIT[0] / 2 - props.PANEL, UNIT[1] / 2 - props.PANEL, yaw)
        if LOW <= lo <= HIGH:
            # parcels go at the aisle edge: only the front 20 cm of the board
            front = Footprint(*board.world(0.0, -board.hy + 0.09), board.hx, 0.06, yaw)
            draft.surface(rack, front, lo, clearance=hi - lo - props.PANEL, room=None, inset=0.02, kind="shelf")
            continue
        for lx in (-0.55, 0.55):
            if draft.rng.random() < 0.5:
                continue
            size = [draft.rng.choice((0.4, 0.45, 0.5)), 0.4, round(draft.rng.uniform(0.22, 0.3), 2)]
            x, y = board.world(lx + draft.rng.uniform(-0.1, 0.1), 0.0)
            draft.content.add(
                draft.name("crate"),
                "box",
                [round(x, 3), round(y, 3), resting_z(lo, "box", size)],
                size=size,
                yaw=yaw,
                color=CRATE,
                mass=4.0,
            )


def _parcels(draft: Draft, wish: Wish) -> None:
    asked = {c: n for c, n in wish.counts.items() if c in ITEMS}
    if not asked:
        asked = {"box": draft.rng.randint(3, 5)}
    for concept, n in asked.items():
        color = wish.colors.get(concept)
        if concept == "box" and color is None:  # parcels come in colors here, so a task can name one
            color = draft.rng.choice(things.PALETTE)
        for item in things.several(concept, n or 3, draft.rng, color):
            if draft.place(item, draft.surfaces) is None:
                draft.notes.append(f"no room was left for a {item.concept}")
