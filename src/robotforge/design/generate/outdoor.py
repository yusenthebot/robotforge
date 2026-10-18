"""Generated outdoor places: a park, a forest, a rocky hillside, a beach, with the terrain the sentence names.

The robot starts at the west end on flat ground. Terrain features (stairs, a ramp, stepping stones, a platform)
line up eastward from there along a path, as a course to cross; hills and rough ground cover patches off to the
sides. Trees, rocks and benches stand clear of the path, of each other and of the features; a picnic table carries
things an arm can take, and loose things lie on flat open ground (a ball left on a slope rolls away: the settling
check moves it).
"""

from __future__ import annotations

from robotforge.design.generate import things
from robotforge.design.generate.draft import Draft
from robotforge.design.generate.space import Footprint, Surface
from robotforge.design.generate.words import ITEMS, Wish

WOOD = [0.62, 0.48, 0.34, 1]
PATH = [0.78, 0.7, 0.55, 1]
WATER = [0.25, 0.5, 0.75, 1]
# terrain features on the course, in the order a robot meets them: (type, length along x, params)
COURSE = {
    "ramp": (5.2, {"angle": 10, "length": 1.5, "landing": 0.8, "width": 1.6, "direction": "up_down"}),
    "stairs": (
        4.2,
        {"steps": 3, "step_height": 0.05, "step_depth": 0.3, "width": 1.6, "landing": 0.8, "direction": "up_down"},
    ),
    "stones": (2.6, {"size": [2.0, 1.4], "stone": 0.28, "gap": 0.05, "height": 0.05, "jitter": 0.012}),
    "platform": (1.6, {"size": [1.2, 1.6], "height": 0.1}),
}
FOREST = ("forest", "wood", "woods", "森林", "树林")


def build(draft: Draft, wish: Wish) -> None:
    rng, c = draft.rng, draft.content
    s = wish.size
    width, depth = round(20.0 * s, 1), round(14.0 * s, 1)
    x0, x1, y0, y1 = -width / 2, width / 2, -depth / 2, depth / 2
    draft.bounds = (x0, x1, y0, y1)
    text = wish.text.lower()
    c.floor = {"style": wish.ground or ("sand" if "beach" in text or "沙滩" in text else "grass")}
    start_x = x0 + 1.5
    # the path east from the start, and the course features on it
    course = [t for t in wish.terrain if t in COURSE]
    x = start_x + 2.0
    for t in course:
        length, params = COURSE[t]
        center = x + length / 2
        if center + length / 2 > x1 - 1.0:
            break
        c.terrain.append(
            {
                "type": t,
                "name": draft.name(t),
                "pos": [round(center, 2), 0.0],
                **params,
                **({"seed": rng.randrange(1000)} if t == "stones" else {}),
            }
        )
        draft.keepout.append(Footprint(center, 0.0, length / 2 + 0.2, 1.1))
        x += length + 1.2
    path_end = min(x + 2.0, x1 - 0.5) if course else x1 - 2.0
    c.add(
        "path",
        "marker",
        [round((start_x + path_end) / 2, 2), 0.0],
        size=[round(path_end - start_x + 1.0, 2), 1.4],
        color=PATH,
    )
    draft.keepout.append(Footprint((start_x + path_end) / 2, 0.0, (path_end - start_x) / 2 + 0.5, 0.9))
    c.add("goal", "marker", [round(path_end, 2), 0.0], size=[0.8], color=[0.92, 0.3, 0.28, 1])
    # covering terrain off the path
    if "hills" in wish.terrain:
        c.terrain.append(
            {
                "type": "hills",
                "name": "hills",
                "pos": [round(x0 + width * 0.62, 2), round(y1 - 3.4 * s, 2)],
                "size": [round(width * 0.55, 1), round(5.0 * s, 1)],
                "height": 0.35,
                "scale": 2.0,
                "seed": rng.randrange(1000),
            }
        )
    if "rough" in wish.terrain:
        c.terrain.append(
            {
                "type": "rough",
                "name": "rough",
                "pos": [round(x0 + width * 0.6, 2), round(y0 + 3.0 * s, 2)],
                "size": [round(width * 0.5, 1), round(4.0 * s, 1)],
                "height": 0.04,
                "scale": 0.3,
                "seed": rng.randrange(1000),
            }
        )
    # the start stays clear
    draft.keepout.append(Footprint(start_x, 0.0, 1.0, 1.0))
    area = Footprint(0.0, 0.0, width / 2 - 0.8, depth / 2 - 0.8)
    if any(w in text for w in ("pond", "lake", "池塘", "湖")) or rng.random() < 0.35:
        spot = draft.in_middle(area, 3.2, 3.2, gap=0.5)
        if spot is not None:
            c.add("pond", "marker", [spot.x, spot.y], size=[3.0], color=WATER)
            draft.keepout.append(spot)
    forest = any(w in text for w in FOREST)
    n_trees = wish.counts.get("tree") or (14 if forest else rng.randint(5, 8))
    for _ in range(n_trees):
        crown = round(rng.uniform(1.2, 2.0), 2)
        spot = draft.in_middle(area, crown, crown, gap=0.6)  # the crown clear of the path and the course
        if spot is not None:
            draft.put(
                "tree",
                "tree",
                Footprint(spot.x, spot.y, crown * 0.35, crown * 0.35),
                [crown, round(crown * rng.uniform(1.6, 2.0), 2)],
            )
    rocky = "rough" in wish.terrain
    for _ in range(wish.counts.get("rock") or (7 if rocky else rng.randint(2, 4))):
        d = round(rng.uniform(0.4, 0.9), 2)
        spot = draft.in_middle(area, d, d, gap=0.5)
        if spot is not None:
            draft.put(
                "rock",
                "rock",
                Footprint(spot.x, spot.y, d / 2, d / 2, round(rng.uniform(0, 360), 1)),
                [d, round(d * 0.55, 2)],
            )
    for _ in range(wish.counts.get("bench") or rng.randint(1, 2)):
        spot = draft.in_middle(area, 1.4, 0.45, gap=0.6)
        if spot is not None:
            draft.put("bench", "bench", spot, [1.4, 0.45, 0.45])
    tables = []
    for _ in range(max(1, wish.counts.get("table") or 1)):
        spot = draft.in_middle(area, 1.6, 0.8, gap=0.8)
        if spot is not None:
            name = draft.put("picnic_table", "table", spot, [1.6, 0.8, 0.72], color=WOOD)
            tables.append(draft.surface(name, spot, 0.72))
    for t in tables:
        t.kind = "table"
    start = draft.slot((start_x, 0.0))
    if start is None:
        raise ValueError("no free floor for a robot to start on")
    draft.map_floor(start.pos)
    _things(draft, wish, tables, area)
    draft.content.ground_slots = [start]
    second = draft.slot((start_x, -2.0))
    if second is not None:
        draft.content.ground_slots.append(second)


def _things(draft: Draft, wish: Wish, tables: list[Surface], area: Footprint) -> None:
    asked = {c: n for c, n in wish.counts.items() if c in ITEMS}
    if not asked:
        asked = {"ball": 1, "cup": 1, "box": 1}
    grass = []  # open ground near the path, where things on the ground lie (flat: off the hills and rough patches)
    covered = [
        Footprint(t["pos"][0], t["pos"][1], t["size"][0] / 2 + 0.3, t["size"][1] / 2 + 0.3)
        for t in draft.content.terrain
        if t["type"] in ("hills", "rough")
    ]
    for k in range(4):
        x = area.x - area.hx * 0.6 + k * area.hx * 0.4
        for y in (-2.0, 2.0, -3.5, 3.5):
            spot = Footprint(x, y, 0.8, 0.6)
            if not any(spot.overlaps(c) for c in covered):
                grass.append(Surface("ground", spot, 0.0, kind="ground"))
    for concept, n in asked.items():
        for item in things.several(concept, n or 2, draft.rng, wish.colors.get(concept)):
            on = tables if item.concept in ("cup", "plate", "bowl", "apple", "can", "book") and tables else grass
            if draft.place(item, on) is None and draft.place(item, grass + tables) is None:
                draft.notes.append(f"no room was left for a {item.concept}")
