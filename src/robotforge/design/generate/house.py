"""Generated homes and offices: rooms side by side with doorways between them, furnished by what each room is for.

Rooms are laid out in one row, or two rows for four or more, each room sized by its kind and the sentence's "small"
or "large", jittered by the seed. Walls are 1.1 m high, a doll's house the camera looks into; every two rooms that
share a wall have a 1.1 m doorway in it, so a robot gets from any room to any other. Each room's floor carries a flat
colored marker named after the room (no collision): "kitchen", "bedroom_2"; move(robot, to="kitchen") goes there and
a task can say "in the kitchen".
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

from robotforge.design import props
from robotforge.design.generate import things
from robotforge.design.generate.draft import FRONT, WALK_GAP, Draft
from robotforge.design.generate.space import Footprint, Surface
from robotforge.design.generate.words import Wish

WALL_HEIGHT = 1.1  # m: low enough to look into from above, high enough that robots do not see over it
WALL_THICK = 0.1
DOOR = 1.1  # m wide doorways
COUNTER = 0.85  # m, a counter's top: a lift picker reaches things standing on it (it reaches up to 0.9 m)
WALL_COLOR = [0.9, 0.88, 0.84, 1]
WOOD = [0.62, 0.48, 0.34, 1]
LIGHT_WOOD = [0.78, 0.64, 0.46, 1]
DARK = [0.2, 0.21, 0.23, 1]
WHITE = [0.92, 0.92, 0.9, 1]
FABRIC = ([0.36, 0.45, 0.6, 1], [0.55, 0.38, 0.32, 1], [0.42, 0.52, 0.4, 1], [0.6, 0.56, 0.5, 1])
# room kind -> (width, depth) in m, and the floor marker's tint (a hint of color over the wood)
ROOM_SIZES = {
    "kitchen": (4.0, 3.6),
    "living_room": (5.0, 4.0),
    "bedroom": (4.0, 3.6),
    "office": (4.4, 3.8),
    "dining_room": (4.2, 3.6),
    "bathroom": (2.8, 2.6),
    "room": (4.4, 3.8),
}
TINTS = {
    "kitchen": [0.85, 0.83, 0.78, 1],
    "living_room": [0.7, 0.6, 0.48, 1],
    "bedroom": [0.66, 0.68, 0.75, 1],
    "office": [0.62, 0.64, 0.62, 1],
    "dining_room": [0.74, 0.62, 0.5, 1],
    "bathroom": [0.8, 0.86, 0.88, 1],
    "room": [0.72, 0.66, 0.58, 1],
}
# What each room has besides its furniture, when the sentence asks for no things of its own
DEFAULT_ITEMS = {
    "kitchen": ("apple", "cup", "cup", "bowl", "plate", "can"),
    "living_room": ("book", "book", "ball", "cup"),
    "bedroom": ("book", "toy", "ball"),
    "office": ("cup", "book", "box", "can"),
    "dining_room": ("plate", "plate", "cup", "apple", "bowl"),
    "bathroom": ("can", "cup"),
    "room": ("cube", "cube", "ball", "cup"),
}
# Which room a thing or piece of furniture belongs in, for one said without a room
HOME_OF = {
    "fridge": "kitchen",
    "counter": "kitchen",
    "apple": "kitchen",
    "plate": "kitchen",
    "bowl": "kitchen",
    "sofa": "living_room",
    "tv": "living_room",
    "coffee_table": "living_room",
    "bed": "bedroom",
    "toy": "bedroom",
    "desk": "office",
    "chair": "dining_room",
}
MAX_ROOMS = 6


def rooms_wanted(wish: Wish) -> list[str]:
    """The kinds of the rooms to lay out, in order."""
    rooms = list(wish.rooms)
    if not rooms:
        homes = [HOME_OF[c] for c in wish.counts if c in HOME_OF]
        rooms = list(dict.fromkeys(homes))
    filler = ["living_room", "kitchen", "bedroom", "office", "dining_room", "bedroom"]
    while len(rooms) < wish.room_count:
        rooms.append(next((r for r in filler if r not in rooms), "bedroom"))
    if not rooms:
        rooms = ["room"]
    return rooms[:MAX_ROOMS]


def build(draft: Draft, wish: Wish) -> None:
    kinds = rooms_wanted(wish)
    rects = _layout(draft, kinds, wish.size)
    draft.wall_thick = WALL_THICK
    names: list[str] = []
    for kind, rect in zip(kinds, rects, strict=True):
        name = draft.name(kind)
        names.append(name)
        draft.rooms[name] = rect
        draft.room_types[name] = kind
        x0, x1, y0, y1 = rect.bounds()
        # the room's floor, a tint over the wood just inside its walls (a marker: nothing collides with it)
        draft.content.add(
            name, "marker", [rect.x, rect.y], size=[round(x1 - x0 - 0.1, 2), round(y1 - y0 - 0.1, 2)], color=TINTS[kind]
        )
    _walls(draft, list(zip(names, rects, strict=True)))
    asked = {c: n for c, n in wish.counts.items()}
    # furniture said in a room ("the bedroom desk") goes in first, so the room's usual pieces do not leave it no space
    placed = {c: r for c, r in wish.rooms_of.items() if c in FURNITURE_SAID and r in kinds}
    for concept, kind in placed.items():
        _asked_furniture(draft, {concept: asked.get(concept, 0)}, names, kinds, only=kind)
    for name, kind in zip(names, kinds, strict=True):
        FURNISH.get(kind, _room)(draft, name, draft.rooms[name])
    _asked_furniture(draft, {c: n for c, n in asked.items() if c not in placed}, names, kinds)
    # The robot starts in the first room (the hall, as it were), so the floor it can get to is known before things
    # go on surfaces: a thing out of every robot's reach would be a task nobody can do.
    start = draft.slot((draft.rooms[names[0]].x, draft.rooms[names[0]].y))
    if start is None:
        raise ValueError("no free floor for a robot to start on")
    _connect(draft, start.pos, names)
    draft.map_floor(start.pos)
    _things(draft, wish, names, kinds)
    slots = [start]
    for name in names[1:]:
        s = draft.slot((draft.rooms[name].x, draft.rooms[name].y))
        if s is not None:
            slots.append(s)
    draft.content.ground_slots = slots
    draft.content.floor = {"style": "wood"}


def _connect(draft: Draft, start: tuple[float, float], names: list[str]) -> None:
    """Take chairs away until a robot gets from its start into every room: a dining table's chairs closed the way
    across a narrow dining room, and the rooms beyond it could not be reached."""
    from robotforge.design.generate.draft import ROBOT_RADIUS
    from robotforge.design.generate.space import FloorMap

    def shut() -> list[str]:
        floor = FloorMap.of(draft.bounds, list(draft.obstacles.values()), ROBOT_RADIUS)
        floor.flood(*start)
        return [n for n in names if not _reached(floor, draft.rooms[n])]

    closed = shut()
    chairs = [n for n in draft.obstacles if "_chair_" in n]
    while closed and chairs:
        chair = chairs.pop()
        del draft.obstacles[chair]
        draft.content.objects = [o for o in draft.content.objects if o["name"] not in (chair, f"{chair}_back")]
        closed = shut()
    if closed:
        draft.notes.append(f"furniture closes off {', '.join(closed)}: a robot cannot get in from the start")


def _reached(floor: Any, room: Footprint) -> bool:
    x0, x1, y0, y1 = room.bounds()
    for x in [x0 + (x1 - x0) * k / 8 for k in range(1, 8)]:
        for y in [y0 + (y1 - y0) * k / 8 for k in range(1, 8)]:
            if floor.reachable(x, y):
                return True
    return False


# ---- rooms and walls ------------------------------------------------------------------------------------


def _layout(draft: Draft, kinds: list[str], scale: float) -> list[Footprint]:
    """Rectangles for the rooms: one row of up to three, else two rows, the house centered on the origin."""
    rng = draft.rng
    sizes = [ROOM_SIZES[k] for k in kinds]
    sizes = [
        (round(w * scale * rng.uniform(0.92, 1.08), 1), round(d * scale * rng.uniform(0.92, 1.08), 1)) for w, d in sizes
    ]
    per_row = len(kinds) if len(kinds) <= 3 else math.ceil(len(kinds) / 2)
    rows = [sizes[i : i + per_row] for i in range(0, len(sizes), per_row)]
    width = max(sum(w for w, _ in row) for row in rows)
    rects: list[Footprint] = []
    y = 0.0
    for row in rows:
        depth = max(d for _, d in row)
        stretch = width / sum(w for w, _ in row)
        x = 0.0
        for w, _ in row:
            w = round(w * stretch, 2)
            rects.append(Footprint(x + w / 2, y + depth / 2, w / 2, depth / 2))
            x += w
        y += depth
    dx, dy = -width / 2, -y / 2
    rects = [Footprint(round(r.x + dx, 3), round(r.y + dy, 3), r.hx, r.hy) for r in rects]
    draft.bounds = (dx - 0.3, -dx + 0.3, dy - 0.3, -dy + 0.3)
    return rects


def _walls(draft: Draft, rooms: list[tuple[str, Footprint]]) -> None:
    """Outer walls round every room, and a wall with a doorway on every edge two rooms share."""
    segments: list[tuple[str, float, float, float, bool]] = []  # (axis, line, a, b, door)
    for i, (_, r) in enumerate(rooms):
        x0, x1, y0, y1 = r.bounds()
        for axis, line, a, b in (("h", y0, x0, x1), ("h", y1, x0, x1), ("v", x0, y0, y1), ("v", x1, y0, y1)):
            shared: list[tuple[float, float]] = []
            for j, (_, o) in enumerate(rooms):
                if j == i:
                    continue
                ox0, ox1, oy0, oy1 = o.bounds()
                if axis == "h" and (abs(oy0 - line) < 1e-6 or abs(oy1 - line) < 1e-6):
                    lo, hi = max(a, ox0), min(b, ox1)
                elif axis == "v" and (abs(ox0 - line) < 1e-6 or abs(ox1 - line) < 1e-6):
                    lo, hi = max(a, oy0), min(b, oy1)
                else:
                    continue
                if hi - lo > 1e-6:
                    shared.append((lo, hi))
                    if j > i:
                        segments.append((axis, line, lo, hi, hi - lo > DOOR + 0.6))
            # what no other room shares is outside wall
            cuts = sorted(shared)
            at = a
            for lo, hi in cuts:
                if lo - at > 1e-6:
                    segments.append((axis, line, at, lo, False))
                at = max(at, hi)
            if b - at > 1e-6:
                segments.append((axis, line, at, b, False))
    merged = _merge(segments)
    k = 0
    for axis, line, a, b, door in merged:
        pieces = [(a, b)]
        if door:
            mid = draft.rng.uniform(a + 0.3 + DOOR / 2, b - 0.3 - DOOR / 2)
            pieces = [(a, mid - DOOR / 2), (mid + DOOR / 2, b)]
            # the doorway and a stretch of floor either side stay clear of furniture
            way = Footprint(mid, line, DOOR / 2, 0.7) if axis == "h" else Footprint(line, mid, 0.7, DOOR / 2)
            draft.keepout.append(way)
        for lo, hi in pieces:
            # walls overlap at the corners by half their thickness, so the corners are closed
            length = hi - lo + (WALL_THICK if not door else 0.0)
            if length < 0.05:
                continue
            k += 1
            center = (lo + hi) / 2
            if axis == "h":
                fp = Footprint(round(center, 3), round(line, 3), length / 2, WALL_THICK / 2, 0.0)
            else:
                fp = Footprint(round(line, 3), round(center, 3), length / 2, WALL_THICK / 2, 90.0)
            draft.walls.add(draft.put(f"wall_{k}", "wall", fp, [length, WALL_THICK, WALL_HEIGHT], color=WALL_COLOR))


def _merge(segments: list[tuple[str, float, float, float, bool]]) -> list[tuple[str, float, float, float, bool]]:
    """Outer wall pieces on one line that touch, as one wall (two rooms' south walls in a row are one wall)."""
    out: list[tuple[str, float, float, float, bool]] = []
    plain = sorted((s for s in segments if not s[4]), key=lambda s: (s[0], round(s[1], 4), s[2]))
    for s in plain:
        if out and out[-1][0] == s[0] and abs(out[-1][1] - s[1]) < 1e-6 and not out[-1][4] and s[2] - out[-1][3] < 1e-6:
            out[-1] = (s[0], s[1], out[-1][2], max(out[-1][3], s[3]), False)
        else:
            out.append(s)
    return out + [s for s in segments if s[4]]


# ---- furniture --------------------------------------------------------------------------------------------


def _top(
    draft: Draft, name: str, fp: Footprint, z: float, room: str, inset: float = 0.04, kind: str = "table"
) -> Surface:
    return draft.surface(name, fp, z, room=room, inset=inset, kind=kind)


def _table(
    draft: Draft,
    room: str,
    rect: Footprint,
    size: tuple[float, float, float],
    wall: bool = False,
    name: str = "table",
    color: list[float] | None = None,
    gap: float = WALK_GAP,
    front: float = FRONT,
) -> str | None:
    length, depth, height = size
    if wall:
        spot = draft.against_wall(rect, length, depth, front=front)
        fp = spot[0] if spot else None
    else:
        fp = draft.in_middle(rect, length, depth, gap)
    if fp is None:
        return None
    full = f"{room}_{name}" if not name.startswith(room) else name
    placed = draft.put(full, "table", fp, [length, depth, height], color=color or draft.rng.choice((WOOD, LIGHT_WOOD)))
    _top(draft, placed, fp, height, room)
    return placed


def _counter(
    draft: Draft,
    room: str,
    rect: Footprint,
    length: float,
    depth: float = 0.6,
    height: float = COUNTER,
    name: str = "counter",
    sides: tuple[str, ...] | None = None,
) -> tuple[str, Footprint] | None:
    spot = draft.against_wall(rect, length, depth, sides)
    if spot is None:
        return None
    fp, _ = spot
    color = draft.rng.choice(([0.3, 0.38, 0.47, 1], [0.55, 0.42, 0.32, 1], [0.86, 0.85, 0.82, 1]))
    placed = draft.put(f"{room}_{name}", "counter", fp, [length, depth, height], color=color)
    return placed, fp


def _shelf(
    draft: Draft, room: str, rect: Footprint, length: float = 1.0, depth: float = 0.35, height: float = 1.6
) -> str | None:
    spot = draft.against_wall(rect, length, depth)
    if spot is None:
        return None
    fp, _ = spot
    size = [length, depth, height]
    placed = draft.put(f"{room}_shelf", "shelf", fp, size, color=draft.rng.choice((WOOD, LIGHT_WOOD)))
    levels = props.shelf_levels(size)
    for lo, hi in zip(levels, levels[1:], strict=False):
        board = Footprint(*fp.world(0.0, -props.PANEL / 2), length / 2 - props.PANEL, depth / 2 - props.PANEL, fp.yaw)
        draft.surface(placed, board, lo, clearance=hi - lo - props.PANEL, room=room, inset=0.03, kind="shelf")
    return placed


def _chest(
    draft: Draft,
    room: str,
    rect: Footprint,
    size: tuple[float, float, float],
    name: str = "drawer",
    near: Footprint | None = None,
) -> str | None:
    """A chest with a drawer, standing on the floor against a wall (a nightstand, a filing chest)."""
    spot = draft.against_wall(rect, size[0], size[1])
    if spot is None:
        return None
    fp, _ = spot
    placed = draft.put(f"{room}_{name}", "drawer", fp, list(size), color=draft.rng.choice((WOOD, [0.92, 0.92, 0.9, 1])))
    _top(draft, placed, fp, size[2], room, inset=0.03, kind="chest")
    return placed


def _on_top(
    draft: Draft, room: str, surface: Surface, shape: str, size: list[float], name: str, **fields
) -> str | None:
    """A fixed thing standing on a surface (a drawer chest on a counter, a TV on its stand), facing the way the
    surface's owner faces."""
    a = surface.area
    hx, hy = props.footprint(shape, size)
    if hx > a.hx or hy > a.hy:
        return None
    for _ in range(30):
        lx = draft.rng.uniform(-a.hx + hx, a.hx - hx)
        # a TV at the back; a mechanism near the front edge, with room beside it on the top for an arm to work it
        ly = a.hy - hy if shape == "box" else min(-a.hy + hy + 0.08, 0.0)
        x, y = a.world(lx, ly)
        fp = Footprint(round(x, 3), round(y, 3), hx, hy, a.yaw)
        if any(fp.overlaps(o, 0.03) for o in surface.items):
            continue
        keep = []
        if shape != "box":  # a mechanism: the top beside it kept clear, where an arm stands to work it
            # a cabinet's handle is on its right (its door hinges on the left and sweeps the front): there only
            towards = (1,) if shape == "cabinet" else (1, -1)
            sides = [Footprint(*a.world(lx + side * (hx + 0.2), 0.0), 0.16, a.hy, a.yaw) for side in towards]
            keep = [z for z in sides if all(a.contains(cx, cy, -1e-3) for cx, cy in z.corners())
                    and not any(z.overlaps(o, 0.0) for o in surface.items)][:1]  # fmt: skip
            if not keep:
                continue
        surface.items += [fp, *keep]
        placed = draft.put(f"{room}_{name}", shape, fp, size, z=surface.z, **fields)
        return placed
    return None


def _kitchen(draft: Draft, room: str, rect: Footprint) -> None:
    x0, x1, y0, y1 = rect.bounds()
    counter = _counter(draft, room, rect, min(2.4, max(x1 - x0, y1 - y0) - 1.3))
    if counter is not None:
        name, fp = counter
        top = _top(draft, name, fp, COUNTER, room, kind="counter")
        _counter_chest(draft, room, top)
    spot = draft.against_wall(rect, 0.7, 0.7)
    if spot is not None:
        fp, _ = spot
        fridge = draft.put(f"{room}_fridge", "box", fp, [0.7, 0.7, 1.8], z=0.9, color=[0.94, 0.94, 0.95, 1], fixed=True)
        draft.obstacles[fridge] = fp
    _table(draft, room, rect, (1.2, 0.8, 0.75))


CHEST = (0.3, 0.3, 0.14)  # a drawer chest on a kitchen counter
ARM_ROOM = 0.5  # m of counter kept clear in front of it: an arm stands there to open it


def _counter_chest(draft: Draft, room: str, top: Surface) -> str | None:
    """A drawer chest at one end of a counter (see _at_end), its top a surface too."""
    name = _at_end(draft, room, top, "drawer", list(CHEST), "drawer", [0.92, 0.92, 0.9, 1])
    if name is not None:
        fp = draft.obstacles_on[name]
        draft.surface(name, fp, top.z + CHEST[2], room=room, inset=0.03, kind="chest")
    return name


def _at_end(draft: Draft, room: str, top: Surface, shape: str, size: list[float], name: str, color: Any) -> str | None:
    """A drawer chest or a cabinet at one end of a top, its front (where its drawer or door comes out) toward the
    top's middle, with the top in front of it kept clear for an arm to stand and work it (as on the kitchen scene's
    island, where the arm faces the drawer from 0.5 m)."""
    a = top.area
    length, depth = size[0], size[1]
    if a.hx < depth / 2 + ARM_ROOM + 0.05 or a.hy < length / 2:
        return None
    ends = [1, -1]
    draft.rng.shuffle(ends)
    for end in ends:
        lx = end * (a.hx - depth / 2 - 0.02)
        x, y = a.world(lx, 0.0)
        yaw = a.yaw + (-90.0 if end > 0 else 90.0)  # its own -y toward the top's middle
        fp = Footprint(round(x, 3), round(y, 3), length / 2, depth / 2, yaw)
        clear = Footprint(*a.world(lx - end * (depth / 2 + ARM_ROOM / 2), 0.0), ARM_ROOM / 2, a.hy, a.yaw)
        if any(fp.overlaps(o, 0.02) or clear.overlaps(o, 0.0) for o in top.items):
            continue
        top.items += [fp, clear]
        placed = draft.put(f"{room}_{name}", shape, fp, size, z=top.z, color=color)
        draft.obstacles_on[placed] = fp
        return placed
    return None


def _living_room(draft: Draft, room: str, rect: Footprint) -> None:
    _sofa(draft, room, rect)
    if not any(o["name"].startswith(f"{room}_coffee_table") for o in draft.content.objects):  # one said goes first
        _table(draft, room, rect, (1.0, 0.55, 0.42), name="coffee_table")
    stand = _counter(draft, room, rect, 1.4, 0.45, 0.5, name="tv_stand")
    if stand is not None:
        name, fp = stand
        top = _top(draft, name, fp, 0.5, room)
        _on_top(draft, room, top, "box", [1.0, 0.06, 0.58], "tv", color=DARK, fixed=True)
    _shelf(draft, room, rect)


def _bedroom(draft: Draft, room: str, rect: Footprint) -> None:
    _bed(draft, room, rect)
    _bedroom_desk(draft, room, rect)  # before the nightstand, which takes the floor a small room's desk needs
    _chest(draft, room, rect, (0.45, 0.4, 0.5), name="nightstand")
    if draft.rng.random() < 0.6:
        _shelf(draft, room, rect, 0.8, 0.32, 1.2)


DESK_SIZES = (  # (length, depth, height) m, against a wall?, walking gap, floor kept before it: roomiest first
    ((1.0, 0.55, 0.75), True, WALK_GAP, FRONT),
    ((0.8, 0.5, 0.75), True, WALK_GAP, FRONT),
    ((0.8, 0.5, 0.75), False, WALK_GAP, FRONT),
    ((0.8, 0.5, 0.75), True, WALK_GAP * 0.6, 0.5),
    ((0.7, 0.45, 0.75), True, WALK_GAP * 0.6, 0.45),
    ((0.7, 0.45, 0.75), False, WALK_GAP * 0.6, 0.45),
)


def put_desk(draft: Draft, room: str, rect: Footprint, name: str = "desk") -> str | None:
    """A desk (or a table of that size) in a room, against a wall, then smaller, then standing free; None when the room
    has no floor for any."""
    for size, wall, gap, front in DESK_SIZES:
        placed = _table(draft, room, rect, size, wall=wall, name=name, gap=gap, front=front)
        if placed is not None:
            return placed
    return None


def _bedroom_desk(draft: Draft, room: str, rect: Footprint) -> None:
    """A bedroom always gets a desk ("bring the cup to the bedroom desk" is the beginner's sentence)."""
    put_desk(draft, room, rect)


def _office(draft: Draft, room: str, rect: Footprint) -> None:
    for k in range(2):
        _table(draft, room, rect, (1.2, 0.65, 0.75), wall=k == 0, name="desk")
    _shelf(draft, room, rect)
    _chest(draft, room, rect, (0.45, 0.5, 0.65), name="drawer")
    spot = draft.in_middle(rect, 0.28, 0.28, gap=0.3)
    if spot is not None:
        draft.put(f"{room}_bin", "bin", spot, [0.28, 0.28, 0.3], color="gray", fixed=True)


def _dining_room(draft: Draft, room: str, rect: Footprint) -> None:
    # room round it for the chairs and, past them, to walk by
    table = _table(draft, room, rect, (1.6, 0.9, 0.75), gap=WALK_GAP + 0.5)
    if table is not None:
        _chairs(draft, room, draft.obstacles[table])
    _counter(draft, room, rect, 1.4, 0.45, 0.85, name="sideboard")
    sideboard = draft.obstacles.get(f"{room}_sideboard")
    if sideboard is not None:
        _top(draft, f"{room}_sideboard", sideboard, 0.85, room)


def _bathroom(draft: Draft, room: str, rect: Footprint) -> None:
    vanity = _counter(draft, room, rect, 0.9, 0.5, 0.85, name="vanity")
    if vanity is not None:
        _top(draft, vanity[0], vanity[1], 0.85, room)
    spot = draft.in_middle(rect, 0.25, 0.25, gap=0.3)
    if spot is not None:
        draft.put(f"{room}_bin", "bin", spot, [0.25, 0.25, 0.28], color="white", fixed=True)


def _room(draft: Draft, room: str, rect: Footprint) -> None:
    _table(draft, room, rect, (1.2, 0.7, 0.75))
    _shelf(draft, room, rect)
    _table(draft, room, rect, (0.9, 0.55, 0.75), wall=True, name="side_table")


def _sofa(draft: Draft, room: str, rect: Footprint) -> None:
    spot = draft.against_wall(rect, 1.9, 0.85)
    if spot is None:
        return
    fp, _ = spot
    color = draft.rng.choice(FABRIC)
    name = draft.name(f"{room}_sofa")
    draft.obstacles[name] = fp
    seat = Footprint(*fp.world(0.0, -0.08), 0.8, 0.34, fp.yaw)
    draft.content.add(name, "box", [seat.x, seat.y, 0.21], size=[1.6, 0.68, 0.42], yaw=fp.yaw, color=color, fixed=True)
    back = Footprint(*fp.world(0.0, 0.34), 0.95, 0.09, fp.yaw)
    shade = [c * 0.85 for c in color[:3]] + [1]
    draft.content.add(
        f"{name}_back", "box", [back.x, back.y, 0.38], size=[1.9, 0.18, 0.76], yaw=fp.yaw, color=shade, fixed=True
    )
    for k, side in enumerate((-1, 1)):
        arm = Footprint(*fp.world(side * 0.875, -0.08), 0.075, 0.34, fp.yaw)
        draft.content.add(
            f"{name}_arm_{k + 1}",
            "box",
            [arm.x, arm.y, 0.29],
            size=[0.15, 0.68, 0.58],
            yaw=fp.yaw,
            color=shade,
            fixed=True,
        )


def _bed(draft: Draft, room: str, rect: Footprint) -> None:
    """A bed with its head to a wall: a frame, a mattress and a pillow (fixed boxes), and its top as a surface."""
    width, length = 1.4, 2.0
    spot = draft.against_wall(rect, width, length)
    if spot is None:
        return
    fp, _ = spot
    name = draft.name(f"{room}_bed")
    draft.obstacles[name] = fp
    draft.content.add(name, "box", [fp.x, fp.y, 0.15], size=[width, length, 0.3], yaw=fp.yaw, color=WOOD, fixed=True)
    draft.content.add(
        f"{name}_mattress",
        "box",
        [fp.x, fp.y, 0.38],
        size=[width - 0.06, length - 0.06, 0.16],
        yaw=fp.yaw,
        color=draft.rng.choice(FABRIC),
        fixed=True,
    )
    pillow = Footprint(*fp.world(0.0, length / 2 - 0.25), 0.5, 0.17, fp.yaw)
    draft.content.add(
        f"{name}_pillow",
        "box",
        [pillow.x, pillow.y, 0.5],
        size=[1.0, 0.34, 0.08],
        yaw=fp.yaw,
        color=[0.94, 0.94, 0.92, 1],
        fixed=True,
    )
    foot = Footprint(*fp.world(0.0, -0.25), width / 2 - 0.12, length / 2 - 0.55, fp.yaw)
    draft.surface(name, foot, 0.46, room=room, inset=0.0, kind="bed")


def _chairs(draft: Draft, room: str, table: Footprint) -> None:
    """Chairs round a table, pulled out a little, each a seat and a back (fixed boxes)."""
    sides = [(0.0, -1), (0.0, 1)] + ([(-1, 0.0), (1, 0.0)] if table.hx > 0.7 else [])
    k = 0
    for sx, sy in sides:
        along = (-0.4, 0.4) if sy and table.hx > 0.7 else (0.0,)
        for a in along:
            lx = sx * (table.hx + 0.3) if sx else a
            ly = sy * (table.hy + 0.3) if sy else 0.0
            x, y = table.world(lx, ly)
            facing = table.yaw + (0.0 if sy == 1 else 180.0 if sy == -1 else -90.0 if sx == 1 else 90.0)
            fp = Footprint(round(x, 3), round(y, 3), 0.22, 0.22, facing)
            if not draft.free(fp, 0.05, ignore=()):
                continue
            k += 1
            name = draft.name(f"{room}_chair_{k}")
            draft.obstacles[name] = fp
            draft.content.add(
                name, "box", [fp.x, fp.y, 0.225], size=[0.44, 0.44, 0.45], yaw=facing, color=WOOD, fixed=True
            )
            back = Footprint(*fp.world(0.0, 0.2), 0.22, 0.03, facing)
            draft.content.add(
                f"{name}_back",
                "box",
                [back.x, back.y, 0.68],
                size=[0.44, 0.05, 0.46],
                yaw=facing,
                color=WOOD,
                fixed=True,
            )


FURNISH: dict[str, Callable[[Draft, str, Footprint], None]] = {
    "kitchen": _kitchen,
    "living_room": _living_room,
    "bedroom": _bedroom,
    "office": _office,
    "dining_room": _dining_room,
    "bathroom": _bathroom,
    "room": _room,
}


# what stands on the floor and may be said with a room (a drawer is the room's own chest, on a counter in a kitchen)
FURNITURE_SAID = ("table", "coffee_table", "desk", "shelf", "counter", "sofa", "bed")


def _asked_furniture(
    draft: Draft, asked: dict[str, int], names: list[str], kinds: list[str], only: str | None = None
) -> None:
    """Furniture the sentence asked for by number ("four desks"), added to the rooms it belongs in until there are
    that many, and mechanisms ("a button", "a cabinet") put where an arm can work them."""

    def count(concept: str) -> int:
        return sum(1 for o in draft.content.objects if re.sub(r"_\d+$", "", o["name"]).endswith(concept))

    for concept, n in asked.items():
        want = max(n, 1)
        rooms = [nm for nm, k in zip(names, kinds, strict=True) if k == only] if only else []
        rooms = rooms or [nm for nm, k in zip(names, kinds, strict=True) if HOME_OF.get(concept) == k] or names
        made = count(concept)
        k = 0
        while made < want and k < 3 * want:
            room = rooms[k % len(rooms)]
            rect = draft.rooms[room]
            k += 1
            if concept in ("table", "desk"):
                done = _table(draft, room, rect, (1.2, 0.65, 0.75), wall=draft.rng.random() < 0.5, name=concept)
            elif concept == "coffee_table":
                done = _table(draft, room, rect, (1.0, 0.55, 0.42), name="coffee_table")
            elif concept == "shelf":
                done = _shelf(draft, room, rect)
            elif concept == "counter":
                done = _counter(draft, room, rect, 1.6)
                if done:
                    _top(draft, done[0], done[1], COUNTER, room)
            elif concept == "drawer":
                done = _chest(draft, room, rect, (0.45, 0.45, 0.6))
            elif concept == "sofa":
                before = len(draft.content.objects)
                _sofa(draft, room, rect)
                done = len(draft.content.objects) > before
            elif concept == "bed":
                before = len(draft.content.objects)
                _bed(draft, room, rect)
                done = len(draft.content.objects) > before
            elif concept in ("cabinet", "button", "switch", "lever", "valve"):
                done = _mechanism(draft, concept, room)
            else:
                break
            if done:
                made += 1


MECHANISM_SIZES = {
    "cabinet": [0.3, 0.24, 0.2],
    "button": [0.04, 0.05],
    "switch": [0.06, 0.04, 0.03],
    "lever": [0.15],
    "valve": [0.16, 0.22],
}


def _mechanism(draft: Draft, concept: str, room: str) -> bool:
    """A cabinet, button, switch, lever or valve on a top in `room` 0.7 to 1.0 m high, at its front edge."""
    tops = [s for s in draft.surfaces if s.room == room and 0.7 <= s.z <= 1.0 and s.clearance > 0.5]
    draft.rng.shuffle(tops)
    if concept == "cabinet":  # at a top's end facing along it, like a drawer chest: its door opens to the arm
        return any(_at_end(draft, room, t, "cabinet", MECHANISM_SIZES["cabinet"], "cabinet", WHITE) for t in tops)
    return any(_on_top(draft, room, top, concept, MECHANISM_SIZES[concept], concept, color="red") for top in tops)


def _put_on(draft: Draft, wish: Wish, item: Any, names: list[str], kinds: list[str]) -> bool:
    """Put a thing on the furniture the sentence said it is on (in the furniture's room, if one was said); False, and a
    note, when there is none with room on top."""
    furniture = wish.on_of[item.concept]
    room_kind = wish.rooms_of.get(furniture)
    rooms = {n for n, k in zip(names, kinds, strict=True) if room_kind is None or k == room_kind}
    tops = [
        s
        for s in draft.surfaces
        if s.room in rooms
        and re.sub(r"_\d+$", "", s.owner).endswith(furniture)
        and s.kind in ("table", "counter", "shelf")
    ]
    if tops and draft.place(item, tops) is not None:
        return True
    where = f"{room_kind.replace('_', ' ')} " if room_kind else ""
    draft.notes.append(f"no room on the {where}{furniture.replace('_', ' ')} for the {item.concept}: put elsewhere")
    return False


def _things(draft: Draft, wish: Wish, names: list[str], kinds: list[str]) -> None:
    """The small things: those the sentence asked for, in the rooms they belong in, and each room's usual ones."""
    from robotforge.design.generate.words import ITEMS

    asked = {c: n for c, n in wish.counts.items() if c in ITEMS}
    by_room: dict[str, list] = {name: [] for name in names}
    for concept, n in asked.items():
        said = wish.rooms_of.get(concept)  # "the red cup in the kitchen" is in the kitchen
        home = [nm for nm, k in zip(names, kinds, strict=True) if k == said]
        home = home or [nm for nm, k in zip(names, kinds, strict=True) if HOME_OF.get(concept) == k] or names
        for k, item in enumerate(things.several(concept, n or 2, draft.rng, wish.colors.get(concept))):
            by_room[home[k % len(home)]].append(item)
    used: dict[str, set[str]] = {}  # concept -> colors given so far: names stay apart ("red_cup", "blue_cup")
    for items in by_room.values():
        for item in items:
            used.setdefault(item.concept, set()).add(str(item.color))
    for name, kind in zip(names, kinds, strict=True):
        usual = [c for c in DEFAULT_ITEMS[kind] if c not in asked]  # "three apples" are three, not four
        if asked:
            usual = usual[: max(1, len(usual) // 2)]
        for concept in usual:
            taken = used.setdefault(concept, set())
            item = things.make(concept, draft.rng)
            if concept in things.COLORED and str(item.color) in taken:
                free = [c for c in things.PALETTE if c not in taken]
                if not free:
                    continue
                item = things.make(concept, draft.rng, draft.rng.choice(free))
            taken.add(str(item.color))
            by_room[name].append(item)
    for name, items in by_room.items():
        tops = [s for s in draft.surfaces if s.room == name]
        for item in [i for i in items if i.concept in wish.on_of]:  # "an apple on the coffee table" goes first
            items.remove(item)
            if not _put_on(draft, wish, item, names, kinds):
                items.append(item)
        for item in items:
            if draft.place(item, tops) is None and draft.place(item, draft.surfaces) is None:
                draft.notes.append(f"no room was left for a {item.concept}")
