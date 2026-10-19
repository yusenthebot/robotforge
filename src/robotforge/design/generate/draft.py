"""A world being generated: its scene content plus what the rules need to know while things are placed.

Furniture goes on the floor by footprints that keep a gap from each other, from walls and from door ways; things go
on surfaces (tops of tables, counters, chests, a shelf's boards, pallets) inside the edges, apart from each other, on
a surface low enough for an arm and next to floor a robot can get to. Every placement is drawn from the world's own
random generator, so a seed gives the same world every time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from robotforge.design import props
from robotforge.design.generate.space import CELL, FloorMap, Footprint, Surface
from robotforge.design.scenes import Content, Slot, resting_z

FURNITURE_GAP = 0.12  # m between two pieces of furniture
WALK_GAP = 0.65  # m of floor kept round furniture in the middle of a room, for a robot to pass
WALL_GAP = 0.03  # m between furniture and the wall it stands against
ITEM_GAP = 0.025  # m between two things on a surface
EDGE = 0.035  # m a thing keeps from a surface's edge
ROBOT_RADIUS = 0.36  # m: the floor map is grown by this (a lift picker is 0.55 m across, and turns with its load)
# m over a thing on a shelf's board to the board above: a hand comes down to it from there (a book stood 0.2 m
# tall under a board 0.37 m above its own, and a lift picker's gripper could not get over it)
SHELF_HEADROOM = 0.25
FRONT = 0.7  # m of floor kept clear before furniture against a wall
REACH = 0.7  # m from free floor to a thing on a surface: an arm on a robot standing there reaches it
LOW, HIGH = 0.3, 0.9  # m: surfaces things are put on, for an arm to reach them (not the floor, not up a shelf)
TRIES = 80


@dataclass(frozen=True)
class Item:
    """A thing to put on a surface: what it is called and its shape, size, color and mass."""

    concept: str
    shape: str
    size: tuple[float, ...]
    color: str | list[float]
    mass: float | None = None
    fixed: bool = False
    named: str | None = None  # its name, if not made from its color and concept

    def half(self) -> tuple[float, float]:
        return props.footprint(self.shape, list(self.size))

    def height(self) -> float:
        if self.shape == "sphere":
            return self.size[0]
        if len(self.size) == 2:  # cylinders, buttons, valves: [diameter, height]
            return self.size[1]
        return self.size[2] if len(self.size) == 3 else 0.05


@dataclass
class Draft:
    rng: random.Random
    content: Content
    bounds: tuple[float, float, float, float]  # x0, x1, y0, y1 of the world (walls or edges)
    obstacles: dict[str, Footprint] = field(default_factory=dict)  # fixed things on the floor, by name
    keepout: list[Footprint] = field(default_factory=list)  # door ways, paths, the robots' starts
    walls: set[str] = field(default_factory=set)  # obstacles that are walls (furniture stands against them)
    obstacles_on: dict[str, Footprint] = field(default_factory=dict)  # things fixed on tops, by name
    wall_thick: float = 0.0  # m: rooms' rectangles run along the middle of their walls
    surfaces: list[Surface] = field(default_factory=list)
    rooms: dict[str, Footprint] = field(default_factory=dict)  # room name -> its floor
    room_types: dict[str, str] = field(default_factory=dict)  # room name -> kitchen, bedroom, ...
    notes: list[str] = field(default_factory=list)
    items: dict[str, tuple[Item, Surface | None]] = field(default_factory=dict)  # loose things, where they stand
    floor_map: FloorMap | None = None

    # ---- names ----

    def taken(self) -> set[str]:
        return {o["name"] for o in self.content.objects} | {t["name"] for t in self.content.terrain}

    def name(self, base: str) -> str:
        base = base[:28]
        taken = self.taken()
        if base not in taken:
            return base
        k = 2
        while f"{base}_{k}" in taken:
            k += 1
        return f"{base}_{k}"

    # ---- the floor ----

    def free(self, fp: Footprint, gap: float, ignore: tuple[str, ...] = (), wall_gap: float | None = None) -> bool:
        """Is `fp` inside the world, out of the door ways and paths, and `gap` from all furniture (`wall_gap` from
        walls: furniture stands against them)?"""
        x0, x1, y0, y1 = self.bounds
        bx0, bx1, by0, by1 = fp.bounds()
        if bx0 < x0 or bx1 > x1 or by0 < y0 or by1 > y1:
            return False
        if any(fp.overlaps(k, 0.0) for k in self.keepout):
            return False
        for n, o in self.obstacles.items():
            if n in ignore:
                continue
            if fp.overlaps(o, gap if n not in self.walls or wall_gap is None else wall_gap):
                return False
        return True

    def against_wall(
        self,
        room: Footprint,
        length: float,
        depth: float,
        sides: tuple[str, ...] | None = None,
        front: float = FRONT,
    ) -> tuple[Footprint, str] | None:
        """A spot along one of the room's walls for something `length` wide and `depth` deep, its back to the wall
        and its front (its own -y) to the room: (footprint, side). The `front` m of floor before it stay clear (a
        robot stands there to reach what is on it; a desk put in a corner behind the bed could not be reached)."""
        x0, x1, y0, y1 = room.bounds()
        inner = self.wall_thick / 2
        x0, x1, y0, y1 = x0 + inner, x1 - inner, y0 + inner, y1 - inner
        order = list(sides or ("north", "south", "east", "west"))
        self.rng.shuffle(order)
        for _ in range(TRIES // 4):
            for side in order:
                if side in ("north", "south"):
                    span = (x0 + length / 2 + 0.05, x1 - length / 2 - 0.05)
                    if span[0] > span[1]:
                        continue
                    x = self.rng.uniform(*span)
                    y = y1 - depth / 2 - WALL_GAP if side == "north" else y0 + depth / 2 + WALL_GAP
                    yaw = 0.0 if side == "north" else 180.0
                else:
                    span = (y0 + length / 2 + 0.05, y1 - length / 2 - 0.05)
                    if span[0] > span[1]:
                        continue
                    y = self.rng.uniform(*span)
                    x = x1 - depth / 2 - WALL_GAP if side == "east" else x0 + depth / 2 + WALL_GAP
                    yaw = -90.0 if side == "east" else 90.0
                fp = Footprint(round(x, 3), round(y, 3), length / 2, depth / 2, yaw)
                ahead = Footprint(*fp.world(0.0, -depth / 2 - front / 2), length / 2, front / 2, yaw) if front else None
                if self.free(fp, FURNITURE_GAP, wall_gap=0.0) and (
                    ahead is None or self.free(ahead, 0.0, wall_gap=0.0)
                ):
                    if ahead is not None:
                        self.keepout.append(ahead)
                    return fp, side
        return None

    def in_middle(self, room: Footprint, length: float, depth: float, gap: float = WALK_GAP) -> Footprint | None:
        """A spot in the room's middle, with room to walk round it."""
        x0, x1, y0, y1 = room.bounds()
        inner = self.wall_thick / 2
        x0, x1, y0, y1 = x0 + inner, x1 - inner, y0 + inner, y1 - inner
        for _ in range(TRIES):
            yaw = self.rng.choice((0.0, 90.0)) if abs(length - depth) > 0.05 else 0.0
            hx, hy = (length / 2, depth / 2) if yaw == 0 else (depth / 2, length / 2)
            lo_x, hi_x = x0 + hx + gap, x1 - hx - gap
            lo_y, hi_y = y0 + hy + gap, y1 - hy - gap
            if lo_x > hi_x or lo_y > hi_y:
                return None
            fp = Footprint(
                round(self.rng.uniform(lo_x, hi_x), 3),
                round(self.rng.uniform(lo_y, hi_y), 3),
                length / 2,
                depth / 2,
                yaw,
            )
            if self.free(fp, gap):
                return fp
        return None

    def put(
        self, name: str, shape: str, fp: Footprint, size: list[float], z: float | None = None, **fields: Any
    ) -> str:
        """Add a piece of furniture standing at `fp` (on the floor, or at height z)."""
        name = self.name(name)
        pos = [fp.x, fp.y] if z is None else [fp.x, fp.y, round(z, 4)]
        self.content.add(name, shape, pos, size=[round(s, 3) for s in size], yaw=fp.yaw, **fields)
        if z is None:
            self.obstacles[name] = fp
        return name

    def surface(
        self,
        owner: str,
        fp: Footprint,
        z: float,
        clearance: float = 2.0,
        room: str | None = None,
        inset: float = EDGE,
        kind: str = "table",
    ) -> Surface:
        area = Footprint(fp.x, fp.y, max(fp.hx - inset, 0.01), max(fp.hy - inset, 0.01), fp.yaw)
        s = Surface(owner, area, z, clearance, room, kind=kind)
        self.surfaces.append(s)
        return s

    # ---- things on surfaces ----

    def spot_on(self, surface: Surface, item: Item) -> tuple[float, float, float] | None:
        hx, hy = item.half()
        headroom = SHELF_HEADROOM if surface.kind == "shelf" else 0.02
        if item.height() > surface.clearance - headroom:
            return None
        a = surface.area
        if hx > a.hx or hy > a.hy:
            return None
        for _ in range(24):
            lx, ly = self.rng.uniform(-a.hx + hx, a.hx - hx), self.rng.uniform(-a.hy + hy, a.hy - hy)
            x, y = a.world(lx, ly)
            yaw = round(a.yaw + self.rng.choice((0.0, 0.0, 90.0, self.rng.uniform(-25, 25))), 1)
            fp = Footprint(x, y, hx, hy, yaw if item.shape not in ("sphere", "cylinder") else 0.0)
            # a turned thing's corners must stay on the top too
            if not all(a.contains(cx, cy, -1e-3) for cx, cy in fp.corners()):
                continue
            if any(fp.overlaps(o, ITEM_GAP) for o in surface.items):
                continue
            if surface.z == 0 and not self.free(fp, 0.1):  # on the ground: clear of what stands there
                continue
            if self.floor_map is not None and not self.floor_map.reachable_within(x, y, REACH):
                continue
            return round(x, 3), round(y, 3), fp.yaw
        return None

    def place(self, item: Item, surfaces: list[Surface], name: str | None = None) -> str | None:
        """Put `item` on one of `surfaces` (tried in a random order), or None when none has room."""
        # tops an arm reaches; and the ground outdoors (what lies there is for pushing or walking to)
        order = [s for s in surfaces if LOW <= s.z <= HIGH or s.kind == "ground"]
        self.rng.shuffle(order)
        for surface in order:
            spot = self.spot_on(surface, item)
            if spot is None:
                continue
            x, y, yaw = spot
            hx, hy = item.half()
            surface.items.append(Footprint(x, y, hx, hy, yaw))
            base = name or item.named or item.concept
            name = self.name(base)
            fields: dict[str, Any] = {"size": list(item.size), "color": item.color, "yaw": yaw}
            if item.mass is not None:
                fields["mass"] = item.mass
            if item.fixed:
                fields["fixed"] = True
            # on the floor or the ground a thing rests where it stands (terrain included); on a top it is set on it
            pos = [x, y] if surface.z == 0 else [x, y, round(resting_z(surface.z, item.shape, list(item.size)), 4)]
            self.content.add(name, item.shape, pos, **fields)
            if not item.fixed:
                self.items[name] = (item, surface)
            return name
        return None

    def take_back(self, name: str) -> None:
        """Remove a thing placed earlier (to place it again elsewhere)."""
        item, surface = self.items.pop(name)
        obj = next(o for o in self.content.objects if o["name"] == name)
        self.content.objects.remove(obj)
        if surface is not None:
            x, y = obj["pos"][0], obj["pos"][1]
            surface.items = [f for f in surface.items if math.hypot(f.x - x, f.y - y) > 1e-6]

    # ---- robots ----

    def map_floor(self, start: tuple[float, float]) -> FloorMap:
        self.floor_map = FloorMap.of(self.bounds, list(self.obstacles.values()), ROBOT_RADIUS)
        self.floor_map.flood(*start)
        return self.floor_map

    def open_spot(self, near: tuple[float, float], clear: float = 0.5) -> tuple[float, float] | None:
        """Free floor nearest `near` with `clear` m to anything: a robot's start."""
        roomy = FloorMap.of(self.bounds, list(self.obstacles.values()) + self.keepout_for_starts(), clear)
        best, best_d = None, math.inf
        rows, cols = roomy.blocked.shape
        for r in range(0, rows, 2):
            for c in range(0, cols, 2):
                if roomy.blocked[r, c]:
                    continue
                x, y = roomy.x0 + c * CELL, roomy.y0 + r * CELL
                d = math.hypot(x - near[0], y - near[1])
                if d < best_d and (self.floor_map is None or self.floor_map.reachable(x, y)):
                    best, best_d = (round(x, 2), round(y, 2)), d
        return best

    def keepout_for_starts(self) -> list[Footprint]:
        return []

    def slot(self, near: tuple[float, float], yaw: float = 0.0) -> Slot | None:
        spot = self.open_spot(near)
        return Slot(spot, yaw) if spot else None
