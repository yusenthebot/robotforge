"""Trips through the air for move(to=...): where an aircraft is asked to go, the way there, and how it went.

Heights are world heights: z 0 is the ground at the world's zero, not the ground under the aircraft, and a point
[x, y, z] is where its center goes. A point at or just above the first surface under it is a landing on that
surface (z 0 on the grass, z 0.75 on a 0.75 m table), and so is a point inside something a little below its top
(z 0 on a hill or a half-buried rock). A point under a tree's crown or a table is a place to hover there, not a
landing on top of what is over it. Only the last point can be a landing.

The way there is straight legs. A leg that would run into something goes over it or round it at the same height,
whichever is shorter and clear all the way. Every leg is checked, the climb at the start and the way down at the
end too (a drone sent to hover beside a tree's trunk once climbed over the tree and came down through its crown),
two ways: the aircraft's own shape (a disc as wide as its rotors reach, as tall as it is) swept along the leg
against the boxes round every solid thing, which no table leg slips through, and rays for the ground, hills and
ramps, whose boxes would fill the air over them. When nothing is clear it says what is in the way, and the caller
adds points around it.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.flight import (
    DOWN,
    SOLID,
    TAKEOFF_HEIGHT,
    UP,
    FlightController,
    FlightError,
    Mission,
    cast,
    ground_height,
)
from robotforge.behaviors.navigate import Destination, NoWay

MAX_POINTS = 50
MAX_ALTITUDE = 30.0
GROUND_PROBE = 50.0  # m above which the top of what stands at a point is looked for
CLEARANCE = 0.6  # m kept above what it climbs over
SIDE_CLEARANCE = 0.4  # m kept beside what it flies round, past the reach of its rotors
ROOM = 0.05  # m kept round its shape when a leg is swept against the boxes of things
LAND_MARGIN = 0.05  # m over a surface (plus its own rest height) that still counts as on it
BURIED = 0.5  # m: a point no deeper than this inside something is a landing on its top (a rock, a hill)
UNDER = 0.15  # m kept between its top and an overhang (a crown, a table) it flies under
COLUMN_STOP = 0.15  # m above the touchdown the way down is checked to (a slope beside the spot is not in the way)
TOUCHDOWN_TOL = 0.15  # m: a landing further than this from where it was planned did not land there
MAX_SLOPE = 25.0  # degrees of ground it lands on (its skids tip over at about 38)
TOO_WIDE = 15.0  # m: something wider than this is flown over, never round (and seen by rays, not its box)
SWEEP_STEP = 0.05  # m between the places along a leg its shape is checked at


def parse_points(text: str) -> list[Any] | None:
    """Points written as text ("[[0, 0, 2], [3, 0, 2]]"), as some agents send them, or None."""
    for read in (json.loads, ast.literal_eval):
        try:
            value = read(text)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(value, (list, tuple)):
            return [list(p) if isinstance(p, (list, tuple)) else p for p in value]
    return None


# ---- the air round the aircraft --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Thing:
    """Something an aircraft can run into: its name (an object, a robot, a terrain patch), its top (m) and its
    shapes."""

    name: str
    top: float
    geoms: tuple[int, ...]


@dataclass(frozen=True)
class Spot:
    """What is at a point in the air: the surface under it (m, None over nothing), or, when the point is inside
    something, what (`inside`), where the point comes out of it going up (`exit`) and the top of everything
    there (`top`: a crown's spheres leave gaps, so they can differ)."""

    surface: float | None
    inside: Thing | None = None
    exit: float | None = None
    top: float | None = None


class Airspace:
    """The world as one aircraft flies through it, as it is now: what is at a point, and what a straight leg runs
    into. The boxes of the solid things are measured once, for every leg of a plan."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, fc: FlightController) -> None:
        from robotforge.behaviors.nav_map import world_extent

        self.model, self.data, self.fc = model, data, fc
        frame = fc.frame
        self.own, self.root = frame.descendants, frame.root
        self.radius, self.rest, self.height = frame.radius, frame.rest_height, frame.height
        flat = (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)
        geoms, lows, highs = [], [], []
        for g in range(model.ngeom):
            group = int(model.geom_group[g])
            if int(model.geom_bodyid[g]) in self.own or int(model.geom_type[g]) in flat:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]) or group >= len(SOLID) or not SOLID[group]:
                continue
            if int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX and np.abs(data.geom_xmat[g][6:]).max() < 0.999:
                continue  # a tilted slab (a ramp): its box would fill the air over the slope; rays see it
            lo, hi = world_extent(model, data, g)
            if np.any(hi - lo > TOO_WIDE):
                continue
            geoms.append(g)
            lows.append(lo)
            highs.append(hi)
        self._geoms = np.array(geoms, dtype=int)
        self._low = np.array(lows, float).reshape(-1, 3)
        self._high = np.array(highs, float).reshape(-1, 3)

    # ---- what is at a point ----

    def spot(self, point: np.ndarray) -> Spot:
        """Whether `point` is out in the air (and over what) or inside something (and how far under its top)."""
        m, d, own, root = self.model, self.data, self.own, self.root
        p = np.asarray(point, float)
        at, inside, hit = p.copy(), None, None
        for _ in range(6):  # a ray up leaves what the point is in through its top (a rock's lumps overlap: go on)
            hit = cast(m, d, at, UP, own, root)
            if hit is None or hit.normal[2] <= 0.0:  # the open sky, or the underside of a crown or a table
                break
            inside = inside or self.thing(hit.geom)
            at = at + UP * (hit.distance + 1e-4)
        sky = np.array([p[0], p[1], GROUND_PROBE])
        top = ground_height(m, d, sky, own, root)
        if inside is not None:
            return Spot(None, inside, float(at[2]), top)
        if hit is None and top is not None and top > p[2] + 1e-6:  # under a height field's surface, or the floor's
            under = cast(m, d, sky, DOWN, own, root)
            return Spot(None, self.thing(under.geom) if under else None, top, top)
        return Spot(ground_height(m, d, p + UP * 1e-3, own, root))

    def crowding(self, point: np.ndarray) -> Thing | None:
        """What the aircraft would touch with its center at `point` (resting on a surface is not touching)."""
        found = self._touching(np.asarray(point, float)[None, :])
        return None if found is None else self.thing(int(self._geoms[found[1]]))

    def thing(self, geom: int) -> Thing:
        """What a geom belongs to (an object, a robot, a terrain patch), the top of that thing and its shapes."""
        from robotforge.behaviors.nav_map import world_extent

        m = self.model
        body = int(m.geom_bodyid[geom])
        if body > 0:
            top_body = int(m.body_rootid[body])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, top_body) or "something"
            geoms = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == top_body]
        else:
            stem = _stem(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
            name = stem or "the ground"
            geoms = [
                g
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) == 0 and _stem(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == stem
            ] or [geom]
        top = max(float(world_extent(m, self.data, g)[1][2]) for g in geoms)
        return Thing(name.split("/", 1)[0], top, tuple(geoms))

    # ---- what a straight leg runs into ----

    def first_hit(self, a: np.ndarray, b: np.ndarray) -> tuple[float, Thing] | None:
        """What a straight flight from a to b runs into first (how far along, m, and what), or None."""
        d = b - a
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return None
        u = d / length
        best: tuple[float, int] | None = None
        steps = max(1, math.ceil(length / SWEEP_STEP))
        along = np.linspace(0.0, length, steps + 1)
        found = self._touching(a + np.outer(along, u), leaving=True)
        if found is not None:
            best = (float(along[found[0]]), int(self._geoms[found[1]]))
        # A ray meets a thing when the aircraft's center would; its shape reaches this much further that way.
        reach = (
            float(np.hypot(u[0], u[1])) * (self.radius + ROOM)
            + max(float(u[2]), 0.0) * (self.height + ROOM)
            + max(-float(u[2]), 0.0) * self.rest
        )
        side = np.cross(u, [0.0, 0.0, 1.0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
        lift = np.cross(side, u)
        r = self.radius
        for offset in (np.zeros(3), side * r, -side * r, lift * 0.5 * r, -lift * 0.5 * r):  # the ground and hills
            hit = cast(self.model, self.data, a + offset, u, self.own, self.root, length)
            if hit is not None and (best is None or hit.distance - reach < best[0]):
                best = (max(0.0, hit.distance - reach), hit.geom)
        return None if best is None else (best[0], self.thing(best[1]))

    def _touching(self, centers: np.ndarray, leaving: bool = False) -> tuple[int, int] | None:
        """The first of `centers` (k x 3) at which the aircraft's shape, a disc of its radius from its rest height
        below its center to its height above, overlaps a solid thing's box, and which box; or None. `leaving`:
        boxes it overlaps at the first center are left out (it is there already, next to a bench it took off
        beside, and can fly away from them)."""
        if not len(self._geoms):
            return None
        bottom = centers[:, None, 2] - self.rest + 0.02  # standing on a table top is not running into it
        top = centers[:, None, 2] + self.height + ROOM
        level = (bottom < self._high[None, :, 2]) & (top > self._low[None, :, 2])
        gap = np.maximum(self._low[None, :, :2] - centers[:, None, :2], 0.0) + np.maximum(
            centers[:, None, :2] - self._high[None, :, :2], 0.0
        )
        near = np.hypot(gap[..., 0], gap[..., 1]) < self.radius + ROOM
        touching = level & near
        if leaving:
            touching[:, touching[0]] = False
        hits = np.argwhere(touching)
        if not len(hits):
            return None
        first = hits[np.argmin(hits[:, 0])]
        return int(first[0]), int(first[1])


def _stem(geom_name: str) -> str:
    """The object a static geom belongs to: "tree_2/3" and "tree_2" are tree_2's, "shed_wall1" is shed's."""
    name = geom_name.split("/", 1)[0]
    for cut in ("_wall", "_trunk", "_crown", "_leg"):
        name = name.split(cut, 1)[0]
    return name


# ---- where move(to=...) asks it to go ----------------------------------------------------------------------


def goals_for(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fc: FlightController,
    to: list[Any],
    named: Destination | None = None,
) -> tuple[list[np.ndarray], bool, dict[str, Any]]:
    """The points a move(to=...) asks an aircraft to fly to, whether it lands at the last one, and how to describe
    the destination. `to` is [x, y] (at the height it flies at, or TAKEOFF_HEIGHT from the ground), [x, y, z], a
    list of such points, or a named thing (`named`, found by the caller), which it flies to 1 m above. See the
    module notes for which points are landings."""
    frame = fc.frame
    here = data.xpos[frame.root]
    ground_here = fc.ground_below(model, data, here)
    cruise_z = float(here[2]) if fc.airborne else (ground_here or 0.0) + frame.rest_height + TAKEOFF_HEIGHT
    if named is not None:
        from robotforge.behaviors.nav_map import world_extent

        top = max((float(world_extent(model, data, g)[1][2]) for g in named.geoms), default=named.z)
        z = max(cruise_z, top + frame.rest_height + TAKEOFF_HEIGHT)
        point = np.array([named.point[0], named.point[1], z])
        return [point], False, {"object": named.name, "at": [round(float(v), 3) for v in point]}
    many = len(to) > 0 and all(isinstance(p, (list, tuple)) for p in to)
    raw = list(to) if many else [to]
    if not raw or len(raw) > MAX_POINTS:
        raise FlightError(f"to is a point [x, y] or [x, y, z], or a list of 1 to {MAX_POINTS} of them")
    space = Airspace(model, data, fc)
    goals: list[np.ndarray] = []
    land = False
    for k, p in enumerate(raw):
        if len(p) not in (2, 3) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
            raise FlightError(f"each point is [x, y] or [x, y, z] in meters, got {p}")
        if max(abs(p[0]), abs(p[1])) > 100.0:
            raise FlightError(f"point {list(p)} is more than 100 m away: positions are in meters")
        if len(p) == 3 and p[2] > MAX_ALTITUDE:
            raise FlightError(f"z {p[2]:g} is above {MAX_ALTITUDE:g} m: heights are in meters")
        z = float(p[2]) if len(p) == 3 else cruise_z
        landing, z = _landing_at(space, np.array([float(p[0]), float(p[1]), z]), k, list(p), len(p) == 3)
        if landing and k < len(raw) - 1:
            raise FlightError(
                f"point {k + 1} {list(p)} is on the ground: only the last point can be a landing (z 0); give the "
                "others a height above the world's zero, e.g. 2"
            )
        land = land or landing
        goals.append(np.array([float(p[0]), float(p[1]), z]))
        cruise_z = z
    last = [round(float(v), 3) for v in goals[-1]]
    described: dict[str, Any] = {"point": last} if len(goals) == 1 else {"points": len(goals), "last": last}
    if land:
        described["land"] = True
    return goals, land, described


def _landing_at(space: Airspace, point: np.ndarray, k: int, asked: list[Any], has_z: bool) -> tuple[bool, float]:
    """Whether a point is a landing, and the height its center goes to there."""
    spot = space.spot(point)
    if spot.exit is None:  # out in the air: a landing only at the surface under it
        floor = (spot.surface if spot.surface is not None else 0.0) + space.rest
        if not has_z or point[2] > floor + LAND_MARGIN:
            return False, float(point[2])
        down = np.array([point[0], point[1], floor])
        if point[2] <= 0.0 and space.first_hit(down, down + UP * (space.height + UNDER)) is not None:
            # "On the ground" where it does not fit under what is over the spot (a ramp's slab, a low step): on top
            sky = np.array([point[0], point[1], GROUND_PROBE])
            top = ground_height(space.model, space.data, sky, space.own, space.root)
            if top is not None and top + space.rest > floor + 1e-3:
                return True, top + space.rest
        return True, floor
    what = spot.inside.name if spot.inside is not None else "the ground"
    top = spot.top if spot.top is not None else spot.exit
    if has_z and top - float(point[2]) <= BURIED:  # a little under the top of a rock, a hill, a crown: on it
        return True, top + space.rest
    if has_z and point[2] <= 0.0:  # z 0 or below: on the ground, whatever is there (a hill)
        return True, spot.exit + space.rest
    if not has_z:
        raise FlightError(
            f"point {k + 1} {asked} is inside {what} at the height it flies at ({point[2]:.2f} m): give a height "
            f"over it, e.g. [{asked[0]}, {asked[1]}, {top + space.rest + 1.0:.1f}], or a point beside it"
        )
    raise FlightError(
        f"point {k + 1} {asked} is inside {what} (its top there is at {top:.2f} m; z is a height above the "
        "world's zero): give a point beside it, or above its top"
    )


# ---- the way there -----------------------------------------------------------------------------------------


@dataclass
class FlightPlan:
    points: list[np.ndarray]
    land: bool
    length: float
    notes: list[str]
    touchdown: np.ndarray | None = None  # where its center comes to rest when it lands


def plan_flight(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fc: FlightController,
    goals: list[np.ndarray],
    land: bool,
) -> FlightPlan:
    """Points a mission flies through for `goals` (world points; with `land`, it lands at the last one): straight
    up first when it stands on the ground, then over or round anything a straight leg would fly into, and
    straight down at the end of a landing. Raises NoWay saying what is in the way when no way is clear."""
    space = Airspace(model, data, fc)
    here = data.xpos[fc.frame.root].copy()
    notes: list[str] = []
    points: list[np.ndarray] = []
    rest = space.rest
    start = here
    if not fc.airborne:  # straight up first: to the first point's height, or TAKEOFF_HEIGHT for a hop to land
        ground = fc.ground_below(model, data, here)
        floor = (ground if ground is not None else here[2] - rest) + rest
        hop = land and len(goals) == 1
        lift = floor + TAKEOFF_HEIGHT if hop else max(goals[0][2], floor + 0.3)
        start = _take_off(space, here, lift, floor, notes)
        points.append(start)
    touchdown = None
    for k, goal in enumerate(goals):
        landing = land and k == len(goals) - 1
        crowded = space.crowding(goal)
        if crowded is not None:
            where = [round(float(v), 2) for v in goal]
            doing = "land" if landing else "hover"
            raise NoWay("blocked", f"there is no room to {doing} at {where}: it would touch {crowded.name}")
        if landing:
            _check_slope(space, goal)
            legs, touchdown = _landing(space, start, goal, notes)
        else:
            legs = _route(space, start, goal, notes)
        points.extend(legs)
        start = points[-1] if points else start
    length = _length(here, points)
    if touchdown is not None:
        length += max(0.0, float(points[-1][2] - touchdown[2]))
    return FlightPlan(points, land, length, notes, touchdown)


def _check_slope(space: Airspace, touchdown: np.ndarray) -> None:
    """Refuse ground too steep to land on: the heights under the ends of its skids, as far apart as it is wide."""
    reach = 0.6 * space.radius
    above = touchdown[2] - space.rest + reach + 0.02  # over ground as steep as 45 degrees, under a ramp's slab
    heights = {}
    for dx, dy in ((reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach)):
        probe = np.array([touchdown[0] + dx, touchdown[1] + dy, above])
        heights[dx, dy] = ground_height(space.model, space.data, probe, space.own, space.root)
    if any(h is None for h in heights.values()):
        return
    gx = (heights[reach, 0.0] - heights[-reach, 0.0]) / (2 * reach)
    gy = (heights[0.0, reach] - heights[0.0, -reach]) / (2 * reach)
    slope = math.degrees(math.atan(math.hypot(gx, gy)))
    if slope > MAX_SLOPE:
        where = [round(float(v), 2) for v in touchdown[:2]]
        raise NoWay(
            "too_steep", f"the ground at {where} slopes {slope:.0f} degrees, too steep to land on; pick a flatter spot"
        )


def _take_off(space: Airspace, here: np.ndarray, lift: float, floor: float, notes: list[str]) -> np.ndarray:
    """Where it climbs to straight up from the ground: `lift`, or under whatever hangs over it."""
    up = np.array([here[0], here[1], lift])
    over = space.first_hit(here, up)
    if over is None:
        return up
    distance, thing = over
    below = float(here[2]) + distance - UNDER
    if below < floor + 0.2:
        raise NoWay("blocked", f"{thing.name} is right over it, so it cannot take off here; move it out from under it")
    notes.append(f"took off to {below:.1f} m, under {thing.name}")
    return np.array([here[0], here[1], below])


def _landing(
    space: Airspace, start: np.ndarray, goal: np.ndarray, notes: list[str]
) -> tuple[list[np.ndarray], np.ndarray]:
    """The way to a landing: across at its height to over the spot and straight down, or, with something over the
    spot (a crown, a table), down first and in under it."""
    touchdown = goal.copy()
    above = np.array([goal[0], goal[1], max(float(start[2]), float(goal[2]))])
    low_end = touchdown + UP * COLUMN_STOP
    if above[2] <= low_end[2] or space.first_hit(above, low_end) is None:
        return _route(space, start, above, notes), touchdown
    over = space.first_hit(low_end, above)  # going up from the spot: the overhang's underside
    where = [round(float(v), 2) for v in goal[:2]]
    if over is None:
        raise NoWay("blocked", f"something is in the way down to the spot to land on {where}; land beside it")
    distance, thing = over
    low = float(low_end[2]) + distance - UNDER
    if low < touchdown[2] + 0.1:
        raise NoWay("blocked", f"{thing.name} is over the spot to land on {where}, too low to fly in under it")
    try:
        down = np.array([start[0], start[1], min(low, float(start[2]))])
        legs = [*_route(space, start, down, notes)]
        legs += _route(space, legs[-1], np.array([goal[0], goal[1], low]), notes)
    except NoWay as exc:
        raise NoWay(
            "blocked",
            f"{thing.name} is over the spot to land on {where} and it cannot come in under it from here ({exc}); "
            f"add a point under it first, e.g. [{where[0]}, {where[1]}, {low:.1f}]",
        ) from exc
    notes.append(f"came in at {low:.1f} m, under {thing.name}, to land")
    return _dedupe(legs), touchdown


def _route(space: Airspace, a: np.ndarray, b: np.ndarray, notes: list[str]) -> list[np.ndarray]:
    """The points from a to b: straight there, or the shorter of up over and round the side of what is in the
    way, checked all the way. Raises NoWay when none is clear."""
    hit = space.first_hit(a, b)
    if hit is None:
        return [b]
    thing = hit[1]
    ways = [w for w in (_over(space, a, b, thing), *_round(space, a, b, thing)) if w is not None]
    clear, blockers = [], []
    for length, legs, note in ways:
        blocker = _blocker(space, a, legs)
        if blocker is None:
            clear.append((length, legs, note))
        else:
            blockers.append(blocker.name)
    if not clear:
        to = [round(float(v), 2) for v in b]
        if thing.top + CLEARANCE + space.rest > MAX_ALTITUDE:
            why = f"too tall to fly over ({thing.top:.1f} m)"
        else:
            others = sorted(set(blockers) - {thing.name})
            runs_into = f"runs into {', '.join(others)}" if others else "is blocked too"
            why = f"and going over or round it {runs_into}"
        raise NoWay(
            "blocked", f"{thing.name} is in the way of the straight flight to {to}, {why}; add points around it"
        )
    _, legs, note = min(clear, key=lambda c: c[0])
    notes.append(note)
    return legs


def _over(space: Airspace, a: np.ndarray, b: np.ndarray, thing: Thing) -> tuple[float, list[np.ndarray], str] | None:
    over = thing.top + CLEARANCE + space.rest
    if over > MAX_ALTITUDE:
        return None
    up = np.array([a[0], a[1], max(float(a[2]), over)])
    across = np.array([b[0], b[1], max(float(b[2]), over)])
    legs = _dedupe([up, across, b])
    return _length(a, legs), legs, f"climbed to {over:.1f} m to fly over {thing.name} (its top is at {thing.top:.1f} m)"


def _round(
    space: Airspace, a: np.ndarray, b: np.ndarray, thing: Thing
) -> list[tuple[float, list[np.ndarray], str] | None]:
    """Round the left and the right of `thing`, at the leg's height: the part of it as high as the leg (a tree's
    trunk at 1 m, not its crown), with SIDE_CLEARANCE beyond the rotors' reach."""
    from robotforge.behaviors.nav_map import world_extent

    d = (b - a)[:2]
    span = float(np.linalg.norm(d))
    if span < 1e-3 or not thing.geoms:
        return []  # straight up or down: nothing to go round
    u = d / span
    n = np.array([-u[1], u[0]])
    low, high = min(a[2], b[2]) - space.rest - 0.1, max(a[2], b[2]) + space.height + 0.1
    boxes = [world_extent(space.model, space.data, g) for g in thing.geoms]
    level = [(lo, hi) for lo, hi in boxes if hi[2] >= low and lo[2] <= high] or boxes
    lo = np.min([box[0][:2] for box in level], axis=0)
    hi = np.max([box[1][:2] for box in level], axis=0)
    if np.any(hi - lo > TOO_WIDE):
        return []
    corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - a[:2]
    s, t = corners @ u, corners @ n
    margin = SIDE_CLEARANCE + space.radius
    s1, s2 = max(0.0, float(s.min()) - margin), min(span, float(s.max()) + margin)
    if s2 <= s1:
        return []
    out: list[tuple[float, list[np.ndarray], str] | None] = []
    for side, offset in (("left", float(t.max()) + margin), ("right", float(t.min()) - margin)):
        legs = []
        for along in (s1, s2):
            xy = a[:2] + u * along + n * offset
            legs.append(np.array([xy[0], xy[1], a[2] + (b[2] - a[2]) * along / span]))
        legs = _dedupe([*legs, b])
        out.append((_length(a, legs), legs, f"flew round the {side} of {thing.name}"))
    return out


def _blocker(space: Airspace, a: np.ndarray, legs: list[np.ndarray]) -> Thing | None:
    """The first thing a path from a through `legs` runs into, or None."""
    path = [a, *legs]
    for p, q in zip(path[:-1], path[1:], strict=True):
        hit = space.first_hit(p, q)
        if hit is not None:
            return hit[1]
    return None


def _dedupe(points: list[np.ndarray]) -> list[np.ndarray]:
    return [p for i, p in enumerate(points) if i == 0 or not np.allclose(p, points[i - 1], atol=1e-3)]


def _length(a: np.ndarray, legs: list[np.ndarray]) -> float:
    path = [a, *legs]
    return float(sum(np.linalg.norm(q - p) for p, q in zip(path[:-1], path[1:], strict=True)))


# ---- the trip -----------------------------------------------------------------------------------------------


@dataclass
class FlightTrip:
    """A trip through the air for move(to=...): the flight controller flies the mission, this watches it and
    reports how it went. It answers like navigate.Navigator, so the tools wait on and report both the same way."""

    robot: str
    fc: FlightController
    mission: Mission
    destination: dict[str, Any]
    plan: Any
    start_time: float
    deadline: float | None
    on_finish: Callable[[FlightTrip], None] = lambda trip: None
    notes: list[str] = field(default_factory=list)
    touchdown: np.ndarray | None = None  # where a landing was planned to put its center
    rate_hz: float = 20.0
    state: str = "flying"  # flying, arrived, stopped
    cause: str | None = None
    note: str | None = None
    end_time: float | None = None
    _last_t: float | None = None
    _travelled: float = 0.0
    _last_p: np.ndarray | None = None
    _worst: float = 0.0  # m, the furthest it strayed from its reference while flying

    @classmethod
    def start(
        cls,
        fc: FlightController,
        plan: FlightPlan,
        destination: dict[str, Any],
        cruise: float,
        now: float,
        seconds: float | None,
        on_finish: Callable[[FlightTrip], None],
    ) -> FlightTrip:
        mission = Mission([p.copy() for p in plan.points], plan.land, cruise)
        fc.fly(mission)
        return cls(
            fc.robot,
            fc,
            mission,
            destination,
            SimpleNamespace(length=plan.length),
            now,
            now + seconds if seconds is not None else None,
            on_finish,
            list(plan.notes),
            plan.touchdown,
        )

    @property
    def done(self) -> bool:
        return self.state in ("arrived", "stopped")

    @property
    def arrived(self) -> bool:
        return self.state == "arrived"

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            return
        t = float(data.time)
        self._last_t = t
        p = data.xpos[self.fc.frame.root].copy()
        if self._last_p is not None:
            self._travelled += float(np.linalg.norm(p - self._last_p))
        self._last_p = p
        if self.fc.ref is not None and self.fc.phase == "flying":
            self._worst = max(self._worst, float(np.linalg.norm(self.fc.ref.sp - p)))
        m = self.mission
        if self.fc.fallen or m.state == "aborted":
            self._finish(t, "stopped", "crashed", "it crashed on the way")
        elif self.fc.mission is not m and not m.done:
            self._finish(t, "stopped", "interrupted", "it was flown another way (keys, move or stop) on the way")
        elif m.state == "arrived":
            self._finish(t, "arrived")
        elif m.state == "landed" and self.fc.phase == "landed":
            self._landed(t, p)
        elif self.deadline is not None and t >= self.deadline - 1e-9:
            self.fc.hover()
            self._finish(t, "stopped", "out_of_time", "it ran out of time on the way and hovers where it was")

    def _landed(self, t: float, p: np.ndarray) -> None:
        """Down: on the spot planned, or somewhere else (on top of something it came down on)."""
        planned = self.touchdown
        if planned is None or (
            float(np.hypot(*(p[:2] - planned[:2]))) <= TOUCHDOWN_TOL and abs(float(p[2] - planned[2])) <= TOUCHDOWN_TOL
        ):
            self._finish(t, "arrived")
            return
        self._finish(
            t,
            "stopped",
            "landed_elsewhere",
            f"it came down at {[round(float(v), 2) for v in p]}, not on the spot planned "
            f"{[round(float(v), 2) for v in planned]}: something was in the way",
        )

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "behavior": "fly",
            "navigating": self.robot,
            "state": self.state,
            "goal": self.destination,
        }
        if not self.done:
            ahead = self.mission.points[self.mission.index :]
            here = self._last_p if self._last_p is not None else ahead[0]
            out["path"] = [[round(float(v), 3) for v in p] for p in [here, *ahead]]
        if self.note:
            out["note"] = self.note
        return out

    def interrupt(self, note: str) -> None:
        if not self.done:
            self._finish(self._last_t if self._last_t is not None else self.start_time, "stopped", "interrupted", note)

    def _finish(self, t: float, state: str, cause: str | None = None, note: str | None = None) -> None:
        self.state, self.cause, self.end_time = state, cause, t
        if note:
            self.note = note
        self.on_finish(self)

    def result(self) -> dict[str, Any]:
        end = self.end_time if self.end_time is not None else (self._last_t or self.start_time)
        out: dict[str, Any] = {
            "arrived": self.arrived,
            "destination": self.destination,
            "path_m": round(float(self.plan.length), 2),
            "distance_m": round(self._travelled, 2),
            "time_s": round(end - self.start_time, 2),
        }
        if self._last_p is not None:
            out["position"] = [round(float(v), 3) for v in self._last_p]
        if self.mission.land:
            out["landed"] = self.fc.phase == "landed"
        else:
            out["hovering"] = self.fc.airborne
            if self.fc.airborne:  # z is a world height: over a hill or a table it is nearer the ground than z says
                out["clearance_m"] = round(self.fc.altitude, 2)
        if self.mission.touchdown_speed is not None:
            out["touchdown_m_s"] = round(self.mission.touchdown_speed, 3)
        out["max_tracking_error_m"] = round(self._worst, 3)
        notes = [*self.notes, *([self.note] if self.note else [])]
        if notes:
            out["note"] = "; ".join(notes)
        if self.cause and not self.arrived:
            out["why"] = self.cause
        return out
