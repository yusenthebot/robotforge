"""Going somewhere: plan a path through the scene, then walk or drive it with the robot's own move controller.

The planner maps what blocks this robot (nav_map), searches the map (nav_grid) and keeps the corners of the
path. The follower is a pure pursuit (Coulter, 1992) on top of whatever moves the robot (a gait, wheels, a
learned policy): it steers toward a point a little ahead on the path, slows down for the end, and at a sharp
corner drives to the corner itself and turns there on the spot, so it never cuts a corner into a wall. Pushed
more than a little off the path, or making no headway for a while, it plans again from where it is; stuck
in the same place again and again, it gives up and says where.

A destination is a point on the floor, or something in the scene: an object or another robot (it stops next
to it, facing it; a robot with an arm stops where the arm can reach it), or a flat mark on the ground (it
goes onto it).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import nav_grid, nav_map
from robotforge.behaviors.nav_grid import Grid, astar, clearance, pull_string
from robotforge.behaviors.nav_map import OBSTACLE, SLOPE, STEP, Body, NavMap, StaticLayer
from robotforge.sim.geometry import collides

RATE_HZ = 20.0  # follower updates per sim second; the walker under it runs at its own rate
ARRIVE_TOL = 0.08  # m from the final point that counts as there
OVERSHOOT = 0.2  # m: a walker that slows late steps past the end; that far past it, it is there too
NEAR_ENOUGH = 0.15  # m: a point a little inside the room a robot keeps from walls, reached this nearly, is reached
FACE_TOL = math.radians(12.0)  # facing an object this closely is facing it
GAP = 0.15  # m left between the robot's front and an object it was sent to
# Where it cannot stop that close to an object it was sent to, up to NEXT_TO m further is still next to it: at the
# edge of the counter or the shelf the object is on (kitchen: 0.5 to 0.8 m from the cups on the island). Further
# out it goes as near as it can get, up to FAR m from the object, and says it did not get there.
NEXT_TO = 0.3
FAR = 3.0
ARM_SHARE = 0.75  # of an arm's stretched-out length: what it reaches with its gripper pointing down
SNAP = 0.5  # m: a point inside an obstacle is replaced by the nearest free spot up to this far beyond the robot
CRUISE_SHARE, CRUISE_MAX = 0.8, 0.5  # default top speed: this share of max_speed, at most this many m/s
TURN_ON_SPOT, TURN_DONE = math.radians(50.0), math.radians(20.0)  # start and stop turning on the spot
SHARP = math.radians(35.0)  # a corner that turns this much is driven to and turned at on the spot
TURN_GAIN = 2.0  # rad/s of turn per rad of heading error
MIN_LOOK = 0.1  # m: the lookahead is halved down to this where the straight way to it is short of room
SLOW_GAIN = 1.2  # m/s of speed per m left to the end or to a sharp corner
MIN_SPEED_SHARE = 0.3  # of the top speed, at the least, until it is there
PROGRESS = 0.05  # m of headway that counts as getting on
STUCK_TIME = 3.0  # s without headway (plus a half turn's time) that counts as stuck
MAX_REPLANS = 4  # plans after the first before it gives up
# Another robot that moved onto the path since it was planned (by more than MOVED m): one this close to the path
# (beyond both robots' half widths) within AHEAD m in front is in the way. The robot whose name comes first plans
# around it after YIELD_FIRST s, the other after YIELD_SECOND s, so two that meet do not swerve the same way at
# once; MAX_YIELDS times at most. One standing still is in the plan already.
AHEAD = 1.0  # m
MARGIN_AROUND = 0.05  # m
MOVED = 0.1  # m
YIELD_FIRST, YIELD_SECOND = 0.3, 2.5  # s
MAX_YIELDS = 6
FALLEN = math.cos(math.radians(60.0))  # its up axis this far from up: it has fallen over
# Turn rate the follower asks for, rad/s, by walker: wheels turn fast, gaits trot round at about 1 rad/s
# (tests/test_behaviors.py turns a dog at 0.8), a humanoid takes its own limit (about 0.4).
TURN_RATE = {"rover": 1.5, "balancer": 1.2, "snake": 0.6}
DEFAULT_TURN_RATE = 1.0


@dataclass(frozen=True)
class Other:
    """Another robot in the world, for making way: its root body and how far it reaches from its middle."""

    name: str
    root: int
    radius: float  # m


class NoWay(ValueError):
    """There is no path; `cause` says why in one word (inside, too_narrow, climb, closed_off, boxed_in)."""

    def __init__(self, cause: str, message: str) -> None:
        super().__init__(message)
        self.cause = cause
        self.destination: dict[str, Any] | None = None  # set by whoever knows what was asked for


@dataclass(frozen=True)
class Arm:
    """An arm the robot carries: where its base is relative to the robot's center, and how far it reaches."""

    forward: float  # m ahead of the robot's center (in its heading frame)
    base_z: float  # m, world height of the arm's base
    reach: float  # m, stretched out


@dataclass(frozen=True)
class Destination:
    point: tuple[float, float]  # where to go, or the middle of what to go to
    name: str | None = None  # the object or robot, None for a point
    onto: bool = False  # a flat mark on the ground: go onto it rather than next to it
    geoms: tuple[int, ...] = ()  # the object's shapes, for how far a cell is from it
    root: int = -1  # its root body (moving objects), left out of the map's obstacles while it is the goal
    z: float = 0.0  # height of its middle, for an arm's reach
    loose: bool = False  # it moves (a free joint): something an arm might pick up, so its reach is checked
    terrain: bool = False  # a terrain patch: go onto its middle if the robot can stand there, else next to it

    def describe(self) -> dict[str, Any]:
        at = [round(self.point[0], 3), round(self.point[1], 3)]
        return {"object": self.name, "at": at} if self.name else {"point": at}


def find_destination(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> Destination | None:
    """The thing named `name` (a body, or the world geoms of a fixed object), or None."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b > 0:
        bodies = {k for k in range(model.nbody) if _descends(model, k, b)}
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]
        root = int(model.body_rootid[b])
    else:
        geoms = []
        for g in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if int(model.geom_bodyid[g]) == 0 and (gname == name or gname.startswith((f"{name}/", f"{name}_wall"))):
                geoms.append(g)
        root = -1
    if not geoms:
        return None
    return _destination_from(model, data, name, geoms, root)


def robot_destination(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> Destination:
    geoms = [g for g in nav_map.robot_geoms(model, robot) if collides(model, g)]
    root = int(model.body_rootid[nav_map.root_body(model, robot)])
    return _destination_from(model, data, robot, geoms, root)


def _destination_from(model, data, name: str, geoms: list[int], root: int) -> Destination:
    solid = [g for g in geoms if collides(model, g)]
    boxes = [nav_map.world_extent(model, data, g) for g in (solid or geoms)]
    low, high = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    middle = (low + high) / 2
    flat = not solid and float(high[2]) < 0.05  # a mark on the ground: nothing collides with it
    loose = root > 0 and int(model.body_weldid[root]) != 0
    return Destination(
        (float(middle[0]), float(middle[1])), name, flat, tuple(solid or geoms), root, float(middle[2]), loose
    )


def _descends(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


# ---- planning ----------------------------------------------------------------------------------------------


@dataclass
class Plan:
    points: np.ndarray  # (n, 2) world x, y: the robot's position first, the corners, the end
    heights: np.ndarray  # (n,) ground under each point, m (for drawing the path)
    length: float  # m
    face: tuple[float, float] | None  # turn to face this point at the end
    exact: bool  # it ends where it was sent (not at the nearest free spot, nor further from an object)
    note: str | None = None
    within_reach: bool | None = None  # for a robot with an arm sent to an object
    short: str | None = None  # why it ends short (a NoWay cause): only near an object, or by a point it cannot reach


class Planner:
    """Plans for one robot to one destination, again and again as the trip goes (static layer kept)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        body: Body,
        destination: Destination,
        arm: Arm | None = None,
        layers: dict | None = None,
    ) -> None:
        self.model, self.robot, self.body, self.destination, self.arm = model, robot, body, destination, arm
        self.layers = layers if layers is not None else {}  # robot -> StaticLayer, shared per compiled world
        self.root = nav_map.root_body(model, robot)
        self.last_map: NavMap | None = None
        self._own: StaticLayer | None = None  # a coarser layer that reaches a far destination, for this trip only

    def _layer(self, data: mujoco.MjData, start: tuple[float, float]) -> StaticLayer:
        # What never moves is the same whatever shape the robot is in (only its geoms are left out), so one layer
        # per robot serves every trip in this world.
        points = [start, self.destination.point]
        for layer in (self._own, self.layers.get(self.robot)):
            if layer is not None and all(_inside_margin(layer.grid, p) for p in points):
                return layer
        grid = nav_map.bounds(self.model, data, self.robot, points)
        layer = nav_map.static_layer(self.model, data, self.robot, grid)
        # A point far outside the scene stretches the map, and a map wider than MAX_CELLS gets coarser cells. Shared,
        # one far trip made every later trip plan on it: after a trip to [60, 60] in the kitchen (9.3 cm cells), a
        # humanoid sent to the fridge stopped 0.45 m short of it. Only a layer as fine as the scene's own is shared.
        if grid.res <= nav_map.bounds(self.model, data, self.robot, [start]).res + 1e-9:
            self.layers[self.robot] = layer
        else:
            self._own = layer
        return layer

    def plan(self, data: mujoco.MjData, blockers: list[tuple[float, float, float]] = ()) -> Plan:
        """A path from where the robot is now. Raises NoWay, saying why, when there is none."""
        start_xy = (float(data.xpos[self.root][0]), float(data.xpos[self.root][1]))
        layer = self._layer(data, start_xy)
        grid = layer.grid
        names = list(layer.names)  # moving things add their names for this plan only
        skip = {self.destination.root} if self.destination.root > 0 else set()
        moving = nav_map.dynamic_columns(self.model, data, self.robot, self.body, grid, names, skip)
        nav = nav_map.build(layer, self.body, moving, _disks(grid, blockers), names)
        self.last_map = nav
        nav.free_from(grid.cell(*start_xy))
        first, escape = self._start(nav, grid.cell(*start_xy), start_xy)
        goals, away = self._goals(nav, data, first)
        penalty = nav.penalty()
        level, cells = next(
            ((k, found) for k, goal in enumerate(goals) if (found := self._search(nav, penalty, first, goal))),
            (-1, None),
        )
        short = None
        if cells is None and away is not None:  # an object it cannot get next to: as near to it as it gets
            near = self._nearest(nav, first, away)
            cells = self._search(nav, penalty, first, near) if near is not None else None
            if cells is not None:
                level, short = len(goals), self._why_not(nav, first, np.logical_or.reduce(goals)).cause
        if cells is None:
            raise self._why_not(nav, first, np.logical_or.reduce(goals))
        corners = pull_string(grid, cells, nav.room, self.body.reach + 0.1)
        points = [start_xy]
        if escape:
            points.append(grid.center(*first))
        points += [grid.center(*c) for c in corners[1:]]
        d = self.destination
        on_point = self._onto(nav)
        exact = level == 0 and (not on_point or self._point_is_free(nav))
        if on_point and exact:
            # The point itself, in place of the search's last cell center next to it: a stub of a few cm from one
            # to the other made the last stretch point anywhere, and a Go2 steering for it on a ramp swung round
            # and slid off (scripts/nav_bench.py, warehouse).
            if len(points) > 1 + escape and math.dist(points[-1], d.point) < 2 * grid.res:
                points[-1] = d.point
            else:
                points.append(d.point)
        pts = _dedupe(np.array(points, float))
        heights = np.array([nav.ground_at(x, y) for x, y in pts])
        plan = Plan(pts, heights, nav_grid.length(pts), None if on_point else d.point, exact)
        if not exact:
            plan.note = self._short_note(nav, cells[-1], on_point, away)
        plan.short = self._point_cause(nav) if on_point and not exact else short
        if self.arm is not None and d.loose and not on_point:
            plan.within_reach = self._reaches(pts[-1])
        return plan

    @staticmethod
    def _search(nav: NavMap, penalty: np.ndarray, first: tuple[int, int], goal: np.ndarray):
        return astar(nav.grid, nav.free, penalty, first, goal, _heuristic(nav.grid, goal))

    def _label(self) -> str:
        d = self.destination
        return d.name or f"[{d.point[0]:.2f}, {d.point[1]:.2f}]"

    def _inside(self, nav: NavMap) -> str:
        """What the destination point is in or on, in words."""
        here = nav.grid.cell(*self.destination.point)
        if not nav.grid.inside(*here):
            return "where the robot does not fit"
        what = nav.blocker(here) or next(iter(_owners_near(nav, here, 3 * nav.grid.res)), None)
        if nav.lethal[here] == SLOPE:
            return f"on a slope{f' of {what}' if what else ''} steeper than it climbs ({self.body.slope_words()})"
        if nav.ground[here] - nav.floor[here] > self.body.step and what:
            return f"on top of or inside {what}"
        return _inside_text(what)

    def _point_cause(self, nav: NavMap) -> str:
        """Why a point it cannot stand on is out of its way, in one word: on a slope too steep, or in something."""
        here = nav.grid.cell(*self.destination.point)
        return "climb" if nav.grid.inside(*here) and nav.lethal[here] == SLOPE else "inside"

    def _point_is_free(self, nav: NavMap) -> bool:
        here = nav.grid.cell(*self.destination.point)
        return nav.grid.inside(*here) and bool(nav.free[here])

    def _onto(self, nav: NavMap) -> bool:
        """Whether it goes to the destination's point itself (a point, a mark on the floor, the middle of a ramp's
        landing or a platform it climbs), rather than stopping next to it, facing it."""
        d = self.destination
        return not d.name or d.onto or (d.terrain and self._point_is_free(nav))

    def _short_note(self, nav: NavMap, end: tuple[int, int], on_point: bool, away: np.ndarray | None) -> str:
        """Why the path ends short of where the robot was sent."""
        d, grid = self.destination, nav.grid
        if on_point:
            gap = math.dist(grid.center(*end), d.point)
            where = f"{self._label()} is {self._inside(nav)}"
            return f"{where}, so it stops at the nearest spot with room for it, {gap:.2f} m off"
        gap = float(away[end]) if away is not None else math.dist(grid.center(*end), d.point)
        between = [n for n in _owners_between(nav, end, grid.cell(*d.point)) if n != d.name][:2]
        lead = ": in the way is " if len(between) == 1 else ": in the way are "
        return f"it gets only as close to {d.name} as {gap:.2f} m{_among(between, lead)}"

    # -- start and goal --

    def _start(self, nav: NavMap, start: tuple[int, int], start_xy) -> tuple[tuple[int, int], bool]:
        """The cell to search from: the robot's own, or the nearest free one when it stands too close to
        something (it then drives straight there first)."""
        grid = nav.grid
        if grid.inside(*start) and nav.free[start]:
            return start, False
        xs, ys = grid.centers()
        dist = np.hypot(xs - start_xy[0], ys - start_xy[1])
        near = nav.free & (dist <= self.body.reach + 0.3)
        if not near.any():
            what = _owners_near(nav, start, self.body.reach + 0.3)
            raise NoWay("boxed_in", f"{self.robot} has no room to move where it stands{_among(what)}")
        k = np.unravel_index(int(np.argmin(np.where(near, dist, np.inf))), grid.shape)
        return (int(k[0]), int(k[1])), True

    def _goals(
        self, nav: NavMap, data: mujoco.MjData, start: tuple[int, int]
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Goal regions from the best to the most lenient (the first one ends where the robot was sent), and
        for an object how far each cell is from its edge (m)."""
        grid, d = nav.grid, self.destination
        xs, ys = grid.centers()
        to_point = np.hypot(xs - d.point[0], ys - d.point[1])
        if self._onto(nav):
            if self._point_is_free(nav):
                return [to_point <= grid.res], None
            # The nearest spot it can get to: free cells nearby can be out of reach too, on top of a wide block
            # (its top is level ground, with walls all round) or behind a wall.
            candidates = nav_grid.reachable(grid, nav.free, start) & (to_point <= self.body.reach + SNAP)
            if not candidates.any():  # nothing near it either: on a step it cannot climb, or deep in something
                raise self._why_not(nav, start, to_point <= grid.res)
            nearest = float(to_point[candidates].min())
            return [candidates & (to_point <= nearest + 0.1)], None
        away = clearance(self._object_mask(nav, data), grid.res, FAR)
        # Its front a hand's width from the object; at least a row of cells beyond the room it needs, or on coarse
        # cells (a map stretched to a far point) no cell falls between the two for a robot with a short front.
        want = max(self.body.front + GAP, self.body.need + grid.res)
        goals = []
        if self.arm is not None and d.loose:
            reach = self._reach_mask(xs, ys)
            if reach is not None:
                goals.append(reach)
        goals += [away <= want, away <= want + NEXT_TO]
        return goals, away

    def _nearest(self, nav: NavMap, start: tuple[int, int], away: np.ndarray) -> np.ndarray | None:
        """The cells it can get to that are nearest an object (within FAR of it), or None."""
        candidates = nav_grid.reachable(nav.grid, nav.free, start) & (away < FAR)
        if not candidates.any():
            return None
        return candidates & (away <= float(away[candidates].min()) + nav.grid.res)

    def _object_mask(self, nav: NavMap, data: mujoco.MjData) -> np.ndarray:
        grid = nav.grid
        mask = np.zeros(grid.rows * grid.cols, bool)
        for g in self.destination.geoms:
            cols = nav_map.geom_columns(self.model, data, g, grid, -1)
            if cols is not None:
                mask[cols.cells] = True
        if not mask.any():
            r, c = grid.cell(*self.destination.point)
            mask[min(max(r, 0), grid.rows - 1) * grid.cols + min(max(c, 0), grid.cols - 1)] = True
        return mask.reshape(grid.shape)

    def _reach_mask(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
        """Cells from which the arm (once the robot faces the object) reaches it, or None if it never can."""
        arm, d = self.arm, self.destination
        dz = d.z - arm.base_z
        level = (ARM_SHARE * arm.reach) ** 2 - dz**2
        if level <= 0:
            return None
        # Less the room a robot leaves when it stops (ARRIVE_TOL), so it reaches from wherever it stops there.
        return np.hypot(xs - d.point[0], ys - d.point[1]) - arm.forward <= math.sqrt(level) - ARRIVE_TOL

    def _reaches(self, at: np.ndarray, heading: float | None = None) -> bool:
        """Whether the arm reaches the object from `at` (the robot's middle), facing it or at `heading`."""
        arm, d = self.arm, self.destination
        if heading is None:
            flat = max(0.0, math.dist(at, d.point) - arm.forward)
        else:
            base = np.asarray(at) + arm.forward * np.array([math.cos(heading), math.sin(heading)])
            flat = math.dist(base, d.point)
        return math.hypot(flat, d.z - arm.base_z) <= ARM_SHARE * arm.reach + 1e-6

    # -- why there is no way --

    def _why_not(self, nav: NavMap, start: tuple[int, int], goal: np.ndarray) -> NoWay:
        grid, body = nav.grid, self.body
        heuristic = _heuristic(grid, goal)
        label = self._label()
        flat = np.zeros(grid.shape)
        # Without the steps and slopes it cannot climb, is there a way? Then they are what is in the way.
        obstacles = nav.lethal == OBSTACLE
        room = clearance(obstacles, grid.res, body.reach + 0.3)
        cells = astar(grid, room >= body.need, flat, start, goal, heuristic)
        if cells is not None:
            return NoWay("climb", f"no way to {label} for {self.robot}: {self._climb(nav, cells)}")
        # As a point (no room needed), is there a way? Then a gap on it is too narrow.
        point = nav.lethal == 0
        cells = astar(grid, point, flat, start, goal, heuristic)
        if cells is not None:
            tight = min(cells, key=lambda c: nav.room[c])
            width = 2 * float(nav.room[tight]) + grid.res
            what = _owners_near(nav, tight, float(nav.room[tight]) + 1.5 * grid.res)
            x, y = grid.center(*tight)
            return NoWay(
                "too_narrow",
                f"no way to {label} for {self.robot}: the way through is about {width:.2f} m wide near "
                f"[{x:.2f}, {y:.2f}]{_among(what, ' between ')}, and it needs {2 * body.need:.2f} m",
            )
        d = self.destination
        here = grid.cell(*d.point)
        if (not d.name or d.onto) and grid.inside(*here) and nav.ground[here] - nav.floor[here] > body.step:
            return NoWay(self._point_cause(nav), f"no way to {label} for {self.robot}: it is {self._inside(nav)}")
        around = _owners_near(nav, nav.grid.cell(*d.point), 1.5, many=4)
        what = f" ({', '.join(around[:-1])} and {around[-1]})" if len(around) > 1 else ""
        what = what or (f" ({around[0]})" if around else "")
        return NoWay("closed_off", f"no way to {label} for {self.robot}: walls or furniture close it off{what}")

    def _climb(self, nav: NavMap, cells: list[tuple[int, int]]) -> str:
        """What the way found without steps and slopes climbs that the robot cannot, in words."""
        body = self.body
        terrain = (nav.lethal == STEP) | (nav.lethal == SLOPE)
        near = _near_terrain(nav)
        # The last one before the end: that way may cross others it could have gone round (the way to the top of
        # the obstacle course's bumps crossed a curb first, and the curb was named), but not what the end is on.
        hit = next((c for c in reversed(cells) if near[c]), None)
        if hit is None:
            return "the way there climbs more than it can"
        cell = _nearest_lethal(nav, hit, terrain)
        what = nav.blocker(cell)
        at = f" ({what})" if what else ""
        if nav.lethal[cell] == SLOPE:
            return f"the only way there is up a slope steeper than it climbs ({body.slope_words()}){at}"
        if what and nav.ground[cell] == nav.floor[cell]:  # a step in a hill's own ground: bumps, not a block's edge
            return f"the only way there crosses {what}, more uneven than it steps ({body.step * 100:.1f} cm)"
        r, c = cell
        window = nav.ground[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2]
        height = float(window.max() - window.min())
        return f"the only way there climbs a {height * 100:.0f} cm step{at}, and it climbs {body.step * 100:.1f} cm"


def _near_terrain(nav: NavMap) -> np.ndarray:
    """Cells where the robot's center would touch a step or slope it cannot climb."""
    terrain = (nav.lethal == STEP) | (nav.lethal == SLOPE)
    return clearance(terrain, nav.grid.res, nav.body.need) < nav.body.need


def _heuristic(grid: Grid, goal: np.ndarray) -> np.ndarray:
    """A lower bound of the distance (m) from each cell to the goal region: to its bounding circle."""
    rows, cols = np.nonzero(goal)
    if len(rows) == 0:
        return np.zeros(grid.shape)
    xs, ys = grid.centers()
    cx, cy = grid.center(float(rows.mean()), float(cols.mean()))
    radius = float(np.hypot((rows - rows.mean()) * grid.res, (cols - cols.mean()) * grid.res).max())
    return np.maximum(0.0, np.hypot(xs - cx, ys - cy) - radius)


def _disks(grid: Grid, blockers) -> np.ndarray | None:
    if not blockers:
        return None
    xs, ys = grid.centers()
    out = np.zeros(grid.shape, bool)
    for x, y, r in blockers:
        out |= np.hypot(xs - x, ys - y) <= r
    return out


def _inside_margin(grid: Grid, p: tuple[float, float]) -> bool:
    r, c = grid.cell(*p)
    return 2 <= r < grid.rows - 2 and 2 <= c < grid.cols - 2


def _dedupe(points: np.ndarray) -> np.ndarray:
    keep = [0] + [i for i in range(1, len(points)) if np.linalg.norm(points[i] - points[i - 1]) > 1e-6]
    return points[keep]


def _nearest_lethal(nav: NavMap, cell: tuple[int, int], among: np.ndarray | None = None) -> tuple[int, int]:
    rows, cols = np.nonzero(nav.lethal != 0 if among is None else among)
    k = int(np.argmin((rows - cell[0]) ** 2 + (cols - cell[1]) ** 2))
    return int(rows[k]), int(cols[k])


def _owners_near(nav: NavMap, cell: tuple[int, int], radius: float, many: int = 2) -> list[str]:
    """Names of what makes the lethal cells within `radius` (m) of a cell lethal, nearest first, at most `many`."""
    grid = nav.grid
    n = math.ceil(radius / grid.res)
    r0, r1 = max(cell[0] - n, 0), min(cell[0] + n + 1, grid.rows)
    c0, c1 = max(cell[1] - n, 0), min(cell[1] + n + 1, grid.cols)
    window = nav.owner[r0:r1, c0:c1]
    rows, cols = np.nonzero((window >= 0) & (nav.lethal[r0:r1, c0:c1] != 0))
    order = np.argsort((rows + r0 - cell[0]) ** 2 + (cols + c0 - cell[1]) ** 2)
    names: list[str] = []
    for k in order:
        name = nav.names[int(window[rows[k], cols[k]])]
        if name not in names:
            names.append(name)
        if len(names) == many:
            break
    return names


def _owners_between(nav: NavMap, a: tuple[int, int], b: tuple[int, int]) -> list[str]:
    """Names of what makes the lethal cells on the straight line from cell a to cell b lethal, in order."""
    n = max(2, math.ceil(3 * math.hypot(b[0] - a[0], b[1] - a[1])) + 1)
    t = np.linspace(0.0, 1.0, n)
    rows = np.clip(np.rint(a[0] + (b[0] - a[0]) * t).astype(int), 0, nav.grid.rows - 1)
    cols = np.clip(np.rint(a[1] + (b[1] - a[1]) * t).astype(int), 0, nav.grid.cols - 1)
    names: list[str] = []
    for r, c in zip(rows, cols, strict=True):
        k = int(nav.owner[r, c])
        if nav.lethal[r, c] and k >= 0 and nav.names[k] not in names:
            names.append(nav.names[k])
    return names


def _among(names: list[str], lead: str = " near ") -> str:
    if not names:
        return ""
    return lead + " and ".join(names)


def _inside_text(what: str | None) -> str:
    return f"inside or right next to {what}" if what else "where the robot does not fit"


# ---- following ----------------------------------------------------------------------------------------------


@dataclass
class Navigator:
    """A controller that follows a Plan by commanding the robot's walker (gait, wheels, policy) at RATE_HZ."""

    robot: str
    walker: Any  # has command(speed, turn) and max_speed
    planner: Planner
    plan: Plan
    cruise: float  # m/s, top speed on the way
    turn_rate: float  # rad/s the follower turns at
    turns_in_place: bool
    start_time: float
    deadline: float | None  # sim time it must be done by
    still_mine: Callable[[], bool] = lambda: True  # False once something else drives the robot
    on_finish: Callable[[Navigator], None] = lambda nav: None
    others: list[Other] = field(default_factory=list)  # the other robots, to make way for when they move
    rate_hz: float = RATE_HZ
    state: str = "following"  # following, facing, arrived, stopped
    note: str | None = None
    cause: str | None = None  # why it stopped short (stuck, fell, out_of_time, interrupted, or a NoWay cause)
    replans: int = 0
    traveled: float = 0.0
    end_time: float | None = None
    first_length: float = field(init=False)
    _seg: int = field(default=0, init=False)
    _spinning: bool = field(default=False, init=False)
    _best_left: float = field(default=math.inf, init=False)
    _since: float = field(default=0.0, init=False)
    _last_xy: np.ndarray | None = field(default=None, init=False)
    _last_yaw: float = field(default=0.0, init=False)
    _counted: np.ndarray | None = field(default=None, init=False)  # where travel was last counted from
    _blockers: list = field(default_factory=list, init=False)
    _waiting: tuple[str, float] | None = field(default=None, init=False)  # (robot in the way, since when)
    _yields: int = field(default=0, init=False)
    _others_at: dict[str, np.ndarray] = field(default_factory=dict, init=False)  # where they were for the plan
    _retried: bool = field(default=False, init=False)  # planned again at the end of a path to a taken point

    def __post_init__(self) -> None:
        self.first_length = self.plan.length
        self._since = self.start_time
        self._reset_path()

    @property
    def done(self) -> bool:
        return self.state in ("arrived", "stopped")

    @property
    def arrived(self) -> bool:
        return self.state == "arrived"

    def _reset_path(self) -> None:
        p = self.plan.points
        self._seg = 0
        self._spinning = False
        self._best_left = math.inf
        seg = np.diff(p, axis=0)
        self._seg_len = np.linalg.norm(seg, axis=1)
        self._after = np.concatenate([np.cumsum(self._seg_len[::-1])[::-1], [0.0]])  # path left from each point

    # -- controller interface --

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            return
        t = float(data.time)
        root = self.planner.root
        xy = data.xpos[root][:2].copy()
        rot = data.xmat[root]
        yaw = math.atan2(rot[3], rot[0])
        # Distance counted in steps of at least 5 cm, so a humanoid's sway from foot to foot is not counted as travel
        if self._counted is None or float(np.linalg.norm(xy - self._counted)) >= 0.05:
            self.traveled += float(np.linalg.norm(xy - self._counted)) if self._counted is not None else 0.0
            self._counted = xy
        self._last_xy, self._last_yaw = xy, yaw
        if not self.still_mine():
            self._finish(t, "stopped", "interrupted", "something else took over the robot (a key, a mode or a reset)")
            return
        if rot[8] < FALLEN:
            self._finish(t, "stopped", "fell", "it fell over on the way")
            return
        if self.deadline is not None and t >= self.deadline - 1e-9:
            s, _ = self._project(xy)
            left = float(self._after[self._seg] - s)
            self._finish(t, "stopped", "out_of_time", f"it ran out of time with {left:.2f} m of its path to go")
            return
        if self.state == "facing":
            self._face(t, xy, yaw)
            return
        self._follow(t, xy, yaw, data)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "behavior": "navigate",
            "navigating": self.robot,
            "state": self.state,
            "goal": self.planner.destination.describe(),
        }
        if not self.done:
            out["path"] = self._remaining_path()
        if self.note:
            out["note"] = self.note
        return out

    # -- following the path --

    def _follow(self, t: float, xy: np.ndarray, yaw: float, data: mujoco.MjData) -> None:
        p = self.plan.points
        end = p[-1]
        to_end = float(np.linalg.norm(end - xy))
        if to_end <= ARRIVE_TOL or (to_end <= OVERSHOOT and float((xy - end) @ (end - p[-2])) > 0):
            self.walker.command(0.0, 0.0)
            d = self.planner.destination
            if self.plan.face is not None:
                self.state = "facing"
                self._spinning = True
            elif self.plan.exact or math.dist(xy, d.point) <= NEAR_ENOUGH:
                self._finish(t, "arrived")
            elif not self._retried and self._try_again(t, data):
                pass  # what stood on the point (another robot) has moved off it: on to the point itself
            else:  # a point inside something: it got as near as it could, which is not there
                self._finish(t, "stopped", self.plan.short or "inside", self.plan.note)
            return
        s, off = self._project(xy)
        if off > max(0.3, 2.0 * self.planner.body.radius):
            self._replan(t, data, "it was pushed off its path")
            return
        if self._make_way(t, xy, s, data):
            return
        left = self._after[self._seg] - s
        if left < self._best_left - PROGRESS:
            self._best_left = left
            self._since = t
        elif t - self._since > STUCK_TIME + math.pi / self.turn_rate:
            # Something the map did not show is in front of it (a step it cannot climb after all, an object
            # too heavy to push): mark a robot-sized spot there so the new plan goes another way.
            body = self.planner.body
            ahead = body.front + body.radius
            self._blockers.append((xy[0] + ahead * math.cos(yaw), xy[1] + ahead * math.sin(yaw), body.radius))
            self._replan(t, data, "something it could not get past was in the way")
            return
        target, corner_left = self._target(xy, s)
        bearing = math.atan2(target[1] - xy[1], target[0] - xy[0])
        alpha = _wrap(bearing - yaw)
        if self.turns_in_place and (abs(alpha) > TURN_ON_SPOT or (self._spinning and abs(alpha) > TURN_DONE)):
            self._spinning = True
            self.walker.command(0.0, self._turn(alpha))
            return
        self._spinning = False
        speed = self.cruise * max(0.0, math.cos(alpha))
        slow = max(MIN_SPEED_SHARE * self.cruise, SLOW_GAIN * min(left, corner_left))
        self.walker.command(min(speed, slow), float(np.clip(TURN_GAIN * alpha, -self.turn_rate, self.turn_rate)))

    def _make_way(self, t: float, xy: np.ndarray, s: float, data: mujoco.MjData) -> bool:
        """Another robot on the path just ahead (it moved there after the plan): stop, and after a moment plan
        again around where it is now. Two robots that meet must not both swerve the same way and meet again, so
        the one whose name comes first waits a moment and the other a few seconds. True while it waits."""
        other = self._robot_ahead(xy, s, data)
        if other is None:
            self._waiting = None
            return False
        self.walker.command(0.0, 0.0)
        if self._waiting is None or self._waiting[0] != other.name:
            self._waiting = (other.name, t)
        self._since = t  # waiting is not being stuck
        wait = YIELD_FIRST if self.robot < other.name else YIELD_SECOND
        if t - self._waiting[1] < wait:
            return True
        self._yields += 1
        self._waiting = None
        if self._yields > MAX_YIELDS:
            self._finish(t, "stopped", "stuck", f"{other.name} kept getting in the way")
            return True
        self._replan(t, data, f"{other.name} was in the way", count=False)  # making way is not failing to plan
        return True

    def _robot_ahead(self, xy: np.ndarray, s: float, data: mujoco.MjData) -> Other | None:
        """The nearest other robot whose middle is within reach of the path over the next AHEAD meters."""
        if not self.others:
            return None
        body = self.planner.body
        ahead = self._path_ahead(s, AHEAD + body.front)
        best, best_d = None, math.inf
        for other in self.others:
            at = data.xpos[other.root][:2]
            then = self._others_at.get(other.name)
            if then is not None and np.linalg.norm(at - then) <= MOVED:
                continue  # where it was when the path was planned: the path goes round it
            d = float(np.linalg.norm(at - xy))
            if d > AHEAD + body.reach + other.radius or d >= best_d:
                continue
            if _distance_to_polyline(at, ahead) < body.radius + other.radius + MARGIN_AROUND:
                best, best_d = other, d
        return best

    def _path_ahead(self, s: float, distance: float) -> np.ndarray:
        """The path from the robot's projection on, `distance` meters of it."""
        p = self.plan.points
        points = [self._point_at(s, 0.0)]
        left, seg, along = distance, self._seg, s
        while left > 0 and seg < len(p) - 1:
            room = self._seg_len[seg] - along
            if room >= left:
                points.append(self._point_at(s, distance))
                break
            points.append(p[seg + 1])
            left -= room
            seg, along = seg + 1, 0.0
        return np.array(points)

    def _project(self, xy: np.ndarray) -> tuple[float, float]:
        """Distance along the path from the start of the current segment, and how far off the path the robot
        is. Moves on to the next segment once the robot is past the current one's end (or at a sharp corner)."""
        p = self.plan.points
        while self._seg < len(p) - 2:
            a, b = p[self._seg], p[self._seg + 1]
            u = (b - a) / max(self._seg_len[self._seg], 1e-9)
            past = float((xy - a) @ u) >= self._seg_len[self._seg]
            at_corner = np.linalg.norm(b - xy) <= ARRIVE_TOL / 2
            if not (past or at_corner):
                break
            self._seg += 1
        a, b = p[self._seg], p[self._seg + 1]
        length = max(self._seg_len[self._seg], 1e-9)
        u = (b - a) / length
        along = float(np.clip((xy - a) @ u, 0.0, length))
        off = float(np.linalg.norm(xy - (a + u * along)))
        return along, off

    def _target(self, xy: np.ndarray, s: float) -> tuple[np.ndarray, float]:
        """The point to steer for, and how far along the path the next sharp corner is (inf without one).
        The point is the one a lookahead ahead on the path, but never past a sharp corner: up to one, the
        robot heads for the corner itself and turns there."""
        p = self.plan.points
        look = float(np.clip(0.25 + 0.8 * self.cruise, 0.3, 0.7))
        left_here = self._after[self._seg] - s  # path left from the robot's projection
        for k in range(self._seg + 1, len(p) - 1):
            if _turn_at(p, k) > SHARP:
                corner_left = left_here - self._after[k]
                return (p[k] if corner_left <= look else self._roomy(xy, s, look)), corner_left
        a, b = p[-2], p[-1]
        if self._seg == len(p) - 2 and float((self._last_xy - a) @ (b - a)) > self._seg_len[-1] ** 2:
            return b.copy(), math.inf  # walked past the end, further than OVERSHOOT: back to it
        # Up to the end the aim stays a lookahead ahead, on the line through the end: aiming at the end itself, the
        # bearing swings across as the robot draws near with a few cm to one side, and a gait's turn lags behind.
        return self._roomy(xy, s, look), math.inf

    def _roomy(self, xy: np.ndarray, s: float, look: float) -> np.ndarray:
        """The path point `look` meters ahead, or a nearer one if the straight way to it has less room than the
        robot needs: steering for a point past two gentle corners cuts across both, and on the obstacle course a
        rover cut the corner where the path met a ramp's toe, put a wheel on the ramp's side and stuck there."""
        target = self._point_at(s, look)
        nav = self.planner.last_map
        if nav is None:
            return target
        limit = min(self.planner.body.need, nav_grid.room_at(nav.grid, nav.room, xy))
        while look > MIN_LOOK and nav_grid.room_along(nav.grid, nav.room, xy, target) < limit - 1e-9:
            look /= 2
            target = self._point_at(s, look)
        return target

    def _point_at(self, s: float, ahead: float) -> np.ndarray:
        """The path point `ahead` meters on from `s` along the current segment, on past the end in its direction."""
        p = self.plan.points
        seg, along = self._seg, s + ahead
        while seg < len(p) - 2 and along > self._seg_len[seg]:
            along -= self._seg_len[seg]
            seg += 1
        return p[seg] + (p[seg + 1] - p[seg]) / max(self._seg_len[seg], 1e-9) * along

    def _turn(self, alpha: float) -> float:
        rate = float(np.clip(TURN_GAIN * alpha, -self.turn_rate, self.turn_rate))
        floor = 0.4 * self.turn_rate  # a gait barely turns at a small command
        return math.copysign(max(abs(rate), floor), alpha)

    def _face(self, t: float, xy: np.ndarray, yaw: float) -> None:
        fx, fy = self.plan.face
        alpha = _wrap(math.atan2(fy - xy[1], fx - xy[0]) - yaw)
        if abs(alpha) <= FACE_TOL or not self.turns_in_place or math.hypot(fx - xy[0], fy - xy[1]) < 0.05:
            self.walker.command(0.0, 0.0)
            if self.plan.short:  # as near as it gets, which is not next to it
                self._finish(t, "stopped", self.plan.short, self.plan.note)
            else:
                self._finish(t, "arrived")
            return
        self.walker.command(0.0, self._turn(alpha))

    def _replan(self, t: float, data: mujoco.MjData, why: str, count: bool = True) -> None:
        """Plan again from where the robot is; `count` says whether this one counts toward giving up."""
        if count and self.replans >= MAX_REPLANS:
            x, y = data.xpos[self.planner.root][:2]
            self._finish(t, "stopped", "stuck", f"it got stuck near [{x:.2f}, {y:.2f}]: {why}")
            return
        self.replans += 1 if count else 0
        self.walker.command(0.0, 0.0)
        try:
            self.plan = self.planner.plan(data, self._blockers)
        except NoWay as exc:
            self._finish(t, "stopped", exc.cause, f"{why}, and then {exc}")
            return
        self._reset_path()
        self._since = t
        self.note_others(data)

    def _try_again(self, t: float, data: mujoco.MjData) -> bool:
        """Once per trip, at the nearest free spot to a point that was taken: plan again, and go on if the point
        itself is free now (a robot standing there when the trip began has walked off)."""
        self._retried = True
        try:
            plan = self.planner.plan(data, self._blockers)
        except NoWay:
            return False
        if not plan.exact:
            return False
        self.plan = plan
        self._reset_path()
        self._since = t
        self.note_others(data)
        return True

    def note_others(self, data: mujoco.MjData) -> None:
        """Remember where the other robots are as the path is planned (see _robot_ahead)."""
        self._others_at = {o.name: data.xpos[o.root][:2].copy() for o in self.others}

    def _finish(self, t: float, state: str, cause: str | None = None, note: str | None = None) -> None:
        self.walker.command(0.0, 0.0)
        self.state = state
        self.cause = cause
        if note:
            self.note = note
        elif state == "arrived" and self.plan.note:
            self.note = self.plan.note
        self.end_time = t
        self.on_finish(self)

    def _remaining_path(self) -> list[list[float]]:
        """The path still ahead (the robot's projection first), x, y and the ground height, for the viewer."""
        p, h = self.plan.points, self.plan.heights
        xy = self._last_xy if self._last_xy is not None else p[0]
        k = self._seg + 1
        out = [[round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(h[max(k - 1, 0)]), 3)]]
        out += [[round(float(p[i][0]), 3), round(float(p[i][1]), 3), round(float(h[i]), 3)] for i in range(k, len(p))]
        return out

    # -- the outcome --

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arrived": self.arrived,
            "destination": self.planner.destination.describe(),
            "path_m": round(self.first_length, 2),
            "distance_m": round(self.traveled, 2),
            "time_s": round((self.end_time if self.end_time is not None else math.nan) - self.start_time, 2),
        }
        if self.replans:
            out["replans"] = self.replans
        if self._yields:
            out["made_way"] = self._yields  # times it planned again around another robot in its way
        reach = self.plan.within_reach
        if reach is not None and self._last_xy is not None:  # from where it really stopped, as it faces
            reach = self.planner._reaches(self._last_xy, self._last_yaw)
        if reach is not None:
            out["within_reach"] = reach
        note = self.note
        if reach is False and self.state == "arrived":
            arm, d = self.planner.arm, self.planner.destination
            extra = f"its arm cannot reach {d.name} from there (its arm reaches about {ARM_SHARE * arm.reach:.2f} m)"
            note = f"{note}; {extra}" if note else extra
        if note:
            out["note"] = note
        if self.cause and not self.arrived:
            out["why"] = self.cause
        return out


def _distance_to_polyline(point: np.ndarray, line: np.ndarray) -> float:
    if len(line) == 1:
        return float(np.linalg.norm(point - line[0]))
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + ab * t[:, None] - point, axis=1).min())


def _turn_at(points: np.ndarray, i: int) -> float:
    """How much the path turns at point i (radians, 0 = straight on)."""
    a, b, c = points[i - 1], points[i], points[i + 1]
    u, v = b - a, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return math.acos(float(np.clip(u @ v / (nu * nv), -1.0, 1.0)))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def default_cruise(max_speed: float) -> float:
    return min(CRUISE_SHARE * max_speed, CRUISE_MAX)


def turn_rate_for(kind: str, walker: Any) -> float:
    rate = TURN_RATE.get(kind, DEFAULT_TURN_RATE)
    own = getattr(walker, "max_turn", None)
    return min(rate, float(own)) if own else rate
