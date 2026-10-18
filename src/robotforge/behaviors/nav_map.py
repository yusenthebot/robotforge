"""What blocks a robot on the floor: a 2D map built from the compiled scene, for nav_grid's search.

Every geom that collides is cut into vertical columns, one per grid cell it covers: where it starts and ends
above that cell (exact for boxes, cylinders and spheres at any tilt; other shapes by their bounding box).
A column that starts at the ground is part of the ground there (a stair, a ramp, the foot of a wall); the
ground of a cell is the top of everything standing in it, over the floor and any heightfield. A column that
starts higher hangs over the cell (a table top, a shelf board, a tree crown).

A cell is no place for the robot's center (lethal) when
- the ground steps up from a neighbor by more than the robot climbs: a wall, a counter, a curb too high
  for it (a step lower than the robot is a "step", anything taller an "obstacle"); or inside the cell, from
  one of its quarter points to another (a gap between stepping stones, narrower than a cell);
- a smooth surface (a ramp, a hill) is steeper there than it climbs (legged robots less steep on hills, whose
  slopes a path crosses at any angle), or a bump on a hill stands out by more than half its step;
- something hangs lower over it than the robot is tall, plus a margin, and never lower than MIN_HEADROOM.
The floor is ignored, and so are loose objects light enough to push aside. Other robots and heavy loose
objects block like furniture. The robot's own shape gives the room it needs around its center: half its
width to pass straight through a gap, the distance to its furthest corner to turn on the spot.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors import Kind
from robotforge.behaviors.nav_grid import Grid, clearance, reachable
from robotforge.design import mechanisms
from robotforge.sim.geometry import collides

G = mujoco.mjtGeom

RES = 0.05  # m per cell: a 20 m park is 400 cells across, and 5 cm table legs still show
MAX_CELLS = 700  # per side; a bigger area gets coarser cells (a 60 m world: 8.6 cm)
EDGE = 1.0  # m of floor kept around everything in the scene
# Room the robot keeps from anything lethal beyond half its width, m: a walking robot sways about 2 cm off its
# line and a path follower cuts a little inside a bend.
MARGIN = 0.05
CLEARANCE = 0.05  # m above the robot's top that something hanging over it must leave
# Robots do not go under anything lower than this, m, however small they are: the 7 cm rover fits under a
# park bench (seat 0.45 m) or into a shelf (boards every 0.4 m), gets stuck there, and nobody sent it to the
# shelf to crawl inside it. Tables (0.72 to 0.75 m) stay open to small robots.
MIN_HEADROOM = 0.6
# A loose object weighing at most this share of a wheeled robot is pushed aside, not driven around. Legged robots
# go round every loose thing on the floor: the template dog walked into a 0.3 kg ball in the park and fell.
PUSHABLE = 0.5
TILTED = 1e-3  # a geom whose top face leans more than this (1 - cos) is a slope, not a flat step

# Why a cell is lethal; where several reasons hold, the higher number is kept (a wall on a slope is a wall).
FREE, SLOPE, STEP, OBSTACLE = 0, 1, 2, 3
# A jump in the ground no higher than this share of the robot's height (and at most STEP_LIKE) is a step too
# high for it (a curb, a stair, a platform's edge); a taller one is an obstacle (a wall, a counter, a pallet).
STEP_SHARE, STEP_LIKE = 0.5, 0.15

# Highest step and steepest slope by kind, from runs on our robots (2026-10-17: driven straight at
# 0.8 x max_speed, at most 0.5 m/s, at curbs and up ramps in lanes):
# - template rover (wheels r 3.5 cm): curb 1.0 cm crossed, 1.5 cm stopped; ramp 15 degrees climbed, 18 stalled;
# - a rover built from parts with 2 cm casters: curb 0.5 cm crossed, 1.0 cm stopped;
# - template dog (gait lift 4.3 cm): curb 5 cm crossed, 6 cm stopped; ramp 15 degrees, 18 stalled;
# - Unitree Go2 on the gait (lift 7.1 cm): curb 5 cm crossed, 7 cm stopped; ramp 10 degrees, 12 fell over;
# - template humanoid: curb 0.5 cm crossed, 1 cm fell over; ramp 8 degrees, 10 fell over. Walking along the park's
#   hills it fell where they leaned 4 degrees across its way (scripts/nav_bench.py), so it keeps to 3.
# And with the follower holding the heading up 3 m wide ramps (no falling off a side): the rover built from parts
# climbed 3 and 7 degrees and stuck at the foot of 5, 9 and 12, its front caster caught on the ramp's toe.
# One curb is not a flight of stairs. Sent by the follower over three steps up to a landing and three down (1.6 m
# wide, 0.3 m treads), straight on and from 0.3 to 0.6 m to one side: the template dog climbed 3 and 3.5 cm steps
# every time and fell once in three on 4 cm (and stuck on the obstacle course's 4 cm stairs in scripts/nav_bench.py);
# the Go2 climbed 2.5 cm every time, fell once in three on 3 cm and every time on 3.5 and 4 cm.
WHEEL_STEP = 0.3  # wheels climb a step of this share of the smallest rolling radius (wheel or caster)
LEG_STEP_SHARE = 0.7  # of the gait's foot lift: the dog's 3.5 cm stairs were 0.82 of its 4.3 cm lift
LEG_STEP_MAX = 0.025  # m, for any legged robot: the Go2's stairs, whatever its 7.1 cm lift
FLAT_STEP = 0.005  # m: robots that only walk on flat ground (humanoids, snakes, learned rhythm walks)
SLOPE_DEG = {
    Kind.ROVER: 15.0,
    # The Go2 climbed 10 degrees straight up 1.5 m ramps at speed (and fell on 12), but on the obstacle course's
    # 10 degree ramp, 1 m slopes, slowing for a point near the top, it stalled and slid back in 4 of 8 start headings
    # (2 of 34 trips in scripts/nav_bench.py): 9 keeps a margin. The template dog climbs 15.
    Kind.QUADRUPED: 9.0,
    Kind.HEXAPOD: 9.0,
    Kind.HUMANOID: 3.0,
    Kind.BALANCER: 5.0,  # not measured: leaning to balance on a slope is hard, so kept low
    Kind.SNAKE: 5.0,  # not measured
}
OTHER_SLOPE_DEG = 5.0  # learned rhythm walks and anything else: not measured
# Up a ramp a path goes straight up its slope; on hills it crosses slopes at any angle, and a gait slides down a
# slope to its side. Sent by the follower across a ramp's face (2026-10-17): at 10 degrees the template dog fell
# and the Go2 fell; at 7.5 the Go2 fell when it turned on the spot first and the dog slid off its path and planned
# again four times; at 5 both walked across. On the obstacle course's bumps (slopes up to 10 degrees) both slid
# off their paths near the top and gave up (scripts/nav_bench.py).
HILL_SLOPE_DEG = {Kind.QUADRUPED: 5.0, Kind.HEXAPOD: 5.0}
CASTER_SLOPE_DEG = 3.0  # wheeled robots with a caster ahead of the drive wheels (see above)
# m beyond the robot's furthest corner, around a ramp, where it goes only straight up or down the ramp (ramp_aim):
# a gait slowing for a corner steps a few cm past it before it turns there.
RUN_UP = 0.1


@dataclass(frozen=True)
class Body:
    """What the planner needs to know about a robot, measured from its shapes where it stands."""

    radius: float  # m, half its width: the room it needs beside its center to go straight through a gap
    reach: float  # m, from its center to its furthest corner: the room it needs to turn on the spot
    front: float  # m, from its center forward to its front
    height: float  # m, from its feet to its top
    mass: float  # kg
    step: float  # m, the highest step it climbs
    slope: float  # rad, the steepest slope it climbs
    pushes: bool = False  # it rolls, and pushes light loose objects aside (see PUSHABLE)
    hill_slope: float | None = None  # rad, the steepest hill (a heightfield) it crosses at any angle; None: slope

    @property
    def hills(self) -> float:
        return self.slope if self.hill_slope is None else min(self.hill_slope, self.slope)

    def slope_words(self) -> str:
        """The steepest slope it climbs, in words (degrees)."""
        up = f"{math.degrees(self.slope):.0f} degrees"
        return up if self.hills == self.slope else f"{up} up a ramp, {math.degrees(self.hills):.0f} on hills"

    @property
    def need(self) -> float:
        """Room (m) a cell needs around it for the robot's center to be there."""
        return self.radius + MARGIN

    @property
    def headroom(self) -> float:
        return max(self.height + CLEARANCE, MIN_HEADROOM)


def measure_body(model: mujoco.MjModel, data: mujoco.MjData, robot: str, kind: Kind, walker: object) -> Body:
    """The robot's footprint (in its own heading frame), height, mass and what it climbs with `walker` (the
    controller that moves it: a legged gait's foot lift and a rover's wheels say what it steps over)."""
    root = root_body(model, robot)
    rot = data.xmat[root].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    heading = np.array([[math.cos(yaw), math.sin(yaw), 0.0], [-math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
    center = data.xpos[root]
    boxes = [_extent(model, data, g, heading, center) for g in robot_geoms(model, robot) if collides(model, g)]
    if not boxes:
        raise ValueError(f"{robot} has no shapes that collide, so there is nothing to plan a path around")
    low = np.min([b[0] for b in boxes], axis=0)
    high = np.max([b[1] for b in boxes], axis=0)
    radius = float(max(-low[1], high[1]))
    reach = float(max(math.hypot(x, y) for x in (low[0], high[0]) for y in (low[1], high[1])))
    lift = getattr(walker, "lift", None)
    slope_deg = SLOPE_DEG.get(kind, OTHER_SLOPE_DEG)
    if kind in (Kind.ROVER, Kind.BALANCER):
        radius_rolling, caster = _rolling_radius(model, data, robot, walker, heading, center)
        step = WHEEL_STEP * radius_rolling
        if caster:
            slope_deg = min(slope_deg, CASTER_SLOPE_DEG)
    elif kind in (Kind.QUADRUPED, Kind.HEXAPOD) and isinstance(lift, float):
        step = min(LEG_STEP_SHARE * lift, LEG_STEP_MAX)
    else:
        step = FLAT_STEP
    slope = math.radians(slope_deg)
    mass = float(model.body_subtreemass[root])
    pushes = kind in (Kind.ROVER, Kind.BALANCER)
    hill = math.radians(HILL_SLOPE_DEG[kind]) if kind in HILL_SLOPE_DEG else None
    return Body(radius, reach, float(high[0]), float(high[2] - low[2]), mass, max(step, 0.003), slope, pushes, hill)


def with_load(model: mujoco.MjModel, data: mujoco.MjData, robot: str, body: Body, load: int) -> Body:
    """`body` grown to take in a load the robot carries (the body `load`, held in its gripper), where it sticks out
    beyond the robot: its width, its corners, its front and its top. Its mass is left out: it does not change
    what the robot climbs or pushes aside."""
    root = root_body(model, robot)
    rot = data.xmat[root].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    heading = np.array([[math.cos(yaw), math.sin(yaw), 0.0], [-math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]])
    center = data.xpos[root]
    root_of = model.body_rootid[model.geom_bodyid]
    geoms = [g for g in range(model.ngeom) if int(root_of[g]) == load and collides(model, g)]
    if not geoms:
        return body
    boxes = [_extent(model, data, g, heading, center) for g in geoms]
    low, high = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
    radius = max(body.radius, float(max(-low[1], high[1])))
    reach = max(body.reach, *(math.hypot(x, y) for x in (low[0], high[0]) for y in (low[1], high[1])))
    own = [_extent(model, data, g, heading, center) for g in robot_geoms(model, robot) if collides(model, g)]
    bottom = min(b[0][2] for b in own) if own else float(low[2])  # its height counts from its lowest point
    return dataclasses.replace(
        body,
        radius=radius,
        reach=float(reach),
        front=max(body.front, float(high[0])),
        height=max(body.height, float(high[2]) - float(bottom)),
    )


def footprint_radius(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> float:
    """How far a robot reaches from its middle (its root) on the floor, m."""
    center = data.xpos[root_body(model, robot)]
    reach = 0.0
    for g in robot_geoms(model, robot):
        if collides(model, g):
            low, high = world_extent(model, data, g)
            for x in (low[0], high[0]):
                for y in (low[1], high[1]):
                    reach = max(reach, math.hypot(x - center[0], y - center[1]))
    return reach


def root_body(model: mujoco.MjModel, robot: str) -> int:
    for b in range(1, model.nbody):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/"):
            return b
    raise ValueError(f"no robot named {robot!r}")


def robot_geoms(model: mujoco.MjModel, robot: str) -> list[int]:
    bodies = {b for b in range(1, model.nbody) if _body_name(model, b).startswith(f"{robot}/")}
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def _rolling_radius(
    model, data, robot: str, walker: object, heading: np.ndarray, center: np.ndarray
) -> tuple[float, bool]:
    """What sets the step a wheeled robot climbs: its drive wheels, or a smaller caster ahead of them; and whether
    it has a caster ahead. A caster behind is dragged up after the wheels: the template rover (trailing caster
    r 1.6 cm) crossed 1 cm, while a rover from parts with a 2 cm caster in front stopped at 1 cm."""
    wheel = float(getattr(walker, "radius", 0.0) or 0.0)
    boxes = {g: _extent(model, data, g, heading, center) for g in robot_geoms(model, robot) if collides(model, g)}
    if not boxes:
        return wheel, False
    bottom = min(b[0][2] for b in boxes.values())
    axle = float(data.xpos[root_body(model, robot)][0])  # fallback when the walker does not say
    wheels = [*getattr(walker, "left", ()), *getattr(walker, "right", ())]
    joints = [int(model.actuator_trnid[a, 0]) for a in wheels]
    if joints:
        axle = float(np.mean([(heading @ (data.xanchor[j] - center))[0] for j in joints]))
    # The drive wheels are not casters, however the robot leans: tilted on the obstacle course's bumps, the template
    # rover's lower wheel stood 1.5 mm ahead of the axle in its heading frame, read as a caster in front, and the
    # rover was given a caster's 3 degree slope limit and found itself boxed in on the slope it stood on.
    driven = {int(model.jnt_bodyid[j]) for j in joints}
    round_types = (G.mjGEOM_SPHERE, G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID)
    ahead = [
        float(model.geom_size[g][0])
        for g, (low, high) in boxes.items()
        if low[2] - bottom < 0.005
        and int(model.geom_type[g]) in round_types
        and int(model.geom_bodyid[g]) not in driven
        and (low[0] + high[0]) / 2 > axle + 1e-3
    ]
    radii = [r for r in (wheel, *ahead) if r > 0]
    return (min(radii) if radii else 0.0), bool(ahead)


def _extent(model, data, g: int, frame: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The low and high corner of a geom's bounding box in a frame (rows: its axes in world coordinates; at
    `origin`): exact for spheres, cylinders, capsules and boxes, whatever way they are turned (a spinning wheel
    keeps its size), the turned bounding box for anything else."""
    rot = frame @ data.geom_xmat[g].reshape(3, 3)
    pos = frame @ (data.geom_xpos[g] - origin)
    size = model.geom_size[g]
    kind = int(model.geom_type[g])
    if kind == G.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif kind in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        axis = rot[:, 2]
        across = np.sqrt(np.clip(1.0 - axis**2, 0.0, 1.0))
        half = size[1] * np.abs(axis) + (size[0] if kind == G.mjGEOM_CAPSULE else size[0] * across)
    elif kind == G.mjGEOM_BOX:
        half = np.abs(rot) @ size[:3]
    else:
        pos = pos + rot @ model.geom_aabb[g, :3]
        half = np.abs(rot) @ model.geom_aabb[g, 3:]
    return pos - half, pos + half


def world_extent(model, data, g: int) -> tuple[np.ndarray, np.ndarray]:
    return _extent(model, data, g, np.eye(3), np.zeros(3))


def _body_name(model: mujoco.MjModel, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""


# ---- columns: where a geom is over each cell ------------------------------------------------------------


@dataclass
class Columns:
    """The part of one geom over the cells it covers: flat cell indices and the bottom and top of the geom
    above each (m)."""

    cells: np.ndarray
    low: np.ndarray
    high: np.ndarray
    owner: int  # index into the map's names
    tilted: bool  # its top leans (a ramp slab): its edges are a slope, not a step
    axis: float | None = None  # rad, the way up its top (a ramp slab), seen from above


def geom_columns(
    model: mujoco.MjModel, data: mujoco.MjData, g: int, grid: Grid, owner: int, pad: float | None = None
) -> Columns | None:
    """Columns of geom `g` over every cell whose square it overlaps (cells are tested at their center against the
    geom grown by half a cell sideways, which is exact for an upright box); with `pad` given, grown by that much
    instead (0: the cells whose center it covers)."""
    pad = grid.res / 2 if pad is None else pad
    low3, high3 = world_extent(model, data, g)
    lo_xy, hi_xy = low3[:2] - pad, high3[:2] + pad
    r0, c0 = grid.cell(lo_xy[0], lo_xy[1])
    r1, c1 = grid.cell(hi_xy[0], hi_xy[1])
    r0, c0, r1, c1 = max(r0, 0), max(c0, 0), min(r1, grid.rows - 1), min(c1, grid.cols - 1)
    if r0 > r1 or c0 > c1:
        return None
    rows, cols = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
    x = grid.x0 + cols.ravel() * grid.res
    y = grid.y0 + rows.ravel() * grid.res
    pos = data.geom_xpos[g]
    rot = data.geom_xmat[g].reshape(3, 3)
    size = model.geom_size[g]
    kind = int(model.geom_type[g])
    upright = abs(rot[2, 2]) > 1 - TILTED
    dx, dy = x - pos[0], y - pos[1]
    if kind == G.mjGEOM_SPHERE or (kind in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE) and upright):
        rho = np.hypot(dx, dy)
        inside = rho <= size[0] + pad
        cap = np.sqrt(np.maximum(0.0, size[0] ** 2 - np.minimum(rho, size[0]) ** 2))
        half = cap if kind == G.mjGEOM_SPHERE else size[1] + (cap if kind == G.mjGEOM_CAPSULE else 0.0)
        low, high = pos[2] - half, pos[2] + half
        tilted = False
    else:
        if kind == G.mjGEOM_BOX:
            center, half_size = pos, np.asarray(size[:3], float)
        else:  # meshes, ellipsoids, tilted cylinders: their bounding box, a little more than the shape
            center, half_size = pos + rot @ model.geom_aabb[g, :3], np.asarray(model.geom_aabb[g, 3:], float)
        low, high, inside = _slab(x - center[0], y - center[1], center[2], rot, half_size, pad)
        tilted = not upright and kind == G.mjGEOM_BOX
    keep = inside & (high >= low)
    if not keep.any():
        return None
    cells = rows.ravel()[keep] * grid.cols + cols.ravel()[keep]
    low, high = np.broadcast_to(low, keep.shape)[keep], np.broadcast_to(high, keep.shape)[keep]
    return Columns(cells, low, high, owner, tilted, _uphill(rot) if tilted else None)


def _uphill(rot: np.ndarray) -> float | None:
    """The way up a tilted box's top face, seen from above (rad): its face that looks most upward leans downhill."""
    i = int(np.argmax(np.abs(rot[2, :])))
    up = rot[:, i] * math.copysign(1.0, rot[2, i])
    if math.hypot(up[0], up[1]) < 1e-9:
        return None
    return math.atan2(-up[1], -up[0])


def _slab(
    dx: np.ndarray, dy: np.ndarray, cz: float, rot: np.ndarray, half: np.ndarray, pad: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where the vertical line through each (dx, dy) (from the box center) enters and leaves an oriented box,
    whose half sizes are grown by `pad` along the horizontal part of each axis (slab method)."""
    low = np.full(dx.shape, -np.inf)
    high = np.full(dx.shape, np.inf)
    inside = np.ones(dx.shape, bool)
    for i in range(3):
        u = rot[:, i]
        h = half[i] + pad * math.hypot(u[0], u[1])
        a = u[0] * dx + u[1] * dy
        if abs(u[2]) < 1e-9:
            inside &= np.abs(a) <= h
            continue
        z1 = cz + (-h - a) / u[2]
        z2 = cz + (h - a) / u[2]
        low = np.maximum(low, np.minimum(z1, z2))
        high = np.minimum(high, np.maximum(z1, z2))
    return low, high, inside


# ---- the map ----------------------------------------------------------------------------------------------


@dataclass
class NavMap:
    """Where a robot's center may go: ground heights, lethal cells and why, and the room around each cell."""

    grid: Grid
    body: Body
    ground: np.ndarray  # m, height of the ground the robot stands on in each cell
    lethal: np.ndarray  # int8 cause (FREE, OBSTACLE, STEP, SLOPE)
    owner: np.ndarray  # int32 index into names of what makes a cell lethal, -1 for none
    names: list[str]
    floor: np.ndarray  # m, the floor and heightfields alone, under whatever stands on them
    # rad, on and near a ramp: the way up it, along which alone the robot moves there (NaN elsewhere; see ramp_aim)
    aim: np.ndarray | None = None
    room: np.ndarray = field(init=False)  # m from each cell center to the nearest lethal cell's edge
    free: np.ndarray = field(init=False)  # cells where the robot's center fits and that it can get to

    def __post_init__(self) -> None:
        self.room = clearance(self.lethal != FREE, self.grid.res, self.body.reach + 0.3)
        fits = self.room >= self.body.need
        # The top of a bench, a crate or a wide wall is level, with room on it, and no way up: only what joins the
        # floor counts (a ramp's landing does, up its ramp). A robot standing up there keeps its own patch.
        floor_level = fits & (self.ground - self.floor <= self.body.step)
        self.free = reachable(self.grid, fits, floor_level)
        self._fits = fits

    def free_from(self, start: tuple[int, int]) -> None:
        """Count as free too what the robot reaches from where it stands (it may stand on a platform)."""
        if self.grid.inside(*start) and self._fits[start] and not self.free[start]:
            self.free = self.free | reachable(self.grid, self._fits, start)

    def penalty(self) -> np.ndarray:
        """Extra cost of a cell (share of a step's length): 0 with room to turn on the spot, rising to 2 at the
        edge of what fits, so paths keep to the middle of corridors and stay off walls where they can."""
        roomy = self.body.reach + 0.1
        short = np.clip((roomy - self.room) / max(roomy - self.body.need, 1e-6), 0.0, 1.0)
        return 2.0 * short**2

    def ground_at(self, x: float, y: float) -> float:
        r, c = self.grid.cell(x, y)
        r, c = min(max(r, 0), self.grid.rows - 1), min(max(c, 0), self.grid.cols - 1)
        return float(self.ground[r, c])

    def blocker(self, cell: tuple[int, int]) -> str | None:
        """The name of what makes a cell lethal, if anything does."""
        k = int(self.owner[cell])
        return self.names[k] if k >= 0 else None


class StaticLayer:
    """The columns of everything that never moves (walls, furniture, terrain) for one robot in one compiled
    world, kept between plans: only loose objects and other robots are measured afresh. `quarters` holds the same
    columns at the four quarter points of every cell (on quarter_grid), for what changes inside a cell."""

    def __init__(
        self,
        grid: Grid,
        floor: np.ndarray,
        smooth_floor: np.ndarray,
        floor_owner: np.ndarray,
        columns: list[Columns],
        names: list[str],
        quarters: list[Columns] = (),
    ):
        self.grid, self.floor, self.smooth_floor, self.floor_owner = grid, floor, smooth_floor, floor_owner
        self.columns, self.names, self.quarters = columns, names, list(quarters)


def quarter_grid(grid: Grid) -> Grid:
    """Cells half as wide, centered on the four quarter points of every cell of `grid`: cell (r, c) holds those
    of (2r .. 2r + 1, 2c .. 2c + 1)."""
    return Grid(grid.x0 - grid.res / 4, grid.y0 - grid.res / 4, grid.res / 2, 2 * grid.rows, 2 * grid.cols)


def bounds(model: mujoco.MjModel, data: mujoco.MjData, robot: str, points: list[tuple[float, float]]):
    """The floor area worth mapping: everything that collides (but the floor plane), the robot and `points`."""
    own = set(robot_geoms(model, robot))
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    for g in range(model.ngeom):
        if g in own or not collides(model, g) or int(model.geom_type[g]) == G.mjGEOM_PLANE:
            continue
        low, high = world_extent(model, data, g)
        xs += [float(low[0]), float(high[0])]
        ys += [float(low[1]), float(high[1])]
    low = (min(xs) - EDGE, min(ys) - EDGE)
    high = (max(xs) + EDGE, max(ys) + EDGE)
    span = max(high[0] - low[0], high[1] - low[1])
    res = max(RES, span / MAX_CELLS)
    return Grid.around(low, high, res)


def static_layer(model: mujoco.MjModel, data: mujoco.MjData, robot: str, grid: Grid) -> StaticLayer:
    """Floor heights (plane and heightfields) and the columns of every geom welded to the world."""
    names: list[str] = []
    floor = np.full(grid.rows * grid.cols, -np.inf)
    smooth = np.zeros(grid.rows * grid.cols, bool)
    floor_owner = np.full(grid.rows * grid.cols, -1, np.int32)  # the heightfield a cell's floor is, by name
    xs, ys = grid.centers()
    own = set(robot_geoms(model, robot))
    columns, quarters = [], []
    fine = quarter_grid(grid)
    for g in range(model.ngeom):
        if g in own or not collides(model, g):
            continue
        body = int(model.geom_bodyid[g])
        if int(model.body_weldid[body]) != 0:
            continue  # it moves: measured afresh for every plan (dynamic_columns)
        kind = int(model.geom_type[g])
        if kind == G.mjGEOM_PLANE:
            if data.geom_xmat[g][8] > 1 - TILTED:  # a floor, not a tilted plane
                floor = np.maximum(floor, float(data.geom_xpos[g][2]))
            continue
        if kind == G.mjGEOM_HFIELD:
            heights = _hfield_heights(model, data, g, xs.ravel(), ys.ravel())
            covered = np.isfinite(heights) & (heights >= floor)
            floor = np.where(covered, heights, floor)
            smooth |= covered
            floor_owner[covered] = _name_index(names, _owner(model, g))
            continue
        owner = _name_index(names, _owner(model, g))
        found = geom_columns(model, data, g, grid, owner)
        if found is not None:
            columns.append(found)
            quartered = geom_columns(model, data, g, fine, owner, pad=0.0)
            if quartered is not None:
                quarters.append(quartered)
    floor = np.where(np.isfinite(floor), floor, 0.0)
    return StaticLayer(grid, floor, smooth, floor_owner, columns, names, quarters)


def dynamic_columns(
    model: mujoco.MjModel, data: mujoco.MjData, robot: str, body: Body, grid: Grid, names: list[str], skip=()
) -> list[Columns]:
    """Columns of what moves: other robots, and loose objects the robot does not push aside (see PUSHABLE).
    Geoms of bodies in `skip` are left out."""
    own = set(robot_geoms(model, robot))
    columns = []
    for g in range(model.ngeom):
        if g in own or not collides(model, g):
            continue
        b = int(model.geom_bodyid[g])
        if int(model.body_weldid[b]) == 0 or int(model.body_rootid[b]) in skip:
            continue
        if mechanisms.is_door_leaf(model, b):
            continue  # a door that swings both ways: pushed open going through, not gone round
        root = int(model.body_rootid[b])
        light = float(model.body_subtreemass[root]) <= PUSHABLE * body.mass
        if body.pushes and light and "/" not in _body_name(model, root):
            continue  # a loose object it can push aside
        found = geom_columns(model, data, g, grid, _name_index(names, _owner(model, g)))
        if found is not None:
            columns.append(found)
    return columns


def build(
    layer: StaticLayer,
    body: Body,
    extra: list[Columns],
    blocked: np.ndarray | None = None,
    names: list[str] | None = None,
) -> NavMap:
    """The map for `body` from the static layer, the moving things (`extra`, whose owners index `names`, the
    layer's names and more) and cells marked by hand (`blocked`, e.g. where the robot got stuck)."""
    grid = layer.grid
    n = grid.rows * grid.cols
    floor = layer.floor
    columns = [*layer.columns, *extra]
    ground = floor.copy()
    rooted = []
    hanging = []
    slabs = []  # (cells, way up) of the ramps' tops
    for col in columns:
        # What starts lower than the robot can pass under is as good as standing on the ground: the robot can
        # only be on top of it. A ramp is a tilted plank whose underside lifts off the floor, a bench seat or a
        # shelf board leaves a gap too low to use; their tops are ground, judged by step and slope.
        stands = col.low < floor[col.cells] + body.headroom
        if stands.any():
            rooted.append((col, stands))
        if not stands.all():
            hanging.append((col, ~stands))
    for col, stands in rooted:
        np.maximum.at(ground, col.cells[stands], col.high[stands])
    on_floor = np.abs(ground - floor) < 1e-6
    owner = np.where(on_floor, layer.floor_owner, -1).astype(np.int32)
    smooth = layer.smooth_floor & on_floor
    for col, stands in rooted:
        cells = col.cells[stands]
        top = col.high[stands] >= ground[cells] - 1e-6
        owner[cells[top]] = col.owner
        if col.tilted:
            smooth[cells[top]] = True
            if col.axis is not None:
                slabs.append((cells[top], col.axis))
    lethal = np.zeros(n, np.int8)
    for col, hangs in hanging:
        cells = col.cells[hangs]
        low, high = col.low[hangs], col.high[hangs]
        over = (low < ground[cells] + body.headroom) & (high > ground[cells] + body.step)
        lethal[cells[over]] = OBSTACLE
        owner[cells[over]] = col.owner
    ground2 = ground.reshape(grid.shape)
    _mark_uneven(layer, body, smooth.reshape(grid.shape), lethal.reshape(grid.shape))
    _mark_steps(ground2, smooth.reshape(grid.shape), body, grid.res, lethal.reshape(grid.shape))
    hills = (layer.smooth_floor & on_floor).reshape(grid.shape)
    _mark_slopes(ground2, smooth.reshape(grid.shape), body, grid.res, lethal.reshape(grid.shape), hills)
    lethal2 = lethal.reshape(grid.shape)
    if blocked is not None:
        lethal2[blocked] = OBSTACLE
    aim = ramp_aim(grid, [(cells[lethal[cells] == FREE], axis) for cells, axis in slabs], body.reach + RUN_UP)
    return NavMap(
        grid,
        body,
        ground2,
        lethal2,
        owner.reshape(grid.shape),
        list(names or layer.names),
        floor.reshape(grid.shape),
        aim,
    )


def ramp_aim(grid: Grid, slabs: list[tuple[np.ndarray, float]], near: float) -> np.ndarray | None:
    """The way up the nearest ramp (rad) in every cell whose center is within `near` (m) of a ramp's top it can stand
    on, NaN elsewhere; None without ramps. There a robot goes only straight up or down the ramp: sent up the
    warehouse dock (8 degrees) at an angle, round its toe and turning on it, the Go2 swung round, slid down sideways
    and fell or stuck, 6 and 8 times in two runs of 28 trips (scripts/nav_bench.py --case dock). Near the ramp too, so
    that it has lined up before its feet reach the toe, and is all the way onto the landing before it turns."""
    slabs = [(cells, axis) for cells, axis in slabs if len(cells)]
    if not slabs:
        return None
    aim = np.full(grid.shape, np.nan)
    best = np.full(grid.shape, np.inf)
    pad = math.ceil(near / grid.res) + 2
    for cells, axis in slabs:
        rows, cols = np.unravel_index(cells, grid.shape)
        r0, r1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, grid.rows)
        c0, c1 = max(int(cols.min()) - pad, 0), min(int(cols.max()) + pad + 1, grid.cols)
        mask = np.zeros((r1 - r0, c1 - c0), bool)
        mask[rows - r0, cols - c0] = True
        away = clearance(mask, grid.res, near + grid.res)
        nearer = (away <= near) & (away < best[r0:r1, c0:c1])
        best[r0:r1, c0:c1][nearer] = away[nearer]
        aim[r0:r1, c0:c1][nearer] = axis
    return aim


def _mark_steps(ground: np.ndarray, smooth: np.ndarray, body: Body, res: float, lethal: np.ndarray) -> None:
    """Mark the higher cell of every pair of side-by-side cells whose ground differs by more than the robot
    climbs. Where either cell is on a smooth surface (a ramp, a hill, or its foot on the floor) the slope itself
    is allowed for too: a 15 degree ramp rises 1.3 cm per 5 cm cell, more than the rover's whole 1 cm step, and
    its first cell read as a 2 cm step up from the floor."""
    rise = res * math.tan(body.slope)
    for axis in (0, 1):
        a = ground.take(range(0, ground.shape[axis] - 1), axis=axis)
        b = ground.take(range(1, ground.shape[axis]), axis=axis)
        either_smooth = smooth.take(range(0, ground.shape[axis] - 1), axis=axis) | smooth.take(
            range(1, ground.shape[axis]), axis=axis
        )
        limit = body.step + np.where(either_smooth, rise, 0.0)
        jump = np.abs(a - b) > limit
        tall = np.abs(a - b) > min(STEP_SHARE * body.height, STEP_LIKE)
        cause = np.where(tall, OBSTACLE, STEP).astype(np.int8)
        first = jump & (a > b)
        second = jump & (b >= a)
        view_a = lethal[:-1, :] if axis == 0 else lethal[:, :-1]
        view_b = lethal[1:, :] if axis == 0 else lethal[:, 1:]
        np.copyto(view_a, np.maximum(view_a, cause), where=first)
        np.copyto(view_b, np.maximum(view_b, cause), where=second)


def _mark_uneven(layer: StaticLayer, body: Body, smooth: np.ndarray, lethal: np.ndarray) -> None:
    """Mark cells whose ground differs inside them by more than the robot climbs, from the ground at their four
    quarter points (2.5 cm apart on 5 cm cells). A cell's ground is the top of everything over any part of it, so
    gaps narrower than a cell vanish from it: the obstacle course's stepping stones, 4 cm apart, read as one
    platform, and the template dog walked onto them, put a foot in a gap and fell (scripts/nav_bench.py)."""
    grid = layer.grid
    floor = layer.floor.reshape(grid.shape)
    quarter = np.repeat(np.repeat(floor, 2, axis=0), 2, axis=1).ravel()  # the floor under each quarter point
    under = quarter.copy()
    for col in layer.quarters:
        stands = col.low < under[col.cells] + body.headroom  # as in build: what it cannot pass under is ground
        np.maximum.at(quarter, col.cells[stands], col.high[stands])
    four = quarter.reshape(grid.rows, 2, grid.cols, 2)
    spread = four.max(axis=(1, 3)) - four.min(axis=(1, 3))
    limit = body.step + np.where(smooth, grid.res * math.tan(body.slope), 0.0)  # a ramp rises inside a cell too
    uneven = spread > limit
    tall = spread > min(STEP_SHARE * body.height, STEP_LIKE)
    np.copyto(lethal, np.maximum(lethal, np.where(tall, OBSTACLE, STEP).astype(np.int8)), where=uneven)


def _mark_slopes(
    ground: np.ndarray, smooth: np.ndarray, body: Body, res: float, lethal: np.ndarray, hills: np.ndarray | None = None
) -> None:
    """Mark smooth cells steeper than the robot climbs, measured over about 20 cm (a robot's stride, so one bump
    of rough ground is not a slope) on the smooth cells only: flat steps and walls have no slope, only edges. On
    hills (heightfields, `hills`) the limit is the robot's hill slope, and a bump standing out from the ground
    around it by more than half the robot's step is a step: the template humanoid (0.5 cm) fell twice on the
    rim of the obstacle course's rough patch, 8 mm bumps on a slope of 3 degrees measured over 20 cm."""
    k = max(1, round(0.1 / res))
    ok = smooth & (lethal == FREE)
    weight = _box_sum(ok.astype(float), k)
    level = np.where(weight > 0, _box_sum(np.where(ok, ground, 0.0), k) / np.maximum(weight, 1e-9), ground)
    gy, gx = np.gradient(level, res)
    hills = np.zeros(ground.shape, bool) if hills is None else hills
    # A slope as steep as the limit is within it: the obstacle course's 10 degree ramp read 10.00 degrees, and half
    # its cells, a hair over in floating point, were too steep for the dog and the Go2, which climb 10.
    limit = np.where(hills, math.tan(body.hills), math.tan(body.slope)) * (1 + 1e-6)
    steep = smooth & (np.hypot(gx, gy) > limit) & (lethal == FREE)
    bumpy = hills & (np.abs(ground - level) > body.step / 2) & (lethal == FREE) & ~steep
    lethal[steep] = SLOPE
    lethal[bumpy] = STEP


def _box_sum(values: np.ndarray, k: int) -> np.ndarray:
    """Sum over the (2k+1) x (2k+1) window around each cell (zeros beyond the edge)."""
    c = np.cumsum(np.cumsum(np.pad(values, ((k + 1, k), (k + 1, k))), axis=0), axis=1)
    n = 2 * k + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def _hfield_heights(model: mujoco.MjModel, data: mujoco.MjData, g: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The heightfield's surface over each point (bilinear between its samples), inf where it does not reach."""
    h = int(model.geom_dataid[g])
    nrow, ncol = int(model.hfield_nrow[h]), int(model.hfield_ncol[h])
    rx, ry, top, _ = (float(v) for v in model.hfield_size[h])
    elevation = np.asarray(model.hfield_data[int(model.hfield_adr[h]) : int(model.hfield_adr[h]) + nrow * ncol])
    elevation = elevation.reshape(nrow, ncol)
    pos = data.geom_xpos[g]
    rot = data.geom_xmat[g].reshape(3, 3)
    lx = (x - pos[0]) * rot[0, 0] + (y - pos[1]) * rot[1, 0]
    ly = (x - pos[0]) * rot[0, 1] + (y - pos[1]) * rot[1, 1]
    u = (lx + rx) / (2 * rx) * (ncol - 1)
    v = (ly + ry) / (2 * ry) * (nrow - 1)
    inside = (u >= 0) & (u <= ncol - 1) & (v >= 0) & (v <= nrow - 1)
    u, v = np.clip(u, 0, ncol - 1), np.clip(v, 0, nrow - 1)
    i0, j0 = np.minimum(v.astype(int), nrow - 2), np.minimum(u.astype(int), ncol - 2)
    fv, fu = v - i0, u - j0
    z = (
        elevation[i0, j0] * (1 - fu) * (1 - fv)
        + elevation[i0, j0 + 1] * fu * (1 - fv)
        + elevation[i0 + 1, j0] * (1 - fu) * fv
        + elevation[i0 + 1, j0 + 1] * fu * fv
    )
    return np.where(inside, pos[2] + z * top, np.inf)


def _owner(model: mujoco.MjModel, g: int) -> str:
    """What a geom belongs to, by the name an agent knows: the object or robot, or the terrain patch."""
    b = int(model.geom_bodyid[g])
    if b != 0:
        name = _body_name(model, int(model.body_rootid[b])) or _body_name(model, b)
        return name.split("/", 1)[0] or f"body {b}"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom {g}"
    return name.split("/", 1)[0]


def _name_index(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError:
        names.append(name)
        return len(names) - 1
