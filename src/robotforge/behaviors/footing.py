"""Where a walking robot's feet go on uneven ground, from what it sees of the ground (ground_scan.py).

The trot and tripod gaits (quadruped.py) plan feet as if the ground were flat and level. On ground that is not,
`Footing` corrects each foot target in the trunk frame:
- the trunk follows the slope it stands on: a plane is fitted through where the feet stand and will land, and the
  gait pitches the trunk with half of that plane's slope instead of leveling it to the horizon (across the slope
  it stays level);
- each foot is placed at the ground's real height under it, off that plane (on stairs: a step up or down);
- a swinging foot lifts higher when something higher than both its ends lies under its way (a riser, a stone);
- a foot about to land on an edge (a stair's nosing, the side of a stone) lands a little before or after it
  instead, where the ground around the foot is flat: the foothold is chosen from heights seen along the way.

Heights of where a swinging foot will land are looked at TERRAIN_HZ times a second while it swings, and frozen
for the last part of the swing, so a foot does not chase a moving target as it comes down. A foot that lands is
measured where it really is (on what it touches), and stands there until it lifts or slides off an edge.

Everything is in the trunk's heading frame (turned with the trunk's yaw only) about the trunk origin, in meters.
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from robotforge.behaviors.ground_scan import ABOVE, GroundScan

TERRAIN_HZ = 25.0
FREEZE = 0.6  # share of a swing after which its landing spot stays as chosen
PITCH_SHARE = 0.5  # of the slope along the heading the trunk follows...
ROLL_SHARE = 0.0  # ...and none across it: following half a side slope, our dog took the obstacle course's stairs at
# an angle, rolled 18 degrees and fell; level across, it crossed them (and 66 of 72 legged_bench runs passed, 64 before)
MAX_TILT = math.radians(30.0)  # the trunk follows slopes up to this steep
PLANE_TAU = 0.5  # gait cycles: the plane the trunk follows is smoothed this much
# Footholds: candidates every FOOTHOLD_STEP leg lengths along the heading, up to FOOTHOLD_REACH steps before or
# after the planned landing spot; a spot is on an edge when the ground a step before and after it differs by more
# than EDGE leg lengths.
FOOTHOLD_STEP = 0.09  # leg lengths (Go2: 2.5 cm)
FOOTHOLD_REACH = 3
# A spot is safe when the ground FOOTHOLD_WINDOW steps either side of it is as high as it: with one step (2.5 cm) a
# Go2's hind foot landed with its edge on a stair's nosing and slid back off it.
FOOTHOLD_WINDOW = 2
EDGE = 0.05  # leg lengths (Go2: 1.4 cm)
CLEARANCE = 0.1  # leg lengths a foot passes above what lies in its way, besides its usual lift
PATH_SAMPLES = (0.25, 0.5, 0.75)
# Leg lengths: higher up or further down than this is a wall or a drop, not a step, and the foot is planned as on
# flat ground. Our dog walking into a 30 cm wall (twice its legs) planned its feet onto the wall's top, wobbled
# against it and recovered as if pushed for 4.3 s at a time.
MAX_STEP = 0.6
# Share of a swing a foot that steps up spends rising before it moves on: it moved on at once and caught the riser
# when the step was close in front of it.
LIFT_FIRST = 0.2
REST_LIMIT = 0.2  # leg lengths: the most a foot's point can sit above the ground it stands on
FOOT_LOOK = 0.25  # leg lengths above a foot that the ground under it is looked at from (past little but the foot)
# Leg lengths across a foot (Go2: 2 cm, its foot's radius). A foot rests on the highest ground under it: on gravel
# the ground under a foot's middle can be a hollow 3 cm under the stones it stands on, and taken for the ground
# there it pushed that leg down into the stones until the trunk tipped (Barkour on 3.5 cm gravel).
FOOTPRINT = 0.07
# A walker looks AHEAD_LEGS leg lengths past its front feet, along its way, in AHEAD_SAMPLES points. Ground that
# rises or falls by more than an edge over them is rough (steps, stones, a ramp) and the gait slows down for it
# (ROUGH_SPEED, in quadruped.py: at full speed the Go2, A1 and Barkour toppled on 8 cm stairs, 1 to 2 of 3 at 80 percent
# and 0 of 3 for the Go2 at 100). A robot that cannot climb as steeply as the stairs it may meet (Spot, library
# max_slope) takes the slope of the line through them for how steep the ground is (stairs, ramps alike; a single
# riser is not): too steep above the limit, and stays so until it is under SLOPE_RELEASE of it.
AHEAD_LEGS = 2.0
AHEAD_SAMPLES = 9
# ...and sees further than it measures the slope: a flat landing between stairs up and stairs down must not have it
# speed up for the way down.
SEE_LEGS = 3.0
SLOPE_RELEASE = 0.9


class Footing:
    """Terrain corrections for the feet of one gait (see the module docstring)."""

    def __init__(self, model: mujoco.MjModel, trunk: int, feet: list[tuple[int, np.ndarray]], leg_length: float):
        self.scan = GroundScan(model, trunk)
        self.trunk = trunk
        self.feet = feet  # (body, point in that body's frame) of each leg's foot
        self.leg_length = leg_length
        self.step = FOOTHOLD_STEP * leg_length
        self.edge = EDGE * leg_length
        self.clearance = CLEARANCE * leg_length
        self.pitch = 0.0  # rad the trunk should be pitched (positive: nose down, as the gait measures it)
        self.roll = 0.0  # rad (positive: left side up)
        self.active = False
        # Until when the ground is rough: a foot stepped up or down by more than an edge, or the feet stood off a
        # plane by that much (steps, stones; not a smooth slope), within the last gait cycle.
        self.rough_until = -math.inf
        self.looks = True  # False: a frozen copy (see frozen)
        self.slope_limit: float | None = None  # m/m: steeper ground ahead is too steep to walk onto (see AHEAD_LEGS)
        self.ahead = 0.0  # m/m: the slope of the ground ahead at the last look
        self.rough_ahead = False  # whether the ground ahead rises or falls by more than an edge
        self.too_steep = False
        n = len(feet)
        self.ground = [0.0] * n  # ground height under each standing foot, or where a swinging one lifted off
        self.land = [0.0] * n  # ...and where a swinging foot will land
        self.shift = [0.0] * n  # m along the trunk's x a foot lands off its planned spot (see FOOTHOLD_STEP)
        self.shift_from = [0.0] * n  # ...and stood off it before this swing
        self.clear = [0.0] * n  # m of extra lift over something in the way
        self.lifted_at = [(0.0, 0.0)] * n  # world x, y where a swinging foot lifted off...
        self.land_at = [(0.0, 0.0)] * n  # ...and where it will land
        self.slope = (0.0, 0.0)  # the ground's rise along and across the heading under the feet (m/m), smoothed
        self._heading = (1.0, 0.0)  # cos and sin of the trunk's heading at the last plan
        self.swinging = [False] * n
        # m from the ground up to each foot's point when it stands (a ball foot's center: its radius), measured
        # standing at the start. A landed foot that touches something stands on ground this far below its point:
        # one landing a centimetre short of a stair's riser was taken for one on the step, as the ground around it
        # looked from above, and its leg pulled up 8 cm until the trunk reared.
        self.rest = [0.0] * n
        self._plane = (0.0, 0.0, 0.0)  # smoothed ground plane a + b x + c y (b, c the trunk should follow)
        self._next_look = -math.inf

    @property
    def needed(self) -> bool:
        return self.scan.needed

    def frozen(self) -> Footing:
        """A copy that plans from what this one has seen and looks at nothing more: for a gait run ahead in time on
        a stand-in for the simulation (control/classic.py), whose feet it cannot see."""
        ghost = copy.copy(self)
        for name in ("ground", "land", "shift", "shift_from", "clear", "lifted_at", "land_at", "swinging", "rest"):
            setattr(ghost, name, list(getattr(self, name)))
        ghost.looks = False
        return ghost

    def reset(self) -> None:
        self.active = False
        self.rough_until = -math.inf
        self.pitch = self.roll = 0.0
        self.ahead, self.too_steep, self.rough_ahead = 0.0, False, False
        self._next_look = -math.inf

    def start(self, model: mujoco.MjModel, data: mujoco.MjData, swinging: list[bool]) -> None:
        """Look at the ground under each foot where it is now."""
        for i in range(len(self.feet)):
            x, y, z = self._foot(data, i)
            ground = self._under_foot(model, data, i, x, y, z)
            self.ground[i] = self.land[i] = z if math.isnan(ground) else ground
            if self._touches(model, data, i) and not math.isnan(ground):
                self.rest[i] = min(max(z - ground, -REST_LIMIT * self.leg_length), REST_LIMIT * self.leg_length)
            self.lifted_at[i] = self.land_at[i] = (x, y)
        self.slope = (0.0, 0.0)
        self.shift = [0.0] * len(self.feet)
        self.shift_from = [0.0] * len(self.feet)
        self.clear = [0.0] * len(self.feet)
        self.swinging = list(swinging)
        self.active = True
        self._fresh = True
        self._next_look = -math.inf

    def _foot(self, data: mujoco.MjData, i: int) -> tuple[float, float, float]:
        body, point = self.feet[i]
        world = data.xpos[body] + data.xmat[body].reshape(3, 3) @ point
        return float(world[0]), float(world[1]), float(world[2])

    def plan(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: float,
        dt: float,
        period: float,
        swings: list[float | None],
        landings: list[tuple[float, float]],
        stands: list[tuple[float, float]],
        swing_time: float,
        velocity: tuple[float, float],
    ) -> None:
        """Before a plan's foot targets: note feet that lifted off or landed, look at where swinging feet will land
        (TERRAIN_HZ), and fit the plane the trunk follows.

        swings: each leg's progress through its swing (0 to 1), None while it stands; landings: where each leg
        lands at the end of its swing as planned, and stands: where each foot is planned now (trunk heading frame,
        about the trunk origin, before any foothold shift)."""
        if not self.looks:
            self._plan_blind(swings, landings, stands, dt, period)
            return
        trunk = data.xpos[self.trunk]
        r = data.xmat[self.trunk]
        yaw = math.atan2(float(r[3]), float(r[0]))
        c, s = math.cos(yaw), math.sin(yaw)
        self._heading = (c, s)
        tx, ty = float(trunk[0]), float(trunk[1])
        top = float(trunk[2]) + ABOVE
        self._now = t
        look = t >= self._next_look
        if look:
            self._next_look = t + 1.0 / TERRAIN_HZ
        for i, u in enumerate(swings):
            lifted = u is not None and not self.swinging[i]
            if u is None and self.swinging[i]:  # landed: measure where the foot really is
                x, y, z = self._foot(data, i)
                touching = self._touches(model, data, i)
                ground = self._under_foot(model, data, i, x, y, z, z - self.rest[i] if touching else None)
                self.ground[i] = self.land[i] if math.isnan(ground) else ground
                self.shift_from[i] = self.shift[i]
                self.clear[i] = 0.0
            elif lifted:  # lifted off: look at once where it goes
                self.lifted_at[i] = self._foot(data, i)[:2]
                self.shift_from[i] = self.shift[i]
            self.swinging[i] = u is not None
            if abs(self.step_up(i)) > self.edge:
                self.rough_until = t + period
            if u is None and look and t < self.rough_until and self._touches(model, data, i):
                # A standing foot can slide off an edge: the ground under it is what it stands on now.
                x, y, z = self._foot(data, i)
                ground = self._under_foot(model, data, i, x, y, z, z - self.rest[i])
                if not math.isnan(ground):
                    self.ground[i] = ground
            if u is not None and u < FREEZE and (look or lifted):
                remaining = (1.0 - u) * swing_time
                lx = landings[i][0] + velocity[0] * remaining
                ly = landings[i][1] + velocity[1] * remaining
                self._look_ahead(model, data, i, (tx + c * lx - s * ly, ty + s * lx + c * ly), (c, s), top)
        if look:
            self._look_far(model, data, (c, s), top)
        self._fit(self._points(swings, landings, stands), dt, period)

    def _look_far(self, model: mujoco.MjModel, data: mujoco.MjData, heading: tuple[float, float], top: float) -> None:
        """The ground in front of the front feet: how rough and how steep (see AHEAD_LEGS)."""
        c, s = heading
        lead = max((self._foot(data, i) for i in range(len(self.feet))), key=lambda f: f[0] * c + f[1] * s)
        n_far = round((AHEAD_SAMPLES - 1) * SEE_LEGS / AHEAD_LEGS) + 1
        ds = np.linspace(0.0, SEE_LEGS * self.leg_length, n_far)
        heights = np.array([self.scan.height(model, data, lead[0] + d * c, lead[1] + d * s, top) for d in ds])
        seen = ~np.isnan(heights)
        if seen.sum() < AHEAD_SAMPLES // 2:
            return
        self.rough_ahead = bool(np.ptp(heights[seen]) > self.edge)
        near = seen.copy()
        near[AHEAD_SAMPLES:] = False
        if near.sum() >= 2:
            self.ahead = float(np.polyfit(ds[near], heights[near], 1)[0])
        if self.slope_limit is not None:
            self.too_steep = self.ahead > self.slope_limit * (SLOPE_RELEASE if self.too_steep else 1.0)

    def _plan_blind(self, swings, landings, stands, dt: float, period: float) -> None:
        for i, u in enumerate(swings):
            if u is None and self.swinging[i]:  # lands where it was going to
                self.ground[i] = self.land[i]
                self.shift_from[i] = self.shift[i]
                self.clear[i] = 0.0
            elif u is not None and not self.swinging[i]:
                self.shift_from[i] = self.shift[i]
            self.swinging[i] = u is not None
        self._fit(self._points(swings, landings, stands), dt, period)

    def _points(self, swings, landings, stands) -> list[tuple[float, float, float]]:
        """Where each foot stands, or will land (trunk heading frame), and the ground's height there."""
        points = []
        for i, u in enumerate(swings):
            if u is None:
                x, y = stands[i]
                points.append((x + self.shift_from[i], y, self.ground[i]))
            else:
                x, y = landings[i]
                points.append((x + self.shift[i], y, self.land[i]))
        return points

    def _touches(self, model: mujoco.MjModel, data: mujoco.MjData, i: int) -> bool:
        """Whether foot `i` touches anything but the robot itself."""
        body = self.feet[i][0]
        through = self.scan.through
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if int(model.geom_bodyid[g1]) == body and not through[g2]:
                return True
            if int(model.geom_bodyid[g2]) == body and not through[g1]:
                return True
        return False

    def _under_foot(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        i: int,
        x: float,
        y: float,
        z: float,
        resting: float | None = None,
    ) -> float:
        """The ground under foot `i` at (x, y, z), seen at its middle and four points around it (see FOOTPRINT): the
        highest of them, or for a foot that touches down with its point at `resting` over the ground (see rest), the
        one nearest that height: a foot beside a step rests on the floor, however much of the step is under it."""
        top, r, body = z + FOOT_LOOK * self.leg_length, FOOTPRINT * self.leg_length, self.feet[i][0]
        seen = []
        for dx, dy in ((0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
            v = self.scan.height(model, data, x + dx, y + dy, top, body)
            if not math.isnan(v):
                seen.append(v)
        if not seen:
            return math.nan
        return max(seen) if resting is None else min(seen, key=lambda v: abs(v - resting))

    def _look_ahead(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        i: int,
        spot: tuple[float, float],
        heading: tuple[float, float],
        top: float,
    ) -> None:
        """Choose where swinging foot `i` lands near `spot` (world x, y) and how high it must lift on its way."""
        c, s = heading
        reach = FOOTHOLD_REACH + FOOTHOLD_WINDOW
        step_x, step_y = c * self.step, s * self.step
        h = {
            k: self.scan.height(model, data, spot[0] + k * step_x, spot[1] + k * step_y, top)
            for k in range(-reach, reach + 1)
        }
        if any(math.isnan(v) for v in h.values()):  # hidden under the robot (its own leg): keep what it chose
            return
        # A spot is flat level or flat along the slope the feet walk on: a smooth ramp is as safe everywhere as flat
        # ground, and a stair's tread is flat although the stairs slope (judged along their slope alone, no spot on
        # a tread was flat and feet came down on the nosings).
        along = {k: v - self.slope[0] * k * self.step for k, v in h.items()}
        best = 0
        for k in sorted(range(-FOOTHOLD_REACH, FOOTHOLD_REACH + 1), key=lambda k: (abs(k), -k)):
            window = range(k - FOOTHOLD_WINDOW, k + FOOTHOLD_WINDOW + 1)
            level = max(h[j] for j in window) - min(h[j] for j in window)
            sloped = max(along[j] for j in window) - min(along[j] for j in window)
            if min(level, sloped) < self.edge:
                best = k
                break
        land = max(h[best - 1], h[best], h[best + 1])  # the foot rests on the highest ground under it
        if abs(land - self.ground[i]) > MAX_STEP * self.leg_length:  # a wall or a drop, not a step: as on flat
            best, land = 0, self.ground[i]
        self.shift[i] = best * self.step
        self.land[i] = land
        x0, y0 = self.lifted_at[i]
        x1, y1 = spot[0] + c * best * self.step, spot[1] + s * best * self.step
        self.land_at[i] = (x1, y1)
        # What lies in the way, higher than both ends (on a smooth slope one end is the highest point).
        over = 0.0
        for f in PATH_SAMPLES:
            v = self.scan.height(model, data, x0 + f * (x1 - x0), y0 + f * (y1 - y0), top)
            if not math.isnan(v):
                over = max(over, v - max(self.ground[i], self.land[i]))
        self.clear[i] = over + self.clearance if 0.25 * self.edge < over < MAX_STEP * self.leg_length else 0.0

    def _fit(self, points: list[tuple[float, float, float]], dt: float, period: float) -> None:
        """Fit z = a + b x + c y through the feet, smooth it and set the trunk's pitch and roll to follow it."""
        pts = np.array(points)
        a_mat = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1]])
        try:
            (a, b, c), *_ = np.linalg.lstsq(a_mat, pts[:, 2], rcond=None)
            k_slope = 1.0 if self._fresh else 1.0 - math.exp(-dt / (PLANE_TAU * period))
            self.slope = (self.slope[0] + k_slope * (b - self.slope[0]), self.slope[1] + k_slope * (c - self.slope[1]))
        except np.linalg.LinAlgError:
            return
        if np.max(np.abs(a_mat @ np.array([a, b, c]) - pts[:, 2])) > self.edge:
            self.rough_until = self._now + period
        pitch = float(np.clip(-math.atan(b) * PITCH_SHARE, -MAX_TILT, MAX_TILT))
        roll = float(np.clip(math.atan(c) * ROLL_SHARE, -MAX_TILT, MAX_TILT))
        k = 1.0 if self._fresh else 1.0 - math.exp(-dt / (PLANE_TAU * period))
        self._fresh = False
        self.pitch += k * (pitch - self.pitch)
        self.roll += k * (roll - self.roll)
        b_d, c_d = -math.tan(self.pitch), math.tan(self.roll)
        level = float(np.mean(pts[:, 2] - b_d * pts[:, 0] - c_d * pts[:, 1]))
        self._plane = (self._plane[0] + k * (level - self._plane[0]), b_d, c_d)

    def step_up(self, i: int) -> float:
        """How much higher (m) swinging leg `i` lands than it lifted off, beyond what the slope it walks on makes
        of its stride: a step up (positive) or down (negative), 0 on even ground, sloped or not. Counted as steps,
        the 2.5 cm a stride rises on a 12 degree ramp made an A1 rise before every step and stall on the ramp."""
        (x0, y0), (x1, y1) = self.lifted_at[i], self.land_at[i]
        c, s = self._heading
        dx, dy = c * (x1 - x0) + s * (y1 - y0), -s * (x1 - x0) + c * (y1 - y0)
        return self.land[i] - self.ground[i] - (self.slope[0] * dx + self.slope[1] * dy)

    def climbs(self, i: int) -> bool:
        """Whether swinging leg `i` steps up onto something (see LIFT_FIRST)."""
        return self.step_up(i) > self.edge

    def offset(self, i: int, x: float, y: float, s: float | None, u: float = 0.0) -> tuple[float, float, float]:
        """(dx, dz, extra lift) for leg `i`'s foot target planned at (x, y), standing (s None) or swinging: u its
        progress through the swing (0 to 1), s how far along its way it has moved (0 to 1).

        A swinging foot rises from the ground it left to the higher of its two ends in the first half of its swing
        and comes down from there to where it lands in the second (besides the gait's own lift): stepping up, it is
        level with the step by mid-swing; stepping down, it stays above the edge it leaves."""
        a, b, c = self._plane
        if s is None:
            dx = self.shift_from[i]
            h = self.ground[i]
            extra = 0.0
        else:
            dx = self.shift_from[i] + s * (self.shift[i] - self.shift_from[i])
            start, end = self.ground[i], self.land[i]
            high = max(start, end)
            h = (start if u < 0.5 else end) + (high - (start if u < 0.5 else end)) * math.sin(math.pi * u)
            extra = self.clear[i]
        return dx, h - (a + b * (x + dx) + c * y), extra
