"""Flying an aircraft: a cascaded flight controller for any rotor layout, and trips through the air.

The controller is the one real multirotors fly with, in three loops, each faster than the one around it:

1. Position (50 cm off means "lean that way"): a smooth reference point moves toward the target with limited
   speed and acceleration; the error from it, and its speed and acceleration, give the acceleration wanted.
   With gravity that is a thrust vector, which says how far to tilt (at most MAX_TILT) and how hard to push.
2. Attitude: the error between the body's orientation and the tilt and heading wanted (on the rotation group,
   so any error, even upside down, has one meaning) asks for body turn rates.
3. Body rates: the rate error times the inertia is the torque to make, which the mixer (airframe.Mixer) turns
   into rotor commands from the rotor layout, giving up yaw first and height second when the rotors run out.

The air is still unless a gust blows: drag (a linear rotor-drag term and quadratic body drag) and the wind act on
the airframe as an external force each control tick. MuJoCo's own fluid model is world-wide and would slow every
robot and ball in the world, so it is left off.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.airframe import Airframe, find_airframe

log = logging.getLogger(__name__)

RATE_HZ = 250.0
TAKEOFF_HEIGHT = 1.0  # m above the ground a takeoff without a height climbs to
MAX_SPEED = 3.0  # m/s across the ground
CRUISE = 1.5  # m/s, a trip's top speed when none is given
MAX_CLIMB = 1.5  # m/s up
MAX_DESCENT = 1.0  # m/s down, away from the ground
LAND_SPEED = 0.3  # m/s down at touchdown
ACCEL = 2.0  # m/s^2 the reference speeds up and slows down at, across
ACCEL_UP = 1.5  # m/s^2 up and down
MAX_TILT = math.radians(30.0)
MAX_TURN = 1.5  # rad/s of yaw
SPOOL_DOWN = 0.5  # s from touchdown to motors off
TOUCHDOWN = 0.02  # m of clearance at which it is down
ARRIVE_TOL = 0.15  # m from the target, with the speed below ARRIVE_SPEED, counts as there
ARRIVE_SPEED = 0.3
UPSIDE_DOWN = 0.0  # the body's up axis below the horizon: crashed
CLEARANCE = 0.6  # m kept above anything a straight leg would hit, when it climbs over it
MAX_ALTITUDE = 30.0
GROUND_PROBE = 50.0  # m above which the ground under a point is looked for
# Gains: position (per axis x, y, z), attitude and body rates (roll, pitch, yaw).
KP = np.array([4.0, 4.0, 9.0])
KD = np.array([3.6, 3.6, 5.4])
KI = np.array([0.6, 0.6, 1.5])
I_LIMIT = 3.0  # m/s^2 the position integral may add
K_ATT = np.array([7.0, 7.0, 2.5])
K_RATE = np.array([20.0, 20.0, 6.0])
# Air: rotor drag per kg of aircraft (N s/m per kg) and the drag coefficient times area per m^2 of plan view.
ROTOR_DRAG = 0.12
BODY_CD = 0.25
AIR_DENSITY = 1.225


class FlightError(ValueError):
    """User-facing: says why it cannot fly there and what to do instead."""


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


@dataclass
class Gust:
    """Wind that rises, blows and dies down (world frame, m/s)."""

    wind: np.ndarray
    start: float
    seconds: float
    ramp: float = 0.3

    def at(self, t: float) -> np.ndarray:
        s = t - self.start
        if s <= 0 or s >= self.seconds + 2 * self.ramp:
            return np.zeros(3)
        k = min(1.0, s / self.ramp, (self.seconds + 2 * self.ramp - s) / self.ramp)
        return self.wind * k

    def over(self, t: float) -> bool:
        return t - self.start >= self.seconds + 2 * self.ramp


@dataclass
class Mission:
    """Points to fly through in order (world, m); with `land`, it lands at the last one."""

    points: list[np.ndarray]
    land: bool
    cruise: float
    index: int = 0
    state: str = "flying"  # flying, arrived, landed, aborted
    touchdown_speed: float | None = None

    @property
    def done(self) -> bool:
        return self.state != "flying"


@dataclass
class Reference:
    """Where the controller wants the aircraft now: a point that moves smoothly toward the target."""

    p: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0

    def toward(self, target: np.ndarray, dt: float, cap: float, down: float = MAX_DESCENT) -> bool:
        """One step along the straight line to `target`; True once there."""
        d = target - self.p
        dist = float(np.linalg.norm(d))
        if dist < 1e-4 or dist <= float(np.linalg.norm(self.v)) * dt and float(self.v @ d) >= 0:
            self.a = -self.v / dt if dt > 0 else np.zeros(3)
            self.p, self.v = target.copy(), np.zeros(3)
            return True
        u = d / dist
        across, up = float(np.hypot(u[0], u[1])), float(u[2])
        top = min(cap / across if across > 1e-6 else math.inf, (MAX_CLIMB if up > 0 else down) / max(abs(up), 1e-6))
        accel = min(ACCEL / across if across > 1e-6 else math.inf, ACCEL_UP / max(abs(up), 1e-6))
        want = u * min(top, math.sqrt(2.0 * accel * dist))
        self._accelerate(want, dt, accel)
        return False

    def drive(self, velocity: np.ndarray, dt: float) -> None:
        """Teleop: follow a commanded velocity (world frame), changing it no faster than the limits."""
        self._accelerate(velocity, dt, max(ACCEL, ACCEL_UP))

    def _accelerate(self, want: np.ndarray, dt: float, accel: float) -> None:
        dv = want - self.v
        step = accel * dt
        n = float(np.linalg.norm(dv))
        if n > step:
            dv = dv * (step / n)
        self.a = dv / dt if dt > 0 else np.zeros(3)
        self.v = self.v + dv
        self.p = self.p + self.v * dt

    def turn_toward(self, yaw: float, dt: float) -> None:
        err = _wrap(yaw - self.yaw)
        self.yaw = _wrap(self.yaw + float(np.clip(err, -MAX_TURN * dt, MAX_TURN * dt)))


class FlightController:
    """Flies one aircraft: takes off, hovers, flies to points or follows keys, lands (see the module notes)."""

    rate_hz = RATE_HZ
    max_speed = MAX_SPEED
    max_turn = MAX_TURN
    max_reverse = MAX_SPEED

    def __init__(self, model: mujoco.MjModel, robot: str, qpos: np.ndarray | None = None) -> None:
        self.robot = robot
        self.frame: Airframe = find_airframe(model, robot, qpos)
        self.model = model
        root = self.frame.root
        joint = int(model.body_jntadr[root])
        self.qadr, self.dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.actuators = np.array(self.frame.actuators, dtype=int)
        self.resting = np.clip(0.0, self.frame.mixer.low, self.frame.mixer.high)  # motors off
        span = self.frame.radius * 2
        self.drag_quad = 0.5 * AIR_DENSITY * BODY_CD * span * span
        self.drag_lin = ROTOR_DRAG * self.frame.mass
        self.phase = "landed"  # landed, flying, landing, spooling, crashed
        self.mode = "hold"  # hold, keys, mission
        self.mission: Mission | None = None
        self.ref: Reference | None = None
        self.target: np.ndarray | None = None  # the point held while hovering
        self.speed = 0.0  # keys: m/s forward, sideways and up, rad/s of turn
        self.side = 0.0
        self.climb = 0.0
        self.turn = 0.0
        self.gust: Gust | None = None
        self.fallen = False
        self.touchdown_speed: float | None = None
        self.saturated = False
        self.altitude = 0.0  # m above the ground under it
        self._t: float | None = None
        self._integral = np.zeros(3)
        self._spool_from = 0.0
        self._started = False
        self._land_ground: float | None = None
        self._last_force = np.zeros(3)
        self._p = np.zeros(3)
        self._v = np.zeros(3)

    # ---- commands -------------------------------------------------------------------------------

    @property
    def airborne(self) -> bool:
        return self.phase in ("flying", "landing")

    def command(self, speed: float = 0.0, turn: float = 0.0, side: float = 0.0, climb: float = 0.0) -> None:
        """Fly by keys: speed forward and side to the left (m/s, along its heading), climb (m/s up), turn (rad/s
        left). A landed aircraft takes off first. All zero: it brakes and hovers where it is."""
        self.speed = float(np.clip(speed, -MAX_SPEED, MAX_SPEED))
        self.side = float(np.clip(side, -MAX_SPEED, MAX_SPEED))
        self.climb = float(np.clip(climb, -MAX_DESCENT, MAX_CLIMB))
        self.turn = float(np.clip(turn, -MAX_TURN, MAX_TURN))
        moving = any(abs(v) > 1e-6 for v in (self.speed, self.side, self.climb, self.turn))
        self.mission = None
        if self.phase in ("landed", "spooling") and moving and self.climb >= 0 and not self.fallen:
            self._take_off()
            self.target = None
        if self.phase in ("flying", "landing"):
            if self.phase == "landing" and moving and self.climb >= 0:
                self.phase = "flying"
            self.mode = "keys"
            self.target = None

    def fly(self, mission: Mission) -> None:
        """Fly a mission (from wherever it is; a landed aircraft takes off straight up first)."""
        if self.fallen:
            raise FlightError(f"{self.robot} has crashed and cannot fly; reset the simulation (sim_control reset)")
        self.mission = mission
        self.mode = "mission"
        self.speed = self.side = self.climb = self.turn = 0.0
        if self.phase in ("landed", "spooling"):
            self._take_off()
        elif self.phase == "landing":
            self.phase = "flying"

    def hover(self) -> None:
        """Stop where it is: brake and hold that point in the air (a landed aircraft stays down)."""
        self.mission = None
        self.speed = self.side = self.climb = self.turn = 0.0
        if self.phase in ("flying", "landing"):
            self.phase = "flying"
            self.mode = "hold"
            self.target = None  # set from where it brakes to, at the next tick

    def gusts(self, wind: list[float] | np.ndarray, seconds: float, now: float) -> None:
        w = np.zeros(3)
        w[: len(wind)] = np.asarray(wind, float)[:3]
        self.gust = Gust(w, now, seconds)

    def _take_off(self) -> None:
        self.phase = "flying"
        self._integral[:] = 0.0
        self._started = False  # the reference starts from where it stands, at the next tick
        self.touchdown_speed = None
        self._land_ground = None
        if self.mission is None:
            self.mode = "keys"

    # ---- the session controller ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "behavior": "fly",
            "robot": self.robot,
            "state": "crashed" if self.fallen else self.phase,
            "altitude_m": round(self.altitude, 2),
            "speed": round(float(np.hypot(self._v[0], self._v[1])), 2),
            "max_speed": MAX_SPEED,
        }
        if self.mode == "keys" and self.airborne:
            out["keys"] = {"forward": self.speed, "left": self.side, "up": self.climb, "turn": self.turn}
        if self.gust is not None and not self.gust.over(self._t or 0.0):
            out["gust_m_s"] = [round(float(v), 2) for v in self.gust.wind]
        if self.touchdown_speed is not None and self.phase in ("landed", "spooling"):
            out["touchdown_m_s"] = round(self.touchdown_speed, 3)
        return out

    def release(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Taken off the robot: the air it applied lets go too (see SimSession.set_controller)."""
        if self.model is model:
            data.xfrc_applied[self.frame.root, :3] = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = float(data.time)
        dt = 1.0 / RATE_HZ if self._t is None else float(np.clip(t - self._t, 1e-4, 0.05))
        self._t = t
        root = self.frame.root
        rot = data.xmat[root].reshape(3, 3)
        self._p = data.xpos[root].copy()
        self._v = data.qvel[self.dadr : self.dadr + 3].copy()
        omega = data.qvel[self.dadr + 3 : self.dadr + 6].copy()
        ground = self.ground_below(model, data, self._p)
        self.altitude = float(self._p[2] - ground - self.frame.rest_height) if ground is not None else self._p[2]
        self._air(data, t)
        if rot[2, 2] < UPSIDE_DOWN and not self.fallen:
            self.fallen = True
            self.phase = "crashed"
            if self.mission is not None and not self.mission.done:
                self.mission.state = "aborted"
        if self.phase in ("landed", "crashed"):
            data.ctrl[self.actuators] = self.resting
            return
        if not self._started:
            self.ref = Reference(self._p.copy(), yaw=math.atan2(rot[1, 0], rot[0, 0]))
            self._started = True
            if self.mode == "keys" and self.phase == "flying" and self.target is None and ground is not None:
                self.target = np.array([*self._p[:2], max(self._p[2], ground + self.frame.rest_height + TAKEOFF_HEIGHT)])
        self._move_reference(dt, ground, rot)
        thrust, torque = self._control(dt, rot, omega)
        if self.phase == "spooling":
            k = max(0.0, 1.0 - (t - self._spool_from) / SPOOL_DOWN)
            thrust *= k
            if k <= 0.0:
                self.phase = "landed"
                data.ctrl[self.actuators] = self.resting
                return
        mixed = self.frame.mixer.mix(thrust, torque)
        self.saturated = mixed.saturated
        data.ctrl[self.actuators] = mixed.controls
        self._touchdown(t)

    # ---- the reference ------------------------------------------------------------------------

    def _move_reference(self, dt: float, ground: float | None, rot: np.ndarray) -> None:
        ref = self.ref
        assert ref is not None
        floor = (ground if ground is not None else -math.inf) + self.frame.rest_height
        if self.phase in ("landing", "spooling"):
            self._descend(dt, floor)
            return
        if self.mode == "mission" and self.mission is not None and not self.mission.done:
            self._follow_mission(dt, floor)
            return
        if self.mode == "keys":
            yaw = ref.yaw
            c, s = math.cos(yaw), math.sin(yaw)
            across = np.array([c * self.speed - s * self.side, s * self.speed + c * self.side])
            if self.target is not None and abs(self.climb) < 1e-6:  # taking off by keys: up to the hover height
                done = abs(self.target[2] - ref.p[2]) < 1e-3
                vz = 0.0 if done else float(np.clip(2.0 * (self.target[2] - ref.p[2]), -MAX_DESCENT, MAX_CLIMB))
                if done:
                    self.target = None
            else:
                self.target = None
                vz = self.climb
            ref.drive(np.array([*across, vz]), dt)
            ref.yaw = _wrap(ref.yaw + self.turn * dt)
            if ref.p[2] < floor + 0.05 and self.climb < 0:  # brought down to the ground by keys: land
                self._start_landing(ground)
            elif ref.p[2] < floor + 0.05:
                ref.p[2], ref.v[2] = floor + 0.05, max(ref.v[2], 0.0)
            return
        # Holding a point: the stopping point of where it was going, then that point.
        if self.target is None:
            v = ref.v
            brake = v * float(np.linalg.norm(v)) / (2 * ACCEL) if np.linalg.norm(v) > 1e-6 else np.zeros(3)
            self.target = ref.p + brake
            self.target[2] = max(self.target[2], floor + 0.05)
        ref.toward(self.target, dt, MAX_SPEED)

    def _follow_mission(self, dt: float, floor: float) -> None:
        ref, mission = self.ref, self.mission
        assert ref is not None and mission is not None
        last = mission.index == len(mission.points) - 1
        point = mission.points[mission.index]
        if last and mission.land:
            above = np.array([point[0], point[1], max(ref.p[2], floor + 0.3)])
            if float(np.hypot(*(point[:2] - ref.p[:2]))) > 1e-3:
                ref.toward(above, dt, mission.cruise)
                self._face(dt, above)
                return
            if float(np.hypot(*(point[:2] - self._p[:2]))) < ARRIVE_TOL and float(np.hypot(*self._v[:2])) < 0.2:
                self._start_landing(None)
            return
        reached = ref.toward(point, dt, mission.cruise)
        self._face(dt, point)
        if not reached:
            return
        if not last:
            mission.index += 1
            return
        if float(np.linalg.norm(point - self._p)) < ARRIVE_TOL and float(np.linalg.norm(self._v)) < ARRIVE_SPEED:
            mission.state = "arrived"
            self.mode = "hold"
            self.target = point.copy()

    def _face(self, dt: float, point: np.ndarray) -> None:
        """Turn its nose the way it flies on a long enough leg (its camera looks where it goes)."""
        ref = self.ref
        assert ref is not None
        if float(np.hypot(point[0] - ref.p[0], point[1] - ref.p[1])) > 0.5:
            ref.turn_toward(math.atan2(point[1] - ref.p[1], point[0] - ref.p[0]), dt)

    def _start_landing(self, ground: float | None) -> None:
        self.phase = "landing"
        self._land_ground = ground

    def _descend(self, dt: float, floor: float) -> None:
        ref = self.ref
        assert ref is not None
        if self.phase == "spooling":
            ref.v[:] = 0.0
            ref.a[:] = 0.0
            return
        height = max(0.0, ref.p[2] - floor)
        speed = float(np.clip(LAND_SPEED + 0.6 * height, LAND_SPEED, MAX_DESCENT))
        ref.drive(np.array([0.0, 0.0, -speed]), dt)
        ref.p[2] = max(ref.p[2], floor - 0.3)  # below the ground a little: it keeps pressing down until it touches

    def _touchdown(self, t: float) -> None:
        if self.phase != "landing" or self.altitude > TOUCHDOWN:
            return
        self.touchdown_speed = float(abs(min(self._v[2], 0.0)))
        self.phase = "spooling"
        self._spool_from = t
        if self.mission is not None and not self.mission.done and self.mission.land:
            self.mission.state = "landed"
            self.mission.touchdown_speed = self.touchdown_speed

    # ---- the three loops ------------------------------------------------------------------------

    def _control(self, dt: float, rot: np.ndarray, omega: np.ndarray) -> tuple[float, np.ndarray]:
        ref = self.ref
        assert ref is not None
        f = self.frame
        g = f.gravity
        err = ref.p - self._p
        if self.phase == "flying":  # on the ground (landing) the integral would only wind up
            self._integral = np.clip(self._integral + err * dt, -I_LIMIT / KI, I_LIMIT / KI)
        acc = ref.a + KP * err + KD * (ref.v - self._v) + KI * self._integral
        force = f.mass * (acc + np.array([0.0, 0.0, g]))
        force[2] = max(force[2], 0.15 * f.mass * g)  # never ask to be pulled down: tilting over is worse
        across = float(np.hypot(force[0], force[1]))
        most = force[2] * math.tan(MAX_TILT)
        if across > most:
            force[:2] *= most / across
        z_want = force / float(np.linalg.norm(force))
        heading = np.array([math.cos(ref.yaw), math.sin(ref.yaw), 0.0])
        y_want = np.cross(z_want, heading)
        y_want /= max(float(np.linalg.norm(y_want)), 1e-9)
        x_want = np.cross(y_want, z_want)
        want = np.column_stack([x_want, y_want, z_want])
        thrust = max(0.0, float(force @ rot[:, 2]))
        e_rot = 0.5 * _vee(want.T @ rot - rot.T @ want)
        rate_want = -K_ATT * e_rot + rot.T @ np.array([0.0, 0.0, self.turn if self.mode == "keys" else 0.0])
        inertia = f.inertia
        torque = inertia @ (K_RATE * (rate_want - omega)) + np.cross(omega, inertia @ omega)
        return thrust, torque

    # ---- the air --------------------------------------------------------------------------------

    def _air(self, data: mujoco.MjData, t: float) -> None:
        """Drag and wind on the airframe, as an external force at its root body (see the module notes)."""
        wind = self.gust.at(t) if self.gust is not None else np.zeros(3)
        if self.gust is not None and self.gust.over(t):
            self.gust = None
        rel = self._v - wind
        speed = float(np.linalg.norm(rel))
        force = -(self.drag_lin + self.drag_quad * speed) * rel
        if self.phase == "landed" and self.gust is None:
            force[:] = 0.0  # on the ground in still air: nothing to push
        data.xfrc_applied[self.frame.root, :3] = force
        self._last_force = force

    def ground_below(self, model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray) -> float | None:
        """Height of whatever is under `point` (not the aircraft itself), or None over nothing."""
        return ground_height(model, data, point, self.frame.descendants, self.frame.root)


def ground_height(
    model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, own: frozenset[int] = frozenset(), root: int = -1
) -> float | None:
    """The height of the first thing straight below `point` that is not one of the bodies `own`."""
    start = np.array(point, float)
    down = np.array([0.0, 0.0, -1.0])
    geom = np.zeros(1, dtype=np.int32)
    for _ in range(6):
        dist = mujoco.mj_ray(model, data, start, down, None, 1, root, geom)
        if dist < 0:
            return None
        hit = start + down * dist
        if int(model.geom_bodyid[geom[0]]) not in own:
            return float(hit[2])
        start = hit + down * 1e-4
    return None


# ---- trips through the air ---------------------------------------------------------------------------


@dataclass
class FlightPlan:
    points: list[np.ndarray]
    land: bool
    length: float
    notes: list[str]


def plan_flight(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fc: FlightController,
    goals: list[np.ndarray],
    land: bool,
) -> FlightPlan:
    """Points a mission flies through for `goals` (world points; with `land`, it lands at the last one): straight
    up first when it stands on the ground, and over anything a straight leg would fly into. Raises FlightError
    when something in the way is too tall to fly over."""
    frame = fc.frame
    here = data.xpos[frame.root].copy()
    notes: list[str] = []
    points: list[np.ndarray] = []
    rest = frame.rest_height
    start = here
    if not fc.airborne:
        first = goals[0]
        ground = fc.ground_below(model, data, here)
        lift = max(first[2] if not (land and len(goals) == 1) else 0.0, (ground or 0.0) + rest + TAKEOFF_HEIGHT)
        if land and len(goals) == 1:
            lift = max(lift, (ground or 0.0) + rest + TAKEOFF_HEIGHT)
        start = np.array([here[0], here[1], lift])
        points.append(start)
    for k, goal in enumerate(goals):
        is_landing = land and k == len(goals) - 1
        end = goal.copy()
        if is_landing:
            end[2] = max(start[2], end[2])  # it lands straight down at the end
        legs = _clear_legs(model, data, fc, start, end, notes)
        points.extend(legs)
        start = legs[-1] if legs else start
        if is_landing:
            points[-1] = np.array([goal[0], goal[1], start[2]])
    path = [here, *points]
    length = float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:], strict=True)))
    if land:
        ground = ground_height(model, data, np.array([*points[-1][:2], points[-1][2]]), frame.descendants, frame.root)
        length += max(0.0, points[-1][2] - (ground or 0.0) - rest)
    return FlightPlan(points, land, length, notes)


def _clear_legs(
    model: mujoco.MjModel, data: mujoco.MjData, fc: FlightController, start: np.ndarray, end: np.ndarray, notes
) -> list[np.ndarray]:
    """The straight leg from start to end, or, when something is in the way, up over it, across and down."""
    blocker = _first_hit(model, data, fc, start, end)
    if blocker is None:
        return [end]
    name, top = blocker
    over = top + CLEARANCE + fc.frame.rest_height
    if over > MAX_ALTITUDE:
        raise FlightError(f"{name} is in the way and too tall to fly over ({top:.1f} m); add a point around it")
    up = np.array([start[0], start[1], max(start[2], over)])
    across = np.array([end[0], end[1], max(end[2], over)])
    for a, b in ((start, up), (up, across)):
        again = _first_hit(model, data, fc, a, b)
        if again is not None:
            raise FlightError(
                f"{name} is in the way of the straight flight to {[round(float(v), 2) for v in end]}, and so is "
                f"{again[0]} when it climbs over; add points around them"
            )
    notes.append(f"climbed to {over:.1f} m to fly over {name} (its top is at {top:.1f} m)")
    legs = [up, across]
    if end[2] < across[2] - 1e-6:
        legs.append(end)
    return [p for i, p in enumerate(legs) if i == 0 or not np.allclose(p, legs[i - 1])]


def _first_hit(
    model: mujoco.MjModel, data: mujoco.MjData, fc: FlightController, a: np.ndarray, b: np.ndarray
) -> tuple[str, float] | None:
    """What a straight flight from a to b runs into (its name and top, m), or None. Rays from the middle and
    the four sides of the aircraft, as wide as its rotors reach."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    u = d / length
    side = np.cross(u, [0.0, 0.0, 1.0])
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([1.0, 0.0, 0.0])
    lift = np.cross(side, u)
    r = fc.frame.radius
    own = fc.frame.descendants
    geom = np.zeros(1, dtype=np.int32)
    best: tuple[float, int] | None = None
    for offset in (np.zeros(3), side * r, -side * r, lift * 0.5 * r, -lift * 0.5 * r):
        start = a + offset
        travelled = 0.0
        for _ in range(6):
            dist = mujoco.mj_ray(model, data, start, u, None, 1, fc.frame.root, geom)
            if dist < 0 or travelled + dist > length:
                break
            g = int(geom[0])
            if int(model.geom_bodyid[g]) in own:
                travelled += dist + 1e-4
                start = start + u * (dist + 1e-4)
                continue
            if best is None or travelled + dist < best[0]:
                best = (travelled + dist, g)
            break
    if best is None:
        return None
    return _thing(model, data, best[1])


def _thing(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> tuple[str, float]:
    """The name of what a geom belongs to (an object, a robot, a terrain patch) and the top of that thing."""
    from robotforge.behaviors.nav_map import world_extent

    body = int(model.geom_bodyid[geom])
    if body > 0:
        top_body = int(model.body_rootid[body])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, top_body) or "something"
        geoms = [g for g in range(model.ngeom) if int(model.body_rootid[model.geom_bodyid[g]]) == top_body]
    else:
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        name = gname.split("/", 1)[0] or "the ground"
        for cut in ("_wall", "_trunk", "_crown", "_leg"):
            name = name.split(cut, 1)[0]
        stem = name
        geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == 0
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(stem)
        ] or [geom]
    top = max(float(world_extent(model, data, g)[1][2]) for g in geoms)
    return name.split("/", 1)[0], top


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
            self._worst = max(self._worst, float(np.linalg.norm(self.fc.ref.p - p)))
        m = self.mission
        if self.fc.fallen or m.state == "aborted":
            self._finish(t, "stopped", "crashed", "it crashed on the way")
        elif self.fc.mission is not m and not m.done:
            self._finish(t, "stopped", "interrupted", "it was flown another way (keys, move or stop) on the way")
        elif m.state in ("arrived", "landed") and (m.state == "arrived" or self.fc.phase == "landed"):
            self._finish(t, "arrived")
        elif self.deadline is not None and t >= self.deadline - 1e-9:
            self.fc.hover()
            self._finish(t, "stopped", "out_of_time", "it ran out of time on the way and hovers where it was")

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"behavior": "fly", "navigating": self.robot, "state": self.state, "goal": self.destination}
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
        out["landed" if self.mission.land else "hovering"] = (
            self.fc.phase == "landed" if self.mission.land else self.fc.airborne
        )
        if self.mission.touchdown_speed is not None:
            out["touchdown_m_s"] = round(self.mission.touchdown_speed, 3)
        out["max_tracking_error_m"] = round(self._worst, 3)
        notes = [*self.notes, *([self.note] if self.note else [])]
        if notes:
            out["note"] = "; ".join(notes)
        if self.cause and not self.arrived:
            out["why"] = self.cause
        return out
