"""Flying an aircraft: a cascaded flight controller for any rotor layout, and trips through the air.

The controller is the one real multirotors fly with, in three loops, each faster than the one around it:

1. Position (50 cm off means "lean that way"): a smooth reference point moves toward the target with limited
   speed and acceleration; the error from it, and its speed and acceleration, give the acceleration wanted.
   With gravity that is a thrust vector, which says how far to tilt (at most MAX_TILT) and how hard to push.
2. Attitude: the error between the body's orientation and the tilt and heading wanted (on the rotation group,
   so any error, even upside down, has one meaning) asks for body turn rates.
3. Body rates: the rate error times the inertia is the torque to make, which the mixer (airframe.Mixer) turns
   into rotor commands from the rotor layout, giving up yaw first and height second when the rotors run out.

A disturbance observer feeds the position loop: the force the rotors make is known (the actuators' forces), the
acceleration is measured (an IMU would), and what is left over is pushing from outside (a gust, drag, a payload
heavier than expected). Subtracting it at once beats an integral, which only notices a push after it has moved
the aircraft away, then overshoots on the way back. A small integral, used only near the target, takes the rest.

Drag and wind are the air's business (behaviors/air.py): to the controller they are outside forces like any
other, and the observer takes them out.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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
LAND_SPEED = 0.22  # m/s down at touchdown
ACCEL = 2.0  # m/s^2 the reference speeds up and slows down at, across
ACCEL_UP = 1.5  # m/s^2 up and down
MAX_TILT = math.radians(35.0)
MAX_TURN = 1.5  # rad/s of yaw
SPOOL_DOWN = 0.5  # s from touchdown to motors off
TOUCHDOWN = 0.02  # m of clearance at which it is down
ARRIVE_TOL = 0.04  # m from the target, with the speed below ARRIVE_SPEED, counts as there
ARRIVE_SPEED = 0.1
UPSIDE_DOWN = 0.0  # the body's up axis below the horizon: crashed
# Gains: position (per axis x, y, z), attitude and body rates (roll, pitch, yaw).
KP = np.array([4.0, 4.0, 9.0])
KD = np.array([4.4, 4.4, 5.4])
KI = np.array([0.4, 0.4, 0.8])
I_LIMIT = 1.0  # m/s^2 the position integral may add
I_ZONE = 0.3  # m: further from the reference than this, the integral waits (it would only wind up)
OBSERVER_TAU = 0.04  # s, how quickly the disturbance estimate follows a push
SMOOTHING = 5.0  # rad/s: the reference's acceleration is low-passed this fast (see Reference)
K_ATT = np.array([7.0, 7.0, 2.5])
K_RATE = np.array([20.0, 20.0, 6.0])


CONTROLLER = (
    "cascaded flight controller: position (with a disturbance observer), then attitude, then body rates; a mixer "
    "from its rotor layout shares thrust and torque among the rotors"
)
STOP_NOTE = "it hovers where it is; to land, move it with to=[x, y, 0] (the spot to land on)"
# Geom groups the altimeter and the look ahead see: what a drone can bump into, not the visual meshes (group 2).
SOLID = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)
DOWN = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 0.0, 1.0])


class FlightError(ValueError):
    """User-facing: says why it cannot fly there and what to do instead."""


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


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
    """Where the controller wants the aircraft now: a point that moves toward the target with limited speed and
    acceleration (p, v, a), smoothed (sp, sv, sa) so its acceleration never jumps. A jump from speeding up to
    slowing down asks the aircraft to flip its tilt at once; it cannot, and overshot a 2 m step by 12 cm."""

    p: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    sp: np.ndarray = field(init=False)
    sv: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sa: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.sp = self.p.copy()

    def smooth(self, dt: float) -> None:
        """A critically damped second-order low-pass of p (no speed fed forward: that adds a zero, which overshoots)."""
        w = SMOOTHING
        self.sa = w * w * (self.p - self.sp) - 2 * w * self.sv
        self.sv = self.sv + self.sa * dt
        self.sp = self.sp + self.sv * dt

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
        root = self.frame.root
        joint = int(model.body_jntadr[root])
        self.dadr = int(model.jnt_dofadr[joint])
        self.actuators = np.array(self.frame.actuators, dtype=int)
        self._pushes = np.array([r.push for r in self.frame.rotors])  # n x 3, body frame
        self._outside = np.zeros(3)  # N, the estimated push from outside (world frame)
        self._com_v: np.ndarray | None = None
        self._first = True
        self.resting = np.clip(0.0, self.frame.mixer.low, self.frame.mixer.high)  # motors off
        self.phase = "landed"  # landed, flying, landing, spooling, crashed
        self.mode = "hold"  # hold, keys, mission
        self.mission: Mission | None = None
        self.ref: Reference | None = None
        self.target: np.ndarray | None = None  # the point held while hovering
        self.speed = 0.0  # keys: m/s forward, sideways and up, rad/s of turn
        self.side = 0.0
        self.climb = 0.0
        self.turn = 0.0
        self.fallen = False
        self.touchdown_speed: float | None = None
        self.saturated = False
        self.altitude = 0.0  # m above the ground under it
        self._t: float | None = None
        self._integral = np.zeros(3)
        self._spool_from = 0.0
        self._started = False
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

    def hold_at(self, point: list[float] | np.ndarray) -> None:
        """Fly to `point` and hold it (taking off first if it stands): the hover task's "cascade" method."""
        self.mission = None
        self.speed = self.side = self.climb = self.turn = 0.0
        if self.phase in ("landed", "spooling"):
            self._take_off()
        self.phase, self.mode = "flying", "hold"
        self.target = np.asarray(point, float).copy()

    def hover(self) -> None:
        """Stop where it is: brake and hold that point in the air (a landed aircraft stays down)."""
        self.mission = None
        self.speed = self.side = self.climb = self.turn = 0.0
        if self.phase in ("flying", "landing"):
            self.phase = "flying"
            self.mode = "hold"
            self.target = None  # set from where it brakes to, at the next tick

    def _take_off(self) -> None:
        self.phase = "flying"
        self._integral[:] = 0.0
        self._started = False  # the reference starts from where it stands, at the next tick
        self.touchdown_speed = None
        if self.mission is None:
            self.mode = "keys"

    # ---- the session controller ----------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "behavior": "fly",
            "robot": self.robot,
            "state": "crashed" if self.fallen else self.phase,
            "altitude_m": round(max(self.altitude, 0.0), 2),
            "speed": round(float(np.hypot(self._v[0], self._v[1])), 2),
            "max_speed": MAX_SPEED,
        }
        if self.mode == "keys" and self.airborne:
            out["keys"] = {"forward": self.speed, "left": self.side, "up": self.climb, "turn": self.turn}
        if self.saturated and self.airborne:  # asked for more than the rotors give: it flies as well as it can
            out["rotors_at_limit"] = True
        if self.touchdown_speed is not None and self.phase in ("landed", "spooling"):
            out["touchdown_m_s"] = round(self.touchdown_speed, 3)
        return out

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
        if self._first:  # placed in the air (a world with a height for it, a reset mid-flight): it hovers there
            self._first = False
            if self.altitude > 0.15 and self.phase == "landed" and rot[2, 2] > 0.5:
                self.phase, self.mode, self.target, self._started = "flying", "hold", None, False
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
                self.target = np.array(
                    [*self._p[:2], max(self._p[2], ground + self.frame.rest_height + TAKEOFF_HEIGHT)]
                )
        self._observe(model, data, dt, rot)
        self._move_reference(dt, ground, rot)
        self.ref.smooth(dt)
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
                self._start_landing()
            elif ref.p[2] < floor + 0.05:
                ref.p[2], ref.v[2] = floor + 0.05, max(ref.v[2], 0.0)
            return
        # Holding a point: where it stops braking from where it is now, then that point. From the smoothed reference
        # (the one it follows): the raw one runs ahead of it, and a drone stopped at 1.5 m/s held 1 m further on.
        if self.target is None:
            ref.p, ref.v = ref.sp.copy(), ref.sv.copy()
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
                self._start_landing()
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

    def _start_landing(self) -> None:
        self.phase = "landing"

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

    def _observe(self, model: mujoco.MjModel, data: mujoco.MjData, dt: float, rot: np.ndarray) -> None:
        """The disturbance observer (see the module notes): measured mass times acceleration, less gravity and
        the rotors' own force, is what pushes from outside. Touching the ground, the ground pushes too: the
        estimate fades out instead, or the take-off would read the ground holding it up as wind."""
        f = self.frame
        fade = min(1.0, dt / OBSERVER_TAU)
        mujoco.mj_subtreeVel(model, data)  # the center of mass's velocity: the root's origin swings as it turns
        com_v = data.subtree_linvel[f.root].copy()
        before, self._com_v = self._com_v, com_v
        if self.phase != "flying" or self.altitude < 0.05 or before is None:
            self._outside *= 1.0 - fade
            return
        acc = (com_v - before) / dt
        rotors = rot @ (self._pushes.T @ data.actuator_force[self.actuators])
        seen = f.mass * (acc + np.array([0.0, 0.0, f.gravity])) - rotors
        self._outside += (seen - self._outside) * fade

    def _control(self, dt: float, rot: np.ndarray, omega: np.ndarray) -> tuple[float, np.ndarray]:
        ref = self.ref
        assert ref is not None
        f = self.frame
        g = f.gravity
        err = ref.sp - self._p
        if self.phase == "flying":  # on the ground (landing) the integral would only wind up
            near = np.abs(err) < I_ZONE
            self._integral = np.where(
                near, np.clip(self._integral + err * dt, -I_LIMIT / KI, I_LIMIT / KI), self._integral
            )
        acc = ref.sa + KP * err + KD * (ref.sv - self._v) + KI * self._integral
        force = f.mass * (acc + np.array([0.0, 0.0, g])) - self._outside
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

    def ground_below(self, model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray) -> float | None:
        """Height of whatever is under `point` (not the aircraft itself), or None over nothing."""
        return ground_height(model, data, point, self.frame.descendants, self.frame.root)


def seen_from_outside(model: mujoco.MjModel, data: mujoco.MjData, frame: Airframe) -> tuple[str, float]:
    """What an aircraft that something else flies (LQR, MPC, a policy, a script) is doing, from its pose alone:
    crashed upside down, flying clear of the ground, or landed; and its height above the ground under it."""
    root = frame.root
    p = data.xpos[root]
    ground = ground_height(model, data, p, frame.descendants, root)
    altitude = float(p[2] - ground - frame.rest_height) if ground is not None else float(p[2])
    if float(data.xmat[root][8]) < UPSIDE_DOWN:
        return "crashed", altitude
    return ("flying" if altitude > 0.05 else "landed"), altitude


def ground_height(
    model: mujoco.MjModel, data: mujoco.MjData, point: np.ndarray, own: frozenset[int] = frozenset(), root: int = -1
) -> float | None:
    """The height of the first thing straight below `point` that is not one of the bodies `own`."""
    hit = cast(model, data, point, DOWN, own, root)
    return None if hit is None else float(point[2]) - hit.distance


@dataclass(frozen=True)
class RayHit:
    distance: float  # m from where the ray starts
    geom: int
    normal: np.ndarray  # the surface's outward normal there: along the ray means it left something it was inside


def cast(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start: np.ndarray,
    direction: np.ndarray,
    own: frozenset[int] = frozenset(),
    root: int = -1,
    reach: float = math.inf,
) -> RayHit | None:
    """The first solid thing a ray from `start` along `direction` (a unit vector) meets within `reach`, past the
    bodies `own` (the aircraft's), or None."""
    at = np.array(start, float)
    way = np.asarray(direction, float)
    geom = np.zeros(1, dtype=np.int32)
    normal = np.zeros(3)
    travelled = 0.0
    for _ in range(6):
        dist = mujoco.mj_ray(model, data, at, way, SOLID, 1, root, geom, normal)
        if dist < 0 or travelled + dist > reach:
            return None
        if int(model.geom_bodyid[geom[0]]) not in own:
            return RayHit(travelled + dist, int(geom[0]), normal.copy())
        travelled += dist + 1e-4
        at = at + way * (dist + 1e-4)
    return None
