"""Piloting boats and underwater robots with their thrusters.

A robot's marine thrusters (site actuators named "..._thruster", see sim/water.py) are found with where they sit
and which way they push, and a least-squares allocation turns the forces and turns a pilot wants into thruster
commands. Both pilots know their own drag from the water model (how hard the water holds them back at a given
speed), so they feed that thrust forward and a small PI loop only corrects the rest:

- BoatPilot: speed and turn on the surface, like a rover's (keyboard, move, and the path follower of move(to=...)).
- RovPilot: the same speed and turn while it holds its depth and heading by PID, plus heave (up and down) and
  going to a point under water (DiveTrip, for move(to=[x, y, z])).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.sim import water as water_model

RATE_HZ = 50.0
TOP_SHARE = 0.85  # of the thrusters' most forward push, spent at top speed: the rest is kept to steer with
GRAVITY = 9.81
DEFAULT_SPEED = 0.5  # m/s, until the pilot has measured its own drag in the water
SURFACE_MARGIN = 0.12  # m an ROV keeps its center below the surface when told to go up
INTEGRATE_NEAR = 0.1  # m/s or rad/s: the speed and turn integrals gather only this near what was asked
DEPTH_GAIN = 0.8  # m/s of climb or dive per meter off the depth it holds: the last stretch closes in 1.25 s
POSITION_GAIN = 0.8  # the same for holding a point (m/s per meter off it)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Thruster:
    actuator: int
    name: str
    push: float  # N at full command
    direction: np.ndarray  # unit vector, in the robot's root body frame
    position: np.ndarray  # from the robot's center of mass, in the root body frame


def root_body(model: mujoco.MjModel, robot: str) -> int:
    prefix = f"{robot}/"
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0 and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
            prefix
        ):
            return b
    raise ValueError(f"robot {robot!r} has no body in the simulation")


def _thruster_actuators(model: mujoco.MjModel, robot: str) -> list[int]:
    prefix, out = f"{robot}/", []
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_SITE:
            continue
        if not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(prefix):
            continue
        site = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(model.actuator_trnid[a, 0])) or ""
        if "thruster" in site.lower():
            out.append(a)
    return out


def thrusters(model: mujoco.MjModel, robot: str, qpos: np.ndarray | None = None) -> list[Thruster]:
    """The robot's marine thrusters, as they sit at `qpos` (the model's own when None)."""
    acts = _thruster_actuators(model, robot)
    if not acts:
        return []
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    root = root_body(model, robot)
    rot = data.xmat[root].reshape(3, 3)
    com = data.subtree_com[root]
    out = []
    for a in acts:
        site = int(model.actuator_trnid[a, 0])
        gear = model.actuator_gear[a][:3]
        push = float(model.actuator_gainprm[a][0])
        if model.actuator_ctrllimited[a]:
            push *= float(np.abs(model.actuator_ctrlrange[a]).max())
        world = data.site_xmat[site].reshape(3, 3) @ gear
        norm = float(np.linalg.norm(world))
        if norm < 1e-9 or push <= 0:
            continue
        out.append(
            Thruster(
                a,
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}",
                push * norm,
                rot.T @ (world / norm),
                rot.T @ (data.site_xpos[site] - com),
            )
        )
    return out


def marine_kind(model: mujoco.MjModel, robot: str) -> str | None:
    """'rov' for a robot whose thrusters push up and down too, 'boat' for one whose thrusters all push flat, None
    for a robot without marine thrusters."""
    if not _thruster_actuators(model, robot):
        return None
    found = thrusters(model, robot)
    if not found:
        return None
    return "rov" if any(abs(t.direction[2]) > 0.7 for t in found) else "boat"


def _allocation(found: list[Thruster], rows: list[int]) -> np.ndarray:
    """The pseudo-inverse taking a wanted wrench (body frame, the components in `rows` of [Fx, Fy, Fz, Mx, My, Mz])
    to thruster commands."""
    columns = [np.concatenate([t.direction, np.cross(t.position, t.direction)]) * t.push for t in found]
    matrix = np.array(columns).T[rows]
    return np.linalg.pinv(matrix)


def _speed_for(force: float, c1: float, c2: float) -> float:
    """The speed at which c1 v + c2 v^2 equals `force`."""
    if c2 > 1e-9:
        return (-c1 + math.sqrt(c1 * c1 + 4 * c2 * force)) / (2 * c2)
    return force / c1 if c1 > 1e-9 else DEFAULT_SPEED


def _fit(steer: np.ndarray, drive: np.ndarray) -> np.ndarray:
    """steer + s * drive with the largest s in [0, 1] that keeps every command within -1..1 (steering comes first:
    a boat at full speed must still turn); steer itself is scaled down if it alone is too much."""
    top = float(np.abs(steer).max()) if len(steer) else 0.0
    if top > 1.0:
        steer = steer / top
    s = 1.0
    for a, b in zip(steer, drive, strict=True):
        if abs(b) < 1e-12:
            continue
        limit = (1.0 - a) / b if b > 0 else (-1.0 - a) / b
        s = min(s, max(limit, 0.0))
    return steer + s * drive


class _Pilot:
    """What both pilots share: the robot's thrusters, its mass and inertia, its velocity in its own frame."""

    rate_hz = RATE_HZ

    def __init__(self, model: mujoco.MjModel, robot: str, found: list[Thruster]) -> None:
        self.robot = robot
        self.root = root_body(model, robot)
        self.thrusters = found
        self.actuators = np.array([t.actuator for t in found], int)
        self.mass = float(model.body_subtreemass[self.root])
        # Roll, pitch and yaw inertia in the robot's own frame: MuJoCo keeps a body's inertia along its principal
        # axes, which for a boat can be any way round (two nearly equal moments).
        principal = np.zeros(9)
        mujoco.mju_quat2Mat(principal, model.body_iquat[self.root])
        principal = principal.reshape(3, 3)
        tensor = principal @ np.diag(model.body_inertia[self.root]) @ principal.T
        self.inertia = np.maximum(np.diag(tensor).copy(), 1e-4)
        self.speed = 0.0
        self.turn = 0.0
        self._vel = np.zeros(6)

    def _motion(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotation, linear velocity (own frame) and angular velocity (world frame) of the robot's root body. XBODY:
        the body's own frame, not its principal axes of inertia (mjOBJ_BODY), which are turned on a boat."""
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, self.root, self._vel, 1)
        rot = data.xmat[self.root].reshape(3, 3)
        return rot, self._vel[3:].copy(), rot @ self._vel[:3]

    def heading(self, data: mujoco.MjData) -> float:
        rot = data.xmat[self.root]
        return math.atan2(rot[3], rot[0])


class BoatPilot(_Pilot):
    """Speed (m/s, along its bow) and turn (rad/s) on the water with flat-pushing thrusters."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        hydro: water_model.Hydro | None = None,
        data: mujoco.MjData | None = None,
    ) -> None:
        found = [t for t in thrusters(model, robot) if abs(t.direction[2]) < 0.5]
        if not found:
            raise ValueError(f"robot {robot!r} has no thrusters that push it along the water; is it a boat?")
        super().__init__(model, robot, found)
        self._pinv = _allocation(found, [0, 5])  # surge force and yaw moment
        self.max_push = sum(t.push * abs(t.direction[0]) for t in found)
        arm = sum(t.push * abs(np.cross(t.position, t.direction)[2]) for t in found)
        self.drag = (0.0, 0.0)
        self.turn_drag = (0.0, 0.0)
        if hydro is not None and data is not None:
            rot = data.xmat[self.root].reshape(3, 3)
            self.drag = hydro.resistance(data, self.root, along=rot[:, 0])
            self.turn_drag = hydro.resistance(data, self.root, about=np.array([0.0, 0.0, 1.0]))
        wet = self.drag[0] + self.drag[1] > 0
        self.max_speed = _speed_for(TOP_SHARE * self.max_push, *self.drag) if wet else DEFAULT_SPEED
        self.max_turn = min(_speed_for(TOP_SHARE * arm, *self.turn_drag), 1.5) if wet else 0.8
        self.max_reverse = 0.5 * self.max_speed
        self.cruise = 0.6 * self.max_speed  # move(to=...)'s pace: a boat carries way, it slows for the corners
        self._ki_speed = 0.0
        self._ki_turn = 0.0

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        self.speed = float(np.clip(speed, -self.max_reverse, self.max_speed))
        self.turn = float(np.clip(turn, -self.max_turn, self.max_turn))

    def status(self) -> dict[str, Any]:
        return {
            "behavior": "sail",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dt = 1.0 / self.rate_hz
        _, v, w = self._motion(model, data)
        c1, c2 = self.drag
        r1, r2 = self.turn_drag
        e_speed = self.speed - v[0]
        e_turn = self.turn - w[2]
        # The integrals only trim what the drag model misses: they gather near the target, not while it speeds
        # up (from a stand they had wound up to 18 N and took it 10 percent past its speed).
        if abs(e_speed) < INTEGRATE_NEAR:
            self._ki_speed = float(np.clip(self._ki_speed + e_speed * dt, -0.3, 0.3))
        if abs(e_turn) < INTEGRATE_NEAR:
            self._ki_turn = float(np.clip(self._ki_turn + e_turn * dt, -0.3, 0.3))
        surge = c1 * self.speed + c2 * self.speed * abs(self.speed) + self.mass * (1.5 * e_speed + 0.5 * self._ki_speed)
        yaw = r1 * self.turn + r2 * self.turn * abs(self.turn) + self.inertia[2] * (4.0 * e_turn + 2.0 * self._ki_turn)
        drive = self._pinv @ np.array([surge, 0.0])
        steer = self._pinv @ np.array([0.0, yaw])
        data.ctrl[self.actuators] = _fit(steer, drive)


class RovPilot(_Pilot):
    """An ROV that holds its depth and heading while it goes: speed and turn as a boat, heave (m/s, up) moves the
    depth it holds, and go_to steers it to a point under water."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        hydro: water_model.Hydro | None = None,
        data: mujoco.MjData | None = None,
    ) -> None:
        found = thrusters(model, robot)
        if not any(abs(t.direction[2]) > 0.7 for t in found):
            raise ValueError(f"robot {robot!r} has no thrusters that push it up and down; is it an ROV?")
        super().__init__(model, robot, found)
        self._pinv = _allocation(found, [0, 1, 2, 3, 4, 5])
        self.water = hydro.water if hydro is not None else None
        self.lift = 0.0  # N it would rise with fully under water, all thrusters off
        self.drag = (0.0, 0.0)
        self.sway_drag = (0.0, 0.0)
        self.heave_drag = (0.0, 0.0)
        self.turn_drag = (0.0, 0.0)
        self.max_push = sum(t.push * abs(t.direction[0]) for t in found)
        if hydro is not None:
            self.lift = hydro.buoyancy(self.root) - self.mass * GRAVITY
        if hydro is not None and data is not None:
            # Its drag as it would be fully under water: it is pulled along at the surface where it starts.
            rot = data.xmat[self.root].reshape(3, 3)
            under = np.full(len(hydro.bodies), -2.0 * hydro.water.level)
            self.drag = hydro.resistance(data, self.root, along=rot[:, 0], lift=under)
            self.sway_drag = hydro.resistance(data, self.root, along=rot[:, 1], lift=under)
            self.heave_drag = hydro.resistance(data, self.root, along=np.array([0.0, 0.0, 1.0]), lift=under)
            self.turn_drag = hydro.resistance(data, self.root, about=np.array([0.0, 0.0, 1.0]), lift=under)
        wet = self.drag[0] + self.drag[1] > 0
        # Its share of the most forward push, with a good part kept for depth and heading.
        self.max_speed = min(_speed_for(0.6 * self.max_push, *self.drag), 1.5) if wet else DEFAULT_SPEED
        self.max_turn = 1.0
        self.max_reverse = 0.6 * self.max_speed
        self.max_heave = 0.4
        self.heave = 0.0
        self.depth_target: float | None = None
        self.yaw_target: float | None = None
        self.goal: np.ndarray | None = None  # a point to go to (DiveTrip)
        self.cruise = 0.6 * self.max_speed
        self._ki_z = 0.0
        self._ki_v = 0.0
        self._ki_xy = np.zeros(2)

    def command(self, speed: float = 0.0, turn: float = 0.0, heave: float | None = None) -> None:
        self.goal = None
        self.speed = float(np.clip(speed, -self.max_reverse, self.max_speed))
        self.turn = float(np.clip(turn, -self.max_turn, self.max_turn))
        if heave is not None:
            self.heave = float(np.clip(heave, -self.max_heave, self.max_heave))

    def go_to(self, point: np.ndarray, cruise: float | None = None) -> None:
        self.goal = np.asarray(point, float)
        self.speed = self.turn = self.heave = 0.0
        if cruise is not None:
            self.cruise = min(abs(cruise), self.max_speed)

    def status(self) -> dict[str, Any]:
        out = {
            "behavior": "dive",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }
        if self.depth_target is not None:
            out["holding_z"] = round(self.depth_target, 3)
        if self.heave:
            out["heave"] = round(self.heave, 3)
        return out

    def _clamp_depth(self, z: float) -> float:
        """Never above the surface: a target in the air would only make it porpoise."""
        return z if self.water is None else min(z, self.water.level - SURFACE_MARGIN)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        dt = 1.0 / self.rate_hz
        rot, v, w = self._motion(model, data)
        pos = data.subtree_com[self.root]
        yaw = self.heading(data)
        world_v = rot @ v
        if self.depth_target is None:
            self.depth_target = float(pos[2])
        if self.yaw_target is None:
            self.yaw_target = yaw
        if self.heave:
            self.depth_target += self.heave * dt
        speed, turn = self.speed, self.turn
        hold_xy = None
        if self.goal is not None:
            self.depth_target = float(self.goal[2])
            away = self.goal[:2] - pos[:2]
            dist = float(np.linalg.norm(away))
            if dist > 0.4:
                self.yaw_target = math.atan2(away[1], away[0])
                off = abs(_wrap(self.yaw_target - yaw))
                speed = min(self.cruise, 0.6 * dist) * max(0.0, math.cos(off)) ** 2
            else:
                hold_xy = self.goal[:2]
        self.depth_target = self._clamp_depth(self.depth_target)
        # Depth: the error asks for a vertical speed (DEPTH_GAIN per meter, at most max_heave; with heave, that
        # speed), and a PI loop on the speed gets it, its heave drag and buoyancy fed forward. The water's own drag
        # damps it: a PD on depth alone was so overdamped that it crept the last ten centimeters for seconds.
        e_z = self.depth_target - float(pos[2])
        vz = float(np.clip(DEPTH_GAIN * e_z, -self.max_heave, self.max_heave))
        if self.heave:
            vz = self.heave
        e_vz = vz - float(world_v[2])
        if abs(e_vz) < INTEGRATE_NEAR:
            self._ki_z = float(np.clip(self._ki_z + e_vz * dt, -0.5, 0.5))
        h1, h2 = self.heave_drag
        fz = h1 * vz + h2 * vz * abs(vz) + self.mass * (3.0 * e_vz + 2.0 * self._ki_z) - self.lift
        # Heading: follow the turn rate asked, or hold the heading (PD, 2 rad/s).
        if turn:
            self.yaw_target = yaw
            r1, r2 = self.turn_drag
            mz = r1 * turn + r2 * turn * abs(turn) + self.inertia[2] * 4.0 * (turn - w[2])
        else:
            mz = self.inertia[2] * (4.0 * _wrap(self.yaw_target - yaw) - 3.6 * w[2])
        # Along and across: the speed asked (drag fed forward) with no drift sideways; at the goal, a point held
        # the same way as the depth (the error asks for a speed, the speed loop gets it).
        if hold_xy is not None:
            e_xy = hold_xy - pos[:2]
            want = POSITION_GAIN * e_xy
            norm = float(np.linalg.norm(want))
            if norm > self.cruise:
                want *= self.cruise / norm
            local_want = rot[:2, :2].T @ want
            e_local = local_want - v[:2]
            self._ki_xy = np.clip(self._ki_xy + e_local * dt, -0.3, 0.3)
            (c1, c2), (s1, s2) = self.drag, self.sway_drag
            fx = c1 * local_want[0] + c2 * local_want[0] * abs(local_want[0])
            fy = s1 * local_want[1] + s2 * local_want[1] * abs(local_want[1])
            fx += self.mass * (2.0 * e_local[0] + 1.0 * self._ki_xy[0])
            fy += self.mass * (2.0 * e_local[1] + 1.0 * self._ki_xy[1])
        else:
            self._ki_xy[:] = 0.0
            c1, c2 = self.drag
            e_v = speed - v[0]
            if abs(e_v) < INTEGRATE_NEAR:
                self._ki_v = float(np.clip(self._ki_v + e_v * dt, -0.3, 0.3))
            fx = c1 * speed + c2 * speed * abs(speed) + self.mass * (1.5 * e_v + 0.5 * self._ki_v)
            fy = -self.mass * 2.0 * v[1]
        # Keep it level: roll and pitch back toward upright (the foam and the battery do most of it).
        up = rot.T @ np.array([0.0, 0.0, 1.0])
        tilt = np.cross(np.array([0.0, 0.0, 1.0]), up)  # body-frame axis to turn about, |tilt| = sin(angle)
        w_body = rot.T @ w
        mx = -self.inertia[0] * (6.0 * -tilt[0] + 3.0 * w_body[0])
        my = -self.inertia[1] * (6.0 * -tilt[1] + 3.0 * w_body[1])
        vertical = rot.T @ np.array([0.0, 0.0, fz])
        hold = self._pinv @ np.array([vertical[0], vertical[1], vertical[2], mx, my, 0.0])
        steer = self._pinv @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, mz])
        drive = self._pinv @ np.array([fx, fy, 0.0, 0.0, 0.0, 0.0])
        data.ctrl[self.actuators] = _fit(_fit(hold, steer), drive)


def pilot_for(kind: str, model: mujoco.MjModel, robot: str, hydro, data) -> _Pilot:
    return RovPilot(model, robot, hydro, data) if kind == "rov" else BoatPilot(model, robot, hydro, data)


NEXT_TO = 0.35  # m left between an ROV's middle and the edge of what it was sent to


def dive_target(
    pilot: RovPilot, to: Any, destination: Any, model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, str | None]:
    """Where an ROV sent to `to` goes: a point [x, y, z] (z left out: the depth it holds), or for an object or a robot
    (`destination`, a navigate.Destination) a spot in front of it at its height. Kept under the surface; the note says
    when the point was moved."""
    here = data.subtree_com[pilot.root]
    holding = pilot.depth_target if pilot.depth_target is not None else float(here[2])
    if destination is None:
        values = [
            float(v)
            for v in (to if isinstance(to, (list, tuple)) else str(to).strip(" []()").replace(",", " ").split())
        ]
        point = np.array([values[0], values[1], values[2] if len(values) == 3 else holding])
    else:
        center = np.array([destination.point[0], destination.point[1], destination.z])
        away = here[:2] - center[:2]
        gap = float(np.linalg.norm(away))
        reach = NEXT_TO + _radius(model, data, destination, center)
        flat = center[:2] + (away / gap * reach if gap > 1e-6 else np.array([reach, 0.0]))
        point = np.array([flat[0], flat[1], center[2]])
    note = None
    if point[2] < 0:
        raise ValueError(f"z {point[2]:g} is below the floor (z is the height above it, in meters)")
    if pilot.water is not None and point[2] > pilot.water.level - SURFACE_MARGIN:
        note = (
            f"z {point[2]:.2f} is at or above the surface ({pilot.water.level:g} m): it goes "
            f"{SURFACE_MARGIN:g} m under it"
        )
        point[2] = pilot.water.level - SURFACE_MARGIN
    return point, note


def _radius(model: mujoco.MjModel, data: mujoco.MjData, destination: Any, center: np.ndarray) -> float:
    """How far what it was sent to reaches from its middle, seen from above."""
    reach = [
        float(np.linalg.norm(data.geom_xpos[g][:2] - center[:2])) + float(model.geom_rbound[g])
        for g in getattr(destination, "geoms", ())
    ]
    return max(reach) if reach else 0.3


# ---- going to a point under water ---------------------------------------------------------------------------

ARRIVE = 0.1  # m from the point that counts as there
STILL = 0.05  # m/s


@dataclass
class _Line:
    """The trip's plan, as move(to=...) reports it: a straight line (no planning under water yet)."""

    points: np.ndarray
    length: float


@dataclass
class DiveTrip:
    """A controller that sends an ROV's pilot to a point and says when it is there (the move(to=...) of an ROV)."""

    robot: str
    pilot: RovPilot
    target: np.ndarray
    start: np.ndarray
    start_time: float
    deadline: float | None
    still_mine: Callable[[], bool] = lambda: True
    on_finish: Callable[[DiveTrip], None] = lambda trip: None
    name: str | None = None  # the object or robot it was sent to
    note: str | None = None
    rate_hz: float = 20.0
    state: str = "diving"
    cause: str | None = None
    end_time: float | None = None
    traveled: float = 0.0
    plan: _Line = field(init=False)
    _last: np.ndarray | None = field(default=None, init=False)
    _last_t: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.plan = _Line(np.array([self.start, self.target]), float(np.linalg.norm(self.target - self.start)))
        self.pilot.go_to(self.target)

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
        pos = data.subtree_com[self.pilot.root].copy()
        if self._last is not None:
            self.traveled += float(np.linalg.norm(pos - self._last))
        self._last = pos
        if not self.still_mine() or self.pilot.goal is None:
            self._finish(t, "stopped", "interrupted", "something else took over the robot (a key, a move or a reset)")
            return
        error = float(np.linalg.norm(self.target - pos))
        speed = float(np.linalg.norm(data.cvel[self.pilot.root][3:]))
        if error <= ARRIVE and speed <= STILL:
            self._finish(t, "arrived")
            return
        if self.deadline is not None and t >= self.deadline - 1e-9:
            self._finish(t, "stopped", "out_of_time", f"it ran out of time {error:.2f} m from the point")

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"behavior": "dive", "navigating": self.robot, "state": self.state, "goal": self._goal()}
        if not self.done and self._last is not None:
            out["path"] = [[round(float(v), 3) for v in self._last], [round(float(v), 3) for v in self.target]]
        if self.note:
            out["note"] = self.note
        return out

    def interrupt(self, note: str) -> None:
        if not self.done:
            self._finish(self._last_t if self._last_t is not None else self.start_time, "stopped", "interrupted", note)

    def _goal(self) -> dict[str, Any]:
        point = [round(float(v), 3) for v in self.target]
        return {"name": self.name, "point": point} if self.name else {"point": point}

    def _finish(self, t: float, state: str, cause: str | None = None, note: str | None = None) -> None:
        self.state, self.cause, self.end_time = state, cause, t
        if note:
            self.note = f"{self.note}; {note}" if self.note else note
        if state == "stopped":
            self.pilot.command(0.0, 0.0)  # it holds the depth and heading it has
        self.on_finish(self)

    def result(self) -> dict[str, Any]:
        end = self.end_time if self.end_time is not None else (self._last_t or self.start_time)
        out: dict[str, Any] = {
            "arrived": self.arrived,
            "destination": self._goal(),
            "path_m": round(self.plan.length, 2),
            "distance_m": round(self.traveled, 2),
            "time_s": round(end - self.start_time, 2),
        }
        if self._last is not None:
            out["error_m"] = round(float(np.linalg.norm(self.target - self._last)), 3)
            if self.pilot.water is not None:
                out["depth_m"] = round(self.pilot.water.level - float(self._last[2]), 3)
        if self.note:
            out["note"] = self.note
        if self.cause and not self.arrived:
            out["why"] = self.cause
        return out
