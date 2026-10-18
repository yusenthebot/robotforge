"""What makes a robot an aircraft: rotors (thrust actuators) and the mixer that shares a command among them.

A rotor is a MuJoCo motor on a site: it pushes along the site's axis (gear[0:3]) and, through gear[3:6],
twists the airframe the other way the propeller spins (the reaction torque). Any layout works: a quadrotor,
a hexacopter built from parts with its rotors where its builder put them, the Skydio X2 from Menagerie, or the
Crazyflie, whose model has one thrust actuator and three moment actuators instead of four rotors.

The allocation matrix B says what one unit of each actuator's control does to the airframe about its center of
mass: thrust along the body's up axis, and roll, pitch and yaw torque. The mixer inverts it (a pseudo-inverse, so
six or eight rotors share the work) and handles saturation the way flight controllers do: roll and pitch first
(an aircraft that cannot hold its attitude falls out of the sky), then the collective thrust (height), then yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

EPS = 1e-9
RANK_TOL = 1e-6  # singular values below this share of the largest: the rotors cannot make that motion
MIN_LIFT = 1.1  # the rotors must lift at least this many times the aircraft's weight to fly
# An aircraft has at least this many rotors pushing up, lifting at least this share of its weight together: one
# fan on a car or a boat pushes it along, and the car is still a car. Short of its weight (but not of half) it is
# still meant to fly, and is told to get stronger rotors.
MIN_ROTORS = 3
MEANT_TO_LIFT = 0.5
UPRIGHT = 0.7  # a rotor whose axis is this close to straight up pushes up
FOOT_BAND = 0.02  # m above its lowest point a part still counts as what it stands on
MIN_FOOT = 0.03  # m


class AirframeError(ValueError):
    """User-facing: says why an aircraft cannot fly and what to change."""


@dataclass(frozen=True)
class Rotor:
    actuator: int
    name: str  # the actuator's name without the robot prefix
    body: int  # the body its site is on
    position: np.ndarray  # site position in the root body frame, relative to the center of mass (m)
    axis: np.ndarray  # thrust direction in the root body frame (unit), zero for a pure moment actuator
    push: np.ndarray  # force on the airframe per newton of actuator force, root body frame (gear[0:3] turned)
    spin: int  # +1: turns counter-clockwise about its axis, -1 clockwise, 0 no reaction torque modelled
    low: float  # control range
    high: float


@dataclass
class Mix:
    """What the mixer could give of what was asked."""

    controls: np.ndarray
    thrust: float  # N, collective thrust it gives
    attitude_share: float  # 1: roll and pitch torque as asked; less: scaled down to fit
    yaw_share: float  # 1: yaw torque as asked
    thrust_clipped: bool  # the collective thrust was changed to make room for the attitude

    @property
    def saturated(self) -> bool:
        return self.attitude_share < 1.0 or self.yaw_share < 1.0 or self.thrust_clipped


@dataclass
class Mixer:
    """Shares [thrust, roll, pitch, yaw torque] among actuators within their ranges (see the module notes)."""

    allocation: np.ndarray  # 4 x n
    low: np.ndarray
    high: np.ndarray
    pinv: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.pinv = np.linalg.pinv(self.allocation)

    def mix(self, thrust: float, torque: np.ndarray) -> Mix:
        lift = self.pinv[:, 0]
        att = self.pinv[:, 1:3] @ np.asarray(torque[:2], float)
        yaw = self.pinv[:, 3] * float(torque[2])
        share = 1.0
        lo, hi = self._thrust_range(att)
        if lo > hi:  # roll and pitch do not fit at any thrust: give as much of them as fits
            a, b = 0.0, 1.0
            for _ in range(24):
                m = 0.5 * (a + b)
                lo_m, hi_m = self._thrust_range(att * m)
                if lo_m <= hi_m:
                    a = m
                else:
                    b = m
            share = a
            att = att * share
            lo, hi = self._thrust_range(att)
        given = float(np.clip(thrust, lo, hi))
        u = att + given * lift
        # Yaw gets what room is left: a slow turn is better than a fall.
        beta = 1.0
        up, down = yaw > EPS, yaw < -EPS
        if up.any():
            beta = min(beta, float(np.min((self.high[up] - u[up]) / yaw[up])))
        if down.any():
            beta = min(beta, float(np.min((self.low[down] - u[down]) / yaw[down])))
        beta = max(0.0, beta)
        u = np.clip(u + beta * yaw, self.low, self.high)
        return Mix(u, float(self.allocation[0] @ u), share, beta, abs(given - thrust) > 1e-6 * max(1.0, abs(thrust)))

    def _thrust_range(self, att: np.ndarray) -> tuple[float, float]:
        """The collective thrusts at which `att` plus that thrust stays within every actuator's range."""
        lift = self.pinv[:, 0]
        lo, hi = -np.inf, np.inf
        pos, neg = lift > EPS, lift < -EPS
        flat = ~(pos | neg)
        if flat.any() and (np.any(att[flat] < self.low[flat] - 1e-9) or np.any(att[flat] > self.high[flat] + 1e-9)):
            return np.inf, -np.inf  # a moment actuator out of its range: no thrust helps
        if pos.any():
            lo = max(lo, float(np.max((self.low[pos] - att[pos]) / lift[pos])))
            hi = min(hi, float(np.min((self.high[pos] - att[pos]) / lift[pos])))
        if neg.any():
            lo = max(lo, float(np.max((self.high[neg] - att[neg]) / lift[neg])))
            hi = min(hi, float(np.min((self.low[neg] - att[neg]) / lift[neg])))
        return lo, hi

    def max_thrust(self) -> float:
        """The most collective thrust it makes with no torque at all (N)."""
        return self.mix(1e9, np.zeros(3)).thrust


@dataclass
class Airframe:
    robot: str
    root: int  # the root body (with the free joint)
    rotors: tuple[Rotor, ...]
    mass: float  # kg, everything the rotors carry
    inertia: np.ndarray  # 3x3 about the center of mass, in the root body frame
    com: np.ndarray  # center of mass in the root body frame
    gravity: float  # m/s^2
    mixer: Mixer
    rest_height: float  # m from the lowest point (feet, skids) up to the root body's origin, standing level
    radius: float  # m from the center to the furthest rotor tip seen from above
    descendants: frozenset[int]  # every body of the aircraft
    height: float = 0.0  # m from the root body's origin up to the top of the aircraft (a GPS dome, a camera)
    foot: float = 0.05  # m from its center out to the furthest point it stands on (a skid's end, a foot), from above

    @property
    def actuators(self) -> list[int]:
        return [r.actuator for r in self.rotors]

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    def describe(self) -> dict:
        lift = self.mixer.max_thrust()
        return {
            "rotors": len([r for r in self.rotors if np.linalg.norm(r.axis) > 0]) or len(self.rotors),
            "mass_kg": round(self.mass, 3),
            "thrust_to_weight": round(lift / self.weight, 2),
        }


def thrust_actuators(model: mujoco.MjModel, robot: str, root: int | None = None) -> list[int]:
    """The robot's actuators that push or twist its airframe through a site on it: motors (the control is the
    force) without a reference site. Cheap enough to call while listing robots."""
    prefix = f"{robot}/"
    found = []
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != mujoco.mjtTrn.mjTRN_SITE or int(model.actuator_trnid[a, 1]) != -1:
            continue
        if int(model.actuator_gaintype[a]) != mujoco.mjtGain.mjGAIN_FIXED:
            continue
        if int(model.actuator_biastype[a]) != mujoco.mjtBias.mjBIAS_NONE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if not name.startswith(prefix) or not np.any(model.actuator_gear[a]):
            continue
        site = int(model.actuator_trnid[a, 0])
        if "thruster" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site) or "").lower():
            continue  # a marine thruster (builder/thrusters.py names them so): it pushes in water, it does not fly
        site_body = int(model.site_bodyid[site])
        if root is not None and int(model.body_weldid[site_body]) != int(model.body_weldid[root]):
            continue  # on a part that moves (a tilting arm): not part of the rigid airframe
        found.append(a)
    return found


def root_of(model: mujoco.MjModel, robot: str) -> int | None:
    """The robot's root body if it moves freely (a free joint), else None."""
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(f"{robot}/") and int(model.body_parentid[b]) == 0:
            start, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
            free = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(start, start + count))
            return b if free else None
    return None


def is_aircraft(model: mujoco.MjModel, robot: str) -> bool:
    """A robot that moves freely with rotors that could carry it (see MIN_ROTORS). Whether they can also steer it
    is find_airframe's business: a builder who gave every rotor the same spin is told why it cannot fly, not that
    it is no aircraft."""
    root = root_of(model, robot)
    if root is None:
        return False
    actuators = thrust_actuators(model, robot, root)
    if len(actuators) < MIN_ROTORS:
        return False
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[key] if key >= 0 else model.qpos0
    mujoco.mj_kinematics(model, data)
    rot = data.xmat[root].reshape(3, 3)
    mass = float(model.body_subtreemass[root])
    up, lift = 0, 0.0
    for a in actuators:
        gain = float(model.actuator_gainprm[a, 0])
        site = data.site_xmat[int(model.actuator_trnid[a, 0])].reshape(3, 3)
        force = gain * (rot.T @ site @ model.actuator_gear[a, :3])
        norm = float(np.linalg.norm(force))
        if norm > EPS and force[2] >= UPRIGHT * norm:
            up += 1
            lift += float(force[2]) * _control_range(model, a, gain, mass, norm)[1]
    gravity = float(np.linalg.norm(model.opt.gravity)) or 9.81
    return up >= MIN_ROTORS and lift >= MEANT_TO_LIFT * mass * gravity


def _control_range(model: mujoco.MjModel, a: int, gain: float, mass: float, norm: float) -> tuple[float, float]:
    """The range of a rotor's control: its own, from its force limit, or (unlimited) enough to lift anything."""
    if model.actuator_ctrllimited[a]:
        low, high = (float(v) for v in model.actuator_ctrlrange[a])
        return low, high
    if model.actuator_forcelimited[a]:
        low, high = (float(v) / max(abs(gain), EPS) for v in model.actuator_forcerange[a])
        return low, high
    return (0.0, 4 * mass * 9.81) if norm > EPS else (-1.0, 1.0)


def viewer_rotors(model: mujoco.MjModel, robots: list[str]) -> list[dict]:
    """What the viewer spins: each rotor that has a body of its own (the drone template's and the builder's
    propellers; a model that draws its propellers into the airframe's mesh, like the Skydio X2, has none to spin).
    Per rotor: its body, the actuator whose command sets its speed, the axis and point it turns about (body
    frame), which way, the command at full throttle and the propeller's radius."""
    out = []
    for robot in robots:
        root = root_of(model, robot)
        if root is None:
            continue
        acts = thrust_actuators(model, robot, root)
        sites = [int(model.actuator_trnid[a, 0]) for a in acts]
        bodies = [int(model.site_bodyid[s]) for s in sites]
        for a, site, body in zip(acts, sites, bodies, strict=True):
            if body == root or bodies.count(body) > 1 or not np.any(model.actuator_gear[a, :3]):
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.site_quat[site])
            axis = rot.reshape(3, 3) @ (model.actuator_gear[a, :3] / np.linalg.norm(model.actuator_gear[a, :3]))
            reaction = float(model.actuator_gear[a, 5])
            discs = [
                float(model.geom_size[g, 0])
                for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) == body and int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_CYLINDER
            ]
            out.append(
                {
                    "body": body,
                    "actuator": a,
                    "center": [round(float(v), 5) for v in model.site_pos[site]],
                    "axis": [round(float(v), 5) for v in axis],
                    "spin": -1 if reaction > 0 else 1,  # the body turns against its propeller
                    "max": round(float(model.actuator_ctrlrange[a, 1]), 5) if model.actuator_ctrllimited[a] else 1.0,
                    "radius": round(max(discs, default=0.05), 4),
                }
            )
    return out


def find_airframe(model: mujoco.MjModel, robot: str, qpos: np.ndarray | None = None) -> Airframe:
    """Measure the aircraft in pose `qpos` (the home pose by default): its rotors, mass, inertia and mixer.
    Raises AirframeError saying why it cannot fly (too heavy, rotors that cannot turn it)."""
    root = root_of(model, robot)
    if root is None:
        raise AirframeError(f"{robot} does not move freely (its base is fixed), so it cannot fly")
    actuators = thrust_actuators(model, robot, root)
    if not actuators:
        raise AirframeError(f"{robot} has no rotors (thrust actuators on its body), so it cannot fly")
    data = mujoco.MjData(model)
    if qpos is None:
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        qpos = model.key_qpos[key] if key >= 0 else model.qpos0
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    rot = data.xmat[root].reshape(3, 3).copy()
    origin = data.xpos[root].copy()
    bodies = frozenset(b for b in range(model.nbody) if int(model.body_rootid[b]) == root)
    mass = float(model.body_subtreemass[root])
    com_world = data.subtree_com[root].copy()
    com = rot.T @ (com_world - origin)
    inertia = _inertia(model, data, bodies, com_world, rot)

    rotors, columns = [], []
    for a in actuators:
        site = int(model.actuator_trnid[a, 0])
        srot = rot.T @ data.site_xmat[site].reshape(3, 3)
        spos = rot.T @ (data.site_xpos[site] - origin) - com
        gear = model.actuator_gear[a]
        gain = float(model.actuator_gainprm[a, 0])
        force = gain * (srot @ gear[:3])
        torque = gain * (srot @ gear[3:6]) + np.cross(spos, force)
        columns.append([force[2], *torque])
        norm = float(np.linalg.norm(force))
        axis = force / norm if norm > EPS else np.zeros(3)
        reaction = float((srot @ gear[3:6]) @ axis) if norm > EPS else 0.0
        low, high = _control_range(model, a, gain, mass, norm)
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").split("/", 1)[-1]
        spin = 0 if abs(reaction) < EPS else (-1 if reaction > 0 else 1)  # the body turns against the propeller
        push = srot @ gear[:3]
        rotors.append(Rotor(a, name, int(model.site_bodyid[site]), spos, axis, push, spin, low, high))
    allocation = np.array(columns, float).T  # 4 x n
    mixer = Mixer(allocation, np.array([r.low for r in rotors]), np.array([r.high for r in rotors]))
    gravity = float(np.linalg.norm(model.opt.gravity)) or 9.81
    frame = Airframe(
        robot=robot,
        root=root,
        rotors=tuple(rotors),
        mass=mass,
        inertia=inertia,
        com=com,
        gravity=gravity,
        mixer=mixer,
        rest_height=_rest_height(model, data, bodies, root),
        radius=_radius(model, data, bodies, origin),
        descendants=bodies,
        height=_height(model, data, bodies, origin),
        foot=_foot(model, data, bodies, origin),
    )
    _check_flies(frame)
    return frame


def _check_flies(frame: Airframe) -> None:
    b = frame.mixer.allocation
    sv = np.linalg.svd(b, compute_uv=False)
    if len(sv) < 4 or sv[-1] < RANK_TOL * sv[0]:
        missing = _missing_motion(b)
        raise AirframeError(f"{frame.robot}'s rotors cannot {missing}; {_layout_hint(frame, missing)}")
    lift = frame.mixer.max_thrust()
    if lift < MIN_LIFT * frame.weight:
        raise AirframeError(
            f"{frame.robot}'s rotors lift at most {lift:.1f} N and it weighs {frame.weight:.1f} N "
            f"({frame.mass:.2f} kg): give it stronger rotors (more thrust), more rotors or a lighter body"
        )


def _missing_motion(b: np.ndarray) -> str:
    """Which of thrust, roll, pitch and yaw the rotors cannot make, in words."""
    _, _, vt = np.linalg.svd(b.T)  # rows of vt beyond the rank: wrench directions no control reaches
    rank = int(np.sum(np.linalg.svd(b, compute_uv=False) > RANK_TOL * np.abs(b).max()))
    names = ["lift it", "roll it", "pitch it", "turn it (yaw)"]
    lacking = {names[int(np.argmax(np.abs(v)))] for v in vt[rank:]}
    return " or ".join(sorted(lacking, key=names.index)) or "control it"


def _layout_hint(frame: Airframe, missing: str) -> str:
    lifting = [r for r in frame.rotors if np.linalg.norm(r.axis) > 0]
    if len(lifting) < 4:
        return (
            f"{len(lifting)} rotors are too few to give thrust and all three turns at once (a tricopter tilts its "
            "tail rotor to turn, which is not built in): use 4, 6 or 8 rotors"
        )
    if "yaw" in missing:
        spins = [r.spin for r in frame.rotors]
        if all(s == spins[0] for s in spins):
            return 'half its rotors must spin the other way (spin: "cw" on every other rotor)'
        return "put the rotors that spin the same way opposite each other"
    if "lift" in missing:
        return "point its rotors up"
    return "spread its rotors around its center (in front of and behind it, left and right of it)"


def _inertia(model: mujoco.MjModel, data: mujoco.MjData, bodies, com: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """The aircraft's inertia about its center of mass, in the root body frame."""
    total = np.zeros((3, 3))
    for b in bodies:
        m = float(model.body_mass[b])
        if m <= 0:
            continue
        r_i = data.ximat[b].reshape(3, 3)
        total += r_i @ np.diag(model.body_inertia[b]) @ r_i.T
        d = data.xipos[b] - com
        total += m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return rot.T @ total @ rot


def _rest_height(model: mujoco.MjModel, data: mujoco.MjData, bodies, root: int) -> float:
    """How high the root body's origin is above the lowest point of the aircraft in the pose measured (its home
    pose: level, standing on its gear), so its height above the ground is known from where its origin is."""
    from robotforge.sim.geometry import collides, lowest_point

    lows = [
        lowest_point(model, data, g)[2] for g in range(model.ngeom) if g_in(model, g, bodies) and collides(model, g)
    ]
    return float(data.xpos[root][2] - min(lows)) if lows else 0.0


def _foot(model: mujoco.MjModel, data: mujoco.MjData, bodies, origin: np.ndarray) -> float:
    """How far from its center (seen from above) the parts it stands on reach: the solid parts that come within
    FOOT_BAND of its lowest point, in the pose measured. A spot to land on must carry it out to there, whichever
    way it faces (a skid off a table's edge tips it over). At least MIN_FOOT: nothing stands on a point."""
    from robotforge.behaviors.nav_map import world_extent
    from robotforge.sim.geometry import collides, lowest_point

    solid = [g for g in range(model.ngeom) if g_in(model, g, bodies) and collides(model, g)]
    if not solid:
        return MIN_FOOT
    bottom = min(float(lowest_point(model, data, g)[2]) for g in solid)
    reach = MIN_FOOT
    for g in solid:
        lo, hi = world_extent(model, data, g)
        if lo[2] > bottom + FOOT_BAND:
            continue
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                reach = max(reach, float(np.hypot(x - origin[0], y - origin[1])))
    return reach


def _radius(model: mujoco.MjModel, data: mujoco.MjData, bodies, origin: np.ndarray) -> float:
    reach = 0.05
    for g in range(model.ngeom):
        if not g_in(model, g, bodies):
            continue
        center = data.geom_xpos[g] - origin
        reach = max(reach, float(np.hypot(center[0], center[1]) + model.geom_rbound[g]))
    return reach


def _height(model: mujoco.MjModel, data: mujoco.MjData, bodies, origin: np.ndarray) -> float:
    """How far its top (drawn parts too: flown under a table, a dome through the tabletop shows) is above its
    origin."""
    from robotforge.behaviors.nav_map import world_extent

    tops = [float(world_extent(model, data, g)[1][2]) for g in range(model.ngeom) if g_in(model, g, bodies)]
    return max(0.0, max(tops, default=float(origin[2])) - float(origin[2]))


def g_in(model: mujoco.MjModel, geom: int, bodies) -> bool:
    return int(model.geom_bodyid[geom]) in bodies
