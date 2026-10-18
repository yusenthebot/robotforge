"""What a robot is like once things are mounted on it: its weight and balance, whether it still stands, how far its
arms reach from where they sit, whether they can hit the robot carrying them and whether they hold what they carry.

Everything is measured on the compiled world with real physics, never guessed from a catalog: the robot settles
for a second from its home pose with its servos holding that pose (it stands, or it sags, tips or falls), the floor
contacts it settles on make its support polygon, and each arm is swung through a fixed set of random poses within
its joint limits, the robot standing still meanwhile. In those poses the robot's centre of mass is compared with the
support polygon (would it tip if the arm reached that way?), the arm's shapes with those of the robot and its other
mounts (could it hit them?), and the gravity torque on each of its joints with what its motor gives (can it hold its
own weight and its end effector's out there?).

A robot that does not stand at home (it tips or falls in the settle, its centre of mass is outside the polygon, or an
arm cannot hold itself at home) is refused, saying what to change. The rest is reported for the agent to read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.sim.geometry import collides, lowest_point
from robotforge.sim.session import fresh_data

SETTLE_S = 1.0  # s the robot stands on its own from its home pose before it is measured
FELL_TILT = 25.0  # degrees: tilted more than this after the settle, it fell over
SAGGED = 0.4  # of its home height above the ground: lower than this after the settle, it collapsed
POSES = 160  # random arm poses swung through
SEED = 7  # the same poses every time, so the same robot reports the same numbers
CONTACT_DEPTH = 0.002  # m: an arm closer than this into its carrier hits it
FLOOR_REACH = 0.03  # m above the floor: a tool point down to here reaches the floor
TIP_MARGIN = 0.01  # m: a centre of mass closer than this to the support polygon's edge is about to tip
# Robots that balance by moving (a humanoid on its policy, a two-wheeled balancer, anything flying or floating) do not
# stand with their servos alone; their settle and support polygon say nothing about them.
MOVING = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
ACTIVE = {"humanoid", "balancer", "aircraft", "boat", "rov", "snake"}


@dataclass
class Checks:
    """The measurements (`report`, for the agent) and, for a robot that cannot stand or hold its arm at home, what is
    wrong and what to change (`refusal`)."""

    report: dict[str, Any] = field(default_factory=dict)
    refusal: str | None = None


def check(model: mujoco.MjModel, robot: str, kind: str, mounted: dict[str, float] | None = None) -> Checks:
    """Measure `robot` (its bodies are "<robot>/...") in the compiled world `model`, a robot of this `kind` (a
    behaviors.Kind value). `mounted` maps the names of what it carries to their mass in kg, for the words."""
    out = Checks()
    bodies = _bodies(model, robot)
    if not bodies:
        raise ValueError(f"no robot {robot!r} in the world")
    root = bodies[0]
    floating = _floating(model, root)
    home = fresh_data(model)
    total = float(model.body_subtreemass[root])
    out.report["mass_kg"] = round(total, 2)
    if mounted:
        out.report["carries_kg"] = {name: round(kg, 2) for name, kg in mounted.items()}
    out.report["center_of_mass"] = _com_words(model, home, root)

    data = home
    polygon: np.ndarray | None = None
    if floating and kind not in ACTIVE:
        data, stand, polygon = _settle(model, home, robot, root)
        out.report["stands"] = stand
        if stand.get("fell"):
            out.refusal = (
                f"{robot} does not stand with this on it: {stand['fell']}; mount it lower or nearer its middle "
                f"(shift), mount something lighter, or put it on a bigger base"
            )
    elif floating:
        out.report["stands"] = {
            "balances": f"a {kind} keeps its balance by moving; its own controller carries the load"
        }

    arms = _arms(model, robot)
    reports = []
    for arm in arms:
        swept = _sweep(model, data, robot, root, arm, polygon)
        reports.append(swept.report)
        if out.refusal is None and swept.refusal:
            out.refusal = swept.refusal
    if arms:
        out.report["arms"] = reports
    if polygon is not None and out.refusal is None:
        margin = _margin(polygon, data.subtree_com[root][:2])
        out.report["stands"]["margin_m"] = round(margin, 3)
        if margin < 0:
            way = _direction(model, data, root, polygon, data.subtree_com[root][:2])
            out.refusal = (
                f"{robot} would tip over: its centre of mass is {-margin * 100:.0f} cm outside what it stands on, "
                f"toward its {way}; shift the mount the other way, mount something lighter, or use a wider base"
            )
    return out


# ---- standing ------------------------------------------------------------------------------------------------------


def _settle(
    model: mujoco.MjModel, home: mujoco.MjData, robot: str, root: int
) -> tuple[mujoco.MjData, dict[str, Any], np.ndarray | None]:
    """The world after SETTLE_S seconds from home, the robot's servos holding their home targets; how it stands
    then; and its support polygon (the convex hull of its contacts with the ground, None if it touches less than
    three points of it)."""
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, home)
    ctrl = home.ctrl.copy()
    height0 = float(home.xpos[root][2]) - _ground(model, home, robot)
    for _ in range(round(SETTLE_S / model.opt.timestep)):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return home, {"fell": "the simulation blew up while it settled"}, None
    up = data.xmat[root].reshape(3, 3) @ (home.xmat[root].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]))
    tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
    height = float(data.xpos[root][2]) - _ground(model, data, robot, floor_only=True)
    stand: dict[str, Any] = {"tilt_deg": round(tilt, 1), "sagged_m": round(max(0.0, height0 - height), 3)}
    if tilt > FELL_TILT:
        stand["fell"] = f"it tipped {tilt:.0f} degrees over within {SETTLE_S:g} s of standing still"
    elif height0 > 0 and height < SAGGED * height0:
        stand["fell"] = f"its body sank from {height0:.2f} m to {height:.2f} m above the ground: its legs gave way"
    points = _support_points(model, data, robot)
    polygon = _hull(points) if len(points) >= 3 else None
    if polygon is None or len(polygon) < 3:
        stand["support"] = "it touches the ground in a line or a point only"
        return data, stand, None
    stand["support_points"] = len(points)
    return data, stand, polygon


def _support_points(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> np.ndarray:
    """Where the robot touches something that is not itself (the floor, the terrain), in world x, y."""
    mine = _geom_owner(model, robot)
    points = []
    for c in data.contact[: data.ncon]:
        a, b = (bool(mine[g]) for g in c.geom)
        if a != b:
            points.append(c.pos[:2].copy())
    return np.array(points) if points else np.zeros((0, 2))


def _ground(model: mujoco.MjModel, data: mujoco.MjData, robot: str, floor_only: bool = False) -> float:
    """The height of the ground under the robot: its lowest colliding point (its feet or wheels stand on it), or
    where it touches the ground when `floor_only` and it does."""
    if floor_only:
        mine = _geom_owner(model, robot)
        heights = [float(c.pos[2]) for c in data.contact[: data.ncon] if bool(mine[c.geom[0]]) != bool(mine[c.geom[1]])]
        if heights:
            return min(heights)
    geoms = [g for g in range(model.ngeom) if _geom_owner(model, robot)[g] and collides(model, g)]
    return min((float(lowest_point(model, data, g)[2]) for g in geoms), default=0.0)


def _hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2D points, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted({(round(float(x), 4), round(float(y), 4)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _margin(polygon: np.ndarray, point: np.ndarray) -> float:
    """How far `point` is inside the convex polygon (m; negative outside it): its distance to the nearest edge."""
    best = math.inf
    outside = 0.0
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        inward = float(edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0])) / length  # left of a CCW edge
        best = min(best, inward)
        if inward < 0:
            t = float(np.clip(np.dot(point - a, edge) / length**2, 0.0, 1.0))
            outside = max(outside, float(np.linalg.norm(point - (a + t * edge))))
    return -outside if best < 0 else best


def _direction(model: mujoco.MjModel, data: mujoco.MjData, root: int, polygon: np.ndarray, point: np.ndarray) -> str:
    """Which way (front, back, left, right, in the robot's own frame) `point` lies from the polygon's middle."""
    away = np.append(point - polygon.mean(axis=0), 0.0)
    local = data.xmat[root].reshape(3, 3).T @ away
    if abs(local[0]) >= abs(local[1]):
        return "front" if local[0] > 0 else "back"
    return "left" if local[1] > 0 else "right"


# ---- arms ------------------------------------------------------------------------------------------------------------


@dataclass
class _Swept:
    report: dict[str, Any]
    refusal: str | None = None


@dataclass(frozen=True)
class _Arm:
    name: str  # as move_arm takes it: "rover/arm", or "panda" for a robot that is an arm
    prefix: str  # name prefix of its bodies and joints: "rover/arm/"
    tcp: int  # site
    joints: tuple[int, ...]  # the hinge and slide joints that move it (not its gripper's fingers)


def _arms(model: mujoco.MjModel, robot: str) -> list[_Arm]:
    """The arms of a robot: one per tool point ("tcp" site), with the joints between its base and that point."""
    arms = []
    for s in range(model.nsite):
        name = model.site(s).name
        if not name.startswith(f"{robot}/") or not (name == f"{robot}/tcp" or name.endswith("/tcp")):
            continue
        prefix = name[: -len("tcp")]
        joints = []
        body = int(model.site_bodyid[s])
        while body > 0:
            for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body]):
                if int(model.jnt_type[j]) in MOVING:
                    joints.append(j)
            body = int(model.body_parentid[body])
        # Only the arm's own joints: a mounted arm's chain runs on into its carrier's body, which has none of these,
        # and a robot's own arm (a mobile manipulator built from parts) has its joints under the robot's prefix.
        joints = [j for j in joints if model.joint(j).name.startswith(prefix)]
        if joints:
            arms.append(_Arm(name.removesuffix("/tcp"), prefix, s, tuple(reversed(joints))))
    return arms


def _sweep(
    model: mujoco.MjModel, settled: mujoco.MjData, robot: str, root: int, arm: _Arm, polygon: np.ndarray | None
) -> _Swept:
    """Swing `arm` through POSES random poses from the settled state and measure its reach, what it can hit, what it
    can hold and, on a robot that stands, which way it would tip it."""
    rng = np.random.default_rng(SEED)
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, settled)
    data.qvel[:] = 0.0
    mine = _geom_owner(model, robot)
    arm_geoms = np.array(
        [model.body(int(model.geom_bodyid[g])).name.startswith(arm.prefix) for g in range(model.ngeom)]
    )
    first = int(model.jnt_bodyid[arm.joints[0]])
    base_body = int(model.body_parentid[first])  # what the arm's first joint turns on: its base, or its carrier
    ground = _ground(model, settled, robot, floor_only=True)
    limits = _limits(model, arm.joints)
    torque_limit = _torque_limits(model, arm.joints)

    def measure() -> tuple[np.ndarray, float, dict[str, float], bool]:
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        bias = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 0, bias)
        worst = 0.0
        for j, limit in zip(arm.joints, torque_limit, strict=True):
            if limit:
                worst = max(worst, abs(float(bias[model.jnt_dofadr[j]])) / limit)
        mujoco.mj_collision(model, data)
        hits: dict[str, float] = {}
        grounded = False  # in the floor, or in something standing round the robot
        for c in data.contact[: data.ncon]:
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if -c.dist < CONTACT_DEPTH:
                continue
            if mine[g1] != mine[g2] and (arm_geoms[g1] or arm_geoms[g2]):
                grounded = True
            if not (mine[g1] and mine[g2]) or arm_geoms[g1] == arm_geoms[g2]:
                continue
            other = g2 if arm_geoms[g1] else g1
            if int(model.body_weldid[model.geom_bodyid[other]]) == int(model.body_weldid[base_body]) and _base_side(
                model, g1, g2, first
            ):
                continue  # the arm's base where it sits on its plate
            what = model.body(int(model.geom_bodyid[other])).name.removeprefix(f"{robot}/")
            hits[what] = max(hits.get(what, 0.0), -float(c.dist))
        return data.site_xpos[arm.tcp].copy(), worst, hits, grounded

    qpos0 = data.qpos.copy()
    _, hold_home, _, _ = measure()
    base = data.xpos[first].copy()
    reach, lowest, highest, outward = 0.0, math.inf, -math.inf, 0.0
    hitting = valid = 0
    hit_bodies: dict[str, int] = {}
    over = 0
    hold_worst = hold_home
    tips: dict[str, int] = {}
    worst_margin = math.inf
    centre = settled.xpos[root][:2]
    for _ in range(POSES):
        data.qpos[:] = qpos0
        for j, (low, high) in zip(arm.joints, limits, strict=True):
            data.qpos[model.jnt_qposadr[j]] = rng.uniform(low, high)
        tcp, hold, hits, grounded = measure()
        if grounded or tcp[2] < ground - CONTACT_DEPTH:
            continue  # a pose through the floor: no arm gets into it
        valid += 1
        reach = max(reach, float(np.linalg.norm(tcp - base)))
        lowest, highest = min(lowest, float(tcp[2]) - ground), max(highest, float(tcp[2]) - ground)
        outward = max(outward, float(np.linalg.norm(tcp[:2] - centre)))
        if hits:
            hitting += 1
            for what in hits:
                hit_bodies[what] = hit_bodies.get(what, 0) + 1
            continue  # a pose the arm cannot get into says nothing about holding or tipping
        hold_worst = max(hold_worst, hold)
        over += hold > 1.0
        if polygon is not None:
            com = data.subtree_com[root][:2]
            margin = _margin(polygon, com)
            worst_margin = min(worst_margin, margin)
            if margin < 0:
                way = _direction(model, data, root, polygon, com)
                tips[way] = tips.get(way, 0) + 1
    report: dict[str, Any] = {
        "arm": arm.name,
        "reach_m": round(reach, 2),
        "tool_height_m": [round(lowest, 2), round(highest, 2)],
        "reaches_floor": bool(lowest <= FLOOR_REACH),
        "reaches_out_m": round(outward, 2),
    }
    free = valid - hitting
    if hitting:
        top = sorted(hit_bodies, key=lambda b: -hit_bodies[b])[:3]
        report["self_collision"] = (
            f"it hits {', '.join(top)} in {round(100 * hitting / max(valid, 1))}% of random poses; move_arm and "
            "pick_and_place plan round them"
        )
    else:
        report["self_collision"] = "none in any of its poses"
    report["holds"] = {
        "at_home": f"{hold_home * 100:.0f}% of its weakest joint's torque",
        "stretched_out": f"up to {hold_worst * 100:.0f}%",
    }
    out = _Swept(report)
    if over and free:
        report["holds"]["too_heavy_in"] = f"{round(100 * over / free)}% of its free poses"
    if polygon is not None and worst_margin < math.inf:
        if tips:
            way = max(tips, key=lambda w: tips[w])
            report["tips_robot"] = (
                f"reaching out toward its {way} it would tip {robot} over in {round(100 * sum(tips.values()) / free)}% "
                "of its free poses; keep loads close"
            )
        else:
            report["tips_robot"] = f"never: the centre of mass stays at least {worst_margin * 100:.0f} cm inside"
    if hold_home > 1.0:
        out.refusal = (
            f"{arm.name} cannot hold itself up with this at home: its joints need {hold_home * 100:.0f}% of the "
            "torque their motors give; mount a lighter end effector or use a stronger arm"
        )
    return out


def _base_side(model: mujoco.MjModel, g1: int, g2: int, first: int) -> bool:
    """Whether one of the two geoms is on the arm's first moving body: that touches the base it turns on."""
    return first in (int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2]))


def _limits(model: mujoco.MjModel, joints: tuple[int, ...]) -> list[tuple[float, float]]:
    out = []
    for j in joints:
        if model.jnt_limited[j]:
            low, high = (float(v) for v in model.jnt_range[j])
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            low, high = -math.pi, math.pi
        else:
            low, high = 0.0, 0.0
        out.append((low, high))
    return out


def _torque_limits(model: mujoco.MjModel, joints: tuple[int, ...]) -> list[float]:
    """The most torque (N*m, or N on a slide) each joint's motor gives, 0.0 when nothing limits it."""
    out = []
    for j in joints:
        limit = 0.0
        for a in range(model.nu):
            drives = model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and int(model.actuator_trnid[a, 0]) == j
            if drives and model.actuator_forcelimited[a]:
                limit = float(np.max(np.abs(model.actuator_forcerange[a]))) * abs(float(model.actuator_gear[a, 0]))
        if model.jnt_actfrclimited[j]:
            cap = float(np.max(np.abs(model.jnt_actfrcrange[j])))
            limit = min(limit, cap) if limit else cap
        out.append(limit)
    return out


# ---- the robot's parts -----------------------------------------------------------------------------------------------


def _bodies(model: mujoco.MjModel, robot: str) -> list[int]:
    return [b for b in range(1, model.nbody) if model.body(b).name.startswith(f"{robot}/")]


def _body_name(model: mujoco.MjModel, geom: int) -> str:
    return model.body(int(model.geom_bodyid[geom])).name


def _geom_owner(model: mujoco.MjModel, robot: str) -> np.ndarray:
    """For each geom, whether it is one of the robot's."""
    prefix = f"{robot}/"
    return np.array([model.body(int(model.geom_bodyid[g])).name.startswith(prefix) for g in range(model.ngeom)])


def _floating(model: mujoco.MjModel, root: int) -> bool:
    j = int(model.body_jntadr[root])
    return model.body_jntnum[root] > 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE


def _com_words(model: mujoco.MjModel, data: mujoco.MjData, root: int) -> dict[str, Any]:
    """The robot's centre of mass in its own frame (x forward, y left, z up from its base's origin), m."""
    rot = data.xmat[root].reshape(3, 3)
    local = rot.T @ (data.subtree_com[root] - data.xpos[root])
    return {"from_base_m": [round(float(v), 3) for v in local]}
