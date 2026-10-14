"""Turn a staged URDF into a robot that sims well: visible, actuated, collision-safe, standing.

Everything that depends on the physics (gains, exclusions, spawn height, floating or fixed
base, a stable starting pose) is decided once, written into a small JSON-friendly `RigPlan`,
and cached. `build_spec` replays a plan on a fresh MjSpec, which is cheap and deterministic.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from robotforge.design.importing.inertia import shapes_radius
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.urdf import JointInfo, StagedUrdf

PLAN_VERSION = 2
GEOM = mujoco.mjtGeom
VISUAL_GROUP = 2
COLLISION_GROUP = 3  # hidden by the viewer, like MuJoCo Menagerie
GRAVITY = 9.81
BASE_OMEGA = 30.0  # rad/s: natural frequency of the joint position loops before tuning
DAMPING_RATIO = 1.0
MIN_JOINT_INERTIA = 1e-5
MAX_TIME_CONSTANT = 0.12  # s: slowest acceptable motor response, including the joint's own damping
WHEEL_TIME_CONSTANT = 0.15  # s
WHEEL_DEFAULT_SPEED = 10.0  # rad/s
WHEEL_ACCELERATION = 4.0  # m/s^2 the wheels can give the robot when no effort limit is given
WHEEL_FULL_TORQUE_ERROR = 0.5  # rad/s of speed error at which a wheel motor reaches its torque limit
DEFAULT_PRISMATIC_SPAN = 0.5  # m, slider range when the URDF gives no limits
TOUCH_TOLERANCE = 0.02  # m: a link this close to the lowest point counts as touching the ground
MAX_HULL_VERTICES = 64
HULL_CAP_THRESHOLD = 200
SELF_COLLISION_SAMPLES = 150
SELF_COLLISION_RATE = 0.4  # pairs overlapping in this share of random poses never collide
HOLD_SECONDS = 3.0
TRACK_SECONDS = 1.0
TRACK_STEP_REV = 0.25  # rad: the small move used to check that sliders follow
TRACK_STEP_SLIDE = 0.03  # m
MAX_HOLD_DRIFT_REV = 0.03  # rad
MAX_HOLD_DRIFT_SLIDE = 0.01  # m
CASTER_FRICTION = 0.05  # a caster or skid that dragged on the floor would stall a wheeled robot
STANCE_SAMPLES = 4000
STANCE_CANDIDATES = 6
STANCE_CLIMB_STEPS = 600
STANCE_REACH = 1.6  # rad: how far a joint may stray from its zero pose while looking for a stance
_WHEEL = re.compile(r"wheel|tire|tyre", re.IGNORECASE)
_PASSIVE = re.compile(r"caster|castor", re.IGNORECASE)
_MOVABLE = ("revolute", "continuous", "prismatic")


class RigError(ValueError):
    """The staged URDF compiles but cannot be turned into a working robot."""


@dataclass
class ActuatorPlan:
    joint: str
    kind: Literal["position", "velocity"]
    kp: float
    kv: float
    ctrl_lo: float
    ctrl_hi: float
    force: float | None  # symmetric limit; None means unlimited
    home: float = 0.0
    force_from_file: bool = False  # True when `force` is the URDF's effort limit (never scaled)


@dataclass
class RigPlan:
    version: int = PLAN_VERSION
    base: Literal["fixed", "floating"] = "fixed"
    root_body: str = ""
    spawn_height: float = 0.0
    actuators: list[ActuatorPlan] = field(default_factory=list)
    excludes: list[tuple[str, str]] = field(default_factory=list)
    armature: dict[str, float] = field(default_factory=dict)
    capped_meshes: list[str] = field(default_factory=list)
    home_qpos: dict[str, float] = field(default_factory=dict)
    friction: dict[str, float] = field(default_factory=dict)  # body -> sliding friction of its collision shapes
    gain_scale: float = 1.0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RigPlan:
        if data.get("version") != PLAN_VERSION:
            raise ValueError("stale plan")
        plan = cls(**{k: v for k, v in data.items() if k not in ("actuators", "excludes")})
        plan.actuators = [ActuatorPlan(**a) for a in data["actuators"]]
        plan.excludes = [(a, b) for a, b in data["excludes"]]
        return plan


# ---- building --------------------------------------------------------------------------------


def build_spec(urdf: Path, plan: RigPlan | None = None, *, name: str | None = None) -> mujoco.MjSpec:
    """Load the staged URDF and apply `plan` (or just the visual/collision split when no plan yet)."""
    try:
        spec = mujoco.MjSpec.from_file(str(urdf))
    except ValueError as exc:
        raise RigError(f"MuJoCo could not read the repaired URDF: {exc}") from exc
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if name:
        spec.modelname = name
    _split_geoms(spec)
    if plan is not None:
        _apply_plan(spec, plan)
    return spec


def _split_geoms(spec: mujoco.MjSpec) -> None:
    """MuJoCo's URDF reader marks visual geoms (group 1, no contacts). Put visuals in a visible
    group and collision shapes in a hidden one, except on links that have no visuals."""
    has_visual: set[str] = set()
    for body in spec.bodies:
        if any(_is_visual(g) for g in body.geoms):
            has_visual.add(body.name)
    for body in spec.bodies:
        for g in body.geoms:
            if _is_visual(g):
                g.group = VISUAL_GROUP
            else:
                g.group = COLLISION_GROUP if body.name in has_visual else VISUAL_GROUP


def _is_visual(geom: mujoco.MjsGeom) -> bool:
    return geom.contype == 0 and geom.conaffinity == 0


def _apply_plan(spec: mujoco.MjSpec, plan: RigPlan) -> None:
    if plan.base == "floating" and plan.root_body:
        spec.body(plan.root_body).add_freejoint(name=_free_name(spec))
    for joint, value in plan.armature.items():
        spec.joint(joint).armature = value
    for a in plan.actuators:
        act = spec.add_actuator(name=a.joint, target=a.joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if a.kind == "position":
            act.set_to_position(kp=a.kp, kv=a.kv)
        else:
            act.set_to_velocity(kv=a.kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [a.ctrl_lo, a.ctrl_hi]
        if a.force is not None:
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-a.force, a.force]
    for first, second in plan.excludes:
        spec.add_exclude(bodyname1=first, bodyname2=second)
    for mesh in plan.capped_meshes:
        spec.mesh(mesh).maxhullvert = MAX_HULL_VERTICES
    if plan.friction:
        for body in spec.bodies:
            mu = plan.friction.get(body.name)
            for g in body.geoms if mu is not None else []:
                if not _is_visual(g):
                    g.friction = [mu, g.friction[1], g.friction[2]]


def _free_name(spec: mujoco.MjSpec) -> str:
    taken = {j.name for j in spec.joints}
    name = "root"
    while name in taken:
        name += "_"
    return name


# ---- planning --------------------------------------------------------------------------------


@dataclass
class HoldResult:
    finite: bool
    drift: float  # worst joint error relative to what counts as "holding" (1.0 = at the limit)
    drift_raw: str  # human readable worst error
    drop: float  # fraction of its starting height a floating base lost
    upright: bool
    tracking: float = 0.0  # worst share of a small commanded move that was not achieved (fixed bases only)
    tracking_raw: str = ""

    @property
    def holds(self) -> bool:
        return self.finite and self.drift <= 1.0 and self.drop < 0.25 and self.upright and self.tracking <= 0.2


@dataclass
class _Draft:
    """A plan plus the observations needed to explain it to the user."""

    plan: RigPlan
    base_reason: str = ""
    passive: list[str] = field(default_factory=list)
    weak: list[str] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)
    lifted: float = 0.0
    result: HoldResult | None = None


def make_plan(
    staged: StagedUrdf,
    urdf: Path,
    report: DoctorReport,
    base: Literal["auto", "fixed", "floating"] = "auto",
) -> RigPlan:
    """Inspect the repaired robot, decide how to drive it, and test that it holds still."""
    home = home_pose(staged)
    draft = _draft(staged, urdf, base, home)
    draft.result = tune_gains(urdf, draft.plan)

    result = draft.result
    if draft.plan.base == "floating" and result is not None and result.finite and _tipped(result):
        better = _search_stance(staged, urdf, draft)
        if better is not None:
            _describe_stance(draft.plan.home_qpos, better.plan.home_qpos, report)
            draft = better
    _emit(draft, report)
    report_hold(draft.plan, draft.result, report)
    return draft.plan


def _tipped(result: HoldResult) -> bool:
    return result.drop >= 0.25 or not result.upright


def _draft(
    staged: StagedUrdf, urdf: Path, base: Literal["auto", "fixed", "floating"], home: dict[str, float]
) -> _Draft:
    spec = build_spec(urdf)
    model, data = _compile(spec)
    plan = RigPlan(root_body=staged.root, home_qpos=dict(home))
    draft = _Draft(plan)
    _cap_hulls(model, plan)
    _pose(model, data, home)

    geo = _Geometry(model, data)
    plan.base, draft.base_reason = _choose_base(staged, geo, base)
    plan.spawn_height, draft.lifted = _spawn_height(plan.base, geo)
    plan.friction = _friction(staged)
    movable = [j for j in staged.joints if j.type in _MOVABLE and not j.mimic_of]
    _plan_actuators(staged, geo, draft, movable)
    plan.excludes = _plan_exclusions(urdf, home)
    return draft


def _friction(staged: StagedUrdf) -> dict[str, float]:
    """Sliding friction per link: the author's Gazebo values, and slippery casters and skids."""
    out = {link: CASTER_FRICTION for link in staged.links if _PASSIVE.search(link)}
    out.update(staged.link_friction)
    return out


def home_pose(staged: StagedUrdf) -> dict[str, float]:
    """Joint values the robot starts at: zero, moved inside the joint limits where zero is not allowed."""
    pose: dict[str, float] = {}
    by_name = {j.name: j for j in staged.joints}
    for j in staged.joints:
        if j.type in ("revolute", "prismatic") and j.lower is not None and j.upper is not None and not j.mimic_of:
            value = min(max(0.0, j.lower), j.upper)
            if value != 0.0:
                pose[j.name] = value
    for j in staged.joints:  # followers sit where their leader's pose puts them
        leader = by_name.get(j.mimic_of or "")
        if leader is not None:
            value = j.mimic_multiplier * pose.get(leader.name, 0.0) + j.mimic_offset
            if value != 0.0:
                pose[j.name] = value
    return pose


def _pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def _compile(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    try:
        model = spec.compile()
    except ValueError as exc:
        raise RigError(f"MuJoCo could not compile the repaired robot: {exc}") from exc
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _cap_hulls(model: mujoco.MjModel, plan: RigPlan) -> None:
    """Collision meshes with many vertices slow every contact check; limit their hull size."""
    heavy = set()
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != GEOM.mjGEOM_MESH or model.geom_contype[geom_id] == 0:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if model.mesh_vertnum[mesh_id] > HULL_CAP_THRESHOLD:
            heavy.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id))
    plan.capped_meshes = sorted(n for n in heavy if n)


class _Geometry:
    """Pose-dependent geometry of a compiled robot: ground clearance, loads, subtree masses."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.children: dict[int, list[int]] = {}
        for b in range(1, model.nbody):
            self.children.setdefault(int(model.body_parentid[b]), []).append(b)
        self.total_mass = float(model.body_mass.sum())
        self.collision_geoms = [
            g for g in range(model.ngeom) if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
        ]
        self.low = self.lowest_by_body()

    def body_name(self, b: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    def joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def subtree(self, body: int) -> list[int]:
        out, stack = [], [body]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.children.get(b, []))
        return out

    def lowest_by_body(self) -> dict[int, float]:
        """Lowest world z of each body's colliding geoms (bodies without any are absent)."""
        low: dict[int, float] = {}
        for g in self.collision_geoms:
            b = int(self.model.geom_bodyid[g])
            low[b] = min(self.geom_low(g)[2], low.get(b, math.inf))
        return low

    def geom_low(self, g: int) -> tuple[float, float, float]:
        """Lowest world point of one geom (primitives analytically, meshes by their vertices)."""
        m, d = self.model, self.data
        pos, rot = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        size, kind = m.geom_size[g], int(m.geom_type[g])
        axis_z = float(rot[2, 2])  # z component of the geom's own z axis
        if kind == GEOM.mjGEOM_SPHERE:
            return float(pos[0]), float(pos[1]), float(pos[2] - size[0])
        if kind == GEOM.mjGEOM_CAPSULE:
            return float(pos[0]), float(pos[1]), float(pos[2] - abs(axis_z) * size[1] - size[0])
        if kind == GEOM.mjGEOM_CYLINDER:
            drop = abs(axis_z) * size[1] + size[0] * math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
            return float(pos[0]), float(pos[1]), float(pos[2] - drop)
        if kind == GEOM.mjGEOM_BOX:
            return float(pos[0]), float(pos[1]), float(pos[2] - float(np.abs(rot[2]) @ size[:3]))
        if kind == GEOM.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            first, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            world = pos + m.mesh_vert[first : first + count] @ rot.T
            lowest = world[int(np.argmin(world[:, 2]))]
            return float(lowest[0]), float(lowest[1]), float(lowest[2])
        center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        return float(pos[0]), float(pos[1]), float(pos[2] + rot[2] @ center - np.abs(rot[2]) @ half)

    @property
    def lowest(self) -> float:
        return min(self.low.values()) if self.low else 0.0

    def touching(self) -> set[int]:
        floor = self.lowest
        return {b for b, z in self.low.items() if z <= floor + TOUCH_TOLERANCE}

    def joint_load(self, jid: int) -> tuple[float, float, float]:
        """(inertia about the joint, mass below it, worst-case gravity torque) for the subtree it moves."""
        m, d = self.model, self.data
        body = int(m.jnt_bodyid[jid])
        anchor, axis = d.xanchor[jid], d.xaxis[jid]
        slide = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        inertia = mass = torque = 0.0
        for b in self.subtree(body):
            mb = float(m.body_mass[b])
            if mb <= 0:
                continue
            r = d.xipos[b] - anchor
            perp = r - axis * float(r @ axis)
            mass += mb
            torque += mb * GRAVITY * (1.0 if slide else float(np.linalg.norm(perp)))
            if not slide:
                rot = d.ximat[b].reshape(3, 3)
                tensor = rot @ np.diag(m.body_inertia[b]) @ rot.T
                inertia += float(axis @ tensor @ axis) + mb * float(perp @ perp)
        return inertia, mass, torque


# ---- base type and spawn height ------------------------------------------------------------


def _choose_base(
    staged: StagedUrdf, geo: _Geometry, requested: Literal["auto", "fixed", "floating"]
) -> tuple[Literal["fixed", "floating"], str]:
    if requested in ("fixed", "floating"):
        return requested, "you asked for it"
    movable = [j for j in staged.joints if j.type in _MOVABLE and not j.mimic_of]
    if not movable:
        return "floating", "it has no moving joints, so it is treated as an object that rests on the floor"
    root = staged.root
    if root.lower() in ("world", "map", "odom"):
        return "fixed", f"the root link is called '{root}', the usual sign of a robot bolted to the world"
    wheels = [
        j.name for j in movable if j.type == "continuous" and _WHEEL.search(j.name) and not _PASSIVE.search(j.name)
    ]
    if wheels:
        return "floating", f"it has wheels ({', '.join(wheels[:3])}), so it can drive around"

    cluster = clusters(staged)
    touching = {cluster[geo.body_name(b)] for b in geo.touching() if geo.body_name(b) in cluster}
    root_cluster = cluster[root]
    others = touching - {root_cluster}
    if others and (root_cluster not in touching or len(others) >= 2):
        return "floating", "it stands on its limbs (legs or wheels touch the ground, the body does not)"
    return "fixed", "its first link sits on the ground and nothing else does, like a robot arm on a table"


def clusters(staged: StagedUrdf) -> dict[str, str]:
    """Group links that are welded together by fixed joints; the label is the group's top link."""
    label = {name: name for name in staged.links}

    def top(name: str) -> str:
        while label[name] != name:
            name = label[name]
        return name

    for j in staged.joints:
        if j.type == "fixed":
            label[j.child] = top(j.parent)
    return {name: top(name) for name in staged.links}


def _spawn_height(base: str, geo: _Geometry) -> tuple[float, float]:
    """(height of the base origin above the floor, how far below the origin the robot reaches)."""
    low = geo.lowest
    if base == "floating":
        return -low + 0.003, 0.0
    if low < -0.001:
        return -low, -low
    return 0.0, 0.0


# ---- actuators -------------------------------------------------------------------------------


def _plan_actuators(staged: StagedUrdf, geo: _Geometry, draft: _Draft, movable: list[JointInfo]) -> None:
    plan = draft.plan
    support = _support_share(staged, geo, plan)
    for j in movable:
        jid = geo.joint_id(j.name)
        if jid < 0:
            continue
        if _PASSIVE.search(j.name):
            draft.passive.append(j.name)
            continue
        if j.type == "continuous" and _WHEEL.search(j.name):
            plan.actuators.append(_wheel_actuator(staged, geo, j))
            continue
        inertia, mass, torque = geo.joint_load(jid)
        actuator = _position_actuator(j, inertia, mass, torque, plan, geo, jid, support)
        actuator.home = plan.home_qpos.get(j.name, 0.0)
        if j.name in plan.home_qpos and not _inside(j, 0.0):
            draft.clamped.append(f"{j.name} (starts at {actuator.home:.3g})")
        if j.effort and plan.base == "fixed" and j.effort < 0.9 * torque:
            draft.weak.append(f"{j.name}: limit {j.effort:.3g}, needs about {torque:.3g} when stretched out")
        plan.actuators.append(actuator)


def _inside(j: JointInfo, value: float) -> bool:
    return j.lower is None or j.upper is None or j.lower <= value <= j.upper


def _support_share(staged: StagedUrdf, geo: _Geometry, plan: RigPlan) -> float:
    """For floating robots, how many ground-touching limbs share the body weight."""
    if plan.base != "floating":
        return 1.0
    cluster = clusters(staged)
    limbs = {cluster[geo.body_name(b)] for b in geo.touching() if geo.body_name(b) in cluster}
    return float(max(1, len(limbs - {cluster[staged.root]}) or len(limbs)))


def _position_actuator(
    j: JointInfo,
    inertia: float,
    mass: float,
    torque: float,
    plan: RigPlan,
    geo: _Geometry,
    jid: int,
    support: float,
) -> ActuatorPlan:
    slide = j.type == "prismatic"
    if slide:
        lo, hi = (
            (j.lower, j.upper) if j.lower is not None else (-DEFAULT_PRISMATIC_SPAN / 2, DEFAULT_PRISMATIC_SPAN / 2)
        )
        effective = max(mass, 1e-3)
    else:
        lo, hi = (j.lower, j.upper) if j.lower is not None else (-math.pi, math.pi)
        effective = max(inertia, MIN_JOINT_INERTIA)
    effective, torque = _share_body_weight(effective, torque, plan, geo, jid, support, slide)
    kp = effective * BASE_OMEGA**2
    kv = 2 * DAMPING_RATIO * effective * BASE_OMEGA
    # The file's own joint damping and friction fight the motor: keep it from turning the joint into molasses.
    kp = max(kp, (j.damping + kv) / MAX_TIME_CONSTANT, j.friction / (0.002 if slide else 0.01))
    kv = max(2 * DAMPING_RATIO * math.sqrt(kp * effective) - j.damping, 0.1 * math.sqrt(kp * effective))
    span = 0.1 if slide else 0.5
    force = j.effort if j.effort else max(4.0 * torque, kp * span)
    return ActuatorPlan(j.name, "position", kp, kv, float(lo), float(hi), float(force), 0.0, bool(j.effort))


def _share_body_weight(
    effective: float, torque: float, plan: RigPlan, geo: _Geometry, jid: int, support: float, slide: bool
) -> tuple[float, float]:
    """A leg joint carries its share of the body above it, not just the limb below it."""
    if plan.base != "floating":
        return effective, torque
    m = geo.model
    below = set(geo.subtree(int(m.jnt_bodyid[jid])))
    share = max(geo.total_mass - sum(float(m.body_mass[b]) for b in below), 0.0) / support
    anchor = geo.data.xanchor[jid]
    reach = 0.0  # lever arm: how far the foot-most part of this limb is from the joint
    for b in below:
        if b in geo.low:
            reach = max(reach, float(np.linalg.norm(geo.data.xipos[b] - anchor)))
    reach = max(reach, 0.05)
    if slide:
        return max(effective, share), torque + share * GRAVITY
    return max(effective, share * reach * reach), torque + share * GRAVITY * reach


def _wheel_actuator(staged: StagedUrdf, geo: _Geometry, j: JointInfo) -> ActuatorPlan:
    """A speed motor: full torque once the wheel is 0.5 rad/s off its command, so rolling resistance barely shows."""
    shapes = staged.link_shapes.get(j.child, [])
    radius = max(shapes_radius(shapes, np.zeros(3)) if shapes else 0.0, 0.02)
    n_wheels = max(1, sum(1 for x in staged.joints if x.type == "continuous" and _WHEEL.search(x.name)))
    load = geo.total_mass / n_wheels
    force = j.effort if j.effort else load * radius * WHEEL_ACCELERATION
    kv = max(force / WHEEL_FULL_TORQUE_ERROR, load * radius * radius / WHEEL_TIME_CONSTANT, 1e-4)
    speed = max(j.velocity or 0.0, WHEEL_DEFAULT_SPEED)
    return ActuatorPlan(
        j.name, "velocity", 0.0, float(kv), -float(speed), float(speed), float(force), 0.0, bool(j.effort)
    )


# ---- self collisions -------------------------------------------------------------------------


def _plan_exclusions(urdf: Path, home: dict[str, float]) -> list[tuple[str, str]]:
    """Link pairs that overlap in their default shape and would fight each other."""
    spec = build_spec(urdf, RigPlan(base="fixed"))
    model, data = _compile(spec)
    _pose(model, data, home)
    rng = np.random.default_rng(0)
    movable = [
        (int(model.jnt_qposadr[j]), *_joint_span(model, j))
        for j in range(model.njnt)
        if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    ]
    chosen = _contact_pairs(model, data)
    counts: dict[tuple[int, int], int] = {}
    for _ in range(SELF_COLLISION_SAMPLES):
        for adr, lo, hi in movable:
            data.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(model, data)
        for pair in _contact_pairs(model, data):
            counts[pair] = counts.get(pair, 0) + 1
    chosen |= {pair for pair, n in counts.items() if n >= SELF_COLLISION_RATE * SELF_COLLISION_SAMPLES}
    names = sorted(tuple(sorted((_body(model, a), _body(model, b)))) for a, b in chosen)
    return [(a, b) for a, b in names]


def _joint_span(model: mujoco.MjModel, j: int) -> tuple[float, float]:
    if model.jnt_limited[j]:
        lo, hi = model.jnt_range[j]
        return float(lo), float(hi)
    return (-math.pi, math.pi) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else (-0.1, 0.1)


def _contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    return pairs


def _body(model: mujoco.MjModel, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


# ---- standing pose search --------------------------------------------------------------------


def _search_stance(staged: StagedUrdf, urdf: Path, current: _Draft) -> _Draft | None:
    """Zero is rarely a pose a legged robot can stand in; look for one by sampling joint angles."""
    best: _Draft | None = None
    for pose in find_stances(staged, urdf, current.plan.home_qpos):
        draft = _draft(staged, urdf, "floating", pose)
        draft.base_reason = current.base_reason
        draft.result = tune_gains(urdf, draft.plan)
        if draft.result is None or not draft.result.finite:
            continue
        if best is None or best.result is None or _score(draft.result) < _score(best.result):
            best = draft
        if draft.result.holds:
            break
    if best is None or best.result is None:
        return None
    if current.result is not None and _score(best.result) >= _score(current.result):
        return None
    return best


def _score(result: HoldResult) -> float:
    return result.drift + 10 * result.drop + (0 if result.upright else 100) + 5 * result.tracking


@dataclass
class _Limb:
    joints: list[JointInfo]
    bodies: set[int]
    hip_joint: str


def find_stances(staged: StagedUrdf, urdf: Path, home: dict[str, float]) -> list[dict[str, float]]:
    """Joint poses, best first, where the feet sit under the hips and the body is over its feet."""
    spec = build_spec(urdf)
    model, data = _compile(spec)
    _pose(model, data, home)
    geo = _Geometry(model, data)
    limbs = _limbs(staged, geo)
    if len(limbs) < 2:
        return []
    search = _StanceSearch(model, data, geo, limbs, geo.body_id(staged.root), home)
    return search.run()


class _StanceSearch:
    """Random sampling followed by hill climbing on a stance score (kinematics only, no simulation)."""

    def __init__(self, model, data, geo: _Geometry, limbs: list[_Limb], root: int, home: dict[str, float]):
        self.model, self.data, self.geo, self.limbs, self.root, self.home = model, data, geo, limbs, root, home
        self.rng = np.random.default_rng(1)
        self.names: list[str] = []
        self.adr: list[int] = []
        lows, highs = [], []
        for limb in limbs:
            for j in limb.joints:
                lo, hi = (j.lower, j.upper) if j.lower is not None and j.upper is not None else (-math.pi, math.pi)
                center = home.get(j.name, 0.0)  # stay within a natural reach of the zero pose
                lo, hi = max(lo, center - STANCE_REACH), min(hi, center + STANCE_REACH)
                self.names.append(j.name)
                self.adr.append(int(model.jnt_qposadr[geo.joint_id(j.name)]))
                lows.append(lo)
                highs.append(hi)
        self.lo, self.hi = np.array(lows), np.array(highs)
        self.start = np.array([home.get(n, 0.0) for n in self.names])
        self.base_q = data.qpos.copy()
        self.reference_height = max(float(data.xpos[root][2] - geo.lowest), 0.05)

    def pose(self, values: np.ndarray) -> dict[str, float]:
        out = dict(self.home)
        out.update({n: float(v) for n, v in zip(self.names, values, strict=True)})
        return out

    def score(self, values: np.ndarray) -> float:
        q = self.base_q.copy()
        q[self.adr] = values
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        deviation = float(np.abs(values - self.start).sum())
        return _stance_score(self.model, self.data, self.geo, self.limbs, self.root, deviation, self.reference_height)

    def run(self) -> list[dict[str, float]]:
        n = len(self.names)
        samples = []
        for k in range(STANCE_SAMPLES):
            if k % 2:
                values = self.rng.uniform(self.lo, self.hi)
            else:
                values = np.clip(self.start + self.rng.normal(0, 0.5, n), self.lo, self.hi)
            samples.append((self.score(values), values))
        samples.sort(key=lambda s: s[0])
        seeds: list[np.ndarray] = []
        for score, values in samples:
            if score == math.inf:
                break
            if all(float(np.abs(values - other).max()) > 0.25 for other in seeds):
                seeds.append(values)
            if len(seeds) >= STANCE_CANDIDATES:
                break
        refined = [self._climb(v) for v in seeds]
        refined.sort(key=lambda s: s[0])
        return [self.pose(v) for score, v in refined if score != math.inf]

    def _climb(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        best, best_score = values, self.score(values)
        step = 0.2
        for _ in range(STANCE_CLIMB_STEPS):
            trial = np.clip(
                best + self.rng.normal(0, step, len(best)) * (self.rng.random(len(best)) < 0.4), self.lo, self.hi
            )
            trial_score = self.score(trial)
            if trial_score < best_score:
                best, best_score = trial, trial_score
            step = max(0.01, step * 0.997)
        return best_score, best


def _limbs(staged: StagedUrdf, geo: _Geometry) -> list[_Limb]:
    """Movable subtrees hanging off the base that reach down toward the ground."""
    cluster = clusters(staged)
    root_cluster = cluster[staged.root]
    child_joints: dict[str, list[JointInfo]] = {}
    for j in staged.joints:
        child_joints.setdefault(j.parent, []).append(j)

    def collect(link: str) -> list[str]:
        out = [link]
        for j in child_joints.get(link, []):
            out += collect(j.child)
        return out

    top = max(geo.low.values(), default=0.0)
    reach_limit = geo.lowest + 0.3 * max(top - geo.lowest, 0.2)
    limbs = []
    for link in [n for n, label in cluster.items() if label == root_cluster]:
        for j in child_joints.get(link, []):
            if j.type not in _MOVABLE or j.mimic_of:
                continue
            links = collect(j.child)
            bodies = {geo.body_id(n) for n in links if geo.body_id(n) >= 0}
            lows = [geo.low[b] for b in bodies if b in geo.low]
            if not lows or min(lows) > reach_limit:
                continue
            joints = [x for x in staged.joints if x.child in links and x.type in _MOVABLE and not x.mimic_of]
            limbs.append(_Limb(joints, bodies, j.name))
    return limbs


def _stance_score(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geo: _Geometry,
    limbs: list[_Limb],
    root: int,
    deviation: float,
    reference_height: float,
) -> float:
    feet, hips = [], []
    for limb in limbs:
        lowest = (0.0, 0.0, math.inf)
        for g in geo.collision_geoms:
            if int(model.geom_bodyid[g]) in limb.bodies:
                point = geo.geom_low(g)
                if point[2] < lowest[2]:
                    lowest = point
        if lowest[2] == math.inf:
            return math.inf
        feet.append(lowest)
        hips.append(data.xanchor[geo.joint_id(limb.hip_joint)])
    feet_arr, hips_arr = np.array(feet), np.array(hips)
    com = data.subtree_com[root][:2]
    com_offset = float(np.linalg.norm(com - feet_arr[:, :2].mean(axis=0)))
    spread = float(feet_arr[:, 2].max() - feet_arr[:, 2].min())
    foot_offsets = feet_arr[:, :2] - hips_arr[:, :2]
    hip_offset = float(np.linalg.norm(foot_offsets, axis=1).mean())
    body_height = float(data.xpos[root][2] - feet_arr[:, 2].mean())
    if body_height < 0.02:
        return math.inf
    asymmetry = _asymmetry(foot_offsets, hips_arr[:, :2] - data.xpos[root][:2])
    margin = _support_margin(feet_arr[:, :2], com)
    # Prefer a tall, left-right symmetric stance close to the zero pose over a crouch that merely balances.
    return (
        4.0 * com_offset
        + 4.0 * spread
        + 2.0 * hip_offset
        + 4.0 * asymmetry
        - 3.0 * min(body_height / reference_height, 0.9)
        + 5.0 * max(0.0, 0.5 - body_height / reference_height)
        - 8.0 * min(margin, 0.08)
        + 0.01 * deviation
    )


def _support_margin(feet: np.ndarray, com: np.ndarray) -> float:
    """Distance from the center of mass to the nearest edge of the support polygon (negative = outside)."""
    pts = sorted({(round(float(x), 5), round(float(y), 5)) for x, y in feet})
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    margin = math.inf
    for i, a in enumerate(hull):
        b = hull[(i + 1) % len(hull)]
        edge = np.array([b[0] - a[0], b[1] - a[1]])
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        margin = min(margin, float((edge[0] * (com[1] - a[1]) - edge[1] * (com[0] - a[0])) / length))
    return 0.0 if margin == math.inf else margin


def _asymmetry(foot_offsets: np.ndarray, hips: np.ndarray) -> float:
    """How far mirrored limbs (same x, opposite y hips) are from mirrored foot placements."""
    total, pairs = 0.0, 0
    for a in range(len(hips)):
        for b in range(a + 1, len(hips)):
            if abs(hips[a, 0] - hips[b, 0]) < 0.02 and abs(hips[a, 1] + hips[b, 1]) < 0.02 and abs(hips[a, 1]) > 0.01:
                total += abs(foot_offsets[a, 0] - foot_offsets[b, 0]) + abs(foot_offsets[a, 1] + foot_offsets[b, 1])
                pairs += 1
    return total / pairs if pairs else 0.0


def _pose_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return max((abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)), default=0.0)


# ---- verification and gain tuning ------------------------------------------------------------


def _test_world(urdf: Path, plan: RigPlan) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """The robot on a floor, at its home pose with every motor commanded to hold it."""
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.worldbody.add_geom(type=GEOM.mjGEOM_PLANE, size=[0, 0, 0.05])
    frame = world.worldbody.add_frame(pos=[0, 0, plan.spawn_height])
    world.attach(build_spec(urdf, plan), prefix="t/", frame=frame)
    model = world.compile()
    data = mujoco.MjData(model)
    for joint, value in plan.home_qpos.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{joint}")]] = value
    for a in plan.actuators:
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"t/{a.joint}")] = a.home
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"t/{plan.root_body}")
    return model, data, root


def _run(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> bool:
    """Step the simulation; False when it blows up."""
    for step in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step % 50 == 0 and not (np.isfinite(data.qpos).all() and np.abs(data.qpos).max() < 1e6):
            return False
    return True


def hold_test(urdf: Path, plan: RigPlan, seconds: float = HOLD_SECONDS) -> HoldResult:
    """Put the robot on a floor with every motor at its home pose, then nudge each joint; check it behaves."""
    model, data, root = _test_world(urdf, plan)
    start_height = float(data.subtree_com[root][2])  # center of mass: robust when the base origin sits on the floor
    if not _run(model, data, seconds):
        return HoldResult(False, math.inf, "the simulation blew up", 1.0, False)

    worst, worst_text = 0.0, "none"
    for a in plan.actuators:
        if a.kind != "position":
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{a.joint}")
        err = abs(float(data.qpos[model.jnt_qposadr[jid]]) - a.home)
        slide = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        ratio = err / (MAX_HOLD_DRIFT_SLIDE if slide else MAX_HOLD_DRIFT_REV)
        if ratio >= worst:
            worst, worst_text = ratio, f"{a.joint} off by {err:.3g} {'m' if slide else 'rad'}"
    drop, upright = 0.0, True
    if plan.base == "floating" and start_height > 1e-3:
        drop = max(0.0, (start_height - float(data.subtree_com[root][2])) / start_height)
        upright = float(data.xmat[root].reshape(3, 3)[2, 2]) > 0.7
    result = HoldResult(True, worst, worst_text, drop, upright)
    if plan.base == "fixed" and result.finite and result.drift <= 1.0:
        result.tracking, result.tracking_raw = _tracking(model, data, plan)
    return result


def _tracking(model: mujoco.MjModel, data: mujoco.MjData, plan: RigPlan) -> tuple[float, str]:
    """Command a small move on every joint at once; return the worst share of it that was not achieved."""
    targets: dict[int, tuple[float, str]] = {}
    for a in plan.actuators:
        if a.kind != "position":
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{a.joint}")
        slide = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        up, down = a.ctrl_hi - a.home, a.home - a.ctrl_lo
        room = max(up, down)
        delta = min(TRACK_STEP_SLIDE if slide else TRACK_STEP_REV, 0.8 * room)
        if delta <= 1e-6:
            continue
        delta = delta if up >= down else -delta
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"t/{a.joint}")
        data.ctrl[aid] = a.home + delta
        targets[jid] = (a.home + delta, a.joint)
    if not targets or not _run(model, data, TRACK_SECONDS):
        return (0.0, "") if not targets else (math.inf, "the simulation blew up")
    worst, text = 0.0, ""
    for jid, (target, name) in targets.items():
        start = plan.home_qpos.get(name, 0.0)
        achieved = float(data.qpos[model.jnt_qposadr[jid]])
        missing = abs(target - achieved) / max(abs(target - start), 1e-9)
        if missing >= worst:
            worst, text = missing, f"{name} reached {achieved:.3g} of {target:.3g}"
    return worst, text


def tune_gains(urdf: Path, plan: RigPlan) -> HoldResult | None:
    """Raise motor stiffness until the robot holds its pose and follows commands; keeps the best in `plan`.

    Returns None when the plan has no motors or the simulation blows up at every setting.
    """
    if not plan.actuators:
        return None
    base = [ActuatorPlan(**asdict(a)) for a in plan.actuators]
    best: tuple[float, float, HoldResult] | None = None
    for scale in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        _scale_gains(plan, base, scale)
        result = hold_test(urdf, plan)
        if result.finite:
            score = _score(result)
            stalled = best is not None and score > 0.97 * best[0]  # more stiffness no longer helps
            if best is None or score < best[0]:
                best = (score, scale, result)
            if result.holds or stalled:
                break
    if best is None:
        _scale_gains(plan, base, 1.0)
        return None
    _, scale, result = best
    _scale_gains(plan, base, scale)
    return result


def _scale_gains(plan: RigPlan, base: list[ActuatorPlan], scale: float) -> None:
    plan.gain_scale = scale
    for target, source in zip(plan.actuators, base, strict=True):
        if source.kind != "position":
            continue
        target.kp = source.kp * scale
        target.kv = source.kv * math.sqrt(scale)
        if not source.force_from_file and source.force is not None:
            target.force = source.force * scale


# ---- reporting -------------------------------------------------------------------------------


def _emit(draft: _Draft, report: DoctorReport) -> None:
    plan = draft.plan
    report.info("base", f"the robot's base is {plan.base}: {draft.base_reason}")
    if plan.base == "floating":
        report.info(
            "placement", f"the robot starts {plan.spawn_height * 100:.1f} cm above the floor so it just touches it"
        )
    elif draft.lifted:
        report.fixed(
            "placement",
            f"part of the robot is {draft.lifted * 100:.1f} cm below its base origin, which would sink into the floor",
            "lifted it so the lowest part rests on the floor",
        )
    if plan.actuators:
        kinds = {"position": 0, "velocity": 0}
        for a in plan.actuators:
            kinds[a.kind] += 1
        report.fixed(
            "actuators",
            "the URDF describes joints but no motors, so sliders and controls would do nothing",
            "added "
            + " and ".join(
                f"{n} {label} motor{'s' if n != 1 else ''}"
                for n, label in ((kinds["position"], "position"), (kinds["velocity"], "wheel (speed)"))
                if n
            )
            + ", sized to the weight each joint carries; joint limits and effort limits from the file are respected",
            [a.joint for a in plan.actuators],
        )
    if draft.passive:
        report.info("actuators", "these joints are casters, which spin freely; they have no motor", draft.passive)
    if plan.friction:
        report.info(
            "friction",
            "set the floor friction of some links (Gazebo mu1 values from the file, or low for casters and skids)",
            [f"{k}: {v:g}" for k, v in sorted(plan.friction.items())],
        )
    if draft.clamped:
        report.fixed(
            "joints",
            "the joint limits do not include zero, where a URDF robot normally starts",
            "start inside the allowed range instead",
            draft.clamped,
        )
    if draft.weak:
        report.unfixed(
            "actuators",
            "the file's effort limits are too weak to hold the robot's own weight when it is stretched out",
            "kept the file's limits, so these joints will sag; "
            "re-import a copy of the URDF with larger <limit effort> values if that is not wanted",
            draft.weak,
        )
    if plan.excludes:
        report.fixed(
            "collision",
            f"{len(plan.excludes)} pair(s) of links overlap in their collision shapes (common in URDFs), "
            "which makes a robot jitter",
            "stopped those pairs from colliding with each other (they still collide with the floor and with objects)",
            [f"{a} / {b}" for a, b in plan.excludes],
        )
    if plan.capped_meshes:
        report.info(
            "collision", f"limited the collision outline of {len(plan.capped_meshes)} detailed mesh(es) to 64 points"
        )


def _describe_stance(old: dict[str, float], new: dict[str, float], report: DoctorReport) -> None:
    changed = [f"{k} = {v:.2f}" for k, v in sorted(new.items()) if abs(v - old.get(k, 0.0)) > 1e-6]
    report.fixed(
        "stance",
        "in its zero pose the robot falls over (its feet are not under its body)",
        "found a standing pose by trying many joint angles, and start there",
        changed,
    )


def report_hold(plan: RigPlan, result: HoldResult | None, report: DoctorReport) -> None:
    if result is None:
        if plan.actuators:
            report.unfixed(
                "stability",
                "the simulation blows up even with soft motors",
                "the robot may not behave; check the masses and joint limits in the file",
            )
        return
    if result.holds:
        extra = f"; stiffness raised {plan.gain_scale:g}x to carry its own weight" if plan.gain_scale > 1 else ""
        report.info(
            "stability",
            f"hold test passed: with every motor at its starting pose the robot stays put for {HOLD_SECONDS:g} s"
            f" (worst joint error: {result.drift_raw}{extra})",
        )
        return
    reasons = []
    if result.drift > 1.0:
        reasons.append(f"joints sag ({result.drift_raw})")
    if result.tracking > 0.2:
        reasons.append(f"joints do not follow commands ({result.tracking_raw})")
    if _tipped(result):
        reasons.append("the body falls over or collapses")
    report.unfixed(
        "stability",
        "the robot does not hold its starting pose by itself: " + " and ".join(reasons),
        "kept the strongest motors the file's effort limits allow. "
        "If the limits are too low, edit <limit effort> in a copy of the URDF; "
        "legged robots also need a balance controller to stay up",
    )
