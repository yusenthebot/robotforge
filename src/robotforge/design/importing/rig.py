"""Plan how to drive an imported robot: base type, motors and gains, self-collision exclusions, a standing pose.

Everything that depends on the physics is decided once, written into a `RigPlan`, and cached
(see `plan.py`); the plan is verified in simulation (`verify.py`) and explained in the doctor report.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from robotforge.design.importing.geometry import GRAVITY, Geometry, clusters, compile_spec, pose_model
from robotforge.design.importing.inertia import shapes_radius
from robotforge.design.importing.plan import MOVABLE, ActuatorPlan, RigPlan, build_spec
from robotforge.design.importing.report import DoctorReport
from robotforge.design.importing.stance import find_stances
from robotforge.design.importing.urdf import JointInfo, StagedUrdf
from robotforge.design.importing.verify import HOLD_SECONDS, HoldResult, score, tipped, tune_gains

GEOM = mujoco.mjtGeom
PLAN_TIME_BUDGET = 40.0  # s: tuning and stance search stop refining after this long (huge robots)
BASE_OMEGA = 30.0  # rad/s: natural frequency of the joint position loops before tuning
DAMPING_RATIO = 1.0
WEAK_MOTOR_RATIO = 4.0  # effort below this multiple of the worst-case gravity torque counts as weak
SAG_TARGET = 0.02  # rad of sag allowed under worst-case gravity torque
OMEGA_LIMIT = 250.0  # rad/s: fastest position loop the 2 ms time step handles comfortably
MIN_JOINT_INERTIA = 1e-4
MAX_TIME_CONSTANT = 0.12  # s: slowest acceptable motor response, including the joint's own damping
WHEEL_TIME_CONSTANT = 0.15  # s
WHEEL_DEFAULT_SPEED = 10.0  # rad/s
WHEEL_GRIP_ACCELERATION = 9.8  # m/s^2: with no effort limit, wheels may push up to the grip of the floor
SKID_STEER_FRICTION = 0.5  # four or more wheels turn by skidding sideways, which needs slippery tires
WHEEL_FULL_TORQUE_ERROR = 0.5  # rad/s of speed error at which a wheel motor reaches its torque limit
DEFAULT_PRISMATIC_SPAN = 0.5  # m, slider range when the URDF gives no limits
FIXED_CLEARANCE = 0.01  # m between a fixed robot's lowest point and the floor when it has to be lifted
HULL_CAP_THRESHOLD = 200
SELF_COLLISION_SAMPLES = 150
SELF_COLLISION_RATE = 0.4  # pairs overlapping in this share of random poses never collide
CASTER_FRICTION = 0.05  # a caster or skid that dragged on the floor would stall a wheeled robot
_WHEEL = re.compile(r"wheel|tire|tyre", re.IGNORECASE)
_PASSIVE = re.compile(r"caster|castor", re.IGNORECASE)
_FINGER = re.compile(r"finger|gripper|claw|thumb|jaw", re.IGNORECASE)


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
    ignore_effort_limits: bool = False,
) -> RigPlan:
    """Inspect the repaired robot, decide how to drive it, and test that it holds still."""
    if ignore_effort_limits:
        for joint in staged.joints:
            joint.effort = None
        report.info(
            "actuators", "ignoring the file's effort limits, as asked; motors are sized from the robot's weight"
        )
    deadline = time.monotonic() + PLAN_TIME_BUDGET
    home = home_pose(staged)
    draft = _draft(staged, urdf, base, home)
    draft.result = tune_gains(urdf, draft.plan, deadline)
    if draft.result is not None and draft.result.blockers:
        # A joint could not make a small move because two links touch: their collision shapes overlap.
        extra = [pair for pair in draft.result.blockers if pair not in draft.plan.excludes]
        draft.plan.excludes = sorted([*draft.plan.excludes, *extra])
        draft.result = tune_gains(urdf, draft.plan, deadline)

    result = draft.result
    if draft.plan.base == "floating" and result is not None and result.finite and tipped(result):
        better = _search_stance(staged, urdf, draft, deadline)
        if better is not None:
            _describe_stance(draft.plan.home_qpos, better.plan.home_qpos, report)
            draft = better
    _emit(draft, report)
    report_hold(draft.plan, draft.result, report)
    return draft.plan


def _draft(
    staged: StagedUrdf, urdf: Path, base: Literal["auto", "fixed", "floating"], home: dict[str, float]
) -> _Draft:
    spec = build_spec(urdf)
    model, data = compile_spec(spec)
    plan = RigPlan(root_body=staged.root, home_qpos=dict(home))
    draft = _Draft(plan)
    _cap_hulls(model, plan)
    pose_model(model, data, home)

    geo = Geometry(model, data)
    plan.base, draft.base_reason = _choose_base(staged, geo, base)
    plan.spawn_height, draft.lifted = _spawn_height(plan.base, geo)
    plan.friction = _friction(staged)
    movable = [j for j in staged.joints if j.type in MOVABLE and not j.mimic_of]
    _plan_actuators(staged, geo, draft, movable)
    plan.excludes = _plan_exclusions(urdf, home)
    return draft


def _friction(staged: StagedUrdf) -> dict[str, float]:
    """Sliding friction per link: the author's Gazebo values, and slippery casters and skids."""
    out = {link: CASTER_FRICTION for link in staged.links if _PASSIVE.search(link)}
    wheels = [j for j in staged.joints if j.type == "continuous" and _WHEEL.search(j.name)]
    if len(wheels) >= 3 and not out:  # skid steering
        out.update({j.child: SKID_STEER_FRICTION for j in wheels})
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


# ---- base type and spawn height ------------------------------------------------------------


def _choose_base(
    staged: StagedUrdf, geo: Geometry, requested: Literal["auto", "fixed", "floating"]
) -> tuple[Literal["fixed", "floating"], str]:
    if requested in ("fixed", "floating"):
        return requested, "you asked for it"
    movable = [j for j in staged.joints if j.type in MOVABLE and not j.mimic_of]
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
    limbs = _limb_roots(staged, cluster, touching - {root_cluster}, root_cluster)
    if len(limbs) >= 2 and not _all_fingers(staged, limbs):
        return "floating", "it stands on its limbs (legs or wheels touch the ground, the body does not)"
    return "fixed", "its first link sits on the ground and nothing else does, like a robot arm on a table"


def _limb_roots(staged: StagedUrdf, cluster: dict[str, str], touching: set[str], root_cluster: str) -> set[str]:
    """The distinct limbs (branches off the base) that the ground-touching link groups belong to."""
    joints = {j.child: j for j in staged.joints}
    roots = set()
    for top in touching:
        current = top
        while current in joints and cluster[joints[current].parent] != root_cluster:
            current = cluster[joints[current].parent]
        roots.add(current)
    return roots


def _all_fingers(staged: StagedUrdf, limb_tops: set[str]) -> bool:
    """True when every ground-touching limb hangs off a finger-like joint (a gripper held pointing down)."""
    joints = {j.child: j for j in staged.joints}
    return all(top in joints and _FINGER.search(joints[top].name) for top in limb_tops)


def _spawn_height(base: str, geo: Geometry) -> tuple[float, float]:
    """(height of the base origin above the floor, how far below the origin the robot reaches)."""
    low = geo.lowest
    if base == "floating":
        return -low + 0.003, 0.0
    if low < -0.001:
        return -low + FIXED_CLEARANCE, -low  # clearance: resting on the floor would make joints push against it
    return 0.0, 0.0


# ---- actuators -------------------------------------------------------------------------------


def _plan_actuators(staged: StagedUrdf, geo: Geometry, draft: _Draft, movable: list[JointInfo]) -> None:
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


def _support_share(staged: StagedUrdf, geo: Geometry, plan: RigPlan) -> float:
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
    geo: Geometry,
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
    # Hold the joint within a small sag even when the arm is stretched out (worst-case gravity torque), as far as
    # the time step allows: the loop must stay slower than OMEGA_LIMIT for the integrator to cope.
    kp = max(kp, min(torque / (0.002 if slide else SAG_TARGET), effective * OMEGA_LIMIT**2))
    if j.effort and j.effort < WEAK_MOTOR_RATIO * torque:
        # A motor this close to the weight it carries saturates on any sizeable step, and a saturated PD
        # controller overshoots into the range where gravity wins. Soften it so ordinary steps stay linear.
        kp = min(kp, j.effort / (0.05 if slide else 0.3))
    kv = max(2 * DAMPING_RATIO * math.sqrt(kp * effective) - j.damping, 0.1 * math.sqrt(kp * effective))
    span = 0.1 if slide else 0.5
    force = j.effort if j.effort else max(4.0 * torque, kp * span)
    return ActuatorPlan(j.name, "position", kp, kv, float(lo), float(hi), float(force), 0.0, bool(j.effort))


def _share_body_weight(
    effective: float, torque: float, plan: RigPlan, geo: Geometry, jid: int, support: float, slide: bool
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
        return max(effective, share), torque + share * GRAVITY * abs(float(geo.data.xaxis[jid][2]))
    return max(effective, share * reach * reach), torque + share * GRAVITY * reach


def _wheel_actuator(staged: StagedUrdf, geo: Geometry, j: JointInfo) -> ActuatorPlan:
    """A speed motor: full torque once the wheel is 0.5 rad/s off its command, so rolling resistance barely shows."""
    shapes = staged.link_shapes.get(j.child, [])
    radius = max(shapes_radius(shapes, np.zeros(3)) if shapes else 0.0, 0.02)
    n_wheels = max(1, sum(1 for x in staged.joints if x.type == "continuous" and _WHEEL.search(x.name)))
    load = geo.total_mass / n_wheels
    skid = sum(1 for x in staged.joints if x.type == "continuous" and _WHEEL.search(x.name)) >= 3
    force = j.effort if j.effort else load * radius * WHEEL_GRIP_ACCELERATION * (3.0 if skid else 1.0)
    kv = max(force / WHEEL_FULL_TORQUE_ERROR, load * radius * radius / WHEEL_TIME_CONSTANT, 1e-4)
    speed = max(j.velocity or 0.0, WHEEL_DEFAULT_SPEED)
    return ActuatorPlan(
        j.name, "velocity", 0.0, float(kv), -float(speed), float(speed), float(force), 0.0, bool(j.effort)
    )


# ---- self collisions -------------------------------------------------------------------------


def _plan_exclusions(urdf: Path, home: dict[str, float]) -> list[tuple[str, str]]:
    """Link pairs that overlap in their default shape and would fight each other."""
    spec = build_spec(urdf, RigPlan(base="fixed"))
    model, data = compile_spec(spec)
    pose_model(model, data, home)
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


def _search_stance(staged: StagedUrdf, urdf: Path, current: _Draft, deadline: float) -> _Draft | None:
    """Zero is rarely a pose a legged robot can stand in; look for one by sampling joint angles."""
    best: _Draft | None = None
    for pose in find_stances(staged, urdf, current.plan.home_qpos):
        draft = _draft(staged, urdf, "floating", pose)
        draft.base_reason = current.base_reason
        draft.result = tune_gains(urdf, draft.plan, deadline)
        if draft.result is None or not draft.result.finite:
            continue
        if best is None or best.result is None or score(draft.result) < score(best.result):
            best = draft
        if draft.result.holds or time.monotonic() > deadline:
            break
    if best is None or best.result is None:
        return None
    if current.result is not None and score(best.result) >= score(current.result):
        return None
    return best


# ---- reporting -------------------------------------------------------------------------------


def _emit(draft: _Draft, report: DoctorReport) -> None:
    plan = draft.plan
    shape = "bolted down (fixed base)" if plan.base == "fixed" else "free-standing (floating base)"
    report.info("base", f"the robot is {shape}: {draft.base_reason}")
    if plan.base == "floating":
        if plan.spawn_height > 0.005:
            report.info(
                "placement", f"the robot starts {plan.spawn_height * 100:.1f} cm above the floor so it just touches it"
            )
    elif draft.lifted:
        report.fixed(
            "placement",
            f"part of the robot is {draft.lifted * 100:.1f} cm below its base origin, which would sink into the floor",
            "lifted it so nothing is stuck in the floor (pass pos=[x, y, z] to place it, or put it on a table)",
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
            "set the floor friction of some links (values from the file, low for casters, "
            "medium for skid-steer wheels)",
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
            "pass ignore_effort_limits=true to size the motors from the robot's weight instead",
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
        reasons.append(f"joints do not follow commands ({result.tracking_raw}; a link may be blocking it)")
    if tipped(result):
        reasons.append("the body falls over or collapses")
    wheeled = any(a.kind == "velocity" for a in plan.actuators)
    advice = (
        "a wheeled robot needs a third point of support (a caster or skid) to stay upright"
        if wheeled and tipped(result)
        else "legged robots also need a balance controller to stay up"
    )
    report.unfixed(
        "stability",
        "the robot does not hold its starting pose by itself: " + " and ".join(reasons),
        "kept the strongest motors the file's effort limits allow. "
        f"If the limits are too low, pass ignore_effort_limits=true; {advice}",
    )
