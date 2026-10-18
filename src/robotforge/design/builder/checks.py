"""Checks a beginner (or an agent) needs before driving a freshly built robot.

The robot is compiled on its own, posed at home, dropped onto a floor and left
alone for a second. That finds what would otherwise show up as a robot that
explodes, jams, sinks into the floor or sags on servos that are too weak, and
says it in words with a fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.behaviors.balance import find_balancer
from robotforge.design import components
from robotforge.design.builder.assemble import Assembly
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.geometry import collides, lowest_point

SPAWN_CLEARANCE = 0.002  # m between the lowest point and the floor
OVERLAP_ERROR = 0.004  # m: deeper than this, two parts are inside each other
SETTLE_SECONDS = 1.0
WEAK = 0.95  # share of an actuator's force limit used just to hold still: it cannot hold its pose
STRAINED = 0.6  # little headroom left for walking or lifting (the humanoid tests fall from about 67 %)
# The same for a robot on four or more feet: a walk holds it on half of them, about twice the load. A 3.2 kg hexapod
# with roll-pitch-pitch legs built in the agent eval, sent forwards for 5 s, made 82 % of the way at 36 % standing,
# 37 % at 46, 6 % at 48 and went backwards from 54 % (the agent's xl430 build); the test dog still made 87 % at 50 %.
WALK_STRAINED = 0.45
_DIVERGENCE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)


@dataclass
class Report:
    spawn_height: float
    mass: float
    notes: list[str] = field(default_factory=list)
    # Each force-limited position servo's share of its force limit used to hold the robot still at home (the
    # numbers behind the "too weak" and "little headroom" notes), by actuator name.
    strain: dict[str, float] = field(default_factory=dict)


def check(assembly: Assembly) -> Report:
    """Measure the robot and find problems. Raises ValueError for robots that cannot work at all."""
    spec = assembly.spec
    model = _compile(spec)
    data = mujoco.MjData(model)
    _home(model, data, assembly)
    mujoco.mj_kinematics(model, data)
    base = model.body("base").id
    geoms = [g for g in range(model.ngeom) if collides(model, g)]
    if assembly.floating:
        considered = geoms
    else:  # a bolted-down robot stands on its base
        considered = [g for g in geoms if model.body_weldid[model.geom_bodyid[g]] == model.body_weldid[base]]
    lowest = min((lowest_point(model, data, g)[2] for g in considered), default=0.0)
    spawn = -float(lowest) + (SPAWN_CLEARANCE if assembly.floating else 0.0)
    mass = float(model.body_subtreemass[base])

    # Drop it onto a floor and watch.
    spec.worldbody.add_geom(name="check_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    spec.worldbody.first_body().pos = [0.0, 0.0, spawn]
    model = _compile(spec)
    data = mujoco.MjData(model)
    _home(model, data, assembly)
    mujoco.mj_forward(model, data)
    _overlaps(model, data)
    strain: dict[str, float] = {}
    notes = _settle(model, data, assembly, strain)
    return Report(spawn_height=spawn, mass=mass, notes=notes, strain=strain)


def _compile(spec: mujoco.MjSpec) -> mujoco.MjModel:
    try:
        return spec.compile()
    except ValueError as exc:
        raise ValueError(f"MuJoCo could not compile the built robot: {exc}") from exc


def _home(model: mujoco.MjModel, data: mujoco.MjData, assembly: Assembly) -> None:
    for joint, value in assembly.home_qpos.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    for act, value in assembly.home_ctrl.items():
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
        if a >= 0:
            data.ctrl[a] = value


def _balances(assembly: Assembly) -> bool:
    """Whether the robot is one the built-in balancer drives (it stands on two driven wheels and nothing else), so
    tipping over when left alone is expected of it. Without this the wheel-legged biped, which balances well, was
    told to get more feet or a caster. Uses the spec checked last (on its floor, at its spawn height)."""
    parent = mujoco.MjSpec()
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations", "noslip_iterations"):
        setattr(parent.option, name, getattr(assembly.spec.option, name))  # the physics it was checked with
    parent.attach(assembly.spec, prefix="robot/", frame=parent.worldbody.add_frame())
    model = parent.compile()
    data = mujoco.MjData(model)
    for joint, value in assembly.home_qpos.items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint}")
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = value
    return find_balancer(model, "robot", data.qpos.copy()) is not None


def part_of(model: mujoco.MjModel, body: int) -> str:
    """The part a body belongs to (mounted robots count as their mount part)."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "base"
    return name.split("/", 1)[0]


def _overlaps(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    floor = model.geom("check_floor").id
    inside: dict[tuple[str, str], float] = {}
    in_floor: dict[str, float] = {}
    for c in data.contact[: data.ncon]:
        if c.dist > -OVERLAP_ERROR:
            continue
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if floor in (g1, g2):
            other = g2 if g1 == floor else g1
            part = part_of(model, int(model.geom_bodyid[other]))
            in_floor[part] = max(in_floor.get(part, 0.0), -c.dist)
            continue
        a, b = sorted((part_of(model, int(model.geom_bodyid[g1])), part_of(model, int(model.geom_bodyid[g2]))))
        if a == b:
            continue  # inside one mounted robot: its own design
        inside[(a, b)] = max(inside.get((a, b), 0.0), -c.dist)
    if in_floor:
        part, depth = max(in_floor.items(), key=lambda kv: kv[1])
        raise ValueError(
            f"part {part!r} goes {depth * 1000:.0f} mm into the floor at the home pose while the base stands on it; "
            "move it up, change its joint's home angle, or make the base taller"
        )
    if inside:
        listed = sorted(inside.items(), key=lambda kv: -kv[1])[:3]
        pairs = "; ".join(f"{a!r} and {b!r} by {d * 1000:.0f} mm" for (a, b), d in listed)
        raise ValueError(
            f"parts overlap at the home pose: {pairs}. Move one of them (at / shift), make it smaller, change a "
            "joint's home angle, or set collide: false on a part that is only for looks"
        )


def _settle(
    model: mujoco.MjModel, data: mujoco.MjData, assembly: Assembly, strain: dict[str, float] | None = None
) -> list[str]:
    base = model.body("base").id
    steps = int(SETTLE_SECONDS / model.opt.timestep)
    servos = [a for a in range(model.nu) if is_position_servo(model, a) and model.actuator_forcelimited[a]]
    peak = np.zeros(model.nu)
    for i in range(steps):
        mujoco.mj_step(model, data)
        if i >= steps // 2:
            np.maximum(peak, np.abs(data.actuator_force), out=peak)
    if not np.isfinite(data.qpos).all() or sum(int(data.warning[int(w)].number) for w in _DIVERGENCE):
        raise ValueError(
            "the robot blows up in the physics simulation (a part flies off or the numbers become NaN). "
            "Usual causes: a very light part on a strong servo, parts pushed into each other, or a tiny link; "
            "make light parts heavier (mass), use a smaller servo, or make links longer"
        )
    notes: list[str] = []
    weak, strained = [], []
    walker = assembly.floating and assembly.feet >= 4
    for a in servos:
        limit = float(model.actuator_forcerange[a][1])
        share = peak[a] / limit if limit > 0 else 0.0
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if strain is not None and name:
            strain[name] = float(share)
        if share >= WEAK:
            weak.append((name, share, limit))
        elif share >= (WALK_STRAINED if walker else STRAINED):
            strained.append((name, share, limit))
    servos_by_strength = sorted(
        (a for a in components.ACTUATORS.values() if a.kind == "servo"), key=lambda a: a.stall_torque
    )
    stronger = ", ".join(f"{a.name} ({a.stall_torque} N*m)" for a in servos_by_strength)
    if weak:
        names = ", ".join(f"{n} ({s:.0%} of {lim:.2f})" for n, s, lim in weak[:4])
        notes.append(
            f"warning: too weak: {names} use all their force just to hold the home pose, so the robot sags. "
            f"Use stronger servos (by strength: {stronger}), add gear: 2 to those parts, or make the robot lighter."
        )
    if strained and walker:
        names = ", ".join(f"{n} ({s:.0%})" for n, s, _ in strained[:4])
        notes.append(
            f"note: little headroom to walk: {names} of their force is used standing on all {assembly.feet} feet; a "
            "walk holds the robot on half of them, about twice that, so it may fall short or slide backwards. "
            f"Stronger servos give it room (by strength: {stronger})."
        )
    elif strained:
        names = ", ".join(f"{n} ({s:.0%})" for n, s, _ in strained[:4])
        notes.append(
            f"note: little headroom: {names} of their force is used standing still; walking or lifting may be too "
            "much for them."
        )
    if assembly.floating:
        up = float(data.xmat[base].reshape(3, 3)[2, 2])
        if up < 0.7 and not _balances(assembly):
            notes.append(
                "note: the robot does not stay upright by itself (it tips over within a second at rest); give it "
                "more feet, a wider stance, a caster, or a balance controller."
            )
    return notes
