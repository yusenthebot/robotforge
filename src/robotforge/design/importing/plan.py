"""The rig plan (everything decided about a robot's physics) and how to replay it on a MuJoCo spec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import mujoco

PLAN_VERSION = 3
VISUAL_GROUP = 2
COLLISION_GROUP = 3  # hidden by the viewer, like MuJoCo Menagerie
MAX_HULL_VERTICES = 64
MOVABLE = ("revolute", "continuous", "prismatic")


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
