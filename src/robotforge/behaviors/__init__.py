"""Built-in behaviors and how to recognize which robots they fit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import mujoco
import numpy as np

from robotforge.sim.actuators import drives_joint, is_velocity_servo


class Kind(StrEnum):
    ARM = "arm"
    QUADRUPED = "quadruped"
    HEXAPOD = "hexapod"
    HUMANOID = "humanoid"
    ROVER = "rover"
    BALANCER = "balancer"  # two wheels on one axle, balances like a Segway
    SNAKE = "snake"  # a chain of side-to-side joints that slithers
    BOAT = "boat"  # floats and drives with thrusters that push along the water
    ROV = "rov"  # an underwater robot: thrusters that push up and down too
    AIRCRAFT = "aircraft"  # rotors (thrust actuators) on a free-moving airframe: flies
    OTHER = "other"


def robot_kind(model: mujoco.MjModel, robot: str, template: str | None = None, by_feet: bool = True) -> Kind:
    """Recognize a robot by its template, by the names our templates use, or by its shape. `by_feet`: a floating robot
    that stands on two feet is a humanoid even when nothing else says so (see stance.py); off, only the names and the
    shapes of dogs, wheels and snakes decide, as before robots were told apart by their feet."""
    if template in {k.value for k in Kind}:
        return Kind(template)

    def has(obj: mujoco.mjtObj, name: str) -> bool:
        return mujoco.mj_name2id(model, obj, f"{robot}/{name}") >= 0

    from robotforge.behaviors.airframe import is_aircraft  # lazy: keeps this package light

    if template == "drone" or is_aircraft(model, robot):
        return Kind.AIRCRAFT

    if has(mujoco.mjtObj.mjOBJ_SITE, "tcp") or (_has_mounted_tool(model, robot) and not _floating(model, robot)):
        return Kind.ARM  # an arm, or a fixed robot built from parts with a gripper mounted on it
    from robotforge.behaviors.marine import marine_kind  # lazy: it imports this package

    marine = marine_kind(model, robot) if _floating(model, robot) else None
    if marine is not None:
        return Kind(marine)  # thrusters: a boat, or an ROV when some push up and down
    from robotforge.behaviors.steering import has_steering

    if has_steering(model, robot):
        return Kind.ROVER  # binding reports unsupported geometry; never silently skid-steer these wheels
    from robotforge.behaviors.balance import find_balancer  # lazy: it imports this package

    if find_balancer(model, robot, home_pose(model)) is not None:
        return Kind.BALANCER
    if has(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel") and has(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel"):
        return Kind.ROVER
    if has(mujoco.mjtObj.mjOBJ_SITE, "front_left_foot"):
        return Kind.QUADRUPED
    if has(mujoco.mjtObj.mjOBJ_SITE, "left_foot") and has(mujoco.mjtObj.mjOBJ_JOINT, "left_knee"):
        return Kind.HUMANOID
    if find_wheels(model, robot) is not None:
        return Kind.ROVER  # an imported or hand-built robot with a driven wheel on each side
    from robotforge.behaviors.stance import stands_on_two_feet  # lazy: it imports this package

    if by_feet and stands_on_two_feet(model, robot, home_pose(model)):
        return Kind.HUMANOID  # nothing else says so, but it stands on two feet under a body taller than they are long
    from robotforge.behaviors.quadruped import discover_legs  # lazy: it imports this package

    found = discover_legs(model, robot, home_pose(model))
    if found is not None:  # library, imported and built walkers
        return Kind.QUADRUPED if len(found[1]) == 4 else Kind.HEXAPOD
    from robotforge.behaviors.snake import find_spine  # lazy: it imports this package

    if find_spine(model, robot, home_pose(model)) is not None:
        return Kind.SNAKE
    return Kind.OTHER


def home_pose(model: mujoco.MjModel):
    """qpos of the world's home keyframe (robots stand in it), or qpos0 without one."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[key].copy() if key >= 0 else model.qpos0.copy()


def home_controls(model: mujoco.MjModel) -> np.ndarray:
    """ctrl of the world's home keyframe (every servo aimed at its home angle), or zeros without one."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_ctrl[key].copy() if key >= 0 else np.zeros(model.nu)


@dataclass(frozen=True)
class WheelSet:
    """Driven wheels of a differential or skid-steer robot, grouped by side."""

    left: tuple[int, ...]  # actuator ids
    right: tuple[int, ...]
    left_signs: tuple[float, ...]  # +1 when a positive speed rolls the wheel forward, -1 when its axis is flipped
    right_signs: tuple[float, ...]
    track: float  # distance between the two sides, meters
    radius: float  # wheel radius, meters


def find_wheels(model: mujoco.MjModel, robot: str) -> WheelSet | None:
    """Find speed-controlled wheels with sideways axes on both sides of `robot`'s first body."""
    from robotforge.behaviors.steering import has_steering

    if has_steering(model, robot):
        return None
    prefix = f"{robot}/"
    root = next(
        (
            b
            for b in range(1, model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
        ),
        None,
    )
    if root is None:
        return None
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frame = data.xmat[root].reshape(3, 3)
    sides: dict[int, list[tuple[int, float, float, float]]] = {1: [], -1: []}
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if not name.startswith(prefix) or not is_velocity_servo(model, a) or not drives_joint(model, a):
            continue
        joint = int(model.actuator_trnid[a, 0])
        if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        axis = frame.T @ data.xaxis[joint]
        lateral = float((frame.T @ (data.xanchor[joint] - data.xpos[root]))[1])
        if abs(axis[1]) < 0.7 or abs(lateral) < 1e-3:
            continue
        sides[1 if lateral > 0 else -1].append((a, lateral, _wheel_radius(model, data, joint), float(np.sign(axis[1]))))
    if not sides[1] or not sides[-1]:
        return None
    left, right = sides[1], sides[-1]
    track = float(np.mean([w[1] for w in left]) - np.mean([w[1] for w in right]))
    radius = float(np.mean([w[2] for w in left + right]))
    return WheelSet(
        tuple(w[0] for w in left),
        tuple(w[0] for w in right),
        tuple(w[3] for w in left),
        tuple(w[3] for w in right),
        track,
        radius,
    )


def _wheel_radius(model: mujoco.MjModel, data: mujoco.MjData, joint: int) -> float:
    """Radius of the wheel on `joint`: how far its colliding shapes reach from the axle, ignoring sideways extent."""
    body = int(model.jnt_bodyid[joint])
    axis, anchor = data.xaxis[joint], data.xanchor[joint]
    reach = 0.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        world = data.geom_xpos[g] + _geom_points(model, g) @ data.geom_xmat[g].reshape(3, 3).T - anchor
        across = world - np.outer(world @ axis, axis)
        reach = max(reach, float(np.linalg.norm(across, axis=1).max()))
    return reach or 0.05


def _geom_points(model: mujoco.MjModel, g: int) -> np.ndarray:
    """Points on a geom's surface in its own frame (enough to measure how far it reaches from an axis)."""
    size, kind = model.geom_size[g], int(model.geom_type[g])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[g])
        first = int(model.mesh_vertadr[mesh])
        return np.asarray(model.mesh_vert[first : first + int(model.mesh_vertnum[mesh])])
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([size[0] * np.cos(angles), size[0] * np.sin(angles)], axis=1)
        return np.vstack([np.column_stack([ring, np.full(24, z)]) for z in (-size[1], size[1])])
    if kind in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        extra = size[1] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0
        r = size[0]
        return np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r + extra], [0, 0, -r - extra]])
    half, center = model.geom_aabb[g, 3:], model.geom_aabb[g, :3]
    return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + center


def _has_mounted_tool(model: mujoco.MjModel, robot: str) -> bool:
    names = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "" for i in range(model.nsite))
    return any(n.startswith(f"{robot}/") and n.endswith("/tcp") for n in names)


def _floating(model: mujoco.MjModel, robot: str) -> bool:
    """True when the robot's root body moves freely (a free joint), false when it is bolted down."""
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0 and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
            f"{robot}/"
        ):
            start, count = model.body_jntadr[b], model.body_jntnum[b]
            return any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(start, start + count))
    return False
