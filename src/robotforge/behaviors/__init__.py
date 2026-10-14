"""Built-in behaviors and how to recognize which robots they fit."""

from __future__ import annotations

from enum import StrEnum

import mujoco


class Kind(StrEnum):
    ARM = "arm"
    QUADRUPED = "quadruped"
    HEXAPOD = "hexapod"
    HUMANOID = "humanoid"
    ROVER = "rover"
    OTHER = "other"


def robot_kind(model: mujoco.MjModel, robot: str, template: str | None = None) -> Kind:
    """Recognize a robot by its template, by the names our templates use, or by its shape."""
    if template in {k.value for k in Kind}:
        return Kind(template)

    def has(obj: mujoco.mjtObj, name: str) -> bool:
        return mujoco.mj_name2id(model, obj, f"{robot}/{name}") >= 0

    if has(mujoco.mjtObj.mjOBJ_SITE, "tcp"):
        return Kind.ARM
    if has(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel") and has(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel"):
        return Kind.ROVER
    if has(mujoco.mjtObj.mjOBJ_SITE, "front_left_foot"):
        return Kind.QUADRUPED
    if has(mujoco.mjtObj.mjOBJ_SITE, "left_foot") and has(mujoco.mjtObj.mjOBJ_JOINT, "left_knee"):
        return Kind.HUMANOID
    from robotforge.behaviors.quadruped import discover_legs  # lazy: it imports this package

    found = discover_legs(model, robot, home_pose(model))
    if found is not None:  # library, imported and built walkers
        return Kind.QUADRUPED if len(found[1]) == 4 else Kind.HEXAPOD
    return Kind.OTHER


def home_pose(model: mujoco.MjModel):
    """qpos of the world's home keyframe (robots stand in it), or qpos0 without one."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[key].copy() if key >= 0 else model.qpos0.copy()
