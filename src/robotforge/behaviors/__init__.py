"""Built-in behaviors and how to recognize which robots they fit."""

from __future__ import annotations

from enum import StrEnum

import mujoco


class Kind(StrEnum):
    ARM = "arm"
    QUADRUPED = "quadruped"
    ROVER = "rover"
    OTHER = "other"


def robot_kind(model: mujoco.MjModel, robot: str, template: str | None = None) -> Kind:
    """Recognize a robot by its template, or by the names our templates use."""
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
    return Kind.OTHER
