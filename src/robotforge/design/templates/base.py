"""Shared pieces for robot templates.

A template turns a small set of beginner-friendly parameters into an MjSpec with
joints, actuators sized from real parts, sensors and a home pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict

from robotforge.assets.bridge import Dress
from robotforge.design import components
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P
from robotforge.sensors import declare

GEOM = mujoco.mjtGeom
# Joint modules run a velocity loop in their driver; model it as damping of at least this fraction of the
# servo's stiffness (seconds). Back-EMF alone leaves a stiff BLDC joint on a heavy limb ringing.
DRIVER_DAMPING = 0.02
DARK = [0.16, 0.17, 0.2, 1.0]
SHELL = [0.9, 0.91, 0.93, 1.0]


class TemplateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: str | list[float] = "blue"


@dataclass
class BuiltRobot:
    """A template's output: the spec plus how it should start."""

    spec: mujoco.MjSpec
    home_qpos: dict[str, float] = field(default_factory=dict)  # joint name -> value
    home_ctrl: dict[str, float] = field(default_factory=dict)  # actuator name -> value
    spawn_height: float = 0.0  # base height above the floor so feet/wheels touch it at home
    floating: bool = False  # True when the root body has a free joint
    notes: list[str] = field(default_factory=list)
    doctor: dict | None = None  # import doctor report (imported files only)


class Template:
    name: ClassVar[str]
    summary: ClassVar[str]
    Params: ClassVar[type[TemplateParams]]

    def build(self, params: TemplateParams) -> BuiltRobot:  # pragma: no cover - interface
        raise NotImplementedError


def new_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.compiler.degree = False  # all angles in radians
    return spec


def servo_joint(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    name: str,
    axis: list[float],
    joint_range: tuple[float, float],
    part: components.Actuator,
    ctrl_range: tuple[float, float] | None = None,
) -> mujoco.MjsJoint:
    """A hinge driven by a position-controlled servo with real torque limits."""
    joint = body.add_joint(
        name=name,
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=axis,
        range=list(joint_range),
        limited=mujoco.mjtLimited.mjLIMITED_TRUE,
        damping=max(part.damping, DRIVER_DAMPING * part.kp),
        armature=part.armature,
        frictionloss=0.02 * part.stall_torque,
    )
    act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=part.kp)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = list(ctrl_range or joint_range)
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-part.stall_torque, part.stall_torque]
    return joint


def servo_case(body: mujoco.MjsBody, part: components.Actuator, pos: list[float], quat=None) -> mujoco.MjsGeom:
    """The visible, massive servo housing at a joint."""
    sx, sy, sz = part.size
    return body.add_geom(
        type=GEOM.mjGEOM_BOX,
        size=[sx / 2, sy / 2, sz / 2],
        pos=pos,
        quat=quat or [1, 0, 0, 0],
        mass=part.mass,
        rgba=DARK,
    )


def imu(spec: mujoco.MjSpec, body: mujoco.MjsBody, prefix: str = "") -> None:
    body.add_site(name=f"{prefix}imu", pos=[0, 0, 0], size=[0.005, 0, 0])
    spec.add_sensor(
        name=f"{prefix}accel",
        type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{prefix}imu",
    )
    spec.add_sensor(
        name=f"{prefix}gyro",
        type=mujoco.mjtSensor.mjSENS_GYRO,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{prefix}imu",
    )
    spec.add_sensor(
        name=f"{prefix}orientation",
        type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=f"{prefix}imu",
    )
    # Declared like add_sensor's (sensors/declare.py), so read_sensors and scripts read it by name.
    native = {"accel": f"{prefix}accel", "gyro": f"{prefix}gyro", "quat": f"{prefix}orientation"}
    declare.declare(
        spec, {"name": f"{prefix}imu", "type": "imu", "body": body.name, "site": f"{prefix}imu", "native": native}
    )


def declare_camera(spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str, fovy: float) -> None:
    """A template's camera, declared like add_sensor's so read_sensors reads it (sensors/declare.py)."""
    declare.declare(
        spec,
        {
            "name": name,
            "type": "camera",
            "body": body.name,
            "camera": name,
            "mode": "rgb",
            "width": 320,
            "height": 240,
            "fov": fovy,
            "range": 10.0,
        },
    )


def declare_touch(spec: mujoco.MjSpec, body: mujoco.MjsBody, name: str, site: str) -> None:
    """A template's touch sensor (a foot's), declared like add_sensor's."""
    definition = {
        "name": name,
        "type": "touch",
        "body": body.name,
        "cells": [site],
        "touch": [name],
        "rows": 1,
        "cols": 1,
    }
    declare.declare(spec, definition)


def dress_servo(
    vis: Visuals,
    body: mujoco.MjsBody,
    part: components.Actuator,
    pos: list[float],
    quat=None,
    horn: tuple[int, int] = (2, 1),
    dress: Dress | None = None,
) -> None:
    """Draw a servo case (as placed by `servo_case`) as a rounded housing with its round horn on the
    face `horn` = (axis, sign) of the case frame, e.g. (0, -1) for the case's -x face. With a `dress` (a style) the
    case is drawn as that style's joint housing instead, filling the case's cross-section."""
    axis, sign = horn
    half = [v / 2 for v in part.size]
    e = np.eye(3)
    x_l = e[(axis + 1) % 3]
    z_l = sign * e[axis]
    rot = np.column_stack([x_l, np.cross(z_l, x_l), z_l])
    if quat is not None:
        rot = M.quat_to_mat(quat) @ rot
    hx, hy, hz = half[(axis + 1) % 3], half[(axis + 2) % 3], half[axis]
    if dress is not None:
        dress.draw(vis, body, "joint_actuator", pos=pos, rot=rot, radius=1.08 * min(hx, hy), length=2 * hz)
        return
    horn_r = 0.78 * min(hx, hy)
    for preset, mesh in P.servo_box(hx, hy, hz, horn_r).items():
        vis.add(body, preset, DARK if preset == "dark" else P.METAL, mesh, pos, rot)
