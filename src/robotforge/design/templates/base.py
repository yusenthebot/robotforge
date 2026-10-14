"""Shared pieces for robot templates.

A template turns a small set of beginner-friendly parameters into an MjSpec with
joints, actuators sized from real parts, sensors and a home pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import mujoco
from pydantic import BaseModel, ConfigDict

from robotforge.design import components

GEOM = mujoco.mjtGeom
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
        damping=part.damping,
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


def servo_case(body: mujoco.MjsBody, part: components.Actuator, pos: list[float], quat=None) -> None:
    """The visible, massive servo housing at a joint."""
    sx, sy, sz = part.size
    body.add_geom(
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
