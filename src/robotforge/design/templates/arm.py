"""Desktop robot arm (3 to 6 joints) with an optional parallel gripper."""

from __future__ import annotations

import math
from typing import Literal

import mujoco
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.base import (
    DARK,
    GEOM,
    SHELL,
    BuiltRobot,
    Template,
    TemplateParams,
    new_spec,
    servo_case,
    servo_joint,
)


class ArmParams(TemplateParams):
    dof: int = Field(5, ge=3, le=6, description="number of arm joints (gripper not counted)")
    reach: float = Field(0.45, ge=0.15, le=1.2, description="shoulder-to-fingertip length in meters")
    base_height: float = Field(0.08, ge=0.02, le=0.5, description="height of the shoulder axis in meters")
    servo: str = Field("sts3215", description="joint actuator from the parts catalog")
    gripper: bool = True
    gripper_opening: float = Field(0.07, ge=0.02, le=0.2, description="max finger gap in meters")
    material: Literal["pla", "aluminum", "carbon"] = "pla"


class Arm(Template):
    name = "arm"
    summary = "Desktop robot arm with 3-6 servo joints and an optional parallel gripper; base fixed to the floor."
    Params = ArmParams

    def build(self, p: ArmParams) -> BuiltRobot:
        part = components.actuator(p.servo)
        density = components.MATERIALS[p.material]
        accent = components.color(p.color)
        spec = new_spec()
        radius = max(0.012, 0.035 * p.reach)
        upper, fore = 0.45 * p.reach, 0.35 * p.reach
        hand = 0.2 * p.reach - (0.05 if p.gripper else 0.0)

        base = spec.worldbody.add_body(name="base")
        base.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[2.2 * radius + 0.02, p.base_height / 4],
            pos=[0, 0, p.base_height / 4],
            rgba=DARK,
            density=components.MATERIALS["aluminum"],
        )

        turret = base.add_body(name="turret", pos=[0, 0, p.base_height / 2])
        servo_joint(spec, turret, "yaw", [0, 0, 1], (-math.pi, math.pi), part)
        turret.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[2 * radius + 0.01, p.base_height / 4],
            pos=[0, 0, p.base_height / 4],
            rgba=SHELL,
            density=density,
        )
        servo_case(turret, part, [0, 0, p.base_height / 2])

        links = [("shoulder", upper), ("elbow", fore)]
        if p.dof >= 4:
            links.append(("wrist_pitch", hand))
        parent, z = turret, p.base_height / 2
        for i, (joint_name, length) in enumerate(links):
            body = parent.add_body(name=f"link{i + 1}", pos=[0, 0, z])
            limit = 2.6 if joint_name != "shoulder" else 1.9
            servo_joint(spec, body, joint_name, [0, 1, 0], (-limit, limit), part)
            body.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[radius * 1.25, part.size[1] / 2 + 0.004],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                mass=part.mass,
            )
            is_last = i == len(links) - 1
            rgba = accent if i == 0 else SHELL
            body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, length],
                size=[radius * (0.8 if is_last else 1.0), 0, 0],
                rgba=rgba,
                density=density,
            )
            parent, z = body, length

        if p.dof >= 5:
            roll = parent.add_body(name="wrist", pos=[0, 0, z])
            if p.dof >= 6:
                servo_joint(spec, roll, "wrist_yaw", [1, 0, 0], (-1.6, 1.6), part)
            servo_joint(spec, roll, "wrist_roll", [0, 0, 1], (-math.pi, math.pi), part)
            roll.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[radius * 1.1, 0.012],
                pos=[0, 0, 0.012],
                rgba=DARK,
                mass=part.mass,
            )
            parent, z = roll, 0.024

        tool = parent.add_body(name="hand", pos=[0, 0, z])
        tool.add_camera(name="wrist_cam", pos=[0.035, 0, 0.0], xyaxes=[0, -1, 0, -1, 0, 0], fovy=70)
        tcp_z = 0.0
        if p.gripper:
            tcp_z = self._gripper(spec, tool, p, radius, accent)
        tool.add_site(name="tcp", pos=[0, 0, tcp_z], size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])
        spec.add_sensor(
            name="tcp_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tcp"
        )

        # A "ready" pose: elbow bent, tool pointing down in front of the base.
        home = {"shoulder": 0.35, "elbow": 1.45}
        if p.dof >= 4:
            home["wrist_pitch"] = math.pi - home["shoulder"] - home["elbow"]
        ctrl = dict(home)
        if p.gripper:
            ctrl["gripper"] = p.gripper_opening / 2
        return BuiltRobot(
            spec=spec,
            home_qpos=home,
            home_ctrl=ctrl,
            spawn_height=0.0,
            floating=False,
            notes=[f"{p.dof}-joint arm, reach {p.reach:.2f} m, {part.name} servos ({part.stall_torque} N*m)"],
        )

    @staticmethod
    def _gripper(spec: mujoco.MjSpec, tool: mujoco.MjsBody, p: ArmParams, radius: float, accent) -> float:
        return add_gripper(spec, tool, p.gripper_opening, accent)


def add_gripper(spec: mujoco.MjSpec, tool: mujoco.MjsBody, opening: float, accent) -> float:
    """A parallel two-finger gripper on `tool`, opening along its y axis, fingers along +z.

    Adds the coupled finger joints, the `gripper` actuator (its control is one finger's travel)
    and returns the height of the point between the fingertips above the tool frame.
    """
    palm_h = 0.02
    half_w = opening / 2 + 0.012
    tool.add_geom(
        type=GEOM.mjGEOM_BOX,
        size=[0.018, half_w, palm_h / 2],
        pos=[0, 0, palm_h / 2],
        rgba=DARK,
        density=components.MATERIALS["pla"],
    )
    finger_len = 0.05
    travel = opening / 2
    for side, sign in (("left", 1), ("right", -1)):
        finger = tool.add_body(name=f"finger_{side}", pos=[0, sign * 0.006, palm_h])
        finger.add_joint(
            name=f"finger_{side}",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, sign, 0],
            range=[0, travel],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=1.0,
            armature=0.001,
        )
        finger.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[0.012, 0.005, finger_len / 2],
            pos=[0, 0, finger_len / 2],
            rgba=accent,
            friction=[1.5, 0.02, 0.002],
            condim=4,
            solref=[0.005, 1],
            mass=0.012,
        )
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_left", name2="finger_right", data=[0, 1, 0, 0, 0] + [0] * 6
    )
    act = spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_position(kp=200)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [0, travel]
    act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.forcerange = [-15, 15]
    return palm_h + finger_len - 0.012
