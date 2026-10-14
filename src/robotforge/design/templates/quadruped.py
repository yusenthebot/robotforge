"""Four-legged walking robot (12 servos: hip, thigh and knee per leg)."""

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
    imu,
    new_spec,
    servo_case,
    servo_joint,
)

LEGS = (("front_left", 1, 1), ("front_right", 1, -1), ("rear_left", -1, 1), ("rear_right", -1, -1))
HOME_THIGH = 0.8  # rad, thigh swung back
HOME_KNEE = -1.6  # rad, knee folded so the foot sits under the hip


class QuadrupedParams(TemplateParams):
    body_length: float = Field(0.32, ge=0.12, le=1.0, description="trunk length in meters")
    body_width: float = Field(0.14, ge=0.06, le=0.5, description="trunk width in meters")
    body_height: float = Field(0.06, ge=0.03, le=0.3, description="trunk thickness in meters")
    leg_length: float = Field(0.24, ge=0.08, le=0.8, description="thigh + shin length in meters")
    servo: str = Field("sts3215", description="joint actuator from the parts catalog")
    material: Literal["pla", "aluminum", "carbon"] = "pla"
    face: bool = Field(True, description="add a friendly face panel on the front")


class Quadruped(Template):
    name = "quadruped"
    summary = "Four-legged robot dog with 12 servos (hip, thigh, knee per leg), IMU, foot touch sensors, front camera."
    Params = QuadrupedParams

    def build(self, p: QuadrupedParams) -> BuiltRobot:
        part = components.actuator(p.servo)
        density = components.MATERIALS[p.material]
        accent = components.color(p.color)
        spec = new_spec()
        L, W, H = p.body_length, p.body_width, p.body_height
        thigh = shin = p.leg_length / 2
        leg_r = max(0.008, 0.06 * thigh)
        foot_r = max(0.01, 0.09 * shin)

        trunk = spec.worldbody.add_body(name="trunk")
        trunk.add_freejoint(name="root")
        # Printed trunks are hollow shells; ~25 % of a solid block is a fair mass estimate.
        trunk.add_geom(type=GEOM.mjGEOM_BOX, size=[L / 2, W / 2, H / 2], rgba=accent, density=density * 0.25)
        trunk.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[L / 2 * 0.7, W / 2 * 0.8, 0.004],
            pos=[0, 0, H / 2 + 0.004],
            rgba=SHELL,
            density=density,
        )
        if p.face:
            trunk.add_geom(
                type=GEOM.mjGEOM_BOX,
                size=[0.004, W / 2 * 0.7, H / 2 * 0.7],
                pos=[L / 2 + 0.004, 0, 0],
                rgba=DARK,
                mass=0.005,
            )
            for side in (1, -1):
                trunk.add_geom(
                    type=GEOM.mjGEOM_SPHERE,
                    size=[H * 0.12],
                    pos=[L / 2 + 0.008, side * W * 0.16, H * 0.08],
                    rgba=[0.55, 0.85, 1.0, 1],
                    mass=0.001,
                    contype=0,
                    conaffinity=0,
                )
        trunk.add_camera(name="front_cam", pos=[L / 2 + 0.01, 0, 0], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
        imu(spec, trunk)

        hip_x, hip_y = L / 2 - part.size[0] / 2, W / 2
        home: dict[str, float] = {}
        for leg, sx, sy in LEGS:
            hip = trunk.add_body(name=f"{leg}_hip", pos=[sx * hip_x, sy * hip_y, 0])
            servo_joint(spec, hip, f"{leg}_hip", [1, 0, 0], (-0.6, 0.6), part)
            servo_case(hip, part, [0, sy * part.size[1] / 2, 0])

            upper = hip.add_body(name=f"{leg}_thigh", pos=[0, sy * (part.size[1] + leg_r), 0])
            servo_joint(spec, upper, f"{leg}_thigh", [0, 1, 0], (-1.2, 2.4), part)
            upper.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[leg_r * 1.6, leg_r],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                mass=part.mass,
            )
            upper.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, -thigh],
                size=[leg_r, 0, 0],
                rgba=SHELL,
                density=density,
            )

            lower = upper.add_body(name=f"{leg}_shin", pos=[0, 0, -thigh])
            servo_joint(spec, lower, f"{leg}_knee", [0, 1, 0], (-2.6, -0.3), part)
            lower.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, -shin],
                size=[leg_r * 0.8, 0, 0],
                rgba=SHELL,
                density=density,
            )
            lower.add_geom(
                type=GEOM.mjGEOM_SPHERE,
                size=[foot_r],
                pos=[0, 0, -shin],
                rgba=DARK,
                friction=[1.2, 0.02, 0.002],
                density=components.MATERIALS["rubber"],
            )
            lower.add_site(name=f"{leg}_foot", pos=[0, 0, -shin], size=[foot_r * 1.05, 0, 0], rgba=[0, 0, 0, 0])
            spec.add_sensor(
                name=f"{leg}_touch",
                type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=f"{leg}_foot",
            )
            home.update({f"{leg}_hip": 0.0, f"{leg}_thigh": HOME_THIGH, f"{leg}_knee": HOME_KNEE})

        stand = (thigh + shin) * math.cos(HOME_THIGH) + foot_r
        return BuiltRobot(
            spec=spec,
            home_qpos=home,
            home_ctrl=dict(home),
            spawn_height=stand + 0.005,
            floating=True,
            notes=[f"quadruped, legs {p.leg_length:.2f} m, {part.name} servos ({part.stall_torque} N*m)"],
        )
