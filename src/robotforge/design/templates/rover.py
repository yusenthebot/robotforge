"""Two-wheel differential-drive rover with a caster ball."""

from __future__ import annotations

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
)


class RoverParams(TemplateParams):
    length: float = Field(0.22, ge=0.08, le=1.0, description="chassis length in meters")
    width: float = Field(0.16, ge=0.06, le=1.0, description="chassis width in meters (wheels outside)")
    wheel_radius: float = Field(0.035, ge=0.015, le=0.3, description="drive wheel radius in meters")
    motor: str = Field("tt_motor", description="wheel motor from the parts catalog")
    material: Literal["pla", "aluminum", "carbon"] = "pla"


class Rover(Template):
    name = "rover"
    summary = "Differential-drive rover: two powered wheels (velocity control), caster ball, IMU, front camera."
    Params = RoverParams

    def build(self, p: RoverParams) -> BuiltRobot:
        part = components.actuator(p.motor)
        density = components.MATERIALS[p.material]
        accent = components.color(p.color)
        spec = new_spec()
        r = p.wheel_radius
        deck_h = max(0.02, 0.5 * r)
        axle_z = 0.0  # chassis origin sits on the wheel axle
        caster_r = r * 0.45

        chassis = spec.worldbody.add_body(name="chassis")
        chassis.add_freejoint(name="root")
        chassis.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[p.length / 2, p.width / 2 - 0.012, deck_h / 2],
            pos=[0, 0, deck_h / 2 + 0.004],
            rgba=accent,
            density=density,
        )
        chassis.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[p.length / 3, p.width / 3, deck_h / 3],
            pos=[-p.length * 0.05, 0, deck_h + deck_h / 3 + 0.004],
            rgba=SHELL,
            density=density,
        )
        chassis.add_geom(
            type=GEOM.mjGEOM_SPHERE,
            size=[caster_r],
            pos=[-p.length / 2 + caster_r, 0, caster_r - r],
            rgba=DARK,
            friction=[0.02, 0.001, 0.0001],
            density=components.MATERIALS["rubber"],
        )
        chassis.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[caster_r * 0.9, (r - caster_r) / 2 + 0.004],
            pos=[-p.length / 2 + caster_r, 0, (caster_r - r) / 2 + caster_r / 2],
            rgba=DARK,
            mass=0.005,
            contype=0,
            conaffinity=0,
        )
        chassis.add_camera(name="front_cam", pos=[p.length / 2, 0, deck_h], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
        imu(spec, chassis)

        for side, sign in (("left", 1), ("right", -1)):
            wheel = chassis.add_body(name=f"{side}_wheel", pos=[p.length / 6, sign * (p.width / 2), axle_z])
            wheel.add_joint(name=f"{side}_wheel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.002)
            wheel.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[r, 0.012],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                friction=[1.2, 0.01, 0.001],
                density=components.MATERIALS["rubber"],
            )
            wheel.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[r * 0.5, 0.0125],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=SHELL,
                mass=0.002,
                contype=0,
                conaffinity=0,
            )
            act = spec.add_actuator(name=f"{side}_wheel", target=f"{side}_wheel", trntype=mujoco.mjtTrn.mjTRN_JOINT)
            # Stiff speed loop: full torque at ~5 % speed error, like a motor driver with feedback.
            act.set_to_velocity(kv=part.stall_torque / (0.05 * part.no_load_speed))
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = [-part.no_load_speed, part.no_load_speed]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.forcerange = [-part.stall_torque, part.stall_torque]
            chassis.add_geom(
                type=GEOM.mjGEOM_BOX,
                size=[part.size[0] / 2, part.size[1] / 2, part.size[2] / 2],
                pos=[p.length / 6, sign * (p.width / 2 - 0.012 - part.size[1] / 2 - 0.004), deck_h / 2 + 0.004],
                rgba=[0.95, 0.8, 0.2, 1],
                mass=part.mass,
            )

        return BuiltRobot(
            spec=spec,
            home_qpos={},
            home_ctrl={},
            spawn_height=r + 0.002,
            floating=True,
            notes=[
                f"rover, wheels r={r:.3f} m, {part.name} motors; controls are wheel speeds in rad/s "
                f"(max {part.no_load_speed})"
            ],
        )
