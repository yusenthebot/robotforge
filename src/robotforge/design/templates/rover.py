"""Two-wheel differential-drive rover with a caster ball."""

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
    declare_camera,
    imu,
    new_spec,
)
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


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

        vis = Visuals(spec)
        chassis = spec.worldbody.add_body(name="chassis")
        chassis.add_freejoint(name="root")
        deck = chassis.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[p.length / 2, p.width / 2 - 0.012, deck_h / 2],
            pos=[0, 0, deck_h / 2 + 0.004],
            rgba=accent,
            density=density,
        )
        top = chassis.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[p.length / 3, p.width / 3, deck_h / 3],
            pos=[-p.length * 0.05, 0, deck_h + deck_h / 3 + 0.004],
            rgba=SHELL,
            density=density,
        )
        ball = chassis.add_geom(
            type=GEOM.mjGEOM_SPHERE,
            size=[caster_r],
            pos=[-p.length / 2 + caster_r, 0, caster_r - r],
            rgba=DARK,
            friction=[0.02, 0.001, 0.0001],
            # MuJoCo takes the larger friction of two touching geoms unless one has priority, so the ball dragged
            # with the floor's grip (1.0) instead of its own 0.02: a brake carrying a share of the weight. Of 96
            # designs (2 motors x 3 materials x 4 lengths x 4 wheel sizes, 3 s at 0.3 m/s) 80 covered under a
            # quarter of their way, among them a 1.7 kg rover on 7 cm wheels and the agent eval's Mars rover;
            # with priority 32 do, all pushing with at most 1.1 % of their weight. The builder's casters had it.
            priority=1,
            density=components.MATERIALS["rubber"],
        )
        socket = chassis.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[caster_r * 0.9, (r - caster_r) / 2 + 0.004],
            pos=[-p.length / 2 + caster_r, 0, (caster_r - r) / 2 + caster_r / 2],
            rgba=DARK,
            mass=0.005,
            contype=0,
            conaffinity=0,
        )
        for geom in (deck, top, ball, socket):
            vis.hide(geom)
        _dress_chassis(vis, chassis, p.length, p.width, deck_h, r, caster_r, accent)
        chassis.add_camera(name="front_cam", pos=[p.length / 2, 0, deck_h], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
        declare_camera(spec, chassis, "front_cam", 75)
        imu(spec, chassis)

        for side, sign in (("left", 1), ("right", -1)):
            wheel = chassis.add_body(name=f"{side}_wheel", pos=[p.length / 6, sign * (p.width / 2), axle_z])
            wheel.add_joint(name=f"{side}_wheel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=0.002)
            tyre = wheel.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[r, 0.012],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                friction=[1.2, 0.01, 0.001],
                density=components.MATERIALS["rubber"],
            )
            hub = wheel.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[r * 0.5, 0.0125],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=SHELL,
                mass=0.002,
                contype=0,
                conaffinity=0,
            )
            vis.hide(tyre)
            vis.hide(hub)
            _dress_wheel(vis, wheel, r, sign, accent)
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

        vis.finish()
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


# ---- looks: visual-only meshes (no mass, no contacts) over the collision primitives above ----------


def _dress_chassis(
    vis: Visuals, chassis, length: float, width: float, deck_h: float, r: float, caster_r: float, accent
):
    hx, hy, hz = length / 2, width / 2 - 0.012, deck_h / 2
    z = hz + 0.004
    corner = min(hy * 0.35, hx * 0.2)
    body = M.rounded_box(hx, hy, hz, min(hz * 0.6, 0.007), segments=4, corner=corner)
    vis.add(chassis, "shell", accent, body, (0, 0, z))
    skid = M.rounded_box(hx * 0.94, hy * 0.9, 0.003, 0.002, segments=2, corner=corner * 0.8)
    vis.add(chassis, "dark", DARK, skid, (0, 0, 0.0035))
    # Light top cover with a dark glass status screen and a metal button.
    tx, ty, tz = length / 3, width / 3, deck_h / 3
    cover = M.rounded_box(tx, ty, tz, min(tz * 0.7, 0.005), segments=3, corner=min(ty * 0.3, 0.014))
    top_z = deck_h + tz + 0.004
    vis.add(chassis, "shell", SHELL, cover, (-length * 0.05, 0, top_z))
    screen = M.rounded_box(tx * 0.42, ty * 0.6, 0.0012, 0.0008, segments=2, corner=0.004)
    vis.add(chassis, "glass", P.GLASS, screen, (-length * 0.05 - tx * 0.38, 0, top_z + tz))
    vis.add(
        chassis,
        "metal",
        P.METAL,
        M.cylinder(0.0045, 0.0022, segments=20, bevel=0.0008, z0=0.0),
        (-length * 0.05 + tx * 0.55, ty * 0.5, top_z + tz - 0.0005),
    )
    vis.add(chassis, "led", P.LED_GREEN, P.led(0.0022)["led"], (-length * 0.05 + tx * 0.55, -ty * 0.5, top_z + tz))
    # Camera lens and headlights on the front face, tail lights at the back.
    lens_r = min(hz * 0.62, 0.0075)
    for preset, mesh in P.lens(lens_r, 0.003).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(chassis, preset, color, mesh, (hx, 0, z), M.Z_TO_X)
    lamp = M.rounded_box(0.0015, hy * 0.13, hz * 0.32, 0.001, segments=2, corner=hz * 0.3)
    for side in (1, -1):
        vis.add(chassis, "led", P.LED_WHITE, lamp, (hx + 0.0003, side * hy * 0.62, z))
        vis.add(chassis, "led", P.LED_RED, lamp, (-hx - 0.0003, side * hy * 0.62, z))
    # Ball caster: a chrome ball held in a dark socket under the tail.
    cx = -hx + caster_r
    ball_z = caster_r - r
    vis.add(chassis, "metal", P.METAL, M.sphere(caster_r, rings=12, segments=24), (cx, 0, ball_z))
    top = 0.004
    socket = M.lathe(
        [
            (0.0, ball_z + caster_r * 0.2),
            (caster_r * 1.06, ball_z + caster_r * 0.2),
            (caster_r * 1.0, top - 0.003),
            (caster_r * 1.3, top - 0.0025),
            (caster_r * 1.3, top),
            (0.0, top),
        ],
        segments=32,
        sharp=(1, 2, 3, 4),
    )
    vis.add(chassis, "dark", DARK, socket, (cx, 0, 0))


def _dress_wheel(vis: Visuals, wheel, r: float, sign: int, accent) -> None:
    outward = M.Z_TO_Y if sign > 0 else M.rot_x(math.pi / 2)
    colors = {"rubber": P.RUBBER, "shell": accent, "dark": DARK, "metal": P.METAL}
    for preset, mesh in P.tyre(r, 0.012).items():
        vis.add(wheel, preset, colors[preset], mesh, (0, 0, 0), outward)
