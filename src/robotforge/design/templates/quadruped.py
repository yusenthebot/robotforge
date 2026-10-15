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
    dress_servo,
    imu,
    new_spec,
    servo_case,
    servo_joint,
)
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

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

        vis = Visuals(spec)
        trunk = spec.worldbody.add_body(name="trunk")
        trunk.add_freejoint(name="root")
        # Printed trunks are hollow shells; ~25 % of a solid block is a fair mass estimate.
        hull = trunk.add_geom(type=GEOM.mjGEOM_BOX, size=[L / 2, W / 2, H / 2], rgba=accent, density=density * 0.25)
        cover = trunk.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[L / 2 * 0.7, W / 2 * 0.8, 0.004],
            pos=[0, 0, H / 2 + 0.004],
            rgba=SHELL,
            density=density,
        )
        covered = [hull, cover]
        if p.face:
            face = trunk.add_geom(
                type=GEOM.mjGEOM_BOX,
                size=[0.004, W / 2 * 0.7, H / 2 * 0.7],
                pos=[L / 2 + 0.004, 0, 0],
                rgba=DARK,
                mass=0.005,
            )
            covered.append(face)
            for side in (1, -1):
                eye = trunk.add_geom(
                    type=GEOM.mjGEOM_SPHERE,
                    size=[H * 0.12],
                    pos=[L / 2 + 0.008, side * W * 0.16, H * 0.08],
                    rgba=[0.55, 0.85, 1.0, 1],
                    mass=0.001,
                    contype=0,
                    conaffinity=0,
                )
                covered.append(eye)
        for geom in covered:
            vis.hide(geom)
        _dress_trunk(vis, trunk, L, W, H, accent, p.face)
        trunk.add_camera(name="front_cam", pos=[L / 2 + 0.01, 0, 0], xyaxes=[0, -1, 0, 0, 0, 1], fovy=75)
        imu(spec, trunk)

        hip_x, hip_y = L / 2 - part.size[0] / 2, W / 2
        home: dict[str, float] = {}
        for leg, sx, sy in LEGS:
            hip = trunk.add_body(name=f"{leg}_hip", pos=[sx * hip_x, sy * hip_y, 0])
            servo_joint(spec, hip, f"{leg}_hip", [1, 0, 0], (-0.6, 0.6), part)
            vis.hide(servo_case(hip, part, [0, sy * part.size[1] / 2, 0]))
            dress_servo(vis, hip, part, [0, sy * part.size[1] / 2, 0], horn=(0, sx))

            upper = hip.add_body(name=f"{leg}_thigh", pos=[0, sy * (part.size[1] + leg_r), 0])
            servo_joint(spec, upper, f"{leg}_thigh", [0, 1, 0], (-1.2, 2.4), part)
            drum = upper.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[leg_r * 1.6, leg_r],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                mass=part.mass,
            )
            thigh_link = upper.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, -thigh],
                size=[leg_r, 0, 0],
                rgba=SHELL,
                density=density,
            )

            lower = upper.add_body(name=f"{leg}_shin", pos=[0, 0, -thigh])
            servo_joint(spec, lower, f"{leg}_knee", [0, 1, 0], (-2.6, -0.3), part)
            shin_link = lower.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, -shin],
                size=[leg_r * 0.8, 0, 0],
                rgba=SHELL,
                density=density,
            )
            foot = lower.add_geom(
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
            for geom in (drum, thigh_link, shin_link, foot):
                vis.hide(geom)
            _dress_leg(vis, upper, lower, sy, thigh, shin, leg_r, foot_r, accent)
        vis.finish()

        stand = (thigh + shin) * math.cos(HOME_THIGH) + foot_r
        return BuiltRobot(
            spec=spec,
            home_qpos=home,
            home_ctrl=dict(home),
            spawn_height=stand + 0.005,
            floating=True,
            notes=[f"quadruped, legs {p.leg_length:.2f} m, {part.name} servos ({part.stall_torque} N*m)"],
        )


# ---- looks: visual-only meshes (no mass, no contacts) over the collision primitives above ----------


def _dress_trunk(vis: Visuals, trunk, L: float, W: float, H: float, accent, face: bool) -> None:
    hx, hy, hz = L / 2, W / 2, H / 2
    corner = min(hy * 0.4, hx * 0.25)
    # Two-tone hull: a glossy upper shell over a dark chassis pan, meeting in a crisp step.
    upper_h = hz * 0.64
    shell = M.rounded_box(hx, hy, upper_h, min(hz * 0.42, corner), segments=4, corner=corner)
    vis.add(trunk, "shell", accent, shell, (0, 0, hz - upper_h))
    pan_h = hz * 0.46
    pan = M.rounded_box(hx - 0.003, hy - 0.003, pan_h, pan_h * 0.6, segments=3, corner=corner - 0.003)
    vis.add(trunk, "dark", DARK, pan, (0, 0, -hz + pan_h))
    # The light top cover sits exactly where the collision plate is.
    lid = M.rounded_box(hx * 0.7, hy * 0.8, 0.004, 0.0032, segments=3, corner=min(hy * 0.3, 0.02))
    vis.add(trunk, "shell", SHELL, lid, (0, 0, hz + 0.004))
    # Status light at the back of the cover and a vent grille on the rear face.
    vis.add(trunk, "led", P.LED_GREEN, P.led(0.0035)["led"], (-hx * 0.62, 0, hz + 0.008))
    grille = P.slats(hy * 0.5, hz * 0.32, 4, 0.0016)
    vis.add(trunk, "dark", DARK, grille, (-hx - 0.0002, 0, hz * 0.18), M.rot_z(math.pi / 2))
    front = hx + 0.008 if face else hx
    if face:
        visor = M.rounded_box(0.004, hy * 0.7, hz * 0.7, 0.0028, segments=3, corner=hz * 0.45)
        vis.add(trunk, "glass", P.GLASS, visor, (hx + 0.004, 0, 0))
        for side in (1, -1):
            eye = M.rounded_box(0.0012, H * 0.075, H * 0.11, 0.001, segments=2, corner=H * 0.07)
            vis.add(trunk, "led", P.LED_CYAN, eye, (front + 0.0006, side * W * 0.17, H * 0.1))
    for preset, mesh in P.lens(H * 0.1, 0.003).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(trunk, preset, color, mesh, (front, 0, -H * 0.12), M.Z_TO_X)


def _dress_leg(vis: Visuals, upper, lower, sy: int, thigh: float, shin: float, leg_r: float, foot_r: float, accent):
    down = M.rot_x(math.pi)  # links are built along +z; legs hang along -z
    side = M.Z_TO_Y if sy > 0 else M.rot_x(math.pi / 2)  # drums face outward
    for preset, mesh in P.drum(leg_r * 1.6, leg_r, leg_r * 1.05).items():
        vis.add(upper, preset, DARK if preset == "dark" else P.METAL, mesh, (0, 0, 0), side)
    width = max(leg_r * 1.55, 0.011)
    link = M.link(thigh, (width, leg_r * 0.82), (leg_r * 1.12, leg_r * 0.78), roundness=0.85, rounding=(0.9, 0.9))
    vis.add(upper, "shell", accent, link, (0, 0, 0), down)
    knee_r = leg_r * 1.22
    for preset, mesh in P.drum(knee_r, leg_r * 0.95, leg_r * 0.8).items():
        vis.add(lower, preset, DARK if preset == "dark" else P.METAL, mesh, (0, 0, 0), side)
    reach = shin - foot_r * 0.8
    shin_link = M.link(reach, (leg_r * 1.08, leg_r * 0.74), (leg_r * 0.78, leg_r * 0.68), roundness=0.9)
    vis.add(lower, "dark", DARK, shin_link, (0, 0, 0), down)
    for preset, mesh in P.ball_foot(foot_r, leg_r * 0.72).items():
        vis.add(lower, preset, P.RUBBER if preset == "rubber" else P.METAL, mesh, (0, 0, -shin))
