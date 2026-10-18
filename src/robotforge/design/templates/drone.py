"""Multirotor drone: a camera drone with 4, 6 or 8 rotors in an X around a rounded shell, on landing skids.

Each rotor is a thrust actuator (a MuJoCo motor on a site at the propeller hub): it pushes up along the rotor's
axis and twists the airframe the other way the propeller spins (gear[5], the reaction torque), so neighbouring
rotors spin in opposite directions and yaw comes from speeding up one set and slowing the other. A first-order
lag on each rotor (a filter actuator) stands for the motor and propeller spinning up. The propeller sits on a
body of its own (no joint: it is welded on) so the viewer can spin it with its command.
"""

from __future__ import annotations

import math
from typing import Literal

import mujoco
import numpy as np
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.base import DARK, GEOM, BuiltRobot, Template, TemplateParams, imu, new_spec
from robotforge.design.visuals import Visuals, aero
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

THRUST_TO_WEIGHT = 2.4  # all rotors at full thrust lift this many times the drone's weight
TORQUE_PER_THRUST = 0.065  # reaction torque per newton of thrust, per meter of propeller diameter
SPIN_UP = (0.015, 0.06)  # s: rotor lag = a + b * diameter (bigger propellers spin up slower)
PROP_FILL = 0.86  # share of the gap between neighbouring rotors a propeller spans
PROP_COLOR = [0.13, 0.14, 0.16, 1.0]
BELL_COLOR = [0.5, 0.52, 0.56, 1.0]
LED_FRONT = [1.0, 0.22, 0.18, 1.0]
LED_BACK = [0.3, 1.0, 0.45, 1.0]


class DroneParams(TemplateParams):
    color: str | list[float] = "white"
    rotors: Literal[4, 6, 8] = Field(4, description="number of rotors: 4 (quadrotor), 6 (hexacopter) or 8 (octocopter)")
    arm_length: float = Field(0.2, ge=0.06, le=1.0, description="m from the middle to each rotor's axis")
    size: float = Field(0.18, ge=0.05, le=1.0, description="length of the body shell in the middle, m")
    mass: float = Field(1.2, ge=0.05, le=30.0, description="kg, everything included (battery, camera)")
    prop: float | None = Field(None, ge=0.03, le=1.5, description="propeller diameter in m (default: as big as fits)")


class Drone(Template):
    name = "drone"
    summary = (
        "Multirotor drone (4, 6 or 8 rotors, thrust actuators): takes off, hovers, flies to points and lands with "
        "its built-in flight controller (move with to=[x, y, z]); camera on a gimbal, IMU."
    )
    Params = DroneParams

    def build(self, p: DroneParams) -> BuiltRobot:
        n, arm, s = p.rotors, p.arm_length, p.size
        hx, hy, hz = s / 2, 0.34 * s, 0.17 * s
        if arm < 0.75 * hx:
            raise ValueError(f"arm_length {arm:g} m is too short for a {s:g} m body: make it at least {0.75 * hx:.3f}")
        gap = 2 * arm * math.sin(math.pi / n)  # between neighbouring rotor axes
        prop = p.prop if p.prop is not None else min(PROP_FILL * gap, 2 * (arm - 0.55 * hx))
        if prop > 0.98 * gap:
            raise ValueError(
                f"propellers of {prop:g} m would hit each other {gap:.3f} m apart; use prop <= {0.98 * gap:.3f}, "
                "longer arms or fewer rotors"
            )
        accent = components.color(p.color)
        g = 9.81
        thrust = THRUST_TO_WEIGHT * p.mass * g / n
        rotor_mass = min(0.045, 0.3 / n) * p.mass
        motor_r, motor_h = aero.motor_size(prop)
        arm_r = max(0.004, 0.032 * s + 0.0025, 0.5 * motor_r)
        arm_z = 0.04 * s
        prop_z = arm_z + arm_r + motor_h
        gear_drop = 0.42 * s + 0.12 * prop
        skid_r = 0.03 * s + 0.002
        skid_y = max(0.62 * s, 0.38 * arm)
        skid_z = -hz - gear_drop

        spec = new_spec()
        vis = Visuals(spec)
        body = spec.worldbody.add_body(name="body")
        body.add_freejoint(name="root")
        rest = p.mass - n * rotor_mass - 0.1 * p.mass - 0.05 * p.mass - 0.04 * p.mass
        hull = body.add_geom(type=GEOM.mjGEOM_BOX, size=[hx * 0.95, hy * 0.95, hz], mass=rest, rgba=accent)
        vis.hide(hull)
        _hull(vis, body, hx, hy, hz, accent)

        for k in range(n):
            angle = 2 * math.pi * (k + 0.5) / n
            spin = 1 if k % 2 == 0 else -1  # counter-clockwise seen from above, then clockwise, round the ring
            c, si = math.cos(angle), math.sin(angle)
            tip = np.array([arm * c, arm * si, arm_z])
            root = np.array([0.3 * s * c, 0.3 * s * si, arm_z])
            capsule = body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                size=[arm_r, 0, 0],
                fromto=[*(root + (tip - root) * 0.15), *tip],
                mass=0.1 * p.mass / n,
                rgba=accent,
            )
            vis.hide(capsule)
            path = np.array(
                [root, root + (tip - root) * 0.5, tip - (tip - root) / np.linalg.norm(tip - root) * motor_r]
            )
            vis.add(body, "shell", accent, M.tube(path, [arm_r * 1.25, arm_r * 1.05, arm_r * 0.95], segments=16))
            mount_top = arm_z + arm_r + motor_h * 0.4
            vis.add(
                body, "dark", DARK, aero.stator_mount(motor_r * 1.08, mount_top - arm_z + arm_r), (*tip[:2], mount_top)
            )
            front = c > 0
            lamp = P.led(motor_r * 0.32)["led"].moved(rot=M.rot_x(math.pi))  # faces down
            vis.add(body, "led", LED_FRONT if front else LED_BACK, lamp, (tip[0], tip[1], arm_z - arm_r - 0.0005))

            rotor = body.add_body(name=f"rotor_{k + 1}", pos=[tip[0], tip[1], prop_z])
            rotor.add_site(name=f"rotor_{k + 1}", size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
            disc = rotor.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[prop / 2, 0.004, 0],
                pos=[0, 0, -motor_h * 0.25],
                mass=rotor_mass,
                rgba=[0.2, 0.2, 0.2, 0.3],
            )
            vis.hide(disc)
            for preset, mesh in aero.bell(motor_r, motor_h * 0.6).items():
                vis.add(
                    rotor, "metal" if preset == "metal" else "dark", BELL_COLOR if preset == "metal" else DARK, mesh
                )
            for preset, mesh in aero.propeller(prop, spin).items():
                if preset == "blades":
                    vis.add(rotor, "shell", PROP_COLOR, mesh, (0, 0, 0.002))
                else:  # the spinner nut: silver on counter-clockwise props, black on clockwise ones, as on real ones
                    vis.add(rotor, "metal", P.METAL if spin > 0 else DARK, mesh, (0, 0, 0.002))
            act = spec.add_actuator(name=f"rotor_{k + 1}", target=f"rotor_{k + 1}", trntype=mujoco.mjtTrn.mjTRN_SITE)
            act.set_to_motor()
            # The propeller turning one way twists the drone the other way: the reaction torque.
            act.gear = [0, 0, 1, 0, 0, -spin * TORQUE_PER_THRUST * prop]
            act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
            act.dynprm = [SPIN_UP[0] + SPIN_UP[1] * prop] + [0.0] * 9
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = [0.0, thrust]

        for side in (1, -1):  # landing skids
            bar = body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                size=[skid_r, 0, 0],
                fromto=[0.62 * s, side * skid_y, skid_z, -0.62 * s, side * skid_y, skid_z],
                mass=0.0175 * p.mass,
                rgba=DARK,
                friction=[1.0, 0.02, 0.002],
            )
            leg = body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                size=[skid_r * 0.9, 0, 0],
                fromto=[0.4 * s, side * 0.6 * hy, -hz * 0.8, 0.4 * s, side * skid_y, skid_z],
                mass=0.00375 * p.mass,
                rgba=DARK,
            )
            leg2 = body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                size=[skid_r * 0.9, 0, 0],
                fromto=[-0.4 * s, side * 0.6 * hy, -hz * 0.8, -0.4 * s, side * skid_y, skid_z],
                mass=0.00375 * p.mass,
                rgba=DARK,
            )
            for geom in (bar, leg, leg2):
                vis.hide(geom)
            mesh = aero.skid(1.24 * s, skid_y, 0.6 * hy, gear_drop - 0.2 * hz, skid_r)
            if side < 0:
                mesh = mesh.moved(scale=(1.0, -1.0, 1.0))
            vis.add(body, "dark", DARK, mesh, (0, 0, -hz * 0.8))

        gimbal_size = 0.3 * s
        cam = body.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[gimbal_size * 0.36, gimbal_size * 0.4, gimbal_size * 0.4],
            pos=[0.3 * s, 0, -hz - gimbal_size * 0.4],
            mass=0.04 * p.mass,
            rgba=DARK,
            contype=0,
            conaffinity=0,
        )
        vis.hide(cam)
        for preset, mesh in aero.gimbal(gimbal_size).items():
            color = {"dark": DARK, "shell": [0.2, 0.21, 0.24, 1.0], "glass": P.GLASS, "metal": P.METAL}[preset]
            vis.add(body, preset, color, mesh, (0.3 * s, 0, -hz))
        tilt = math.radians(20.0)  # the gimbal looks a little down, the way camera drones film
        body.add_camera(
            name="front_cam",
            pos=[0.3 * s + gimbal_size * 0.48, 0, -hz - gimbal_size * 0.55],
            xyaxes=[0, -1, 0, math.sin(tilt), 0, math.cos(tilt)],
            fovy=80,
        )
        imu(spec, body)
        vis.finish()
        lift = THRUST_TO_WEIGHT
        return BuiltRobot(
            spec=spec,
            spawn_height=-skid_z + skid_r + 0.002,
            floating=True,
            notes=[
                f"drone with {n} rotors, {prop:.2f} m propellers, {p.mass:g} kg; its rotors lift {lift:g} times its "
                "weight. Fly it with move: to=[x, y, z] flies there (z 0 lands), a list of points follows them"
            ],
        )


def _hull(vis: Visuals, body, hx: float, hy: float, hz: float, accent) -> None:
    """A rounded shell in the robot's color over a dark belly, a GPS dome on top, a status light at the back."""
    per = 6

    def ring(z: float, k: float) -> np.ndarray:
        return M.at_z(M.rounded_rect(hx * k, hy * k, 0.55 * min(hx, hy) * k, per), z)

    upper = [
        ring(hz * z, k) for z, k in ((0.0, 1.0), (0.3, 0.985), (0.6, 0.93), (0.82, 0.82), (0.95, 0.64), (1.0, 0.42))
    ]
    lower = [ring(-hz * z, k * 0.985) for z, k in ((0.0, 1.0), (0.3, 0.99), (0.6, 0.95), (0.85, 0.84), (1.0, 0.64))]
    vis.add(body, "shell", accent, M.loft(upper))
    vis.add(body, "dark", DARK, M.loft(lower[::-1]))
    dome_r = 0.22 * min(hx, hy)
    dome = M.lathe(
        [(0.0, 0.0), (dome_r, 0.0)]
        + [(dome_r * math.cos(a), dome_r * 0.5 * math.sin(a)) for a in np.linspace(0.2, 1.5, 5)],
        segments=28,
        sharp=(1,),
    )
    vis.add(body, "dark", [0.24, 0.25, 0.28, 1.0], dome, (-0.25 * hx, 0, hz * 0.97))
    vis.add(body, "led", LED_BACK, P.led(0.06 * hy)["led"], (-0.62 * hx, 0, hz * 0.84))
    visor = M.rounded_box(0.012 * hx + 0.001, hy * 0.42, hz * 0.18, hz * 0.1, segments=3)
    vis.add(body, "glass", P.GLASS, visor, (hx * 0.985, 0, hz * 0.25))
