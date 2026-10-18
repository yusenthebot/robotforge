"""An underwater ROV (remotely operated vehicle), laid out like the BlueROV2 that many labs use.

Six thrusters: four in the corners turned 45 degrees (they push forward, sideways and turn it), two vertical
ones on the sides (up, down and roll). Like a real ROV it is trimmed a little lighter than the water it
displaces, so it drifts up to the surface when it loses power; buoyancy foam on top and the battery underneath
keep it level. Its depth and heading are held by the pilot in behaviors/marine.py.
"""

from __future__ import annotations

import math

import mujoco
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.base import DARK, GEOM, SHELL, BuiltRobot, Template, TemplateParams, imu, new_spec
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

WATER_DENSITY = 1000.0
TRIM = 0.985  # its mass as a share of the water it displaces: 1.5 percent buoyant, so it floats up unpowered
SPIN_UP = 0.1  # s
FRAME = [0.07, 0.07, 0.08, 1.0]
TUBE = [0.78, 0.86, 0.92, 1.0]


class RovParams(TemplateParams):
    color: str | list[float] = "yellow"
    length: float = Field(0.46, ge=0.25, le=2.0, description="frame length in meters (width and height scale)")
    thrust: float | None = Field(None, gt=0, le=1000, description="newtons per thruster (default from the size)")


class Rov(Template):
    name = "rov"
    summary = (
        "Underwater ROV: six thrusters (four vectored, two vertical), buoyancy foam, front camera, IMU and a "
        "depth sensor (observe reads it in meters under the surface); near neutral buoyancy. move drives it, "
        "move(to=[x, y, z]) dives to a point (z above the floor), holding depth and heading."
    )
    Params = RovParams

    def build(self, p: RovParams) -> BuiltRobot:
        L = p.length
        W, Hh = 0.74 * L, 0.55 * L  # width and height of the frame
        accent = components.color(p.color)
        spec = new_spec()
        vis = Visuals(spec)
        body = spec.worldbody.add_body(name="frame")
        body.add_freejoint(name="root")

        # What displaces water, low to high: skids, battery and electronics tubes, side plates, buoyancy foam.
        r_main, r_batt = 0.11 * L, 0.075 * L
        parts = [  # (type, size, pos, density) -- densities make it bottom-heavy before the trim below
            ("battery", GEOM.mjGEOM_CYLINDER, [r_batt, 0.36 * L], [0.0, 0.0, -0.3 * Hh], 2600.0, "y"),
            ("electronics", GEOM.mjGEOM_CYLINDER, [r_main, 0.4 * L], [0.04 * L, 0.0, -0.02 * Hh], 900.0, "x"),
            ("left_plate", GEOM.mjGEOM_BOX, [0.5 * L, 0.012 * L, 0.48 * Hh], [0, 0.5 * W, 0], 1400.0, None),
            ("right_plate", GEOM.mjGEOM_BOX, [0.5 * L, 0.012 * L, 0.48 * Hh], [0, -0.5 * W, 0], 1400.0, None),
            ("foam", GEOM.mjGEOM_BOX, [0.42 * L, 0.42 * W, 0.11 * Hh], [-0.04 * L, 0, 0.36 * Hh], 250.0, None),
            ("left_skid", GEOM.mjGEOM_BOX, [0.48 * L, 0.03 * L, 0.03 * L], [0, 0.42 * W, -0.47 * Hh], 950.0, None),
            ("right_skid", GEOM.mjGEOM_BOX, [0.48 * L, 0.03 * L, 0.03 * L], [0, -0.42 * W, -0.47 * Hh], 950.0, None),
        ]
        geoms, volume = [], 0.0
        for name, kind, size, pos, density, axis in parts:
            quat = [1, 0, 0, 0]
            if axis == "x":
                quat = [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]
            elif axis == "y":
                quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
            v = _volume(kind, size)
            volume += v
            geoms.append((name, kind, size, pos, quat, density * v))
        raw = sum(g[5] for g in geoms)
        mass = TRIM * WATER_DENSITY * volume
        for name, kind, size, pos, quat, part_mass in geoms:
            geom = body.add_geom(name=name, type=kind, size=size, pos=pos, quat=quat, mass=part_mass, rgba=accent)
            vis.hide(geom)
        # Lead ballast on the skids makes up the trim, placed fore or aft so the center of mass sits right under the
        # center of buoyancy: level, with no pitch for the thrusters to fight.
        buoyancy_x = sum(_volume(k, size) * pos[0] for _, k, size, pos, _, _ in geoms) / volume
        ballast = mass - raw
        ballast_x = (mass * buoyancy_x - sum(m * pos[0] for _, _, _, pos, _, m in geoms)) / ballast
        body.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[0.06 * L, 0.3 * W, 0.02 * L],
            pos=[float(max(-0.4 * L, min(0.4 * L, ballast_x))), 0, -0.44 * Hh],
            mass=ballast,
            contype=0,
            conaffinity=0,
            group=3,
        )
        thrust = p.thrust if p.thrust is not None else round(2.5 * mass, 1)
        prop = 0.09 * L
        # Four vectored thrusters at the corners, each pushing along its axis 45 degrees off forward (forward-out at
        # the front, backward-out at the back, the BlueROV2 layout), and two vertical ones on the sides.
        corners = [
            ("front_left", 0.36 * L, 0.33 * W, 45.0),
            ("front_right", 0.36 * L, -0.33 * W, -45.0),
            ("back_left", -0.36 * L, 0.33 * W, 135.0),
            ("back_right", -0.36 * L, -0.33 * W, -135.0),
        ]
        for name, x, y, yaw in corners:
            a = math.radians(yaw)
            site_quat = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
            body.add_site(name=f"{name}_thruster", pos=[x, y, -0.12 * Hh], quat=site_quat, size=[prop, 0, 0])
            _thruster(spec, f"{name}_thruster", thrust, [1, 0, 0])
        for name, y in (("left_vertical", 0.36 * W), ("right_vertical", -0.36 * W)):
            body.add_site(name=f"{name}_thruster", pos=[0.02 * L, y, 0.14 * Hh], size=[prop, 0, 0])
            _thruster(spec, f"{name}_thruster", thrust, [0, 0, 1])
        body.add_camera(
            name="front_cam", pos=[0.04 * L + 0.4 * L + 0.02 * L, 0, -0.02 * Hh], xyaxes=[0, -1, 0, 0, 0, 1], fovy=80
        )
        # A pressure sensor: MuJoCo has no depth sensor, so this measures where the site is and observe turns that
        # into meters under the surface (sim/water.py, Hydro.depth_sensors).
        body.add_site(name="depth", pos=[-0.3 * L, 0, -0.1 * Hh], size=[0.01, 0, 0])
        spec.add_sensor(
            name="depth", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="depth"
        )
        imu(spec, body)
        _dress(vis, body, L, W, Hh, accent, r_main, r_batt, prop, corners)
        vis.finish()
        return BuiltRobot(
            spec=spec,
            spawn_height=0.5 * Hh + 0.03 * L + 0.002,
            floating=True,
            notes=[
                f"ROV, {L:.2f} m, {mass:.1f} kg, 1.5 percent buoyant, six thrusters of {thrust:g} N (-1 to 1); "
                "it dives in a world with water: move(to=[x, y, z]) goes to a point under water and holds it"
            ],
        )


def _volume(kind: int, size: list[float]) -> float:
    if kind == GEOM.mjGEOM_BOX:
        return 8 * size[0] * size[1] * size[2]
    return math.pi * size[0] ** 2 * 2 * size[1]


def _thruster(spec: mujoco.MjSpec, site: str, thrust: float, along: list[float]) -> None:
    act = spec.add_actuator(name=site, target=site, trntype=mujoco.mjtTrn.mjTRN_SITE)
    act.gear = [*along, 0, 0, 0]
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-1, 1]
    act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
    act.dynprm[0] = SPIN_UP
    act.gainprm[0] = thrust


# ---- looks ------------------------------------------------------------------------------------------------


def _dress(vis: Visuals, body, L, W, Hh, accent, r_main, r_batt, prop, corners) -> None:
    # Side frames in black plastic, cut out to a ring with a cross brace (the collision plate under them is whole,
    # but what shows is a frame you can see the tubes and thrusters through).
    bar = 0.055 * L
    hx, hz = 0.5 * L, 0.48 * Hh
    for sign in (1, -1):
        y = sign * 0.5 * W
        parts = [
            M.rounded_box(hx, 0.012 * L, bar / 2, 0.008 * L, segments=2).moved((0, y, hz - bar / 2)),
            M.rounded_box(hx, 0.012 * L, bar / 2, 0.008 * L, segments=2).moved((0, y, -hz + bar / 2)),
            M.rounded_box(bar / 2, 0.012 * L, hz, 0.008 * L, segments=2).moved((hx - bar / 2, y, 0)),
            M.rounded_box(bar / 2, 0.012 * L, hz, 0.008 * L, segments=2).moved((-hx + bar / 2, y, 0)),
            M.rounded_box(bar / 2, 0.012 * L, hz, 0.008 * L, segments=2).moved((0.05 * L, y, 0)),
        ]
        vis.add(body, "dark", FRAME, M.merge(parts))
        handle = M.rounded_box(0.12 * L, 0.014 * L, 0.03 * L, 0.01 * L, segments=2)
        vis.add(body, "dark", DARK, handle, (-0.05 * L, y, hz + 0.02 * L))
    # Buoyancy foam on top, in the robot's color, with rounded edges.
    foam = M.rounded_box(0.42 * L, 0.42 * W, 0.11 * Hh, 0.04 * L, segments=3, corner=0.06 * L)
    vis.add(body, "shell", accent, foam, (-0.04 * L, 0, 0.36 * Hh))
    # The clear electronics tube with a dome at the front (the camera looks out of it) and a black end cap.
    tube = M.cylinder(r_main, 0.8 * L, segments=40, bevel=0.004)
    vis.add(body, "glass", TUBE, tube.moved((0, 0, 0), M.Z_TO_X), (0.04 * L, 0, -0.02 * Hh))
    dome = M.sphere(r_main * 0.98, rings=12, segments=40).moved(scale=(0.6, 1, 1))
    vis.add(body, "glass", TUBE, dome, (0.44 * L, 0, -0.02 * Hh))
    cap = M.cylinder(r_main * 1.05, 0.04 * L, segments=40, bevel=0.003)
    vis.add(body, "dark", DARK, cap.moved((0, 0, 0), M.Z_TO_X), (-0.37 * L, 0, -0.02 * Hh))
    for preset, mesh in P.lens(0.35 * r_main, 0.25 * r_main).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(body, preset, color, mesh, (0.4 * L, 0, -0.02 * Hh), M.Z_TO_X)
    # The battery tube underneath, across the frame.
    batt = M.cylinder(r_batt, 0.72 * L, segments=32, bevel=0.003)
    vis.add(body, "aluminum", [0.32, 0.36, 0.42, 1.0], batt.moved((0, 0, 0), M.Z_TO_Y), (0, 0, -0.3 * Hh))
    # Lights either side of the dome.
    for sign in (1, -1):
        lamp = M.cylinder(0.035 * L, 0.03 * L, segments=24, bevel=0.002)
        vis.add(body, "dark", DARK, lamp.moved((0, 0, 0), M.Z_TO_X), (0.43 * L, sign * 0.27 * W, 0.12 * Hh))
        vis.add(
            body,
            "led",
            P.LED_WHITE,
            M.cylinder(0.027 * L, 0.004, segments=20),
            (0.447 * L, sign * 0.27 * W, 0.12 * Hh),
            M.Z_TO_X,
        )
    # Skids.
    for sign in (1, -1):
        skid = M.rounded_box(0.48 * L, 0.03 * L, 0.03 * L, 0.012 * L, segments=2)
        vis.add(body, "dark", FRAME, skid, (0, sign * 0.42 * W, -0.47 * Hh))

    # Thrusters: a motor in a duct, black, with a light propeller.
    def thruster(pos, rot):
        motor = M.lathe([(0.0, -0.9), (0.45, -0.8), (0.5, 0.0), (0.45, 0.8), (0.0, 0.9)], segments=20).moved(scale=prop)
        duct = M.lathe(
            [(1.05, -0.55), (1.22, -0.5), (1.25, 0.0), (1.2, 0.55), (1.05, 0.5), (1.02, 0.0)], segments=32
        ).moved(scale=prop)
        blades = M.merge(
            M.rounded_box(0.98 * prop, 0.12 * prop, 0.05 * prop, 0.02 * prop, segments=1).moved(
                (0, 0, 0), M.rot_z(k * 2 * math.pi / 3) @ M.rot_x(0.5)
            )
            for k in range(3)
        )
        vis.add(body, "dark", DARK, motor, pos, rot)
        vis.add(body, "dark", DARK, duct, pos, rot)
        vis.add(body, "shell", SHELL, blades, pos, rot)

    for _, x, y, yaw in corners:
        thruster((x, y, -0.12 * Hh), M.rot_z(math.radians(yaw)) @ M.Z_TO_X)
    for y in (0.36 * W, -0.36 * W):
        thruster((0.02 * L, y, 0.14 * Hh), None)
