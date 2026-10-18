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
    declare_camera,
    new_spec,
    servo_case,
    servo_joint,
)
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


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

        vis = Visuals(spec)
        base = spec.worldbody.add_body(name="base")
        pedestal = base.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[2.2 * radius + 0.02, p.base_height / 4],
            pos=[0, 0, p.base_height / 4],
            rgba=DARK,
            density=components.MATERIALS["aluminum"],
        )

        turret = base.add_body(name="turret", pos=[0, 0, p.base_height / 2])
        servo_joint(spec, turret, "yaw", [0, 0, 1], (-math.pi, math.pi), part)
        column = turret.add_geom(
            type=GEOM.mjGEOM_CYLINDER,
            size=[2 * radius + 0.01, p.base_height / 4],
            pos=[0, 0, p.base_height / 4],
            rgba=SHELL,
            density=density,
        )
        shoulder_servo = servo_case(turret, part, [0, 0, p.base_height / 2])
        for geom in (pedestal, column, shoulder_servo):
            vis.hide(geom)
        _dress_base(vis, base, turret, p.base_height, 2.2 * radius + 0.02, 2 * radius + 0.01)

        links = [("shoulder", upper), ("elbow", fore)]
        if p.dof >= 4:
            links.append(("wrist_pitch", hand))
        parent, z = turret, p.base_height / 2
        for i, (joint_name, length) in enumerate(links):
            body = parent.add_body(name=f"link{i + 1}", pos=[0, 0, z])
            limit = 2.6 if joint_name != "shoulder" else 1.9
            servo_joint(spec, body, joint_name, [0, 1, 0], (-limit, limit), part)
            drum = body.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[radius * 1.25, part.size[1] / 2 + 0.004],
                quat=[0.7071068, 0.7071068, 0, 0],
                rgba=DARK,
                mass=part.mass,
            )
            is_last = i == len(links) - 1
            rgba = accent if i == 0 else SHELL
            tube = body.add_geom(
                type=GEOM.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, length],
                size=[radius * (0.8 if is_last else 1.0), 0, 0],
                rgba=rgba,
                density=density,
            )
            vis.hide(drum)
            vis.hide(tube)
            link_r = radius * (0.8 if is_last else 1.0)
            _dress_link(vis, body, length, radius * 1.25, part.size[1] / 2 + 0.004, link_r, rgba, is_last)
            parent, z = body, length

        if p.dof >= 5:
            roll = parent.add_body(name="wrist", pos=[0, 0, z])
            if p.dof >= 6:
                servo_joint(spec, roll, "wrist_yaw", [1, 0, 0], (-1.6, 1.6), part)
            servo_joint(spec, roll, "wrist_roll", [0, 0, 1], (-math.pi, math.pi), part)
            wrist = roll.add_geom(
                type=GEOM.mjGEOM_CYLINDER,
                size=[radius * 1.1, 0.012],
                pos=[0, 0, 0.012],
                rgba=DARK,
                mass=part.mass,
            )
            vis.hide(wrist)
            _dress_wrist(vis, roll, radius * 1.1, 0.024)
            parent, z = roll, 0.024

        tool = parent.add_body(name="hand", pos=[0, 0, z])
        # Just in front of its lens (_dress_wrist_camera puts the lens 7.5 mm out, 1 mm deep): from the palm it
        # looked out through its own bezel, a black ring round the middle of every picture.
        tool.add_camera(name="wrist_cam", pos=[0.035, 0, 0.0095], xyaxes=[0, -1, 0, -1, 0, 0], fovy=70)
        declare_camera(spec, tool, "wrist_cam", 70)
        tcp_z = 0.0
        if p.gripper:
            tcp_z = add_gripper(spec, tool, p.gripper_opening, accent, vis)
        _dress_wrist_camera(vis, tool, p.gripper)
        vis.finish()
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


def add_gripper(spec: mujoco.MjSpec, tool: mujoco.MjsBody, opening: float, accent, vis: Visuals | None = None) -> float:
    """A parallel two-finger gripper on `tool`, opening along its y axis, fingers along +z.

    Adds the coupled finger joints, the `gripper` actuator (its control is one finger's travel)
    and returns the height of the point between the fingertips above the tool frame. Its looks go
    into `vis` (or into its own Visuals, finished here, when none is given).
    """
    own = vis is None
    vis = vis or Visuals(spec)
    palm_h = 0.02
    half_w = opening / 2 + 0.012
    palm = tool.add_geom(
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
        pad = finger.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[0.012, 0.005, finger_len / 2],
            pos=[0, 0, finger_len / 2],
            rgba=accent,
            friction=[1.5, 0.02, 0.002],
            condim=4,
            solref=[0.005, 1],
            mass=0.012,
        )
        vis.hide(pad)
        _dress_finger(vis, finger, sign, finger_len, accent)
    vis.hide(palm)
    _dress_palm(vis, tool, half_w, palm_h, travel)
    if own:
        vis.finish()
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


# ---- looks: visual-only meshes (no mass, no contacts) over the collision primitives above ----------


def _dress_base(vis: Visuals, base, turret, height: float, base_r: float, turret_r: float) -> None:
    """A dark pedestal with a bolted flange and a light ring, under a light turret column."""
    h = height / 2
    bevel = min(0.005, h * 0.2)
    vis.add(base, "dark", DARK, M.cylinder(base_r, h, segments=48, bevel=bevel, z0=0.0))
    vis.add(base, "metal", P.METAL, M.cylinder(base_r * 1.015, h * 0.1, segments=48, bevel=h * 0.03, z0=0.0))
    bolts = P.bolt_circle(base_r * 0.86, 6, base_r * 0.055, 0.0025, start=math.pi / 6)
    vis.add(base, "metal", P.METAL, bolts, (0, 0, h - 0.0005))
    ring = M.torus(turret_r * 1.06, 0.0018, segments=48, sides=8)
    vis.add(base, "led", P.LED_CYAN, ring, (0, 0, h + 0.0005))
    column = M.cylinder(turret_r, h, segments=48, bevel=min(0.006, h * 0.25), z0=0.0)
    vis.add(turret, "shell", SHELL, column)
    seam = M.cylinder(turret_r * 1.004, 0.0012, segments=48, z0=h * 0.32)
    vis.add(turret, "dark", DARK, seam)


def _dress_link(
    vis: Visuals, body, length: float, drum_r: float, drum_half: float, link_r: float, rgba, is_last: bool
) -> None:
    """A joint drum on the pitch axis (y) and a tapered rounded limb up to the next joint."""
    half_y = link_r * 1.02
    drum_half = max(drum_half, half_y + 0.003)
    for preset, mesh in P.drum(drum_r, drum_half, drum_r * 0.72).items():
        vis.add(body, preset, DARK if preset == "dark" else P.METAL, mesh, (0, 0, 0), M.Z_TO_Y)
    if is_last:  # the short wrist link: a straight round neck out of the drum
        start = end = (link_r * 1.02, half_y)
    else:
        start, end = (drum_r * 0.98, half_y), (link_r * 0.95, half_y * 0.95)
    limb = M.link(length, start, end, roundness=0.85, rounding=(1.0, 1.0), stations=8)
    vis.add(body, "shell", rgba, limb)


def _dress_wrist(vis: Visuals, roll, r: float, height: float) -> None:
    """The wrist roll unit: a dark drum ending in a metal tool flange with bolts."""
    vis.add(roll, "dark", DARK, M.cylinder(r, height * 0.8, segments=36, bevel=0.002, z0=0.0))
    flange = M.cylinder(r * 0.92, height * 0.2, segments=36, bevel=0.0008, z0=height * 0.8)
    vis.add(roll, "metal", P.METAL, flange)
    vis.add(roll, "dark", DARK, P.bolt_circle(r * 0.7, 4, r * 0.08, 0.0012, math.pi / 4), (0, 0, height - 0.0003))


def _dress_wrist_camera(vis: Visuals, tool, gripper: bool) -> None:
    """The wrist camera beside the palm, looking along the fingers (+z), centered on `wrist_cam`."""
    x = 0.035
    inner = 0.018 if gripper else 0.012
    hx = (x + 0.008 - inner) / 2
    housing = M.rounded_box(hx, 0.012, 0.0075, 0.0025, segments=2, corner=0.004)
    vis.add(tool, "dark", DARK, housing, (inner + hx, 0, 0.0))
    for preset, mesh in P.lens(0.0055, 0.0025).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(tool, preset, color, mesh, (x, 0, 0.0075))
    vis.add(tool, "led", P.LED_GREEN, P.led(0.0012, 0.0008)["led"], (x, 0.009, 0.0075))


def _dress_palm(vis: Visuals, tool, half_w: float, palm_h: float, travel: float) -> None:
    """The gripper body: a dark housing with light side covers and the metal rails the fingers ride on."""
    body = M.rounded_box(0.018, half_w, palm_h / 2, 0.004, segments=3, corner=0.005)
    vis.add(tool, "dark", DARK, body, (0, 0, palm_h / 2))
    for side in (1, -1):
        cover = M.rounded_box(0.0012, half_w * 0.86, palm_h * 0.32, 0.001, segments=2, corner=0.004)
        vis.add(tool, "shell", SHELL, cover, (side * 0.0185, 0, palm_h * 0.5))
    for x in (0.009, -0.009):
        rail = M.cylinder(0.0022, 2 * half_w * 0.92, segments=12).moved(rot=M.Z_TO_Y)
        vis.add(tool, "metal", P.METAL, rail, (x, 0, palm_h + 0.0008))


def _dress_finger(vis: Visuals, finger, sign: int, length: float, accent) -> None:
    """An accent finger with a rubber grip pad flush with its inner (closing) face."""
    t = 0.005  # half thickness of the collision box along y
    pad_t = 0.0009
    body_t = t - pad_t
    body = M.rounded_box(0.012, body_t, length / 2, 0.0025, segments=2, corner=0.003)
    vis.add(finger, "shell", accent, body, (0, sign * pad_t, length / 2))
    pad = M.rounded_box(0.0112, pad_t, length * 0.4, 0.0008, segments=2, corner=0.002)
    vis.add(finger, "rubber", P.RUBBER, pad, (0, -sign * (t - pad_t), length * 0.56))
    screw = P.bolt(0.0016, 0.0008).moved(rot=M.rot_x(-sign * math.pi / 2))
    vis.add(finger, "metal", P.METAL, screw, (0, sign * t, length * 0.18))
