"""Desktop robot arm (3 to 6 joints) with an optional parallel gripper."""

from __future__ import annotations

import math
from typing import Literal

import mujoco
from pydantic import Field

from robotforge.assets.bridge import Dress
from robotforge.design import components, end_effectors
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
    end_effector: Literal["gripper", "suction", "magnet"] = Field(
        "gripper",
        description="what the hand holds with: two fingers, a suction cup (flat things) or an electromagnet (steel)",
    )
    style: str | None = Field(
        None, description="how it is drawn, not how it moves: cobot (default), lab, hobby, rugged (see list_templates)"
    )


class Arm(Template):
    name = "arm"
    summary = "Desktop robot arm with 3-6 servo joints and an optional parallel gripper; base fixed to the floor."
    Params = ArmParams

    def build(self, p: ArmParams) -> BuiltRobot:
        part = components.actuator(p.servo)
        density = components.MATERIALS[p.material]
        accent = components.color(p.color)
        dress = Dress(p.style, p.material, color=accent)
        spec = new_spec()
        radius = max(0.012, 0.035 * p.reach)
        upper, fore = 0.45 * p.reach, 0.35 * p.reach
        # What the hand holds with: an end effector chosen by name wins; "gripper" is the fingers, unless gripper
        # is false (an arm with a bare flange, as before end effectors had a name).
        holder = p.end_effector if p.end_effector != "gripper" else ("gripper" if p.gripper else None)
        tool_length = end_effectors.TOOLS[holder].length if holder in end_effectors.TOOLS else 0.05
        hand = 0.2 * p.reach - (tool_length if holder else 0.0)

        vis = Visuals(spec)
        dress.attach(vis)
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
        _dress_base(vis, base, turret, p.base_height, 2.2 * radius + 0.02, 2 * radius + 0.01, dress)

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
            _dress_link(vis, body, length, radius * 1.25, part.size[1] / 2 + 0.004, link_r, i == 0, is_last, dress)
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
            _dress_wrist(vis, roll, radius * 1.1, 0.024, dress)
            parent, z = roll, 0.024

        tool = parent.add_body(name="hand", pos=[0, 0, z])
        # Just in front of its lens (_dress_wrist_camera puts the lens 7.5 mm out, 1 mm deep): from the palm it
        # looked out through its own bezel, a black ring round the middle of every picture.
        tool.add_camera(name="wrist_cam", pos=[0.035, 0, 0.0095], xyaxes=[0, -1, 0, -1, 0, 0], fovy=70)
        declare_camera(spec, tool, "wrist_cam", 70)
        tcp_z = 0.0
        if holder == "gripper":
            tcp_z = add_gripper(spec, tool, p.gripper_opening, accent, vis, dress)
        elif holder is not None:  # a suction cup or an electromagnet where the fingers would be
            tcp_z = end_effectors.add_tool(spec, tool, holder, vis, holder)
        _dress_wrist_camera(vis, tool, holder == "gripper")
        vis.finish()
        tool.add_site(name="tcp", pos=[0, 0, tcp_z], size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])
        # The shoulder's drum sits in the pedestal it turns on (6 mm deep, folded down): their contact held the base
        # joint still against the servo's full torque under MuJoCo's no-slip pass (behaviors/fetch.py).
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = "base", "link1"
        spec.add_sensor(
            name="tcp_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="tcp"
        )

        # A "ready" pose: elbow bent, tool pointing down in front of the base.
        home = {"shoulder": 0.35, "elbow": 1.45}
        if p.dof >= 4:
            home["wrist_pitch"] = math.pi - home["shoulder"] - home["elbow"]
        ctrl = dict(home)
        if holder == "gripper":
            ctrl["gripper"] = p.gripper_opening / 2
        elif holder is not None:
            ctrl["gripper"] = 0.0  # off: it holds once told to close
        notes = [f"{p.dof}-joint arm, reach {p.reach:.2f} m, {part.name} servos ({part.stall_torque} N*m)"]
        if holder in end_effectors.TOOLS:
            info = end_effectors.TOOLS[holder]
            notes.append(f"{info.title} on the hand: holds {info.holds}; gripper close turns it on, open lets go")
        return BuiltRobot(spec=spec, home_qpos=home, home_ctrl=ctrl, spawn_height=0.0, floating=False, notes=notes)


def add_gripper(
    spec: mujoco.MjSpec,
    tool: mujoco.MjsBody,
    opening: float,
    accent,
    vis: Visuals | None = None,
    dress: Dress | None = None,
) -> float:
    """A parallel two-finger gripper on `tool`, opening along its y axis, fingers along +z.

    Adds the coupled finger joints, the `gripper` actuator (its control is one finger's travel)
    and returns the height of the point between the fingertips above the tool frame. Its looks go
    into `vis` (or into its own Visuals, finished here, when none is given).
    """
    own = vis is None
    dress = dress or Dress(color=accent)
    if vis is None:
        vis = Visuals(spec)
        dress.attach(vis)
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
        _dress_finger(vis, finger, sign, finger_len, dress)
    vis.hide(palm)
    _dress_palm(vis, tool, half_w, palm_h, dress)
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


def _dress_base(vis: Visuals, base, turret, height: float, base_r: float, turret_r: float, dress: Dress) -> None:
    """A bolted mounting plate and column under a turret joint (a rotary housing turning about the vertical)."""
    h = height / 2
    column = min(0.95, max(0.3, 0.97 * turret_r / base_r))
    dress.draw(vis, base, "pedestal", radius=base_r, height=h, column=column)
    dress.draw(vis, turret, "joint_actuator", pos=(0.0, 0.0, h / 2), radius=turret_r, length=h, gland=False)


def _dress_link(
    vis: Visuals,
    body,
    length: float,
    drum_r: float,
    drum_half: float,
    link_r: float,
    first: bool,
    is_last: bool,
    dress: Dress,
) -> None:
    """A joint housing on the pitch axis (y) and a tapered shell up to the next joint."""
    half_y = max(drum_half, link_r * 1.02 + 0.003)
    palette = dress.colored if first else dress.shell
    dress.draw(vis, body, "joint_actuator", rot=M.Z_TO_Y, radius=1.18 * drum_r, length=2 * half_y, gland_angle=180.0)
    if length < 0.004:  # a hand shorter than its own tool leaves no link to draw
        return
    root = link_r * (0.95 if is_last else 1.0)
    width = min(2.2, max(0.5, 0.92 * half_y / root))
    dress.draw(
        vis,
        body,
        "link_shell",
        palette,
        length=length,
        radius=root,
        width=width,
        tip=1.0 if is_last else 0.86,
        waist=0.0 if is_last else 0.07,
    )


def _dress_wrist(vis: Visuals, roll, r: float, height: float, dress: Dress) -> None:
    """The wrist roll unit: a small rotary housing that ends in the tool flange."""
    dress.draw(vis, roll, "joint_actuator", pos=(0.0, 0.0, height / 2), radius=r, length=height, gland=False, bolts=4)


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


def _dress_palm(vis: Visuals, tool, half_w: float, palm_h: float, dress: Dress) -> None:
    """The gripper body: a housing with side covers and the rails the fingers ride on, on a bolted flange."""
    dress.draw(vis, tool, "gripper_body", width=2 * half_w, depth=0.036, height=palm_h)


def _dress_finger(vis: Visuals, finger, sign: int, length: float, dress: Dress) -> None:
    """A finger in the accent color with a ridged rubber grip pad on its inner (closing) face."""
    dress.draw(
        vis,
        finger,
        "grip_pad",
        dress.colored,
        rot=M.rot_z(-sign * math.pi / 2),
        length=length,
        width=0.024,
        thickness=0.01,
    )
