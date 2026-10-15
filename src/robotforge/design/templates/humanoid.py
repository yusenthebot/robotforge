"""Small two-legged servo humanoid (6 servos per leg, 3 per arm, 2 in the neck).

In the spirit of Robotis OP3 and hobby humanoids. Joint signs are mirrored on
the right side so the same number makes a mirror-image pose: positive pitch
lifts a limb forward, positive roll moves it outward, positive knee and elbow
bend them. The torso is the root body (free joint "root").
"""

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

SIDES = (("left", 1), ("right", -1))
LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
HEAD_JOINTS = ("neck_yaw", "neck_pitch")
HOME_BEND = 0.35  # rad on hip and ankle pitch (knee bends twice as much): knees soft, ready to step
HOME_ARM = {"shoulder_pitch": 0.1, "shoulder_roll": 0.12, "elbow": 0.35}
EYE = [0.55, 0.85, 1.0, 1.0]
MAX_WALKING_HEIGHT = 0.75  # m; taller (heavy-legged) humanoids fell under quick command changes
REFERENCE_HEIGHT = 0.45  # m; every length below is given at this height and scales with it
_Y_CYL = [0.7071068, 0.7071068, 0, 0]  # cylinder axis along y


class HumanoidParams(TemplateParams):
    height: float = Field(
        0.45,
        ge=0.3,
        le=1.0,
        description="standing height in meters; 0.3-0.6 walks on sts3215 servos, taller needs stronger",
    )
    servo: str = Field("sts3215", description="joint actuator from the parts catalog")
    material: Literal["pla", "aluminum", "carbon"] = "pla"
    arms: bool = Field(True, description="add two 3-joint arms")
    head: bool = Field(True, description="add a 2-joint head with a camera")


class Dims:
    """Body proportions for one height (meters)."""

    def __init__(self, height: float):
        s = height / REFERENCE_HEIGHT
        self.scale = s
        self.foot = (0.10 * s, 0.056 * s, 0.012 * s)  # length, width, plate thickness
        self.ankle = 0.032 * s  # ankle axes above the sole
        self.shin = 0.095 * s
        self.thigh = 0.095 * s
        self.hip_drop = 0.026 * s  # hip axes below the torso box
        self.hip_y = 0.042 * s  # hip axes either side of the middle
        self.torso = (0.075 * s, 0.12 * s, 0.13 * s)  # depth, width, height
        self.leg_r = 0.016 * s
        self.arm_r = 0.013 * s
        self.upper_arm = 0.075 * s
        self.forearm = 0.065 * s
        self.head = (0.038 * s, 0.042 * s, 0.034 * s)  # ellipsoid radii

    def stand_height(self, bend: float = HOME_BEND) -> float:
        """Torso origin above the soles with hips, knees and ankles at `bend`."""
        return self.ankle + (self.thigh + self.shin) * math.cos(bend) + self.hip_drop + self.torso[2] / 2


class Humanoid(Template):
    name = "humanoid"
    summary = (
        "Two-legged humanoid with 20 servos (6 per leg, 3 per arm, 2 in the neck), flat feet with touch "
        "sensors, torso IMU and a head camera. Can stand, balance, walk slowly and strike poses."
    )
    Params = HumanoidParams

    def build(self, p: HumanoidParams) -> BuiltRobot:
        part = components.actuator(p.servo)
        density = components.MATERIALS[p.material]
        accent = components.color(p.color)
        d = Dims(p.height)
        spec = new_spec()
        depth, width, tall = d.torso

        vis = Visuals(spec)
        torso = spec.worldbody.add_body(name="torso")
        torso.add_freejoint(name="root")
        # Printed torsos are hollow shells; ~25 % of a solid block is a fair mass estimate.
        hull = torso.add_geom(
            type=GEOM.mjGEOM_BOX, size=[depth / 2, width / 2, tall / 2], rgba=accent, density=density * 0.25
        )
        chest = torso.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[0.004 * d.scale, width / 2 * 0.72, tall / 2 * 0.62],
            pos=[depth / 2 + 0.002 * d.scale, 0, tall * 0.06],
            rgba=SHELL,
            density=density,
        )
        pelvis_h = 0.022 * d.scale
        pelvis = torso.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[depth / 2 * 0.92, width / 2 * 0.94, pelvis_h / 2],
            pos=[0, 0, -tall / 2 + pelvis_h / 2 - 0.004 * d.scale],
            rgba=DARK,
            density=density,
        )
        # Battery and controller board ride inside the torso (a 2S pack plus a small board at 0.45 m).
        battery = _hidden_mass(torso, 0.2 * d.scale**3, [0, 0, 0], [depth * 0.3, width * 0.3, tall * 0.25])
        imu(spec, torso)
        for geom in (hull, chest, pelvis, battery):
            vis.hide(geom)
        _dress_torso(vis, torso, d, accent, pelvis_h)

        home: dict[str, float] = {}
        for side, sy in SIDES:
            home.update(self._leg(spec, vis, torso, side, sy, d, part, density, accent))
            if p.arms:
                home.update(self._arm(spec, vis, torso, side, sy, d, part, density, accent))
        if p.head:
            self._head(spec, vis, torso, d, part, density)

        model = spec.compile()  # before the looks are added: the notes only need masses
        vis.finish()
        mass = float(model.body_subtreemass[1])
        notes = [
            f"humanoid, {p.height:.2f} m tall, {mass:.2f} kg, {model.nu} {part.name} servos ({part.stall_torque} N*m)"
        ]
        notes += _strength_warnings(model, part, mass, d)
        return BuiltRobot(
            spec=spec,
            home_qpos=home,
            home_ctrl=dict(home),
            spawn_height=d.stand_height() + 0.002,
            floating=True,
            notes=notes,
        )

    @staticmethod
    def _leg(spec, vis, torso, side: str, sy: int, d: Dims, part, density: float, accent) -> dict[str, float]:
        depth, width, tall = d.torso
        s = d.scale
        # The hip yaw servo sits upright inside the pelvis, right above its leg.
        vis.hide(servo_case(torso, part, [0, sy * d.hip_y, -tall / 2 + part.size[2] / 2 + 0.004 * s]))
        _visual_only(torso)

        # Hip yaw, roll and pitch axes cross at one point, so the whole thigh hangs off a 3-axis hip.
        thigh = torso.add_body(name=f"{side}_thigh", pos=[0, sy * d.hip_y, -tall / 2 - d.hip_drop])
        servo_joint(spec, thigh, f"{side}_hip_yaw", [0, 0, sy], (-0.6, 0.6), part)
        servo_joint(spec, thigh, f"{side}_hip_roll", [sy, 0, 0], (-0.4, 0.6), part)
        servo_joint(spec, thigh, f"{side}_hip_pitch", [0, -1, 0], (-1.0, 1.9), part)
        hip_cap = _joint_cap(thigh, part, 2, d.leg_r * 1.3, part.size[1] / 2 + 0.006 * s, _Y_CYL)
        thigh_tube = thigh.add_geom(
            type=GEOM.mjGEOM_CAPSULE,
            fromto=[0, 0, -0.012 * s, 0, 0, -d.thigh + 0.012 * s],
            size=[d.leg_r, 0, 0],
            rgba=SHELL,
            density=density,
        )

        shin = thigh.add_body(name=f"{side}_shin", pos=[0, 0, -d.thigh])
        servo_joint(spec, shin, f"{side}_knee", [0, 1, 0], (0.0, 2.4), part)
        knee_cap = _joint_cap(shin, part, 1, d.leg_r * 1.2, part.size[1] / 2 + 0.004 * s, _Y_CYL)
        shin_tube = shin.add_geom(
            type=GEOM.mjGEOM_CAPSULE,
            fromto=[0, 0, -0.012 * s, 0, 0, -d.shin + 0.016 * s],
            size=[d.leg_r * 0.9, 0, 0],
            rgba=accent,
            density=density,
        )

        # Ankle pitch and roll also cross at one point, so the sole stays flat with simple math.
        foot = shin.add_body(name=f"{side}_foot", pos=[0, 0, -d.shin])
        servo_joint(spec, foot, f"{side}_ankle_pitch", [0, -1, 0], (-0.9, 1.2), part)
        servo_joint(spec, foot, f"{side}_ankle_roll", [sy, 0, 0], (-0.5, 0.5), part)
        case_z = max(-d.ankle + d.foot[2] + part.size[2] / 2 + 0.001 * s, -part.size[2] / 2)
        ankle_case = servo_case(foot, part, [0, 0, case_z])
        _visual_only(foot)
        # The ankle roll servo nests inside the pitch servo's housing (hidden, but it weighs as much).
        ankle_mass = _hidden_mass(foot, part.mass, [0, 0, case_z], [v * 0.8 for v in part.size])
        length, wide, plate = d.foot
        sole = [0.006 * s, sy * 0.003 * s, -d.ankle]  # slightly toe-heavy and outboard, like a real foot
        sole_geom = foot.add_geom(
            name=f"{side}_sole",
            type=GEOM.mjGEOM_BOX,
            size=[length / 2, wide / 2, plate / 2],
            pos=[sole[0], sole[1], sole[2] + plate / 2],
            rgba=DARK,
            friction=[1.0, 0.02, 0.002],
            density=components.MATERIALS["pla"],
        )
        top_plate = foot.add_geom(
            type=GEOM.mjGEOM_BOX,
            size=[length / 2 * 0.86, wide / 2 * 0.8, 0.002 * s],
            pos=[sole[0] + 0.002 * s, sole[1], sole[2] + plate + 0.0015 * s],
            rgba=accent,
            mass=0.001,
            contype=0,
            conaffinity=0,
        )
        foot.add_site(
            name=f"{side}_foot",
            type=GEOM.mjGEOM_BOX,
            pos=sole,
            size=[length / 2 + 0.002, wide / 2 + 0.002, 0.004 * s],
            rgba=[0, 0, 0, 0],
        )
        for geom in (hip_cap, thigh_tube, knee_cap, shin_tube, ankle_case, ankle_mass, sole_geom, top_plate):
            vis.hide(geom)
        _dress_leg(vis, thigh, shin, foot, sy, d, part, case_z, sole, accent)
        spec.add_sensor(
            name=f"{side}_foot_touch",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{side}_foot",
        )
        return {
            f"{side}_hip_pitch": HOME_BEND,
            f"{side}_knee": 2 * HOME_BEND,
            f"{side}_ankle_pitch": HOME_BEND,
        }

    @staticmethod
    def _arm(spec, vis, torso, side: str, sy: int, d: Dims, part, density: float, accent) -> dict[str, float]:
        depth, width, tall = d.torso
        s = d.scale
        shoulder_y = width / 2 + d.arm_r + 0.008 * s
        shoulder_z = tall / 2 - 0.022 * s
        # Shoulder pitch servo inside the torso side, shaft pointing out.
        vis.hide(
            servo_case(
                torso,
                part,
                [0, sy * (width / 2 - part.size[1] / 2 - 0.002 * s), shoulder_z],
                quat=[0.7071068, 0, 0, 0.7071068],
            )
        )
        _visual_only(torso)
        upper = torso.add_body(name=f"{side}_upper_arm", pos=[0, sy * shoulder_y, shoulder_z])
        servo_joint(spec, upper, f"{side}_shoulder_pitch", [0, -1, 0], (-1.0, 3.1), part)
        servo_joint(spec, upper, f"{side}_shoulder_roll", [sy, 0, 0], (-0.15, 2.8), part)
        shoulder_cap = _joint_cap(upper, part, 1, d.arm_r * 1.45, d.arm_r * 1.2, _Y_CYL)
        upper_tube = upper.add_geom(
            type=GEOM.mjGEOM_CAPSULE,
            fromto=[0, 0, -0.008 * s, 0, 0, -d.upper_arm + 0.008 * s],
            size=[d.arm_r, 0, 0],
            rgba=SHELL,
            density=density,
        )
        fore = upper.add_body(name=f"{side}_forearm", pos=[0, 0, -d.upper_arm])
        servo_joint(spec, fore, f"{side}_elbow", [0, -1, 0], (-0.1, 2.4), part)
        elbow_cap = _joint_cap(fore, part, 1, d.arm_r * 1.3, d.arm_r * 1.1, _Y_CYL)
        fore_tube = fore.add_geom(
            type=GEOM.mjGEOM_CAPSULE,
            fromto=[0, 0, -0.008 * s, 0, 0, -d.forearm],
            size=[d.arm_r * 0.9, 0, 0],
            rgba=accent,
            density=density,
        )
        hand = fore.add_geom(
            type=GEOM.mjGEOM_SPHERE,
            size=[d.arm_r * 1.25, 0, 0],
            pos=[0, 0, -d.forearm - d.arm_r * 0.6],
            rgba=DARK,
            density=components.MATERIALS["rubber"],
        )
        for geom in (shoulder_cap, upper_tube, elbow_cap, fore_tube, hand):
            vis.hide(geom)
        _dress_arm(vis, torso, upper, fore, sy, d, shoulder_z, accent)
        fore.add_site(
            name=f"{side}_hand", pos=[0, 0, -d.forearm - d.arm_r * 0.6], size=[0.005, 0, 0], rgba=[0, 0, 0, 0]
        )
        return {f"{side}_{j}": v for j, v in HOME_ARM.items()}

    @staticmethod
    def _head(spec, vis, torso, d: Dims, part, density: float) -> None:
        depth, width, tall = d.torso
        s = d.scale
        rx, ry, rz = d.head
        # Neck yaw servo stands on the shoulders like a short neck.
        vis.hide(
            servo_case(torso, part, [0, 0, tall / 2 - part.size[2] / 2 + 0.01 * s], quat=[0.7071068, 0, 0, 0.7071068])
        )
        _visual_only(torso)
        head = torso.add_body(name="head", pos=[0, 0, tall / 2 + 0.012 * s])
        servo_joint(spec, head, "neck_yaw", [0, 0, 1], (-1.4, 1.4), part)
        servo_joint(spec, head, "neck_pitch", [0, 1, 0], (-0.6, 0.8), part)
        neck_cap = _joint_cap(head, part, 1, 0.009 * s, part.size[1] / 2, _Y_CYL)
        center = [0.004 * s, 0, rz + 0.004 * s]
        skull = head.add_geom(
            type=GEOM.mjGEOM_ELLIPSOID, size=[rx, ry, rz], pos=center, rgba=SHELL, density=density * 0.3
        )
        # Dark visor with two glowing eyes, so it is obvious which way the robot faces.
        visor = head.add_geom(
            type=GEOM.mjGEOM_ELLIPSOID,
            size=[rx * 0.5, ry * 0.78, rz * 0.5],
            pos=[center[0] + rx * 0.58, 0, center[2] + rz * 0.05],
            rgba=DARK,
            mass=0.002,
            contype=0,
            conaffinity=0,
        )
        covered = [neck_cap, skull, visor]
        for side in (1, -1):
            eye = head.add_geom(
                type=GEOM.mjGEOM_SPHERE,
                size=[rz * 0.17, 0, 0],
                pos=[center[0] + rx * 0.98, side * ry * 0.3, center[2] + rz * 0.08],
                rgba=EYE,
                mass=0.001,
                contype=0,
                conaffinity=0,
            )
            covered.append(eye)
        for geom in covered:
            vis.hide(geom)
        _dress_head(vis, torso, head, d, center)
        head.add_camera(name="head_cam", pos=[center[0] + rx * 1.02, 0, center[2]], xyaxes=[0, -1, 0, 0, 0, 1], fovy=70)


def _strength_warnings(model: mujoco.MjModel, part: components.Actuator, mass: float, d: Dims) -> list[str]:
    """Tell beginners when the chosen servos cannot carry this robot (limits measured in simulation)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com_height = d.stand_height() + float(data.subtree_com[1][2])
    # On one leg the knee holds the whole robot at the thigh's lever arm. A grid of 12 heights x 5 servos
    # (walk, back, turn, tapped and mashed keys, random commands) stayed up to 56 % of stall, fell from 67 %.
    knee = mass * 9.81 * d.thigh * math.sin(HOME_BEND)
    # A position servo must be stiffer than the robot's tendency to topple over its ankles.
    topple = mass * 9.81 * com_height
    out = []
    if knee > 0.6 * part.stall_torque:
        out.append(
            f"warning: {part.name} servos ({part.stall_torque} N*m) are too weak for a {mass:.1f} kg humanoid: "
            f"standing on one leg needs about {knee:.1f} N*m at the knee, so it will likely fall when walking. "
            "Choose a stronger servo (see list_templates) or a smaller height."
        )
    elif part.kp < 1.2 * topple:
        out.append(
            f"warning: {part.name} servos are too springy for a {mass:.1f} kg humanoid this tall; it may wobble "
            "and fall when walking. Choose a stronger servo or a smaller height."
        )
    elif d.scale * REFERENCE_HEIGHT > MAX_WALKING_HEIGHT + 1e-9:
        out.append(
            f"note: walking is tuned for humanoids up to {MAX_WALKING_HEIGHT} m tall; this one stands and poses, "
            "but quick changes of speed or direction may topple it."
        )
    return out


def _visual_only(body: mujoco.MjsBody) -> None:
    """Servo housings are drawn and weighed but do not collide (they sit right at the joints)."""
    geom = body.geoms[-1]
    geom.contype = 0
    geom.conaffinity = 0


def _joint_cap(body, part, servos: int, radius: float, half_len: float, quat) -> mujoco.MjsGeom:
    """A dark cylinder along a joint axis carrying the mass of the servos at that joint."""
    return body.add_geom(
        type=GEOM.mjGEOM_CYLINDER,
        size=[radius, half_len, 0],
        quat=quat,
        rgba=DARK,
        mass=part.mass * servos,
        contype=0,
        conaffinity=0,
    )


def _hidden_mass(body, mass: float, pos: list[float], size: list[float]) -> mujoco.MjsGeom:
    return body.add_geom(
        type=GEOM.mjGEOM_BOX,
        size=[v / 2 for v in size],
        pos=pos,
        mass=mass,
        rgba=DARK,
        contype=0,
        conaffinity=0,
    )


# ---- looks: visual-only meshes (no mass, no contacts) over the collision primitives above ----------


def _drum(vis: Visuals, body, r: float, half: float, cap_r: float, pos=(0.0, 0.0, 0.0)) -> None:
    for preset, mesh in P.drum(r, half, cap_r).items():
        vis.add(body, preset, DARK if preset == "dark" else P.METAL, mesh, pos, M.Z_TO_Y)


def _dress_torso(vis: Visuals, torso, d: Dims, accent, pelvis_h: float) -> None:
    depth, width, tall = d.torso
    s = d.scale
    hx, hy, hz = depth / 2, width / 2, tall / 2
    shell = M.rounded_box(hx, hy, hz, 0.012 * s, segments=4, corner=0.018 * s)
    vis.add(torso, "shell", accent, shell)
    plate = M.rounded_box(0.004 * s, hy * 0.72, hz * 0.62, 0.0035 * s, segments=3, corner=0.02 * s)
    vis.add(torso, "shell", SHELL, plate, (hx + 0.002 * s, 0, tall * 0.06))
    # A fine dark gap around the chest plate, so it reads as a fitted cover rather than a sticker.
    gap = M.rounded_box(
        0.0032 * s, hy * 0.72 + 0.0014 * s, hz * 0.62 + 0.0014 * s, 0.002 * s, segments=2, corner=0.021 * s
    )
    vis.add(torso, "dark", DARK, gap, (hx + 0.0012 * s, 0, tall * 0.06))
    vis.add(
        torso,
        "led",
        P.LED_CYAN,
        P.led(0.0045 * s, 0.0012 * s)["led"],
        (hx + 0.006 * s, 0, tall * 0.06 + hz * 0.34),
        M.Z_TO_X,
    )
    pelvis = M.rounded_box(hx * 0.92, hy * 0.94, pelvis_h / 2, pelvis_h * 0.3, segments=3, corner=0.012 * s)
    vis.add(torso, "dark", DARK, pelvis, (0, 0, -hz + pelvis_h / 2 - 0.004 * s))
    # Battery hatch with vent slats on the back.
    hatch = M.rounded_box(0.0025 * s, hy * 0.6, hz * 0.5, 0.0015 * s, segments=2, corner=0.008 * s)
    vis.add(torso, "dark", DARK, hatch, (-hx - 0.001 * s, 0, tall * 0.04))
    vents = P.slats(hy * 0.42, hz * 0.3, 4, 0.0012 * s)
    vis.add(torso, "dark", DARK, vents, (-hx - 0.0035 * s, 0, tall * 0.04), M.rot_z(math.pi / 2))


def _dress_leg(vis: Visuals, thigh, shin, foot, sy: int, d: Dims, part, case_z: float, sole, accent) -> None:
    s = d.scale
    down = M.rot_x(math.pi)
    hip_half = part.size[1] / 2 + 0.006 * s
    _drum(vis, thigh, d.leg_r * 1.3, hip_half, d.leg_r * 0.95)
    limb = M.link(d.thigh, (d.leg_r * 1.15, d.leg_r * 1.02), (d.leg_r * 1.02, d.leg_r * 0.98), roundness=0.8)
    vis.add(thigh, "shell", SHELL, limb, (0, 0, 0), down)
    _drum(vis, shin, d.leg_r * 1.2, part.size[1] / 2 + 0.004 * s, d.leg_r * 0.85)
    reach = d.shin - 0.012 * s
    limb = M.link(reach, (d.leg_r * 1.02, d.leg_r * 0.92), (d.leg_r * 0.9, d.leg_r * 0.86), roundness=0.8)
    vis.add(shin, "shell", accent, limb, (0, 0, 0), down)
    # Ankle: the pitch servo with its horn facing out, above a layered foot.
    dress_servo(vis, foot, part, [0, 0, case_z], horn=(1, sy))
    length, wide, plate = d.foot
    x, y, z = sole
    rubber = 0.004 * s
    vis.add(
        foot,
        "rubber",
        P.RUBBER,
        M.rounded_box(length / 2, wide / 2, rubber / 2, rubber * 0.4, 2, 0.01 * s),
        (x, y, z + rubber / 2),
    )
    body = plate - rubber
    vis.add(
        foot,
        "dark",
        DARK,
        M.rounded_box(length / 2 - 0.0006, wide / 2 - 0.0006, body / 2, body * 0.35, 2, 0.01 * s),
        (x, y, z + rubber + body / 2 - 0.0002),
    )
    cover = M.rounded_box(length / 2 * 0.86, wide / 2 * 0.8, 0.0022 * s, 0.0016 * s, 3, 0.008 * s)
    vis.add(foot, "shell", accent, cover, (x + 0.002 * s, y, z + plate + 0.0015 * s))


def _dress_arm(vis: Visuals, torso, upper, fore, sy: int, d: Dims, shoulder_z: float, accent) -> None:
    depth, width, tall = d.torso
    s = d.scale
    down = M.rot_x(math.pi)
    # The shoulder servo's horn shows on the torso side, under the arm.
    out = M.Z_TO_Y if sy > 0 else M.rot_x(math.pi / 2)
    for preset, mesh in P.horn(d.arm_r * 1.05, 0.0022 * s, screws=4).items():
        vis.add(torso, preset, DARK if preset == "dark" else P.METAL, mesh, (0, sy * width / 2, shoulder_z), out)
    _drum(vis, upper, d.arm_r * 1.45, d.arm_r * 1.2, d.arm_r * 1.0)
    limb = M.link(d.upper_arm, (d.arm_r * 1.12, d.arm_r * 1.02), (d.arm_r * 1.0, d.arm_r * 0.98), roundness=0.8)
    vis.add(upper, "shell", SHELL, limb, (0, 0, 0), down)
    _drum(vis, fore, d.arm_r * 1.3, d.arm_r * 1.1, d.arm_r * 0.9)
    limb = M.link(d.forearm, (d.arm_r * 0.98, d.arm_r * 0.92), (d.arm_r * 0.92, d.arm_r * 0.88), roundness=0.8)
    vis.add(fore, "shell", accent, limb, (0, 0, 0), down)
    r = d.arm_r * 1.25
    mitt = M.rounded_box(r * 1.0, r * 0.82, r * 1.2, r * 0.7, segments=4, corner=r * 0.75)
    vis.add(fore, "dark", DARK, mitt, (0, 0, -d.forearm - d.arm_r * 0.6))


def _dress_head(vis: Visuals, torso, head, d: Dims, center) -> None:
    depth, width, tall = d.torso
    s = d.scale
    rx, ry, rz = d.head
    # A short neck column rising from the shoulders.
    neck = M.cylinder(0.013 * s, 0.016 * s, segments=28, bevel=0.002 * s, z0=tall / 2 - 0.004 * s)
    vis.add(torso, "dark", DARK, neck)
    _drum(vis, head, 0.009 * s, 0.0124 * s, 0.006 * s)
    skull = M.rounded_box(rx, ry, rz, rz * 0.62, segments=5, corner=ry * 0.62)
    vis.add(head, "shell", SHELL, skull, center)
    cx, _, cz = center
    visor = M.rounded_box(0.004 * s, ry * 0.74, rz * 0.48, 0.003 * s, segments=3, corner=rz * 0.4)
    vis.add(head, "glass", P.GLASS, visor, (cx + rx - 0.0022 * s, 0, cz + rz * 0.05))
    for side in (1, -1):
        eye = M.rounded_box(0.001 * s, 0.0042 * s, 0.0058 * s, 0.0008 * s, segments=2, corner=0.0036 * s)
        vis.add(head, "led", P.LED_CYAN, eye, (cx + rx + 0.0018 * s, side * ry * 0.32, cz + rz * 0.12))
        for preset, mesh in P.horn(rz * 0.38, 0.0025 * s, screws=0).items():
            rot = M.Z_TO_Y if side > 0 else M.rot_x(math.pi / 2)
            color = DARK if preset == "dark" else P.METAL
            vis.add(head, preset, color, mesh, (cx * 0.5, side * (ry - 0.0012 * s), cz), rot)
    for preset, mesh in P.lens(0.0042 * s, 0.0018 * s).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(head, preset, color, mesh, (cx + rx + 0.0016 * s, 0, cz - rz * 0.2), M.Z_TO_X)
