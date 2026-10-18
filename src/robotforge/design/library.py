"""Robot library: well-known real robots from MuJoCo Menagerie, fetched on first use.

The catalog is small and curated for beginners. Each robot is downloaded on its
first use, only its own folder (see menagerie.py), and compiled like an imported
file, plus a home pose from its keyframe, a spawn height that puts it on the
floor and, for torque-motor robots, position servos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from robotforge import wording
from robotforge.design import menagerie
from robotforge.design.templates.base import BuiltRobot
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.geometry import collides, lowest_point

SPAWN_CLEARANCE = 0.002  # m between the lowest point and the floor, so nothing starts inside it
TCP_RGBA = [1.0, 0.6, 0.1, 0.6]  # the tool point, as on our own arms


class LibraryError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


@dataclass(frozen=True)
class Tool:
    """Where an arm holds things. The grasp point becomes the "tcp" site that move_arm, pick_and_place and reach
    steer, as on our own arms, pointing the way the gripper points. An arm without a gripper also gets a "flange"
    site there: a gripper from the library mounts on it (see mounting.py)."""

    frame: str  # a body or a site of the model: the flange, or the grasp point itself
    reach: float = 0.0  # m from the frame's origin to the grasp point, along `axis`
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)  # the way the gripper points, in the frame's axes
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, the grasp point moved off the axis (one-jaw grippers)
    gripper: str | None = None  # actuator that opens and closes it, renamed "gripper" like ours
    # True when the gripper closes at the top of its control range (Robotiq's 0 open, 255 closed). It is turned
    # around, so that for every arm the top of the range opens and the bottom closes, as move_arm expects.
    closes_high: bool = False
    grip_force: float | None = None  # N or N*m: a gentler limit on the gripper's force than the model's
    arm: str = ""  # on a robot with two arms, each arm's name prefix ("left/"); "" on a one-armed robot
    cannot_pick: str = ""  # why pick_and_place does not work with it yet, which refuses it; "" when it does


@dataclass(frozen=True)
class Payload:
    """What a walking robot carries on top: the robots mounted on it (mounting.py), by their mass, measured with the
    library's arms on the real model (docs/BUILDER.md has the table). Up to `full` kg it walks as fast as it does
    without; up to `most` kg at `slow` m/s at most; anything heavier is refused."""

    full: float
    most: float
    slow: float

    def describe(self) -> str:
        if self.most <= self.full:
            return f"up to {self.most:g} kg"
        return f"up to {self.full:g} kg at full speed, {self.most:g} kg at {self.slow:g} m/s"


@dataclass(frozen=True)
class LibraryRobot:
    name: str
    title: str
    kind: Literal["quadruped", "arm", "humanoid", "gripper"]  # gripper: a hand to mount on an arm (mounting.py)
    summary: str
    maker: str
    license: str  # of the Menagerie model folder (see its LICENSE file)
    folder: str  # Menagerie directory
    mjcf: str  # model file inside it
    home_key: str | None = "home"  # keyframe that becomes our home pose
    # For a robot with no keyframe anywhere: joint angles (rad) of its home pose; the others stay at zero.
    home: tuple[tuple[str, float], ...] = ()
    # Torque-motor models get position servos with this stiffness (N*m/rad), keeping their torque
    # limits: that is how the real robots are commanded (joint targets + PD in the motor driver).
    servo_kp: float | None = None
    # Joint position servos that are too soft for the robot's weight get this stiffness (N*m/rad) and damping
    # (N*m*s/rad), keeping their force limits.
    stiffer_servos: tuple[float, float] | None = None
    max_speed: float | None = None  # m/s, for walkers slower than the gait's general estimate
    tools: tuple[Tool, ...] = ()  # arms: where each arm holds things (see Tool)
    mass: float = 0.0  # kg, of the model at the pinned commit
    # Degrees to turn a fixed-base model about the vertical so its front faces +x, as every robot here does.
    facing: float = 0.0
    payload: Payload | None = None  # walkers: what they carry (see Payload)
    # Arms whose real controllers push the torques gravity needs through their motors, which the Menagerie model
    # leaves out: arm motions do the same for them (see ArmMotion._hold). Not for the PiPER, whose model cancels
    # gravity on its links itself.
    holds_against_gravity: bool = False


CATALOG: dict[str, LibraryRobot] = {
    r.name: r
    for r in [
        LibraryRobot(
            "go2",
            "Unitree Go2",
            "quadruped",
            "Popular 15 kg robot dog with 12 joints; walks with move and the keyboard.",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_go2",
            "go2.xml",
            mass=15.21,
            # Carries (measured, see docs/BUILDER.md) a Z1 at full speed; an xArm 7 at 0.1 m/s (a Panda toppled it
            # turning, a UR5e standing).
            payload=Payload(4.7, 12.1, 0.1),
            servo_kp=100.0,  # what Menagerie uses for Go1, which has the same motor family
        ),
        LibraryRobot(
            "go1",
            "Unitree Go1",
            "quadruped",
            "Go2's predecessor, 12 kg, 12 joints; walks with move and the keyboard.",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_go1",
            "go1.xml",
            mass=12.74,
            # Carries (measured, see docs/BUILDER.md) a Z1 at full speed; an xArm 7 at 0.15 m/s (a Panda toppled it).
            payload=Payload(4.7, 12.1, 0.15),
        ),
        LibraryRobot(
            "a1",
            "Unitree A1",
            "quadruped",
            "Research robot dog, 12 kg, 12 joints; walks with move and the keyboard.",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_a1",
            "a1.xml",
            mass=12.45,
            # Carries (measured, see docs/BUILDER.md) a Z1 at full speed; an xArm 7 at 0.1 m/s (a Panda toppled it).
            payload=Payload(4.7, 12.1, 0.1),
        ),
        LibraryRobot(
            "spot",
            "Boston Dynamics Spot",
            "quadruped",
            "Large industrial robot dog with 12 joints and room on its back for an arm; this model weighs 50 kg (the "
            "real one 32 kg).",
            "Boston Dynamics (model by Clearpath Robotics)",
            "BSD-3-Clause",
            "boston_dynamics_spot",
            "spot.xml",
            mass=50.34,
            # Carries (measured, see docs/BUILDER.md) an xArm 7 at full speed; a Panda at 0.15 m/s (a UR5e toppled it
            # turning).
            payload=Payload(12.1, 17.5, 0.15),
            max_speed=0.35,  # measured: tops out at 0.36-0.39 m/s however long the stride
        ),
        LibraryRobot(
            "panda",
            "Franka Emika Panda",
            "arm",
            "7-joint research arm with a two-finger gripper, the classic manipulation robot.",
            "Franka Robotics",
            "Apache-2.0",
            "franka_emika_panda",
            "panda.xml",
            mass=17.45,
            tools=(Tool("hand", 0.1034, gripper="actuator8"),),  # Franka's own flange-to-TCP distance
        ),
        LibraryRobot(
            "so_arm100",
            "SO-ARM100",
            "arm",
            "Low-cost 3D-printed 5-joint arm with gripper, used with LeRobot; picks small things up to about 0.27 m in "
            "front of it.",
            "The Robot Studio",
            "Apache-2.0",
            "trs_so_arm100",
            "so_arm100.xml",
            mass=1.17,
            # The jaws reach along the fixed jaw's -y; the pads span 0.047-0.106 m. Grasped in the middle of the pads
            # (0.077 m), the jaw tips went 9 mm into the floor at a 4 cm cube's middle, so it could not pick from the
            # floor at all; 16 mm short of the tips they clear it by 4 mm. The grasp point sits 26 mm off the fixed
            # jaw's pad (x 0.008): that jaw comes down 6 mm beside the cube, not on its edge, and the moving jaw
            # pushes the cube against it. 5 of 6 floor layouts within 0.27 m; at 0.12 m off the pad, 1 of 6.
            tools=(Tool("Fixed_Jaw", 0.09, axis=(0.0, -1.0, 0.0), shift=(-0.018, 0.0, 0.0), gripper="Jaw"),),
            # The model faces -y. Mounted that way on a dog's back its folded arm stuck out 27 cm over the dog's side:
            # a Go2 drifted 0.17 m sideways in 10 s and a Barkour walked backwards (-0.4 m) instead of 2.5 m forwards.
            facing=90.0,
        ),
        LibraryRobot(
            "piper",
            "AgileX PiPER",
            "arm",
            "Light 6-joint desktop arm (0.63 m reach) with a parallel gripper, popular for teaching.",
            "AgileX Robotics (model by Omar Rayyan)",
            "MIT",
            "agilex_piper",
            "piper.xml",
            mass=2.37,
            # centre of the fingertip pads (the pads span 0.075-0.135 m past the flange)
            tools=(Tool("link6", 0.12, gripper="gripper"),),
        ),
        LibraryRobot(
            "g1",
            "Unitree G1",
            "humanoid",
            "1.3 m humanoid with 29 joints; walks with move and the keyboard on a learned (UniLab) policy.",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_g1",
            "g1.xml",
            mass=33.34,
            # Carries (measured, see docs/BUILDER.md) a PiPER at full speed (1 m/s); a Z1 at 0.3 m/s (it fell at 1 m/s;
            # an xArm 7 at 0.3).
            payload=Payload(2.4, 4.7, 0.3),
            home_key="stand",
        ),
        LibraryRobot(
            "anymal_c",
            "ANYbotics ANYmal C",
            "quadruped",
            "45 kg industrial inspection robot dog with a lidar on its back; walks with move and the keyboard.",
            "ANYbotics",
            "BSD-3-Clause",
            "anybotics_anymal_c",
            "anymal_c.xml",
            mass=44.97,
            # Carries (measured, see docs/BUILDER.md) an xArm 7 at full speed; a UR5e with a Robotiq at 0.2 m/s (a Panda
            # rocked it 27 degrees turning at 0.36).
            payload=Payload(12.1, 22.1, 0.2),
            # Lower than the stance of Menagerie's MJX keyframe (0.52, 0.785 rad), where the legs are nearly straight:
            # on those our trot swung a knee through straight and it fell within 6 s; on these it walks 0.3 m/s.
            home=tuple(
                (f"{leg}_{joint}", sign * angle)
                for leg, sign in (("LF", 1), ("RF", 1), ("LH", -1), ("RH", -1))
                for joint, angle in (("HFE", 0.8), ("KFE", -1.4))
            ),
            # Menagerie's kp 100 and the joints' own damping of 1 let the 45 kg robot sag from 0.44 to 0.27 m and rock:
            # its knees hit the floor 15 to 36% of the time, it tilted up to 45 degrees turning at keyboard commands,
            # and at 0.6 m/s it veered off. At kp 300 alone it fell over turning. With damping 10 it walks 0.3 and
            # 0.6 m/s within 4 degrees, turns as told within 6, and nothing but its feet touches the floor.
            stiffer_servos=(300.0, 10.0),
        ),
        LibraryRobot(
            "barkour",
            "Google Barkour vB",
            "quadruped",
            "Google DeepMind's 14 kg agility robot dog with a head camera; walks with move and the keyboard.",
            "Google DeepMind",
            "Apache-2.0",
            "google_barkour_vb",
            "barkour_vb.xml",
            mass=14.5,
            # Carries (measured, see docs/BUILDER.md) a PiPER at full speed; a Z1 rocked it 16 to 21 degrees even at 0.1
            # m/s.
            payload=Payload(2.4, 2.4, 0.3),
            # On the stance of Menagerie's scene keyframe (hip 0.5, knee 1.0 rad) our trot walked it backwards
            # (-1.4 m in 10 s at 0.3 m/s); on these bent knees it walks 3.0 m, straight.
            home=tuple(
                (f"{joint}_{leg}", angle)
                for leg in ("front_left", "hind_left", "front_right", "hind_right")
                for joint, angle in (("hip", 0.7), ("knee", 1.4))
            ),
        ),
        LibraryRobot(
            "op3",
            "Robotis OP3",
            "humanoid",
            "51 cm, 3 kg open-source humanoid with 20 servos, a RoboCup favourite; walks with move and the keyboard.",
            "Robotis",
            "Apache-2.0",
            "robotis_op3",
            "op3.xml",
            mass=3.15,
            # Carries (measured, see docs/BUILDER.md) our 0.08 kg gripper template; a 1 kg Robotiq on its head toppled
            # it turning, an SO-100 walking.
            payload=Payload(0.1, 0.1, 0.1),
            # No keyframe in Menagerie: knees soft like our template's (hip and ankle 0.35 rad, knee 0.7). Each joint
            # turns the way its axis says; the right leg's pitch axes point the other way.
            home=(
                ("l_hip_pitch", -0.35),
                ("l_knee", 0.7),
                ("l_ank_pitch", 0.35),
                ("r_hip_pitch", 0.35),
                ("r_knee", -0.7),
                ("r_ank_pitch", -0.35),
                ("l_sho_roll", 1.25),  # arms down at its sides (at zero they stick out sideways)
                ("r_sho_roll", -1.25),
            ),
        ),
        LibraryRobot(
            "t1",
            "Booster T1",
            "humanoid",
            "1.2 m, 32 kg humanoid with 23 joints, built for robot soccer; walks slowly (0.12 m/s) with move and the "
            "keyboard.",
            "Booster Robotics",
            "Apache-2.0",
            "booster_t1",
            "t1.xml",
            mass=31.61,
            # Carries (measured, see docs/BUILDER.md) an SO-100 at full speed; a PiPER at 0.06 m/s (it fell at 0.12; a
            # Z1 even at 0.05).
            payload=Payload(1.2, 2.4, 0.06),
            # On Menagerie's kp 75, kv 5 servos the balance gait kept it standing within 2 degrees but it fell at its
            # first steps, even at 0.05 m/s, and a 0.4 m/s shove knocked it over. Stiffer and damped, it walks 0.05
            # to 0.14 m/s within 6 degrees, turns at keyboard commands, and takes 0.4 m/s shoves within 9.
            stiffer_servos=(150.0, 12.0),
            max_speed=0.12,  # measured: 1.13 m in 10 s at 0.12; at the gait's own 0.17 it fell within 2 s
        ),
        LibraryRobot(
            "z1",
            "Unitree Z1",
            "arm",
            "Unitree's 4.7 kg 6-joint arm (0.74 m reach) for robot dogs; move_arm works, but its one-jaw claw "
            "cannot hold a cube in pick_and_place yet.",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_z1",
            "z1_gripper.xml",
            mass=4.69,
            holds_against_gravity=True,  # on its servos alone it sagged 3 to 4 mm 0.33 to 0.45 m out
            # The grasp point is the middle of the fingertip pads (0.172-0.2 m along link06's x), 1 cm above the fixed
            # jaw's pad: a cube rests on that jaw and the moving one closes onto it (at 0 rad; -1.52 is wide open).
            tools=(
                Tool(
                    "link06",
                    0.186,
                    axis=(1.0, 0.0, 0.0),
                    shift=(0.0, 0.0, 0.01),
                    gripper="motorGripper",
                    closes_high=True,
                    grip_force=6.0,  # its 30 N*m flung a 40 g cube out of the jaws at 20 m/s
                    # 1 of 3 floor picks (2026-10-17): it lifts the cube, then lets go of it as the arm swings
                    cannot_pick="its one-jaw claw lets go of a cube as the arm swings",
                ),
            ),
        ),
        LibraryRobot(
            "ur5e",
            "Universal Robots UR5e",
            "arm",
            "The classic 6-joint industrial cobot (0.85 m reach) with a bare flange: mount a gripper (robotiq_2f85).",
            "Universal Robots",
            "BSD-3-Clause",
            "universal_robots_ur5e",
            "ur5e.xml",
            mass=20.99,
            # On its servos alone it sagged 7 to 17 mm 0.47 to 0.73 m out, and on ANYmal its grasps closed 2 cm beside
            # the cube.
            holds_against_gravity=True,
            tools=(Tool("attachment_site"),),
        ),
        LibraryRobot(
            "xarm7",
            "UFACTORY xArm 7",
            "arm",
            "7-joint arm (0.7 m reach) with UFACTORY's parallel gripper.",
            "UFACTORY",
            "BSD-3-Clause",
            "ufactory_xarm7",
            "xarm7.xml",
            mass=12.07,
            holds_against_gravity=True,  # on its servos alone it sagged 11 to 17 mm 0.35 to 0.55 m out
            # link_tcp is the fingertips; the pads end 2 mm short of them and reach 3.9 cm up. Grasped at the tips, a
            # 4 cm cube was held by its top 18 mm and slipped out on 4 of 6 carries. 15 mm deeper the pads hold 2.4 cm
            # of it and still clear the floor by 7 mm: 6 of 6.
            tools=(Tool("link_tcp", -0.015, gripper="gripper", closes_high=True),),
        ),
        LibraryRobot(
            "aloha",
            "ALOHA 2",
            "arm",
            "Two-armed teleoperation station (two 6-joint arms with grippers facing each other); arms aloha/left and "
            "aloha/right.",
            "Trossen Robotics and Google DeepMind",
            "BSD-3-Clause",
            "aloha",
            "aloha.xml",
            mass=8.52,
            home_key="neutral_pose",
            tools=(
                Tool("left/gripper", axis=(1.0, 0.0, 0.0), gripper="left/gripper", arm="left/"),
                Tool("right/gripper", axis=(1.0, 0.0, 0.0), gripper="right/gripper", arm="right/"),
            ),
        ),
        LibraryRobot(
            "robotiq_2f85",
            "Robotiq 2F-85",
            "gripper",
            "The most common industrial two-finger gripper (85 mm stroke); mount it on the flange of an arm (ur5e).",
            "Robotiq (model from ROS-Industrial)",
            "BSD-2-Clause",
            "robotiq_2f85",
            "2f85.xml",
            mass=1.05,
            home_key=None,
            tools=(Tool("pinch", gripper="fingers_actuator", closes_high=True),),
        ),
    ]
}


def _titles() -> dict[str, str]:
    return {wording.squash(r.title): r.name for r in CATALOG.values()}  # "Unitree Go2" is go2


def canonical(name: str) -> str:
    """The library robot `name` means ("Go2", "Unitree Go2"), or `name` itself when there is none."""
    return wording.match(name, CATALOG, _titles()) or name


def get(name: str) -> LibraryRobot:
    found = wording.match(name, CATALOG, _titles())
    if found is None:
        close = wording.near(name, CATALOG, _titles())
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise LibraryError(f"no robot {name!r} in the library.{hint} Choose one of: {', '.join(CATALOG)}")
    return CATALOG[found]


def kind_of(name: str | None) -> str | None:
    """What a library robot is ("arm", "gripper"), or None for no library robot."""
    return get(name).kind if name is not None else None


def catalog() -> list[dict[str, Any]]:
    """Library robots for agents: what they are, their license, download size and whether they are here yet."""
    return [
        {
            "name": r.name,
            "title": r.title,
            "kind": r.kind,
            "summary": r.summary,
            "maker": r.maker,
            "license": r.license,
            "mass_kg": r.mass,
            **({"carries": r.payload.describe()} if r.payload is not None else {}),
            "download_mb": download_mb(r.name),
            "downloaded": is_downloaded(r.name),
        }
        for r in CATALOG.values()
    ]


def download_mb(name: str) -> float:
    return round(menagerie.download_size(get(name).folder) / 1e6, 1)


def is_downloaded(name: str) -> bool:
    return menagerie.local_folder(get(name).folder) is not None


def model_dir(robot: LibraryRobot, progress: menagerie.Progress | None = None) -> Path:
    """The robot's Menagerie folder, downloading it on first use."""
    return menagerie.fetch(robot.folder, progress)


# ---- building ---------------------------------------------------------------------------


def build(name: str, params: dict[str, Any] | None = None) -> BuiltRobot:
    """Load a library robot as a spec ready to attach into a world."""
    robot = get(name)
    if params:
        raise LibraryError(f"library robots are used as built and take no params (got {', '.join(params)})")
    spec = _read(model_dir(robot) / robot.mjcf)
    key_qpos, key_ctrl = _read_key(spec, robot.home_key)
    if key_qpos is None and robot.home:
        key_qpos = _home_from_joints(spec, robot)
    notes = [f"{robot.title} from MuJoCo Menagerie ({robot.license}, {robot.maker})"]
    if robot.servo_kp is not None:
        converted = _motors_to_servos(spec, robot.servo_kp)
        if converted:
            notes.append(
                f"{converted} torque motors driven as position servos (kp {robot.servo_kp:g}, torque limits kept)"
            )
    if robot.stiffer_servos is not None:
        kp, kv = robot.stiffer_servos
        stiffened = _stiffen_servos(spec, kp, kv)
        if stiffened:
            notes.append(f"{stiffened} joint servos stiffened to kp {kp:g}, kv {kv:g} (force limits kept)")
    if robot.tools:
        key_ctrl = _add_tools(spec, robot, key_ctrl)
    for geom in list(spec.worldbody.geoms):  # the world brings its own floor and lights
        spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    for key in list(spec.keys):
        spec.delete(key)
    root = spec.worldbody.first_body()
    if root is None:
        raise LibraryError(f"{robot.title}: the model has no bodies")
    floating = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints)
    if robot.facing and not floating:
        turn, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, [0.0, 0.0, 1.0], np.radians(robot.facing))
        mujoco.mju_mulQuat(quat, turn, np.asarray(root.quat, float))
        pos = np.zeros(3)
        mujoco.mju_rotVecQuat(pos, np.asarray(root.pos, float), turn)
        root.quat, root.pos = quat.tolist(), pos.tolist()
    if floating:
        # Place the base by its keyframe orientation; height comes from the lowest point below. Without a keyframe
        # it faces +x like every robot here: a file's own placement is its scene's (ANYmal C's faces -x).
        root.pos = [0.0, 0.0, 0.0]
        root.quat = key_qpos[3:7].tolist() if key_qpos is not None else [1.0, 0.0, 0.0, 0.0]
    home_qpos, home_ctrl, spawn = _home_pose(spec, key_qpos, key_ctrl, floating)
    for tool in robot.tools:  # a gripper with no keyframe to say otherwise starts open, as the arms' keyframes have it
        gripper = f"{tool.arm}gripper"
        actuator = next((a for a in spec.actuators if a.name == gripper), None)
        if tool.gripper is not None and gripper not in home_ctrl and actuator is not None:
            home_ctrl[gripper] = float(actuator.ctrlrange[1])
    return BuiltRobot(
        spec=spec, home_qpos=home_qpos, home_ctrl=home_ctrl, spawn_height=spawn, floating=floating, notes=notes
    )


def _home_from_joints(spec: mujoco.MjSpec, robot: LibraryRobot) -> np.ndarray:
    model = spec.compile()
    qpos = model.qpos0.copy()
    if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # level and facing +x
    for joint, angle in robot.home:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if j < 0:
            raise LibraryError(f"{robot.title}: the model has no joint {joint!r} for its home pose")
        qpos[model.jnt_qposadr[j]] = angle
    return qpos


def _read(path: Path) -> mujoco.MjSpec:
    try:
        return mujoco.MjSpec.from_file(str(path))
    except ValueError as exc:
        raise LibraryError(f"MuJoCo could not read {path}: {exc}") from exc


def _add_tools(spec: mujoco.MjSpec, robot: LibraryRobot, key_ctrl: np.ndarray | None) -> np.ndarray | None:
    """Put each arm's "tcp" site (and "flange", without a gripper) on the model and name its gripper "gripper".
    Returns the keyframe's controls, with those of a gripper that was turned around turned with it."""
    model = spec.compile()  # resolves where sites really are (frames, other ways of writing an orientation)
    actuators = list(spec.actuators)
    for tool in robot.tools:
        body, pos, quat = _tool_frame(spec, model, robot, tool)
        axis = np.asarray(tool.axis, float) / np.linalg.norm(tool.axis)
        turn, tip = np.zeros(4), np.zeros(4)
        mujoco.mju_quatZ2Vec(turn, axis)
        mujoco.mju_mulQuat(tip, quat, turn)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        grasp = pos + rot.reshape(3, 3) @ (tool.reach * axis + np.asarray(tool.shift, float))
        body.add_site(name=f"{tool.arm}tcp", pos=grasp.tolist(), quat=tip.tolist(), size=[0.006, 0, 0], rgba=TCP_RGBA)
        if tool.gripper is None:  # a bare flange: a gripper can be mounted here
            body.add_site(name=f"{tool.arm}flange", pos=pos.tolist(), quat=tip.tolist(), size=[0.004, 0, 0], group=5)
            continue
        index = next((i for i, a in enumerate(actuators) if a.name == tool.gripper), None)
        if index is None:
            raise LibraryError(f"{robot.title}: the model has no gripper actuator {tool.gripper!r}")
        actuators[index].name = f"{tool.arm}gripper"
        if tool.grip_force is not None:
            actuators[index].forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuators[index].forcerange = [-tool.grip_force, tool.grip_force]
        if tool.closes_high:
            low, high = _turn_around(actuators[index], robot)
            if key_ctrl is not None and key_ctrl.size == len(actuators):
                key_ctrl = key_ctrl.copy()
                key_ctrl[index] = low + high - key_ctrl[index]
    return key_ctrl


def _tool_frame(
    spec: mujoco.MjSpec, model: mujoco.MjModel, robot: LibraryRobot, tool: Tool
) -> tuple[mujoco.MjsBody, np.ndarray, np.ndarray]:
    """The body a tool's frame is on, and the frame's position and orientation in that body."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tool.frame)
    if site >= 0:
        body = spec.body(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site])))
        return body, model.site_pos[site].copy(), model.site_quat[site].copy()
    body = spec.body(tool.frame)
    if body is None:
        raise LibraryError(f"{robot.title}: the model has no body or site {tool.frame!r} to put the tool point on")
    return body, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])


def _turn_around(actuator: mujoco.MjsActuator, robot: LibraryRobot) -> tuple[float, float]:
    """Make an actuator read its control the other way round within its range (c becomes low + high - c), exactly:
    force = gain * c + bias is affine in c. Returns the range."""
    if actuator.gaintype != mujoco.mjtGain.mjGAIN_FIXED or actuator.dyntype != mujoco.mjtDyn.mjDYN_NONE:
        raise LibraryError(f"{robot.title}: cannot turn the gripper actuator {actuator.name!r} around")
    low, high = (float(v) for v in actuator.ctrlrange)
    gain = np.array(actuator.gainprm, float)
    bias = np.array(actuator.biasprm, float)
    bias[0] += gain[0] * (low + high)
    gain[0] = -gain[0]
    actuator.gainprm = gain.tolist()
    actuator.biasprm = bias.tolist()
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    return low, high


def _read_key(spec: mujoco.MjSpec, name: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if name is None:
        return None, None
    key = next((k for k in spec.keys if k.name == name), None)
    if key is None:
        return None, None
    qpos = np.asarray(key.qpos, float) if len(key.qpos) else None
    ctrl = np.asarray(key.ctrl, float) if len(key.ctrl) else None
    return qpos, ctrl


def _motors_to_servos(spec: mujoco.MjSpec, kp: float) -> int:
    """Turn joint torque motors into position servos with the same torque limits."""
    joints = {j.name: j for j in spec.joints}
    count = 0
    for act in spec.actuators:
        is_motor = (
            act.trntype == mujoco.mjtTrn.mjTRN_JOINT
            and act.gaintype == mujoco.mjtGain.mjGAIN_FIXED
            and act.biastype == mujoco.mjtBias.mjBIAS_NONE
            and act.dyntype == mujoco.mjtDyn.mjDYN_NONE
        )
        joint = joints.get(act.target)
        torque = float(np.abs(np.asarray(act.ctrlrange, float)).max() * abs(act.gear[0]))
        if not is_motor or joint is None or torque <= 0:  # no torque limit known: leave it alone
            continue
        if joint.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        act.gear = [1, 0, 0, 0, 0, 0]
        act.set_to_position(kp=kp)  # damping comes from the model's joints
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-torque, torque]
        joint_range = np.asarray(joint.range, float)
        limited = joint.limited == mujoco.mjtLimited.mjLIMITED_TRUE or (
            joint.limited == mujoco.mjtLimited.mjLIMITED_AUTO and joint_range[0] < joint_range[1]
        )
        if limited:
            if spec.compiler.degree and joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                joint_range = np.radians(joint_range)  # spec ranges keep the file's units; controls are radians
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = joint_range.tolist()
        else:  # a continuous joint: any target is fine
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        count += 1
    return count


def _stiffen_servos(spec: mujoco.MjSpec, kp: float, kv: float) -> int:
    """Give the joint position servos this stiffness and damping (force = kp * (target - q) - kv * qdot)."""
    count = 0
    for act in spec.actuators:
        gain, bias = list(act.gainprm), list(act.biasprm)
        servo = (
            act.trntype == mujoco.mjtTrn.mjTRN_JOINT
            and act.gaintype == mujoco.mjtGain.mjGAIN_FIXED
            and act.biastype == mujoco.mjtBias.mjBIAS_AFFINE
            and act.dyntype == mujoco.mjtDyn.mjDYN_NONE
            and gain[0] > 0
            and bias[1] == -gain[0]
        )
        if servo:
            gain[0], bias[1], bias[2] = kp, -kp, -kv
            act.gainprm, act.biasprm = gain, bias
            count += 1
    return count


def _home_pose(
    spec: mujoco.MjSpec, key_qpos: np.ndarray | None, key_ctrl: np.ndarray | None, floating: bool
) -> tuple[dict[str, float], dict[str, float], float]:
    """Joint and actuator values of the home pose, and the base height that puts it on the floor."""
    model = spec.compile()
    data = mujoco.MjData(model)
    if key_qpos is not None and key_qpos.size == model.nq:
        data.qpos[:] = key_qpos
    if floating:
        root = model.body_jntadr[1]  # the free joint of the first body
        data.qpos[model.jnt_qposadr[root] : model.jnt_qposadr[root] + 3] = 0.0
    mujoco.mj_kinematics(model, data)
    home_qpos: dict[str, float] = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            home_qpos[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = float(data.qpos[model.jnt_qposadr[j]])
    home_ctrl: dict[str, float] = {}
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        on_joint = int(model.actuator_trntype[a]) == mujoco.mjtTrn.mjTRN_JOINT
        if on_joint and is_position_servo(model, a):
            # A servo on a joint should aim where the joint is, or the robot jumps at the start.
            home_ctrl[name] = float(data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]])
        elif key_ctrl is not None and key_ctrl.size == model.nu:
            home_ctrl[name] = float(key_ctrl[a])
    # Floating robots stand on their lowest point; fixed ones stand on their base.
    on_base = model.body_weldid[model.geom_bodyid] == 0
    geoms = [g for g in range(model.ngeom) if collides(model, g) and (floating or on_base[g])]
    lowest = min((lowest_point(model, data, g)[2] for g in geoms), default=0.0)
    return home_qpos, home_ctrl, -lowest + (SPAWN_CLEARANCE if floating else 0.0)
