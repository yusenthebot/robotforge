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
from robotforge.design import menagerie, rotors
from robotforge.design.rotors import RotorLayout
from robotforge.design.templates.base import BuiltRobot
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.geometry import collides, lowest_point

SPAWN_CLEARANCE = 0.002  # m between the lowest point and the floor, so nothing starts inside it


class LibraryError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


@dataclass(frozen=True)
class LibraryRobot:
    name: str
    title: str
    kind: Literal["quadruped", "arm", "humanoid", "aircraft"]
    summary: str
    maker: str
    license: str  # of the Menagerie model folder (see its LICENSE file)
    folder: str  # Menagerie directory
    mjcf: str  # model file inside it
    home_key: str | None = "home"  # keyframe that becomes our home pose
    # Torque-motor models get position servos with this stiffness (N*m/rad), keeping their torque
    # limits: that is how the real robots are commanded (joint targets + PD in the motor driver).
    servo_kp: float | None = None
    max_speed: float | None = None  # m/s, for walkers slower than the gait's general estimate
    # Arms: the body the tool hangs from and how far along its +z the grasp point is (m). It becomes
    # the "tcp" site that move_arm, pick_and_place and reach steer, as on our own arms.
    tool: tuple[str, float] | None = None
    gripper: str | None = None  # actuator that opens the gripper, renamed "gripper" like ours
    rotors: RotorLayout | None = None  # aircraft: real rotors in place of the model's own actuators (rotors.py)


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
        ),
        LibraryRobot(
            "spot",
            "Boston Dynamics Spot",
            "quadruped",
            "Large industrial robot dog, 32 kg, 12 joints (no arm).",
            "Boston Dynamics (model by Clearpath Robotics)",
            "BSD-3-Clause",
            "boston_dynamics_spot",
            "spot.xml",
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
            tool=("hand", 0.1034),  # Franka's own flange-to-TCP distance
            gripper="actuator8",
        ),
        LibraryRobot(
            "so_arm100",
            "SO-ARM100",
            "arm",
            "Low-cost 3D-printed 5-joint arm with gripper, used with LeRobot.",
            "The Robot Studio",
            "Apache-2.0",
            "trs_so_arm100",
            "so_arm100.xml",
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
            tool=("link6", 0.12),  # centre of the fingertip pads (the pads span 0.075-0.135 m past the flange)
            gripper="gripper",
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
            home_key="stand",
        ),
        # Aircraft fly with robotforge's flight controller, from the rotors the model has (behaviors/airframe.py).
        # They start on the ground with their motors idle, so no hover keyframe.
        LibraryRobot(
            "skydio_x2",
            "Skydio X2",
            "aircraft",
            "1.3 kg quadrotor, four thrust rotors; flies with move(to=[x, y, z]) and the keyboard.",
            "Skydio",
            "Apache-2.0",
            "skydio_x2",
            "x2.xml",
            home_key=None,
        ),
        LibraryRobot(
            "crazyflie",
            "Bitcraze Crazyflie 2",
            "aircraft",
            "27 g palm-sized quadrotor for research (thrust and three body moments); flies with move and the keyboard.",
            "Bitcraze (model by whoenig)",
            "MIT",
            "bitcraze_crazyflie_2",
            "cf2.xml",
            home_key=None,
            rotors=rotors.CRAZYFLIE,
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
    path = model_dir(robot) / robot.mjcf
    try:
        spec = mujoco.MjSpec.from_file(str(path))
    except ValueError as exc:
        raise LibraryError(f"MuJoCo could not read {path}: {exc}") from exc
    key_qpos, key_ctrl = read_key(spec, robot.home_key)
    notes = [f"{robot.title} from MuJoCo Menagerie ({robot.license}, {robot.maker})"]
    if robot.servo_kp is not None:
        converted = _motors_to_servos(spec, robot.servo_kp)
        if converted:
            notes.append(
                f"{converted} torque motors driven as position servos (kp {robot.servo_kp:g}, torque limits kept)"
            )
    if robot.tool is not None:
        _add_tool(spec, robot)
    if robot.rotors is not None:
        notes.append(rotors.fit(spec, robot.rotors))
    for geom in list(spec.worldbody.geoms):  # the world brings its own floor and lights
        spec.delete(geom)
    for light in list(spec.worldbody.lights):
        spec.delete(light)
    for key in list(spec.keys):
        spec.delete(key)
    if spec.worldbody.first_body() is None:
        raise LibraryError(f"{robot.title}: the model has no bodies")
    home_qpos, home_ctrl, spawn, floating = stand_at_home(spec, key_qpos, key_ctrl)
    return BuiltRobot(
        spec=spec, home_qpos=home_qpos, home_ctrl=home_ctrl, spawn_height=spawn, floating=floating, notes=notes
    )


def stand_at_home(
    spec: mujoco.MjSpec, key_qpos: np.ndarray | None, key_ctrl: np.ndarray | None
) -> tuple[dict[str, float], dict[str, float], float, bool]:
    """Ready a robot's spec (with at least one body) to stand in a world in its home pose: a floating base moves to
    the origin, turned the way its keyframe has it. Returns the home joint values, the actuator values, the height
    that puts it on the floor and whether its base floats."""
    root = spec.worldbody.first_body()
    floating = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints)
    if floating:
        # Place the base by its keyframe orientation; height comes from the lowest point below.
        root.pos = [0.0, 0.0, 0.0]
        if key_qpos is not None:
            root.quat = key_qpos[3:7].tolist()
    home_qpos, home_ctrl, spawn = home_pose(spec, key_qpos, key_ctrl, floating)
    return home_qpos, home_ctrl, spawn, floating


def _add_tool(spec: mujoco.MjSpec, robot: LibraryRobot) -> None:
    body_name, reach = robot.tool
    body = spec.body(body_name)
    if body is None:
        raise LibraryError(f"{robot.title}: the model has no body {body_name!r} to put the tool point on")
    body.add_site(name="tcp", pos=[0, 0, reach], size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])
    if robot.gripper and robot.gripper != "gripper":
        actuator = next((a for a in spec.actuators if a.name == robot.gripper), None)
        if actuator is None:
            raise LibraryError(f"{robot.title}: the model has no gripper actuator {robot.gripper!r}")
        actuator.name = "gripper"


def read_key(spec: mujoco.MjSpec, name: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def home_pose(
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
