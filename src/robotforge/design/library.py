"""Robot library: well-known real robots from MuJoCo Menagerie, fetched on first use.

The catalog is small and curated for beginners. Models come from the Menagerie
commit that the robot_descriptions package pins. robot_descriptions itself would
clone all of Menagerie (about 2 GB on disk, a few minutes) to give you one robot,
so robotforge fetches only the folder of the robot you ask for with a shallow,
sparse git checkout (tens of MB, seconds) into the same cache directory.
An existing full robot_descriptions clone at the pinned commit is reused as is.
"""

from __future__ import annotations

import difflib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from robotforge.design.templates.base import BuiltRobot
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.geometry import collides, lowest_point

log = logging.getLogger(__name__)

SPAWN_CLEARANCE = 0.002  # m between the lowest point and the floor, so nothing starts inside it
FETCH_TIMEOUT = 900  # s for one robot folder over a slow connection
FETCH_ATTEMPTS = 3  # proxies and flaky Wi-Fi drop GitHub connections now and then


class LibraryError(ValueError):
    """User-facing: says what went wrong and what to do about it."""


@dataclass(frozen=True)
class LibraryRobot:
    name: str
    title: str
    kind: Literal["quadruped", "arm", "humanoid"]
    summary: str
    maker: str
    license: str  # of the Menagerie model folder (see its LICENSE file)
    folder: str  # Menagerie directory
    mjcf: str  # model file inside it
    home_key: str | None = "home"  # keyframe that becomes our home pose
    # Torque-motor models get position servos with this stiffness (N*m/rad), keeping their torque
    # limits: that is how the real robots are commanded (joint targets + PD in the motor driver).
    servo_kp: float | None = None


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
            "g1",
            "Unitree G1",
            "humanoid",
            "1.3 m humanoid with 29 joints; stands in place (no walking yet).",
            "Unitree Robotics",
            "BSD-3-Clause",
            "unitree_g1",
            "g1.xml",
            home_key="stand",
        ),
    ]
}


def get(name: str) -> LibraryRobot:
    try:
        return CATALOG[name]
    except KeyError:
        close = difflib.get_close_matches(name, list(CATALOG), n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise LibraryError(f"no robot {name!r} in the library.{hint} Choose one of: {', '.join(CATALOG)}") from None


def catalog() -> list[dict[str, Any]]:
    """Library robots for agents: what they are, their license and whether they are downloaded yet."""
    return [
        {
            "name": r.name,
            "title": r.title,
            "kind": r.kind,
            "summary": r.summary,
            "maker": r.maker,
            "license": r.license,
            "downloaded": _cached_folder(r.folder) is not None,
        }
        for r in CATALOG.values()
    ]


# ---- fetching from Menagerie ----------------------------------------------------------


def _menagerie_pin() -> tuple[str, str]:
    """(url, commit) of Menagerie as pinned by the installed robot_descriptions."""
    from robot_descriptions._repositories import REPOSITORIES

    repo = REPOSITORIES["mujoco_menagerie"]
    return repo.url, repo.commit


def _cache_root() -> Path:
    return Path(os.environ.get("ROBOT_DESCRIPTIONS_CACHE", "~/.cache/robot_descriptions")).expanduser()


def _sparse_root(commit: str) -> Path:
    return _cache_root() / "robotforge" / f"mujoco_menagerie-{commit[:12]}"


def _cached_folder(folder: str) -> Path | None:
    """Where a Menagerie folder already is on disk, without downloading anything."""
    override = os.environ.get("ROBOTFORGE_MENAGERIE")
    if override:
        path = Path(override).expanduser() / folder
        return path if path.is_dir() else None
    _, commit = _menagerie_pin()
    full = _cache_root() / "mujoco_menagerie"
    head = full / ".git" / "HEAD"
    if (full / folder).is_dir() and head.is_file() and head.read_text().strip() == commit:
        return full / folder
    sparse = _sparse_root(commit) / folder
    return sparse if sparse.is_dir() else None


def model_dir(robot: LibraryRobot) -> Path:
    """The robot's Menagerie folder, downloading it on first use."""
    found = _cached_folder(robot.folder)
    if found is not None:
        return found
    if os.environ.get("ROBOTFORGE_MENAGERIE"):
        raise LibraryError(f"{robot.folder} is not in ROBOTFORGE_MENAGERIE={os.environ['ROBOTFORGE_MENAGERIE']}")
    url, commit = _menagerie_pin()
    target = _sparse_root(commit) / robot.folder
    _sparse_fetch(url, commit, robot.folder, target)
    return target


def _sparse_fetch(url: str, commit: str, folder: str, target: Path) -> None:
    """Shallow, sparse checkout of one folder, moved into place atomically when complete."""
    git = shutil.which("git")
    if git is None:
        raise LibraryError("downloading library robots needs git; install it from https://git-scm.com and try again")
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s from MuJoCo Menagerie (first use only)", folder)
    # Cleanup errors are ignored because git marks pack files read-only on Windows.
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".fetch-", ignore_cleanup_errors=True) as tmp:
        _git(git, ["init", "-q", tmp])
        _git(git, ["-C", tmp, "remote", "add", "origin", url])
        _git(git, ["-C", tmp, "sparse-checkout", "set", "--no-cone", f"/{folder}/"])
        # Both of these talk to GitHub: the checkout downloads the folder's files (a partial clone).
        _git(git, ["-C", tmp, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit], FETCH_ATTEMPTS)
        _git(git, ["-C", tmp, "checkout", "-q", "FETCH_HEAD"], FETCH_ATTEMPTS)
        fetched = Path(tmp) / folder
        if not fetched.is_dir():
            raise LibraryError(f"Menagerie at {commit[:12]} has no folder {folder!r}")
        try:
            fetched.rename(target)
        except OSError:
            if not target.is_dir():  # another process finishing first is fine; anything else is not
                raise


def _git(git: str, args: list[str], attempts: int = 1) -> None:
    for attempt in range(1, attempts + 1):
        try:
            # Output is captured: on stdio MCP servers stdout belongs to the protocol.
            done = subprocess.run([git, *args], capture_output=True, text=True, timeout=FETCH_TIMEOUT, check=False)
        except subprocess.TimeoutExpired as exc:
            raise LibraryError("downloading the robot timed out; check your internet connection and try again") from exc
        if done.returncode == 0:
            return
        detail = " / ".join((done.stderr or done.stdout).strip().splitlines()[-3:])
        if attempt < attempts:
            log.warning("git failed (attempt %d of %d), retrying: %s", attempt, attempts, detail)
            time.sleep(2.0 * attempt)
    raise LibraryError(
        "could not download the robot from GitHub (MuJoCo Menagerie); check your internet connection "
        f"or proxy. git said: {detail}"
    )


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
    key_qpos, key_ctrl = _read_key(spec, robot.home_key)
    notes = [f"{robot.title} from MuJoCo Menagerie ({robot.license}, {robot.maker})"]
    if robot.servo_kp is not None:
        converted = _motors_to_servos(spec, robot.servo_kp)
        if converted:
            notes.append(
                f"{converted} torque motors driven as position servos (kp {robot.servo_kp:g}, torque limits kept)"
            )
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
    if floating:
        # Place the base by its keyframe orientation; height comes from the lowest point below.
        root.pos = [0.0, 0.0, 0.0]
        if key_qpos is not None:
            root.quat = key_qpos[3:7].tolist()
    home_qpos, home_ctrl, spawn = _home_pose(spec, key_qpos, key_ctrl, floating)
    return BuiltRobot(
        spec=spec, home_qpos=home_qpos, home_ctrl=home_ctrl, spawn_height=spawn, floating=floating, notes=notes
    )


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
