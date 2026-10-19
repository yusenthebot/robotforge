"""Repairs for MJCF files written by other people, found when the open catalogue was tried on real robots.

MuJoCo reads MJCF strictly, and robots in the wild break it in a few common ways. `read` opens a file and, when
MuJoCo refuses for a known reason, reads a repaired copy kept next to it (so meshes and includes still resolve) and
says what was done. `drive_motors` turns plain torque motors into position servos, which is how real robots are
commanded and the only way a robot made of torque motors stands up without a controller.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

REPAIRED_SUFFIX = ".robotforge.xml"
NUMBER_LIST = re.compile(r'(\w+)="([-+0-9.eE,\s]*,[-+0-9.eE,\s]*)"')  # a number list written with commas
INCLUDE = re.compile(r'<include\s+file="([^"]+)"')
RAD_ERROR = 0.25  # rad of error at which a servo gives its rated torque
DAMPING = 0.04  # servo damping as a share of its stiffness, where the joint has too little of its own
MAX_REPAIRS = 8  # files repaired for one robot (an include tree)


def read(path: Path) -> tuple[mujoco.MjSpec, list[str]]:
    """The spec of an MJCF file and what was repaired to read it. Raises ValueError as MuJoCo does when it cannot be."""
    try:
        return mujoco.MjSpec.from_file(str(path)), []
    except ValueError as exc:
        first = exc
    repaired = _repair(path)
    if not repaired:
        raise first
    copy = path.with_name(path.stem + REPAIRED_SUFFIX)
    try:
        return mujoco.MjSpec.from_file(str(copy)), repaired
    except ValueError:
        raise first from None


def _repair(path: Path) -> list[str]:
    """Write repaired copies of the file and of what it includes; what was repaired, or [] if nothing known applies."""
    done: list[str] = []
    todo, seen = [path], set()
    while todo and len(seen) < MAX_REPAIRS:
        source = todo.pop()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        text = source.read_text(encoding="utf-8", errors="replace")
        fixed, count = NUMBER_LIST.subn(lambda m: f'{m[1]}="{re.sub(r"[,\s]+", " ", m[2]).strip()}"', text)
        includes = [source.parent / name for name in INCLUDE.findall(text)]
        if count:
            done.append(f"{source.name}: {count} number lists written with commas")
        if includes:  # the copy must include the repaired copies of what it includes
            fixed = INCLUDE.sub(lambda m: m[0].replace(m[1], _copy_name(m[1])), fixed)
        if count or includes:
            source.with_name(source.stem + REPAIRED_SUFFIX).write_text(fixed, encoding="utf-8")
        todo += includes
    return done if done else []


def _copy_name(name: str) -> str:
    path = Path(name)
    return path.with_name(path.stem + REPAIRED_SUFFIX).as_posix() if path.suffix else name


def drive_motors(spec: mujoco.MjSpec) -> int:
    """Turn the torque motors on limited joints into position servos with the same torque limit, stiff enough to give
    that torque at 0.25 rad of error. Wheels (joints without a range) keep their torque motors; so does the last joint
    of a chain (a continuous wrist), but a joint without a range that carries limited joints (a thigh whose file
    gives hip and calf a range and itself none) is a limb joint and is driven over the full turn. Returns how many."""
    joints = {j.name: j for j in spec.joints}
    count = 0
    for act in spec.actuators:
        joint = joints.get(act.target)
        motor = (
            act.trntype == mujoco.mjtTrn.mjTRN_JOINT
            and act.gaintype == mujoco.mjtGain.mjGAIN_FIXED
            and act.biastype == mujoco.mjtBias.mjBIAS_NONE
            and act.dyntype == mujoco.mjtDyn.mjDYN_NONE
        )
        torque = float(np.abs(np.asarray(act.ctrlrange, float)).max() * abs(act.gear[0]))
        if not motor or joint is None or torque <= 0:
            continue
        if _limited(spec, joint):
            low, high = np.asarray(joint.range, float)
            if spec.compiler.degree and joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                low, high = np.radians([low, high])
        elif joint.type == mujoco.mjtJoint.mjJNT_HINGE and _carries_limited_joints(spec, joint):
            low, high = -np.pi, np.pi
        else:
            continue
        kp = torque / RAD_ERROR
        damping = float(np.asarray(joint.damping, float).ravel()[0])
        kv = DAMPING * kp if damping < 0.5 * DAMPING * kp else 0.0
        act.gear = [1, 0, 0, 0, 0, 0]
        act.set_to_position(kp=kp, kv=kv)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-torque, torque]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [low, high]
        count += 1
    return count


def _limited(spec: mujoco.MjSpec, joint: mujoco.MjsJoint) -> bool:
    if joint.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return False
    if joint.limited == mujoco.mjtLimited.mjLIMITED_TRUE:
        return True
    low, high = joint.range
    return joint.limited == mujoco.mjtLimited.mjLIMITED_AUTO and low < high


def _carries_limited_joints(spec: mujoco.MjSpec, joint: mujoco.MjsJoint) -> bool:
    """Whether a body further out on the chain than `joint` has a limited joint: `joint` is in a limb, not a wheel."""
    todo = list(joint.parent.bodies)
    own = [j for j in joint.parent.joints if j is not joint]
    if any(_limited(spec, j) for j in own):
        return True
    while todo:
        body = todo.pop()
        if any(_limited(spec, j) for j in body.joints):
            return True
        todo += body.bodies
    return False
