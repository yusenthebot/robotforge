"""What a catalogue robot is and what it can do, read from its structure rather than from what its makers tagged it.

`finish` runs when the robot is built: it puts the arm's "tcp" (and "flange") sites on a fixed-base arm, which is what
move_arm, pick_and_place and the mounting tools steer by. `describe` runs on the compiled world: it classifies the
robot (arm, quadruped, humanoid, wheeled, hand, aircraft, other) and says in one line what it is and which of the
tools work on it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.design import catalogue
from robotforge.design.templates.base import BuiltRobot

log = logging.getLogger(__name__)

MIN_ARM_JOINTS = 3  # a chain with fewer movable joints does not reach in space
MAX_ARMS = 2
MIN_ARM_REACH = 0.2  # m of links from the first joint to the wrist: shorter chains are fingers
HAND_JOINTS = 5  # a branch of the wrist with up to this many joints is a finger, not another arm
TCP_RGBA = [1.0, 0.6, 0.1, 0.6]  # the tool point, as on our own arms
MOVING = frozenset({int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)})  # joints that move


@dataclass(frozen=True)
class Flange:
    body: int
    joints: int  # movable joints from the base to here
    base_y: float  # sideways position of the arm's first body, to name two arms left and right


def finish(built: BuiltRobot, robot: catalogue.Entry | str) -> BuiltRobot:
    """Give a fixed-base arm the tool point and flange the arm tools steer by (a robot from any file: `robot` is the
    catalogue entry or just a name for the log). One that has a tool point already is left as its file draws it."""
    if built.floating or any(site.name.endswith("tcp") for site in built.spec.sites):
        return built
    name = robot if isinstance(robot, str) else robot.id
    try:
        model = built.spec.compile()
    except ValueError as exc:  # the file itself is checked by the world's compile, which says what is wrong
        log.debug("catalogue: %s not compiled for its tool point: %s", name, exc)
        return built
    flanges = arm_flanges(model, qpos=model.qpos0)
    if not flanges or len(flanges) > MAX_ARMS:  # three or more reaching chains are fingers or legs, not arms
        return built
    flanges.sort(key=lambda f: f.base_y)
    names = [""] if len(flanges) == 1 else [f"{side}/" for side in ("right", "left")] if len(flanges) == 2 else None
    if names is None:
        names = [f"arm{i + 1}/" for i in range(len(flanges))]
    for flange, prefix in zip(flanges, names, strict=True):
        _add_tool_sites(built.spec, model, flange, prefix)
    arms = "arm" if len(flanges) == 1 else f"{len(flanges)} arms"
    built.notes.append(f"found its {arms}: tool point 'tcp' and mounting 'flange' added at the wrist")
    return built


def arm_flanges(model: mujoco.MjModel, qpos: np.ndarray) -> list[Flange]:
    """The wrists of a fixed-base robot: where a serial chain of motor-driven joints ends in a hand, a gripper or
    nothing."""
    powered = {
        int(model.actuator_trnid[a, 0])
        for a in range(model.nu)
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
    }
    own_joints = [
        sum(int(model.jnt_type[j]) in MOVING and j in powered for j in _joints(model, b)) for b in range(model.nbody)
    ]  # joints with a motor: an arm moves by its motors, a hand's tendon-driven fingers do not make it one
    joints_in = np.array(own_joints)  # in the subtree of each body, its own included
    for b in range(model.nbody - 1, 0, -1):
        joints_in[model.body_parentid[b]] += joints_in[b]
    children: dict[int, list[int]] = {}
    for b in range(1, model.nbody):
        children.setdefault(int(model.body_parentid[b]), []).append(b)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)

    found: list[Flange] = []

    def walk(body: int, so_far: int, first: int | None) -> None:
        moving = [c for c in children.get(body, []) if joints_in[c] > 0]
        long_enough = first is not None and _length(model, first, body) >= MIN_ARM_REACH
        if not moving:
            if so_far >= MIN_ARM_JOINTS and long_enough:
                found.append(Flange(body, so_far, float(data.xpos[first][1])))
            return
        if len(moving) >= 2 and so_far >= MIN_ARM_JOINTS and all(joints_in[c] <= HAND_JOINTS for c in moving):
            if long_enough:
                found.append(Flange(body, so_far, float(data.xpos[first][1])))
            return
        for c in moving:
            walk(c, so_far + own_joints[c], first if first is not None else (c if own_joints[c] else None))

    for root in children.get(0, []):
        if joints_in[root] and not any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in _joints(model, root)):
            walk(root, own_joints[root], root if own_joints[root] else None)
    return found


def _length(model: mujoco.MjModel, first: int, last: int) -> float:
    """How long the links from body `first` down to body `last` are laid end to end (an arm folded at home is as
    long as when it is stretched)."""
    total = 0.0
    while last not in (first, 0):
        total += float(np.linalg.norm(model.body_pos[last]))
        last = int(model.body_parentid[last])
    return total


def _joints(model: mujoco.MjModel, body: int) -> range:
    return range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))


def _add_tool_sites(spec: mujoco.MjSpec, model: mujoco.MjModel, flange: Flange, prefix: str) -> None:
    """A "tcp" site pointing the way the wrist points, and a bare "flange" site where a gripper can be mounted."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body = flange.body
    rot = data.xmat[body].reshape(3, 3)
    axis_world = _wrist_axis(model, data, body)
    axis = rot.T @ axis_world
    quat = np.zeros(4)
    mujoco.mju_quatZ2Vec(quat, axis)
    reach = _hand_reach(model, data, body, axis_world)
    node = spec.body(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
    node.add_site(
        name=f"{prefix}tcp", pos=(axis * reach).tolist(), quat=quat.tolist(), size=[0.006, 0, 0], rgba=TCP_RGBA
    )
    if reach == 0.0:  # a bare wrist: a gripper from the library mounts here
        node.add_site(name=f"{prefix}flange", pos=[0.0, 0.0, 0.0], quat=quat.tolist(), size=[0.004, 0, 0], group=5)


def _wrist_axis(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    """The way the tool points, in the world: from the last joint's anchor to the wrist, or the wrist's own z."""
    parent = int(model.body_parentid[body])
    while parent > 0 and not any(int(model.jnt_type[j]) in MOVING for j in _joints(model, parent)):
        parent = int(model.body_parentid[parent])
    anchor = data.xpos[parent] if parent > 0 else data.xpos[body]
    for j in reversed([j for b in (body, parent) if b > 0 for j in _joints(model, b)]):
        if int(model.jnt_type[j]) in MOVING:
            anchor = data.xanchor[j]
            break
    direction = data.xpos[body] - anchor
    if np.linalg.norm(direction) < 1e-3:
        direction = data.xmat[body].reshape(3, 3)[:, 2]
    return direction / np.linalg.norm(direction)


def _hand_reach(model: mujoco.MjModel, data: mujoco.MjData, body: int, axis: np.ndarray) -> float:
    """How far the geoms below the wrist reach along the tool axis: the grasp point is out where fingers close."""
    below = {b for b in range(body, model.nbody) if _descends(model, b, body)}
    reach = 0.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in below and model.geom_contype[g] + model.geom_conaffinity[g] > 0:
            tip = float((data.geom_xpos[g] - data.xpos[body]) @ axis) + float(np.max(model.geom_size[g])) * 0.5
            reach = max(reach, tip)
    return round(reach * 0.85, 4) if below != {body} and reach > 0.02 else 0.0


def _descends(model: mujoco.MjModel, body: int, top: int) -> bool:
    while body > 0:
        if body == top:
            return True
        body = int(model.body_parentid[body])
    return False
