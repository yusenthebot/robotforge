"""Dexterous hands from MuJoCo Menagerie (LEAP, Allegro, Shadow) as end effectors an arm's gripper commands work.

An arm holds with one "gripper" actuator: move_arm, gripper and pick_and_place open it at the top of its range and
close it at the bottom. A hand has sixteen to twenty joints, so the fingers that grasp are driven together by one
tendon instead, the way underactuated grippers are built: a fixed tendon runs over each grasping joint with a weight
in proportion to how far that joint travels from open to closed, a position actuator on it is the gripper, and a
spring in each of those joints holds the hand open and shares the closing out among them. A finger that touches the
object stops while the others carry on closing round it, and the squeeze is the actuator's force limit spread over
them. The joints that only shape the hand (spreading the fingers, turning the thumb to face them) keep their own
servos, held at a grasp preshape.

Each hand is turned so that it mounts like the gripper of an arm: its mounting face on the origin, pointing along +z
away from the arm, with its grasp point ("tcp") where the fingers meet. LEAP mounts palm down by the back of its palm
(as it ships on xArm and UR arms), Allegro by its wrist, Shadow by the base of its forearm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

CLOSE_TAU = 0.25  # s: the gripper's target eases toward what it is told with this time constant
SETTLE_RATIO = 20.0  # the tendon servo's stiffness over the springs': closing goes 95% of the way when nothing is held


@dataclass(frozen=True)
class Hand:
    """How a hand from the library grasps: which joints close and how far, and where it mounts."""

    flex: tuple[tuple[str, float, float], ...]  # (joint, open, closed) of the joints the gripper closes
    hold: tuple[tuple[str, float], ...]  # (joint, angle) the shaping joints keep with their own servos
    mount_pos: tuple[float, float, float]  # where the hand's root body goes, in the mounting frame (m)
    mount_quat: tuple[float, float, float, float]  # and how it is turned there
    tcp: tuple[float, float, float]  # the grasp point in the mounting frame (m)
    spring: float  # N*m/rad of each closing joint's spring (holds the open hand up against gravity)
    squeeze: float  # N*m the most-travelling joint gets at the actuator's force limit


def fit(spec: mujoco.MjSpec, hand: Hand) -> dict[str, float]:
    """Turn `spec` (a Menagerie hand) into an end effector: mounted on its face, its closing joints on one tendon
    with a "gripper" servo, a "tcp" site at its grasp point. Returns the home controls (open, shaped)."""
    root = spec.worldbody.first_body()
    root.pos, root.quat = list(hand.mount_pos), list(hand.mount_quat)
    joints = {j.name: j for j in spec.joints}
    missing = [n for n, *_ in hand.flex if n not in joints] + [n for n, _ in hand.hold if n not in joints]
    if missing:
        raise ValueError(f"the hand model has no joints {', '.join(missing)}")
    closing = {n for n, *_ in hand.flex}
    for actuator in list(spec.actuators):  # the closing joints' own servos go: the tendon moves them
        if actuator.target in closing or actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON:
            spec.delete(actuator)
    for tendon in list(spec.tendons):  # Shadow's coupled finger joints: a closing tendon of our own replaces them
        spec.delete(tendon)
    travel = np.array([closed - opened for _, opened, closed in hand.flex])
    scale = float(np.linalg.norm(travel))
    tendon = spec.add_tendon(name="grasp")
    for (name, opened, _), delta in zip(hand.flex, travel, strict=True):
        joint = joints[name]
        joint.stiffness = [hand.spring, 0.0, 0.0]
        joint.springref = opened
        tendon.wrap_joint(name, float(-delta / scale))
    length_open = float(sum(-d / scale * o for d, (_, o, _) in zip(travel, hand.flex, strict=True)))
    # Past fully closed, so a hand that has closed round something keeps squeezing it.
    length_closed = length_open - scale * 1.3
    servo = spec.add_actuator(name="gripper", target="grasp", trntype=mujoco.mjtTrn.mjTRN_TENDON)
    kp = SETTLE_RATIO * hand.spring
    servo.set_to_position(kp=kp, kv=0.05 * kp)
    # The target eases to where it is told (a first-order filter), so the fingers and the thumb close together: told
    # at once, the fingers swept a cube 16 cm along the floor before the thumb came round to meet them.
    servo.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
    servo.dynprm[0] = CLOSE_TAU
    servo.actlimited = mujoco.mjtLimited.mjLIMITED_FALSE
    servo.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    servo.ctrlrange = [length_closed, length_open]
    force = hand.squeeze * scale / float(np.max(np.abs(travel)))
    servo.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    servo.forcerange = [-force, force]
    _tcp_site(spec, root, hand)
    home = {"gripper": length_open}
    for actuator in spec.actuators:
        if actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT and actuator.target in dict(hand.hold):
            home[actuator.name] = dict(hand.hold)[actuator.target]
    return home


def open_pose(hand: Hand) -> dict[str, float]:
    """The hand's joints open, as it starts: the closing joints at their open angles, the shaping ones held."""
    return {name: opened for name, opened, _ in hand.flex} | dict(hand.hold)


def _tcp_site(spec: mujoco.MjSpec, root: mujoco.MjsBody, hand: Hand) -> None:
    """The grasp point, on the root body: the mounting frame's point `hand.tcp` in the root's own frame."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(hand.mount_quat, float))
    local = rot.reshape(3, 3).T @ (np.asarray(hand.tcp, float) - np.asarray(hand.mount_pos, float))
    quat = np.zeros(4)  # pointing along the mounting frame's +z, as every gripper's tool point does
    mujoco.mju_negQuat(quat, np.asarray(hand.mount_quat, float))
    root.add_site(name="tcp", pos=local.tolist(), quat=quat.tolist(), size=[0.006, 0, 0], rgba=[1, 0.6, 0.1, 0.6])


def _rot(axis: tuple[float, float, float], degrees: float) -> tuple[float, float, float, float]:
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, float), math.radians(degrees))
    return tuple(float(v) for v in quat)


LEAP = Hand(
    # Found by a search (kinematic screening, then picking blocks, balls and cans in simulation): open, the thumb is
    # raised beside the palm's wrist end and the fingers lie almost straight; closing, the fingers curl down and back
    # while the thumb swings down to meet them, 0.14 m under the back of the palm.
    flex=(
        *((f"{f}_mcp", 0.078, 1.283) for f in ("if", "mf", "rf")),
        *((f"{f}_pip", 0.232, 0.698) for f in ("if", "mf", "rf")),
        *((f"{f}_dip", 0.199, 0.477) for f in ("if", "mf", "rf")),
        ("th_cmc", 0.816, 1.54),
        ("th_axl", 0.545, 0.258),
        ("th_mcp", -0.07, 0.92),
        ("th_ipl", 0.24, -0.485),
    ),
    hold=(("if_rot", 0.0), ("mf_rot", 0.0), ("rf_rot", 0.0)),
    mount_pos=(0.035, -0.03, 0.012),
    mount_quat=(0.0, 1.0, 0.0, 0.0),
    tcp=(0.0047, -0.0164, 0.1427),
    spring=0.4,
    squeeze=1.5,
)


ALLEGRO = Hand(
    # Found the same way: the thumb turned to face the fingers, which curl in under the palm to meet it 0.1 m under the
    # back of the palm.
    flex=(
        *((f"{f}j1", -0.121, 0.696) for f in ("ff", "mf", "rf")),
        *((f"{f}j2", -0.155, 1.43) for f in ("ff", "mf", "rf")),
        *((f"{f}j3", 0.229, 0.399) for f in ("ff", "mf", "rf")),
        ("thj0", 1.137, 1.371),
        ("thj1", 0.894, 0.267),
        ("thj2", 0.901, 1.094),
        ("thj3", 1.41, 0.583),
    ),
    hold=(("ffj0", 0.0), ("mfj0", 0.0), ("rfj0", 0.0)),
    mount_pos=(0.047, 0.0, 0.029),
    mount_quat=(0.0, 0.7071068, 0.0, 0.7071068),
    tcp=(0.0605, -0.0181, 0.0996),
    spring=0.3,
    squeeze=1.0,
)
