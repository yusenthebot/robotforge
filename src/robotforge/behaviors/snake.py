"""Serpentine gait for snake robots: a wave of bends runs from head to tail and pushes the body forward.

A snake robot is a chain of segments joined by side-to-side (yaw) servos. Real snakes
and snake robots move because their skin or little passive wheels slide easily along
the body but not sideways; a bending wave travelling backwards then pushes against
the ground and the body glides forward. The wave is a sine: every joint follows the
same swing, a little later than the one in front of it (Hirose's serpenoid curve).
Speed sets how fast the wave travels; turning adds a steady bend, steered by the
direction from the tail to the head.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.sim.actuators import drives_joint, is_position_servo

MIN_SEGMENTS = 4
AMPLITUDE = math.radians(40)  # each joint's swing on a snake of up to TUNED_SEGMENTS joints
TUNED_SEGMENTS = 8
WAVES = 1.0  # bends along the body at once
SPEED_PER_HZ = 0.2  # m/s per wave per second, measured on an 8-segment snake with passive wheels
TUNED_SPINE = 0.56  # m from its first yaw joint to its last (seven 8 cm segments)
MAX_HZ = 1.4
HEADING_GAIN = 0.8  # rad of steady bend per rad of heading error
MAX_BIAS = 0.35  # rad
SMOOTHING = 0.5  # s


def find_spine(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None) -> list[int] | None:
    """The yaw joints of a snake robot, head to tail, or None if `robot` is not shaped like one.

    A snake is a floating robot whose longest chain of powered hinges with upright axes (at the home
    pose) has at least four joints.
    """
    prefix = f"{robot}/"
    bodies = [b for b in range(1, model.nbody) if (_name(model, b)).startswith(prefix)]
    roots = [b for b in bodies if model.body_parentid[b] == 0]
    if len(roots) != 1:
        return None
    root = roots[0]
    free = range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])
    if not any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in free):
        return None
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if home_qpos is None else home_qpos
    mujoco.mj_kinematics(model, data)
    powered = {
        int(model.actuator_trnid[a, 0])
        for a in range(model.nu)
        if drives_joint(model, a) and is_position_servo(model, a)
    }

    def yaw_joints(body: int) -> list[int]:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        return [
            j
            for j in range(start, start + count)
            if j in powered and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and abs(data.xaxis[j][2]) > 0.9
        ]

    parents = {int(model.body_parentid[b]) for b in bodies}
    best: list[int] = []
    for leaf in (b for b in bodies if b not in parents):
        path, body = [], leaf
        while body != root:
            path = yaw_joints(body) + path
            body = int(model.body_parentid[body])
        if len(path) > len(best):
            best = path
    return best if len(best) >= MIN_SEGMENTS else None


def _spine_length(model: mujoco.MjModel, joints: list[int], home_qpos: np.ndarray) -> float:
    """Meters from the first joint of the spine to the last, along it."""
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    anchors = data.xanchor[joints]
    return float(np.sum(np.linalg.norm(np.diff(anchors, axis=0), axis=1)))


def _name(model: mujoco.MjModel, body: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""


class SnakeGait:
    rate_hz = 50.0

    def __init__(self, model: mujoco.MjModel, robot: str, home_qpos: np.ndarray):
        joints = find_spine(model, robot, home_qpos)
        if joints is None:
            raise ValueError(f"robot {robot!r} is not a snake (a chain of at least {MIN_SEGMENTS} side-to-side servos)")
        self.robot = robot
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}
        self.actuators = np.array([joint_to_act[j] for j in joints])
        self.head = int(model.body_rootid[model.jnt_bodyid[joints[0]]])
        self.tail = int(model.jnt_bodyid[joints[-1]])
        limits = np.array([model.jnt_range[j] for j in joints])
        # Hirose's serpenoid: a joint swings 2 a sin(pi waves / n) for a body winding a either way. More joints at the
        # same swing wind the body into loops: a 12-joint snake went 29 to 56 % as far as an 8-joint one. Past the
        # tuned count each joint swings less, so the body winds as the tuned snake's does: without it some 10 to 14
        # joint snakes went backwards, with it all went at least a third of the way (2026-10-19 sweep).
        share = math.sin(math.pi * WAVES / len(joints)) / math.sin(math.pi * WAVES / TUNED_SEGMENTS)
        swing = AMPLITUDE * min(1.0, share)
        self.amplitude = float(min(swing, 0.8 * np.min(np.abs(limits))))
        self.offsets = -2 * math.pi * WAVES * np.arange(len(joints)) / len(joints)
        # One wave runs along the whole body, so each wave carries a snake about a share of its own length: a
        # 0.6-size snake went a third as far as asked and a 12-segment one up to three times (2026-10-19 sweep).
        self.speed_per_hz = SPEED_PER_HZ * _spine_length(model, joints, home_qpos) / TUNED_SPINE
        self.max_speed = self.speed_per_hz * MAX_HZ
        self.command_speed = self.command_turn = 0.0
        self.speed = self.turn = 0.0
        self._phase = 0.0
        self._level = 0.0  # 0 = lying straight, 1 = full wave
        self._heading_ref: float | None = None
        self._last_t: float | None = None

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        self.command_speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.command_turn = float(np.clip(turn, -1.0, 1.0))

    def status(self) -> dict:
        return {
            "behavior": "slither",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }

    def heading(self, data: mujoco.MjData) -> float:
        """Which way the body as a whole points: from the tail to the head (the head itself swings)."""
        d = data.xpos[self.head] - data.xpos[self.tail]
        return math.atan2(d[1], d[0])

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = float(data.time)
        dt = 0.0 if self._last_t is None or t < self._last_t else t - self._last_t
        self._last_t = t
        alpha = 1.0 - math.exp(-dt / SMOOTHING)
        self.speed += alpha * (self.command_speed - self.speed)
        self.turn += alpha * (self.command_turn - self.turn)
        moving = abs(self.command_speed) > 1e-3 or abs(self.command_turn) > 1e-3
        self._level += alpha * ((1.0 if moving else 0.0) - self._level)
        # A snake turns by curving while it slithers, so turning on the spot still needs a (slow) wave.
        hz = max(abs(self.speed) / self.speed_per_hz, 0.6 * abs(self.turn))
        direction = -1.0 if self.speed < 0 else 1.0
        self._phase = (self._phase + direction * 2 * math.pi * min(hz, MAX_HZ) * dt) % (2 * math.pi)
        heading = self.heading(data)
        if self._heading_ref is None or not moving:
            self._heading_ref = heading
        self._heading_ref += self.turn * dt
        error = (self._heading_ref - heading + math.pi) % (2 * math.pi) - math.pi
        # Bending every joint the same way curves the body; a curve bent to the right (negative angles,
        # the tail swung left) steers the snake left.
        bias = -float(np.clip(HEADING_GAIN * error, -MAX_BIAS, MAX_BIAS)) * self._level
        targets = self._level * self.amplitude * np.sin(self._phase + self.offsets) + bias
        data.ctrl[self.actuators] = targets
