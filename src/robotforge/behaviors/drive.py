"""Differential drive: turn a forward speed and turn rate into left/right wheel speeds."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.behaviors import WheelSet, find_wheels


def rover_drive(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray | None = None):
    """Share steering discovery between all runtime rover entry points."""
    from robotforge.behaviors.steering import SteeringDrive, steering_binding

    modules = steering_binding(model, robot, home_qpos)
    return SteeringDrive(model, robot, modules) if modules is not None else DiffDrive(model, robot)


def drive_binding(model: mujoco.MjModel, robot: str) -> WheelSet:
    """Resolve the same wheel geometry used by the differential drive, without creating it."""
    from robotforge.behaviors.steering import has_steering

    if has_steering(model, robot):
        raise ValueError(f"{robot}: wheels have steering joints; use the independent steering mover")
    has_template_wheels = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/left_wheel") >= 0
    if not has_template_wheels:
        wheels = find_wheels(model, robot)
        if wheels is None:
            raise ValueError(f"robot {robot!r} has no driven wheels on both sides; is it a rover?")
        return wheels
    actuators = []
    for name in ("left_wheel", "right_wheel"):
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/{name}")
        if a < 0:
            raise ValueError(f"robot {robot!r} has no actuator {name!r}; is it a rover?")
        actuators.append(a)
    left_body = model.body(f"{robot}/left_wheel").id
    right_body = model.body(f"{robot}/right_wheel").id
    track = float(np.linalg.norm(model.body_pos[left_body] - model.body_pos[right_body]))
    wheel_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == left_body)
    radius = float(model.geom_size[wheel_geom][0])
    return WheelSet((actuators[0],), (actuators[1],), (1.0,), (1.0,), track, radius)


def drive_speed(model: mujoco.MjModel, wheels: WheelSet) -> float:
    return float(model.actuator_ctrlrange[wheels.left[0]][1]) * wheels.radius


class DiffDrive:
    rate_hz = 50.0

    def __init__(self, model: mujoco.MjModel, robot: str):
        self.robot = robot
        wheels = drive_binding(model, robot)
        self.left, self.right = wheels.left, wheels.right
        self.left_signs, self.right_signs = wheels.left_signs, wheels.right_signs
        self.track, self.radius = wheels.track, wheels.radius
        self.max_wheel = float(model.actuator_ctrlrange[self.left[0]][1])
        self.max_speed = drive_speed(model, wheels)
        self.speed = 0.0
        self.turn = 0.0

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        self.speed = float(np.clip(speed, -self.max_speed, self.max_speed))
        self.turn = float(np.clip(turn, -4.0, 4.0))

    def status(self) -> dict:
        return {
            "behavior": "drive",
            "robot": self.robot,
            "speed": round(self.speed, 3),
            "turn": round(self.turn, 3),
            "max_speed": round(self.max_speed, 3),
        }

    def push_limit(self, model: mujoco.MjModel) -> float | None:
        """The most the wheels can push the robot with, in newtons (each motor's stall torque over the wheel
        radius), or None when a motor has no torque limit."""
        wheels = [*self.left, *self.right]
        if not all(model.actuator_forcelimited[a] for a in wheels):
            return None
        return float(sum(model.actuator_forcerange[a][1] for a in wheels) / self.radius)

    def spin(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """How fast the wheels turn as a fraction of the speed they are commanded (0 = stalled), 1 when idle."""
        wanted = abs(self.speed) / self.radius
        if wanted < 1e-6:
            return 1.0
        wheels = [*self.left, *self.right]
        turning = [abs(data.qvel[model.jnt_dofadr[model.actuator_trnid[a][0]]]) for a in wheels]
        return float(np.mean(turning) / wanted)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        left = (self.speed - self.turn * self.track / 2) / self.radius
        right = (self.speed + self.turn * self.track / 2) / self.radius
        # If a wheel would exceed its limit, scale both so the turn radius is kept.
        scale = max(1.0, abs(left) / self.max_wheel, abs(right) / self.max_wheel)
        data.ctrl[list(self.left)] = np.array(self.left_signs) * (left / scale)
        data.ctrl[list(self.right)] = np.array(self.right_signs) * (right / scale)


def heading(data: mujoco.MjData, body: int) -> float:
    return math.atan2(data.xmat[body][3], data.xmat[body][0])
