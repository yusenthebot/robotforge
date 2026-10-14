"""Differential drive: turn a forward speed and turn rate into left/right wheel speeds."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.behaviors import find_wheels


class DiffDrive:
    rate_hz = 50.0

    def __init__(self, model: mujoco.MjModel, robot: str):
        self.robot = robot
        has_template_wheels = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/left_wheel") >= 0
        if has_template_wheels:
            self.left, self.right = (self._actuator(model, "left_wheel"),), (self._actuator(model, "right_wheel"),)
            self.left_signs = self.right_signs = (1.0,)
            left_body = model.body(f"{robot}/left_wheel").id
            right_body = model.body(f"{robot}/right_wheel").id
            self.track = float(np.linalg.norm(model.body_pos[left_body] - model.body_pos[right_body]))
            wheel_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == left_body)
            self.radius = float(model.geom_size[wheel_geom][0])
        else:  # imported or hand-built robots: any speed-controlled wheels on both sides
            wheels = find_wheels(model, robot)
            if wheels is None:
                raise ValueError(f"robot {robot!r} has no driven wheels on both sides; is it a rover?")
            self.left, self.right = wheels.left, wheels.right
            self.left_signs, self.right_signs = wheels.left_signs, wheels.right_signs
            self.track, self.radius = wheels.track, wheels.radius
        self.max_wheel = float(model.actuator_ctrlrange[self.left[0]][1])
        self.max_speed = self.max_wheel * self.radius
        self.speed = 0.0
        self.turn = 0.0

    def _actuator(self, model: mujoco.MjModel, name: str) -> int:
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.robot}/{name}")
        if a < 0:
            raise ValueError(f"robot {self.robot!r} has no actuator {name!r}; is it a rover?")
        return a

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

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        left = (self.speed - self.turn * self.track / 2) / self.radius
        right = (self.speed + self.turn * self.track / 2) / self.radius
        # If a wheel would exceed its limit, scale both so the turn radius is kept.
        scale = max(1.0, abs(left) / self.max_wheel, abs(right) / self.max_wheel)
        data.ctrl[list(self.left)] = np.array(self.left_signs) * (left / scale)
        data.ctrl[list(self.right)] = np.array(self.right_signs) * (right / scale)


def heading(data: mujoco.MjData, body: int) -> float:
    return math.atan2(data.xmat[body][3], data.xmat[body][0])
