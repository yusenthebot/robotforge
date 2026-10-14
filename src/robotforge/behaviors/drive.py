"""Differential drive: turn a forward speed and turn rate into left/right wheel speeds."""

from __future__ import annotations

import math

import mujoco
import numpy as np


class DiffDrive:
    rate_hz = 50.0

    def __init__(self, model: mujoco.MjModel, robot: str):
        self.robot = robot

        def act(name: str) -> int:
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/{name}")
            if a < 0:
                raise ValueError(f"robot {robot!r} has no actuator {name!r}; is it a rover?")
            return a

        self.left, self.right = act("left_wheel"), act("right_wheel")
        left_body = model.body(f"{robot}/left_wheel").id
        right_body = model.body(f"{robot}/right_wheel").id
        self.track = float(np.linalg.norm(model.body_pos[left_body] - model.body_pos[right_body]))
        wheel_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == left_body)
        self.radius = float(model.geom_size[wheel_geom][0])
        self.max_wheel = float(model.actuator_ctrlrange[self.left][1])
        self.max_speed = self.max_wheel * self.radius
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

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        left = (self.speed - self.turn * self.track / 2) / self.radius
        right = (self.speed + self.turn * self.track / 2) / self.radius
        # If a wheel would exceed its limit, scale both so the turn radius is kept.
        scale = max(1.0, abs(left) / self.max_wheel, abs(right) / self.max_wheel)
        data.ctrl[self.left] = left / scale
        data.ctrl[self.right] = right / scale


def heading(data: mujoco.MjData, body: int) -> float:
    return math.atan2(data.xmat[body][3], data.xmat[body][0])
