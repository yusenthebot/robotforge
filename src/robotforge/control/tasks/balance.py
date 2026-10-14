"""balance: keep a cart-pole's pole upright by pushing the cart, and keep the cart near the middle."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.control.tasks.base import Batch, Task, TaskParams

FALL_ANGLE = 0.8  # rad (about 46 degrees): past this the pole is considered dropped


class BalanceParams(TaskParams):
    start_angle: float = Field(0.0, ge=-0.6, le=0.6, description="pole tilt at the start of an episode, radians")
    randomize: float = Field(0.1, ge=0.0, le=0.5, description="random extra tilt at each env reset, radians")


class Balance(Task):
    name = "balance"
    summary = "Keep the pole of a cart-pole upright by moving the cart, and keep the cart near the middle."
    robot_hint = "a cart-pole (add_robot with template 'cartpole')"
    methods = ("lqr", "pd", "mpc", "mppi", "ars", "none")
    Params = BalanceParams
    default_seconds = 6.0
    control_hz = 100.0
    alive_bonus = 2.0
    mpc_defaults = {"horizon": 0.8, "knots": 4, "samples": 32, "noise": 0.3}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: BalanceParams
        mj = mujoco.mjtObj.mjOBJ_JOINT
        self.slider = self.robot.need(mj, "slider", self.robot_hint)
        self.hinge = self.robot.need(mj, "hinge", self.robot_hint)
        self.x_adr, self.th_adr = model.jnt_qposadr[self.slider], model.jnt_qposadr[self.hinge]
        self.xd_adr, self.thd_adr = model.jnt_dofadr[self.slider], model.jnt_dofadr[self.hinge]

    def state(self, data: mujoco.MjData) -> tuple[float, float, float, float]:
        """(cart position, pole angle from upright in (-pi, pi], cart speed, pole spin)."""
        th = float(np.arctan2(np.sin(data.qpos[self.th_adr]), np.cos(data.qpos[self.th_adr])))
        return float(data.qpos[self.x_adr]), th, float(data.qvel[self.xd_adr]), float(data.qvel[self.thd_adr])

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(self.state(data), dtype=np.float32)

    def observation_labels(self) -> list[str]:
        return ["cart position", "pole angle", "cart speed", "pole spin"]

    def cost(self, batch: Batch) -> np.ndarray:
        x, th = batch.qpos[..., self.x_adr], batch.qpos[..., self.th_adr]
        xd, thd = batch.qvel[..., self.xd_adr], batch.qvel[..., self.thd_adr]
        u = batch.ctrl[..., self.actuators[0]]
        return 5.0 * (1.0 - np.cos(th)) + 2.0 * x * x + 0.1 * xd * xd + 0.05 * thd * thd + 0.01 * u * u

    def failed(self, data: mujoco.MjData) -> bool:
        return abs(self.state(data)[1]) > FALL_ANGLE

    def success(self, data: mujoco.MjData) -> bool:
        return abs(self.state(data)[1]) < 0.1

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        x, th, _, _ = self.state(data)
        return {"pole_angle_deg": round(float(np.degrees(th)), 2), "cart_position_m": round(x, 3)}

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        r = self.params.randomize
        data.qpos[self.th_adr] = self.params.start_angle + rng.uniform(-r, r)
        data.qpos[self.x_adr] = rng.uniform(-0.1, 0.1)

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.thd_adr] += 1.5 * strength
        return f"shoved the pole: it suddenly spins at {1.5 * strength:.1f} rad/s"
