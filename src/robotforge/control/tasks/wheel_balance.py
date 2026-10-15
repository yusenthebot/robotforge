"""balance, two-wheeled version: keep a Segway-like robot upright on its wheels and near where it started.

The same "balance" task name as the cart-pole: make_task picks this one when the robot
is a two-wheeled balancer (see behaviors.balance). Costs read the floating base straight
from qpos, so MPC can score every predicted future in one numpy call.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.behaviors import balance
from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams, home_qpos


class WheelBalanceParams(TaskParams):
    start_angle: float = Field(0.0, ge=-0.4, le=0.4, description="lean at the start of an episode, radians")
    randomize: float = Field(0.05, ge=0.0, le=0.3, description="random extra lean at each env reset, radians")


class WheelBalance(Task):
    name = "balance"
    summary = "Keep a two-wheeled robot upright by rolling its wheels under it, and keep it near its spot."
    robot_hint = "a two-wheeled balancing robot (wheels on one axle, nothing else touching the floor)"
    methods = ("lqr", "mpc", "mppi", "none")
    Params = WheelBalanceParams
    default_seconds = 6.0
    control_hz = 100.0
    alive_bonus = 2.0
    mpc_defaults = {"horizon": 0.6, "knots": 4, "samples": 48, "noise": 0.15, "plan_dt": 0.005}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: WheelBalanceParams
        found = balance.find_balancer(model, robot, home_qpos(model))
        if found is None:
            raise TaskError(f"robot {robot!r} is not {self.robot_hint}")
        self.b = found
        self.qa, self.da = found.qadr, found.dadr
        self.lean = 0.0
        self._lean_known = False
        self.anchor = home_qpos(model)[self.qa : self.qa + 2].copy()
        self.top_speed = found.top_speed

    @staticmethod
    def fits(model: mujoco.MjModel, robot: str) -> bool:
        return balance.find_balancer(model, robot, home_qpos(model)) is not None

    def _balance_lean(self) -> float:
        """Lean (radians) at which the center of mass sits over the axle; found once, from the model."""
        if not self._lean_known:
            _, self.lean = balance.equilibrium(self.model, self.b, home_qpos(self.model))
            self._lean_known = True
        return self.lean

    # ---- vectorized trunk numbers ----

    def pitch(self, qpos: np.ndarray) -> np.ndarray:
        w, x, y, z = (qpos[..., self.qa + 3 + i] for i in range(4))
        zx, zy, zz = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)
        hx, hy = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z)
        norm = np.sqrt(hx * hx + hy * hy) + 1e-9
        return np.arctan2((zx * hx + zy * hy) / norm, zz)

    def state(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(lean from balance, lean rate, distance from the start spot)."""
        lean = float(self.pitch(data.qpos)) - self._balance_lean()
        drift = float(np.linalg.norm(data.qpos[self.qa : self.qa + 2] - self.anchor))
        return lean, float(data.qvel[self.da + 4]), drift

    # ---- task interface ----

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        lean, rate, _ = self.state(data)
        offset = data.qpos[self.qa : self.qa + 2] - self.anchor
        return np.array([lean, rate, *offset, *data.qvel[self.da : self.da + 2]], dtype=np.float32)

    def observation_labels(self) -> list[str]:
        return ["lean", "lean rate", "offset x", "offset y", "speed x", "speed y"]

    def cost(self, batch: Batch) -> np.ndarray:
        lean = self.pitch(batch.qpos) - self._balance_lean()
        offset = batch.qpos[..., self.qa : self.qa + 2] - self.anchor
        speed = batch.qvel[..., self.da : self.da + 2]
        spin = batch.qvel[..., self.da + 4]
        u = batch.ctrl[..., self.actuators] / max(self.top_speed, 1e-6)
        return (
            5.0 * (1.0 - np.cos(lean))
            + 2.0 * np.sum(offset**2, axis=-1)
            + 0.2 * np.sum(speed**2, axis=-1)
            + 0.02 * spin**2
            + 0.01 * np.sum(u**2, axis=-1)
        )

    def failed(self, data: mujoco.MjData) -> bool:
        return abs(self.state(data)[0]) > balance.FALL_ANGLE

    def success(self, data: mujoco.MjData) -> bool:
        lean, _, drift = self.state(data)
        return abs(lean) < 0.1 and drift < 0.25

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        lean, _, drift = self.state(data)
        return {"lean_deg": round(float(np.degrees(lean)), 2), "drift_m": round(drift, 3)}

    def begin(self, data: mujoco.MjData) -> None:
        self.anchor = data.qpos[self.qa : self.qa + 2].copy()

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        mujoco.mj_kinematics(self.model, data)
        r = self.params.randomize
        _, left = balance.heading_frame(data, self.b)
        balance.set_lean(
            self.model,
            data,
            self.b,
            balance.axle_point(data, self.b),
            left,
            self.params.start_angle + rng.uniform(-r, r),
        )

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.da + 4] += 1.5 * strength
        return f"shoved the body: it suddenly tips at {1.5 * strength:.1f} rad/s"

    def lqr_controller(self, model: mujoco.MjModel, data: mujoco.MjData, q_weights=None, r_weight=None):
        """The same LQR balancer `move` uses, with optional weights [lean, lean rate, position, speed]."""
        weights = balance.Q_WEIGHTS if q_weights is None else tuple(float(w) for w in q_weights)
        if len(weights) != 4 or any(not np.isfinite(w) or w < 0 for w in weights):
            raise TaskError("q_weights are 4 numbers >= 0: [lean, lean rate, position, speed]")
        r = balance.R_WEIGHT if r_weight is None else float(r_weight)
        if not (np.isfinite(r) and r > 0):
            raise TaskError("r_weight must be a number > 0")
        try:
            return balance.WheelBalancer(model, self.robot.name, home_qpos(model), self.b, weights, r)
        except ValueError as exc:
            raise TaskError(str(exc)) from exc
