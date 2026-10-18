"""hover: hold an aircraft still at a point in the air, and get back there after a push.

Three ways to fly it, the same task for each, so they can be compared:

- "cascade": robotforge's flight controller (behaviors/flight.py), the way real multirotors fly: position, then
  attitude, then body rates, with a mixer from the rotor layout.
- "lqr": one linear rule for every rotor at once. MuJoCo linearizes the aircraft hovering at the target (a step of
  physics as x' = A x + B u, with the motors' lag in it), and the Riccati equation turns how much each error
  matters into the gains. Nothing in it knows about tilting to move: it falls out of the physics.
- "mpc": predictive sampling, which tries out futures every 20 ms with the full model and keeps the best.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.behaviors.airframe import AirframeError, find_airframe
from robotforge.control.actuation import ActionScale
from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams

TAKEOFF = 1.5  # m above where it stands, the target of an aircraft on the ground
FALLEN = 0.0  # the body's up axis below the horizon: it has turned over
LOST = 10.0  # m from the target: it has flown off
PUSH_SPEED = 1.5  # m/s sideways, the push of push_at (a gust's worth of a shove)
UP, DOWN = 0.3, 0.03  # m of clearance: up in the air, and back on the ground


class HoverParams(TaskParams):
    target: list[float] | None = Field(
        None, description="[x, y, z] to hold, m (default: where it hovers now, or 1.5 m above where it stands)"
    )


class Hover(Task):
    name = "hover"
    summary = "Hold an aircraft still at a point in the air, and come back to it after a push."
    robot_hint = (
        "an aircraft: the drone template, a library drone (skydio_x2, crazyflie) or a robot with rotor parts "
        "(methods cascade, lqr, mpc, none)"
    )
    methods = ("cascade", "lqr", "mpc", "none")
    Params = HoverParams
    default_seconds = 6.0
    control_hz = 100.0
    alive_bonus = 1.0
    mpc_defaults = {"horizon": 0.6, "knots": 4, "samples": 48, "noise": 0.12, "plan_dt": 0.005}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: HoverParams
        try:
            self.frame = find_airframe(model, robot)
        except AirframeError as exc:
            raise TaskError(f"{exc}; hover needs {self.robot_hint}") from exc
        self.actuators = np.array(self.frame.actuators, dtype=int)  # the rotors (not an arm it may carry)
        joint = int(model.body_jntadr[self.frame.root])
        self.qadr, self.dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        self.hover_ctrl = self.frame.mixer.mix(self.frame.weight, np.zeros(3)).controls
        self.span = np.maximum(self.frame.mixer.high - self.frame.mixer.low, 1e-9)
        self.target = np.array(self.params.target, float) if self.params.target is not None else None
        self._was_up = False
        if self.target is not None and self.target.shape != (3,):
            raise TaskError(f"target is [x, y, z] in meters, got {self.params.target}")

    # ---- where it should be ----

    def begin(self, data: mujoco.MjData) -> None:
        if self.params.target is None:
            here = data.xpos[self.frame.root].copy()
            airborne = (
                float(np.linalg.norm(data.qvel[self.dadr : self.dadr + 3])) > 0.05 or self._clearance(data) > 0.15
            )
            self.target = here if airborne else here + np.array([0.0, 0.0, TAKEOFF])
        # A copy of a live world (a comparison) keeps the air's last push; nothing pushes it here but the push test.
        data.xfrc_applied[self.frame.root, :3] = 0.0

    def _clearance(self, data: mujoco.MjData) -> float:
        from robotforge.behaviors.flight import ground_height

        ground = ground_height(self.model, data, data.xpos[self.frame.root], self.frame.descendants, self.frame.root)
        return float(data.xpos[self.frame.root][2] - (ground or 0.0) - self.frame.rest_height)

    def _target(self, data: mujoco.MjData | None = None) -> np.ndarray:
        if self.target is None:
            if data is None:
                raise TaskError("the hover task was not started (begin) before it was asked where to hover")
            self.begin(data)
        assert self.target is not None
        return self.target

    # ---- what the algorithms read ----

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        q = data.qpos[self.qadr + 3 : self.qadr + 7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, q)
        return np.concatenate(
            [
                data.qpos[self.qadr : self.qadr + 3] - self._target(data),
                data.qvel[self.dadr : self.dadr + 3],
                rot.reshape(3, 3)[2],  # which way is up, in the body's frame
                data.qvel[self.dadr + 3 : self.dadr + 6],
            ]
        ).astype(np.float32)

    def observation_labels(self) -> list[str]:
        return [
            *(f"offset {a}" for a in "xyz"),
            *(f"speed {a}" for a in "xyz"),
            *(f"up seen from the body {a}" for a in "xyz"),
            *(f"turn rate {a}" for a in "xyz"),
        ]

    def cost(self, batch: Batch) -> np.ndarray:
        target = self._target()
        p = batch.qpos[..., self.qadr : self.qadr + 3] - target
        x, y = batch.qpos[..., self.qadr + 4], batch.qpos[..., self.qadr + 5]
        tilt = 2.0 * (x * x + y * y)  # 1 - cos(tilt)
        v = batch.qvel[..., self.dadr : self.dadr + 3]
        w = batch.qvel[..., self.dadr + 3 : self.dadr + 6]
        du = (batch.ctrl[..., self.actuators] - self.hover_ctrl) / self.span
        return (
            4.0 * np.sum(p * p, axis=-1)
            + 2.0 * tilt
            + 0.3 * np.sum(v * v, axis=-1)
            + 0.02 * np.sum(w * w, axis=-1)
            + 0.05 * np.sum(du * du, axis=-1)
        )

    def failed(self, data: mujoco.MjData) -> bool:
        """Turned over, flown off, or back on the ground after it had been up (it fell)."""
        up = 1.0 - 2.0 * float(data.qpos[self.qadr + 4] ** 2 + data.qpos[self.qadr + 5] ** 2)
        if up < FALLEN or float(np.linalg.norm(data.xpos[self.frame.root] - self._target(data))) > LOST:
            return True
        height = self._clearance(data)
        self._was_up = self._was_up or height > UP
        return self._was_up and height < DOWN

    def success(self, data: mujoco.MjData) -> bool:
        off = float(np.linalg.norm(data.xpos[self.frame.root] - self._target(data)))
        return off < 0.1 and float(np.linalg.norm(data.qvel[self.dadr : self.dadr + 3])) < 0.2

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        up = 1.0 - 2.0 * float(data.qpos[self.qadr + 4] ** 2 + data.qpos[self.qadr + 5] ** 2)
        return {
            "position_error_m": round(float(np.linalg.norm(data.xpos[self.frame.root] - self._target(data))), 3),
            "height_m": round(float(data.xpos[self.frame.root][2]), 3),
            "tilt_deg": round(math.degrees(math.acos(float(np.clip(up, -1.0, 1.0)))), 1),
        }

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.dadr] += PUSH_SPEED * strength
        return f"shoved it sideways: it suddenly moves at {PUSH_SPEED * strength:.1f} m/s"

    def nominal_ctrl(self, data: mujoco.MjData) -> np.ndarray:
        return self.hover_ctrl.copy()

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        if self.target is not None:
            out["target"] = [round(float(v), 3) for v in self.target]
        return out

    # ---- the controllers that only fit aircraft (see control/methods.py) ----

    own_methods = ("cascade",)

    def own_controller(self, method: str, model: mujoco.MjModel, data: mujoco.MjData, **params: Any):
        if params:
            raise TaskError(f"cascade takes no settings (got {', '.join(params)}): its gains are tuned for any layout")
        from robotforge.behaviors.flight import FlightController

        fc = FlightController(model, self.robot.name)
        fc.hold_at(self._target(data))
        return fc

    def lqr_controller(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_weights: list[float] | None = None,
        r_weight: float = 1.0,
    ) -> HoverLQR:
        return HoverLQR(self, model, data, q_weights, r_weight)


class HoverLQR:
    """ "lqr" for hover: u = u_hover - K (x - x_hover) on the full state of the aircraft (see the module notes)."""

    def __init__(
        self,
        task: Hover,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        q_weights: list[float] | None = None,
        r_weight: float = 1.0,
    ):
        from robotforge.control.classic import solve_dare

        self.task = task
        target = task._target(data)
        acts = task.actuators
        dofs = np.arange(task.dadr, task.dadr + 6)
        acts_with_lag = [int(model.actuator_actadr[a]) for a in acts if model.actuator_actadr[a] >= 0]
        self.qpos_eq = data.qpos.copy()
        self.qpos_eq[task.qadr : task.qadr + 3] = target
        q = data.qpos[task.qadr + 3 : task.qadr + 7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        self.qpos_eq[task.qadr + 3 : task.qadr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.ctrl_eq = task.hover_ctrl.copy()
        lin = mujoco.MjData(model)
        lin.qpos[:] = self.qpos_eq
        lin.ctrl[:] = data.ctrl
        lin.ctrl[acts] = self.ctrl_eq
        for a, eq in zip(acts, self.ctrl_eq, strict=True):
            if model.actuator_actadr[a] >= 0:
                lin.act[model.actuator_actadr[a]] = eq  # the motors spun up to hover
        mujoco.mj_forward(model, lin)
        nx = 2 * model.nv + model.na
        A, B = np.zeros((nx, nx)), np.zeros((nx, model.nu))
        mujoco.mjd_transitionFD(model, lin, 1e-6, True, A, B, None, None)
        self.idx = np.concatenate([dofs, model.nv + dofs, 2 * model.nv + np.array(acts_with_lag, dtype=int)])
        self.A, self.B = A[np.ix_(self.idx, self.idx)], B[np.ix_(self.idx, acts)]
        weights = (
            np.asarray(q_weights, float)
            if q_weights is not None
            else np.array(
                [20.0, 20.0, 40.0, 30.0, 30.0, 5.0, 4.0, 4.0, 6.0, 0.5, 0.5, 0.2] + [0.0] * len(acts_with_lag)
            )
        )
        if weights.shape != (len(self.idx),) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise TaskError(
                f"q_weights needs {len(self.idx)} numbers >= 0: position (3), rotation (3), speed (3), turn rate (3)"
                + (f", motor lag ({len(acts_with_lag)})" if acts_with_lag else "")
            )
        if not 0 < r_weight <= 1e6:
            raise TaskError(f"r_weight must be above 0, got {r_weight}")
        scale = np.diag(1.0 / task.span)  # rotor commands measured as a share of their range
        self.Q = np.diag(weights)
        self.R = r_weight * scale @ scale
        P = solve_dare(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R + self.B.T @ P @ self.B, self.B.T @ P @ self.A)
        closed = np.abs(np.linalg.eigvals(self.A - self.B @ self.K))
        if not (np.all(np.isfinite(self.K)) and closed.max() < 1.0):
            raise TaskError("LQR found no gains that hold this aircraft with these weights; try other q_weights")
        self.scale = ActionScale(model, acts)
        self.acts = acts
        self.act_adr = np.array(acts_with_lag, dtype=int)
        self.lag_eq = self.ctrl_eq[np.array([model.actuator_actadr[a] >= 0 for a in acts], dtype=bool)]
        self.rate_hz = 1.0 / model.opt.timestep
        self._dq = np.zeros(model.nv)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_differentiatePos(model, self._dq, 1.0, self.qpos_eq, data.qpos)
        d = self.task.dadr
        x = np.concatenate([self._dq[d : d + 6], data.qvel[d : d + 6], data.act[self.act_adr] - self.lag_eq])
        data.ctrl[self.acts] = self.scale.clip(self.ctrl_eq - self.K @ x)

    def status(self) -> dict[str, Any]:
        return {"controller": "lqr", "states": len(self.idx), "rotors": len(self.acts)}
