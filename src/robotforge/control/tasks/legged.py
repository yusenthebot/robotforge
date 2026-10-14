"""stand and walk: tasks for robots with a floating base and legs (quadrupeds, later humanoids).

Everything is read straight from the root free joint: height and tilt from its
position and quaternion, speed from its velocity. That keeps the costs cheap
enough to score every state of every MPC rollout.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams, home_qpos


class _Legged(Task):
    robot_hint = "a legged robot that moves freely, e.g. one from the quadruped template"

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        root = self.robot.free_joint()
        if root < 0:
            raise TaskError(f"robot {robot!r} is fixed to the world; {self.name} needs {self.robot_hint}")
        self.qa, self.da = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
        self.trunk = int(model.jnt_bodyid[root])
        joint_to_act = {int(model.actuator_trnid[a, 0]): int(a) for a in self.actuators}
        self.joints = [j for j in self.robot.hinge_joints() if j in joint_to_act]
        self.joint_qadr = np.array([model.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.joint_dadr = np.array([model.jnt_dofadr[j] for j in self.joints], dtype=int)
        home = home_qpos(model)
        self.home_joints = home[self.joint_qadr]
        # Template robots spawn 5 mm above the floor and settle a little under gravity.
        self.height = float(self.params.height) if self.params.height else float(home[self.qa + 2]) - 0.01
        self.anchor = home[self.qa : self.qa + 2].copy()
        self.start_time = 0.0

    # ---- trunk quantities, vectorized over [..., nq] / [..., nv] arrays ----

    def _quat(self, qpos: np.ndarray) -> tuple[np.ndarray, ...]:
        q = qpos[..., self.qa + 3 : self.qa + 7]
        return q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    def up_z(self, qpos: np.ndarray) -> np.ndarray:
        """1 when the trunk is level, 0 on its side, -1 upside down."""
        _, x, y, _ = self._quat(qpos)
        return 1.0 - 2.0 * (x * x + y * y)

    def heading(self, qpos: np.ndarray) -> np.ndarray:
        w, x, y, z = self._quat(qpos)
        return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

    def forward_lateral(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.heading(qpos)
        vx, vy = qvel[..., self.da], qvel[..., self.da + 1]
        c, s = np.cos(h), np.sin(h)
        return c * vx + s * vy, -s * vx + c * vy

    def _posture_cost(self, batch: Batch) -> np.ndarray:
        return np.sum((batch.qpos[..., self.joint_qadr] - self.home_joints) ** 2, axis=-1)

    # ---- task interface ----

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        qpos, qvel = data.qpos, data.qvel
        rot = data.xmat[self.trunk].reshape(3, 3)
        gravity_in_body = rot.T @ np.array([0.0, 0.0, -1.0])
        lin_body = rot.T @ qvel[self.da : self.da + 3]
        return np.concatenate(
            [
                [qpos[self.qa + 2]],
                gravity_in_body,
                lin_body,
                qvel[self.da + 3 : self.da + 6],
                qpos[self.joint_qadr] - self.home_joints,
                qvel[self.joint_dadr],
                self._extra_obs(),
            ]
        ).astype(np.float32)

    def _extra_obs(self) -> np.ndarray:
        return np.zeros(0)

    def observation_labels(self) -> list[str]:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        return (
            ["trunk height"]
            + [f"gravity in body {a}" for a in "xyz"]
            + [f"body speed {a}" for a in "xyz"]
            + [f"body spin {a}" for a in "xyz"]
            + [f"{n} angle from home" for n in names]
            + [f"{n} speed" for n in names]
        )

    def failed(self, data: mujoco.MjData) -> bool:
        return bool(data.qpos[self.qa + 2] < 0.5 * self.height or self.up_z(data.qpos) < 0.5)

    def begin(self, data: mujoco.MjData) -> None:
        self.anchor = data.qpos[self.qa : self.qa + 2].copy()
        self.start_time = float(data.time)

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        data.qpos[self.joint_qadr] += rng.uniform(-0.05, 0.05, size=len(self.joint_qadr))

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        h = float(self.heading(data.qpos))
        side = np.array([-np.sin(h), np.cos(h)])
        data.qvel[self.da : self.da + 2] += 0.6 * strength * side
        return f"shoved the body sideways at {0.6 * strength:.1f} m/s"

    def _pose_report(self, data: mujoco.MjData) -> dict[str, Any]:
        return {
            "height_m": round(float(data.qpos[self.qa + 2]), 3),
            "upright": round(float(self.up_z(data.qpos)), 3),
            "position": [round(float(v), 3) for v in data.qpos[self.qa : self.qa + 3]],
        }


class StandParams(TaskParams):
    height: float | None = Field(None, gt=0, le=2.0, description="trunk height to hold in meters; default: home")


class Stand(_Legged):
    name = "stand"
    summary = "Stay upright at a chosen height without drifting, and recover from shoves."
    methods = ("hold", "mpc", "mppi")
    Params = StandParams
    default_seconds = 5.0
    control_hz = 50.0
    alive_bonus = 2.0
    mpc_defaults = {"horizon": 0.35, "knots": 3, "samples": 32, "noise": 0.05}

    def cost(self, batch: Batch) -> np.ndarray:
        qpos, qvel = batch.qpos, batch.qvel
        height = qpos[..., self.qa + 2] - self.height
        drift = qpos[..., self.qa : self.qa + 2] - self.anchor
        lin = qvel[..., self.da : self.da + 3]
        ang = qvel[..., self.da + 3 : self.da + 6]
        return (
            100.0 * height**2
            + 5.0 * (1.0 - self.up_z(qpos))
            + 5.0 * np.sum(drift**2, axis=-1)
            + 1.0 * np.sum(lin**2, axis=-1)
            + 0.05 * np.sum(ang**2, axis=-1)
            + 0.1 * self._posture_cost(batch)
        )

    def success(self, data: mujoco.MjData) -> bool:
        return not self.failed(data) and abs(data.qpos[self.qa + 2] - self.height) < 0.03

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        drift = float(np.linalg.norm(data.qpos[self.qa : self.qa + 2] - self.anchor))
        return {**self._pose_report(data), "target_height_m": round(self.height, 3), "drift_m": round(drift, 3)}


class WalkParams(TaskParams):
    speed: float = Field(0.3, ge=-1.0, le=1.5, description="forward speed to walk at, m/s (negative = backwards)")
    turn: float = Field(0.0, ge=-2.0, le=2.0, description="turn rate, rad/s (positive = left)")
    height: float | None = Field(None, gt=0, le=2.0, description="trunk height to keep in meters; default: home")


class Walk(_Legged):
    name = "walk"
    summary = "Walk forward at a commanded speed (and turn rate) without falling."
    methods = ("gait", "mpc", "mppi")
    Params = WalkParams
    default_seconds = 6.0
    control_hz = 50.0
    alive_bonus = 2.0
    mpc_defaults = {"horizon": 0.4, "knots": 4, "samples": 48, "noise": 0.12}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: WalkParams
        self.start = np.zeros(2)

    def _extra_obs(self) -> np.ndarray:
        return np.array([self.params.speed, self.params.turn])

    def observation_labels(self) -> list[str]:
        return [*super().observation_labels(), "commanded speed", "commanded turn"]

    def begin(self, data: mujoco.MjData) -> None:
        super().begin(data)
        self.start = data.qpos[self.qa : self.qa + 2].copy()

    def cost(self, batch: Batch) -> np.ndarray:
        qpos, qvel = batch.qpos, batch.qvel
        fwd, lat = self.forward_lateral(qpos, qvel)
        yaw_rate = qvel[..., self.da + 5]
        height = qpos[..., self.qa + 2] - self.height
        return (
            5.0 * (fwd - self.params.speed) ** 2
            + 2.0 * lat**2
            + 0.5 * (yaw_rate - self.params.turn) ** 2
            + 50.0 * height**2
            + 5.0 * (1.0 - self.up_z(qpos))
            + 0.02 * self._posture_cost(batch)
        )

    def success(self, data: mujoco.MjData) -> bool:
        if self.failed(data):
            return False
        elapsed = max(1e-6, data.time - self.start_time)
        speed = self._progress(data) / elapsed
        return elapsed > 1.0 and abs(speed - self.params.speed) <= 0.5 * max(abs(self.params.speed), 0.1)

    def _progress(self, data: mujoco.MjData) -> float:
        """Distance covered along the starting heading direction (negative when walking backwards)."""
        moved = data.qpos[self.qa : self.qa + 2] - self.start
        return float(np.linalg.norm(moved)) * (1.0 if self.params.speed >= 0 else -1.0)

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        elapsed = max(1e-6, data.time - self.start_time)
        dist = self._progress(data)
        return {**self._pose_report(data), "distance_m": round(dist, 3), "avg_speed_m_s": round(dist / elapsed, 3)}
