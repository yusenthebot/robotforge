"""drive_to: drive a wheeled robot to a goal point on the floor and stop there."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams, home_qpos, smooth_abs


class DriveToParams(TaskParams):
    goal: list[float] | str | None = Field(
        None, description="[x, y] in meters, an object name, or empty for a random point about 1 m away"
    )
    tolerance: float = Field(0.05, gt=0, le=1.0, description="success when this close to the goal, meters")


class DriveTo(Task):
    name = "drive_to"
    summary = "Drive a wheeled robot to a goal point on the floor and stop there."
    robot_hint = "a wheeled robot that moves freely, e.g. one from the rover template"
    methods = ("pd", "mpc", "mppi")
    Params = DriveToParams
    default_seconds = 6.0
    control_hz = 20.0
    mpc_defaults = {"horizon": 1.0, "knots": 4, "samples": 32, "noise": 0.25, "plan_dt": 0.02}
    goal_prefixes = ("goal",)  # observation entries that are the goal (kept when only sensors give the rest)

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: DriveToParams
        root = self.robot.free_joint()
        if root < 0:
            raise TaskError(f"robot {robot!r} is fixed to the world; drive_to needs {self.robot_hint}")
        self.qa, self.da = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
        self.left = self.robot.find(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel")
        self.right = self.robot.find(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel")
        if self.left < 0 or self.right < 0:
            raise TaskError(f"robot {robot!r} has no left_wheel/right_wheel motors; drive_to needs {self.robot_hint}")
        self.actuators = np.array([self.left, self.right])
        self.goal = np.zeros(2)
        self._fresh_goal = False  # set by reset(): an episode already drew its random goal
        self._resolve_goal(home_qpos(model), np.random.default_rng(0))

    def _resolve_goal(self, qpos: np.ndarray, rng: np.random.Generator) -> None:
        g = self.params.goal
        if isinstance(g, list):
            if len(g) not in (2, 3):
                raise TaskError("drive_to goal is [x, y] in meters")
            self.goal = np.asarray(g[:2], float)
        elif isinstance(g, str):
            self.goal = self._object_xy(g, qpos)
        else:
            angle = rng.uniform(-np.pi, np.pi)
            self.goal = qpos[self.qa : self.qa + 2] + rng.uniform(0.6, 1.2) * np.array([np.cos(angle), np.sin(angle)])

    def _object_xy(self, name: str, qpos: np.ndarray) -> np.ndarray:
        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, data)
        for obj, table in ((mujoco.mjtObj.mjOBJ_BODY, data.xpos), (mujoco.mjtObj.mjOBJ_GEOM, data.geom_xpos)):
            idx = mujoco.mj_name2id(self.model, obj, name)
            if idx >= 0:
                return table[idx][:2].copy()
        raise TaskError(f"no object named {name!r} to drive to; give [x, y] or an object name")

    def _heading(self, qpos: np.ndarray) -> np.ndarray:
        q = qpos[..., self.qa + 3 : self.qa + 7]
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))

    def position(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qa : self.qa + 2].copy()

    def distance(self, data: mujoco.MjData) -> float:
        return float(np.linalg.norm(self.position(data) - self.goal))

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        pos = self.position(data)
        h = float(self._heading(data.qpos))
        c, s = np.cos(h), np.sin(h)
        to_goal = self.goal - pos
        goal_body = np.array([c * to_goal[0] + s * to_goal[1], -s * to_goal[0] + c * to_goal[1]])
        v = data.qvel[self.da : self.da + 2]
        return np.array(
            [*pos, c, s, c * v[0] + s * v[1], data.qvel[self.da + 5], *goal_body, np.linalg.norm(to_goal)],
            dtype=np.float32,
        )

    def observation_labels(self) -> list[str]:
        return [
            "x",
            "y",
            "heading cos",
            "heading sin",
            "forward speed",
            "turn rate",
            "goal ahead",
            "goal left",
            "goal distance",
        ]

    def cost(self, batch: Batch) -> np.ndarray:
        pos = batch.qpos[..., self.qa : self.qa + 2]
        dist = np.linalg.norm(pos - self.goal, axis=-1)
        speed2 = np.sum(batch.qvel[..., self.da : self.da + 2] ** 2, axis=-1)
        # Moving is fine far from the goal, but the robot should arrive slowly and stop.
        near = np.exp(-dist / 0.15)
        return 5.0 * smooth_abs(dist, 0.05) + 2.0 * near * speed2

    def terminal_cost(self, batch: Batch) -> np.ndarray:
        return 3.0 * self.cost(batch.last())[:, 0]

    def success(self, data: mujoco.MjData) -> bool:
        return self.distance(data) < self.params.tolerance

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        return {
            "distance_m": round(self.distance(data), 3),
            "position": [round(float(v), 3) for v in self.position(data)],
            "goal": [round(float(v), 3) for v in self.goal],
        }

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.params.goal is None:
            self._resolve_goal(data.qpos.copy(), rng)
            self._fresh_goal = True

    def begin(self, data: mujoco.MjData) -> None:
        if isinstance(self.params.goal, str):
            self.goal = self._object_xy(self.params.goal, data.qpos.copy())
        elif self.params.goal is None and not self._fresh_goal:
            # A live run without a goal: pick one about a meter from where the robot is now.
            self._resolve_goal(data.qpos.copy(), np.random.default_rng(0))
        self._fresh_goal = False

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.da + 5] += 3.0 * strength
        return f"spun the robot at {3.0 * strength:.1f} rad/s"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "goal": [round(float(v), 3) for v in self.goal]}
