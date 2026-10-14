"""reach: move an arm's tool point (its "tcp" site) to a target point in space."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams, home_qpos, smooth_abs


class ReachParams(TaskParams):
    target: list[float] | str | None = Field(
        None, description="[x, y, z] in meters, an object name, or empty for a random reachable point"
    )
    tolerance: float = Field(0.01, gt=0, le=0.2, description="success when the tool is this close, meters")


class Reach(Task):
    name = "reach"
    summary = "Move an arm's tool to a target point and hold it there."
    robot_hint = "an arm with a tool point named 'tcp' (any robot from the arm template)"
    methods = ("ik", "pd", "cartesian", "mpc", "mppi")
    Params = ReachParams
    default_seconds = 4.0
    control_hz = 50.0
    mpc_defaults = {"horizon": 0.4, "knots": 3, "samples": 32, "noise": 0.06}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: ReachParams
        self.site = self.robot.need(mujoco.mjtObj.mjOBJ_SITE, "tcp", self.robot_hint)
        self.site_sensor = self.robot.framepos_sensor(self.site)
        hinges = self.robot.hinge_joints()
        if not hinges:
            raise TaskError(f"robot {robot!r} has no hinge joints to move")
        joint_to_act = {int(model.actuator_trnid[a, 0]): int(a) for a in self.actuators}
        # Drive only the arm joints; a gripper does not help reach a point.
        self.joints = [j for j in hinges if j in joint_to_act]
        if not self.joints:
            raise TaskError(f"none of {robot!r}'s hinge joints has an actuator")
        self.actuators = np.array([joint_to_act[j] for j in self.joints])
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self._fk = mujoco.MjData(model)
        self.target = np.zeros(3)
        self._resolve_target(home_qpos(model), np.random.default_rng(0))

    # ---- target ----

    def _resolve_target(self, qpos: np.ndarray, rng: np.random.Generator) -> None:
        t = self.params.target
        if isinstance(t, list):
            if len(t) != 3:
                raise TaskError("reach target is [x, y, z] in meters")
            self.target = np.asarray(t, float)
        elif isinstance(t, str):
            self.target = self._object_position(t, qpos)
        else:
            self.target = self.random_target(qpos, rng)

    def _object_position(self, name: str, qpos: np.ndarray) -> np.ndarray:
        m = self.model
        self._fk.qpos[:] = qpos
        mujoco.mj_kinematics(m, self._fk)
        lookups = (
            (mujoco.mjtObj.mjOBJ_BODY, self._fk.xpos),
            (mujoco.mjtObj.mjOBJ_GEOM, self._fk.geom_xpos),  # fixed objects are plain geoms
            (mujoco.mjtObj.mjOBJ_SITE, self._fk.site_xpos),
        )
        for obj, table in lookups:
            idx = mujoco.mj_name2id(m, obj, name)
            if idx >= 0:
                return table[idx].copy()
        raise TaskError(f"no object named {name!r} to reach; give [x, y, z] or an object name")

    def random_target(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A point the tool can certainly reach: the tool position of a random joint configuration."""
        m, d = self.model, self._fk
        base = d.xpos[m.jnt_bodyid[self.joints[0]]]
        lo = np.maximum(m.jnt_range[self.joints, 0], -2.0)
        hi = np.minimum(m.jnt_range[self.joints, 1], 2.0)
        for _ in range(200):
            d.qpos[:] = qpos
            d.qpos[self.qadr] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[self.site].copy()
            base = d.xpos[m.jnt_bodyid[self.joints[0]]]
            if p[2] > base[2] + 0.03 and np.linalg.norm(p[:2] - base[:2]) > 0.1:
                return p
        return d.site_xpos[self.site].copy()

    def tool(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site].copy()

    def distance(self, data: mujoco.MjData) -> float:
        return float(np.linalg.norm(self.tool(data) - self.target))

    # ---- task interface ----

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        tool = self.tool(data)
        return np.concatenate(
            [data.qpos[self.qadr], data.qvel[self.dadr], tool, self.target, self.target - tool]
        ).astype(np.float32)

    def observation_labels(self) -> list[str]:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        return (
            [f"{n} angle" for n in names]
            + [f"{n} speed" for n in names]
            + [f"tool {a}" for a in "xyz"]
            + [f"target {a}" for a in "xyz"]
            + [f"to target {a}" for a in "xyz"]
        )

    def cost(self, batch: Batch) -> np.ndarray:
        tool = batch.site_pos(self.site, self.site_sensor)
        dist = np.linalg.norm(tool - self.target, axis=-1)
        speed = np.sum(batch.qvel[..., self.dadr] ** 2, axis=-1)
        return 10.0 * smooth_abs(dist, 0.02) + 0.02 * speed

    def terminal_cost(self, batch: Batch) -> np.ndarray:
        return 5.0 * self.cost(batch.last())[:, 0]

    def success(self, data: mujoco.MjData) -> bool:
        return self.distance(data) < self.params.tolerance

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        return {
            "distance_m": round(self.distance(data), 4),
            "tool": [round(float(v), 3) for v in self.tool(data)],
            "target": [round(float(v), 3) for v in self.target],
        }

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.params.target is None:
            self.target = self.random_target(data.qpos.copy(), rng)

    def begin(self, data: mujoco.MjData) -> None:
        if isinstance(self.params.target, str):  # the object may have moved since the task was made
            self.target = self._object_position(self.params.target, data.qpos.copy())

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.dadr[: min(3, len(self.dadr))]] += 2.0 * strength
        return f"knocked the first joints with {2.0 * strength:.1f} rad/s"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "target": [round(float(v), 3) for v in self.target]}
