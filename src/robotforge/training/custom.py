"""A task written as a reward function, for any robot.

    from robotforge.extend import task
    from robotforge.training.custom import reward_task

    @task("run_forward", summary="run along +x without falling", params={"speed": 0.5})
    def build(model, robot, params):
        return reward_task(
            model, robot, params,
            reward=lambda ctx: ctx.base_velocity()[0] * 2 - 0.01 * ctx.effort(),
            failed=lambda ctx: ctx.height() < 0.08 or ctx.up() < 0.5,
            success=lambda ctx: ctx.base_pos()[0] > 3.0,
            seconds=8,
        )

Then `train_policy(task="run_forward", robot="dog")` trains it with any algorithm. The observation is the robot's
joint angles and speeds, plus its base's height, tilt and speeds when it moves freely, plus whatever `observe`
returns (a list of numbers), plus the sensors on the robot as for any task. Planners (MPC) need a cost they can
score in bulk and so do not apply to a reward task.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams


class RewardContext:
    """What a reward, success or failure function sees: the robot's state, by the robot's own names."""

    def __init__(self, task: RewardTask) -> None:
        self.task = task
        self.model = task.model
        self.data: mujoco.MjData = None  # type: ignore[assignment]  # set before every call
        self.params = task.settings
        self.rng = np.random.default_rng(0)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def _find(self, kind: mujoco.mjtObj, name: str) -> int:
        for candidate in (self.task.robot.prefix + name, name):
            found = mujoco.mj_name2id(self.model, kind, candidate)
            if found >= 0:
                return found
        raise TaskError(f"no {kind.name.removeprefix('mjOBJ_').lower()} named {name!r} on {self.task.robot.name}")

    def body(self, name: str) -> np.ndarray:
        """World position of a body of the robot (or of the world), by name."""
        return self.data.xpos[self._find(mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def site(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self._find(mujoco.mjtObj.mjOBJ_SITE, name)].copy()

    def joint(self, name: str) -> float:
        j = self._find(mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qpos[self.model.jnt_qposadr[j]])

    def set_joint(self, name: str, value: float) -> None:
        """Put a joint at `value` (radians or meters): for a reset function that randomizes the start."""
        j = self._find(mujoco.mjtObj.mjOBJ_JOINT, name)
        self.data.qpos[self.model.jnt_qposadr[j]] = value

    def joint_velocity(self, name: str) -> float:
        j = self._find(mujoco.mjtObj.mjOBJ_JOINT, name)
        return float(self.data.qvel[self.model.jnt_dofadr[j]])

    def base_pos(self) -> np.ndarray:
        """Where the robot's base is (its free joint), or its first body when it is fixed."""
        return self.data.xpos[self.task.base_body].copy()

    def base_velocity(self) -> np.ndarray:
        """The base's speed along world x, y, z."""
        task = self.task
        if task.da < 0:
            return np.zeros(3)
        return self.data.qvel[task.da : task.da + 3].copy()

    def height(self) -> float:
        return float(self.base_pos()[2])

    def up(self) -> float:
        """1 when the base is level, 0 on its side, -1 upside down."""
        return float(self.data.xmat[self.task.base_body][8])

    def effort(self) -> float:
        """Sum of the commanded actuator values' squares: a cost for using the motors hard."""
        ctrl = self.data.ctrl[self.task.actuators]
        return float(np.sum(np.square(ctrl)))


Fn = Callable[[RewardContext], Any]


class RewardTask(Task):
    """A task whose reward, success and failure are functions of a RewardContext."""

    name = "custom"
    summary = "a task written as a reward function"
    robot_hint = "any robot with actuators"
    methods: tuple[str, ...] = ()
    Params = TaskParams
    control_hz = 50.0

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        settings: dict[str, Any] | None = None,
        *,
        reward: Fn,
        success: Fn | None = None,
        failed: Fn | None = None,
        reset: Callable[[RewardContext, np.random.Generator], None] | None = None,
        observe: Fn | None = None,
        labels: list[str] | None = None,
        seconds: float = 5.0,
        task_name: str = "custom",
        alive_bonus: float = 0.0,
    ) -> None:
        super().__init__(model, robot, None)
        self.settings = dict(settings or {})
        self.name = task_name
        self.default_seconds = float(seconds)
        self.alive_bonus = float(alive_bonus)
        self._reward, self._success, self._failed, self._reset = reward, success, failed, reset
        self._observe, self._observe_labels = observe, list(labels or [])
        root = self.robot.free_joint()
        self.da = int(model.jnt_dofadr[root]) if root >= 0 else -1
        self.qa = int(model.jnt_qposadr[root]) if root >= 0 else -1
        bodies = self.robot._names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        self.base_body = int(model.jnt_bodyid[root]) if root >= 0 else int(bodies[0])
        joints = [j for j in self.robot.hinge_joints()]
        slides = [j for j in self.robot.joints if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE]
        self.joints = joints + slides
        self.joint_q = np.array([model.jnt_qposadr[j] for j in self.joints], dtype=int)
        self.joint_d = np.array([model.jnt_dofadr[j] for j in self.joints], dtype=int)
        self.context = RewardContext(self)

    def _ctx(self, data: mujoco.MjData) -> RewardContext:
        self.context.data = data
        return self.context

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        ctx = self._ctx(data)
        parts = [data.qpos[self.joint_q], data.qvel[self.joint_d]]
        if self.da >= 0:
            parts += [[ctx.height(), ctx.up()], data.qvel[self.da : self.da + 6]]
        if self._observe is not None:
            parts.append(np.asarray(self._observe(ctx), dtype=float).ravel())
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts]).astype(np.float32)

    def observation_labels(self) -> list[str]:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}" for j in self.joints]
        names = [n.removeprefix(self.robot.prefix) for n in names]
        out = [f"{n} position" for n in names] + [f"{n} speed" for n in names]
        if self.da >= 0:
            out += ["base height", "base level", *(f"base {k}" for k in ("vx", "vy", "vz", "wx", "wy", "wz"))]
        return out + self._observe_labels

    def step_cost(self, data: mujoco.MjData) -> float:
        value = float(self._reward(self._ctx(data)))
        if not np.isfinite(value):
            raise TaskError("the reward function returned NaN or infinity")
        return -value

    def cost(self, batch: Batch) -> np.ndarray:
        raise TaskError("a task written as a reward function has no planner cost; train a policy for it instead")

    def success(self, data: mujoco.MjData) -> bool:
        return bool(self._success(self._ctx(data))) if self._success else False

    def failed(self, data: mujoco.MjData) -> bool:
        return bool(self._failed(self._ctx(data))) if self._failed else False

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        if self._reset is not None:
            self._reset(self._ctx(data), rng)

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        out: dict[str, Any] = {"reward_now": round(-self.step_cost(data), 4)}
        if self.da >= 0:
            out["height_m"] = round(self._ctx(data).height(), 3)
            out["position"] = [round(float(v), 3) for v in self._ctx(data).base_pos()]
        return out


def reward_task(
    model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None, **functions: Any
) -> RewardTask:
    """Build a RewardTask (see the module notes for the functions it takes)."""
    return RewardTask(model, robot, params, **functions)
