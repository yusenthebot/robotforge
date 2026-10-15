"""RobotEnv: a Gym-style environment for any task on any robot, with no extra dependencies.

It follows the Gymnasium API exactly (reset(seed) -> (obs, info); step(action) ->
(obs, reward, terminated, truncated, info); observation_space / action_space with
low, high, shape, dtype, sample()), so reinforcement learning libraries can use it
through a one-line wrapper (see `robotforge.control.gym_compat.as_gymnasium`).

Actions are always in [-1, 1], one per actuator the task controls, mapped onto each
actuator's range. In "residual" mode the action is a correction added on top of a
classic controller (IK for reach, LQR for balance, the trot gait for walk...),
which lets a learner start from something that already roughly works.

Reward per step = (alive bonus - task cost) x step duration, so an episode's return
is the same "reward" the comparison table reports for classic controllers and MPC.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import ValidationError

from robotforge import wording
from robotforge.control.actuation import ActionScale
from robotforge.control.methods import make_controller
from robotforge.control.spaces import Box
from robotforge.control.tasks import Task, TaskError, make_task
from robotforge.control.tasks.base import divergences, readable
from robotforge.sim.session import SimSession

DEFAULT_ROBOTS: dict[str, tuple[str, str]] = {
    "reach": ("arm", "arm"),
    "balance": ("cartpole", "cartpole"),
    "stand": ("quadruped", "dog"),
    "walk": ("quadruped", "dog"),
    "drive_to": ("rover", "rover"),
}
MARKER = "goal_marker"
UNSTABLE_PENALTY = 10.0  # reward lost when the simulation diverges (ends the episode)


class _Residual:
    """Runs a classic controller and adds the agent's correction to what it commands."""

    def __init__(self, base, task: Task, scale: ActionScale):
        self.base, self.task, self.scale = base, task, scale
        self.rate_hz = base.rate_hz
        self.offset = np.zeros(len(task.actuators))
        self.base_ctrl: np.ndarray | None = None

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.base.update(model, data)
        self.base_ctrl = data.ctrl[self.task.actuators].copy()
        self.apply(data)

    def apply(self, data: mujoco.MjData) -> None:
        if self.base_ctrl is not None:
            data.ctrl[self.task.actuators] = self.scale.clip(self.base_ctrl + self.offset)

    def status(self) -> dict[str, Any]:
        return {"controller": "residual", "base": self.base.status()}


class RobotEnv:
    """One robot, one task, Gym-style. Owns its own physics (never touches the live world)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: Task,
        *,
        control_hz: float | None = None,
        episode_seconds: float | None = None,
        action_mode: Literal["direct", "residual"] = "direct",
        residual_scale: float = 0.2,
        render_mode: Literal["rgb_array"] | None = None,
    ):
        if action_mode not in ("direct", "residual"):
            raise TaskError("action_mode is 'direct' (actions are the commands) or 'residual' (corrections)")
        if render_mode not in (None, "rgb_array"):
            raise TaskError("render_mode can only be 'rgb_array' (or None)")
        self.task = task
        self.model = task.model
        self.data = mujoco.MjData(self.model)
        self.session = SimSession(model=self.model, data=self.data)
        self.action_mode = action_mode
        self.residual_scale = float(residual_scale)
        self.render_mode = render_mode
        dt = self.model.opt.timestep
        hz = float(control_hz or task.control_hz)
        if not 0 < hz <= 1.0 / dt:
            raise TaskError(f"control_hz must be in (0, {1.0 / dt:.0f}] for this model")
        self.substeps = max(1, round(1.0 / (hz * dt)))
        self.control_dt = self.substeps * dt
        self.episode_seconds = float(episode_seconds or 2 * task.default_seconds)
        self.max_steps = max(1, math.ceil(self.episode_seconds / self.control_dt - 1e-9))
        self.scale = ActionScale(self.model, task.actuators)
        self.action_space = Box(-1.0, 1.0, (len(task.actuators),), np.float32)
        self.np_random = np.random.default_rng()
        self._key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self._marker = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MARKER)
        self._residual: _Residual | None = None
        self._steps = 0
        self._renderer = None
        self._home()
        obs = task.observation(self.data)
        self.observation_space = Box(-np.inf, np.inf, obs.shape, np.float32)
        self.spec = None  # Gymnasium reads env.spec; we are not registered anywhere

    # ---- Gym API ----

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        self._home()
        self.task.reset(self.data, self.np_random)
        mujoco.mj_forward(self.model, self.data)
        self.task.begin(self.data)
        self._show_goal()
        for name in list(self.session.controllers):
            self.session.set_controller(name, None)
        self._residual = None
        if self.action_mode == "residual":
            base = make_controller(self.task.methods[0], self.task, self.model, self.data)
            self._residual = _Residual(base, self.task, self.scale)
            self.session.set_controller("base", self._residual)
        self._steps = 0
        return self.task.observation(self.data), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError("action contains NaN or infinity")
        a = np.clip(a, -1.0, 1.0)
        if self._residual is not None:
            self._residual.offset = a * self.residual_scale * self.scale.half
            self._residual.apply(self.data)
        else:
            self.data.ctrl[self.task.actuators] = self.scale.to_ctrl(a)
        before = divergences(self.data)
        self.session.step(self.substeps)
        self._steps += 1
        unstable = divergences(self.data) > before or not np.isfinite(self.data.qpos).all()
        if unstable:
            # The physics blew up (MuJoCo already reset it). End the episode from a valid home pose so
            # the observation is meaningful, with a penalty so blowing up is never a way out of costs.
            self._home()
            self._show_goal()
            obs, cost = self.task.observation(self.data), 0.0
            reward = -UNSTABLE_PENALTY
        else:
            obs, cost = self.task.observation(self.data), self.task.step_cost(self.data)
            reward = (self.task.alive_bonus - cost) * self.control_dt
        terminated = unstable or self.task.failed(self.data)
        truncated = not terminated and self._steps >= self.max_steps
        info = self._info(cost)
        if unstable:
            info["unstable"] = True
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        from robotforge.sim.render import SnapshotRenderer

        if self._renderer is None:
            self._renderer = SnapshotRenderer()
        return self._renderer.rgb(self.session, "default", 480, 360)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ---- helpers ----

    @property
    def unwrapped(self) -> RobotEnv:
        return self

    def _home(self) -> None:
        if self._key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _show_goal(self) -> None:
        if self._marker < 0:
            return
        goal = getattr(self.task, "target", None)
        if goal is None and hasattr(self.task, "goal"):
            goal = np.array([*self.task.goal, 0.005])
        if goal is not None:
            self.data.mocap_pos[self.model.body_mocapid[self._marker]] = goal
            mujoco.mj_forward(self.model, self.data)

    def _info(self, cost: float | None = None) -> dict[str, Any]:
        success = bool(self.task.success(self.data))
        # "is_success" is the key Stable-Baselines3's Monitor and EvalCallback read for success rates.
        info: dict[str, Any] = {"time": round(float(self.data.time), 4), "success": success, "is_success": success}
        if cost is not None:
            info["cost"] = cost
        return info

    def __repr__(self) -> str:
        return f"RobotEnv(task={self.task.name!r}, robot={self.task.robot.name!r}, mode={self.action_mode!r})"


def task_world(task: str, robot_template: str | None = None, robot_params: dict[str, Any] | None = None):
    """A small world for `task`: the default robot (or `robot_template`) on a floor, plus a goal marker.

    Returns (model, robot name)."""
    from robotforge.design.world import World

    if task not in DEFAULT_ROBOTS:
        raise TaskError(f"unknown task {wording.short(task)}; choose one of: {', '.join(DEFAULT_ROBOTS)}")
    template, robot = DEFAULT_ROBOTS[task]
    template = robot_template or template
    from robotforge.design import templates

    try:
        templates.get(template).Params.model_validate(robot_params or {})  # readable errors before compiling
        world = World.model_validate(
            {"name": f"{task}_lab", "robots": [{"name": robot, "template": template, "params": robot_params or {}}]}
        )
    except ValidationError as exc:
        raise TaskError(readable(exc, "robot settings")) from exc
    except ValueError as exc:  # e.g. an unknown template
        raise TaskError(str(exc)) from exc
    return compile_task_world(task, world), robot


def compile_task_world(task: str, world, base_dir=None) -> mujoco.MjModel:
    """Compile a design World for `task`, adding the goal marker that reach and drive_to show."""
    from robotforge.design.world import compile_world

    try:
        compiled = compile_world(world, base_dir)
    except ValidationError as exc:
        raise TaskError(readable(exc, "robot settings")) from exc
    except ValueError as exc:  # template parameters are checked while compiling
        raise TaskError(str(exc)) from exc
    spec = compiled.spec
    if task in ("reach", "drive_to"):
        marker = spec.worldbody.add_body(name=MARKER, mocap=True, pos=[0.3, 0.0, 0.2])
        size = [0.012, 0, 0] if task == "reach" else [0.05, 0.003, 0]
        shape = mujoco.mjtGeom.mjGEOM_SPHERE if task == "reach" else mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.add_geom(type=shape, size=size, rgba=[0.2, 0.9, 0.4, 0.6], contype=0, conaffinity=0, group=1)
    return spec.compile()


def make_env(
    task: str,
    *,
    robot_template: str | None = None,
    robot_params: dict[str, Any] | None = None,
    task_params: dict[str, Any] | None = None,
    model: mujoco.MjModel | None = None,
    robot: str | None = None,
    **env_kwargs: Any,
) -> RobotEnv:
    """Build an environment for a built-in task.

    With no model, a small world is made for the task (arm for reach, cart-pole for balance,
    quadruped for stand and walk, rover for drive_to). Pass `model` and `robot` to use any
    compiled world instead. Extra keywords go to RobotEnv (control_hz, episode_seconds,
    action_mode, residual_scale, render_mode)."""
    if model is None:
        model, default_robot = task_world(task, robot_template, robot_params)
        robot = robot or default_robot
    elif robot is None:
        raise TaskError("when passing a model, also say which robot in it (robot='name')")
    return RobotEnv(make_task(task, model, robot, task_params), **env_kwargs)
