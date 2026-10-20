"""A task saved with a world (design/tasks) as a learning task: the same Gym-style environment as the built-in tasks.

    from robotforge.control.tasks.world_task import world_env
    env = world_env("~/robotforge/worlds/home", "bring_red_cup")
    obs, info = env.reset(seed=0)            # the robot and the cup start where the task's randomization draws
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    info["success"]                          # the task's own success check (design/tasks/check.py)

The observation is the robot's middle and heading, its hand, the thing and the place (zeros where a task has none)
and the robot's joint angles and speeds. The cost is the check's distance still to go, minus a bonus while the task is
done, so an episode's return is the reward the check gives. An attempt fails when a robot that walks falls over.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np

from robotforge.control.tasks.base import Batch, Task, TaskError, TaskParams
from robotforge.design.tasks import check
from robotforge.design.tasks.live import Probe, randomize
from robotforge.design.tasks.spec import TaskEntry


class WorldTask(Task):
    name: ClassVar[str] = "world_task"
    summary: ClassVar[str] = "a task saved with a world: its own success check and reward"
    robot_hint: ClassVar[str] = "the task's robot"
    methods: ClassVar[tuple[str, ...]] = ()
    Params: ClassVar[type[TaskParams]] = TaskParams
    control_hz: ClassVar[float] = 25.0
    goal_prefixes: ClassVar[tuple[str, ...]] = ("hand_", "thing_", "place_")  # the goal (kept with sensors only)

    def __init__(self, model: mujoco.MjModel, entry: TaskEntry, world: Any):
        super().__init__(model, entry.robot, None)
        self.entry, self.world = entry, world
        self.default_seconds = entry.seconds  # type: ignore[misc]
        self.probe = Probe(model, entry, world)
        joints = [j for j in self.robot.joints if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                                       mujoco.mjtJoint.mjJNT_SLIDE)]  # fmt: skip
        self._qpos = np.array([model.jnt_qposadr[j] for j in joints], dtype=int)
        self._qvel = np.array([model.jnt_dofadr[j] for j in joints], dtype=int)
        reg = check.region(world, entry.target) if entry.target else None
        self._place = (
            np.array([reg.area.x, reg.area.y, 0.0 if math.isinf(reg.low) else reg.low]) if reg else np.zeros(3)
        )
        self._scratch = mujoco.MjData(model)
        self._start: check.State | None = None
        self.walks = entry.robot and self.probe.floating

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        s = self.probe.state(data)
        robot = np.array(s.robot) if s.robot else np.zeros(3)
        yaw = 0.0
        if self.probe.root > 0:
            m = data.xmat[self.probe.root]
            yaw = math.atan2(m[3], m[0])
        hand = np.array(s.hand) if s.hand else np.zeros(3)
        thing = np.array(s.things.get(self.entry.object, (0.0, 0.0, 0.0)))
        return np.concatenate(
            [
                robot,
                [math.cos(yaw), math.sin(yaw)],
                hand,
                thing,
                self._place,
                data.qpos[self._qpos],
                data.qvel[self._qvel],
            ]
        ).astype(np.float32)

    def observation_labels(self) -> list[str]:
        labels = ["robot_x", "robot_y", "robot_z", "heading_cos", "heading_sin", "hand_x", "hand_y", "hand_z"]
        labels += ["thing_x", "thing_y", "thing_z", "place_x", "place_y", "place_z"]
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}" for j in self.robot.joints]
        hinge = [n for n, j in zip(names, self.robot.joints, strict=True) if self.model.jnt_type[j] in (2, 3)]
        return labels + [f"{n}_pos" for n in hinge] + [f"{n}_vel" for n in hinge]

    def cost(self, batch: Batch) -> np.ndarray:
        b, t = batch.shape
        out = np.empty((b, t))
        d = self._scratch
        for i in range(b):
            for k in range(t):
                d.qpos[:] = batch.qpos[i, k]
                mujoco.mj_kinematics(self.model, d)
                v = check.judge(self.entry, self.world, self.probe.state(d), self._start)
                out[i, k] = v.distance - (check.SUCCESS_REWARD if v.done else 0.0)
        return out

    def step_cost(self, data: mujoco.MjData) -> float:
        v = self.verdict(data)
        return v.distance - (check.SUCCESS_REWARD if v.done else 0.0)

    def verdict(self, data: mujoco.MjData) -> check.Verdict:
        return check.judge(self.entry, self.world, self.probe.state(data), self._start)

    def success(self, data: mujoco.MjData) -> bool:
        return self.verdict(data).done

    def failed(self, data: mujoco.MjData) -> bool:
        s = self.probe.state(data)
        return bool(self.walks and s.upright is False)

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        return self.verdict(data).as_dict()

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        mujoco.mj_forward(self.model, data)
        randomize(self.model, data, self.entry, rng)

    def begin(self, data: mujoco.MjData) -> None:
        self._start = self.probe.state(data)

    def describe(self) -> dict[str, Any]:
        return {"task": self.entry.name, "robot": self.entry.robot, "goal": self.entry.describe()}


def world_env(world: str | Path, task: str | None = None, **env_kwargs: Any):
    """A RobotEnv for a task saved with a world (a world folder, its world.yaml, or a saved world's name). `task` is
    the task's name; the world's first task when left out. Extra keywords go to RobotEnv (control_hz,
    episode_seconds, render_mode); actions are direct (there is no classic controller to correct)."""
    from robotforge.control.env import RobotEnv
    from robotforge.design.store import WorldStore

    store = WorldStore.open(str(world))
    saved = store.load()
    if not saved.tasks:
        raise TaskError(f"world {saved.name!r} has no tasks; add one with add_task")
    names = [t.name for t in saved.tasks]
    entry = saved.tasks[0] if task is None else next((t for t in saved.tasks if t.name == task), None)
    if entry is None:
        raise TaskError(f"world {saved.name!r} has no task {task!r}; its tasks: {', '.join(names)}")
    model = store.compile().model
    env_kwargs.setdefault("episode_seconds", entry.seconds)
    return RobotEnv(WorldTask(model, entry, saved), **env_kwargs)
