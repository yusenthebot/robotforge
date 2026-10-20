"""One environment factory for any robot and any task.

    from robotforge.training import make_env

    env = make_env("walk", world=world, robot="dog")            # a robot of a world, a built-in task
    env = make_env("bring_red_cup", world="home", robot="arm")   # a task saved with a world (add_task)
    env = make_env("reach", robot={"name": "arm", "template": "arm", "sensors": [{"type": "imu"}]})

The observation is assembled from the sensors the robot actually carries (obs.py): `observation="auto"` adds
their channels to the task's own observation when there are any, "sensors" gives the task's goal plus the sensors
only (a robot that learns from what it can sense), "task" keeps the task's own observation. `env.layout` says
which numbers sit where, `describe(env)` documents both spaces.

The environment is a RobotEnv (Gym API, no gymnasium needed; control/gym_compat wraps it for libraries), so every
algorithm, built-in or from a plugin, sees the same interface. An EnvSpec is plain data: a training process
rebuilds the same environment from it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotforge import wording
from robotforge.control.env import DEFAULT_ROBOTS, RobotEnv, compile_task_world, refuse_contact_task
from robotforge.control.tasks import TASKS, Task, TaskError, make_task
from robotforge.control.tasks.base import readable
from robotforge.training.obs import DEPTH_SIZE, LIDAR_BINS, Block, SensorObservation

OBSERVATIONS = ("auto", "task", "task+sensors", "sensors")
Observation = Literal["auto", "task", "task+sensors", "sensors"]


class EnvSpec(BaseModel):
    """Everything needed to build the same environment again, in another process."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    task: str
    robot: str
    task_params: dict[str, Any] = Field(default_factory=dict)
    world: dict[str, Any]  # a World, as World.model_dump() writes it
    world_dir: str | None = None  # resolves relative files of the world
    scope: Literal["robot", "world"] = "robot"
    observation: Observation = "auto"
    lidar_bins: int = Field(LIDAR_BINS, ge=1, le=360)
    depth_size: list[int] = Field(default_factory=lambda: list(DEPTH_SIZE))
    action_mode: Literal["direct", "residual"] = "direct"
    residual_scale: float = Field(0.2, gt=0, le=1.0)
    control_hz: float | None = Field(None, gt=0)
    episode_seconds: float | None = Field(None, gt=0, le=600)
    push: float = Field(0.0, ge=0, le=5.0)


class ObservedTask:
    """A task whose observation is assembled from its own numbers and the robot's sensors.

    Everything else (cost, success, reset, push ...) is the wrapped task's, so MPC, comparisons and RobotEnv work
    on it as on the task itself."""

    def __init__(self, task: Task, mode: str, sensors: SensorObservation | None) -> None:
        self._task = task
        self.mode = mode
        self.sensors = sensors
        labels = list(task.observation_labels())
        if mode == "sensors":
            prefixes = tuple(getattr(task, "goal_prefixes", ()))
            self._task_index = np.array([i for i, label in enumerate(labels) if label.startswith(prefixes)], dtype=int)
            labels = [labels[i] for i in self._task_index]
            name = "task goal"
        else:
            self._task_index = None
            name = "task"
        self.blocks: list[Block] = []
        at = 0
        if mode != "sensors" or labels:
            self.blocks.append(Block(name, "task", 0, tuple(labels)))
            at = len(labels)
        if sensors is not None:
            for block in sensors.blocks:
                self.blocks.append(Block(block.source, block.kind, at + block.start, block.labels))
        self._labels = [
            f"{b.source}: {label}" if b.kind != "task" else label for b in self.blocks for label in b.labels
        ]

    def __getattr__(self, name: str) -> Any:
        if name == "_task":  # not built yet (copy, pickle): do not look for it in itself
            raise AttributeError(name)
        return getattr(self._task, name)

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        parts = []
        if self.blocks and self.blocks[0].kind == "task":
            own = np.asarray(self._task.observation(data), dtype=np.float64)
            parts.append(own if self._task_index is None else own[self._task_index])
        if self.sensors is not None:
            parts.append(self.sensors.vector(data))
        return np.concatenate(parts).astype(np.float32)

    def observation_labels(self) -> list[str]:
        return list(self._labels)

    def prime(self, data: mujoco.MjData) -> None:
        if self.sensors is not None:
            self.sensors.prime(data)


class TrainingEnv(RobotEnv):
    """RobotEnv for learning: sensor observations and, optionally, random shoves while it trains."""

    def __init__(self, task: Task | ObservedTask, *, push: float = 0.0, **kwargs: Any) -> None:
        self.push = float(push)
        self._next_shove = math.inf
        super().__init__(task, **kwargs)  # type: ignore[arg-type]

    @property
    def layout(self) -> list[dict[str, Any]]:
        """Where each source's numbers sit in the observation."""
        task = self.task
        if isinstance(task, ObservedTask):
            return [b.describe() for b in task.blocks]
        labels = tuple(task.observation_labels())
        return [Block("task", "task", 0, labels).describe()]

    def _schedule_shove(self) -> None:
        if self.push > 0:
            self._next_shove = float(self.data.time) + float(self.np_random.uniform(1.5, 4.0))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if isinstance(self.task, ObservedTask) and self.task.sensors is not None:
            self.task.prime(self.data)
            obs = self.task.observation(self.data)
        self._schedule_shove()
        return obs, info

    def step(self, action):
        if self.push > 0 and self.data.time >= self._next_shove:
            rng = self.np_random
            try:
                self.task.push(self.data, float(rng.uniform(0.3, 1.0) * self.push * rng.choice([-1.0, 1.0])))
            except TaskError:
                self.push = 0.0  # this task cannot be shoved: train without
            self._schedule_shove()
        return super().step(action)

    def close(self) -> None:
        task = self.task
        if isinstance(task, ObservedTask) and task.sensors is not None:
            task.sensors.close()
        super().close()


# ---- worlds and tasks ------------------------------------------------------------------------------------


def _as_world(world: Any) -> tuple[Any, Path | None]:
    """A World and the folder its relative files live in, from a World, a dict, or a saved world's name or path."""
    from robotforge.design.world import World

    if isinstance(world, World):
        return world, None
    if isinstance(world, dict):
        try:
            return World.model_validate(world), None
        except ValidationError as exc:
            raise TaskError(readable(exc, "world")) from exc
    from robotforge.design.store import WorldStore

    store = WorldStore.open(str(world))
    return store.load(), store.dir


def _default_world(task: str, robot: Any) -> tuple[dict[str, Any], str]:
    """A world for `robot` (a dict like a world.yaml robot entry, or a template or library name) alone on a floor."""
    from robotforge.design import templates
    from robotforge.design.library import CATALOG
    from robotforge.design.world import World

    if robot is None:
        if task not in DEFAULT_ROBOTS:
            raise TaskError(f"say which robot (robot=...) or which world it is in (world=...) for task {task!r}")
        template, name = DEFAULT_ROBOTS[task]
        entry: dict[str, Any] = {"name": name, "template": template}
    elif isinstance(robot, str):
        if robot in CATALOG:
            entry = {"name": robot, "library": robot}
        else:
            try:
                templates.get(robot)
            except (KeyError, ValueError) as exc:
                raise TaskError(
                    f"{wording.short(robot)} is neither a template nor a library robot; give a robot in a world "
                    "(world=...), or a robot entry as a dict"
                ) from exc
            entry = {"name": robot, "template": robot}
    elif isinstance(robot, dict):
        entry = dict(robot)
        if "name" not in entry:
            raise TaskError("a robot entry needs a name")
    else:
        raise TaskError("robot is a name, or a dict like a world.yaml robot entry")
    try:
        world = World.model_validate({"name": "training", "robots": [entry]})
    except ValidationError as exc:
        raise TaskError(readable(exc, "robot")) from exc
    return world.model_dump(mode="json"), entry["name"]


def _carried(world: Any, robot: str) -> list[Any]:
    """The robot's entry; what it carries (an arm on a dog) is part of the entry."""
    found = [r for r in world.robots if r.name == robot]
    if not found:
        raise TaskError(wording.unknown("robot", robot, sorted(r.name for r in world.robots)))
    return found


def _scoped(world: Any, robot: str, scope: str) -> Any:
    if scope == "world":
        return world
    return world.model_copy(update={"robots": _carried(world, robot), "objects": [], "tasks": []})


def env_spec(
    task: str,
    *,
    robot: Any = None,
    world: Any = None,
    world_dir: str | Path | None = None,
    scope: str | None = None,
    task_params: dict[str, Any] | None = None,
    observation: str = "auto",
    lidar_bins: int = LIDAR_BINS,
    depth_size: tuple[int, int] = DEPTH_SIZE,
    action_mode: str = "direct",
    residual_scale: float = 0.2,
    control_hz: float | None = None,
    episode_seconds: float | None = None,
    push: float = 0.0,
) -> EnvSpec:
    """The plain-data description of an environment (see make_env for the arguments)."""
    refuse_contact_task(task)
    if observation not in OBSERVATIONS:
        raise TaskError(f"observation is one of {', '.join(OBSERVATIONS)}, not {wording.short(observation)}")
    if world is None:
        data, name = _default_world(task, robot)
    else:
        found, opened = _as_world(world)
        world_dir = opened or world_dir  # a saved world's own folder wins
        data = found.model_dump(mode="json")
        if isinstance(robot, str):
            name = robot
        elif isinstance(robot, dict):
            raise TaskError("with a world, robot is the name of a robot in it")
        elif len(found.robots) == 1:
            name = found.robots[0].name
        elif robot is None and task in DEFAULT_ROBOTS and DEFAULT_ROBOTS[task][1] in [r.name for r in found.robots]:
            name = DEFAULT_ROBOTS[task][1]
        else:
            raise TaskError(
                f"say which robot to train; robots in this world: {', '.join(r.name for r in found.robots)}"
            )
    saved = {t["name"] for t in data.get("tasks", [])}
    chosen = scope or ("world" if task in saved else "robot")
    if task in saved and chosen != "world":
        raise TaskError(f"task {task!r} is saved with the world and judges its objects, so it needs scope='world'")
    try:
        return EnvSpec(
            task=task,
            robot=name,
            task_params=task_params or {},
            world=data,
            world_dir=str(world_dir) if world_dir else None,
            scope=chosen,  # type: ignore[arg-type]
            observation=observation,  # type: ignore[arg-type]
            lidar_bins=lidar_bins,
            depth_size=list(depth_size),
            action_mode=action_mode,  # type: ignore[arg-type]
            residual_scale=residual_scale,
            control_hz=control_hz,
            episode_seconds=episode_seconds,
            push=push,
        )
    except ValidationError as exc:
        raise TaskError(readable(exc, "environment settings")) from exc


def task_on(spec: EnvSpec, model: mujoco.MjModel, world: Any = None) -> Task:
    """The spec's task on `model`: a built-in one, a task saved with `world`, or a plugin's."""
    if spec.task in TASKS:
        return make_task(spec.task, model, spec.robot, spec.task_params)
    entry = next((t for t in getattr(world, "tasks", []) if t.name == spec.task), None)
    if entry is not None:
        from robotforge.control.tasks.world_task import WorldTask

        return WorldTask(model, entry, world)
    from robotforge.extend import REGISTRY, ensure_user_plugins

    ensure_user_plugins()
    plugin = REGISTRY.get("task", spec.task)
    if plugin is not None:
        return plugin.meta["build"](model, spec.robot, {**plugin.meta["params"], **spec.task_params})
    known = [*TASKS, *(t.name for t in getattr(world, "tasks", [])), *REGISTRY.of("task")]
    raise TaskError(wording.unknown("task", spec.task, known, where="built in, saved with the world or from plugins"))


def _build_task(spec: EnvSpec) -> Task:
    from robotforge.design.world import World, compile_world

    full = World.model_validate(spec.world)
    world = _scoped(full, spec.robot, spec.scope)
    base = Path(spec.world_dir) if spec.world_dir else None
    try:
        compiler = compile_task_world if spec.task in TASKS else None  # built-in tasks may add a goal marker
        model = compiler(spec.task, world, base) if compiler else compile_world(world, base).model
    except ValidationError as exc:
        raise TaskError(readable(exc, "robot settings")) from exc
    return task_on(spec, model, world)


def observed(task: Task, spec: EnvSpec) -> Task | ObservedTask:
    """The task with the observation the spec asks for: the robot's sensors added to it, or standing in for it."""
    mode = spec.observation
    if mode == "task":
        return task
    probe = SensorObservation(task.model, spec.robot, lidar_bins=spec.lidar_bins, depth_size=tuple(spec.depth_size))
    if not probe.sensors:
        probe.close()
        if mode in ("task+sensors", "sensors"):
            left_out = "".join(f" ({s['sensor']}: {s['why']})" for s in probe.skipped)
            raise TaskError(
                f"robot {spec.robot!r} carries no sensor an observation can use{left_out}, so there is nothing "
                "to assemble it from; put some on it, e.g. add_sensor(robot=..., type='imu'), or use "
                "observation='task'"
            )
        return task
    return ObservedTask(task, "sensors" if mode == "sensors" else "task+sensors", probe)


def build_env(spec: EnvSpec | dict[str, Any]) -> TrainingEnv:
    """The environment an EnvSpec describes (each process builds its own)."""
    if isinstance(spec, dict):
        try:
            spec = EnvSpec.model_validate(spec)
        except ValidationError as exc:
            raise TaskError(readable(exc, "environment settings")) from exc
    return TrainingEnv(
        observed(_build_task(spec), spec),
        push=spec.push,
        control_hz=spec.control_hz,
        episode_seconds=spec.episode_seconds,
        action_mode=spec.action_mode,
        residual_scale=spec.residual_scale,
    )


def make_env(task: str, **kwargs: Any) -> TrainingEnv:
    """An environment for `task` on a robot.

    task: a built-in task (reach, balance, stand, walk, drive_to, hover), a task saved with the world
        (add_task), or one a plugin added.
    robot: a robot of `world` by name; with no world, a template or library name, or a dict like a world.yaml
        robot entry (its `sensors` too), placed alone on a floor. Default: the task's usual robot.
    world: a World, its dict, or a saved world's name or path. Default: the robot alone on a flat floor.
    world_dir: the folder a World or dict was opened from, for robots given as files.
    scope: "robot" (the robot, its mounts, the floor and terrain; the default) or "world" (everything in it,
        needed by saved tasks).
    task_params: the task's own settings, e.g. {"speed": 0.4}.
    observation: "auto" (the task's observation plus the robot's sensors, when it has any), "task", "task+sensors"
        or "sensors" (the task's goal plus the sensors only).
    lidar_bins, depth_size: how lidars and depth cameras are shrunk into the observation.
    action_mode, residual_scale, control_hz, episode_seconds: as for RobotEnv.
    push: random shoves while training, as strong as this (0 = none).
    """
    return build_env(env_spec(task, **kwargs))


def describe(env: TrainingEnv) -> dict[str, Any]:
    """Both spaces in words: what each number is, where it sits, and how actions are scaled."""
    from robotforge.control.rl.runtime import actuator_names

    task = env.task
    names = actuator_names(task)
    sensors = getattr(task, "sensors", None)
    return {
        "task": getattr(task, "name", type(task).__name__),
        "robot": task.robot.name,
        "observation": {
            "size": int(env.observation_space.shape[0]),
            "dtype": "float32",
            "sources": env.layout,
            "left_out": sensors.skipped if sensors is not None else [],
        },
        "action": {
            "size": int(env.action_space.shape[0]),
            "range": [-1.0, 1.0],
            "mode": env.action_mode,
            "actuators": names,
            "meaning": (
                "corrections added to the built-in controller"
                if env.action_mode == "residual"
                else "each value maps onto its actuator's range"
            ),
        },
        "control_hz": round(1.0 / env.control_dt, 3),
        "episode_steps": int(env.max_steps),
        "episode_seconds": round(float(env.episode_seconds), 3),
    }
