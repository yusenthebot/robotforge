"""One live view of built-in, saved-world and plugin tasks, without running factories to list them."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import mujoco
import numpy as np
from pydantic import BaseModel, ValidationError

from robotforge import wording
from robotforge.control.tasks.base import Robot, Task, TaskError, readable


def _finite_params(params: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """Copy JSON-shaped parameters; tuples remain accepted for Python callers."""

    def normalize(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        if isinstance(value, np.generic):
            return normalize(value.item())
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(normalize(item) for item in value)
        return value

    def check(value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                check(item)
            return
        if isinstance(value, dict) and all(isinstance(key, str) for key in value):
            for item in value.values():
                check(item)
            return
        raise ValueError("parameters must contain finite JSON values with string keys")

    if params is None:
        return {}
    try:
        if not isinstance(params, dict):
            raise ValueError("parameters must be an object")
        values = normalize(params)
        check(values)
        return copy.deepcopy(values)
    except (ValueError, RecursionError) as exc:
        raise TaskError(f"task {name!r}: {exc}") from exc


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable metadata and binding rules; the plugin registry remains the source of truth.

    ``methods=None`` means the factory must be bound to discover its methods. Saved-world
    definitions retain their saved configuration: legacy task_params do not override it.
    """

    name: str
    summary: str
    robot_hint: str
    methods: tuple[str, ...] | None
    source: str
    schema: type[BaseModel] | None = None
    origin: str | None = None
    _defaults: dict[str, Any] = field(default_factory=dict, repr=False)
    _factory: Callable[..., Task] | None = field(default=None, repr=False, compare=False)
    _entry: Any = field(default=None, repr=False, compare=False)
    _world: Any = field(default=None, repr=False, compare=False)

    @property
    def defaults(self) -> dict[str, Any]:
        """Fresh copies of declared defaults; callers cannot change later bindings."""
        return copy.deepcopy(self._defaults)

    def validate_params(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        values = {**self.defaults, **_finite_params(params, self.name)}
        if self.source == "world":
            return {}  # world entries are configured through their saved TaskEntry
        if self.schema is None:
            return values
        try:
            validated = self.schema.model_validate(values)
        except ValidationError as exc:
            raise TaskError(readable(exc, f"task {self.name!r}", list(self.schema.model_fields))) from exc
        return _finite_params(validated.model_dump(mode="python", by_alias=True), self.name)

    def describe(self) -> dict[str, Any]:
        schema = (
            self.schema.model_json_schema()
            if self.schema
            else {
                "type": "object",
                "properties": {key: {"default": value} for key, value in self.defaults.items()},
                "additionalProperties": True,
            }
        )
        # Decorator defaults override schema defaults, just as they do when binding.
        schema = copy.deepcopy(schema)
        for key, value in self.defaults.items():
            if key in schema.get("properties", {}):
                schema["properties"][key]["default"] = value
        if "required" in schema:
            required = [key for key in schema["required"] if key not in self.defaults]
            if required:
                schema["required"] = required
            else:
                del schema["required"]
        params = {
            key: {k: v for k, v in prop.items() if k in ("type", "default", "description", "minimum", "maximum")}
            for key, prop in schema.get("properties", {}).items()
        }
        return {
            "name": self.name,
            "summary": self.summary,
            "robot": self.robot_hint,
            "methods": list(self.methods) if self.methods is not None else None,
            "params": params,
            "schema": schema,
            "source": self.source,
            **({"from": self.origin} if self.origin else {}),
        }

    def bind(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None) -> Task:
        from robotforge.control.tasks.balance import Balance
        from robotforge.control.tasks.wheel_balance import WheelBalance

        definition = self
        if self._factory is Balance and WheelBalance.fits(model, robot):
            definition = replace(self, schema=WheelBalance.Params, _factory=WheelBalance)
        values = definition.validate_params(params)
        if self.source == "world":
            from robotforge.control.tasks.world_task import WorldTask

            task = WorldTask(model, self._entry, self._world)
            expected_robot = self._entry.robot  # saved tasks have always selected their own robot
        else:
            if definition._factory is None:
                raise TaskError(f"task {self.name!r}: no factory is registered")
            expected_robot = robot
            task = definition._factory(model, robot, values)
        _validate_bound(task, model, expected_robot, self.name)
        # Keep Task.name for built-in controller dispatch, including factories wrapping Reach.
        task._registered_task_name = self.name
        return task


def _validate_bound(task: Any, model: mujoco.MjModel, robot: str, name: str) -> None:
    what = f"task {name!r}: factory"
    if not isinstance(task, Task):
        raise TaskError(f"{what} must return a Task, got {type(task).__name__}")
    if getattr(task, "model", None) is not model:
        raise TaskError(f"{what} returned a task bound to a different model")
    owner = getattr(task, "robot", None)
    if (
        not isinstance(owner, Robot)
        or owner.model is not model
        or not (owner.name == robot or owner.name.startswith(f"{robot}/"))
    ):
        raise TaskError(f"{what} returned a task bound to a different robot")
    try:
        indices = np.asarray(getattr(task, "actuators", None))
    except (TypeError, ValueError) as exc:
        raise TaskError(f"{what} must select a flat array of actuator indices") from exc
    if (
        indices.ndim != 1
        or indices.size == 0
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= model.nu)
        or np.unique(indices).size != indices.size
        or not np.isin(indices, owner.actuators).all()
    ):
        raise TaskError(f"{what} must select valid, unique actuator indices belonging to robot {robot!r}")


def task_definitions(world: Any = None) -> dict[str, TaskDefinition]:
    """Current definitions, with built-in > saved-world > plugin precedence."""
    from robotforge.control.tasks import TASKS
    from robotforge.extend import REGISTRY, ensure_user_plugins

    ensure_user_plugins()
    definitions = {
        name: TaskDefinition(name, cls.summary, cls.robot_hint, tuple(cls.methods), "builtin", cls.Params, _factory=cls)
        for name, cls in TASKS.items()
    }
    for entry in getattr(world, "tasks", []):
        definitions.setdefault(
            entry.name,
            TaskDefinition(
                entry.name,
                entry.describe(),
                "the task's saved robot",
                (),
                "world",
                _entry=entry,
                _world=world,
            ),
        )
    for name, item in REGISTRY.of("task").items():
        if name in definitions:
            continue
        meta = item.meta
        definitions.setdefault(
            name,
            TaskDefinition(
                name,
                item.summary,
                meta.get("robot_hint", "the robot accepted by the task factory"),
                meta.get("methods"),
                "plugin",
                meta.get("schema"),
                item.plugin,
                _finite_params(meta.get("params"), name),
                meta["build"],
            ),
        )
    return definitions


def resolve_task(name: str, world: Any = None) -> TaskDefinition:
    definitions = task_definitions(world)
    canonical = wording.match(name, definitions) if isinstance(name, str) else None
    if canonical is None:
        raise TaskError(f"unknown task {wording.short(name)}; choose one of: {', '.join(definitions)}")
    return definitions[canonical]


def bind_task(
    name: str,
    model: mujoco.MjModel,
    robot: str,
    params: dict[str, Any] | None = None,
    *,
    world: Any = None,
) -> Task:
    return resolve_task(name, world).bind(model, robot, params)


def registered_task_name(task: Task) -> str:
    """Registration identity for persistence and policy checks, separate from controller dispatch."""
    return getattr(task, "_registered_task_name", task.name)


def check_training_task(task: Task) -> None:
    """Apply existing training restrictions to the implementation, including plugin wrappers."""
    from robotforge.control.env import refuse_contact_task

    refuse_contact_task(task.name)


def describe_task(task: Task) -> dict[str, Any]:
    """Describe a binding using its registration identity, preserving custom task details."""
    description = copy.deepcopy(task.describe())
    description["task"] = registered_task_name(task)
    if description["task"] != task.name:
        description["implementation_task"] = task.name
    return description


def task_registry_revision() -> tuple:
    """A hashable fingerprint, including replacements and metadata changes at the same count."""
    from robotforge.extend import REGISTRY, ensure_user_plugins

    ensure_user_plugins()
    return tuple(
        (
            name,
            id(item),
            id(item.meta.get("build")),
            id(item.meta.get("schema")),
            item.summary,
            item.plugin,
            repr(item.meta.get("params")),
            repr(item.meta.get("methods")),
            item.meta.get("robot_hint"),
            json.dumps(item.meta["schema"].model_json_schema(), sort_keys=True) if item.meta.get("schema") else None,
        )
        for name, item in REGISTRY.of("task").items()
    )
