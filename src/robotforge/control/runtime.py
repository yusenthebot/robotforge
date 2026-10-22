"""Shared discovery and binding for live control adapters.

Execution and pacing belong to the calling adapter. Task identity, stored policy
settings and method validation are shared with training and policy playback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from robotforge.control.tasks import Task, TaskError


def contract_catalog(world: Any = None) -> dict[str, Any]:
    from robotforge.control import tasks
    from robotforge.control.methods import METHODS, method_catalog
    from robotforge.training.mcp_tools import algorithm_catalog

    return {
        "tasks": tasks.catalog(world=world),
        "algorithms": dict(METHODS),
        "method_schemas": method_catalog(),
        "training_algorithms": algorithm_catalog(),
    }


def effective_task_params(
    task: str,
    method: str,
    params: dict[str, Any] | None,
    world_dir: Path | None = None,
) -> dict[str, Any] | None:
    """A policy's stored task settings, followed by explicit goal overrides."""
    from robotforge.control.methods import POLICY_PREFIX, resolve_method
    from robotforge.control.rl import store
    from robotforge.control.rl.runtime import TrainedPolicy

    method = resolve_method(method)
    if not method.startswith(POLICY_PREFIX):
        return params
    policy = TrainedPolicy.load(store.find_policy(method.removeprefix(POLICY_PREFIX), world_dir))
    spec = policy.meta.get("env") or {}
    wanted = spec.get("task") or policy.meta.get("task")
    # World task names are already canonicalized by the request's resolver.
    from robotforge import wording

    if wording.squash(wanted) != wording.squash(task):
        raise TaskError(f"policy {policy.name!r} was trained for the {wanted} task, not {task}")
    stored = spec.get("task_params") if spec else policy.meta.get("task_params")
    return {**copy.deepcopy(stored or {}), **copy.deepcopy(params or {})}


@dataclass(frozen=True)
class BoundControl:
    task: Task
    method: str
    params: dict[str, Any] | None


def bind_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot: str,
    task: str,
    method: str,
    task_params: dict[str, Any] | None = None,
    method_params: dict[str, Any] | None = None,
    *,
    world: Any = None,
    world_dir: Path | None = None,
) -> BoundControl:
    from robotforge.control.methods import check, resolve_method, validate_params
    from robotforge.control.rl.runtime import policy_params
    from robotforge.control.task_registry import bind_task, resolve_task

    definition = resolve_task(task, world)
    method = resolve_method(method)
    settings = effective_task_params(definition.name, method, task_params, world_dir)
    bound = bind_task(definition.name, model, robot, settings, world=world)
    if bound.robot.name != robot and not bound.robot.name.startswith(robot + "/"):
        raise TaskError(f"task {definition.name!r} belongs to robot {bound.robot.name!r}, not {robot!r}")
    check(bound, method)
    bound.begin(data)
    given = policy_params(method, method_params, world_dir, bound)
    given = validate_params(method, bound, given)
    return BoundControl(bound, method, given)


def start_control(
    playground: Any,
    robot: str,
    task: str,
    method: str,
    task_params: dict[str, Any] | None = None,
    method_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach only after the shared request validates and its model is still current.

    Callers provide execution admission: a viewer command lease or a script's
    ExecutionRun.call_thread. This function never starts another simulation clock.
    """
    from robotforge.control.methods import make_controller
    from robotforge.control.task_registry import describe_task
    from robotforge.experiments.live import attach_record

    session = playground.session
    with session.lock:
        admitted_generation = session.generation
    world = playground.current_world()
    directory = playground.store.dir if playground.store is not None else None
    names = playground.behaviors.robot_names()
    robot = playground.behaviors.resolve(robot) if names else robot
    with session.lock:
        if session.generation != admitted_generation:
            raise TaskError("the world changed while preparing the task; run it again")
        model, data, generation = session.model, copy.copy(session.data), session.generation
    request = bind_control(
        model, data, robot, task, method, task_params, method_params, world=world, world_dir=directory
    )
    ctl = make_controller(request.method, request.task, model, data, request.params)
    with session.lock:
        if session.generation != generation:
            raise TaskError("the world was reset or changed while the controller was being prepared; run it again")
        request.task.begin(session.data)
        experiment = attach_record(
            session,
            robot,
            ctl,
            getattr(ctl, "task", request.task),
            request.method,
            world=world,
            world_dir=directory,
            method_params=request.params,
            entrypoint="script_or_viewer_control",
            overrides=task_params,
        )
    return {
        "robot": robot,
        "task": describe_task(request.task),
        "method": request.method,
        "controller": ctl.status(),
        "experiment": experiment,
    }
