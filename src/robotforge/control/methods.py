"""The algorithm menu: which methods exist, what they are in one sentence, and how to build one."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import ValidationError

from robotforge.control import classic
from robotforge.control.mpc import MPCConfig, SamplingMPC
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import home_ctrl, readable

METHODS: dict[str, str] = {
    "ik": "Inverse kinematics + servo position control: solve for joint angles once and send them to the servos.",
    "pd": "PD feedback: push back in proportion to the error and its rate (with gravity compensation for arms).",
    "cartesian": "Operational-space control: a virtual spring pulls the tool straight to the target, no IK needed.",
    "lqr": "Linear-quadratic regulator: the optimal feedback gains for a model linearized around balance.",
    "mpc": "Predictive sampling MPC: simulate many possible futures every 20 ms and follow the best one.",
    "mppi": "MPPI: like predictive sampling, but blends all futures weighted by how good they are.",
    "ars": "Reinforcement learning (Augmented Random Search): learn a linear policy by trial and error, then run it.",
    "hold": "Hold the home pose with every servo (what the robot does when nobody controls it).",
    "gait": "The built-in trot gait: legs follow scripted foot paths, solved with leg IK.",
    "none": "No control at all, to see what physics does on its own.",
}

# Settings that work well per task, so a beginner gets a good result without tuning.
_LQR_WEIGHTS: dict[str, dict[str, Any]] = {
    "balance": {"q_weights": [5.0, 50.0, 0.5, 1.0], "r_weight": 0.5},
}


def check(task: Task, method: str) -> None:
    if method not in METHODS:
        raise TaskError(f"unknown method {method!r}; choose one of: {', '.join(METHODS)}")
    if method not in task.methods:
        raise TaskError(
            f"{method} does not fit the {task.name} task; for {task.name} use one of: {', '.join(task.methods)}"
        )


def make_controller(
    method: str, task: Task, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any] | None = None
):
    """Build the controller for `method` on `task`, starting from the state in `data`."""
    check(task, method)
    params = dict(params or {})
    try:
        if method in ("mpc", "mppi"):
            cfg = {**task.mpc_defaults, **params, "method": "predictive" if method == "mpc" else "mppi"}
            base_name = cfg.pop("base", None)
            base = _base(base_name, task, model, params) if base_name else None
            return SamplingMPC(task, model, MPCConfig.model_validate(cfg), base=base)
        if method == "ik":
            return classic.ServoIK(task, model, data)
        if method == "cartesian":
            return classic.CartesianPD(task, model, data, **params)
        if method == "lqr":
            return classic.LQR(task, model, data, **{**_LQR_WEIGHTS.get(task.name, {}), **params})
        if method == "pd":
            if task.name == "reach":
                return classic.JointPD(task, model, data, **params)
            if task.name == "balance":
                return classic.StateFeedback(task, **params)
            return classic.GoToGoal(task, model, **params)
        if method == "hold":
            return classic.Hold(task, home_ctrl(model)[task.actuators])
        if method == "none":
            return classic.Hold(task, np.zeros(len(task.actuators)), label="none")
        if method == "gait":
            return _gait(task, model, params)
        if method == "ars":
            from robotforge.control.ars import train_controller

            return train_controller(task, model, params)
    except TypeError as exc:  # unknown keyword in params
        raise TaskError(f"{method}: bad params {sorted(params)}: {exc}") from exc
    except ValidationError as exc:
        raise TaskError(readable(exc, f"{method} settings")) from exc
    raise TaskError(f"{method} is not available for {task.name} yet")


def _base(name: str, task: Task, model: mujoco.MjModel, params: dict[str, Any]):
    """A controller that MPC corrects instead of planning from scratch."""
    if name == "gait":
        return classic.GaitBase(task, _gait(task, model, {}))
    raise TaskError(f"MPC can only be put on top of the gait for now, not {name!r}; use base=null for plain MPC")


def _gait(task: Task, model: mujoco.MjModel, params: dict[str, Any]):
    from robotforge.behaviors import quadruped
    from robotforge.control.tasks.base import home_qpos

    try:
        gait = quadruped.for_robot(model, task.robot.name, home_qpos(model))
    except ValueError as exc:
        raise TaskError(f"the built-in gait needs a four-legged robot: {exc}") from exc
    speed = float(params.get("speed", getattr(task.params, "speed", 0.0)))
    turn = float(params.get("turn", getattr(task.params, "turn", 0.0)))
    gait.command(speed, turn)
    return gait
