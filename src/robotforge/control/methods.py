"""The algorithm menu: which methods exist, what they are in one sentence, and how to build one."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import ValidationError

from robotforge import wording
from robotforge.control import classic, impedance
from robotforge.control.method_params import parameter_model, validate_params
from robotforge.control.mpc import MPCConfig, SamplingMPC
from robotforge.control.tasks import Task, TaskError
from robotforge.control.tasks.base import home_ctrl, readable

# Full names of methods, taken for the short name. Other names agents use are only offered as "did you mean":
# a PID controller is not quite the PD one here, and "sampling MPC" could be mpc or mppi.
LONG_NAMES = {"inverse_kinematics": "ik", "model_predictive_control": "mpc"}
NICKNAMES = {"pid": "pd", "pid_control": "pd", "sampling_mpc": "mppi", "jacobian": "cartesian", "hold_position": "hold"}
METHODS: dict[str, str] = {
    "ik": "Inverse kinematics + servo position control: solve for joint angles once and send them to the servos.",
    "pd": "PD feedback: push back in proportion to the error and its rate (with gravity compensation for arms).",
    "cartesian": "Operational-space control: a virtual spring pulls the tool straight to the target, no IK needed.",
    "joint_impedance": "Joint impedance: a soft spring and damper in every joint toward the IK answer, with gravity "
    "held up; the arm gives way when it touches something.",
    "cartesian_impedance": "Cartesian impedance: the tool is a mass on a spring and damper toward the target "
    "(torque = J^T F + gravity), so it gives way like a spring in space.",
    "osc": "Operational-space control: the tool answers like a unit mass in every pose (tool inertia, Coriolis and "
    "gravity cancelled), and along a surface it closes a force loop on the touch it feels.",
    "lqr": "Linear-quadratic regulator: the optimal feedback gains for a model linearized around balance.",
    "mpc": "Predictive sampling MPC: simulate many possible futures every 20 ms and follow the best one.",
    "mppi": "MPPI: like predictive sampling, but blends all futures weighted by how good they are.",
    "ars": "Reinforcement learning (Augmented Random Search): learn a linear policy by trial and error, then run it.",
    "hold": "Hold the home pose with every servo (what the robot does when nobody controls it).",
    "gait": "The built-in trot gait: legs follow scripted foot paths, solved with leg IK.",
    "cascade": "Cascaded flight control: position, then attitude, then body rates, and a mixer from the rotor layout "
    "(how real drones fly).",
    "none": "No control at all, to see what physics does on its own.",
}

# Settings that work well per task, so a beginner gets a good result without tuning.
_LQR_WEIGHTS: dict[str, dict[str, Any]] = {
    "balance": {"q_weights": [5.0, 50.0, 0.5, 1.0], "r_weight": 0.5},
}


POLICY_PREFIX = "policy:"  # "policy:<name>" runs a trained neural network policy (see control/rl)
LEARNED = ("ppo", "sac")  # algorithms that are trained first, so they are named but not in METHODS
LEARNED_HINT = (
    "{method} is trained, not built on the spot: train a policy with train_policy(algo={method!r}), "
    "then run it with method='policy:<its name>'"
)


def explain(method: str) -> str:
    """The one-line explanation of a method, including trained policies."""
    method = resolve_method(method)
    if method.startswith(POLICY_PREFIX):
        return f"A trained neural network policy ({method.removeprefix(POLICY_PREFIX)}), learned by trial and error."
    return METHODS[method]


def resolve_method(name: str) -> str:
    """Resolve unambiguous human spellings, preserving a trained policy's case-sensitive name."""
    if not isinstance(name, str):
        raise TaskError("method must be a string")
    prefix, colon, suffix = name.partition(":")
    if colon and prefix.lower() == "policy":
        return POLICY_PREFIX + suffix
    given = name.strip()
    method = wording.match(given, (*METHODS, *LEARNED), LONG_NAMES)
    if method is None:
        close = wording.near(given, METHODS, NICKNAMES)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise TaskError(f"unknown method {given!r}.{hint} Choose one of: {', '.join(METHODS)}")
    return method


def check(task: Task, method: str) -> None:
    method = resolve_method(method)
    if method.startswith(POLICY_PREFIX):
        from robotforge.control.rl.store import check_name

        check_name(method.removeprefix(POLICY_PREFIX))
        return
    if method in LEARNED:
        raise TaskError(LEARNED_HINT.format(method=method))
    if method not in task.methods:
        raise TaskError(
            f"{method} does not fit the {task.name} task; for {task.name} use one of: {', '.join(task.methods)}"
        )
    if method in getattr(task, "own_methods", ()):
        if not callable(getattr(task, "own_controller", None)):
            raise TaskError(f"{method} requires the task's own_controller adapter")
        return
    if method == "cascade":
        raise TaskError("cascade requires the task's own_controller adapter")
    if method == "lqr" and hasattr(task, "lqr_controller") and not callable(task.lqr_controller):
        raise TaskError("lqr requires a callable lqr_controller adapter")
    path = method in impedance.IMPEDANCE_METHODS or (method in impedance.PATH_CONTROLLERS and hasattr(task, "contact"))
    required: tuple[str, ...] = ()
    if path or method in ("ik", "cartesian") or (method == "pd" and task.name == "reach"):
        required = ("site", "joints", "qadr", "dadr", "target_source" if path else "target")
    elif method == "pd":
        required = ("hinge", "state") if task.name == "balance" else ("goal", "position", "_heading")
    missing = [name for name in required if not hasattr(task, name)]
    if missing:
        raise TaskError(f"{method} requires a compatible task adapter; {task.name} is missing {', '.join(missing)}")


def base_method(task: Task) -> str:
    """The first declared non-policy base; malformed declarations fail without a fallback."""
    for given in task.methods:
        method = resolve_method(given)
        if method == "ars" or method.startswith(POLICY_PREFIX):
            continue
        check(task, method)
        return method
    raise TaskError(f"{task.name} has no supported base controller")


def method_schema(method: str, task: Task | None = None) -> dict[str, Any]:
    """Settings schema for a bound task, or named variants for task-dependent adapters."""
    method = resolve_method(method)
    if task is not None:
        check(task, method)
        schema = parameter_model(method, task).model_json_schema()
        if method == "lqr" and not hasattr(task, "lqr_controller"):
            for key, value in _LQR_WEIGHTS.get(task.name, {}).items():
                schema["properties"][key]["default"] = value
        return schema
    from robotforge.control.tasks import TASKS, WheelBalance

    variants = []
    for cls in (*TASKS.values(), WheelBalance):
        if method.startswith(POLICY_PREFIX) or method in cls.methods:
            schema = parameter_model(method, cls).model_json_schema()
            if method == "lqr" and not hasattr(cls, "lqr_controller"):
                for key, value in _LQR_WEIGHTS.get(cls.name, {}).items():
                    schema["properties"][key]["default"] = value
            variants.append({"task": cls.name, "adapter": cls.__name__, "schema": schema})
    schemas = []
    for item in variants:
        if item["schema"] not in schemas:
            schemas.append(item["schema"])
    if len(schemas) == 1:
        return schemas[0]
    return {"anyOf": schemas, "x-task-variants": variants}


def method_catalog(task: Task | None = None) -> list[dict[str, Any]]:
    """Live control methods; training Algorithm plugins have their separate registry."""
    names = METHODS if task is None else {name: METHODS[name] for name in task.methods if name in METHODS}
    return [
        {"name": name, "summary": summary, "domain": "control", "params": method_schema(name, task)}
        for name, summary in names.items()
    ]


def make_controller(
    method: str, task: Task, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any] | None = None
):
    """Build the controller for `method` on `task`, starting from the state in `data`."""
    method = resolve_method(method)
    check(task, method)
    params = validate_params(method, task, params)
    if method.startswith(POLICY_PREFIX):
        from robotforge.control.rl.runtime import policy_controller

        return policy_controller(method.removeprefix(POLICY_PREFIX), task, model, data, params)
    try:
        if method in getattr(task, "own_methods", ()):  # a task that brings the controller for it (hover: cascade)
            return task.own_controller(method, model, data, **params)
        if method in ("mpc", "mppi"):
            cfg = {**task.mpc_defaults, **params, "method": "predictive" if method == "mpc" else "mppi"}
            base_name = cfg.pop("base", None)
            base = _base(base_name, task, model, params) if base_name else None
            return SamplingMPC(task, model, MPCConfig.model_validate(cfg), base=base)
        if method in impedance.IMPEDANCE_METHODS or (method in impedance.PATH_CONTROLLERS and hasattr(task, "contact")):
            return impedance.make_arm_controller(method, task, model, data, params)
        if method == "ik":
            return classic.ServoIK(task, model, data)
        if method == "cartesian":
            return classic.CartesianPD(task, model, data, **params)
        if method == "lqr":
            if hasattr(task, "lqr_controller"):  # tasks with their own model-based design (wheel balancers)
                return task.lqr_controller(model, data, **params)
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
        raise TaskError(f"the built-in gait needs a four- or six-legged robot: {exc}") from exc
    speed = float(params.get("speed", getattr(task.params, "speed", 0.0)))
    turn = float(params.get("turn", getattr(task.params, "turn", 0.0)))
    gait.command(speed, turn)
    return gait
