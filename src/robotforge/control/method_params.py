"""Controller settings contracts, derived from the adapters that consume them.

Inspection never instantiates a controller or runs a solver. Validation keeps caller values
and omits defaults, so the existing adapters remain responsible for tuning and control math.
"""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Annotated, Any, Literal, get_type_hints

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from robotforge.control import classic, impedance
from robotforge.control.mpc import MPCConfig
from robotforge.control.tasks.base import Task, TaskError, readable


class MPCParams(MPCConfig):
    base: Literal["gait"] | None = None


class PolicyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str | Path | None = None
    world_dir: str | Path | None = None


def _source(method: str, task: Task | type[Task]) -> tuple[Any, bool]:
    """The callable whose keywords the selected adapter consumes, and whether **kwargs are public."""
    if method in getattr(task, "own_methods", ()):
        from robotforge.control.tasks.hover import Hover

        source = getattr(task.own_controller, "__func__", task.own_controller)
        return task.own_controller, source is not Hover.own_controller
    if method in impedance.IMPEDANCE_METHODS or (method in impedance.PATH_CONTROLLERS and hasattr(task, "contact")):
        return {**impedance.PATH_CONTROLLERS, **impedance.CONTROLLERS}[method]._configure, False
    if method == "lqr":
        return getattr(task, "lqr_controller", classic.LQR.__init__), True
    if method == "pd":
        cls = {"reach": classic.JointPD, "balance": classic.StateFeedback}.get(task.name, classic.GoToGoal)
        return cls.__init__, False
    if method == "cartesian":
        return classic.CartesianPD.__init__, False
    if method == "ars":
        from robotforge.control.ars import train_ars

        return train_ars, False
    return classic.ServoIK.__init__, False  # ik, hold, none: no user settings


def _bounds(source: Any, key: str) -> dict[str, float]:
    """Existing adapter limits, also exposed in JSON Schema (no new tuning limits)."""
    arm = (
        classic.JointPD.__init__,
        classic.CartesianPD.__init__,
        impedance.JointPDPath._configure,
        impedance.JointImpedance._configure,
        impedance.CartesianImpedance._configure,
        impedance.OperationalSpace._configure,
    )
    if source in arm:
        high = {
            "omega": 200.0,
            "zeta": 10.0,
            "rot_omega": 200.0,
            "soft_omega": 200.0,
            "normal_omega": 200.0,
            "force_gain": 20.0,
            "force_integral": 200.0,
            "normal_damping": 0.95,
            "rot_stiffness": 1e4,
        }.get(key)
        return {"gt": 0.0, "le": high} if high is not None else {}
    if source is classic.GoToGoal.__init__ and key in ("k_turn", "k_speed"):
        return {"gt": 0.0, "le": 100.0}
    from robotforge.control.tasks.hover import Hover, HoverLQR
    from robotforge.control.tasks.wheel_balance import WheelBalance

    if key == "r_weight":
        if source in (classic.LQR.__init__, Hover.lqr_controller, HoverLQR.__init__):
            return {"gt": 0.0, "le": 1e6}
        if source is WheelBalance.lqr_controller:
            return {"gt": 0.0}
    return {}


def parameter_model(method: str, task: Task | type[Task]) -> type[BaseModel]:
    if method.startswith("policy:"):
        return PolicyParams
    owned = method in getattr(task, "own_methods", ())
    if method in ("mpc", "mppi") and not owned:
        defaults = {**task.mpc_defaults, "method": "predictive" if method == "mpc" else "mppi"}
        overrides = {}
        for key, value in defaults.items():
            if key in MPCParams.model_fields:
                info = copy.deepcopy(MPCParams.model_fields[key])
                info.default = value
                overrides[key] = (info.annotation, info)
        return create_model("MPCSettings", __base__=MPCParams, **overrides)
    if method == "gait" and not owned:
        fields = {}
        for key in ("speed", "turn"):
            field = task.Params.model_fields.get(key)
            default = getattr(getattr(task, "params", None), key, field.default if field else 0.0)
            fields[key] = (float, default)
        return create_model("GaitSettings", __config__=ConfigDict(extra="forbid", allow_inf_nan=False), **fields)
    source, allow_kwargs = _source(method, task)
    source = getattr(source, "__func__", source)
    hints = get_type_hints(source)
    from robotforge.control.tasks.hover import Hover
    from robotforge.control.tasks.wheel_balance import WheelBalance

    built_in_lqr = source in (classic.LQR.__init__, Hover.lqr_controller, WheelBalance.lqr_controller)
    built_in_spring = source in (impedance.JointImpedance._configure, impedance.CartesianImpedance._configure)
    fields: dict[str, Any] = {}
    extras = False
    for name, param in inspect.signature(source).parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            extras = allow_kwargs
            continue
        if name in {"self", "task", "model", "data", "method", "env"} or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            continue
        annotation = hints.get(name, Any)
        # Python callers may pass callbacks and an existing policy to legacy ARS. They are
        # deliberately retained, even though JSON clients cannot manufacture these objects.
        if method == "ars" and not owned and name in ("callback", "policy", "stop"):
            annotation = Any
        if name == "q_weights" and built_in_lqr:
            annotation = list[Annotated[float, Field(ge=0)]] | None
        if name == "r_weight" and param.default is None and built_in_lqr:
            annotation = float | None
        if name in ("stiffness", "soft_stiffness") and built_in_spring:
            positive = Annotated[float, Field(gt=0, le=1e5)]
            annotation = positive | list[positive]
            if param.default is None:
                annotation = annotation | None
        default = ... if param.default is inspect.Parameter.empty else param.default
        default = getattr(task, "gains", {}).get(method, {}).get(name, default)
        limits = _bounds(source, name)
        if source is classic.StateFeedback.__init__ and name == "gains":
            annotation = Annotated[list[float], Field(min_length=4, max_length=4)] | None
        fields[name] = (annotation, Field(default=default, **limits))
    if method == "ars" and not owned:
        fields["episode_seconds"] = (float, Field(default=task.default_seconds))
    return create_model(
        f"{method.title().replace('_', '')}Settings",
        __config__=ConfigDict(extra="allow" if extras else "forbid", allow_inf_nan=False),
        **fields,
    )


def _finite(value: Any, path: str = "params") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, np.ndarray) and value.ndim == 0:
            _finite(value.item(), path)
        else:
            for index, child in enumerate(value):
                _finite(child, f"{path}[{index}]")
    elif isinstance(value, Real) and not math.isfinite(value):
        raise TaskError(f"{path} must be finite")


def validate_params(method: str, task: Task, params: dict[str, Any] | None) -> dict[str, Any]:
    """Check settings without inserting defaults or replacing the caller's numeric values."""
    if params is not None and not isinstance(params, dict):
        raise TaskError(f"{method} settings must be a dictionary")
    supplied = dict(params or {})
    _finite(supplied)
    try:
        # Arrays are accepted by existing NumPy-based adapters; use lists only for schema checking.
        values = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in supplied.items()}
        owned = method in getattr(task, "own_methods", ())
        if method not in ("mpc", "mppi", "gait") and not method.startswith("policy:"):
            source = _source(method, task)[0]
            source = getattr(source, "__func__", source)
            for key, value in supplied.items():
                limits = _bounds(source, key)
                if limits and value is not None:
                    classic.positive(key, value, limits.get("le", math.inf))
        validated = parameter_model(method, task).model_validate(values)
        if method == "ars" and not owned and not 1 <= validated.top <= validated.directions:
            raise TaskError("top must be between 1 and the number of directions")
        if method in ("joint_impedance", "cartesian_impedance") and not owned:
            size = len(task.actuators) if method == "joint_impedance" else 3
            for key in ("stiffness", "soft_stiffness"):
                if supplied.get(key) is not None:
                    impedance.per_axis(key, supplied[key], size)
    except ValidationError as exc:
        unknown = any(error["type"] == "extra_forbidden" for error in exc.errors())
        what = f"{method}: bad params" if unknown and method not in ("mpc", "mppi") else f"{method} settings"
        raise TaskError(readable(exc, what)) from exc
    except (TypeError, ValueError) as exc:
        raise TaskError(f"{method} settings: {exc}") from exc
    return supplied
