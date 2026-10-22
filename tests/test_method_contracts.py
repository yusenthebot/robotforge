"""Public method names and settings agree across catalogs and controller construction."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from robotforge.control import make_task, task_world
from robotforge.control.method_params import validate_params
from robotforge.control.methods import (
    METHODS,
    base_method,
    check,
    make_controller,
    method_catalog,
    method_schema,
    resolve_method,
)
from robotforge.control.tasks import TaskError


@pytest.fixture(scope="module")
def tasks():
    result = {}
    for name in ("reach", "balance", "drive_to", "stand", "walk", "hover", "press"):
        model, robot = task_world(name)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        result[name] = (make_task(name, model, robot), model, data)
    return result


@pytest.mark.parametrize(
    "given, expected",
    [
        ("Inverse Kinematics", "ik"),
        ("the PD", "pd"),
        ("MODEL-PREDICTIVE-CONTROL", "mpc"),
        ("Policy:demo", "policy:demo"),
        ("Policy:MiXeD", "policy:MiXeD"),
        (" joint-impedance ", "joint_impedance"),
    ],
)
def test_resolve_unambiguous_method_names(given, expected):
    assert resolve_method(given) == expected


def test_nicknames_remain_suggestions_and_training_algorithms_are_distinct(tasks):
    with pytest.raises(TaskError, match="Did you mean pd"):
        resolve_method("PID")
    with pytest.raises(TaskError, match="is trained"):
        check(tasks["reach"][0], "PPO")
    assert "ppo" not in {row["name"] for row in method_catalog()}
    assert {row["domain"] for row in method_catalog()} == {"control"}


@pytest.mark.parametrize(
    "task_name, method",
    [
        ("reach", "ik"),
        ("reach", "pd"),
        ("reach", "cartesian"),
        ("reach", "joint_impedance"),
        ("reach", "cartesian_impedance"),
        ("reach", "osc"),
        ("balance", "pd"),
        ("balance", "lqr"),
        ("balance", "ars"),
        ("drive_to", "pd"),
        ("stand", "hold"),
        ("hover", "none"),
        ("walk", "gait"),
        ("hover", "cascade"),
        ("hover", "lqr"),
        ("reach", "mpc"),
        ("reach", "mppi"),
        ("reach", "policy:demo"),
        ("press", "ik"),
        ("press", "pd"),
        ("press", "osc"),
    ],
)
def test_unknown_params_fail_at_boundary_before_construction(tasks, task_name, method):
    task, _, _ = tasks[task_name]
    with pytest.raises(TaskError, match="typo"):
        make_controller(method, task, None, None, {"typo": 1})


@pytest.mark.parametrize(
    "task_name, method, settings",
    [
        ("reach", "pd", {"omega": 12.0, "zeta": 0.8}),
        ("balance", "pd", {"gains": [-0.2, 4, 0.5, 0.6]}),
        ("balance", "lqr", {"q_weights": [5, 50, 0.5, 1], "r_weight": 0.5}),
        ("drive_to", "pd", {"k_speed": 1.2}),
        ("walk", "gait", {"speed": -0.2, "turn": 0.3}),
        ("reach", "cartesian_impedance", {"stiffness": [100, 200, 300], "normal_damping": 0.9}),
        ("reach", "osc", {"soft_omega": 8, "force_integral": 3}),
        ("reach", "mpc", {"method": "mppi", "samples": 2, "base": None}),
        ("reach", "policy:demo", {"path": Path("demo"), "world_dir": "world"}),
        ("balance", "ars", {"iterations": 0, "callback": lambda *_: None, "stop": lambda: True}),
    ],
)
def test_valid_settings_keep_values_and_do_not_insert_defaults(tasks, task_name, method, settings):
    task = tasks[task_name][0]
    assert validate_params(method, task, settings) == settings
    schema = method_schema(method, task)
    assert set(settings) <= schema["properties"].keys()
    assert validate_params(method, task, {}) == {}


def test_array_settings_remain_accepted(tasks):
    gains = np.array([0.3, 4, 0.5, 0.6])
    assert validate_params("pd", tasks["balance"][0], {"gains": gains})["gains"] is gains


@pytest.mark.parametrize("settings", [{"gains": [1, 2, float("nan"), 4]}, {"gains": np.array([1, 2, 3, np.inf])}])
def test_nested_nonfinite_settings_fail(tasks, settings):
    with pytest.raises(TaskError, match="finite"):
        validate_params("pd", tasks["balance"][0], settings)


def test_bound_schema_uses_task_specific_pd_adapter(tasks):
    assert set(method_schema("pd", tasks["reach"][0])["properties"]) == {"omega", "zeta"}
    assert set(method_schema("pd", tasks["balance"][0])["properties"]) == {"gains"}
    assert set(method_schema("pd", tasks["drive_to"][0])["properties"]) == {"k_turn", "k_speed"}
    assert method_schema("pd", tasks["reach"][0])["properties"]["omega"]["maximum"] == 200
    assert {row["name"] for row in method_catalog()} == METHODS.keys()
    json.dumps(method_catalog())


def test_alias_controller_matches_canonical_output(tasks):
    task, model, data = tasks["reach"]
    direct = make_controller("pd", task, model, data, {"omega": 12})
    alias = make_controller("the PD", task, model, data, {"omega": 12})
    direct.update(model, data)
    expected = data.ctrl.copy()
    alias.update(model, data)
    np.testing.assert_array_equal(expected, data.ctrl)
    assert direct.status() == alias.status()


def test_incompatible_plugin_declaration_fails_before_solver():
    task = SimpleNamespace(name="plugin", methods=("ik",))
    with pytest.raises(TaskError, match="compatible task adapter"):
        check(task, "Inverse Kinematics")
    with pytest.raises(TaskError, match="compatible task adapter"):
        make_controller("ik", task, None, None)


def test_empty_residual_base_is_actionable():
    with pytest.raises(TaskError, match="no supported base controller"):
        base_method(SimpleNamespace(name="plugin", methods=()))


def test_wheel_lqr_preserves_optional_defaults_and_has_no_new_upper_bound():
    from robotforge.control.tasks.wheel_balance import WheelBalance

    task = object.__new__(WheelBalance)
    settings = {"r_weight": 1e8, "q_weights": None}
    assert validate_params("lqr", task, settings) == settings
    assert validate_params("lqr", task, {"r_weight": None}) == {"r_weight": None}
    with pytest.raises(TaskError, match="typo"):
        validate_params("lqr", task, {"typo": 1})


def test_mpc_schema_preserves_task_defaults_and_selected_optimizer(tasks):
    task = tasks["reach"][0]
    assert method_schema("mpc", task)["properties"]["horizon"]["default"] == task.mpc_defaults["horizon"]
    assert method_schema("mppi", task)["properties"]["method"]["default"] == "mppi"
    assert method_schema("lqr", tasks["balance"][0])["properties"]["r_weight"]["default"] == 0.5


def test_own_controller_signature_controls_its_settings():
    class OwnTask:
        name = "own"
        methods = own_methods = ("mpc",)

        def own_controller(self, method, model, data, stiffness: float = -1):
            return stiffness

    task = OwnTask()
    assert make_controller("mpc", task, None, None, {"stiffness": -2}) == -2
    with pytest.raises(TaskError, match="samples"):
        make_controller("mpc", task, None, None, {"samples": 2})


def test_policy_prefix_resolution_preserves_suffix_for_validation(tasks):
    assert resolve_method("POLICY:SomePolicy ") == "policy:SomePolicy "
    with pytest.raises(TaskError, match="lowercase"):
        check(tasks["reach"][0], "POLICY:SomePolicy")


def test_own_keyword_adapter_keeps_supported_kwargs_and_rejects_nested_nan():
    class OwnTask:
        name = "own"
        methods = own_methods = ("pd",)

        def own_controller(self, method, model, data, **settings):
            return settings

    task = OwnTask()
    assert make_controller("pd", task, None, None, {"nested": {"value": -2}}) == {"nested": {"value": -2}}
    with pytest.raises(TaskError, match="finite"):
        make_controller("pd", task, None, None, {"nested": {"value": float("nan")}})


def test_mpc_validation_retains_existing_error_prefix(tasks):
    task = tasks["reach"][0]
    with pytest.raises(TaskError, match="mpc settings: replan_hz"):
        validate_params("mpc", task, {"replan_hz": 1000})


def test_declared_specialized_methods_need_callable_adapters():
    from types import SimpleNamespace

    from robotforge.control.methods import check

    with pytest.raises(TaskError, match="own_controller"):
        check(SimpleNamespace(name="custom", methods=("cascade",)), "cascade")
    with pytest.raises(TaskError, match="callable lqr_controller"):
        check(SimpleNamespace(name="custom", methods=("lqr",), lqr_controller=None), "lqr")
