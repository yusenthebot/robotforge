"""Shared task discovery, parameter boundaries and binding ownership."""

from dataclasses import FrozenInstanceError

import mujoco
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, Field

from robotforge.control import task_world
from robotforge.control.task_registry import (
    bind_task,
    check_training_task,
    describe_task,
    registered_task_name,
    resolve_task,
    task_definitions,
    task_registry_revision,
)
from robotforge.control.tasks import TASKS, TaskError, catalog, make_task
from robotforge.control.tasks.base import Robot
from robotforge.control.tasks.reach import Reach
from robotforge.control.tasks.world_task import WorldTask
from robotforge.design.tasks.spec import TaskEntry
from robotforge.design.world import World
from robotforge.extend import REGISTRY, PluginError, task
from robotforge.extend.registry import Item


@pytest.fixture(autouse=True)
def isolated_plugins(monkeypatch):
    monkeypatch.setattr(REGISTRY, "items", {})
    monkeypatch.setattr(REGISTRY, "user_loaded", True)


@pytest.fixture
def arm():
    return task_world("reach")


def test_catalog_is_lazy_and_builtin_mapping_is_compatible():
    @task("wrapped_reach", summary="A wrapper", methods=("ik",), robot_hint="an arm")
    def build(model, robot, params):
        raise AssertionError("catalog must not invoke factories")

    assert "wrapped_reach" not in TASKS
    rows = {row["name"]: row for row in catalog()}
    assert rows["reach"]["params"]["tolerance"]["default"] == 0.01
    assert rows["reach"]["schema"]["additionalProperties"] is False
    assert rows["wrapped_reach"]["methods"] == ["ik"]
    assert rows["wrapped_reach"]["source"] == "plugin"
    assert rows["wrapped_reach"]["robot"] == "an arm"
    definition = resolve_task("The Wrapped-Reach")
    assert definition.name == "wrapped_reach"
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"
    with pytest.raises(TaskError, match="unknown task"):
        resolve_task("does_not_exist")


def test_wrapped_builtin_keeps_dispatch_name_and_does_not_change_physics(arm):
    model, robot = arm
    data = mujoco.MjData(model)
    before = (model.qpos0.copy(), model.actuator_gainprm.copy(), data.qpos.copy(), data.ctrl.copy())

    @task("wrapped_reach")
    def build(model, robot, params):
        return Reach(model, robot, params)

    assert resolve_task("wrapped_reach").methods is None
    bound = make_task("Wrapped Reach", model, robot, {"tolerance": 0.02})
    assert bound.name == "reach"
    assert describe_task(bound)["task"] == "wrapped_reach"
    assert describe_task(bound)["implementation_task"] == "reach"
    assert bound.describe()["task"] == "reach"
    assert registered_task_name(bound) == "wrapped_reach"
    assert registered_task_name(Reach(model, robot)) == "reach"
    assert bound.methods == Reach.methods
    for previous, current in zip(before, (model.qpos0, model.actuator_gainprm, data.qpos, data.ctrl), strict=True):
        np.testing.assert_array_equal(previous, current)


def test_legacy_unknown_options_and_defaults_are_copied_between_bindings(arm):
    model, robot = arm
    defaults = {"nested": {"values": [1]}}
    observed = []

    @task("legacy", params=defaults)
    def build(model, robot, params):
        observed.append(params)
        params["nested"]["values"].append(2)
        return Reach(model, robot)

    defaults["nested"]["values"].append(9)
    definition = resolve_task("legacy")
    definition.defaults["nested"]["values"].append(10)
    caller = {"extra": {"list": [3]}}
    bind_task("legacy", model, robot, caller)
    bind_task("legacy", model, robot)
    assert observed[0] == {"nested": {"values": [1, 2]}, "extra": {"list": [3]}}
    assert observed[1] == {"nested": {"values": [1, 2]}}
    assert caller == {"extra": {"list": [3]}}
    assert REGISTRY.get("task", "legacy").meta["params"] == {"nested": {"values": [1]}}


def test_schema_defaults_bounds_strictness_and_extra_policy(arm):
    class Options(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        tolerance: float = Field(0.01, gt=0, le=0.2)
        weights: list[float] = Field(default_factory=lambda: [1.0])

    observed = []

    @task("typed", schema=Options, params={"tolerance": 0.02})
    def build(model, robot, params):
        observed.append(params)
        return Reach(model, robot, {"tolerance": params["tolerance"]})

    definition = resolve_task("typed")
    assert definition.validate_params() == {"tolerance": 0.02, "weights": [1.0]}
    assert definition.describe()["schema"]["properties"]["tolerance"]["default"] == 0.02
    for values in ({"tolerance": -1}, {"tolerance": "0.1"}, {"oops": 1}):
        with pytest.raises(TaskError, match="typed"):
            bind_task("typed", *arm, values)
    assert not observed
    bind_task("typed", *arm, {"tolerance": 0.03})
    assert observed == [{"tolerance": 0.03, "weights": [1.0]}]

    class OpenOptions(BaseModel):
        model_config = ConfigDict(extra="allow")
        enabled: bool = True

    task("open_options", schema=OpenOptions)(lambda model, robot, params: Reach(model, robot))
    assert resolve_task("open_options").validate_params({"unknown": [2]}) == {"enabled": True, "unknown": [2]}


@pytest.mark.parametrize("values", [{"x": float("nan")}, {"nested": [float("inf")]}, {1: 2}, {"x": object()}, []])
def test_invalid_json_parameters_never_reach_factory(arm, values):
    @task("finite")
    def build(model, robot, params):
        raise AssertionError("must validate first")

    with pytest.raises(TaskError, match="parameters"):
        bind_task("finite", *arm, values)


@pytest.mark.parametrize("fault", ["result", "model", "robot", "index", "duplicate", "foreign", "fractional", "ragged"])
def test_wrong_factory_results_are_rejected(arm, fault):
    model, robot = arm

    @task("broken")
    def build(model, robot, params):
        if fault == "result":
            return {"not": "a task"}
        bound = Reach(model, robot)
        if fault == "model":
            bound.model = mujoco.MjModel.from_xml_string("<mujoco/>")
        if fault == "robot":
            bound.robot.name = "somebody_else"
        if fault == "index":
            bound.actuators = np.array([model.nu])
        if fault == "duplicate":
            bound.actuators = np.array([0, 0])
        if fault == "foreign":
            bound.robot = Robot(model, robot)
            bound.robot._names = lambda *args: []
        if fault == "ragged":
            bound.actuators = [[0], [1, 2]]
        if fault == "fractional":
            bound.actuators = np.array([0.5])
        return bound

    with pytest.raises(TaskError, match="factory"):
        bind_task("broken", model, robot)


def test_live_reload_and_in_place_metadata_changes_invalidate_revision():
    task("reloadable", methods=("ik",))(lambda model, robot, params: None)
    before = task_registry_revision()
    old = resolve_task("reloadable")
    REGISTRY.remove("task", "reloadable")
    task("reloadable", methods=("pd",))(lambda model, robot, params: None)
    after = task_registry_revision()
    assert before != after
    assert resolve_task("reloadable").methods == ("pd",)
    assert old.methods == ("ik",)
    REGISTRY.get("task", "reloadable").meta["params"]["new_default"] = 2
    assert after != task_registry_revision()
    REGISTRY.remove("task", "reloadable")
    assert "reloadable" not in task_definitions()


def test_saved_world_precedence_robot_selection_and_legacy_params(arm):
    model, robot = arm
    task("saved")(lambda model, robot, params: (_ for _ in ()).throw(AssertionError("plugin lost precedence")))
    world = World(
        tasks=[TaskEntry(name="saved", kind="go_to", robot=robot), TaskEntry(name="reach", kind="go_to", robot=robot)]
    )
    definitions = task_definitions(world)
    assert definitions["reach"].source == "builtin"
    assert definitions["saved"].source == "world"
    bound = bind_task("The Saved", model, "legacy_other_robot", {"ignored_legacy_option": 1}, world=world)
    assert isinstance(bound, WorldTask)
    assert bound.robot.name == robot
    assert registered_task_name(bound) == "saved"
    assert bound.name == "world_task"
    # Even direct registry entries cannot displace built-in names.
    REGISTRY.add(Item("task", "reach", "collision", None, meta={"build": None}))
    assert resolve_task("reach", world).source == "builtin"


def test_decorator_rejects_invalid_metadata_and_defaults():
    for metadata in ({"schema": object}, {"methods": ["ik"]}, {"robot_hint": 1}, {"params": {"n": float("nan")}}):
        with pytest.raises(PluginError):
            task("invalid", **metadata)(lambda model, robot, params: None)


def test_wheel_balance_substitution_uses_its_original_defaults_and_bounds():
    from built_robots import GALLERY

    from robotforge.control.tasks.wheel_balance import WheelBalance
    from robotforge.design.world import compile_world

    model = compile_world(World.model_validate({"robots": [{"name": "seg", "build": GALLERY["balancer"]}]})).model
    bound = bind_task("Balance", model, "seg")
    assert isinstance(bound, WheelBalance)
    assert bound.params.randomize == 0.05
    with pytest.raises(TaskError, match="randomize"):
        bind_task("balance", model, "seg", {"randomize": 0.4})


def test_real_file_reload_replaces_same_count_definition(tmp_path):
    plugin = tmp_path / "registered_task.py"
    plugin.write_text(
        "from robotforge.extend import task\n"
        '@task("file_task", params={"a": 1}, methods=("ik",))\n'
        "def build(model, robot, params): return None\n"
    )
    try:
        first = REGISTRY.load_file(plugin, "user", "registered_task.py")
        assert first.error is None
        revision = task_registry_revision()
        plugin.write_text(plugin.read_text().replace('("ik",)', '("pd",)').replace('"a": 1', '"a": 2'))
        second = REGISTRY.load_file(plugin, "user", "registered_task.py")
        assert second.error is None
        assert task_registry_revision() != revision
        assert resolve_task("file_task").methods == ("pd",)
        assert resolve_task("file_task").defaults == {"a": 2}
    finally:
        REGISTRY._unload(plugin)


def test_training_restrictions_follow_plugin_implementation():
    from robotforge.control.tasks.contact import Press

    model, robot = task_world("press")
    task("custom_press")(lambda model, robot, params: Press(model, robot, params))
    bound = bind_task("custom_press", model, robot)
    assert registered_task_name(bound) == "custom_press"
    assert bound.name == "press"
    with pytest.raises(TaskError, match="press has no gym env or training"):
        check_training_task(bound)


def test_strict_schema_and_legacy_tuple_defaults_keep_their_python_type():
    class Options(BaseModel):
        model_config = ConfigDict(strict=True)
        target: tuple[float, float, float]

    task("tuple_options", schema=Options, params={"target": (0.1, 0.2, 0.3)})(lambda *args: None)
    assert resolve_task("tuple_options").validate_params() == {"target": (0.1, 0.2, 0.3)}
    assert "required" not in resolve_task("tuple_options").describe()["schema"]
    task("legacy_tuple", params={"values": (1, 2)})(lambda *args: None)
    assert resolve_task("legacy_tuple").validate_params() == {"values": (1, 2)}


def test_python_numpy_task_settings_remain_supported_and_finite(arm):
    model, robot = arm
    bound = make_task("reach", model, robot, {"tolerance": np.float32(0.02), "target": np.array([0.3, 0.1, 0.2])})
    assert bound.params.tolerance == pytest.approx(0.02)
    with pytest.raises(TaskError, match="finite"):
        make_task("reach", model, robot, {"target": np.array([0.3, np.nan, 0.2])})
