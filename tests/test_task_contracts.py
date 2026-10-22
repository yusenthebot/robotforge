"""The MCP and Python control boundaries bind the same registered task."""

import json

import mujoco
import pytest
from mcp import Client
from pydantic import BaseModel, ConfigDict, Field

from robotforge.control.runtime import bind_control, contract_catalog, start_control
from robotforge.control.tasks import TaskError
from robotforge.control.tasks.reach import Reach
from robotforge.extend import registry, task
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    with registry.isolated():

        class Goal(BaseModel):
            model_config = ConfigDict(extra="forbid")
            target: list[float] = Field(default_factory=lambda: [0.25, 0.1, 0.2], min_length=3, max_length=3)
            tolerance: float = Field(0.02, gt=0)

        @task("registered_reach", schema=Goal, methods=("ik", "pd"))
        def build(model, robot, params):
            return Reach(model, robot, params)

        playground = Playground(session=SimSession.from_xml("<mujoco/>"))
        playground.new_world("contracts")
        playground.add_robot("arm", template="arm")
        yield playground


def test_shared_binding_preserves_implementation_dispatch_and_checks_before_attach(pg):
    result = start_control(pg, "arm", "Registered Reach", "Inverse Kinematics")
    assert result["task"]["task"] == "registered_reach"
    assert result["task"]["implementation_task"] == "reach"
    before = pg.session.controllers["arm"]
    with pytest.raises(TaskError, match="tolerance"):
        start_control(pg, "arm", "registered_reach", "ik", {"tolerance": -1})
    assert pg.session.controllers["arm"] is before
    with pytest.raises(TaskError, match="banana"):
        start_control(pg, "arm", "registered_reach", "ik", method_params={"banana": 1})
    assert pg.session.controllers["arm"] is before
    pg.session.step(100)
    assert pg.session.data.time > 0


@pytest.mark.anyio
async def test_mcp_discovers_runs_and_compares_plugin_task_with_same_contract(pg):
    async with Client(build_server(pg, 0)) as client:
        listed = await client.call_tool("list_algorithms", {})
        catalog = json.loads(listed.content[0].text)
        for key, value in contract_catalog(pg.current_world()).items():
            assert catalog[key] == value
        result = await client.call_tool(
            "run_controller",
            {
                "robot": "arm",
                "task": "Registered Reach",
                "method": "IK",
                "seconds": 0.04,
                "task_params": {"tolerance": 0.03},
            },
        )
        assert not result.is_error, result
        out = json.loads(result.content[0].text)
        assert out["task"]["task"] == "registered_reach"
        assert out["method"] == "ik"
        result = await client.call_tool(
            "compare_controllers",
            {
                "robot": "arm",
                "task": "registered_reach",
                "methods": ["IK", "PD"],
                "seconds": 0.04,
            },
        )
        assert not result.is_error, result
        assert json.loads(result.content[0].text)["task"]["task"] == "registered_reach"


def test_model_reset_during_build_cannot_attach_stale_controller(pg, monkeypatch):
    import robotforge.control.methods as methods

    original = methods.make_controller

    def build(*args, **kwargs):
        controller = original(*args, **kwargs)
        pg.session.reset()
        return controller

    monkeypatch.setattr(methods, "make_controller", build)
    with pytest.raises(TaskError, match="reset or changed"):
        start_control(pg, "arm", "registered_reach", "ik")
    assert not pg.session.controllers


def test_comparison_rejects_unselected_method_settings(pg):
    from robotforge.control.compare import compare_methods

    with pytest.raises(TaskError, match="not selected"):
        compare_methods(
            "registered_reach",
            ["ik"],
            0.04,
            model=pg.session.model,
            robot="arm",
            world=pg.current_world(),
            method_params={"pd": {}},
        )


def test_policy_saved_settings_must_match_task_before_binding(pg, monkeypatch):
    from types import SimpleNamespace

    from robotforge.control.rl import store
    from robotforge.control.rl.runtime import TrainedPolicy

    monkeypatch.setattr(store, "find_policy", lambda *args: None)
    monkeypatch.setattr(
        TrainedPolicy,
        "load",
        lambda *args: SimpleNamespace(
            name="example",
            meta={"task": "reach", "env": {"task": "registered_reach", "task_params": {"tolerance": 0.04}}},
        ),
    )
    from robotforge.control.runtime import effective_task_params

    assert effective_task_params("registered_reach", "policy:example", {"target": [0.2, 0.1, 0.3]}) == {
        "tolerance": 0.04,
        "target": [0.2, 0.1, 0.3],
    }
    with pytest.raises(TaskError, match="trained for"):
        bind_control(pg.session.model, pg.session.data, "arm", "reach", "policy:example")


def test_comparison_keeps_contact_duration_from_initialized_probe(monkeypatch):
    import robotforge.control.compare as comparison
    from robotforge.control import task_world
    from robotforge.control.tasks import make_task

    model, robot = task_world("wipe")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    task = make_task("wipe", model, robot)
    task.begin(data)
    captured = []
    monkeypatch.setattr(comparison, "compare", lambda *args: captured.append(args[4]) or [])
    comparison.compare_methods("wipe", ["ik"], model=model, robot=robot, data=data, seed=None)
    assert captured == [task.default_seconds]
