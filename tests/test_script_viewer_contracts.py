"""Script and viewer adapters share metadata, validation and live controller ownership."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import textwrap
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from robotforge.control.runtime import contract_catalog, start_control
from robotforge.control.tasks.reach import Reach
from robotforge.design.tasks.spec import TaskEntry
from robotforge.extend import REGISTRY, algorithm, task
from robotforge.playground import Playground
from robotforge.script_runner import ScriptRunner
from robotforge.sim.execution import ExecutionError, RunLimits
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(REGISTRY, "items", {})
    monkeypatch.setattr(REGISTRY, "user_loaded", True)
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("contracts")
    playground.add_robot("arm", template="arm")
    return playground


def attach(pg):
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    return hub


def rows(catalog):
    return {row["name"]: row for row in catalog["tasks"]}


def test_catalog_parity_is_lazy_and_cached(pg, monkeypatch):
    @task("custom_reach", methods=("ik",), summary="Custom reach")
    def build(model, robot, params):
        raise AssertionError("metadata inspection must not bind a plugin task")

    expected = contract_catalog(pg.current_world())
    inspect = Mock(wraps=contract_catalog)
    monkeypatch.setattr("robotforge.control.runtime.contract_catalog", inspect)
    hub = attach(pg)
    runner = ScriptRunner(pg, lambda: False)
    assert runner._op_task_catalog({}) == expected
    assert hub.scene_extras()["task_contracts"] == expected
    assert hub.robot_stream()["task_contracts"] == expected
    assert hub.robot_stream() is None
    assert inspect.call_count == 2  # one script request and one cached viewer catalog
    assert rows(expected)["custom_reach"]["methods"] == ["ik"]
    assert not pg.session.controllers
    assert pg.session.data.time == 0
    hub.scene_extras()["task_contracts"]["tasks"].clear()
    assert hub.scene_extras()["task_contracts"] == expected


def test_unchanged_robot_poll_does_not_copy_the_task_catalog(pg, monkeypatch):
    hub = attach(pg)
    hub.robot_stream()
    deepcopy = copy.deepcopy
    catalog_copies = []

    def inspect(value, *args, **kwargs):
        if isinstance(value, dict) and ("method_schemas" in value or "task_contracts" in value):
            catalog_copies.append(value)
        return deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", inspect)
    assert hub.robot_stream() is None
    assert not catalog_copies


def test_viewer_messages_do_not_expose_private_broadcast_snapshot(pg):
    hub = attach(pg)
    first = hub.robot_stream()
    expected = copy.deepcopy(first)
    first["task_contracts"]["tasks"][0]["params"].clear()
    first["robots"][0]["capabilities"]["manual"]["actuators"].clear()
    first["modes"]["arm"].clear()
    assert hub.robot_stream() is None
    scene = hub.scene_extras()
    assert scene["task_contracts"] == expected["task_contracts"]
    assert scene["robots"][0]["capabilities"] == expected["robots"][0]["capabilities"]
    assert scene["modes"] == expected["modes"]
    scene["task_contracts"]["tasks"].clear()
    assert hub.robot_stream() is None
    assert hub.scene_extras()["task_contracts"] == expected["task_contracts"]


def test_failed_message_copy_keeps_pending_broadcast(pg, monkeypatch):
    hub = attach(pg)
    deepcopy = copy.deepcopy

    def refuse_message(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("type") == "robot_info":
            raise ValueError("cannot detach message")
        return deepcopy(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(copy, "deepcopy", refuse_message)
        with pytest.raises(ValueError, match="cannot detach message"):
            hub.robot_stream()
    assert hub.robot_stream()["task_contracts"] == contract_catalog(pg.current_world())
    assert hub.robot_stream() is None


def test_metadata_replacement_refreshes_existing_viewers_without_consuming_update(pg):
    @task("custom_reach", methods=("ik",), summary="First")
    def build(model, robot, params):
        raise AssertionError("inspection must be lazy")

    hub = attach(pg)
    other = attach(pg)
    assert rows(hub.robot_stream()["task_contracts"])["custom_reach"]["summary"] == "First"
    other.robot_stream()
    version = pg.session.version
    REGISTRY.remove("task", "custom_reach")

    @task("custom_reach", methods=("pd",), summary="Replacement")
    def replacement(model, robot, params):
        raise AssertionError("inspection must be lazy after hot reload")

    # A welcome response must not consume a broadcast intended for connected viewers.
    welcome = hub.scene_extras()["task_contracts"]
    assert rows(welcome)["custom_reach"]["methods"] == ["pd"]
    assert hub.robot_stream()["task_contracts"] == welcome
    assert other.robot_stream()["task_contracts"] == welcome
    assert hub.robot_stream() is None
    assert pg.session.version == version
    REGISTRY.remove("task", "custom_reach")
    assert "custom_reach" not in rows(hub.robot_stream()["task_contracts"])


@pytest.mark.anyio
async def test_script_attaches_and_steps_shared_plugin_controller(pg, tmp_path):
    @task("custom_reach", methods=("ik",))
    def build(model, robot, params):
        return Reach(model, robot, params)

    path = tmp_path / "controller.py"
    path.write_text(
        textwrap.dedent("""\
        import hashlib, json
        from robotforge.script import connect
        sim = connect()
        catalog = sim.task_catalog()
        result = sim.run_controller("arm", "Custom Reach", "Inverse Kinematics", {"target": [0.2, 0.1, 0.2]})
        assert sim.time == 0
        sim.step(0.1)
        digest = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        print(json.dumps({"catalog_hash": digest, "result": result, "time": sim.time}))
    """)
    )
    before = pg.session.data.qpos.copy()
    runner = ScriptRunner(pg, lambda: False)
    result = await runner.run(path, max_sim_seconds=1, max_wall_seconds=20)
    assert result["ok"], result
    output = json.loads(result["output"])
    expected = hashlib.sha256(json.dumps(contract_catalog(pg.current_world()), sort_keys=True).encode()).hexdigest()
    assert output["catalog_hash"] == expected
    assert output["result"]["task"]["task"] == "custom_reach"
    assert output["result"]["method"] == "ik"
    assert output["time"] == pytest.approx(0.1)
    assert not np.array_equal(pg.session.data.qpos, before)
    assert runner._controlled["arm"] is pg.session.controllers["arm"]
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_refused_script_request_does_not_own_or_stop_existing_controller(pg, tmp_path):
    start_control(pg, "arm", "reach", "ik")
    previous = pg.session.controllers["arm"]
    path = tmp_path / "refused.py"
    path.write_text(
        "from robotforge.script import connect\nsim = connect()\n"
        "sim.run_controller('arm', 'reach', 'ik', method_params={'unknown': 1})\n"
    )
    runner = ScriptRunner(pg, lambda: False)
    result = await runner.run(path, max_sim_seconds=1, max_wall_seconds=20)
    assert not result["ok"]
    assert "ScriptError" in result["output"] and "unknown" in result["output"]
    assert pg.session.controllers["arm"] is previous
    assert not runner._command_robots and not runner._controlled
    assert pg.session.data.time == 0


@pytest.mark.anyio
async def test_script_controller_requires_execution(pg):
    runner = ScriptRunner(pg, lambda: False)
    result = await runner._dispatch({"op": "run_controller", "robot": "arm", "task": "reach", "method": "ik"})
    assert result == {"error": "the script has no active execution"}
    assert not pg.session.controllers


def test_viewer_task_command_attaches_without_taking_clock_and_refuses_active_run(pg):
    hub = attach(pg)
    request = {"robot": "arm", "task": "reach", "method": "ik"}
    result = hub.slow_handlers["task_run"](request)
    assert result["task"]["task"] == "reach"
    previous = pg.session.controllers["arm"]
    assert pg.session.clock_owner is None and pg.session.data.time == 0
    with pg.session.execution.run("script", RunLimits(sim_seconds=1)):
        with pytest.raises(ExecutionError, match="already owned"):
            hub.slow_handlers["task_run"](request)
        assert pg.session.controllers["arm"] is previous
    assert not pg.session.execution._commands


def test_saved_world_task_metadata_changes_refresh_without_recompile(pg):
    hub = attach(pg)
    hub.robot_stream()
    version = pg.session.version
    world = pg.current_world()
    world.tasks.append(TaskEntry(name="press_button", kind="press", robot="arm", object="button"))
    pg.store.file.write_text(world.to_yaml())
    catalog = hub.robot_stream()["task_contracts"]
    assert rows(catalog)["press_button"]["source"] == "world"
    assert catalog == ScriptRunner(pg, lambda: False)._op_task_catalog({})
    world.tasks[0].object = "new_button"
    pg.store.file.write_text(world.to_yaml())
    assert "new_button" in rows(hub.robot_stream()["task_contracts"])["press_button"]["summary"]
    assert pg.session.version == version


def test_algorithm_reload_refreshes_contract_catalog(pg):
    @algorithm("custom_algo", params={"gain": 1})
    def train(ctx):
        raise AssertionError("inspection must not train")

    hub = attach(pg)
    first = hub.robot_stream()["task_contracts"]
    REGISTRY.remove("algorithm", "custom_algo")

    @algorithm("custom_algo", params={"gain": 2})
    def replacement(ctx):
        raise AssertionError("inspection must not train")

    second = hub.robot_stream()["task_contracts"]
    assert second != first
    assert second == contract_catalog(pg.current_world())
    assert hub.robot_stream() is None


@pytest.mark.anyio
async def test_cancelled_script_dispatch_tracks_successful_worker_attachment(pg):
    started, proceed = threading.Event(), threading.Event()

    @task("slow_reach", methods=("ik",))
    def build(model, robot, params):
        started.set()
        assert proceed.wait(5), "test did not release the controller worker"
        return Reach(model, robot, params)

    runner = ScriptRunner(pg, lambda: False)
    with pg.session.execution.run("script", RunLimits(sim_seconds=1)) as execution:
        runner._execution = execution
        dispatch = asyncio.create_task(
            runner._dispatch({"op": "run_controller", "robot": "arm", "task": "slow_reach", "method": "ik"})
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            dispatch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await dispatch
        finally:
            proceed.set()
            await execution.drain_workers()
        assert runner._command_robots == {"arm"}
        assert runner._controlled["arm"] is pg.session.controllers["arm"]
        assert "robot:arm" in execution.resources
