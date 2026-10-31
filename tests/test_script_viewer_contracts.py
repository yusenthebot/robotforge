"""Script and viewer adapters share metadata, validation and live controller ownership."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import textwrap
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robotforge.control.runtime import contract_catalog, start_control
from robotforge.control.tasks.reach import Reach
from robotforge.design.tasks.spec import TaskEntry
from robotforge.design.world import RobotEntry, World, compile_world
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


def test_robot_poll_reads_saved_world_once_per_callback(pg, monkeypatch):
    hub = attach(pg)
    load = Mock(wraps=pg.store.load)
    monkeypatch.setattr(pg.store, "load", load)
    first = hub.robot_stream()
    assert first["robots"][0]["name"] == "arm"
    assert load.call_count == 1
    for _ in range(3):
        load.reset_mock()
        assert hub.robot_stream() is None
        assert load.call_count == 1


def test_lesson_robot_poll_uses_lesson_rows_and_saved_task_catalog(pg, monkeypatch):
    saved = pg.current_world()
    saved.tasks.append(TaskEntry(name="saved_press", kind="press", robot="arm", object="button"))
    pg.store.file.write_text(saved.to_yaml(), encoding="utf-8")
    lesson_world = World(
        name="lesson",
        robots=[RobotEntry(name="lesson_dog", template="quadruped")],
        tasks=[TaskEntry(name="lesson_press", kind="press", robot="lesson_dog", object="button")],
    )
    pg.lessons.lesson = SimpleNamespace(world=lesson_world)
    pg.session.load_model(compile_world(lesson_world).model, "lesson")
    policy = SimpleNamespace(spec=SimpleNamespace(name="lesson_walk", commands=SimpleNamespace(lin_vel_x=[0, 0.4])))
    monkeypatch.setattr("robotforge.bridge.policies.for_robot", lambda _: policy)
    hub = attach(pg)
    load = Mock(wraps=pg.store.load)
    monkeypatch.setattr(pg.store, "load", load)
    message = hub.robot_stream()
    assert load.call_count == 1
    assert [row["name"] for row in message["robots"]] == ["lesson_dog"]
    assert message["learned"] == {"lesson_dog": {"policy": "lesson_walk", "max_speed": 0.4, "active": False}}
    assert "learned" in message["modes"]["lesson_dog"]
    assert message["task_contracts"] == contract_catalog(saved)
    assert "saved_press" in rows(message["task_contracts"])
    assert "lesson_press" not in rows(message["task_contracts"])
    load.reset_mock()
    assert hub.robot_stream() is None
    assert load.call_count == 1


def test_robot_poll_without_saved_world_keeps_model_prefix_discovery(pg, monkeypatch):
    pg.store = None
    expected = pg.behaviors.robots()
    hub = attach(pg)
    monkeypatch.setattr(pg.behaviors, "_world", Mock(side_effect=AssertionError("unexpected world reread")))
    message = hub.robot_stream()
    assert message["robots"] == expected
    assert message["learned"] == {}
    assert hub.robot_stream() is None


@pytest.mark.parametrize("phase", ["world_read", "inspection", "message_copy"])
def test_robot_poll_drops_replaced_model_and_retries_next_callback(pg, monkeypatch, phase):
    hub = attach(pg)
    hub.robot_stream()
    replacement = pg.current_world()
    replacement.robots = [RobotEntry(name="replacement_arm", template="arm")]
    compiled = compile_world(replacement)
    replaced = False

    def replace_once():
        nonlocal replaced
        if not replaced:
            replaced = True
            pg.store.file.write_text(replacement.to_yaml(), encoding="utf-8")
            pg.session.load_model(compiled.model, "replacement")

    describe = Mock(wraps=pg.behaviors._describe_robots)
    monkeypatch.setattr(pg.behaviors, "_describe_robots", describe)
    if phase == "world_read":
        load = pg.store.load

        def read_then_replace():
            world = load()
            replace_once()
            return world

        monkeypatch.setattr(pg.store, "load", read_then_replace)
    elif phase == "inspection":

        def describe_then_replace(world):
            result = describe(world)
            replace_once()
            return result

        monkeypatch.setattr(pg.behaviors, "_describe_robots", describe_then_replace)
    else:
        deepcopy = copy.deepcopy

        def copy_then_replace(value, *args, **kwargs):
            result = deepcopy(value, *args, **kwargs)
            if isinstance(value, dict) and value.get("type") == "robot_info":
                replace_once()
            return result

        # Make an emission pending even though the model has not changed yet.
        pending = pg.current_world()
        pending.tasks.append(TaskEntry(name="new_press", kind="press", robot="arm", object="button"))
        pg.store.file.write_text(pending.to_yaml(), encoding="utf-8")
        monkeypatch.setattr(copy, "deepcopy", copy_then_replace)

    assert hub.robot_stream() is None
    assert replaced
    if phase == "world_read":
        assert not describe.called  # stale world rows must never inspect the replacement model
    message = hub.robot_stream()
    assert message["version"] == pg.session.version
    assert [row["name"] for row in message["robots"]] == ["replacement_arm"]
    assert "replacement_arm" in message["modes"] and "arm" not in message["modes"]
    assert hub.robot_stream() is None


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


def test_saved_world_task_metadata_changes_refresh_without_recompile(pg, monkeypatch):
    hub = attach(pg)
    hub.robot_stream()
    version = pg.session.version
    world = pg.current_world()
    load = Mock(wraps=pg.store.load)
    monkeypatch.setattr(pg.store, "load", load)
    world.tasks.append(TaskEntry(name="press_button", kind="press", robot="arm", object="button"))
    pg.store.file.write_text(world.to_yaml())
    catalog = hub.robot_stream()["task_contracts"]
    assert load.call_count == 1
    assert rows(catalog)["press_button"]["source"] == "world"
    assert catalog == ScriptRunner(pg, lambda: False)._op_task_catalog({})
    world.tasks[0].object = "new_button"
    pg.store.file.write_text(world.to_yaml())
    load.reset_mock()
    assert "new_button" in rows(hub.robot_stream()["task_contracts"])["press_button"]["summary"]
    assert load.call_count == 1
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
