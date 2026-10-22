"""Actual MCP controller measurements own one clock and preserve cancellation/record boundaries."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from robotforge.experiments import ExperimentHandle, artifact_path, load_record
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.execution import EndReason
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    value = Playground(session=SimSession.from_xml("<mujoco/>"))
    value.new_world("owned_controller")
    value.add_robot("pole", template="cartpole")
    return value


def decoded(result):
    return json.loads("".join(block.text for block in result.content if getattr(block, "type", "") == "text"))


def request(seconds=0.3, **extra):
    return {"robot": "pole", "task": "balance", "method": "lqr", "seconds": seconds, **extra}


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def execution_records(pg):
    return [
        r for p in (pg.store.dir / "experiments").glob("*/experiment.json") if (r := load_record(p)).kind == "execution"
    ]


def slow_physics(pg, monkeypatch):
    original = pg.session.step

    def slower(n, **kwargs):
        time.sleep(0.001)
        return original(n, **kwargs)

    monkeypatch.setattr(pg.session, "step", slower)


@pytest.mark.anyio
async def test_push_phases_share_exact_budget_and_link_terminal_records(pg):
    server = build_server(pg, 0)
    out = decoded(await server.call_tool("run_controller", request(push_at=0.1)))
    run = load_record(out["execution"]["path"])
    child = load_record(out["experiment"]["path"])
    assert run.status == child.status == "completed"
    assert run.origins == [out["experiment"]]
    assert run.metrics["reason"] == "completed"
    assert run.budget_used["steps"] == child.budget_used["physics_steps"] == 150
    assert run.budget_used["sim_seconds"] == pytest.approx(0.3)
    assert child.algorithm["push"]["physics_step"] == 50
    assert out["keeps_running"] and out["sim_seconds"] == 0.3
    assert pg.session.clock_owner is None and pg.session.controllers["pole"] is not None
    for record in (run, child):
        for ref in record.artifacts:
            artifact_path(pg.store.dir / "experiments" / record.id, ref)


@pytest.mark.anyio
async def test_headless_playing_paused_and_disconnect_have_identical_physics(pg, monkeypatch):
    hubs = []
    attach = pg.attach_viewer
    monkeypatch.setattr(pg, "attach_viewer", lambda hub: (hubs.append(hub), attach(hub)))
    server = build_server(pg, 0)
    hub = hubs[0]
    values, shown = [], []
    for mode in ("headless", "playing", "paused", "disconnect"):
        pg.session.reset()
        pg.session.paused = mode == "paused"
        hub.clients.clear()
        if mode != "headless":
            hub.clients.add(object())
        task = asyncio.create_task(server.call_tool("run_controller", request(push_at=0.1)))
        if mode == "disconnect":
            await until(lambda: pg.session.data.time >= 0.08)
            hub.clients.clear()
        out = decoded(await task)
        record = load_record(out["execution"]["path"])
        assert record.budget_used["steps"] == 150
        values.append(tuple(getattr(pg.session.data, key).copy() for key in ("qpos", "qvel", "ctrl")))
        shown.append(out["shown_live"])
    assert shown == [False, True, False, True]
    for state in values[1:]:
        for got, expected in zip(state, values[0], strict=True):
            np.testing.assert_array_equal(got, expected)


@pytest.mark.anyio
async def test_cancel_stops_only_owned_controller_and_finishes_both_records(pg, monkeypatch):
    slow_physics(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("run_controller", request(10)))
    await until(lambda: pg.session.data.time > 0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pg.session.clock_owner is None and "pole" not in pg.session.controllers
    assert not any(key.startswith("__monitor__") for key in pg.session.controllers)
    run = execution_records(pg)[0]
    assert run.status == "cancelled" and run.metrics["reason"] == "cancelled"
    assert load_record(run.origins[0]["path"]).status == "cancelled"
    assert pg.session.execution.last_result.reason == EndReason.CANCELLED
    assert decoded(await server.call_tool("run_controller", request(0.04)))["keeps_running"]


@pytest.mark.anyio
async def test_reset_invalidates_without_touching_replacement_or_reading_old_task(pg, monkeypatch):
    slow_physics(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("run_controller", request(10)))
    await until(lambda: pg.session.data.time > 0)
    pg.session.reset()
    replacement = SimpleNamespace(rate_hz=500, update=lambda model, data: None)
    pg.session.set_controller("pole", replacement)
    with pytest.raises(ToolError, match="reset during the run"):
        await task
    assert pg.session.controllers["pole"] is replacement
    run = execution_records(pg)[0]
    assert run.status == "failed" and run.metrics["reason"] == "invalidated"
    assert run.budget_used["sim_seconds"] > 0
    assert "unavailable_after_invalidation" in run.metrics["final_state"]
    assert pg.session.execution.last_result.reason == EndReason.INVALIDATED


@pytest.mark.anyio
async def test_lease_refuses_legacy_mutation_and_stop_interrupts_owned_target(pg, monkeypatch):
    slow_physics(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("run_controller", request(10)))
    await until(lambda: pg.session.data.time > 0)
    controller = pg.session.controllers["pole"]
    for tool, args in (("simulate", {"seconds": 1}), ("set_controls", {"controls": {"pole/slide": 9}})):
        with pytest.raises(ToolError, match="already owned"):
            await server.call_tool(tool, args)
    assert pg.session.controllers["pole"] is controller
    await server.call_tool("stop", {"robot": "pole"})
    with pytest.raises(ToolError, match="stopped"):
        await task
    run = execution_records(pg)[0]
    assert run.status == "cancelled" and run.metrics["reason"] == "cancelled"


@pytest.mark.anyio
async def test_cancel_preparation_keeps_lease_and_event_loop_responsive(pg, monkeypatch):
    from robotforge.control import mcp_tools

    started, release = threading.Event(), threading.Event()
    original = mcp_tools.make_controller

    def prepare(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(mcp_tools, "make_controller", prepare)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("run_controller", request()))
    await until(started.is_set)
    task.cancel()
    try:
        # Cancellation drains the worker asynchronously; this timeout must still get CPU time.
        await asyncio.sleep(0.03)
        assert pg.session.body_poses() is not None
        assert not task.done()
        assert pg.session.clock_owner == "run_controller"
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done() and pg.session.clock_owner == "run_controller"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert "pole" not in pg.session.controllers and pg.session.clock_owner is None
    assert execution_records(pg)[0].status == "cancelled"


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_final_recording_failure_does_not_mask_cancel_or_lose_applied_command(pg, monkeypatch, cancel):
    original = ExperimentHandle.artifact

    def disk_full(handle, *args, **kwargs):
        if handle.record.kind == "execution":
            raise OSError("execution disk full")
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(ExperimentHandle, "artifact", disk_full)
    server = build_server(pg, 0)
    if cancel:
        slow_physics(pg, monkeypatch)
        task = asyncio.create_task(server.call_tool("run_controller", request(10)))
        await until(lambda: pg.session.data.time > 0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "pole" not in pg.session.controllers
        assert pg.session.execution.last_result.reason == EndReason.CANCELLED
    else:
        out = decoded(await server.call_tool("run_controller", request(0.04)))
        assert out["keeps_running"] and "pole" in pg.session.controllers
        assert out["execution"]["recording_error"] == "execution disk full"
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_preparation_uses_original_deadline_and_never_attaches_after_timeout(pg, monkeypatch):
    from robotforge import mcp_server
    from robotforge.control import mcp_tools

    original = mcp_tools.make_controller

    def prepare(*args):
        time.sleep(0.06)
        return original(*args)

    monkeypatch.setattr(mcp_tools, "make_controller", prepare)
    monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.03)
    server = build_server(pg, 0)
    with pytest.raises(ToolError, match="wall time limit"):
        await server.call_tool("run_controller", request())
    run = execution_records(pg)[0]
    assert run.status == "stopped" and run.metrics["reason"] == "wall_limit"
    assert run.budget_used["steps"] == 0
    assert pg.session.clock_owner is None and "pole" not in pg.session.controllers


@pytest.mark.anyio
async def test_cancel_rests_rover_velocity_actuators_without_changing_another_robot(pg, monkeypatch):
    pg.add_robot("rover", template="rover", pos=[1, 0])
    slow_physics(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(
        server.call_tool(
            "run_controller",
            {
                "robot": "rover",
                "task": "drive_to",
                "method": "pd",
                "seconds": 10,
                "task_params": {"goal": [2.0, 0.0]},
            },
        )
    )
    await until(lambda: pg.session.data.time > 0.1)
    assert np.max(np.abs(pg.session.data.ctrl)) > 0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "rover" not in pg.session.controllers
    np.testing.assert_array_equal(pg.session.data.ctrl, 0)
    pg.session.step(10)
    np.testing.assert_array_equal(pg.session.data.ctrl, 0)


@pytest.mark.anyio
async def test_push_does_not_renew_the_measurement_wall_budget(pg, monkeypatch):
    from robotforge import mcp_server
    from robotforge.sim import execution as kernel

    clock = [0.0]
    monkeypatch.setattr(kernel, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.25)
    original = pg.session.step

    def slow_batch(n, **kwargs):
        clock[0] += 0.1
        return original(n, **kwargs)

    monkeypatch.setattr(pg.session, "step", slow_batch)
    server = build_server(pg, 0)
    out = decoded(await server.call_tool("run_controller", request(0.08, push_at=0.04)))
    run = load_record(out["execution"]["path"])
    child = load_record(out["experiment"]["path"])
    assert run.status == "stopped" and run.metrics["reason"] == "wall_limit"
    assert run.budget_used["steps"] == child.budget_used["physics_steps"] == 20
    assert child.algorithm["push"]["physics_step"] == 20
    assert out["sim_seconds"] == 0.04 and "computer time" in out["stopped_early"]


@pytest.mark.anyio
async def test_stop_and_recording_failure_still_record_cancelled(pg, monkeypatch):
    original = ExperimentHandle.artifact

    def disk_full(handle, *args, **kwargs):
        if handle.record.kind == "execution":
            raise OSError("execution disk full")
        return original(handle, *args, **kwargs)

    monkeypatch.setattr(ExperimentHandle, "artifact", disk_full)
    slow_physics(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("run_controller", request(10)))
    await until(lambda: pg.session.data.time > 0)
    await server.call_tool("stop", {"robot": "pole"})
    with pytest.raises(ToolError, match="stopped"):
        await task
    run = execution_records(pg)[0]
    assert run.status == "cancelled" and run.termination["reason"] == "cancelled"
    assert "recording_failed" in run.termination["error"]
    assert pg.session.clock_owner is None
