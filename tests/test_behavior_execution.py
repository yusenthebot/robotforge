"""Pose/gripper ownership, unchanged queued behavior, and actual script boundaries."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from robotforge.experiments import ExperimentHandle, artifact_path, load_record
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.script_runner import ScriptRunner
from robotforge.sim.execution import EndReason, RunLimits
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    value = Playground(session=SimSession.from_xml("<mujoco/>"))
    value.new_world("owned_behavior")
    value.add_robot("kid", template="humanoid")
    value.add_robot("arm", template="arm", pos=[0.8, 0])
    return value


def decoded(result):
    return json.loads("".join(b.text for b in result.content if getattr(b, "type", "") == "text"))


def request(tool):
    return {"robot": "kid", "name": "wave"} if tool == "pose" else {"robot": "arm", "state": "close"}


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def slow_steps(pg, monkeypatch):
    step = pg.session.step

    def slower(n, **kwargs):
        time.sleep(0.002)
        return step(n, **kwargs)

    monkeypatch.setattr(pg.session, "step", slower)


def records(pg):
    return [
        r for p in (pg.store.dir / "experiments").glob("*/experiment.json") if (r := load_record(p)).kind == "execution"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_normal_results_record_actual_budget_and_leave_controller_attached(pg, tool):
    server = build_server(pg, 0)
    out = decoded(await server.call_tool(tool, request(tool)))
    record = load_record(out["execution"]["path"])
    robot = request(tool)["robot"]
    assert record.status == "completed" and record.metrics["behavior_done"]
    assert record.environment["entrypoint"] == f"mcp_{tool}"
    assert record.task == {"domain": "behavior", "robot": robot, "name": tool}
    assert record.algorithm["name"] == tool and record.seeds["mode"] == "current_state"
    assert record.budget_used["steps"] > 0 and record.budget_used["generation_valid"]
    assert out["sim_seconds"] == round(record.budget_used["sim_seconds"], 2)
    assert not out["shown_live"] and pg.session.clock_owner is None
    assert robot in pg.session.controllers
    assert ({"robot", "pose", "upright"} if tool == "pose" else {"gripper"}) <= out.keys()
    for ref in record.artifacts:
        artifact_path(out["execution"]["path"], ref)


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_four_viewing_states_keep_identical_steps_and_physics(pg, monkeypatch, tool):
    hubs = []
    attach = pg.attach_viewer
    monkeypatch.setattr(pg, "attach_viewer", lambda hub: (hubs.append(hub), attach(hub)))
    server = build_server(pg, 0)
    hub = hubs[0]
    states, counts, shown = [], [], []
    for mode in ("headless", "playing", "paused", "disconnect"):
        pg.session.reset()
        pg.session.paused = mode == "paused"
        hub.clients.clear()
        if mode != "headless":
            hub.clients.add(object())
        task = asyncio.create_task(server.call_tool(tool, request(tool)))
        if mode == "disconnect":
            await until(lambda: pg.session.data.time >= 0.06)
            hub.clients.clear()
        out = decoded(await task)
        record = load_record(out["execution"]["path"])
        states.append(tuple(getattr(pg.session.data, k).copy() for k in ("qpos", "qvel", "ctrl")))
        counts.append(record.budget_used["steps"])
        shown.append(out["shown_live"])
    assert counts == [counts[0]] * 4 and shown == [False, True, False, True]
    for state in states[1:]:
        for got, expected in zip(state, states[0], strict=True):
            np.testing.assert_array_equal(got, expected)


@pytest.mark.anyio
async def test_lease_refuses_both_commands_before_any_mutation_or_record(pg):
    server = build_server(pg, 0)
    before = pg.session.controllers.copy()
    with pg.session.execution.run("another", RunLimits(1, 2)):
        for tool in ("pose", "gripper"):
            with pytest.raises(ToolError, match="already owned"):
                await server.call_tool(tool, request(tool))
    assert pg.session.controllers == before and not records(pg)


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_actual_stop_interrupts_only_matching_resource_and_releases_lease(pg, monkeypatch, tool):
    slow_steps(pg, monkeypatch)
    hubs = []
    attach = pg.attach_viewer
    monkeypatch.setattr(pg, "attach_viewer", lambda hub: (hubs.append(hub), attach(hub)))
    server = build_server(pg, 0)
    hubs[0].clients.add(object())
    task = asyncio.create_task(server.call_tool(tool, request(tool)))
    await until(lambda: pg.session.data.time >= 0.04)
    with pytest.raises(ToolError):
        await server.call_tool("stop", {"robot": "missing"})
    await server.call_tool("stop", {"robot": "arm" if tool == "pose" else "kid"})
    assert pg.session.clock_owner == tool and not task.done()
    await server.call_tool("stop", {"robot": request(tool)["robot"]})
    with pytest.raises(ToolError, match="stopped"):
        await task
    record = records(pg)[0]
    assert record.status == "cancelled" and record.metrics["reason"] == "cancelled"
    assert pg.session.clock_owner is None
    if tool == "pose":
        assert pg.session.controllers["kid"].command_speed == 0
    else:
        assert "arm" not in pg.session.controllers
    assert decoded(await server.call_tool(tool, request(tool)))["execution"]


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["reset", "model", "controller"])
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_invalidation_preserves_replacement_and_closes_record(pg, monkeypatch, tool, change):
    slow_steps(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool(tool, request(tool)))
    await until(lambda: pg.session.data.time >= 0.04)
    if change == "reset":
        pg.session.reset()
    elif change == "model":
        pg.session.load_model(SimSession.from_xml("<mujoco/>").model, "replacement")
    robot = request(tool)["robot"]
    replacement = SimpleNamespace(rate_hz=500, update=lambda model, data: None)
    pg.session.set_controller(robot, replacement)
    with pytest.raises(ToolError, match="reset|changed"):
        await task
    assert pg.session.controllers[robot] is replacement and pg.session.clock_owner is None
    record = records(pg)[0]
    assert record.status == "failed" and record.metrics["reason"] == "invalidated"
    if change != "controller":
        assert not record.metrics["generation_valid"]
        assert "unavailable_after_invalidation" in record.metrics["final_state"]


@pytest.mark.anyio
async def test_cancel_gripper_on_moving_carrier_preserves_base_and_position_servo_hold(pg, monkeypatch):
    pg.add_robot("rover", template="rover", mounts=[{"template": "arm"}], pos=[2, 0])
    pg.behaviors.move("rover", 0.2, 0)
    carrier = pg.session.controllers["rover"]
    slow_steps(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("gripper", {"robot": "rover", "state": "close"}))
    await until(lambda: pg.session.data.time >= 0.06)
    m = pg.session.model
    ids = [i for i in range(m.nu) if m.actuator(i).name.startswith("rover/arm/")]
    held = pg.session.data.ctrl[ids].copy()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "rover/arm" not in pg.session.controllers and pg.session.controllers["rover"] is carrier
    assert carrier.status()["speed"] == 0.2
    np.testing.assert_array_equal(pg.session.data.ctrl[ids], held)
    pg.session.step(10)
    np.testing.assert_array_equal(pg.session.data.ctrl[ids], held)
    record = records(pg)[0]
    assert record.task["robot"] == "rover/arm" and record.status == "cancelled"


@pytest.mark.anyio
async def test_reused_arm_retains_queued_motion_and_five_second_cap(pg):
    arm = pg.behaviors.set_gripper("arm", "open")
    arm.queue.clear()
    arm.set_gripper("open", wait=6.0)
    server = build_server(pg, 0)
    out = decoded(await server.call_tool("gripper", request("gripper")))
    record = load_record(out["execution"]["path"])
    assert pg.session.controllers["arm"] is arm and not arm.done
    assert out["sim_seconds"] == 5.0 and record.budget_used["steps"] == 2500
    assert not record.metrics["behavior_done"]
    assert arm.queue[-1].label == "gripper close"


@pytest.mark.anyio
async def test_already_done_pose_can_finish_without_a_physics_step(pg, monkeypatch):
    ctrl = SimpleNamespace(done=True, rate_hz=500, update=lambda model, data: None)

    def strike(*args):
        pg.session.set_controller("kid", ctrl)
        return ctrl

    monkeypatch.setattr(pg.behaviors, "strike_pose", strike)
    out = decoded(await build_server(pg, 0).call_tool("pose", request("pose")))
    assert out["sim_seconds"] == 0
    assert load_record(out["execution"]["path"]).budget_used["steps"] == 0
    assert pg.session.controllers["kid"] is ctrl and pg.session.clock_owner is None


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_wall_limit_preserves_applied_controller_and_records_partial_budget(pg, monkeypatch, tool):
    from robotforge import mcp_server
    from robotforge.sim import execution

    clock = [0.0]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.15)
    step = pg.session.step

    def slower(n, **kwargs):
        clock[0] += 0.1
        return step(n, **kwargs)

    monkeypatch.setattr(pg.session, "step", slower)
    out = decoded(await build_server(pg, 0).call_tool(tool, request(tool)))
    record = load_record(out["execution"]["path"])
    assert record.status == "stopped" and record.metrics["reason"] == "wall_limit"
    assert 0 < record.budget_used["steps"] < 100
    assert "computer time" in out["stopped_early"] and not record.metrics["behavior_done"]
    assert request(tool)["robot"] in pg.session.controllers and pg.session.clock_owner is None


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_recording_failure_does_not_mask_cancellation_or_applied_gripper(pg, monkeypatch, cancel):
    artifact = ExperimentHandle.artifact

    def disk_full(handle, *args, **kwargs):
        if handle.record.kind == "execution":
            raise OSError("execution disk full")
        return artifact(handle, *args, **kwargs)

    monkeypatch.setattr(ExperimentHandle, "artifact", disk_full)
    server = build_server(pg, 0)
    if cancel:
        slow_steps(pg, monkeypatch)
        task = asyncio.create_task(server.call_tool("gripper", request("gripper")))
        await until(lambda: pg.session.data.time >= 0.04)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "arm" not in pg.session.controllers
        assert pg.session.execution.last_result.reason == EndReason.CANCELLED
    else:
        out = decoded(await server.call_tool("gripper", request("gripper")))
        assert out["execution"]["recording_error"] == "execution disk full"
        assert "arm" in pg.session.controllers
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_real_script_wrappers_keep_bool_none_results_and_one_cumulative_budget(pg, tmp_path):
    path = tmp_path / "commands.py"
    path.write_text(
        "from robotforge.script import connect\ns=connect()\nr=s.robot('kid').pose('wave')\n"
        "g=s.gripper('arm','close')\ns.robot('arm').gripper.open()\nprint(type(r).__name__,g is None)\n",
        encoding="utf-8",
    )
    out = await ScriptRunner(pg, lambda: False).run(path, max_sim_seconds=10)
    assert out["ok"] and "bool True" in out["output"]
    assert out["sim_seconds"] > 3 and pg.session.execution.last_result.sim_seconds <= 10
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_script_polling_does_not_adopt_a_replacement_after_identity_failure(pg, monkeypatch):
    runner = ScriptRunner(pg, lambda: False)
    slow_steps(pg, monkeypatch)
    with pg.session.execution.run("script", RunLimits(10, 10)) as run:
        runner._execution = run
        task = asyncio.create_task(runner._dispatch({"op": "gripper", **request("gripper")}))
        await until(lambda: pg.session.data.time >= 0.04)
        original = runner._controlled["arm"]
        replacement = SimpleNamespace(done=True, rate_hz=500, update=lambda model, data: None)
        pg.session.set_controller("arm", replacement)
        out = await task
        assert "controller changed" in out["error"]
        assert "result" in await runner._dispatch({"op": "time"})
        assert runner._controlled["arm"] is original
        runner._remember_command("arm")
        assert runner._controlled["arm"] is replacement


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_final_metric_failure_keeps_success_result_and_attached_controller(pg, monkeypatch, tool):
    update = ExperimentHandle.update

    def disk_full(handle, **changes):
        if "behavior_done" in changes.get("metrics", {}):
            raise OSError("metric disk full")
        return update(handle, **changes)

    monkeypatch.setattr(ExperimentHandle, "update", disk_full)
    out = decoded(await build_server(pg, 0).call_tool(tool, request(tool)))
    assert out["execution"]["recording_error"] == "metric disk full"
    assert request(tool)["robot"] in pg.session.controllers and pg.session.clock_owner is None
    assert load_record(out["execution"]["path"]).termination["reason"] == "recording_failed"


@pytest.mark.anyio
@pytest.mark.parametrize("tool", ["pose", "gripper"])
async def test_expired_snapshot_preamble_never_issues_a_behavior(pg, monkeypatch, tool):
    from robotforge import mcp_server

    snapshot = ExperimentHandle.snapshot_environment
    before = pg.session.controllers.copy()

    def slower(*args, **kwargs):
        value = snapshot(*args, **kwargs)
        time.sleep(0.06)
        return value

    monkeypatch.setattr(ExperimentHandle, "snapshot_environment", slower)
    monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.03)
    with pytest.raises(ToolError, match="wall time limit"):
        await build_server(pg, 0).call_tool(tool, request(tool))
    assert pg.session.controllers == before and pg.session.clock_owner is None
    record = records(pg)[0]
    assert record.status == "stopped" and record.metrics["reason"] == "wall_limit"
    assert record.budget_used["steps"] == 0


@pytest.mark.anyio
async def test_cancelled_pose_keeps_existing_zero_speed_balance_and_pose_stop_semantics(pg, monkeypatch):
    slow_steps(pg, monkeypatch)
    task = asyncio.create_task(build_server(pg, 0).call_tool("pose", request("pose")))
    await until(lambda: pg.session.data.time >= 0.06)
    ctrl = pg.session.controllers["kid"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert pg.session.controllers["kid"] is ctrl
    assert ctrl.command_speed == ctrl.command_turn == 0
    # Existing stop intentionally retains a pending/current pose on its balancing controller.
    pose_time = ctrl._pose_t
    assert ctrl._pose is not None or ctrl._pending is not None
    pg.session.step(100)
    assert ctrl._pose_t > pose_time
    assert records(pg)[0].status == "cancelled" and pg.session.clock_owner is None
