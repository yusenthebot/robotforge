"""Arm-family ownership at planner, controller, reset, budget and script boundaries."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from test_behavior_execution import decoded, records, slow_steps, until
from test_fetch import _wall_world

from robotforge.behaviors import catch as catching
from robotforge.behaviors import fetch, relay
from robotforge.behaviors.execution import ArmExecution
from robotforge.experiments import ExperimentHandle, load_record
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.script_runner import ScriptRunner
from robotforge.sim.execution import EndReason, ExecutionError, RunLimits
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("arm_execution", "wood")
    yield playground
    playground.renderer.close()


def recorded_budget(wall=5.0):
    updates = []
    recorded = SimpleNamespace(
        watched=lambda: False,
        wall_limit=lambda: wall,
        call=SimpleNamespace(record=SimpleNamespace(update=lambda **change: updates.append(change))),
    )
    return recorded, updates


@pytest.mark.anyio
async def test_cancelled_mcp_planner_retains_lease_until_worker_finishes_and_disarms_relay(pg, monkeypatch):
    pg.add_robot("arm", template="arm")
    pg.add_robot("other", template="arm", pos=[1, 0])
    entered, release = threading.Event(), threading.Event()
    callbacks, roots = [], []

    def plan(*args):
        giver = pg.behaviors.set_gripper("arm", "close")
        root = relay.Relay(
            giver,
            "other",
            "cube",
            np.zeros(2),
            lambda: True,
            lambda: callbacks.append("late second arm"),
            on_done=lambda: callbacks.append("late completion"),
        )
        roots.append(root)
        pg.session.set_controller(relay.KEY + "arm", root)
        entered.set()
        if not release.wait(3):
            raise TimeoutError("test did not release the planner")
        return root, {"object": "cube", "region": None}

    monkeypatch.setattr(pg.behaviors, "hand_over", plan)
    task = asyncio.create_task(
        build_server(pg, 0).call_tool(
            "pick_and_place", {"robot": "arm", "object": "cube", "target": "hold", "hand_to": "other"}
        )
    )
    try:
        await until(entered.is_set)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        assert pg.session.clock_owner == "pick_and_place"
        assert pg.session.execution.active is not None
        task.cancel()  # a second cancellation must not release a still-mutating planner
        await asyncio.sleep(0.01)
        assert not task.done() and pg.session.clock_owner == "pick_and_place"
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
    root = roots[0]
    assert root.note and root.on_done is None
    assert isinstance(root.start(), str)
    root.update(pg.session.model, pg.session.data)
    pg.session.step(20)
    assert not callbacks
    assert relay.KEY + "arm" not in pg.session.controllers
    assert "arm" not in pg.session.controllers
    assert pg.session.execution.active is None and pg.session.clock_owner is None
    assert records(pg)[0].status == "cancelled"


@pytest.mark.anyio
@pytest.mark.parametrize("alias", [False, True])
async def test_foreign_controller_cannot_replace_owned_arm_then_run_the_old_actor_in_same_step(pg, monkeypatch, alias):
    if alias:
        pg.add_robot("picker", design="lift_picker")
        name, endpoint = "picker", "picker/arm"
    else:
        pg.add_robot("arm", template="arm")
        name = endpoint = "arm"
    called = []
    replacement = SimpleNamespace(rate_hz=500, update=lambda m, d: called.append("replacement"))

    def foreign_update(model, data):
        called.append("foreign")
        pg.session.set_controller(endpoint, replacement)

    # The foreign actor is first in the existing controller order, before the arm is attached.
    pg.session.set_controller("foreign", SimpleNamespace(rate_hz=500, update=foreign_update))
    before = pg.session.steps_taken
    with pg.session.execution.run("arm_test", RunLimits(1, 5)) as run:
        with pytest.raises(ExecutionError, match="outside the arm command") as exc:
            async with ArmExecution(pg, run) as family:
                family.reserve(name)
                arm = await family.plan(pg.behaviors.set_gripper, name, "close")
                assert endpoint in family.expected
                monkeypatch.setattr(arm, "update", lambda m, d: called.append("stale arm"))
                await family.advance(lambda: False, 0.02, "motion")
        assert exc.value.reason == EndReason.INVALIDATED
        assert run.current
    assert called == ["foreign"]
    assert pg.session.steps_taken == before
    assert pg.session.controllers[endpoint] is replacement
    assert not any(isinstance(c, relay.Relay) for c in pg.session.controllers.values())


@pytest.mark.anyio
async def test_cancelling_alias_scope_removes_mounted_arm_and_keeps_foreign_base_command(pg):
    pg.add_robot("picker", design="lift_picker")
    with pg.session.execution.run("arm_test", RunLimits(1, 5)) as run, pytest.raises(asyncio.CancelledError):
        async with ArmExecution(pg, run) as family:
            family.reserve("picker")
            await family.plan(pg.behaviors.set_gripper, "picker", "close")
            # A new command on a previously empty carrier slot is foreign, even though the alias names it.
            replacement = SimpleNamespace(rate_hz=500, update=lambda m, d: None)
            pg.session.set_controller("picker", replacement)
            raise asyncio.CancelledError
    assert "picker/arm" not in pg.session.controllers
    assert pg.session.controllers["picker"] is replacement


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["reset", "model"])
async def test_real_errand_restores_old_model_hold_settings_before_reset_or_swap(pg, change):
    _wall_world(pg)
    model = pg.session.model
    fields = ("actuator_gainprm", "actuator_biasprm", "geom_condim", "geom_solmix", "geom_solimp", "geom_solref")
    before = {name: getattr(model, name).copy() for name in fields}
    noslip = model.opt.noslip_iterations
    with (
        pg.session.execution.run("pick_and_place", RunLimits(2, 30)) as run,
        pytest.raises(ExecutionError, match="reset|changed"),
    ):
        async with ArmExecution(pg, run) as family:
            family.reserve("picker")
            motion, _ = await family.plan(pg.behaviors.pick_and_place, "picker", "cube", "goal")
            assert isinstance(motion, fetch.Errand)
            await family.advance(lambda: motion._noslip is not None, 0.05, "first update")
            assert model.opt.noslip_iterations > noslip
            assert motion.arm.carry_weight and motion._squeezed is not None
            assert np.any(model.geom_condim != before["geom_condim"])
            if change == "reset":
                pg.session.reset()
            else:
                pg.session.load_model(SimSession.from_xml("<mujoco/>").model, "replacement")
                assert pg.session.model is not model
            assert model.opt.noslip_iterations == noslip
            for name in fields:
                np.testing.assert_array_equal(getattr(model, name), before[name])
            assert not motion.arm.carry_weight
            assert motion._noslip is motion._squeezed is motion._pads is motion._rolls is None
            assert motion.done
            family.check()
    assert pg.behaviors.errand("picker") is None


@pytest.mark.anyio
async def test_phase_budget_reconfiguration_keeps_elapsed_simulation_and_original_wall_start(pg, monkeypatch):
    from robotforge.sim import execution

    clock = [100.0]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    recorded, updates = recorded_budget()
    with pg.session.execution.run("arm_test", RunLimits(20, 100)) as run:
        async with ArmExecution(pg, run) as family:
            await family.advance(lambda: False, 0.02, "carrier")
            clock[0] = 103.0  # preparation took three seconds after the lease began
            family.budget(recorded, [0.02, 0.03])
            assert run.deadline == 110.0
            assert run.sim_seconds == pytest.approx(0.02)
            result = await family.advance(lambda: False, 0.08, "motion")
            assert "sim_limit" in result["stopped_early"]
            assert run.steps == 25 and run.sim_seconds == pytest.approx(0.05)
            assert [p["steps"] for p in family.phases] == [10, 15]
            assert updates[-1]["budget"]["includes_preparation"]
            assert updates[-1]["budget"]["wall_seconds"] == 10


@pytest.mark.anyio
async def test_phase_budget_does_not_give_slow_preparation_a_new_wall_deadline(pg, monkeypatch):
    from robotforge.sim import execution

    clock = [0.0]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    recorded, _ = recorded_budget(wall=1.0)
    with pg.session.execution.run("arm_test", RunLimits(20, 100)) as run:
        with pytest.raises(ExecutionError, match="wall time limit") as exc:
            async with ArmExecution(pg, run) as family:
                clock[0] = 2.1
                family.budget(recorded, [1, 0.5])
        assert exc.value.reason == EndReason.WALL_LIMIT
        assert run.steps == 0 and run.deadline == 2.0


@pytest.mark.anyio
async def test_actual_moving_carrier_wait_is_charged_to_same_sim_budget(pg):
    pg.add_robot("rover", template="rover", mounts=[{"template": "arm"}])
    pg.behaviors.move("rover", 0.2, 0)
    pg.run_fast(lambda: False, 0.5)
    before = pg.session.steps_taken
    with pg.session.execution.run("move_arm", RunLimits(10, 20)) as run:
        async with ArmExecution(pg, run) as family:
            response = await family.carrier_stands("rover")
            assert "rover stopped first" in response["carrier_stopped"]
            waited = run.sim_seconds
            assert waited > 0 and pg.session.controllers["rover"].status()["speed"] == 0
            assert family.phases[0]["name"] == "carrier_stands"
            # Deliberately leave only 0.1 seconds after waiting, to detect a fresh phase clock.
            run.limits = RunLimits(sim_seconds=waited + 0.1, wall_seconds=20)
            result = await family.advance(lambda: False, 1, "motion")
            assert "sim_limit" in result["stopped_early"]
            assert run.sim_seconds == pytest.approx(waited + 0.1)
            assert run.steps == pg.session.steps_taken - before
            assert run.steps == sum(p["steps"] for p in family.phases)


@pytest.mark.anyio
async def test_mcp_move_arm_keeps_preexisting_queue_and_exact_low_level_physics(pg):
    pg.add_robot("arm", template="arm")

    def queue():
        return pg.behaviors.move_arm("arm", [0.2, -0.15, 0.2], "close")

    queue()
    direct = pg.behaviors.move_arm("arm", [0.2, 0.15, 0.2], "open")
    before = pg.session.steps_taken
    pg.run_fast(lambda: direct.done, 20)
    steps = pg.session.steps_taken - before
    state = {k: getattr(pg.session.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
    pg.session.reset()
    queued = queue()
    output = decoded(
        await build_server(pg, 0).call_tool(
            "move_arm", {"robot": "arm", "position": [0.2, 0.15, 0.2], "gripper": "open"}
        )
    )
    record = load_record(output["execution"]["path"])
    assert pg.session.controllers["arm"] is queued and queued.done
    assert record.budget["phase_caps"] == [20.0] and record.budget["sim_seconds"] == 20.0
    assert record.metrics["phases"][0]["limit_seconds"] == 20.0
    assert record.budget_used["steps"] == steps
    for key, expected in state.items():
        np.testing.assert_array_equal(getattr(pg.session.data, key), expected)


@pytest.mark.anyio
async def test_mcp_stop_keeps_cancelled_reason_when_it_removes_owned_arm(pg, monkeypatch):
    pg.add_robot("arm", template="arm")
    slow_steps(pg, monkeypatch)
    server = build_server(pg, 0)
    task = asyncio.create_task(server.call_tool("move_arm", {"robot": "arm", "position": [0.2, 0.15, 0.2]}))
    await until(lambda: pg.session.data.time >= 0.04)
    await server.call_tool("stop", {"robot": "arm"})
    with pytest.raises(ToolError, match="stopped"):
        await task
    assert "arm" not in pg.session.controllers
    assert pg.session.clock_owner is None
    record = records(pg)[0]
    assert record.status == "cancelled" and record.metrics["reason"] == "cancelled"


@pytest.mark.anyio
async def test_real_script_can_replace_arm_with_later_explicit_commands_and_drive(pg, tmp_path):
    pg.add_robot("arm", template="arm")
    pg.add_robot("rover", template="rover", pos=[2, 0])
    script = tmp_path / "later_commands.py"
    script.write_text(
        "from robotforge.script import connect\n"
        "s=connect()\n"
        "p=s.move_arm('arm',[0.2,0.15,0.2],gripper='open')\n"
        "s.stop('arm')\n"
        "g=s.gripper('arm','close')\n"
        "s.move('rover',0.2)\n"
        "s.step(0.1)\n"
        "s.stop('rover')\n"
        "print('later commands',len(p),g is None)\n",
        encoding="utf-8",
    )
    result = await ScriptRunner(pg, lambda: False).run(script, max_sim_seconds=5, max_wall_seconds=10)
    assert result["ok"], result
    assert "later commands 3 True" in result["output"]
    assert pg.session.execution.last_result.reason == EndReason.COMPLETED
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_expired_snapshot_does_not_stop_a_moving_carrier_or_issue_arm_motion(pg, monkeypatch):
    from robotforge import mcp_server
    from robotforge.sim import execution

    pg.add_robot("rover", template="rover", mounts=[{"template": "arm"}])
    pg.behaviors.move("rover", 0.2, 0)
    pg.run_fast(lambda: False, 0.1)
    before = pg.session.controllers.copy()
    state = {k: getattr(pg.session.data, k).copy() for k in ("qpos", "qvel", "ctrl")}
    clock = [0.0]
    monkeypatch.setattr(execution, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(mcp_server, "FAST_WALL_LIMIT", 0.03)
    snapshot = ExperimentHandle.snapshot_environment

    def delayed(*args, **kwargs):
        value = snapshot(*args, **kwargs)
        clock[0] += 0.1
        return value

    monkeypatch.setattr(ExperimentHandle, "snapshot_environment", delayed)
    with pytest.raises(ToolError, match="wall time limit"):
        await build_server(pg, 0).call_tool("move_arm", {"robot": "rover", "position": [0.2, 0, 0.3]})
    assert pg.session.controllers == before
    assert pg.session.controllers["rover"].status()["speed"] == 0.2
    for key, expected in state.items():
        np.testing.assert_array_equal(getattr(pg.session.data, key), expected)
    record = records(pg)[0]
    assert record.status == "stopped" and record.metrics["reason"] == "wall_limit"
    assert record.budget_used["steps"] == 0
    assert pg.session.clock_owner is None


def settling_playground(start_time, timestep=0.002):
    """A moving physical state makes an extra legacy settling batch observable in every comparison."""
    session = SimSession.from_xml(
        f'<mujoco><option timestep="{timestep}"/><worldbody><body>'
        '<joint name="slide" type="slide" axis="1 0 0"/><geom size=".1"/>'
        '</body></worldbody><actuator><motor joint="slide"/></actuator></mujoco>'
    )
    session.data.time = start_time
    session.data.ctrl[:] = 0.25
    return Playground(session=session)


def legacy_mcp_settle(pg, seconds):
    """The previous headless MCP outer loop, including its 0.1 s slices and final epsilon."""
    start = float(pg.session.data.time)
    while pg.session.data.time - start < seconds - 1e-9:
        pg.run_fast(lambda: False, min(0.1, seconds - (pg.session.data.time - start)))


@pytest.mark.anyio
@pytest.mark.parametrize("timestep,start_time", [(0.002, t) for t in (0.0, 0.1, 6.64, 28.72, 34.94)] + [(0.1, 5.0)])
@pytest.mark.parametrize("seconds", [0.3, 0.5])
async def test_legacy_settlement_matches_nested_mcp_elapsed_rule_exactly(timestep, start_time, seconds):
    legacy, owned = settling_playground(start_time, timestep), settling_playground(start_time, timestep)
    try:
        legacy_mcp_settle(legacy, seconds)
        if timestep == 0.1 and seconds == 0.3:
            assert legacy.session.steps_taken == 4
        with owned.session.execution.run("arm_test", RunLimits(2, 5)) as run:
            async with ArmExecution(owned, run) as family:
                await family.advance(lambda: False, seconds, "settle", legacy_elapsed=True)
                assert run.steps == legacy.session.steps_taken
                assert family.phases[-1]["limit_seconds"] == seconds
        assert owned.session.data.time == legacy.session.data.time
        for key in ("qpos", "qvel", "ctrl"):
            np.testing.assert_array_equal(getattr(owned.session.data, key), getattr(legacy.session.data, key))
    finally:
        legacy.renderer.close()
        owned.renderer.close()


@pytest.mark.anyio
@pytest.mark.parametrize("limit", ["sim", "steps"])
async def test_strict_whole_run_budget_clips_legacy_extra_settlement_chunk(limit):
    legacy, owned = settling_playground(6.64), settling_playground(6.64)
    try:
        legacy_mcp_settle(legacy, 0.5)
        assert legacy.session.steps_taken > 250  # this starting float exposes the extra 20 ms batch
        limits = RunLimits(sim_seconds=0.5, wall_seconds=5) if limit == "sim" else RunLimits(steps=250, wall_seconds=5)
        with owned.session.execution.run("arm_test", limits) as run:
            async with ArmExecution(owned, run) as family:
                result = await family.advance(lambda: False, 0.5, "settle", legacy_elapsed=True)
                assert run.steps == owned.session.steps_taken == 250
                assert run.sim_seconds == pytest.approx(0.5)
                assert ("sim_limit" if limit == "sim" else "step_limit") in result["stopped_early"]
                assert family.phases[-1]["limit_seconds"] == 0.5
                again = await family.advance(lambda: False, 0.5, "second settle", legacy_elapsed=True)
                assert again["sim_seconds"] == 0 and run.steps == 250
    finally:
        legacy.renderer.close()
        owned.renderer.close()


@pytest.mark.anyio
async def test_late_catch_then_errand_descendant_adopts_and_stops_existing_stationary_gait(pg):
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.3, 0.1, 0.5])
    pg.behaviors.move("picker", 0.0, 0.0)
    gait = pg.session.controllers["picker"]
    body = pg.session.model.body("cube").id
    callbacks = []
    with pg.session.execution.run("pick_and_place", RunLimits(2, 5)) as run:
        async with ArmExecution(pg, run) as family:
            family.reserve("picker")
            with family.command():
                arm = pg.behaviors.set_gripper("picker", "close")
                meeting = catching.Meeting(np.zeros(3), np.zeros(2), arm.q.copy(), None, 0.0, np.zeros(3), arm.q.copy())
                caught = catching.Catch(arm, body, 0.02, meeting, then=lambda *args: callbacks.append("catch"))
                pg.session.set_controller("picker/arm", caught)
                family.bind(caught)
            assert id(gait) not in family.owned  # a fixed catch initially has no authority over its carrier
            after = catching.Then(arm, lambda: callbacks.append("then"))
            errand = fetch.Errand(
                manager=pg.behaviors,
                robot="picker",
                arm=arm,
                obj="cube",
                body=body,
                half=0.02,
                radius=0.03,
                hold=np.zeros(3),
                hold_q=arm.q.copy(),
                pick=None,
                drop=fetch.Spot((0.5, 0.0), 0.0, (0.6, 0.0, 0.5)),
                place=fetch.Place([np.array([0.6, 0.0, 0.5])], 0.5, {}, "goal"),
            )
            # The real state machines install these descendants on later updates. Exercise the exact object
            # graph without running an entire conveyor scenario or replacing the existing stationary gait.
            caught.after, after.after = after, errand
            family.check()
            assert id(after) in family.owned and id(errand) in family.owned and id(gait) in family.owned
            assert family.expected["picker"] is gait
            gait.command(0.2, 0.1)  # the adopted errand can now use the same gait for its carry
            family.cancel()
            assert pg.session.controllers["picker"] is gait
            assert gait.status()["speed"] == gait.status()["turn"] == 0.0
            assert "picker/arm" not in pg.session.controllers
            assert errand.done and caught.then is None
            assert after.note and isinstance(after.start(), str)
            assert not callbacks


@pytest.mark.anyio
async def test_repeated_script_cancellation_reaps_child_and_waits_for_late_arm_planner(pg, tmp_path, monkeypatch):
    pg.add_robot("arm", template="arm")
    script = tmp_path / "blocked_arm.py"
    script.write_text(
        "from robotforge.script import connect\ns=connect()\ns.move_arm('arm',[0.2,0.15,0.2],gripper='open')\n",
        encoding="utf-8",
    )
    entered, release = threading.Event(), threading.Event()
    original_move = pg.behaviors.move_arm
    attached = []

    def blocked_move(*args, **kwargs):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release the arm planner")
        arm = original_move(*args, **kwargs)
        attached.append(arm)
        return arm

    monkeypatch.setattr(pg.behaviors, "move_arm", blocked_move)
    processes = []
    original_spawn = asyncio.create_subprocess_exec

    async def capture_spawn(*args, **kwargs):
        process = await original_spawn(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_spawn)
    runner = ScriptRunner(pg, lambda: False)
    task = asyncio.create_task(runner.run(script, max_sim_seconds=5, max_wall_seconds=10))
    try:
        await until(entered.is_set)
        lease = pg.session.execution.active
        assert lease is not None and pg.session.clock_owner == "script"
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert pg.session.execution.active is lease and pg.session.clock_owner == "script"
            assert not attached  # the real planner will still attach its controller after release
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    assert len(attached) == 1
    assert len(processes) == 1 and processes[0].returncode is not None
    assert await asyncio.wait_for(processes[0].wait(), 1) != 0
    assert "arm" not in pg.session.controllers
    assert pg.session.execution.active is None and pg.session.clock_owner is None
    assert runner._execution is None and runner._arms is None
    assert pg.session.execution.last_result.reason == EndReason.CANCELLED
