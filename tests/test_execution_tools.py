"""move and real child scripts share a session lease without changing their public tool shapes."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

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
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("execution")
    pg.add_robot("rover", template="rover")
    return pg


async def until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.005)


def slow_steps(pg, monkeypatch):
    step = pg.session.step

    def slower(n, **kwargs):
        time.sleep(0.002)
        step(n, **kwargs)

    monkeypatch.setattr(pg.session, "step", slower)


def decoded(result):
    return json.loads("".join(block.text for block in result.content if getattr(block, "type", "") == "text"))


@pytest.mark.anyio
async def test_move_lease_rejects_another_move_or_script_before_controller_mutation(pg, tmp_path, monkeypatch):
    slow_steps(pg, monkeypatch)
    server = build_server(pg, 0)
    script = tmp_path / "must_not_run.py"
    marker = tmp_path / "unexpected"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8")
    moving = asyncio.create_task(server.call_tool("move", {"robot": "rover", "speed": 0.3, "seconds": 60}))
    try:
        await until(lambda: pg.session.clock_owner == "move")
        controller = pg.session.controllers["rover"]
        with pytest.raises(ToolError, match="already owned"):
            await server.call_tool("move", {"robot": "rover", "speed": -0.2, "seconds": None})
        with pytest.raises(ValueError, match="already owned"):
            await ScriptRunner(pg, lambda: False).run(script)
        assert pg.session.controllers["rover"] is controller
        assert controller.status()["speed"] > 0
        assert not marker.exists()
    finally:
        moving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await moving
    assert pg.session.clock_owner is None
    assert pg.session.controllers["rover"].status()["speed"] == 0
    assert pg.session.execution.last_result.reason == EndReason.CANCELLED


@pytest.mark.anyio
async def test_script_lease_rejects_move_and_simulate_without_changing_controls(pg, tmp_path):
    ready = tmp_path / "ready"
    script = tmp_path / "waiting.py"
    script.write_text(
        "import time\nfrom pathlib import Path\nfrom robotforge.script import connect\n"
        "sim = connect()\nsim.move('rover', .3)\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
    server = build_server(pg, 0)
    try:
        await until(ready.exists)
        controller = pg.session.controllers["rover"]
        for tool, args in (
            ("move", {"robot": "rover", "speed": -0.3, "seconds": None}),
            ("simulate", {"seconds": 1, "controls": {"rover/left_wheel": 9}}),
        ):
            with pytest.raises(ToolError, match="already owned"):
                await server.call_tool(tool, args)
        assert pg.session.controllers["rover"] is controller
        assert controller.status()["speed"] == 0.3
        assert pg.session.data.time == 0
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert pg.session.clock_owner is None
    assert pg.session.controllers["rover"].status()["speed"] == 0


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["move", "script"])
async def test_reset_invalidates_old_execution_and_cleanup_preserves_the_new_controller(
    pg, tmp_path, monkeypatch, operation
):
    ready = tmp_path / "ready"
    if operation == "move":
        slow_steps(pg, monkeypatch)
        server = build_server(pg, 0)
        running = asyncio.create_task(server.call_tool("move", {"robot": "rover", "seconds": 60}))
        await until(lambda: pg.session.clock_owner == "move")
    else:
        script = tmp_path / "waiting.py"
        script.write_text(
            "import time\nfrom pathlib import Path\nfrom robotforge.script import connect\n"
            "sim = connect()\nsim.move('rover', .3)\nsim.step(.1)\n"
            f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n",
            encoding="utf-8",
        )
        running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
        await until(ready.exists)
    try:
        pg.session.reset()
        pg.behaviors.move("rover", 0.7, 0)
        replacement = pg.session.controllers["rover"]
        if operation == "move":
            with pytest.raises(ToolError, match="reset or changed"):
                await asyncio.wait_for(running, 3)
        else:
            result = await asyncio.wait_for(running, 3)
            assert not result["ok"] and "reset or changed" in result["error"]
            assert result["sim_seconds"] == pytest.approx(0.1)
        assert pg.session.controllers["rover"] is replacement
        assert replacement.status()["speed"] == 0.7
        assert pg.session.clock_owner is None
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.anyio
async def test_continuous_move_releases_its_lease_and_timed_move_keeps_legacy_result_fields(pg):
    server = build_server(pg, 0)
    started = decoded(await server.call_tool("move", {"robot": "rover", "seconds": None}))
    assert set(started) == {"started", "pose"}
    assert pg.session.clock_owner is None
    pg.simulate(0.1)
    out = decoded(await server.call_tool("move", {"robot": "rover", "speed": 0.3, "seconds": 1}))
    assert {"from", "to", "distance_m", "turned_deg", "stopped", "sim_seconds", "shown_live", "behavior"} <= out.keys()
    assert out["stopped"] and out["distance_m"] > 0.2
    assert out["sim_seconds"] == pytest.approx(1)


@pytest.mark.anyio
async def test_competing_gripper_is_refused_before_installing_a_controller(pg):
    pg.add_robot("arm", template="arm", pos=[1, 0])
    server = build_server(pg, 0)
    with pg.session.execution.run("script", RunLimits()):
        before = dict(pg.session.controllers)
        with pytest.raises(ToolError, match="already owned"):
            await server.call_tool("gripper", {"robot": "arm", "state": "close"})
        assert pg.session.controllers == before


@pytest.mark.anyio
async def test_stop_only_interrupts_a_valid_target_owned_by_the_run(pg):
    pg.add_robot("other", template="rover", pos=[1, 0])
    server = build_server(pg, 0)
    with pg.session.execution.run("move", RunLimits()) as run:
        run.resources.add("robot:rover")
        with pytest.raises(ToolError, match="no robot"):
            await server.call_tool("stop", {"robot": "missing"})
        run.check()
        await server.call_tool("stop", {"robot": "other"})
        run.check()
        await server.call_tool("stop", {"robot": "rover"})
        with pytest.raises(ValueError, match="stopped"):
            run.check()


@pytest.mark.anyio
async def test_cancelled_script_stops_its_mounted_arm_controller(pg, tmp_path):
    pg.update_robot("rover", mounts=[{"template": "arm"}])
    pg.behaviors.move("rover", 0.3, 0)  # belongs to an earlier caller, not this arm-only script
    base = pg.session.controllers["rover"]
    ready = tmp_path / "ready"
    script = tmp_path / "grip.py"
    script.write_text(
        "import time\nfrom pathlib import Path\nfrom robotforge.script import connect\n"
        "sim = connect()\nsim.gripper('rover/arm', 'close')\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
    try:
        await until(ready.exists)
        assert "rover/arm" in pg.session.controllers
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert "rover/arm" not in pg.session.controllers
    assert pg.session.controllers["rover"] is base
    assert base.status()["speed"] == 0.3
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_legacy_replay_cannot_mutate_state_under_an_owned_clock(pg):
    pg.demonstrations.start("rover", "demo")
    pg.simulate(0.1)
    pg.demonstrations.stop("demo")
    server = build_server(pg, 0)
    with pg.session.execution.run("script", RunLimits()):
        hooks, controllers = list(pg.session.step_hooks), dict(pg.session.controllers)
        with pytest.raises(ToolError, match="already owned"):
            await server.call_tool(
                "demonstration", {"action": "replay", "name": "demo", "mode": "joints", "wait": False}
            )
        assert pg.session.step_hooks == hooks
        assert pg.session.controllers == controllers


@pytest.mark.anyio
async def test_a_refused_script_command_does_not_claim_an_existing_controller(pg, tmp_path):
    pg.behaviors.move("rover", 0.3, 0)
    base = pg.session.controllers["rover"]
    ready = tmp_path / "ready"
    script = tmp_path / "bad_command.py"
    script.write_text(
        "import time\nfrom pathlib import Path\nfrom robotforge.script import connect\nsim = connect()\n"
        "try:\n    sim.move('rover', speed='bad')\nexcept Exception:\n    pass\n"
        f"Path({str(ready)!r}).touch()\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
    try:
        await until(ready.exists)
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert pg.session.controllers["rover"] is base
    assert base.status()["speed"] == 0.3
