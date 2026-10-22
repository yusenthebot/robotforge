"""Loaded files and script lifetimes must not leak into the next operation."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from robotforge.playground import Playground, PlaygroundError
from robotforge.script_runner import WINDOWS_BOOTSTRAP, ScriptRunner
from robotforge.sim.execution import RunLimits
from robotforge.sim.session import ModelLoadError, SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "worlds"))
    return Playground(session=SimSession.from_xml("<mujoco/>"))


def test_mjcf_open_detaches_saved_world_without_editing_it(pg, tmp_path):
    pg.new_world("original")
    store = pg.store
    original = store.file.read_bytes()
    pg.last_doctor = {"old": {"note": "old import"}}
    model = tmp_path / "external.xml"
    model.write_text(
        '<mujoco><worldbody><body name="external"><geom size=".1"/></body></worldbody></mujoco>', encoding="utf-8"
    )

    pg.open_model(str(model))

    assert pg.store is None
    assert pg.current_world() is None
    assert pg.last_doctor == {}
    assert pg.session.source == str(model.resolve())
    with pytest.raises(PlaygroundError, match="no world is open"):
        pg.add_object("unexpected", shape="box", size=[0.1, 0.1, 0.1], pos=[0, 0])
    assert store.file.read_bytes() == original
    assert pg.session.model.body("external").id > 0


def test_failed_mjcf_open_keeps_saved_world_and_model(pg, tmp_path):
    pg.new_world("original")
    store, model, data = pg.store, pg.session.model, pg.session.data
    pg.last_doctor = {"original": {"note": "keep"}}
    invalid = tmp_path / "broken.xml"
    invalid.write_text("<mujoco><bad/></mujoco>", encoding="utf-8")

    with pytest.raises(ModelLoadError):
        pg.open_model(str(invalid))

    assert pg.store is store
    assert pg.session.model is model
    assert pg.session.data is data
    assert pg.last_doctor == {"original": {"note": "keep"}}
    pg.add_object("still_editable", shape="box", size=[0.1, 0.1, 0.1], pos=[0, 0])
    assert pg.current_world().objects[0].name == "still_editable"


def test_script_joints_include_robot_after_observation_limit():
    bodies = "".join(
        f'<body><joint name="{robot}/j{j}"/><geom size=".01"/></body>'
        for robot, count in (("first", 45), ("second", 3))
        for j in range(count)
    )
    pg = Playground(session=SimSession.from_xml(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"))
    runner = ScriptRunner(pg, watched=lambda: False)
    pg.session.data.qpos[:] = [j / 100 for j in range(48)]

    assert len(runner._op_joints({"robot": "first"})) == 45
    assert runner._op_joints({"robot": "second"}) == {"j0": 0.45, "j1": 0.46, "j2": 0.47}
    with pytest.raises(PlaygroundError, match="no joints found"):
        runner._op_joints({"robot": "missing"})


async def wait_for_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not path.exists():
            await asyncio.sleep(0.01)


@pytest.mark.anyio
async def test_cancel_script_reaps_real_child_and_closes_rpc(pg, tmp_path, monkeypatch):
    ready = tmp_path / "ready.json"
    script = tmp_path / "waiting.py"
    script.write_text(
        "import json, os, time\nfrom pathlib import Path\n"
        "from robotforge.script import connect\nsim = connect()\n"
        f"Path({str(ready)!r}).write_text(json.dumps({{'port': int(os.environ['ROBOTFORGE_SCRIPT_PORT'])}}), "
        "encoding='utf-8')\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    children = []
    spawn = asyncio.create_subprocess_exec

    async def capture_child(*args, **kwargs):
        proc = await spawn(*args, **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_child)
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
    try:
        await wait_for_file(ready)
        assert pg.session.clock_owner == "script"
        # Another runner must not overwrite the owner or release the first one's clock.
        with pytest.raises(PlaygroundError, match="already owned"):
            await ScriptRunner(pg, lambda: False).run(script)
        assert len(children) == 1
        assert pg.session.clock_owner == "script"
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)

    assert children[0].returncode is not None
    assert await children[0].wait() != 0
    assert pg.session.clock_owner is None
    port = json.loads(ready.read_text(encoding="utf-8"))["port"]
    with pytest.raises(OSError):
        await asyncio.open_connection("127.0.0.1", port)
    assert not any(
        task is not asyncio.current_task() and "ScriptRunner.run.<locals>" in task.get_coro().__qualname__
        for task in asyncio.all_tasks()
    )
    script.write_text("print('next run')\n", encoding="utf-8")
    result = await ScriptRunner(pg, lambda: False).run(script)
    assert result["ok"], result


@pytest.mark.anyio
async def test_cancel_during_child_spawn_still_reaps_it(pg, tmp_path, monkeypatch):
    script = tmp_path / "sleeping.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    spawned = asyncio.Event()
    release = asyncio.Event()
    children = []
    spawn = asyncio.create_subprocess_exec

    async def delayed_spawn(*args, **kwargs):
        proc = await spawn(*args, **kwargs)
        children.append(proc)
        spawned.set()
        await release.wait()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script))
    try:
        await asyncio.wait_for(spawned.wait(), 5)
        running.cancel()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(running, 5)
    assert children[0].returncode is not None
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_wall_timeout_reaps_real_child(pg, tmp_path):
    script = tmp_path / "sleeping.py"
    script.write_text("import time\nprint('started', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    result = await ScriptRunner(pg, lambda: False).run(script, max_wall_seconds=0.2)
    assert not result["ok"]
    assert result["exit_code"] != 0
    assert "wall time" in result["error"]
    assert "started" in result["output"]
    assert pg.session.clock_owner is None


@pytest.mark.anyio
async def test_windows_bootstrap_gates_script_and_preserves_arguments(tmp_path):
    ready = tmp_path / "started"
    script = tmp_path / "gated.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        f"Path({str(ready)!r}).write_text('started', encoding='utf-8')\n"
        "print(json.dumps({'args': sys.argv[1:], 'stdin': sys.stdin.read()}))\n",
        encoding="utf-8",
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        WINDOWS_BOOTSTRAP,
        str(script),
        "hello",
        "two words",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.sleep(0.05)
        assert not ready.exists()
        output, _ = await asyncio.wait_for(proc.communicate(b"1"), 2)
        assert proc.returncode == 0
        assert json.loads(output) == {"args": ["hello", "two words"], "stdin": ""}
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


@pytest.mark.anyio
@pytest.mark.parametrize("ending", ["timeout", "cancel", "exit"])
async def test_script_cleanup_kills_descendants_inheriting_stdout(pg, tmp_path, ending):
    ready = tmp_path / "descendant.pid"
    survived = tmp_path / "descendant_survived"
    script = tmp_path / "parent.py"
    descendant = (
        "import time\nfrom pathlib import Path\ntime.sleep(.8)\n"
        f"Path({str(survived)!r}).write_text('still running', encoding='utf-8')\n"
        "time.sleep(2)\n"
    )
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}])\n"
        f"Path({str(ready)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        + ("" if ending == "exit" else "time.sleep(60)\n"),
        encoding="utf-8",
    )
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script, max_wall_seconds=0.3))
    try:
        await wait_for_file(ready)
        started = time.monotonic()
        if ending == "cancel":
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 2)
        else:
            result = await asyncio.wait_for(running, 2)
            assert result["ok"] == (ending == "exit"), result
        assert time.monotonic() - started < 1.5
        assert pg.session.clock_owner is None
        await asyncio.sleep(0.9)
        assert not survived.exists(), "descendant survived cleanup"
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
        if os.name != "nt" and ready.exists():
            # Keep the regression itself tidy even when run against a broken implementation.
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(ready.read_text(encoding="utf-8")), signal.SIGKILL)


@pytest.mark.anyio
@pytest.mark.skipif(os.name == "nt", reason="POSIX sessions can deliberately escape the runner's process group")
async def test_detached_descendant_pipe_drain_is_bounded(pg, tmp_path):
    ready = tmp_path / "detached.pid"
    script = tmp_path / "detached.py"
    script.write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        f"Path({str(ready)!r}).write_text(str(child.pid), encoding='utf-8')\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    running = asyncio.create_task(ScriptRunner(pg, lambda: False).run(script, max_wall_seconds=0.3))
    try:
        await wait_for_file(ready)
        result = await asyncio.wait_for(running, 2)
        assert not result["ok"]
        assert "wall time" in result["error"]
        assert pg.session.clock_owner is None
    finally:
        if ready.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(ready.read_text(encoding="utf-8")), signal.SIGKILL)
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)


@pytest.mark.anyio
async def test_headless_advance_yields_for_cancellation(pg, monkeypatch):
    runner = ScriptRunner(pg, lambda: False)
    steps = 0
    real_step = pg.session.step

    def slow_step(count, **kwargs):
        nonlocal steps
        steps += 1
        time.sleep(0.002)
        real_step(count, **kwargs)

    monkeypatch.setattr(pg.session, "step", slow_step)
    with pg.session.execution.run("script", RunLimits(sim_seconds=60)) as run:
        runner._execution = run
        running = asyncio.create_task(runner._advance(lambda: False, 60))
        await asyncio.sleep(0.03)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert 0 < steps < 100
    assert pg.session.data.time < 2


@pytest.mark.anyio
async def test_headless_advance_checks_wall_deadline(pg):
    runner = ScriptRunner(pg, lambda: False)
    with pg.session.execution.run("script", RunLimits(sim_seconds=60, wall_seconds=0.001)) as run:
        runner._execution = run
        await asyncio.sleep(0.002)
        with pytest.raises(PlaygroundError, match="wall time limit"):
            await runner._advance(lambda: False, 60)
    assert pg.session.data.time == 0
