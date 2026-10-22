"""One execution contract for live and headless clocks, including invalidation and cumulative limits."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from robotforge.sim.execution import EndReason, ExecutionError, RunLimits
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub

XML = """<mujoco><option timestep=".002"/><worldbody><body>
<joint name="slide" type="slide" axis="1 0 0"/><geom size=".1"/>
</body></worldbody><actuator><motor joint="slide"/></actuator></mujoco>"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session():
    return SimSession.from_xml(XML)


@pytest.mark.parametrize(
    "limits", [{"sim_seconds": 0}, {"wall_seconds": float("nan")}, {"steps": True}, {"steps": 1.5}]
)
def test_bad_limits_are_refused(limits):
    with pytest.raises(ValueError):
        RunLimits(**limits)


def test_claim_is_exclusive_and_stale_cleanup_cannot_release_a_new_claim(session):
    old = session.execution.run("move", RunLimits())
    with old:
        with pytest.raises(ExecutionError, match="already owned"), session.execution.run("script", RunLimits()):
            pytest.fail("second caller acquired the clock")
        with pytest.raises(ExecutionError, match="already owned"):
            session.step(1)
        assert session.advance(1) == 0
        assert session.clock_owner == "move"
    with session.execution.run("script", RunLimits()) as new:
        old.__exit__(None, None, None)
        assert session.execution.active is new
        assert session.clock_owner == "script"
    session.step(1)  # ordinary stepping is still available after release
    assert session.steps_taken == 1


@pytest.mark.anyio
async def test_watching_only_changes_pacing_not_physics_or_step_count():
    states = []
    for mode in ("headless", "watched", "changing"):
        session = SimSession.from_xml(XML)
        session.speed = 4
        session.data.ctrl[:] = 1
        calls = 0

        def watched(mode=mode):
            nonlocal calls
            calls += 1
            return mode == "watched" or mode == "changing" and calls % 2 == 0

        # No viewer task advances this session: a watched run must still finish.
        with session.execution.run("move", RunLimits(sim_seconds=0.2, wall_seconds=1), watched) as run:
            result = await asyncio.wait_for(run.advance(lambda: False, 0.2), 1)
            assert result.steps == 100
            assert result.reason == EndReason.DURATION
            assert result.shown_live == (mode != "headless")
            states.append(session.data.qpos.copy())
    for state in states[1:]:
        np.testing.assert_array_equal(state, states[0])


@pytest.mark.anyio
async def test_limits_accumulate_between_phases_and_never_add_a_fractional_step(session):
    with session.execution.run("script", RunLimits(sim_seconds=0.035)) as run:
        await run.advance(lambda: False, 0.03)
        last = await run.advance(lambda: False, 0.1)
        assert last.reason == EndReason.SIM_LIMIT
        assert run.steps == 17
        assert run.sim_seconds == pytest.approx(0.034)
        again = await run.advance(lambda: False, 0.1)
        assert again.steps == 0 and again.reason == EndReason.SIM_LIMIT
    with session.execution.run("move", RunLimits(steps=7)) as run:
        await run.advance(lambda: False, 0.01)
        last = await run.advance(lambda: False, 0.1)
        assert last.reason == EndReason.STEP_LIMIT
        assert run.steps == 7


@pytest.mark.anyio
async def test_wall_budget_is_shared_with_work_outside_physics(session):
    with session.execution.run("script", RunLimits(wall_seconds=0.01)) as run:
        await asyncio.sleep(0.02)
        result = await run.advance(lambda: False, 1)
        assert result.reason == EndReason.WALL_LIMIT
        assert result.steps == 0
        with pytest.raises(ExecutionError, match="wall time"):
            await run.wait(lambda: False)


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["reset", "swap"])
async def test_invalidation_is_checked_before_a_stale_completion_predicate(session, change):
    with session.execution.run("move", RunLimits()) as run:
        await run.advance(lambda: False, 0.02)
        version = session.version
        if change == "reset":
            session.reset()
            assert session.version == version
        else:
            session.load_model(SimSession.from_xml("<mujoco/>").model, "replacement")

        def stale():
            pytest.fail("old predicate accessed the replacement world")

        with pytest.raises(ExecutionError, match="reset or changed"):
            await run.advance(stale, 1)
        assert run.sim_seconds == pytest.approx(0.02)  # unaffected by reset's clock rewind
        assert not run.current
        assert session.clock_owner == "move"  # retained until the owner unwinds
    assert session.clock_owner is None


@pytest.mark.anyio
async def test_reset_during_a_step_stops_the_batch_before_stepping_the_new_world(session):
    session.step_hooks.append(session.reset)
    with session.execution.run("move", RunLimits()) as run:
        with pytest.raises(ExecutionError):
            await run.advance(lambda: False, 1)
        assert run.steps == 1
        assert run.sim_seconds == pytest.approx(0.002)
        assert session.data.time == 0


@pytest.mark.anyio
async def test_cancellation_yields_and_releases_the_claim(session, monkeypatch):
    step = session.step

    def slow_step(n, **kwargs):
        time.sleep(0.002)
        step(n, **kwargs)

    monkeypatch.setattr(session, "step", slow_step)

    async def execute():
        with session.execution.run("move", RunLimits()) as run:
            await run.advance(lambda: False, 60)

    task = asyncio.create_task(execute())
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.clock_owner is None
    assert session.execution.last_result.reason == EndReason.CANCELLED
    assert session.steps_taken < 1000


@pytest.mark.anyio
async def test_clock_rewind_is_not_reported_as_completed_duration(session, monkeypatch):
    session.data.time = 1
    step = session.step

    def rewind(n, **kwargs):
        step(n, **kwargs)
        session.data.time = 0

    monkeypatch.setattr(session, "step", rewind)
    with session.execution.run("move", RunLimits()) as run, pytest.raises(ExecutionError, match="clock rewound"):
        await run.advance(lambda: False, 1)


def test_viewer_single_step_is_refused_without_pausing_an_active_run(session):
    hub = ViewerHub(session)
    with session.execution.run("script", RunLimits()):
        with pytest.raises(ExecutionError, match="already owned"):
            hub.handle({"type": "cmd", "cmd": "step"})
        assert not session.paused
        assert session.steps_taken == 0


@pytest.mark.anyio
async def test_wall_deadline_is_checked_between_slow_controller_updates(session):
    session.set_controller("slow", SimpleNamespace(rate_hz=500, update=lambda m, d: time.sleep(0.02)))
    with session.execution.run("move", RunLimits(wall_seconds=0.01)) as run:
        result = await run.advance(lambda: False, 1)
        assert result.reason == EndReason.WALL_LIMIT
        assert result.steps <= 1
        assert result.wall_seconds < 0.1  # one unavoidable update, not ten updates in a batch


@pytest.mark.anyio
async def test_legacy_command_cancellation_retains_admission_until_real_work_finishes(session):
    entered, release = threading.Event(), threading.Event()

    def worker():
        entered.set()
        assert release.wait(2)

    async def invoke():
        await asyncio.to_thread(worker)

    task = asyncio.create_task(session.execution.call_command("old planner", invoke))
    try:
        async with asyncio.timeout(1):
            while not entered.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ExecutionError, match="busy with old planner"), session.execution.run("move", RunLimits()):
            pytest.fail("new execution acquired a still-running planner")
    finally:
        release.set()
        async with asyncio.timeout(2):
            while session.execution._pending:
                await asyncio.sleep(0.001)
    with session.execution.run("move", RunLimits()):
        pass


@pytest.mark.anyio
async def test_cancelled_script_worker_cannot_start_late_after_invalidation(session):
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)
    release = threading.Event()
    blocker = asyncio.create_task(asyncio.to_thread(release.wait, 2))
    changed = []
    try:
        with session.execution.run("script", RunLimits()) as run:
            worker = asyncio.create_task(run.call_thread(lambda: changed.append(True)))
            await asyncio.sleep(0.01)
            worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await worker
            run.interrupt()
            release.set()
            await run.drain_workers()
            assert not changed
    finally:
        release.set()
        await blocker
        pool.shutdown(wait=True)
