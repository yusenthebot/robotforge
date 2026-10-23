"""Controller identity and invalidation within one owned simulation step."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from robotforge.sim.execution import EndReason, ExecutionError, RunLimits
from robotforge.sim.session import SimSession


@pytest.fixture
def session():
    return SimSession.from_xml('<mujoco><option timestep=".002"/></mujoco>')


def test_invalidation_keeps_identity_for_cleanup_without_changing_generation(session):
    generation = session.generation
    with session.execution.run("arm", RunLimits()) as run:
        run.invalidate("the arm controller was replaced")
        assert run.current
        assert session.generation == generation
        assert run.reason == EndReason.INVALIDATED
        with pytest.raises(ExecutionError, match="arm controller was replaced") as exc:
            run.check_valid()
        assert exc.value.reason == EndReason.INVALIDATED
        assert run.current
    assert session.execution.last_result.reason == EndReason.INVALIDATED
    assert session.execution.active is None
    assert session.clock_owner is None


def test_explicit_stop_keeps_cancellation_reason_when_its_cleanup_invalidates_controller(session):
    with session.execution.run("arm", RunLimits()) as run:
        run.interrupt("stopped")
        with pytest.raises(ExecutionError) as cancelled:
            run.check_valid()
        assert cancelled.value.reason == EndReason.CANCELLED
        run.invalidate("replaced")
        with pytest.raises(ExecutionError, match="stopped") as invalidated:
            run.check_valid()
        assert invalidated.value.reason == EndReason.CANCELLED


@pytest.mark.parametrize("fails", [False, True])
def test_actor_is_visible_only_on_update_thread_and_cleared_afterwards(session, fails):
    assert session.updating_controller is None
    with ThreadPoolExecutor(max_workers=1) as pool:

        def update(model, data):
            assert session.updating_controller is controller
            assert session.in_controller
            assert pool.submit(lambda: session.updating_controller).result(timeout=1) is None
            assert not pool.submit(lambda: session.in_controller).result(timeout=1)
            if fails:
                raise ValueError("controller failed")

        controller = SimpleNamespace(rate_hz=500, update=update)
        session.set_controller("arm", controller)
        if fails:
            with pytest.raises(ValueError, match="controller failed"):
                session.step()
        else:
            session.step()
    assert session.updating_controller is None
    assert not session.in_controller
    assert session._updating_controller is None


def test_hook_invalidation_blocks_later_controller_from_old_snapshot(session):
    called = []
    replacement = SimpleNamespace(rate_hz=500, update=lambda m, d: called.append("replacement"))
    stale = SimpleNamespace(rate_hz=500, update=lambda m, d: called.append("stale"))

    def update(model, data):
        called.append("first")
        session.set_controller("arm", replacement)

    first = SimpleNamespace(rate_hz=500, update=update)
    session.set_controller("first", first)
    session.set_controller("arm", stale)
    with session.execution.run("arm", RunLimits()) as run:

        def replaced(name, old, new, reason):
            assert session.updating_controller is first
            assert (name, old, new, reason) == ("arm", stale, replacement, "replaced")
            run.invalidate("another controller replaced the arm")

        session.controller_hooks.append(replaced)
        with pytest.raises(ExecutionError, match="another controller replaced") as exc:
            session.step(owner=run)
        assert exc.value.reason == EndReason.INVALIDATED
        assert run.current
        assert called == ["first"]
        assert session.steps_taken == 0
        assert session.data.time == 0
        assert session.controllers["arm"] is replacement
        assert session.updating_controller is None
        assert not session.in_controller


@pytest.mark.parametrize("owned", [False, True])
def test_controller_rates_and_order_stay_the_same(session, owned):
    called = []

    def controller(name, rate):
        def update(model, data):
            assert session.updating_controller is item
            called.append((name, round(float(data.time), 6)))

        item = SimpleNamespace(rate_hz=rate, update=update)
        return item

    session.set_controller("fast", controller("fast", 500))
    session.set_controller("slow", controller("slow", 250))
    if owned:
        with session.execution.run("arm", RunLimits()) as run:
            session.step(5, owner=run)
    else:
        session.step(5)
    assert called == [
        ("fast", 0.0),
        ("slow", 0.0),
        ("fast", 0.002),
        ("fast", 0.004),
        ("slow", 0.004),
        ("fast", 0.006),
        ("fast", 0.008),
        ("slow", 0.008),
    ]
    assert session.steps_taken == 5
    assert session.data.time == pytest.approx(0.01)


def test_unowned_step_retains_snapshot_order_when_a_controller_is_replaced(session):
    called = []
    replacement = SimpleNamespace(rate_hz=500, update=lambda m, d: called.append("replacement"))

    def first(model, data):
        called.append("first")
        session.set_controller("arm", replacement)

    session.set_controller("first", SimpleNamespace(rate_hz=500, update=first))
    session.set_controller("arm", SimpleNamespace(rate_hz=500, update=lambda m, d: called.append("old")))
    session.step(2)
    assert called == ["first", "old", "first", "replacement"]
