"""Bounded held driving owns only its latest command, not a robot indefinitely."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("held")
    pg.add_robot("rover", template="rover")
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    owner = object()
    hub.clients.add(owner)
    return pg, hub, owner


def hold(hub, owner, **values):
    hub.handle(
        {
            "type": "teleop",
            "robot": "rover",
            "speed": 0.2,
            "hold_ms": 750,
            "version": hub.session.version,
            "generation": hub.session.generation,
            **values,
        },
        owner,
    )


def speed(pg, robot="rover"):
    return pg.session.controllers[robot].speed


def test_expiry_stops_without_physics_or_a_connected_viewer(setup):
    pg, hub, owner = setup
    hold(hub, owner)
    pg.session.paused = True
    before = pg.session.data.time
    hub.clients.clear()
    hub.drive_holds.expire(math.inf)
    assert speed(pg) == 0
    assert pg.session.data.time == before
    assert not hub.drive_holds._holds


def test_disconnect_releases_only_that_owners_latest_hold(setup):
    pg, hub, first = setup
    second = object()
    hub.clients.add(second)
    hold(hub, first)
    hold(hub, second, speed=0.1)
    hub.forget(first)
    assert speed(pg) == pytest.approx(0.1)
    hub.forget(second)
    assert speed(pg) == 0


@pytest.mark.parametrize("replacement", ["move", "stop", "reset", "controller", "navigate"])
def test_stale_hold_cannot_stop_a_new_command_or_scene(setup, replacement):
    pg, hub, owner = setup
    hold(hub, owner)
    if replacement == "move":
        pg.behaviors.move("rover", 0.1, 0)
    elif replacement == "stop":
        pg.behaviors.stop("rover")
    elif replacement == "reset":
        pg.session.reset()
        pg.behaviors.move("rover", 0.1, 0)
    elif replacement == "controller":
        pg.session.set_controller("rover", None)
        pg.behaviors.move("rover", 0.1, 0)
    else:
        trip = pg.behaviors.navigate("rover", [2, 0])
        assert not trip.done
    controller = pg.session.controllers["rover"]
    state = controller.status()
    hub.forget(owner)
    hub.drive_holds.expire(math.inf)
    assert pg.session.controllers["rover"] is controller
    assert controller.status() == state
    if replacement == "navigate":
        assert not trip.done


def test_old_protocol_keeps_its_unbounded_command(setup):
    pg, hub, owner = setup
    hold(hub, owner)
    hub.handle({"type": "teleop", "robot": "rover", "speed": 0.1})
    hub.forget(owner)
    hub.drive_holds.expire(math.inf)
    assert speed(pg) == pytest.approx(0.1)


@pytest.mark.parametrize("duration", [True, False, None, 0, -1, 249, 1501, 750.0, "750", float("nan")])
def test_invalid_hold_is_refused_before_movement(setup, duration):
    pg, hub, owner = setup
    before = pg.session.controllers.get("rover")
    with pytest.raises(ValueError, match="hold_ms"):
        hold(hub, owner, hold_ms=duration)
    assert pg.session.controllers.get("rover") is before
    assert not hub.drive_holds._holds


@pytest.mark.parametrize("duration", [250, 1500])
def test_hold_accepts_bounded_integer_deadlines(setup, duration):
    pg, hub, owner = setup
    hold(hub, owner, hold_ms=duration)
    assert speed(pg) > 0
    hub.drive_holds.expire(math.inf)
    assert speed(pg) == 0


@pytest.mark.parametrize("axis", ["speed", "turn", "side", "climb", "heave"])
def test_nonfinite_axes_do_not_partially_replace_a_held_command(setup, axis):
    pg, hub, owner = setup
    hold(hub, owner)
    before = pg.session.controllers["rover"].status()
    with pytest.raises(ValueError, match="finite"):
        hold(hub, owner, **{axis: float("nan")})
    assert pg.session.controllers["rover"].status() == before
    assert hub.drive_holds._holds


def test_unconnected_or_nonteleop_holds_are_refused(setup):
    pg, hub, owner = setup
    for other in (None, object()):
        with pytest.raises(ValueError, match="connected viewer"):
            hold(hub, other)
    with pytest.raises(ValueError, match="teleop"):
        hub.handle({"type": "cmd", "cmd": "reset", "hold_ms": 750}, owner)
    assert not hub.drive_holds._holds


@pytest.mark.parametrize("template", ["quadruped", "boat", "rov", "drone"])
def test_expiry_stops_each_mobile_profile_and_all_vertical_axes(setup, template):
    pg, hub, owner = setup
    if template in {"boat", "rov"}:
        pg.set_floor("sand", water=1.0)
    pg.add_robot("other", template=template, pos=[2, 0])
    extra = {"heave": 0.2} if template == "rov" else {"side": 0.1, "climb": 0.2} if template == "drone" else {}
    hold(hub, owner, robot="other", **extra)
    pilot = pg.session.controllers["other"]
    hub.forget(owner)
    status = pilot.status()
    assert status.get("speed", status.get("command_speed", 0)) == 0
    assert status.get("turn", 0) == 0
    if template == "rov":
        assert pilot.heave == 0
    if template == "drone":
        assert pilot.side == 0
        assert pilot.climb == 0


def test_new_move_and_its_ownership_hook_are_one_atomic_boundary(setup):
    pg, hub, owner = setup
    hold(hub, owner)
    entered, finish = threading.Event(), threading.Event()

    def between_command_and_release(robot, event):
        if event.get("op") == "move" and event.get("speed") == 0.1:
            entered.set()
            assert finish.wait(5)

    # Deliberately block before the registry sees the new command.
    pg.behaviors.command_hooks.insert(0, between_command_and_release)
    with ThreadPoolExecutor(max_workers=1) as pool:
        new_move = pool.submit(pg.behaviors.move, "rover", 0.1, 0)
        try:
            assert entered.wait(5)
            acquired = pg.session.lock.acquire(blocking=False)
            if acquired:
                pg.session.lock.release()
            assert not acquired, "expiry could stop the new move before its ownership hook"
        finally:
            finish.set()
        new_move.result(5)
    hub.drive_holds.expire(math.inf)
    assert speed(pg) == pytest.approx(0.1)


def test_late_navigation_notification_cannot_revoke_a_newer_hold(setup):
    pg, hub, first = setup
    second = object()
    hub.clients.add(second)
    hold(hub, first)
    entered, finish = threading.Event(), threading.Event()

    def delayed_go(robot, event):
        if event.get("op") == "go":
            entered.set()
            assert finish.wait(5)

    pg.behaviors.command_hooks.insert(0, delayed_go)
    with ThreadPoolExecutor(max_workers=1) as pool:
        navigation = pool.submit(pg.behaviors.navigate, "rover", [2, 0])
        try:
            assert entered.wait(5)
            hold(hub, second, speed=0.1)
        finally:
            finish.set()
        navigation.result(5)
    assert hub.drive_holds._holds["rover"].owner is second
    hub.forget(second)
    assert speed(pg) == 0


def test_nullable_optional_axes_keep_the_legacy_meaning(setup):
    pg, hub, _ = setup
    hub.handle({"type": "teleop", "robot": "rover", "speed": 0.1, "side": None, "climb": None, "heave": None})
    assert speed(pg) == pytest.approx(0.1)


@pytest.mark.parametrize("key", ["version", "generation"])
@pytest.mark.parametrize("value", [None, True, -1, "current"])
def test_stale_context_cannot_move_or_stop_the_current_controller(setup, key, value):
    pg, hub, owner = setup
    pg.behaviors.move("rover", 0.1, 0)
    before = pg.session.controllers["rover"].status()
    with pytest.raises(ValueError, match="scene changed"):
        hold(hub, owner, speed=0, **{key: value})
    assert pg.session.controllers["rover"].status() == before
    assert not hub.drive_holds._holds


def test_viewer_observers_preserve_multi_waypoint_flight_commands(setup):
    import json

    pg, _, _ = setup
    pg.add_robot("drone", template="drone", pos=[2, 0])
    events = []
    pg.behaviors.command_hooks.append(lambda robot, event: events.append((robot, event)))
    path = [[2, 0, 1], [3, 0, 1], [2, 0, 0]]
    trip = pg.behaviors.navigate("drone", path)
    assert not trip.done
    assert events[-1][0] == "drone" and events[-1][1]["op"] == "go"
    assert json.loads(events[-1][1]["to"]) == path
