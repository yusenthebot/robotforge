"""Sensing II: two robot vacuums on the same lidar rule, each with its own stop distance, speed, noise and filter."""

import warnings

import mujoco
import pytest

from robotforge.learn import autopilot, sensors
from robotforge.learn.engine import LessonError
from robotforge.playground import Playground


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground.with_demo()


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _clean(pg: Playground) -> dict:
    pg.lessons.handle({"action": "press", "control": "run"})
    _play(pg, autopilot.RUN_SECONDS + 0.5)
    return _play(pg, 0.1)


def test_two_arenas_stack_front_to_back_and_settings_wait_their_turn(pg):
    described = pg.lessons.start("lidar_autopilot")
    rig = described["rig"]
    assert rig["kind"] == "cleaner"
    (ax, ay), (bx, by) = rig["arenas"]["a"], rig["arenas"]["b"]
    assert ax == bx and by - ay == pytest.approx(2 * autopilot.OFFSET)  # A in front, B behind
    with pytest.raises(LessonError, match="later step"):
        pg.lessons.handle({"action": "set", "control": "stop_a", "value": 3})


def test_a_stop_distance_inside_the_body_pins_the_rover_to_the_wall(pg):
    pg.lessons.start("lidar_autopilot")
    _play(pg, 0.5)
    update = _clean(pg)
    a, b = update["results"]["clean"]["a"], update["results"]["clean"]["b"]
    assert (a["stop"], b["stop"]) == (0.15, 0.3)
    assert a["pressed"] > 5.0 and b["pressed"] < 1.0  # 0.15 m from the lidar leaves 3 cm past the nose
    assert b["covered"] > a["covered"] and a["distance"] < b["distance"]
    assert "stop" in update["passed"]
    painted = update["live"]["cells"]
    assert painted["a"] and painted["b"] and all(0 <= i < autopilot.CELLS for c in painted["b"] for i in c)


def test_each_run_starts_both_rovers_from_scratch(pg):
    pg.lessons.start("lidar_autopilot")
    _play(pg, 0.5)
    first = _clean(pg)["results"]["clean"]
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "set", "control": "stop_a", "value": 3})
    update = _clean(pg)
    again = update["results"]["clean"]
    assert again["a"]["stop"] == 0.9 and again["a"]["bumps"] == 0  # far too careful to touch anything
    assert again["b"] == first["b"]  # B's noise-free run repeats exactly: the start and odometry were reset
    assert "timid" in update["passed"]


def test_both_arenas_run_on_the_sensors_lessons_integrator():
    """Attaching keeps the parent's physics options: a parent left at MuJoCo's defaults ran both rovers on the Euler
    integrator and logged "Attach conflict when attaching 'sensors_lesson' ... integrator: parent has 0 (default),
    child has 3". The arena's own options must win."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = autopilot.AutopilotLesson().build()
    assert not [w for w in caught if "Attach conflict" in str(w.message)]
    alone = sensors.SensorsLesson().build()
    assert model.opt.integrator == alone.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.timestep == alone.opt.timestep
