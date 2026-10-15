"""Inverse kinematics on six-joint arms: our template arm, then the PiPER (offline stand-in)."""

from pathlib import Path

import pytest

from robotforge.learn import ik_arm
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("ik_arm")
    return playground


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _set(pg: Playground, **values: float) -> None:
    for control, value in values.items():
        pg.lessons.handle({"action": "set", "control": control, "value": value})


def test_the_hand_follows_a_target_in_3d(pg):
    update = _play(pg, 1.0)
    live = update["live"]
    assert live["arm"] == "ours" and len(live["ellipsoid"]["axes"]) == 3
    _set(pg, tx=0.35, ty=0.1, tz=0.2)
    update = _play(pg, 1.5)
    assert update["samples"][-1]["err"] < ik_arm.REACHED * 1000
    _set(pg, tx=0.2, ty=-0.15, tz=0.35)
    assert "drag" in _play(pg, 1.5)["passed"]


def test_stretching_flattens_the_ellipsoid_and_small_damping_flails(pg):
    pg.lessons.passed.add("drag")
    pg.lessons.handle({"action": "goto", "step": 1})
    _set(pg, tx=0.6, ty=0.0, tz=0.2)  # past our arm's reach
    update = _play(pg, 2.0)
    assert update["samples"][-1]["w"] < ik_arm.EDGE and "ellipsoid" in update["passed"]
    assert max(s["step"] for s in update["samples"]) < ik_arm.CALM  # lambda 0.05: stretched out, calm
    pg.lessons.handle({"action": "next"})
    _set(pg, tx=0.3)
    _play(pg, 1.5)
    _set(pg, damping=0, tx=0.85)  # lambda 0.0001
    update = _play(pg, 2.0)
    assert max(s["step"] for s in update["samples"]) > ik_arm.FLAIL
    assert "damping" in update["passed"]


def test_the_same_steps_work_on_the_piper(pg):
    pg.lessons.passed.update({"drag", "ellipsoid", "damping", "down"})
    pg.lessons.handle({"action": "goto", "step": 4})
    _set(pg, arm=1, tx=0.55, ty=0.0, tz=0.25)  # beyond our arm, inside the PiPER's reach
    update = _play(pg, 3.0)
    assert update["live"]["arm"] == "piper" and update["samples"][-1]["err"] < ik_arm.REACHED * 1000
    assert "piper" in update["passed"]
