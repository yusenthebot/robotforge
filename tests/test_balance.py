"""Humanoid balance: the capture point predicts a fall, crouching helps, the G1 steps (offline stand-in)."""

from pathlib import Path

import pytest

from robotforge.behaviors import humanoid
from robotforge.learn import balance
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("balance")
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


def _push(pg: Playground, strength: float) -> dict:
    _set(pg, strength=strength)
    pg.lessons.handle({"action": "press", "control": "push"})
    _play(pg, balance.PUSH_AT + balance.WATCH + 0.2)
    return _play(pg, 0.1)


def _next(pg: Playground, guess: int | None = None) -> None:
    pg.lessons.handle({"action": "next"})
    if guess is not None:
        pg.lessons.handle({"action": "answer", "choice": guess})


def test_the_capture_point_says_when_a_push_is_too_much(pg):
    update = _play(pg, 1.5)
    live = update["live"]
    assert len(live["support"]) >= 3 and live["com"] and live["capture"]
    rows = {r["show"]: r for r in live["formula"] if r["show"] == "f_edge"}
    assert rows  # the foot edge row exists for the steps that show it

    pg.lessons.handle({"action": "answer", "choice": 0})
    gentle = _push(pg, 0.1)
    assert gentle["results"]["push"]["stood"] and "stand" in gentle["passed"]

    _next(pg, guess=1)
    results = {}
    for strength in (0.3, 0.4):
        update = _push(pg, strength)
        results[strength] = update["results"]["push"]
    held, fell = results[0.3], results[0.4]
    assert held["stood"] and held["cp_cm"] < held["edge_cm"]  # inside the feet: the ankles bring it back
    assert not fell["stood"] and fell["cp_cm"] > fell["edge_cm"]  # past the toes: down it goes
    assert "edge" in update["passed"] and update["results"]["summary"]["en"].startswith("Capture point")

    _next(pg, guess=0)
    _set(pg, direction=1)
    update = _push(pg, 0.5)
    side = update["results"]["push"]
    assert side["stood"] and side["edge_cm"] > held["edge_cm"] and "side" in update["passed"]  # wider sideways

    _next(pg, guess=1)
    _set(pg, direction=0, crouch=0.3)
    update = _push(pg, 0.4)
    low = update["results"]["push"]
    assert low["stood"] and low["height_cm"] < fell["height_cm"] - 3 and low["cp_cm"] < fell["cp_cm"]
    assert "crouch" in update["passed"]


def test_the_g1_steps_under_a_push_that_would_floor_ours(pg):
    for step in ("stand", "edge", "side", "crouch"):
        pg.lessons.passed.add(step)
    pg.lessons.handle({"action": "goto", "step": 4})
    pg.lessons.handle({"action": "answer", "choice": 1})
    _set(pg, robot=1)
    model = pg.session.model
    feet = [model.body("g1/left_ankle_roll_link").id, model.body("g1/right_ankle_roll_link").id]
    before = pg.session.data.xpos[feet][:, 0].mean()
    update = _push(pg, 0.8)
    push = update["results"]["push"]
    assert push["robot"] == "g1" and push["stood"] and push["cp_cm"] > push["edge_cm"]  # past its toes, yet up
    assert pg.session.data.xpos[feet][:, 0].mean() > before + 0.1  # because it stepped forward
    assert "g1" in update["passed"]
    runs = pg.lessons.ctx.runs
    assert runs[-1]["params"]["robot"] == 1 and runs[-1]["x"][0] == pytest.approx(0, abs=0.02)


def test_a_crouch_sinks_the_hips_and_stays_there(pg):
    ctrl = humanoid.for_template(pg.session.model, "ours", pg.session.model.key("home").qpos)
    with pytest.raises(ValueError, match="between 0 and 0.35"):
        ctrl.crouch(0.6)
    ctrl.crouch(0.2)
    assert ctrl.status()["behavior"] == "pose"
