"""PID II: the small arm's gains on the PiPER's shoulder (offline stand-in with the real link masses)."""

from pathlib import Path

import pytest

from robotforge.learn import pd_piper
from robotforge.learn.engine import LessonError
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("pid_piper")
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


def _test(pg: Playground) -> dict:
    pg.lessons.handle({"action": "press", "control": "test"})
    _play(pg, 3.3)
    update = _play(pg, 0.1)
    return update


def _next(pg: Playground, guess: int | None = None) -> None:
    pg.lessons.handle({"action": "next"})
    if guess is not None:
        pg.lessons.handle({"action": "answer", "choice": guess})


def test_the_shoulder_is_a_torque_motor_under_real_gravity(pg):
    model = pg.session.model
    assert not model.body_gravcomp.any()  # the file's free gravity cancelling is off
    act = model.actuator(pd_piper.JOINT).id
    assert model.actuator_biasprm[act].tolist() == [0.0] * len(model.actuator_biasprm[act])
    update = _play(pg, 1.0)
    rows = {r["show"]: r for r in update["live"]["formula"]}
    zeta = float(rows["f_z"]["value"])
    assert 0.3 < zeta < 0.5  # Kp 40 and Kd 3 on a shoulder about five times heavier to turn
    assert float(rows["f_g"]["value"].split()[0]) > 4.0  # gravity needs several N*m
    others = [pg.session.model.joint(f"piper/joint{i}").qposadr[0] for i in (1, 3, 4)]
    home = model.key("home").qpos
    assert all(abs(pg.session.data.qpos[a] - home[a]) < 0.05 for a in others)  # held by their own servos


def test_the_lesson_from_sagging_to_a_model_free_challenge(pg):
    pg.lessons.handle({"action": "answer", "choice": 1})
    update = _test(pg)
    copied = update["results"]["test"]
    assert copied["end_err"] > 5 and "copy" in update["passed"]  # the small arm's gains sag on the PiPER

    _next(pg, guess=1)
    _set(pg, gravity=1)
    update = _test(pg)
    assert update["results"]["test"]["overshoot_pct"] > 10 and update["results"]["test"]["end_err"] < 0.5
    assert "ff" in update["passed"]

    _next(pg)
    _set(pg, kp=80, kd=10)  # zeta close to 1
    update = _test(pg)
    matched = update["results"]["test"]
    assert matched["overshoot_pct"] < 2 and matched["settle_s"] < 0.4 and "zeta" in update["passed"]

    _next(pg, guess=1)
    _set(pg, payload=1)
    loaded = _test(pg)["results"]["test"]
    assert loaded["end_err"] > 1.0  # the model thinks the box is empty
    _set(pg, ki=20)
    update = _test(pg)
    assert update["results"]["test"]["end_err"] < 0.5 and "payload" in update["passed"]

    _next(pg)
    easy = _test(pg)
    assert "challenge" not in easy["passed"]  # with the model it is too easy: the challenge is without
    assert easy["results"]["test"]["stars"] == 3 and easy["results"]["test"]["note"]["en"].startswith("This one")
    _set(pg, gravity=0, ki=60)
    update = _test(pg)
    assert update["results"]["test"]["stars"] == 3 and "challenge" in update["passed"]
    runs = pg.lessons.ctx.runs
    assert runs[-1]["params"]["payload"] == 1 and runs[-1]["params"]["gravity"] == 0

    _next(pg, guess=1)
    assert "quiz" in _play(pg, 0.1)["passed"]


def test_the_box_weighs_only_when_full_and_sweeps_leave_the_live_arm_alone(pg):
    lesson = pg.lessons.lesson
    model = pg.session.model
    with pytest.raises(LessonError, match="later step"):
        _set(pg, payload=1)
    for step in ("copy", "ff", "zeta"):
        pg.lessons.passed.add(step)
    pg.lessons.handle({"action": "goto", "step": 3})
    _set(pg, payload=1)
    assert model.body_mass[lesson.payload_body] == pd_piper.PAYLOAD
    _set(pg, payload=0)
    assert model.body_mass[lesson.payload_body] == pd_piper.EMPTY
    started = pg.session.data.time
    _set(pg, gravity=1)
    pg.lessons.handle({"action": "sweep", "control": "kd"})
    assert pg.session.data.time == started
    overshoot = [p["overshoot_pct"] for p in pg.lessons.ctx.sweep["points"]]
    assert overshoot[0] > 20 and overshoot[-1] < 1  # more Kd, less overshoot


def test_the_dial_lies_in_the_shoulders_plane(pg):
    rig = pg.lessons.describe_active()["rig"]
    assert rig["kind"] == "joint" and rig["body"] == pg.session.model.body("piper/link2").id
    axis, zero, swing = (pytest.importorskip("numpy").array(rig[k]) for k in ("axis", "zero", "swing"))
    assert abs(axis @ zero) < 1e-3 and abs(axis @ swing) < 1e-3 and abs(zero @ swing) < 1e-3
    assert 0.4 < rig["length"] < 0.7
