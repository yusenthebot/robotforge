"""Twins: copies of one robot with their own settings, put through the same test side by side."""

import pytest

from robotforge.learn import twins
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


def test_each_twin_gets_its_own_settings_under_its_own_heading(pg):
    described = pg.lessons.start("pid_twins")
    assert described["kind"] == "bench"
    own = [c for c in described["controls"] if c.get("group")]
    assert [c["id"] for c in own] == ["kp_a", "kd_a", "kp_b", "kd_b", "kp_c", "kd_c"]
    assert set(described["groups"]) == {"a", "b", "c"} and described["groups"]["b"]["color"] == "blue"
    with pytest.raises(LessonError, match="later step"):
        pg.lessons.handle({"action": "set", "control": "kp_b", "value": 5})  # B and C wait for step 3


def test_three_joints_take_the_same_step_and_show_what_kd_does(pg):
    pg.lessons.start("pid_twins")
    _play(pg, 0.5)
    pg.lessons.handle({"action": "press", "control": "test"})
    update = _play(pg, twins.pd.TEST_SECONDS + 0.5)
    a, b, c = (update["results"]["triplet"][n] for n in "abc")
    assert b["overshoot_pct"] > c["overshoot_pct"] + 20  # the same Kp: the weak brake flies past
    assert c["stars"] >= 2 and a["settle_s"] is None  # soft and unbraked never settles
    assert "compare" in update["passed"]
    assert {"q_a", "q_b", "q_c", "target"} <= set(update["samples"][-1])
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "set", "control": "kp_a", "value": 40})
    pg.lessons.handle({"action": "set", "control": "kd_a", "value": 3})
    pg.lessons.handle({"action": "press", "control": "test"})
    _play(pg, twins.pd.TEST_SECONDS + 0.5)
    assert "tune_a" in _play(pg, 0.1)["passed"]  # tuned like C, A earns its stars


def test_lqr_twins_trade_error_for_effort(pg):
    pg.lessons.start("lqr_twins")
    _play(pg, 1.0)
    pg.lessons.handle({"action": "press", "control": "push"})
    _play(pg, twins.WATCH_SECONDS + 0.3)
    update = _play(pg, 0.1)
    a, b = update["results"]["push"]["a"], update["results"]["push"]["b"]
    assert b["effort"] < 0.6 * a["effort"]  # R 20 against R 0.2: far gentler...
    assert b["lean"] >= a["lean"] and b["settle"] > a["settle"]  # ...for a little more lean, a slower recovery
    assert "trade" in update["passed"]
    pg.lessons.handle({"action": "next"})
    gain = pg.session.controllers["b"].K.copy()
    for control, value in (("r_b", 1), ("q_b", 4), ("q_a", 0)):
        pg.lessons.handle({"action": "set", "control": control, "value": value})
    assert abs(pg.session.controllers["b"].K[0, 1]) > abs(gain[0, 1])  # new weights, new gains at once
    _play(pg, 1.0)
    pg.lessons.handle({"action": "press", "control": "push"})
    _play(pg, twins.WATCH_SECONDS + 0.3)
    update = _play(pg, 0.1)
    a, b = update["results"]["push"]["a"], update["results"]["push"]["b"]
    assert b["lean"] < 0.7 * a["lean"] and b["travel"] > a["travel"]  # big Q: upright, at the cart's expense
    assert "weights" in update["passed"]


def test_reset_puts_the_twins_controllers_back(pg):
    pg.lessons.start("lqr_twins")
    _play(pg, 0.5)
    pg.session.reset()
    assert {"a", "b", "watch"} <= set(pg.session.controllers)
    _play(pg, 2.0)
    model, data = pg.session.model, pg.session.data
    for name in ("a", "b"):  # both still balancing their poles
        assert abs(data.qpos[model.joint(f"{name}/hinge").qposadr[0]]) < 0.05


def _until(pg: Playground, key: str, busy, limit: float = 40.0) -> dict:
    for _ in range(int(limit)):
        update = _play(pg, 1.0)
        if key in update["results"] and not busy(pg.lessons.lesson):
            return _play(pg, 0.1)
    raise AssertionError(f"no {key} result")


def test_mpc_twins_show_that_a_too_short_horizon_falls_first(pg):
    pg.lessons.start("mpc_twins")
    _play(pg, 0.5)
    pg.lessons.handle({"action": "press", "control": "shove"})
    update = _until(pg, "shoves", lambda lesson: lesson.next_shove is not None)
    first = update["results"]["shoves"]
    assert {r["noise"] for r in first.values()} == {0.05, 0.3} and all(r["survived"] >= 2 for r in first.values())
    assert "bold" in update["passed"]
    pg.lessons.handle({"action": "next"})
    for control, value in (("noise_a", 3), ("horizon_a", 0)):
        pg.lessons.handle({"action": "set", "control": control, "value": value})
    pg.lessons.handle({"action": "press", "control": "shove"})
    update = _until(pg, "shoves", lambda lesson: lesson.next_shove is not None)
    a, b = update["results"]["shoves"]["a"], update["results"]["shoves"]["b"]
    # Shoving is chaotic and the tries random, so one run's order is not asserted; the lesson quotes the
    # averages of six seeds instead (0.1 s: 6 levels, 0.35 s: 8.5, 0.6 s: 6.8).
    assert (a["horizon"], b["horizon"]) == (0.1, 0.35) and min(a["survived"], b["survived"]) >= 1
    assert "horizon" in update["passed"]
    rows = {r["label"]["en"]: r for r in update["live"]["table"]["rows"] if "cells" in r}
    assert rows["Physics steps per plan"]["cells"] == ["1,600", "5,600"]  # samples x horizon / step


def test_gait_twins_trade_energy_for_speed_and_clearance(pg):
    pg.lessons.start("gait_twins")
    _play(pg, 0.5)
    pg.lessons.handle({"action": "press", "control": "flat"})
    update = _until(pg, "flat", lambda lesson: lesson.test is not None)
    a, b = update["results"]["flat"]["a"], update["results"]["flat"]["b"]
    assert a["top"] > 1.5 * b["top"] and b["cot"] < a["cot"]  # quick steps: faster, dearer
    assert abs(a["speed"] - b["speed"]) < 0.03 and not (a["fell"] or b["fell"])  # both make the same speed
    pg.lessons.handle({"action": "next"})
    for control, value in (("period_a", 1), ("period_b", 1), ("lift_a", 0), ("lift_b", 2)):
        pg.lessons.handle({"action": "set", "control": control, "value": value})
    pg.lessons.handle({"action": "press", "control": "flat"})
    update = _until(pg, "flat", lambda lesson: lesson.test is not None)
    low, high = update["results"]["flat"]["a"], update["results"]["flat"]["b"]
    assert low["cot"] < 0.5 * high["cot"]  # a low lift is cheap on the flat...
    pg.lessons.handle({"action": "press", "control": "curb"})
    update = _until(pg, "curb", lambda lesson: lesson.test is not None)
    low, high = update["results"]["curb"]["a"], update["results"]["curb"]["b"]
    assert not low["crossed"] and high["crossed"]  # ...and catches on the curb
    assert "lift" in update["passed"]
