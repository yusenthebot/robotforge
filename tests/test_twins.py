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
