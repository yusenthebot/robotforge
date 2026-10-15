"""Lessons: a learner can walk the PD lesson end to end, and every step's check means what it says."""

from __future__ import annotations

import json

import numpy as np
import pytest
from starlette.testclient import TestClient

from robotforge.learn import LessonError
from robotforge.playground import Playground
from robotforge.viewer.server import ViewerHub, create_app


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("home")
    playground.add_robot("dog", template="quadruped")
    return playground


def _play(pg: Playground, seconds: float) -> dict:
    """Run physics the way the viewer does, ticking the lesson about 20 times a second."""
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _set(pg: Playground, **values: float) -> None:
    for control, value in values.items():
        pg.lessons.handle({"action": "set", "control": control, "value": value})


def _step_id(pg: Playground) -> str:
    return pg.lessons.lesson.steps[pg.lessons.step].id


def test_the_pd_lesson_can_be_walked_end_to_end(pg):
    described = pg.lessons.start("pd_control")
    assert [s["id"] for s in described["steps"]] == ["feel", "p", "d", "why_low", "ff_i", "challenge"]
    assert described["rig"]["kind"] == "joint" and described["camera"]["eye"]
    lessons = pg.lessons

    # 1: swing the arm by hand (a drag, and the arm really moves)
    assert _step_id(pg) == "feel"
    with pytest.raises(LessonError, match="later step"):
        _set(pg, kp=10)  # the gains unlock later
    arm = pg.session.model.body("arm").id
    pg.session.start_drag(arm, pg.session.data.site("tip").xpos.copy())  # grab the weight
    pg.session.move_drag(np.array([0.3, -0.06, 0.62]))
    _play(pg, 1.0)
    pg.session.end_drag()
    update = _play(pg, 0.2)
    assert "feel" in update["passed"]
    with pytest.raises(LessonError, match="not open yet"):
        lessons.handle({"action": "goto", "step": 3})  # skipping ahead is not allowed
    lessons.handle({"action": "next"})

    # 2: P only swings back and forth
    assert _step_id(pg) == "p"
    _set(pg, kp=20, target=90)
    update = _play(pg, 4.0)
    assert "p" in update["passed"] and update["samples"]
    lessons.handle({"action": "next"})

    # 3: D calms it down, but it stops short of the target
    _set(pg, kd=0.8)
    update = _play(pg, 3.0)
    assert "d" in update["passed"]
    assert 2 < update["live"]["error"] < 10  # the steady-state error the next step asks about
    lessons.handle({"action": "next"})

    # 4: the quiz: a wrong answer does not pass, the right one does
    lessons.handle({"action": "answer", "choice": 0})
    update = _play(pg, 0.1)
    assert update["quiz"]["right"] is False and "why_low" not in update["passed"]
    lessons.handle({"action": "answer", "choice": 1})
    update = _play(pg, 0.1)
    assert update["quiz"]["right"] is True and "why_low" in update["passed"]
    lessons.handle({"action": "next"})

    # 5: gravity compensation removes the error
    _set(pg, gravity=1)
    update = _play(pg, 2.0)
    assert "ff_i" in update["passed"] and abs(update["live"]["error"]) < 1
    lessons.handle({"action": "next"})

    # 6: the challenge: P+D alone gets two stars, adding the feedforward gets three
    _set(pg, gravity=0, kp=40, kd=3)
    lessons.handle({"action": "press", "control": "test"})
    update = _play(pg, 3.5)
    assert update["results"]["test"]["stars"] == 2
    _set(pg, gravity=1)
    lessons.handle({"action": "press", "control": "test"})
    update = _play(pg, 3.5)
    assert update["results"]["test"]["stars"] == 3 and "challenge" in update["passed"]


def test_reset_keeps_the_lesson_running_and_exit_brings_the_world_back(pg):
    pg.lessons.start("pd_control")
    pg.session.reset()
    assert "lesson" in pg.session.controllers  # the reset hook put the controller back
    assert pg.store is None  # the lesson's rig is not a saved world
    pg.lessons.exit()
    assert pg.store is not None and pg.store.load().name == "home"
    assert pg.session.model.body("dog/trunk") is not None
    assert pg.lessons.tick() is None


def test_changing_the_world_ends_the_lesson(pg):
    pg.lessons.start("pd_control")
    pg.load_scene("park")
    assert pg.lessons.tick() == {"type": "lesson", "ended": True}
    assert not pg.lessons.active


def test_unknown_lessons_and_actions_are_explained(pg):
    with pytest.raises(LessonError, match="unknown lesson"):
        pg.lessons.start("juggling")
    with pytest.raises(LessonError, match="no lesson is running"):
        pg.lessons.handle({"action": "next"})
    pg.lessons.start("pd_control")
    with pytest.raises(LessonError, match="unknown lesson action"):
        pg.lessons.handle({"action": "dance"})
    with pytest.raises(LessonError, match="no control"):
        pg.lessons.handle({"action": "set", "control": "volume", "value": 1})


def test_the_viewer_starts_a_lesson_and_receives_its_updates(pg):
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    assert hub.scene_message()["lessons"][0]["id"] == "pd_control"
    with TestClient(create_app(hub)) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "lesson_open", "action": "start", "lesson": "pd_control", "id": 7}))
        got_reply = got_update = False
        for _ in range(600):
            msg = ws.receive()
            if not msg.get("text"):
                continue
            data = json.loads(msg["text"])
            if data["type"] == "reply" and data["id"] == 7:
                assert data["ok"] and data["data"]["id"] == "pd_control"
                got_reply = True
            elif data["type"] == "lesson" and "step" in data:
                got_update = True
            if got_reply and got_update:
                break
        assert got_reply and got_update
    assert hub.scene_message()["lesson"]["id"] == "pd_control"
