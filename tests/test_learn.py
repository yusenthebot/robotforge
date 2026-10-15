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


def _test(pg: Playground) -> dict:
    """Press Run test and play it out; returns the update after the result came in."""
    pg.lessons.handle({"action": "press", "control": "test"})
    _play(pg, 3.3)
    return _play(pg, 0.1)


def test_the_pd_lesson_can_be_walked_end_to_end(pg):
    described = pg.lessons.start("pd_control")
    assert [s["id"] for s in described["steps"]] == ["feel", "p", "d", "why_low", "ff_i", "challenge"]
    assert described["rig"]["kind"] == "joint" and described["camera"]["eye"]
    assert described["steps"][1]["quiz"]["predict"] is True
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
    assert update["live"]["readout"]["en"].startswith("Controller off")
    with pytest.raises(LessonError, match="not open yet"):
        lessons.handle({"action": "goto", "step": 3})  # skipping ahead is not allowed
    lessons.handle({"action": "next"})

    # 2: P only: two tests with different Kp both swing; the step waits for a guess first
    assert _step_id(pg) == "p"
    _set(pg, kp=10)
    first = _test(pg)
    _set(pg, kp=40)
    second = _test(pg)
    assert "p" not in second["passed"]  # no guess yet
    assert first["results"]["test"]["settle_s"] is None and second["results"]["test"]["settle_s"] is None
    lessons.handle({"action": "answer", "choice": 2})  # a wrong guess still counts as a guess...
    with pytest.raises(LessonError, match="already made your guess"):
        lessons.handle({"action": "answer", "choice": 1})
    update = _play(pg, 0.1)
    assert "p" in update["passed"]
    assert update["quiz"]["right"] is False and update["quiz"]["explain"]  # ...and is told afterwards
    rows = {r["show"] for r in update["live"]["formula"]}
    assert {"f_p", "f_d", "f_i", "f_g"} <= rows and "f_p" in update["shown"] and "f_d" not in update["shown"]
    lessons.handle({"action": "next"})

    # 3: D: guess, sweep Kd, then a run that settles fast without overshoot
    lessons.handle({"action": "answer", "choice": 1})
    update = _play(pg, 0.1)
    assert update["quiz"] == {"chosen": 1, "pending": True}  # no verdict before the experiment
    lessons.handle({"action": "sweep", "control": "kd"})
    settle = {p["x"]: p["settle_s"] for p in lessons.ctx.sweep["points"]}
    assert settle[3] < settle[6] and settle[3] < settle[1]  # a sweet spot: too little wobbles, too much creeps
    _set(pg, kd=3)
    update = _test(pg)
    assert "d" in update["passed"] and update["quiz"]["right"] is True
    pd_end = update["results"]["test"]["end_err"]
    lessons.handle({"action": "next"})

    # 4: steady-state error: twice the Kp, half the error
    lessons.handle({"action": "answer", "choice": 0})
    _set(pg, kp=80, kd=5)
    update = _test(pg)
    assert "why_low" in update["passed"]
    assert update["results"]["test"]["end_err"] == pytest.approx(pd_end / 2, rel=0.15)
    lessons.handle({"action": "next"})

    # 5: feedforward removes the error at once, the integral only slowly
    lessons.handle({"action": "answer", "choice": 0})
    _set(pg, kp=40, kd=3, gravity=1)
    ff = _test(pg)["results"]["test"]
    _set(pg, gravity=0, ki=10)
    update = _test(pg)
    assert "ff_i" in update["passed"]
    assert ff["end_err"] < 0.2 < update["results"]["test"]["end_err"] < pd_end
    lessons.handle({"action": "next"})

    # 6: the challenge: P+D alone gets two stars, adding the feedforward gets three
    _set(pg, ki=0)
    assert _test(pg)["results"]["test"]["stars"] == 2
    _set(pg, gravity=1)
    update = _test(pg)
    assert update["results"]["test"]["stars"] == 3 and "challenge" in update["passed"]

    # the experiment record keeps the last five tests, with their settings and curves, to lay over each other
    runs = lessons.ctx.runs
    assert len(runs) == 5 and [r["params"]["gravity"] for r in runs[-2:]] == [0, 1]
    assert runs[-1]["metrics"]["stars"] == 3 and runs[-1]["x"][0] == pytest.approx(0, abs=0.02)


def test_a_sweep_shows_what_one_gain_does(pg):
    pg.lessons.start("pd_control")
    lessons = pg.lessons
    with pytest.raises(LessonError, match="later step"):
        lessons.handle({"action": "sweep", "control": "kp"})  # the gains unlock later
    for step in ("feel", "p", "d", "why_low", "ff_i"):  # jump to the challenge, where every knob is open
        lessons.passed.add(step)
    lessons.handle({"action": "goto", "step": 5})
    _set(pg, kd=1.0)
    started = pg.session.data.time
    lessons.handle({"action": "sweep", "control": "kp"})
    assert pg.session.data.time == started  # sweeps run on copies; the live joint is not disturbed
    sweep = lessons.ctx.sweep
    overshoot = [p["overshoot_pct"] for p in sweep["points"]]
    assert sweep["control"] == "kp" and len(overshoot) == 9
    assert overshoot[0] < 1 and overshoot[-1] > 25 and overshoot == sorted(overshoot)  # more Kp, more overshoot
    assert pg.lessons.tick()["bench"]["sweep"]["control"] == "kp"  # sent with the next update after a change
    assert pg.lessons.tick()["bench"] is None  # and not again until the record changes
    lessons.handle({"action": "clear_runs"})
    assert lessons.ctx.sweep is None and not lessons.ctx.runs


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
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1:8737") as client,
        client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws,
    ):
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


def test_the_ik_lesson_can_be_walked_end_to_end(pg):
    from robotforge.learn import ik

    described = pg.lessons.start("inverse_kinematics")
    assert described["rig"]["kind"] == "planar_arm"
    lessons = pg.lessons
    # 1: forward kinematics with the angle sliders
    q1, q2 = ik.analytic(*ik.START_TARGET, elbow_up=True)
    _set(pg, theta1=round(np.degrees(q1)), theta2=round(np.degrees(q2)))
    assert "fk" in _play(pg, 1.0)["passed"]
    lessons.handle({"action": "next"})
    # 2: analytic IK follows the target to three places
    _set(pg, method=1)
    for tx, tz in ((0.35, 0.05), (0.05, 0.4), (-0.25, 0.25)):
        _set(pg, tx=tx, tz=tz)
        update = _play(pg, 1.0)
        assert update["live"]["err_cm"] < 2
    assert "ik" in update["passed"]
    lessons.handle({"action": "next"})
    # 3: out of reach, the arm points straight at the target
    _set(pg, tx=0.6, tz=0.45)
    update = _play(pg, 1.2)
    assert update["live"]["reachable"] is False and "workspace" in update["passed"]
    assert abs(update["live"]["q2"]) < 4  # stretched (nearly) straight
    lessons.handle({"action": "next"})
    # 4: both elbow solutions reach the same point
    _set(pg, tx=0.3, tz=0.05)
    up = _play(pg, 1.0)["live"]["q2"]
    _set(pg, elbow_up=0)
    update = _play(pg, 1.2)
    assert up < 0 < update["live"]["q2"] and update["live"]["err_cm"] < 2 and "elbow" in update["passed"]
    lessons.handle({"action": "next"})
    # 5: iterative IK closes a fresh error step by step
    _set(pg, method=2)
    errors = [_play(pg, 0.5)["live"]["err_cm"]]
    for _ in range(8):
        lessons.handle({"action": "press", "control": "iterate"})
        errors.append(_play(pg, 0.4)["live"]["err_cm"])
    assert errors[0] > 20 and errors[-1] < 1 and all(b <= a + 0.5 for a, b in zip(errors, errors[1:], strict=False))
    lessons.handle({"action": "next"})
    # 6: the quiz
    lessons.handle({"action": "answer", "choice": 1})
    assert _play(pg, 0.1)["passed"][-1] == "why_iterate"


def test_choice_controls_reject_options_that_do_not_exist(pg):
    pg.lessons.start("inverse_kinematics")
    pg.lessons.step = 1  # where the method unlocks
    with pytest.raises(LessonError, match="options 0 to 2"):
        pg.lessons.handle({"action": "set", "control": "method", "value": 5})


def test_the_ik_rig_stays_stable_under_wild_input(pg):
    pg.lessons.start("inverse_kinematics")
    rng = np.random.default_rng(1)
    for k in range(200):
        if k % 10 == 0:
            _set(pg, theta1=float(rng.uniform(-90, 180)), theta2=float(rng.uniform(-150, 150)))
        _play(pg, 0.05)
    assert np.isfinite(pg.session.data.qacc).all() and pg.session.data.time > 9.9


def test_the_sensors_lesson_can_be_driven_end_to_end(pg):
    """A scripted driver plays the sensors lesson: encoders, IMU, lidar, bumper, camera, quiz."""
    import math

    from robotforge.learn import sensors

    described = pg.lessons.start("sensors")
    assert described["drive"] == {"max_speed": sensors.MAX_SPEED, "turn": sensors.MAX_TURN}
    lessons, session = pg.lessons, pg.session
    body = session.model.body("rover").id

    def drive(speed: float, turn: float) -> None:
        lessons.handle({"action": "drive", "speed": speed, "turn": turn})

    def heading_error(point) -> float:
        x, y = session.data.xpos[body][:2]
        heading = math.atan2(session.data.xmat[body][3], session.data.xmat[body][0])
        want = math.atan2(point[1] - y, point[0] - x)
        return math.atan2(math.sin(want - heading), math.cos(want - heading))

    def face(point) -> None:
        for _ in range(200):
            err = heading_error(point)
            if abs(math.degrees(err)) < 3:
                break
            drive(0, max(-1.5, min(1.5, 2.5 * err)))
            _play(pg, 0.05)
        drive(0, 0)
        _play(pg, 0.3)

    drive(0.3, 0)
    _play(pg, 3.5)
    drive(0, 0)
    update = _play(pg, 0.5)
    assert "encoders" in update["passed"] and abs(update["live"]["odom"] - update["live"]["true"]) < 0.05
    lessons.handle({"action": "next"})

    drive(0, 2.0)
    _play(pg, 3.6)
    drive(0, 0)
    update = _play(pg, 0.5)
    assert "imu" in update["passed"]
    assert abs(update["live"]["accel"][2] - 9.81) < 0.3  # an accelerometer at rest feels gravity
    lessons.handle({"action": "next"})

    x, _ = session.data.xpos[body][:2]
    face((x, -sensors.ARENA))
    for _ in range(300):
        update = _play(pg, 0.05)
        if 0 <= update["live"]["ranges"][0] < 0.3:
            break
        drive(0.15, 0)
    drive(0, 0)
    update = _play(pg, 2.5)
    assert "lidar" in update["passed"] and update["live"]["touch"] == 0
    lessons.handle({"action": "next"})

    drive(-0.2, 0)
    _play(pg, 1.0)
    face(sensors.CRATE)
    drive(0.25, 0)
    for _ in range(200):
        if "bumper" in _play(pg, 0.05)["passed"]:
            break
    drive(0, 0)
    lessons.handle({"action": "next"})

    drive(-0.2, 0)
    _play(pg, 1.0)
    face(sensors.BALL)
    for _ in range(400):
        update = _play(pg, 0.05)
        if update["results"]["ball"]["distance"] < 1.0:
            break
        drive(0.3, max(-1.0, min(1.0, 2 * heading_error(sensors.BALL))))
    drive(0, 0)
    update = _play(pg, 0.5)
    assert "camera" in update["passed"]
    lessons.handle({"action": "next"})
    lessons.handle({"action": "answer", "choice": 1})
    assert _play(pg, 0.1)["passed"][-1] == "drift"


def test_driving_is_only_for_lessons_with_a_robot_to_drive(pg):
    pg.lessons.start("pd_control")
    with pytest.raises(LessonError, match="no driving"):
        pg.lessons.handle({"action": "drive", "speed": 0.2, "turn": 0})


def _until_trained(pg: Playground, lesson, limit: float = 600.0) -> None:
    """Keep the physics going until the lesson's training thread ends (slow CI runners need minutes)."""
    import time

    deadline = time.monotonic() + limit
    while lesson.thread.is_alive() and time.monotonic() < deadline:
        _play(pg, 0.2)
    assert not lesson.thread.is_alive(), f"training did not finish within {limit:.0f} s"


def test_the_rl_lesson_trains_a_balancer_while_you_watch(pg, monkeypatch):
    from robotforge.learn import rl

    monkeypatch.setattr(rl, "PACE", 0.0)  # no need to slow it down for a test
    pg.lessons.start("reinforcement_learning")
    lessons = pg.lessons
    with pytest.raises(LessonError, match="later step"):
        lessons.handle({"action": "press", "control": "train"})
    lessons.handle({"action": "press", "control": "release"})
    assert "fall" in _play(pg, 3.0)["passed"]  # no control: it falls
    lessons.handle({"action": "next"})
    lessons.handle({"action": "answer", "choice": 1})
    _play(pg, 0.1)
    lessons.handle({"action": "next"})
    for _ in range(3):
        lessons.handle({"action": "press", "control": "random"})
        _play(pg, 1.0)
    lessons.handle({"action": "next"})
    lessons.handle({"action": "press", "control": "train"})
    lesson = lessons.lesson
    _until_trained(pg, lesson)
    update = _play(pg, 0.5)
    assert "train" in update["passed"] and update["results"]["train"]["mean"] >= rl.TRAINED_RETURN
    update = _play(pg, 2 * rl.SHOW_SECONDS)
    assert update["results"]["last_attempt"] >= rl.SHOW_SECONDS - 0.1  # up for a whole attempt, not fallen
    runs = lessons.ctx.runs
    assert len(runs) == 1 and runs[0]["metrics"]["learned_at"] is not None and len(runs[0]["y"]) == rl.ITERATIONS
    lessons.handle({"action": "next"})
    lessons.handle({"action": "set", "control": "lr", "value": 0.02})  # too small a step to learn in time
    lessons.handle({"action": "set", "control": "fast", "value": 1})
    lessons.handle({"action": "press", "control": "train"})
    _until_trained(pg, lesson)
    update = _play(pg, 0.3)
    assert "tune" in update["passed"]
    slow = lessons.ctx.runs[-1]
    assert slow["params"]["lr"] == 0.02 and slow["metrics"]["learned_at"] is None
    assert update["bench"] is None  # the record goes out once per change, not on every update
    # The cart now runs the step-0.02 weights, which never learned. Pushing them used to be a dead end: the step
    # could not pass and nothing said why. Push puts the best trained weights back first, then pushes.
    lessons.handle({"action": "next"})
    lessons.handle({"action": "press", "control": "push"})
    update = _play(pg, 0.1)
    assert update["results"]["summary"]["en"].startswith("The newest weights never learned")
    assert "best learned weights" in update["live"]["readout"]["en"]
    update = _play(pg, rl.PUSH_AFTER + 4.0)
    assert "push" in update["passed"]
    lessons.handle({"action": "press", "control": "push"})  # weights that learned are pushed at once, as before
    assert lesson.live_ctrl.pushed_at == pg.session.data.time and not lesson.live_ctrl.pending_restart
    pg.lessons.exit()
    assert lesson.stopping.is_set()


def test_every_control_of_every_lesson_unlocks_in_some_step(pg):
    """A knob no step unlocks can never be used (the learn-to-walk lesson shipped with two such)."""
    from robotforge.learn import LESSONS

    for lesson_id in LESSONS:
        described = pg.lessons.start(lesson_id)
        lesson = pg.lessons.lesson
        unlocked = {c for step in lesson.steps for c in step.unlock}
        controls = {c["id"] for c in described["controls"]}
        assert controls <= unlocked, f"{lesson_id}: {sorted(controls - unlocked)} never unlock"
    pg.lessons.exit()
