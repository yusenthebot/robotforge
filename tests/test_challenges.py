"""Challenges: each one can be beaten, scored with stars, and retried after a reset."""

from __future__ import annotations

import math

import pytest

from robotforge.learn import challenges
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


def _heading_to(pg: Playground, body: int, point) -> tuple[float, float]:
    data = pg.session.data
    x, y = data.xpos[body][:2]
    heading = math.atan2(data.xmat[body][3], data.xmat[body][0])
    want = math.atan2(point[1] - y, point[0] - x)
    return math.atan2(math.sin(want - heading), math.cos(want - heading)), math.dist((x, y), point)


def test_challenges_are_listed_apart_from_lessons(pg):
    kinds = {c["id"]: c["kind"] for c in pg.lessons.catalog()}
    assert kinds["pd_control"] == "lesson"
    assert {k for k, v in kinds.items() if v == "challenge"} == {"maze_escape", "shove_survival", "obstacle_course"}


def test_the_maze_can_be_driven_out_of_and_is_timed_against_par(pg, monkeypatch):
    monkeypatch.setattr(challenges.random, "randrange", lambda n: 4273)
    described = pg.lessons.start("maze_escape")
    lesson = pg.lessons.lesson
    assert lesson.par == round(challenges.maze_path_length(4273) / challenges.DRIVE_SPEED)
    assert str(lesson.par) in described["steps"][0]["body"]["en"]
    assert _play(pg, 0.5)["live"]["readout"]["en"].startswith("time 0.0 s")  # the clock waits for the first key
    body = pg.session.model.body("rover/chassis").id
    for waypoint in challenges.maze_route(4273)[1:]:
        for _ in range(400):
            err, dist = _heading_to(pg, body, waypoint)
            if dist < 0.2:
                break
            speed = 0.4 if abs(err) < 0.3 else 0.0
            pg.lessons.handle({"action": "drive", "speed": speed, "turn": max(-2.0, min(2.0, 3 * err))})
            update = _play(pg, 0.05)
            if "run" in update["results"]:
                break
    update = _play(pg, 0.5)
    run = update["results"]["run"]
    assert run["stars"] >= 2 and run["time"] > 10 and "escape" in update["passed"]
    pg.session.reset()  # Reset starts a fresh attempt
    assert "run" not in _play(pg, 0.2)["results"]


def test_shoves_topple_a_held_pose_before_mpc(pg):
    levels = {}
    for choice in (0, 2):
        pg.lessons.start("shove_survival")
        pg.lessons.lesson.rng.seed(1)
        _play(pg, 0.5)
        pg.lessons.handle({"action": "set", "control": "controller", "value": choice})
        _play(pg, 0.5)
        pg.lessons.handle({"action": "press", "control": "start"})
        for _ in range(40):
            update = _play(pg, 1.0)
            if "run" in update["results"]:
                break
        levels[choice] = update["results"]["run"]["levels"]
    assert levels[0] < levels[2]  # looking ahead keeps it on its feet longer
    assert update["results"]["stars"] >= 2


def test_holding_w_takes_the_dog_past_the_stairs(pg):
    """Regression: W walked the dog at 0.35 m/s, where it fell on the stairs in all 8 of our key-driven runs, and the
    tip to slow down there asked for a speed no key gives. At 0.3 m/s 6 of 8 got past them."""
    from robotforge.learn import playthrough

    described = pg.lessons.start("obstacle_course")
    assert described["drive"]["max_speed"] * playthrough.KEY_SHARE == pytest.approx(0.3, abs=0.01)
    assert "slow down" not in described["steps"][0]["body"]["en"]
    pg.lessons.exit()
    run = playthrough.play(pg, "obstacle_course").results["run"]  # W held, A and D tapped back to the middle
    assert run["stars"] >= 2, f"fell at {run['distance']} m, before the end of the stairs"


def test_the_obstacle_course_scores_how_far_the_dog_gets(pg):
    pg.lessons.start("obstacle_course")
    lesson = pg.lessons.lesson
    assert [name for name, _ in lesson.marks] == ["ramp", "stairs", "rough"]
    body = pg.session.model.body("dog/trunk").id
    update = None
    for _ in range(240):
        data = pg.session.data
        heading = math.atan2(data.xmat[body][3], data.xmat[body][0])
        steer = max(-1.0, min(1.0, -2 * data.xpos[body][1] - 1.5 * heading))  # keep to the middle of the course
        pg.lessons.handle({"action": "drive", "speed": 0.5, "turn": steer})
        update = _play(pg, 0.5)
        if "run" in update["results"]:
            break
    run = update["results"]["run"]
    assert run["distance"] > lesson.marks[1][1]  # past the stairs at least
    assert run["stars"] >= 2 and "course" in update["passed"]
