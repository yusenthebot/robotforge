"""RL II: any body learns to walk in the viewer, trained on a background thread."""

import time

import pytest

from robotforge.learn import walk_rl
from robotforge.playground import Playground


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _train(pg: Playground) -> dict:
    pg.lessons.handle({"action": "press", "control": "train"})
    deadline = time.monotonic() + 300
    while pg.lessons.lesson.thread.is_alive():
        assert time.monotonic() < deadline, "training did not finish"
        _play(pg, 0.1)
    return _play(pg, 0.1)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground.with_demo()


def test_the_dog_learns_and_the_viewer_replays_the_newest_rhythm(pg):
    described = pg.lessons.start("learn_to_walk")
    robots = next(c for c in described["controls"] if c["id"] == "robot")["options"]
    assert [o["en"] for o in robots][:3] == ["Template dog", "Three-legged starfish (built from parts)", "Humanoid"]
    pg.lessons.handle({"action": "answer", "choice": 2})
    pg.lessons.handle({"action": "set", "control": "iterations", "value": 20})
    update = _train(pg)
    trained = update["results"]["training"]
    assert trained["robot"] == "dog" and trained["rounds"] == 20 and trained["distance"] > 0.5
    assert "rhythm" in update["passed"]
    run = pg.lessons.ctx.runs[-1]
    assert run["x"] == list(range(1, 21)) and run["y"][-1] > run["y"][0]  # a learning curve
    lesson = pg.lessons.lesson
    assert lesson.players["dog"].rhythm.theta.any()  # the live dog plays what was learned
    walk = _play(pg, walk_rl.ATTEMPT + 0.5)["samples"]
    assert max(s["x"] for s in walk) > 0.2  # and really walks in the viewer


def test_your_own_robot_joins_the_list(pg):
    pg.new_world("mine")
    pg.add_robot("crab", build=walk_rl.STAR)
    described = pg.lessons.start("learn_to_walk")
    robots = next(c for c in described["controls"] if c["id"] == "robot")["options"]
    assert robots[-1]["en"] == "Your robot: crab"
    assert pg.session.model.body("yours/base") is not None
    pg.lessons.exit()
    assert pg.store.load().name == "mine"  # the learner's world comes back untouched
