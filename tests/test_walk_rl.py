"""RL II: any body learns to walk in the viewer, trained on a background thread."""

import time

import pytest

from robotforge.behaviors.manager import BehaviorError
from robotforge.learn import walk_rl
from robotforge.playground import Playground


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _samples(pg: Playground, seconds: float) -> list[dict]:
    """Every plot sample over `seconds`, not just the last tick's (that one may fall right after a replay restarts)."""
    out = []
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        out += pg.lessons.tick()["samples"]
    return out


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
    walk = _samples(pg, walk_rl.ATTEMPT + 0.5)  # at least one whole replay
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


def test_your_robot_keeps_what_it_learned_and_walks_that_way_at_home(pg, monkeypatch):
    # The flow, not the quality of the turns (test_rhythm_walk checks that): a lighter turn search keeps a slow
    # CI machine inside the wait below (Windows took over 120 s for the full two starts of 30 rounds a side).
    from robotforge.behaviors import rhythm_walk

    monkeypatch.setattr(rhythm_walk, "TURN_ROUNDS", 20)
    monkeypatch.setattr(rhythm_walk, "TURN_STARTS", 1)
    pg.new_world("mine")
    pg.add_robot("crab", build=walk_rl.STAR)
    with pytest.raises(BehaviorError, match="RL II"):  # no gait code knows a starfish
        pg.behaviors.move("crab", 0.2, 0.0)
    described = pg.lessons.start("learn_to_walk")
    assert any(c["id"] == "keep" for c in described["controls"])
    lesson = pg.lessons.lesson
    assert "crab" in described["steps"][0]["body"]["en"]  # sent here to teach crab: it can from the first step
    pg.lessons.handle({"action": "press", "control": "keep"})
    assert "Pick your robot" in pg.lessons.ctx.results["summary"]["en"]
    pg.lessons.handle({"action": "set", "control": "robot", "value": 3})
    assert lesson.active == "yours"
    pg.lessons.handle({"action": "press", "control": "keep"})
    assert "press Learn first" in pg.lessons.ctx.results["summary"]["en"]
    pg.lessons.handle({"action": "set", "control": "iterations", "value": 30})
    trained = _train(pg)["results"]["training"]
    pg.lessons.handle({"action": "press", "control": "keep"})
    kept = pg.lessons.ctx.results["kept"]
    assert kept["robot"] == "crab" and kept["speed"] == pytest.approx(trained["distance"] / walk_rl.ATTEMPT, abs=1e-3)
    lesson.turning.join(timeout=600)  # then it learns to turn, on its own, and keeps that too
    assert not lesson.turning.is_alive(), "turn learning did not finish"  # say so, rather than a bare False
    assert pg.lessons.ctx.results["kept"]["turns"]
    pg.lessons.exit()

    start = pg.behaviors.pose("crab")
    assert pg.behaviors.move("crab", 0.2, 0.0)["controller"] == "learned rhythm"
    pg.run_fast(lambda: False, walk_rl.ATTEMPT)
    gone = pg.behaviors.pose("crab")["position"][0] - start["position"][0]
    assert gone > 0.5 * trained["distance"], f"learned {trained['distance']} m, walked {gone:.2f} m at home"
    heading = pg.behaviors.pose("crab")["heading_deg"]
    assert pg.behaviors.move("crab", 0.0, 0.8)["turning"] == "left"
    pg.run_fast(lambda: False, 2.0)
    assert (pg.behaviors.pose("crab")["heading_deg"] - heading + 540) % 360 - 180 > 30


def test_the_robot_no_gait_code_knows_is_the_one_offered(pg):
    pg.new_world("mixed")
    pg.add_robot("rex", template="quadruped")
    pg.add_robot("crab", build=walk_rl.STAR, pos=[0, 1])
    described = pg.lessons.start("learn_to_walk")
    robots = next(c for c in described["controls"] if c["id"] == "robot")["options"]
    assert robots[-1]["en"] == "Your robot: crab"
