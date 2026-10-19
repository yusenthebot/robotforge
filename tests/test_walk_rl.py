"""RL II: any body learns to walk in the viewer, trained on a background thread."""

import time
from pathlib import Path

import pytest
from importer_fixtures import critter_mjcf, critter_urdf, write

from robotforge.behaviors import rhythm_walk
from robotforge.behaviors.manager import BehaviorError
from robotforge.control.cpg import HEADING_WEIGHT, Body, Reward
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


def test_the_starfish_step_drops_the_energy_cost_and_a_walk_lost_to_it_says_why(pg):
    """Regression: the energy cost of 0.3 from the reward step carried over, so the starfish learned to stand still
    and its step could never pass. The step now says to set the cost back to 0, and a training lost to it says so."""
    described = pg.lessons.start("learn_to_walk")
    body = next(s for s in described["steps"] if s["id"] == "body")["body"]
    assert "Set Energy cost back to 0" in body["en"] and "「能耗的代价」调回 0" in body["zh"]
    lessons = pg.lessons
    lessons.passed.update({"rhythm", "reward"})
    lessons.handle({"action": "goto", "step": 2})
    for control, value in (("robot", 1), ("energy", 2), ("iterations", 10)):  # what the reward step left behind
        lessons.handle({"action": "set", "control": control, "value": value})
    update = _train(pg)
    assert update["results"]["training"]["distance"] < walk_rl.KEEP_MIN and "body" not in update["passed"]
    assert update["results"]["summary"]["en"].endswith("set Energy cost back to 0 and learn again.")


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
    monkeypatch.setattr(rhythm_walk, "WALKING_ROUNDS", 10)
    monkeypatch.setattr(rhythm_walk, "STEER_ROUNDS", 10)
    monkeypatch.setattr(rhythm_walk, "WALK_CHECKS", rhythm_walk.WALK_CHECKS[::5])  # a tenth of a second apart
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


@pytest.mark.parametrize("file", ["critter.urdf", "critter.xml"])
def test_a_robot_imported_from_a_file_learns_keeps_its_walk_and_walks_at_home(pg, monkeypatch, file):
    """A URDF, or an MJCF written by hand. Regression (MJCF): its unnamed motors belonged to no robot, so the lesson
    never offered it, and its home keyframe was dropped."""
    # How well turns are learned is the starfish's test; this one is about the imported file finding its way.
    monkeypatch.setattr(rhythm_walk, "TURN_ROUNDS", 8)
    monkeypatch.setattr(rhythm_walk, "TURN_STARTS", 1)
    monkeypatch.setattr(rhythm_walk, "WALKING_ROUNDS", 4)
    monkeypatch.setattr(rhythm_walk, "WALK_CHECKS", rhythm_walk.WALK_CHECKS[::5])
    monkeypatch.setattr(rhythm_walk, "STEER_ROUNDS", 4)
    pg.new_world("mine")
    write(pg.store.dir / file, critter_urdf() if file.endswith(".urdf") else critter_mjcf())
    pg.add_robot("critter", file=file)  # a path relative to the world folder, as world.yaml keeps it
    assert pg.behaviors.kind("critter").value == "other"  # two-joint legs: no gait code knows it
    with pytest.raises(BehaviorError, match="RL II"):
        pg.behaviors.move("critter", 0.2, 0.0)
    described = pg.lessons.start("learn_to_walk")
    robots = next(c for c in described["controls"] if c["id"] == "robot")["options"]
    assert robots[-1]["en"] == "Your robot: critter"
    lesson = pg.lessons.lesson
    pg.lessons.handle({"action": "set", "control": "robot", "value": 3})
    pg.lessons.handle({"action": "set", "control": "iterations", "value": 15})
    trained = _train(pg)["results"]["training"]
    assert trained["robot"] == "yours" and trained["distance"] > walk_rl.KEEP_MIN
    pg.lessons.handle({"action": "press", "control": "keep"})
    assert pg.lessons.ctx.results["kept"]["robot"] == "critter"
    lesson.turning.join(timeout=120)
    assert pg.lessons.ctx.results["kept"]["turns"], pg.lessons.ctx.results["summary"]["en"]
    kept = rhythm_walk.load_walk(lesson.own[1], "critter")  # the learner's world folder
    solo = lesson.solo["yours"]
    body = Body.find(solo, "yours")
    learned_around = dict(zip(rhythm_walk.joint_names(solo, body), solo.key("home").qpos[body.qadr], strict=True))
    assert kept.home == pytest.approx(learned_around)  # it says what it learned around
    pg.lessons.exit()

    start = pg.behaviors.pose("critter")
    assert pg.behaviors.move("critter", 0.2, 0.0)["controller"] == "learned rhythm"
    pg.run_fast(lambda: False, walk_rl.ATTEMPT)
    end = pg.behaviors.pose("critter")
    gone = end["position"][0] - start["position"][0]
    assert gone > 0.5 * trained["distance"], f"learned {trained['distance']} m, walked {gone:.2f} m at home"
    assert pg.behaviors.move("critter", 0.0, 0.8)["turning"] == "left"
    pg.run_fast(lambda: False, 2.0)
    assert (pg.behaviors.pose("critter")["heading_deg"] - end["heading_deg"] + 540) % 360 - 180 > 10


MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
# Which of the Menagerie stand-ins, added by file, RL II offers, and what the others are told instead.
OFFERED = {
    "unitree_g1": None,  # 29 position servos
    "unitree_go2": "torque motors",  # the file's legs; the library turns them into servos, a file is kept as it is
    "agilex_piper": "fixed to the world",
    "franka_emika_panda": "fixed to the world",
    "trs_so_arm100": "fixed to the world",
    "bitcraze_crazyflie_2": "no built-in way to move",  # drones: no joint but the free one
    "skydio_x2": "0 servo joints",
    "anybotics_anymal_c": None,  # 12 position servos
    "robotis_op3": None,  # 20 position servos
    "universal_robots_ur5e": "no built-in way to move",  # an arm the file does not fix to the world
    "aloha": "no built-in way to move",
    "robotiq_2f85": "no built-in way to move",  # a gripper on its own
    "leap_hand": "fixed to the world",  # hands: the file fixes the palm where it is
    "wonik_allegro": "fixed to the world",
}


@pytest.mark.parametrize("path", sorted(MENAGERIE.glob("*/*.xml")), ids=lambda p: p.parent.name)
def test_rl_ii_offers_a_robot_from_an_mjcf_file_only_if_it_can_learn_and_says_why_not(pg, path):
    """Regression: the Go2's file drives its legs with torque motors, which a rhythm cannot aim, yet move told the
    learner to teach it in RL II, which never listed it."""
    why = OFFERED[path.parent.name]
    pg.new_world("mine")
    pg.add_robot("it", file=str(path))
    if why != "fixed to the world":  # an arm on the floor is told it cannot move around at all
        with pytest.raises(BehaviorError) as exc:
            pg.behaviors.move("it", 0.2, 0.0, "rhythm")
        assert ("Learn, pick 'RL II" in str(exc.value)) == (why is None)
        assert why is None or why in str(exc.value)
    described = pg.lessons.start("learn_to_walk")
    robots = next(c for c in described["controls"] if c["id"] == "robot")["options"]
    assert (robots[-1]["en"] == "Your robot: it") == (why is None)
    pg.lessons.exit()


def test_shoved_trainings_do_not_count_the_heading_and_the_panel_says_so(pg):
    """A shove turns the robot whatever its rhythm does: counting the heading then made the humanoid's reflex
    trainings worse (a steady walk in 1 of 12 instead of 5). Shoved, the reward leaves the heading out."""
    assert walk_rl.lesson_reward(0.0, shove=False).heading == HEADING_WEIGHT
    assert walk_rl.lesson_reward(0.1, shove=True) == Reward(energy=0.1, survive=True, heading=0.0)
    pg.lessons.start("learn_to_walk")
    lesson, ctx = pg.lessons.lesson, pg.lessons.ctx
    model, data = pg.session.model, pg.session.data

    def reward_shown() -> str:
        return next(f for f in lesson.live(model, data, ctx)["formula"] if f["show"] == "f_reward")["expr"]["en"]

    assert reward_shown().startswith(f"distance * (1 - {HEADING_WEIGHT:g} * turn)")
    ctx.values["shove"] = 1.0
    assert reward_shown().startswith("distance - ")
