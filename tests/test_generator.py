"""Robots generated from a sentence: read into a plan, built from parts, kept with the world, changed by another
sentence, and moving as soon as they are added."""

from __future__ import annotations

import numpy as np
import pytest

from robotforge.design import generator
from robotforge.design.generator import fit
from robotforge.design.generator.plan import Plan
from robotforge.design.store import WorldStore
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

# The milestone's fifteen sentences and what each must become: (kind, legs, wheels, arms) where they matter.
FIFTEEN = [
    ("a six-legged robot with two arms", "hexapod", {"legs": 6, "arms": 2}),
    ("a tall wheeled courier with a lidar", "rover", {"cargo": True, "lidar": True}),
    ("a snake with a gripper head", "snake", {"arms": 1}),
    ("a small orange robot dog", "quadruped", {"legs": 4, "color": "orange"}),
    ("a big carbon fiber quadruped with long legs and a camera head", "quadruped", {"material": "carbon"}),
    ("a spider robot in black and red", "hexapod", {"round": True, "color": "black", "accent": "red"}),
    ("a six-wheeled mars rover with an arm", "rover", {"wheels": 6, "arms": 1}),
    ("a self-balancing robot on two wheels", "balancer", {}),
    ("a robot arm with six joints and a suction cup", "arm", {"joints": 6, "hand": "suction"}),
    ("a long green snake with 12 segments", "snake", {"segments": 12, "color": "green"}),
    ("a centaur in white and gold", "quadruped", {"arms": 2, "color": "white"}),
    ("a tall humanoid", "humanoid", {}),
    ("a wheeled robot with two arms", "rover", {"arms": 2}),
    ("一只带激光雷达的蓝色四足机器狗", "quadruped", {"legs": 4, "lidar": True, "color": "blue"}),
    ("a tentacle arm with a gripper", "arm", {"tentacle": True}),
]


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("generated")
    return playground


# ---- reading sentences -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("sentence", "kind", "fields"), FIFTEEN, ids=[s for s, *_ in FIFTEEN])
def test_each_sentence_becomes_the_robot_it_says(sentence, kind, fields):
    made = generator.generate(sentence)
    assert made.kind == kind
    for name, value in fields.items():
        assert getattr(made.plan, name) == value, name
    assert made.plan.sentences == (sentence,)


def test_sizes_are_factors_and_lengths_in_meters_set_them():
    assert generator.generate("a tall humanoid").plan.height > 1
    assert generator.generate("a small dog").plan.size < 1
    assert generator.generate("a very long rover").plan.length > generator.generate("a long rover").plan.length
    dog = generator.generate("a dog with long legs").plan
    assert dog.height > 1 and dog.length == 1  # the legs are long, not the body
    rover = generator.generate("a wheeled robot 54 cm long").plan
    assert rover.length == pytest.approx(1.5)  # the rover is 0.36 m long at factor 1
    huge = generator.generate("a wheeled robot 2 m long")
    assert huge.plan.length == 1.8 and any("length" in n for n in huge.notes)


def test_the_same_sentence_makes_the_same_robot_and_a_seed_another():
    a, b = generator.generate("a robot dog"), generator.generate("a robot dog")
    assert a.source == b.source
    other = generator.generate("a robot dog", seed=a.plan.seed + 1)
    assert other.source != a.source and other.kind == a.kind


def test_what_cannot_be_built_says_what_to_say_instead():
    with pytest.raises(ValueError, match="legs .*wheels"):
        generator.generate("a nice robot")
    with pytest.raises(ValueError, match="drone"):
        generator.generate("a flying robot with four propellers")
    eight = generator.generate("a spider with eight legs")
    assert eight.plan.legs == 6 and any("8 legs became 6" in n for n in eight.notes)
    three = generator.generate("a rover with three arms")
    assert three.plan.arms == 2 and any("at most two arms" in n for n in three.notes)


# ---- changing a generated robot by another sentence ----------------------------------------------------------------


def test_changes_apply_to_the_plan_they_are_given():
    dog = generator.generate("a small orange robot dog").plan
    longer = generator.change(dog, "make it longer").plan
    assert longer.length > dog.length and longer.size == dog.size and longer.color == "orange"
    six = generator.change(dog, "six legs instead of four")
    assert six.kind == "hexapod" and six.plan.seed == dog.seed
    assert generator.change(dog, "another one").plan.seed != dog.seed
    assert generator.change(dog, "wheels instead of legs").kind == "rover"
    assert generator.change(dog, "make it red and add a lidar").plan.lidar is True
    with pytest.raises(ValueError, match="what to change"):
        generator.change(dog, "hmm")


def test_adding_an_arm_keeps_the_body():
    """'add an arm' read the word arm as the body and turned the dog into a standing arm."""
    dog = generator.generate("a robot dog").plan
    armed = generator.change(dog, "add an arm")
    assert armed.kind == "quadruped" and armed.plan.arms == 1
    assert generator.change(armed.plan, "remove the arm").plan.arms == 0


def test_a_plan_round_trips_through_what_the_world_keeps():
    made = generator.generate("a six-wheeled mars rover with an arm")
    again = Plan.from_dict(made.plan.to_dict())
    assert again == made.plan
    with pytest.raises(ValueError, match="unknown fields"):
        Plan.from_dict({"body": "legged", "tails": 2})


# ---- the builder's measurement chooses the leg joints -------------------------------------------------------------


def test_bigger_walkers_get_stronger_leg_joints_by_what_the_builder_measures():
    small = generator.generate("a robot dog").source["build"]
    big = generator.generate("a big robot dog with long legs").source["build"]
    leg = next(p for p in small["parts"] if p["name"] == "left_shin")
    assert "actuator" not in leg  # the robot's own hobby servo holds a small dog
    leg = next(p for p in big["parts"] if p["name"] == "left_shin")
    assert leg["actuator"] in fit.LADDER[2:]  # a joint module, maybe geared
    assert fit.strain(big) <= fit.WALK_STRAIN


# ---- in a world ----------------------------------------------------------------------------------------------------


def test_add_robot_describe_builds_it_and_keeps_its_plan(pg):
    result = pg.add_robot("mantis", describe="a six-legged robot with two arms")
    gen = result["generated"]
    assert gen["kind"] == "hexapod" and "mantis/left_hand" in gen["how"]
    entry = pg.current_world().robots[0]
    assert entry.build is not None and entry.generated["legs"] == 6
    assert pg.behaviors.kind("mantis").value == "hexapod"
    assert pg.behaviors.arms("mantis") == ["mantis/left_hand", "mantis/right_hand"]
    # world.yaml keeps the plan, and reopening the world reads it back
    loaded = WorldStore.open(str(pg.store.file)).load()
    assert loaded.robots[0].generated == entry.generated


def test_update_robot_describe_changes_a_generated_robot(pg):
    pg.add_robot("dog", describe="a small robot dog")
    result = pg.update_robot("dog", describe="six legs instead of four")
    assert result["generated"]["kind"] == "hexapod"
    assert pg.behaviors.kind("dog").value == "hexapod"
    assert pg.current_world().robots[0].generated["sentences"] == ["a small robot dog", "six legs instead of four"]
    pg.update_part("dog", "robot", {"color": "purple"})
    result = pg.update_robot("dog", describe="make it longer")
    assert any("changed by hand" in n for n in result["notes"])


def test_describe_is_a_whole_robot_and_only_for_generated_robots(pg):
    with pytest.raises(PlaygroundError, match="leave out template"):
        pg.add_robot("x", describe="a robot dog", template="quadruped")
    with pytest.raises(PlaygroundError, match="params may only give a seed"):
        pg.add_robot("x", describe="a robot dog", params={"color": "red"})
    pg.add_robot("a", describe="a robot dog", params={"seed": 4})
    assert pg.current_world().robots[0].generated["seed"] == 4
    pg.add_robot("t", template="quadruped", pos=[2, 0])
    with pytest.raises(PlaygroundError, match="not made from a sentence"):
        pg.update_robot("t", describe="make it longer")


@pytest.mark.parametrize(
    ("sentence", "speed"),
    [("a small orange robot dog", 0.25), ("a tall wheeled courier with a lidar", 0.25), ("a snake", 0.1)],
)
def test_a_generated_robot_moves_as_soon_as_it_is_added(pg, sentence, speed):
    pg.add_robot("bot", describe=sentence)
    pg.run_fast(lambda: False, 1.0)
    start = pg.behaviors.pose("bot")["position"]
    pg.behaviors.move("bot", speed, 0.0)
    pg.run_fast(lambda: False, 4.0)
    end = pg.behaviors.pose("bot")
    assert end["upright"]
    assert np.hypot(end["position"][0] - start[0], end["position"][1] - start[1]) > 0.25 * speed * 4


@pytest.mark.parametrize(("sentence", "kind", "fields"), FIFTEEN, ids=[s for s, *_ in FIFTEEN])
def test_each_of_the_fifteen_builds_stands_and_moves(pg, sentence, kind, fields):
    pg.add_robot("bot", describe=sentence)
    assert pg.behaviors.kind("bot").value == kind
    pg.run_fast(lambda: False, 1.5)
    pose = pg.behaviors.pose("bot")
    assert pose["upright"]
    if kind == "arm":
        return
    speed = {"snake": 0.1, "humanoid": 0.15}.get(kind, 0.25)
    pg.behaviors.move("bot", speed, 0.0)
    pg.run_fast(lambda: False, 5.0)
    end = pg.behaviors.pose("bot")
    assert end["upright"]
    moved = np.hypot(end["position"][0] - pose["position"][0], end["position"][1] - pose["position"][1])
    assert moved > (0.2 if kind == "snake" else 0.5 * speed * 5)
