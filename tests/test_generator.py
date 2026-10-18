"""Robots generated from a sentence: read into a plan, built from parts, kept with the world, changed by another
sentence, and moving as soon as they are added."""

from __future__ import annotations

from typing import Any

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


def test_bodies_said_in_other_words():
    """'two-legged' made a quadruped, a 'mobile manipulator' a standing arm without a base, and wheels on a dog were
    dropped without a word."""
    assert generator.generate("a two-legged walker").kind == "humanoid"
    mobile = generator.generate("a mobile manipulator with a suction cup")
    assert mobile.kind == "rover" and mobile.plan.arms == 1 and mobile.plan.hand == "suction"
    dog = generator.generate("a dog with wheels")
    assert dog.kind == "quadruped" and any("wheel_dog" in n for n in dog.notes)


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


def test_legs_said_as_legged_change_the_legs_not_the_body():
    """'long-legged' and 'make the legs longer' lengthened the body: 'legged' did not match the leg patterns."""
    said = (("a long-legged insect", True), ("a long legged dog", True), ("a short-legged dog", False))
    for sentence, taller in said:
        plan = generator.generate(sentence).plan
        assert plan.length == 1 and (plan.height > 1) == taller, sentence
    dog = generator.generate("a robot dog").plan
    for sentence, taller in (("make the legs longer", True), ("make the legs shorter", False)):
        plan = generator.change(dog, sentence).plan
        assert plan.length == 1 and (plan.height > 1) == taller, sentence


@pytest.mark.parametrize(
    ("sentence", "field", "value"),
    [
        ("一辆六轮火星车", "wheels", 6),
        ("带两个手臂的机器狗", "arms", 2),
        ("一个有两条手臂的六足机器人", "arms", 2),
        ("一条有十二节的蛇", "segments", 12),
        ("一个6轴机械臂", "joints", 6),
    ],
)
def test_chinese_counts_after_other_characters(sentence, field, value):
    """A count after another Chinese character was skipped: the lookbehind took CJK characters for word characters."""
    assert getattr(generator.generate(sentence).plan, field) == value


def test_meters_win_over_size_words():
    """'a small robot dog 50 cm long' was 0.37 m: the meters were turned into a factor before 'small' shrank it."""
    dog = generator.generate("a small robot dog 40 cm long").source["build"]["base"]["size"][0]
    rover = generator.generate("a small wheeled robot 40 cm long").source["build"]["base"]["size"][0]
    assert dog == pytest.approx(0.4, rel=0.07) and rover == pytest.approx(0.4, rel=0.07)  # within the seed's jitter
    assert generator.generate("a small humanoid 60 cm tall").source["params"]["height"] == pytest.approx(0.6)
    assert generator.generate("a big humanoid 40 cm tall").source["params"]["height"] == pytest.approx(0.4)


def test_negations_and_small_changes_read_as_said():
    two = generator.generate("a wheeled robot with two arms").plan
    assert generator.change(two, "remove an arm").plan.arms == 1
    assert generator.change(two, "remove one arm").plan.arms == 1
    assert generator.change(two, "remove the arms").plan.arms == 0
    assert generator.generate("a dog that does not have an arm").plan.arms == 0
    assert generator.generate("a robot dog without an arm").plan.arms == 0
    dog = generator.generate("a robot dog").plan
    assert generator.change(dog, "make it not so long").plan.length < 1
    assert generator.change(dog, "less long").plan.length < 1
    blue = generator.change(dog, "make it blue, don't make it red").plan
    assert blue.color == "blue" and blue.accent != "red"
    assert generator.generate("a blue robot dog, not red").plan.accent != "red"
    again = generator.change(dog, "make it longer again").plan
    assert again.seed == dog.seed and again.length > 1  # the same robot, longer: not another variation
    assert generator.change(dog, "another one").plan.seed != dog.seed


def test_a_plan_keeps_its_last_sentences_only():
    plan = generator.generate("a robot dog").plan
    for _ in range(25):
        plan = generator.change(plan, "make it longer").plan
    assert len(plan.sentences) == 20 and plan.sentences[-1] == "make it longer"


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


def _carried(source: Any) -> dict[str, set[str]]:
    """The sensor types and the hands (gripper, suction, magnet) anywhere in a robot's source."""
    found: dict[str, set[str]] = {"sensors": set(), "hands": set()}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("template") in ("gripper", "suction_cup", "magnet"):
                found["hands"].add({"suction_cup": "suction"}.get(node["template"], node["template"]))
            if node.get("template") == "arm" or "upper" in node.get("params", {}):
                found["hands"].add(node.get("params", {}).get("end_effector", "gripper"))
            if "end_effector" in node and isinstance(node["end_effector"], str):
                found["hands"].add(node["end_effector"])
            for sensor in node.get("sensors", []):
                found["sensors"].add(sensor if isinstance(sensor, str) else sensor["type"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(source)
    return found


@pytest.mark.parametrize(
    "sentence",
    [
        "a snake with a suction head",
        "a tentacle arm with a suction cup",
        "a tentacle with a magnet",
        "a tall humanoid with a lidar",
        "a robot arm with a lidar and a camera",
        "a headless robot dog with a camera",
        "a wheeled robot with two arms and suction cups",
        "a six-wheeled mars rover with a magnet arm",
        "a robot dog with a lidar and an arm",
        "a blind spider",
        "a self-balancing robot with an arm",
    ],
)
def test_what_the_plan_says_is_what_is_built(sentence):
    """A snake or tentacle asked for a suction cup or magnet got a gripper, and a humanoid kept lidar: true in its
    plan, with no lidar and no word about it."""
    made = generator.generate(sentence)
    plan, carried = made.plan, _carried(made.source)
    if plan.body == "humanoid":
        assert not plan.lidar and any("lidar" in n for n in made.notes)
        assert made.source["params"]["head"] == plan.head == plan.camera
        return
    assert ("lidar" in carried["sensors"]) == plan.lidar, "lidar"
    assert ("camera" in carried["sensors"]) == plan.camera, "camera"
    assert carried["hands"] == ({plan.hand} if plan.arms else set()), "hands"


# ---- the builder's measurement chooses the leg joints -------------------------------------------------------------


def test_bigger_walkers_get_stronger_leg_joints_by_what_the_builder_measures():
    small = generator.generate("a robot dog").source["build"]
    big = generator.generate("a big robot dog with long legs").source["build"]
    leg = next(p for p in small["parts"] if p["name"] == "left_shin")
    assert "actuator" not in leg  # the robot's own hobby servo holds a small dog
    leg = next(p for p in big["parts"] if p["name"] == "left_shin")
    assert leg["actuator"] in fit.LADDER[1:]  # a joint module, maybe geared
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


def test_a_name_taken_is_not_a_failed_build(pg):
    """A second add_robot with the same name was reported as a builder failure, with advice to change the seed."""
    pg.add_robot("s", describe="a snake")
    with pytest.raises(PlaygroundError, match="already exists") as err:
        pg.add_robot("s", describe="a robot dog")
    assert "builder's checks" not in str(err.value) and "seed" not in str(err.value)


def test_a_failed_build_says_what_to_change(pg, monkeypatch):
    def broken(build, which=fit.is_leg):
        raise ValueError("part 'x': too heavy")

    monkeypatch.setattr(fit, "strain", broken)
    with pytest.raises(PlaygroundError, match="builder's checks: part 'x'.*params=\\{'seed'"):
        pg.add_robot("w", describe="a wheeled robot")


@pytest.mark.parametrize(
    "sentence",
    ["a snake with a suction head", "a tentacle with a magnet", "a wheeled robot with two arms and suction cups"],
)
def test_tools_asked_for_are_built_and_work(pg, sentence):
    result = pg.add_robot("bot", describe=sentence)
    hands = generator.generate(sentence).hands("bot")
    assert hands and all(h in result["generated"]["how"] for h in hands)
    for state in ("close", "open"):
        motion = pg.behaviors.set_gripper(pg.behaviors.resolve(hands[0], arm=True), state)
        pg.run_fast(lambda m=motion: m.done, 5.0)
        assert motion.done


@pytest.mark.parametrize(
    ("sentence", "cube", "target"),
    [
        ("a robot arm with six joints and a suction cup", [0.28, -0.1], "bin"),
        ("a tentacle arm with a gripper", [0.25, 0.1], [0.05, 0.3]),
        ("a six-wheeled mars rover with an arm", [1.5, 0.0], [0.0, 0.8]),
    ],
)
def test_generated_arms_pick_and_place(pg, sentence, cube, target):
    """Arms reach: a standing arm puts a cube in a bin, a rover fetches one 1.5 m away."""
    pg.add_robot("bot", describe=sentence)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=cube, color="red")
    if target == "bin":
        pg.add_object("bin", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.3], color="blue")
    pg.run_fast(lambda: False, 1.0)
    hand = generator.generate(sentence).hands("bot")[0]
    motion, info = pg.behaviors.pick_and_place(pg.behaviors.resolve(hand, arm=True), "cube", target)
    pg.run_fast(lambda: motion.done, 60.0)
    pg.run_fast(lambda: False, 0.5)
    assert pg.behaviors.placed("cube", info["region"])["success"]


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
    # Measured over three seeds each (scripts/gen_bench.py fifteen): snakes 56 to 95 % of the way, the rest 81 %+.
    assert moved > (0.4 if kind == "snake" else 0.5) * speed * 5


@pytest.mark.parametrize("change", ["make it shorter", "make the legs shorter", "make it narrower"])
def test_a_dog_changed_by_a_sentence_still_walks(pg, change):
    """A robot dog made shorter went -0.08 m in 5 s at 0.25 m/s: its hips came closer than two leg lengths."""
    pg.add_robot("dog", describe="a robot dog")
    pg.update_robot("dog", describe=change)
    pg.run_fast(lambda: False, 2.0)
    start = pg.behaviors.pose("dog")
    pg.behaviors.move("dog", 0.25, 0.0)
    pg.run_fast(lambda: False, 5.0)
    end = pg.behaviors.pose("dog")
    assert end["upright"]
    assert end["position"][0] - start["position"][0] > 0.5 * 0.25 * 5
