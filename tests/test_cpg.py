"""Learning to walk by rhythm and random search (control/cpg.py)."""

import pytest

from robotforge.control.cpg import Body, CPGError, Reward, Rhythm, Trainer
from robotforge.design.world import World, compile_world
from robotforge.learn.walk_rl import STAR


def _solo(name: str, **entry) -> object:
    return compile_world(World.model_validate({"name": "solo", "robots": [{"name": name, **entry}]})).model


def test_a_still_rhythm_stands_and_a_short_search_walks_forward():
    model = _solo("dog", template="quadruped")
    trainer = Trainer(model, "dog", Reward(), seconds=2.0)
    rhythm = Rhythm.still(trainer.body.joints)
    standing = trainer.rollout(rhythm)
    assert abs(standing["distance"]) < 0.02 and not standing["fell"]
    log = trainer.train(rhythm, iterations=8, seed=1)
    assert log.attempts == 8 * 17 and len(log.distances) == 8
    assert trainer.rollout(rhythm)["distance"] > 0.2  # a few rounds already get it moving


def test_any_body_with_servo_joints_can_learn_but_a_fixed_arm_cannot():
    star = Body.find(_solo("star", build=STAR), "star")
    assert star.joints == 6 and len(star.qpos) == 7 + 6
    with pytest.raises(CPGError, match="fixed to the world"):
        Body.find(_solo("arm", template="arm"), "arm")


def test_an_energy_price_makes_the_search_save_energy():
    model = _solo("dog", template="quadruped")
    free = Trainer(model, "dog", Reward(energy=0.0), seconds=2.0)
    thrifty = Trainer(model, "dog", Reward(energy=0.3), seconds=2.0)
    rhythm = Rhythm.still(free.body.joints)
    free.train(rhythm, iterations=6, seed=2)
    walk = free.rollout(rhythm)
    assert thrifty.rollout(rhythm)["reward"] < walk["reward"]  # the same walk scores less when energy costs
