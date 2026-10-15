"""Learning to walk by rhythm and random search (control/cpg.py)."""

import numpy as np
import pytest

from robotforge.control.cpg import Body, CPGError, Push, Reward, Rhythm, Trainer
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


def test_reflexes_turn_body_tilt_into_joint_corrections_and_shoves_land():
    model = _solo("dog", template="quadruped")
    trainer = Trainer(model, "dog", Reward(survive=True), seconds=2.0)
    n = trainer.body.joints
    rhythm = Rhythm.still(n, reflexes=True)
    assert len(rhythm.theta) == 7 * n
    rhythm.theta[3 * n :] = 0.1  # each joint adds 0.1 rad per unit of each sense
    tilted = rhythm.targets(trainer.home, 0.0, senses=np.array([0.2, 0.0, 0.0, 0.0]))
    assert tilted == pytest.approx(trainer.home + 0.02)
    still = Rhythm.still(n)
    calm = trainer.rollout(still)
    shoved = trainer.rollout(still, push=Push(0.5, (0.0, 1.5)))
    assert shoved["drift"] > calm["drift"] + 0.05  # the shove really moved it sideways
    result = trainer.push_test(still, 0.4, trials=3)
    assert result["trials"] == 3 and 0 <= result["survived"] <= 3


def test_an_imported_robot_drives_its_unlimited_joints_but_not_its_wheels(tmp_path, monkeypatch):
    """A URDF joint without limits imports unlimited, its servo commanded within half a turn either way: it is a
    servo joint like any other, bounded by that. Wheels get speed servos with bounded commands too, but a speed
    is no angle to aim at, so they are not driven."""
    from importer_fixtures import critter_urdf, rover_urdf, write

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    model = _solo("critter", file=str(write(tmp_path / "critter.urdf", critter_urdf())))
    body = Body.find(model, "critter")
    hips = [i for i, a in enumerate(body.actuators) if "hip" in model.actuator(int(a)).name]
    assert body.joints == 8 and len(hips) == 4
    assert not model.jnt_limited[model.joint("critter/hip0").id]
    assert body.lower[hips] == pytest.approx(-np.pi) and body.upper[hips] == pytest.approx(np.pi)
    with pytest.raises(CPGError, match="0 servo joints"):
        Body.find(_solo("rover", file=str(write(tmp_path / "rover.urdf", rover_urdf()))), "rover")
