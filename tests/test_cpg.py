"""Learning to walk by rhythm and random search (control/cpg.py)."""

import math

import mujoco
import numpy as np
import pytest

from robotforge.control.cpg import (
    CONTROL_HZ,
    DRIFT_WEIGHT,
    FADE,
    FALL_PENALTY,
    HEADING_WEIGHT,
    STEER_SPAN,
    WANDER_WEIGHT,
    Body,
    CPGError,
    Push,
    Reward,
    Rhythm,
    Start,
    Steering,
    Trainer,
    fade_in,
)
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


def test_a_forward_walk_counts_less_the_more_its_heading_turned():
    """An imported A1 learned a walk of 1.46 m in 4 s that turned 42 degrees: forward was an arc. The distance now
    counts for less the more the heading turned (HEADING_WEIGHT of it per radian)."""
    model = _solo("dog", template="quadruped")
    trainer = Trainer(model, "dog", Reward(), seconds=2.0)
    rhythm = Rhythm.still(trainer.body.joints)
    trainer.train(rhythm, iterations=4, seed=1)
    walk = trainer.rollout(rhythm)
    assert not walk["fell"] and walk["distance"] > 0.1 and abs(walk["turned"]) > 0.01
    straight = walk["distance"] - DRIFT_WEIGHT * walk["drift"]
    assert walk["reward"] == pytest.approx(straight - HEADING_WEIGHT * abs(walk["turned"]) * walk["distance"])
    # Never less than nothing: past 1 / weight rad of turning, walking further would have scored worse.
    harsh = Trainer(model, "dog", Reward(heading=2.0 / abs(walk["turned"])), seconds=2.0).rollout(rhythm)
    assert harsh["reward"] == pytest.approx(-DRIFT_WEIGHT * walk["drift"])


def test_a_steering_moves_the_walk_by_its_gains_as_far_as_the_heading_is_off():
    model = _solo("star", build=STAR)
    trainer = Trainer(model, "star", Reward(), seconds=1.0)
    n = trainer.body.joints
    walk = Rhythm(np.concatenate([np.zeros(n), np.full(n, 0.3), np.zeros(n)]))
    steering = Steering(walk, np.concatenate([np.full(n, 0.1), np.zeros(2 * n)]))  # offsets only
    home, t = trainer.home, 0.3
    assert steering.targets(home, t) == pytest.approx(walk.targets(home, t))  # on course: the walk as learned
    assert steering.targets(home, t, error=STEER_SPAN / 2) == pytest.approx(walk.targets(home, t) + 0.05)
    assert steering.targets(home, t, error=-3.0) == pytest.approx(walk.targets(home, t) - 0.1)  # at most its gains
    assert walk.targets(home, t, error=1.0) == pytest.approx(walk.targets(home, t))  # a rhythm does not steer


def test_holding_a_heading_scores_the_mean_over_its_aims_measured_along_each():
    model = _solo("dog", template="quadruped")
    trainer = Trainer(model, "dog", Reward(), seconds=2.0)
    rhythm = Rhythm.still(trainer.body.joints)
    trainer.train(rhythm, iterations=4, seed=1)
    ahead = trainer.rollout(rhythm)
    aside = trainer.rollout(rhythm, aim=math.pi / 2)  # the same walk aiming to the left: forward is across the aim
    assert abs(aside["distance"]) == pytest.approx(ahead["drift"])
    assert aside["drift"] == pytest.approx(ahead["distance"])
    hold = Trainer(model, "dog", Reward(hold=1.0, aims=(0.3, -0.3)), seconds=2.0)
    steering = Steering.still(rhythm, trainer.body.joints)
    tries = [hold.rollout(steering, aim=a) for a in (0.3, -0.3)]
    assert hold._score(steering, steering.theta, None, None) == pytest.approx(np.mean([t["reward"] for t in tries]))
    for t in tries:  # progress along the aim, less the share it strayed from it on average
        assert t["stray"] > 0.1 and t["reward"] == pytest.approx(
            t["distance"] - min(t["stray"], 1.0) * max(t["distance"], 0.0) - DRIFT_WEIGHT * t["drift"]
        )


def test_a_rhythm_can_start_mid_stride_faded_in_from_the_targets_of_the_moment():
    model = _solo("star", build=STAR)
    trainer = Trainer(model, "star", Reward(), seconds=1.0)
    n = trainer.body.joints
    swing = Rhythm(np.concatenate([np.zeros(n), np.full(n, 0.5), np.zeros(n)]))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for k in range(round(0.5 * CONTROL_HZ)):  # half a second of the swing
        data.ctrl[trainer.body.actuators] = swing.targets(trainer.home, k / CONTROL_HZ)
        mujoco.mj_step(model, data, trainer.substeps)
    moments = [Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())]
    assert np.abs(moments[0].qvel).max() > 0.1  # mid-stride: moving
    a, b = np.zeros(3), np.ones(3)
    assert fade_in(a, b, 0.0) == pytest.approx(a) and fade_in(a, b, FADE) == pytest.approx(b)
    assert fade_in(a, b, FADE / 2) == pytest.approx(0.5 * b)
    still = Rhythm.still(n)
    from_stride = trainer.rollout(still, start=moments[0])
    assert from_stride["energy"] > 2 * trainer.rollout(still)["energy"]  # it had a stride to stop


def test_a_fall_forfeits_a_turn_begun_mid_stride_but_not_one_from_standing():
    """Regression: a fall cost one point while a fast spin earned up to 14 rad in 4 s, so turns that spun the
    imported A1 onto its back when started out of its walk still scored best. From standing the reward is what it
    always was, so the turns learned from standing do not change (forfeiting there too slowed the A1's)."""
    model = _solo("dog", template="quadruped")
    trainer = Trainer(model, "dog", Reward(turn=1.0), seconds=2.0)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    pos, vel = trainer.body.qpos, trainer.body.qvel
    data.qpos[pos[2]] += 0.1
    data.qpos[pos[3:7]] = [math.cos(math.radians(20)), math.sin(math.radians(20)), 0.0, 0.0]  # leaning 40 degrees
    data.qvel[vel[5]] = 8.0  # spinning to the left
    data.qvel[vel[3]] = 2.0  # and rolling over
    toppling = Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    spun = trainer.rollout(Rhythm.still(trainer.body.joints), start=toppling)
    assert spun["fell"] and spun["turned"] > 1.0  # rad: it spun a long way before it went down
    assert spun["reward"] == pytest.approx(-FALL_PENALTY - WANDER_WEIGHT * spun["wander"])
    model.key_qpos[0], model.key_qvel[0] = toppling.qpos, toppling.qvel  # the same, as the pose it stands in
    standing = trainer.rollout(Rhythm.still(trainer.body.joints))
    assert standing["fell"] and standing["turned"] > 1.0
    assert standing["reward"] == pytest.approx(standing["turned"] - FALL_PENALTY - WANDER_WEIGHT * standing["wander"])
    forfeit = Trainer(model, "dog", Reward(turn=1.0, forfeit=True), seconds=2.0)  # asked to, from standing too
    assert forfeit.rollout(Rhythm.still(trainer.body.joints))["reward"] == pytest.approx(
        -FALL_PENALTY - WANDER_WEIGHT * standing["wander"]
    )


def test_a_turn_can_be_rewarded_for_its_first_moments_only():
    """Turns tuned to begin out of the walk were slow to get going when rewarded for all 4 s: rewarded for their
    first second only (a fall in any of the 4 s still losing it), more of them reached 60% of their rate in it."""
    model = _solo("dog", template="quadruped")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    spinning = Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    spinning.qvel[Body.find(model, "dog").qvel[5]] = 4.0  # rad/s of yaw, slowing down on its feet
    still = Rhythm.still(Body.find(model, "dog").joints)
    short = Trainer(model, "dog", Reward(turn=1.0), seconds=0.5).rollout(still, start=spinning)
    whole = Trainer(model, "dog", Reward(turn=1.0), seconds=2.0).rollout(still, start=spinning)
    early = Trainer(model, "dog", Reward(turn=1.0, early=0.5), seconds=2.0).rollout(still, start=spinning)
    assert not whole["fell"] and short["turned"] > 0.1 and abs(whole["turned"] - short["turned"]) > 0.001
    assert early["reward"] == pytest.approx(short["turned"] - WANDER_WEIGHT * whole["wander"])
