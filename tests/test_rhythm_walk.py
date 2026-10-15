"""A walk a robot learned for itself, kept with its world: move and the keyboard drive the robot with it."""

import math

import pytest
from pydantic import ValidationError

from robotforge import modes
from robotforge.behaviors.manager import BehaviorError
from robotforge.behaviors.quadruped import QuadrupedGait
from robotforge.behaviors.rhythm_walk import RhythmWalker, SavedWalk, joint_names, load_walk, save_walk, walk_file
from robotforge.control.cpg import Body, Reward, Rhythm, Trainer
from robotforge.design.world import World, compile_world
from robotforge.learn.walk_rl import STAR
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab")
    return playground


def _walk(robot: str, joints: list[str], theta: list[float] | None = None, **extra) -> SavedWalk:
    return SavedWalk(
        robot=robot,
        joints=joints,
        freq=2.0,
        theta=theta if theta is not None else [0.0] * (3 * len(joints)),
        speed=0.2,
        distance=0.8,
        seconds=4.0,
        **extra,
    )


def test_a_saved_walk_checks_its_numbers():
    with pytest.raises(ValidationError, match="expected 6"):
        _walk("crab", ["a", "b"], [0.0] * 5)
    with pytest.raises(ValidationError, match="not finite"):
        _walk("crab", ["a", "b"], [0.0] * 5 + [math.nan])
    with pytest.raises(ValidationError, match="not a robotforge name"):
        _walk("../crab", ["a", "b"])
    assert len(_walk("crab", ["a", "b"], [0.0] * 14, reflexes=True).theta) == 14  # 3 + 4 reflex gains per joint
    with pytest.raises(ValidationError, match="turn rhythm needs 6"):
        _walk("crab", ["a", "b"], turn_left=[0.0] * 4, turn_right=[0.0] * 6)
    assert _walk("crab", ["a", "b"], turn_left=[0.0] * 6, turn_right=[0.0] * 6).turns


def test_a_kept_walk_survives_the_round_trip(tmp_path):
    walk = _walk("crab", ["a", "b"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], survived=9, shoves=12)
    path = save_walk(tmp_path, walk)
    assert path == tmp_path / "walks" / "crab.json"
    assert load_walk(tmp_path, "crab") == walk
    assert load_walk(tmp_path, "other") is None
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="delete it"):
        load_walk(tmp_path, "crab")
    with pytest.raises(ValueError):
        walk_file(tmp_path, "../../etc")


def test_a_robot_no_gait_knows_is_told_how_to_learn_a_walk(pg):
    added = pg.add_robot("crab", build=STAR)
    assert any("crab: no gait code knows this body" in n and "RL II" in n for n in added["notes"])
    assert "RL II" in next(r for r in pg.behaviors.robots() if r["name"] == "crab")["walk_hint"]
    assert pg.behaviors.kind("crab").value == "other"
    with pytest.raises(BehaviorError, match="RL II"):
        pg.behaviors.move("crab", 0.2, 0.0)
    pg.add_robot("arm", template="arm")
    with pytest.raises(BehaviorError) as exc:  # an arm bolted to the floor cannot learn to walk
        pg.behaviors.move("arm", 0.2, 0.0)
    assert "RL II" not in str(exc.value)


def test_a_kept_walk_drives_a_robot_built_from_parts(pg):
    pg.add_robot("crab", build=STAR)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    body = Body.find(solo, "crab")
    rhythm = Rhythm.still(body.joints)
    trainer = Trainer(solo, "crab", Reward(), 4.0, rhythm.freq)
    trainer.train(rhythm, iterations=25, seed=3)
    learned = trainer.rollout(rhythm)["distance"]
    assert learned > 0.2, f"the starfish only learned {learned:.2f} m"
    walk = SavedWalk(
        robot="crab",
        joints=joint_names(solo, body),
        freq=rhythm.freq,
        theta=list(rhythm.theta),
        speed=learned / 4.0,
        distance=learned,
        seconds=4.0,
    )
    save_walk(pg.store.dir, walk)

    listed = next(r for r in pg.behaviors.robots() if r["name"] == "crab")
    assert "walk_hint" not in listed
    assert any("crab walks with the walk it learned" in n for n in pg.reload_world()["notes"])
    assert not listed["learned_walk"]["turns"] and listed["max_speed"] == pytest.approx(learned / 4.0, abs=1e-3)
    assert modes.available(pg, listed, learned=False)[0] == "drive"  # the keyboard can drive it now

    start = pg.behaviors.pose("crab")
    status = pg.behaviors.move("crab", 0.2, 0.5)
    assert status["controller"] == "learned rhythm" and "not learned to turn" in status["note"]
    assert modes.current(pg)["crab"] == "drive"
    pg.run_fast(lambda: False, 4.0)
    end = pg.behaviors.pose("crab")
    gone = end["position"][0] - start["position"][0]
    assert gone > 0.5 * learned, f"learned {learned:.2f} m in 4 s alone, walked {gone:.2f} m in the world"
    assert end["upright"]
    assert pg.behaviors.stop("crab")["stopped"]
    assert not pg.session.controllers["crab"].walking  # holds its home pose


def test_a_walk_for_other_joints_asks_to_learn_again(pg):
    pg.add_robot("crab", build=STAR)
    save_walk(pg.store.dir, _walk("crab", ["a", "b"]))
    with pytest.raises(BehaviorError, match="learn the walk again"):
        pg.behaviors.move("crab", 0.2, 0.0)


def test_a_damaged_walk_file_is_named_when_moving_but_does_not_break_the_list(pg):
    pg.add_robot("crab", build=STAR)
    path = walk_file(pg.store.dir, "crab")
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(BehaviorError, match="delete it"):
        pg.behaviors.move("crab", 0.2, 0.0)
    assert "learned_walk" not in next(r for r in pg.behaviors.robots() if r["name"] == "crab")


def test_a_dog_keeps_its_gait_and_offers_its_learned_walk_as_a_mode(pg):
    pg.add_robot("dog", template="quadruped")
    body = Body.find(pg.session.model, "dog")
    save_walk(pg.store.dir, _walk("dog", joint_names(pg.session.model, body)))
    pg.behaviors.move("dog", 0.2, 0.0)
    assert isinstance(pg.session.controllers["dog"], QuadrupedGait)  # the gait still comes first: it turns
    listed = next(r for r in pg.behaviors.robots() if r["name"] == "dog")
    assert "rhythm" in modes.available(pg, listed, learned=False)
    modes.apply(pg, "dog", "rhythm")
    assert isinstance(pg.session.controllers["dog"], RhythmWalker)
    assert modes.current(pg)["dog"] == "rhythm"
    pg.behaviors.move("dog", 0.3, 0.0)  # the keyboard keeps whichever walker was picked
    assert isinstance(pg.session.controllers["dog"], RhythmWalker)
    modes.apply(pg, "dog", "drive")
    assert isinstance(pg.session.controllers["dog"], QuadrupedGait)


def test_a_kept_walk_learns_to_turn_and_move_turns_the_robot(pg):
    """Turning on the spot is learned after the walk (one rhythm each way); a turn command then plays it."""
    from robotforge.behaviors.rhythm_walk import learn_turns

    pg.add_robot("crab", build=STAR)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    body = Body.find(solo, "crab")
    walk = _walk("crab", joint_names(solo, body))
    turned = learn_turns(solo, "crab", walk, seed=1)
    assert turned.turns and turned.turn_rate > 0.5  # rad/s; the starfish learns about 1 in 30 rounds
    save_walk(pg.store.dir, turned)
    assert next(r for r in pg.behaviors.robots() if r["name"] == "crab")["learned_walk"]["turns"]

    def heading_after(turn: float) -> float:
        before = pg.behaviors.pose("crab")["heading_deg"]
        status = pg.behaviors.move("crab", 0.0, turn)
        assert status["turning"] == ("left" if turn > 0 else "right")
        pg.run_fast(lambda: False, 2.0)  # under half a turn, so the wrapped heading keeps its sign
        pg.behaviors.stop("crab")
        pg.run_fast(lambda: False, 0.5)
        return (pg.behaviors.pose("crab")["heading_deg"] - before + 540) % 360 - 180

    assert heading_after(0.8) > 40  # degrees, to the left
    assert heading_after(-0.8) < -40
    assert pg.behaviors.pose("crab")["upright"]
