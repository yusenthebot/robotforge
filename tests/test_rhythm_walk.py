"""A walk a robot learned for itself, kept with its world: move and the keyboard drive the robot with it."""

import math

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from robotforge import modes
from robotforge.behaviors.manager import BehaviorError
from robotforge.behaviors.quadruped import QuadrupedGait
from robotforge.behaviors.rhythm_walk import (
    CALM_MAX,
    RhythmWalker,
    SavedWalk,
    joint_names,
    load_walk,
    save_walk,
    walk_file,
)
from robotforge.control.cpg import FADE, Body, Reward, Rhythm, Trainer
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
    kept_before = {
        "robot": "crab",
        "joints": ["a", "b"],
        "freq": 2.0,
        "theta": [0.0] * 6,
        "speed": 0.2,
        "distance": 0.8,
        "seconds": 4.0,
        "turn_left": [0.0] * 6,
        "turn_right": [0.0] * 6,
        "turn_rate": 1.0,
    }
    old = SavedWalk.model_validate(kept_before)  # format 1 files from before still load
    assert not old.turns_from_walking and old.home is None
    with pytest.raises(ValidationError, match="home pose"):
        _walk("crab", ["a", "b"], home={"a": 0.0, "c": 0.0})


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


def _swinging(pg, **extra) -> RhythmWalker:
    """The starfish with a made-up walk and turns (big swings, all in step), walking mid-stride."""
    pg.add_robot("crab", build=STAR)
    model = pg.session.model
    names = joint_names(model, Body.find(model, "crab"))
    swing = [0.0] * len(names) + [0.5] * len(names) + [0.0] * len(names)
    turn = [0.0] * len(names) + [0.3] * len(names) + [1.0] * len(names)
    save_walk(pg.store.dir, _walk("crab", names, swing, turn_left=turn, turn_right=turn, **extra))
    pg.behaviors.move("crab", 0.2, 0.0)
    walker = pg.session.controllers["crab"]
    pg.run_fast(lambda: False, 0.1)
    assert walker.t0 == pytest.approx(0.0, abs=0.03)  # standing still already: the walk starts at once, as learned
    pg.run_fast(lambda: False, 1.1)  # mid-stride
    return walker


def test_a_turn_out_of_the_walk_starts_at_once_faded_in_from_the_stride(pg):
    """Regression: every switch held home until the robot stood still (1 s at most), since each rhythm was learned
    from standing: a 1 s turn right after walking turned the imported A1 16 degrees of the 88 it promised. Turns
    that also learned to start out of the walk now start at once, faded in from the stride."""
    walker = _swinging(pg, turns_from_walking=True)
    stride = pg.session.data.ctrl[walker.actuators].copy()
    pg.behaviors.stop("crab")  # what the move tool does between commands: the next comes at once
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "left" and walker.t0 is not None and walker.steady_from is None
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(stride, abs=0.05)  # no jump to home
    pg.run_fast(lambda: False, FADE)
    turn = walker.rhythms["left"].targets(walker.home, pg.session.data.time - walker.t0)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(np.clip(turn, walker.lower, walker.upper), abs=0.05)


def test_other_switches_while_moving_steady_on_home_until_calm(pg):
    """Out of a turn, or into one learned only from standing (files kept before turns learned out of the walk), the
    robot steadies on its home pose first, but only until it is calm: CALM_MAX at most, not until it stands still."""
    walker = _swinging(pg)
    spin = walker.body.qvel[5]  # the trunk's yaw rate: 5 rad/s is far from calm
    pg.session.data.qvel[spin] = 5.0
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "left" and walker.t0 is None and walker.steady_from is not None  # steadying on home
    assert pg.run_fast(lambda: walker.t0 is not None, 2 * CALM_MAX) <= CALM_MAX + 0.05  # then it turns
    pg.run_fast(lambda: False, 0.6)
    pg.session.data.qvel[spin] = 5.0
    pg.behaviors.move("crab", 0.2, 0.0)  # out of the turn into the walk
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "forward" and walker.t0 is None
    assert pg.run_fast(lambda: walker.t0 is not None, 2 * CALM_MAX) <= CALM_MAX + 0.05
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0)
    home = np.clip(walker.home, walker.lower, walker.upper)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # stop fades back to the home pose


def test_a_clock_turned_back_mid_switch_starts_the_rhythm_afresh(pg):
    """A data reset under the walker (without SimSession.reset) left the end of a fade in the future: the joints
    would have held the old targets until the clock came round again. It starts the asked rhythm afresh instead."""
    walker = _swinging(pg, turns_from_walking=True)
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.1)  # fading into the turn
    assert walker.fade_from is not None
    model, data = pg.session.model, pg.session.data
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pg.run_fast(lambda: False, 0.04)
    assert walker.playing == "left" and walker.t0 == pytest.approx(0.0) and walker.fade_from is None  # as learned


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
    assert turned.turns_from_walking
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


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_move_reports_a_half_turn_as_turned_not_wrapped(pg):
    """Regression: move reported turned_deg as the difference of the two headings, which wraps at 180: a half
    turn to the left (182 degrees) read as -177.6, a turn to the right."""
    import json as _json

    from mcp import Client

    from robotforge.behaviors.rhythm_walk import learn_turns
    from robotforge.cli import find_free_port
    from robotforge.mcp_server import build_server

    pg.add_robot("crab", build=STAR)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    walk = learn_turns(solo, "crab", _walk("crab", joint_names(solo, Body.find(solo, "crab"))), seed=1)
    save_walk(pg.store.dir, walk)
    seconds = 1.2 * math.pi / walk.turn_rate  # past a half turn at the slower side's rate
    async with Client(build_server(pg, find_free_port(8790))) as client:
        result = await client.call_tool("move", {"robot": "crab", "speed": 0.0, "turn": 0.8, "seconds": seconds})
        turned = _json.loads(result.content[0].text)["turned_deg"]
    assert turned > 180, f"turned {turned} degrees"


def test_a_walk_kept_before_files_named_their_home_plays_an_mjcf_robot_as_it_learned(pg):
    """Regression: an MJCF robot's home keyframe became its home pose, but a walk kept before that was learned around
    the file's zero pose (the keyframe was dropped then): played around the crouch, a critter's walk went 0.88 m in
    4 s instead of 1.25. Such files play around the zero pose; files kept now say what they learned around."""
    from importer_fixtures import CRITTER_HOME, critter_mjcf, write

    write(pg.store.dir / "critter.xml", critter_mjcf())
    pg.add_robot("critter", file="critter.xml")
    model = pg.session.model
    names = joint_names(model, Body.find(model, "critter"))
    hip, knee = CRITTER_HOME
    crouch = [hip if j.startswith("hip") else knee for j in names]
    assert pg.behaviors.home_qpos()[Body.find(model, "critter").qadr] == pytest.approx(crouch)  # the world's home
    save_walk(pg.store.dir, _walk("critter", names))
    pg.behaviors.move("critter", 0.2, 0.0, "rhythm")
    assert pg.session.controllers["critter"].home == pytest.approx(np.zeros(len(names)))
    save_walk(pg.store.dir, _walk("critter", names, home=dict(zip(names, crouch, strict=True))))
    pg.behaviors.move("critter", 0.2, 0.0, "rhythm")
    assert pg.session.controllers["critter"].home == pytest.approx(crouch)
