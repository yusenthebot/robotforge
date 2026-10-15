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
    SETTLE_MAX,
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
    assert old.home is None and old.steer is None and not old.turns_from_walking
    with pytest.raises(ValidationError, match="steering needs 6"):
        _walk("crab", ["a", "b"], steer=[0.0] * 3)
    with pytest.raises(ValidationError, match="home pose"):
        _walk("crab", ["a", "b"], home={"a": 0.0, "c": 0.0})
    both = _walk("crab", ["a", "b"], turn_left=[0.0] * 6, turn_right=[0.0] * 6)
    walking = both.model_copy(
        update={
            "turn_left_walking": [0.0] * 6,
            "turn_right_walking": [0.0] * 6,
            "turn_left_walking_at": [0.0, 0.4],
            "switching": "fade",
        }
    )
    assert walking.turns_from_walking and SavedWalk.model_validate(walking.model_dump()) == walking
    assert old.switching is None and old.turn_left_walking_at is None  # switches as walks kept before did
    for bad in ([], [1.0], [math.nan]):
        with pytest.raises(ValidationError, match="stride phases"):
            _walk("crab", ["a", "b"], turn_left_walking_at=bad)
    with pytest.raises(ValidationError):
        _walk("crab", ["a", "b"], switching="snap")


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


def _swinging(pg, walking: bool = False, kept_before: bool = False, **more) -> RhythmWalker:
    """The starfish with a made-up walk and turns (big swings, all in step; the turns' walking versions swing the
    other way round, if `walking`), walking mid-stride. A walk kept now names its home pose and how it switches;
    `kept_before` leaves both out, as files kept before did."""
    pg.add_robot("crab", build=STAR)
    model = pg.session.model
    body = Body.find(model, "crab")
    names = joint_names(model, body)
    swing = [0.0] * len(names) + [0.5] * len(names) + [0.0] * len(names)
    turn = [0.0] * len(names) + [0.3] * len(names) + [1.0] * len(names)
    extra: dict = more
    if not kept_before:
        extra |= {"home": dict(zip(names, pg.behaviors.home_qpos()[body.qadr], strict=True)), "switching": "fade"}
    if walking:
        out_of_walk = [0.0] * len(names) + [0.3] * len(names) + [-1.0] * len(names)
        extra |= {"turn_left_walking": out_of_walk, "turn_right_walking": out_of_walk}
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
    walker = _swinging(pg, walking=True)
    stride = pg.session.data.ctrl[walker.actuators].copy()
    pg.behaviors.stop("crab")  # what the move tool does between commands: the next comes at once
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "left" and walker.rhythm == "left walking" and walker.steady_from is None
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(stride, abs=0.05)  # no jump to home
    pg.run_fast(lambda: False, FADE)
    turn = walker.rhythms["left walking"].targets(walker.home, walker.clock - walker.t0)  # as of its last update
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(np.clip(turn, walker.lower, walker.upper), abs=0.05)


def test_a_turn_that_fell_from_some_points_of_the_stride_waits_for_one_it_did_not(pg):
    """Regression: a turn that tipped the template dog over from 2 of 21 moments of its walk was not kept for
    starting out of the walk at all, so the robot steadied first and a 1 s turn straight after walking turned 14% of
    its rate. It now walks on to the next point of the stride it never fell from (less than a stride) and turns
    there."""
    walker = _swinging(pg, walking=True, turn_left_walking_at=[0.3])
    assert walker.walking_at == {"left": [0.3], "right": None}
    walked_from, freq = walker.t0, walker.walk.freq
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.playing == "left" and walker.rhythm == "forward" and walker.turn_at is not None  # walking on
    pg.behaviors.move("crab", 0.2, 0.0)  # let go before it began: it walks on, as it was
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.rhythm == "forward" and walker.t0 == walked_from and walker.turn_at is None
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    asked = walker.clock
    phase = ((asked - walked_from) * freq) % 1.0
    assert walker.turn_at - asked == pytest.approx(((0.3 - phase) % 1.0) / freq, abs=0.011)
    began = pg.run_fast(lambda: walker.rhythm == "left walking", 1.0)
    assert began < 1.0 / freq + 0.05  # within a stride
    assert ((walker.t0 - walked_from) * freq) % 1.0 == pytest.approx(0.3, abs=0.03)  # at the point it was safe from
    assert walker.fade_from is not None  # faded in from the stride


def test_a_turn_checked_safe_from_every_point_of_the_stride_begins_at_once(pg):
    walker = _swinging(pg, walking=True, turn_left_walking_at=[0.3])
    pg.behaviors.move("crab", 0.0, -0.8)  # the right turn may begin anywhere
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.rhythm == "right walking" and walker.turn_at is None


def test_other_switches_while_moving_steady_on_home_until_calm(pg):
    """Out of a turn, or into one learned only from standing (files kept before turns learned out of the walk), the
    robot steadies on its home pose first, but only until it is calm: CALM_MAX at most, not until it stands still."""
    walker = _swinging(pg)
    assert "left walking" not in walker.rhythms
    spin = walker.body.qvel[5]  # the trunk's yaw rate: 5 rad/s is far from calm
    pg.session.data.qvel[spin] = 5.0
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "left" and walker.rhythm is None and walker.steady_from is not None  # steadying on home
    assert pg.run_fast(lambda: walker.t0 is not None, 2 * CALM_MAX) <= CALM_MAX + 0.05  # then it turns
    assert walker.rhythm == "left"
    pg.run_fast(lambda: False, 0.6)
    pg.session.data.qvel[spin] = 5.0
    pg.behaviors.move("crab", 0.2, 0.0)  # out of the turn into the walk
    pg.run_fast(lambda: False, 0.02)
    assert walker.playing == "forward" and walker.t0 is None
    assert pg.run_fast(lambda: walker.t0 is not None, 2 * CALM_MAX) <= CALM_MAX + 0.05
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0)
    home = np.clip(walker.home, walker.lower, walker.upper)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # stop holds the home pose


def test_a_turn_checked_to_need_it_is_left_by_holding_home_at_once(pg):
    """Regression: fading to home out of a template dog walk's fast right turn tipped it over in 3 of 40 stops
    (holding home at once, as before: none). How to leave each turn is checked after it is learned and kept."""
    walker = _swinging(pg, turn_left_exit="hold", turn_right_exit="fade")
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: walker.rhythm == "left", CALM_MAX + 0.1)
    pg.run_fast(lambda: False, 0.5)  # turning
    pg.session.data.qvel[walker.body.qvel[5]] = 1.0  # rad/s of yaw: calm, but not still
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    home = np.clip(walker.home, walker.lower, walker.upper)
    assert walker.holding and pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # home at once
    pg.behaviors.move("crab", 0.0, -0.8)  # asked while it holds: it holds until still, then turns as learned
    waited = pg.run_fast(lambda: walker.t0 is not None, 2 * SETTLE_MAX)
    assert waited < SETTLE_MAX + 0.05 and walker.rhythm == "right" and walker.fade_from is None
    pg.run_fast(lambda: False, 0.5)
    pg.session.data.qvel[walker.body.qvel[5]] = 1.0
    pg.behaviors.stop("crab")  # the right turn fades
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert not walker.holding and walker.fade_from is not None


def test_turn_exits_keep_the_way_that_tips_the_robot_over_less(monkeypatch):
    from robotforge.behaviors import rhythm_walk

    tried = []

    def falls(model, robot, walk, turn, lead, walked):
        style = walk.turn_left_exit if turn > 0 else walk.turn_right_exit
        tried.append((turn, style, walked))
        return turn < 0 and style == "fade" and lead > 2.0  # fading out of the right turn tips it, late in it

    monkeypatch.setattr(rhythm_walk, "_falls_stopping", falls)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    names = joint_names(solo, Body.find(solo, "crab"))
    turns = {"turn_left": [0.0] * 3 * len(names), "turn_right": [0.0] * 3 * len(names), "turn_rate": 1.0}
    kept = rhythm_walk.learn_turn_exits(solo, "crab", _walk("crab", names, **turns))
    assert kept.turn_left_exit == "fade" and kept.turn_right_exit == "hold"  # a tie fades
    assert len(tried) == 2 * 2 * 2 * len(rhythm_walk.EXIT_LEADS)  # each side, each way, from standing and walking
    assert rhythm_walk.learn_turn_exits(solo, "crab", _walk("crab", names)).turn_left_exit is None  # no turns yet


def test_a_walk_kept_before_switches_as_it_always_did(pg):
    """Regression: with faster switching (stop fading to home, rhythms starting once the robot is calm), the walk the
    imported A1 kept first ended on its back when the sim ran on between move calls, as it does with the viewer
    open: in 7 of 8 waits from 0 to 5 s; switched as before, in 1. A file without a home pose, kept before, stops at
    once on home and waits on it until the robot stands still."""
    walker = _swinging(pg, kept_before=True)
    assert walker.kept_before
    pg.session.data.qvel[walker.body.qvel[5]] = 1.0  # rad/s of yaw: calm, but not still
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    home = np.clip(walker.home, walker.lower, walker.upper)
    assert walker.rhythm is None and pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # home at once
    waited = pg.run_fast(lambda: walker.t0 is not None, 2 * SETTLE_MAX)
    assert 3 / walker.rate_hz < waited < SETTLE_MAX and walker.fade_from is None  # until still, then as learned
    pg.run_fast(lambda: False, 1.0)  # turning
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # stop: home at once


def test_stop_fades_a_walk_kept_now_to_home(pg):
    """In 810 tries of each switch on its own (6 A1 walks), stopping a walk or a turn by holding home at once tipped
    the A1 37 times, fading to home never."""
    walker = _swinging(pg, walking=True)
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0)  # turning
    turning = pg.session.data.ctrl[walker.actuators].copy()
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(turning, abs=0.1)  # no jump
    pg.run_fast(lambda: False, FADE)
    assert pg.session.data.ctrl[walker.actuators] == pytest.approx(np.clip(walker.home, walker.lower, walker.upper))


def test_a_clock_turned_back_mid_switch_starts_the_rhythm_afresh(pg):
    """A data reset under the walker (without SimSession.reset) left the end of a fade in the future: the joints
    would have held the old targets until the clock came round again. It starts the asked rhythm afresh instead."""
    walker = _swinging(pg, walking=True)
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.1)  # fading into the turn
    assert walker.fade_from is not None
    model, data = pg.session.model, pg.session.data
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    pg.run_fast(lambda: False, 0.04)
    assert walker.rhythm == "left" and walker.t0 == pytest.approx(0.0) and walker.fade_from is None  # as learned


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


def test_a_walking_turn_begins_only_at_points_of_the_stride_it_fell_from_at_no_moment():
    """Turns tuned to begin out of the walk spin faster, but some tipped robots over from moments they were not tuned
    on. Each is checked from moments of the walk; the points of the stride it fell from are left out (at 2 Hz a
    moment 0.6 s in is 0.2 of a stride along), and one that fell from every point is not kept."""
    from robotforge.behaviors.rhythm_walk import _mean_wait, _walking_turn
    from robotforge.control.cpg import Start

    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "dog", "template": "quadruped"}]}))
    model = solo.model
    check = Trainer(model, "dog", Reward(turn=1.0), 2.0)
    first = Trainer(model, "dog", Reward(turn=1.0), 1.0)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    standing = Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    pos, vel = check.body.qpos, check.body.qvel
    data.qpos[pos[2]] += 0.1
    data.qpos[pos[3:7]] = [math.cos(math.radians(20)), math.sin(math.radians(20)), 0.0, 0.0]  # leaning 40 degrees
    data.qvel[vel[3]], data.qvel[vel[5]] = 2.0, 8.0  # rolling over and spinning: nothing it does keeps it up
    toppling = Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    n = check.body.joints
    swing = Rhythm(np.concatenate([np.zeros(n), np.full(n, 0.3), np.zeros(n)]))
    times = (0.5, 0.6, 1.0, 0.7)  # stride points 0, 0.2, 0 again and 0.4
    assert _walking_turn((swing,), 1.0, check, first, [standing] * 4, times)[1] is None  # anywhere
    assert _walking_turn((swing,), 1.0, check, first, [standing, toppling, standing, standing], times)[1] == [0.0, 0.4]
    assert _walking_turn((swing,), 1.0, check, first, [standing, standing, toppling, standing], times)[1] == [0.2, 0.4]
    assert _walking_turn((swing,), 1.0, check, first, [toppling, standing, standing, toppling], times) is None  # 1 of 3
    assert _walking_turn((swing,), 1.0, check, first, [toppling] * 4, times) is None
    assert _mean_wait([0.0]) == pytest.approx(0.5)  # half a stride on average
    assert _mean_wait([0.0, 0.5]) == pytest.approx(0.25)


def test_a_walk_that_falls_before_its_first_moment_keeps_no_walking_turns(monkeypatch):
    """Regression: the file said its turns were learned out of the walk even when the walk gave no moment to begin
    from (it fell first), so they would have started mid-stride untrained."""
    from robotforge.behaviors import rhythm_walk

    monkeypatch.setattr(rhythm_walk, "TURN_ROUNDS", 2)
    monkeypatch.setattr(rhythm_walk, "TURN_STARTS", 1)
    monkeypatch.setattr(rhythm_walk, "walk_moments", lambda model, robot, walk, times: [])  # down before 0.5 s
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    turned = rhythm_walk.learn_turns(solo, "crab", _walk("crab", joint_names(solo, Body.find(solo, "crab"))), seed=1)
    assert turned.turns and turned.turn_left_walking is None and turned.turn_right_walking is None


def test_a_turn_from_standing_that_ends_on_its_back_is_learned_again_with_the_fall_forfeiting_it(monkeypatch):
    """Regression: a fall cost a point while a fast spin earned many, so the imported A1 kept a left turn that put it
    on its back from standing in all of 15 tries (1 of 6 walks). Such a side learns again, a fall forfeiting the
    turn; the others learn as they always did."""
    from robotforge.behaviors import rhythm_walk

    calls = []

    def learned(model, robot, joints, reward, seed):
        calls.append((reward.turn, reward.forfeit))
        fell = reward.turn > 0 and not reward.forfeit  # the left turn falls until the fall forfeits it
        return Rhythm.still(joints.body.joints), {"fell": float(fell), "turned": reward.turn * (6.0 if fell else 4.0)}

    monkeypatch.setattr(rhythm_walk, "_standing_turn", learned)
    monkeypatch.setattr(rhythm_walk, "learn_walking_turns", lambda model, robot, walk, seed=0: walk)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    turned = rhythm_walk.learn_turns(solo, "crab", _walk("crab", joint_names(solo, Body.find(solo, "crab"))))
    assert calls == [(1.0, False), (1.0, True), (-1.0, False)]
    assert turned.turn_rate == pytest.approx(4.0 / 4.0)  # rad/s, from the turn kept


def test_a_kept_walk_that_curves_learns_to_hold_its_heading(pg, monkeypatch):
    """The reward's heading term straightened most learned walks, but not the imported A1's: open loop they turned
    42 degrees in 4 s on average. A kept walk learns to steer back toward where it was asked to go, kept if that
    holds its heading better over 20 s played as move plays it."""
    from robotforge.behaviors import rhythm_walk

    monkeypatch.setattr(rhythm_walk, "STEER_ROUNDS", 8)  # a short search is enough for this starfish walk
    pg.add_robot("crab", build=STAR)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    trainer = Trainer(solo, "crab", Reward(heading=0.0), 4.0)
    rhythm = Rhythm.still(trainer.body.joints)
    trainer.train(rhythm, iterations=8, seed=3)
    walk = _walk("crab", joint_names(solo, trainer.body), [float(v) for v in rhythm.theta])
    assert math.degrees(rhythm_walk.heading_drift(solo, "crab", walk)) > 30  # open loop it curves to the left
    held = rhythm_walk.learn_steering(solo, "crab", walk, seed=1)
    assert held.steer is not None and held.describe()["holds_heading"]
    save_walk(pg.store.dir, held)

    def heading_change(seconds: float) -> float:
        before = pg.behaviors.pose("crab")["heading_deg"]
        pg.behaviors.move("crab", 0.2, 0.0)
        pg.run_fast(lambda: False, seconds)
        return (pg.behaviors.pose("crab")["heading_deg"] - before + 540) % 360 - 180

    assert abs(heading_change(rhythm_walk.STEER_CHECK)) < 15  # degrees, over 20 s
    still = _walk("crab", walk.joints)  # standing still cannot steer: nothing is kept
    assert rhythm_walk.learn_steering(solo, "crab", still, seed=1) == still


def test_a_walk_thrown_off_course_steers_back_to_the_heading_it_started_with(pg):
    """Regression: the aim followed a robot more than 17 degrees off for good, and steering came in over the walk's
    first 2 s, so a walk that drifted early held the new heading: an A1 walk went -35 degrees in 8 s and stayed
    there. The aim stays within STEER_LEASH of the robot but comes back to the course by STEER_RELAX."""
    from robotforge.behaviors import rhythm_walk

    walker = _swinging(pg)
    model, data = pg.session.model, pg.session.data

    def face(yaw: float) -> None:
        data.qpos[walker.body.qpos[3:7]] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_kinematics(model, data)

    walker.course = walker.aim = 0.0
    face(-1.0)  # spun a long way to the right
    assert walker._heading_error(data, 10.0, 0.02) == pytest.approx(rhythm_walk.STEER_SPAN)  # corrects fully
    assert walker.aim == pytest.approx(-1.0 + rhythm_walk.STEER_LEASH)  # its aim no further off than the leash
    face(-0.1)  # back near the course: the aim is never further than the leash either way, then comes back
    for _ in range(50):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(-0.1 - rhythm_walk.STEER_LEASH + rhythm_walk.STEER_RELAX, abs=0.003)
    for _ in range(500):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(0.0)  # all the way back
    assert walker._heading_error(data, 1.0, 0.02) == pytest.approx(0.1 / rhythm_walk.STEER_RAMP)  # easing in


def test_a_walk_asked_for_out_of_a_turn_holds_the_heading_it_was_asked_at(pg):
    """Regression: the walk held the heading the robot faced once it had steadied, so an imported A1 coming out of a
    fast turn walked off 47 degrees further round than where it was when asked to walk."""
    walker = _swinging(pg)
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: walker.rhythm == "left", CALM_MAX + 0.1)
    pg.run_fast(lambda: False, 0.5)
    from robotforge.control.cpg import trunk_yaw

    pg.session.data.qvel[walker.body.qvel[5]] = 12.0  # still spinning hard to the left
    asked = trunk_yaw(pg.session.data, walker.body)
    pg.behaviors.move("crab", 0.2, 0.0)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.rhythm is None  # steadying
    pg.run_fast(lambda: walker.t0 is not None, 2 * CALM_MAX)
    started = trunk_yaw(pg.session.data, walker.body)
    assert walker.course == pytest.approx(asked, abs=0.05)  # rad: where it was asked, not where it steadied
    assert abs(started - asked) > 0.15  # it spun on while it steadied


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
async def test_move_reports_a_half_turn_as_turned_not_wrapped(pg, monkeypatch):
    """Regression: move reported turned_deg as the difference of the two headings, which wraps at 180: a half
    turn to the left (182 degrees) read as -177.6, a turn to the right."""
    import json as _json

    from mcp import Client

    from robotforge.behaviors import rhythm_walk
    from robotforge.behaviors.rhythm_walk import learn_turns

    monkeypatch.setattr(rhythm_walk, "WALKING_ROUNDS", 2)  # it turns from standing here
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
