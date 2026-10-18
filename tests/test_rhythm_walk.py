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
    STOP_GRACE,
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


def _swinging(pg, walking: bool = False, kept_before: bool = False, walked: float = 1.7, **more) -> RhythmWalker:
    """The starfish with a made-up walk and turns (big swings, all in step; the turns' walking versions swing the
    other way round, if `walking`), walking mid-stride, `walked` seconds into the walk. A walk kept now names its home
    pose and how it switches; `kept_before` leaves both out, as files kept before did."""
    pg.add_robot("crab", build=STAR)
    model = pg.session.model
    body = Body.find(model, "crab")
    names = joint_names(model, body)
    swing = [0.0] * len(names) + [0.5] * len(names) + [0.0] * len(names)
    turn = [0.0] * len(names) + [0.3] * len(names) + [1.0] * len(names)
    extra: dict = {}
    if not kept_before:
        extra |= {"home": dict(zip(names, pg.behaviors.home_qpos()[body.qadr], strict=True)), "switching": "fade"}
    if walking:
        out_of_walk = [0.0] * len(names) + [0.3] * len(names) + [-1.0] * len(names)
        extra |= {"turn_left_walking": out_of_walk, "turn_right_walking": out_of_walk}
    extra |= more
    save_walk(pg.store.dir, _walk("crab", names, swing, turn_left=turn, turn_right=turn, **extra))
    pg.behaviors.move("crab", 0.2, 0.0)
    walker = pg.session.controllers["crab"]
    pg.run_fast(lambda: False, 0.1)
    assert walker.t0 == pytest.approx(0.0, abs=0.03)  # standing still already: the walk starts at once, as learned
    pg.run_fast(lambda: False, walked - 0.1)  # mid-stride
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


def test_a_turn_soon_after_a_stop_steadies_first_instead_of_skipping_the_stride_check(pg):
    """Regression: a turn asked for within a quarter second of a stop counted as out of the walk, but the walk no
    longer played, so its walking version began at once wherever the stride was (here at 0.48 to 0.88 of it where
    only 0.3 was checked safe), from joints half way home. It turns as from standing, once steady, as after any other
    stop."""
    walker = _swinging(pg, walking=True, turn_left_walking_at=[0.3])
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, STOP_GRACE + 0.06)  # letting go of W, then pressing A a moment after it stopped
    assert walker.rhythm is None
    pg.behaviors.move("crab", 0.0, 0.8)
    began = pg.run_fast(lambda: walker.rhythm is not None, 2 * CALM_MAX)
    assert walker.playing == "left" and walker.rhythm == "left", f"{walker.rhythm} after {began:.2f} s"


def test_a_quick_key_change_turns_straight_out_of_the_walk(pg):
    """Regression: letting go of W and pressing A a moment later sends a stop between the two; taken at once, the robot
    steadied first, and a 1 s turn one update after letting go of W turned the template dog 27 to 70% of its rate. A
    stop asked while the walk plays comes STOP_GRACE later: a turn asked by then begins straight out of the walk (at a
    point of the stride it was checked safe from), and the walk asked again plays on."""
    walker = _swinging(pg, walking=True, turn_left_walking_at=[0.3])
    walked_from = walker.t0
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 0.02)
    assert walker.rhythm == "forward" and walker.stop_at is not None and walker.status()["walking"] is False
    pg.behaviors.move("crab", 0.2, 0.0)  # W again before the stop came: it walks on as it was
    pg.run_fast(lambda: False, 0.02)
    assert walker.rhythm == "forward" and walker.t0 == walked_from and walker.stop_at is None
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 0.02)
    pg.behaviors.move("crab", 0.0, 0.8)  # A one update after letting go of W
    began = pg.run_fast(lambda: walker.rhythm == "left walking", 1.0)
    assert ((walker.t0 - walked_from) * walker.walk.freq) % 1.0 == pytest.approx(0.3, abs=0.03), f"after {began:.2f}"
    pg.behaviors.stop("crab")  # out of a turn the stop comes at once
    pg.run_fast(lambda: False, 0.02)
    assert walker.rhythm is None and walker.stop_at is None
    pg.behaviors.move("crab", 0.2, 0.0)
    pg.run_fast(lambda: walker.rhythm == "forward", 2 * CALM_MAX)
    pg.run_fast(lambda: False, 1.0)
    pg.behaviors.stop("crab")  # nothing asked by then: it steadies on home
    assert pg.run_fast(lambda: walker.rhythm is None, 1.0) == pytest.approx(STOP_GRACE, abs=0.025)


@pytest.mark.parametrize(("walked", "steadies"), [(0.5, True), (1.2, True), (1.7, False)])
def test_a_turn_asked_early_in_the_walk_steadies_first(pg, walked, steadies):
    """Regression: in random runs of commands, 6 template dog walks fell 16 times in 14400 commands against 7 switched
    as before, 10 of them in or after a turn begun out of a walk 0.06 to 1.04 s old, often one started out of a turn;
    with turns begun only from walks 1 s old, 8 of 11 falls more in 86400 commands came out of walks 1.0 to 1.47 s old.
    Asked before the walk has played WALKED_MIN (1.5 s), the robot steadies first."""
    walker = _swinging(pg, walking=True, walked=walked)
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.playing == "left"
    if steadies:  # steadying on home, or already still and turning as from standing
        assert walker.rhythm in (None, "left")
    else:
        assert walker.rhythm == "left walking"


def test_a_turn_out_of_a_walk_begun_off_its_course_steadies_first(pg):
    """Regression: asked out of a turn, a walk began after the robot had spun on while it steadied, and steered hard
    back to where it was asked to go; a turn begun straight out of that walk met a stride it was never checked from.
    In 1800 random turn, walk, turn sequences 12 template dog walks fell 15 times against 5 switched as 7a4aa92 did; a
    turn out of a walk that began more than COURSE_GATE off its course now steadies first: 7 (scripts/rhythm_bench.py
    oos --paths after --bench-seed 3)."""
    from robotforge.behaviors import rhythm_walk
    from robotforge.control.cpg import trunk_yaw

    walker = _swinging(pg, walking=True)
    assert walker.began_off < rhythm_walk.COURSE_GATE  # asked standing still, facing its course
    data = pg.session.data
    walker.course = trunk_yaw(data, walker.body) + 2 * rhythm_walk.COURSE_GATE  # as if asked before it spun on
    walker._start(data, data.time, "forward", fade=False)
    assert walker.began_off == pytest.approx(2 * rhythm_walk.COURSE_GATE)
    pg.run_fast(lambda: False, 1.7)  # past WALKED_MIN
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    assert walker.playing == "left" and walker.rhythm in (None, "left")  # steadying, or still and turning as learned


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


def _fake_play(tried: list, held_walk_falls: bool):
    """A stand-in for play_alone: out of the walk the right turn falls when it begins straight out of it, whichever
    way it is left; a held stop falls if `held_walk_falls`. Turns give 0.6 rad in a first second straight out of the
    walk, 0.3 steadying first."""
    from robotforge.behaviors.rhythm_walk import Played

    def play(model, robot, walk, commands):
        turns = [c for c in commands if c[1]]
        walked = commands[0][0] > 0
        if not turns:
            tried.append((walk.switching, "walk"))
            fell = held_walk_falls and walk.switching == "hold"
            return [Played(0.0, False), Played(0.0, fell)]
        side = "right" if turns[0][1] < 0 else "left"
        walking, exit_ = getattr(walk, f"turn_{side}_walking") is not None, getattr(walk, f"turn_{side}_exit")
        tried.append((walk.switching, side, walking, exit_))
        fell = walked and side == "right" and walking
        out = [Played(0.0, False), Played((0.6 if walking else 0.3) * (-1 if side == "right" else 1), False)]
        return out + [Played(0.0, fell)]

    return play


@pytest.mark.parametrize("held_walk_falls", [False, True])
def test_switches_keep_the_first_way_that_tips_the_robot_over_least(monkeypatch, held_walk_falls):
    """Regression: leaving each turn was checked only for stops, so a template dog walk's right turn kept to fade fell
    in 2 of 25 walk, turn, walk sequences and another kept to hold in 10; and fading everywhere, 6 dog walks still fell
    12 times in 2100 bench sequences, holding (as before) in none. How the walk steadies (holding first, fading only if
    that tips it over less), whether each turn begins straight out of the walk and how it is left are now chosen
    together, on the sequences move and the keyboard give."""
    from robotforge.behaviors import rhythm_walk

    tried: list = []
    monkeypatch.setattr(rhythm_walk, "play_alone", _fake_play(tried, held_walk_falls))
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    names = joint_names(solo, Body.find(solo, "crab"))
    still = [0.0] * 3 * len(names)
    turns = {"turn_left": still, "turn_right": still, "turn_rate": 1.0}
    walking = {"turn_left_walking": still, "turn_right_walking": still, "turn_right_walking_at": [0.2]}
    kept = rhythm_walk.learn_switches(solo, "crab", _walk("crab", names, **turns, **walking))
    style = "fade" if held_walk_falls else "hold"
    assert kept.switching == style
    assert kept.turn_left_walking is not None and kept.turn_left_exit == style  # no falls: the style's own way out
    assert kept.turn_right_walking is None and kept.turn_right_walking_at is None and kept.turn_right_exit == style
    checks = len(rhythm_walk.switch_checks(2.0, 1.0))
    assert tried.count((style, "left", True, style)) == checks  # the first way on each side, checked in full
    assert tried.count((style, "right", False, style)) == checks
    assert all(t[0] == "hold" for t in tried) != held_walk_falls  # fading is only tried if holding fell
    # rad/s, each side's first second out of the walk: reporting the slower alone said 26 deg/s for a template dog walk
    # whose 1 s turns right after walking went 70 degrees left and 159 right
    assert kept.turn_rates_from_walk == pytest.approx({"left": 0.6, "right": 0.3})
    assert kept.describe()["turn_deg_s_from_walk"] == {"left": 34, "right": 17}  # degrees
    assert rhythm_walk.learn_switches(solo, "crab", _walk("crab", names)).turn_left_exit is None  # no turns yet


@pytest.mark.parametrize(
    ("walking_falls", "steady_falls", "kept_walking"), [(1, 0, True), (2, 0, False), (2, 2, False), (2, 3, True)]
)
def test_a_walking_turn_that_falls_twice_is_dropped_unless_steadying_falls_more(
    monkeypatch, walking_falls, steady_falls, kept_walking
):
    """Regression: a walking turn was dropped on a single fall more than steadying first, in about 85 checks: 3 of 6
    imported A1 walks lost theirs, their 1 s turns after walking slowed to about half the rate, and the bench fell as
    often as before. Kept unless it fell 2 more, a template dog walk's left turn that fell twice, as often as
    steadying first, fell in 4 of its 100 bench sequences, steadying first in none. One fall is noise; two drop it
    unless steadying first falls more."""
    from robotforge.behaviors import rhythm_walk
    from robotforge.behaviors.rhythm_walk import Played

    seen: dict[tuple, int] = {}

    def play(model, robot, walk, commands):
        """The left turn out of the walk falls in the first few sequences of each way checked."""
        turns = [c for c in commands if c[1]]
        if not turns:
            return [Played(0.0, False), Played(0.0, False)]
        fell = False
        if turns[0][1] > 0 and commands[0][0] > 0:
            walking = walk.turn_left_walking is not None
            way = (walk.switching, walk.turn_left_exit, walking)
            seen[way] = seen.get(way, 0) + 1
            fell = seen[way] <= (walking_falls if walking else steady_falls)
        return [Played(0.0, False), Played(0.5, False), Played(0.0, fell)]

    monkeypatch.setattr(rhythm_walk, "play_alone", play)
    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "crab", "build": STAR}]})).model
    names = joint_names(solo, Body.find(solo, "crab"))
    still = [0.0] * 3 * len(names)
    walk = _walk("crab", names, turn_left=still, turn_right=still, turn_rate=1.0, turn_left_walking=still)
    kept = rhythm_walk.learn_switches(solo, "crab", walk)
    assert (kept.turn_left_walking is not None) == kept_walking


@pytest.mark.parametrize("kept", ["before", "to hold"])
def test_a_walk_kept_before_or_to_hold_switches_as_walks_always_did(pg, kept):
    """Regression: with faster switching (stop fading to home, rhythms starting once the robot is calm), the walk the
    imported A1 kept first ended on its back when the sim ran on between move calls, as it does with the viewer
    open: in 7 of 8 waits from 0 to 5 s; switched as before, in 1. A file without a home pose, kept before, stops at
    once on home and waits on it until the robot stands still; so does one checked to hold (switching: hold)."""
    walker = _swinging(pg, kept_before=kept == "before", **({"switching": "hold"} if kept == "to hold" else {}))
    assert walker.holds and walker.grace == 0.0  # no turn to begin out of the walk: a stop comes at once, as before
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


def test_a_walk_kept_to_hold_still_begins_its_walking_turns_straight_out_of_the_walk(pg):
    """A template dog walk never fell when it held home before other switches, and kept the turns that begin straight
    out of the walk safely: a 1 s turn after walking turned 80% of its rate on the median instead of 47%."""
    walker = _swinging(pg, walking=True, switching="hold")
    assert walker.holds
    pg.behaviors.stop("crab")  # what the move tool does between commands: the next comes at once
    pg.behaviors.move("crab", 0.0, 0.8)
    pg.run_fast(lambda: False, 0.02)
    assert walker.rhythm == "left walking" and walker.steady_from is None and walker.fade_from is not None
    pg.run_fast(lambda: False, 0.5)
    pg.behaviors.stop("crab")
    pg.run_fast(lambda: False, 1.0 / walker.rate_hz)
    home = np.clip(walker.home, walker.lower, walker.upper)
    assert walker.holding and pg.session.data.ctrl[walker.actuators] == pytest.approx(home)  # stop: home at once


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


def test_a_walking_turn_that_comes_near_falling_is_left_out(monkeypatch):
    """Regression: begun out of the walk only at points of the stride it never fell from, a turn could still come
    near to falling there, and in random walk-turn-walk sequences the turns of template dog and imported A1 walks fell
    more often than the robots steadied first (11 in 2880 against 6 for the dog, 162 in 1440 against 147 for the A1).
    Each that fell had leaned more than 30 degrees from some moment of its checks. A turn that leans the trunk
    ROOM_TILT or more from a moment at its points is not begun out of the walk."""
    from robotforge.behaviors import rhythm_walk
    from robotforge.behaviors.rhythm_walk import _walking_turn
    from robotforge.control.cpg import Start

    solo = compile_world(World.model_validate({"name": "solo", "robots": [{"name": "dog", "template": "quadruped"}]}))
    model = solo.model
    check = Trainer(model, "dog", Reward(turn=1.0), 2.0)
    first = Trainer(model, "dog", Reward(turn=1.0), 1.0)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    standing = Start(data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    n = check.body.joints
    swing = Rhythm(np.concatenate([np.zeros(n), np.full(n, 0.3), np.zeros(n)]))
    lean = check.rollout(swing, start=standing)["tilt"]
    assert 0.0 < lean < math.radians(60.0)  # it sways, and stays up
    times = (0.5, 0.6, 0.7, 0.8)
    monkeypatch.setattr(rhythm_walk, "ROOM_TILT", lean + 1e-3)
    assert _walking_turn((swing,), 1.0, check, first, [standing] * 4, times) is not None
    monkeypatch.setattr(rhythm_walk, "ROOM_TILT", lean - 1e-3)
    assert _walking_turn((swing,), 1.0, check, first, [standing] * 4, times) is None


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
    walker.origin = pg.session.data.xpos[walker.body.trunk][:2].copy()  # on its line, which runs along +x
    face(-1.0)  # spun a long way to the right
    assert walker._heading_error(data, 10.0, 0.02) == pytest.approx(rhythm_walk.STEER_SPAN)  # corrects fully
    assert walker.aim == pytest.approx(-1.0 + rhythm_walk.STEER_LEASH)  # its aim no further off than the leash
    face(-0.1)  # back near the course: the aim is never further than the leash either way, then comes back
    for _ in range(50):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(-0.1 - rhythm_walk.STEER_LEASH + rhythm_walk.STEER_RELAX, abs=0.003)
    for _ in range(500):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(0.0, abs=1e-3)  # all the way back
    assert walker._heading_error(data, 1.0, 0.02) == pytest.approx(0.1 / rhythm_walk.STEER_RAMP, abs=1e-3)  # easing in


def test_a_walk_beside_its_line_steers_back_onto_it(pg):
    """Regression: holding its heading, a walk that went a little crabwise walked on beside the line it started on:
    over 20 s from standing, 8 kept template dog walks ended 0.10 m to the side per metre walked on average and 6
    imported A1 walks 0.18 (scripts/rhythm_bench.py straight). It aims back at the line, LINE_AHEAD further along it,
    at most LINE_CUT off the course."""
    from robotforge.behaviors import rhythm_walk
    from robotforge.control.cpg import trunk_yaw

    walker = _swinging(pg)
    data = pg.session.data
    course = trunk_yaw(data, walker.body)
    walker.course = walker.aim = course  # facing along its line, 0.3 m to the left of it
    left = np.array([-math.sin(course), math.cos(course)])
    walker.origin = data.xpos[walker.body.trunk][:2] - 0.3 * left
    for _ in range(500):
        error = walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(course - math.atan2(0.3, rhythm_walk.LINE_AHEAD))
    assert error < -0.2  # rad: it steers right, back toward the line
    walker.origin = data.xpos[walker.body.trunk][:2] + 5.0 * left  # far to the right of it: at most LINE_CUT
    for _ in range(500):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(course + rhythm_walk.LINE_CUT)
    walker.origin = data.xpos[walker.body.trunk][:2].copy()  # on it: the aim comes back to the course
    for _ in range(500):
        walker._heading_error(data, 10.0, 0.02)
    assert walker.aim == pytest.approx(course)


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


def test_a_kept_walk_learns_to_turn_and_move_turns_the_robot(pg, monkeypatch):
    """Turning on the spot is learned after the walk (one rhythm each way); a turn command then plays it."""
    from robotforge.behaviors import rhythm_walk
    from robotforge.behaviors.rhythm_walk import learn_turns

    # Checked from every update of the walk the walking turns take minutes here; a tenth of a second apart, as before
    monkeypatch.setattr(rhythm_walk, "WALK_CHECKS", rhythm_walk.WALK_CHECKS[::5])
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
    monkeypatch.setattr(rhythm_walk, "WALK_CHECKS", rhythm_walk.WALK_CHECKS[::5])
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
