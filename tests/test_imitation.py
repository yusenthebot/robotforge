"""The imitation lesson: demonstrations recorded through the lesson's own controls, two learners copying them,
and roll-outs on the real physics that land where they were measured.

The demonstrations come from the playtester's scripted learner (learn/playthrough.py), which moves the same
sliders and toggle a learner moves; the cube spots come from the lesson's seed, so every run sees the same ones.
That the playtester plays the whole lesson through is tests/test_playthrough.py's job.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotforge.learn import cloning as bc
from robotforge.learn import imitation
from robotforge.learn.engine import LessonError
from robotforge.learn.playthrough import (
    GRASP,
    HIGH,
    Player,
    Press,
    Set,
    _drag,
    _show_a_pick,
    _show_more,
    _take_it_from_here,
)
from robotforge.playground import Playground


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    yield playground
    playground.lessons.exit()


def _player(pg: Playground) -> Player:
    pg.lessons.start("imitation")
    p = Player(pg)
    p.budget(math.inf, math.inf)
    return p


def _wait(p: Player, done, seconds: float = 300.0) -> None:
    """Play until `done`; at the viewer's pace while the lesson works on its own thread, which a physics loop at
    full speed would starve."""
    for _ in range(round(seconds / 0.25)):
        if done():
            return
        p.play(0.25, realtime=p.lesson.busy)
    raise AssertionError("never finished")


def _finish_try(p: Player) -> dict:
    _wait(p, lambda: p.lesson.hands.pilot is None and p.lesson.pending is None and not p.lesson.busy)
    p.play(0.1)
    return p.lesson.trial


def _finish_job(p: Player) -> None:
    _wait(p, lambda: not p.lesson.busy)
    p.play(0.1)


def _unlock_all(p: Player) -> None:
    """Jump the step pointer to the last step so every control is usable (the checks are tested elsewhere)."""
    lessons = p.pg.lessons
    lessons.passed.update(s.id for s in lessons.lesson.steps)
    lessons.step = len(lessons.lesson.steps) - 1


def test_the_lesson_builds_with_its_desk_arm_cube_and_bin(pg):
    described = pg.lessons.start("imitation")
    model = pg.session.model
    assert {"cube", "bin", "desk"} <= {model.body(b).name for b in range(model.nbody)} | {
        model.geom(g).name for g in range(model.ngeom)
    }
    assert described["rig"]["kind"] == "imitate" and described["drive"]["robot"] is False
    assert [s["id"] for s in described["steps"]] == [s.id for s in imitation.STEPS]
    # the arm's links are held up by gravity compensation, set when the model was compiled (see build_model)
    arm_bodies = [b for b in range(model.nbody) if model.body(b).name.startswith("arm/")]
    assert model.body_gravcomp[arm_bodies].min() == 1.0


def test_a_demonstration_is_recorded_as_see_do_pairs(pg):
    p = _player(pg)
    _show_a_pick(p)
    lesson = p.lesson
    assert p.ctx.results["demos"] == 1 and "show" in pg.lessons.passed
    demo = lesson.demos[0]
    assert demo.X.shape[1] == 7 and demo.Y.shape == (len(demo.X), 4) and 30 <= len(demo.X) <= 100
    assert np.allclose(demo.spot, bc.HOME_SPOT)
    # the close is labelled where the learner closed: at the bottom, over the cube
    first_close = int(np.argmax(demo.Y[:, 3] >= 0.5))
    hand, cube = demo.X[first_close, 3:6], demo.X[first_close, :3]
    assert np.linalg.norm(hand[:2] - cube[:2]) < 0.01 and hand[2] - cube[2] < 0.012
    update = p.play(0.1)
    table = update["live"]["table"]
    assert [r["label"]["en"] for r in table["rows"] if "label" in r] == ["#1"]
    assert {"demos", "tests"} <= set(update["live"].get("drawing") or lesson.drawing())


def test_a_held_cube_stays_in_the_gripper_at_a_learners_pace(pg):
    """Grasped, lifted 10 cm and held still for 15 s: still in the gripper. Without the no-slip pass (cloning.NOSLIP)
    it crept down 3.6 mm a second and was back on the desk 7.5 s after the lift (measured 2026-10-17)."""
    p = _player(pg)
    p.play(0.1)
    x, y, _ = p.update["live"]["cube"]
    _drag(p, x, y, HIGH)
    _drag(p, x, y, GRASP)
    p.play(0.5)
    Set(grip=1).apply(p)
    p.play(0.5)
    _drag(p, x, y, HIGH)
    p.play(15.0)
    live = p.update["live"]
    assert np.linalg.norm(np.subtract(live["hand"], live["cube"])) < 0.01
    assert live["cube"][2] - bc.DESK > HIGH - 0.03


def test_time_before_the_first_move_is_not_recorded(pg):
    """A learner reads the step for a minute before moving: the demonstration keeps a second of that, not 600
    samples of the arm standing at home (which outnumbered the real samples and set the network back)."""
    p = _player(pg)
    p.play(60.0)
    _show_a_pick(p)
    demo = p.lesson.demos[0]
    assert len(demo.X) <= 100 and demo.seconds < 15.0
    # at home: the first samples while the fingers open and the arm settles, then PAUSE_KEEP of standing still
    standing = [k for k in range(len(demo.X)) if np.linalg.norm(demo.X[k, 3:6] - demo.X[0, 3:6]) < 0.001]
    assert len(standing) <= bc.PAUSE_KEEP + 2


def test_new_cube_while_it_trains_calls_the_try_off(pg):
    """Let it try with the network, then New cube before the training ends: no try takes the arm over afterwards,
    and the demonstration the learner starts is recorded under the new cube's spot."""
    p = _player(pg)
    _show_a_pick(p)
    _unlock_all(p)
    Set(method=1, where=2).apply(p)
    Press("try").apply(p)
    assert p.lesson.busy  # training on the lesson's thread
    Press("new_cube").apply(p)
    spot = list(p.lesson.recording.spot)
    _finish_job(p)
    p.play(1.0)
    lesson = p.lesson
    assert lesson.hands.pilot is None and lesson.pending is None and lesson.trial is None
    assert lesson.recording is not None and np.allclose(lesson.recording.spot, spot)
    assert np.allclose(p.update["live"]["cube"][:2], spot, atol=0.002)  # the cube stayed where New cube put it
    assert not any(e.startswith("try:") for e in p.ctx.events)


def test_labels_skip_pauses_and_close_only_where_the_learner_closed():
    """A pause is labelled with the move that ended it; the way down is labelled 'go down', not 'close' (the copies
    closed 2 to 3 cm over the cube when the last 3 cm of the way down were labelled with the grip change)."""
    cube = np.array([0.25, 0.0, bc.DESK + 0.02])
    rows = []
    hand = np.array([0.25, 0.0, bc.DESK + 0.10])
    for _ in range(5):  # a pause above the cube
        rows.append((np.array([*cube, *hand, 0.07]), np.array([*hand, 0.0]), False))
    for k in range(1, 5):  # 2 cm a sample straight down to the grasp point, 7.5 cm lower
        target = hand + [0, 0, -0.02 * k] if k < 4 else np.array([0.25, 0.0, bc.DESK + 0.025])
        rows.append((np.array([*cube, *target + [0, 0, 0.02], 0.07]), np.array([*target, 0.0]), False))
    bottom = np.array([0.25, 0.0, bc.DESK + 0.025])
    rows.append((np.array([*cube, *bottom, 0.07]), np.array([*bottom, 0.0]), False))  # arrived, paused
    rows.append((np.array([*cube, *bottom, 0.07]), np.array([*bottom, 1.0]), False))  # closes
    rows.append((np.array([*cube, *bottom, 0.05]), np.array([*bottom, 1.0]), True))  # fingers moving: left out
    X, Y = bc.label(rows)
    assert len(X) == len(rows) - 1
    assert Y[0, 2] < bc.DESK + 0.10 - 0.02 and Y[0, 3] == 0.0  # the pause: the move down that ended it
    on_the_way = [k for k in range(len(X)) if X[k, 5] > bc.DESK + 0.035]  # a centimetre or more above the bottom
    assert len(on_the_way) == 9 and all(Y[k, 3] == 0.0 for k in on_the_way)  # never "close" before the bottom
    assert Y[-2, 3] == 1.0 and Y[-1, 3] == 1.0  # at the bottom: close


def test_copying_works_beside_a_demonstration_and_fails_far_from_it(pg):
    p = _player(pg)
    _show_a_pick(p)
    _unlock_all(p)
    Set(where=0, method=0).apply(p)
    Press("try").apply(p)
    near = _finish_try(p)
    assert near["ok"] is True and near["distance"] == pytest.approx(imitation.NEAR_OFFSET, abs=1e-3)
    Set(where=1).apply(p)
    Press("try").apply(p)
    far = _finish_try(p)
    # measured 2026-10-17: the corner of the region 24 cm from the only demonstration; it closes on air there
    assert far["ok"] is False and far["distance"] > 0.2
    assert "try:near:ok" in p.ctx.events and "try:far:missed" in p.ctx.events
    with pytest.raises(LessonError, match="wait for"):
        Press("try").apply(p)
        Press("try").apply(p)


@pytest.mark.slow  # 25 s: trains the network six times (CI runs the slow tests on Ubuntu)
def test_tests_and_the_sweep_give_the_measured_success(pg):
    """Five demonstrations by the scripted learner, then the three methods at the 24 spots and the sweep.

    Measured on macOS arm64, 2026-10-17, with the no-slip pass and SURE 0.5: copying 4% of the 24 spots, 100%
    beside a demonstration, 0% 6 cm or more away; the network 88%, 100%, 62% (sweep: 4, 21, 67, 71, 88% with 1 to
    5 demonstrations); the hand-written pick 100% everywhere. On 12 sets of five demonstrations (4 recorded through
    the viewer, 3 by a deliberately rough teacher) the network ranged 71 to 92%, 100% beside a demonstration in
    every set: the tolerances leave that much room for other machines' floating point."""
    p = _player(pg)
    _show_a_pick(p)
    _show_more(p)
    assert p.ctx.results["demos"] == imitation.DEMOS_WANTED
    _unlock_all(p)
    results = {}
    for method in range(3):
        Set(method=method).apply(p)
        Press("test").apply(p)
        _finish_job(p)
        results[imitation.METHODS[method]] = p.ctx.results["test"]
    knn, mlp, ik = results["knn"], results["mlp"], results["ik"]
    assert knn["success"] <= 20 and knn["near"] == 100 and knn["far"] <= 10
    assert mlp["success"] >= 50 and mlp["near"] >= 80 and mlp["success"] > knn["success"]
    assert ik["success"] == 100 and ik["near"] == 100 and ik["far"] == 100
    runs = p.ctx.runs
    assert [r["params"]["method"] for r in runs] == [0.0, 1.0, 2.0]
    assert "use" not in runs[2]["params"]  # the hand-written pick learned from no demonstrations
    assert runs[0]["x"][0] == 1.0 and runs[0]["y"][0] == 100.0  # copying: within 2 cm it works...
    assert max(runs[0]["y"][1:]) <= 34  # ...and further off it hardly ever does
    # the sweep runs on the lesson's thread and fills in its points one by one
    Set(method=1).apply(p)
    p.handle("sweep", control="use")
    _finish_job(p)
    points = p.ctx.sweep["points"]
    assert [pt["x"] for pt in points] == [1.0, 2.0, 3.0, 4.0, 5.0] and "swept:mlp" in p.ctx.events
    assert points[0]["success"] < points[-1]["success"]
    assert points[-1]["success"] == pytest.approx(mlp["success"], abs=1e-6)  # the same five, the same 24 spots


def test_the_hand_written_pick_needs_no_demonstrations(pg):
    p = _player(pg)
    _unlock_all(p)
    Set(method=2, where=2).apply(p)
    Press("try").apply(p)
    trial = _finish_try(p)
    assert trial["ok"] is True and trial["distance"] is None
    with pytest.raises(LessonError, match="show it once first"):
        Set(method=0).apply(p)
        Press("try").apply(p)


def test_w_a_s_d_move_the_target_along_the_desk(pg):
    p = _player(pg)
    start = dict(p.ctx.values)
    p.handle("drive", speed=0.14, turn=0.0)  # W, as the viewer sends it (70% of the lesson's 0.2 m/s)
    p.play(0.4)
    p.handle("drive", speed=0.0, turn=-0.14)  # D
    p.play(0.5)
    p.handle("drive", speed=0.0, turn=0.0)
    p.play(1.0)
    v = p.ctx.values
    assert v["tx"] - start["tx"] == pytest.approx(0.056, abs=0.01)
    assert v["ty"] - start["ty"] == pytest.approx(-0.07, abs=0.01)
    hand = p.update["live"]["hand"]
    assert np.linalg.norm(np.subtract(hand[:2], [v["tx"], v["ty"]])) < 0.01  # the gripper went there


def test_reset_starts_the_demonstration_over_and_keeps_the_ones_recorded(pg):
    p = _player(pg)
    _show_a_pick(p)
    Press("new_cube").apply(p)
    p.play(1.0)
    pg.session.reset()  # R in the viewer
    update = p.play(0.2)
    lesson = p.lesson
    assert pg.lessons.active and len(lesson.demos) == 1 and lesson.recording is not None
    assert np.allclose(update["live"]["cube"][:2], bc.HOME_SPOT, atol=1e-3)
    assert update["live"]["recording"] is True


def test_a_correction_taken_over_from_a_far_try_teaches_it_that_spot(pg):
    """DAgger in one round: copying misses the far spot; the learner takes over on the way and finishes the pick;
    the same spot again, and the copy now gets there (measured 2026-10-17 with 1 and with 5 demonstrations, and
    taking over 1.5, 2.5 or 3.5 s in or after the miss: every retry succeeded)."""
    p = _player(pg)
    _show_a_pick(p)
    _unlock_all(p)
    with pytest.raises(LessonError, match="press Let it try first"):
        Press("take_over").apply(p)
    Set(method=0, where=1).apply(p)
    Press("try", 2.0).apply(p)
    spot = list(p.lesson.trial["spot"])
    Press("take_over").apply(p)
    assert p.lesson.hands.pilot is None and p.lesson.recording.correction
    _take_it_from_here(p)
    assert "corrected" in p.ctx.events and p.lesson.demos[-1].correction
    rows = [r for r in p.update["live"]["table"]["rows"] if "label" in r]
    assert rows[-1]["label"]["en"] == "#2 fix"
    Press("try").apply(p)
    again = _finish_try(p)
    assert again["spot"] == spot and again["ok"] is True and "after_fix:far:ok" in p.ctx.events
    # still far from every demonstration: what got it there is the correction made at this very spot
    assert again["distance"] > 0.2 and "at the spot you corrected" in p.ctx.results["summary"]["en"]
    with pytest.raises(LessonError, match="nothing to correct"):
        Press("take_over").apply(p)


def test_a_sweep_ends_on_every_demonstration_and_copying_points_to_the_network(pg, monkeypatch):
    """Seven demonstrations: the sweep's counts stop at 6 in its values, and it trains on all seven too. A sweep
    with copying stays low, so it does not pass the step and says to sweep with the network."""
    spots = bc.test_spots()[:3]  # three of the 24: the counts are what this is about
    monkeypatch.setattr(bc, "test_spots", lambda: spots)
    p = _player(pg)
    while p.ctx.results.get("demos", 0) < 7:
        if p.ctx.results.get("demos", 0):
            Press("new_cube", 0.3).apply(p)
        _show_a_pick(p)
    _unlock_all(p)
    pg.lessons.passed.discard("how_many")
    Set(method=0).apply(p)
    p.handle("sweep", control="use")
    _finish_job(p)
    assert [pt["x"] for pt in p.ctx.sweep["points"]] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert "swept:knn" in p.ctx.events and not imitation._swept(p.ctx)
    assert "small neural network" in p.ctx.results["summary"]["en"]


def test_the_panel_never_shows_minus_zero():
    assert imitation._xyz([-0.001, 0.0, 0.254]) == "0, 0, 25"


def test_a_try_that_hovers_over_the_cube_never_closed(monkeypatch):
    """A network unsure at the cube (grip 0.6 against a threshold of 0.7, as SURE was) holds still there and fails:
    it never closed the gripper, and the reason must say so (it said 'closed the gripper 0 cm from the cube and
    missed it')."""
    monkeypatch.setattr(bc, "SURE", 0.7)
    model = bc.build_model()
    spot = np.array(bc.HOME_SPOT)

    def unsure(obs: np.ndarray) -> np.ndarray:
        return np.array([obs[0], obs[1], bc.DESK + 0.025, 0.6])

    out = bc.rollout(model, unsure, spot)
    assert not out.success and out.why["en"] == "never closed the gripper"


def test_a_pause_after_the_first_move_is_kept_whole():
    """Standing at home before the first move keeps 1 s; a 3 s pause at the cube before closing keeps all of it:
    those are the samples labelled 'close', and cutting them to 1 s took the network from 79% to 38% on slow
    demonstrations."""
    cube = np.array([0.25, 0.0, bc.DESK + 0.02])
    home, bottom = np.array([0.22, 0.0, bc.DESK + 0.11]), np.array([0.25, 0.0, bc.DESK + 0.025])
    rec = bc.Recording(cube[:2], 0.0)
    for k in range(30):  # reading the step
        rec.add(np.array([*cube, *home, 0.07]), home, 0.0, False, k * bc.DT)
    assert len(rec.rows) == 1 + bc.PAUSE_KEEP and rec.started == pytest.approx(29 * bc.DT - bc.PAUSE_KEEP * bc.DT)
    rec.add(np.array([*cube, *home, 0.07]), bottom, 0.0, False, 3.0)  # the first move
    for k in range(30):  # at the cube, finding the toggle
        rec.add(np.array([*cube, *bottom, 0.07]), bottom, 0.0, False, 3.1 + k * bc.DT)
    assert len(rec.rows) == 1 + bc.PAUSE_KEEP + 1 + 30
