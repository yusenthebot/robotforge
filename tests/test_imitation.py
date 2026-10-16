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
from robotforge.learn.playthrough import Player, Press, Set, _show_a_pick, _show_more
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


def test_tests_and_the_sweep_give_the_measured_success(pg):
    """Five demonstrations by the scripted learner, then the three methods at the 24 spots and the sweep.

    Measured on macOS arm64, 2026-10-17: copying 4% of the 24 spots, 100% beside a demonstration, 0% 6 cm or more
    away; the network 92%, 100%, 75% (sweep: 0, 29, 67, 71, 92% with 1 to 5 demonstrations); the hand-written pick
    100% everywhere. On 12 sets of five demonstrations (4 through the viewer) the network ranged 62 to 95%, always
    100% beside a demonstration: the tolerances leave that much room for other machines' floating point."""
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
    assert runs[0]["x"][0] == 1.0 and runs[0]["y"][0] == 100.0  # copying: within 2 cm it works...
    assert max(runs[0]["y"][1:]) <= 34  # ...and further off it hardly ever does
    # the sweep runs on the lesson's thread and fills in its points one by one
    Set(method=1).apply(p)
    p.handle("sweep", control="use")
    _finish_job(p)
    points = p.ctx.sweep["points"]
    assert [pt["x"] for pt in points] == [1.0, 2.0, 3.0, 4.0, 5.0] and "swept:done" in p.ctx.events
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
