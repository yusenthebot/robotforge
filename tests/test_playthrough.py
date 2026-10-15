"""Every lesson can be played from its first step to its last by doing what its steps say.

The moves are in robotforge.learn.playthrough; scripts/playtest.py plays them all and prints a table. The
lessons that train for a minute or more are marked slow and left out of the default run:
`uv run pytest -m slow tests/test_playthrough.py` plays them.
"""

from pathlib import Path

import pytest

from robotforge.learn import LESSONS
from robotforge.learn.playthrough import RECIPES, Recipe, Set, Wait, play
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    return Playground.with_demo()


def _lessons() -> list:
    return [
        pytest.param(lesson_id, id=lesson_id, marks=[pytest.mark.slow] if RECIPES[lesson_id].slow else [])
        for lesson_id in LESSONS
    ]


@pytest.mark.parametrize("lesson_id", _lessons())
def test_the_lesson_can_be_played_to_the_end(pg, lesson_id):
    report = play(pg, lesson_id)
    assert report.ok, report.explain()
    assert not pg.lessons.active  # and left again, so nothing trains on in the background


def test_every_step_of_every_lesson_has_its_moves(pg):
    assert set(RECIPES) == set(LESSONS)
    for lesson_id, cls in LESSONS.items():
        steps = {s.id for s in cls().steps}
        assert set(RECIPES[lesson_id].steps) == steps, lesson_id


def test_a_stuck_step_says_what_was_tried_and_which_check_failed(pg):
    idle = Recipe({"feel": (Wait(1.0),)}, sim_limit=5.0)
    report = play(pg, "pd_control", idle)
    assert not report.ok and report.passed == 0 and report.stuck.id == "feel"
    assert report.stuck.tried == ["wait 1 s"]
    assert report.stuck.failed.startswith("check never passed: _swung_by_hand")
    assert "stuck at step 'feel'" in report.explain()

    early = Recipe({"feel": (Set(kp=40),)})  # the gains unlock in a later step
    stuck = play(pg, "pd_control", early).stuck
    assert stuck.failed == "the lesson refused: kp unlocks in a later step"


def test_a_step_that_takes_too_long_runs_out_of_time(pg):
    slow = Recipe({"feel": (Wait(10.0),)}, sim_limit=2.0)
    stuck = play(pg, "pd_control", slow).stuck
    assert stuck.failed == "out of time: the step's 2 s of physics ran out" and stuck.sim == pytest.approx(2.0)
