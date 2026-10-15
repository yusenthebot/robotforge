"""Quadruped gaits: which legs step together, the step period, and the Go2 (offline stand-in)."""

from pathlib import Path

import numpy as np
import pytest

from robotforge.learn import gait
from robotforge.learn.engine import LessonError
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("gait")
    return playground


def _play(pg: Playground, seconds: float) -> tuple[dict, list[dict]]:
    """Run the physics as the viewer does; returns the last update and every plot sample on the way."""
    update, samples = None, []
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
        samples += update["samples"]
    return update, samples


def _set(pg: Playground, **values: float) -> None:
    for control, value in values.items():
        pg.lessons.handle({"action": "set", "control": control, "value": value})


def _walk(pg: Playground) -> tuple[dict, list[dict]]:
    pg.lessons.handle({"action": "press", "control": "walk"})
    _, samples = _play(pg, gait.TEST_SECONDS + 0.2)
    update, _ = _play(pg, 0.1)
    return update, samples


def _next(pg: Playground, guess: int | None = None) -> None:
    pg.lessons.handle({"action": "next"})
    if guess is not None:
        pg.lessons.handle({"action": "answer", "choice": guess})


def _together(samples: list[dict], a: str, b: str) -> float:
    """How often two feet are up or down at the same time while walking."""
    walking = [s for s in samples if 0 < sum(s[k] for k in ("fl", "fr", "rl", "rr")) < 4]
    return float(np.mean([s[a] == s[b] for s in walking]))


def test_the_trot_moves_diagonal_feet_together_and_rocks_least(pg):
    with pytest.raises(LessonError, match="later step"):
        _set(pg, pattern=1)
    pg.lessons.handle({"action": "answer", "choice": 0})
    update, samples = _walk(pg)
    trot = update["results"]["walk"]
    assert "trot" in update["passed"] and trot["distance"] > 0.8 and not trot["fell"]
    assert _together(samples, "fl", "rr") > 0.8 > 0.3 > _together(samples, "fl", "fr")
    assert update["results"]["summary"]["en"].startswith("Covered")
    live = update["live"]
    assert len(live["com"]) == 2 and {r["show"] for r in live["formula"]} >= {"f_legs", "f_stride", "f_max"}

    _next(pg, guess=0)
    _set(pg, pattern=1)
    update, samples = _walk(pg)
    pace = update["results"]["walk"]
    assert "pace" in update["passed"] and pace["roll"] > 3 * trot["roll"]
    assert _together(samples, "fl", "rl") > 0.55 > _together(samples, "fl", "rr")  # the rocking blurs the timing

    _next(pg, guess=1)
    _set(pg, pattern=2)
    update, _ = _walk(pg)
    bound = update["results"]["walk"]
    assert "bound" in update["passed"] and bound["pitch"] > 3 * trot["pitch"]
    assert bound["distance"] < trot["distance"] / 2  # nodding, it hardly gets anywhere


def test_quick_steps_go_faster_and_the_go2_goes_further(pg):
    for step in ("trot", "pace", "bound"):
        pg.lessons.passed.add(step)
    pg.lessons.handle({"action": "goto", "step": 3})
    pg.lessons.handle({"action": "answer", "choice": 1})
    _set(pg, pattern=0, speed=0.6, period=0.5)
    slow = _walk(pg)[0]["results"]["walk"]
    _set(pg, period=0.25)
    update, _ = _walk(pg)
    quick = update["results"]["walk"]
    assert quick["distance"] > 1.6 * slow["distance"] and "period" in update["passed"]
    rows = {r["show"]: r for r in update["live"]["formula"]}
    assert "note" in rows["f_max"]  # 0.6 m/s is past T 0.25's limit for our dog

    _next(pg, guess=1)
    _set(pg, period=0.36, dog=1)
    update, _ = _walk(pg)
    go2 = update["results"]["walk"]
    assert go2["dog"] == "go2" and "go2" in update["passed"]
    ours = [r for r in pg.lessons.ctx.runs if r["params"]["dog"] == 0 and r["params"]["period"] == 0.5]
    assert go2["distance"] > ours[-1]["metrics"]["distance"]  # longer legs, longer strides

    _next(pg)
    _set(pg, period=0.25)
    update, _ = _walk(pg)
    assert update["results"]["walk"]["distance"] >= gait.CHALLENGE_DISTANCE and "challenge" in update["passed"]


def test_every_walk_starts_from_the_line_with_fresh_settings(pg):
    pg.lessons.handle({"action": "answer", "choice": 0})
    first = _walk(pg)[0]["results"]["walk"]
    again = _walk(pg)[0]["results"]["walk"]
    assert again["distance"] == pytest.approx(first["distance"], abs=0.02)  # the same start, the same walk
    runs = pg.lessons.ctx.runs
    assert len(runs) == 2 and runs[0]["x"][0] == pytest.approx(0, abs=0.02) and runs[0]["y"][-1] > 0.8
