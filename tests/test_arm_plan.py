"""Motion planning lesson: straight across hits, the planned path goes round, on our arm and the PiPER."""

from pathlib import Path

import pytest

from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.lessons.start("arm_planning")
    return playground


def _play(pg: Playground, seconds: float) -> dict:
    update = None
    for _ in range(round(seconds * 20)):
        pg.session.step(round(0.05 / pg.session.model.opt.timestep))
        update = pg.lessons.tick()
    return update


def _go(pg: Playground, control: str) -> dict:
    pg.lessons.handle({"action": "press", "control": control})
    for _ in range(60):
        _play(pg, 0.5)
        run = pg.lessons.lesson.run
        if run is None or run["finished"]:
            break
    return _play(pg, 0.1)


def _set(pg: Playground, **values: float) -> None:
    for control, value in values.items():
        pg.lessons.handle({"action": "set", "control": control, "value": value})


def test_straight_hits_the_fence_and_the_plan_goes_round(pg):
    first = _play(pg, 0.05)
    assert first["live"]["drawing"]["goal"]  # the goal is drawn from the start...
    assert "drawing" not in _play(pg, 0.05)["live"]  # ...and sent again only when it changes
    pg.lessons.describe_active()
    assert "drawing" in _play(pg, 0.05)["live"]  # or when a viewer (re)connects
    pg.lessons.handle({"action": "answer", "choice": 1})
    straight = _go(pg, "straight")
    assert straight["results"]["arm_run"]["hit"] == 1.0 and "straight" in straight["passed"]
    pg.lessons.handle({"action": "next"})
    pg.lessons.handle({"action": "answer", "choice": 0})
    pg.lessons.handle({"action": "press", "control": "plan"})
    drawn = _play(pg, 0.05)["live"].get("drawing") or {}  # the one tick after the plan
    assert drawn.get("trees") and len(drawn.get("path", [])) > 2 and drawn.get("clash")
    planned = _go(pg, "plan")
    run = planned["results"]["arm_run"]
    assert run["hit"] == 0.0 and run["nodes"] > 2 and run["turned"] < run["raw"]  # round it, and shortcut
    assert "rrt" in planned["passed"]
    record = pg.lessons.ctx.runs[-1]
    assert record["y"][-1] < 0.02  # the hand ends at the target


def test_the_piper_plans_round_its_fence_too(pg):
    for step in ("straight", "rrt", "smooth", "step"):
        pg.lessons.passed.add(step)
    pg.lessons.handle({"action": "goto", "step": 4})
    _set(pg, arm=1)
    assert _go(pg, "straight")["results"]["arm_run"]["hit"] == 1.0
    planned = _go(pg, "plan")
    assert planned["results"]["arm_run"]["arm"] == "piper" and planned["results"]["arm_run"]["hit"] == 0.0
    assert "piper" in planned["passed"]
