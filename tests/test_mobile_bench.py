"""The mobile manipulation bench (scripts/mobile_bench.py): its table, and a few real cells on the offline stand-ins."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "mobile_bench.py"
FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("mobile_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mobile_bench"] = module
    spec.loader.exec_module(module)
    return module


def _run(robot: str, job: str, seed: int, **extra):
    return {"robot": robot, "job": job, "seed": seed, **extra}


def test_a_cell_every_seed_of_which_was_refused_for_what_the_robot_cannot_do_is_not_supported_not_a_failure(bench):
    runs = [_run("spot_z1", "fetch", s, success=False, why="refused:cannot_pick") for s in range(4)]
    runs += [_run("mobile_panda", "fetch", s, success=s != 2, why=None if s != 2 else "missed") for s in range(4)]
    text = bench.report(runs)
    assert "spot_z1        fetch    n/s      not supported: refused:cannot_pick" in text
    assert "mobile_panda   fetch    3/4      missed x1" in text
    assert "measured cells: 3/4 runs succeeded" in text  # the 4 refused runs are left out of the totals


def test_a_refusal_that_only_some_seeds_met_stays_a_measured_failure(bench):
    runs = [_run("lift_picker", "conveyor", 0, success=True)]
    runs += [_run("lift_picker", "conveyor", 1, success=False, why="refused:too_late")]
    text = bench.report(runs)
    assert "lift_picker    conveyor 1/2      refused:too_late x1" in text


def test_a_refusal_names_its_cause(bench):
    assert bench.cause("mm/arm does not reach the height of the belt (0.62 m) from anywhere") == (
        "refused:cannot_reach_belt"
    )
    assert bench.cause("it gets to sooner than the arm: the box is at the end of the belt, sooner than") == (
        "refused:too_late"
    )
    assert bench.cause("nowhere mm can stand") == "refused:nowhere_to_stand"


@pytest.mark.slow
@pytest.mark.parametrize(("job", "seed"), [("door", 0), ("button", 1), ("conveyor", 7)])
def test_a_mobile_panda_does_the_job_in_the_bench(bench, monkeypatch, tmp_path, job, seed):
    """Real cells of the bench, small enough to run: the swing door, the button, the box off the running belt."""
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))  # run_one sets it itself; put back after
    result = bench.run_one("mobile_panda", job, seed)
    assert result["success"], result
