"""The lightweight budget's bookkeeping (scripts/bench.py): comparing runs, checking budgets, finding the last run.

The measurements themselves need Chrome, uv and minutes of machine time; these tests need none of them.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "bench.py"
BUDGET = Path(__file__).parents[1] / "docs" / "bench" / "budget.json"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def test_importing_the_script_stays_light():
    """Its own process measures the others: robotforge, mujoco and the MCP SDK load only where a part needs them."""
    here = str(SCRIPT.parent)
    code = f"import json, sys; sys.path.insert(0, {here!r}); import bench; print(json.dumps([*sys.modules]))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = {name.split(".")[0] for name in json.loads(done.stdout)}
    assert not loaded & {"robotforge", "mujoco", "mcp", "numpy"}


def test_compare_says_better_or_worse_by_the_metrics_direction(bench):
    m = bench.metric
    now = {
        "imports.cli_ms": m(300, "ms", "lower"),
        "physics.dog.steps_per_s": m(30000, "steps/s", "higher"),
        "install.installed_mb": m(160.5, "MB", "lower"),
        "mcp.move_ms": m(70, "ms", "lower"),
        "viewer.fps": m(60, "fps", "higher"),
        "physics.new.steps_per_s": m(1000, "steps/s", "higher"),
    }
    before = {
        "imports.cli_ms": m(600, "ms", "lower"),  # halved: better
        "physics.dog.steps_per_s": m(45000, "steps/s", "higher"),  # a third slower: worse
        "install.installed_mb": m(160.0, "MB", "lower"),  # +0.3 %: sizes are exact, but this is within 2 %
        "mcp.move_ms": m(56, "ms", "lower"),  # +25 %: within the noise of a shared machine
        "viewer.fps": m(None, "fps", "higher"),  # not measured last time
    }
    out = bench.compare(now, before)
    assert out["imports.cli_ms"] == {"before": 600.0, "change": -0.5, "verdict": "better"}
    assert out["physics.dog.steps_per_s"]["verdict"] == "worse"
    assert out["install.installed_mb"]["verdict"] == "same"
    assert out["mcp.move_ms"]["verdict"] == "same"
    assert out["viewer.fps"]["verdict"] == "new"
    assert out["physics.new.steps_per_s"] == {"before": None, "change": None, "verdict": "new"}


def test_a_tiny_timing_that_wobbles_is_not_a_regression(bench):
    """An observe call measured 1.8 ms and then 3.4 ms on a busy machine: +90 percent, but within 2 ms."""
    m = bench.metric
    out = bench.compare({"mcp.observe_ms": m(3.4, "ms", "lower")}, {"mcp.observe_ms": m(1.8, "ms", "lower")})
    assert out["mcp.observe_ms"]["verdict"] == "same"
    out = bench.compare({"mcp.observe_ms": m(6.0, "ms", "lower")}, {"mcp.observe_ms": m(1.8, "ms", "lower")})
    assert out["mcp.observe_ms"]["verdict"] == "worse"


def test_measured_memory_wobbles_but_a_size_on_disk_does_not(bench):
    m = bench.metric
    peak = bench.compare(
        {"physics.dog.peak_mb": m(160, "MB", "lower", noise=bench.MEMORY_NOISE)},
        {"physics.dog.peak_mb": m(155, "MB", "lower", noise=bench.MEMORY_NOISE)},
    )
    assert peak["physics.dog.peak_mb"]["verdict"] == "same"


def test_a_size_that_grows_past_two_percent_is_worse(bench):
    m = bench.metric
    now, before = {"install.installed_mb": m(165, "MB", "lower")}, {"install.installed_mb": m(160, "MB", "lower")}
    out = bench.compare(now, before)
    assert out["install.installed_mb"]["verdict"] == "worse"


def test_budgets_flag_what_is_over_and_what_was_not_measured(bench):
    m = bench.metric
    now = {
        "imports.cli_ms": m(900, "ms", "lower"),
        "physics.dog.steps_per_s": m(20000, "steps/s", "higher"),
        "viewer.fps": m(60, "fps", "higher"),
        "mcp.start_s": m(None, "s", "lower"),
    }
    budgets = {
        "imports.cli_ms": {"max": 800, "why": "cold start"},
        "physics.dog.steps_per_s": {"min": 25000},
        "viewer.fps": {"min": 50},
        "mcp.start_s": {"max": 3},
        "install.installed_mb": {"max": 200},
    }
    rows = {r["metric"]: r for r in bench.check_budgets(now, budgets)}
    assert rows["imports.cli_ms"] == {
        "metric": "imports.cli_ms",
        "value": 900,
        "why": "cold start",
        "limit": 800.0,
        "kind": "max",
        "status": "over",
    }
    assert rows["physics.dog.steps_per_s"]["status"] == "over"  # a minimum not reached
    assert rows["viewer.fps"]["status"] == "ok"
    assert rows["mcp.start_s"]["status"] == "unmeasured"  # it ran but measured nothing
    assert rows["install.installed_mb"]["status"] == "unmeasured"  # that part did not run
    with pytest.raises(ValueError, match="needs a max or a min"):
        bench.check_budgets(now, {"viewer.fps": {"why": "no limit"}})


def test_a_value_at_its_limit_is_within_budget(bench):
    m = bench.metric
    rows = bench.check_budgets(
        {"a.x": m(10, "ms", "lower"), "a.y": m(5, "fps", "higher")}, {"a.x": {"max": 10}, "a.y": {"min": 5}}
    )
    assert [r["status"] for r in rows] == ["ok", "ok"]


def test_the_previous_run_is_the_newest_by_its_start_not_by_file_name(bench, tmp_path):
    assert bench.previous_run(tmp_path) is None
    (tmp_path / "2026-10-18-aaaaaaa.json").write_text(json.dumps({"started": "2026-10-18T09:00:00+08:00"}))
    (tmp_path / "2026-10-17-zzzzzzz.json").write_text(json.dumps({"started": "2026-10-18T21:00:00+08:00"}))
    (tmp_path / "budget.json").write_text(json.dumps({"started": "2099-01-01T00:00:00+08:00", "budgets": {}}))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "notes.json").write_text(json.dumps({"something": "else"}))
    assert bench.previous_run(tmp_path).name == "2026-10-17-zzzzzzz.json"


def test_flatten_names_metrics_by_section_and_skips_errors(bench):
    m = bench.metric
    results = {
        "imports": {"cli_ms": m(1, "ms", "lower")},
        "physics": {"dog.steps_per_s": m(2, "steps/s", "higher"), "go2_error": "Traceback ..."},
        "viewer": {"error": "RuntimeError: Google Chrome not found"},
    }
    assert set(bench.flatten(results)) == {"imports.cli_ms", "physics.dog.steps_per_s"}


def test_import_offenders_sum_self_time_by_package(bench):
    log = "\n".join(
        [
            "import time: self [us] | cumulative | imported package",
            "import time:       500 |        500 |     numpy._core",
            "import time:      1500 |       2000 |   numpy",
            "import time:      3000 |       3000 |   mcp.types",
            "import time:       200 |       3200 | mcp",
            "import time:       100 |       5300 | robotforge.cli",
            "something else on stderr",
        ]
    )
    assert bench.import_offenders(log, top=2) == [["mcp", 3.2], ["numpy", 2.0]]


def test_the_table_shows_the_change_and_the_budget(bench):
    m = bench.metric
    now = {"imports.cli_ms": m(300, "ms", "lower")}
    compared = bench.compare(now, {"imports.cli_ms": m(600, "ms", "lower")})
    budgets = bench.check_budgets(now, {"imports.cli_ms": {"max": 500}})
    header, row = bench.table(now, compared, budgets)
    assert "budget" in header
    assert "300" in row and "600" in row and "-50%" in row and "<= 500" in row and "ok (better)" in row


def test_the_budget_file_names_metrics_the_bench_measures(bench):
    data = json.loads(BUDGET.read_text(encoding="utf-8"))
    assert data["budgets"], "budget.json has no budgets"
    for name, budget in data["budgets"].items():
        section, _, rest = name.partition(".")
        assert section in bench.SECTIONS, name
        if section == "physics":
            world = rest.split(".")[0]
            assert world in bench.WORLDS, name
        assert ("max" in budget) != ("min" in budget), f"{name}: give a max or a min"
        assert budget.get("why"), f"{name}: say where the number comes from"
