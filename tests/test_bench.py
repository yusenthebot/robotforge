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
    # A Go2 world peaked at 480 MB and then 535 MB on the same build: its meshes' compile wobbles more
    go2 = bench.WORLDS["go2_walk"].memory_noise
    out = bench.compare({"x": m(535, "MB", "lower", noise=go2)}, {"x": m(480, "MB", "lower", noise=go2)})
    assert out["x"]["verdict"] == "same"
    out = bench.compare({"x": m(175, "MB", "lower", noise=bench.MEMORY_NOISE)}, {"x": m(155, "MB", "lower")})
    assert out["x"]["verdict"] == "worse"  # the dog's 13 percent is a change (trimesh and Pillow loaded again)


def test_a_tiny_share_of_controller_time_that_moves_is_not_a_regression(bench):
    """objects_20 spends 0.0046 and then 0.0124 of its time in controllers (it has none): not '+170% worse'."""
    m = bench.metric
    out = bench.compare({"x": m(0.0124, "share", "lower")}, {"x": m(0.0046, "share", "lower")})
    assert out["x"]["verdict"] == "same"
    out = bench.compare({"x": m(0.25, "share", "lower")}, {"x": m(0.10, "share", "lower")})
    assert out["x"]["verdict"] == "worse"


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
    rows = {r["metric"]: r for r in bench.check_budgets(now, budgets, ran={"imports", "physics.dog", "viewer", "mcp"})}
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
    assert rows["install.installed_mb"]["status"] == "not run"  # that part did not run
    # Without knowing what ran, a missing number is unmeasured: never quietly fine
    assert bench.check_budgets(now, {"install.installed_mb": {"max": 200}})[0]["status"] == "unmeasured"
    with pytest.raises(ValueError, match="needs a max or a min"):
        bench.check_budgets(now, {"viewer.fps": {"why": "no limit"}})


def test_a_value_at_its_limit_is_within_budget(bench):
    m = bench.metric
    rows = bench.check_budgets(
        {"a.x": m(10, "ms", "lower"), "a.y": m(5, "fps", "higher")}, {"a.x": {"max": 10}, "a.y": {"min": 5}}
    )
    assert [r["status"] for r in rows] == ["ok", "ok"]


def _unit(name: str) -> str:
    for end, unit in (("steps_per_s", "steps/s"), ("share", "share"), ("_mb", "MB"), ("_kb", "KB"), ("_pct", "%")):
        if name.endswith(end):
            return unit
    return {"fps": "fps"}.get(name.rsplit(".", 1)[-1]) or (
        "ms" if name.endswith("_ms") else "s" if name.endswith("_s") else ""
    )


def _results(bench, value: float = 1.0, only: set[str] | None = None) -> dict:
    """Results with every metric a complete run measures (or those of the parts in `only`)."""
    out: dict = {}
    for name in sorted(bench.known_metrics()):
        if only is None or bench.part_of(name) in only:
            section, _, rest = name.partition(".")
            out.setdefault(section, {})[rest] = bench.metric(value, _unit(name), "lower")
    return out


def _record(bench, started: str, **extra) -> str:
    return json.dumps({"started": started, "results": _results(bench), **extra})


def test_the_previous_run_is_the_newest_by_its_start_not_by_file_name(bench, tmp_path):
    assert bench.previous_run(tmp_path) is None
    (tmp_path / "2026-10-18-aaaaaaa.json").write_text(_record(bench, "2026-10-18T09:00:00+08:00"))
    (tmp_path / "2026-10-17-zzzzzzz.json").write_text(_record(bench, "2026-10-18T21:00:00+08:00"))
    (tmp_path / "budget.json").write_text(json.dumps({"started": "2099-01-01T00:00:00+08:00", "budgets": {}}))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "notes.json").write_text(json.dumps({"something": "else"}))
    (tmp_path / "odd.json").write_text(json.dumps({"started": "yesterday", "results": {}}))
    assert bench.previous_run(tmp_path).name == "2026-10-17-zzzzzzz.json"


def test_runs_are_ordered_by_their_moment_across_utc_offsets(bench, tmp_path):
    """09:00 at UTC-7 is 16:00 UTC, later than 10:00 at UTC+8 (02:00 UTC), though "09" sorts first as text."""
    (tmp_path / "a.json").write_text(_record(bench, "2026-10-18T09:00:00-07:00"))
    (tmp_path / "b.json").write_text(_record(bench, "2026-10-18T10:00:00+08:00"))
    assert bench.previous_run(tmp_path).name == "a.json"


def test_a_partial_or_quick_run_is_never_the_previous_run(bench, tmp_path):
    """A one-world run must not become what the next complete run compares with."""
    (tmp_path / "2026-10-18-full.json").write_text(_record(bench, "2026-10-18T09:00:00+08:00"))
    partial = {"started": "2026-10-18T10:00:00+08:00", "partial": True, "results": _results(bench)}
    (tmp_path / "2026-10-18-new-partial.json").write_text(json.dumps(partial))
    (tmp_path / "quick.json").write_text(_record(bench, "2026-10-18T11:00:00+08:00", quick=True))
    # Saved before records said whether they were partial: only the dog in its physics, as the bug left it
    stub = {"started": "2026-10-18T12:00:00+08:00", "results": _results(bench, only={"physics.dog_idle"})}
    (tmp_path / "2026-10-18-old.json").write_text(json.dumps(stub))
    errored = {"started": "2026-10-18T13:00:00+08:00", "results": {**_results(bench), "viewer": {"error": "no"}}}
    (tmp_path / "errored.json").write_text(json.dumps(errored))
    assert bench.previous_run(tmp_path).name == "2026-10-18-full.json"
    assert bench.missing_parts(stub["results"])[:4] == ["install", "imports", "viewer", "mcp"]


def _fake_main(bench, monkeypatch, results: dict, load: float = 1.0):
    errors = [f"{part}: {found['error']}" for part, found in results.items() if "error" in found]  # as run() says
    monkeypatch.setattr(bench, "run", lambda sections, quick, shots, worlds=None: (results, errors))
    monkeypatch.setattr(bench, "git_sha", lambda: "abc1234")
    monkeypatch.setattr(bench, "machine", lambda: {"load": [load, load, load]})
    monkeypatch.setattr(bench.os, "getloadavg", lambda: (load, load, load))


def test_a_partial_run_neither_replaces_a_complete_one_nor_becomes_the_baseline(bench, tmp_path, monkeypatch, capsys):
    """The run that lost objects_20: a complete record, then `--only physics --worlds dog_idle` at the same commit
    on the same day wrote over it, and the next run compared with the stub."""
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"budgets": {}}))
    out = tmp_path / "runs"
    _fake_main(bench, monkeypatch, _results(bench))
    assert bench.main(["--out", str(out), "--budget", str(budget)]) == 0
    (full,) = out.glob("*.json")
    assert full.name.endswith("-abc1234.json")
    complete = full.read_text()

    _fake_main(bench, monkeypatch, _results(bench, 2.0, only={"physics.dog_idle"}))
    args = ["--only", "physics", "--worlds", "dog_idle", "--out", str(out), "--budget", str(budget)]
    assert bench.main(args) == 0
    assert "not saved: a quick or partial run" in capsys.readouterr().out
    assert [p.name for p in out.glob("*.json")] == [full.name] and full.read_text() == complete

    assert bench.main([*args, "--save"]) == 0
    partial = out / full.name.replace(".json", "-partial.json")
    assert {p.name for p in out.glob("*.json")} == {full.name, partial.name}
    assert full.read_text() == complete
    assert json.loads(partial.read_text())["covers"] == ["physics.dog_idle", "physics.machine"]
    assert bench.previous_run(out) == full


def test_exit_codes_say_over_busy_or_unchecked(bench, tmp_path, monkeypatch, capsys):
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "budgets": {
                    "physics.dog_idle.steps_per_s": {"min": 5.0, "why": "a speed"},
                    "imports.cli_heavy": {"max": 0, "why": "a count"},
                    "mcp.idle_cpu_pct": {"max": 2, "why": "needs ps"},
                }
            }
        )
    )
    m = bench.metric

    def results(speed: float = 10.0, heavy: int = 0, idle: float | None = 0.5) -> dict:
        found = _results(bench)
        found["physics"]["dog_idle.steps_per_s"] = m(speed, "steps/s", "higher")
        found["imports"]["cli_heavy"] = m(heavy, "", "lower")
        found["mcp"]["idle_cpu_pct"] = m(idle, "%", "lower")
        return found

    args = ["--no-save", "--budget", str(budget), "--out", str(tmp_path / "runs")]

    def failing(found: dict, part: str) -> dict:
        return {**found, part: {"error": "RuntimeError: Google Chrome not found"}}

    cases = [
        (results(), 1.0, 0),
        (results(speed=1.0), 1.0, 1),  # too slow on a calm machine: a regression
        (results(speed=1.0), 999.0, 3),  # too slow on a busy one: cannot tell
        (results(speed=1.0, heavy=1), 999.0, 1),  # a count is a count, however busy the machine
        (results(idle=None), 1.0, 2),  # no ps: the budget went unchecked, and says so
        # Over budget wins over a part that failed or a budget unchecked: neither may hide a regression
        (failing(results(heavy=1), "viewer"), 1.0, 1),
        (results(heavy=1, idle=None), 1.0, 1),
        (failing(results(), "viewer"), 1.0, 2),
        (failing(results(speed=1.0), "viewer"), 999.0, 2),  # a failure outranks a busy machine's maybe
    ]
    for found, load, code in cases:
        _fake_main(bench, monkeypatch, found, load)
        assert bench.main(args) == code, (load, code)
    assert "UNMEASURED  mcp.idle_cpu_pct" in capsys.readouterr().out


def test_a_slower_robotforge_is_over_budget_not_a_busy_machine(bench, tmp_path, monkeypatch, capsys):
    """One extra mj_forward per step in Session.step: objects_20 and the dog ran half as fast while the machine
    check, in the same processes, ran as fast as ever. That is a regression (exit 1), not a busy machine (3).
    Numbers from that run; floors from the real budget.json."""
    real = json.loads(BUDGET.read_text(encoding="utf-8"))["budgets"]
    names = ("physics.objects_20.steps_per_s", "physics.dog_idle.steps_per_s", bench.MACHINE)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"budgets": {name: real[name] for name in names}}))
    m = bench.metric
    found = _results(bench)
    found["physics"]["objects_20.steps_per_s"] = m(4652, "steps/s", "higher", machine=222000)
    found["physics"]["dog_idle.steps_per_s"] = m(29533, "steps/s", "higher", machine=225000)
    found["physics"]["machine.steps_per_s"] = m(222000, "steps/s", "higher", slowest="objects_20")
    _fake_main(bench, monkeypatch, found, load=5.0)
    assert bench.main(["--no-save", "--budget", str(budget), "--out", str(tmp_path / "runs")]) == 1
    out = capsys.readouterr().out
    assert "OVER BUDGET  physics.objects_20.steps_per_s" in out and "OVER BUDGET  physics.dog_idle.steps_per_s" in out
    assert "INCONCLUSIVE" not in out and "busy" not in out
    # The same speeds with the machine check under its floor: then the machine was slow, and nobody can tell
    found["physics"]["machine.steps_per_s"] = m(120000, "steps/s", "higher", slowest="objects_20")
    assert bench.main(["--no-save", "--budget", str(budget), "--out", str(tmp_path / "runs")]) == 3
    assert "the machine check ran 120,000 steps/s, under its floor" in capsys.readouterr().out


def test_a_budget_on_a_metric_that_does_not_exist_stops_the_run(bench, tmp_path, monkeypatch, capsys):
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"budgets": {"physics.dog_idle.step_per_s": {"min": 5, "why": "typo"}}}))
    with pytest.raises(ValueError, match="step_per_s"):
        bench.load_budgets(budget)
    monkeypatch.setattr(bench, "run", lambda *a, **k: pytest.fail("measured despite a broken budget file"))
    assert bench.main(["--no-save", "--budget", str(budget)]) == 2
    assert "physics.dog_idle.step_per_s" in capsys.readouterr().out


def test_a_busy_machine_makes_timings_over_budget_inconclusive(bench):
    m = bench.metric
    budgets = {bench.MACHINE: {"min": 165000}, "physics.objects_20.steps_per_s": {"min": 8500}}
    calm = {bench.MACHINE: m(220000, "steps/s", "higher")}
    slow = {bench.MACHINE: m(120000, "steps/s", "higher")}
    assert bench.busy_reason([5.0, 8.0], 18, calm, budgets) is None
    assert "load average 40 on 18 cores" in bench.busy_reason([12.0, 40.0], 18, calm, budgets)
    assert "under its floor" in bench.busy_reason([5.0], 18, slow, budgets)
    # A world under its floor says nothing about the machine, however slow: it may be the code
    world_slow = {**calm, "physics.objects_20.steps_per_s": m(4652, "steps/s", "higher")}
    assert bench.busy_reason([5.0], 18, world_slow, budgets) is None
    now = {
        "physics.dog_idle.steps_per_s": m(10, "steps/s", "higher"),
        "physics.dog_idle.peak_mb": m(900, "MB", "lower"),
        "imports.cli_heavy": m(2, "", "lower"),
        "physics.dog_idle.controllers_share": m(0.9, "share", "lower"),
    }
    budgets = {name: {"min": 1000} if name.endswith("steps_per_s") else {"max": 0.5} for name in now}
    rows = {r["metric"]: r["status"] for r in bench.check_budgets(now, budgets, busy=True)}
    assert rows == {
        "physics.dog_idle.steps_per_s": "inconclusive",
        "physics.dog_idle.peak_mb": "over",
        "imports.cli_heavy": "over",
        "physics.dog_idle.controllers_share": "over",
    }


def test_speeds_from_runs_without_the_machine_check_compare_as_measured(bench):
    """Older runs have no machine check: no world stands in for it (objects_20 runs robotforge's step too), so their
    speeds compare as they are, with the wide noise of raw timings."""
    m = bench.metric
    before = {
        "physics.objects_20.steps_per_s": m(10000, "steps/s", "higher"),
        "physics.dog_idle.steps_per_s": m(100000, "steps/s", "higher"),
        "physics.go2_walk.steps_per_s": m(20000, "steps/s", "higher", machine=220000),
    }
    now = {
        "physics.objects_20.steps_per_s": m(6000, "steps/s", "higher", machine=130000),
        "physics.dog_idle.steps_per_s": m(80000, "steps/s", "higher", machine=130000),
        "physics.go2_walk.steps_per_s": m(15000, "steps/s", "higher", machine=220000),
    }
    out = bench.compare(now, before)
    assert out["physics.objects_20.steps_per_s"] == {"before": 10000.0, "change": -0.4, "verdict": "worse"}
    assert out["physics.dog_idle.steps_per_s"] == {"before": 100000.0, "change": -0.2, "verdict": "same"}
    # Both runs timed the check with go2_walk: relative, and the machine did not move, so 25 percent is the code
    assert out["physics.go2_walk.steps_per_s"]["relative"] and out["physics.go2_walk.steps_per_s"]["verdict"] == "worse"


def test_speeds_compare_relative_to_the_machine_check_in_their_own_process(bench):
    """The load drifts within a run: the dog, timed first, saw a machine half as fast; the control, timed last,
    a machine as fast as before. The machine check timed with each world tells them apart."""
    m = bench.metric
    before = {
        "physics.objects_20.steps_per_s": m(10000, "steps/s", "higher", machine=280000),
        "physics.dog_idle.steps_per_s": m(100000, "steps/s", "higher", machine=280000),
    }
    now = {
        "physics.objects_20.steps_per_s": m(10000, "steps/s", "higher", machine=280000),
        "physics.dog_idle.steps_per_s": m(52000, "steps/s", "higher", machine=140000),
    }
    out = bench.compare(now, before)
    assert out["physics.dog_idle.steps_per_s"]["verdict"] == "same" and out["physics.dog_idle.steps_per_s"]["relative"]
    assert out["physics.objects_20.steps_per_s"]["verdict"] == "same"
    now["physics.dog_idle.steps_per_s"] = m(52000, "steps/s", "higher", machine=280000)  # same machine: the code
    assert bench.compare(now, before)["physics.dog_idle.steps_per_s"]["verdict"] == "worse"
    # Over the machine check, speeds wobble less, so a 20 percent change shows; 10 percent is still noise
    now["physics.dog_idle.steps_per_s"] = m(80000, "steps/s", "higher", machine=280000)
    assert bench.compare(now, before)["physics.dog_idle.steps_per_s"]["verdict"] == "worse"
    now["physics.dog_idle.steps_per_s"] = m(90000, "steps/s", "higher", machine=280000)
    assert bench.compare(now, before)["physics.dog_idle.steps_per_s"]["verdict"] == "same"


def test_the_machine_check_costs_the_same_every_time(bench):
    """Restarted from the same pose, the check's best time is steady on one machine (a busy one may move it some)."""
    rates = []
    for _ in range(3):
        check = bench.MachineCheck(slices=20)
        for _repeat in range(3):
            for k in range(20):
                check.time(k)
        rates.append(check.steps_per_s)
    assert max(rates) / min(rates) < 1.25, rates


def test_a_stale_lock_says_what_uv_found(bench, monkeypatch):
    """uv pip check prints the broken requirements on stderr, not stdout: the message must carry them."""
    found = "The package `httpx` requires `certifi`, but it's not installed"

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=found + "\n")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="uv.lock does not satisfy the wheel") as caught:
        bench._pip_check("uv", Path("python"))
    assert str(caught.value).endswith(found)
    monkeypatch.setattr(bench.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "", ""))
    bench._pip_check("uv", Path("python"))


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
    known = bench.known_metrics()
    for name, budget in data["budgets"].items():
        assert name in known, f"{name}: no part of the bench measures it"
        assert ("max" in budget) != ("min" in budget), f"{name}: give a max or a min"
        assert budget.get("why"), f"{name}: say where the number comes from"
    assert bench.MACHINE in data["budgets"], "the machine check needs a floor, or a busy machine goes unnoticed"


def test_the_known_metrics_are_what_a_real_run_measured(bench):
    """The newest complete run in docs/bench, measured for real, names no metric the list lacks."""
    newest = bench.previous_run(BUDGET.parent)
    assert newest is not None, "docs/bench has no complete run"
    measured = set(bench.flatten(json.loads(newest.read_text(encoding="utf-8"))["results"]))
    assert measured <= bench.known_metrics(), sorted(measured - bench.known_metrics())
