"""How light is robotforge? Measure it the same way every time and hold it to a budget.

    uv run python scripts/bench.py [--only install imports physics viewer mcp] [--worlds dog_idle ...]
        [--quick] [--save | --no-save] [--wait-calm MIN] [--budget docs/bench/budget.json]
        [--against docs/bench/<run>.json] [--shots <folder>]
    uv run python scripts/bench.py --ab <rev> [--worlds ...] [--rounds 3]

Measures, each in fresh processes:
- install: the built wheel, and the wheel plus its runtime dependencies at the versions uv.lock pins, installed in a
  clean uv venv in a temp folder (bytes on disk, packages with their versions, the biggest ones, and the download
  those wheels make);
- imports: `import robotforge`, `robotforge.cli` (what every command loads) and `robotforge.mcp_server`, best of
  several runs, the packages that take the time (python -X importtime) and which heavy modules got loaded;
- physics: headless steps per second (process time, so a busy machine slows it less; each 50 ms slice the best
  of several repeats) for standard worlds, the share of that time spent in controllers, and peak memory;
- viewer: cold start from running `robotforge open <world>` to the first frame drawn in headless Chrome, frames
  per second and the time one frame takes to draw, and the viewer server's peak memory;
- mcp: `robotforge mcp` started to tools listed, the latency of common calls, and the server's memory.

Writes docs/bench/<date>-<sha>.json, prints a table that compares with the previous complete run there and with
the budgets in docs/bench/budget.json. Exits 1 when a budget is exceeded (even if another part failed), else 2 when a
measurement failed or a budget went unchecked, else 3 when only timings went over on a busy machine, else 0. Busy:
the load average above the cores, or the machine check (a MuJoCo model built here, no robotforge code, timed in
each world's process) under its floor in budget.json; then timings over budget say nothing about the code: rerun
calm. A quick or partial run (--quick, --only, --worlds) is not saved unless --save, and then as
<date>-<sha>-partial.json, which no later run compares with. Physics speeds are compared relative to that machine
check, so a busier machine than last time reads less as a slower robotforge (a world moves a little more than the
check, so the noise floor widens where the check moved a lot); --wait-calm waits for the load to drop first. Two runs an
hour apart on a shared machine can still differ more than a change does, so --ab measures a change instead:
imports, cold start and the physics worlds on this tree and on revision <rev>, alternately.
The physics, viewer and MCP parts use the real Menagerie Go2 (downloaded once into the cache); the viewer part
needs Google Chrome (or CHROME=...). Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "docs" / "bench"
BUDGET_FILE = BENCH_DIR / "budget.json"
SECTIONS = ("install", "imports", "physics", "viewer", "mcp")
# Modules robotforge must not load until a feature needs them: mesh import and decomposition, image handling and
# the learning stack. The MCP server and the viewer start without any of them.
HEAVY_MODULES = ("trimesh", "manifold3d", "coacd", "PIL", "gymnasium", "torch", "stable_baselines3", "onnxruntime")
IMPORT_TARGETS = ("robotforge", "robotforge.cli", "robotforge.mcp_server")
TOP_N = 8  # rows of "where the time or the bytes go" kept in a result
# A change within this share or this amount is noise, by unit: sizes on disk are exact; speeds and times on a shared
# machine are not (the same commit measured 103,000 and 75,000 steps/s for the idle dog at the same load average,
# minutes apart), and a 2 ms call that takes 3 ms once is not a regression. Measured memory wobbles a few percent.
NOISE = {
    "MB": (0.02, 1.0),
    "KB": (0.02, 10.0),
    "": (0.0, 0.0),
    "ms": (0.3, 2.0),
    "s": (0.3, 0.05),
    "%": (0.3, 1.0),
    "share": (0.3, 0.02),  # of step time in controllers: 0.005 against 0.012 is no change worth a line
}
TIME_NOISE = (0.3, 0.0)
MEMORY_NOISE = (0.05, 5.0)  # peak and resident memory of a process, in MB
# Speeds over the machine check timed with each world (MachineCheck), against 0d0b444 on unchanged physics code: in
# 72 world runs at load 21 to 161, -11 to +4 percent where that world's check moved less than a quarter (raw speeds
# -20 to +28), and -23 to +7 where it moved more (raw -53 to +57; at 15 percent, 8 of those 54 read "worse"): a
# world slows or speeds 1.2 to 1.4 times as much as the small check, more so with Python controllers (objects_20,
# with none, stayed within 2 percent), so the further the machine moved, the more a relative change leans its way.
MACHINE_RELATIVE_NOISE = 0.15
MACHINE_MOVED = 1.25  # the check this much faster or slower than in the run compared with
MACHINE_MOVED_NOISE = 0.25
# A process that compiles the Go2's OBJ meshes peaks 480 MB one run and 535 MB the next on the same build.
MESH_MEMORY_NOISE = (0.15, 20.0)
# Units whose numbers a busy machine changes: over budget on a busy machine says nothing about the code. The
# controllers' share is a ratio of two timings, but a busy machine slows Python more than MuJoCo's C: at load 161
# the standing dog's share went from 0.12 to 0.19, over its 0.18, with no code change.
LOAD_SENSITIVE = ("steps/s", "ms", "s", "fps", "%", "share")
# How fast this machine ran during the run: the slowest of the machine checks timed in each world's process
# (MachineCheck, a model built in this script and stepped by MuJoCo alone). No world can serve: even objects_20,
# which has no controllers, is built by robotforge (its solver settings set most of its cost) and steps through
# Session.step (1.1 us of its 134 us per step), so a slowdown in that shared code would read as a busy machine.
MACHINE = "physics.machine.steps_per_s"

# Every metric each part can produce, so budget.json cannot name one that does not exist (a typo would turn a
# budget off without a word) and a renamed metric shows up as an error, not as a budget gone quiet.
INSTALL_METRICS = ("wheel_kb", "installed_mb", "packages", "download_mb")
IMPORT_METRICS = ("ms", "cpu_ms", "heavy")  # per import target, e.g. imports.cli_cpu_ms
PHYSICS_METRICS = ("steps_per_s", "controllers_share", "peak_mb")  # per world, e.g. physics.dog_idle.peak_mb
VIEWER_METRICS = ("cold_start_s", "fps", "frame_ms", "server_peak_mb", "idle_cpu_pct")
MCP_CALLS = ("new_world", "add_robot", "move", "observe")
MCP_METRICS = ("start_s", *(f"{call}_ms" for call in MCP_CALLS), "server_rss_mb", "idle_cpu_pct")


# ---- results: metrics, comparison, budgets ---------------------------------------------------------------


def metric(value: float | int | None, unit: str, better: str, **info: Any) -> dict[str, Any]:
    """One measured number. `better` is "lower" or "higher"; `info` is context kept with it (small)."""
    out: dict[str, Any] = {"value": None if value is None else round(float(value), 4), "unit": unit, "better": better}
    out.update(info)
    return out


def flatten(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{section: {name: metric}} -> {"section.name": metric}, skipping a section's notes and errors."""
    return {
        f"{section}.{name}": m
        for section, found in results.items()
        for name, m in found.items()
        if isinstance(m, dict) and "value" in m
    }


def import_key(target: str) -> str:
    """robotforge.cli -> cli: how an import target is named in the results."""
    return target.removeprefix("robotforge.")


def known_metrics() -> set[str]:
    """Every "section.name" a complete run measures."""
    names = {f"install.{n}" for n in INSTALL_METRICS}
    names |= {f"imports.{import_key(t)}_{n}" for t in IMPORT_TARGETS for n in IMPORT_METRICS}
    names |= {f"physics.{w}.{n}" for w in WORLDS for n in PHYSICS_METRICS} | {MACHINE}
    names |= {f"viewer.{n}" for n in VIEWER_METRICS}
    return names | {f"mcp.{n}" for n in MCP_METRICS}


def part_of(name: str) -> str:
    """The part of a run a metric comes from: its section, or physics.<world> for a physics metric."""
    section, _, rest = name.partition(".")
    return f"physics.{rest.split('.')[0]}" if section == "physics" else section


def parts(sections: list[str], worlds: list[str]) -> set[str]:
    """The parts a run with these sections and physics worlds covers."""
    out = {s for s in sections if s != "physics"}
    return out | ({part_of(MACHINE), *(f"physics.{w}" for w in worlds)} if "physics" in sections else set())


def missing_parts(results: dict[str, Any]) -> list[str]:
    """What a complete run has and these results do not (or got only an error for)."""
    missing = [s for s in SECTIONS if not isinstance(results.get(s), dict) or "error" in results[s]]
    if "physics" not in missing:
        missing += [f"physics.{w}" for w in WORLDS if f"{w}.steps_per_s" not in results["physics"]]
    return missing


def is_baseline(record: dict[str, Any]) -> bool:
    """Whether later runs may compare with this one: complete and not quick. Records say whether they are partial;
    older ones did not, so for those it is read from their results."""
    if record.get("quick") or record.get("partial"):
        return False
    return "partial" in record or not missing_parts(record.get("results", {}))


def _noise(m: dict[str, Any]) -> tuple[float, float]:
    if m.get("noise"):
        return tuple(m["noise"])
    return NOISE.get(m.get("unit", ""), TIME_NOISE)


def compare(now: dict[str, dict[str, Any]], before: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per metric: the previous value, the change and a verdict (better, worse, same, new), from flattened results.
    A world's speed changes relative to the machine ("relative": True) when both runs timed the machine check in
    that world's process: a world that slowed by as much as the machine did is the machine, not robotforge; when
    the machine moved a lot, the world moved a little more, so the floor widens. Runs from before the check
    compare as measured, with the wider noise of raw timings."""
    out = {}
    for name, m in now.items():
        old = before.get(name)
        value = m.get("value")
        if old is None or old.get("value") is None or value is None:
            out[name] = {"before": None, "change": None, "verdict": "new"}
            continue
        prev = float(old["value"])
        scale, (share, amount) = None, _noise(m)
        if m.get("unit") == "steps/s" and m.get("machine") and old.get("machine"):
            scale = m["machine"] / old["machine"]
            moved = not 1 / MACHINE_MOVED <= scale <= MACHINE_MOVED
            share, amount = (MACHINE_MOVED_NOISE if moved else MACHINE_RELATIVE_NOISE), 0.0
        relative = scale is not None
        ref = prev * scale if relative else prev
        change = (value - ref) / abs(ref) if ref else (0.0 if value == ref else math.inf)
        if abs(change) <= share or abs(value - ref) <= amount:
            verdict = "same"
        else:
            verdict = "better" if (change < 0) == (m.get("better") == "lower") else "worse"
        out[name] = {"before": prev, "change": round(change, 4), "verdict": verdict}
        if relative:
            out[name]["relative"] = True
    return out


def busy_reason(loads: list[float], cores: int, now: dict[str, dict[str, Any]], budgets: dict[str, Any]) -> str | None:
    """Why this run's timings cannot be held to their budgets, or None: the load average went above the number of
    cores, or the machine check ran under its floor (then the machine, not the code, was slow). Only the check
    tells: it runs no robotforge code, so a slowdown in robotforge leaves it where it was and fails the budgets."""
    peak = max(loads, default=0.0)
    if peak > cores:
        return f"load average {peak:.0f} on {cores} cores"
    value, floor = (now.get(MACHINE) or {}).get("value"), (budgets.get(MACHINE) or {}).get("min")
    if value is not None and floor is not None and value < floor:
        return f"the machine check ran {value:,.0f} steps/s, under its floor of {floor:,.0f}"
    return None


def check_budgets(
    now: dict[str, dict[str, Any]],
    budgets: dict[str, dict[str, Any]],
    ran: set[str] | None = None,
    busy: bool = False,
) -> list[dict[str, Any]]:
    """Each budget against its metric: ok, over (exceeded), inconclusive (a timing over on a busy machine),
    unmeasured (its part ran but gave no number) or not run (`ran`, the parts this run covered, left it out)."""
    out = []
    for name, budget in budgets.items():
        m = now.get(name)
        value = None if m is None else m.get("value")
        row: dict[str, Any] = {"metric": name, "value": value, "why": budget.get("why", "")}
        if "max" in budget:
            row["limit"], row["kind"] = float(budget["max"]), "max"
        elif "min" in budget:
            row["limit"], row["kind"] = float(budget["min"]), "min"
        else:
            raise ValueError(f"budget {name!r} needs a max or a min")
        if value is None:
            row["status"] = "unmeasured" if ran is None or part_of(name) in ran else "not run"
        elif (row["kind"] == "max" and value > row["limit"]) or (row["kind"] == "min" and value < row["limit"]):
            row["status"] = "inconclusive" if busy and m.get("unit") in LOAD_SENSITIVE else "over"
        else:
            row["status"] = "ok"
        out.append(row)
    return out


def load_budgets(path: Path) -> dict[str, dict[str, Any]]:
    """The budgets in `path` (none when there is no file); a budget on a metric the bench does not measure is an
    error, so a typo cannot switch a budget off."""
    if not path.is_file():
        return {}
    budgets = json.loads(path.read_text(encoding="utf-8")).get("budgets", {})
    unknown = sorted(set(budgets) - known_metrics())
    if unknown:
        raise ValueError(f"{path.name} has budgets on metrics the bench does not measure: {', '.join(unknown)}")
    return budgets


def wait_for_calm(seconds: float) -> None:
    """Wait (at most `seconds`) until the last minute's load average drops below the number of cores."""
    if not hasattr(os, "getloadavg"):
        return
    end, cores = time.monotonic() + seconds, os.cpu_count() or 1
    while os.getloadavg()[0] >= cores and time.monotonic() < end:
        print(f"waiting for a calm machine: load {os.getloadavg()[0]:.0f} on {cores} cores", file=sys.stderr)
        time.sleep(30)


def previous_run(directory: Path) -> Path | None:
    """The newest complete run in `directory` by the time it records (partial and quick runs, and the budget
    file, are not runs to compare with)."""
    runs = []
    for p in directory.glob("*.json"):
        if p.name == BUDGET_FILE.name:
            continue
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
            # astimezone: times with different UTC offsets, and one without (taken as local), compare correctly
            started = dt.datetime.fromisoformat(record["started"]).astimezone()
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            continue
        if is_baseline(record):
            runs.append((started, p.name, p))
    return max(runs)[2] if runs else None


def run_name(started: dt.datetime, sha: str, partial: bool = False) -> str:
    return f"{started:%Y-%m-%d}-{sha}{'-partial' if partial else ''}.json"


# ---- the table ----------------------------------------------------------------------------------------------


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "-"
    if unit in ("steps/s", "KB") or (unit == "" and float(value).is_integer()):
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}" if unit in ("", "share") else f"{value:.2f}"


def table(
    now: dict[str, dict[str, Any]],
    compared: dict[str, dict[str, Any]],
    budgets: list[dict[str, Any]],
    busy: bool = False,
) -> list[str]:
    """One row per metric. On a busy machine raw timings get no better or worse (the change is shown, not
    judged); speeds over the machine check still do, since the check moved with the machine."""
    by_metric = {b["metric"]: b for b in budgets}
    lines = [f"{'metric':<40} {'now':>10} {'unit':<8} {'before':>10} {'change':>8}  {'budget':<14} status"]
    for name, m in now.items():
        c = compared.get(name, {})
        b = by_metric.get(name)
        change = "" if c.get("change") is None else f"{100 * c['change']:+.0f}%{'*' if c.get('relative') else ''}"
        limit = "" if b is None else f"{'<=' if b['kind'] == 'max' else '>='} {_fmt(b['limit'], m['unit'])}"
        status = "" if b is None else b["status"]
        unjudged = busy and m["unit"] in LOAD_SENSITIVE and not c.get("relative")
        if c.get("verdict") in ("better", "worse") and not unjudged:
            status = f"{status} ({c['verdict']})".strip()
        lines.append(
            f"{name:<40} {_fmt(m['value'], m['unit']):>10} {m['unit']:<8} "
            f"{_fmt(c.get('before'), m['unit']):>10} {change:>8}  {limit:<14} {status}"
        )
    return lines


# ---- small helpers ------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
    return out.stdout.strip()


def git_sha() -> str:
    """The commit measured; "-dirty" when the code (src, scripts) or its dependencies (pyproject.toml, uv.lock)
    have changes not committed yet."""
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    changed = _git("status", "--porcelain", "--", "src", "scripts", "pyproject.toml", "uv.lock")
    return f"{sha}-dirty" if changed else sha


def machine() -> dict[str, Any]:
    import mujoco
    import numpy

    info: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.mac_ver()[0] or platform.release(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
    }
    with contextlib.suppress(OSError, AttributeError):
        info["load"] = [round(v, 1) for v in os.getloadavg()]
    return info


def _rss_mb(ru_maxrss: int) -> float:
    """ru_maxrss is in bytes on macOS and in kilobytes on Linux."""
    return ru_maxrss / 2**20 if sys.platform == "darwin" else ru_maxrss / 1024


def _robotforge_command() -> str:
    found = shutil.which("robotforge", path=str(Path(sys.executable).parent))
    if not found:
        raise RuntimeError("the robotforge command is not installed next to this Python (uv sync)")
    return found


def _child_env(home: Path) -> dict[str, str]:
    return {**os.environ, "ROBOTFORGE_HOME": str(home), "ROBOTFORGE_NO_BROWSER": "1", "PYTHONUNBUFFERED": "1"}


def _with_source(env: dict[str, str], src: Path | None) -> dict[str, str]:
    """`env` with robotforge's source taken from `src` (another commit's), ahead of the installed one."""
    if src is not None:
        env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _stop(proc: subprocess.Popen) -> float | None:
    """Stop a child and return its peak memory (MB) where the platform tells (os.wait4)."""
    if proc.poll() is None:
        proc.terminate()
    if hasattr(os, "wait4") and proc.returncode is None:
        try:
            _, status, usage = os.wait4(proc.pid, 0)
        except ChildProcessError:
            proc.wait()
            return None
        proc.returncode = os.waitstatus_to_exitcode(status)
        return round(_rss_mb(usage.ru_maxrss), 1)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)
    if proc.returncode is None:
        proc.kill()
        proc.wait()
    return None


def _ps(*args: str) -> subprocess.CompletedProcess | None:
    """ps (macOS and Linux); None where there is none (Windows), and those numbers go unmeasured."""
    try:
        return subprocess.run(["ps", *args], capture_output=True, text=True, check=False)
    except OSError:
        return None


def _cpu_seconds(pid: int) -> float | None:
    """Processor time a process has used so far (s): /proc on Linux, ps elsewhere."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    done = _ps("-o", "time=", "-p", str(pid))
    text = done.stdout.strip() if done else ""
    if not text:
        return None
    days, _, clock = text.rpartition("-")
    seconds = 0.0
    for part in clock.split(":"):
        seconds = seconds * 60 + float(part)
    return seconds + 86400 * int(days or 0)


def _cpu_share(pid: int, seconds: float) -> float | None:
    """The share of one core a process uses over the next `seconds` (percent)."""
    before = _cpu_seconds(pid)
    time.sleep(seconds)
    after = _cpu_seconds(pid)
    return None if before is None or after is None else 100.0 * (after - before) / seconds


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float, proc: subprocess.Popen | None = None) -> None:
    import urllib.request

    end = time.monotonic() + timeout
    while True:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"the server exited with code {proc.returncode} before it answered")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                resp.read()
                return
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(0.02)


# ---- install ------------------------------------------------------------------------------------------------


def _uv_install(command: list[str]) -> None:
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=False)
    if done.returncode != 0:  # no network: what the uv cache has will do
        done = subprocess.run([*command, "--offline"], capture_output=True, text=True, encoding="utf-8", check=False)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(command[1:3])} failed: {done.stderr.strip()[-400:]}")


def bench_install(quick: bool) -> dict[str, Any]:
    """The wheel and its runtime dependencies at the versions uv.lock pins, so the size moves only when robotforge
    or its lock does (a plain `uv pip install <wheel>` would take whatever upstream released since)."""
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found on PATH")
    with tempfile.TemporaryDirectory(prefix="rf-bench-install-") as tmp:
        tmp_path = Path(tmp)
        dist = tmp_path / "dist"
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(dist), str(ROOT)], check=True, capture_output=True)
        wheel = next(dist.glob("robotforge-*.whl"))
        locked = tmp_path / "locked.txt"
        export = [uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "-o", str(locked)]
        subprocess.run(export, cwd=ROOT, check=True, capture_output=True)
        venv = tmp_path / "venv"
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run([uv, "venv", "--python", version, str(venv)], check=True, capture_output=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        _uv_install([uv, "pip", "install", "--python", str(python), "-r", str(locked)])
        _uv_install([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
        _pip_check(uv, python)
        site = Path(
            subprocess.run(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            ).stdout.strip()
        )
        found = _installed(site)
        total = sum(size for _version, size in found.values())
        download = _download_size(found)
        top = sorted(found.items(), key=lambda kv: -kv[1][1])[:TOP_N]
        versions = sorted(f"{name}=={version}" for name, (version, _size) in found.items())
        return {
            "wheel_kb": metric(wheel.stat().st_size / 1024, "KB", "lower"),
            "installed_mb": metric(
                total / 2**20,
                "MB",
                "lower",
                top=[[f"{name}=={version}", round(size / 2**20, 1)] for name, (version, size) in top],
            ),
            "packages": metric(len(found), "", "lower", versions=versions),
            "download_mb": metric(None if download is None else download / 2**20, "MB", "lower"),
        }


def _pip_check(uv: str, python: Path) -> None:
    """The locked set must satisfy what the wheel asks for, or uv.lock is behind pyproject.toml. uv prints what
    is wrong on stderr."""
    check = subprocess.run(
        [uv, "pip", "check", "--python", str(python)], capture_output=True, text=True, encoding="utf-8", check=False
    )
    if check.returncode != 0:
        said = (check.stderr.strip() or check.stdout.strip()).splitlines()
        reason = " ".join(line for line in said if not line.startswith("Using Python"))[-400:]  # not the temp venv
        raise RuntimeError(f"uv.lock does not satisfy the wheel (uv lock?): {reason}")


def _installed(site: Path) -> dict[str, tuple[str, int]]:
    """Version and bytes on disk per installed distribution, from each one's RECORD (pip and setuptools left out)."""
    found: dict[str, tuple[str, int]] = {}
    claimed: set[Path] = set()
    for info in site.glob("*.dist-info"):
        name, _, version = info.name.removesuffix(".dist-info").partition("-")
        name = name.lower().replace("_", "-")
        if name in ("pip", "setuptools", "wheel"):
            continue
        total = 0
        record = info / "RECORD"
        lines = record.read_text(encoding="utf-8").splitlines() if record.is_file() else []
        for line in lines:
            rel = line.rsplit(",", 2)[0]
            path = (site / rel).resolve()
            if path in claimed or not path.is_file():
                continue
            claimed.add(path)
            total += path.stat().st_size
        found[name] = (version, total)
    return found


def _download_size(installed: dict[str, tuple[str, int]]) -> int | None:
    """Bytes of the wheels uv.lock lists for this platform, for the installed distributions at their versions."""
    try:
        import tomllib

        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        return None
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    supported = {str(t): rank for rank, t in enumerate(sys_tags())}
    total = 0
    for package in lock.get("package", []):
        name = package["name"].lower().replace("_", "-")
        if name not in installed or name == "robotforge":
            continue
        if package.get("version") != installed[name][0]:
            return None  # installed is not what the lock lists: its wheel sizes would describe another set
        best: tuple[int, int] | None = None
        for wheel in package.get("wheels", []):
            filename = wheel["url"].rsplit("/", 1)[-1]
            tags = parse_wheel_filename(filename)[3]
            rank = min((supported[str(t)] for t in tags if str(t) in supported), default=None)
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, int(wheel.get("size", 0)))
        if best is None:
            return None  # a package uv would build from source here: no wheel size to count
        total += best[1]
    return total


# ---- imports ------------------------------------------------------------------------------------------------

_IMPORT_PROBE = """
import json, sys, time
t0, c0 = time.perf_counter(), time.process_time()
import {target}
wall, cpu = time.perf_counter() - t0, time.process_time() - c0
heavy = [m for m in {heavy!r} if m in sys.modules]
print(json.dumps({{"wall": wall, "cpu": cpu, "heavy": heavy, "modules": len(sys.modules)}}))
"""


def _import_once(target: str, importtime: bool = False, src: Path | None = None) -> tuple[dict[str, Any], str]:
    code = _IMPORT_PROBE.format(target=target, heavy=HEAVY_MODULES)
    args = [sys.executable, *(["-X", "importtime"] if importtime else []), "-c", code]
    env = _with_source(dict(os.environ), src)
    # Run from a neutral folder, so `import robotforge` finds the installed package, not a folder named so
    done = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", check=False, cwd=tempfile.gettempdir(), env=env
    )
    if done.returncode != 0:
        raise RuntimeError(f"import {target} failed: {done.stderr.strip()[-400:]}")
    return json.loads(done.stdout.strip().splitlines()[-1]), done.stderr


def import_offenders(importtime_log: str, top: int = TOP_N) -> list[list[Any]]:
    """Where an import's time goes, by top-level package: self time summed (ms), the biggest first."""
    by_package: dict[str, int] = {}
    for line in importtime_log.splitlines():
        if not line.startswith("import time:") or "self [us]" in line:
            continue
        try:
            self_us, _cumulative, name = line.removeprefix("import time:").split("|")
        except ValueError:
            continue
        package = name.strip().split(".")[0]
        by_package[package] = by_package.get(package, 0) + int(self_us)
    ranked = sorted(by_package.items(), key=lambda kv: -kv[1])[:top]
    return [[name, round(us / 1000, 1)] for name, us in ranked]


def bench_imports(quick: bool) -> dict[str, Any]:
    runs = 3 if quick else 7
    out: dict[str, Any] = {}
    for target in IMPORT_TARGETS:
        _import_once(target)  # compiles any stale .pyc first, so every counted run reads cached bytecode
        samples = [_import_once(target)[0] for _ in range(runs)]
        best = min(samples, key=lambda s: s["wall"])
        _, log = _import_once(target, importtime=True)
        key = import_key(target)
        out[f"{key}_ms"] = metric(
            1000 * best["wall"], "ms", "lower", modules=best["modules"], offenders=import_offenders(log)
        )
        # Process time: what the import costs this machine, less disturbed by other work on it than the wall clock
        out[f"{key}_cpu_ms"] = metric(1000 * min(s["cpu"] for s in samples), "ms", "lower")
        out[f"{key}_heavy"] = metric(len(best["heavy"]), "", "lower", loaded=best["heavy"])
    return out


# ---- physics ------------------------------------------------------------------------------------------------

CHUNK = 0.05  # s of sim per timed slice
MAX_WALL_PER_WORLD = 120.0  # s; a world that is far too slow stops early and says so
MACHINE_STEPS = 25  # steps of the machine check after each timed slice: about 0.1 ms


class MachineCheck:
    """How fast this machine runs MuJoCo at the moments a world is timed: a chain of 20 links that touch nothing,
    restarted from the same pose every time (so on the same machine it always costs the same), stepped right after
    each timed slice in the world's own process and kept, like the world's slices, as each slice's best over the
    repeats. It runs no robotforge code, so it says how busy the machine was and nothing about robotforge (the
    slowest check of a run is physics.machine, whose floor marks a busy machine). A world's speed over it moves
    less with how busy the machine was than its speed alone, and the load drifts within one run, so the first
    worlds and the last can see different machines. On a very busy machine both still wobble, so a small change
    needs --ab to be sure of."""

    def __init__(self, slices: int, links: int = 20) -> None:
        import mujoco

        self._mujoco = mujoco
        link = (
            '<body pos="0 0 -0.1"><joint type="hinge" axis="0 1 0" damping="0.01"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" contype="0" conaffinity="0"/>'
        )
        xml = f'<mujoco><worldbody><body pos="0 0 3">{link * links}{"</body>" * links}</body></worldbody></mujoco>'
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.best = [math.inf] * slices

    def time(self, k: int) -> None:
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.3
        c0 = time.process_time()
        self._mujoco.mj_step(self.model, self.data, nstep=MACHINE_STEPS)
        self.best[k] = min(self.best[k], time.process_time() - c0)

    @property
    def steps_per_s(self) -> float:
        measured = [t for t in self.best if math.isfinite(t)]
        return len(measured) * MACHINE_STEPS / max(sum(measured), 1e-9)


@dataclass(frozen=True)
class World:
    """A standard world: built once, then reopened for every repeat so each starts from the same state."""

    build: Callable[[Any], None]  # makes the world on a Playground (saved as world.yaml)
    start: Callable[[Any], Any]  # sets the robots going on the reopened world; returns what `check` needs
    check: Callable[[Any, Any], dict[str, Any]]  # what the run did (small), so a broken world is not timed quietly
    seconds: float = 4.0  # sim time measured
    warm: float = 0.5  # sim time run before measuring
    memory_noise: tuple[float, float] = MEMORY_NOISE  # how much its peak memory wobbles from run to run


def _nothing(pg: Any) -> None:
    return None


def _dog(pg: Any) -> None:
    pg.add_robot("dog", template="quadruped")


def _go2(pg: Any) -> None:
    pg.add_robot("go2", library="go2")


def _humanoid(pg: Any) -> None:
    pg.add_robot("humanoid", template="humanoid")


def _walk(robot: str, speed: float = 0.3, walker: str | None = "gait") -> Callable[[Any], Any]:
    def start(pg: Any) -> Any:
        pg.behaviors.move(robot, speed, 0.0, walker=walker)
        return pg.behaviors.pose(robot)["position"]

    return start


def _went(robot: str) -> Callable[[Any, Any], dict[str, Any]]:
    def check(pg: Any, start: Any) -> dict[str, Any]:
        pose = pg.behaviors.pose(robot)
        return {"moved_m": round(math.dist(start[:2], pose["position"][:2]), 3), "upright": pose["upright"]}

    return check


def _stood(robot: str) -> Callable[[Any, Any], dict[str, Any]]:
    return lambda pg, _start: {"upright": pg.behaviors.pose(robot)["upright"]}


def _arm_world(pg: Any) -> None:
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")


def _arm_start(pg: Any) -> Any:
    _arm, info = pg.behaviors.pick_and_place("arm", "cube", "tray")
    return info


def _arm_check(pg: Any, info: Any) -> dict[str, Any]:
    return {"placed": bool(pg.behaviors.placed("cube", info["region"])["success"])}


def _warehouse(pg: Any) -> None:
    pg.add_robot("go2", library="go2")
    pg.load_scene("warehouse")


def _warehouse_start(pg: Any) -> Any:
    pg.behaviors.navigate("go2", "dock")
    return pg.behaviors.pose("go2")["position"]


def _objects(pg: Any) -> None:
    """20 objects on and above a table: ten on its top, ten dropped onto them, so they collide and settle."""
    pg.add_object("table", shape="table", size=[0.9, 0.6, 0.4], pos=[0.0, 0.0])
    shapes = ("box", "sphere", "cylinder", "box", "sphere")
    sizes = {"box": [0.06, 0.06, 0.06], "sphere": [0.035], "cylinder": [0.03, 0.08]}
    for k in range(20):
        layer, slot = divmod(k, 10)
        x, y = -0.32 + 0.16 * (slot % 5), -0.12 + 0.24 * (slot // 5)
        shape = shapes[(k + layer) % len(shapes)]
        z = 0.47 + 0.12 * layer
        pg.add_object(f"thing_{k}", shape=shape, size=sizes[shape], pos=[x + 0.03 * layer, y, z], color="red")


def _objects_check(pg: Any, _start: Any) -> dict[str, Any]:
    import mujoco

    m, d = pg.session.model, pg.session.data
    names = {b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)}
    things = [b for b, name in names.items() if name.startswith("thing")]
    return {"on_table": sum(1 for b in things if d.xpos[b][2] > 0.4), "objects": len(things)}


WORLDS: dict[str, World] = {
    # Never moved: stands on its gait from the start, as a beginner finds it.
    "dog_idle": World(_dog, _nothing, _stood("dog")),
    "dog_walk": World(_dog, _walk("dog"), _went("dog")),
    "go2_idle": World(_go2, _nothing, _stood("go2"), memory_noise=MESH_MEMORY_NOISE),
    "go2_walk": World(_go2, _walk("go2"), _went("go2"), memory_noise=MESH_MEMORY_NOISE),
    # The learned walk (a neural network policy) standing still: it holds its spot.
    "go2_rl_idle": World(_go2, _walk("go2", 0.0, "rl"), _went("go2"), memory_noise=MESH_MEMORY_NOISE),
    # Keeps its balance from the start: a walking controller (footsteps, ZMP preview, leg IK) standing still.
    "humanoid_idle": World(_humanoid, _nothing, _stood("humanoid")),
    "arm_pick": World(_arm_world, _arm_start, _arm_check, seconds=8.0, warm=0.0),
    "warehouse_nav": World(_warehouse, _warehouse_start, _went("go2"), seconds=6.0, memory_noise=MESH_MEMORY_NOISE),
    "objects_20": World(_objects, _nothing, _objects_check, seconds=3.0, warm=0.0),
}


def physics_world(name: str, repeats: int) -> dict[str, Any]:
    """Runs in a worker process: build the world, then time it `repeats` times and once more with the
    controllers timed apart."""
    try:
        import resource
    except ImportError:  # Windows: peak memory goes unmeasured
        resource = None

    from robotforge.playground import Playground

    world = WORLDS[name]
    pg = Playground.with_demo()
    t0 = time.perf_counter()
    pg.new_world(f"bench_{name}")
    world.build(pg)
    built = time.perf_counter() - t0
    slices = max(1, round(world.seconds / CHUNK))
    best = [math.inf] * slices  # each slice's least process time over the repeats
    machine = MachineCheck(slices)
    done: dict[str, Any] = {}
    share = 0.0
    started = time.monotonic()
    for rep in range(repeats + 1):
        pg.open_world(f"bench_{name}")  # the same start every time: compiled afresh, controllers new
        s = pg.session
        chunk = max(1, round(CHUNK / s.model.opt.timestep))
        state = world.start(pg)
        pg.run_fast(lambda: False, world.warm)
        # The last run times the controllers apart (which costs a little itself), and is not counted for speed.
        timed = rep == repeats
        in_controllers = 0.0
        if timed:
            original = s._run_controllers

            def run_controllers(original=original) -> None:
                nonlocal in_controllers
                t = time.perf_counter()
                original()
                in_controllers += time.perf_counter() - t

            s._run_controllers = run_controllers
        wall0 = time.perf_counter()
        for k in range(slices):
            c0 = time.process_time()
            s.step(chunk)
            if not timed:
                best[k] = min(best[k], time.process_time() - c0)
                machine.time(k)
            if time.monotonic() - started > MAX_WALL_PER_WORLD:
                done["note"] = f"stopped after {MAX_WALL_PER_WORLD:.0f} s"
                break
        if timed:
            share = in_controllers / max(time.perf_counter() - wall0, 1e-9)
            del s._run_controllers
            done.update(world.check(pg, state))
        if "note" in done:
            break
    m = pg.session.model
    measured = [t for t in best if math.isfinite(t)]
    rate = len(measured) * chunk / max(sum(measured), 1e-9)
    return {
        "steps_per_s": rate,
        "machine": machine.steps_per_s,
        "rtf": rate * m.opt.timestep,
        "controllers_share": share,
        "peak_mb": _rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if resource else None,
        "build_s": built,
        "bodies": m.nbody - 1,
        "geoms": m.ngeom,
        "timestep": m.opt.timestep,
        **done,
    }


def _physics_worker(name: str, repeats: int, src: Path | None = None) -> dict[str, Any]:
    """One world timed in a fresh process; `src`: robotforge's source from another commit instead of this one."""
    with tempfile.TemporaryDirectory(prefix="rf-bench-phys-") as home:
        env = _with_source(_child_env(Path(home)), src)
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--repeats", str(repeats)]
        done = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[-600:])
    return json.loads(done.stdout.strip().splitlines()[-1])


def bench_physics(quick: bool, worlds: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in worlds or list(WORLDS):
        try:
            r = _physics_worker(name, 2 if quick else 4)
        except RuntimeError as exc:
            out[f"{name}_error"] = str(exc)
            continue
        info = {k: r[k] for k in r if k not in ("steps_per_s", "rtf", "controllers_share", "peak_mb")}
        info["machine"] = round(info["machine"])  # steps/s of the machine check, timed in the same process
        out[f"{name}.steps_per_s"] = metric(r["steps_per_s"], "steps/s", "higher", **info)
        out[f"{name}.controllers_share"] = metric(r["controllers_share"], "share", "lower")
        out[f"{name}.peak_mb"] = metric(r["peak_mb"], "MB", "lower", noise=WORLDS[name].memory_noise)
    checks = {
        name: out[f"{name}.steps_per_s"]["machine"] for name in worlds or list(WORLDS) if f"{name}.steps_per_s" in out
    }
    if checks:  # the slowest moment of the run: a world timed then may have crossed its floor with no code change
        slowest = min(checks, key=checks.get)
        out["machine.steps_per_s"] = metric(checks[slowest], "steps/s", "higher", slowest=slowest)
    return out


@contextlib.contextmanager
def _source_at(rev: str) -> Iterator[Path]:
    """robotforge's source as it was at revision `rev`, unpacked in a temp folder."""
    import io
    import tarfile

    archive = subprocess.run(["git", "archive", "--format=tar", rev, "src"], cwd=ROOT, capture_output=True, check=False)
    if archive.returncode != 0:
        raise SystemExit(f"cannot read {rev}: {archive.stderr.decode(errors='replace').strip()}")
    with tempfile.TemporaryDirectory(prefix="rf-bench-ab-") as tmp:
        with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
            tar.extractall(tmp, filter="data")
        yield Path(tmp) / "src"


def ab_imports(old: Path, rounds: int) -> None:
    """Import times on this tree and on another commit's source, alternately: best process time of each."""
    for target in IMPORT_TARGETS:
        _import_once(target)  # bytecode compiled first, for both
        _import_once(target, src=old)
        now, then = [], []
        for _ in range(rounds):
            now.append(_import_once(target)[0]["cpu"])
            then.append(_import_once(target, src=old)[0]["cpu"])
        a, b = 1000 * min(then), 1000 * min(now)
        print(f"import {target:<22} {a:>7.0f} -> {b:>7.0f} ms process time ({b / a - 1:+.0%})", flush=True)


def ab_cold_start(old: Path, rounds: int) -> None:
    """Cold start to the first frame on this tree and on another commit's source, alternately: best of each."""
    with tempfile.TemporaryDirectory(prefix="rf-bench-viewer-") as tmp:
        home = Path(tmp)
        dog = _make_world(home, "bench_dog", _dog)
        with _chrome(home) as chrome_port:
            now, then = [], []
            for _ in range(rounds + 1):  # the first of each warms the disk cache and is not counted
                now.append(asyncio.run(_cold_start(chrome_port, dog, home))["first_frame_s"])
                then.append(asyncio.run(_cold_start(chrome_port, dog, home, src=old))["first_frame_s"])
    a, b = min(then[1:]), min(now[1:])
    print(f"cold start to the first frame {a:>6.2f} -> {b:>6.2f} s ({b / a - 1:+.0%})", flush=True)


def ab_physics(rev: str, worlds: list[str], rounds: int) -> list[dict[str, Any]]:
    """The physics worlds timed on this tree and on `rev`, alternately, so both see the same machine load:
    on a shared machine two runs an hour apart differ more than a change does. Steps per second: the best round
    of each (each round is already the best of four per 50 ms slice); controllers' share: the median round."""
    rows = []
    with _source_at(rev) as old:
        ab_imports(old, 2 * rounds + 1)
        try:
            ab_cold_start(old, rounds)
        except (SystemExit, RuntimeError, OSError) as exc:  # no Chrome here: the rest still counts
            print(f"cold start not compared: {exc}", flush=True)
        for name in worlds:
            now: list[dict[str, Any]] = []
            then: list[dict[str, Any]] = []
            for _ in range(rounds):
                now.append(_physics_worker(name, 4))
                then.append(_physics_worker(name, 4, old))
            row = {
                "world": name,
                "steps_per_s": [max(r["steps_per_s"] for r in then), max(r["steps_per_s"] for r in now)],
                "controllers_share": [
                    statistics.median(r["controllers_share"] for r in then),
                    statistics.median(r["controllers_share"] for r in now),
                ],
            }
            rows.append(row)
            (a, b), (ca, cb) = row["steps_per_s"], row["controllers_share"]
            print(f"{name:<16} {a:>9,.0f} -> {b:>9,.0f} steps/s ({b / a - 1:+.0%})   controllers {ca:.0%} -> {cb:.0%}")
    return rows


# ---- viewer (headless Chrome) -------------------------------------------------------------------------------

VIEW_W, VIEW_H, VIEW_SCALE = 1280, 800, 2  # CSS px and device pixel ratio, as the README's screenshots
IDLE_SECONDS = 5.0  # processor use of an idle server is measured over this long
READY_TIMEOUT = 90.0


@contextlib.contextmanager
def _chrome(home: Path) -> Iterator[int]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import screenshots

    port = _free_port()
    proc = subprocess.Popen(
        [
            screenshots._chrome(),
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={home / 'chrome'}",
            "--hide-scrollbars",
            "--no-first-run",
            "--use-angle=metal" if sys.platform == "darwin" else "--use-gl=angle",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/json/version", 30, proc)
        yield port
    finally:
        _stop(proc)


_FIRST_FRAME = """
new Promise((resolve) => {
  const ready = () => window.__robotforge && window.__robotforge.scene && window.__robotforge.time > 0;
  const drawn = () => requestAnimationFrame(() => requestAnimationFrame(() => resolve(true)));
  const poll = () => (ready() ? drawn() : setTimeout(poll, 5));
  poll();
})
"""

_FPS = """
new Promise((resolve) => {
  let n = 0;
  const t0 = performance.now();
  const tick = (now) => { n++; if (now - t0 < %d) requestAnimationFrame(tick); else resolve(n * 1000 / (now - t0)); };
  requestAnimationFrame(tick);
})
"""

_FRAME_MS = """
(() => {
  const { scene, camera, renderer } = window.__robotforge.three();
  const gl = renderer.getContext();
  const px = new Uint8Array(4);
  const times = [];
  for (let k = 0; k < 40; k++) {
    const t0 = performance.now();
    renderer.render(scene, camera);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);  // waits for the GPU to finish the frame
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return { median: times[20], drawing_buffer: [gl.drawingBufferWidth, gl.drawingBufferHeight],
           calls: renderer.info.render.calls, triangles: renderer.info.render.triangles };
})()
"""


async def _open_tab(chrome_port: int) -> tuple[Any, Any, Any, str]:
    import websockets

    sys.path.insert(0, str(ROOT / "scripts"))
    import screenshots

    version = json.loads(await asyncio.to_thread(_http_body, f"http://127.0.0.1:{chrome_port}/json/version"))
    ws = await websockets.connect(version["webSocketDebuggerUrl"], max_size=None)
    browser = screenshots.Page(ws)
    target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
    session = (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"]
    page = screenshots.Page(ws, session)
    await page.call(
        "Emulation.setDeviceMetricsOverride", width=VIEW_W, height=VIEW_H, deviceScaleFactor=VIEW_SCALE, mobile=False
    )
    await page.call("Page.enable")
    await page.call("Page.bringToFront")
    await page.call("Emulation.setFocusEmulationEnabled", enabled=True)
    storage = "localStorage.setItem('robotforge.coached', '1'); localStorage.setItem('robotforge.world', '0');"
    await page.call("Page.addScriptToEvaluateOnNewDocument", source=f"try {{ {storage} }} catch (e) {{}}")
    return ws, browser, page, target


def _http_body(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=10) as resp:
        return resp.read()


async def _cold_start(
    chrome_port: int, world_dir: Path, home: Path, idle: float = 0.0, src: Path | None = None
) -> dict[str, Any]:
    """From running the command to the first frame with the world in it, in a tab opened beforehand; with `idle`,
    then the server's processor use over that many seconds while the page shows the world (real time, nobody
    touching anything)."""
    ws, browser, page, target = await _open_tab(chrome_port)
    port = _free_port()
    cmd = [_robotforge_command(), "open", str(world_dir), "--no-browser", "--port", str(port)]
    env = _with_source(_child_env(home), src)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        await asyncio.to_thread(_wait_http, f"http://127.0.0.1:{port}/", READY_TIMEOUT, proc)
        served = time.perf_counter() - t0
        await page.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang=en")
        await asyncio.wait_for(page.eval(_FIRST_FRAME), READY_TIMEOUT)
        first = time.perf_counter() - t0
        idle_cpu = await asyncio.to_thread(_cpu_share, proc.pid, idle) if idle > 0 else None
    finally:
        peak = await asyncio.to_thread(_stop, proc)
        await browser.call("Target.closeTarget", targetId=target)
        await ws.close()
    return {"served_s": served, "first_frame_s": first, "server_peak_mb": peak, "idle_cpu": idle_cpu}


async def _frame_rate(chrome_port: int, world_dir: Path, home: Path, shots: Path | None) -> dict[str, Any]:
    ws, browser, page, target = await _open_tab(chrome_port)
    port = _free_port()
    cmd = [_robotforge_command(), "open", str(world_dir), "--no-browser", "--port", str(port)]
    proc = subprocess.Popen(cmd, env=_child_env(home), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        await asyncio.to_thread(_wait_http, f"http://127.0.0.1:{port}/", READY_TIMEOUT, proc)
        await page.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang=en")
        await asyncio.wait_for(page.eval(_FIRST_FRAME), READY_TIMEOUT)
        await asyncio.sleep(2.0)  # the camera's opening glide and the first shadow maps
        fps = await page.eval(_FPS % 3000)
        frame = await page.eval(_FRAME_MS)
        if shots is not None:
            import base64

            data = (await page.call("Page.captureScreenshot", format="png"))["data"]
            (shots / "bench_viewer.png").write_bytes(base64.b64decode(data))
    finally:
        peak = await asyncio.to_thread(_stop, proc)
        await browser.call("Target.closeTarget", targetId=target)
        await ws.close()
    return {"fps": fps, "frame": frame, "server_peak_mb": peak}


def _make_world(home: Path, name: str, build: Callable[[Any], None]) -> Path:
    """A world saved in `home`, built in a child process so this one stays light."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import bench; "
        "from robotforge.playground import Playground; pg = Playground.with_demo(); "
        f"pg.new_world({name!r}); bench.{build.__name__}(pg)"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts")], env=_child_env(home), check=True, capture_output=True
    )
    return home / "worlds" / name


def bench_viewer(quick: bool, shots: Path | None = None) -> dict[str, Any]:
    runs = 2 if quick else 4
    with tempfile.TemporaryDirectory(prefix="rf-bench-viewer-") as tmp:
        home = Path(tmp)
        dog = _make_world(home, "bench_dog", _dog)
        warehouse = _make_world(home, "bench_warehouse", _warehouse)
        with _chrome(home) as chrome_port:
            starts = [asyncio.run(_cold_start(chrome_port, dog, home, IDLE_SECONDS * (k == 0))) for k in range(runs)]
            rate = asyncio.run(_frame_rate(chrome_port, warehouse, home, shots))
    best = min(starts[1:] or starts, key=lambda r: r["first_frame_s"])  # the first run also warms the disk cache
    peaks = [r["server_peak_mb"] for r in starts if r["server_peak_mb"] is not None]
    return {
        "cold_start_s": metric(best["first_frame_s"], "s", "lower", served_s=round(best["served_s"], 3)),
        "fps": metric(rate["fps"], "fps", "higher", world="warehouse + go2", size=[VIEW_W, VIEW_H, VIEW_SCALE]),
        "frame_ms": metric(rate["frame"]["median"], "ms", "lower", **{k: rate["frame"][k] for k in rate["frame"]}),
        "server_peak_mb": metric(
            max(peaks) if peaks else None, "MB", "lower", noise=MEMORY_NOISE, warehouse=rate["server_peak_mb"]
        ),
        # The viewer server simulating an idle dog in real time for an open page: what leaving it open costs.
        "idle_cpu_pct": metric(starts[0]["idle_cpu"], "%", "lower", seconds=IDLE_SECONDS),
    }


# ---- MCP ----------------------------------------------------------------------------------------------------


def _rss_of(pid: int) -> float | None:
    """A process's resident memory now (MB), from ps (macOS and Linux)."""
    done = _ps("-o", "rss=", "-p", str(pid))
    return int(done.stdout.strip()) / 1024 if done and done.stdout.strip().isdigit() else None


def _server_pid(port: int) -> int | None:
    """The `robotforge mcp --port <port>` process this one started."""
    done = _ps("-A", "-o", "pid=,ppid=,command=")
    for line in done.stdout.splitlines() if done else []:
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[1]) == os.getpid() and "mcp" in parts[2] and f"--port {port}" in parts[2]:
            return int(parts[0])
    return None


async def _mcp_session(home: Path, calls: int) -> dict[str, Any]:
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    port = _free_port()
    params = StdioServerParameters(
        command=_robotforge_command(), args=["mcp", "--port", str(port)], env=_child_env(home)
    )
    latencies: dict[str, list[float]] = {call: [] for call in MCP_CALLS}
    rss: list[float] = []

    async def timed(label: str, tool: str, /, **args: Any) -> Any:
        t = time.perf_counter()
        result = await client.call_tool(tool, args)
        latencies[label].append(time.perf_counter() - t)
        if result.is_error:
            text = "".join(getattr(c, "text", "") for c in result.content)
            raise RuntimeError(f"{tool} failed: {text[:300]}")
        return result

    t0 = time.perf_counter()
    async with Client(params, read_timeout_seconds=READY_TIMEOUT) as client:
        tools = (await client.list_tools()).tools
        started = time.perf_counter() - t0
        pid = await asyncio.to_thread(_server_pid, port)
        for k in range(calls):
            await timed("new_world", "new_world", name=f"bench_{k}")
            await timed("add_robot", "add_robot", name="dog", template="quadruped")
            await timed("move", "move", robot="dog", speed=0.3, seconds=1.0)
            await timed("observe", "observe", include_image=False)
            if pid is not None:
                found = await asyncio.to_thread(_rss_of, pid)
                if found is not None:
                    rss.append(found)
        # Between calls, with no viewer open: an agent session spends most of its time like this.
        idle = await asyncio.to_thread(_cpu_share, pid, IDLE_SECONDS) if pid is not None else None
    return {
        "start_s": started,
        "tools": len(tools),
        "latencies": latencies,
        "rss": max(rss) if rss else None,
        "idle_cpu": idle,
    }


def bench_mcp(quick: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rf-bench-mcp-") as tmp:
        r = asyncio.run(_mcp_session(Path(tmp), 3 if quick else 5))
    out = {"start_s": metric(r["start_s"], "s", "lower", tools=r["tools"])}
    for name, times in r["latencies"].items():
        out[f"{name}_ms"] = metric(1000 * statistics.median(times), "ms", "lower", calls=len(times))
    out["server_rss_mb"] = metric(r["rss"], "MB", "lower", noise=MEMORY_NOISE)
    out["idle_cpu_pct"] = metric(r["idle_cpu"], "%", "lower", seconds=IDLE_SECONDS)
    return out


# ---- main ---------------------------------------------------------------------------------------------------


def run(
    sections: list[str], quick: bool, shots: Path | None, worlds: list[str] | None = None
) -> tuple[dict[str, Any], list[str]]:
    benches: dict[str, Callable[[], dict[str, Any]]] = {
        "install": lambda: bench_install(quick),
        "imports": lambda: bench_imports(quick),
        "physics": lambda: bench_physics(quick, worlds),
        "viewer": lambda: bench_viewer(quick, shots),
        "mcp": lambda: bench_mcp(quick),
    }
    results: dict[str, Any] = {}
    errors: list[str] = []
    for section in sections:
        t0 = time.perf_counter()
        print(f"... {section}", file=sys.stderr, flush=True)
        try:
            results[section] = benches[section]()
        except Exception as exc:  # one part failing should not lose the others
            while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]  # anyio wraps what went wrong in task groups
            errors.append(f"{section}: {type(exc).__name__}: {exc}")
            results[section] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
        errors += [f"{section}: {k}: {v[-200:]}" for k, v in results[section].items() if k.endswith("_error")]
        known = known_metrics()
        errors += [
            f"{name}: measured, but known_metrics() does not list it (add it there)"
            for name in flatten({section: results[section]})
            if name not in known
        ]
        print(f"    {section} took {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, help="parts to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="fewer repeats: a rough look, not saved")
    parser.add_argument(
        "--wait-calm", type=float, default=0.0, metavar="MIN", help="first wait up to MIN minutes for a calm machine"
    )
    saving = parser.add_mutually_exclusive_group()
    saving.add_argument("--no-save", action="store_true", help="print only; write nothing to docs/bench")
    saving.add_argument(
        "--save", action="store_true", help="also save a quick or partial run, as <date>-<sha>-partial.json"
    )
    parser.add_argument("--out", type=Path, default=BENCH_DIR)
    parser.add_argument("--budget", type=Path, default=BUDGET_FILE)
    parser.add_argument("--against", type=Path, help="compare with this run (default: the newest complete one)")
    parser.add_argument("--shots", type=Path, help="save a picture of the viewer frame measured here")
    parser.add_argument("--worlds", nargs="*", choices=list(WORLDS), help="physics: only these worlds")
    parser.add_argument("--ab", metavar="REV", help="time imports and physics on this tree and REV alternately")
    parser.add_argument("--rounds", type=int, default=3, help="--ab: rounds per world")
    parser.add_argument("--worker", help=argparse.SUPPRESS)  # internal: time one physics world in this process
    parser.add_argument("--repeats", type=int, default=4, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(physics_world(args.worker, args.repeats)))
        return 0
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    if args.ab:
        print(f"{args.ab} -> this tree ({git_sha()}), load {machine().get('load')}", flush=True)
        ab_physics(args.ab, args.worlds or list(WORLDS), args.rounds)
        return 0
    try:  # before minutes of measuring: a budget on a metric that does not exist is a mistake to fix first
        budget_spec = load_budgets(args.budget)
    except ValueError as exc:
        print(f"ERROR  {exc}")
        return 2
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    if args.wait_calm > 0:
        wait_for_calm(args.wait_calm * 60)
    sections = args.only or list(SECTIONS)
    worlds = args.worlds or list(WORLDS)
    ran = parts(sections, worlds)
    # Only a complete run is a baseline: a partial one saved under the same name would replace it, and the next
    # complete run would compare with a stub.
    partial = args.quick or ran != parts(list(SECTIONS), list(WORLDS))
    started = dt.datetime.now().astimezone()
    sha = git_sha()
    before_machine = machine()
    results, errors = run(sections, args.quick, args.shots, args.worlds)
    load_after = [round(v, 1) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None
    record = {
        "started": started.isoformat(timespec="seconds"),
        "sha": sha,
        "quick": args.quick,
        "partial": partial,
        "covers": sorted(ran),
        "machine": before_machine,
        "load_after": load_after,
        "results": results,
    }
    now = flatten(results)
    against = args.against or previous_run(args.out)
    before: dict[str, Any] = {}
    if against is not None and against.is_file():
        old = json.loads(against.read_text(encoding="utf-8"))
        before = flatten(old.get("results", {}))
        then, here = old.get("machine", {}).get("load", ["?"])[0], before_machine.get("load", ["?"])[0]
        print(f"compared with {against.name} (load {then} then, {here} now)")
    compared = compare(now, before)
    cores = os.cpu_count() or 1
    loads = [load[0] for load in (before_machine.get("load"), load_after) if load]
    busy = busy_reason(loads, cores, now, budget_spec)
    budgets = check_budgets(now, budget_spec, ran, busy is not None)
    print("\n".join(table(now, compared, budgets, busy is not None)))
    if any(c.get("relative") for c in compared.values()):
        print(
            "\n* steps/s changes allow for the machine's speed, timed in each world's own process. A busier machine"
            " than last time slows a world (most, one with Python controllers) about 1.2 to 1.4 times as much as the"
            " check, a faster one speeds it as much: where the check moved more than a quarter, the change leans that"
            " way and needs 25 percent to count"
        )
    if busy:
        print(f"\nnote: {busy}: timings run slow on a machine this busy, so timings over budget are inconclusive")
    over = [b for b in budgets if b["status"] == "over"]
    inconclusive = [b for b in budgets if b["status"] == "inconclusive"]
    unmeasured = [b for b in budgets if b["status"] == "unmeasured"]
    for b in over:
        print(f"OVER BUDGET  {b['metric']} = {b['value']} ({b['kind']} {b['limit']}): {b['why']}")
    for b in inconclusive:
        print(f"INCONCLUSIVE  {b['metric']} = {b['value']} ({b['kind']} {b['limit']}), on a busy machine")
    for b in unmeasured:
        print(f"UNMEASURED  {b['metric']}: its part ran but gave no number, so this budget went unchecked")
    for e in errors:
        print(f"ERROR  {e}")
    if errors and not args.no_save:
        # A run with parts missing would become the next run's "previous": not a record.
        print("\nnot saved: parts of the run failed")
    elif partial and not args.save and not args.no_save:
        print("\nnot saved: a quick or partial run (--save writes it as <date>-<sha>-partial.json)")
    elif not args.no_save:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / run_name(started, sha, partial)
        record["budgets"] = {b["metric"]: b["status"] for b in budgets}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    # A budget exceeded is the answer even when another part failed: the failure must not hide a regression
    if over:
        return 1
    if errors or unmeasured:
        return 2
    return 3 if inconclusive else 0


if __name__ == "__main__":
    sys.exit(main())
