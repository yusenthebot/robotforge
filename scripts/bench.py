"""How light is robotforge? Measure it the same way every time and hold it to a budget.

    uv run python scripts/bench.py [--only install imports physics viewer mcp] [--worlds dog_idle ...]
        [--quick] [--no-save] [--wait-calm MIN] [--budget docs/bench/budget.json]
        [--against docs/bench/<run>.json] [--shots <folder>]
    uv run python scripts/bench.py --ab <rev> [--worlds ...] [--rounds 3]

Measures, each in fresh processes:
- install: the built wheel, and the wheel plus its runtime dependencies installed in a clean uv venv in a temp
  folder (bytes on disk, packages, the biggest ones, and the download those wheels make per uv.lock);
- imports: `import robotforge`, `robotforge.cli` (what every command loads) and `robotforge.mcp_server`, best of
  several runs, the packages that take the time (python -X importtime) and which heavy modules got loaded;
- physics: headless steps per second (process time, so a busy machine slows it less; each 50 ms slice the best
  of several repeats) for standard worlds, the share of that time spent in controllers, and peak memory;
- viewer: cold start from running `robotforge open <world>` to the first frame drawn in headless Chrome, frames
  per second and the time one frame takes to draw, and the viewer server's peak memory;
- mcp: `robotforge mcp` started to tools listed, the latency of common calls, and the server's memory.

Writes docs/bench/<date>-<sha>.json, prints a table that compares with the previous run there and with the
budgets in docs/bench/budget.json, and exits 1 when a budget is exceeded (2 when a measurement failed).
Timings on a busy machine run slow: the run records the load average and says so next to a failed budget, and
--wait-calm waits for the load to drop first. Two runs an hour apart on a shared machine can differ more than a
change does, so --ab measures a change instead: the physics worlds on this tree and on revision <rev>, alternately.
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
# A change smaller than this share is noise, by unit: sizes are exact, times on a shared machine are not.
NOISE = {"MB": 0.02, "KB": 0.02, "": 0.0}
TIME_NOISE = 0.15


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


def _noise(unit: str) -> float:
    return NOISE.get(unit, TIME_NOISE)


def compare(now: dict[str, dict[str, Any]], before: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per metric: the previous value, the change and a verdict (better, worse, same, new), from flattened results."""
    out = {}
    for name, m in now.items():
        old = before.get(name)
        value = m.get("value")
        if old is None or old.get("value") is None or value is None:
            out[name] = {"before": None, "change": None, "verdict": "new"}
            continue
        prev = float(old["value"])
        change = (value - prev) / abs(prev) if prev else (0.0 if value == prev else math.inf)
        if abs(change) <= _noise(m.get("unit", "")):
            verdict = "same"
        else:
            verdict = "better" if (change < 0) == (m.get("better") == "lower") else "worse"
        out[name] = {"before": prev, "change": round(change, 4), "verdict": verdict}
    return out


def check_budgets(now: dict[str, dict[str, Any]], budgets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each budget against its metric: ok, over (exceeded) or unmeasured (that part did not run)."""
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
            row["status"] = "unmeasured"
        elif (row["kind"] == "max" and value > row["limit"]) or (row["kind"] == "min" and value < row["limit"]):
            row["status"] = "over"
        else:
            row["status"] = "ok"
        out.append(row)
    return out


def load_budgets(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("budgets", {})


def wait_for_calm(seconds: float) -> None:
    """Wait (at most `seconds`) until the last minute's load average drops below the number of cores."""
    if not hasattr(os, "getloadavg"):
        return
    end, cores = time.monotonic() + seconds, os.cpu_count() or 1
    while os.getloadavg()[0] >= cores and time.monotonic() < end:
        print(f"waiting for a calm machine: load {os.getloadavg()[0]:.0f} on {cores} cores", file=sys.stderr)
        time.sleep(30)


def previous_run(directory: Path) -> Path | None:
    """The newest run in `directory` by the time it records (the budget file is not a run)."""
    runs = []
    for p in directory.glob("*.json"):
        if p.name == BUDGET_FILE.name:
            continue
        try:
            started = json.loads(p.read_text(encoding="utf-8")).get("started")
        except (OSError, ValueError):
            continue
        if started:
            runs.append((started, p))
    return max(runs)[1] if runs else None


def run_name(started: dt.datetime, sha: str) -> str:
    return f"{started:%Y-%m-%d}-{sha}.json"


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
    now: dict[str, dict[str, Any]], compared: dict[str, dict[str, Any]], budgets: list[dict[str, Any]]
) -> list[str]:
    by_metric = {b["metric"]: b for b in budgets}
    lines = [f"{'metric':<40} {'now':>10} {'unit':<8} {'before':>10} {'change':>8}  {'budget':<14} status"]
    for name, m in now.items():
        c = compared.get(name, {})
        b = by_metric.get(name)
        change = "" if c.get("change") is None else f"{100 * c['change']:+.0f}%"
        limit = "" if b is None else f"{'<=' if b['kind'] == 'max' else '>='} {_fmt(b['limit'], m['unit'])}"
        status = "" if b is None else b["status"]
        if c.get("verdict") in ("better", "worse"):
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
    """The commit measured; "-dirty" when the code (src, scripts) has changes not committed yet."""
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    return f"{sha}-dirty" if _git("status", "--porcelain", "--", "src", "pyproject.toml") else sha


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


def bench_install(quick: bool) -> dict[str, Any]:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv not found on PATH")
    with tempfile.TemporaryDirectory(prefix="rf-bench-install-") as tmp:
        tmp_path = Path(tmp)
        dist = tmp_path / "dist"
        subprocess.run([uv, "build", "--wheel", "--out-dir", str(dist), str(ROOT)], check=True, capture_output=True)
        wheel = next(dist.glob("robotforge-*.whl"))
        venv = tmp_path / "venv"
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run([uv, "venv", "--python", version, str(venv)], check=True, capture_output=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        install = [uv, "pip", "install", "--python", str(python), str(wheel)]
        done = subprocess.run(install, capture_output=True, text=True, encoding="utf-8", check=False)
        if done.returncode != 0:  # no network: what the uv cache has will do
            done = subprocess.run([*install, "--offline"], capture_output=True, text=True, encoding="utf-8")
            if done.returncode != 0:
                raise RuntimeError(f"uv pip install failed: {done.stderr.strip()[-400:]}")
        site = Path(
            subprocess.run(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            ).stdout.strip()
        )
        sizes = _installed_sizes(site)
        total = sum(sizes.values())
        download = _download_size(sizes)
        top = sorted(sizes.items(), key=lambda kv: -kv[1])[:TOP_N]
        return {
            "wheel_kb": metric(wheel.stat().st_size / 1024, "KB", "lower"),
            "installed_mb": metric(
                total / 2**20, "MB", "lower", top=[[name, round(size / 2**20, 1)] for name, size in top]
            ),
            "packages": metric(len(sizes), "", "lower"),
            "download_mb": metric(None if download is None else download / 2**20, "MB", "lower"),
        }


def _installed_sizes(site: Path) -> dict[str, int]:
    """Bytes on disk per installed distribution, from each one's RECORD (pip and setuptools left out)."""
    sizes: dict[str, int] = {}
    claimed: set[Path] = set()
    for info in site.glob("*.dist-info"):
        name = info.name.split("-")[0].lower().replace("_", "-")
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
        sizes[name] = total
    return sizes


def _download_size(installed: dict[str, int]) -> int | None:
    """Bytes of the wheels uv.lock lists for this platform, for the installed distributions."""
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


def _import_once(target: str, importtime: bool = False) -> tuple[dict[str, Any], str]:
    code = _IMPORT_PROBE.format(target=target, heavy=HEAVY_MODULES)
    args = [sys.executable, *(["-X", "importtime"] if importtime else []), "-c", code]
    # Run from a neutral folder, so `import robotforge` finds the installed package, not a folder named so
    done = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", check=False, cwd=tempfile.gettempdir()
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
        key = target.replace("robotforge.", "") if target != "robotforge" else "robotforge"
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


@dataclass(frozen=True)
class World:
    """A standard world: built once, then reopened for every repeat so each starts from the same state."""

    build: Callable[[Any], None]  # makes the world on a Playground (saved as world.yaml)
    start: Callable[[Any], Any]  # sets the robots going on the reopened world; returns what `check` needs
    check: Callable[[Any, Any], dict[str, Any]]  # what the run did (small), so a broken world is not timed quietly
    seconds: float = 4.0  # sim time measured
    warm: float = 0.5  # sim time run before measuring


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
    "go2_idle": World(_go2, _nothing, _stood("go2")),
    "go2_walk": World(_go2, _walk("go2"), _went("go2")),
    # The learned walk (a neural network policy) standing still: it holds its spot.
    "go2_rl_idle": World(_go2, _walk("go2", 0.0, "rl"), _went("go2")),
    # Keeps its balance from the start: a walking controller (footsteps, ZMP preview, leg IK) standing still.
    "humanoid_idle": World(_humanoid, _nothing, _stood("humanoid")),
    "arm_pick": World(_arm_world, _arm_start, _arm_check, seconds=8.0, warm=0.0),
    "warehouse_nav": World(_warehouse, _warehouse_start, _went("go2"), seconds=6.0),
    "objects_20": World(_objects, _nothing, _objects_check, seconds=3.0, warm=0.0),
}


def physics_world(name: str, repeats: int) -> dict[str, Any]:
    """Runs in a worker process: build the world, then time it `repeats` times and once more with the
    controllers timed apart."""
    import resource

    from robotforge.playground import Playground

    world = WORLDS[name]
    pg = Playground.with_demo()
    t0 = time.perf_counter()
    pg.new_world(f"bench_{name}")
    world.build(pg)
    built = time.perf_counter() - t0
    slices = max(1, round(world.seconds / CHUNK))
    best = [math.inf] * slices  # each slice's least process time over the repeats
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
        "rtf": rate * m.opt.timestep,
        "controllers_share": share,
        "peak_mb": _rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "build_s": built,
        "bodies": m.nbody - 1,
        "geoms": m.ngeom,
        "timestep": m.opt.timestep,
        **done,
    }


def _physics_worker(name: str, repeats: int, src: Path | None = None) -> dict[str, Any]:
    """One world timed in a fresh process; `src`: robotforge's source from another commit instead of this one."""
    with tempfile.TemporaryDirectory(prefix="rf-bench-phys-") as home:
        env = _child_env(Path(home))
        if src is not None:
            env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
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
        out[f"{name}.steps_per_s"] = metric(r["steps_per_s"], "steps/s", "higher", **info)
        out[f"{name}.controllers_share"] = metric(r["controllers_share"], "share", "lower")
        out[f"{name}.peak_mb"] = metric(r["peak_mb"], "MB", "lower")
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


def ab_physics(rev: str, worlds: list[str], rounds: int) -> list[dict[str, Any]]:
    """The physics worlds timed on this tree and on `rev`, alternately, so both see the same machine load:
    on a shared machine two runs an hour apart differ more than a change does. Steps per second: the best round
    of each (each round is already the best of four per 50 ms slice); controllers' share: the median round."""
    rows = []
    with _source_at(rev) as old:
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


async def _cold_start(chrome_port: int, world_dir: Path, home: Path) -> dict[str, Any]:
    """From running the command to the first frame with the world in it, in a tab opened beforehand."""
    ws, browser, page, target = await _open_tab(chrome_port)
    port = _free_port()
    cmd = [_robotforge_command(), "open", str(world_dir), "--no-browser", "--port", str(port)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=_child_env(home), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        await asyncio.to_thread(_wait_http, f"http://127.0.0.1:{port}/", READY_TIMEOUT, proc)
        served = time.perf_counter() - t0
        await page.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang=en")
        await asyncio.wait_for(page.eval(_FIRST_FRAME), READY_TIMEOUT)
        first = time.perf_counter() - t0
    finally:
        peak = await asyncio.to_thread(_stop, proc)
        await browser.call("Target.closeTarget", targetId=target)
        await ws.close()
    return {"served_s": served, "first_frame_s": first, "server_peak_mb": peak}


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
            starts = [asyncio.run(_cold_start(chrome_port, dog, home)) for _ in range(runs)]
            rate = asyncio.run(_frame_rate(chrome_port, warehouse, home, shots))
    best = min(starts[1:] or starts, key=lambda r: r["first_frame_s"])  # the first run also warms the disk cache
    peaks = [r["server_peak_mb"] for r in starts if r["server_peak_mb"] is not None]
    return {
        "cold_start_s": metric(best["first_frame_s"], "s", "lower", served_s=round(best["served_s"], 3)),
        "fps": metric(rate["fps"], "fps", "higher", world="warehouse + go2", size=[VIEW_W, VIEW_H, VIEW_SCALE]),
        "frame_ms": metric(rate["frame"]["median"], "ms", "lower", **{k: rate["frame"][k] for k in rate["frame"]}),
        "server_peak_mb": metric(max(peaks) if peaks else None, "MB", "lower", warehouse=rate["server_peak_mb"]),
    }


# ---- MCP ----------------------------------------------------------------------------------------------------


def _rss_of(pid: int) -> float | None:
    """A process's resident memory now (MB), from ps (macOS and Linux)."""
    done = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return int(done.stdout.strip()) / 1024 if done.stdout.strip().isdigit() else None


def _server_pid(port: int) -> int | None:
    """The `robotforge mcp --port <port>` process this one started."""
    done = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True, check=False)
    for line in done.stdout.splitlines():
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
    latencies: dict[str, list[float]] = {"new_world": [], "add_robot": [], "move": [], "observe": []}
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
    return {"start_s": started, "tools": len(tools), "latencies": latencies, "rss": max(rss) if rss else None}


def bench_mcp(quick: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rf-bench-mcp-") as tmp:
        r = asyncio.run(_mcp_session(Path(tmp), 3 if quick else 5))
    out = {"start_s": metric(r["start_s"], "s", "lower", tools=r["tools"])}
    for name, times in r["latencies"].items():
        out[f"{name}_ms"] = metric(1000 * statistics.median(times), "ms", "lower", calls=len(times))
    out["server_rss_mb"] = metric(r["rss"], "MB", "lower")
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
        print(f"    {section} took {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    return results, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="*", choices=SECTIONS, help="parts to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="fewer repeats: a rough look, not saved")
    parser.add_argument(
        "--wait-calm", type=float, default=0.0, metavar="MIN", help="first wait up to MIN minutes for a calm machine"
    )
    parser.add_argument("--no-save", action="store_true", help="print only; write nothing to docs/bench")
    parser.add_argument("--out", type=Path, default=BENCH_DIR)
    parser.add_argument("--budget", type=Path, default=BUDGET_FILE)
    parser.add_argument("--against", type=Path, help="compare with this run (default: the newest in --out)")
    parser.add_argument("--shots", type=Path, help="save a picture of the viewer frame measured here")
    parser.add_argument("--worlds", nargs="*", choices=list(WORLDS), help="physics: only these worlds")
    parser.add_argument("--ab", metavar="REV", help="physics only: time this tree and revision REV alternately")
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
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    if args.wait_calm > 0:
        wait_for_calm(args.wait_calm * 60)
    started = dt.datetime.now().astimezone()
    sha = git_sha()
    before_machine = machine()
    results, errors = run(args.only or list(SECTIONS), args.quick, args.shots, args.worlds)
    record = {
        "started": started.isoformat(timespec="seconds"),
        "sha": sha,
        "quick": args.quick,
        "machine": before_machine,
        "load_after": [round(v, 1) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None,
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
    budgets = check_budgets(now, load_budgets(args.budget))
    print("\n".join(table(now, compared, budgets)))
    load = before_machine.get("load", [0.0])[0]
    busy = load > (os.cpu_count() or 1)
    if busy:
        print(f"\nnote: load average {load} on {os.cpu_count()} cores: timings run slow on a machine this busy")
    over = [b for b in budgets if b["status"] == "over"]
    for b in over:
        print(f"OVER BUDGET  {b['metric']} = {b['value']} ({b['kind']} {b['limit']}): {b['why']}")
    for e in errors:
        print(f"ERROR  {e}")
    if not (args.no_save or args.quick):
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / run_name(started, sha)
        record["budgets"] = {b["metric"]: b["status"] for b in budgets}
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    if errors:
        return 2
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
