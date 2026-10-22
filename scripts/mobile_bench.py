"""Mobile manipulation jobs in the home scene, run headless over robots and seeds: success rates and failures by cause.

    uv run python scripts/mobile_bench.py [--robots go2_arm mobile_panda ...] [--jobs fetch drawer ...]
                                          [--seeds 8] [--seed-start 0] [--workers 3] [--json out.jsonl]

The jobs, each as an agent would do it with the tools (pick_and_place, operate, move):
- door: from the living room, through the swing door into the kitchen (move with to="kitchen").
- fetch: "go to the kitchen and bring the apple to the living-room table" (pick_and_place goes over by itself).
- drawer: open the chest of drawers in the kitchen and take the block out onto the kitchen table.
- button: press the button on the living room's side table.
- conveyor: in the factory, beside the running belt (the start varies along it by 1.5 m and away from it by 0.85 m),
  catch the last box as it rides by and take it to the workbench (pick_and_place drives to a place beside the belt
  where the arm reaches if the arm does not reach from the start, catches the box, and carries it; too late for
  that, it waits for the box at the end stop and fetches it there).
A seed jitters the things on the tables (the scene's own seed) and where and how the robot starts. Seeds 0-7 are the
ones the fixes were tuned on: measure on others (--seed-start 8 --seeds 20). A cell where every seed was refused for
what the robot or its arm cannot do (a refusal before anything moves, UNSUPPORTED) prints as "not supported", not as
a measured failure. The demo videos (scripts/demos.py mobile_*) run the same jobs in real time. Maintainer tool, not
part of robotforge.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROBOTS: dict[str, dict[str, Any]] = {
    "go2_arm": {"design": "go2_arm"},
    "mobile_panda": {"design": "mobile_panda"},
    "lift_picker": {"design": "lift_picker"},
    "spot_z1": {"library": "spot", "mounts": [{"library": "z1"}]},
    "anymal_ur5e": {"library": "anymal_c", "mounts": [{"library": "ur5e"}, {"library": "robotiq_2f85"}]},
}
JOBS = ("door", "fetch", "drawer", "button", "conveyor")
BESIDE_BELT = (-1.3, 0.2)  # in the factory: where along the belt (x) the robot starts, north of it, facing it
BELT_Y = (0.45, 1.3)  # and how far from it (y; the belt runs along y = 0)
BELT_SPEED = 0.1  # m/s
NAME = "mm"  # the robot's name in every run
STARTS = ((2.4, -1.5), (1.4, -1.7), (3.0, -0.6))  # in the living room, clear of its furniture
DOOR_PUSHED = 45.0  # degrees the door must have swung for the robot to have pushed through it
OPERATE_S, ERRAND_S, TRIP_S = 150.0, 200.0, 90.0  # sim seconds each may take

# Refusals that say what the robot (or its library entry) cannot do at all, the same for every seed.
UNSUPPORTED = (
    "cannot_reach_belt",
    "cannot_pick",
    "nowhere_to_stand",
    "too_high_or_low",
    "cannot_hold",
    "gripper_too_small",
    "too_heavy",
)

Wait = Callable[[Callable[[], bool], float], None]


def build(pg: Any, robot: str, seed: int, job: str = "fetch") -> None:
    """The home (seed `seed`) with `robot` at one of the living room's starting spots, turned some way; for the
    conveyor, the factory with the robot beside its belt, anywhere from 1.3 m upstream of where the last box starts to
    0.2 m downstream, 0.45 to 1.3 m from the belt, facing it within 40 degrees (the box takes 13 s to the end stop:
    further up the belt the robot is too late)."""
    rng = random.Random(seed)
    if job == "conveyor":
        pg.load_scene("factory", seed=seed, starter=False, options={"boxes": 6, "speed": BELT_SPEED})
        x = rng.uniform(*BESIDE_BELT)
        pg.add_robot(NAME, pos=[x, rng.uniform(*BELT_Y)], yaw=-90 + rng.uniform(-40, 40), **ROBOTS[robot])
        return
    pg.load_scene("home", seed=seed, starter=False)
    x, y = STARTS[seed % len(STARTS)]
    pg.add_robot(
        NAME, pos=[x + rng.uniform(-0.1, 0.1), y + rng.uniform(-0.1, 0.1)], yaw=rng.uniform(90, 270), **ROBOTS[robot]
    )


def _arm(pg: Any) -> str:
    return pg.behaviors.resolve(NAME, arm=True)


def _mechanism(pg: Any, target: str) -> dict[str, Any]:
    s = pg.session
    from robotforge.sim import mechanisms

    shapes = {o.name: o.shape for o in pg.store.load().objects if o.name == target}
    with s.lock:
        return mechanisms.read(s.model, s.data, shapes, s.controllers.get(mechanisms.KEY))[target]


def operate(pg: Any, target: str, action: str, wait: Wait) -> dict[str, Any]:
    """operate as the MCP tool does it: plan (it goes over if it must), wait, read the result."""
    from robotforge.behaviors import operate as operating
    from robotforge.behaviors import visit

    op, found, _ = operating.start(pg.behaviors, _arm(pg), target, action, None)
    wait(lambda: op.done, OPERATE_S)
    wait(lambda: False, 0.3)
    s, state = pg.session, _mechanism(pg, found)
    with s.lock:
        if isinstance(op, visit.Visit):
            return op.result(s.model, s.data, state)
        return operating.result(op, s.model, s.data, state)


def pick_and_place(pg: Any, obj: str, target: str, wait: Wait) -> dict[str, Any]:
    from robotforge.behaviors import fetch

    motion, info = pg.behaviors.pick_and_place(_arm(pg), obj, target)
    if isinstance(motion, fetch.Errand):
        wait(lambda: motion.done, ERRAND_S)
    else:
        wait(lambda: motion.done, 40.0)
    wait(lambda: False, 0.5)
    out = pg.behaviors.placed(info["object"], info["region"])
    if isinstance(motion, fetch.Errand):
        out.update(motion.result())
    return out


def job_door(pg: Any, wait: Wait) -> dict[str, Any]:
    d, m = pg.session.data, pg.session.model
    adr = int(m.joint("door/leaf").qposadr[0])
    peak = [0.0]
    trip = pg.behaviors.navigate(NAME, "kitchen")

    def done() -> bool:
        peak[0] = max(peak[0], abs(math.degrees(float(d.qpos[adr]))))
        return trip.done

    wait(done, TRIP_S)
    result = trip.result()
    through = bool(result.get("arrived")) and peak[0] >= DOOR_PUSHED
    out = {"success": through, "door_peak_deg": round(peak[0], 1), "time_s": result.get("time_s")}
    if not through:
        out["why"] = result.get("why") or ("door_not_pushed" if result.get("arrived") else "stuck")
        out["note"] = result.get("note")
    return out


def job_fetch(pg: Any, wait: Wait) -> dict[str, Any]:
    out = pick_and_place(pg, "apple", "living_table", wait)
    return out


def job_drawer(pg: Any, wait: Wait) -> dict[str, Any]:
    opened = operate(pg, "chest", "open", wait)
    if not opened.get("success"):
        return {"success": False, "why": f"open:{opened.get('why', 'short')}", "note": opened.get("note")}
    out = pick_and_place(pg, "block", "kitchen_table", wait)
    res = dict(out)
    res["opened_m"] = opened.get("moved")
    if not out.get("success"):
        res["why"] = f"take:{out.get('why', 'not_there')}"
    return res


def job_button(pg: Any, wait: Wait) -> dict[str, Any]:
    pressed = operate(pg, "button", "press", wait)
    out = {"success": bool(pressed.get("pressed")), "visit_s": pressed.get("visit_s")}
    if not out["success"]:
        out["why"], out["note"] = pressed.get("why", "not_pressed"), pressed.get("note")
    return out


def job_conveyor(pg: Any, wait: Wait) -> dict[str, Any]:
    from robotforge.behaviors import catch, catch_stand

    motion, info = pg.behaviors.pick_and_place(_arm(pg), "box_6", "workbench")
    catching = isinstance(motion, (catch.Catch, catch_stand.Approach))
    wait(lambda: motion.done, motion.time_limit if catching else 40.0)
    wait(lambda: False, 0.5)
    region = motion.region if catching else info["region"]
    out = pg.behaviors.placed("box_6", region) if region is not None else {"success": False}
    res = {"success": bool(out.get("success")), "approach": isinstance(motion, catch_stand.Approach)}
    if catching:
        caught = motion.result()
        res["caught_at"] = caught.get("caught_at")
        res["stood_at"] = caught.get("stood_at")
        if not res["success"]:
            res["why"] = "missed" if not caught.get("caught") else (caught.get("why") or "not_there")
            res["note"] = caught.get("note")
            named = cause(res["note"] or "")  # the arm cannot put it on the workbench, wherever the robot stands
            if named == "refused:cannot_place":
                res["why"] = named
    elif not res["success"]:
        res["why"] = "not_caught"
    return res


JOB_FUNCS: dict[str, Callable[[Any, Wait], dict[str, Any]]] = {
    "door": job_door,
    "fetch": job_fetch,
    "drawer": job_drawer,
    "button": job_button,
    "conveyor": job_conveyor,
}


def cause(text: str) -> str:
    """A short cause for a refusal's message (what the tools raise before anything moves)."""
    for key, words in (
        ("cannot_reach_belt", ("height of the belt",)),
        ("too_late", ("sooner than",)),
        ("no_place_at_belt", ("anywhere beside the belt",)),
        ("cannot_catch", ("is riding on",)),
        ("cannot_place", ("free spot on the",)),
        ("nowhere_to_stand", ("nowhere", "no room")),
        ("too_high_or_low", ("too high", "too low")),
        ("no_way", ("no way",)),
        ("cannot_pick", ("cannot pick and place",)),
        ("cannot_hold", ("cannot take hold",)),
        ("gripper_too_small", ("opens only",)),
        ("too_heavy", ("heavy", "weighs")),
    ):
        if any(w in text for w in words):
            return f"refused:{key}"
    return "refused:other"


def run_one(robot: str, job: str, seed: int, artifacts: str | None = None) -> dict[str, Any]:
    """Run an isolated case, retaining its world, stage trace and final physics state when requested."""
    previous_home = os.environ.get("ROBOTFORGE_HOME")
    temporary = tempfile.TemporaryDirectory(prefix="rf-mobile-") if artifacts is None else None
    home = Path(temporary.name) if temporary else Path(artifacts) / f"{robot}-{job}-{seed}"
    if artifacts is not None:
        home.mkdir(parents=True, exist_ok=False)
    os.environ["ROBOTFORGE_HOME"] = str(home)
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    t0 = time.monotonic()
    out: dict[str, Any] = {"robot": robot, "job": job, "seed": seed}
    pg = None
    stages: list[dict[str, Any]] = []
    last = None

    def record() -> None:
        nonlocal last
        from dataclasses import asdict

        from robotforge.behaviors.fetch import Errand

        for controller in pg.session.controllers.values():
            if not isinstance(controller, Errand):
                continue
            key = (controller.step, controller.state, controller._replans, controller._tries)
            if key == last:
                continue
            last = key
            stages.append(
                {
                    "time": round(float(pg.session.data.time), 3),
                    "step": controller.step,
                    "state": controller.state,
                    "replans": controller._replans,
                    "tries": controller._tries,
                    "base": list(controller._pose(pg.session.data)[0]),
                    "heading": controller._pose(pg.session.data)[1],
                    "pick": asdict(controller.pick) if controller.pick else None,
                    "drop": asdict(controller.drop),
                }
            )

    def wait(until: Callable[[], bool], seconds: float) -> None:
        def observe() -> bool:
            record()
            return until()

        pg.run_fast(observe, seconds)
        record()

    try:
        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world(f"{robot}_{job}_{seed}")
        build(pg, robot, seed, job)
        pg.run_fast(lambda: False, 1.0)
        out.update(JOB_FUNCS[job](pg, wait))
        out["sim_s"] = round(float(pg.session.data.time), 1)
        if not pg.behaviors.pose(NAME)["upright"]:
            out.update(success=False, why="fell")
    except Exception as exc:  # a refusal before it moves is a result too: the cause is counted
        text = str(exc)
        known = type(exc).__name__ in ("OperateError", "BehaviorError", "FetchError", "NoWay")
        out.update(success=False, why=cause(text) if known else f"error:{type(exc).__name__}", note=text[:300])
        if not known:
            out["trace"] = traceback.format_exc()[-800:]
    finally:
        try:
            out["wall_s"] = round(time.monotonic() - t0, 1)
            out["stages"] = stages
            if artifacts is not None:
                import numpy as np

                out["artifacts"] = str(home.resolve())
                if pg is not None:
                    data = pg.session.data
                    np.savez_compressed(
                        home / "final-state.npz", qpos=data.qpos, qvel=data.qvel, ctrl=data.ctrl, time=data.time
                    )
                (home / "result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        finally:
            if previous_home is None:
                os.environ.pop("ROBOTFORGE_HOME", None)
            else:
                os.environ["ROBOTFORGE_HOME"] = previous_home
            if temporary:
                temporary.cleanup()
    return out


def read_cases(path: Path) -> list[tuple[str, str, int]]:
    """Validate a frozen case list before starting any worker or simulation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("case manifest must have version 1")
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("case manifest must have a nonempty cases list")
    cases = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("robot"), str)
            or not isinstance(row.get("job"), str)
            or row.get("robot") not in ROBOTS
            or row.get("job") not in JOBS
            or type(row.get("seed")) is not int
            or row["seed"] < 0
        ):
            raise ValueError(f"invalid mobile case: {row!r}")
        case = (row["robot"], row["job"], row["seed"])
        if case in cases:
            raise ValueError(f"duplicate mobile case: {case!r}")
        cases.append(case)
    return cases


def provenance(runs: list[tuple[str, str, int]]) -> dict[str, Any]:
    """Identify the exact source, model pin and environment, including uncommitted source edits."""
    from robotforge.design.menagerie import COMMIT

    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    inputs = sorted(p for p in (root / "src").rglob("*") if p.suffix in {".py", ".json", ".yaml", ".xml", ".onnx"})
    for path in inputs + [Path(__file__).resolve(), root / "uv.lock"]:
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return {
        "version": 1,
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": source.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "robotforge")},
        "menagerie_commit": COMMIT,
        "menagerie_override": os.environ.get("ROBOTFORGE_MENAGERIE"),
        "plugins": os.environ.get("ROBOTFORGE_PLUGINS"),
        "standard_models_and_plugins": not os.environ.get("ROBOTFORGE_MENAGERIE")
        and os.environ.get("ROBOTFORGE_PLUGINS") == "",
        "limits_sim_s": {"operate": OPERATE_S, "errand": ERRAND_S, "trip": TRIP_S},
        "cases": [{"robot": r, "job": j, "seed": s} for r, j, s in runs],
    }


def unsupported(row: dict[str, Any]) -> bool:
    return (
        not row.get("success")
        and str(row.get("why", "")).removeprefix("refused:") in UNSUPPORTED
        and (str(row.get("why", "")).startswith("refused:"))
    )


def report(results: list[dict[str, Any]]) -> str:
    """The table: per robot and job the successes over the runs, and the failures by cause. A cell where every run was
    refused for the same thing the robot cannot do reads "not supported (cause)" and is left out of the totals."""
    table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        table[(r["robot"], r["job"])].append(r)
    lines = ["robot          job      success  failures by cause"]
    won = ran = 0
    for (robot, job), rows in table.items():
        causes = Counter(r.get("why") or "?" for r in rows if not r.get("success"))
        if all(unsupported(r) for r in rows) and len(causes) == 1:
            lines.append(f"{robot:14} {job:8} n/s      not supported: {next(iter(causes))}")
            continue
        good = sum(1 for r in rows if r.get("success"))
        won, ran = won + good, ran + len(rows)
        text = ", ".join(f"{c} x{n}" for c, n in causes.most_common()) or "-"
        lines.append(f"{robot:14} {job:8} {good}/{len(rows):<6} {text}")
    lines.append(f"measured cells: {won}/{ran} runs succeeded (not-supported cells left out)")
    lines.append(
        f"all attempts: {sum(bool(r.get('success')) for r in results)}/{len(results)} succeeded; "
        f"{sum(str(r.get('why', '')).startswith('refused:') for r in results)} refused"
    )
    return "\n".join(lines)


def main() -> None:
    # Benchmarks should not accidentally load a developer's personal plugins.
    os.environ.setdefault("ROBOTFORGE_PLUGINS", "")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robots", nargs="+", choices=list(ROBOTS), default=list(ROBOTS))
    parser.add_argument("--jobs", nargs="+", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=0, help="the first seed (seeds 0-7 are the tuning ones)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--json", help="write each run as a line of JSON here")
    parser.add_argument("--cases", type=Path, help="frozen JSON case list, overriding robots/jobs/seeds")
    parser.add_argument("--artifacts", type=Path, help="new directory for manifest, worlds and diagnostics")
    args = parser.parse_args()
    if args.workers < 1 or args.seeds < 1 or args.seed_start < 0:
        parser.error("workers and seeds must be positive; seed-start must be nonnegative")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    runs = [(r, j, s) for r in args.robots for j in args.jobs for s in seeds]
    if args.cases:
        try:
            runs = read_cases(args.cases)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    metadata = provenance(runs)
    if args.artifacts:
        args.artifacts.mkdir(parents=True, exist_ok=False)
        (args.artifacts / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        root = Path(__file__).resolve().parents[1]
        patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        (args.artifacts / "source.patch").write_bytes(patch)
    if args.json:
        Path(args.json + ".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    results = []
    sink = open(args.json, "w", encoding="utf-8") if args.json else None  # noqa: SIM115
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(
            run_one, *zip(*runs, strict=True), [str(args.artifacts) if args.artifacts else None] * len(runs)
        ):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("trace", "stages")}), flush=True)
            if sink:
                sink.write(json.dumps(result) + "\n")
                sink.flush()
    table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        table[(r["robot"], r["job"])].append(r)
    print(report(results))
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
