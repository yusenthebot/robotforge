"""Measure navigation: many start and goal pairs per scene and robot; success, time and failures by cause.

    uv run python scripts/nav_bench.py [--scenes kitchen ...] [--robots dog rover ...] [--trials 20] [--seed 0]
        [--json out.jsonl]
    uv run python scripts/nav_bench.py --case dock|ramps|narrow|dead_ends [--robots ...] [--trials 20]
    uv run python scripts/nav_bench.py --report a.jsonl b.jsonl ...

Each trial puts the robot at a start with update_robot (the world is compiled afresh, so every controller starts
clean), sends it to a goal as move(to=...) does (BehaviorManager.navigate, with the tool's 60 s limit) and runs
the physics headless until it arrives or stops. A trip that runs out of time is sent on again, as the tool's note
tells an agent to, up to MAX_CALLS calls; the report says how many arrived within one call. Starts and goals are
drawn from the floor the robot's own map calls free, inside the scene (not the margin round it), at least
MIN_APART apart, with room to turn on the spot whichever way the robot faces ("roomy": an easy start, and goals
the planner already calls reachable); about one goal in four is a named object of the scene. With --anywhere
SHARE, that share of the goals is any point of the scene instead (in a counter, on a rack, behind a wall): those
are reported apart, by what the robot made of them (arrived, stopped at the nearest free spot, no way and why).
The cases (--case) aim at what the random draw rarely gives: up the warehouse dock from the floor (dock), onto
every ramp's landing in the warehouse and the obstacle course (ramps), anywhere in a maze of 0.6 m corridors,
starts and goals without room to turn on the spot (narrow), and into and out of the maze's dead ends (dead_ends).
Two answers are the planner's, not failed trips: "no way" (it did not set off) and "short" (its first plan said
it would end short, beside what it cannot reach, and it did). A trip that set off for its goal and ended short
anyway (planning again on the way found no way on) failed. Uses the real Menagerie Go2 (downloaded once into
the cache). Maintainer tool.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SCENES = ("kitchen", "warehouse", "maze", "park", "obstacle_course", "desk_lab")
MIN_APART = 1.5  # m between start and goal
OBJECT_SHARE = 0.25  # of the goals, a named object rather than a point
MAX_CALLS = 5  # move(to=...) calls of 60 s at most per trip: 5 min, 27 m for a humanoid at 0.09 m/s
SETTLE = 0.5  # s a robot stands at its start before it is sent off
SHORT = ("inside", "closed_off", "climb", "too_narrow")  # a trip that ends short for one of these was told so
CASES = ("dock", "ramps", "narrow", "dead_ends")


def robots() -> dict[str, dict[str, Any]]:
    from screenshots import MOBILE_ARM

    return {
        "dog": {"template": "quadruped"},
        "rover": {"template": "rover"},
        "go2": {"library": "go2"},
        "humanoid": {"template": "humanoid"},
        "built": {"build": MOBILE_ARM},  # a rover built from parts, with an arm on top
    }


def free_spots(pg: Any, robot: str, roomy: bool = True) -> tuple[Any, np.ndarray]:
    """The robot's map where it stands, and the cells inside the scene its middle can be in and get to; `roomy`:
    only those with room to turn on the spot."""
    from robotforge.behaviors import nav_grid, nav_map

    x, y = pg.behaviors.pose(robot)["position"][:2]
    trip = pg.behaviors.navigate(robot, [x + 0.05, y])
    pg.behaviors.stop(robot)
    nav = trip.planner.last_map
    grid = nav.grid
    edge = math.ceil(nav_map.EDGE / grid.res)
    inside = np.zeros(grid.shape, bool)
    inside[edge:-edge, edge:-edge] = True
    spots = nav.free & (nav.room >= nav.body.reach + 0.05) if roomy else nav.free  # turns on the spot either way
    reach = nav_grid.reachable(grid, nav.free, grid.cell(x, y))
    return nav, spots & inside & reach


def pick(rng: np.random.Generator, nav: Any, spots: np.ndarray, near: tuple[float, float] | None):
    """A random spot (x, y), at least MIN_APART from `near`."""
    xs, ys = nav.grid.centers()
    ok = spots.copy()
    if near is not None:
        ok &= np.hypot(xs - near[0], ys - near[1]) >= MIN_APART
    cells = np.argwhere(ok)
    if not len(cells):
        return None
    r, c = cells[rng.integers(len(cells))]
    return (float(xs[r, c]), float(ys[r, c]))


def trial(pg: Any, robot: str, start: tuple[float, float], yaw: float, goal: Any) -> dict[str, Any]:
    from robotforge.behaviors.navigate import NoWay

    if start is None or goal is None:
        return {"start": start, "goal": goal, "outcome": "skipped", "why": "no start or goal of this kind"}
    from robotforge.playground import MAX_SIM_SECONDS

    pg.update_robot(robot, pos=[round(start[0], 3), round(start[1], 3)], yaw=round(yaw, 1))
    pg.run_fast(lambda: False, SETTLE)
    record: dict[str, Any] = {"start": [round(start[0], 2), round(start[1], 2)], "yaw": round(yaw, 1), "goal": goal}
    calls, spent, walked, replans, said = 0, 0.0, 0.0, 0, None
    wall = time.perf_counter()
    while True:
        calls += 1
        t0 = time.perf_counter()
        try:
            trip = pg.behaviors.navigate(robot, goal, seconds=MAX_SIM_SECONDS)
        except NoWay as exc:
            if calls == 1:
                return {**record, "outcome": "no_way", "why": exc.cause, "note": str(exc)}
            result = {"arrived": False, "why": exc.cause, "note": str(exc), "time_s": 0.0, "distance_m": 0.0}
            break
        if calls == 1:
            record.update(plan_ms=round(1000 * (time.perf_counter() - t0), 1), path_m=round(trip.plan.length, 2))
            said = trip.plan.short  # what the first plan said it would end short of, if anything
        pg.run_fast(lambda trip=trip: trip.done, MAX_SIM_SECONDS + 1.0)
        result = trip.result()
        spent += result["time_s"]
        walked += result["distance_m"]
        replans += result.get("replans", 0)
        if result.get("why") != "out_of_time" or calls == MAX_CALLS:
            break
    pose = pg.behaviors.pose(robot)
    why = result.get("why")
    # Short as the first plan said, or the planner changed its mind on the way (it set off, then found no way on).
    told = said is not None and why in SHORT and ", and then" not in (result.get("note") or "")
    outcome = "arrived" if result["arrived"] else "short" if told else "failed"
    record.update(
        outcome=outcome,
        why=why,
        note=result.get("note"),
        calls=calls,
        time_s=round(spent, 2),
        distance_m=round(walked, 2),
        replans=replans,
        upright=pose["upright"],
        end=[round(v, 2) for v in pose["position"][:2]],
        wall_s=round(time.perf_counter() - wall, 2),
        within_reach=result.get("within_reach"),
    )
    return record


def run(
    scene: str, robot: str, source: dict[str, Any], trials: int, seed: int, out, anywhere: float = 0.0
) -> list[dict[str, Any]]:
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-navbench-")
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world(f"bench_{scene}_{robot}")
    pg.add_robot(robot, **source)
    pg.load_scene(scene, seed=seed)
    nav, spots = free_spots(pg, robot)
    objects = [o.name for o in pg.current_world().objects]
    rng = np.random.default_rng(seed)
    records = []
    edge = nav_map_edge()
    low = (nav.grid.x0 + edge, nav.grid.y0 + edge)
    high = (
        nav.grid.x0 + (nav.grid.cols - 1) * nav.grid.res - edge,
        nav.grid.y0 + (nav.grid.rows - 1) * nav.grid.res - edge,
    )
    for k in range(trials):
        start = pick(rng, nav, spots, None)
        kind = "anywhere" if rng.random() < anywhere else "object" if rng.random() < OBJECT_SHARE else "spot"
        if kind == "anywhere":
            goal: Any = [round(float(rng.uniform(low[0], high[0])), 2), round(float(rng.uniform(low[1], high[1])), 2)]
        elif kind == "object":
            goal = str(rng.choice(objects))
        else:
            point = pick(rng, nav, spots, start)
            goal = [round(point[0], 2), round(point[1], 2)] if point else None
        if start is None or goal is None:
            continue
        outcome = trial(pg, robot, start, rng.uniform(-180, 180), goal)
        record = {"scene": scene, "robot": robot, "trial": k, "goal_kind": kind, **outcome}
        records.append(record)
        out.write(json.dumps(record) + "\n")
        out.flush()
        keys = ("goal", "outcome", "why", "calls", "time_s", "path_m", "replans", "wall_s")
        brief = {key: record.get(key) for key in keys}
        print(f"{scene} {robot} {k}: {brief}", flush=True)
    return records


# ---- targeted cases ---------------------------------------------------------------------------------------------

NARROW_MAZE = {"cell": 0.7}  # 0.6 m between the hedges: the Go2 needs 0.86 m to turn on the spot
DOCK_LANDING = ((5.05, 7.05), (-2.6, -0.2))  # m, x and y of the warehouse dock's landing


def run_case(case: str, robot: str, source: dict[str, Any], trials: int, seed: int, out) -> list[dict[str, Any]]:
    """Trials of one case (see the module's docstring) for one robot."""
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-navbench-")
    rng = np.random.default_rng(seed)
    scenes = {"dock": ["warehouse"], "ramps": ["warehouse", "obstacle_course"], "narrow": ["maze"]}
    records = []
    for scene in scenes.get(case, ["maze"]):
        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world(f"case_{case}_{scene}_{robot}")
        pg.add_robot(robot, **source)
        pg.load_scene(scene, seed=seed, options=NARROW_MAZE if case == "narrow" else None)
        nav, roomy = free_spots(pg, robot)
        _, anywhere = free_spots(pg, robot, roomy=False)
        for k, (start, yaw, goal) in enumerate(_case_trips(case, nav, roomy, anywhere, rng, trials)):
            outcome = trial(pg, robot, start, yaw, goal)
            record = {"scene": scene, "robot": robot, "trial": k, "goal_kind": case, **outcome}
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            brief = {key: record.get(key) for key in ("start", "goal", "outcome", "why", "time_s", "replans")}
            print(f"{case} {scene} {robot} {k}: {brief}", flush=True)
    return records


def _case_trips(case: str, nav: Any, roomy: np.ndarray, anywhere: np.ndarray, rng, trials: int):
    """(start, yaw in degrees, goal) for each trial of a case."""
    xs, ys = nav.grid.centers()
    if case == "dock":
        on = (xs > DOCK_LANDING[0][0] + 0.3) & (xs < DOCK_LANDING[0][1] - 0.3) & (ys > DOCK_LANDING[1][0] + 0.3)
        landing = anywhere & on & (ys < DOCK_LANDING[1][1] - 0.3)
        floor = roomy & (nav.ground - nav.floor < 0.01)
        for k in range(trials):
            goal = "dock" if k % 2 else _round(pick(rng, nav, landing, None))
            yield pick(rng, nav, floor, (6.0, -1.4)), float(rng.uniform(-180, 180)), goal
    elif case == "ramps":
        raised = anywhere & (nav.ground - nav.floor > 0.1)  # up a ramp: the dock, the course's landing
        floor = roomy & (nav.ground - nav.floor < 0.01)
        for _ in range(trials if raised.any() else 0):
            start = pick(rng, nav, floor, None)
            yield start, float(rng.uniform(-180, 180)), _round(pick(rng, nav, raised, start))
    elif case == "narrow":
        for _ in range(trials):
            start = pick(rng, nav, anywhere, None)
            yield start, fitting_yaw(nav, start, rng), _round(pick(rng, nav, anywhere, start))
    else:  # dead_ends: into one from a roomy start, then out of one, facing its end
        ends = dead_ends(nav, anywhere)
        for k in range(trials if ends else 0):
            x, y, way_in = ends[rng.integers(len(ends))]
            if k % 2 == 0:
                yield pick(rng, nav, roomy, (x, y)), float(rng.uniform(-180, 180)), [round(x, 2), round(y, 2)]
            else:
                yield (x, y), math.degrees(way_in), _round(pick(rng, nav, roomy, (x, y)))


def fitting_yaw(nav: Any, start: tuple[float, float] | None, rng) -> float:
    """A heading (degrees) the robot fits in where it stands: any, with room to turn on the spot; else along the way
    with the longest free run ahead and behind (along a corridor), forward or back. A heading across a 0.6 m
    corridor put the Go2 into both its walls."""
    if start is None:
        return 0.0
    if nav.room[nav.grid.cell(*start)] >= nav.body.reach + 0.05:
        return float(rng.uniform(-180, 180))
    ways = np.linspace(0.0, math.pi, 16, endpoint=False)
    runs = [_free_run(nav, start, way) + _free_run(nav, start, way + math.pi) for way in ways]
    way = float(ways[int(np.argmax(runs))]) + (math.pi if rng.random() < 0.5 else 0.0)
    return math.degrees(way + rng.uniform(-0.1, 0.1))


def _free_run(nav: Any, start: tuple[float, float], way: float, most: float = 1.0) -> float:
    """How far (m, up to `most`) the robot's middle stays on free cells going from `start` along `way` (rad)."""
    run = 0.0
    while run < most:
        cell = nav.grid.cell(start[0] + (run + 0.05) * math.cos(way), start[1] + (run + 0.05) * math.sin(way))
        if not (nav.grid.inside(*cell) and nav.free[cell]):
            break
        run += 0.05
    return run


def dead_ends(nav: Any, free: np.ndarray, ring: float = 0.6) -> list[tuple[float, float, float]]:
    """Free cells at the end of a dead end, at least 0.8 m apart: (x, y, the way in, rad). Around one, on a circle
    `ring` m across, the free cells make a single short arc: the way it is entered."""
    angles = np.linspace(-math.pi, math.pi, 36, endpoint=False)
    found: list[tuple[float, float, float]] = []
    for r, c in np.argwhere(free)[::3]:
        x, y = nav.grid.center(int(r), int(c))
        if any(math.dist((x, y), e[:2]) < 0.8 for e in found):
            continue
        cells = [nav.grid.cell(x + ring * math.cos(a), y + ring * math.sin(a)) for a in angles]
        open_ = np.array([nav.grid.inside(*cell) and bool(free[cell]) for cell in cells])
        arcs = int(np.sum(open_ & ~np.roll(open_, 1)))
        if arcs == 1 and open_.sum() <= 6:  # one way out, at most 60 degrees of the circle
            out = math.atan2(np.sin(angles[open_]).mean(), np.cos(angles[open_]).mean())
            found.append((x, y, out + math.pi))
    return found


def _round(point: tuple[float, float] | None) -> list[float] | None:
    return None if point is None else [round(point[0], 2), round(point[1], 2)]


# ---- the report ------------------------------------------------------------------------------------------------


def nav_map_edge() -> float:
    from robotforge.behaviors import nav_map

    return nav_map.EDGE


def report(all_records: list[dict[str, Any]]) -> str:
    """Markdown tables: per scene and robot, then failures and the planner's answers by cause; goals anywhere and
    the targeted cases apart."""
    cases = [r for r in all_records if r.get("goal_kind") in CASES and r["outcome"] != "skipped"]
    records = [r for r in all_records if r.get("goal_kind") in ("spot", "object")]
    anywhere = [r for r in all_records if r.get("goal_kind") == "anywhere"]
    lines: list[str] = []
    if records:  # random trips (not only targeted cases)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            groups[(r["scene"], r["robot"])].append(r)
        lines += [
            "| scene | robot | trips | arrived | in one call | median time (s) | median path (m) | speed (m/s) "
            "| replans | short | no way |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for (scene, robot), rs in sorted(groups.items()):
            trips = [r for r in rs if r["outcome"] in ("arrived", "failed")]
            ok = [r for r in trips if r["outcome"] == "arrived"]
            once = sum(r.get("calls", 1) == 1 for r in ok)
            times = [r["time_s"] for r in ok]
            paths = [r["path_m"] for r in ok]
            speed = sum(paths) / sum(times) if times else float("nan")
            short = sum(r["outcome"] == "short" for r in rs)
            no_way = sum(r["outcome"] == "no_way" for r in rs)
            rate = f"{len(ok)}/{len(trips)} ({100 * len(ok) / len(trips):.0f}%)" if trips else "-"
            lines.append(
                f"| {scene} | {robot} | {len(trips)} | {rate} | {once} | {_median(times)} | {_median(paths)} | "
                f"{speed:.2f} | {sum(r.get('replans', 0) for r in trips)} | {short} | {no_way} |"
            )
        failed = Counter((r["robot"], r["why"]) for r in records if r["outcome"] == "failed")
        short = Counter((r["robot"], r["why"]) for r in records if r["outcome"] == "short")
        refused = Counter((r["robot"], r["why"]) for r in records if r["outcome"] == "no_way")
        lines += ["", "Failed trips by cause: " + (_counts(failed) or "none") + "."]
        lines += [
            "Ended short as the planner said (beside what it cannot reach), by cause: "
            + (_counts(short) or "none")
            + "."
        ]
        lines += ["No way (the planner's answer) by cause: " + (_counts(refused) or "none") + "."]
        total = [r for r in records if r["outcome"] in ("arrived", "failed")]
        arrived = [r for r in total if r["outcome"] == "arrived"]
        if total:
            once = sum(r.get("calls", 1) == 1 for r in arrived)
            lines.append(
                f"Overall: {len(arrived)} of {len(total)} trips arrived ({100 * len(arrived) / len(total):.1f}%), "
                f"{once} within one move call of 60 s ({100 * once / len(total):.1f}%)."
            )
        noted = sum(bool(r.get("note")) for r in arrived)
        unreachable = sum(r.get("within_reach") is False for r in arrived)
        lines.append(
            f"Of those that arrived, {noted} said something (next to the counter or shelf an object is on, not at its "
            f"usual distance), and {unreachable} stood next to an object their arm cannot reach."
        )
    plans = [r["plan_ms"] for r in all_records if "plan_ms" in r]
    if plans:
        lines.append(f"Planning took {statistics.median(plans):.0f} ms at the median, {max(plans):.0f} ms at most.")
    if anywhere:
        answers = Counter(
            "arrived" if r["outcome"] == "arrived" else f"{_word(r['outcome'])}: {r['why']}" for r in anywhere
        )
        lines += ["", f"Goals anywhere in the scene: {len(anywhere)}; " + _plain(answers) + "."]
    if cases:
        lines += [
            "",
            "| case | robot | trips | arrived | failed, by cause | short | no way |",
            "|---|---|---|---|---|---|---|",
        ]
        by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in cases:
            by_case[(r["goal_kind"], r["robot"])].append(r)
        for (case, robot), rs in sorted(by_case.items()):
            trips = [r for r in rs if r["outcome"] in ("arrived", "failed")]
            ok = sum(r["outcome"] == "arrived" for r in trips)
            failed = Counter(r["why"] for r in trips if r["outcome"] == "failed")
            short = Counter(r["why"] for r in rs if r["outcome"] == "short")
            no_way = Counter(r["why"] for r in rs if r["outcome"] == "no_way")
            rate = f"{ok}/{len(trips)}" if trips else "-"
            lines.append(
                f"| {case} | {robot} | {len(trips)} | {rate} | {_plain(failed) or '-'} | {_plain(short) or '-'} | "
                f"{_plain(no_way) or '-'} |"
            )
    return "\n".join(lines)


def _word(outcome: str) -> str:
    return {"no_way": "no way", "short": "stopped short", "failed": "failed"}[outcome]


def _plain(counter: Counter) -> str:
    return ", ".join(f"{what} {n}" for what, n in counter.most_common())


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def _counts(counter: Counter) -> str:
    return ", ".join(f"{robot} {why}: {n}" for (robot, why), n in sorted(counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=list(SCENES))
    parser.add_argument("--robots", nargs="*", default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anywhere", type=float, default=0.0, help="share of goals at any point of the scene")
    parser.add_argument("--json", type=Path, default=None, help="write one JSON line per trial here")
    parser.add_argument("--report", nargs="*", type=Path, help="print the tables for these JSON line files")
    parser.add_argument("--case", choices=CASES, help="run one targeted case instead of random trips per scene")
    args = parser.parse_args()
    if args.report:
        records = [json.loads(line) for path in args.report for line in path.read_text(encoding="utf-8").splitlines()]
        print(report(records))
        return
    sources = robots()
    names = args.robots or list(sources)
    out_path = args.json or Path(tempfile.mkdtemp()) / "nav_bench.jsonl"
    records = []
    with out_path.open("w", encoding="utf-8") as out:
        if args.case:
            for name in names:
                records += run_case(args.case, name, sources[name], args.trials, args.seed, out)
        for scene in [] if args.case else args.scenes:
            for name in names:
                records += run(scene, name, sources[name], args.trials, args.seed, out, args.anywhere)
    print(report(records))
    print(f"trials written to {out_path}")


if __name__ == "__main__":
    main()
