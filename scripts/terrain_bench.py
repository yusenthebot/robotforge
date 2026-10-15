"""How well do the built-in walking and driving behaviors cope with terrain? Measure it.

    uv run python scripts/terrain_bench.py [--robots dog,go2,rover,humanoid] [--terrains ...] [--jobs 4]

Each run puts one robot 0.6 m before a terrain patch that crosses its path, commands it
straight ahead with `move` (60 % of its top speed, like holding W in the viewer) and steps
the physics. It reports the distance covered, whether it fell (trunk tilted past 45 degrees)
and whether it made it: past the far end of the patch and still upright. The library Go2
is skipped unless it is already downloaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

ROBOTS = {
    "dog": {"template": "quadruped"},
    "go2": {"library": "go2"},
    "rover": {"template": "rover"},
    "humanoid": {"template": "humanoid"},
}
TERRAINS: dict[str, dict | None] = {
    "flat": None,
    "ramp5": {"type": "ramp", "angle": 5, "length": 1.2, "landing": 0.6},
    "ramp10": {"type": "ramp", "angle": 10, "length": 1.2, "landing": 0.6},
    "ramp15": {"type": "ramp", "angle": 15, "length": 1.2, "landing": 0.6},
    "stairs4cm": {"type": "stairs", "steps": 3, "step_height": 0.04, "step_depth": 0.3, "landing": 0.6},
    "stairs8cm": {"type": "stairs", "steps": 3, "step_height": 0.08, "step_depth": 0.3, "landing": 0.6},
    "rough3cm": {"type": "rough", "size": [3.0, 1.6], "height": 0.03, "seed": 1},
    "rough6cm": {"type": "rough", "size": [3.0, 1.6], "height": 0.06, "seed": 1},
    "hills": {"type": "hills", "size": [4.0, 2.4], "height": 0.2, "scale": 1.0, "seed": 1},  # slopes up to 27 deg
}
GAP = 0.6  # m of flat floor between the robot and the patch
FLAT_RUN = 3.0  # m a robot must cover on flat ground to count as a success
SPEED_SHARE = 0.6


@dataclass
class Result:
    robot: str
    terrain: str
    success: bool
    distance: float
    fell: bool
    climbed: float  # highest trunk rise over its start height, m
    seconds: float
    note: str = ""


def run(robot: str, terrain_name: str) -> Result:
    from robotforge.design import library, terrain
    from robotforge.playground import Playground

    if robot == "go2" and not library.is_downloaded("go2"):
        return Result(robot, terrain_name, False, 0.0, False, 0.0, 0.0, "skipped: Go2 not downloaded")
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_bench_")
    pg = Playground.with_demo()
    pg.new_world(f"bench_{robot}_{terrain_name}")
    pg.add_robot(robot, **ROBOTS[robot])
    patch = TERRAINS[terrain_name]
    end = FLAT_RUN
    if patch is not None:
        length = terrain.parse(dict(patch)).footprint()[0]
        pg.add_terrain(pos=[GAP + length / 2, 0.0], **_split(patch))
        end = GAP + length + 0.2
    behaviors = pg.behaviors
    start = behaviors.pose(robot)["position"]
    top = next(r["max_speed"] for r in behaviors.robots() if r["name"] == robot)
    speed = SPEED_SHARE * top
    limit = min(60.0, 2.5 * (end + 0.5) / speed + 3.0)
    behaviors.move(robot, speed, 0.0)
    fell, climbed, t = False, 0.0, 0.0
    while t < limit:
        t += pg.run_fast(lambda: False, 0.1)
        pose = behaviors.pose(robot)
        climbed = max(climbed, pose["position"][2] - start[2])
        if not pose["upright"]:
            fell = True
            break
        if pose["position"][0] - start[0] > end + 0.3:
            break
    pose = behaviors.pose(robot)
    distance = pose["position"][0] - start[0]
    success = not fell and distance > end
    return Result(robot, terrain_name, success, round(distance, 2), fell, round(climbed, 3), round(t, 1))


def _split(patch: dict) -> dict:
    """add_terrain(type, ..., params) from a patch dict."""
    rest = {k: v for k, v in patch.items() if k != "type"}
    return {"type": patch["type"], "params": rest}


def _run(args: tuple[str, str]) -> Result:
    try:
        return run(*args)
    except Exception as exc:  # report the failure in the table rather than lose the whole batch
        return Result(args[0], args[1], False, 0.0, False, 0.0, 0.0, f"error: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--terrains", default=",".join(TERRAINS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    jobs = [(r, t) for r in args.robots.split(",") for t in args.terrains.split(",")]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_run, jobs))
    width = max(len(t) for t in TERRAINS)
    print(f"{'robot':<9} {'terrain':<{width}}  ok  dist m  fell  climb m  sim s  note")
    for r in results:
        ok = "yes" if r.success else "no"
        fell = "yes" if r.fell else "-"
        print(
            f"{r.robot:<9} {r.terrain:<{width}}  {ok:<3} {r.distance:6.2f}  {fell:<4}  {r.climbed:7.3f}  "
            f"{r.seconds:5.1f}  {r.note}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=1)
    return 0 if all(math.isfinite(r.distance) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
