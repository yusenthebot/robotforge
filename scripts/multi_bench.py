"""Two robots in one world: handing an object from one arm to another, and rovers whose paths cross.

    uv run python scripts/multi_bench.py                    # both, 12 handovers and 4 crossings
    uv run python scripts/multi_bench.py --handovers 20 --crossings 0 --seed 5

Handover: two Pandas 1.3 m apart, a cube somewhere on the left arm's side of the table, a pad on the right arm's
side; the left arm hands it to the right one (`pick_and_place(..., hand_to=...)`). It counts the cube lying on the
pad, and the closest the two arms came (a contact deeper than 5 mm is a collision).
Crossing: two rovers sent across each other's way so that they get to the middle together, a T meeting, and three
rovers meeting from three sides; it counts arrivals, the closest the robots came and the time against the same run
with no right of way.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("ROBOTFORGE_HOME", tempfile.mkdtemp(prefix="rf-multi-"))

from robotforge.playground import Playground  # noqa: E402

ARMS_APART = 0.65
WORLDS = itertools.count()
CRASH = 0.005  # m: arms pressed this far into each other have collided


def handover(rng: np.random.Generator) -> dict:
    cube = [float(rng.uniform(-0.40, -0.20)), float(rng.uniform(-0.25, 0.25))]
    pad_y = float(rng.uniform(0.1, 0.3)) * (1 if rng.random() < 0.5 else -1)
    pg = Playground.with_demo()
    pg.new_world(f"bench{next(WORLDS)}", "wood")
    pg.add_robot("left", library="panda", pos=[-ARMS_APART, 0.0], yaw=0)
    pg.add_robot("right", library="panda", pos=[ARMS_APART, 0.0], yaw=180)
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], pos=cube, color="red", mass=0.1)
    pg.add_object("pad", shape="box", size=[0.2, 0.2, 0.01], pos=[ARMS_APART - 0.35, pad_y], color="blue", fixed=True)
    pg.run_fast(lambda: False, 1.0)
    out: dict = {"cube": [round(v, 2) for v in cube], "pad_y": round(pad_y, 2)}
    try:
        handed, info = pg.behaviors.hand_over("left", "right", "cube", "pad")
    except Exception as exc:  # noqa: BLE001 - a refusal is a result
        return {**out, "ok": False, "why": f"refused: {str(exc)[:80]}"}
    model, data = pg.session.model, pg.session.data
    pressed, t0 = 0.0, data.time
    while not handed.done and data.time - t0 < handed.time_limit:
        pg.run_fast(lambda: False, 0.1)
        for i in range(data.ncon):
            con = data.contact[i]
            names = {model.body(model.geom_bodyid[con.geom1]).name, model.body(model.geom_bodyid[con.geom2]).name}
            if any(n.startswith("left/") for n in names) and any(n.startswith("right/") for n in names):
                pressed = max(pressed, -float(con.dist))
    pg.run_fast(lambda: False, 0.5)
    ok = bool(pg.behaviors.placed("cube", info["region"])["success"])
    return {
        **out,
        "ok": ok,
        "arms_pressed_mm": round(1000 * pressed, 1),
        "seconds": round(data.time - t0, 1),
        "why": "" if ok else (handed.note or "the cube is not on the pad"),
    }


def crossing(kind: str, traffic: bool) -> dict:
    pg = Playground.with_demo()
    pg.new_world(f"cross{next(WORLDS)}", "wood")
    starts = {
        "cross": [(-1.5, 0.0, 0), (0.0, -1.5, 90)],
        "tee": [(-1.5, 0.0, 0), (0.1, -1.5, 90)],
        "three": [(-1.5, 0.0, 0), (0.75, -1.3, 120), (0.75, 1.3, -120)],
    }[kind]
    goals = {
        "cross": [(1.6, 0.0), (0.0, 1.6)],
        "tee": [(1.6, 0.0), (0.1, 0.0)],
        "three": [(1.6, 0.0), (-0.8, 1.4), (-0.8, -1.4)],
    }[kind]
    names = [f"r{i}" for i in range(len(starts))]
    for name, (x, y, yaw) in zip(names, starts, strict=True):
        pg.add_robot(name, template="rover", pos=[x, y], yaw=yaw)
    pg.run_fast(lambda: False, 0.5)
    trips = []
    for name, goal in zip(names, goals, strict=True):
        try:
            trips.append(pg.behaviors.navigate(name, list(goal), seconds=60))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "why": f"refused: {str(exc)[:80]}"}
    if not traffic:
        for trip in trips:
            trip.crossings = None
    data = pg.session.data
    t0, closest = data.time, 9.0
    while not all(t.done for t in trips) and data.time - t0 < 61:
        pg.run_fast(lambda: False, 0.1)
        poses = [pg.behaviors.pose(n)["position"][:2] for n in names]
        closest = min(closest, *(math.dist(a, b) for i, a in enumerate(poses) for b in poses[i + 1 :]))
    results = [t.result() for t in trips]
    return {
        "ok": all(r["arrived"] for r in results),
        "closest_m": round(closest, 2),
        "seconds": round(max(r["time_s"] for r in results), 1),
        "waited_s": round(sum(r.get("waited_s", 0.0) for r in results), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--handovers", type=int, default=12)
    ap.add_argument("--crossings", type=int, default=1, help="repeats of each crossing kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    report: dict = {"handover": [], "crossing": {}}
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for i in range(args.handovers):
        r = handover(rng)
        report["handover"].append(r)
        print(
            f"handover {i:2d} cube {r['cube']} pad y {r['pad_y']:+.2f}: {'ok' if r['ok'] else 'FAIL'} "
            f"{r.get('seconds', '')}s pressed {r.get('arms_pressed_mm', '-')} mm {r['why']}",
            flush=True,
        )
    for kind in ("cross", "tee", "three"):
        for _ in range(args.crossings):
            on, off = crossing(kind, True), crossing(kind, False)
            report["crossing"].setdefault(kind, []).append({"right_of_way": on, "reactive_only": off})
            print(f"{kind:6s} right of way {on}  |  making way only {off}", flush=True)
    if report["handover"]:
        ok = sum(r["ok"] for r in report["handover"])
        bad = sum(r.get("arms_pressed_mm", 0) > 1000 * CRASH for r in report["handover"])
        print(f"\nhandover: {ok}/{len(report['handover'])} on the pad, {bad} with the arms pressed into each other")
    print(f"({time.time() - t0:.0f} s)")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
