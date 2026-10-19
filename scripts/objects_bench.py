"""Pick and place every modelled object with real Menagerie arms, over seeds, and say how it went and why it failed.

    uv run python scripts/objects_bench.py [--arms panda,ur5e] [--objects mug,can,...] [--seeds 4] [--out FILE]

A table with a real arm in front of it (a Panda, or a UR5e with a Robotiq 2F-85) and a tray; each trial adds one object
(a shape and a seed), has the arm pick it up and put it in the tray with `pick_and_place`, and reads what happened from
the simulation: refused (the arm could not take hold of it, with its reason), slipped (lifted, then fell on the way),
never lifted, ended tipped over, ended outside the tray. Prints the success rate per object and per cause and writes
the rows as JSON. Maintainer tool: it needs the real Menagerie models (downloaded once).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ARMS = {
    "panda": ("franka_emika_panda", None),
    "ur5e": ("ur5e", [{"library": "robotiq_2f85"}]),
}
OBJECTS = (
    "mug cup bowl plate bottle can apple orange pear banana package tin brick squeeze_bottle screwdriver hammer wrench"
).split()
TABLE_Z = 0.4
PICK = (0.45, -0.22)
TRAY = (0.45, 0.28)


def world(arm: str) -> Any:
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp()
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("bench")
    library, mounts = ARMS[arm]
    pg.add_robot("arm", library=library, mounts=mounts, pos=[0, 0, TABLE_Z])
    pg.add_object("table", shape="table", size=[1.4, 1.2, TABLE_Z], pos=[0.4, 0], fixed=True)
    pg.add_object("tray", shape="tray", size=[0.3, 0.26, 0.05], pos=[TRAY[0], TRAY[1], TABLE_Z], color="blue")
    pg.run_fast(lambda: False, 0.5)
    return pg


def trial(pg: Any, shape: str, seed: int) -> dict[str, Any]:
    import numpy as np

    pg.add_object("thing", shape=shape, pos=[PICK[0], PICK[1], TABLE_Z], seed=seed)
    pg.run_fast(lambda: False, 0.8)
    m, d = pg.session.model, pg.session.data
    body = m.body("thing").id
    start = d.xpos[body].copy()
    row: dict[str, Any] = {"shape": shape, "seed": seed}
    arm = pg.behaviors.resolve("arm", arm=True)
    t0 = time.time()
    try:
        motion, info = pg.behaviors.pick_and_place(arm, "thing", "tray")
    except Exception as exc:  # the planner's own refusal: the reason is the answer
        row.update(result="refused", why=str(exc)[:240])
        pg.remove("thing")
        return row
    best = [start[2]]

    def watch() -> bool:
        best.append(float(d.xpos[body][2]))
        return bool(motion.done)

    pg.run_fast(watch, 45)
    pg.run_fast(lambda: False, 1.0)
    end = d.xpos[body].copy()
    result = pg.behaviors.placed("thing", info["region"])
    lifted = max(best) - start[2]
    row.update(
        grasp=(info.get("grasp") or {}).get("kind"),
        lifted_cm=round(lifted * 100, 1),
        tilt_deg=result.get("tilt_deg"),
        end=[round(float(v), 3) for v in end],
        wall_s=round(time.time() - t0, 1),
    )
    tilt = float(result.get("tilt_deg") or 0.0)
    tipped_ok = shape in ("hammer", "screwdriver", "wrench", "banana", "book")  # lying things may lean on the rim
    if result["success"] and float(end[2]) > TABLE_Z + 0.09:
        row.update(result="failed", cause="still in the fingers, held in the air over the tray")
    elif result["success"] and (tilt < 25 or tipped_ok):
        row["result"] = "ok"
    elif result["success"]:
        row.update(result="failed", cause="ended tipped over in the tray")
    elif lifted < 0.02:
        row.update(result="failed", cause="never lifted (the grip did not take hold)")
    elif float(np.hypot(end[0] - start[0], end[1] - start[1])) < 0.15:
        row.update(result="failed", cause="slipped out of the fingers on the way")
    else:
        row.update(result="failed", cause="landed outside the tray")
    pg.remove("thing")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="panda,ur5e")
    parser.add_argument("--objects", default=",".join(OBJECTS))
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for arm in args.arms.split(","):
        pg = world(arm)
        for shape in args.objects.split(","):
            outcomes = []
            for seed in range(args.seeds):
                row = {"arm": arm, **trial(pg, shape, seed)}
                rows.append(row)
                outcomes.append(row["result"])
                note = row.get("cause") or row.get("why") or ""
                print(f"{arm:6s} {shape:15s} seed {seed}: {row['result']:8s} {row.get('grasp') or '':8s} {note}", flush=True)
            print(f"{arm:6s} {shape:15s} {outcomes.count('ok')}/{len(outcomes)}", flush=True)
    print()
    for arm in args.arms.split(","):
        mine = [r for r in rows if r["arm"] == arm]
        ok = sum(r["result"] == "ok" for r in mine)
        print(f"{arm}: {ok}/{len(mine)} placed ({100 * ok / max(len(mine), 1):.0f}%)")
        causes = Counter(r.get("cause") or ("refused" if r["result"] == "refused" else "") for r in mine if r["result"] != "ok")
        for cause, n in causes.most_common():
            print(f"   {n:3d}  {cause}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
