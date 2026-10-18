"""Mobile manipulation jobs in the home scene, run headless over robots and seeds: success rates and failures by cause.

    uv run python scripts/mobile_bench.py [--robots go2_arm mobile_panda ...] [--jobs fetch drawer ...]
                                          [--seeds 5] [--workers 3] [--json out.jsonl]

The jobs, each as an agent would do it with the tools (pick_and_place, operate, move):
- door: from the living room, through the swing door into the kitchen (move with to="kitchen").
- fetch: "go to the kitchen and bring the apple to the living-room table" (pick_and_place goes over by itself).
- drawer: open the chest of drawers in the kitchen and take the block out onto the kitchen table.
- button: press the button on the living room's side table.
A seed jitters the things on the tables (the scene's own seed) and where and how the robot starts. The demo videos
(scripts/demos.py mobile_*) run the same jobs in real time. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

ROBOTS: dict[str, dict[str, Any]] = {
    "go2_arm": {"design": "go2_arm"},
    "mobile_panda": {"design": "mobile_panda"},
    "lift_picker": {"design": "lift_picker"},
    "spot_z1": {"library": "spot", "mounts": [{"library": "z1"}]},
    "anymal_ur5e": {"library": "anymal_c", "mounts": [{"library": "ur5e"}, {"library": "robotiq_2f85"}]},
}
JOBS = ("door", "fetch", "drawer", "button")
NAME = "mm"  # the robot's name in every run
STARTS = ((2.4, -1.5), (1.4, -1.7), (3.0, -0.6))  # in the living room, clear of its furniture
DOOR_PUSHED = 45.0  # degrees the door must have swung for the robot to have pushed through it
OPERATE_S, ERRAND_S, TRIP_S = 150.0, 200.0, 90.0  # sim seconds each may take

Wait = Callable[[Callable[[], bool], float], None]


def build(pg: Any, robot: str, seed: int) -> None:
    """The home (seed `seed`) with `robot` at one of the living room's starting spots, turned some way."""
    rng = random.Random(seed)
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
    return {k: out[k] for k in ("success", "why", "note", "errand_s", "driven_m") if k in out}


def job_drawer(pg: Any, wait: Wait) -> dict[str, Any]:
    opened = operate(pg, "chest", "open", wait)
    if not opened.get("success"):
        return {"success": False, "why": f"open:{opened.get('why', 'short')}", "note": opened.get("note")}
    out = pick_and_place(pg, "block", "kitchen_table", wait)
    res = {k: out[k] for k in ("success", "why", "note") if k in out}
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


JOB_FUNCS: dict[str, Callable[[Any, Wait], dict[str, Any]]] = {
    "door": job_door,
    "fetch": job_fetch,
    "drawer": job_drawer,
    "button": job_button,
}


def cause(text: str) -> str:
    """A short cause for a refusal's message (what the tools raise before anything moves)."""
    for key, words in (
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


def run_one(robot: str, job: str, seed: int) -> dict[str, Any]:
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf-mobile-")
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    t0 = time.monotonic()
    out: dict[str, Any] = {"robot": robot, "job": job, "seed": seed}
    try:
        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world(f"{robot}_{job}_{seed}")
        build(pg, robot, seed)
        pg.run_fast(lambda: False, 1.0)
        out.update(JOB_FUNCS[job](pg, lambda until, seconds: pg.run_fast(until, seconds)))
        out["sim_s"] = round(float(pg.session.data.time), 1)
        if not pg.behaviors.pose(NAME)["upright"]:
            out.update(success=False, why="fell")
    except Exception as exc:  # a refusal before it moves is a result too: the cause is counted
        text = str(exc)
        known = type(exc).__name__ in ("OperateError", "BehaviorError", "FetchError", "NoWay")
        out.update(success=False, why=cause(text) if known else f"error:{type(exc).__name__}", note=text[:300])
        if not known:
            out["trace"] = traceback.format_exc()[-800:]
    out["wall_s"] = round(time.monotonic() - t0, 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robots", nargs="*", default=list(ROBOTS))
    parser.add_argument("--jobs", nargs="*", default=list(JOBS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--json", help="write each run as a line of JSON here")
    args = parser.parse_args()
    runs = [(r, j, s) for r in args.robots for j in args.jobs for s in range(args.seeds)]
    results = []
    sink = open(args.json, "w", encoding="utf-8") if args.json else None  # noqa: SIM115
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_one, *zip(*runs, strict=True)):
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
            if sink:
                sink.write(json.dumps(result) + "\n")
                sink.flush()
    table: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        table[(r["robot"], r["job"])].append(r)
    print("\nrobot          job      success  failures by cause")
    for (robot, job), rows in table.items():
        good = sum(1 for r in rows if r.get("success"))
        causes = Counter(r.get("why") or "?" for r in rows if not r.get("success"))
        text = ", ".join(f"{c} x{n}" for c, n in causes.most_common()) or "-"
        print(f"{robot:14} {job:8} {good}/{len(rows):<6} {text}")
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
