"""Does a humanoid stand on its own, under a load and a push? Measure it, robot by robot.

    uv run python scripts/balance_bench.py [--robots g1,t1,h1_2,...|all] [--jobs 3] [--json out.json]
                                           [--shoves 0.1,0.2,0.3,0.4] [--load 0.2]

Every run adds one robot to an empty world and leaves it alone with whatever stands it up when it appears (what a
beginner who adds it gets: the walking gait of the tuned humanoids, the whole-body balance of the rest; see
behaviors/whole_body.py). It then watches, from the same state each time:
- stand: six seconds standing, upright (trunk tilt under 20 degrees) and no lower than 85 % of its height;
- load: a backpack of LOAD (a share of the robot's own mass) added to the trunk, standing on for five seconds more;
- shove: 0.1, 0.2, ... m/s added to the trunk's velocity forwards, backwards, to the left and to the right in one
  go; the table gives the largest one that it was still standing after, upright, for five seconds (0 for none).
The robots are the library's humanoids and the catalogue's two-legged robots (`all`); downloads happen on first use.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

LIBRARY = ("g1", "t1", "op3")
SETTLE_S = 6.0
WATCH_S = 5.0
UPRIGHT = 20.0  # degrees of trunk tilt
HEIGHT_KEPT = 0.85
SHOVES = (0.1, 0.2, 0.3, 0.4)
DIRECTIONS = (("forward", (1.0, 0.0)), ("back", (-1.0, 0.0)), ("left", (0.0, 1.0)), ("right", (0.0, -1.0)))
LOAD = 0.2
LOAD_S, LOAD_STEPS = 1.5, 15


@dataclass
class Result:
    robot: str
    controller: str = ""
    kind: str = ""
    mass: float = 0.0
    stands: bool = False
    tilt: float = 0.0  # degrees at the end of the stand
    why: str = ""  # why it does not stand
    load_ok: bool | None = None
    shove: dict[str, float] = field(default_factory=dict)  # direction -> largest m/s survived
    seconds: float = 0.0
    note: str = ""


def _tilt(up: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, up))))


def run(robot: str, shoves: tuple[float, ...] = SHOVES, load: float = LOAD) -> Result:
    import mujoco
    import numpy as np

    from robotforge.design import catalogue_probe
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

    out = Result(robot)
    with tempfile.TemporaryDirectory(prefix="rf-balance-") as scratch:
        pg = Playground.with_demo()
        pg.store = WorldStore.create("balance", parent=Path(scratch))
        pg._load_world(pg.store.compile(), created=True)
        token = catalogue_probe._probing.set(True)  # the bench is its own trial
        try:
            pg.add_robot("r", library=robot)
        finally:
            catalogue_probe._probing.reset(token)
        session = pg.session
        m, d = session.model, session.data
        info = next((r for r in pg.behaviors.robots() if r["name"] == "r"), {})
        out.kind = str(info.get("kind", ""))
        ctrl = session.controllers.get("r")
        out.controller = type(ctrl).__name__ if ctrl is not None else "none"
        roots = (b for b in range(1, m.nbody) if m.body_parentid[b] == 0)
        trunk = next(b for b in roots if (m.body(b).name or "").startswith("r/"))
        root_dof = int(m.jnt_dofadr[m.body_jntadr[trunk]])
        out.mass = round(float(m.body_subtreemass[trunk]), 1)

        def tilt() -> float:
            return _tilt(float(d.xmat[trunk][8]))

        start_z = float(d.xpos[trunk][2])

        def standing(height: float) -> bool:
            high_enough = float(d.xpos[trunk][2]) > HEIGHT_KEPT * height
            return tilt() < UPRIGHT and high_enough and bool(np.isfinite(d.qpos).all())

        ran = pg.run_fast(lambda: False, SETTLE_S)
        z0 = float(d.xpos[trunk][2])
        out.tilt = round(tilt(), 1)
        out.stands = ran >= SETTLE_S - 1e-6 and standing(max(start_z, z0)) and z0 > HEIGHT_KEPT * start_z
        if not out.stands:
            out.why = f"tilt {out.tilt} deg, height {z0:.2f} m of {start_z:.2f} m after {SETTLE_S:.0f} s"
            return out
        snap = (d.qpos.copy(), d.qvel.copy(), d.time, d.ctrl.copy(), d.act.copy())

        def restore() -> None:
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:], d.time = snap[0], snap[1], snap[2]
            d.ctrl[:] = snap[3]
            if m.na:
                d.act[:] = snap[4]
            mujoco.mj_forward(m, d)
            with contextlib.suppress(AttributeError):  # a learned walk reads whether it has fallen off the robot
                ctrl.fallen = False

        if load:
            was = float(m.body_mass[trunk]), m.body_inertia[trunk].copy()
            extra = load * out.mass
            for step in range(1, LOAD_STEPS + 1):  # taken up over a second and a half, as a hand puts a load on
                m.body_mass[trunk] = was[0] + extra * step / LOAD_STEPS
                m.body_inertia[trunk] = was[1] * m.body_mass[trunk] / was[0]
                mujoco.mj_setConst(m, d)
                pg.run_fast(lambda: False, LOAD_S / LOAD_STEPS)
            pg.run_fast(lambda: False, WATCH_S)
            out.load_ok = standing(z0)
            m.body_mass[trunk], m.body_inertia[trunk] = was
            mujoco.mj_setConst(m, d)
            restore()
        yaw = math.atan2(d.xmat[trunk][3], d.xmat[trunk][0])
        for name, (fx, fy) in DIRECTIONS:
            best = 0.0
            for speed in shoves:
                restore()
                world = np.array([math.cos(yaw) * fx - math.sin(yaw) * fy, math.sin(yaw) * fx + math.cos(yaw) * fy])
                d.qvel[root_dof : root_dof + 2] += speed * world
                pg.run_fast(lambda: False, WATCH_S)
                if not standing(z0):
                    break
                best = speed
            out.shove[name] = best
    return out


def _run(args: tuple[str, tuple[float, ...], float]) -> Result:
    import time

    start = time.time()
    try:
        result = run(*args)
    except Exception as exc:  # report the failure in the table rather than lose the whole batch
        result = Result(args[0], note=f"error: {type(exc).__name__}: {str(exc)[:160]}")
    result.seconds = round(time.time() - start, 1)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robots", default=",".join(LIBRARY), help="comma separated, or 'all' for the catalogue too")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--shoves", default=",".join(str(s) for s in SHOVES))
    parser.add_argument("--load", type=float, default=LOAD, help="share of the robot's mass added to the trunk")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    if args.robots == "all":
        from robotforge.design import catalogue

        names = [*LIBRARY, *(e.id for e in catalogue.entries().values() if e.kind == "humanoid")]
    else:
        names = args.robots.split(",")
    shoves = tuple(float(s) for s in args.shoves.split(","))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_run, [(n, shoves, args.load) for n in names]))
    print("robot                  kind      controller             kg     stands  load   shove f/b/l/r (m/s)   s")
    for r in results:
        shove = "/".join(f"{r.shove.get(n, 0):.1f}" for n, _ in DIRECTIONS) if r.shove else "-"
        load = "-" if r.load_ok is None else ("yes" if r.load_ok else "no")
        print(
            f"{r.robot:<22} {r.kind:<9} {r.controller:<22} {r.mass:6.1f} {'yes' if r.stands else 'no':<7} {load:<5} "
            f"{shove:<22} {r.seconds:5.1f}  {r.why or r.note}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    return 0 if not any(r.note.startswith("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
