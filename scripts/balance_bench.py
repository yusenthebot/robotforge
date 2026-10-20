"""Does a humanoid stand on its own, under a load and a push? Measure it, robot by robot.

    uv run python scripts/balance_bench.py [--robots g1,t1,h1_2,...|all] [--jobs 3] [--json out.json]
                                           [--shoves 0.1,0.2,0.3,0.4] [--load 0.2]

Every run adds one robot to an empty world and leaves it alone with whatever stands it up when it appears (what a
beginner who adds it gets: the walking gait of the tuned humanoids, the whole-body balance of the rest; see
behaviors/whole_body.py). It then watches, from the same state each time:
- stand: six seconds standing, upright (trunk tilt under 20 degrees), no lower than 85 % of its height and never
  fallen over on the way (a balance that declared the robot fallen and fell back on its servos, however upright
  it ends, is not standing);
- load: a backpack of LOAD (a share of the robot's own mass) added to the trunk, standing on for five seconds more;
- shove: 0.1, 0.2, ... m/s added to the trunk's velocity forwards, backwards, to the left and to the right in one
  go (a velocity impulse on the trunk, not a force on a spring); every speed is tried, and the table gives the
  largest one that it was still standing after, upright, for five seconds (0 for none; a star marks a direction
  where a smaller shove had thrown it and a larger one did not).
The load is a mass added to the trunk in the model, which the balance reads from the model on every tick: it is a
load the controller can feel, not one it has to guess. "leg" is how straight its legs are standing (height of the
trunk over the feet divided by the length of the leg as drawn; 1.00 is straight, a squat is under 0.80); "cpu s/s" is
the CPU time one second of simulation took, measured in the process (the machine's load does not count).
The robots are the library's humanoids and the catalogue's two-legged robots (`all`); downloads happen on first use.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import tempfile
import time
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
    shove_gaps: list[str] = field(default_factory=list)  # directions: a smaller shove threw it, a larger did not
    leg: float = 0.0  # trunk height over the feet / the length of the leg as drawn
    cpu_per_s: float = 0.0  # CPU seconds a second of simulation took
    seconds: float = 0.0
    note: str = ""


def _tilt(up: float) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, up))))


def _leg_straightness(m, d, trunk: int) -> float:
    """How straight the legs stand: the height of the trunk over the feet divided by the length of the leg as drawn
    (the offsets of its bodies, one below the other, from the foot up to the trunk). The largest over the feet that
    touch anything fixed in the world."""
    import mujoco

    feet = set()
    for c in d.contact[: d.ncon]:
        for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if m.geom_bodyid[a] != 0 and m.body_rootid[m.geom_bodyid[a]] == trunk and m.geom_bodyid[b] == 0:
                feet.add(int(m.geom_bodyid[a]))
    best = 0.0
    for foot in feet:
        length, body = 0.0, foot
        while body != trunk and body != 0:
            length += float(mujoco.mju_norm3(m.body_pos[body]))
            body = int(m.body_parentid[body])
        if body == trunk and length > 1e-6:
            best = max(best, (float(d.xpos[trunk][2]) - float(d.xpos[foot][2])) / length)
    return best


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
        falls0 = 0  # falls the controller counted before each trial (whole-body balance counts, the others do not)

        def standing(height: float) -> bool:
            high_enough = float(d.xpos[trunk][2]) > HEIGHT_KEPT * height
            held = getattr(ctrl, "falls", 0) == falls0  # a balance that gave up and holds the servos is not standing
            return tilt() < UPRIGHT and high_enough and held and bool(np.isfinite(d.qpos).all())

        cpu0 = time.process_time()
        ran = pg.run_fast(lambda: False, SETTLE_S)
        out.cpu_per_s = round((time.process_time() - cpu0) / max(ran, 1e-6), 2)
        z0 = float(d.xpos[trunk][2])
        out.leg = round(_leg_straightness(m, d, trunk), 2)
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
                ctrl.falls = falls0

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
            best, failed = 0.0, False
            for speed in shoves:  # every speed, not up to the first failure: a push is a single event
                restore()
                world = np.array([math.cos(yaw) * fx - math.sin(yaw) * fy, math.sin(yaw) * fx + math.cos(yaw) * fy])
                d.qvel[root_dof : root_dof + 2] += speed * world
                pg.run_fast(lambda: False, WATCH_S)
                if standing(z0):
                    best = speed
                    if failed:
                        out.shove_gaps.append(name)
                        failed = False
                else:
                    failed = True
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
    print(
        "robot                  kind      controller             kg     stands  load   leg   shove f/b/l/r (m/s)"
        "   cpu s/s   s"
    )
    for r in results:
        shove = (
            "/".join(f"{r.shove.get(n, 0):.1f}{'*' if n in r.shove_gaps else ''}" for n, _ in DIRECTIONS)
            if r.shove
            else "-"
        )
        load = "-" if r.load_ok is None else ("yes" if r.load_ok else "no")
        print(
            f"{r.robot:<22} {r.kind:<9} {r.controller:<22} {r.mass:6.1f} {'yes' if r.stands else 'no':<7} {load:<5} "
            f"{r.leg:<5.2f} {shove:<22} {r.cpu_per_s:6.2f}   {r.seconds:5.1f}  {r.why or r.note}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    return 0 if not any(r.note.startswith("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
