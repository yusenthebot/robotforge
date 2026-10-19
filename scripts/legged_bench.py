"""How well does every legged robot in the library walk? Measure it, robot by robot and ground by ground.

    uv run python scripts/legged_bench.py [--robots go2,a1,...] [--cases flat,slope,...] [--trials 5] [--speed 0.6]
                                          [--jobs 4] [--json out.json]

Every run adds one robot to an empty world and drives it with `move`, the way an agent or the
keyboard does (no walker named: each robot walks on what its library entry makes it walk with).
Cases:
- flat, slope, stairs, gravel, grass: walk straight ahead at 60 % of the robot's top speed across a
  patch that starts 0.6 m in front of it (flat: 3 m of open floor). A success crosses the patch,
  still upright and over it rather than around it (a robot that turns off the side of a 1.2 m wide
  flight of stairs also gets past its far end).
- turn: told to turn on the spot at 1 rad/s (or the fastest it turns) for TURN_SECONDS; a success turns at least
  90 degrees (A or D held that long turns it round a corner) and stays within
  a leg length of where it stood.
- shove: walking, 0.5 m/s added to the trunk's sideways velocity at 2 s; a success is upright and
  walking on 5 s later.
- drag: walking, the viewer's drag spring pulls the trunk 0.4 m sideways over 2 s; as for shove.
- payload: the heaviest library arm the robot carries at full speed (library.Payload) mounted on its
  back, walking 3 m on flat ground.
For each it reports success, the average forward speed (m/s along the way it faced at the start),
whether it fell (trunk tilted past 45 degrees), the heading drift (degrees, end against start, for the
straight walks; the turn rate for turn) and how far it strayed sideways. Library robots are downloaded
on first use; set ROBOTFORGE_MENAGERIE to a checkout to work offline. With --trials N each case runs N times from
starts a little back or ahead, aside and turned (the first from the usual start) and the table counts successes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

ROBOTS = ("go2", "go1", "a1", "anymal_c", "spot", "barkour", "g1", "t1", "op3")
# (Unitree H1 is not in the library: docs/LEGGED.md says why)
GAP = 0.6  # m of flat floor between the robot and the patch
FLAT_RUN = 3.0  # m a robot must cover on flat ground
SPEED_SHARE = 0.6
FELL_TILT = 45.0  # degrees
# Ground patches, each across the robot's way. Stairs go up three steps and down three; gravel is fine, sharp
# rubble a few centimetres high; grass a soft, slightly uneven lawn.
PATCHES: dict[str, dict] = {
    "slope": {"type": "ramp", "angle": 12, "length": 1.5, "landing": 0.8},
    "stairs": {"type": "stairs", "steps": 3, "step_height": 0.08, "step_depth": 0.3, "landing": 0.8},
    "gravel": {"type": "rough", "size": [3.0, 2.0], "height": 0.035, "scale": 0.06, "seed": 3, "style": "rock"},
    "grass": {"type": "rough", "size": [3.0, 2.0], "height": 0.02, "scale": 0.3, "seed": 5, "style": "grass"},
}


def patch_of(case: str) -> dict | None:
    """The ground patch of a case. "stairs@8x30x3" is a flight of 3 steps 8 cm high and 30 cm deep (stairs at other
    sizes, to find what a robot can and cannot climb)."""
    name, _, size = case.partition("@")
    patch = PATCHES.get(name)
    if patch is None or not size:
        return patch
    height, depth, steps = (float(v) for v in size.split("x"))
    return {**patch, "steps": int(steps), "step_height": height / 100, "step_depth": depth / 100}


CASES = ("flat", *PATCHES, "turn", "shove", "drag", "payload")
# The heaviest arm each robot carries at full speed (library.Payload), by the arms' masses.
PAYLOADS = {
    "go2": "z1",
    "go1": "z1",
    "a1": "z1",
    "anymal_c": "xarm7",
    "spot": "xarm7",
    "barkour": "piper",
    "g1": "piper",
    "t1": "so_arm100",
}
SHOVE = 0.5  # m/s
DRAG, DRAG_SECONDS, DRAG_STEPS = 0.4, 2.0, 40
WATCH = 5.0  # s after a shove or a drag
TURN_SECONDS = 6.0  # a humanoid takes a second to get its first step going


@dataclass
class Result:
    robot: str
    case: str
    success: bool
    speed: float  # m/s forward, along the start heading
    fell: bool
    drift: float  # degrees of heading change (turn: degrees per second turned)
    off: float  # m sideways from the start line at the end (turn: from where it stood)
    seconds: float
    walker: str = ""
    note: str = ""
    trial: int = 0


def _start(trial: int) -> tuple[list[float], float]:
    """Where a trial starts: trial 0 at the origin facing +x, later ones up to 15 cm back or 5 cm ahead, 10 cm to
    either side and 5 degrees off, seeded by the trial's number, so each trial meets the ground differently."""
    if trial == 0:
        return [0.0, 0.0], 0.0
    import numpy as np

    rng = np.random.default_rng(trial)
    return [float(rng.uniform(-0.15, 0.05)), float(rng.uniform(-0.1, 0.1))], float(rng.uniform(-5.0, 5.0))


def _setup(robot: str, case: str, trial: int = 0):
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_legged_")
    pg = Playground.with_demo()
    pg.new_world(f"legged_{robot}_{case}")
    mounts = [{"library": PAYLOADS[robot]}] if case == "payload" and robot in PAYLOADS else None
    pos, yaw = _start(trial)
    pg.add_robot(robot, library=robot, mounts=mounts, pos=pos, yaw=yaw)
    patch = patch_of(case)
    if patch is not None:
        rest = {k: v for k, v in patch.items() if k not in ("type", "style")}
        from robotforge.design import terrain

        length, _ = terrain.parse({k: v for k, v in patch.items() if k != "style"}).footprint()
        pg.add_terrain(type=patch["type"], pos=[GAP + length / 2, 0.0], style=patch.get("style"), params=rest)
    return pg


def _trunk(pg, robot: str) -> tuple[int, int]:
    import mujoco

    m = pg.session.model
    for j in range(m.njnt):
        body = int(m.jnt_bodyid[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(body).name.startswith(f"{robot}/"):
            return body, int(m.jnt_dofadr[j])
    raise ValueError(f"{robot} has no free trunk")


def _walker(pg, robot: str) -> str:
    """What walks it: the gait, the balance controller, a policy (by its name)."""
    ctrl = pg.session.controllers.get(robot)
    spec = getattr(ctrl, "spec", None)
    return spec.name if spec is not None else type(ctrl).__name__


def run(robot: str, case: str, trial: int = 0, share: float = SPEED_SHARE, walker: str | None = None) -> Result:
    """One measured run; the scratch home it made is removed again."""
    try:
        return _measure(robot, case, trial, share, walker)
    finally:
        home = os.environ.get("ROBOTFORGE_HOME", "")
        if Path(home).name.startswith("rf_legged_"):
            shutil.rmtree(home, ignore_errors=True)


def _measure(robot: str, case: str, trial: int, share: float, walker: str | None) -> Result:
    import numpy as np

    if case == "payload" and robot not in PAYLOADS:
        return Result(robot, case, False, 0.0, False, 0.0, 0.0, 0.0, note="skipped: carries no library arm")
    pg = _setup(robot, case, trial)
    behaviors, session = pg.behaviors, pg.session
    data = session.data
    trunk, dof = _trunk(pg, robot)
    info = next(r for r in behaviors.robots() if r["name"] == robot)
    top = float(info.get("max_speed", 0.3))
    start = data.xpos[trunk].copy()
    rot = data.xmat[trunk].reshape(3, 3)
    yaw0 = math.atan2(rot[1, 0], rot[0, 0])
    ahead = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
    left = np.array([-math.sin(yaw0), math.cos(yaw0), 0.0])
    tilt_max = 0.0
    turned = 0.0
    last_yaw = yaw0

    def watch(seconds: float, until=None) -> bool:
        """Step `seconds`; True when it fell. Follows the heading (turned) as it goes."""
        nonlocal tilt_max, turned, last_yaw
        steps = max(1, round(0.02 / pg.session.model.opt.timestep))
        t0 = data.time
        while data.time - t0 < seconds:
            session.step(steps)
            zz = float(data.xmat[trunk][8])
            tilt_max = max(tilt_max, math.degrees(math.acos(max(-1.0, min(1.0, zz)))))
            r = data.xmat[trunk]
            yaw = math.atan2(r[3], r[0])
            turned += (yaw - last_yaw + math.pi) % (2 * math.pi) - math.pi
            last_yaw = yaw
            if tilt_max > FELL_TILT:
                return True
            if until is not None and until():
                return False
        return False

    def forward() -> float:
        return float((data.xpos[trunk] - start) @ ahead)

    def sideways() -> float:
        return float((data.xpos[trunk] - start) @ left)

    t_start = float(data.time)
    if case == "turn":
        behaviors.move(robot, 0.0, 1.0, walker=walker)
        walker = _walker(pg, robot)
        fell = watch(TURN_SECONDS)
        seconds = float(data.time) - t_start
        rate = math.degrees(turned) / seconds
        strayed = float(np.linalg.norm((data.xpos[trunk] - start)[:2]))
        leg = 0.5 * float(start[2])
        ok = not fell and math.degrees(turned) >= 90.0 and strayed < max(leg, 0.15)
        return Result(robot, case, ok, 0.0, fell, round(rate, 1), round(strayed, 2), round(seconds, 1), walker)

    speed = share * top
    behaviors.move(robot, speed, 0.0, walker=walker)
    walker = _walker(pg, robot)
    patch = patch_of(case)
    end, half_width, length = FLAT_RUN, math.inf, 0.0
    if patch is not None:
        from robotforge.design import terrain

        length, width = terrain.parse({k: v for k, v in patch.items() if k != "style"}).footprint()
        end, half_width = GAP + length + 0.2, width / 2
    note = ""
    if case in ("shove", "drag"):
        fell = watch(2.0)
        if not fell and case == "shove":
            data.qvel[dof : dof + 3] += SHOVE * left
        elif not fell:
            grab = data.xpos[trunk].copy()
            session.start_drag(trunk, grab)
            for k in range(1, DRAG_STEPS + 1):
                session.move_drag(grab + left * DRAG * k / DRAG_STEPS + ahead * speed * DRAG_SECONDS * k / DRAG_STEPS)
                if watch(DRAG_SECONDS / DRAG_STEPS):
                    fell = True
                    break
            session.end_drag()
        if not fell:
            fell = watch(WATCH - 1.0)
        before = data.xpos[trunk].copy()
        if not fell:
            fell = watch(1.0)
        walking = float((data.xpos[trunk] - before) @ ahead) > 0.4 * speed
        ok = not fell and walking
        seconds = float(data.time) - t_start
        note = "" if walking or fell else "stopped walking"
    else:
        # A patch lies across the world's x axis wherever the trial started (see _start); flat ground is measured
        # from the start.
        x0 = 0.0 if patch is None else float(start[0])

        def along() -> float:
            return x0 + forward() if patch is None else float(data.xpos[trunk][0])

        limit = min(60.0, 2.5 * (end - x0 + 0.5) / max(speed, 0.02) + 3.0)
        off_patch = 0.0
        fell = False
        t0 = float(data.time)
        while data.time - t0 < limit and not fell:
            fell = watch(0.1)
            if GAP <= along() <= GAP + length:
                off_patch = max(off_patch, abs(float(data.xpos[trunk][1])))
            if along() > end + 0.3:
                break
        seconds = float(data.time) - t_start
        around = off_patch > half_width
        ok = not fell and along() > end and not around
        if around and along() > end:
            note = "went around the patch"
        elif not ok and not fell:
            note = f"stalled at x {along():.2f} of {end:.2f} m"
    drift = (math.degrees(turned) + 180.0) % 360.0 - 180.0
    return Result(
        robot,
        case,
        ok,
        round(forward() / max(seconds, 1e-6), 3),
        fell,
        round(drift, 1),
        round(sideways(), 2),
        round(seconds, 1),
        walker,
        note,
    )


def _run(args: tuple) -> Result:
    try:
        return dataclasses.replace(run(*args), trial=args[2])
    except Exception as exc:  # report the failure in the table rather than lose the whole batch
        note = f"error: {type(exc).__name__}: {exc}"
        return Result(args[0], args[1], False, 0.0, False, 0.0, 0.0, 0.0, note=note, trial=args[2])


def skipped(r: Result) -> bool:
    """A case the robot cannot be given (no library arm to carry), not one it failed."""
    return r.note.startswith("skipped")


def summary(results: list[Result]) -> str:
    """One line per robot and case over its trials: how many succeeded, mean speed, falls, mean |drift|."""
    groups: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        groups.setdefault((r.robot, r.case), []).append(r)
    lines = [f"{'robot':<9} {'case':<8} success  speed m/s  falls  |drift| deg  walker"]
    for (robot, case), rs in groups.items():
        if all(skipped(r) for r in rs):
            lines.append(f"{robot:<9} {case:<8}   -        -          -        -     {rs[0].note}")
            continue
        ok = sum(r.success for r in rs)
        speed = sum(r.speed for r in rs) / len(rs)
        falls = sum(r.fell for r in rs)
        drift = sum(abs(r.drift) for r in rs) / len(rs)
        unit = "deg/s" if case == "turn" else ""
        lines.append(
            f"{robot:<9} {case:<8} {ok:>2}/{len(rs):<2}    {speed:5.2f}     {falls:>3}    {drift:6.1f}{unit:<5}  "
            f"{rs[0].walker}"
        )
    ran = [r for r in results if not skipped(r)]
    total = sum(r.success for r in ran)
    lines.append(f"{total} of {len(ran)} runs passed ({len(results) - len(ran)} skipped)")
    return "\n".join(lines)


def table(results: list[Result]) -> str:
    lines = [f"{'robot':<9} {'case':<8} ok   speed  fell  drift   off m  sim s  {'walker':<18}  note"]
    for r in results:
        ok = "yes" if r.success else "no"
        fell = "yes" if r.fell else "-"
        lines.append(
            f"{r.robot:<9} {r.case:<8} {ok:<4} {r.speed:5.2f}  {fell:<4} {r.drift:6.1f}  {r.off:6.2f}  "
            f"{r.seconds:5.1f}  {r.walker:<18}  {r.note}"
        )
    ran = [r for r in results if not skipped(r)]
    passed = sum(r.success for r in ran)
    lines.append(f"{passed} of {len(ran)} passed ({len(results) - len(ran)} skipped)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--trials", type=int, default=1, help="runs per robot and case, from different starts")
    parser.add_argument("--speed", type=float, default=SPEED_SHARE, help="share of each robot's top speed")
    parser.add_argument("--walker", default=None, help="walk with this walker (move's walker), not the default")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    jobs = [
        (r, c, t, args.speed, args.walker)
        for r in args.robots.split(",")
        for c in args.cases.split(",")
        for t in range(args.trials)
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_run, jobs))
    print(table(results) if args.trials == 1 else summary(results))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=1)
    return 0 if not any(r.note.startswith("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
