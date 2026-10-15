"""Do our quadrupeds take a push? Measure it.

    uv run python scripts/push_bench.py [--robots dog,go2] [--states stand,trot] [--pushes ...] [--jobs 4]

Each run puts one robot on flat ground under its built-in gait (`move(robot, ..., walker="gait")`),
standing still or trotting at 0.3 m/s, lets it settle for two seconds and then pushes it:
- shove0.5: 0.5 m/s added to the trunk's sideways velocity in one go (front0.5: forwards);
- drag0.4: the viewer's drag spring pulls the trunk origin 0.4 m to the left over 2 s, then lets go
  (dragtop0.4: grabbed 4 cm higher, on its back, where a mouse usually picks it; dragrear0.4 and
  dragfront0.4: on its back, 12 cm behind or ahead of the middle).
It watches the next five seconds and reports whether the robot fell (tilted past 45 degrees) or
flipped (upside down), its largest tilt, where it ended up sideways from where it was pushed (for a
trotting robot that includes any change of heading, also reported), the lowest its trunk sank, and
whether it recovered: upright, at its usual height and, standing, still again; or trotting, still
walking forward. The library Go2 needs ROBOTFORGE_MENAGERIE (the test fixtures work offline) or a
download.
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

ROBOTS = {"dog": {"template": "quadruped"}, "go2": {"library": "go2"}}
STATES = {"stand": 0.0, "trot": 0.3}  # commanded forward speed, m/s
PUSHES = ("shove0.3", "shove0.5", "shove0.7", "front0.7", "drag0.4", "dragtop0.4")
SETTLE = 2.0  # s on the gait before the push
WATCH = 5.0  # s watched after the push (after letting go, for a drag)
DRAG_SECONDS, DRAG_STEPS = 2.0, 40  # the drag target moves in 40 even steps over 2 s, like a hand
# Where the hand grabs, in the trunk's frame (m): its origin, its back, the back near its tail or its head.
GRABS = {"top": [0.0, 0.0, 0.04], "rear": [-0.12, 0.0, 0.03], "front": [0.12, 0.0, 0.03], "": [0.0, 0.0, 0.0]}
FELL_TILT = 45.0  # degrees
RECOVERED_TILT = 10.0  # degrees


@dataclass
class Result:
    robot: str
    state: str
    push: str
    fell: bool
    flipped: bool
    max_tilt: float  # degrees
    displaced: float  # m sideways (trunk's left), from just before the push to the end
    sank: float  # lowest trunk height over its height before the push
    recovered: bool
    final_tilt: float
    forward: float  # m/s forward over the last second (trotting)
    turned: float  # degrees the heading changed from just before the push to the end
    note: str = ""


def run(robot: str, state: str, push: str) -> Result:
    import mujoco
    import numpy as np

    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_push_")
    pg = Playground.with_demo()
    pg.new_world(f"push_{robot}_{state}_{push}".replace(".", "_"))
    pg.add_robot(robot, **ROBOTS[robot])
    session, behaviors = pg.session, pg.behaviors
    model, data = session.model, session.data
    trunk = next(
        int(model.jnt_bodyid[j])
        for j in range(model.njnt)
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        and model.body(int(model.jnt_bodyid[j])).name.startswith(f"{robot}/")
    )
    dof = int(model.jnt_dofadr[model.body_jntadr[trunk]])
    speed = STATES[state]
    legged = behaviors.kind(robot).value == "quadruped"  # a hexapod has only the built-in gait
    behaviors.move(robot, speed, 0.0, walker="gait" if legged else None)
    pg.run_fast(lambda: False, SETTLE)

    rot = data.xmat[trunk].reshape(3, 3)
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    ahead = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    start = data.xpos[trunk].copy()
    height = float(start[2])
    worst_tilt, lowest, fell, flipped = 0.0, height, False, False

    def watch(seconds: float) -> None:
        nonlocal worst_tilt, lowest, fell, flipped
        for _ in range(round(seconds / model.opt.timestep)):  # every physics step, so the drag keeps its pace
            session.step()
            zz = float(data.xmat[trunk][8])
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, zz))))
            worst_tilt = max(worst_tilt, tilt)
            lowest = min(lowest, float(data.xpos[trunk][2]))
            fell = fell or tilt > FELL_TILT
            flipped = flipped or zz < 0.0

    if push.startswith("shove"):
        data.qvel[dof : dof + 3] += float(push.removeprefix("shove")) * left
    elif push.startswith("front"):
        data.qvel[dof : dof + 3] += float(push.removeprefix("front")) * ahead
    else:
        where = next(w for w in GRABS if push.startswith(f"drag{w}"))
        distance = float(push.removeprefix(f"drag{where}"))
        grab = data.xpos[trunk] + rot @ GRABS[where]
        session.start_drag(trunk, grab)
        for k in range(1, DRAG_STEPS + 1):
            session.move_drag(grab + left * distance * k / DRAG_STEPS + ahead * speed * DRAG_SECONDS * k / DRAG_STEPS)
            watch(DRAG_SECONDS / DRAG_STEPS)
        session.end_drag()
    watch(WATCH - 1.0)
    before = data.xpos[trunk].copy()
    watch(1.0)
    end = data.xpos[trunk]
    final_tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[trunk][8])))))
    rot = data.xmat[trunk].reshape(3, 3)
    turned = (math.degrees(math.atan2(rot[1, 0], rot[0, 0]) - yaw) + 180.0) % 360.0 - 180.0
    forward = float((end - before) @ ahead)
    still = float(np.linalg.norm(end - before)) < 0.05
    walking = forward > 0.5 * speed
    recovered = (
        not fell and final_tilt < RECOVERED_TILT and float(end[2]) > 0.85 * height and (walking if speed > 0 else still)
    )
    return Result(
        robot,
        state,
        push,
        fell,
        flipped,
        round(worst_tilt, 1),
        round(float((end - start) @ left), 3),
        round(lowest / height, 2),
        recovered,
        round(final_tilt, 1),
        round(forward, 3),
        round(turned, 1),
    )


def _run(args: tuple[str, str, str]) -> Result:
    try:
        return run(*args)
    except Exception as exc:  # report the failure in the table rather than lose the whole batch
        note = f"error: {type(exc).__name__}: {exc}"
        return Result(*args, False, False, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0, note)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--robots", default=",".join(ROBOTS))
    parser.add_argument("--states", default=",".join(STATES))
    parser.add_argument("--pushes", default=",".join(PUSHES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args(argv)
    jobs = [(r, s, p) for r in args.robots.split(",") for s in args.states.split(",") for p in args.pushes.split(",")]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(_run, jobs))
    print("robot    state  push        fell  flip  max tilt  displaced m  sank  recovered  final tilt  fwd m/s  turned")
    for r in results:
        print(
            f"{r.robot:<8} {r.state:<6} {r.push:<11} {'yes' if r.fell else '-':<5} {'yes' if r.flipped else '-':<5} "
            f"{r.max_tilt:7.1f}  {r.displaced:11.3f}  {r.sank:4.2f}  {'yes' if r.recovered else 'no':<9}  "
            f"{r.final_tilt:10.1f}  {r.forward:7.3f}  {r.turned:6.1f}  {r.note}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=1)
    return 0 if not any(r.note.startswith("error") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
