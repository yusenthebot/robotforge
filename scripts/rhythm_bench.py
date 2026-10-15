"""Do learned rhythm walks go straight, and switch quickly between walking and turning? Measure it.

    uv run python scripts/rhythm_bench.py forward [--robots dog,star,humanoid] [--seeds 1,2,3] [--jobs 4]
        [--rounds 30] [--energy 0.1] [--reflexes] [--shove] [--heading W]
    uv run python scripts/rhythm_bench.py switch [--robots dog,star] [--seeds 1,2,3] [--jobs 4] [--stress 4]
        [--idle 3] [--keep DIR | --walks DIR]
    add --import a1=path/to/a1.urdf (or any URDF or MJCF file) to bench a robot from a file too

forward: each run teaches one robot to walk forward the way the RL II lesson's Learn button does
(control/cpg.py: random search from standing still, 4 s tries at 2 Hz, the seed as the lesson's n-th
training, with the lesson's energy cost, reflexes and shoves when asked) and then replays the learned rhythm
once for 4 s: how far it got, how much its heading changed (degrees, positive is to the left), how far it
drifted sideways, the energy it spent, whether it fell and, trained with shoves, how many of 12 shoves it
stayed up for. --heading overrides the reward's weight on the heading (for tuning it; shoved trainings
do not count the heading, as in the lesson).

switch: each run teaches a walk, its heading hold and the turns the way "Keep this walk for my robot" does
(learn_steering, learn_turns), keeps them in a world and drives the robot through it three times:
- move: like the MCP move tool, forward 4 s, turn left 1 s, forward 2 s, each call ending with a stop, the next
  call at once (no viewer open: the sim waits for the next call);
- idle: the same with --idle seconds (3) of sim time after each call, as when the viewer is open and the sim keeps
  running while the agent thinks;
- keys: like holding W, pressing A for a second while W is still held, then letting go of A (the viewer's
  teleop messages, no stop in between).
For every leg it reports the distance, the heading change and whether it stayed up (after the idle time too); for
the 1 s turn also the share of what its learned turn rate promises in that second. Each path ends with a stop
and 2 s of standing, reported as the leg "stop". With --stress N it then gives N runs of 60
random commands (forward, left, right or stop, each for 0.3 to 1.5 s, the next at once like a key change) and
counts the falls (trunk past 60 degrees at the end of a command), with the switch each one came on. Those runs
carry on from wherever the last command left the robot, so one fall changes everything after it; --trials N
instead tries every switch out of walking or turning (F>L, L>F, R>S, ...) N times on its own, from standing:
the first command for 2 to 2.5 s, the second for 1.5 s, and counts the falls after the switch (and any before).
--keep DIR saves each trained walk as DIR/<robot>_<seed>.json; --walks DIR plays those instead of training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

ROBOTS: dict[str, dict] = {"dog": {"template": "quadruped"}, "humanoid": {"template": "humanoid"}}
SECONDS = 4.0  # s per try, as in the lesson (walk_rl.ATTEMPT)
FELL_COS = 0.5  # control/cpg.py FELL: cosine of the trunk's tilt below which the robot is down
FREQ = 2.0  # Hz, the lesson's default
TURN_SECONDS = 1.0  # the short turn command between two walks
STAND_SECONDS = 2.0  # after the last command's stop
STRESS_COMMANDS = 60  # per stress run
COMMANDS = {"F": (0.2, 0.0), "L": (0.0, 1.0), "R": (0.0, -1.0), "S": (0.0, 0.0)}


@dataclass
class Forward:
    robot: str
    seed: int
    distance: float  # m along +x in 4 s
    heading: float  # degrees turned in those 4 s, positive to the left
    drift: float  # m sideways
    energy: float  # J
    fell: bool
    survived: int | None  # of 12 shoves, when trained with shoves
    seconds: float  # s of training


@dataclass
class Leg:
    name: str
    distance: float  # m the trunk moved over the ground
    heading: float  # degrees, positive to the left
    upright: bool  # leaning less than 45 degrees at the end
    share: float | None = None  # of the learned turn rate's promise, for the turn
    up: bool = True  # not fallen: leaning less than 60 degrees at the end (the walker's own FELL)


@dataclass
class Switch:
    robot: str
    seed: int
    path: str  # move or keys
    learned: float  # m in 4 s in training
    turn_rate: float  # deg/s, the slower side
    legs: list[Leg] = field(default_factory=list)
    switches: int = 0  # in the stress runs
    falls: dict[str, int] = field(default_factory=dict)  # switch ("F>L": walking, then left) -> falls on it
    trials: dict[str, list[int]] = field(default_factory=dict)  # switch -> [falls after it, falls before it, tries]


def _entry(robot: str, opts: dict[str, Any]) -> dict:
    if robot in opts["imports"]:
        return {"file": opts["imports"][robot]}
    if robot == "star":
        from robotforge.learn.walk_rl import STAR

        return {"build": STAR}
    return ROBOTS[robot]


def _solo(robot: str, entry: dict):
    from robotforge.design.world import World, compile_world

    return compile_world(World.model_validate({"name": "solo", "robots": [{"name": robot, **entry}]})).model


def _learn(robot: str, entry: dict, seed: int, opts: dict[str, Any]):
    from robotforge.control import cpg
    from robotforge.learn import walk_rl

    if hasattr(walk_rl, "lesson_reward"):  # as the lesson's Learn button trains
        reward = walk_rl.lesson_reward(opts["energy"], opts["shove"])
    else:  # code from before the heading counted, for before-and-after numbers
        reward = cpg.Reward(energy=opts["energy"], survive=opts["shove"])
    if opts["heading"] is not None and getattr(reward, "heading", 0.0):
        reward = replace(reward, heading=opts["heading"])
    model = _solo(robot, entry)
    trainer = cpg.Trainer(model, robot, reward, SECONDS, FREQ)
    rhythm = cpg.Rhythm.still(trainer.body.joints, FREQ, reflexes=opts["reflexes"])
    pushes = walk_rl.SHOVE if opts["shove"] else 0.0
    log = trainer.train(rhythm, iterations=opts["rounds"], seed=seed, pushes=pushes)
    return model, trainer, rhythm, log


def run_forward(robot: str, seed: int, opts: dict[str, Any]) -> Forward:
    from robotforge.learn import walk_rl

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_rhythm_")
    _, trainer, rhythm, log = _learn(robot, _entry(robot, opts), seed, opts)
    walk = trainer.rollout(rhythm)
    shoved = trainer.push_test(rhythm, walk_rl.SHOVE, walk_rl.PUSH_TRIALS) if opts["shove"] else None
    return Forward(
        robot=robot,
        seed=seed,
        distance=round(walk["distance"], 3),
        heading=round(math.degrees(walk["turned"]), 1),
        drift=round(walk["drift"], 3),
        energy=round(walk["energy"], 1),
        fell=bool(walk["fell"]),
        survived=int(shoved["survived"]) if shoved else None,
        seconds=round(log.seconds, 1),
    )


def run_switch(robot: str, seed: int, opts: dict[str, Any]) -> list[Switch]:
    import numpy as np

    from robotforge.behaviors import rhythm_walk
    from robotforge.behaviors.rhythm_walk import SavedWalk, joint_names, learn_turns, save_walk
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_rhythm_")
    entry = _entry(robot, opts)
    if opts["walks"]:
        walk = SavedWalk.model_validate_json(Path(opts["walks"], f"{robot}_{seed}.json").read_text(encoding="utf-8"))
    else:
        model, trainer, rhythm, _ = _learn(robot, entry, seed, opts)
        learned = trainer.rollout(rhythm)["distance"]
        joints = joint_names(model, trainer.body)
        # What it learned around, as the lesson keeps it (code from before kept no home).
        home = {"home": dict(zip(joints, map(float, trainer.home), strict=True))}
        walk = SavedWalk(
            robot=robot,
            joints=joints,
            **(home if "home" in SavedWalk.model_fields else {}),
            freq=rhythm.freq,
            reflexes=rhythm.reflexes,
            theta=[float(v) for v in rhythm.theta],
            speed=max(0.0, learned) / SECONDS,
            distance=learned,
            seconds=SECONDS,
        )
        if hasattr(rhythm_walk, "learn_steering"):  # absent before the heading hold, for benching older code
            walk = rhythm_walk.learn_steering(model, robot, walk, seed=seed)
        walk = learn_turns(model, robot, walk, seed=seed)
        if opts["keep"]:
            Path(opts["keep"]).mkdir(parents=True, exist_ok=True)
            Path(opts["keep"], f"{robot}_{seed}.json").write_text(json.dumps(walk.model_dump()), encoding="utf-8")
    rate = walk.turn_rate or 0.0

    def world(name: str):
        pg = Playground.with_demo()
        pg.new_world(f"switch_{robot}_{name}")
        pg.add_robot(robot, **entry)
        save_walk(pg.store.dir, walk)
        return pg

    out = []
    for path in ("move", "idle", "keys"):
        pg = world(path)
        result = Switch(robot, seed, path, round(walk.distance, 3), round(math.degrees(rate), 1))
        legs = (("forward", 0.2, 0.0, SECONDS), ("left", 0.0, 1.0, TURN_SECONDS), ("forward again", 0.2, 0.0, 2.0))
        idle = opts["idle"] if path == "idle" else 0.0
        for name, speed, turn, seconds in legs:
            if path == "keys" and name == "left":
                speed = 0.2  # W is still held while A is down: the turn wins
            result.legs.append(_leg(pg, robot, name, speed, turn, seconds, stop=path != "keys", idle=idle))
            if not result.legs[-1].up:
                break
        result.legs.append(_leg(pg, robot, "stop", 0.0, 0.0, STAND_SECONDS, stop=True))
        if rate > 0 and len(result.legs) > 2:
            result.legs[1].share = round(math.radians(result.legs[1].heading) / (rate * TURN_SECONDS), 2)
        out.append(result)
    for run in range(opts["stress"]):
        rng = np.random.default_rng(1000 * seed + run)
        pg, last = world(f"stress{run}"), "S"
        for _ in range(STRESS_COMMANDS):
            key = str(rng.choice([k for k in COMMANDS if k != last]))
            switch, last = f"{last}>{key}", key
            leg = _leg(pg, robot, switch, *COMMANDS[key], float(rng.uniform(0.3, 1.5)), stop=False)
            out[0].switches += 1
            if not leg.up:
                out[0].falls[switch] = out[0].falls.get(switch, 0) + 1
                pg.session.reset()
                last = "S"
    if opts["trials"]:
        pg = world("trials")
        rng = np.random.default_rng(7000 + seed)
        for first in "FLR":
            for then in "FLRS":
                if then == first:
                    continue
                count = out[0].trials.setdefault(f"{first}>{then}", [0, 0, 0])
                for _ in range(opts["trials"]):
                    pg.session.reset()
                    count[2] += 1
                    lead = 2.0 + float(rng.uniform(0.0, 0.5))
                    if not _leg(pg, robot, first, *COMMANDS[first], lead, stop=False).up:
                        count[1] += 1
                    elif not _leg(pg, robot, then, *COMMANDS[then], 1.5, stop=False).up:
                        count[0] += 1
    return out


def _leg(pg, robot: str, name: str, speed: float, turn: float, seconds: float, stop: bool, idle: float = 0.0) -> Leg:
    """One command for `seconds`, the heading followed as it goes (the ends alone wrap at 180). Distance and
    heading are the command's; with `idle`, uprightness is judged after that much more sim time with no command."""
    from robotforge.behaviors.manager import BehaviorError

    before = pg.behaviors.pose(robot)
    try:
        pg.behaviors.move(robot, speed, turn, "rhythm")
    except BehaviorError:  # it is down and cannot get up
        return Leg(name, 0.0, 0.0, False, up=False)
    heading = {"last": before["heading_deg"], "turned": 0.0}

    def track() -> bool:
        now = pg.behaviors.pose(robot)["heading_deg"]
        heading["turned"] += (now - heading["last"] + 180) % 360 - 180
        heading["last"] = now
        return False

    pg.run_fast(track, seconds)
    if stop:
        pg.behaviors.stop(robot)
    after = pg.behaviors.pose(robot)
    moved = math.dist(after["position"][:2], before["position"][:2])
    turned = heading["turned"]
    if idle:
        pg.run_fast(lambda: False, idle)
        after = pg.behaviors.pose(robot)
    up = float(pg.session.data.xmat[pg.behaviors._root_body(robot)][8]) >= FELL_COS
    return Leg(name, round(moved, 3), round(turned, 1), after["upright"], up=up)


def _forward_job(args: tuple[str, int, dict[str, Any]]) -> dict:
    return asdict(run_forward(*args))


def _switch_job(args: tuple[str, int, dict[str, Any]]) -> list[dict]:
    return [asdict(r) for r in run_switch(*args)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("forward", "switch"))
    parser.add_argument("--robots", default="dog,star,humanoid")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--import", dest="imports", action="append", default=[], metavar="NAME=FILE")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--energy", type=float, default=0.0, help="the reward's energy cost w")
    parser.add_argument("--reflexes", action="store_true")
    parser.add_argument("--shove", action="store_true", help="shove every try, points for staying up")
    parser.add_argument("--heading", type=float, default=None, help="override the reward's heading weight")
    parser.add_argument("--stress", type=int, default=0, help="switch: random-command runs to count falls in")
    parser.add_argument("--idle", type=float, default=3.0, help="switch: s of sim time after each call (idle path)")
    parser.add_argument("--trials", type=int, default=0, help="switch: tries of each switch on its own")
    parser.add_argument("--keep", help="switch: save the trained walks here")
    parser.add_argument("--walks", help="switch: play the walks saved here instead of training")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print one JSON line per run")
    args = parser.parse_args(argv)
    imports = {}
    for item in args.imports:
        name, _, file = item.partition("=")
        if not name or not file:
            parser.error(f"--import takes NAME=FILE, got {item!r}")
        imports[name] = str(Path(file).expanduser().resolve())
    opts = {
        "imports": imports,
        "rounds": args.rounds,
        "energy": args.energy,
        "reflexes": args.reflexes,
        "shove": args.shove,
        "heading": args.heading,
        "stress": args.stress,
        "idle": args.idle,
        "trials": args.trials,
        "keep": args.keep,
        "walks": args.walks,
    }
    names = [r for r in args.robots.split(",") if r]
    robots = names + [n for n in imports if n not in names]
    jobs = [(r, int(s), opts) for r in robots for s in args.seeds.split(",")]
    worker = _forward_job if args.what == "forward" else _switch_job
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(worker, jobs))
    if args.what == "forward":
        _print_forward(results, args.json)
    else:
        _print_switch([r for batch in results for r in batch], args.json)
    return 0


def _print_forward(results: list[dict], as_json: bool) -> None:
    if as_json:
        for r in results:
            print(json.dumps(r))
        return
    print(
        f"{'robot':10} {'seed':>4} {'dist m':>7} {'head deg':>8} {'drift m':>7} {'J':>6} {'fell':>5} {'up':>3} {'s':>5}"
    )
    for r in results:
        up = "" if r["survived"] is None else r["survived"]
        print(
            f"{r['robot']:10} {r['seed']:>4} {r['distance']:>7.2f} {r['heading']:>8.1f} {r['drift']:>7.2f} "
            f"{r['energy']:>6.1f} {'yes' if r['fell'] else '':>5} {up:>3} {r['seconds']:>5.1f}"
        )
    for robot in dict.fromkeys(r["robot"] for r in results):
        mine = [r for r in results if r["robot"] == robot]
        dist = sum(r["distance"] for r in mine) / len(mine)
        head = sum(abs(r["heading"]) for r in mine) / len(mine)
        print(f"{robot:10} mean distance {dist:.2f} m, mean |heading| {head:.1f} deg over {len(mine)} seeds")


def _print_switch(results: list[dict], as_json: bool) -> None:
    if as_json:
        for r in results:
            print(json.dumps(r))
        return
    for r in results:
        legs = "; ".join(
            f"{leg['name']} {leg['distance']:.2f} m {leg['heading']:+.0f} deg{'' if leg['upright'] else ' FELL'}"
            + (f" ({leg['share']:.0%} of the rate)" if leg["share"] is not None else "")
            for leg in r["legs"]
        )
        print(
            f"{r['robot']} seed {r['seed']} {r['path']}: learned {r['learned']:.2f} m, {r['turn_rate']:.0f} deg/s; "
            f"{legs}"
        )
        if r["trials"]:
            where = ", ".join(f"{k} {n[0]}" + (f" (+{n[1]} before)" if n[1] else "") for k, n in r["trials"].items())
            falls, tries = (sum(n[i] for n in r["trials"].values()) for i in (0, 2))
            print(f"  trials: {falls} falls in {tries} switches: {where}")
        if r["switches"]:
            where = ", ".join(f"{k} {n}" for k, n in sorted(r["falls"].items()))
            print(
                f"  stress: {sum(r['falls'].values())} falls in {r['switches']} switches{': ' + where if where else ''}"
            )


if __name__ == "__main__":
    sys.exit(main())
