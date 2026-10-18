"""Do learned rhythm walks go straight, and switch quickly between walking and turning? Measure it.

    uv run python scripts/rhythm_bench.py forward [--robots dog,star,humanoid] [--seeds 1,2,3] [--jobs 4]
        [--rounds 30] [--energy 0.1] [--reflexes] [--shove] [--heading W]
    uv run python scripts/rhythm_bench.py switch [--robots dog,star] [--seeds 1,2,3] [--jobs 4] [--stress 4]
        [--idle 3] [--long 20] [--min-share 0.6] [--keep DIR | --walks DIR]
    uv run python scripts/rhythm_bench.py leads --walks DIR [--robots dog,star] [--seeds 1,2,3] [--paths move,gap]
        [--jobs 4] [--json]
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
  teleop messages, no stop in between);
- long: forward for --long seconds (20) from standing, the heading reported every 4 s (leg "forward Ns").
For every leg it reports the distance, the heading change and whether it stayed up (after the idle time too); for
the 1 s turn also the share of what its learned turn rate promises in that second. Each path ends with a stop
and 2 s of standing, reported as the leg "stop". It ends with a summary of the 1 s turns straight out of the walk
(move and keys paths) and of the long walks, and exits with 1 if a turn got less than --min-share (0.6) of its
rate. With --stress N it then gives N runs of 60
random commands (forward, left, right or stop, each for 0.3 to 1.5 s, the next at once like a key change) and
counts the falls (trunk past 60 degrees at the end of a command), with the switch each one came on. Those runs
carry on from wherever the last command left the robot, so one fall changes everything after it; --trials N
instead tries every switch out of walking or turning (F>L, L>F, R>S, ...) N times on its own, from standing:
the first command for 2 to 2.5 s, the second for 1.5 s, and counts the falls after the switch (and any before),
and the degrees turned after it: in the first second (a turn's share of its rate) and in all 1.5 s.
--keep DIR saves each trained walk as DIR/<robot>_<seed>.json; --walks DIR plays those instead of training.
switch's move, idle and keys paths turn left once, after one lead: one sequence says little about falls, which come
at some points of the stride and not others.

leads: plays walks kept with switch --keep through each path, both turn directions, after each of 25 leads of
walking (2.0 to 4.48 s, every update of a 2 Hz stride once): move and idle as above (a 1 s turn, then 2 to 2.48 s of
walking, every update of a stride once too, then a stop and 2 s of standing); quick, a key change by hand: W let go,
A or D pressed one update later for a second, let go, W pressed one update later; and gap: the same with 0.04 to 0.3
s between the keys. (Holding W while A is down plays exactly as move: a turn wins over W, and move's stop is skipped
as the next call comes at once.) Each sequence starts from standing in a reset world. It reports the falls (trunk
past 60 degrees at any update) per robot, path and side with the call each came in (s<seed>c<call>; call 0 is the
walk before the turn), the 1 s turn's share of the learned rate (per walk the median over its leads, and over all
tries how many reached --min-share), and per robot how many distinct sequences fell after the walk before the turn.
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
    # switch -> degrees turned in each try's first TURN_SECONDS after it, and in all its 1.5 s (tries that stayed up)
    turned: dict[str, list[list[float]]] = field(default_factory=dict)


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
        # What it learned around and how it switches, as the lesson keeps it (code from before kept neither).
        kept = {"home": dict(zip(joints, map(float, trainer.home), strict=True)), "switching": "fade"}
        walk = SavedWalk(
            robot=robot,
            joints=joints,
            **{k: v for k, v in kept.items() if k in SavedWalk.model_fields},
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
    if opts["long"]:
        pg = world("long")
        result = Switch(robot, seed, "long", round(walk.distance, 3), round(math.degrees(rate), 1))
        result.legs = _long(pg, robot, opts["long"])
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
                        continue
                    head = _leg(pg, robot, then, *COMMANDS[then], TURN_SECONDS, stop=False)
                    rest = _leg(pg, robot, then, *COMMANDS[then], 1.5 - TURN_SECONDS, stop=False) if head.up else head
                    if not rest.up:
                        count[0] += 1
                    else:
                        out[0].turned.setdefault(f"{first}>{then}", []).append(
                            [head.heading, round(head.heading + rest.heading, 1)]
                        )
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


def _long(pg, robot: str, seconds: float) -> list[Leg]:
    """Forward for `seconds` from standing in one command: a leg every 4 s, each with the distance and heading
    change from the start."""
    legs = []
    start = pg.behaviors.pose(robot)
    pg.behaviors.move(robot, 0.2, 0.0, "rhythm")
    heading = {"last": start["heading_deg"], "turned": 0.0}

    def track() -> bool:
        now = pg.behaviors.pose(robot)["heading_deg"]
        heading["turned"] += (now - heading["last"] + 180) % 360 - 180
        heading["last"] = now
        return False

    for k in range(math.ceil(seconds / 4.0)):
        pg.run_fast(track, min(4.0, seconds - 4.0 * k))
        track()
        now = pg.behaviors.pose(robot)
        up = float(pg.session.data.xmat[pg.behaviors._root_body(robot)][8]) >= FELL_COS
        moved = math.dist(now["position"][:2], start["position"][:2])
        legs.append(
            Leg(
                f"forward {min(4.0 * (k + 1), seconds):g}s",
                round(moved, 3),
                round(heading["turned"], 1),
                now["upright"],
                up=up,
            )
        )
        if not up:
            break
    pg.behaviors.stop(robot)
    return legs


# leads: when in the stride a turn is asked for decides whether it tips the robot over, so each sequence is tried
# after 25 leads of walking, 2.0 to 4.48 s: every update of a 2 Hz stride (0.02 s apart) once, over five strides.
LEADS = tuple(round(2.0 + 0.1 * k + 0.02 * (k // 5), 2) for k in range(25))
GAPS = (0.04, 0.1, 0.16, 0.22, 0.3)  # s between letting go of W and pressing A or D (path gap), cycled over leads
# The quickest key change by hand that is not two keys down at once: one update. run_fast steps whole updates of
# 0.02 s until this much has passed (0.02 itself sometimes came to two, the sum a hair short).
QUICK = 0.01  # s
PATHS = ("move", "idle", "quick", "gap")
# Seconds of walking after the turn, one per lead, over a stride in another order than the leads: a walk started after
# a turn starts its stride afresh, so a fixed 2 s stopped it at one point of its stride every time. Waiting 0.06 s
# before stopping instead of 0.1 moved that point, and one A1 walk's stops after the turn went from 3 falls in 50 to 11.
WALK_ON = tuple(round(2.0 + 0.02 * ((7 * k) % 25), 2) for k in range(25))


def _sequence(
    path: str, turn: float, lead: float, idle: float, gap: float, on: float = 2.0
) -> list[tuple[str, float, float, float]]:
    """The calls of one sequence: (what, speed, turn, then seconds of sim time). "move" is a move call (a key change),
    "stop" a stop; the turn command's seconds are the 1 s turn measured."""
    if path == "move":  # the MCP move tool with seconds, the next call at once
        return [
            ("move", 0.2, 0.0, lead),
            ("stop", 0, 0, 0.0),
            ("move", 0.0, turn, TURN_SECONDS),
            ("stop", 0, 0, 0.0),
            ("move", 0.2, 0.0, on),
            ("stop", 0, 0, STAND_SECONDS),
        ]
    if path == "idle":  # the same with the viewer open: the sim runs on while the agent thinks
        return [
            ("move", 0.2, 0.0, lead),
            ("stop", 0, 0, idle),
            ("move", 0.0, turn, TURN_SECONDS),
            ("stop", 0, 0, idle),
            ("move", 0.2, 0.0, on),
            ("stop", 0, 0, STAND_SECONDS),
        ]
    if path == "quick":  # W let go and A or D pressed one update later for a second, then W again as quickly
        gap = QUICK
    # gap: W let go, A or D pressed a moment later for a second, then W again
    return [
        ("move", 0.2, 0.0, lead),
        ("move", 0.0, 0.0, gap),
        ("move", 0.0, turn, TURN_SECONDS),
        ("move", 0.0, 0.0, gap),
        ("move", 0.2, 0.0, on),
        ("move", 0.0, 0.0, STAND_SECONDS),
    ]


def run_leads(robot: str, seed: int, opts: dict[str, Any]) -> list[dict]:
    """Every path, both turn directions, every lead, on a saved walk: whether it fell (trunk past 60 degrees at any
    update) and in which call, and the degrees of the 1 s turn."""
    from robotforge.behaviors.rhythm_walk import SavedWalk, save_walk
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_rhythm_")
    walk = SavedWalk.model_validate_json(Path(opts["walks"], f"{robot}_{seed}.json").read_text(encoding="utf-8"))
    pg = Playground.with_demo()
    pg.new_world(f"leads_{robot}")
    pg.add_robot(robot, **_entry(robot, opts))
    save_walk(pg.store.dir, walk)
    rate = math.degrees(walk.turn_rate or 0.0)
    out = []
    for path in opts["paths"]:
        for side, turn in (("L", 1.0), ("R", -1.0)):
            for k, lead in enumerate(LEADS):
                pg.session.reset()
                calls = _sequence(path, turn, lead, opts["idle"], GAPS[k % len(GAPS)], WALK_ON[k])
                fell, turned = _play(pg, robot, calls, next(i for i, c in enumerate(calls) if c[2]))
                share = round(turn * turned / rate, 2) if turned is not None and rate > 0 else None
                out.append(
                    {
                        "robot": robot,
                        "seed": seed,
                        "path": path,
                        "side": side,
                        "lead": lead,
                        "fell": fell,
                        "turned": turned,
                        "share": share,
                    }
                )
    return out


def _play(
    pg, robot: str, calls: list[tuple[str, float, float, float]], measure: int | None
) -> tuple[int | None, float | None]:
    """Make the calls one after the other: the call during or after which the robot fell (None: it stayed up), and
    the degrees it turned in the first TURN_SECONDS of call `measure` (None if it fell before, or none is measured)."""
    root = pg.behaviors._root_body(robot)
    state = {"fell": False, "turned": 0.0, "last": 0.0}

    def heading() -> float:
        rot = pg.session.data.xmat[root]
        return math.degrees(math.atan2(float(rot[3]), float(rot[0])))

    def watch() -> bool:
        now = heading()
        state["turned"] += (now - state["last"] + 180) % 360 - 180
        state["last"] = now
        state["fell"] = state["fell"] or float(pg.session.data.xmat[root][8]) < FELL_COS
        return state["fell"]

    turned = None
    for i, (what, speed, angle, seconds) in enumerate(calls):
        if what == "stop":
            pg.behaviors.stop(robot)
        else:
            pg.behaviors.move(robot, speed, angle, "rhythm")
        state["last"], state["turned"] = heading(), 0.0
        first = min(seconds, TURN_SECONDS) if i == measure else seconds
        if first:
            pg.run_fast(watch, first)
        watch()
        if i == measure:
            turned = round(state["turned"], 1)
            if seconds > first and not state["fell"]:
                pg.run_fast(watch, seconds - first)
        if state["fell"]:
            return i, turned
    return None, turned


def _leads_job(args: tuple[str, int, dict[str, Any]]) -> list[dict]:
    return run_leads(*args)


# oos: sequences drawn at random, none on a grid the switch checks or leads play, each played by the walk as kept, by
# the same walk switched as 7a4aa92 switched every kept walk but holding its heading (steered), and as 7a4aa92 plays
# the file (base: switched so, its heading not held, around the world's home pose).
OOS_PATHS = ("move", "idle", "held", "keys", "chain")
SWITCHINGS = ("kept", "steered", "base")
OOS_LEAD = (0.2, 6.0)  # s of walking before the turn
OOS_TURN = (0.3, 2.5)  # s of turning
OOS_ON = (0.3, 4.0)  # s of walking after it
OOS_IDLE = (0.05, 3.0)  # s the sim runs on after a move call ends, the viewer open while the agent thinks
OOS_GAP = (0.0, 0.4)  # s between letting go of one key and pressing the next (0: the next update)
OOS_CHAIN = (6, 10)  # key changes in a chain
OOS_HOLD = (0.2, 3.0)  # s each key of a chain is held


def as_base(walk):
    """The walk switched as 7a4aa92 switches every kept walk: any switch while moving holds home until the robot stands
    still (SETTLE_MAX at most) and starts the next rhythm as learned, and a stop holds home at once."""
    sides = ("left", "right")
    cleared = {f"turn_{s}_walking{part}": None for s in sides for part in ("", "_at")}
    return walk.model_copy(update={**cleared, "switching": None, **{f"turn_{s}_exit": None for s in sides}})


def _oos_cases(path: str, count: int, rng) -> list[tuple[list[tuple[str, float, float, float]], int | None]]:
    """`count` sequences of calls for `path` (as _sequence gives them), turning left and right in turn, and the index of
    the turn whose first second is measured (None: none is)."""

    def u(span: tuple[float, float]) -> float:
        return round(float(rng.uniform(*span)), 3)

    out = []
    for k in range(count):
        side = 1.0 if k % 2 == 0 else -1.0
        speed, turn = u((0.05, 0.5)), side * u((0.2, 2.0))
        lead, turning, on = u(OOS_LEAD), u(OOS_TURN), u(OOS_ON)
        if path in ("move", "idle"):  # the move tool with seconds: each call ends with a stop
            idle = (u(OOS_IDLE), u(OOS_IDLE)) if path == "idle" else (0.0, 0.0)
            calls = [
                ("move", speed, 0.0, lead),
                ("stop", 0.0, 0.0, idle[0]),
                ("move", 0.0, turn, turning),
                ("stop", 0.0, 0.0, idle[1]),
                ("move", speed, 0.0, on),
                ("stop", 0.0, 0.0, STAND_SECONDS),
            ]
            out.append((calls, 2))
        elif path == "held":  # walking with seconds=null (or W held), then the turn (A or D pressed too), then walking
            calls = [("move", speed, 0.0, lead), ("move", speed, turn, turning), ("move", speed, 0.0, on)]
            out.append(([*calls, ("move", 0.0, 0.0, STAND_SECONDS)], 1))
        elif path == "keys":  # W let go, A or D pressed a moment later, let go, W again a moment later
            gaps = [max(QUICK, u(OOS_GAP)) for _ in range(2)]
            calls = [
                ("move", speed, 0.0, lead),
                ("move", 0.0, 0.0, gaps[0]),
                ("move", 0.0, side, turning),
                ("move", 0.0, 0.0, gaps[1]),
                ("move", speed, 0.0, on),
                ("move", 0.0, 0.0, STAND_SECONDS),
            ]
            out.append((calls, 2))
        else:  # chain: random key changes, each held a while
            calls, last = [], "S"
            for _ in range(int(rng.integers(OOS_CHAIN[0], OOS_CHAIN[1] + 1))):
                key = str(rng.choice([c for c in COMMANDS if c != last]))
                calls.append(("move", *COMMANDS[key], u(OOS_HOLD)))
                last = key
            out.append(([*calls, ("move", 0.0, 0.0, STAND_SECONDS)], None))
    return out


def run_oos(robot: str, seed: int, opts: dict[str, Any]) -> list[dict]:
    """Every out-of-sample sequence on a saved walk, played as kept and switched as 7a4aa92 switched: whether it fell
    (trunk past 60 degrees at any update) and in which call, and the degrees of the measured turn's first second."""
    import numpy as np

    from robotforge.behaviors.rhythm_walk import SavedWalk, save_walk
    from robotforge.playground import Playground

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="rf_rhythm_")
    kept = SavedWalk.model_validate_json(Path(opts["walks"], f"{robot}_{seed}.json").read_text(encoding="utf-8"))
    plain = {"steer": None, "home": None}
    walks = {"kept": kept, "steered": as_base(kept), "base": as_base(kept).model_copy(update=plain)}
    pg = Playground.with_demo()
    pg.new_world(f"oos_{robot}")
    pg.add_robot(robot, **_entry(robot, opts))
    out = []
    for p, path in enumerate(opts["paths"]):
        rng = np.random.default_rng([opts["bench_seed"], seed, p, sum(map(ord, robot))])
        for k, (calls, measure) in enumerate(_oos_cases(path, opts["cases"], rng)):
            for switching in opts["switchings"]:
                save_walk(pg.store.dir, walks[switching])
                pg.session.reset()
                fell, turned = _play(pg, robot, calls, measure)
                out.append(
                    {
                        "robot": robot,
                        "seed": seed,
                        "path": path,
                        "case": k,
                        "switching": switching,
                        "fell": fell,
                        "calls": len(calls),
                        "turned": turned,
                        "lead": calls[0][3],
                        "side": None if measure is None else ("L" if calls[measure][2] > 0 else "R"),
                        "measured": None if measure is None else min(TURN_SECONDS, calls[measure][3]),
                    }
                )
    return out


def _oos_job(args: tuple[str, int, dict[str, Any]]) -> list[dict]:
    return run_oos(*args)


def _print_oos(results: list[dict], rates: dict[tuple[str, int], float]) -> None:
    """Per robot, path and switching: the sequences that fell after the walk they start with (and during it), which
    walks they were, and the measured turns' first second as a share of the rate from standing (median, over turns
    asked after WALKED_MIN or more of walking)."""
    from robotforge.behaviors.rhythm_walk import WALKED_MIN

    for robot in dict.fromkeys(r["robot"] for r in results):
        mine = [r for r in results if r["robot"] == robot]
        totals = {}
        for path in dict.fromkeys(r["path"] for r in mine):
            for switching in dict.fromkeys(r["switching"] for r in mine):
                runs = [r for r in mine if r["path"] == path and r["switching"] == switching]
                after = [r for r in runs if r["fell"] not in (None, 0)]
                during = sum(r["fell"] == 0 for r in runs)
                where: dict[str, int] = {}
                for r in after:
                    where[f"s{r['seed']}"] = where.get(f"s{r['seed']}", 0) + 1
                shares = sorted(
                    r["turned"] * (1 if r["side"] == "L" else -1) / (rates[(robot, r["seed"])] * r["measured"])
                    for r in runs
                    if r["turned"] is not None and r["lead"] >= WALKED_MIN and rates[(robot, r["seed"])] > 0
                )
                share = f"; turn after the walk {shares[len(shares) // 2]:.0%} of the rate" if shares else ""
                print(
                    f"{robot:9} {path:5} {switching:4}: {len(after):3} fell in {len(runs)} after the walk"
                    + (f" ({', '.join(f'{k} {n}' for k, n in sorted(where.items()))})" if where else "")
                    + (f", {during} during it" if during else "")
                    + share
                )
                totals[switching] = totals.get(switching, 0) + len(after)
        print(f"{robot:9} all: " + ", ".join(f"{s} {n} falls" for s, n in totals.items()))


def _print_leads(results: list[dict], min_share: float) -> None:
    """Falls per robot, path and side (and in which call), and the 1 s turn's share of the rate: per walk the median
    over leads, and over all tries how many reached `min_share`."""
    for robot in dict.fromkeys(r["robot"] for r in results):
        mine = [r for r in results if r["robot"] == robot]
        seeds = sorted({r["seed"] for r in mine})
        for path in dict.fromkeys(r["path"] for r in mine):
            for side in "LR":
                runs = [r for r in mine if r["path"] == path and r["side"] == side]
                falls = [r for r in runs if r["fell"] is not None]
                where = {}
                for r in falls:
                    where[f"s{r['seed']}c{r['fell']}"] = where.get(f"s{r['seed']}c{r['fell']}", 0) + 1
                shares = [r["share"] for r in runs if r["share"] is not None and r["fell"] is None]
                medians = []
                for seed in seeds:
                    own = sorted(r["share"] for r in runs if r["seed"] == seed and r["share"] is not None)
                    medians.append(f"{own[len(own) // 2]:.0%}" if own else "-")
                reached = sum(s >= min_share for s in shares)
                print(
                    f"{robot:9} {path:5} {side}: {len(falls):3} falls in {len(runs)}"
                    + (f" ({', '.join(f'{k} {n}' for k, n in sorted(where.items()))})" if where else "")
                    + f"; 1 s turn: median per walk {' '.join(medians)}; >= {min_share:.0%} in {reached} of "
                    f"{len(shares)}"
                )
        switch = sum(r["fell"] not in (None, 0) for r in mine)
        before = sum(r["fell"] == 0 for r in mine)
        print(f"{robot:9} all: {switch} distinct sequences fell after the walk ({before} during it) of {len(mine)}")


def _forward_job(args: tuple[str, int, dict[str, Any]]) -> dict:
    return asdict(run_forward(*args))


def _switch_job(args: tuple[str, int, dict[str, Any]]) -> list[dict]:
    return [asdict(r) for r in run_switch(*args)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("forward", "switch", "leads", "oos"))
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
    parser.add_argument("--long", type=float, default=20.0, help="switch: s of the long forward walk (0: none)")
    parser.add_argument("--min-share", type=float, default=0.6, help="switch: least share of the turn rate in 1 s")
    parser.add_argument("--trials", type=int, default=0, help="switch: tries of each switch on its own")
    parser.add_argument("--keep", help="switch: save the trained walks here")
    parser.add_argument("--walks", help="switch: play the walks saved here instead of training")
    parser.add_argument(
        "--paths", help=f"leads: which of {', '.join(PATHS)} (all); oos: which of {', '.join(OOS_PATHS)} (all)"
    )
    parser.add_argument("--cases", type=int, default=40, help="oos: sequences per path and walk")
    parser.add_argument("--bench-seed", type=int, default=0, help="oos: draws other sequences")
    parser.add_argument("--switchings", default=",".join(SWITCHINGS), help="oos: which of " + ", ".join(SWITCHINGS))
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
        "long": args.long,
        "trials": args.trials,
        "keep": args.keep,
        "walks": args.walks,
        "paths": [p for p in (args.paths or ",".join(OOS_PATHS if args.what == "oos" else PATHS)).split(",") if p],
        "cases": args.cases,
        "bench_seed": args.bench_seed,
        "switchings": [s for s in args.switchings.split(",") if s],
    }
    names = [r for r in args.robots.split(",") if r]
    robots = names + [n for n in imports if n not in names]
    jobs = [(r, int(s), opts) for r in robots for s in args.seeds.split(",")]
    if args.what in ("leads", "oos") and not args.walks:
        parser.error(f"{args.what} plays saved walks: give --walks DIR (from switch --keep DIR)")
    if args.what == "oos":
        unknown = set(opts["paths"]) - set(OOS_PATHS) or set(opts["switchings"]) - set(SWITCHINGS)
        if unknown:
            parser.error(f"oos does not know {', '.join(sorted(unknown))}")
        missing = [f"{r}_{s}" for r, s, _ in jobs if not Path(args.walks, f"{r}_{s}.json").is_file()]
        if missing:  # robots kept with different seeds share a folder
            print(f"no walk kept for {', '.join(missing)}: left out", file=sys.stderr)
        jobs = [(r, s, o) for r, s, o in jobs if f"{r}_{s}" not in missing]
    worker = {"forward": _forward_job, "switch": _switch_job, "leads": _leads_job, "oos": _oos_job}[args.what]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(worker, jobs))
    if args.what == "oos":
        from robotforge.behaviors.rhythm_walk import SavedWalk

        flat = [r for batch in results for r in batch]
        if args.json:
            for r in flat:
                print(json.dumps(r))
        rates = {}
        for r, s, _ in jobs:
            walk = SavedWalk.model_validate_json(Path(args.walks, f"{r}_{s}.json").read_text(encoding="utf-8"))
            rates[(r, s)] = math.degrees(walk.turn_rate or 0.0)
        _print_oos(flat, rates)
        return 0
    if args.what == "leads":
        flat = [r for batch in results for r in batch]
        if args.json:
            for r in flat:
                print(json.dumps(r))
        _print_leads(flat, args.min_share)
        return 0
    if args.what == "forward":
        _print_forward(results, args.json)
        return 0
    flat = [r for batch in results for r in batch]
    _print_switch(flat, args.json)
    return 0 if _summary(flat, args.min_share, sys.stderr if args.json else sys.stdout) else 1


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


def _summary(results: list[dict], min_share: float, out) -> bool:
    """Per robot: the 1 s turns straight out of the walk (move and keys paths; degrees and share of the rate), the
    long walks' heading at their end; whether every turn got at least `min_share` of its rate."""
    ok = True
    for robot in dict.fromkeys(r["robot"] for r in results):
        mine = [r for r in results if r["robot"] == robot]
        for path in ("move", "keys"):
            turns = [(r["seed"], r["legs"][1]) for r in mine if r["path"] == path and len(r["legs"]) > 2]
            shares = [leg["share"] for _, leg in turns if leg["share"] is not None]
            if not shares:
                continue
            low = [f"seed {seed} {leg['share']:.0%}" for seed, leg in turns if (leg["share"] or 0.0) < min_share]
            ok = ok and not low
            print(
                f"{robot} {path}: 1 s turn out of the walk "
                + ", ".join(f"{leg['heading']:+.0f} deg ({leg['share']:.0%})" for _, leg in turns)
                + f"; mean {sum(shares) / len(shares):.0%}"
                + (f"; under {min_share:.0%}: {', '.join(low)}" if low else ""),
                file=out,
            )
        turned: dict[str, list[list[float]]] = {}
        for r in mine:
            for switch, tries in r.get("turned", {}).items():
                turned.setdefault(switch, []).extend(tries)
        rates = {r["seed"]: r["turn_rate"] for r in mine}
        for switch in ("F>L", "F>R"):
            sign = 1.0 if switch.endswith("L") else -1.0
            shares = [
                sign * t[0] / (rates[r["seed"]] * TURN_SECONDS)
                for r in mine
                for t in r.get("turned", {}).get(switch, [])
                if rates[r["seed"]] > 0
            ]
            if shares:
                low = sum(v < min_share for v in shares)
                print(
                    f"{robot} trials {switch}: 1 s turn out of the walk, mean {sum(shares) / len(shares):.0%} of the "
                    f"rate; under {min_share:.0%} in {low} of {len(shares)} tries",
                    file=out,
                )
        for switch in ("L>F", "R>F"):
            ahead = [abs(t[1]) for t in turned.get(switch, [])]
            if ahead:
                print(
                    f"{robot} trials {switch}: heading change in 1.5 s of walking after the turn, mean "
                    f"{sum(ahead) / len(ahead):.0f} deg, over 30 in {sum(a > 30 for a in ahead)} of {len(ahead)}",
                    file=out,
                )
        long = [r for r in mine if r["path"] == "long"]
        if long:
            ends = [r["legs"][-1] for r in long]
            print(
                f"{robot} long: "
                + ", ".join(
                    f"{leg['name']} {leg['heading']:+.0f} deg {leg['distance']:.1f} m" + ("" if leg["up"] else " FELL")
                    for leg in ends
                )
                + f"; mean |heading| {sum(abs(leg['heading']) for leg in ends) / len(ends):.0f} deg",
                file=out,
            )
    return ok


if __name__ == "__main__":
    sys.exit(main())
