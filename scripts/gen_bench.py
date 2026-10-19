"""Do robots generated from a sentence stand and move? Measure it (docs/BUILDER.md's tables come from here).

    uv run python scripts/gen_bench.py fifteen [--variations 3] [--jobs 4]
    uv run python scripts/gen_bench.py changes [--variations 2] [--jobs 4]
    uv run python scripts/gen_bench.py sweep --body quadruped [--length 0.6,1,1.4] [--width 1] [--height 1]
        [--size 1] [--arms 0] [--set segments=8,12 --set material=pla,aluminum] [--seeds 5] [--jobs 4]
    uv run python scripts/gen_bench.py say "a robot dog" ["make it shorter" ...] [--variations 3]
    uv run python scripts/gen_bench.py arms [--jobs 4]

Each run adds the robot to an empty world the way add_robot(name, describe=...) does (sweep: the build a plan makes,
added as a build), lets it stand for STAND seconds, then moves it forward for MOVE seconds at the speed its kind
walks (SPEED) and reports how far it got from where it stood, as a share of the commanded distance, and the
largest tilt of its trunk while moving. A run passes when it stays upright and gets at least half the commanded
distance. Arms only stand. --variations N runs each sentence with its own seed and the next N-1 seeds.

fifteen: the milestone's fifteen sentences (tests/test_generator.py FIFTEEN). changes: a first sentence and one
change ("a robot dog", then "make it shorter"). sweep: a grid of plans for one body (quadruped, hexapod, spider,
rover, balancer, snake, arm, humanoid). say: one first sentence and any changes, in order.

arms: every armed sentence in ARMED stands for a second, then its first hand picks a 4 cm cube put on the floor
REACH[0] m beyond its tool point (away from the robot's middle; the next distance in REACH when that one is refused)
and puts it 0.15 m to the side with pick_and_place (a robot that goes may walk or drive to reach; a magnet's cube is
steel): placed or the refusal.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STAND = 2.0  # s
MOVE = 5.0  # s
SPEED = {"snake": 0.1, "humanoid": 0.15}  # m/s; every other kind 0.25
PASS = 0.5  # of the commanded distance

FIFTEEN = (
    "a six-legged robot with two arms",
    "a tall wheeled courier with a lidar",
    "a snake with a gripper head",
    "a small orange robot dog",
    "a big carbon fiber quadruped with long legs and a camera head",
    "a spider robot in black and red",
    "a six-wheeled mars rover with an arm",
    "a self-balancing robot on two wheels",
    "a robot arm with six joints and a suction cup",
    "a long green snake with 12 segments",
    "a centaur in white and gold",
    "a tall humanoid",
    "a wheeled robot with two arms",
    "一只带激光雷达的蓝色四足机器狗",
    "a tentacle arm with a gripper",
)
CHANGES = (
    ("a robot dog", "make it shorter"),
    ("a robot dog", "make the legs shorter"),
    ("a robot dog", "make the legs longer"),
    ("a robot dog", "make it longer"),
    ("a robot dog", "make it narrower"),
    ("a short-legged robot dog", None),
    ("a very short robot dog", None),
    ("a narrow short dog", None),
    ("a compact robot dog", None),
    ("a long-legged insect", None),
    ("a robot dog", "six legs instead of four"),
    ("a six-legged robot", "make it shorter"),
    ("a wheeled robot", "make it shorter"),
    ("a small robot dog 50 cm long", None),
)
ARMED = (
    "a six-legged robot with two arms",
    "a snake with a gripper head",
    "a six-wheeled mars rover with an arm",
    "a robot arm with six joints and a suction cup",
    "a centaur in white and gold",
    "a wheeled robot with two arms",
    "a tentacle arm with a gripper",
    "a robot dog with an arm",
    "a snake with a suction head",
    "a wheeled robot with two arms and suction cups",
    "a tentacle with a magnet",
)
REACH = (0.06, 0.0, -0.06, -0.12)  # m beyond the tool point, tried in turn
BODIES = {
    "quadruped": {"body": "legged", "legs": 4},
    "hexapod": {"body": "legged", "legs": 6},
    "spider": {"body": "legged", "legs": 6, "round": True},
    "rover": {"body": "wheeled"},
    "balancer": {"body": "balancer"},
    "snake": {"body": "snake"},
    "arm": {"body": "arm"},
    "humanoid": {"body": "humanoid"},
}


@dataclass(frozen=True)
class Run:
    label: str
    sentences: tuple[str, ...] = ()
    seed: int | None = None
    plan: tuple[tuple[str, Any], ...] = ()  # a plan to build directly (sweep), as items


def measure(run: Run) -> dict[str, Any]:
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="gen_bench_")  # each run its own empty world
    import numpy as np

    from robotforge.design import generator
    from robotforge.playground import Playground, PlaygroundError
    from robotforge.sim.session import SimSession

    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("bench")
    notes: list[str] = []
    try:
        if run.plan:
            made = generator.realize(generator.Plan(**dict(run.plan)))
            notes = list(made.notes)
            src = made.source
            pg.add_robot("bot", template=src.get("template"), params=src.get("params"), build=src.get("build"))
        else:
            params = {"seed": run.seed} if run.seed is not None else None
            result = pg.add_robot("bot", describe=run.sentences[0], params=params)
            notes += result.get("generated", {}).get("notes", [])
            for sentence in run.sentences[1:]:
                result = pg.update_robot("bot", describe=sentence)
                notes += result.get("generated", {}).get("notes", [])
    except (PlaygroundError, ValueError) as exc:
        return {"label": run.label, "error": str(exc)[:200]}
    kind = pg.behaviors.kind("bot").value
    pg.run_fast(lambda: False, STAND)
    stood = pg.behaviors.pose("bot")
    out: dict[str, Any] = {"label": run.label, "kind": kind, "stands": stood["upright"], "notes": notes}
    if kind == "arm":
        out["passed"] = stood["upright"]
        return out
    speed = SPEED.get(kind, 0.25)
    worst = [0.0]
    root = pg.behaviors._root_body("bot")

    def tilt() -> bool:
        z = float(pg.session.data.xmat[root].reshape(3, 3)[2, 2])
        worst[0] = max(worst[0], math.degrees(math.acos(max(-1.0, min(1.0, z)))))
        return False

    pg.behaviors.move("bot", speed, 0.0)
    pg.run_fast(tilt, MOVE)
    end = pg.behaviors.pose("bot")
    start, stop = np.array(stood["position"][:2]), np.array(end["position"][:2])
    heading = math.radians(stood["heading_deg"])
    forward = float(np.dot(stop - start, [math.cos(heading), math.sin(heading)]))
    commanded = speed * MOVE
    out |= {
        "speed": speed,
        "forward": round(forward, 3),
        "share": round(forward / commanded, 2),
        "tilt": round(worst[0]),
        "upright": end["upright"],
        "passed": end["upright"] and forward >= PASS * commanded,
    }
    return out


def reach(sentence: str) -> dict[str, Any]:
    """pick_and_place from where the robot stands (see the module's arms)."""
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="gen_bench_")
    import mujoco
    import numpy as np

    from robotforge.design import generator
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    refused = ""
    for k, beyond in enumerate(REACH):
        pg = Playground(session=SimSession.from_xml("<mujoco/>"))
        pg.new_world(f"arms{k}")
        pg.add_robot("bot", describe=sentence)
        pg.run_fast(lambda: False, 1.0)
        hand = generator.generate(sentence).hands("bot")[0]
        m, d = pg.session.model, pg.session.data
        tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{hand}/tcp")][:2].copy()
        middle = np.array(pg.behaviors.pose("bot")["position"][:2])
        out = tcp - middle
        out = out / np.linalg.norm(out) if np.linalg.norm(out) > 0.03 else np.array([1.0, 0.0])
        cube = tcp + beyond * out
        target = tcp + 0.15 * np.array([-out[1], out[0]])
        steel = {"material": "steel"} if generator.generate(sentence).plan.hand == "magnet" else {}
        pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[float(v) for v in cube], color="red", **steel)
        pg.run_fast(lambda: False, 0.5)
        try:
            arm = pg.behaviors.resolve(hand, arm=True)
            motion, info = pg.behaviors.pick_and_place(arm, "cube", [float(v) for v in target])
        except ValueError as exc:
            refused = str(exc)
            continue
        pg.run_fast(lambda m=motion: m.done, 90.0)
        pg.run_fast(lambda: False, 0.5)
        placed = pg.behaviors.placed("cube", info["region"])["success"]
        reach = float(np.linalg.norm(cube - middle))
        went = "went" in info or "driven_m" in info
        return {"label": sentence, "hand": hand, "beyond": beyond, "from_middle": round(reach, 2), "placed": placed,
                "went": went}  # fmt: skip
    return {"label": sentence, "refused": refused[:160]}


def _seeds(sentence: str, variations: int) -> list[int]:
    from robotforge.design.generator import seed_of

    first = seed_of(sentence)
    return [first + k for k in range(variations)]


def _runs(args: argparse.Namespace) -> list[Run]:
    if args.mode == "fifteen":
        return [Run(s, (s,), seed) for s in FIFTEEN for seed in _seeds(s, args.variations)]
    if args.mode == "changes":
        return [
            Run(f"{a} -> {b}" if b else a, (a, b) if b else (a,), seed)
            for a, b in CHANGES
            for seed in _seeds(a, args.variations)
        ]
    if args.mode == "say":
        return [Run(" -> ".join(args.sentences), tuple(args.sentences), seed)
                for seed in _seeds(args.sentences[0], args.variations)]  # fmt: skip
    base = BODIES[args.body]
    more = [_setting(text) for text in args.set]
    grid = itertools.product(
        *(_floats(getattr(args, k)) for k in ("length", "width", "height", "size")),
        (int(a) for a in args.arms.split(",")),
        *(values for _, values in more),
        range(args.seeds),
    )
    runs = []
    for length, width, height, size, arms, *extra, seed in grid:
        plan = {**base, "length": length, "width": width, "height": height, "size": size, "arms": arms, "seed": seed}
        plan |= {name: value for (name, _), value in zip(more, extra, strict=True)}
        said = "".join(f" {name}={value}" for (name, _), value in zip(more, extra, strict=True))
        label = f"{args.body} L{length} W{width} H{height} S{size} A{arms}{said} seed {seed}"
        runs.append(Run(label, plan=tuple(plan.items())))
    return runs


def _setting(text: str) -> tuple[str, list[Any]]:
    """ "segments=8,12" -> ("segments", [8, 12]): any other plan field to sweep."""
    name, _, values = text.partition("=")

    def value(v: str) -> Any:
        return int(v) if v.isdigit() else float(v) if v.replace(".", "", 1).isdigit() else v

    return name, [value(v) for v in values.split(",")]


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def _row(r: dict[str, Any]) -> str:
    if "error" in r:
        return f"| {r['label']} | error: {r['error']} | | | |"
    if r["kind"] == "arm":
        return f"| {r['label']} | arm | stands {'yes' if r['stands'] else 'NO'} | | |"
    mark = "" if r["passed"] else " MISS" if r["upright"] else " FELL"
    note = "; ".join(r["notes"])[:90]
    return f"| {r['label']} | {r['kind']} | {r['forward']:.2f} m ({r['share']:.0%}){mark} | {r['tilt']} | {note} |"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("fifteen", "changes", "sweep", "say", "arms"))
    parser.add_argument("sentences", nargs="*")
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument("--body", choices=sorted(BODIES), default="quadruped")
    for name in ("length", "width", "height", "size"):
        parser.add_argument(f"--{name}", default="1.0")
    parser.add_argument("--arms", default="0")
    parser.add_argument("--set", action="append", default=[], help="another plan field to sweep, e.g. segments=8,12")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if args.mode == "say" and not args.sentences:
        parser.error("say needs a sentence")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if args.mode == "arms":
        with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, 4))) as pool:
            done = list(pool.map(reach, ARMED))
        print("| Sentence | Hand | Cube | Placed |")
        print("|---|---|---|---|")
        for r in done:
            if "refused" in r:
                print(f"| {r['label']} | | | refused: {r['refused']} |")
            else:
                where = f"{r['beyond']:+.2f} m past the hand, {r['from_middle']:.2f} m from the middle"
                where += ", went over to it" if r["went"] else ""
                print(f"| {r['label']} | {r['hand']} | {where} | {'yes' if r['placed'] else 'NO'} |")
        return 0 if all(r.get("placed") for r in done) else 1
    runs = _runs(args)
    with ProcessPoolExecutor(max_workers=max(1, min(args.jobs, 4))) as pool:
        results = list(pool.map(measure, runs))
    print("| Run | Kind | Forward in 5 s | Max tilt | Notes |")
    print("|---|---|---|---|---|")
    for r in results:
        print(_row(r))
    moved = [r for r in results if "error" not in r]
    passed = sum(bool(r.get("passed")) for r in moved)
    print(f"\n{passed} of {len(results)} passed (upright and at least {PASS:.0%} of the commanded distance)")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
