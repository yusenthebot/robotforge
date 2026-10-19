"""Make a suite of generated worlds with a task each, for agent evaluations and reinforcement learning.

    uv run python scripts/task_suite.py --count 20 --seed 0 --out suite
    uv run python scripts/task_suite.py --count 12 --hard --out hard_suite   # mugs on shelves, stacks, pours, ...
    uv run python scripts/agent_eval.py --suite suite          # an agent does each one; the task's check judges it

Each case is suite/worlds/<case>/world.yaml (a world generated from a sentence, its robot and its task) and a line in
suite/suite.json with the world's sentence, the task's sentence and the prompt a user would type. Open one in
Python as a learning environment with robotforge.control.tasks.world_task.world_env("suite/worlds/case_01").
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("suite"))
    parser.add_argument("--zh", type=float, default=0.3, help="share of cases asked in Chinese")
    parser.add_argument("--hard", action="store_true", help="the hard tasks on a bench (upright, stack, insert, ...)")
    args = parser.parse_args(argv)
    from robotforge.design.tasks.suite import make_hard_suite, make_suite

    start = time.monotonic()
    cases = (
        make_hard_suite(args.count, args.seed, args.out)
        if args.hard
        else make_suite(args.count, args.seed, args.out, args.zh)
    )
    for c in cases:
        print(f"{c.name}  {c.kind:<9} {c.robot:<7} {c.lang}  {c.task}")
    print(f"{len(cases)} cases in {time.monotonic() - start:.0f} s: {args.out / 'suite.json'}")
    return 0 if len(cases) == args.count else 1


if __name__ == "__main__":
    sys.exit(main())
