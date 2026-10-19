"""Success rates of the hard tasks with real Menagerie arms, over seeds, with the cause of every failure.

    uv run python scripts/objects_tasks.py [--kinds upright,stack,insert,sort] [--arms panda,ur5e] [--seeds 8] [--out FILE]

Each trial builds a bench (design/tasks/hard_bench.py: the arm on a table, the fixtures and things of the task, placed
from the seed), adds the task with add_task, has the scripted expert make the pick_and_place calls and reads the
task's own success check. Prints a table of successes per task and arm, and the failures by cause. Maintainer tool:
it needs the real Menagerie models (downloaded once).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kinds", default="upright,stack,insert,sort")
    parser.add_argument("--arms", default="panda,ur5e")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from robotforge.design.tasks import hard_bench

    rows = []
    for kind in args.kinds.split(","):
        for arm in args.arms.split(","):
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                try:
                    row = hard_bench.run(kind, seed, arm)
                except Exception as exc:  # a trial that blows up is a failure with its own cause
                    row = {"kind": kind, "seed": seed, "arm": arm, "done": False, "why": "", "steps": [],
                           "cause": f"crashed: {type(exc).__name__}: {exc}"[:200]}  # fmt: skip
                rows.append(row)
                print(f"{kind:8s} {arm:6s} seed {seed}: {'ok' if row['done'] else 'FAILED':6s} {row.get('cause') or row['why']}", flush=True)
    print()
    print(f"{'task':10s} {'arm':6s} placed")
    for kind in args.kinds.split(","):
        for arm in args.arms.split(","):
            mine = [r for r in rows if r["kind"] == kind and r["arm"] == arm]
            ok = sum(r["done"] for r in mine)
            print(f"{kind:10s} {arm:6s} {ok}/{len(mine)}")
    print()
    causes = Counter(_group(r["cause"]) for r in rows if not r["done"])
    for cause, n in causes.most_common():
        print(f"  {n:3d}  {cause}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


def _group(cause: str) -> str:
    """A cause without the names of the things in it, so the same failure counts together."""
    import re

    return re.sub(r"\b(?:tin|brick|book|mug|bottle|rack|apple|orange|can)(?:_\d+)?\b", "<thing>", cause)[:110]


if __name__ == "__main__":
    sys.exit(main())
