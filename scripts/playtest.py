"""Play every lesson from its first step to its last, the way a learner would, and say where it gets stuck.

    uv run python scripts/playtest.py [--lessons pd_control,gait] [--quick] [--json out.json]

For each lesson it opens a fresh playground, makes the moves in robotforge.learn.playthrough (answer the
guess, turn the knobs, press the buttons, let time pass), steps the physics as the viewer does and ticks
the lesson 20 times a simulated second. A step must pass within its time limit, or the lesson stops there.
The table lists the steps passed, the simulated and wall-clock seconds, and for a stuck step what was
tried and which check never held. `--quick` leaves out the lessons that train for minutes.

The real robots (PiPER, Go2, G1) come from ROBOTFORGE_MENAGERIE; when it is unset this uses the offline
stand-ins in tests/fixtures/menagerie, so it needs no network. Worlds are written to a temporary
ROBOTFORGE_HOME unless one is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "menagerie"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lessons", help="comma-separated lesson ids (default: all)")
    parser.add_argument("--quick", action="store_true", help="skip the lessons that train for minutes")
    parser.add_argument("--steps", action="store_true", help="also list every step with its times")
    parser.add_argument("--json", type=Path, help="also write the full reports here")
    args = parser.parse_args()

    if "ROBOTFORGE_MENAGERIE" not in os.environ and FIXTURES.is_dir():
        os.environ["ROBOTFORGE_MENAGERIE"] = str(FIXTURES)
    home = None
    if "ROBOTFORGE_HOME" not in os.environ:
        home = tempfile.TemporaryDirectory(prefix="robotforge-playtest-")
        os.environ["ROBOTFORGE_HOME"] = home.name

    from robotforge.learn import LESSONS
    from robotforge.learn.playthrough import RECIPES, play
    from robotforge.playground import Playground

    ids = args.lessons.split(",") if args.lessons else list(LESSONS)
    unknown = [i for i in ids if i not in LESSONS]
    if unknown:
        parser.error(f"unknown lessons: {', '.join(unknown)}; lessons: {', '.join(LESSONS)}")
    if args.quick:
        ids = [i for i in ids if not (i in RECIPES and RECIPES[i].slow)]

    print(f"menagerie: {os.environ['ROBOTFORGE_MENAGERIE']}", flush=True)
    reports = []
    for lesson_id in ids:
        report = play(Playground.with_demo(), lesson_id)
        reports.append(report)
        mark = "ok" if report.ok else "STUCK"
        print(f"  {lesson_id:<24} {report.passed}/{report.total}  {mark}", flush=True)
        if args.steps:
            for step in report.steps:
                state = "passed" if step.passed else "STUCK"
                print(f"      {step.id:<14} {state:<7} {step.sim:>7.1f} s sim {step.wall:>7.1f} s wall", flush=True)

    print()
    print(f"{'lesson':<24} {'steps':>7} {'sim s':>8} {'wall s':>8}  stuck at")
    for r in reports:
        stuck = r.stuck.id if r.stuck else ("(could not start)" if r.error else "")
        print(f"{r.lesson:<24} {f'{r.passed}/{r.total}':>7} {r.sim:>8.1f} {r.wall:>8.1f}  {stuck}")
    total_ok = sum(r.ok for r in reports)
    print(f"\n{total_ok} of {len(reports)} lessons played to the end.")
    for r in reports:
        if not r.ok:
            print()
            print(r.explain())

    if args.json:
        args.json.write_text(
            json.dumps([{**asdict(r), "ok": r.ok} for r in reports], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if home is not None:
        home.cleanup()
    return 0 if total_ok == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
