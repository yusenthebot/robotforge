"""Seed sweeps of ZenoBench's skill scenarios: how often each one succeeds, not whether one run did.

MuJoCo is deterministic, so one run per scenario shows a single outcome, and some outcomes hinge on a
degree (a tipped bottle, how far a rolling pin rolls). A sweep runs `tools/verify_skills.py` once per
seed; seed 0 is the scene exactly as authored, seed n > 0 nudges every free object by up to 1 mm and
0.5 degrees first (`ZENOBENCH_SEED`, see `world.World._jitter_start`). The summary puts each scenario's
pass rate next to Isaac Sim's recorded result (`skill_library/verification/STATUS.md`).
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STATUS = Path("skill_library/verification/STATUS.md")
SCENARIOS = Path("skill_library/verification/scenarios.json")


@dataclass(frozen=True)
class Outcome:
    success: bool
    steps_ok: int
    steps: int
    failure: str  # the first failing step's error, or "" on success


def scenario_ids(root: Path) -> list[str]:
    data = json.loads((root / SCENARIOS).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("scenarios", [])
    return [row["id"] for row in rows]


def run(root: Path, out: Path, seeds: int, jobs: int, ids: list[str]) -> None:
    """One verify_skills run per seed into out/seed<k>; a seed whose results are all there is skipped,
    so an interrupted sweep resumes where it stopped."""
    for seed in range(seeds):
        where = out / f"seed{seed}"
        if all((where / i / "result.json").exists() for i in ids):
            print(f"sweep: seed {seed} already done", flush=True)
            continue
        missing = [i for i in ids if not (where / i / "result.json").exists()]
        command = [sys.executable, "-m", "robotforge.zenobench.python", "tools/verify_skills.py"]
        command += ["--out", str(where), "--jobs", str(jobs)]
        for i in missing:
            command += ["--id", i]
        print(f"sweep: seed {seed}, {len(missing)} scenarios", flush=True)
        subprocess.run(command, cwd=root, env={**os.environ, "ZENOBENCH_SEED": str(seed)}, check=False)


def outcomes(out: Path) -> dict[str, list[Outcome]]:
    """Scenario -> one outcome per seed that ran it."""
    found: dict[str, list[Outcome]] = collections.defaultdict(list)
    for result in sorted(out.glob("seed*/*/result.json")):
        r = json.loads(result.read_text(encoding="utf-8"))
        steps = r.get("steps", [])
        ok = sum(1 for s in steps if s.get("success"))
        failed = next((s for s in steps if not s.get("success")), None)
        failure = "" if failed is None else str(failed.get("error") or failed.get("error_code") or "failed")
        found[r["id"]].append(Outcome(bool(r.get("success")), ok, len(steps), failure))
    return dict(found)


def isaac(root: Path) -> dict[str, str]:
    """Scenario -> Isaac Sim's recorded result ("pass 7/7")."""
    table, rows = False, {}
    for line in (root / STATUS).read_text(encoding="utf-8").splitlines():
        if line.startswith("## Scenarios"):
            table = True
        elif table and line.startswith("| ") and not line.startswith(("| Scenario", "|---")):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = f"{cells[1]} {cells[2]}"
    return rows


def summary(root: Path, out: Path) -> str:
    found, recorded = outcomes(out), isaac(root)
    lines = ["| Scenario | Isaac Sim | MuJoCo passes | Steps passed | Most common failure |", "|---|---|---|---|---|"]
    rates = []
    names = sorted(found)  # the scenarios this sweep ran
    for name in names:
        runs = found.get(name, [])
        wins = sum(o.success for o in runs)
        rates.append(wins / len(runs) if runs else 0.0)
        reach = sorted({f"{o.steps_ok}/{o.steps}" for o in runs})
        common = collections.Counter(o.failure[:90] for o in runs if not o.success).most_common(1)
        lines.append(
            f"| {name} | {recorded.get(name, '-')} | {wins}/{len(runs)} | {', '.join(reach)} | "
            f"{common[0][0] if common else ''} |"
        )
    isaac_pass = sum(recorded.get(n, "").startswith("pass") for n in names)
    always = sum(1 for runs in found.values() if runs and all(o.success for o in runs))
    never = sum(1 for runs in found.values() if runs and not any(o.success for o in runs))
    lines += [
        "",
        f"Isaac Sim (one run each): {isaac_pass}/{len(names)} pass. "
        f"MuJoCo: {sum(rates):.1f} of {len(found)} expected to pass, {always} always pass, {never} never pass, "
        f"{len(found) - always - never} sometimes.",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse

    from robotforge.zenobench import source

    ap = argparse.ArgumentParser(prog="robotforge zenobench sweep", description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5, help="seeds 0..n-1; 0 is the scene as authored")
    ap.add_argument("--jobs", type=int, default=4, help="scenarios run at once within a seed")
    ap.add_argument("--out", default="runs/sweep", help="relative to the ZenoBench checkout")
    ap.add_argument("--id", action="append", default=[], help="a scenario (repeatable); default all")
    ap.add_argument("--summary", action="store_true", help="only summarise what is in --out")
    args = ap.parse_args(argv)
    root = source.check()
    out = (root / args.out).resolve()
    if not args.summary:
        known = scenario_ids(root)
        unknown = [i for i in args.id if i not in known]
        if unknown:
            print(f"error: unknown scenario {', '.join(unknown)}", file=sys.stderr)
            return 2
        run(root, out, args.seeds, args.jobs, args.id or known)
    text = summary(root, out)
    (out / "summary.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
