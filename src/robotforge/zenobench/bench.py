"""ZenoBench's whole-task benchmark on MuJoCo next to Isaac Sim's recorded `benchmark/RESULTS.md`.

`benchmark/run_episode.py --batch` writes one `<task>/<episode>/result.json` per episode; this module reads
those from one or more run directories (a later directory replaces an episode, so a retest can sit on top of
a full batch) and compares them, episode by episode, with RESULTS.md's per-episode table
(`| Task | Seed | Variant | Planner | Route | Outcome | ...`). Episodes are grouped the way ZenoBench reports
them: scripted routes, single-skill probes, and the fixed / retry / oracle planner baselines.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

RESULTS = Path("benchmark/RESULTS.md")
GROUPS = ("routes", "probes", "fixed", "retry", "oracle", "recovery")

Key = tuple[str, int, str, str, str]  # task, seed, variant, planner, route


@dataclass
class Group:
    passed: int = 0
    runs: int = 0
    isaac_passed: int = 0
    isaac_runs: int = 0
    differences: list[str] = field(default_factory=list)


def isaac(root: Path) -> dict[Key, bool]:
    """Episode -> whether it passed on Isaac Sim."""
    found = {}
    for line in (root / RESULTS).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and re.match(r"^[PT]\d\d_", cells[0]) and cells[1].isdigit():
            found[(cells[0], int(cells[1]), cells[2], cells[3], cells[4])] = cells[5] == "pass"
    return found


def _failure(result: dict) -> str:
    """The first call that did not do what the route expected of it."""
    for call in result.get("calls", []):
        feedback = call.get("feedback") or {}
        ok, expect = feedback.get("success"), call.get("expect")
        if (expect == "ok" and ok is False) or (expect == "fail" and ok is True):
            what = feedback.get("error") or ("succeeded" if ok else "failed")
            return f"{call.get('skill')} (expected {expect}): {what}"
    return "" if result.get("success") else "task goal not met"


def ours(dirs: list[Path]) -> dict[Key, tuple[bool, str]]:
    """Episode -> (passed, first unexpected call) on MuJoCo."""
    found = {}
    for d in dirs:
        for path in sorted(d.glob("*/*/result.json")):
            r = json.loads(path.read_text(encoding="utf-8"))
            planner, route = r.get("planner") or "scripted", r.get("route") or "nominal"
            key = (r["task_id"], int(r["seed"]), r["variant"], planner, route)
            found[key] = (bool(r["success"]), _failure(r))
    return found


def _group(key: Key) -> str:
    task, planner = key[0], key[3]
    if task.startswith("P"):
        return "probes"
    return "routes" if planner == "scripted" else planner


def compare(root: Path, dirs: list[Path]) -> dict[str, Group]:
    recorded, groups = isaac(root), collections.defaultdict(Group)
    for key, (ok, failure) in sorted(ours(dirs).items()):
        g = groups[_group(key)]
        g.passed += ok
        g.runs += 1
        if key not in recorded:
            continue
        g.isaac_passed += recorded[key]
        g.isaac_runs += 1
        if recorded[key] != ok:
            side = "Isaac pass, MuJoCo fail" if recorded[key] else "Isaac fail, MuJoCo pass"
            g.differences.append(f"{key[0]} s{key[1]} {key[2]} {key[4]}: {side}" + (f" - {failure}" if failure else ""))
    return dict(groups)


def report(groups: dict[str, Group]) -> str:
    lines = ["| Batch | MuJoCo | Isaac Sim (same episodes) |", "|---|---|---|"]
    for name in [g for g in GROUPS if g in groups]:
        g = groups[name]
        lines.append(f"| {name} | {g.passed}/{g.runs} | {g.isaac_passed}/{g.isaac_runs} |")
    for name in [g for g in GROUPS if g in groups and groups[g].differences]:
        lines += ["", f"{name}:"] + [f"- {d}" for d in groups[name].differences]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    import argparse

    from robotforge.zenobench import source

    ap = argparse.ArgumentParser(prog="robotforge zenobench compare", description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run_episode.py --out directories, relative to the ZenoBench checkout")
    ap.add_argument("--md", help="also write the comparison to this Markdown file")
    args = ap.parse_args(argv)
    root = source.check()
    dirs = [(root / d).resolve() for d in args.runs]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        print(f"error: no run directory {', '.join(missing)}", file=sys.stderr)
        return 2
    text = report(compare(root, dirs))
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0
