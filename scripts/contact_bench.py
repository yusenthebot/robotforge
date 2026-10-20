"""Which controller holds a force? Measure it, over several starts.

    uv run python scripts/contact_bench.py [--arms panda,ur5e] [--tasks wipe,press] [--methods osc,...]
        [--seeds 5] [--top 0.25] [--force N] [--jobs 3]

Each arm gets a wiper pad and a table `--top` m high under the pad's start. Every control method runs the same
task from `--seeds` starts (the tool a few cm to the side and higher or lower, ContactTask.reset), with the task's own
default time, and the tables give the mean over the starts and, in brackets, the lowest and highest. Wipe: the force
the filtered wrist sensor reads, its spread within a run, how often the unfiltered sensor reads under 0.3 of the force
asked (raw dropout), how far the tool strayed from the path, the share of the path wiped, and how many starts
succeeded. Press: the force it settles at and the highest it reaches in `hold` and `stop` mode. The library arms need
ROBOTFORGE_MENAGERIE (the test fixtures work offline) or a download.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

METHODS = ("ik", "pd", "joint_impedance", "cartesian_impedance", "osc")
# Where the arm's pad starts at home (x, y): the table goes under it, so the job is on the table and not the floor.
UNDER_THE_TOOL = {"panda": (0.5, 0.0), "ur5e": (-0.134, 0.492)}


def _world(arm: str, top: float):
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world(f"bench_{arm}", "checker")
    pg.add_robot("arm", library=arm)
    pg.update_robot("arm", end_effector="wiper")
    pg.add_object("desk", shape="table", size=[0.6, 0.6, top], pos=list(UNDER_THE_TOOL[arm]))
    return pg


def _runs(job: tuple[str, str, str, int, float, float]) -> dict[str, list[float]]:
    arm, task, method, seeds, top, force = job
    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp(prefix="contact_bench_")  # its own worlds folder per job
    import mujoco

    from robotforge.control import make_task
    from robotforge.control.compare import run

    pg = _world(arm, top)
    model, data0 = pg.session.model, copy.copy(pg.session.data)
    mujoco.mj_forward(model, data0)
    out: dict[str, list[float]] = {}
    for seed in range(seeds):
        modes = ({"mode": "hold"}, {"mode": "stop"}) if task == "press" else ({},)
        for extra in modes:
            data = copy.copy(data0)
            t = make_task(task, model, "arm", {"force": force, **extra})
            t.reset(data, np.random.default_rng(seed))
            mujoco.mj_forward(model, data)
            t.begin(data)
            res = run(t, model, data, method, t.default_seconds)
            rep = res.report
            tag = extra.get("mode", "")
            row = {"success": float(res.success)}
            if task == "wipe":
                row |= {
                    k: float(rep[k])
                    for k in (
                        "force_mean_n",
                        "force_std_n",
                        "lost_contact_pct",
                        "raw_dropout_pct",
                        "track_rms_mm",
                        "path_wiped_pct",
                    )
                    if k in rep
                }
            else:
                row |= {k: float(rep[k]) for k in ("settled_n", "peak_n") if k in rep}
            for k, v in row.items():
                out.setdefault(f"{tag}:{k}", []).append(v)
    return out


def _cell(values: list[float], digits: int = 1) -> str:
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    mean = float(np.mean(values))
    if abs(hi - lo) < 10 ** (-digits):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", default="panda,ur5e")
    parser.add_argument("--tasks", default="wipe,press")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top", type=float, default=0.25, help="table height, m")
    parser.add_argument("--force", type=float, default=None, help="N asked (default 8 for wipe, 6 for press)")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)
    if "ROBOTFORGE_MENAGERIE" not in os.environ:
        fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "menagerie"
        if fixtures.is_dir():
            os.environ["ROBOTFORGE_MENAGERIE"] = str(fixtures)
    arms, tasks = args.arms.split(","), args.tasks.split(",")
    jobs = [
        (arm, task, method, args.seeds, args.top, args.force or (8.0 if task == "wipe" else 6.0))
        for task in tasks
        for arm in arms
        for method in args.methods.split(",")
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(_runs, jobs))
    for task in tasks:
        print(f"\n{task}, table top {args.top} m, {args.seeds} starts (mean, and lowest to highest in brackets)\n")
        if task == "wipe":
            print("| arm | method | force mean (N) | force std (N) | lost contact (%) | raw dropout (%) |", end="")
            print(" path error (mm) | wiped (%) | succeeded |")
            print("|---|---|---|---|---|---|---|---|---|")
        else:
            print("| arm | method | hold: settled (N) | hold: peak (N) | stop: peak (N) | succeeded (hold, stop) |")
            print("|---|---|---|---|---|---|")
        rows = [(j, r) for j, r in zip(jobs, results, strict=True) if j[1] == task]
        for (arm, _, method, seeds, *_), r in rows:
            if task == "wipe":
                wins = int(sum(r.get(":success", [])))
                print(
                    f"| {arm} | {method} | {_cell(r.get(':force_mean_n', []))} | {_cell(r.get(':force_std_n', []))} "
                    f"| {_cell(r.get(':lost_contact_pct', []))} | {_cell(r.get(':raw_dropout_pct', []))} |"
                    f" {_cell(r.get(':track_rms_mm', []))} | {_cell(r.get(':path_wiped_pct', []))} | {wins}/{seeds} |"
                )
            else:
                hold, stop = int(sum(r.get("hold:success", []))), int(sum(r.get("stop:success", [])))
                print(
                    f"| {arm} | {method} | {_cell(r.get('hold:settled_n', []))} | {_cell(r.get('hold:peak_n', []))} "
                    f"| {_cell(r.get('stop:peak_n', []))} | {hold}/{seeds}, {stop}/{seeds} |"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
