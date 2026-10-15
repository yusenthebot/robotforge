"""`robotforge unilab ...` and `robotforge walkers`: the command line side of the bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from robotforge.bridge import policies, unilab
from robotforge.bridge.spec import SpecError

POLL_SECONDS = 5.0


def add_parsers(sub: argparse._SubParsersAction) -> None:
    sub.add_parser("walkers", help="list the learned walking policies robotforge can play")
    p = sub.add_parser("unilab", help="train walking policies with UniLab (external RL framework)")
    usub = p.add_subparsers(dest="unilab_command", required=True)
    usub.add_parser("check", help=f"check the UniLab checkout named by ${unilab.ENV_VAR}")
    t = usub.add_parser("train", help="train a policy in the background, export it to ONNX and install it")
    t.add_argument(
        "--task", default="go2_joystick_flat", help=f"one of {', '.join(sorted(unilab.TASKS))}, or a task in --conf-dir"
    )
    t.add_argument(
        "--conf-dir", type=Path, default=None, help="folder with task/<task>/mujoco.yaml (a robot of your own)"
    )
    t.add_argument("--iterations", type=int, default=1000)
    t.add_argument("--name", default="go2_walk_trained", help="name of the installed policy")
    t.add_argument("--num-envs", type=int, default=None, help="parallel simulations (UniLab's default: 1024)")
    t.add_argument("--wait", action="store_true", help="follow progress until the job ends")
    s = usub.add_parser("status", help="progress of a training job (the newest by default)")
    s.add_argument("job", nargs="?", default=None)
    s.add_argument("--wait", action="store_true", help="follow progress until the job ends")
    c = usub.add_parser("cancel", help="stop a training job")
    c.add_argument("job", nargs="?", default=None)
    e = usub.add_parser("export", help="export an existing UniLab run directory and install it")
    e.add_argument("run", type=Path)
    e.add_argument("--name", required=True)
    e.add_argument("--task", default=None, help="UniLab task owner, e.g. go2_joystick_flat (default: inferred)")


def run(args: argparse.Namespace) -> int:
    try:
        if args.command == "walkers":
            return _print([p.describe() for p in policies.all_policies().values()])
        command = args.unilab_command
        if command == "check":
            return _print(unilab.check())
        if command == "train":
            started = unilab.start_training(args.task, args.iterations, args.name, args.num_envs, args.conf_dir)
            _print(started)
            return _follow(started["job"]) if args.wait else 0
        if command == "status":
            return _follow(args.job) if args.wait else _print(unilab.job_status(args.job))
        if command == "cancel":
            return _print(unilab.cancel(args.job))
        if command == "export":
            return _print(unilab.export_run(args.run, args.name, args.task))
    except (unilab.UniLabError, SpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args}")  # argparse only lets known commands through


def _follow(job: str | None) -> int:
    last = None
    while True:
        status = unilab.job_status(job)
        line = _progress_line(status)
        if line != last:
            print(line, flush=True)
            last = line
        if status.get("state") in ("done", "failed"):
            _print(status)
            return 0 if status["state"] == "done" else 1
        time.sleep(POLL_SECONDS)


def _progress_line(status: dict[str, Any]) -> str:
    parts = [status.get("job", "?"), status.get("state", "?")]
    if status.get("iterations"):
        parts.append(f"iteration {status.get('iteration', 0)}/{status['iterations']}")
    if "mean_reward" in status:
        parts.append(f"reward {status['mean_reward']:.1f}")
    if status.get("eta_s") is not None and status.get("state") == "training":
        parts.append(f"about {status['eta_s'] / 60:.1f} min left")
    return "  ".join(str(p) for p in parts)


def _print(data: object) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    return 0
