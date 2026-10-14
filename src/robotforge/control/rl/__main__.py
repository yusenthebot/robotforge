"""Command line: train a policy, or (internal) run one training job.

    python -m robotforge.control.rl train balance --minutes 2
    python -m robotforge.control.rl train walk --algo ppo --timesteps 600000 --name dog_walk
    python -m robotforge.control.rl list

This module stays light (no PyTorch at import): multiprocessing workers may import it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robotforge.control.rl", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train a policy on a built-in task and wait for it")
    t.add_argument("task", help="reach, balance, stand, walk or drive_to")
    t.add_argument("--algo", choices=["ppo", "sac"])
    t.add_argument("--timesteps", type=int)
    t.add_argument("--minutes", type=float)
    t.add_argument("--n-envs", type=int)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--name")
    t.add_argument("--mode", choices=["direct", "residual"], dest="action_mode")
    t.add_argument("--task-params", type=json.loads, default=None, help="JSON, e.g. '{\"speed\": 0.4}'")
    t.add_argument("--save-dir", type=Path)
    sub.add_parser("list", help="list trained policies")
    j = sub.add_parser("job", help=argparse.SUPPRESS)
    j.add_argument("folder", type=Path)
    args = parser.parse_args(argv)

    if args.command == "job":
        from robotforge.control.rl.train import run_job  # imports PyTorch: only in the training process

        return run_job(args.folder)
    if args.command == "list":
        from robotforge.control.rl.store import list_policies

        print(json.dumps(list_policies(), indent=1))
        return 0
    from robotforge.control.rl.api import train
    from robotforge.control.tasks import TaskError

    try:
        policy = train(
            args.task,
            args.algo,
            timesteps=args.timesteps,
            minutes=args.minutes,
            n_envs=args.n_envs,
            seed=args.seed,
            name=args.name,
            action_mode=args.action_mode,
            task_params=args.task_params,
            save_dir=args.save_dir,
        )
    except TaskError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"policy": str(policy.folder), "final": policy.meta.get("final")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
