"""`robotforge zenobench ...`: fetch ZenoBench and run its episodes, scripts and HTTP service on MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
import sys


def add_parsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("zenobench", help="ZenoBench (Zeno Malo house tasks) on MuJoCo, without Isaac Sim")
    zsub = p.add_subparsers(dest="zenobench_command", required=True)
    zsub.add_parser("fetch", help="clone ZenoBench at the verified commit with its Git LFS scenes (about 1 GB)")
    zsub.add_parser("check", help="where the checkout is and whether its scenes are present")
    e = zsub.add_parser(
        "episode",
        help="run benchmark/run_episode.py (arguments after --)",
        description="Example: robotforge zenobench episode -- --tasks skill_library/tasks.json "
        "--task collect_fruits --planner oracle",
    )
    e.add_argument("args", nargs=argparse.REMAINDER)
    r = zsub.add_parser("python", help="run a ZenoBench script or `-m module` as Isaac Lab's python would")
    r.add_argument("args", nargs=argparse.REMAINDER)
    s = zsub.add_parser("serve", help="ZenoBench's env_service HTTP server for an external planner")
    s.add_argument("args", nargs=argparse.REMAINDER, help="env_service.server arguments, e.g. --port 8800")
    w = zsub.add_parser(
        "sweep",
        help="run the skill scenarios over several seeds and compare pass rates with Isaac Sim's",
        description="Example: robotforge zenobench sweep -- --seeds 5 --jobs 6 (arguments after --)",
    )
    w.add_argument("args", nargs=argparse.REMAINDER)
    c = zsub.add_parser(
        "compare",
        help="compare benchmark episodes run here with Isaac Sim's benchmark/RESULTS.md",
        description="Example: robotforge zenobench compare -- runs/bench/accept runs/bench/probes (arguments after --)",
    )
    c.add_argument("args", nargs=argparse.REMAINDER)


def _rest(args: list[str]) -> list[str]:
    return args[1:] if args and args[0] == "--" else args


def run(args: argparse.Namespace) -> int:
    from robotforge.zenobench import source

    command = args.zenobench_command
    try:
        if command == "fetch":
            print(source.fetch())
            return 0
        if command == "check":
            root = source.root()
            try:
                source.check(root)
                ok = True
            except source.SourceError as exc:
                ok, root = False, str(exc)
            print(json.dumps({"ready": ok, "root": str(root), "commit": source.COMMIT}, indent=2))
            return 0 if ok else 1
        from robotforge.zenobench import python

        root = source.check()
        os.chdir(root)  # ZenoBench's scripts take paths relative to the repository
        if command == "episode":
            return python.main([str(root / "benchmark" / "run_episode.py"), *_rest(args.args)])
        if command == "python":
            return python.main(_rest(args.args))
        if command == "sweep":
            from robotforge.zenobench import sweep

            return sweep.main(_rest(args.args))
        if command == "compare":
            from robotforge.zenobench import bench

            return bench.main(_rest(args.args))
        if command == "serve":
            rest = _rest(args.args)
            if "--worker-python" not in rest:
                rest = [*rest, "--worker-python", str(python.wrapper())]
            return python.main(["-m", "env_service.server", *rest])
    except source.SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
