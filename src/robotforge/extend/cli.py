"""`robotforge plugins` and `robotforge export`: the plugin list and the exporter without an agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_parsers(sub: argparse._SubParsersAction) -> None:
    sub.add_parser(
        "plugins", help="list the plugins robotforge finds in ~/.robotforge/plugins and any that fail to load"
    )
    p = sub.add_parser("export", help="write a world, or one robot, as MJCF or URDF with meshes for other simulators")
    p.add_argument("world", help="the name or folder of a world")
    p.add_argument("--robot", default=None, help="export just this robot (required for URDF)")
    p.add_argument("--format", choices=["mjcf", "urdf"], default="mjcf")
    p.add_argument("--to", type=Path, default=None, help="folder to write into (default: exports/ in the world)")
    p.add_argument("--overwrite", action="store_true", help="replace an earlier export in a --to folder that is not empty")


def run(args: argparse.Namespace) -> int:
    from robotforge.extend.registry import REGISTRY

    if args.command == "plugins":
        REGISTRY.load_user()
        print(json.dumps(REGISTRY.report(), indent=2, ensure_ascii=False))
        return 1 if REGISTRY.problems() else 0
    return _export(args)


def _export(args: argparse.Namespace) -> int:
    from robotforge.design import export
    from robotforge.design.store import WorldStore
    from robotforge.extend.registry import REGISTRY

    try:
        store = WorldStore.open(args.world)
        world = store.load()
        for why in REGISTRY.problems().values():
            print(f"note: a plugin was not loaded: {why}", file=sys.stderr)
        entry = None
        if args.robot:
            entry = next((r for r in world.robots if r.name == args.robot), None)
            if entry is None:
                known = ", ".join(r.name for r in world.robots) or "none"
                raise export.ExportError(f"the world has no robot named {args.robot!r}; its robots: {known}")
        name = entry.name if entry is not None else world.name
        default = store.dir / "exports" / f"{name}_{args.format}"
        folder, own = export.target_folder(args.to, default, store.dir, args.overwrite)
        result = export.export_world(world, entry, store.dir, args.format, folder, own_folder=own)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["check"].get("same") else 1
