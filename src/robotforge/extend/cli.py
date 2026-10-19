"""`robotforge plugins` and `robotforge export`: the plugin list and the exporter without an agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_parsers(sub: argparse._SubParsersAction) -> None:
    q = sub.add_parser(
        "plugins",
        help="list the plugins robotforge finds in ~/.robotforge/plugins and any that fail to load; "
        "`plugins show WORLD` prints the Python files of a world's plugins/ folder and `plugins trust WORLD` "
        "approves them after you have read them",
    )
    q.add_argument("action", nargs="?", choices=["list", "show", "trust"], default="list")
    q.add_argument("world", nargs="?", default=None, help="the name or folder of a world (show and trust)")
    q.add_argument("--yes", action="store_true", help="trust without asking (you have read the files)")
    p = sub.add_parser("export", help="write a world, or one robot, as MJCF or URDF with meshes for other simulators")
    p.add_argument("world", help="the name or folder of a world")
    p.add_argument("--robot", default=None, help="export just this robot (required for URDF)")
    p.add_argument("--format", choices=["mjcf", "urdf"], default="mjcf")
    p.add_argument("--to", type=Path, default=None, help="folder to write into (default: exports/ in the world)")
    p.add_argument(
        "--overwrite", action="store_true", help="replace an earlier export in a --to folder that is not empty"
    )


def run(args: argparse.Namespace) -> int:
    from robotforge.extend.registry import REGISTRY

    if args.command == "plugins":
        if args.action != "list":
            return _show_or_trust(args)
        REGISTRY.load_user()
        print(json.dumps(REGISTRY.report(), indent=2, ensure_ascii=False))
        return 1 if REGISTRY.problems() else 0
    return _export(args)


def _show_or_trust(args: argparse.Namespace) -> int:
    """`robotforge plugins show WORLD` prints what a world's plugins/ folder holds; `trust WORLD` prints it and, once
    the person at the terminal types yes, approves it. This is the only way to trust a world's own Python."""
    from robotforge.design.store import WorldStore
    from robotforge.extend.registry import REGISTRY

    if not args.world:
        print(f"error: robotforge plugins {args.action} needs a world (its name or folder)", file=sys.stderr)
        return 2
    try:
        directory = WorldStore.open(args.world).dir
        shown = REGISTRY.show(directory)
        if not shown["files"]:
            print(f"error: {directory} has no plugins/ folder with files", file=sys.stderr)
            return 2
        for rel, text in shown["files"].items():
            print(f"==== {rel}\n{text}")
        if shown.get("cannot_be_trusted"):
            print(f"error: cannot be trusted: {'; '.join(shown['cannot_be_trusted'])}", file=sys.stderr)
            return 2
        if args.action == "show":
            print(f"({'trusted' if shown['trusted'] else 'not trusted'}: {shown['folder']})")
            return 0
        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    "error: trusting needs you to type yes at a terminal (or --yes once you have read it)",
                    file=sys.stderr,
                )
                return 2
            names = ", ".join(shown["files"])
            if (
                input(
                    f"\nThis is Python that will run on this computer with your rights ({names}). Trust it? [yes/N] "
                ).strip()
                != "yes"
            ):
                print("not trusted")
                return 1
        names = REGISTRY.trust(directory)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"trusted {len(names)} files: {', '.join(names)}")
    return 0


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
