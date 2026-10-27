"""Command line entry: `robotforge open <model>`, `robotforge demo`, `robotforge mcp`, `robotforge unilab ...`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import socket
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path

import mujoco
import uvicorn

from robotforge.design import importer
from robotforge.design.importing.report import render_text
from robotforge.design.store import WorldStore
from robotforge.playground import Playground
from robotforge.sim.session import ModelLoadError, SimSession
from robotforge.viewer.server import ViewerHub, create_app

DEMO_MODEL = Path(__file__).parent / "assets" / "demo.xml"
DEFAULT_PORT = 8737

log = logging.getLogger("robotforge")


def _server_socket() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform == "win32":
        # SO_REUSEADDR means "share the port" on Windows, so the probe would call a port in use free.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        # Same option uvicorn uses, so sockets lingering in TIME_WAIT don't count as busy.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with _server_socket() as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def listen_on_free_port(start: int, host: str = "127.0.0.1", attempts: int = 50) -> socket.socket:
    """A socket listening on the first free port from `start`, for the viewer to serve on.

    Holding the port closes the gap between finding it free and binding it: two MCP servers started at the same
    moment (two agent sessions) both found 8737 free, and the second one's viewer then failed to bind and took
    its MCP server down with it, so that session had no robotforge tools at all."""
    for port in range(start, start + attempts):
        sock = _server_socket()
        try:
            sock.bind((host, port))
            sock.listen()  # at once: until a socket listens, Linux lets a second SO_REUSEADDR socket bind its port
        except OSError:
            sock.close()
            continue
        return sock
    if attempts == 1:
        raise RuntimeError(f"port {start} is already in use")
    raise RuntimeError(f"no free port in {start}-{start + attempts - 1}")


def has_display(platform: str = sys.platform, env: Mapping[str, str] = os.environ) -> bool:
    """Windows and macOS always have a desktop; on Linux there is one only under X11 or Wayland."""
    if platform.startswith("linux"):
        return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))
    return True


def open_in_browser(url: str) -> bool:
    """Open `url` in a desktop browser if this machine has one.

    Without a display Python's webbrowser falls back to text browsers like lynx, which block until
    quit and, inside an MCP server, would read the protocol stream from stdin.
    """
    if not has_display():
        return False
    try:
        return webbrowser.open(url)
    except webbrowser.Error:
        return False


def find_free_port(start: int, host: str = "127.0.0.1", attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        if port_is_free(port, host):
            return port
    raise RuntimeError(f"no free port in {start}-{start + attempts - 1}")


def load_playground(path: Path, options: dict | None = None) -> Playground:
    """Open an MJCF file, a URDF/xacro/mesh (through the import doctor), or a world folder / world.yaml."""
    path = path.expanduser()
    if importer.is_importable(path):
        compiled = importer.compile_standalone(path, options)
        session = SimSession(model=compiled.model, data=mujoco.MjData(compiled.model), source=str(path.resolve()))
        session.reset()
        playground = Playground(session=session)
        playground.last_doctor = compiled.doctor
        return playground
    world_file = path / "world.yaml" if path.is_dir() else path
    if world_file.name == "world.yaml" and world_file.exists():
        store = WorldStore(world_file.parent)
        compiled = store.compile()
        session = SimSession(model=compiled.model, data=mujoco.MjData(compiled.model), source=str(world_file))
        session.reset()
        return Playground(session=session, store=store, _play_world_descriptor=compiled.world_sha256)
    return Playground(session=SimSession.from_path(path))


async def serve(playground: Playground, host: str, port: int, open_browser: bool) -> None:
    hub = ViewerHub(playground.session)
    playground.attach_viewer(hub)
    app = create_app(hub)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://{host}:{port}/"
    playground.viewer_url = url
    print(f"robotforge viewer: {url}  (model: {playground.session.source})", flush=True)
    if open_browser:
        open_in_browser(url)
    await server.serve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotforge", description="Beginner-friendly robot playground.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_open = sub.add_parser("open", help="open an MJCF/URDF/xacro/mesh model or a world folder in the viewer")
    p_open.add_argument("model", type=Path)
    p_open.add_argument("--base", choices=["auto", "fixed", "floating"], default=None, help="URDF/mesh: base type")
    p_open.add_argument("--decompose", action="store_true", help="split concave collision meshes (slower)")
    p_open.add_argument("--scale", type=float, default=None, help="plain mesh: size multiplier")
    p_open.add_argument(
        "--respect-effort-limits",
        action="store_true",
        help="URDF: keep the file's torque limits even where they are too weak to hold the robot up",
    )
    p_open.add_argument(
        "--xacro-arg", action="append", default=[], metavar="KEY=VALUE", help="value for a <xacro:arg> (repeatable)"
    )
    sub.add_parser("demo", help="open the built-in demo scene")
    p_mcp = sub.add_parser("mcp", help="run as an MCP server on stdio (for Claude, Codex, ...)")
    p_mcp.add_argument("--model", type=Path, default=None, help="model to load at start (default: demo)")
    p_mcp.add_argument("--port", type=int, default=None, help="viewer port (default: first free from 8737)")
    from robotforge.bridge import cli as bridge_cli  # `policies` and `unilab ...`

    bridge_cli.add_parsers(sub)
    from robotforge.extend import cli as extend_cli  # `plugins` and `export`

    extend_cli.add_parsers(sub)
    from robotforge.zenobench import cli as zenobench_cli  # `zenobench ...` (optional extra)

    zenobench_cli.add_parsers(sub)
    for p in (p_open, sub.choices["demo"]):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=None, help=f"default: first free port from {DEFAULT_PORT}")
        p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    args = parser.parse_args(argv)
    if args.command in ("walkers", "unilab"):
        return bridge_cli.run(args)
    if args.command in ("plugins", "export"):
        return extend_cli.run(args)
    if args.command == "zenobench":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        return zenobench_cli.run(args)
    if args.command == "mcp":
        from robotforge import mcp_server  # imported lazily: the viewer CLI does not need the MCP SDK

        mcp_server.run(model=args.model, port=args.port)
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model_path = DEMO_MODEL if args.command == "demo" else args.model
    options = {}
    if args.command == "open":
        options = {k: v for k, v in (("base", args.base), ("scale", args.scale)) if v is not None}
        if args.decompose:
            options["decompose"] = True
        if args.respect_effort_limits:
            options["respect_effort_limits"] = True
        if args.xacro_arg:
            pairs = [a.partition("=") for a in args.xacro_arg]
            if any(not sep for _, sep, _ in pairs):
                print("error: --xacro-arg needs KEY=VALUE, for example --xacro-arg hand=true", file=sys.stderr)
                return 2
            options["xacro_args"] = {k: v for k, _, v in pairs}
    try:
        playground = load_playground(model_path, options)
    except (ModelLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for report in playground.last_doctor.values():
        print(render_text(report), flush=True)

    if args.port is None:
        port = find_free_port(DEFAULT_PORT, args.host)
    elif port_is_free(args.port, args.host):
        port = args.port
    else:
        print(f"error: port {args.port} is already in use", file=sys.stderr)
        return 2
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(playground, args.host, port, open_browser=not args.no_browser))
    return 0
