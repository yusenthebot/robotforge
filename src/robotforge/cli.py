"""Command line entry: `robotforge open <model>` and `robotforge demo`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import socket
import sys
import webbrowser
from pathlib import Path

import uvicorn

from robotforge.sim.session import ModelLoadError, SimSession
from robotforge.viewer.server import ViewerHub, create_app

DEMO_MODEL = Path(__file__).parent / "assets" / "demo.xml"
DEFAULT_PORT = 8737

log = logging.getLogger("robotforge")


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    # Same socket option uvicorn uses, so sockets lingering in TIME_WAIT don't count as busy.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def find_free_port(start: int, host: str = "127.0.0.1", attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        if port_is_free(port, host):
            return port
    raise RuntimeError(f"no free port in {start}-{start + attempts - 1}")


async def serve(session: SimSession, host: str, port: int, open_browser: bool) -> None:
    hub = ViewerHub(session)
    app = create_app(hub)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    url = f"http://{host}:{port}/"
    print(f"robotforge viewer: {url}  (model: {session.source})", flush=True)
    if open_browser:
        webbrowser.open(url)
    await server.serve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robotforge", description="Beginner-friendly robot playground.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_open = sub.add_parser("open", help="open an MJCF or URDF model in the viewer")
    p_open.add_argument("model", type=Path)
    sub.add_parser("demo", help="open the built-in demo scene")
    p_mcp = sub.add_parser("mcp", help="run as an MCP server on stdio (for Claude, Codex, ...)")
    p_mcp.add_argument("--model", type=Path, default=None, help="model to load at start (default: demo)")
    p_mcp.add_argument("--port", type=int, default=None, help="viewer port (default: first free from 8737)")
    for p in (p_open, sub.choices["demo"]):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=None, help=f"default: first free port from {DEFAULT_PORT}")
        p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")

    args = parser.parse_args(argv)
    if args.command == "mcp":
        from robotforge import mcp_server  # imported lazily: the viewer CLI does not need the MCP SDK

        mcp_server.run(model=args.model, port=args.port)
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    model_path = DEMO_MODEL if args.command == "demo" else args.model
    try:
        session = SimSession.from_path(model_path)
    except ModelLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.port is None:
        port = find_free_port(DEFAULT_PORT, args.host)
    elif port_is_free(args.port, args.host):
        port = args.port
    else:
        print(f"error: port {args.port} is already in use", file=sys.stderr)
        return 2
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve(session, args.host, port, open_browser=not args.no_browser))
    return 0
