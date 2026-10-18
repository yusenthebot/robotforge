"""MCP tools for robots working the things around them: operate drawers, doors, buttons, switches, levers and valves.

Registered by robotforge.mcp_server.build_server via `register(...)`, one line there.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from robotforge.behaviors import operate as operating
from robotforge.playground import Playground
from robotforge.sim import mechanisms

OPERATE_SECONDS = 90.0  # sim seconds an operation may take: a valve turned a full turn takes about 50


def register(server: MCPServer, playground: Playground, run_until: Callable[..., Awaitable[dict[str, Any]]]) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    @server.tool()
    async def operate(
        robot: str,
        target: str,
        action: Literal["open", "close", "press", "turn"],
        amount: float | None = None,
    ) -> str:
        """Have an arm work a mechanism: it takes hold of the handle and moves it along the joint's own line or arc.
        drawer: open / close, amount in m (default: all the way). cabinet: open / close the door, amount in degrees
        (default 90 open, all the way closed). button: press (it springs back). switch, lever: press flips it to the
        other side; turn tips it by amount degrees (+ toward -y). valve: open (counter-clockwise) / close / turn by
        amount degrees (default 90; past 90 it lets go and takes hold again until the wheel is there). Works with
        fingers or a suction cup; a magnet holds only the steel bars of drawers and doors (and pushes buttons). The
        result: moved (m or degrees), the state now, success (judged against how far the thing can go), and notes
        when the arm could not follow the whole way or did not take hold; a handle out of reach is refused with its
        distance. Put things in an open drawer with pick_and_place(robot, object, drawer)."""
        with _tool_errors():
            op, found, shape = operating.start(playground.behaviors, robot, target, action, amount)
            run = await run_until(lambda: op.done, OPERATE_SECONDS)
            settle = await run_until(lambda: False, 0.3)
            session = playground.session
            with session.lock:
                state = mechanisms.read(
                    session.model, session.data, {found: shape}, session.controllers.get(mechanisms.KEY)
                )
                report = operating.result(op, session.model, session.data, state.get(found, {}))
            seconds = round(run["sim_seconds"] + settle["sim_seconds"], 2)
            return _json({**report, "sim_seconds": seconds, "shown_live": run["shown_live"]})
