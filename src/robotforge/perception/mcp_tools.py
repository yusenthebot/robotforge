"""The MCP tool for mapping and localization: `mapping` starts a map, explores, localizes, and reports how good the
map and the pose are. Going somewhere on the robot's own map is `move(robot, to=..., map="own")`.

Registered by robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

from robotforge.playground import Playground, PlaygroundError

MAX_EXPLORE_SECONDS = 300.0
Action = Literal["start", "status", "explore", "localize", "stop", "clear"]


def register(
    server: MCPServer,
    playground: Playground,
    run_until: Callable[..., Awaitable[dict[str, Any]]],
) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    hub = playground.perception

    @server.tool(annotations=ToolAnnotations(openWorldHint=False))
    async def mapping(robot: str, action: Action = "status", options: dict[str, Any] | None = None) -> list:
        """A robot draws a map of where it goes and finds itself in it, from its own sensors (a lidar and an odometry,
        added if it has none). action: "start" begins an occupancy-grid map (options: pose "estimate" = a particle
        filter corrects the odometry's drift against the map, "odometry" = the odometry alone, "truth" = the exact pose;
        resolution in m (0.05), particles (150), lidar and odometry = the name of one the robot has, or options for
        the one to add); the map then grows while the robot moves (move, move(to=...), keys). "explore" drives it to
        the nearest frontier again and again (options: seconds (60, at most 300), area [x0, y0, x1, y1], speed) until
        nothing reachable is unseen. "status" gives the map's size, the pose estimate with its error against the true
        pose (and the odometry's own error), and the map's accuracy against the scene (precision, wall_error_m,
        recall, unsafe_free); options {"image": true} also returns the map as a picture. "localize" keeps the map as it
        is and finds the robot in it (options: init "pose" = from where it believes it is, "global" = from nowhere,
        particles). "stop" stops growing the map (kept for localizing and navigating); "clear" throws the map away.
        The viewer draws the growing map, the particles and the pose. move(robot, to=..., map="own") then goes
        there by the robot's own map and pose alone (it plans through space it has seen free; a name or point to go
        to is the task, the way is its own)."""
        with _tool_errors():
            opts = dict(options or {})
            if action == "start":
                out = await asyncio.to_thread(hub.start, robot, opts)
            elif action == "status":
                image = bool(opts.pop("image", False))
                if opts:
                    raise PlaygroundError(f"status takes only the option image, not {', '.join(map(repr, opts))}")
                out = await asyncio.to_thread(hub.status, robot)
                if image:
                    png = await asyncio.to_thread(hub.picture, robot)
                    return [_json(out), Image(data=png, format="png")]
            elif action == "explore":
                out = await explore(robot, opts)
            elif action == "localize":
                init = str(opts.pop("init", "pose"))
                particles = opts.pop("particles", None)
                if opts:
                    raise PlaygroundError(f"localize takes init and particles, not {', '.join(map(repr, opts))}")
                out = await asyncio.to_thread(hub.localize, robot, init, particles)
            elif action == "stop":
                out = hub.stop(robot)
            else:
                out = hub.clear(robot)
            return [_json(out)]

    async def explore(robot: str, opts: dict[str, Any]) -> dict[str, Any]:
        seconds = float(opts.pop("seconds", 60.0))
        area = opts.pop("area", None)
        speed = opts.pop("speed", None)
        if opts:
            raise PlaygroundError(f"explore takes seconds, area and speed, not {', '.join(map(repr, opts))}")
        if not 1.0 <= seconds <= MAX_EXPLORE_SECONDS:
            raise PlaygroundError(f"seconds is how long it may explore, 1 to {MAX_EXPLORE_SECONDS:g}, got {seconds:g}")
        if area is not None and (
            not isinstance(area, list) or len(area) != 4 or area[0] >= area[2] or area[1] >= area[3]
        ):
            raise PlaygroundError("area is [x0, y0, x1, y1] in meters, the low corner first")
        if speed is not None and not 0.02 <= float(speed) <= 5:
            raise PlaygroundError(f"speed is its top speed on the way, 0.02 to 5 m/s, got {speed}")
        mapper = hub.mapper(robot)  # refuses a robot that is not mapping, before anything moves
        session = playground.session
        started = float(session.data.time)
        trips = hub.explore_trips(mapper.robot, seconds, area, None if speed is None else float(speed))

        def next_trip() -> tuple[Any, Any]:
            try:
                return next(trips)
            except StopIteration as done:  # not raised into a future, which cannot carry it
                return None, done.value or {}

        while True:
            trip, limit = await asyncio.to_thread(next_trip)
            if trip is None:
                summary = limit
                break
            await run_until(lambda t=trip: t.done, limit + 1.0)
        out = await asyncio.to_thread(hub.status, robot)
        return {"explored": {**summary, "sim_seconds": round(float(session.data.time) - started, 1)}, **out}
