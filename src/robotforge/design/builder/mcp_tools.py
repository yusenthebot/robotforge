"""MCP tools for robots built from parts: the cheat-sheet and part-by-part editing.

Registered by robotforge.mcp_server.build_server via `register(...)`; add_robot(build=...)
itself lives with the other add_robot sources in the server.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from robotforge import wording
from robotforge.design.builder.guide import guide
from robotforge.playground import Playground


def register(server: MCPServer, playground: Playground) -> None:
    @server.tool()
    def builder_guide() -> str:
        """How to build any robot from parts: the build schema (base, parts, joints, actuators, shapes, anchors,
        sensors, mirror, repeat, mounting a template such as an arm on a rover), the parts catalog, the built-in
        behaviors a built robot gets and a complete four-legged example. Read it before add_robot(build=...) or
        add_part."""
        return _json(guide())

    @server.tool()
    async def add_part(robot: str, part: dict[str, Any]) -> str:
        """Add a part to a robot built from parts. The robot is rebuilt and checked; a part that overlaps another or
        would make it fail is refused with an explanation and nothing changes.

        Example (a head with a camera on top of the base):
        add_part(robot="dog", part={"name": "head", "at": "front", "joint": {"axis": "z", "range": [-60, 60]},
                 "shape": "sphere", "size": [0.06], "direction": "forward", "sensors": ["camera"]})"""
        with _tool_errors():
            return _json(await asyncio.to_thread(playground.add_part, robot, part))

    @server.tool()
    async def update_part(robot: str, part: str, changes: dict[str, Any]) -> str:
        """Change fields of one part of a robot built from parts; nested fields (joint, actuator, repeat, a mounted
        robot's params and home) merge and null resets a field to its default. part="base" edits the base,
        part="robot" the robot-wide settings (color, material, servo, motor, linear, servo_caps). Copies made by
        mirror or repeat follow their source.

        Examples: update_part(robot="dog", part="left_thigh", changes={"size": [0.022, 0.15], "joint": {"home":
        50}}); update_part(robot="dog", part="robot", changes={"servo": "xl430"})"""
        with _tool_errors():
            return _json(await asyncio.to_thread(playground.update_part, robot, part, changes))

    @server.tool()
    async def remove_part(robot: str, part: str) -> str:
        """Remove a part from a robot built from parts, together with every part that hangs on it.

        Example: remove_part(robot="dog", part="head")"""
        with _tool_errors():
            return _json(await asyncio.to_thread(playground.remove_part, robot, part))


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Anticipated failures become ToolError so the agent sees the actual message."""
    try:
        yield
    except (ValueError, KeyError) as exc:
        raise ToolError(str(exc)) from exc


def _json(data: object) -> str:
    return wording.dumps(data)
