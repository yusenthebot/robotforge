"""The mount tool: put any robot, end effector or sensor on any body of any robot in one call (see combine.py).

add_robot(mounts=...) and update_robot(mounts=...) still take a whole list; mount adds or swaps one thing, says what
the result is like (weight, balance, reach, what its arm can hit and hold) and keeps a finished combination as a
design. Taking one off is remove("robot/mount"), the tool that removes everything else. Registered by
robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from robotforge.playground import Playground


def register(server: MCPServer, playground: Playground) -> None:
    from robotforge.design import combine
    from robotforge.mcp_server import _download_with_progress, _json, _tool_errors  # mcp_server imports this module

    @server.tool()
    async def mount(
        ctx: Context,
        on: str,
        what: str | None = None,
        at: str | None = None,
        shift: list[float] | None = None,
        rotate: list[float] | None = None,
        name: str | None = None,
        params: dict[str, Any] | None = None,
        adapter: bool = True,
        save_as: str | None = None,
        overwrite: bool = False,
    ) -> str:
        """Mount anything on anything: a robot (a library arm such as ur5e or piper, a template), an end effector (a
        library gripper or hand: robotiq_2f85, panda_hand, the dexterous leap_hand and allegro_hand; or gripper,
        suction, magnet) or a sensor type (camera, lidar, lidar3d, imu, force_torque, ...) on robot `on`, or on one of
        its mounts as "robot/mount" ("rover/arm"). An end effector goes on that arm's flange and swaps out the one there
        (a quick change); a robot goes on top, or at a body name; a sensor on the arm's wrist or the robot's base, or at
        a body name. at: "top", "tool" (a flange) or a body; shift [x, y, z] m and rotate [roll, pitch, yaw] degrees
        from there. A robot stands on an adapter plate sized to both sides (adapter=false: none). params: template
        params or sensor options. Every mount is checked once on: mass, centre of mass, whether a walking or wheeled
        carrier still stands and how much margin it keeps, each arm's reach from its mount, what it can hit of its
        carrier and whether it can hold its tool stretched out; a robot that would tip or fall, or an arm that cannot
        hold itself, is refused saying what to change. save_as keeps the robot with all it carries as a design
        (add_robot(design=...) adds it again, mount(what=<design>) mounts it whole); with no what it saves as it is. A
        name already saved is refused unless overwrite=true, and a design cannot be saved under the name of one it carries.
        remove("rover/arm") takes a mount off with what is on it."""
        with _tool_errors():
            sort = None
            if what is not None:
                try:
                    sort, fields = combine.classify(what)
                except ValueError:
                    sort, fields = None, {}
            wanted = [fields["library"]] if sort in ("robot", "tool") and fields.get("library") else []
            if sort == "design":
                wanted = combine.libraries(fields["design"])
            for robot in wanted:
                await _download_with_progress(robot, ctx)
            options = {"at": at, "shift": shift, "rotate": rotate, "name": name, "params": params}
            options |= {"adapter": adapter, "save_as": save_as, "overwrite": overwrite}
            done = await asyncio.to_thread(playground.mount, what, on, **options)
            return _json(done)
