"""The MCP tool for demonstrations: `demonstration` records what a robot does (a teleoperated motion, a script, a
behavior), replays it on the same robot or another of its kind, and exports it as JSON or NPZ.

Registered by robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from robotforge.demonstrations import RATE_HZ
from robotforge.playground import Playground

Action = Literal["start", "stop", "replay", "export", "list"]
MARGIN = 3.0  # s a replay may run past the demonstration's own length before it is given up on


def register(
    server: MCPServer,
    playground: Playground,
    run_until: Callable[..., Awaitable[dict[str, Any]]],
) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    demos = playground.demonstrations

    @server.tool(annotations=ToolAnnotations(openWorldHint=False))
    async def demonstration(
        action: Action = "list",
        robot: str | None = None,
        name: str | None = None,
        mode: Literal["commands", "joints"] = "commands",
        format: Literal["json", "npz"] = "json",
        to: str | None = None,
        rate_hz: float = RATE_HZ,
        sensors: list[str] | None = None,
        overwrite: bool = False,
        wait: bool = True,
    ) -> str:
        """Record, replay and export demonstrations of a robot. "start" (robot, name, rate_hz, sensors) begins keeping,
        at rate_hz frames a simulated second, the robot's joint positions, its actuator targets, its base pose and its
        sensor readings (the cheap ones by default; sensors=[...] names the ones to read, [] none), and the commands it
        is given from now on, whoever gives them (the keyboard or a gamepad in the viewer, move, move_arm, a script).
        Then act, and "stop" (name, optional when one is recording) ends it: the result says how long it was and what
        commands it holds, and saves it as demonstrations/<name>.json in the world. "replay" (name, robot, mode)
        plays one on a robot of the same kind, which may be another one than recorded it (a Go2 replays a template
        dog's walk, a Panda an arm's reach; arm targets are kept relative to the arm's base): mode "commands" (the
        default) gives the recorded commands again at their times, mode "joints" puts the robot in the first pose and
        plays the actuator targets back as they were, for the same robot (or the same actuators), open loop. With
        wait=true (default) it returns when the replay is over, shown live if the viewer is open; wait=false starts it
        and returns. "export" (name, format "json" or "npz", to = a file name) writes the data for training in the
        world's demonstrations folder: the NPZ has time, joint_pos, joint_names, ctrl, ctrl_names, base_pos, base_quat,
        sensors, sensor_names and the commands; LeRobot's parquet needs a library robotforge does not install. "list"
        shows the demonstrations kept and what is recording or replaying. Scripts do the same with robot.record() and
        robot.replay(name)."""
        with _tool_errors():
            if action == "start":
                if robot is None:
                    raise ValueError("start needs robot: the robot to record")
                return _json(demos.start(robot, name, rate_hz, sensors, overwrite))
            if action == "stop":
                return _json(demos.stop(name))
            if action == "replay":
                if name is None:
                    raise ValueError("replay needs name: the demonstration to play (action=list shows them)")
                replay = demos.replay(name, robot, mode)
                if wait:
                    run = await run_until(lambda: replay.done, replay.demo.seconds + MARGIN)
                    if not replay.done:
                        replay.cancel()
                    return _json({**replay.status(), "shown_live": run["shown_live"]})
                return _json({"started": replay.status()})
            if action == "export":
                if name is None:
                    raise ValueError("export needs name: the demonstration to write (action=list shows them)")
                return _json(demos.export(name, format, to))
            return _json(demos.listing())
