"""robotforge as an MCP server (stdio) for Claude, Codex and other MCP hosts.

One process hosts the MCP tools and the local viewer, sharing a single
simulation. stdout carries the MCP protocol, so nothing here may print to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import webbrowser
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import uvicorn
from mcp.server.apps import Apps, ResourceCsp
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image

from robotforge.cli import DEFAULT_PORT, find_free_port
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import ModelLoadError, SimSession
from robotforge.viewer.server import ViewerHub, create_app

log = logging.getLogger(__name__)

VIEWER_UI_URI = "ui://robotforge/viewer.html"
HOST = "127.0.0.1"

INSTRUCTIONS = """\
robotforge is a robot playground backed by the MuJoCo physics engine, made for beginners.

Design: new_world creates a world (saved as world.yaml). list_templates shows the robot templates
(arm, quadruped, rover) and their parameters; add_robot builds one from a template or imports an
MJCF/URDF file. add_object places boxes, balls, cylinders, tables and bins. update_robot,
update_object, remove, set_floor and undo edit the world; every edit is validated and the live
viewer updates immediately. Names are lowercase with underscores. Units are meters, kilograms,
radians for joints and degrees for yaw.

Behavior: describe_model lists actuator names (robot actuators are prefixed "<robot>/").
set_controls / simulate make robots move; observe reports state with an image so you can check
your work. open_viewer gives the user a live 3D view; offer it early.

Explain what you are doing in plain language and briefly teach the idea behind it
(joints, actuators, control): the user may be new to robotics.
"""


def _viewer_page(port: int) -> str:
    url = f"http://{HOST}:{port}/"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>robotforge viewer</title>
<style>html,body{{margin:0;height:100%;background:#1a1f26}}iframe{{border:0;width:100%;height:100%;min-height:480px}}</style>
</head><body><iframe src="{url}" title="robotforge viewer" allow="fullscreen"></iframe></body></html>
"""


def build_server(playground: Playground, port: int) -> MCPServer:
    hub = ViewerHub(playground.session)
    playground.viewer_url = f"http://{HOST}:{port}/"
    apps = Apps()
    origins = [f"http://{HOST}:{port}", f"http://localhost:{port}"]
    apps.add_html_resource(
        VIEWER_UI_URI,
        _viewer_page(port),
        title="robotforge viewer",
        csp=ResourceCsp(frame_domains=origins, connect_domains=[*origins, f"ws://{HOST}:{port}"]),
        prefers_border=False,
    )

    # Apps snapshots its tools when MCPServer is constructed, so register them first.
    @apps.tool(
        resource_uri=VIEWER_UI_URI,
        description=(
            "Open the live 3D viewer. Users can play/pause, move actuator sliders and drag the robot. "
            "Returns the viewer URL; hosts that support MCP Apps show it inline."
        ),
    )
    def open_viewer(launch_browser: bool = True) -> str:
        if launch_browser and not os.environ.get("ROBOTFORGE_NO_BROWSER"):
            webbrowser.open(playground.viewer_url)
        return _json({"viewer_url": playground.viewer_url, "model": playground.session.source})

    @contextlib.asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        config = uvicorn.Config(
            create_app(hub), host=HOST, port=port, log_level="warning", access_log=False, log_config=None
        )
        viewer = uvicorn.Server(config)
        task = asyncio.create_task(viewer.serve())
        log.info("viewer at %s", playground.viewer_url)
        try:
            yield
        finally:
            viewer.should_exit = True
            with contextlib.suppress(Exception):
                await task
            playground.renderer.close()

    server = MCPServer(
        "robotforge",
        title="robotforge robot playground",
        instructions=INSTRUCTIONS,
        extensions=[apps],
        lifespan=lifespan,
    )

    @server.tool()
    def list_templates() -> str:
        """List robot templates (arm, quadruped, rover) with their parameters, defaults and limits,
        and the parts catalog (servos and motors) that templates can use."""
        from robotforge.design import components, templates

        parts = {
            n: {"kind": a.kind, "stall_torque_nm": a.stall_torque, "speed_rad_s": a.no_load_speed, "note": a.note}
            for n, a in components.ACTUATORS.items()
        }
        return _json({"templates": templates.catalog(), "parts": parts, "colors": list(components.COLORS)})

    @server.tool()
    def new_world(name: str, floor: Literal["checker", "grass", "wood", "concrete"] = "checker") -> str:
        """Create a new world (or reopen it if one with this name exists) and show it in the viewer."""
        with _tool_errors():
            return _json(playground.new_world(name, floor))

    @server.tool()
    def open_world(name_or_path: str) -> str:
        """Open an existing world by name or by path to its folder or world.yaml."""
        with _tool_errors():
            return _json(playground.open_world(name_or_path))

    @server.tool()
    def add_robot(
        name: str,
        template: str | None = None,
        file: str | None = None,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float = 0.0,
    ) -> str:
        """Add a robot to the open world, from a template (see list_templates) or an MJCF/URDF file.

        params override template defaults, e.g. {"leg_length": 0.3, "color": "orange"}.
        pos is [x, y] in meters (height is automatic for templates); yaw is in degrees.
        """
        with _tool_errors():
            return _json(playground.add_robot(name, template, file, params, pos, yaw))

    @server.tool()
    def update_robot(
        name: str, params: dict[str, Any] | None = None, pos: list[float] | None = None, yaw: float | None = None
    ) -> str:
        """Change a robot's template parameters (merged into existing ones), position or heading."""
        with _tool_errors():
            return _json(playground.update_robot(name, params, pos, yaw))

    @server.tool()
    def add_object(
        name: str,
        shape: Literal["box", "sphere", "cylinder", "table", "bin"] = "box",
        size: list[float] | None = None,
        pos: list[float] | None = None,
        color: str | list[float] | None = None,
        mass: float | None = None,
        fixed: bool = False,
    ) -> str:
        """Add an object. size in meters: box/table/bin [length, width, height], sphere [diameter],
        cylinder [diameter, height]. pos is [x, y] (resting on the floor) or [x, y, z].
        fixed objects never move; a bin is an open-top container."""
        with _tool_errors():
            return _json(
                playground.add_object(name, shape=shape, size=size, pos=pos, color=color, mass=mass, fixed=fixed)
            )

    @server.tool()
    def update_object(
        name: str,
        size: list[float] | None = None,
        pos: list[float] | None = None,
        color: str | list[float] | None = None,
        mass: float | None = None,
        fixed: bool | None = None,
    ) -> str:
        """Change an object's size, position, color, mass or fixed flag."""
        with _tool_errors():
            return _json(playground.update_object(name, size=size, pos=pos, color=color, mass=mass, fixed=fixed))

    @server.tool()
    def remove(name: str) -> str:
        """Remove a robot or object from the world."""
        with _tool_errors():
            return _json(playground.remove(name))

    @server.tool()
    def set_floor(style: Literal["checker", "grass", "wood", "concrete"]) -> str:
        """Change the floor look and feel."""
        with _tool_errors():
            return _json(playground.set_floor(style))

    @server.tool()
    def undo() -> str:
        """Undo the last world edit."""
        with _tool_errors():
            return _json(playground.undo())

    @server.tool()
    def show_world() -> str:
        """Show the world.yaml of the open world (the editable description of everything in it)."""
        with _tool_errors():
            return _json(playground.show_world())

    @server.tool()
    def open_model(path: str) -> str:
        """Load a robot or scene from an MJCF (.xml) or URDF (.urdf) file, replacing the current one.

        Returns the model summary (actuators, joints, free bodies, sensors, views).
        """
        with _tool_errors():
            return _json(playground.open_model(path))

    @server.tool()
    def describe_model() -> str:
        """Summarize the loaded model: actuator names and ranges, joints, free bodies, sensors, views."""
        return _json(playground.describe())

    @server.tool()
    def set_controls(controls: dict[str, float]) -> str:
        """Set actuator targets by name, e.g. {"shoulder": 0.8}. Values are clamped to each actuator's range.

        Controls persist until changed. Use simulate to let time pass.
        """
        with _tool_errors():
            return _json(playground.set_controls(controls))

    @server.tool()
    def simulate(seconds: float, controls: dict[str, float] | None = None) -> str:
        """Advance physics by `seconds` (max 60) as fast as possible, optionally setting controls first.

        Returns the new time and which free bodies moved (from/to positions in meters).
        """
        with _tool_errors():
            return _json(playground.simulate(seconds, controls))

    @server.tool()
    def sim_control(action: Literal["play", "pause", "reset"]) -> str:
        """Play or pause real-time simulation in the viewer, or reset the model to its initial state."""
        with _tool_errors():
            return _json(playground.control(action))

    @server.tool()
    def observe(view: str = "default", width: int = 640, height: int = 480, include_image: bool = True) -> list:
        """Report the current state (joint angles, controls, free body positions, sensors, contacts) and,
        by default, a rendered image.

        `view` is a preset (default, front, back, left, right, top) or a camera name from describe_model.
        """
        with _tool_errors():
            content: list = [_json(playground.state())]
            if include_image:
                content.append(Image(data=playground.snapshot(view, width, height), format="png"))
            return content

    return server


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn anticipated failures into ToolError so the agent sees the actual message."""
    try:
        yield
    except (PlaygroundError, ModelLoadError, IndexError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


def run(model: Path | None = None, port: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    session = SimSession.from_path(model) if model else None
    playground = Playground(session=session) if session else Playground.with_demo()
    chosen = port if port is not None else find_free_port(DEFAULT_PORT, HOST)
    build_server(playground, chosen).run("stdio")
