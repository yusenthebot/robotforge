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
import re
import time
import webbrowser
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from mcp.server.apps import Apps, ResourceCsp
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

from robotforge.behaviors.manager import BehaviorError
from robotforge.cli import DEFAULT_PORT, find_free_port
from robotforge.playground import Playground, PlaygroundError
from robotforge.script_runner import ScriptRunner
from robotforge.sim.render import RenderUnavailable
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

Behavior: move walks a quadruped or drives a rover; pick_and_place, move_arm and gripper work arms;
robots lists robots with their positions. For anything custom, set_controls / simulate drive actuators
directly (names from describe_model, prefixed "<robot>/"). observe reports state with an image so you
can check your work. open_viewer gives the user a live 3D view (they can also drive with WASD);
offer it early, and when it is open, actions play out in real time for the user to watch.

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
    playground.attach_viewer(hub)
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

    async def run_until(done, max_seconds: float) -> dict[str, Any]:
        """Let time pass until `done()`: in real time if someone is watching, else as fast as possible."""
        session = playground.session
        start = session.data.time
        watched = bool(hub.clients) and not session.paused
        if watched:
            deadline = time.monotonic() + max_seconds * 3 + 5  # guard against a slow machine
            while not done() and session.data.time - start < max_seconds and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        else:
            playground.run_fast(done, max_seconds)
        return {"sim_seconds": round(float(session.data.time - start), 2), "shown_live": watched}

    @server.tool()
    async def move(robot: str, speed: float = 0.2, turn: float = 0.0, seconds: float | None = 3.0) -> str:
        """Walk a quadruped or drive a rover. speed in m/s (negative = backwards), turn in rad/s (positive =
        left). With `seconds`, moves for that long then stops and reports how far it went; with seconds=null
        it keeps moving until `stop`. If the viewer is open the user watches it happen in real time."""
        with _tool_errors():
            before = playground.behaviors.pose(robot)
            status = playground.behaviors.move(robot, speed, turn)
            if seconds is None:
                return _json({"started": status, "pose": before})
            if not 0 < seconds <= 60:
                raise PlaygroundError("seconds must be in (0, 60]")
            run = await run_until(lambda: False, seconds)
            playground.behaviors.stop(robot)
            after = playground.behaviors.pose(robot)
            moved = float(np.linalg.norm(np.subtract(after["position"][:2], before["position"][:2])))
            turned = (after["heading_deg"] - before["heading_deg"] + 180) % 360 - 180
            return _json(
                {
                    "from": before,
                    "to": after,
                    "distance_m": round(moved, 3),
                    "turned_deg": round(turned, 1),
                    **run,
                    "behavior": status,
                }
            )

    @server.tool()
    def stop(robot: str) -> str:
        """Stop a robot's current behavior (a walking robot settles into a stand)."""
        with _tool_errors():
            return _json(playground.behaviors.stop(robot))

    @server.tool()
    async def move_arm(robot: str, position: list[float], gripper: Literal["open", "close"] | None = None) -> str:
        """Move an arm's gripper to [x, y, z] (meters, world frame) pointing down, optionally opening or
        closing the gripper on the way. Fails with an explanation if the point is out of reach."""
        with _tool_errors():
            arm = playground.behaviors.move_arm(robot, position, gripper)
            run = await run_until(lambda: arm.done, 20.0)
            tcp = playground.session.data.site_xpos[arm.site]
            return _json({"tool_position": [round(float(v), 3) for v in tcp], **run})

    @server.tool()
    async def gripper(robot: str, state: Literal["open", "close"]) -> str:
        """Open or close an arm's gripper."""
        with _tool_errors():
            arm = playground.behaviors.set_gripper(robot, state)
            run = await run_until(lambda: arm.done, 5.0)
            return _json({"gripper": state, **run})

    @server.tool()
    async def pick_and_place(robot: str, object: str, target: str | list[float]) -> str:
        """Have an arm pick up a small object and put it on/in a target: another object's name (a bin, table
        or box) or a floor position [x, y]. Plans the whole motion, checks reach first, runs it and reports
        whether the object really ended up there."""
        with _tool_errors():
            arm, info = playground.behaviors.pick_and_place(robot, object, target)
            run = await run_until(lambda: arm.done, 40.0)
            settle = await run_until(lambda: False, 0.5)
            result = playground.behaviors.placed(object, info["region"])
            return _json(
                {
                    **result,
                    "object": object,
                    "target": target,
                    "sim_seconds": run["sim_seconds"] + settle["sim_seconds"],
                    "shown_live": run["shown_live"],
                }
            )

    @server.tool()
    def write_script(name: str, code: str) -> str:
        """Save a Python robot script as behaviors/<name>.py in the open world. Scripts use:

            from robotforge.script import connect
            sim = connect()
            sim.time; sim.step(seconds)                      # time passes only when you step
            sim.position(name) -> [x, y, z]; sim.heading(robot) -> degrees
            sim.joints(robot) -> {joint: value}; sim.sensors(robot) -> {sensor: value}
            sim.set_controls({"arm/shoulder": 0.5})          # raw actuator targets
            sim.move(robot, speed, turn); sim.stop(robot)      # walk / drive (keeps going until stop)
            sim.move_arm(robot, [x, y, z], gripper="open"|"close"|None)   # waits until arrived
            sim.gripper(robot, "open"|"close"); sim.pick_and_place(robot, obj, target) -> bool
            print(...)                                         # output is returned by run_script

        Write clear, commented code: the user may read it to learn. Then call run_script."""
        with _tool_errors():
            store = playground._require_store()
            path = _script_path(store.dir, name)
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as f:  # LF on every OS
                f.write(code)
            return _json({"saved": str(path), "lines": code.count("\n") + 1})

    @server.tool()
    def read_script(name: str) -> str:
        """Show a saved script from the open world's behaviors folder."""
        with _tool_errors():
            path = _script_path(playground._require_store().dir, name)
            if not path.exists():
                raise PlaygroundError(f"no script named {name!r}")
            return path.read_text(encoding="utf-8")

    @server.tool(annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False))
    async def run_script(name: str, max_sim_seconds: float = 60.0) -> str:
        """Run behaviors/<name>.py against the live world. Runs Python on this computer (not a sandbox),
        in lockstep with the simulation, shown live if the viewer is open. Returns the script output,
        exit code and any traceback."""
        with _tool_errors():
            path = _script_path(playground._require_store().dir, name)
            if not path.exists():
                raise PlaygroundError(f"no script named {name!r}; write it with write_script first")
            if not 0 < max_sim_seconds <= 300:
                raise PlaygroundError("max_sim_seconds must be in (0, 300]")
            runner = ScriptRunner(playground, lambda: bool(hub.clients) and not playground.session.paused)
            return _json(await runner.run(path, max_sim_seconds=max_sim_seconds))

    @server.tool()
    def robots() -> str:
        """List robots in the world with their kind (arm, quadruped, rover), where they are, which way they
        face, whether they are upright, and any behavior running on them."""
        with _tool_errors():
            out = []
            for info in playground.behaviors.robots():
                with contextlib.suppress(BehaviorError):
                    info.update(playground.behaviors.pose(info["name"]))
                out.append(info)
            return _json({"robots": out, "behaviors": playground.behaviors.status()})

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
        by default, a rendered image (on machines without OpenGL the state comes back with a note instead).

        `view` is a preset (default, front, back, left, right, top) or a camera name from describe_model.
        """
        with _tool_errors():
            state = playground.state()
            image = None
            if include_image:
                try:
                    image = playground.snapshot(view, width, height)
                except RenderUnavailable as exc:
                    # Headless machines (CI, servers, VMs) have no OpenGL: keep the state, say why there is no image.
                    state["note"] = (
                        f"No image. {exc} Everything else in this state is accurate, "
                        "and open_viewer shows the live 3D view in a browser."
                    )
            content: list = [_json(state)]
            if image is not None:
                content.append(Image(data=image, format="png"))
            return content

    return server


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn anticipated failures into ToolError so the agent sees the actual message."""
    try:
        yield
    except (PlaygroundError, BehaviorError, ModelLoadError, IndexError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


SCRIPT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _script_path(world_dir: Path, name: str) -> Path:
    stem = name[:-3] if name.endswith(".py") else name
    if not SCRIPT_NAME.match(stem):
        raise PlaygroundError("script names use lowercase letters, digits and _, e.g. pick_cube")
    return world_dir / "behaviors" / f"{stem}.py"


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


def run(model: Path | None = None, port: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    session = SimSession.from_path(model) if model else None
    playground = Playground(session=session) if session else Playground.with_demo()
    chosen = port if port is not None else find_free_port(DEFAULT_PORT, HOST)
    build_server(playground, chosen).run("stdio")
