"""robotforge as an MCP server (stdio) for Claude, Codex and other MCP hosts.

One process hosts the MCP tools and the local viewer, sharing a single
simulation. stdout carries the MCP protocol, so nothing here may print to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import uvicorn
from mcp.server.apps import Apps, ResourceCsp
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

from robotforge import lessons, tool_input, wording
from robotforge.behaviors.manager import BehaviorError
from robotforge.bridge import mcp_tools as policy_tools
from robotforge.cli import DEFAULT_PORT, find_free_port, open_in_browser
from robotforge.control import mcp_tools as control_tools
from robotforge.design import mcp_tools as scene_tools
from robotforge.design.builder import edit as build_edit
from robotforge.design.builder import mcp_tools as builder_tools
from robotforge.design.terrain import StyleName
from robotforge.design.world import ShapeName
from robotforge.playground import MAX_SIM_SECONDS, Playground, PlaygroundError
from robotforge.script_runner import ScriptRunner
from robotforge.sim.render import RenderUnavailable
from robotforge.sim.session import ModelLoadError, SimSession
from robotforge.viewer.server import ViewerHub, create_app

log = logging.getLogger(__name__)

VIEWER_UI_URI = "ui://robotforge/viewer.html"
FAST_WALL_LIMIT = 60.0  # computer seconds one unwatched run may take
HOST = "127.0.0.1"

INSTRUCTIONS = wording.unwrap(
    """\
robotforge is a beginner-friendly robot playground on the MuJoCo physics engine.

Design: new_world creates a world (saved as world.yaml). add_robot builds a robot from a template (arm,
quadruped, humanoid, rover, cartpole, gripper; list_templates), takes a real one from the library
(list_library: Unitree Go2, Franka Panda, ...) or imports a file (URDF, xacro, MJCF, or a mesh such as
.stl/.obj/.dae/.glb; the import doctor repairs URDFs and meshes and reports in the result's "doctor" field).
add_robot(build=...) builds any robot from parts (hexapod, snake, self-balancing robot, arm on a rover):
read builder_guide, then edit with add_part / update_part / remove_part. add_object places boxes, balls,
cylinders, tables, bins, furniture and architecture (walls, shelves, counters, pallets, trees, benches,
rocks, flat ground markers). update_robot, update_object, remove, set_floor and undo edit the world; every
edit is validated and the live viewer updates at once. Robots and objects an edit does not change stay where
they are (new terrain or a scene starts them fresh). Names are lowercase with underscores; other capitals
and spaces are accepted ("Red Cube" is saved as red_cube, "the dog" finds dog). Units: meters, kilograms,
radians for joints, degrees for yaw (the builder takes degrees for joints too).

Scenes and terrain: load_scene fills the world with a ready-made scene (desk_lab, kitchen, warehouse, park,
maze, obstacle_course; list_scenes describes them). set_terrain and add_terrain shape the ground (hills,
rough ground, stairs, ramps, stepping stones, curbs, platforms); robots and objects stand on the ground
under them. To build a scene from a sentence, combine load_scene, add_terrain and add_object, then check it
with observe(view="overview").

Behavior: move walks a quadruped, hexapod or humanoid, drives a rover or two-wheeled balancer and makes a
snake slither; pose makes a humanoid wave, squat, bow or cheer; pick_and_place, move_arm and gripper work
arms (an arm mounted on a robot is "robot/part", e.g. "rover/arm"); robots lists robots with kind, arms and
positions. For anything custom, set_controls / simulate drive actuators directly (names from describe_model,
prefixed "<robot>/"). observe reports state with an image so you can check your work. open_viewer gives the
user a live 3D view (they can also drive with WASD): offer it early; while it is open, actions play out in
real time.

Algorithms: one robot and task can be solved by different control methods (IK, PD, LQR, sampling MPC, ...).
list_algorithms explains them; run_controller drives a robot with one (live in the viewer);
compare_controllers runs several on the same task and returns a table to discuss. train_policy teaches a
robot a task by reinforcement learning (PPO, SAC) in the background; follow it with training_status, then
run or compare it as method "policy:<name>".

Learning path: lessons lists nine short beginner lessons (joints, driving, gaits, inverse kinematics,
sensors, control, scripts, training, importing a URDF). When the user asks for one ("teach me lesson 3",
"第三课"), call lessons(number) and follow its plan. The viewer's Learn button (学习) also has hands-on lessons
(PID, inverse kinematics, motion planning, sensors, gaits, balance, reinforcement learning, face-offs
against the real PiPER, Franka Panda and Unitree G1, star challenges): suggest them to beginners who want to
understand robots (open_viewer, then press Learn).

Learned walkers: list_walkers shows neural-network walking policies (a Go2 walk ships with robotforge);
move(..., walker="rl") uses one instead of the built-in gait. Any robot that walks on jointed legs (built
from parts, imported from a URDF, even one with a built-in gait) can learn a walk of its own in the viewer's
RL II lesson ("Your robot: <name>", then "Keep this walk for my robot"); robots then lists its learned_walk
and walker="rhythm" uses it (forward at its own pace; turning on the spot once its turns are learned). If
the user wants such a robot to walk and robots or add_robot show a walk_hint, say first that it can learn a
walk by itself in about ten seconds and keep it, which teaches more and walks better than a hand-written
gait; offer to open the viewer, and write a gait script only if they would rather not. UniLab, an external
RL framework, trains new ones: unilab_check, unilab_train, unilab_job.

Explain what you are doing in plain language and briefly teach the idea behind it (joints, actuators,
control): the user may be new to robotics. Check results (numbers, observe) before saying something worked.
"""
)


class _Server(MCPServer):
    """An MCPServer that reads a call forgivingly and answers a failed one briefly (see tool_input)."""

    async def call_tool(self, name: str, arguments: dict[str, Any], context: Context | None = None):
        names = [t.name for t in self._tool_manager.list_tools()]
        name = name if name in names else tool_input.tool_name(name, names) or name
        tool = self._tool_manager.get_tool(name)
        if tool is None:
            raise ToolError(tool_input.unknown_tool(name, names))
        arguments = tool_input.tidy(name, tool.parameters, arguments)
        try:
            return await super().call_tool(name, arguments, context)
        except UnexpectedToolError:
            raise
        except ToolError as exc:
            raise ToolError(tool_input.error_text(name, tool.parameters, exc)) from exc


def _viewer_page(port: int) -> str:
    url = f"http://{HOST}:{port}/"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>robotforge viewer</title>
<style>html,body{{margin:0;height:100%;background:#1a1f26}}iframe{{border:0;width:100%;height:100%;min-height:480px}}</style>
</head><body><iframe src="{url}" title="robotforge viewer" allow="fullscreen"></iframe></body></html>
"""


def _plain_tools(server: MCPServer) -> None:
    """Every tool registered on `server` from here on returns its text only and is described without the
    docstring's indentation and line breaks. Tools return JSON text; declaring an output schema made the SDK
    send each result a second time as structured content, and the indentation was 8 spaces on every line of 50
    descriptions, all of which an agent's model reads again on every turn."""
    register = server.tool

    def tool(*args: Any, **kwargs: Any):
        kwargs.setdefault("structured_output", False)
        decorate = register(*args, **kwargs)

        def apply(fn):
            if fn.__doc__ and "description" not in kwargs:
                fn.__doc__ = wording.unwrap(inspect.cleandoc(fn.__doc__))
            return decorate(fn)

        return apply

    server.tool = tool  # type: ignore[method-assign]


def _lean_schemas(server: MCPServer) -> None:
    """The generated argument schemas carry words that tell an agent nothing, and it reads them on every turn: a
    "title" that repeats each parameter's name ("Name", "Speed"), `"default": null` on every optional parameter
    (leaving it out means the same) and `"additionalProperties": true` (the JSON Schema default). Arguments are
    still checked against the function's own model, so only the advertised copy changes. (A parameter that is
    itself called title, default or additionalProperties is a dict: kept.)"""

    def lean(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: lean(v)
                for k, v in node.items()
                if not (k == "title" and isinstance(v, str))
                and not (k == "default" and v is None)
                and not (k == "additionalProperties" and v is True)
            }
        if isinstance(node, list):
            return [lean(v) for v in node]
        return node

    for tool in server._tool_manager.list_tools():  # noqa: SLF001 (the SDK has no public setter)
        tool.parameters = lean(tool.parameters)


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
            "Open the live 3D viewer (play/pause, actuator sliders, drag the robot). "
            "Returns the viewer URL; hosts with MCP Apps show it inline."
        ),
    )
    def open_viewer(launch_browser: bool = True) -> str:
        if launch_browser and not os.environ.get("ROBOTFORGE_NO_BROWSER"):
            open_in_browser(playground.viewer_url)
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

    server = _Server(
        "robotforge",
        title="robotforge robot playground",
        instructions=INSTRUCTIONS,
        extensions=[apps],
        lifespan=lifespan,
    )
    _plain_tools(server)

    @server.tool()
    def list_templates() -> str:
        """List robot templates (arm, quadruped, humanoid, rover, cartpole, gripper) with parameters, defaults and
        limits, and the parts catalog (servos, motors, linear actuators). For other robots build from parts
        (builder_guide)."""
        from robotforge.design import components, templates

        parts = {
            n: {"kind": a.kind, "stall_torque_nm": a.stall_torque, "speed_rad_s": a.no_load_speed, "note": a.note}
            for n, a in components.ACTUATORS.items()
        }
        return _json({"templates": templates.catalog(), "parts": parts, "colors": list(components.COLORS)})

    @server.tool()
    def list_library() -> str:
        """List real robots that add_robot(library=...) loads: Unitree Go2 and other robot dogs (move and the keyboard
        walk them), arms like the Franka Panda and AgileX PiPER (move_arm and pick_and_place work) and the Unitree
        G1 humanoid (walks with a learned policy). The first add_robot of each downloads it from MuJoCo Menagerie
        (size in download_mb, usually 5-30 s) and caches it."""
        from robotforge.design import library

        with _tool_errors():
            return _json({"robots": library.catalog()})

    @server.tool()
    def new_world(name: str, floor: StyleName = "checker") -> str:
        """Create a world (or reopen it if the name exists) and show it in the viewer."""
        with _tool_errors():
            return _json(playground.new_world(name, floor))

    @server.tool()
    def open_world(name_or_path: str) -> str:
        """Open an existing world by name or by path to its folder or world.yaml."""
        with _tool_errors():
            return _json(playground.open_world(name_or_path))

    @server.tool()
    async def add_robot(
        ctx: Context,
        name: str,
        template: str | None = None,
        file: str | None = None,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float = 0.0,
        library: str | None = None,
        build: dict[str, Any] | None = None,
    ) -> str:
        """Add a robot: from a template (list_templates), the robot library (list_library, e.g. library="go2"), a file,
        or built from parts (build, see builder_guide).

        build is a base plus parts, e.g. a two-wheeled balancer: {"base": {"size": [0.07, 0.14, 0.22]}, "motor":
        "jga25", "parts": [{"name": "left_wheel", "at": [0, 0.095, -0.11], "shape": "wheel", "size": [0.12, 0.025],
        "mirror": true}]}.

        file is an MJCF (.xml), URDF, xacro or mesh (.stl .obj .dae .glb ...). URDFs and meshes go through the
        import doctor, which repairs mesh paths, formats, mass and inertia, adds motors and picks a fixed or
        floating base; the result's "doctor" field lists what it fixed and could not. For files, params are import
        options: base (auto, fixed, floating), decompose (true splits concave collision meshes into convex pieces),
        respect_effort_limits (by default torque limits too weak to hold the robot up are relaxed and reported; true
        keeps the file's), scale and mass (meshes), xacro_args.

        A library robot downloads on first use (4-55 MB, usually 5-30 s, with progress updates); if the call timed
        out, call again: the download continues and a repeated identical call reports the robot. For templates,
        params override defaults, e.g. {"leg_length": 0.3, "color": "orange"}. pos is [x, y] in meters (height is
        automatic); yaw is in degrees."""
        with _tool_errors():
            for wanted in [library, *build_edit.libraries_in(build)]:
                if wanted is not None:
                    await _download_with_progress(wanted, ctx)
            # A cold import of a URDF or mesh file converts meshes and simulates the robot for up to a
            # minute: off the event loop, so the viewer and other requests keep running meanwhile.
            added = await asyncio.to_thread(
                playground.add_robot, name, template, file, params, pos, yaw, library=library, build=build
            )
            return _json(added)

    @server.tool()
    def update_robot(
        name: str,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float | None = None,
        build: dict[str, Any] | None = None,
    ) -> str:
        """Change a robot's template params (merged), position or heading. For a robot built from parts, build replaces
        its whole description (add_part / update_part change one part)."""
        with _tool_errors():
            return _json(playground.update_robot(name, params, pos, yaw, build))

    @server.tool()
    def add_object(
        name: str,
        shape: ShapeName = "box",
        size: list[float] | None = None,
        pos: list[float] | None = None,
        color: str | list[float] | None = None,
        mass: float | None = None,
        fixed: bool | None = None,
        yaw: float = 0.0,
    ) -> str:
        """Add an object. size in meters: box/table/bin [length, width, height]; sphere [diameter]; cylinder, pillar,
        rock [diameter, height]; wall [length, thickness, height]; shelf, counter, bench [length, depth, height];
        door_frame [width, depth, height]; pallet [length, width, height]; tree [crown diameter, height]; marker
        [diameter] or [length, width]. Every shape has a default size and color; color is a name, a hex code like
        "#9c6b3f" or [r, g, b]. pos is [x, y] (rests on the ground or terrain there) or [x, y, z]; yaw is in
        degrees. fixed objects never move (walls, shelves, counters, trees, benches and rocks are fixed unless
        fixed=false). A bin is an open-top container; a marker is a flat mark nothing collides with (goals, lanes).
        Ramps and stairs are terrain: see add_terrain."""
        with _tool_errors():
            fields = {"size": size, "pos": pos, "color": color, "mass": mass, "fixed": fixed}
            return _json(playground.add_object(name, shape=shape, yaw=yaw, **fields))

    @server.tool()
    def update_object(
        name: str,
        size: list[float] | None = None,
        pos: list[float] | None = None,
        color: str | list[float] | None = None,
        mass: float | None = None,
        fixed: bool | None = None,
        yaw: float | None = None,
    ) -> str:
        """Change an object's size, position, heading (yaw, degrees), color, mass or fixed flag."""
        with _tool_errors():
            fields = {"size": size, "pos": pos, "color": color, "mass": mass, "fixed": fixed, "yaw": yaw}
            return _json(playground.update_object(name, **fields))

    @server.tool()
    def remove(name: str) -> str:
        """Remove a robot, object or terrain patch from the world."""
        with _tool_errors():
            return _json(playground.remove(name))

    @server.tool()
    def set_floor(style: StyleName) -> str:
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
        result: dict[str, Any] = {}
        if not watched:
            # Fast, but in short slices that yield to the event loop (viewer, other requests), and never
            # longer than FAST_WALL_LIMIT of computer time, however slow a controller is.
            deadline = time.monotonic() + FAST_WALL_LIMIT
            while not done() and session.data.time - start < max_seconds - 1e-9:
                if time.monotonic() > deadline:
                    result["stopped_early"] = (
                        f"stopped after {FAST_WALL_LIMIT:.0f} s of computer time; the controller is too slow "
                        "for this machine (for MPC use fewer samples or a shorter horizon)"
                    )
                    break
                playground.run_fast(done, min(0.1, max_seconds - (session.data.time - start)))
                await asyncio.sleep(0)
        return {"sim_seconds": round(float(session.data.time - start), 2), "shown_live": watched, **result}

    @server.tool()
    async def move(
        robot: str, speed: float = 0.2, turn: float = 0.0, seconds: float | None = 3.0, walker: str | None = None
    ) -> str:
        """Walk a quadruped, hexapod or humanoid, drive a rover or two-wheeled balancer, make a snake slither, or walk
        a robot with a walk it learned itself (RL II lesson). speed in m/s (negative = backwards), turn in rad/s
        (positive = left); more than the robot can do is clamped, and the result says so. Each robot's max_speed is
        in `robots` (humanoids: about 0.1 m/s, slow to start and stop). With seconds it moves that long, stops and
        reports the distance; seconds=null keeps moving until `stop`. The user watches in real time if the viewer is
        open. walker (legged robots): "gait" = the built-in controller, "rl" = this robot's learned policy
        (list_walkers), or a policy name; omitted keeps the current one. "rhythm" (a robot with learned_walk in
        `robots`) = the walk it learned in the viewer, forward at its own pace, turning on the spot. Keyboard
        driving uses the same walker."""
        with _tool_errors():
            robot = playground.behaviors.resolve(robot)
            if seconds is not None and not 0 < seconds <= MAX_SIM_SECONDS:  # before starting, or the robot keeps going
                raise PlaygroundError(
                    f"{wording.seconds_range('seconds', seconds, MAX_SIM_SECONDS)}; "
                    "seconds=null keeps moving until stop"
                )
            before = playground.behaviors.pose(robot)
            status = playground.behaviors.move(robot, speed, turn, walker)
            if seconds is None:
                return _json({"started": status, "pose": before})
            # The heading is followed as it goes: the difference of the two ends wraps at 180, and a half turn
            # to the left read as -177.6 degrees, a turn to the right.
            heading = {"last": before["heading_deg"], "turned": 0.0}

            def track() -> bool:
                now = playground.behaviors.pose(robot)["heading_deg"]
                heading["turned"] += (now - heading["last"] + 180) % 360 - 180
                heading["last"] = now
                return False

            run = await run_until(track, seconds)
            playground.behaviors.stop(robot)
            settled = playground.behaviors.settled(robot)
            if settled is not None:  # a humanoid finishes its step first; measure where it really stops
                extra = await run_until(lambda: track() or settled(), 3.0)
                run["settle_seconds"] = extra["sim_seconds"]
                run["sim_seconds"] = round(run["sim_seconds"] + extra["sim_seconds"], 2)
            track()
            after = playground.behaviors.pose(robot)
            moved = float(np.linalg.norm(np.subtract(after["position"][:2], before["position"][:2])))
            turned = heading["turned"]
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

    control_tools.register(server, playground, run_until)
    scene_tools.register(server, playground)
    builder_tools.register(server, playground)
    lessons.register(server)
    policy_tools.register(server)

    @server.tool()
    def stop(robot: str) -> str:
        """Stop a robot's current behavior (a walking robot settles into a stand)."""
        with _tool_errors():
            return _json(playground.behaviors.stop(playground.behaviors.resolve(robot)))

    @server.tool()
    async def pose(robot: str, name: Literal["wave", "squat", "arms_up", "bow", "look_around", "stand"]) -> str:
        """Make a humanoid strike a pose: wave, squat (deep knee bend), arms_up (cheer), bow, look_around (turns its
        head) or stand (neutral). A walking humanoid stops first; it keeps its balance. Waits until the pose is done
        and reports whether it stayed up."""
        with _tool_errors():
            robot = playground.behaviors.resolve(robot)
            ctrl = playground.behaviors.strike_pose(robot, name)
            run = await run_until(lambda: ctrl.done, 20.0)
            after = playground.behaviors.pose(robot)
            return _json({"robot": robot, "pose": name, "upright": after["upright"], **run})

    @server.tool()
    async def move_arm(robot: str, position: list[float], gripper: Literal["open", "close"] | None = None) -> str:
        """Move an arm's gripper to [x, y, z] (meters, world frame) pointing down, optionally opening or closing the
        gripper on the way. Fails with the distance and reach if the point cannot be reached. robot is an arm, or a
        robot carrying one ("rover", or "rover/arm" for the mounted arm)."""
        with _tool_errors():
            arm = playground.behaviors.move_arm(playground.behaviors.resolve(robot), position, gripper)
            run = await run_until(lambda: arm.done, 20.0)
            tcp = playground.session.data.site_xpos[arm.site]
            return _json({"tool_position": [round(float(v), 3) for v in tcp], **run})

    @server.tool()
    async def gripper(robot: str, state: Literal["open", "close"]) -> str:
        """Open or close an arm's gripper."""
        with _tool_errors():
            arm = playground.behaviors.set_gripper(playground.behaviors.resolve(robot), state)
            run = await run_until(lambda: arm.done, 5.0)
            return _json({"gripper": state, **run})

    @server.tool()
    async def pick_and_place(robot: str, object: str, target: str | list[float]) -> str:
        """Have an arm pick up a small object and put it on/in a target: another object's name (bin, table, box) or a
        floor position [x, y]. Plans the motion, checks reach first, runs it and reports whether the object ended up
        there."""
        with _tool_errors():
            arm, info = playground.behaviors.pick_and_place(playground.behaviors.resolve(robot), object, target)
            object = info["object"]  # as the world spells it
            run = await run_until(lambda: arm.done, 40.0)
            settle = await run_until(lambda: False, 0.5)
            result = playground.behaviors.placed(object, info["region"])
            return _json(
                {
                    **result,
                    "object": object,
                    "target": target,
                    "sim_seconds": round(run["sim_seconds"] + settle["sim_seconds"], 2),
                    "shown_live": run["shown_live"],
                }
            )

    @server.tool()
    def write_script(name: str, code: str) -> str:
        """Save a Python robot script as behaviors/<name>.py in the open world. Scripts use:

            from robotforge.script import connect
            sim = connect()
            sim.time; sim.step(seconds)  # time passes only when you step
            sim.position(name) -> [x, y, z]; sim.heading(robot) -> degrees
            sim.joints(robot) -> {joint: value}; sim.sensors(robot) -> {sensor: value}
            sim.set_controls({"arm/shoulder": 0.5})  # raw actuator targets
            sim.move(robot, speed, turn); sim.stop(robot)  # walk / drive (keeps going until stop)
            sim.pose(robot, "wave") -> bool  # humanoid poses; waits, True if upright
            sim.move_arm(robot, [x, y, z], gripper="open"|"close"|None)  # waits until arrived
            sim.gripper(robot, "open"|"close"); sim.pick_and_place(robot, obj, target) -> bool
            sim.run_policy(robot, policy_name)  # a trained policy drives it until stop
            print(...)  # output is returned by run_script

        Write clear, commented code: the user may read it to learn. Code that does not parse is refused. Then call
        run_script."""
        with _tool_errors():
            store = playground._require_store()
            path = _script_path(store.dir, name)
            code, fenced = _unfence(code)
            try:
                compile(code, path.name, "exec")  # parses only: nothing runs
            except (SyntaxError, ValueError) as exc:
                where = f"line {exc.lineno}: " if isinstance(exc, SyntaxError) and exc.lineno else ""
                raise PlaygroundError(
                    f"the code has a syntax error ({where}{getattr(exc, 'msg', exc)}); nothing was saved. "
                    "Fix it and send the whole script again"
                ) from None
            path.parent.mkdir(exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as f:  # LF on every OS
                f.write(code)
            saved = {"saved": str(path), "lines": code.count("\n") + 1}
            return _json({**saved, "note": "the ``` fence around the code was removed"} if fenced else saved)

    @server.tool()
    def read_script(name: str) -> str:
        """Show a saved script from the open world's behaviors folder."""
        with _tool_errors():
            path = _script_path(playground._require_store().dir, name)
            if not path.exists():
                raise PlaygroundError(_no_script(name, path))
            return path.read_text(encoding="utf-8")

    @server.tool(annotations=ToolAnnotations(destructiveHint=True, openWorldHint=False))
    async def run_script(name: str, max_sim_seconds: float = 60.0) -> str:
        """Run behaviors/<name>.py against the live world: Python on this computer (not a sandbox), in lockstep with
        the simulation, shown live if the viewer is open. Returns the output, exit code and any traceback."""
        with _tool_errors():
            path = _script_path(playground._require_store().dir, name)
            if not path.exists():
                raise PlaygroundError(_no_script(name, path))
            if not 0 < max_sim_seconds <= 300:
                raise PlaygroundError(wording.seconds_range("max_sim_seconds", max_sim_seconds, 300))
            runner = ScriptRunner(playground, lambda: bool(hub.clients) and not playground.session.paused)
            return _json(await runner.run(path, max_sim_seconds=max_sim_seconds))

    @server.tool()
    def robots() -> str:
        """List robots with their kind (arm, quadruped, hexapod, humanoid, rover, balancer, snake, other), the arms
        they carry, position, heading, whether upright, and any running behavior."""
        with _tool_errors():
            out = []
            for info in playground.behaviors.robots():
                with contextlib.suppress(BehaviorError):
                    info.update(playground.behaviors.pose(info["name"]))
                out.append(info)
            return _json({"robots": out, "behaviors": playground.behaviors.status()})

    @server.tool()
    def open_model(path: str, options: dict[str, Any] | None = None) -> str:
        """Load a robot or scene from an MJCF (.xml), URDF, xacro or mesh file, replacing the current one. URDFs and
        meshes are repaired by the import doctor (the result's "doctor" field says what was fixed). options
        (URDF/mesh only): base (auto, fixed, floating), decompose, respect_effort_limits, scale, mass, xacro_args.
        Returns the model summary (actuators, joints, free bodies, sensors, views)."""
        with _tool_errors():
            return _json(playground.open_model(path, options))

    @server.tool()
    def describe_model() -> str:
        """Summarize the loaded model: actuator names and ranges, joints, free bodies, sensors, views."""
        return _json(playground.describe())

    @server.tool()
    def set_controls(controls: dict[str, float]) -> str:
        """Set actuator targets by name, e.g. {"shoulder": 0.8} (the robot prefix is optional when only one robot has
        the name). Values are clamped to each actuator's range, with a note. Controls persist until changed;
        simulate lets time pass."""
        with _tool_errors():
            return _json(playground.set_controls(controls))

    @server.tool()
    def simulate(seconds: float, controls: dict[str, float] | None = None) -> str:
        """Advance physics by `seconds` (max 60) as fast as possible, optionally setting controls first. Returns the
        new time and which free bodies moved (from/to positions in meters)."""
        with _tool_errors():
            return _json(playground.simulate(seconds, controls))

    @server.tool()
    def sim_control(action: Literal["play", "pause", "reset"]) -> str:
        """Play or pause real-time simulation in the viewer, or reset the model to its initial state."""
        with _tool_errors():
            return _json(playground.control(action))

    @server.tool()
    def observe(view: str = "default", width: int = 640, height: int = 480, include_image: bool = True) -> list:
        """Report the state (joint angles, controls, free body positions, sensors, contacts) and, by default, a
        rendered image (without OpenGL the state comes back with a note instead). `view` is a preset (default,
        front, back, left, right, top, overview for the whole scene with terrain and furniture) or a camera name
        from describe_model."""
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
                except PlaygroundError:
                    raise  # a wrong view name: the message says which views exist
                except Exception as exc:  # the state is still worth having when drawing fails
                    log.exception("rendering failed")
                    state["note"] = (
                        f"No image: rendering failed ({type(exc).__name__}: {exc}). Everything else in this state "
                        "is accurate, and open_viewer shows the live 3D view in a browser."
                    )
            content: list = [_json(state)]
            if image is not None:
                content.append(Image(data=image, format="png"))
            return content

    _lean_schemas(server)
    return server


PROGRESS_INTERVAL = 1.0  # s between progress notifications during a download
# Downloads in flight, by library robot: a call retried after a client timeout joins the running
# download instead of starting a second one.
_downloads: dict[str, tuple[asyncio.Future, dict[str, Any]]] = {}


async def _download_with_progress(name: str, ctx: Context) -> None:
    """Download a library robot off the event loop (the viewer stays live), reporting progress."""
    from robotforge.design import library

    robot = library.get(name)
    if library.is_downloaded(name):
        return
    running = _downloads.get(name)
    if running is None or running[0].done():
        state: dict[str, Any] = {
            "fraction": 0.0,
            "message": f"downloading {robot.title} ({library.download_mb(name)} MB)",
            "started": time.monotonic(),
        }

        def progress(fraction: float, message: str) -> None:  # called from the download thread
            state.update(fraction=fraction, message=f"{robot.title}: {message}")

        # If the client gives up, the download thread keeps going and fills the cache for the next call.
        running = (asyncio.ensure_future(asyncio.to_thread(library.model_dir, robot, progress)), state)
        _downloads[name] = running
    task, state = running
    while True:
        elapsed = time.monotonic() - state["started"]
        await ctx.report_progress(round(100 * state["fraction"], 1), 100, f"{state['message']} ({elapsed:.0f} s)")
        done, _ = await asyncio.wait({task}, timeout=PROGRESS_INTERVAL)
        if done:
            break
    task.result()  # re-raises a failed download


MAX_CRASH_TEXT = 200  # characters of an unexpected error's own text that an agent is shown


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn failures into ToolError so the agent sees the actual message. The MCP SDK would answer anything
    else with a bare "Error executing tool" and keep the reason in a log the agent and user never see."""
    try:
        yield
    except (PlaygroundError, BehaviorError, ModelLoadError, IndexError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    except ToolError:
        raise
    except Exception as exc:
        log.exception("unexpected error in a tool")
        what = f"{type(exc).__name__}: {exc}"
        if len(what) > MAX_CRASH_TEXT:  # an OSError repeats the whole path it was given, a page long for a pasted blob
            what = what[:MAX_CRASH_TEXT] + "..."
        raise ToolError(
            f"robotforge hit an unexpected error: {what}. Other tools still work; if it "
            "repeats, it is worth reporting with this message"
        ) from exc


SCRIPT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
CODE_FENCE = re.compile(r"\A\s*```[\w-]*[ \t]*\n(.*?)\n?[ \t]*```\s*\Z", re.DOTALL)


def _unfence(code: str) -> tuple[str, bool]:
    """Code as an agent often sends it, inside a markdown fence. Python cannot start with backticks, so the
    fence is never part of the script."""
    fenced = CODE_FENCE.match(code)
    return (fenced[1] + "\n", True) if fenced else (code, False)


def _script_path(world_dir: Path, name: str) -> Path:
    stem = wording.squash(name[:-3] if name.lower().endswith(".py") else name)  # "Pick Cube" is pick_cube
    if not SCRIPT_NAME.match(stem):
        raise PlaygroundError(
            f"script name {wording.short(name)} must use lowercase letters, digits and _, start with a letter, "
            "e.g. pick_cube"
        )
    return world_dir / "behaviors" / f"{stem}.py"


def _no_script(name: str, path: Path) -> str:
    have = sorted(p.stem for p in path.parent.glob("*.py"))
    text = wording.unknown("script", name, have, where="saved in this world")
    return text if have else f"{text} write_script saves one"


def _json(data: object) -> str:
    return wording.dumps(data)


def run(model: Path | None = None, port: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")  # stderr
    session = SimSession.from_path(model) if model else None
    playground = Playground(session=session) if session else Playground.with_demo()
    chosen = port if port is not None else find_free_port(DEFAULT_PORT, HOST)
    build_server(playground, chosen).run("stdio")
