"""The plugins tool: what plugins added, load them again, trust a world's own, run a controller or behavior.

Registered by robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from robotforge import wording
from robotforge.design import export
from robotforge.extend import routines
from robotforge.extend.registry import REGISTRY, WORLD_FOLDER
from robotforge.playground import Playground, PlaygroundError

MAX_CONTROLLER_SECONDS = 60.0
MAX_BEHAVIOR_SECONDS = 300.0

RunUntil = Callable[[Callable[[], bool], float], Awaitable[dict[str, Any]]]
RunScript = Callable[[Path, float, list[str]], Awaitable[dict[str, Any]]]


def register(server: MCPServer, playground: Playground, run_until: RunUntil, run_script: RunScript) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    @server.tool(annotations=ToolAnnotations(openWorldHint=False))
    async def plugins(
        action: Literal["list", "reload", "trust", "run"] = "list",
        name: str | None = None,
        robot: str | None = None,
        options: dict[str, Any] | None = None,
        seconds: float | None = None,
    ) -> str:
        """The user's own code that adds parts, sensors, controllers and behaviors. list: what is loaded (each with
        what it is for and its options) and every plugin file that failed or was not loaded, with why. reload: read the
        plugin folders again after a file was edited or added (~/.robotforge/plugins, and the open world's plugins/
        folder). trust: load the open world's plugins/ folder as it is now. run: start the controller or behavior
        `name` on `robot` with `options`; a controller runs inside the simulation for `seconds` (default 5, at most
        60) and keeps running afterwards until stop(robot), a behavior is a script that runs to its end (`seconds` is
        its time limit, default 60) and returns what it returns. A world's plugin files load only when they were
        written here (write_script with plugin=true) or trusted: they are Python that runs on this computer with the
        user's rights, so read each one (read_script) and tell the user what it does before trusting a world you did
        not make. Plugin parts are shapes for add_object, plugin sensors types for add_sensor."""
        with _tool_errors():
            store = playground.store
            if action == "trust":
                if store is None:
                    raise PlaygroundError("no world is open; plugins(action='trust') trusts the open world's plugins")
                if not REGISTRY.trust(store.dir):
                    raise PlaygroundError(f"the world has no {WORLD_FOLDER}/ folder with plugin files")
            elif action == "reload":
                REGISTRY.reload()
                if store is not None:
                    REGISTRY.load_world(store.dir)
            elif store is not None:
                REGISTRY.load_world(store.dir)
            if action == "run":
                return _json(await _run(name, robot, options, seconds))
            return _json(REGISTRY.report())

    async def _run(name: str | None, robot: str | None, options: dict[str, Any] | None, seconds: float | None):
        if not name or not robot:
            raise PlaygroundError("plugins(action='run') needs name (a controller or behavior) and robot")
        name = wording.squash(name)
        robot = playground.behaviors.resolve(robot)
        if REGISTRY.get("controller", name) is not None:
            return await _run_controller(name, robot, options, seconds)
        if REGISTRY.get("behavior", name) is not None:
            return await _run_behavior(name, robot, options, seconds)
        known = [*REGISTRY.of("controller"), *REGISTRY.of("behavior")]
        raise PlaygroundError(wording.unknown("controller or behavior", name, known, where="from plugins"))

    async def _run_controller(name: str, robot: str, options: dict[str, Any] | None, seconds: float | None):
        seconds = 5.0 if seconds is None else seconds
        if not 0 < seconds <= MAX_CONTROLLER_SECONDS:
            raise PlaygroundError(wording.seconds_range("seconds", seconds, MAX_CONTROLLER_SECONDS))
        session = playground.session

        def read(sensor: str) -> Any:
            reading, _ = playground.read_sensors(robot, [sensor])
            return next(iter(reading["sensors"].values()))

        started = routines.start_controller(session, robot, name, options, read)
        run = await run_until(lambda: _stopped(session, robot, name), seconds)
        with session.lock:
            attached = session.controllers.get(robot)
            mine = isinstance(attached, routines.PluginController) and attached.name == name
            status = attached.status() if mine else None
        out: dict[str, Any] = {**started, **run, "keeps_running": bool(mine and not status.get("stopped_with_error"))}
        if status:
            out["status"] = status
        if mine:
            out["release_with"] = f"stop(robot={robot!r})"
        else:
            out["note"] = f"something else took over {robot} during the run"
        return out

    def _stopped(session: Any, robot: str, name: str) -> bool:
        attached = session.controllers.get(robot)
        return not (isinstance(attached, routines.PluginController) and attached.name == name) or bool(attached.failure)

    async def _run_behavior(name: str, robot: str, options: dict[str, Any] | None, seconds: float | None):
        seconds = 60.0 if seconds is None else seconds
        if not 0 < seconds <= MAX_BEHAVIOR_SECONDS:
            raise PlaygroundError(wording.seconds_range("seconds", seconds, MAX_BEHAVIOR_SECONDS))
        path, args, used = routines.behavior_command(name, robot, options)
        with tempfile.TemporaryDirectory(prefix="robotforge_behavior_") as folder:
            script = Path(folder) / "run_behavior.py"
            script.write_text(routines.BEHAVIOR_SCRIPT, encoding="utf-8", newline="\n")
            out = await run_script(script, seconds, args)
        result, text = routines.parse_result(out.get("output", ""))
        out["output"] = text
        return {"behavior": name, "robot": robot, "options": used, "result": result, **out}

    @server.tool(annotations=ToolAnnotations(openWorldHint=False))
    async def export_model(
        robot: str | None = None, format: Literal["mjcf", "urdf"] = "mjcf", to: str | None = None
    ) -> str:
        """Write the open world, or one robot, for other simulators. format="mjcf" (MuJoCo's XML, the whole world
        or with robot=NAME just that robot on the floor) or "urdf" (one robot: robot is required), with the meshes as
        OBJ files in a meshes/ folder beside it. to is a folder (default exports/ in the world's folder). The file is
        loaded again in MuJoCo and compared with the world (same bodies, joints, masses, vertices): "check" says
        so. URDF cannot say free joints, loops, capsules or ball joints; "notes" lists what changed. Open the result
        with python -m mujoco.viewer --mjcf=FILE (an MJCF) or any URDF tool."""
        with _tool_errors():
            store = playground._require_store()
            world = store.load()
            if format == "urdf" and not robot:
                raise PlaygroundError("URDF is one robot at a time: give robot=NAME (robots lists them)")
            entry = None
            if robot:
                top = playground.behaviors.resolve(robot).split("/")[0]
                entry = next((r for r in world.robots if r.name == top), None)
                if entry is None:
                    raise PlaygroundError(wording.unknown("robot", robot, [r.name for r in world.robots]))
            default = store.dir / "exports" / f"{entry.name if entry else world.name}_{format}"
            folder = Path(to).expanduser() if to else default
            return _json(
                await asyncio.to_thread(export.export_world, world, entry, store.dir, format, folder, to is None)
            )
