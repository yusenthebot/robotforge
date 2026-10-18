"""MCP tools for scenes and terrain: list and load ready-made scenes, shape the ground.

Registered by robotforge.mcp_server.build_server via `register(...)`, so the server
file needs one line for all of them.
"""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from robotforge.design.scenes import SCENES
from robotforge.design.terrain import TYPES, StyleName
from robotforge.playground import Playground

SceneName = Literal[tuple(SCENES)]  # desk_lab, kitchen, factory, warehouse, park, maze, obstacle_course
TerrainType = Literal[tuple(TYPES)]  # hills, rough, stairs, ramp, stones, curb, platform


def register(server: MCPServer, playground: Playground) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    @server.tool()
    def list_scenes() -> str:
        """List ready-made scenes (desk_lab, kitchen, factory, warehouse, park, maze, obstacle_course) with their
        options, the terrain types add_terrain can lay down with every parameter, and the ground styles."""
        return _json(playground.list_scenes())

    @server.tool()
    def load_scene(
        preset: SceneName,
        seed: int = 0,
        keep_robots: bool = True,
        starter: bool = True,
        options: dict[str, float] | None = None,
    ) -> str:
        """Fill the open world with a ready-made scene, replacing its floor, terrain and objects (undo restores them);
        with no world open, opens one named after the scene. Robots are kept and moved to the scene's places (arms
        onto the desk or counter); keep_robots=false removes them. A world left without robots gets the scene's
        starter robot unless starter=false. seed changes the layout (maze paths, tree spots, crates); options size
        some scenes, e.g. load_scene("maze", seed=3, options={"rows": 7, "cols": 7}). Afterwards it is ordinary
        world.yaml content: move, remove or add objects freely. The result's "scene" field says where robots went
        and what to try."""
        with _tool_errors():
            return _json(playground.load_scene(preset, seed, keep_robots, starter, options))

    @server.tool()
    def set_terrain(
        type: Literal["flat", *TYPES], style: StyleName | None = None, params: dict[str, Any] | None = None
    ) -> str:
        """Make the ground one kind of terrain around the origin, replacing all terrain patches; "flat" removes them.
        Robots and objects are lifted onto it. Examples: set_terrain("hills"); set_terrain("rough", params={"size":
        [6, 6], "height": 0.04}). list_scenes shows each type's parameters; for several patches use add_terrain."""
        with _tool_errors():
            return _json(playground.set_terrain(type, style, params))

    @server.tool()
    def add_terrain(
        type: TerrainType,
        name: str | None = None,
        pos: list[float] | None = None,
        yaw: float = 0.0,
        style: StyleName | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Lay a terrain patch on the floor, centered at pos [x, y] and turned by yaw degrees. Stairs and ramps climb
        along the patch's own +x. Types and their main params (meters, degrees):

          hills {"size": [6, 6], "height": 0.25, "scale": 1.5, "seed": 0} smooth rolling hills
          rough {"size": [3, 3], "height": 0.03, "scale": 0.25, "seed": 0} small bumps
          stairs {"steps": 4, "step_height": 0.05, "step_depth": 0.3, "width": 1.2, "direction": "up_down"}
          ramp {"angle": 10, "length": 1.5, "width": 1.2, "direction": "up_down", "landing": 0.8}
          stones {"size": [2.4, 1.2], "stone": 0.25, "gap": 0.08, "height": 0.08} stepping stones
          curb {"size": [0.15, 2.0], "height": 0.06}; platform {"size": [1, 1], "height": 0.1}

        Example: add_terrain("ramp", pos=[2, 0], params={"angle": 15}). style is the look and grip (grass, dirt, sand,
        rock, snow is slippery, concrete, wood, tile). The result notes each patch's extent and slope."""
        with _tool_errors():
            return _json(playground.add_terrain(type, name, pos, yaw, style, params))

    @server.tool()
    def update_terrain(
        name: str,
        pos: list[float] | None = None,
        yaw: float | None = None,
        style: StyleName | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Change a terrain patch: move or turn it, restyle it, or change parameters, e.g.
        update_terrain("ramp1", params={"angle": 15}). Remove it with remove(name)."""
        with _tool_errors():
            return _json(playground.update_terrain(name, pos, yaw, style, params))
