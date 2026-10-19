"""The plugins tool: what plugins added, load them again, trust a world's own.

Registered by robotforge.mcp_server.build_server via `register(...)`.
"""

from __future__ import annotations

from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from robotforge.extend.registry import REGISTRY, WORLD_FOLDER
from robotforge.playground import Playground, PlaygroundError


def register(server: MCPServer, playground: Playground) -> None:
    from robotforge.mcp_server import _json, _tool_errors  # here: mcp_server imports this module

    @server.tool(annotations=ToolAnnotations(openWorldHint=False))
    def plugins(action: Literal["list", "reload", "trust"] = "list") -> str:
        """The user's own code that adds parts, sensors, controllers and behaviors. list: what is loaded (each with
        what it is for) and every plugin file that failed or was not loaded, with why. reload: read the plugin
        folders again after a file was edited or added (~/.robotforge/plugins, and the open world's plugins/ folder).
        trust: load the open world's plugins/ folder as it is now. A world's plugin files load only when they were
        written here (write_script with plugin=true) or trusted: they are Python that runs on this computer with the
        user's rights, so read each one (read_script) and tell the user what it does before trusting a world you did
        not make. Plugin parts are shapes for add_object, plugin sensors types for add_sensor."""
        with _tool_errors():
            store = playground.store
            if action == "trust":
                if store is None:
                    raise PlaygroundError("no world is open; plugins(action='trust') trusts the open world's plugins")
                names = REGISTRY.trust(store.dir)
                if not names:
                    raise PlaygroundError(f"the world has no {WORLD_FOLDER}/ folder with plugin files")
            elif action == "reload":
                REGISTRY.reload()
                if store is not None:
                    REGISTRY.load_world(store.dir)
            elif store is not None:
                REGISTRY.load_world(store.dir)
            return _json(REGISTRY.report())
