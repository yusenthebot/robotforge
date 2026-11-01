"""Read-only selection of one rigid support and all props carried by it."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pydantic import Field

from robotforge.playground import PlaygroundError
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import _parse
from robotforge.viewer.object_edit import SHAPES, _owners, _support_context
from robotforge.viewer.object_group import Guard, _check, _members, _support_graph

if TYPE_CHECKING:
    from robotforge.playground import Playground
    from robotforge.viewer.object_edit import _SupportContext


class Select(Guard):
    name: str = Field(min_length=1, max_length=32)


def _reachable(context: _SupportContext, world, root: str) -> dict[str, frozenset[str]]:
    """Resolve only reachable owners; diamonds are valid, cycles and partial results are not."""
    editable = {entry.name for entry in world.objects if entry.shape in SHAPES}
    edges: dict[str, frozenset[str]] = {}
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise PlaygroundError("the selected layout has a cyclic support relationship")
        if name in edges:
            return
        if name not in editable:
            raise PlaygroundError(f"{name} is not an editable rigid prop; its support layout cannot be selected")
        if len(edges) == 16:
            raise PlaygroundError("the support layout contains more than 16 props; select a smaller layout")
        active.add(name)
        edges[name] = context.supported(name)
        for carried in sorted(edges[name]):
            visit(carried)
        active.remove(name)

    visit(root)
    return edges


def handle(pg: Playground, message: dict) -> dict:
    """Select atomically, without issuing a layout preparation or consuming Undo."""
    request = _parse(Select, message, "support layout selection")
    with pg.session.execution.command("viewer_select_object_group"), pg.lessons._lock, pg.session.lock:
        store = pg._require_store()
        world = _check(pg, store, request)
        # Even forward/contact bookkeeping belongs to these private copies.
        model = copy.copy(pg.session.model)
        owners = _owners(model, world)
        saved = _support_context(model, fresh_data(model), owners, forwarded=True)
        live = _support_context(model, copy.copy(pg.session.data), owners)
        saved_edges = _reachable(saved, world, request.name)
        live_edges = _reachable(live, world, request.name)
        if saved_edges != live_edges:
            raise PlaygroundError("the saved and live support layouts differ; reset the simulation and select again")
        names = [entry.name for entry in world.objects if entry.name in live_edges]
        for context in (saved, live):
            _members(model, context.data, world, names)
            _support_graph(model, context.data, world, set(names))
        # Keep the same file/version/native admission checks at the return boundary.
        _check(pg, store, request)
        return {
            "name": request.name,
            "names": names,
            "version": request.version,
            "generation": request.generation,
            "world_sha256": request.world_sha256,
        }
