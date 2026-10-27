"""Persist prop finishes independently of physical material, with guarded rebuild and Undo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import mujoco

from robotforge.design import layouts, menagerie
from robotforge.design.world import ObjectEntry, World, compile_world
from robotforge.playground import PlaygroundError
from robotforge.sim import carry
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.object_edit import Selection, UndoEdit, _guard

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

SHAPES = {"box", "sphere", "cylinder"}


class SetSurface(Selection):
    surface_material: Literal["plastic", "wood", "steel"] | None


@dataclass(frozen=True)
class SurfaceEdit:
    store: WorldStore
    version: int
    generation: int
    token: str
    world_sha256: str
    before: ObjectEntry
    after: ObjectEntry
    index: int


def _check(pg: Playground, store: WorldStore, guard: Selection) -> World:
    world = _guard(pg, store, guard.version, guard.generation, guard.world_sha256)
    check_supported_restore(pg, "changing a prop's surface")
    if pg.session.model.nflex:
        raise PlaygroundError("surface editing needs a rigid world; choose another map")
    return world


def handle(pg: Playground, message: dict) -> dict:
    """Keep current poses, then reset motion and pause; never change physical material."""
    undoing = message.get("action") == "undo_object_edit"
    request = _parse(UndoEdit if undoing else SetSurface, message, "prop surface")
    with pg.session.execution.command("viewer_undo_surface" if undoing else "viewer_set_surface"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            if undoing:
                receipt = pg._play_edit
                if (
                    not isinstance(receipt, SurfaceEdit)
                    or receipt.store is not store
                    or (request.token, request.version, request.generation)
                    != (receipt.token, receipt.version, receipt.generation)
                ):
                    raise PlaygroundError("that surface edit is no longer available to undo")
                before, after, index = receipt.after, receipt.before, receipt.index
                guard = Selection(
                    name=before.name,
                    version=receipt.version,
                    generation=receipt.generation,
                    world_sha256=receipt.world_sha256,
                )
                world = _check(pg, store, guard)
                if index >= len(world.objects) or world.objects[index] != before:
                    raise PlaygroundError("the prop changed; select it again")
            else:
                guard = request
                world = _check(pg, store, guard)
                index = next((i for i, obj in enumerate(world.objects) if obj.name == request.name), None)
                if index is None or world.objects[index].shape not in SHAPES:
                    raise PlaygroundError("select an existing box, ball or cylinder surface")
                before = world.objects[index]
                after = ObjectEntry.model_validate(
                    {**before.model_dump(), "surface_material": request.surface_material}
                )
                if before == after:
                    return {
                        "object": before.name,
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "unchanged": True,
                    }
            candidate = world.model_copy(deep=True)
            candidate.objects[index] = after
            candidate = World.model_validate(candidate.model_dump())
            original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
            if layouts.sha(original) != guard.world_sha256:
                raise PlaygroundError("the saved world changed; select the prop again")
        with menagerie.cached_only():
            compiled = compile_world(candidate, store.dir)
        fresh_data(compiled.model)
        prepared = layouts.PreparedWorld(
            original, candidate.to_yaml().encode("utf-8"), compiled, compiled.to_xml().encode("utf-8")
        )
        if len(prepared.after) > layouts.MAX_WORLD_BYTES:
            raise PlaygroundError("the world is too large for a local surface edit")
        with pg.lessons._lock, pg.session.lock:
            _check(pg, store, guard)
            if store.file.read_bytes() != original:
                raise PlaygroundError("the saved world changed; select the prop again")
            # Capture after preparation: a running prop may have moved while compilation finished.
            live = carry.take(pg.session.model, pg.session.data)
            rollback = BuiltinLayoutState.capture(pg)
            try:
                with layouts._install_world(store, prepared) as notes:
                    result = pg._load_world(compiled)
                    carry._apply(pg.session.model, pg.session.data, live, lambda _name: True)
                    mujoco.mj_forward(pg.session.model, pg.session.data)
                    pg.session.paused = True
                    pg._places = None
                    response = {
                        "object": before.name,
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "paused": True,
                        "undo_edit": None,
                    }
                    if not undoing:
                        receipt = SurfaceEdit(
                            store,
                            pg.session.version,
                            pg.session.generation,
                            uuid4().hex,
                            layouts.sha(prepared.after),
                            before,
                            after,
                            index,
                        )
                        pg._play_edit = receipt
                        response["undo_edit"] = {
                            "token": receipt.token,
                            "version": receipt.version,
                            "generation": receipt.generation,
                        }
                response["notes"] = [*result["notes"], *notes]
            except BaseException as exc:
                rollback.restore()
                try:
                    recovered = store.file.read_bytes() == original
                except OSError:
                    recovered = False
                if not recovered or getattr(exc, "code", None) == "layout_recovery":
                    pg.session.paused = True
                    pg._play_world_sha256 = None
                    raise PlaygroundError("surface rollback failed; reopen the world before continuing") from exc
                raise
            return response
