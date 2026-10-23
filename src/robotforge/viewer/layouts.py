"""Viewer layout actions use the same authored world, guarded against stale scene transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotforge.design import layouts
from robotforge.design.world import World
from robotforge.sim.execution import ExecutionError
from robotforge.sim.session import fresh_data
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore

if TYPE_CHECKING:
    from robotforge.playground import Playground


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Guard(Empty):
    version: int = Field(ge=1)
    generation: int = Field(ge=0)
    world_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Save(Guard):
    name: str = Field(min_length=1, max_length=256)


class Target(Guard):
    layout_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _current(pg: Playground, request: Guard | None = None):
    if pg.store is None:
        raise layouts.LayoutError("no_world", "Choose a map before saving layouts")
    if pg.lessons.lesson is not None:
        raise layouts.LayoutError("lesson", "Leave the lesson before using saved layouts")
    raw = layouts._read(pg.store.file, layouts.MAX_WORLD_BYTES)
    digest = layouts.sha(raw)
    if digest != pg._play_world_sha256 or (
        request is not None
        and (
            pg.session.version != request.version
            or pg.session.generation != request.generation
            or digest != request.world_sha256
        )
    ):
        raise layouts.LayoutError("changed", "The scene or saved world changed; refresh the layouts")
    world = World.from_yaml(raw.decode("utf-8"))
    if layouts.sha(world.model_dump_json().encode("utf-8")) != pg._play_world_descriptor:
        raise layouts.LayoutError("changed", "The saved world differs from the simulation; reopen the world")
    return pg.store, raw


def _response(pg, raw, entries, **result):
    return {
        "available": True,
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": layouts.sha(raw),
        "entries": entries,
        "limit": layouts.MAX_LAYOUTS,
        **result,
    }


def handle(pg: Playground, message: dict) -> dict:
    action = message.get("action")
    schema = {"layout_list": Empty, "layout_save": Save, "layout_restore": Target, "layout_remove": Target}.get(action)
    if schema is None:
        raise layouts.LayoutError("invalid_request", "Unknown layout action")
    try:
        request = schema.model_validate({k: v for k, v in message.items() if k not in {"type", "action", "id"}})
    except ValidationError as exc:
        raise layouts.LayoutError("invalid_request", f"Invalid layout request: {exc}") from exc
    try:
        if action == "layout_list":
            with pg.lessons._lock, pg.session.lock:
                if pg.store is None or pg.lessons.lesson is not None:
                    return {
                        "available": False,
                        "reason": "layout_lesson" if pg.lessons.lesson is not None else "layout_no_world",
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "entries": [],
                        "limit": layouts.MAX_LAYOUTS,
                    }
                store, raw = _current(pg)
                token = _response(pg, raw, [])
            return {**token, "entries": layouts.list_layouts(store)}
        with pg.session.execution.command(action):
            with pg.lessons._lock, pg.session.lock:
                store, raw = _current(pg, request)
                entries = layouts.list_layouts(store)
                if action == "layout_save":
                    saved = layouts.save_layout(store, request.name, layouts.sha(raw))
                    return _response(pg, raw, [saved, *entries], name=saved["name"])
                if action == "layout_remove":
                    layouts.remove_layout(store, request.layout_id, request.sha256)
                    return _response(pg, raw, [e for e in entries if e["id"] != request.layout_id])
                check_supported_restore(pg)
            prepared = layouts.prepare_restore(store, request.layout_id, request.sha256)
            fresh_data(prepared.compiled.model)  # preflight authored initial state before changing any active data
            with pg.lessons._lock, pg.session.lock:
                _current(pg, request)
                if pg.store is not store or prepared.before != raw:
                    raise layouts.LayoutError("changed", "The world changed while preparing restoration")
                check_supported_restore(pg)
                saved = BuiltinLayoutState.capture(pg)
                try:
                    with layouts.install(store, prepared) as notes:
                        result = pg._load_world(prepared.compiled)
                        pg.session.paused = True
                except BaseException as exc:
                    saved.restore()
                    # A second I/O failure during disk rollback needs explicit repair before more simulation.
                    try:
                        recovered = store.file.read_bytes() == raw
                    except OSError:
                        recovered = False
                    if not recovered or getattr(exc, "code", None) == "layout_recovery":
                        pg.session.paused = True
                        pg._play_world_sha256 = None
                        raise layouts.LayoutError(
                            "recovery", "Restore rollback failed; reopen the world before continuing"
                        ) from exc
                    raise
                pg._play_addition = pg._play_edit = pg._places = None
                return _response(pg, prepared.after, entries, name=prepared.name, notes=[*result["notes"], *notes])
    except ExecutionError as exc:
        raise layouts.LayoutError("busy", str(exc)) from exc
    except OSError as exc:
        raise layouts.LayoutError("io", f"The layout files could not be updated: {exc}") from exc
