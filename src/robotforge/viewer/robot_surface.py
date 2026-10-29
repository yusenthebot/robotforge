"""Guarded, opt-in robot part finishes; source geometry and physical parameters stay intact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Self
from uuid import uuid4

import mujoco
from pydantic import Field, model_validator

from robotforge.design import body_surfaces, layouts, menagerie
from robotforge.design.world import RobotEntry, World, compile_world
from robotforge.playground import PlaygroundError
from robotforge.sim import carry
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.object_edit import Selection, UndoEdit, _guard
from robotforge.viewer.robot_edit import _eligible

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground


class SetBodySurface(Selection):
    body: str | None = Field(default=None, min_length=1, max_length=160)
    bodies: list[Annotated[str, Field(min_length=1, max_length=160)]] | None = Field(
        default=None, min_length=1, max_length=64
    )
    surface_material: Literal["plastic", "wood", "steel"] | None

    @model_validator(mode="after")
    def select_parts(self) -> Self:
        if len(self.model_fields_set & {"body", "bodies"}) != 1 or (self.body is None and self.bodies is None):
            raise ValueError("give exactly one of body or bodies")
        if self.bodies is not None and len(set(self.bodies)) != len(self.bodies):
            raise ValueError("bodies must be unique")
        return self


@dataclass(frozen=True)
class RobotSurfaceEdit:
    store: WorldStore
    version: int
    generation: int
    token: str
    world_sha256: str
    before: RobotEntry
    after: RobotEntry
    index: int
    bodies: tuple[str, ...]
    batch: bool


def metadata(pg: Playground) -> dict:
    with pg._edit_lock, pg.lessons._lock, pg.session.lock:
        if pg.store is None or pg.lessons.lesson is not None or not pg._play_world_sha256:
            return {"robots": []}
        try:
            world = _guard(pg, pg.store, pg.session.version, pg.session.generation, pg._play_world_sha256, "robot")
        except PlaygroundError:
            return {"robots": []}
        robots = []
        for entry in world.robots:
            if entry.mounts or not _eligible(pg.session.model, entry.name):
                continue
            parts = body_surfaces.parts(pg.session.model, entry.name)
            if parts:
                robots.append(
                    {
                        "name": entry.name,
                        "parts": [
                            {**part, "surface_material": entry.body_surfaces.get(part["body"])} for part in parts
                        ],
                    }
                )
        return {"robots": robots, "version": pg.session.version, "world_sha256": pg._play_world_sha256}


def _check(pg: Playground, store: WorldStore, guard: Selection) -> World:
    world = _guard(pg, store, guard.version, guard.generation, guard.world_sha256, "robot")
    check_supported_restore(pg, "changing a robot part surface")
    if pg.session.model.nflex:
        raise PlaygroundError("part surfaces need a rigid world")
    return world


def handle(pg: Playground, message: dict) -> dict:
    undoing = message.get("action") == "undo_robot_surface"
    request = _parse(UndoEdit if undoing else SetBodySurface, message, "robot surface")
    with pg.session.execution.command("viewer_undo_robot_surface" if undoing else "viewer_set_robot_surface"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            if undoing:
                receipt = pg._play_edit
                if (
                    not isinstance(receipt, RobotSurfaceEdit)
                    or receipt.store is not store
                    or (request.token, request.version, request.generation)
                    != (receipt.token, receipt.version, receipt.generation)
                ):
                    raise PlaygroundError("that part surface edit is no longer available to undo")
                before, after, index, bodies, batch = (
                    receipt.after,
                    receipt.before,
                    receipt.index,
                    receipt.bodies,
                    receipt.batch,
                )
                guard = Selection(
                    name=before.name,
                    version=receipt.version,
                    generation=receipt.generation,
                    world_sha256=receipt.world_sha256,
                )
                world = _check(pg, store, guard)
                if index >= len(world.robots) or world.robots[index] != before:
                    raise PlaygroundError("the robot changed; select it again")
            else:
                guard = request
                world = _check(pg, store, guard)
                index = next((i for i, entry in enumerate(world.robots) if entry.name == request.name), None)
                if index is None:
                    raise PlaygroundError("select an existing robot part")
                before = world.robots[index]
                batch = request.bodies is not None
                if request.bodies is not None:
                    bodies = tuple(request.bodies)
                else:
                    assert request.body is not None  # SetBodySurface enforces exactly one target form.
                    bodies = (request.body,)
                eligible = {p["body"] for p in body_surfaces.parts(pg.session.model, before.name)}
                if before.mounts or not _eligible(pg.session.model, before.name) or not set(bodies) <= eligible:
                    raise PlaygroundError("select an opaque, untextured part of an unmounted rigid robot")
                finishes = dict(before.body_surfaces)
                for body in bodies:
                    if request.surface_material is None:
                        finishes.pop(body, None)
                    else:
                        finishes[body] = request.surface_material
                if len(finishes) > 64:
                    raise PlaygroundError("a robot can have at most 64 styled parts; restore other parts first")
                after = RobotEntry.model_validate({**before.model_dump(), "body_surfaces": finishes})
                if before == after:
                    return {
                        "robot": before.name,
                        **({"bodies": list(bodies)} if batch else {"body": bodies[0]}),
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "unchanged": True,
                    }
            candidate = world.model_copy(deep=True)
            candidate.robots[index] = after
            candidate = World.model_validate(candidate.model_dump())
            original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
            if layouts.sha(original) != guard.world_sha256:
                raise PlaygroundError("the saved world changed; select the robot again")
        with menagerie.cached_only():
            compiled = compile_world(candidate, store.dir)
        fresh_data(compiled.model)
        prepared = layouts.PreparedWorld(original, candidate.to_yaml().encode(), compiled, compiled.to_xml().encode())
        if len(prepared.after) > layouts.MAX_WORLD_BYTES:
            raise PlaygroundError("the world is too large for a local surface edit")
        with pg.lessons._lock, pg.session.lock:
            _check(pg, store, guard)
            if store.file.read_bytes() != original:
                raise PlaygroundError("the saved world changed; select the robot again")
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
                        "robot": before.name,
                        **({"bodies": list(bodies)} if batch else {"body": bodies[0]}),
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "paused": True,
                        "undo_surface": None,
                    }
                    if not undoing:
                        receipt = RobotSurfaceEdit(
                            store,
                            pg.session.version,
                            pg.session.generation,
                            uuid4().hex,
                            layouts.sha(prepared.after),
                            before,
                            after,
                            index,
                            bodies,
                            batch,
                        )
                        pg._play_edit = receipt
                        response["undo_surface"] = {
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
