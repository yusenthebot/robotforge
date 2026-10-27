"""Reposition exact authored terrain blocks with saved/live clearance and one-use Undo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, field_validator

from robotforge.design import terrain
from robotforge.playground import PlaygroundError
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import Coordinate, _parse
from robotforge.viewer.object_edit import Selection, UndoEdit, _owners
from robotforge.viewer.terrain_create import _check, _clear_world, _commit, _occupants

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

Block = terrain.Ramp | terrain.Stairs


class Reposition(Selection):
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    turns: Literal[0, 1, 2, 3] = 0

    @field_validator("turns", mode="before")
    @classmethod
    def integer_turns(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("turns must be an integer quarter turn")
        return value


@dataclass(frozen=True)
class TerrainEdit:
    store: WorldStore
    version: int
    generation: int
    token: str
    world_sha256: str
    before: Block
    after: Block
    index: int


def metadata(pg: Playground) -> dict:
    with pg._edit_lock, pg.lessons._lock, pg.session.lock:
        if pg.store is None or pg.lessons.lesson is not None or not pg._play_world_sha256:
            return {"terrains": []}
        from robotforge.viewer.layouts import Guard

        guard = Guard(version=pg.session.version, generation=pg.session.generation, world_sha256=pg._play_world_sha256)
        try:
            world = _check(pg, pg.store, guard)
        except (PlaygroundError, ValueError):
            return {"terrains": []}
        owners = _owners(pg.session.model, world)
        entries = []
        for block in world.terrain:
            if not isinstance(block, Block):
                continue
            geoms = [g for g in owners if owners[g] == block.name]
            if geoms:
                entries.append(
                    {
                        "name": block.name,
                        "type": block.type,
                        "pos": block.pos,
                        "yaw": block.yaw,
                        "boxes": terrain.box_geometry(block),
                        "geoms": geoms,
                    }
                )
        return {"terrains": entries, "version": pg.session.version, "world_sha256": pg._play_world_sha256}


def handle(pg: Playground, message: dict) -> dict:
    undoing = message.get("action") == "undo_terrain_edit"
    request = _parse(UndoEdit if undoing else Reposition, message, "terrain edit")
    with pg.session.execution.command("viewer_undo_terrain_edit" if undoing else "viewer_edit_terrain"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            if undoing:
                receipt = pg._play_terrain_edit
                if (
                    not isinstance(receipt, TerrainEdit)
                    or receipt.store is not store
                    or (request.token, request.version, request.generation)
                    != (receipt.token, receipt.version, receipt.generation)
                ):
                    raise PlaygroundError("that terrain move is no longer available to undo")
                from robotforge.viewer.layouts import Guard

                guard = Guard(version=receipt.version, generation=receipt.generation, world_sha256=receipt.world_sha256)
                world = _check(pg, store, guard)
                index, before, after = receipt.index, receipt.after, receipt.before
                if index >= len(world.terrain) or world.terrain[index] != before:
                    raise PlaygroundError("the moved terrain changed; select it again")
            else:
                guard = request
                world = _check(pg, store, guard)
                index = next((i for i, block in enumerate(world.terrain) if block.name == request.name), None)
                if index is None or not isinstance(world.terrain[index], Block):
                    raise PlaygroundError("select an editable ramp or stair block")
                before = world.terrain[index]
                if len(request.pos) == 3 and abs(request.pos[2]) > 0.001:
                    raise PlaygroundError("place local terrain on the ground, not on another surface")
                after = terrain.parse(
                    {**before.model_dump(), "pos": request.pos[:2], "yaw": before.yaw + 90 * request.turns}
                )
                if before == after:
                    return {
                        "terrain": before.name,
                        "version": pg.session.version,
                        "generation": pg.session.generation,
                        "unchanged": True,
                    }
            old_model = pg.session.model
        _clear_world(world, after, excluding=before.name)
        saved = fresh_data(old_model)
        blocked = _occupants(old_model, saved, world, before, excluding=before.name)
        blocked.update(_occupants(old_model, saved, world, after, excluding=before.name))
        candidate = world.model_copy(deep=True)
        candidate.terrain[index] = after

        def publish(version: int, generation: int, digest: str) -> dict:
            if undoing:
                return {"terrain": before.name}
            receipt = TerrainEdit(store, version, generation, uuid4().hex, digest, before, after, index)
            pg._play_terrain_edit = receipt
            return {
                "terrain": before.name,
                "undo_terrain_edit": {"token": receipt.token, "version": version, "generation": generation},
            }

        return _commit(
            pg,
            store,
            guard,
            candidate,
            after,
            blocked,
            clearance=(before, after),
            excluding=before.name,
            on_publish=publish,
        )
