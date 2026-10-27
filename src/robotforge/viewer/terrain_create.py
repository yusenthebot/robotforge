"""Guarded local terrain additions on clear ground, sharing authored layout transactions."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import mujoco
from pydantic import Field, field_validator

from robotforge.design import layouts, menagerie, terrain
from robotforge.design.world import World, compile_world
from robotforge.playground import PlaygroundError, _unused
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import Coordinate, UndoAddition, _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.layouts import Guard
from robotforge.viewer.object_edit import _bounds, _guard, _owners

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

PRESETS = ("ramp", "stairs")
GAP = 0.01  # Keep separate terrain features and occupants clear of each other.


class Placement(Guard):
    preset: Literal["ramp", "stairs"]
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    yaw: Literal[0, 90, 180, 270] = 0

    @field_validator("yaw", mode="before")
    @classmethod
    def integer_turn(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("yaw must be an integer quarter turn")
        return value


@dataclass(frozen=True)
class TerrainAddition:
    store: WorldStore
    version: int
    generation: int
    token: str
    name: str
    entry: str
    world_sha256: str


def catalog() -> list[dict]:
    return [{"name": kind, "boxes": terrain.box_geometry(terrain.parse({"type": kind}))} for kind in PRESETS]


def _check(pg: Playground, store: WorldStore, request: Guard) -> World:
    world = _guard(pg, store, request.version, request.generation, request.world_sha256, "terrain")
    check_supported_restore(pg, "editing local terrain")
    if pg.session.model.nflex:
        raise PlaygroundError("local terrain needs a rigid world; choose another map")
    return world


def _overlap(a, b) -> bool:
    return min(a[1] - b[0], b[1] - a[0], a[3] - b[2], b[3] - a[2]) > -GAP


def _clear_world(world: World, patch, *, excluding: str | None = None) -> None:
    box = terrain._world_bounds(patch)
    for other in world.terrain:
        if other.name == excluding:
            continue
        if _overlap(box, terrain._world_bounds(other)):
            raise PlaygroundError(f"that spot is occupied by {other.name}; choose clear ground")
    water = world.water
    if (
        water
        and water.level + water.waves > 0
        and (water.area is None or _overlap(box, (water.area[0], water.area[2], water.area[1], water.area[3])))
    ):
        raise PlaygroundError("choose dry ground for local terrain")


def _occupants(model, data, world: World, patch, *, excluding: str | None = None) -> set[str]:
    """Conservative XY clearance, including overhead/fixed objects and complete imported robots."""
    mujoco.mj_forward(model, data)
    owners = _owners(model, world)
    box = terrain._world_bounds(patch)
    found = set()
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]) or owners[g] == excluding:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        low, high = _bounds(model, data, g)
        if _overlap(box, (low[0], high[0], low[1], high[1])):
            found.add(owners[g])
    return found


def handle(pg: Playground, message: dict) -> dict:
    """Called under the playground edit lock; no candidate is published during preflight."""
    request = _parse(Placement, message, "terrain placement")
    if len(request.pos) == 3 and abs(request.pos[2]) > 0.001:
        raise PlaygroundError("place local terrain on the ground, not on another surface")
    with pg.session.execution.command("viewer_add_terrain"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            world = _check(pg, store, request)
            old_model = pg.session.model
        taken = {e.name for e in [*world.robots, *world.objects, *world.terrain]}
        patch = terrain.parse(
            {
                "type": request.preset,
                "name": _unused(request.preset, taken),
                "pos": request.pos[:2],
                "yaw": request.yaw,
                "style": world.floor.style,
            }
        )
        _clear_world(world, patch)
        blocked = _occupants(old_model, fresh_data(old_model), world, patch)
        candidate = world.model_copy(deep=True)
        candidate.terrain.append(patch)
        return _commit(pg, store, request, candidate, patch, blocked)


def _commit(
    pg,
    store,
    request,
    candidate: World,
    patch,
    blocked: set[str],
    *,
    removing: bool = False,
    clearance=None,
    excluding: str | None = None,
    on_publish: Callable[[int, int, str], dict] | None = None,
) -> dict:
    original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
    if layouts.sha(original) != request.world_sha256:
        raise PlaygroundError("the saved world changed; choose a spot again")
    with menagerie.cached_only():
        compiled = compile_world(candidate, store.dir)
    fresh_data(compiled.model)
    prepared = layouts.PreparedWorld(
        original, candidate.to_yaml().encode("utf-8"), compiled, compiled.to_xml().encode("utf-8")
    )
    if len(prepared.after) > layouts.MAX_WORLD_BYTES:
        raise PlaygroundError("the world is too large for a local terrain edit")
    with pg.lessons._lock, pg.session.lock:
        world = _check(pg, store, request)
        if store.file.read_bytes() != original:
            raise PlaygroundError("the saved world changed; choose a spot again")
        for area in clearance if clearance is not None else (patch,):
            blocked.update(
                _occupants(
                    pg.session.model,
                    copy.copy(pg.session.data),
                    world,
                    area,
                    excluding=excluding if excluding is not None else patch.name if removing else None,
                )
            )
        if blocked:
            raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; move away first")
        rollback = BuiltinLayoutState.capture(pg)
        try:
            with layouts._install_world(store, prepared) as notes:
                result = pg._load_world(compiled)
                pg.session.paused = True
                pg._places = None
                response = {
                    "version": pg.session.version,
                    "generation": pg.session.generation,
                    "paused": True,
                }
                digest = layouts.sha(prepared.after)
                if on_publish is not None:
                    response.update(on_publish(pg.session.version, pg.session.generation, digest))
                elif not removing:
                    receipt = TerrainAddition(
                        store,
                        pg.session.version,
                        pg.session.generation,
                        uuid4().hex,
                        patch.name,
                        patch.model_dump_json(),
                        digest,
                    )
                    pg._play_addition = receipt
                    response.update(
                        terrain=patch.name,
                        undo={"token": receipt.token, "version": receipt.version, "name": receipt.name},
                    )
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
                raise PlaygroundError("terrain rollback failed; reopen the world before continuing") from exc
            raise
        return response


def undo(pg: Playground, message: dict) -> dict:
    request = _parse(UndoAddition, message, "undo terrain")
    with pg.session.execution.command("viewer_undo_terrain"):
        receipt = pg._play_addition
        if (
            not isinstance(receipt, TerrainAddition)
            or request.token != receipt.token
            or request.version != receipt.version
        ):
            raise PlaygroundError("that terrain addition is no longer available to undo")
        guard = Guard(version=receipt.version, generation=receipt.generation, world_sha256=receipt.world_sha256)
        with pg.lessons._lock, pg.session.lock:
            world = _check(pg, receipt.store, guard)
            match = [p for p in world.terrain if p.name == receipt.name and p.model_dump_json() == receipt.entry]
            if len(match) != 1:
                raise PlaygroundError("the added terrain changed; it cannot be undone")
            patch = match[0]
            blocked = _occupants(pg.session.model, fresh_data(pg.session.model), world, patch, excluding=patch.name)
        candidate = world.model_copy(deep=True)
        candidate.terrain.remove(patch)
        result = _commit(pg, receipt.store, guard, candidate, patch, blocked, removing=True)
        return {**result, "removed": receipt.name}
