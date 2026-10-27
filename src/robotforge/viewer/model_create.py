"""Place curated library models and saved designs through server-owned source IDs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import mujoco
from pydantic import BaseModel, ConfigDict, Field

from robotforge.design import combine, layouts, library, menagerie
from robotforge.design.world import RobotEntry, World, compile_world
from robotforge.playground import PlaygroundError
from robotforge.sim import carry, placement
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import Addition, Coordinate, UndoAddition, _object_obstacles, _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.object_edit import _owners, _supported, _task_users
from robotforge.viewer.robot_edit import _eligible, _platform_fits, _surface_position

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground

FEATURED = ("go2", "panda")


class ModelPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source: Literal["library", "design"]
    model: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    version: int = Field(ge=1)
    generation: int = Field(ge=0)


@dataclass(frozen=True)
class ModelAddition(Addition):
    """Distinguish protected model removal while retaining the existing Undo API."""


def catalog() -> list[dict]:
    """Metadata only: listing never compiles a model or starts a download."""
    entries = [
        {"name": name, "source": "design", "title": name.replace("_", " ").title(), "local": True}
        for name in combine.designs()[:16]
        if _design_name(name)
    ]
    for name in FEATURED:
        robot = library.get(name)
        entries.append(
            {
                "name": name,
                "source": "library",
                "title": robot.title,
                "kind": robot.kind,
                "summary": robot.summary,
                "downloaded": library.is_downloaded(name),
                "download_mb": library.download_mb(name),
            }
        )
    return entries


def _design_name(name: str) -> bool:
    from robotforge.design.world import NAME_RE

    return NAME_RE.fullmatch(name) is not None


def _fields(request: ModelPlacement) -> dict:
    if request.source == "library":
        if request.model not in FEATURED:
            raise PlaygroundError("select a model offered in the viewer")
        robot = library.get(request.model)
        from robotforge.viewer.progress import download_callback, report

        report("download")
        library.model_dir(robot, download_callback("download"))
        return {"library": request.model}
    path = combine.designs_home() / f"{request.model}.yaml"
    if not path.is_file() or path.stat().st_size > layouts.MAX_WORLD_BYTES:
        raise PlaygroundError("the saved design is missing or too large; import and save it again")
    return combine.entry_fields(request.model)


def _check(pg: Playground, store: WorldStore, version: int, generation: int, original: bytes) -> World:
    if (
        pg.store is not store
        or pg.lessons.lesson is not None
        or pg.session.version != version
        or pg.session.generation != generation
        or store.file.read_bytes() != original
        or pg._play_world_sha256 != layouts.sha(original)
    ):
        raise PlaygroundError("the world changed; select the model and its position again")
    check_supported_restore(pg, "adding or removing a model")
    world = World.from_yaml(original.decode("utf-8"))
    if any(not _eligible(pg.session.model, entry.name) for entry in world.robots):
        raise PlaygroundError("this world's robot state cannot be preserved by model placement")
    return world


def place(pg: Playground, message: dict) -> dict:
    request = _parse(ModelPlacement, message, "model placement")
    from robotforge.viewer.progress import report

    with pg.session.execution.command("viewer_add_model"):
        store = pg._require_store()
        original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
        with pg.lessons._lock, pg.session.lock:
            world = _check(pg, store, request.version, request.generation, original)
        fields = _fields(request)
        names = {entry.name for entries in (world.robots, world.objects, world.terrain) for entry in entries}
        name, number = request.model, 1
        while name in names:
            number += 1
            suffix = f"_{number}"
            name = request.model[: 32 - len(suffix)] + suffix
        entry = RobotEntry.model_validate({**fields, "name": name, "pos": request.pos, "yaw": 0})
        with menagerie.cached_only():
            report("prepare")
            with pg.lessons._lock, pg.session.lock:
                _check(pg, store, request.version, request.generation, original)
                position, platform = _surface_position(pg, world, entry, request.pos)
                platform_name = pg.session.model.geom(platform).name if platform is not None else None
            entry = entry.model_copy(update={"pos": position})
            world.robots.append(entry)
            return _commit(pg, store, request.version, request.generation, original, world, entry, platform_name)


@menagerie.cached_only()
def _commit(
    pg: Playground,
    store: WorldStore,
    version: int,
    generation: int,
    original: bytes,
    candidate: World,
    entry: RobotEntry,
    platform_name: str | None = None,
    *,
    removing: bool = False,
) -> dict:
    from robotforge.viewer.progress import report

    report("compile")
    compiled = compile_world(candidate, store.dir)
    if any(not _eligible(compiled.model, robot.name) for robot in candidate.robots):
        raise PlaygroundError("this model's runtime state cannot be preserved; use open_model to inspect it separately")
    prepared = layouts.PreparedWorld(
        original, candidate.to_yaml().encode("utf-8"), compiled, compiled.to_xml().encode("utf-8")
    )
    if len(prepared.after) > layouts.MAX_WORLD_BYTES:
        raise PlaygroundError("the resulting world is too large")
    model, saved = compiled.model, fresh_data(compiled.model)
    owners = _owners(model, candidate)
    platform = model.geom(platform_name).id if platform_name is not None else None
    blocked = set()
    if not removing:
        blocked.update(placement.blocked(model, saved, entry.name))
        blocked.update(_object_obstacles(model, saved, entry.name, owners))
        _platform_fits(model, saved, entry.name, platform, owners)
    with pg.lessons._lock, pg.session.lock:
        world = _check(pg, store, version, generation, original)
        if removing:
            previous = pg.session.model
            previous_owners = _owners(previous, world)
            supported = _supported(previous, fresh_data(previous), entry.name, previous_owners)
            supported.update(_supported(previous, copy.copy(pg.session.data), entry.name, previous_owners))
            if supported:
                raise PlaygroundError(f"{entry.name} supports {', '.join(sorted(supported))}; move those objects first")
        live = carry.take(pg.session.model, pg.session.data)
        manual = {name for name in pg.behaviors.hands_off() if name != entry.name}
        current = copy.copy(saved)
        carry._apply(model, current, live, lambda name: carry.owner(name) != entry.name)
        mujoco.mj_forward(model, current)
        if not removing:
            blocked.update(placement.blocked(model, current, entry.name))
            blocked.update(_object_obstacles(model, current, entry.name, owners))
            _platform_fits(model, current, entry.name, platform, owners)
        if blocked:
            raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; choose an open spot")
        keep = sorted({r.name for r in world.robots if r.name != entry.name})
        rollback = BuiltinLayoutState.capture(pg)
        report("publish")
        try:
            with layouts._install_world(store, prepared) as notes:
                result = pg._load_world(compiled)
                carry._apply(model, pg.session.data, live, lambda name: carry.owner(name) != entry.name)
                mujoco.mj_forward(model, pg.session.data)
                pg.behaviors.let_go(manual)
                pg.sensors.rebuild(kept=keep)
                pg.perception.rebuild(keep)
                pg.session.paused = True
        except BaseException as exc:
            rollback.restore()
            try:
                recovered = store.file.read_bytes() == original
            except OSError:
                recovered = False
            if not recovered or getattr(exc, "code", None) == "layout_recovery":
                pg.session.paused = True
                pg._play_world_sha256 = None
                raise layouts.LayoutError(
                    "recovery", "model placement could not recover its files; reopen the world"
                ) from exc
            raise
        receipt = (
            None
            if removing
            else ModelAddition(
                store,
                pg.session.version,
                pg.session.generation,
                uuid4().hex,
                "robot",
                entry.name,
                entry.model_dump_json(),
                layouts.sha(prepared.after),
            )
        )
        pg._play_addition = receipt
        pg._play_edit = pg._play_robot_edit = None
        return {
            "removed" if removing else "robot": entry.name,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "notes": [*result["notes"], *notes],
            "undo": {"token": receipt.token, "version": receipt.version, "name": entry.name} if receipt else None,
        }


def undo(pg: Playground, message: dict) -> dict:
    request = _parse(UndoAddition, message, "undo model addition")
    with pg.session.execution.command("viewer_undo_model"):
        receipt = pg._play_addition
        if (
            not isinstance(receipt, ModelAddition)
            or receipt.token != request.token
            or receipt.version != request.version
        ):
            raise PlaygroundError("that model addition is no longer available to undo")
        store = receipt.store
        original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
        with pg.lessons._lock, pg.session.lock:
            world = _check(pg, store, receipt.version, receipt.generation, original)
            if layouts.sha(original) != receipt.world_sha256:
                raise PlaygroundError("the saved world changed; that addition cannot be undone")
        entry = RobotEntry.model_validate_json(receipt.entry)
        if [r for r in world.robots if r.name == entry.name] != [entry]:
            raise PlaygroundError("the added model changed; it cannot be undone")
        if _task_users(world, entry.name) or any(carry.owner(t.robot) == entry.name for t in world.tasks):
            raise PlaygroundError("the added model is used by a task; edit that task first")
        world.robots.remove(entry)
        return _commit(pg, store, receipt.version, receipt.generation, original, world, entry, removing=True)
