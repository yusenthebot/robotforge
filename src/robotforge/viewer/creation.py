"""Local, click-to-place starters. No file paths, downloads or arbitrary builder input."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.design.world import ObjectEntry, RobotEntry, World
    from robotforge.playground import Playground

STARTERS = ("rover", "quadruped", "arm", "humanoid")
Coordinate = Annotated[float, Field(ge=-100, le=100, allow_inf_nan=False)]


class Placement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    template: Literal["rover", "quadruped", "arm", "humanoid"]
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    version: int = Field(ge=1)


class ObjectPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    preset: Literal["ball", "block", "pin", "barrier", "table"]
    size: Literal["small", "normal", "large"] = "normal"
    yaw: Literal[0, 90, 180, 270] = 0
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    version: int = Field(ge=1)

    @field_validator("yaw", mode="before")
    @classmethod
    def integer_turn(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("yaw must be an integer quarter turn")
        return value


class UndoAddition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: int = Field(ge=1)


@dataclass(frozen=True)
class Addition:
    store: WorldStore
    version: int
    generation: int
    token: str
    kind: str
    name: str
    entry: str
    world_sha256: str


SCALES = {"small": 0.75, "normal": 1.0, "large": 1.5}
OBJECTS = {
    "ball": {"shape": "sphere", "size": [0.18], "color": "orange", "mass": 0.12, "material": "plastic"},
    "block": {"shape": "box", "size": [0.18, 0.18, 0.18], "color": "#b88f5c", "mass": 0.6, "material": "wood"},
    "pin": {"shape": "cylinder", "size": [0.10, 0.24], "color": "blue", "mass": 0.15, "material": "plastic"},
    "barrier": {"shape": "box", "size": [0.8, 0.12, 0.22], "color": "#b88f5c", "fixed": True, "material": "wood"},
    "table": {"shape": "table", "size": [0.8, 0.5, 0.5], "color": "#b88f5c", "fixed": True},
}


def object_catalog() -> list[dict]:
    # Only preview dimensions, never arbitrary shape/build parameters from the client.
    return [{"name": name, "shape": spec["shape"], "size": list(spec["size"])} for name, spec in OBJECTS.items()]


def _parse[Model: BaseModel](schema: type[Model], message: dict, label: str) -> Model:
    from robotforge.playground import PlaygroundError

    try:
        return schema.model_validate({k: v for k, v in message.items() if k not in {"type", "action", "id"}})
    except ValidationError as exc:
        raise PlaygroundError(f"invalid {label}: {exc}") from None


def catalog() -> list[dict[str, str]]:
    from robotforge.design.templates import TEMPLATES

    return [{"name": name, "summary": TEMPLATES[name].summary} for name in STARTERS]


def place(pg: Playground, message: dict) -> dict:
    """Called with the playground edit lock held; preflight never publishes a temporary world."""
    from robotforge.design import terrain
    from robotforge.design.world import RobotEntry, spawn_height

    request = _parse(Placement, message, "robot placement")

    def entry(world: World, name: str) -> RobotEntry:
        result = RobotEntry(name=name, template=request.template, pos=request.pos)
        if len(request.pos) == 3:
            ground = terrain.Ground.build(world.terrain, world.floor.style)
            on_ground = abs(request.pos[2] - ground.height(*request.pos[:2])) <= 0.001
            pos = request.pos[:2] if on_ground else [*request.pos[:2], request.pos[2] + spawn_height(result)]
            result = result.model_copy(update={"pos": pos})
        return result

    return _place(pg, request, "robot", "dog" if request.template == "quadruped" else request.template, entry)


def place_object(pg: Playground, message: dict) -> dict:
    from robotforge.design import terrain
    from robotforge.design.world import ObjectEntry

    request = _parse(ObjectPlacement, message, "object placement")

    def entry(world: World, name: str) -> ObjectEntry:
        spec = {**OBJECTS[request.preset]}
        scale = SCALES[request.size]
        spec["size"] = [value * scale for value in spec["size"]]
        if "mass" in spec:
            spec["mass"] *= scale**3
        pos = request.pos[:2]
        if len(request.pos) == 3:
            ground = terrain.Ground.build(world.terrain, world.floor.style)
            if abs(request.pos[2] - ground.height(*pos)) > 0.001:
                height = spec["size"][-1]
                pos = [*pos, request.pos[2] + (0 if spec["shape"] == "table" else height / 2)]
        return ObjectEntry(name=name, **spec, pos=pos, yaw=request.yaw)

    return _place(pg, request, "object", request.preset, entry)


def _place(
    pg: Playground,
    request: Placement | ObjectPlacement,
    kind: str,
    base: str,
    make_entry: Callable[[World, str], RobotEntry | ObjectEntry],
) -> dict:
    import copy

    import mujoco

    from robotforge.design.world import compile_world
    from robotforge.playground import PlaygroundError, _unused
    from robotforge.sim import carry, placement
    from robotforge.sim.session import fresh_data

    if pg.lessons.lesson is not None:
        raise PlaygroundError("leave the lesson before adding robots or objects")
    store = pg._require_store()
    with pg.session.execution.command(f"viewer_add_{kind}"):
        with pg.session.lock:
            if pg.session.version != request.version:
                raise PlaygroundError("the scene changed; choose a spot in the current scene")
            generation = pg.session.generation
        saved_world = store.file.read_bytes()
        world = store.load()
        taken = {r.name for r in world.robots} | {o.name for o in world.objects} | {p.name for p in world.terrain}
        name = _unused(base, taken)
        candidate = world.model_copy(deep=True)
        entry = make_entry(world, name)
        getattr(candidate, f"{kind}s").append(entry)
        compiled = compile_world(candidate, store.dir)
        saved = fresh_data(compiled.model)
        occupied = placement.blocked if kind == "robot" else _object_obstacles
        blocked = set(occupied(compiled.model, saved, name))
        # Lessons take their runner lock before the session lock when swapping models.
        with pg.lessons._lock, pg.session.lock:
            if (
                pg.store is not store
                or pg.lessons.lesson is not None
                or pg.session.version != request.version
                or pg.session.generation != generation
                or store.file.read_bytes() != saved_world
            ):
                raise PlaygroundError("the simulation changed; choose a spot again")
            current = copy.copy(saved)
            live = carry.take(pg.session.model, pg.session.data)
            # Unlike carry.restore, preflight must not move an old robot out of a clash.
            # Read live poses at commit: the viewer can keep stepping while compilation runs.
            keep = {r.name for r in world.robots} | {o.name for o in world.objects}
            carry._apply(compiled.model, current, live, lambda joint: carry.owner(joint) in keep)
            mujoco.mj_forward(compiled.model, current)
            blocked.update(occupied(compiled.model, current, name))
            if blocked:
                raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; choose an open spot")
            if kind == "robot":
                result = pg.add_robot(name, template=request.template, pos=entry.pos)
            else:
                result = pg._edit(lambda w: w.objects.append(entry), exact_carry=True)
            receipt = Addition(
                store,
                pg.session.version,
                pg.session.generation,
                uuid4().hex,
                kind,
                name,
                entry.model_dump_json(),
                hashlib.sha256(store.file.read_bytes()).hexdigest(),
            )
            pg._play_addition = receipt
            return {
                kind: name,
                "version": pg.session.version,
                "notes": result["notes"],
                "undo": {"token": receipt.token, "version": receipt.version, "name": name},
            }


def _object_obstacles(model, data, name: str) -> dict[str, float]:
    """Explicit distances include fixed-fixed pairs that MuJoCo's contact generation omits."""
    import mujoco
    import numpy as np

    from robotforge.sim.placement import owner

    mujoco.mj_forward(model, data)
    solid = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    mine = [g for g in solid if owner(model, g) == name]
    others = [g for g in solid if owner(model, g) != name and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    boxes = {}
    for g in [*mine, *others]:
        rot, local = data.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
        center, half = data.geom_xpos[g] + rot @ local[:3], np.abs(rot) @ local[3:]
        boxes[g] = center - half, center + half
    blocked = {}
    for a in mine:
        for b in others:
            if np.any(boxes[a][0] > boxes[b][1]) or np.any(boxes[b][0] > boxes[a][1]):
                continue
            distance = mujoco.mj_geomDistance(model, data, a, b, 0.01, None)
            if abs(distance) <= 0.002:
                # Native CCD can report zero for deeply overlapping coincident boxes/cylinders.
                # Query libccd on this private candidate only; never change the committed model's options.
                flags = model.opt.disableflags
                try:
                    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                    distance = mujoco.mj_geomDistance(model, data, a, b, 0.01, None)
                finally:
                    model.opt.disableflags = flags
            if distance < -0.002:
                other = owner(model, b)
                blocked[other] = max(blocked.get(other, 0.0), -distance)
    return blocked


def undo_addition(pg: Playground, message: dict) -> dict:
    from robotforge.playground import PlaygroundError

    request = _parse(UndoAddition, message, "undo add")
    with pg.session.execution.command("viewer_undo_add"), pg.lessons._lock, pg.session.lock:
        receipt = pg._play_addition
        if receipt is None or request.token != receipt.token:
            raise PlaygroundError("that addition is no longer available to undo")
        if (
            pg.store is not receipt.store
            or pg.lessons.lesson is not None
            or pg.session.version != request.version
            or pg.session.version != receipt.version
            or pg.session.generation != receipt.generation
            or hashlib.sha256(receipt.store.file.read_bytes()).hexdigest() != receipt.world_sha256
        ):
            pg._play_addition = None
            raise PlaygroundError("the simulation changed; that addition can no longer be undone")

        def change(world: World) -> None:
            entries = getattr(world, f"{receipt.kind}s")
            matches = [e for e in entries if e.name == receipt.name and e.model_dump_json() == receipt.entry]
            if len(matches) != 1:
                raise PlaygroundError("the added entity changed; it cannot be undone")
            entries.remove(matches[0])

        # This is an exact inverse edit, not a history pop that might discard another writer's work.
        result = pg._edit(change, exact_carry=True)
        pg._play_addition = None
        return {"removed": receipt.name, "version": pg.session.version, "notes": result["notes"]}
