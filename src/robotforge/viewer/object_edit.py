"""Guarded rigid-prop edits and copies with a one-use inverse in the local viewer."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from robotforge.sim import carry, placement
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import Coordinate, Heading, _object_obstacles, _parse

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.design.world import ObjectEntry
    from robotforge.playground import Playground

SHAPES = {"sphere", "box", "cylinder", "table", "bin"}
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    generation: int = Field(ge=0)
    world_sha256: Digest


class Reposition(Selection):
    pos: list[Coordinate] = Field(min_length=2, max_length=3)
    yaw: Heading


class UndoEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    version: int = Field(ge=1)
    generation: int = Field(ge=0)


@dataclass(frozen=True)
class Edit:
    store: WorldStore
    version: int
    generation: int
    token: str
    world_sha256: str
    name: str
    before: ObjectEntry | None
    after: ObjectEntry | None
    index: int
    pose: carry.Snapshot


def metadata(pg: Playground) -> dict:
    """Exact object/geom identities, never guessed from a preset-like display name."""
    with pg._edit_lock, pg.lessons._lock, pg.session.lock:
        store = pg.store
        if store is None or pg.lessons.lesson is not None:
            return {"objects": []}
        raw = store.file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != pg._play_world_sha256:
            return {"objects": []}
        from robotforge.design.world import World

        world = World.from_yaml(raw.decode("utf-8"))
        if hashlib.sha256(world.model_dump_json().encode("utf-8")).hexdigest() != pg._play_world_descriptor:
            return {"objects": []}
        model = pg.session.model
        owners = _owners(model, world)
        objects = []
        for entry in world.objects:
            if entry.shape not in SHAPES:
                continue
            geoms = [g for g in range(model.ngeom) if owners[g] == entry.name]
            objects.append(
                {
                    "name": entry.name,
                    "shape": entry.shape,
                    "size": entry.size,
                    "yaw": entry.yaw,
                    "geoms": geoms,
                    **(
                        {"surface_material": entry.surface_material}
                        if entry.shape in {"box", "sphere", "cylinder"}
                        else {}
                    ),
                }
            )
        return {
            "objects": objects,
            "version": pg.session.version,
            "world_sha256": hashlib.sha256(raw).hexdigest(),
        }


def _owners(model, world):
    from robotforge.design.visuals import PRESETS

    owners = {g: placement.owner(model, g) for g in range(model.ngeom)}
    # The compiler's unrotated fixed bins keep their four collision walls on worldbody.
    walls = {f"{o.name}_wall{i}": o.name for o in world.objects if o.shape == "bin" for i in range(1, 5)}
    for g in owners:
        if not model.geom_bodyid[g] and model.geom(g).name in walls:
            owners[g] = walls[model.geom(g).name]
        if model.geom_bodyid[g] or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.mesh(int(model.geom_dataid[g])).name
        # Visuals.finish names each authored mesh rf_<exact owner>_<preset>[group index].
        # Anonymous fixed visual geoms cannot be identified by their worldbody alone.
        matches = set()
        for obj in world.objects:
            for preset in PRESETS:
                base = f"rf_{obj.name}_{preset}"
                tail = mesh.removeprefix(base)
                if mesh.startswith(base) and (not tail or (tail.isdigit() and int(tail) >= 2)):
                    matches.add(obj.name)
        if len(matches) == 1:
            owners[g] = matches.pop()
    return owners


def _bounds(model, data, g):
    rot, box = data.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
    center, half = data.geom_xpos[g] + rot @ box[:3], np.abs(rot) @ box[3:]
    return center - half, center + half


@dataclass
class _SupportContext:
    """One posed state, with lazy upward owner edges and reusable geometry bounds."""

    model: mujoco.MjModel
    data: mujoco.MjData
    owners: dict[int, str]
    solid: list[int]
    boxes: dict[int, tuple[np.ndarray, np.ndarray]]
    owner_geoms: dict[str, list[int]]
    contacts: dict[str, set[str]]
    edges: dict[str, frozenset[str]]

    def supported(self, name: str) -> frozenset[str]:
        if name not in self.edges:
            self.edges[name] = frozenset(_supported_in_state(self, name))
        return self.edges[name]


def _support_context(model, data, owners, *, forwarded=False) -> _SupportContext:
    """Prepare once; callers own the model/data and must keep this state unchanged."""
    if not forwarded:
        mujoco.mj_forward(model, data)
    contacts: dict[str, set[str]] = {}
    for contact in data.contact[: data.ncon]:
        a, b = int(contact.geom1), int(contact.geom2)
        if a < 0 or b < 0 or contact.dist > 0.008 or owners[a] == owners[b]:
            continue
        if contact.frame[2] > 0.2:
            contacts.setdefault(owners[a], set()).add(owners[b])
        if contact.frame[2] < -0.2:
            contacts.setdefault(owners[b], set()).add(owners[a])
    solid = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    owner_geoms: dict[str, list[int]] = {}
    for geom in solid:
        owner_geoms.setdefault(owners[geom], []).append(geom)
    return _SupportContext(
        model, data, owners, solid, {g: _bounds(model, data, g) for g in solid}, owner_geoms, contacts, {}
    )


def _supported_in_state(context: _SupportContext, name: str) -> set[str]:
    model, data, owners = context.model, context.data, context.owners
    solid, boxes = context.solid, context.boxes
    supported = set(context.contacts.get(name, ()))
    for a in context.owner_geoms.get(name, ()):
        low, high = boxes[a]
        for b in solid:
            if owners[b] == name or model.geom_type[b] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            bottom, top = boxes[b]
            if np.any(bottom > high + 0.008) or np.any(low > top + 0.008):
                continue
            points = np.zeros(6)
            if mujoco.mj_geomDistance(model, data, a, b, 0.009, points) >= 0.008:
                continue
            # Native fixed-fixed pairs have no contact normal. Recover the outward normal at
            # the nearest point on our primitive, so tilted tops do not rely on a world AABB top.
            rot = data.geom_xmat[a].reshape(3, 3)
            point = rot.T @ (points[:3] - data.geom_xpos[a])
            kind, size = model.geom_type[a], model.geom_size[a]
            normal = np.zeros(3)
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                ratios = np.abs(point) / np.maximum(size, 1e-12)
                axes = np.flatnonzero(ratios >= ratios.max() - 1e-4)
                # A closest point at an edge can belong to two faces; retain its upward face.
                axis = max(axes, key=lambda i: rot[2, i] * np.sign(point[i]))
                normal[axis] = np.sign(point[axis])
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                normal = point / max(float(np.linalg.norm(point)), 1e-12)
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                if abs(point[2]) / max(size[1], 1e-12) >= np.linalg.norm(point[:2]) / max(size[0], 1e-12):
                    normal[2] = np.sign(point[2])
                else:
                    normal[:2] = point[:2] / max(float(np.linalg.norm(point[:2])), 1e-12)
            if (rot @ normal)[2] > 0.2 and top[2] > points[2] and bottom[2] >= points[2] - 0.008:
                supported.add(owners[b])
    return supported


def _supported(model, data, name, owners):
    """Upward near-contacts, including static pairs missing from the native contact list."""
    return set(_support_context(model, data, owners).supported(name))


def _task_users(world, name):
    users = []
    for task in world.tasks:
        names = [
            task.object,
            task.target,
            *(task.items or []),
            *(task.into or {}),
            *((task.into or {}).values()),
            *task.start.objects,
        ]
        if any(value and value.split("/", 1)[0] == name for value in names):
            users.append(task.name)
    return users


def edit(pg: Playground, message: dict) -> dict:
    from robotforge.playground import PlaygroundError

    removing = message.get("action") == "remove_object"
    request = _parse(Selection if removing else Reposition, message, "object edit")
    with pg.session.execution.command("viewer_edit_object"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            world = _guard(pg, store, request.version, request.generation, request.world_sha256)
            matches = [o for o in world.objects if o.name == request.name and o.shape in SHAPES]
            if len(matches) != 1:
                raise PlaygroundError("select an existing editable prop")
            before = matches[0]
            if users := _task_users(world, before.name):
                raise PlaygroundError(f"{before.name} is used by task {', '.join(users)}; edit the task first")
        after = None
        if not removing:
            after = before.model_copy(update={"pos": _surface_pos(world, before, request.pos), "yaw": request.yaw})
        index = world.objects.index(before)
        return _commit(pg, store, request, before, after, index)


def _surface_pos(world, entry, position):
    from robotforge.design import terrain

    pos = position[:2]
    if len(position) == 3:
        ground = terrain.Ground.build(world.terrain, world.floor.style)
        if abs(position[2] - ground.height(*pos)) > 0.001:
            pos = [*pos, position[2] + (0 if entry.shape in {"table", "bin"} else entry.size[-1] / 2)]
    return pos


def duplicate(pg: Playground, message: dict) -> dict:
    """Copy server-owned authored settings, leaving the selected source and its task bindings alone."""
    from robotforge.design.world import ObjectEntry
    from robotforge.playground import PlaygroundError

    request = _parse(Reposition, message, "object copy")
    with pg.session.execution.command("viewer_duplicate_object"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            world = _guard(pg, store, request.version, request.generation, request.world_sha256)
            source = next((o for o in world.objects if o.name == request.name and o.shape in SHAPES), None)
            if source is None:
                raise PlaygroundError("select an existing editable prop")
        taken = {entry.name for entries in (world.robots, world.objects, world.terrain) for entry in entries}
        n = 2
        while True:
            suffix = f"_{n}"
            name = source.name[: 32 - len(suffix)] + suffix
            if name not in taken:
                break
            n += 1
        clone = ObjectEntry.model_validate(
            {**source.model_dump(), "name": name, "pos": _surface_pos(world, source, request.pos), "yaw": request.yaw}
        )
        return _commit(pg, store, request, None, clone, len(world.objects))


def _guard(pg, store, version, generation, digest, subject="prop"):
    from robotforge.design.world import World
    from robotforge.playground import PlaygroundError

    raw = store.file.read_bytes()
    if (
        pg.store is not store
        or pg.lessons.lesson is not None
        or pg.session.version != version
        or pg.session.generation != generation
        or pg._play_world_sha256 != digest
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise PlaygroundError(f"the simulation changed; select the {subject} again")
    world = World.from_yaml(raw.decode("utf-8"))
    if hashlib.sha256(world.model_dump_json().encode("utf-8")).hexdigest() != pg._play_world_descriptor:
        raise PlaygroundError("the saved world differs from the running model; reload the world")
    return world


def _commit(pg, store, request, before, after, index, restore=None):
    from robotforge.design.world import World, compile_world
    from robotforge.playground import PlaygroundError

    name = before.name if before is not None else after.name
    original = store.file.read_bytes()
    if hashlib.sha256(original).hexdigest() != request.world_sha256:
        raise PlaygroundError("the saved world changed; select the prop again")
    world = World.from_yaml(original.decode("utf-8"))

    def change(candidate):
        if store.file.read_bytes() != original:
            raise PlaygroundError("the saved world changed; select the prop again")
        found = next((o for o in candidate.objects if o.name == name), None)
        if found != before:
            raise PlaygroundError("the selected prop changed; select it again")
        if found is not None:
            candidate.objects.remove(found)
        if after is not None:
            candidate.objects.insert(index, after)

    candidate = world.model_copy(deep=True)
    change(candidate)
    compiled = compile_world(candidate, store.dir)
    saved = fresh_data(compiled.model)
    blocked = (
        set(_object_obstacles(compiled.model, saved, name, _owners(compiled.model, candidate)))
        if after is not None
        else set()
    )
    # Original saved support must remain valid on reset as well as in the running scene.
    if before is not None:
        old = compile_world(world, store.dir)
        supported = _supported(old.model, fresh_data(old.model), name, _owners(old.model, world))
    else:
        supported = set()
    with pg.lessons._lock, pg.session.lock:
        _guard(pg, store, request.version, request.generation, request.world_sha256)
        if before is not None:
            supported.update(_supported(pg.session.model, pg.session.data, name, _owners(pg.session.model, world)))
        if supported:
            raise PlaygroundError(f"{name} supports {', '.join(sorted(supported))}; move those objects first")
        live = carry.take(pg.session.model, pg.session.data)
        current = copy.copy(saved)
        carry._apply(compiled.model, current, live, lambda joint: carry.owner(joint) != name)
        target = restore if restore is not None else carry.take(compiled.model, saved)
        carry._apply(compiled.model, current, target, lambda joint: carry.owner(joint) == name)
        if after is not None:
            blocked.update(_object_obstacles(compiled.model, current, name, _owners(compiled.model, candidate)))
        if blocked:
            raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; choose an open spot")
        result = pg._edit(change, exact_carry=True)
        carry._apply(pg.session.model, pg.session.data, target, lambda joint: carry.owner(joint) == name)
        mujoco.mj_forward(pg.session.model, pg.session.data)
        pg._play_addition = None
        undo = None
        if restore is None:
            pose = carry.Snapshot(free={k: v for k, v in live.free.items() if carry.owner(k) == name})
            receipt = Edit(
                store,
                pg.session.version,
                pg.session.generation,
                uuid4().hex,
                hashlib.sha256(store.file.read_bytes()).hexdigest(),
                name,
                before,
                after,
                index,
                pose,
            )
            pg._play_edit = receipt
            undo = {"token": receipt.token, "version": receipt.version, "generation": receipt.generation}
        else:
            pg._play_edit = None
        return {
            "removed" if after is None else "object": name,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "notes": result["notes"],
            "undo_edit": undo,
        }


def undo(pg: Playground, message: dict) -> dict:
    from robotforge.playground import PlaygroundError

    request = _parse(UndoEdit, message, "undo object edit")
    with pg.session.execution.command("viewer_undo_object_edit"):
        receipt = pg._play_edit
        if not isinstance(receipt, Edit) or receipt.token != request.token:
            raise PlaygroundError("that object edit is no longer available to undo")
        with pg.lessons._lock, pg.session.lock:
            _guard(pg, receipt.store, request.version, request.generation, receipt.world_sha256)
            if request.version != receipt.version or request.generation != receipt.generation:
                raise PlaygroundError("that object edit is no longer available to undo")
        selection = Selection(
            name=receipt.name, version=request.version, generation=request.generation, world_sha256=receipt.world_sha256
        )
        return _commit(pg, receipt.store, selection, receipt.after, receipt.before, receipt.index, receipt.pose)
