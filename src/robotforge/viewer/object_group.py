"""Transient, server-owned rigid layouts: one compile, one saved edit, one guarded inverse."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from robotforge.design.store import ApplyArtifacts, _directory_identity
from robotforge.design.world import ObjectEntry
from robotforge.playground import PlaygroundError
from robotforge.sim import carry
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import Coordinate, Heading, _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.object_edit import SHAPES, Digest, UndoEdit, _bounds, _guard, _owners, _supported

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.design.world import World
    from robotforge.playground import Playground

CONTACT = 0.008
PENETRATION = 0.002
UPRIGHT = 1e-4


class Guard(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: int = Field(ge=1)
    generation: int = Field(ge=0)
    world_sha256: Digest


class Prepare(Guard):
    names: list[str] = Field(min_length=2, max_length=16)

    @field_validator("names")
    @classmethod
    def exact_unique_names(cls, names: list[str]) -> list[str]:
        if len(set(names)) != len(names) or any(not name or len(name) > 32 for name in names):
            raise ValueError("select 2 to 16 distinct exact prop names")
        return names


class Duplicate(Guard):
    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    pos: list[Coordinate] = Field(min_length=3, max_length=3)
    yaw: Heading


@dataclass(frozen=True)
class Member:
    entry: ObjectEntry
    pos: tuple[float, float, float]
    yaw: float
    geoms: tuple[int, ...]


@dataclass(frozen=True)
class PreparedGroup:
    store: WorldStore
    version: int
    generation: int
    world_sha256: str
    token: str
    pose_sha256: str
    anchor: tuple[float, float, float]
    members: tuple[Member, ...]


@dataclass(frozen=True)
class GroupCopy:
    store: WorldStore
    version: int
    generation: int
    world_sha256: str
    token: str
    entries: tuple[ObjectEntry, ...]


def _check(pg: Playground, store: WorldStore, guard: Guard) -> World:
    world = _guard(pg, store, guard.version, guard.generation, guard.world_sha256, "layout")
    if not pg.session.paused:
        raise PlaygroundError("pause the simulation before copying a layout")
    check_supported_restore(pg, "copying a prop layout")
    if pg.session.model.nflex:
        raise PlaygroundError("layout copying needs a rigid world; choose another map")
    return world


def _solid(model) -> list[int]:
    return [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]


def _pair_enabled(model, a: int, b: int) -> bool:
    return bool(
        (model.geom_contype[a] & model.geom_conaffinity[b]) or (model.geom_contype[b] & model.geom_conaffinity[a])
    )


def _members(model, data, world, names) -> tuple[Member, ...]:
    owners = _owners(model, world)
    members = []
    for name in names:
        entry = next((obj for obj in world.objects if obj.name == name and obj.shape in SHAPES), None)
        if entry is None:
            raise PlaygroundError("select 2 to 16 existing editable rigid props")
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise PlaygroundError(f"{name} has no editable collision geometry")
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0:
            pos, rot = data.xpos[body].copy(), data.xmat[body].reshape(3, 3)
        else:
            pos, rot = data.geom_xpos[geom].copy(), data.geom_xmat[geom].reshape(3, 3)
            if entry.shape == "bin":
                # The unrotated fixed bin has no body: its named geom is the 5 mm base.
                pos -= rot[:, 2] * model.geom_size[geom, 2]
        if not np.all(np.isfinite(pos)) or np.linalg.norm(rot[:, 2] - [0, 0, 1]) > UPRIGHT:
            raise PlaygroundError(f"{name} is tilted; make every selected prop upright before copying")
        yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        members.append(
            Member(
                entry.model_copy(deep=True),
                tuple(float(v) for v in pos),
                yaw,
                tuple(g for g in owners if owners[g] == name),
            )
        )
    return tuple(members)


def _pose_digest(model, data, members: tuple[Member, ...]) -> str:
    digest = hashlib.sha256()
    # Include all selected geometry, not only free joints: fixed bins have anonymous walls.
    for member in members:
        digest.update(member.entry.name.encode("utf-8"))
        for geom in member.geoms:
            digest.update(np.asarray(data.geom_xpos[geom], dtype="<f8").tobytes())
            digest.update(np.asarray(data.geom_xmat[geom], dtype="<f8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Foot:
    geom: int
    points: np.ndarray
    radius: float = 0.0


def _feet(model, data, name, owners, solid) -> list[Foot]:
    geoms = [g for g in solid if owners[g] == name]
    bottom = min(_bounds(model, data, g)[0][2] for g in geoms)
    feet = []
    for geom in geoms:
        low, _ = _bounds(model, data, geom)
        if abs(low[2] - bottom) > CONTACT:
            continue
        # NumPy scalar equality with MuJoCo enums is asymmetric in tuple membership.
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        rot, center = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            # A full rectangular face fits a convex support exactly iff all four corners fit.
            points = np.array(
                [center + rot @ [x * size[0], y * size[1], -size[2]] for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
            )
            feet.append(Foot(geom, points))
        elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            height = size[0] if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size[1]
            radius = 0.0 if kind == mujoco.mjtGeom.mjGEOM_SPHERE else float(size[0])
            feet.append(Foot(geom, np.array([center - rot[:, 2] * height]), radius))
        else:
            raise PlaygroundError(f"{name} has an unsupported contact footprint")
    if not feet:
        raise PlaygroundError(f"{name} has no supported base")
    return feet


def _on_top(model, data, foot: Foot, support: int) -> float | None:
    """Exact containment on one horizontal primitive face; no AABB or collision-only support guess."""
    kind, size = model.geom_type[support], model.geom_size[support]
    rot, center = data.geom_xmat[support].reshape(3, 3), data.geom_xpos[support]
    if np.linalg.norm(rot[:, 2] - [0, 0, 1]) > UPRIGHT:
        return None
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        top = 0.0
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        top = float(size[2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        top = float(size[1])
    else:
        return None
    local = (foot.points - center) @ rot
    gaps = local[:, 2] - top
    if np.any(gaps < -PENETRATION) or np.any(gaps > CONTACT):
        return None
    # Micrometre tolerance only, so a partially overhanging foot cannot use the 8 mm contact tolerance.
    if kind == mujoco.mjtGeom.mjGEOM_BOX and np.any(np.abs(local[:, :2]) + foot.radius > size[:2] + 1e-6):
        return None
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER and np.any(
        np.linalg.norm(local[:, :2], axis=1) + foot.radius > size[0] + 1e-6
    ):
        return None
    return float(center[2] + top)


def _support_graph(model, data, world, names: set[str], *, target: bool = False) -> float:
    owners, solid = _owners(model, world), _solid(model)
    edges: dict[str, set[str]] = {name: set() for name in names}
    external: dict[str, list[tuple[str, float]]] = {name: [] for name in names}
    for name in names:
        for foot in _feet(model, data, name, owners, solid):
            supports = [
                (owners[g], height)
                for g in solid
                if owners[g] != name and _pair_enabled(model, foot.geom, g)
                if (height := _on_top(model, data, foot, g)) is not None
            ]
            internal = [owner for owner, _ in supports if owner in names]
            if internal:
                edges[name].update(internal)
            elif supports:
                external[name].append(supports[0])
            else:
                raise PlaygroundError(f"{name} is not fully supported on a horizontal surface; include its support")
        # One object partially supported by another selected object and partly by the outside
        # cannot be represented as an independent root in a preserved layout.
        if edges[name] and external[name]:
            raise PlaygroundError(f"{name} spans selected and external supports; select its full support layout")

    def roots(name, seen):
        if name in seen:
            raise PlaygroundError("the selected layout has a cyclic support relationship")
        if not edges[name]:
            return external[name]
        return [root for below in edges[name] for root in roots(below, seen | {name})]

    contacts = [root for name in names for root in roots(name, set())]
    if not contacts or max(height for _, height in contacts) - min(height for _, height in contacts) > PENETRATION:
        raise PlaygroundError("the layout roots must share one horizontal support height")
    if target and len({owner for owner, _ in contacts}) != 1:
        raise PlaygroundError("place every layout root on the same horizontal support")
    return float(sum(height for _, height in contacts) / len(contacts))


def _collisions(model, data, world, names: set[str]) -> None:
    """Selected-selected and selected-external penetration, including fixed pairs and planes."""
    mujoco.mj_forward(model, data)
    owners, solid = _owners(model, world), _solid(model)
    boxes = {g: _bounds(model, data, g) for g in solid}
    selected = [g for g in solid if owners[g] in names]
    selected_set = set(selected)
    for a in selected:
        for b in solid:
            if owners[a] == owners[b] or (b in selected_set and b <= a):
                continue
            if not _pair_enabled(model, a, b):
                continue
            plane = next((g for g in (a, b) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None)
            if plane is not None:
                other = b if plane == a else a
                # For horizontal planes this is exact even for spheres and cylinders.
                normal = data.geom_xmat[plane].reshape(3, 3)[:, 2]
                if np.linalg.norm(normal - [0, 0, 1]) <= UPRIGHT:
                    distance = boxes[other][0][2] - data.geom_xpos[plane, 2]
                else:
                    distance = mujoco.mj_geomDistance(model, data, a, b, 0.01, None)
            else:
                if np.any(boxes[a][0] > boxes[b][1]) or np.any(boxes[b][0] > boxes[a][1]):
                    continue
                distance = mujoco.mj_geomDistance(model, data, a, b, 0.01, None)
                if abs(distance) <= PENETRATION:
                    flags = model.opt.disableflags
                    try:
                        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_NATIVECCD
                        distance = mujoco.mj_geomDistance(model, data, a, b, 0.01, None)
                    finally:
                        model.opt.disableflags = flags
            if distance < -PENETRATION - 1e-9:
                raise PlaygroundError(f"layout occupied: {owners[a]} overlaps {owners[b]}; choose a clear spot")


def _prepare(pg, request: Prepare):
    store = pg._require_store()
    world = _check(pg, store, request)
    # All queries use private model/data: collision fallback and forward never touch the live simulation.
    model, data = copy.copy(pg.session.model), copy.copy(pg.session.data)
    mujoco.mj_forward(model, data)
    members = _members(model, data, world, request.names)
    names = set(request.names)
    _collisions(model, data, world, names)
    height = _support_graph(model, data, world, names)
    anchor = (
        sum(member.pos[0] for member in members) / len(members),
        sum(member.pos[1] for member in members) / len(members),
        height,
    )
    prepared = PreparedGroup(
        store,
        request.version,
        request.generation,
        request.world_sha256,
        uuid4().hex,
        _pose_digest(model, data, members),
        anchor,
        members,
    )
    pg._play_group = prepared
    return {
        "token": prepared.token,
        "pose_sha256": prepared.pose_sha256,
        "anchor": list(anchor),
        "version": request.version,
        "generation": request.generation,
        "world_sha256": request.world_sha256,
        "members": [
            {
                "name": member.entry.name,
                "shape": member.entry.shape,
                "size": member.entry.size,
                "pos": list(member.pos),
                "yaw": member.yaw,
                "geoms": list(member.geoms),
                "offset": [member.pos[i] - anchor[i] for i in range(3)],
            }
            for member in members
        ],
    }


def _check_prepared(pg, request, prepared):
    if not isinstance(prepared, PreparedGroup) or (
        prepared.token,
        prepared.version,
        prepared.generation,
        prepared.world_sha256,
    ) != (request.token, request.version, request.generation, request.world_sha256):
        raise PlaygroundError("that layout preparation is no longer available; select the props again")
    world = _check(pg, prepared.store, request)
    data = copy.copy(pg.session.data)
    mujoco.mj_forward(pg.session.model, data)
    if _pose_digest(pg.session.model, data, prepared.members) != prepared.pose_sha256:
        raise PlaygroundError("the selected layout pose changed; select the props again")
    return world


def _clones(world, prepared, request) -> tuple[ObjectEntry, ...]:
    taken = {entry.name for entries in (world.objects, world.robots, world.terrain) for entry in entries}
    angle = math.radians(request.yaw)
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    clones = []
    for member in prepared.members:
        count = 2
        while True:
            suffix = f"_{count}"
            name = member.entry.name[: 32 - len(suffix)] + suffix
            if name not in taken:
                break
            count += 1
        taken.add(name)
        xy = rot @ (np.array(member.pos[:2]) - prepared.anchor[:2]) + request.pos[:2]
        pos = [float(xy[0]), float(xy[1]), member.pos[2] + request.pos[2] - prepared.anchor[2]]
        if not all(math.isfinite(value) and abs(value) <= 100 for value in pos):
            raise PlaygroundError("the copied layout extends beyond the supported coordinates")
        clones.append(
            ObjectEntry.model_validate(
                {**member.entry.model_dump(), "name": name, "pos": pos, "yaw": member.yaw + request.yaw}
            )
        )
    return tuple(clones)


@dataclass(frozen=True)
class DiskState:
    files: dict[Path, bytes | None]
    directories: dict[Path, tuple[int, int, int] | None]


def _disk_state(store):
    paths = [store.file, store.build_dir / "world.xml", *store.history.glob("*.yaml")]
    if any(path.is_symlink() for path in paths) or store.history.is_symlink() or store.build_dir.is_symlink():
        raise PlaygroundError("layout editing needs regular world, build and history files")
    return DiskState(
        {path: path.read_bytes() if path.exists() else None for path in (store.file, store.build_dir / "world.xml")},
        {path: _directory_identity(path) for path in (*store.dir.parents, store.dir, store.build_dir, store.history)},
    )


def _restore_disk(store, saved, candidate_world, candidate_xml, artifacts):
    """Restore only bytes owned by this transaction, preserving an outside writer's work."""

    def same_directory(path):
        expected = saved.directories[path]
        if expected is None:
            expected = artifacts.created_directories.get(path)
        return _directory_identity(path) == expected

    # Check every ancestor before touching any child: checking world.xml itself does
    # not prevent following a replaced build/.history directory into an outside tree.
    same_root = all(same_directory(path) for path in (*store.dir.parents, store.dir))
    external_world = not same_root
    for path, candidate in ((store.file, candidate_world), (store.build_dir / "world.xml", candidate_xml)):
        if not same_root or (path.parent == store.build_dir and not same_directory(store.build_dir)):
            external_world = True
            continue
        # A replacement symlink or missing authoritative file also belongs to the outside
        # writer. Never follow it or recreate it as part of this failed transaction.
        current = path.read_bytes() if path.exists() and not path.is_symlink() else None
        original = saved.files[path]
        owned = not path.is_symlink() and current in (original, candidate)
        if path == store.file:
            external_world = not owned
        if not owned or current == original:
            continue
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(original)
    if not same_root or not same_directory(store.history):
        return True
    for path, contents in artifacts.created_history.items():
        if path.exists() and not path.is_symlink() and path.read_bytes() == contents:
            path.unlink()
    # Only a recorded successful prune belongs to this edit. Missing older records
    # that the store never removed were deleted externally and must stay deleted.
    for path, raw in artifacts.pruned_history.items():
        if not path.exists() and not path.is_symlink():
            path.write_bytes(raw)
    return external_world


def _commit(pg, store, guard, change, validate, entries, *, undoing=False):
    """WorldStore owns the sole compile; post-compile validation precedes every persisted byte."""
    original = store.file.read_bytes()
    original_model, original_data = pg.session.model, pg.session.data
    original_arrays = {
        name: getattr(original_data, name).copy() for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")
    }
    original_time, original_steps = original_data.time, pg.session.steps_taken
    rollback = None
    disk = None
    preparation = pg._play_group
    candidate_bytes = None
    candidate_xml = None
    artifacts = ApplyArtifacts()

    def same_runtime():
        return (
            pg.session.model is original_model
            and pg.session.data is original_data
            and pg.session.data.time == original_time
            and pg.session.steps_taken == original_steps
            and all(np.array_equal(getattr(pg.session.data, name), values) for name, values in original_arrays.items())
        )

    def post_compile(candidate, compiled):
        nonlocal rollback, disk, candidate_bytes, candidate_xml
        _check(pg, store, guard)
        if not same_runtime():
            raise PlaygroundError("the simulation pose changed during preparation; select the layout again")
        validate(candidate, compiled)
        _check(pg, store, guard)
        if store.file.read_bytes() != original or not same_runtime():
            raise PlaygroundError("the world or pose changed during validation; select the layout again")
        candidate_bytes = candidate.to_yaml().encode("utf-8")
        candidate_xml = compiled.to_xml().encode("utf-8")
        disk = _disk_state(store)
        rollback = BuiltinLayoutState.capture(pg)

    try:
        result = pg._edit(change, exact_carry=True, post_compile=post_compile, artifacts=artifacts)
        pg.session.paused = True
        pg._places = None
        pg._play_group = None
        receipt = (
            None
            if undoing
            else GroupCopy(
                store,
                pg.session.version,
                pg.session.generation,
                hashlib.sha256(store.file.read_bytes()).hexdigest(),
                uuid4().hex,
                entries,
            )
        )
        pg._play_edit = receipt
        return {
            "removed" if undoing else "objects": [entry.name for entry in entries],
            "version": pg.session.version,
            "generation": pg.session.generation,
            "paused": True,
            "notes": result["notes"],
            "undo_edit": None
            if receipt is None
            else {"token": receipt.token, "version": receipt.version, "generation": receipt.generation},
        }
    except BaseException:
        if rollback is not None:
            rollback.restore()
            pg._play_group = preparation
            try:
                if _restore_disk(store, disk, candidate_bytes, candidate_xml, artifacts):
                    # The restored runtime no longer describes the external authoritative
                    # file. Require a reload even if that file happens to be valid YAML.
                    pg.session.paused = True
                    pg._play_world_sha256 = None
                    pg._play_world_descriptor = None
                    pg._play_group = None
            except OSError as recovery:
                pg.session.paused = True
                pg._play_world_sha256 = None
                pg._play_world_descriptor = None
                pg._play_group = None
                raise PlaygroundError("layout rollback failed; reopen the world before continuing") from recovery
        raise


def _duplicate(pg, request: Duplicate):
    prepared = pg._play_group
    world = _check_prepared(pg, request, prepared)
    entries = _clones(world, prepared, request)
    names = {entry.name for entry in entries}

    def change(candidate):
        _check_prepared(pg, request, prepared)
        if any(entry.name in {obj.name for obj in candidate.objects} for entry in entries):
            raise PlaygroundError("the copied prop names are no longer available")
        candidate.objects.extend(entries)

    def validate(candidate, compiled):
        _check_prepared(pg, request, prepared)
        saved = fresh_data(compiled.model)
        current = copy.copy(saved)
        live = carry.take(pg.session.model, pg.session.data)
        carry._apply(compiled.model, current, live, lambda name: carry.owner(name) not in names)
        mujoco.mj_forward(compiled.model, current)
        for data in (saved, current):
            _collisions(compiled.model, data, candidate, names)
            height = _support_graph(compiled.model, data, candidate, names, target=True)
            if abs(height - request.pos[2]) > PENETRATION:
                raise PlaygroundError("the target point is not on the layout's horizontal support")

    return _commit(pg, prepared.store, request, change, validate, entries)


def _undo(pg, request: UndoEdit):
    receipt = pg._play_edit
    if not isinstance(receipt, GroupCopy) or (receipt.token, receipt.version, receipt.generation) != (
        request.token,
        request.version,
        request.generation,
    ):
        raise PlaygroundError("that layout copy is no longer available to undo")
    guard = Guard(version=request.version, generation=request.generation, world_sha256=receipt.world_sha256)
    world = _check(pg, receipt.store, guard)
    names = {entry.name for entry in receipt.entries}
    if tuple(obj for obj in world.objects if obj.name in names) != receipt.entries:
        raise PlaygroundError("the copied props changed; this layout copy can no longer be undone")

    def no_external_load():
        model = copy.copy(pg.session.model)
        owners = _owners(model, world)
        for data in (fresh_data(model), copy.copy(pg.session.data)):
            supported = set().union(*(_supported(model, data, name, owners) for name in names)) - names
            if supported:
                raise PlaygroundError(
                    f"the copied layout supports {', '.join(sorted(supported))}; move those objects first"
                )

    no_external_load()

    def change(candidate):
        candidate.objects[:] = [obj for obj in candidate.objects if obj.name not in names]

    def validate(candidate, compiled):
        _check(pg, receipt.store, guard)
        no_external_load()

    return _commit(pg, receipt.store, guard, change, validate, receipt.entries, undoing=True)


def handle(pg: Playground, message: dict) -> dict:
    action = message.get("action")
    schema = Prepare if action == "prepare_object_group" else UndoEdit if action == "undo_object_edit" else Duplicate
    request = _parse(schema, message, "prop layout")
    with pg.session.execution.command("viewer_prop_layout"), pg.lessons._lock, pg.session.lock:
        if isinstance(request, Prepare):
            return _prepare(pg, request)
        if isinstance(request, UndoEdit):
            return _undo(pg, request)
        return _duplicate(pg, request)
