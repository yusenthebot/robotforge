"""Design-start placement for exact top-level robots, with protected publication and undo."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import mujoco
import numpy as np

from robotforge.design import layouts, terrain
from robotforge.design.world import RobotEntry, World, compile_world, spawn_height
from robotforge.playground import PlaygroundError
from robotforge.sim import carry, placement
from robotforge.sim.session import fresh_data
from robotforge.viewer.creation import _object_obstacles, _parse
from robotforge.viewer.layout_state import BuiltinLayoutState, check_supported_restore
from robotforge.viewer.object_edit import Reposition, Selection, UndoEdit, _bounds, _guard, _owners, _supported

if TYPE_CHECKING:
    from robotforge.design.store import WorldStore
    from robotforge.playground import Playground


@dataclass(frozen=True)
class RobotEdit:
    store: WorldStore
    version: int
    generation: int
    token: str
    world_sha256: str
    before: RobotEntry
    after: RobotEntry
    pose: carry.Snapshot
    hands_off: frozenset[str]


def handle(pg: Playground, message: dict) -> dict:
    """Keep shared transaction error codes meaningful in the robot-start UI."""
    try:
        return undo(pg, message) if message.get("action") == "undo_robot_start" else edit(pg, message)
    except layouts.LayoutError as exc:
        raise RobotStartError(exc) from exc


class RobotStartError(PlaygroundError):
    def __init__(self, cause: layouts.LayoutError) -> None:
        super().__init__(str(cause))
        self.code = "robot_start_" + cause.code.removeprefix("layout_")


def _eligible(model: mujoco.MjModel, name: str) -> bool:
    if model.nmocap or model.nflex:
        return False
    prefix = name + "/"
    bodies = {b for b in range(1, model.nbody) if model.body(b).name.startswith(prefix)}
    if not bodies:
        return False
    for j in range(model.njnt):
        if (
            model.jnt_bodyid[j] in bodies
            and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
            and not model.joint(j).name
        ):
            return False
    # Anonymous actuators cannot be carried/restored by the existing named-state contract.
    if any(not model.actuator(a).name for a in range(model.nu)):
        return False
    for eq in range(model.neq):
        kind = model.eq_type[eq]
        ids = (int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]))
        if kind in (mujoco.mjtEq.mjEQ_CONNECT, mujoco.mjtEq.mjEQ_WELD):
            inside = [i in bodies for i in ids]
        elif kind == mujoco.mjtEq.mjEQ_JOINT:
            inside = [i >= 0 and int(model.jnt_bodyid[i]) in bodies for i in ids]
        else:
            # Tendon/flex constraints require a different ownership contract.
            return False
        if any(inside) and not all(inside):
            return False
    return True


def metadata(pg: Playground) -> dict:
    """Only exact authored roots; mounted children move with their whole carrier."""
    with pg._edit_lock, pg.lessons._lock, pg.session.lock:
        if pg.store is None or pg.lessons.lesson is not None or not pg._play_world_sha256:
            return {"robots": []}
        try:
            world = _guard(pg, pg.store, pg.session.version, pg.session.generation, pg._play_world_sha256, "robot")
        except PlaygroundError:
            return {"robots": []}
        model = pg.session.model
        owners = _owners(model, world)
        entries = [
            {"name": entry.name, "yaw": entry.yaw, "geoms": [g for g in owners if owners[g] == entry.name]}
            for entry in world.robots
            if _eligible(model, entry.name)
        ]
        return {"robots": entries, "version": pg.session.version, "world_sha256": pg._play_world_sha256}


def _check(pg: Playground, store: WorldStore, request: Selection) -> World:
    world = _guard(pg, store, request.version, request.generation, request.world_sha256, "robot")
    check_supported_restore(pg, "changing a robot start")
    return world


def _surface_position(
    pg: Playground, world: World, entry: RobotEntry, position: list[float]
) -> tuple[list[float], int | None]:
    """Resolve floor height or an authored, fixed horizontal box/table top on the server."""
    xy = position[:2]
    ground = terrain.Ground.build(world.terrain, world.floor.style)
    floor = ground.height(*xy)
    if len(position) == 2 or abs(position[2] - floor) <= 0.005:
        if world.water and world.water.covers(*xy) and world.water.level > floor + 0.01:
            raise PlaygroundError("choose dry ground or a fixed tabletop for the robot start")
        return xy, None
    model, data = pg.session.model, pg.session.data
    for obj in world.objects:
        if obj.shape not in {"box", "table"} or not (obj.fixed or obj.shape == "table"):
            continue
        try:
            geom = model.geom(obj.name).id
        except KeyError:
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-8):
            continue
        center, size = data.geom_xpos[geom], model.geom_size[geom]
        top = center[2] + size[2]
        if world.water and world.water.covers(*xy) and top <= world.water.level + 0.01:
            continue
        local = rotation.T @ (np.asarray([*xy, top]) - center)
        if abs(position[2] - top) <= 0.005 and np.all(np.abs(local[:2]) < size[:2] - 0.002):
            return [*xy, float(top) + spawn_height(entry, pg.store.dir)], geom
    raise PlaygroundError("choose ground or the flat top of a fixed table or block")


def _platform_fits(
    model: mujoco.MjModel, data: mujoco.MjData, name: str, platform: int | None, owners: dict[int, str]
) -> None:
    if platform is None:
        return
    rotation = data.geom_xmat[platform].reshape(3, 3)
    center, size = data.geom_xpos[platform], model.geom_size[platform]
    for g in owners:
        if owners[g] != name or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        low, high = _bounds(model, data, g)
        for x in (low[0], high[0]):
            for y in (low[1], high[1]):
                local = rotation.T @ (np.array([x, y, center[2]]) - center)
                if np.any(np.abs(local[:2]) > size[:2] - 0.002):
                    raise PlaygroundError("the whole robot start must fit on the tabletop; choose a larger surface")


def _pose(snap: carry.Snapshot, name: str) -> carry.Snapshot:
    return carry.Snapshot(
        **{
            field: {key: value for key, value in getattr(snap, field).items() if carry.owner(key) == name}
            for field in ("free", "joints", "controls")
        }
    )


def edit(pg: Playground, message: dict) -> dict:
    request = _parse(Reposition, message, "robot start")
    with pg.session.execution.command("viewer_robot_start"):
        store = pg._require_store()
        with pg.lessons._lock, pg.session.lock:
            world = _check(pg, store, request)
            before = next((entry for entry in world.robots if entry.name == request.name), None)
            if before is None or not _eligible(pg.session.model, before.name):
                raise PlaygroundError("select an editable top-level robot")
            pos, platform = _surface_position(pg, world, before, request.pos)
            platform_name = pg.session.model.geom(platform).name if platform is not None else None
        after = before.model_copy(update={"pos": pos, "yaw": request.yaw})
        return _commit(pg, store, request, before, after, platform_name)


def _commit(
    pg: Playground,
    store: WorldStore,
    request: Selection,
    before: RobotEntry,
    after: RobotEntry,
    platform_name: str | None = None,
    restore: carry.Snapshot | None = None,
    restore_hands_off: frozenset[str] = frozenset(),
) -> dict:
    original = layouts._read(store.file, layouts.MAX_WORLD_BYTES)
    if layouts.sha(original) != request.world_sha256:
        raise PlaygroundError("the saved world changed; select the robot again")
    candidate = World.from_yaml(original.decode("utf-8"))
    if next((entry for entry in candidate.robots if entry.name == before.name), None) != before:
        raise PlaygroundError("the selected robot changed; select it again")
    candidate.robots[candidate.robots.index(before)] = after
    compiled = compile_world(candidate, store.dir)
    prepared = layouts.PreparedWorld(
        original, candidate.to_yaml().encode("utf-8"), compiled, compiled.to_xml().encode("utf-8")
    )
    if len(prepared.after) > layouts.MAX_WORLD_BYTES:
        raise PlaygroundError("the world is too large for a robot-start edit")
    model = compiled.model
    saved = fresh_data(model)
    owners = _owners(model, candidate)
    blocked = set(placement.blocked(model, saved, before.name))
    blocked.update(_object_obstacles(model, saved, before.name, owners))
    platform = model.geom(platform_name).id if platform_name is not None else None
    _platform_fits(model, saved, before.name, platform, owners)
    with pg.lessons._lock, pg.session.lock:
        world = _check(pg, store, request)
        if store.file.read_bytes() != original:
            raise PlaygroundError("the saved world changed; select the robot again")
        old_model = pg.session.model
        old_owners = _owners(old_model, world)
        supported = _supported(old_model, fresh_data(old_model), before.name, old_owners)
        supported.update(_supported(old_model, copy.copy(pg.session.data), before.name, old_owners))
        if supported:
            raise PlaygroundError(f"{before.name} supports {', '.join(sorted(supported))}; move those objects first")
        live = carry.take(old_model, pg.session.data)
        hands_off = pg.behaviors.hands_off()
        current = copy.copy(saved)
        carry._apply(model, current, live, lambda joint: carry.owner(joint) != before.name)
        target = restore if restore is not None else _pose(carry.take(model, saved), before.name)
        carry._apply(model, current, target, lambda joint: carry.owner(joint) == before.name)
        mujoco.mj_forward(model, current)
        blocked.update(placement.blocked(model, current, before.name))
        blocked.update(_object_obstacles(model, current, before.name, owners))
        _platform_fits(model, current, before.name, platform, owners)
        if blocked:
            raise PlaygroundError(f"that spot is occupied by {', '.join(sorted(blocked))}; choose an open spot")
        names = {entry.name for entry in world.robots}
        names.update(pg.behaviors.robot_names())
        names.update(pg.sensors.suite.robots())
        names.update(pg.perception.mappers)
        keep = sorted(name for name in names if carry.owner(name) != before.name)
        rollback = BuiltinLayoutState.capture(pg)
        try:
            with layouts._install_world(store, prepared) as notes:
                result = pg._load_world(compiled)
                carry._apply(model, pg.session.data, live, lambda joint: carry.owner(joint) != before.name)
                carry._apply(model, pg.session.data, target, lambda joint: carry.owner(joint) == before.name)
                mujoco.mj_forward(model, pg.session.data)
                manual = {name for name in hands_off if carry.owner(name) != before.name}
                pg.behaviors.let_go(manual | set(restore_hands_off))
                # Match existing edits: unchanged robots keep estimates and manually held poses.
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
                raise RobotStartError(
                    layouts.LayoutError("recovery", "robot-start rollback failed; reopen the world before continuing")
                ) from exc
            raise
        pg._play_addition = pg._play_edit = None
        undo = None
        if restore is None:
            receipt = RobotEdit(
                store,
                pg.session.version,
                pg.session.generation,
                uuid4().hex,
                hashlib.sha256(prepared.after).hexdigest(),
                before,
                after,
                _pose(live, before.name),
                frozenset(name for name in hands_off if carry.owner(name) == before.name),
            )
            pg._play_robot_edit = receipt
            undo = {"token": receipt.token, "version": receipt.version, "generation": receipt.generation}
        else:
            pg._play_robot_edit = None
        return {
            "robot": before.name,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "notes": [*result["notes"], *notes],
            "undo_robot_start": undo,
        }


def undo(pg: Playground, message: dict) -> dict:
    request = _parse(UndoEdit, message, "undo robot start")
    with pg.session.execution.command("viewer_undo_robot_start"):
        receipt = pg._play_robot_edit
        if receipt is None or receipt.token != request.token:
            raise PlaygroundError("that robot start is no longer available to undo")
        with pg.lessons._lock, pg.session.lock:
            if request.version != receipt.version or request.generation != receipt.generation:
                raise PlaygroundError("that robot start is no longer available to undo")
            selection = Selection(
                name=receipt.before.name,
                version=request.version,
                generation=request.generation,
                world_sha256=receipt.world_sha256,
            )
            _check(pg, receipt.store, selection)
        return _commit(
            pg,
            receipt.store,
            selection,
            receipt.after,
            receipt.before,
            restore=receipt.pose,
            restore_hands_off=receipt.hands_off,
        )
