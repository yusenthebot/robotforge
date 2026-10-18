"""The playground: one simulation, one viewer, and the operations agents call.

Every MCP tool is a thin wrapper over a method here, so the logic is testable
without an MCP client. Methods return plain dicts that serialize to JSON.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from pydantic import BaseModel, ValidationError

from robotforge import modes, wording
from robotforge.behaviors import airframe, robot_kind
from robotforge.behaviors.manager import BehaviorManager
from robotforge.design import gallery, importer, scenes, terrain
from robotforge.design.builder import edit as build_edit
from robotforge.design.builder import plate
from robotforge.design.store import WorldStore, worlds_home
from robotforge.design.world import (
    CompiledWorld,
    Floor,
    MountEntry,
    ObjectEntry,
    RobotEntry,
    World,
    compile_world,
    spawn_height,
    tidy_name,
)
from robotforge.learn import LESSONS, LessonRunner
from robotforge.sim import carry, finder, lidar, placement, vision
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.render import VIEWS, SnapshotRenderer
from robotforge.sim.session import SimSession, fresh_data
from robotforge.viewer.server import ViewerHub

DEMO_MODEL = Path(__file__).parent / "assets" / "demo.xml"
MAX_SIM_SECONDS = 60.0
_LIST_LIMIT = 40
ACTUATORS_LISTED = 8  # actuator names shown in an error; a robot has a dozen, a world several robots

_JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


# Positions and sizes are meters. The largest scene is 20 m across, so a number beyond these can only be
# centimetres or millimetres; refusing it beats a robot placed 150 m away that nobody can find.
MAX_POSITION = 100.0
MAX_OBJECT_SIZE = 50.0
PLURAL = {"pos": "positions", "size": "sizes"}

AREA_TERRAIN = ("hills", "rough")  # terrain types that cover the ground rather than sit on it
FEATURE_GAP = 0.6  # m of flat floor between a robot and a feature put in front of it


class PlaygroundError(ValueError):
    """A user-facing error: the message tells the agent what to do instead."""


def _one_edit_at_a_time(method):
    """Edits read, change and recompile the whole world, so two at once (tool calls run in worker
    threads) would lose one of them or repeat a slow import; the second waits for the first."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._edit_lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Playground:
    session: SimSession
    viewer_url: str | None = None
    renderer: SnapshotRenderer = field(default_factory=SnapshotRenderer)
    store: WorldStore | None = None
    last_doctor: dict[str, Any] = field(default_factory=dict)  # robot name -> import doctor report
    behaviors: BehaviorManager = field(init=False)
    modes: dict[str, tuple[str, Any]] = field(default_factory=dict, init=False)  # robot -> (mode, controller)
    lessons: LessonRunner = field(init=False)
    _edit_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    # The ground places of the scene last loaded, and the world it went into: a robot added there without pos
    # starts on the first free one. Not saved: a world opened later starts robots round [0, 0].
    _places: tuple[Path, list[scenes.Slot]] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.behaviors = BehaviorManager(
            self.session, self._acting_world, lambda: self.store.dir if self.store is not None else None
        )
        self.lessons = LessonRunner(self, LESSONS)
        self.session.reset_hooks.append(self.lessons.after_reset)

    def attach_viewer(self, hub: ViewerHub) -> None:
        """Let the viewer drive robots from the keyboard and list them in its UI."""

        def teleop(msg: dict[str, Any]) -> None:
            walker = msg.get("walker")  # set only when the user switches between the gait and a learned policy
            speed, turn = float(msg.get("speed", 0.0)), float(msg.get("turn", 0.0))
            air = {k: float(msg[k]) for k in ("side", "climb") if msg.get(k)}  # aircraft: sideways, up and down
            self.behaviors.move(str(msg["robot"]), speed, turn, str(walker) if walker else None, **air)

        def go_to(msg: dict[str, Any]) -> dict[str, Any]:
            """A double-click on the ground: go there along a planned path, as move(to=...) does, until it arrives.
            Planned off the hub's event loop: 60 to 220 ms in the scenes, over a second for a place closed off in the
            park (working out why), and the stream stood still for that long."""
            to = msg["to"] if isinstance(msg["to"], str) else [float(v) for v in msg["to"]]
            speed = float(msg["speed"]) if msg.get("speed") is not None else None
            walker = msg.get("walker")
            trip = self.behaviors.navigate(str(msg["robot"]), to, speed, str(walker) if walker else None)
            return {"path_m": round(trip.plan.length, 2)}

        def extras() -> dict[str, Any]:
            world = self.current_world()
            robots = self.behaviors.robots()
            learned = self.behaviors.learned_walkers()
            return {
                "robots": robots,
                "learned": learned,
                "world": world.name if world is not None else None,  # the page title; None for a loaded file
                "floor": world.floor.style if world is not None else None,
                "terrain": [t.type for t in world.terrain] if world is not None else [],
                "modes": {r["name"]: modes.available(self, r, r["name"] in learned) for r in robots},
                "catalog": viewer_catalog(),
                "lessons": self.lessons.catalog(),
                "lesson": self.lessons.describe_active(),
                "rotors": airframe.viewer_rotors(self.session.model, [r["name"] for r in robots]),
            }

        def world_request(msg: dict[str, Any]) -> dict[str, Any]:
            action = msg.get("action")
            if action == "scene":
                if msg.get("preset") == "empty":
                    return _short(self.clear_scene())
                return _short(self.load_scene(str(msg["preset"]), seed=int(msg.get("seed", 0))))
            if action == "terrain":
                return _short(self.terrain_for_play(str(msg["terrain"]), msg.get("robot")))
            if action == "floor":
                return _short(self.set_floor(str(msg["style"])))
            raise PlaygroundError(f"unknown world action {action!r}")

        def lesson_open(msg: dict[str, Any]) -> dict[str, Any]:
            if msg.get("action") == "start":
                return self.lessons.start(str(msg["lesson"]))
            if msg.get("action") == "exit":
                return self.lessons.exit()
            raise PlaygroundError(f"unknown lesson action {msg.get('action')!r}")

        hub.handlers["teleop"] = teleop
        hub.handlers["lesson"] = self.lessons.handle
        hub.slow_handlers["lesson_open"] = lesson_open
        hub.slow_handlers["go_to"] = go_to

        def lesson_bench(msg: dict[str, Any]) -> dict[str, Any]:
            """Sweeps run many tests on copies of the rig: off the hub thread, answered when done."""
            self.lessons.handle(msg)
            return {"done": msg.get("action")}

        hub.slow_handlers["lesson_bench"] = lesson_bench
        hub.lesson_stream = self.lessons.tick
        hub.slow_handlers["world"] = world_request
        hub.slow_handlers["mode"] = lambda msg: modes.apply(self, str(msg["robot"]), str(msg["mode"]))
        hub.scene_extras = extras
        hub.state_extras = lambda: {"mode": modes.current(self)}

    def current_world(self) -> World | None:
        return self.store.load() if self.store is not None else None

    def _acting_world(self) -> World | None:
        """The world behaviors act in: a running lesson's own world (its objects to pick), else the open one."""
        lessons = getattr(self, "lessons", None)
        world = getattr(lessons.lesson, "world", None) if lessons is not None and lessons.lesson else None
        return world if world is not None else self.current_world()

    @classmethod
    def with_demo(cls) -> Playground:
        return cls(session=SimSession.from_path(DEMO_MODEL))

    # ---- loading and describing ----------------------------------------------------

    @_one_edit_at_a_time
    def open_model(self, path: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Load an MJCF (.xml), a URDF/xacro or a plain mesh. URDFs and meshes go through the import doctor."""
        p = Path(path).expanduser()
        if importer.is_importable(p):
            if not p.is_file():
                # The path as the caller wrote it (Path would print it with backslashes on Windows)
                raise PlaygroundError(f"file not found: {path} (give the path of {importer.formats_hint()})")
            try:
                compiled = importer.compile_standalone(p, options)
            except ValueError as exc:
                raise PlaygroundError(str(exc)) from exc
            self.store = None  # the loaded model is no longer a saved world
            self.session.load_model(compiled.model, str(p.resolve()))
            self.last_doctor = compiled.doctor
            result = self.describe()
            result["doctor"] = next(iter(compiled.doctor.values()), None)
            return result
        if p.is_dir():
            raise PlaygroundError(f"{p.name} is a folder; give a file: {importer.formats_hint()}")
        if p.suffix.lower() not in {".xml", ".mjcf"}:
            raise PlaygroundError(f"{p.name}: expected {importer.formats_hint()}")
        self.session.load_path(p.resolve())
        return self.describe()

    def describe(self) -> dict[str, Any]:
        m = self.session.model
        with self.session.lock:
            joints = [
                {
                    "name": _name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                    "type": _JOINT_TYPES.get(int(m.jnt_type[j]), "unknown"),
                    "range": [round(float(x), 4) for x in m.jnt_range[j]] if m.jnt_limited[j] else None,
                }
                for j in range(m.njnt)
            ]
            actuators = [self._actuator_info(a) for a in range(m.nu)]
            sensors = [_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(m.nsensor)]
            cameras = [_name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)]
            free_bodies = [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in _free_bodies(m)]
            flying = _aircraft(m, self.behaviors)
        return {
            **({"aircraft": flying} if flying else {}),
            "model": self.session.source,
            "counts": {"bodies": m.nbody - 1, "joints": m.njnt, "actuators": m.nu, "sensors": m.nsensor},
            "actuators": actuators,
            "joints": _limit(joints),
            "free_bodies": _limit(free_bodies),
            "sensors": _limit(sensors),
            "views": sorted(VIEWS) + [c for c in cameras if c],
            "timestep": m.opt.timestep,
            "viewer_url": self.viewer_url,
        }

    def _actuator_info(self, a: int) -> dict[str, Any]:
        m = self.session.model
        info: dict[str, Any] = {"name": _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"}
        if m.actuator_ctrllimited[a]:
            info["range"] = [round(float(x), 4) for x in m.actuator_ctrlrange[a]]
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            info["joint"] = _name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
        return info

    # ---- worlds (design) ------------------------------------------------------------

    @_one_edit_at_a_time
    def new_world(self, name: str, floor: str = "checker") -> dict[str, Any]:
        self.store = WorldStore.create(name, floor)
        result = self._load_world(self.store.compile(), created=not self.store.reopened)
        saved = result["world"]
        if self.store.reopened:
            note = f"{saved} already existed: opened as it was, floor and contents unchanged"
        elif saved != name:
            note = f"saved as {saved!r}; open_world takes either name"  # a name in Chinese, or with spaces
        else:
            return result
        return {**result, "notes": [note, *result["notes"]]}

    @_one_edit_at_a_time
    def open_world(self, name_or_path: str) -> dict[str, Any]:
        store = WorldStore.open(name_or_path)
        try:
            compiled = store.compile()
        except ValueError as exc:
            # Where the file is: an agent told only "unknown template 'quadrupd'" looked for it in ~/.robotforge,
            # read the troubleshooting page and searched the disk before it found and fixed it (agent eval).
            raise PlaygroundError(f"{exc} (the world is {store.file}: fix it there, then open_world again)") from exc
        # Only now: a world that did not open used to become the one edits went to while the simulation still
        # ran the last one, so robots listed the broken file's dog and add_object failed on its template.
        self.store = store
        return self._load_world(compiled)

    @_one_edit_at_a_time
    def add_robot(
        self,
        name: str,
        template: str | None = None,
        file: str | None = None,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float | None = None,
        library: str | None = None,
        build: dict[str, Any] | None = None,
        design: str | None = None,
        mounts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        given, name = name, tidy_name(name)
        _check_meters("pos", pos, MAX_POSITION)
        if design is not None:
            build = design_build(design, template=template, file=file, library=library, build=build, params=params)
        try:
            entry = RobotEntry(
                name=name,
                template=template,
                file=file,
                library=library,
                build=build,
                params=params or {},
                pos=pos or [0.0, 0.0],
                yaw=yaw or 0.0,
                mounts=mounts or [],
            )
        except ValidationError as exc:
            raise PlaygroundError(_validation_message(exc)) from exc
        world = self.current_world()
        existing = next((r for r in world.robots if r.name == name), None) if world else None
        # Without a pos (or a yaw) the call asked for no place (or heading), and the first one was given the scene's
        # or moved clear of furniture (in the kitchen [0, 0] is inside the island): the same call again is still
        # the same robot.
        unasked = {**({"pos": entry.pos} if pos is None else {}), **({"yaw": entry.yaw} if yaw is None else {})}
        if existing is not None and existing.model_copy(update=unasked) == entry:
            # Typically a retry after the client timed out while the first call was still downloading.
            return self._unchanged(
                f"{name} is already in the world exactly like this (at {_xy(existing.pos)}); nothing to do"
            )

        def change(w: World) -> None:
            w.robots.append(entry)

        result = self._edit(change)
        if pos is None:  # a place of the scene's, or clear of what stands at [0, 0]
            free, facing, placed = self._start_for(name, entry.pos, None if yaw is None else entry.yaw)
        else:
            free, facing, placed = None, None, self._clear_start(name, entry.pos)
        if free is not None:

            def move(w: World) -> None:
                robot = _find(w.robots, name, "robot")
                robot.pos = free
                if facing is not None:
                    robot.yaw = facing

            result = self._edit(move, snapshot=False)
        if name in self.last_doctor:  # URDFs, xacro and meshes come with the doctor's report
            result["doctor"] = self.last_doctor[name]
        return _noted(_renamed(result, given, name), _yaw_note(yaw), placed)

    @_one_edit_at_a_time
    def update_robot(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float | None = None,
        build: dict[str, Any] | None = None,
        mounts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        _check_meters("pos", pos, MAX_POSITION)
        kept: list[str] = []
        mounted = None if mounts is None else _mounts(mounts)
        found: list[str] = []
        swap = self._replacing(name, build) if build is not None and _whole(build) else None
        replaced = [swap[0]] if swap else []
        if swap:
            build = swap[1]  # a lidar alone for a template or library robot: the robot carried with it on a plate

        def change(w: World) -> None:
            robot = _find(w.robots, name, "robot")
            found.append(robot.name)
            if not params and pos is None and yaw is None and build is None and mounts is None:
                raise PlaygroundError("nothing to change: give params, pos, yaw, build or mounts")
            if params:
                robot.params = {**robot.params, **params}
            if pos is not None:
                robot.pos = pos
            if yaw is not None:
                robot.yaw = yaw
            if mounted is not None:  # the whole list: [] takes every mounted robot off
                robot.mounts = mounted
            if build is not None:
                if robot.build is None and _whole(build):
                    # A whole robot from parts in place of a template, library robot or file, where it stands: both
                    # lidar sessions in the eval sent the maze's starter rover a build with a lidar (update_robot,
                    # then add_part), were refused, and removed it to add the same robot again. _replacing has
                    # checked that it carries the old robot or is one of the same kind.
                    robot.template = robot.file = robot.library = None
                    robot.params = {}
                    robot.build = build
                    return
                entry = _built(robot)
                if not (entry.build or {}).get("parts") or "parts" in build:
                    entry.build = build
                    return
                # A build without parts changes the fields it gives and keeps the parts: an agent in the eval sent
                # build={"servo": ..., "base": {...}} to give its hexapod stronger servos, got a bare box and had
                # to send all 19 parts again.
                if not build:
                    raise PlaygroundError(
                        f"build {{}} changes nothing on {robot.name}: give the fields to change (servo, base, ...), or "
                        '"parts" to replace the whole robot'
                    )
                entry.build = _merged_build(entry.build, build)
                kept.append(
                    f'{robot.name}: kept its parts and changed only {", ".join(build)}; a build without "parts" '
                    'changes the fields it gives (send "parts" to replace them)'
                )

        result = self._edit(change)
        placed = self._clear_start(found[0], pos) if pos is not None else None
        return _noted(result, _yaw_note(yaw), *replaced, *kept, placed)

    def _replacing(self, name: str, build: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        """The note for a whole build sent to a template, library or file robot, and the build to save, or None for
        one built from parts. A build that carries the old robot on a part, or makes a robot of the same kind,
        replaces it where it stands; sensors alone go on a plate that carries the robot; any other is refused with
        the build that keeps it. Rebuilt on the word of the add_part refusal, a quadruped given a base with a lidar
        became a 1 kg box with no joints, and a Go2 given a lidar puck a 1.1 kg cylinder; the only sign was "now
        built from parts"."""
        store = self._require_store()
        entry = _find(store.load().robots, name, "robot")
        if entry.build is not None:
            return None
        source, mount = _source(entry), plate.mount_of(entry)
        if mount is not None and plate.mounts(build, mount):
            return f"{entry.name}: now built from parts, carrying {source} as it was", build
        if mount is not None and _sensors_only(build):
            carried, note = self._carry(entry, build, lambda b: b, "with the sensors on the plate")
            return note, carried
        before, joints = self.behaviors.kind(entry.name).value, _moving_joints(self.session.model, entry.name)
        try:
            model = self._compiled_with(entry.name, build)
        except (ValueError, ValidationError):
            return None  # the edit itself says what is wrong with the build
        after, now = robot_kind(model, entry.name).value, _moving_joints(model, entry.name)
        if after == before:
            note = (
                f"{entry.name}: now built from parts (it was {source}), still {_a(before)}, with {_joints(now)} where "
                f"it had {joints}"
            )
            return note, build
        raise PlaygroundError(
            self._keep_it(
                entry,
                build,
                f"this build would turn {entry.name}, {_a(before)} with {_joints(joints)} from {source}, into a robot "
                f"of kind {after} with {_joints(now)}: the {before} would be gone",
            )
        )

    def _carry(
        self, entry: RobotEntry, build: dict[str, Any], add: Callable[[dict[str, Any]], dict[str, Any]], what: str
    ) -> tuple[dict[str, Any], str]:
        """`entry`, a template or library robot, carried under a plate as the gallery's go2_arm carries a Go2, with
        `build`'s base sensors on the plate and `add` applied to that build (add_part's part), and the note saying
        so; refused (PlaygroundError) when the result would no longer be the same kind of robot. In both of the
        reviewer's sessions the agent first called add_part to give a robot from the library or a template a lidar,
        and was refused with this build to send instead: one failed call in 9 or 10."""
        mount, kind, source = plate.mount_of(entry), self.behaviors.kind(entry.name).value, _source(entry)
        if mount is None:
            raise PlaygroundError(
                f"{entry.name} comes from a file: parts and sensors go on robots built from parts, and one from a "
                "file cannot be carried on a plate; build one from parts (builder_guide)"
            )
        bare = plate.carried(self.session.model, entry.name, mount, {k: v for k, v in build.items() if k != "parts"})
        whole = add(bare)  # a part that is wrong says so here, as on any robot built from parts
        try:
            after = robot_kind(self._compiled_with(entry.name, bare), entry.name).value
            if after == kind:  # the plate keeps it what it was; then the part has its say (it may overlap the robot)
                after = robot_kind(self._compiled_with(entry.name, whole), entry.name).value
        except ValidationError as exc:  # as an edit says it (a part with no size)
            raise PlaygroundError(_validation_message(exc)) from exc
        if after != kind:
            limit = " (a current limit: a humanoid cannot ride a plate yet)" if kind == "humanoid" else ""
            raise PlaygroundError(
                f"{entry.name} comes from {source}, and parts and sensors go on robots built from parts; carried on a "
                f"plate it would no longer be {_a(kind)}{limit}: build one from parts instead (builder_guide)"
            )
        hung = bare["parts"][0]["name"]
        note = (
            f"{entry.name} comes from {source}, so it now rides a {bare['base']['mass']:g} kg plate as its base, "
            f"{what} (the gallery's go2_arm layout), still {_a(kind)}; its joints are now {entry.name}/{hung}/..."
        )
        return whole, note

    def _keep_it(self, entry: RobotEntry, build: dict[str, Any], why: str) -> str:
        """`why` a change to a robot not built from parts is refused, and the build that keeps the robot: carried
        under a plate, as the gallery's go2_arm carries a Go2, with `build`'s parts and base sensors on the plate.
        Offered only when the plate and the robot compile and are still the same kind of robot (a humanoid is not,
        so far)."""
        mount, kind, source = plate.mount_of(entry), self.behaviors.kind(entry.name).value, _source(entry)
        if mount is None:
            return f"{why}. A robot from a file cannot be carried on a plate: build one from parts (builder_guide)"
        model = self.session.model
        bare = plate.carried(model, entry.name, mount, {k: v for k, v in build.items() if k != "parts"})
        try:
            still = robot_kind(self._compiled_with(entry.name, bare), entry.name).value == kind
        except (ValueError, ValidationError):
            still = False
        if not still:
            limit = " (a current limit: a humanoid cannot ride a plate yet)" if kind == "humanoid" else ""
            return f"{why}. Carried on a plate, {source} would no longer be {_a(kind)}{limit}: build one from parts"
        recipe = json.dumps(plate.carried(model, entry.name, mount, build))
        return (
            f"{why}. Keep it on a plate: update_robot({json.dumps(entry.name)}, build={recipe}); joints become "
            f"{entry.name}/{bare['parts'][0]['name']}/..."
        )

    def _compiled_with(self, name: str, build: dict[str, Any]) -> mujoco.MjModel:
        """The open world compiled with robot `name` built from `build` instead, saving nothing."""
        store = self._require_store()
        world = store.load().model_copy(deep=True)
        robot = next(r for r in world.robots if r.name == name)
        robot.template = robot.file = robot.library = None
        robot.params, robot.build = {}, build
        return compile_world(World.model_validate(world.model_dump()), store.dir).model

    def _worlds(self) -> tuple[mujoco.MjModel, list[mujoco.MjData]]:
        """The open world as it starts (every robot at its saved place, as a reset or reload_world puts them) and as
        it is now: a place has to be free in both."""
        with self.session.lock:
            model, live = self.session.model, copy.copy(self.session.data)
        return model, [fresh_data(model), live]

    def _clear_start(self, robot: str, pos: list[float]) -> str | None:
        """What `robot`, put at `pos` on purpose, stands inside (furniture, a wall, a ramp, another robot, or another
        robot's start), as a note saying where is free; it stays there. None when it stands clear."""
        model, (start, live) = self._worlds()
        now = placement.inside(model, live, robot)
        then = {k: v for k, v in placement.inside(model, start, robot).items() if k not in now}
        if not now and not then:
            return None
        spot = placement.free_place(model, [start, live], robot, ((x, y, 0.0) for x, y in placement.rings(*pos[:2])))
        where = f"; {_xy(spot)} is free" if spot is not None else ""
        hit = now or then
        what, deep = _names(hit), max(hit.values()) * 100
        if now:
            return (
                f"{robot} starts inside {what} at {_xy(pos)} ({deep:.0f} cm deep): it will be pushed out or get stuck "
                f"there; give it a free pos{where}"
            )
        return (
            f"{robot} starts at {_xy(pos)} where {what} starts too ({deep:.0f} cm into it): after a reset or "
            f"reload_world they stand inside each other; give it a free pos{where}"
        )

    def _start_for(
        self, robot: str, pos: list[float], yaw: float | None
    ) -> tuple[list[float] | None, float | None, str | None]:
        """Where `robot`, added without pos (so at `pos`, [0, 0]), is to start instead, the heading to give it (None
        to keep it), and the note saying so. After load_scene it is the first of the scene's robot_places free in
        the world as it starts and as it is now, facing the way the place does unless `yaw` was given; with every
        place taken, or no scene, the nearest free place round the first place or round `pos`, if something
        stands at `pos`. Moved clear of the kitchen's island to the nearest free place, 0.1 m from it, a dog told
        to walk over to the island was already there, and a rover added next went under a Go2's belly."""
        model, worlds = self._worlds()
        if placement.movable(model, robot) is None:  # fixed to the floor: only told
            return None, None, self._clear_start(robot, pos)
        slots = self._scene_places()

        def turn(heading: float) -> float:  # added facing 0 degrees, or the yaw given (kept)
            return 0.0 if yaw is not None else heading

        for slot in slots:
            found = placement.free_place(model, worlds, robot, [(*slot.pos, turn(slot.yaw))])
            if found is not None:
                facing = None if yaw is not None else float(slot.yaw)
                said = "" if facing is None else f", facing {facing:g} degrees"
                where = [found[0], found[1]]
                return where, facing, f"{robot} starts at {_xy(where)}{said}, the first free one of robot_places"
        if slots:
            around, heading, why = slots[0].pos, slots[0].yaw, "the scene's robot_places are all taken"
        else:
            hit: dict[str, float] = {}
            for world in worlds:
                for name, much in placement.blocked(model, world, robot).items():
                    hit[name] = max(hit.get(name, 0.0), much)
            if not hit:
                return None, None, None
            inside = any(placement.inside(model, w, robot) for w in worlds)
            around, heading = pos[:2], 0.0
            why = f"{_xy(pos)} is {'inside' if inside else 'too close to'} {_names(hit)}"
        rings = placement.rings(*around)
        found = placement.free_place(model, worlds, robot, ((x, y, turn(heading)) for x, y in rings))
        if found is None:
            return None, None, self._clear_start(robot, pos)
        facing = None if yaw is not None or not slots else float(heading)
        free = [found[0], found[1], *pos[2:]]
        return free, facing, f"{robot}: {why}, so it starts at {_xy(free)}, the nearest free place"

    def _scene_places(self) -> list[scenes.Slot]:
        """The ground places of the scene loaded into the open world, if one was loaded this session."""
        if self._places is None or self.store is None or self._places[0] != self.store.dir:
            return []
        return self._places[1]

    # ---- robots built from parts ------------------------------------------------------

    @_one_edit_at_a_time
    def add_part(self, robot: str, part: dict[str, Any]) -> dict[str, Any]:
        return self._edit_build(robot, lambda b: build_edit.add_part(b, part), adding=part)

    @_one_edit_at_a_time
    def update_part(self, robot: str, part: str, changes: dict[str, Any]) -> dict[str, Any]:
        return self._edit_build(robot, lambda b: build_edit.update_part(b, part, changes))

    @_one_edit_at_a_time
    def remove_part(self, robot: str, part: str) -> dict[str, Any]:
        removed: list[str] = []

        def change(b: dict[str, Any]) -> dict[str, Any]:
            new, gone = build_edit.remove_part(b, part)
            removed.extend(gone)
            return new

        result = self._edit_build(robot, change)
        result["removed_parts"] = removed
        return result

    def _edit_build(
        self, robot: str, change_build: Callable[[dict[str, Any]], dict[str, Any]], adding: dict | None = None
    ) -> dict[str, Any]:
        entry = _find(self._require_store().load().robots, robot, "robot")
        if entry.build is None and isinstance(adding, dict):  # a part for a template or library robot: on a plate
            name = adding.get("name", "the part")
            whole, note = self._carry(entry, {}, change_build, f"with {name} on the plate")

            def carry_it(w: World) -> None:
                found = _find(w.robots, robot, "robot")
                found.template = found.file = found.library = None
                found.params, found.build = {}, whole

            return _noted(self._edit(carry_it), note)

        def change(w: World) -> None:
            entry = _built(_find(w.robots, robot, "robot"))
            entry.build = change_build(entry.build)

        return self._edit(change)

    @_one_edit_at_a_time
    def add_object(self, name: str, **fields: Any) -> dict[str, Any]:
        given, name = name, tidy_name(name)
        _check_meters("pos", fields.get("pos"), MAX_POSITION)
        _check_meters("size", fields.get("size"), MAX_OBJECT_SIZE)
        size, cube = fields.get("size"), None
        if fields.get("shape", "box") == "box" and size is not None and len(size) == 1:
            fields = {**fields, "size": [size[0]] * 3}  # one number is the edge of a cube
            cube = f"box size {size} is a cube of {size[0]:g} m"
        try:
            entry = ObjectEntry(name=name, **{k: v for k, v in fields.items() if v is not None})
        except ValidationError as exc:  # pydantic's own wording buries the fix under types and a docs link
            raise PlaygroundError(_validation_message(exc)) from exc

        def change(w: World) -> None:
            w.objects.append(entry)

        return _noted(_renamed(self._edit(change), given, name), cube, _yaw_note(fields.get("yaw")))

    @_one_edit_at_a_time
    def update_object(self, name: str, **fields: Any) -> dict[str, Any]:
        _check_meters("pos", fields.get("pos"), MAX_POSITION)
        _check_meters("size", fields.get("size"), MAX_OBJECT_SIZE)

        def change(w: World) -> None:
            obj = _find(w.objects, name, "object")
            if all(value is None for value in fields.values()):
                raise PlaygroundError(f"nothing to change: give {', '.join(fields)}")
            for key, value in fields.items():
                if value is not None:
                    setattr(obj, key, value)

        return _noted(self._edit(change), _yaw_note(fields.get("yaw")))

    @_one_edit_at_a_time
    def remove(self, name: str) -> dict[str, Any]:
        def change(w: World) -> None:
            names = [*(r.name for r in w.robots), *(o.name for o in w.objects), *(t.name for t in w.terrain)]
            found = wording.match(name, names, plurals=False)  # "dogs" may be dog, dog2 or both: ask, do not delete
            if found is None:
                text = wording.unknown("robot, object or terrain patch", name, names, many="names")
                raise PlaygroundError(
                    f"{text}. The floor is changed with set_floor" if "floor" in name.lower() else text
                )
            w.robots = [r for r in w.robots if r.name != found]
            w.objects = [o for o in w.objects if o.name != found]
            w.terrain = [t for t in w.terrain if t.name != found]

        return self._edit(change)

    @_one_edit_at_a_time
    def reload_world(self) -> dict[str, Any]:
        """Compile the open world again and load it, e.g. when coming back from a lesson."""
        return self._load_world(self._require_store().compile())

    @_one_edit_at_a_time
    def clear_scene(self) -> dict[str, Any]:
        """Back to an empty checker floor: no terrain, no objects. Robots stay where they are."""
        if self.store is None:
            self.store = WorldStore.create("playground")

        def change(w: World) -> None:
            w.floor = Floor()
            w.terrain = []
            w.objects = []

        self._places = None  # the scene's places went with it
        return self._edit(change)

    @_one_edit_at_a_time
    def set_floor(self, style: str) -> dict[str, Any]:
        def change(w: World) -> None:
            w.floor.style = style

        return self._edit(change)

    @_one_edit_at_a_time
    def undo(self) -> dict[str, Any]:
        store = self._require_store()
        store.undo()
        return self._load_world(store.compile())

    def show_world(self) -> dict[str, Any]:
        """The open world's world.yaml. With none open it says so and lists the saved worlds, not an error: agents
        call it first to see where things stand (both go2_arm_kitchen sessions on the branch began with it, and it
        counted as a failed call)."""
        if self.store is None:
            home = worlds_home()
            saved = sorted(p.parent.name for p in home.glob("*/world.yaml")) if home.is_dir() else []
            return {
                "world": None,
                "saved_worlds": _limit(saved),
                "notes": ["no world is open: new_world makes one, load_scene a ready-made one, open_world a saved one"],
            }
        store = self.store
        return {"world_dir": str(store.dir), "world_yaml": store.file.read_text(encoding="utf-8")}

    def _edit(self, change, snapshot: bool = True) -> dict[str, Any]:
        """Apply `change` to the open world and load it; snapshot=False finishes the edit before it (one undo)."""
        store = self._require_store()
        before = store.load()
        with self.session.lock:
            live = carry.take(self.session.model, self.session.data)
            hands_off = self.behaviors.hands_off()
        try:
            after, compiled = store.apply(change, snapshot=snapshot)
        except ValidationError as exc:
            raise PlaygroundError(_validation_message(exc)) from exc
        result = self._load_world(compiled)
        # What the edit did not touch stays where it is (a box added does not send the dog back to its start).
        # New ground under them (terrain, a scene) starts everything fresh: their heights would no longer fit.
        if before.terrain == after.terrain:
            unchanged = [r.name for r in after.robots if r in before.robots]
            unchanged += [o.name for o in after.objects if o in before.objects]
            with self.session.lock:
                carried = carry.restore(self.session.model, self.session.data, live, unchanged)
                self.behaviors.let_go(hands_off.intersection(carried))  # legs set by hand stay set
        return result

    def _unchanged(self, note: str) -> dict[str, Any]:
        """The world summary without recompiling: the simulation keeps running as it is."""
        store = self._require_store()
        world = store.load()
        return {
            "world": world.name,
            "world_dir": str(store.dir),
            "created": False,
            "unchanged": True,
            "robots": [r.name for r in world.robots],
            "objects": [o.name for o in world.objects],
            "notes": [note],
            "viewer_url": self.viewer_url,
        }

    def _require_store(self) -> WorldStore:
        if self.store is None:
            raise PlaygroundError("no world is open; call new_world first (or open_world for an existing one)")
        return self.store

    def _load_world(self, compiled: CompiledWorld, created: bool = False) -> dict[str, Any]:
        store = self._require_store()
        self.session.load_model(compiled.model, str(store.file))
        self.last_doctor = compiled.doctor
        world = store.load()
        summary = self.describe()
        return {
            "world": world.name,
            "world_dir": str(store.dir),
            "created": created,
            "robots": [r.name for r in world.robots],
            "objects": _limit([o.name for o in world.objects]),
            **({"terrain": [t.name for t in world.terrain]} if world.terrain else {}),
            "notes": [*compiled.notes, *self.behaviors.walk_notes()],
            "actuators": summary["actuators"],
            "viewer_url": self.viewer_url,
        }

    # ---- scenes and terrain (design/scenes.py, design/terrain.py) ------------------------

    def list_scenes(self) -> dict[str, Any]:
        return {
            "scenes": scenes.catalog(),
            "terrain_types": terrain.catalog(),
            "ground_styles": list(terrain.STYLES),
        }

    @_one_edit_at_a_time
    def load_scene(
        self,
        preset: str,
        seed: int = 0,
        keep_robots: bool = True,
        starter: bool = True,
        options: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Replace the floor, terrain and objects with a ready-made scene (opening a world named after it if
        none is open). Kept robots move to the scene's places; an empty world gets the scene's starter robot."""
        try:
            content = scenes.build(preset, seed, options)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
        if self.store is None:
            self.store = WorldStore.create(preset, content.floor.get("style", "checker"))
        base_dir = self.store.dir
        placed: list[str] = []

        def change(w: World) -> None:
            robots = list(w.robots) if keep_robots else []
            if not robots and starter:
                robots = [RobotEntry(**r) for r in content.starter]
            placed.extend(scenes.place(robots, content, lambda entry: spawn_height(entry, base_dir)))
            taken = {r.name for r in robots}
            w.robots = robots
            w.floor = Floor.model_validate(content.floor)
            w.terrain = [terrain.parse(t) for t in content.terrain]
            w.objects = [ObjectEntry(**{**o, "name": _unused(o["name"], taken)}) for o in content.objects]

        result = self._edit(change)
        self._places = (self.store.dir, list(content.ground_slots))
        result["scene"] = {
            "preset": preset,
            "seed": seed,
            "placed": placed,
            "robot_places": [list(s.pos) for s in content.ground_slots],
            "arm_places": [[*s.pos, s.z] for s in content.arm_slots],
            "tips": content.tips,
        }
        return result

    @_one_edit_at_a_time
    def set_terrain(self, type: str, style: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make the whole ground one terrain patch centered on the origin; type "flat" removes all terrain."""
        patches = [] if type == "flat" else [self._patch({**(params or {}), "type": type, "style": style})]

        def change(w: World) -> None:
            w.terrain = patches

        return self._edit(change)

    def terrain_for_play(self, type: str, robot: str | None = None) -> dict[str, Any]:
        """The viewer's terrain picker: hills and rough ground cover the area around the origin; a feature
        (stairs, ramp, stones, curb, platform) is put just in front of `robot`, facing the way it looks."""
        if type in ("flat", *AREA_TERRAIN):
            return self.set_terrain(type)
        try:
            length = terrain.parse({"type": type}).footprint()[0]
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
        world = self._require_store().load()
        entry = next((r for r in world.robots if r.name == robot), None)
        x, y = entry.pos if entry is not None else (0.0, 0.0)
        yaw = entry.yaw if entry is not None else 0.0
        ahead = FEATURE_GAP + length / 2
        heading = np.radians(yaw)
        pos = [round(x + ahead * np.cos(heading), 3), round(y + ahead * np.sin(heading), 3)]
        return self.set_terrain(type, params={"pos": pos, "yaw": yaw})

    @_one_edit_at_a_time
    def add_terrain(
        self,
        type: str,
        name: str | None = None,
        pos: list[float] | None = None,
        yaw: float = 0.0,
        style: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        given, name = name, tidy_name(name) if name else None
        _check_meters("pos", pos, MAX_POSITION)
        fields = {"name": name, "pos": pos, "yaw": yaw, "style": style}
        patch = self._patch({**(params or {}), "type": type, **{k: v for k, v in fields.items() if v is not None}})

        def change(w: World) -> None:
            w.terrain.append(patch)

        result = _noted(self._edit(change), _yaw_note(yaw))
        return _renamed(result, given, name) if name else result

    @_one_edit_at_a_time
    def update_terrain(
        self,
        name: str,
        pos: list[float] | None = None,
        yaw: float | None = None,
        style: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _check_meters("pos", pos, MAX_POSITION)

        def change(w: World) -> None:
            old = _find(w.terrain, name, "terrain patch")
            fields = {"pos": pos, "yaw": yaw, "style": style}
            if not params and all(value is None for value in fields.values()):
                raise PlaygroundError("nothing to change: give pos, yaw, style or params")
            data = {**old.model_dump(), **(params or {}), **{k: v for k, v in fields.items() if v is not None}}
            data["type"], data["name"] = old.type, old.name  # a patch keeps its type and name
            w.terrain = [self._patch(data) if t.name == old.name else t for t in w.terrain]

        return _noted(self._edit(change), _yaw_note(yaw))

    @staticmethod
    def _patch(data: dict[str, Any]) -> Any:
        if data.get("style") is None:
            data.pop("style", None)
        try:
            return terrain.parse(data)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc

    # ---- control and simulation ----------------------------------------------------

    def set_controls(self, controls: dict[str, float]) -> dict[str, Any]:
        applied, notes = self._apply_controls(controls)
        return {"applied": applied, **({"notes": notes} if notes else {})}

    def _apply_controls(self, controls: dict[str, float]) -> tuple[dict[str, float], list[str]]:
        """Set the targets; the values as applied (an actuator clamps to its range), and a note for each clamp."""
        applied: dict[str, float] = {}
        notes: list[str] = []
        m = self.session.model
        with self.session.lock:
            for name, value in controls.items():
                a = self._actuator_id(name)
                asked = float(value)
                self.session.set_ctrl(a, asked)
                actual = self._actuator_info(a)["name"]
                applied[actual] = round(float(self.session.data.ctrl[a]), 4)
                if m.actuator_ctrllimited[a] and abs(applied[actual] - asked) > 1e-6:
                    low, high = (float(x) for x in m.actuator_ctrlrange[a])
                    note = f"{actual}: {asked:g} is outside {low:g} to {high:g}; clamped to {applied[actual]:g}"
                    if is_position_servo(m, a) and abs(asked) <= 360 and abs(asked) > 5 * max(abs(low), abs(high)):
                        note += f" (joint targets are radians; {asked:g} degrees is {math.radians(asked):.2f})"
                    notes.append(note)
        return applied, notes

    def simulate(
        self, seconds: float, controls: dict[str, float] | None = None, wind: list[float] | None = None
    ) -> dict[str, Any]:
        if not 0 < seconds <= MAX_SIM_SECONDS:
            raise PlaygroundError(wording.seconds_range("seconds", seconds, MAX_SIM_SECONDS))
        notes: list[str] = []
        blown: dict[str, Any] | None = None
        with self.session.lock:
            if controls:
                _, notes = self._apply_controls(controls)
            steps = max(1, round(seconds / self.session.model.opt.timestep))
            before = self._free_body_positions()
            if wind is not None:
                from robotforge.behaviors import air

                try:
                    blown = air.blow(self.session, wind, seconds, steps)
                except air.WindError as exc:
                    raise PlaygroundError(str(exc)) from exc
            else:
                self.session.step(steps)
            after = self._free_body_positions()
            unstable = not np.isfinite(self.session.data.qpos).all()
        moved = {
            name: {"from": before[name], "to": pos}
            for name, pos in after.items()
            if np.linalg.norm(np.subtract(pos, before[name])) > 0.005
        }
        result: dict[str, Any] = {
            "time": round(float(self.session.data.time), 4),
            "steps": steps,
            "moved": moved,
        }
        if notes:
            result["notes"] = notes
        if blown is not None:
            result["wind"] = blown
        if unstable:
            result["warning"] = "simulation became unstable (NaN state); reset and check the model"
        return result

    def run_fast(self, until: Callable[[], bool], max_seconds: float) -> float:
        """Step as fast as possible until `until()` or `max_seconds` of sim time; returns sim seconds run."""
        with self.session.lock:
            start = self.session.data.time
            chunk = max(1, round(0.02 / self.session.model.opt.timestep))
            while not until() and self.session.data.time - start < max_seconds:
                self.session.step(chunk)
            return float(self.session.data.time - start)

    def control(self, action: str) -> dict[str, Any]:
        s = self.session
        if action == "play":
            s.paused = False
        elif action == "pause":
            s.paused = True
        elif action == "reset":
            s.reset()
        else:
            raise PlaygroundError(f"unknown action {action!r}; use play, pause or reset")
        return {"action": action, "paused": s.paused, "time": round(float(s.data.time), 4)}

    # ---- observation ----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        m, d = self.session.model, self.session.data
        with self.session.lock:
            joints = {}
            for j in range(m.njnt):
                jtype = int(m.jnt_type[j])
                if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    joints[_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"] = round(
                        float(d.qpos[m.jnt_qposadr[j]]), 4
                    )
            ctrl = {self._actuator_info(a)["name"]: round(float(d.ctrl[a]), 4) for a in range(m.nu)}
            sensors = self.sensor_readings()
            lidars = lidar.readings(m, d)
            listed = _limit_dict(sensors)
            if lidars and "..." in listed:  # a 64-ray lidar alone is 64 sensors: say where all its rays are
                listed["..."] += "; lidars has every ray of each lidar"
            return {
                "time": round(float(d.time), 4),
                "paused": self.session.paused,
                "joints": _limit_dict(joints),
                "controls": _limit_dict(ctrl),
                "free_bodies": _limit_dict(self._free_body_positions()),
                "sensors": listed,
                **({"lidars": lidars} if lidars else {}),
                "contacts": int(d.ncon),
            }

    def sensor_readings(self) -> dict[str, Any]:
        """Every sensor's reading by name, none left out (a script reads them all; observe lists the first ones).
        A rangefinder on a robot reads past the robot's own parts, as its lidar does (sim/lidar.seen)."""
        m, d = self.session.model, self.session.data
        with self.session.lock:
            out: dict[str, Any] = {}
            for s in range(m.nsensor):
                adr, dim = m.sensor_adr[s], m.sensor_dim[s]
                if dim == 1 and int(m.sensor_type[s]) == mujoco.mjtSensor.mjSENS_RANGEFINDER:
                    values = [round(lidar.seen(m, d, s), 4)]
                else:
                    values = [round(float(v), 4) for v in d.sensordata[adr : adr + dim]]
                out[_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"] = values[0] if dim == 1 else values
            return out

    def snapshot(self, view: str = "default", width: int = 640, height: int = 480) -> bytes:
        try:
            return self.renderer.png(self.session, view, width, height)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc

    def find(
        self, view: str, color: str, width: int = 640, height: int = 480, image: bool = True
    ) -> tuple[dict[str, Any], bytes | None]:
        """observe(find=...): the things of `color` that `view` sees, located by color and depth, and (with
        `image`) the picture as the 3D view shows it, each find boxed and numbered."""
        try:
            finder.colors(color)
            layers = self.renderer.layers(
                self.session, view, width, height, geoms=True, exposure=vision.EXPOSURE, shown=image
            )
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
        found = finder.find(layers, color)
        marked = finder.marked_png(layers.shown, found) if layers.shown is not None else None
        return finder.summary(view, color, found), marked

    # ---- helpers ----------------------------------------------------------------------

    def _actuator_id(self, name: str) -> int:
        """The actuator called `name`: exactly, in other capitals, or without its robot's prefix ("shoulder" for
        "arm/shoulder") when only one robot has it."""
        m = self.session.model
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a >= 0:
            return a
        names = [self._actuator_info(i)["name"] for i in range(m.nu)]
        found = wording.match(name, names)
        if found is None:
            tail = "/" + wording.squash(name)
            ends = [n for n in names if wording.squash(n).endswith(tail)]
            found = ends[0] if len(ends) == 1 else None
        if found is not None:
            return names.index(found)
        close = wording.near(name, names)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        if len(names) > ACTUATORS_LISTED:  # a few dozen names would only bury the one that was meant
            listing = f"Names look like {names[0]!r}; describe_model lists them all"
        else:
            listing = f"Available: {', '.join(names)}"
        raise PlaygroundError(f"no actuator named {name!r}.{hint} {listing}")

    def _free_body_positions(self) -> dict[str, list[float]]:
        m, d = self.session.model, self.session.data
        return {
            _name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}": [round(float(x), 4) for x in d.xpos[b]]
            for b in _free_bodies(m)
        }


def _unused(name: str, taken: set[str]) -> str:
    """`name`, or name_2, name_3, ... if a robot already uses it."""
    n, out = 1, name
    while out in taken:
        n += 1
        out = f"{name}_{n}"
    return out


def design_build(design: str, **others: Any) -> dict[str, Any]:
    """The build a gallery design starts from. A design is the robot's whole description, so it comes alone;
    the world keeps it as the robot's build, ready to change part by part."""
    extra = [k for k, v in others.items() if v]  # params={} is no params
    if extra:
        raise PlaygroundError(
            f"design is a whole robot on its own; leave out {', '.join(extra)}. Change it after adding it, part by "
            'part: update_part, e.g. part="robot", changes={"color": "red"}'
        )
    try:
        return gallery.build(design)
    except ValueError as exc:
        raise PlaygroundError(str(exc)) from None


def _source(robot: RobotEntry) -> str:
    if robot.template:
        return f"the {robot.template} template"
    return f"the robot library ({robot.library})" if robot.library else "a file"


def _whole(build: dict[str, Any]) -> bool:
    """A build that describes a whole robot (a base or parts), not a few fields to change on one."""
    return isinstance(build, dict) and ("base" in build or "parts" in build)


def _sensors_only(build: dict[str, Any]) -> bool:
    """A build that is nothing but sensors for the base: {"base": {"sensors": [...]}}."""
    base = build.get("base")
    return set(build) == {"base"} and isinstance(base, dict) and set(base) == {"sensors"} and bool(base["sensors"])


def _built(robot: RobotEntry) -> RobotEntry:
    if robot.build is None:
        raise PlaygroundError(
            f"{robot.name} comes from {_source(robot)}; parts and sensors (a lidar, a camera) are edited only on "
            "robots built from parts. add_part on it carries it on a plate and adds the part there; to build on a "
            'template, mount it as a part: {"name": "arm", "robot": {"template": "arm"}}'
        )
    return robot


CREATED_BY = {"robot": "add_robot", "object": "add_object", "terrain patch": "add_terrain"}


def _find(items: list, name: str, kind: str):
    """The entry called `name` (in other capitals too); an error listing what there is when none is."""
    names = [i.name for i in items]
    found = wording.match(name, names)
    if found is None:
        raise PlaygroundError(wording.unknown(kind, name, names, make=CREATED_BY[kind]))
    return items[names.index(found)]


def _check_meters(label: str, values: list[float] | None, limit: float) -> None:
    """Refuse a position or size too big to be anything but centimetres or millimetres taken for meters."""
    if not values or not all(isinstance(v, int | float) for v in values) or max(abs(v) for v in values) <= limit:
        return
    text = f"{label} {list(values)} is out of range: {PLURAL[label]} are meters, at most {limit:g} in a scene"
    if max(abs(v) for v in values) / 100 <= limit:
        text += f"; use meters (in cm that would be {[round(v / 100, 3) for v in values]})"
    raise PlaygroundError(text)


def _merged_build(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """A robot's build with the fields of `new` changed and its parts kept, as update_part(robot, "robot" or "base")
    changes them, so null puts a field back to its default and an unknown field is refused the same way."""
    robot_wide = {key: value for key, value in new.items() if key != "base"}
    out = build_edit.update_part(old, "robot", robot_wide) if robot_wide else old
    base = new.get("base")
    if isinstance(base, dict) and base:
        out = build_edit.update_part(out, "base", base)  # {"base": {"mass": 0.8}} keeps its shape and size
    elif "base" in new:
        out = {**out, "base": base}  # not a mapping: the build's own check says what a base is
    return out


def _moving_joints(model: mujoco.MjModel, robot: str) -> int:
    """How many hinges and slides robot `robot` has in a compiled world."""
    moving = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    return sum(
        int(model.jnt_type[j]) in moving
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(f"{robot}/")
        for j in range(model.njnt)
    )


def _a(kind: str) -> str:
    """A robot kind with its article: an arm, a quadruped, a robot of kind other."""
    if kind == "other":
        return "a robot of kind other"
    return f"an {kind}" if kind[0] in "aeiou" else f"a {kind}"


def _joints(count: int) -> str:
    return f"{count} joint" if count == 1 else f"{count} joints"


def _xy(pos: list[float] | tuple[float, ...]) -> str:
    return f"[{pos[0]:g}, {pos[1]:g}]"


def _names(found: dict[str, float]) -> str:
    """The first three names of what a robot stands in or by, most first."""
    return ", ".join(list(found)[:3]) + (" and more" if len(found) > 3 else "")


def _noted(result: dict[str, Any], *notes: str | None) -> dict[str, Any]:
    """Add the notes that are not None to the result's "notes"."""
    added = [n for n in notes if n]
    if added:
        result["notes"] = [*result.get("notes", []), *added]
    return result


def _yaw_note(yaw: float | None) -> str | None:
    """yaw is degrees; a fraction of a radian-sized number (1.57, 3.14) is almost surely radians."""
    if yaw is None or yaw == round(yaw) or not 0 < abs(yaw) <= 2 * math.pi + 0.01:
        return None
    return f"yaw {yaw:g} is degrees, so it turns only {yaw:g} degrees; if that was radians, use {math.degrees(yaw):.0f}"


def _renamed(result: dict[str, Any], given: str, name: str) -> dict[str, Any]:
    """Say so when a name was saved in its tidy form ("Red Cube" as red_cube), since later calls may use either."""
    if given != name:
        result["notes"] = [*result.get("notes", []), f"{given!r} is saved as {name!r}: names are lowercase with _"]
    return result


class _Mounts(BaseModel):
    mounts: list[MountEntry]


def _mounts(mounts: list[Any]) -> list[MountEntry]:
    """A mounts list checked as add_robot checks it, so that errors say where: "mounts[0]: ..."."""
    try:
        return _Mounts.model_validate({"mounts": mounts}).mounts
    except ValidationError as exc:
        raise PlaygroundError(_validation_message(exc)) from exc


def _validation_message(exc: ValidationError) -> str:
    return wording.validation_message(exc)


def _name(m: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(m, obj, idx) or ""


def _aircraft(m: mujoco.MjModel, behaviors) -> list[dict[str, Any]]:
    """The robots that fly: their rotors (thrust actuators), what flies them now and what they are doing."""
    from robotforge.behaviors import Kind

    out = []
    for robot in behaviors.robot_names():
        if behaviors.kind(robot) != Kind.AIRCRAFT:
            continue
        root = airframe.root_of(m, robot)
        names = [_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in airframe.thrust_actuators(m, robot, root)]
        info = behaviors.flight_info(robot)
        keep = ("controller", "state", "altitude_m", "cannot_fly")
        out.append({"robot": robot, "rotors": names, **{k: info[k] for k in keep if k in info}})
    return out


def _free_bodies(m: mujoco.MjModel) -> list[int]:
    """Bodies attached to the world by a free joint: robots' bases and loose objects."""
    return [int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]


def _limit(items: list) -> list:
    return items if len(items) <= _LIST_LIMIT else [*items[:_LIST_LIMIT], f"... {len(items) - _LIST_LIMIT} more"]


def _limit_dict(d: dict) -> dict:
    if len(d) <= _LIST_LIMIT:
        return d
    keys = list(d)[:_LIST_LIMIT]
    return {**{k: d[k] for k in keys}, "...": f"{len(d) - _LIST_LIMIT} more"}


def viewer_catalog() -> dict[str, Any]:
    """What the viewer's World panel offers: scene presets, ground styles with their colors, terrain types."""
    return {
        "scenes": [{"name": s["name"], "title": s["title"], "summary": s["summary"]} for s in scenes.catalog()],
        "grounds": {name: [round(c, 3) for c in style.rgba[:3]] for name, style in terrain.STYLES.items()},
        "terrains": ["flat", *terrain.TYPES],
    }


def _short(result: dict[str, Any]) -> dict[str, Any]:
    """The viewer only needs what changed, not the full world summary."""
    return {k: result[k] for k in ("world", "robots", "terrain", "notes") if k in result}
