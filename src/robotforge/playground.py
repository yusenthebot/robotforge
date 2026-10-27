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
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
from pydantic import BaseModel, ValidationError

from robotforge import modes, wording
from robotforge.behaviors import airframe, robot_kind
from robotforge.behaviors.manager import BehaviorManager
from robotforge.demonstrations import Demonstrations
from robotforge.design import catalogue, combine, gallery, importer, scenes, terrain
from robotforge.design import mechanisms as mechanism_shapes
from robotforge.design.builder import edit as build_edit
from robotforge.design.builder import plate
from robotforge.design.store import WorldStore, worlds_home
from robotforge.design.water import WaterEntry
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
from robotforge.extend import plugin_notes
from robotforge.flow_board import FlowBoard
from robotforge.learn import LESSONS, LessonRunner
from robotforge.perception.hub import PerceptionHub
from robotforge.sensors import edit as sensor_edit
from robotforge.sensors import without_sensors
from robotforge.sensors.hub import SensorHub
from robotforge.sim import carry, finder, lidar, mechanisms, placement, vision
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.render import VIEWS, SnapshotRenderer
from robotforge.sim.session import SimSession, fresh_data
from robotforge.viewer.held_drive import HeldDrive
from robotforge.viewer.server import ViewerHub

if TYPE_CHECKING:
    from robotforge.design import generator
    from robotforge.viewer.creation import Addition
    from robotforge.viewer.object_edit import Edit
    from robotforge.viewer.robot_edit import RobotEdit
    from robotforge.viewer.terrain_create import TerrainAddition
    from robotforge.viewer.terrain_edit import TerrainEdit

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
    sensors: SensorHub = field(init=False)  # the sensors robots carry, sampled while time passes (sensors/)
    perception: PerceptionHub = field(init=False)  # the maps robots draw and localize in (perception/)
    flow: FlowBoard = field(default_factory=FlowBoard, init=False)  # the behavior tree a script shows (flow.py)
    demonstrations: Demonstrations = field(init=False)  # recordings, replays and exports (demonstrations.py)
    _edit_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _shapes: tuple[int, dict[str, str]] = field(default=(0, {}), init=False, repr=False)  # see _mechanism_shapes
    _soft: tuple[int, set[int]] = field(default=(-1, set()), init=False, repr=False)  # see _soft_bodies
    # The ground places of the scene last loaded, and the world it went into: a robot added there without pos
    # starts on the first free one. Not saved: a world opened later starts robots round [0, 0].
    _places: tuple[Path, list[scenes.Slot]] | None = field(default=None, init=False, repr=False)
    # Each task's state as its attempt began (pick_up and press compare with it), and the probes that read tasks
    # from the model, for the model they were made for (design/tasks/live.py)
    _task_starts: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _probes: tuple[int, dict[str, Any]] = field(default=(-1, {}), init=False, repr=False)
    _play_addition: Addition | TerrainAddition | None = field(default=None, init=False, repr=False)
    _play_edit: Edit | None = field(default=None, init=False, repr=False)
    _play_robot_edit: RobotEdit | None = field(default=None, init=False, repr=False)
    _play_terrain_edit: TerrainEdit | None = field(default=None, init=False, repr=False)
    _play_world_sha256: str | None = field(default=None, init=False, repr=False)
    _play_world_descriptor: str | None = field(default=None, repr=False)
    _layout_reset_hooks: tuple[Any, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        initial_hooks = len(self.session.reset_hooks)
        if self.store is not None:
            import hashlib

            raw = self.store.file.read_bytes()
            descriptor = hashlib.sha256(
                World.from_yaml(raw.decode("utf-8")).model_dump_json().encode("utf-8")
            ).hexdigest()
            if descriptor == self._play_world_descriptor:
                self._play_world_sha256 = hashlib.sha256(raw).hexdigest()
        self.behaviors = BehaviorManager(
            self.session, self._acting_world, lambda: self.store.dir if self.store is not None else None
        )
        self.lessons = LessonRunner(self, LESSONS)
        self.session.reset_hooks.append(self.lessons.after_reset)
        self.sensors = SensorHub(self.session)
        self.perception = PerceptionHub(self)
        self.demonstrations = Demonstrations(self)
        self.session.reset_hooks.append(self.flow.clear)  # a reset starts the world over: the tree's progress is stale
        # belts keep running and button presses are counted, through resets
        mechanisms.attach(self.session, self._mechanism_shapes)
        self.session.reset_hooks.append(self._task_starts.clear)  # a reset starts every task's attempt again
        self._layout_reset_hooks = tuple(self.session.reset_hooks[initial_hooks:])

    def attach_viewer(self, hub: ViewerHub) -> None:
        """Let the viewer drive robots from the keyboard and list them in its UI."""

        hub.drive_holds.bind(self.behaviors)

        def teleop(msg: dict[str, Any]) -> HeldDrive | None:
            walker = msg.get("walker")  # set only when the user switches between the gait and a learned policy
            axes = {key: float(msg.get(key, 0.0)) for key in ("speed", "turn")}
            axes.update(
                {key: float(msg[key]) if msg.get(key) is not None else 0.0 for key in ("side", "climb", "heave")}
            )
            if not all(math.isfinite(value) for value in axes.values()):
                raise ValueError("teleop axes must be finite numbers")
            for key, expected in (("version", self.session.version), ("generation", self.session.generation)):
                if key in msg and (type(msg[key]) is not int or msg[key] != expected):
                    raise ValueError("The driving scene changed; release and press again")
                if "hold_ms" in msg and key not in msg:
                    raise ValueError("held driving requires version and generation")
            robot = self.behaviors.resolve(str(msg["robot"]))
            self.behaviors.move(
                robot,
                axes["speed"],
                axes["turn"],
                str(walker) if walker else None,
                side=axes["side"],
                climb=axes["climb"],
            )
            if msg.get("heave") is not None:  # an underwater robot going up or down (keys E and Q)
                self.behaviors.heave(robot, axes["heave"])
            if "hold_ms" not in msg:
                return None
            controller = self.session.controllers.get(robot)
            model, generation = self.session.model, self.session.generation

            def release() -> None:
                # A trip can reuse its pilot. Never stop a newer navigation or replacement controller.
                if (
                    self.session.model is model
                    and self.session.generation == generation
                    and self.session.controllers.get(robot) is controller
                    and f"__navigate__{robot}" not in self.session.controllers
                ):
                    self.behaviors.heave(robot, 0.0)
                    self.behaviors.stop(robot)

            return HeldDrive(robot, release)

        def teleop_arm(msg: dict[str, Any]) -> None:
            """An arm's tool jogged from a gamepad: velocity [x, y, z] m/s in the world, and/or the gripper. The hand
            sends it again every fraction of a second while a stick is held; an arm not told again holds still."""
            velocity = [float(v) for v in msg.get("velocity", [0.0, 0.0, 0.0])]
            grip = msg.get("gripper")
            self.behaviors.jog_arm(str(msg["robot"]), velocity, str(grip) if grip else None)

        def go_to(msg: dict[str, Any]) -> dict[str, Any]:
            """A double-click on the ground: go there along a planned path, as move(to=...) does, until it arrives.
            Planned off the hub's event loop: 60 to 220 ms in the scenes, over a second for a place closed off in the
            park (working out why), and the stream stood still for that long."""
            to = msg["to"] if isinstance(msg["to"], str) else [float(v) for v in msg["to"]]
            speed = float(msg["speed"]) if msg.get("speed") is not None else None
            walker = msg.get("walker")
            trip = self.behaviors.navigate(str(msg["robot"]), to, speed, str(walker) if walker else None)
            return {"path_m": round(trip.plan.length, 2)}

        contract_key: tuple | None = None
        contracts: dict[str, Any] = {}

        def task_contracts(world: World | None, *, copy_result: bool = True) -> dict[str, Any]:
            from robotforge.control.runtime import contract_catalog
            from robotforge.control.task_registry import task_registry_revision
            from robotforge.extend import REGISTRY

            nonlocal contract_key, contracts
            key = (
                self.session.version,
                task_registry_revision(),
                tuple(
                    (name, id(item), id(item.obj), item.summary, item.plugin, repr(item.meta))
                    for name, item in REGISTRY.of("algorithm").items()
                ),
                tuple(entry.model_dump_json() for entry in world.tasks) if world is not None else (),
            )
            if key != contract_key:
                contracts = contract_catalog(world)
                contract_key = key
            return copy.deepcopy(contracts) if copy_result else contracts

        def task_run(msg: dict[str, Any]) -> dict[str, Any]:
            from robotforge.control.runtime import start_control

            # Admission belongs inside this worker so queued requests cannot outlive their lease.
            with self.session.execution.command("task_run"):
                return start_control(
                    self,
                    msg["robot"],
                    msg["task"],
                    msg["method"],
                    msg.get("task_params"),
                    msg.get("method_params"),
                )

        def extras() -> dict[str, Any]:
            from robotforge.viewer.object_edit import metadata
            from robotforge.viewer.robot_edit import metadata as robot_metadata
            from robotforge.viewer.terrain_edit import metadata as terrain_metadata

            world = self.current_world()
            robots = self.behaviors.robots()
            learned = self.behaviors.learned_walkers()
            return {
                "robots": robots,
                "object_edit": metadata(self),
                "robot_edit": robot_metadata(self),
                "terrain_edit": terrain_metadata(self),
                "task_contracts": task_contracts(world),
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
            if action in {"layout_list", "layout_save", "layout_restore", "layout_remove"}:
                return self.layouts_for_play(msg)
            if action == "add_robot":
                return self.add_robot_for_play(msg)
            if action == "add_model":
                return self.add_model_for_play(msg)
            if action == "add_terrain":
                return self.add_terrain_for_play(msg)
            if action in {"edit_terrain", "undo_terrain_edit"}:
                return self.edit_terrain_for_play(msg)
            if action in {"edit_robot_start", "undo_robot_start"}:
                return self.edit_robot_start_for_play(msg)
            if action in {"edit_object", "remove_object", "undo_object_edit", "duplicate_object"}:
                return self.edit_object_for_play(msg)
            if action in {"add_object", "undo_add"}:
                return self.object_for_play(msg)
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
        hub.handlers["teleop_arm"] = teleop_arm
        hub.handlers["lesson"] = self.lessons.handle
        hub.slow_handlers["lesson_open"] = lesson_open
        hub.slow_handlers["go_to"] = go_to

        def pick_and_place(msg: dict[str, Any]) -> dict[str, Any]:
            """Pick an object up and put it on or in a target, as the pick_and_place tool does, without waiting for
            it to finish: a robot that goes fetches it (scripts/screenshots.py pictures one on its way)."""
            arm = self.behaviors.resolve(str(msg["robot"]), arm=True)
            target = msg["target"] if isinstance(msg["target"], str) else [float(v) for v in msg["target"]]
            _, info = self.behaviors.pick_and_place(arm, str(msg["object"]), target)
            return {"object": info["object"], "target": info["target"]}

        hub.slow_handlers["pick_and_place"] = pick_and_place

        def lesson_bench(msg: dict[str, Any]) -> dict[str, Any]:
            """Sweeps run many tests on copies of the rig: off the hub thread, answered when done."""
            self.lessons.handle(msg)
            return {"done": msg.get("action")}

        hub.slow_handlers["lesson_bench"] = lesson_bench
        hub.lesson_stream = self.lessons.tick
        hub.slow_handlers["world"] = world_request
        hub.slow_handlers["task_run"] = task_run
        hub.slow_handlers["mode"] = lambda msg: modes.apply(self, str(msg["robot"]), str(msg["mode"]))
        hub.scene_extras = extras
        last_description: dict[str, Any] | None = None

        def robot_update() -> dict[str, Any] | None:
            nonlocal last_description
            # Called only by the broadcast loop: a joining client's welcome must
            # never consume an update intended for the other viewers.
            version = self.session.version
            rows = self.behaviors.robots()
            rows = [{k: v for k, v in r.items() if k not in ("flight", "balance")} for r in rows]
            learned = self.behaviors.learned_walkers()
            description = {
                "type": "robot_info",
                "version": version,
                "robots": rows,
                "task_contracts": task_contracts(self.current_world(), copy_result=False),
                "learned": learned,
                "modes": {r["name"]: modes.available(self, r, r["name"] in learned) for r in rows},
            }
            if self.session.version != version or description == last_description:
                return None
            # Keep the cached catalog private, but copy only a message we actually send.
            emitted = copy.deepcopy(description)
            last_description = description
            return emitted

        hub.robot_stream = robot_update
        hub.state_extras = lambda: {"mode": modes.current(self), **self.demonstrations.viewer_state()}
        hub.flow = self.flow
        hub.sensor_stream = self.sensors.viewer_message
        hub.map_stream = self.perception.viewer_message
        from robotforge.training.live import install as install_training

        install_training(self, hub)  # the Training panel: the run that changed last, its play and stop buttons

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
        # Compile before detaching the saved world: a bad file must leave it editable.
        loaded = SimSession.from_path(p.resolve())
        self.store = None
        self.last_doctor = {}
        self._places = None
        self.session.load_model(loaded.model, loaded.source)
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
            sensors = [n for s in range(m.nsensor) if not _added_sensor(n := _name(m, mujoco.mjtObj.mjOBJ_SENSOR, s))]
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
        opened = self._load_world(compiled)
        opened["notes"].extend(plugin_notes(store.dir))  # said once, as the world opens, not with every edit after
        return opened

    @_one_edit_at_a_time
    def layouts_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Save/restore authored layouts in the current world's original asset directory."""
        from robotforge.viewer.layouts import handle

        return handle(self, message)

    @_one_edit_at_a_time
    def add_robot_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Place an allowlisted local starter from the viewer without overwriting an existing robot."""
        from robotforge.viewer.creation import place

        return place(self, message)

    @_one_edit_at_a_time
    def object_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Place a local prop or undo the exact latest viewer addition."""
        from robotforge.viewer.creation import place_object, undo_addition

        return undo_addition(self, message) if message.get("action") == "undo_add" else place_object(self, message)

    @_one_edit_at_a_time
    def add_terrain_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Add a fixed local terrain preset on clear ground; pause at saved starts."""
        from robotforge.viewer.terrain_create import handle

        return handle(self, message)

    @_one_edit_at_a_time
    def edit_terrain_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Move an authored ramp/stair block on clear ground, or undo its latest move."""
        from robotforge.viewer.terrain_edit import handle

        return handle(self, message)

    @_one_edit_at_a_time
    def edit_object_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Reposition/copy/remove an exact existing rigid prop, or undo that one edit."""
        from robotforge.viewer.object_edit import duplicate, edit, undo

        action = message.get("action")
        if action == "duplicate_object":
            return duplicate(self, message)
        return undo(self, message) if action == "undo_object_edit" else edit(self, message)

    @_one_edit_at_a_time
    def add_model_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Place a curated model or saved design selected by its server-owned ID."""
        from robotforge.viewer.model_create import place

        return place(self, message)

    @_one_edit_at_a_time
    def edit_robot_start_for_play(self, message: dict[str, Any]) -> dict[str, Any]:
        """Edit an exact authored robot's start without changing its configuration."""
        from robotforge.viewer.robot_edit import handle

        return handle(self, message)

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
        end_effector: str | None = None,
        describe: str | None = None,
    ) -> dict[str, Any]:
        given, name = name, tidy_name(name)
        _check_meters("pos", pos, MAX_POSITION)
        catalogued = None
        if library is not None:  # the tuned library first, then the open catalogue (list_library finds both)
            try:
                library, catalogued = catalogue.source_of(library)
            except catalogue.CatalogueError as exc:
                raise PlaygroundError(str(exc)) from exc
        saved: dict[str, Any] = {}
        if design is not None and gallery.canonical(design) is None and combine.saved_design(wording.squash(design)):
            extra = [k for k, v in dict(template=template, file=file, library=library, build=build).items() if v]
            if extra or params or describe is not None:
                raise PlaygroundError(
                    f"design is a whole robot on its own; leave out {', '.join(extra or ['params'])} (mount adds to it)"
                )
            saved = combine.entry_fields(wording.squash(design))  # a combination kept with mount(save_as=...)
            design = None
        made = None
        if describe is not None:
            made = described_robot(describe, params, template=template, file=file, library=library, build=build)
            if design is not None:
                raise PlaygroundError("give describe or design, not both: each is a whole robot")
            template, params, build = made.source.get("template"), made.source.get("params"), made.source.get("build")
        if design is not None:
            build = design_build(design, template=template, file=file, library=library, build=build, params=params)
        try:
            if saved:
                fields = {**saved, "name": name, "pos": pos or [0.0, 0.0], "yaw": yaw or 0.0}
                fields["mounts"] = [*saved.get("mounts", []), *(mounts or [])]
                if end_effector is not None:
                    fields["end_effector"] = end_effector
                entry = RobotEntry.model_validate(fields)
            else:
                entry = RobotEntry(
                    name=name,
                    template=template,
                    file=file,
                    library=library,
                    catalogue=catalogued,
                    build=build,
                    params=params or {},
                    pos=pos or [0.0, 0.0],
                    yaw=yaw or 0.0,
                    mounts=mounts or [],
                    end_effector=end_effector,
                    generated=made.plan.to_dict() if made is not None else None,
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

        # The builder has checked a generated robot (generator.realize): what the world refuses now (a name taken, a
        # place outside it) is not the sentence's fault and comes back as it is.
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

            result = self._edit(move, snapshot=False)  # a new result: the generated report goes on it below
        if made is not None:
            result["generated"] = made.describe(name)
        if name in self.last_doctor:  # URDFs, xacro and meshes come with the doctor's report
            result["doctor"] = self.last_doctor[name]
        if catalogued is not None:
            result["catalogue"] = self._about_catalogue_robot(name, catalogued)
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
        end_effector: str | None = None,
        describe: str | None = None,
    ) -> dict[str, Any]:
        _check_meters("pos", pos, MAX_POSITION)
        kept: list[str] = []
        mounted = None if mounts is None else _mounts(mounts)
        found: list[str] = []
        remade: list[generator.Generated] = []
        if describe is not None and (params or build is not None):
            raise PlaygroundError(
                "describe changes a robot made from a sentence by another sentence; give params or build separately"
            )
        swap = self._replacing(name, build) if build is not None and _whole(build) else None
        replaced = [swap[0]] if swap else []
        if swap:
            build = swap[1]  # a lidar alone for a template or library robot: the robot carried with it on a plate

        def change(w: World) -> None:
            robot = _find(w.robots, name, "robot")
            found.append(robot.name)
            asked = (params, build, mounts, end_effector, describe)
            if not params and pos is None and yaw is None and all(a is None for a in asked[1:]):
                raise PlaygroundError(
                    "nothing to change: give params, pos, yaw, build, mounts, end_effector or describe"
                )
            if describe is not None:
                remade.append(_regenerate(robot, describe, kept))
            if params:
                robot.params = {**robot.params, **params}
            if pos is not None:
                robot.pos = pos
            if yaw is not None:
                robot.yaw = yaw
            if mounted is not None:  # the whole list: [] takes every mounted robot off
                robot.mounts = mounted
            if end_effector is not None:  # a tool change, checked by the world's own validation
                robot.end_effector = end_effector
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
        if remade:
            result["generated"] = remade[0].describe(found[0])
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
            after = robot_kind(self._compiled_with(entry.name, bare), entry.name, by_feet=False).value
            if after == kind:  # the plate keeps it what it was; then the part has its say (it may overlap the robot)
                after = robot_kind(self._compiled_with(entry.name, whole), entry.name, by_feet=False).value
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
            still = robot_kind(self._compiled_with(entry.name, bare), entry.name, by_feet=False).value == kind
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

    @_one_edit_at_a_time
    def mount(self, what: str | None, on: str, **options: Any) -> dict[str, Any]:
        """Mount a robot, an end effector or a sensor on any body of a robot (combine.py), checked once it is on;
        save_as keeps the result as a design."""
        try:
            return combine.mount(self, what, on, **options)
        except ValueError as exc:
            if isinstance(exc, PlaygroundError):
                raise
            raise PlaygroundError(str(exc)) from None

    @_one_edit_at_a_time
    def add_sensor(self, robot: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Put a sensor on any robot (sensors/edit.py); remove("robot/sensor") takes it off."""
        return sensor_edit.add(self, robot, fields)

    def read_sensors(
        self,
        robot: str,
        names: list[str] | None = None,
        image: bool = False,
        full: bool = False,
        points: int | None = None,
    ) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        world = self.current_world()
        robots = [r.name for r in world.robots] if world is not None else []
        try:
            return self.sensors.read(self.sensors.resolve_robot(robot, robots), names, image, full, points)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc

    def show_sensors(self, names: list[str]) -> list[str]:
        """Draw these sensors in the viewer (open_viewer(show=...)); returns their full names."""
        world = self.current_world()
        robots = [r.name for r in world.robots] if world is not None else []
        try:
            return self.sensors.show(names, robots)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc

    # ---- robots built from parts ------------------------------------------------------

    @_one_edit_at_a_time
    def add_part(self, robot: str, part: dict[str, Any]) -> dict[str, Any]:
        return self._edit_build(robot, lambda b: build_edit.add_part(b, part), adding=part)

    @_one_edit_at_a_time
    def update_part(self, robot: str, part: str, changes: dict[str, Any]) -> dict[str, Any]:
        def follow(entry: RobotEntry, before: dict[str, Any]) -> None:  # sensors on a renamed part go with it
            old = set(build_edit.part_names(before)) - set(build_edit.part_names(entry.build))
            new = set(build_edit.part_names(entry.build)) - set(build_edit.part_names(before))
            if len(old) == 1 and len(new) == 1:
                sensor_edit.follow_rename(entry, old.pop(), new.pop())

        return self._edit_build(robot, lambda b: build_edit.update_part(b, part, changes), follow)

    @_one_edit_at_a_time
    def remove_part(self, robot: str, part: str) -> dict[str, Any]:
        removed: list[str] = []
        dropped: list[str] = []

        def change(b: dict[str, Any]) -> dict[str, Any]:
            new, gone = build_edit.remove_part(b, part)
            removed.extend(gone)
            return new

        def take_off(entry: RobotEntry, _before: dict[str, Any]) -> None:  # what sat on a removed part goes too
            dropped.extend(sensor_edit.drop_on(entry, removed))

        result = self._edit_build(robot, change, take_off)
        result["removed_parts"] = removed
        if dropped:
            result["removed_sensors"] = dropped
        return result

    def _edit_build(
        self,
        robot: str,
        change_build: Callable[[dict[str, Any]], dict[str, Any]],
        then: Callable[[RobotEntry, dict[str, Any]], None] | None = None,
        adding: dict | None = None,
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
            before = entry.build
            entry.build = change_build(entry.build)
            if then is not None:
                then(entry, before)

        return self._edit(change)

    @_one_edit_at_a_time
    def add_object(self, name: str, **fields: Any) -> dict[str, Any]:
        given, name = name, tidy_name(name)
        described, said = fields.pop("describe", None), []
        if described is not None or fields.get("seed") is not None or fields.get("params"):
            from robotforge.objects import describe as describing

            try:
                fields, said = describing.resolve(fields, described)
            except ValueError as exc:
                raise PlaygroundError(str(exc)) from exc
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

        return _noted(_renamed(self._edit(change), given, name), cube, _yaw_note(fields.get("yaw")), *said)

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
        world = self.current_world()
        if world is not None and combine.find_mount(world, name) is not None:  # "rover/arm": a mounted robot
            return combine.take_off(self, *combine.find_mount(world, name))
        sensor = sensor_edit.find(world, name) if world is not None else None
        if sensor is not None:  # "go2/lidar": a sensor added to a robot
            return sensor_edit.remove(self, *sensor)

        def change(w: World) -> None:
            names = [*(r.name for r in w.robots), *(o.name for o in w.objects), *(t.name for t in w.terrain)]
            names += [t.name for t in w.tasks if t.name not in names]
            found = wording.match(name, names, plurals=False)  # "dogs" may be dog, dog2 or both: ask, do not delete
            if found is None:
                kinds = "robot, object, terrain patch or task" if w.tasks else "robot, object or terrain patch"
                text = wording.unknown(kinds, name, names, many="names")
                raise PlaygroundError(
                    f"{text}. The floor is changed with set_floor" if "floor" in name.lower() else text
                )
            w.robots = [r for r in w.robots if r.name != found]
            w.objects = [o for o in w.objects if o.name != found]
            w.terrain = [t for t in w.terrain if t.name != found]
            w.tasks = [t for t in w.tasks if t.name != found]

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
            w.water = None
            w.objects = []

        self._places = None  # the scene's places went with it
        return self._edit(change)

    @_one_edit_at_a_time
    def set_floor(self, style: str | None = None, water: Any = None) -> dict[str, Any]:
        """The floor's look and grip, and the water over it: water=1.2 (or {"level": 1.2, "waves": 0.03}) fills
        the world with water up to 1.2 m, the floor becoming its bottom; water=0 or false takes it away."""
        if style is None and water is None:
            raise PlaygroundError("give style (the floor's look) or water (a level in meters, 0 for none)")
        entry: WaterEntry | None = None
        if water is not None and water is not False and water != 0:
            try:
                entry = WaterEntry.model_validate(water)
            except ValidationError as exc:
                raise PlaygroundError(f"water: {_validation_message(exc)}") from exc

        def change(w: World) -> None:
            if style is not None:
                w.floor.style = style
            if water is not None:
                w.water = entry

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

    def _edit(self, change, snapshot: bool = True, *, exact_carry: bool = False) -> dict[str, Any]:
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
        # New ground under them (terrain, a scene, water) starts everything fresh: their heights would no longer fit.
        if before.terrain == after.terrain and before.water == after.water:
            kept = [without_sensors(r) for r in before.robots]  # a sensor added or removed moves no robot
            unchanged = [r.name for r in after.robots if without_sensors(r) in kept]
            unchanged += [o.name for o in after.objects if o in before.objects]
            with self.session.lock:
                if exact_carry:
                    keep = set(unchanged)
                    carry._apply(self.session.model, self.session.data, live, lambda name: carry.owner(name) in keep)
                    mujoco.mj_forward(self.session.model, self.session.data)
                    carried = unchanged
                else:
                    carried = carry.restore(self.session.model, self.session.data, live, unchanged)
                self.behaviors.let_go(hands_off.intersection(carried))  # legs set by hand stay set
                # First samples from where the robots now are; their sensors' estimates carry on (odometry).
                self.sensors.rebuild(kept=carried)
                self.perception.rebuild(carried)
        return result

    def _about_catalogue_robot(self, name: str, catalogue_id: str) -> dict[str, Any]:
        """What an open catalogue robot turned out to be, from its structure, and which tools work on it."""
        from robotforge.design import catalogue_describe, catalogue_probe

        entry = catalogue.get(catalogue_id)
        described = catalogue_describe.describe(
            self.session.model, name, self.behaviors.home_qpos(), entry, self.behaviors
        )
        if described.kind in catalogue_probe.LEGGED and described.legs:  # the gait is generic: try it, once
            checked = catalogue_probe.trial(entry)
            if checked is not None:
                described = described.with_trial(checked)
        return {"name": entry.id, **described.to_dict()}

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
        import hashlib

        store = self._require_store()
        self._play_world_descriptor = compiled.world_sha256
        raw = store.file.read_bytes()
        descriptor = hashlib.sha256(World.from_yaml(raw.decode("utf-8")).model_dump_json().encode("utf-8")).hexdigest()
        self._play_world_sha256 = hashlib.sha256(raw).hexdigest() if descriptor == compiled.world_sha256 else None
        self.session.load_model(compiled.model, str(store.file))
        self._play_addition = self._play_edit = self._play_robot_edit = self._play_terrain_edit = None
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

    # ---- tasks (design/tasks) --------------------------------------------------------------

    @_one_edit_at_a_time
    def add_task(
        self, sentence: str, robot: str | None = None, name: str | None = None, seed: int = 0
    ) -> dict[str, Any]:
        """Turn a sentence into a task saved with the open world: what has to happen, judged by a success check
        (observe reports it under "tasks"), with a reward and randomized starts for learning. What the task needs and
        the world lacks is added: the thing it names, a bin, a robot that can do it."""
        from robotforge.design.tasks import make

        store = self._require_store()
        world = store.load()
        infos = [
            make.RobotInfo(r["name"], r["kind"], list(r.get("arms") or []), tuple(_entry_pos(world, r["name"])))
            for r in self.behaviors.robots()
        ]
        try:
            plan = make.plan(world, sentence, infos, robot, name, seed, lambda e: spawn_height(e, store.dir))
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
        task = plan.task
        notes = list(plan.notes)
        if plan.add_objects:

            def furnish(w: World) -> None:
                w.objects += [ObjectEntry(**o) for o in plan.add_objects]

            self._edit(furnish)
        if plan.add_robot is not None:
            fields = dict(plan.add_robot)
            build = fields.pop("build", None)
            added = self.add_robot(fields.pop("name"), build=build, **fields)
            for spot in plan.arm_spots[1:]:  # an arm put beside a mechanism: a place where it stands clear
                if not any(n.startswith(f"{task.robot} starts") for n in added.get("notes", [])):
                    break
                added = self.update_robot(task.robot, pos=spot["pos"], yaw=spot["yaw"])
                clash = self._clear_start(task.robot, spot["pos"])
                added["notes"] = [clash] if clash else []
            notes += [n for n in added.get("notes", []) if n.startswith(f"{task.robot} ")][:2]
            entry = next(r for r in self.current_world().robots if r.name == task.robot)
            task.start.robot = [float(entry.pos[0]), float(entry.pos[1]), float(entry.yaw)]

        def save(w: World) -> None:
            w.tasks = [t for t in w.tasks if t.name != task.name] + [task]

        result = self._edit(save)
        self._task_starts.pop(task.name, None)
        status = self.task_status().get(task.name, {})
        if status.get("done") and task.kind == "go_to":  # "go to the kitchen" from the kitchen: start it elsewhere
            moved = self._start_elsewhere(task)
            if moved is not None:
                notes.append(moved)
                status = self.task_status().get(task.name, {})
        if status.get("done"):
            notes.append(f"{task.describe()} is so already: the task is done before it starts")
        result["task"] = {**task.model_dump(mode="json", exclude={"start", "randomize"}), "goal": task.describe(),
                          "status": status}  # fmt: skip
        result["notes"] = notes + [n for n in result.get("notes", []) if "task" in n]
        return result

    def _start_elsewhere(self, task: Any) -> str | None:
        """Move a go_to task's robot (and the task's start) to the first of the scene's places from which it still
        has to go; a note saying so, or None when there is none."""
        from robotforge.design.tasks.check import State, judge

        world = self.current_world()
        for slot in self._scene_places():
            if judge(task, world, State(robot=(slot.pos[0], slot.pos[1], 0.0))).done:
                continue

            def move(w: World, slot: scenes.Slot = slot) -> None:
                robot = _find(w.robots, task.robot, "robot")
                robot.pos, robot.yaw = [slot.pos[0], slot.pos[1]], slot.yaw
                saved = next(t for t in w.tasks if t.name == task.name)
                saved.start.robot = [slot.pos[0], slot.pos[1], slot.yaw]

            self._edit(move, snapshot=False)
            self._task_starts.pop(task.name, None)
            return f"{task.robot} starts at {_xy(slot.pos)}, away from {task.target}, so it has somewhere to go"
        return None

    def task_status(self) -> dict[str, dict[str, Any]]:
        """Each task of the open world: its goal and the check's verdict now (done, distance still to go, reward,
        why)."""
        world = self.current_world()
        if world is None or not world.tasks:
            return {}
        from robotforge.design.tasks.live import Probe

        version, probes = self._probes
        if version != self.session.version:
            probes = {}
            self._task_starts.clear()
        m, d = self.session.model, self.session.data
        out = {}
        with self.session.lock:
            watcher = self.session.controllers.get(mechanisms.KEY)
            for task in world.tasks:
                probe = probes.get(task.name)
                if probe is None or probe.task != task:
                    probe = probes[task.name] = Probe(m, task, world)
                start = self._task_starts.get(task.name)
                if start is None:
                    start = self._task_starts[task.name] = probe.state(d, watcher)
                verdict = probe.verdict(d, start, watcher)
                out[task.name] = {"goal": task.describe(), "robot": task.robot, **verdict.as_dict()}
        self._probes = (self.session.version, probes)
        return out

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
        preset: str | None = None,
        seed: int = 0,
        keep_robots: bool = True,
        starter: bool = True,
        options: dict[str, float] | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Replace the floor, terrain and objects with a ready-made scene, or one generated from `description` (a
        sentence: design/generate), opening a world named after it if none is open. Kept robots move to the scene's
        places; an empty world gets the scene's starter robot."""
        generated = None
        try:
            if description is not None:
                from robotforge.design import generate  # only when asked: it is not needed to start

                if preset is not None:
                    raise ValueError(
                        "give preset (a ready-made scene) or description (a sentence to generate one from), not both"
                    )
                if options:
                    raise ValueError("options size the ready-made scenes; a description says the size in words")
                generated = generate.generate(description, seed)
                content = generated.content
            elif preset is None:
                raise ValueError(f"give preset (one of {', '.join(scenes.SCENES)}) or description (a sentence)")
            else:
                content = scenes.build(preset, seed, options)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc
        if preset is None:
            from robotforge.design import generate

            preset = generate.world_name(description)
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
            w.water = WaterEntry.model_validate(content.water) if content.water else None
            w.objects = [ObjectEntry(**{**o, "name": _unused(o["name"], taken)}) for o in content.objects]

        result = self._edit(change)
        self._places = (self.store.dir, list(content.ground_slots))
        result["scene"] = {
            **({"generated": generated.summary()} if generated is not None else {"preset": preset}),
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
        x, y = entry.pos[:2] if entry is not None else (0.0, 0.0)
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
            self.session.execution.require_idle()
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
        """Step as fast as possible until `until()` or `max_seconds` of sim time; returns sim seconds run. A simulation
        that blows up is reset to its start by MuJoCo (its clock goes back to 0): the run ends there, and the seconds
        returned are then short of `max_seconds` (it used to wait forever for a clock that kept starting over)."""
        with self.session.lock:
            start = last = self.session.data.time
            chunk = max(1, round(0.02 / self.session.model.opt.timestep))
            while not until() and self.session.data.time - start < max_seconds:
                self.session.step(chunk)
                if self.session.data.time < last:
                    return max(0.0, float(last - start))
                last = self.session.data.time
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
            soft = self._soft_bodies()
            for j in range(m.njnt):
                jtype = int(m.jnt_type[j])
                if int(m.jnt_bodyid[j]) in soft:  # a rope's or cloth's hundreds: its summary is in "mechanisms"
                    continue
                if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    joints[_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"] = round(
                        float(d.qpos[m.jnt_qposadr[j]]), 4
                    )
            ctrl = {self._actuator_info(a)["name"]: round(float(d.ctrl[a]), 4) for a in range(m.nu)}
            # add_sensor's own MuJoCo sensors: read_sensors and observe's sensor_summary report those
            sensors = {k: v for k, v in self.sensor_readings().items() if not _added_sensor(k)}
            lidars = lidar.readings(m, d)
            listed = _limit_dict(sensors)
            if lidars and "..." in listed:  # a 64-ray lidar alone is 64 sensors: say where all its rays are
                listed["..."] += "; lidars has every ray of each lidar"
            out = {
                "time": round(float(d.time), 4),
                "paused": self.session.paused,
                "joints": _limit_dict(joints),
                "controls": _limit_dict(ctrl),
                "free_bodies": _limit_dict(self._free_body_positions()),
                "sensors": listed,
                **({"lidars": lidars} if lidars else {}),
                "contacts": int(d.ncon),
                **self._water_state(),
            }
            watcher = self.session.controllers.get(mechanisms.KEY)
            moving = mechanisms.read(m, d, self._mechanism_shapes(), watcher)
            if moving:  # drawers, doors, buttons, switches, levers, valves, belts, ropes, cloths
                out["mechanisms"] = _limit_dict(moving)
            return out

    def _soft_bodies(self) -> set[int]:
        """The bodies of the world's ropes and cloths, whose unnamed joints observe leaves out: one 0.6 m cloth and
        a rope filled its 40 joints with "joint8" to "joint40" and pushed out the joints of anything added later."""
        version, bodies = self._soft
        if version != self.session.version:
            m = self.session.model
            shapes = self._mechanism_shapes().items()
            prefixes = tuple(f"{name}/" for name, shape in shapes if shape in ("rope", "cloth"))
            named = (_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody))
            bodies = {b for b, name in enumerate(named) if prefixes and name.startswith(prefixes)}
            self._soft = (self.session.version, bodies)
        return bodies

    def _mechanism_shapes(self) -> dict[str, str]:
        """The world's mechanisms and soft things by name, read once per compiled model: scripts poll state()
        hundreds of times a second, and the world file only changes with the model."""
        version, shapes = self._shapes
        if version != self.session.version:
            world = self._acting_world()
            objects = world.objects if world is not None else []
            shapes = {o.name: o.shape for o in objects if o.shape in mechanism_shapes.SHAPES}
            self._shapes = (self.session.version, shapes)
        return shapes

    def _water_state(self) -> dict[str, Any]:
        """For a world with water: its level, and how deep in it the things it has touched are (share under the
        surface, depth of their middle), so an agent can tell what floats and what sank."""
        hydro = self.session.water
        if hydro is None:
            return {}
        return {"water": {"level": hydro.water.level, "afloat": hydro.report(self.session.data)}}

    def sensor_readings(self) -> dict[str, Any]:
        """Every sensor's reading by name, none left out (a script reads them all; observe lists the first ones).
        A rangefinder on a robot reads past the robot's own parts, as its lidar does (sim/lidar.seen)."""
        m, d = self.session.model, self.session.data
        with self.session.lock:
            out: dict[str, Any] = {}
            depths = self.session.water.depth_sensors(d) if self.session.water is not None else {}
            for s in range(m.nsensor):
                adr, dim = m.sensor_adr[s], m.sensor_dim[s]
                if dim == 1 and int(m.sensor_type[s]) == mujoco.mjtSensor.mjSENS_RANGEFINDER:
                    values = [round(lidar.seen(m, d, s), 4)]
                else:
                    values = [round(float(v), 4) for v in d.sensordata[adr : adr + dim]]
                reading = depths.get(s, values[0] if dim == 1 else values)  # a depth sensor reads m under the water
                out[_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"] = reading
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


def _entry_pos(world: World, robot: str) -> list[float]:
    entry = next((r for r in world.robots if r.name == robot), None)
    return [float(v) for v in entry.pos[:2]] if entry is not None else [0.0, 0.0]


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


def described_robot(describe: str, params: dict[str, Any] | None, **others: Any) -> generator.Generated:
    """The robot a sentence describes (add_robot(describe=...)): a whole robot, so it comes alone; params may only
    pick its seed (another seed, another variation of the same sentence)."""
    from robotforge.design import generator  # lazy: only robots made from a sentence need it

    extra = [k for k, v in others.items() if v]
    params = dict(params or {})
    seed = params.pop("seed", None)
    if params:
        extra.append(f"params {', '.join(params)}")
    if extra:
        raise PlaygroundError(
            f"describe makes the whole robot from the sentence; leave out {', '.join(extra)} (params may only give a "
            'seed). Change it afterwards with another sentence, update_robot(name, describe="make it longer"), or '
            "part by part with update_part"
        )
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise PlaygroundError(f"params seed is a whole number 0 or more, got {wording.short(seed)}")
    try:
        return generator.generate(describe, seed)
    except generator.SentenceError as exc:
        raise PlaygroundError(str(exc)) from None
    except generator.BuildError as exc:  # its parts did not pass the builder's checks
        plan, _ = generator.read(describe, seed=seed)
        raise PlaygroundError(_generated_failed(describe, plan.seed, exc)) from None


def _regenerate(robot: RobotEntry, describe: str, notes: list[str]) -> generator.Generated:
    """Change a robot made from a sentence as another sentence says, in place: its plan changes and its build (or
    template) is made again from it. Edits made part by part since are replaced, and a note says so."""
    from robotforge.design import generator  # lazy: only robots made from a sentence need it

    if robot.generated is None:
        raise PlaygroundError(
            f"{robot.name} was not made from a sentence, so describe cannot change it; add_robot(name, describe=...) "
            "makes one. Change this one with build, add_part or update_part"
        )
    old = generator.Plan.from_dict(robot.generated)
    try:
        made = generator.change(old, describe)
    except generator.SentenceError as exc:
        raise PlaygroundError(str(exc)) from None
    except generator.BuildError as exc:
        raise PlaygroundError(_generated_failed(describe, old.seed, exc, change=True)) from None
    before = generator.realize(old).source
    if before != {k: v for k, v in (("build", robot.build), ("template", robot.template)) if v is not None} | (
        {"params": robot.params} if robot.template else {}
    ):
        notes.append(f"{robot.name} was made again from its sentences, so the parts changed by hand since are replaced")
    robot.template = made.source.get("template")
    robot.params = made.source.get("params") or {}
    robot.build = made.source.get("build")
    robot.generated = made.plan.to_dict()
    return made


def _generated_failed(describe: str, seed: int, exc: Exception, change: bool = False) -> str:
    """What to do when a robot made from a sentence does not pass the builder's checks."""
    other = 'say "another one" for another variation' if change else f"try params={{'seed': {seed + 1}}}"
    return (
        f"the robot made from {describe!r} did not pass the builder's checks: {exc}. Say it another way (fewer arms, "
        f"a smaller size), or {other}"
    )


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


def _added_sensor(name: str) -> bool:
    """A MuJoCo sensor that add_sensor made (rf_<sensor>_...): its readings come through read_sensors."""
    return name.startswith("rf_") or "/rf_" in name


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
    from robotforge.viewer.creation import catalog, object_catalog
    from robotforge.viewer.model_create import catalog as model_catalog
    from robotforge.viewer.terrain_create import catalog as terrain_catalog

    return {
        "robot_templates": catalog(),
        "robot_models": model_catalog(),
        "object_presets": object_catalog(),
        "terrain_presets": terrain_catalog(),
        "scenes": [{"name": s["name"], "title": s["title"], "summary": s["summary"]} for s in scenes.catalog()],
        "grounds": {name: [round(c, 3) for c in style.rgba[:3]] for name, style in terrain.STYLES.items()},
        "terrains": ["flat", *(t for t in terrain.TYPES if t != "basin")],  # a hollow needs water: the lake scene
    }


def _short(result: dict[str, Any]) -> dict[str, Any]:
    """The viewer only needs what changed, not the full world summary."""
    return {k: result[k] for k in ("world", "robots", "terrain", "notes") if k in result}
