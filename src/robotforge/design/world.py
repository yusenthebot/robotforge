"""world.yaml: the single source of truth for a playground world, and its compiler to MuJoCo.

A world has a floor, terrain (hills, stairs, ramps, ... see terrain.py), robots (from templates,
the robot library, imported files or built from parts) and loose or fixed objects (including
furniture and architecture, see props.py). Every robot's names are prefixed with "<robot>/" so
several robots can share a world without clashing. Robots and objects without an explicit height
rest on the ground under them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from robotforge import wording
from robotforge.design import (
    builder,
    catalogue,
    components,
    end_effectors,
    importer,
    library,
    mechanisms,
    menagerie,
    mjcf_doctor,
    mjcf_stance,
    mounting,
    props,
    templates,
    terrain,
)
from robotforge.design import water as waters
from robotforge.design.tasks.spec import TaskEntry
from robotforge.design.terrain import TerrainPatch
from robotforge.design.visuals import props as looks
from robotforge.design.water import WaterEntry
from robotforge.sensors import fields as source_fields
from robotforge.sensors.spec import SensorEntry
from robotforge.sim import water as water_physics

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# A hand-written `color: #9c6b3f` parses as no color at all (an unquoted # starts a YAML comment).
BARE_HEX_COLOR = re.compile(r"(?:^[ \t-]*|[{,][ \t]*)color:[ \t]*(#[0-9a-fA-F]+)(?![\w-])", re.MULTILINE)
ShapeName = Literal[tuple(props.SHAPES)]  # box, sphere, cylinder, table, bin, wall, shelf, ... (built in)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unset(value: object) -> bool:
    """For fields added after worlds were first saved: left out of a world.yaml while unset, so a world that does not
    use them saves exactly as it did."""
    return value is None


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError(
            f"{wording.short(value)} must use lowercase letters, digits and _, start with a letter and have at most "
            "32 characters, e.g. 'red_cube'"
        )
    return value


def tidy_name(name: str) -> str:
    """ "Red Cube" as red_cube: capitals and spaces have one reading. A name that is still not valid is returned
    as given, so the check says what is wrong with it."""
    loose = wording.squash(name)
    return loose if NAME_RE.match(loose) else name


class Floor(_Strict):
    style: terrain.StyleName = "checker"
    friction: float = Field(1.0, gt=0, le=3)


class MountEntry(_Strict):
    """A whole robot carried by another one (see mounting.py): {"library": "piper"} on a Go2's back."""

    name: str | None = None  # default: what it is ("arm", "gripper")
    library: str | None = None
    template: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    at: str | None = None  # "top" (default), "tool" (an arm's flange; default for a gripper there) or a body name
    shift: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # m, from `at`, in its frame
    rotate: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])  # [roll, pitch, yaw] in degrees
    # A combination saved with mount(save_as=...) (see combine.py), mounted whole: a UR5e with its LEAP hand.
    design: str | None = Field(None, exclude_if=_unset)
    # An adapter plate between the mount and what carries it, sized to both (see mounting.adapter_plate). Left out
    # (worlds saved before there were plates), there is none.
    adapter: bool | None = Field(None, exclude_if=_unset)

    @field_validator("name", "design")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return _check_name(v) if v is not None else None

    @field_validator("template")
    @classmethod
    def _template(cls, v: str | None) -> str | None:
        return templates.canonical(v) if v is not None else None

    @field_validator("library")
    @classmethod
    def _library(cls, v: str | None) -> str | None:
        return library.canonical(v) if v is not None else None

    @model_validator(mode="after")
    def _one_source(self) -> MountEntry:
        if [self.template, self.library, self.design].count(None) != 2:
            raise ValueError(
                'a mount is {"library": "piper"}, {"template": "arm", "params": {...}} or {"design": "<a saved one>"}'
            )
        if self.library is not None:
            library.get(self.library)
        elif self.template is not None:
            templates.get(self.template)
        if len(self.shift) != 3 or len(self.rotate) != 3:
            raise ValueError("a mount's shift is [x, y, z] in meters and its rotate [roll, pitch, yaw] in degrees")
        return self


class RobotEntry(_Strict):
    name: str
    template: str | None = None
    file: str | None = None  # MJCF or URDF path, relative to the world directory
    library: str | None = None  # a robot from the library, e.g. "go2" (see library.CATALOG)
    # An open robot from the catalogue, e.g. "a1_mj" (see catalogue.py): fetched on first use, then read as a file.
    catalogue: str | None = Field(None, exclude_if=_unset)
    build: dict[str, Any] | None = None  # built from parts: {"base": {...}, "parts": [...]} (see builder)
    params: dict[str, Any] = Field(default_factory=dict)
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0])  # x, y (and optional z) in meters
    yaw: float = 0.0  # degrees, counter-clockwise seen from above
    mounts: list[MountEntry] = Field(default_factory=list)  # robots it carries (see mounting.py)
    # What its arm holds with, whatever the arm came with: "suction" (a suction cup), "magnet" (an electromagnet),
    # or "gripper" (the arm's own fingers); None keeps it as built. See end_effectors.py.
    end_effector: str | None = Field(None, exclude_if=_unset)
    sensors: list[SensorEntry] = Field(default_factory=list)  # added to any robot (add_sensor), see sensors/
    # A robot made from a sentence (add_robot(describe=...)): the plan its build came from (body, counts, factors,
    # seed and the sentences so far), so a later sentence changes it (update_robot(describe=...)). See generator/.
    generated: dict[str, Any] | None = Field(None, exclude_if=_unset)

    @field_validator("generated")
    @classmethod
    def _generated(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is not None:
            from robotforge.design.generator.plan import Plan  # lazy: only worlds with generated robots need it

            Plan.from_dict(v)
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @field_validator("end_effector")
    @classmethod
    def _end_effector(cls, v: str | None) -> str | None:
        return end_effectors.check(v) if v is not None else None

    @field_validator("sensors")
    @classmethod
    def _sensor_names(cls, v: list[SensorEntry]) -> list[SensorEntry]:
        names = [s.name for s in v if s.name]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"sensor names must be unique on a robot: {', '.join(map(repr, dupes))} is used twice")
        return v

    @field_validator("template")
    @classmethod
    def _template(cls, v: str | None) -> str | None:
        return templates.canonical(v) if v is not None else None

    @field_validator("library")
    @classmethod
    def _library(cls, v: str | None) -> str | None:
        return library.canonical(v) if v is not None else None

    @field_validator("catalogue")
    @classmethod
    def _catalogue(cls, v: str | None) -> str | None:
        return catalogue.get(v).id if v is not None else None

    @model_validator(mode="after")
    def _one_source(self) -> RobotEntry:
        if [self.template, self.file, self.library, self.build, self.catalogue].count(None) != 4:
            raise ValueError(f"robot {self.name!r}: give exactly one of 'template', 'library', 'file' or 'build'")
        if self.library is not None:
            library.get(self.library)
        if self.build is not None:
            if self.params:
                raise ValueError(f"robot {self.name!r}: a built robot has no params; put everything in 'build'")
            try:
                builder.parse(self.build)
            except ValueError as exc:
                raise ValueError(f"robot {self.name!r}: {exc}") from None
        if len(self.pos) not in (2, 3):
            raise ValueError(f"robot {self.name!r}: pos is [x, y] or [x, y, z]")
        return self

    @property
    def mjcf_file(self) -> bool:
        """Whether it comes from an MJCF file of its own (URDF and mesh files go through the import doctor)."""
        if self.catalogue is not None:
            return catalogue.get(self.catalogue).format == "mjcf"
        return self.file is not None and not importer.is_importable(self.file)


def _shape_list() -> str:
    """The shapes an unknown name is told about: the plain ones and plugins' by name, the modelled objects (mug,
    bottle, ...) as an example and a count, so the message stays short as they grow in number."""
    from robotforge.objects.registry import BUILT_IN

    plain = [f"'{s}'" for s in props.SHAPES if s in props.BUILT_IN and s not in BUILT_IN]
    plugins = [f"'{s}'" for s in props.SHAPES if s not in props.BUILT_IN]
    modelled = [s for s in props.SHAPES if s in BUILT_IN]
    named = (
        [f'\'{modelled[0]}\', ... ({len(modelled)} modelled objects: shape="mug" or describe="a mug")']
        if modelled
        else []
    )
    return ", ".join([*plain, *named, *plugins])


RESISTING = ("drawer", "cabinet", "switch", "lever", "valve")  # what ObjectEntry.resist applies to


class ObjectEntry(_Strict):
    name: str
    shape: str = "box"  # a built-in shape (ShapeName) or one a plugin added (robotforge.extend)
    size: list[float] = Field(default_factory=lambda: [0.05, 0.05, 0.05])
    pos: list[float] = Field(default_factory=lambda: [0.3, 0.0])  # x, y (and optional z); z defaults to resting
    yaw: float = 0.0  # degrees, counter-clockwise seen from above
    color: str | list[float] = "red"
    mass: float | None = Field(None, gt=0, description="kg; default from size and the material's density")
    fixed: bool = False
    # What a loose box, ball or can is made of: its default mass, and steel is what an electromagnet holds.
    material: Literal["plastic", "wood", "steel"] | None = Field(None, exclude_if=_unset)
    # A finish changes only the drawn surface; density and magnetic affinity still use material.
    surface_material: Literal["plastic", "wood", "steel"] | None = Field(None, exclude_if=_unset)
    speed: float | None = Field(
        None, ge=-0.5, le=0.5, description="m/s a conveyor's belt runs at (conveyors only)", exclude_if=_unset
    )
    # A source's strength: degrees of rise at a heat source, ppm at a gas source (sensors/fields.py)
    strength: float | None = Field(None, gt=0, le=10000, exclude_if=_unset)
    # A generated object (mug, bottle, hammer, ...: robotforge.objects) varies with its seed, and the caller may fix
    # some of what varies (params, e.g. {"handle": "d"}); the same shape, size, seed and params build the same object.
    seed: int | None = Field(None, ge=0, le=2**31 - 1, exclude_if=_unset)
    params: dict[str, Any] | None = Field(None, exclude_if=_unset)
    # What a drawer, door, lever, switch or valve takes to move besides its own friction: N at its handle (rust)
    resist: float | None = Field(None, gt=0, le=500, exclude_if=_unset)

    @model_validator(mode="before")
    @classmethod
    def _shape_defaults(cls, data: Any) -> Any:
        """Fields left out take the shape's defaults: a wall is fixed and off-white, a box small and red."""
        if isinstance(data, dict):
            shape = data.get("shape", "box")
            if isinstance(shape, str) and shape not in props.SHAPES:
                data = {**data, "shape": wording.match(shape, props.SHAPES) or shape}  # "Table", "benches"
            info = props.SHAPES.get(data.get("shape", "box"))
            if info is not None:
                data = {"size": list(info.size), "color": _plain(info.color), "fixed": info.fixed, **data}
        return data

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @field_validator("shape")
    @classmethod
    def _shape(cls, v: str) -> str:
        if v in props.SHAPES:
            return v
        found = wording.match(v, props.SHAPES)
        if found is not None:
            return found
        from pydantic_core import PydanticCustomError

        from robotforge import extend

        if extend.REGISTRY.untrusted:  # the shape may be one a plugin file would add if it were trusted
            raise ValueError(extend.unknown_part(v, list(props.SHAPES)))
        # The message every unknown name gets, with the shapes there are (plugins' too, once loaded)
        expected = _shape_list()
        raise PydanticCustomError("literal_error", "Input should be {expected}", {"expected": expected})

    @model_validator(mode="after")
    def _sizes(self) -> ObjectEntry:
        info = props.SHAPES[self.shape]
        if len(self.size) not in info.size_lengths or any(s <= 0 for s in self.size):
            raise ValueError(f"object {self.name!r}: {self.shape} size is {info.size_hint} in meters")
        try:
            mechanisms.check_size(self.shape, self.size, info.size_hint)
        except ValueError as exc:
            raise ValueError(f"object {self.name!r}: {exc}") from exc
        if len(self.pos) not in (2, 3):
            raise ValueError(f"object {self.name!r}: pos is [x, y] or [x, y, z]")
        components.color(self.color)
        if self.speed is not None and self.shape != "conveyor":
            raise ValueError(f"object {self.name!r}: speed is a conveyor's belt speed; a {self.shape} has none")
        if self.strength is not None and self.shape not in props.SOURCES:
            raise ValueError(
                f"object {self.name!r}: strength is for a heat_source or gas_source; a {self.shape} has none"
            )
        if self.resist is not None and self.shape not in RESISTING:
            raise ValueError(
                f"object {self.name!r}: resist is how hard a {', '.join(RESISTING)} is to move; a {self.shape} has none"
            )
        if self.material is not None and self.shape not in ("box", "sphere", "cylinder"):
            raise ValueError(f"object {self.name!r}: material is for loose boxes, balls and cylinders")
        if self.surface_material is not None and self.shape not in ("box", "sphere", "cylinder"):
            raise ValueError(f"object {self.name!r}: surface_material is for boxes, balls and cylinders")
        if self.seed is not None or self.params:
            self._generated_fields()
        return self

    def _generated_fields(self) -> None:
        from robotforge.objects.registry import GENERATORS

        gen = GENERATORS.get(self.shape)
        if gen is None:
            raise ValueError(
                f"object {self.name!r}: seed and params are for generated objects ({', '.join(sorted(GENERATORS))}); "
                f"a {self.shape} has none"
            )
        try:
            self.params = gen.check_params(self.params) or None
        except ValueError as exc:
            raise ValueError(f"object {self.name!r}: {exc}") from exc


class World(_Strict):
    name: str = "my_world"
    floor: Floor = Field(default_factory=Floor)
    terrain: list[TerrainPatch] = Field(default_factory=list)
    water: WaterEntry | None = None  # a lake, a harbor, a pool: see design/water.py
    robots: list[RobotEntry] = Field(default_factory=list)
    objects: list[ObjectEntry] = Field(default_factory=list)
    # Goals with success checks (design/tasks): add_task; a world without tasks saves as it always did
    tasks: list[TaskEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> World:
        terrain.assign_names(self.terrain, {r.name for r in self.robots} | {o.name for o in self.objects})
        names = [r.name for r in self.robots] + [o.name for o in self.objects] + [t.name for t in self.terrain]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"{', '.join(map(repr, dupes))} already exists: names must be unique across robots, objects and "
                "terrain. Pick another, or change it with update_robot, update_object or update_terrain"
            )
        tasks = [t.name for t in self.tasks]
        twice = sorted({n for n in tasks if tasks.count(n) > 1})
        if twice:
            raise ValueError(f"task names must be unique: {', '.join(map(repr, twice))} is used twice")
        return self

    # ---- persistence ----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, text: str) -> World:
        for bare in BARE_HEX_COLOR.finditer(text):
            line_start = text.rfind("\n", 0, bare.start()) + 1
            if "#" in text[line_start : bare.start()]:
                continue  # inside a comment: YAML ignores it, and so do we
            line = text.count("\n", 0, bare.start()) + 1
            raise ValueError(
                f'world.yaml line {line}: put the hex color in quotes, color: "{bare[1]}"; '
                "without them YAML reads # as the start of a comment"
            )
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)  # the parser's own report quotes the lines around it
            at = f" at line {mark.line + 1}" if mark is not None else ""
            problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
            raise ValueError(f"world.yaml is not valid YAML{at}: {problem}") from exc
        if not isinstance(data, dict):
            raise ValueError("world.yaml must be a mapping")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # a hand edit; the agent reads this, not pydantic's report
            raise ValueError(f"world.yaml is not valid, fix it in the file. {_file_problems(exc, data)}") from exc

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_defaults=False)
        if data.get("water") is None:
            data.pop("water", None)  # a world without water saves as it always did
        if not data.get("tasks"):
            data.pop("tasks", None)  # nor does one without tasks
        for robot in data["robots"]:  # most robots carry nothing and have no added sensors: saved as they always were
            for extra in ("mounts", "sensors"):
                if not robot.get(extra):
                    robot.pop(extra, None)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


ENTRY_KINDS = {"robots": "robot", "objects": "object", "terrain": "terrain patch"}  # the lists of a world.yaml
MAX_ENTRIES = 3  # entries named in one answer; fix these and the rest follow


def _inside_entry(err: dict[str, Any], tag: object) -> dict[str, Any]:
    """An error with the entry's position in its list dropped. A terrain patch is a union picked by its type:
    pydantic words a wrong or missing type as a mismatch of tags, which is a type that is not one of the types."""
    ctx = err.get("ctx", {})
    if err["type"] in ("union_tag_invalid", "union_tag_not_found"):
        expected = ", ".join(f"'{name}'" for name in terrain.TYPES)
        return {
            **err,
            "type": "literal_error",
            "loc": ("type",),
            "input": ctx.get("tag"),
            "ctx": {"expected": expected},
        }
    return {**err, "loc": tuple(part for part in err["loc"][2:] if part != tag)}


def _file_problems(exc: ValidationError, data: dict[str, Any]) -> str:
    """What is wrong with a world.yaml, entry by entry: "object ball: 'colour' is not a known field; did you mean
    color?". pydantic names an entry by its position in the list, which means nothing to someone reading the answer."""
    fields = {"robots": list(RobotEntry.model_fields), "objects": list(ObjectEntry.model_fields)}
    by_entry: dict[tuple[str, int] | None, list[dict[str, Any]]] = {}
    for err in exc.errors():
        loc = err["loc"]
        entry = (loc[0], loc[1]) if len(loc) >= 2 and loc[0] in ENTRY_KINDS and isinstance(loc[1], int) else None
        by_entry.setdefault(entry, []).append(err)
    sentences = []
    for entry, errors in list(by_entry.items())[:MAX_ENTRIES]:
        if entry is None:
            sentences.append(wording.validation_message(errors))
            continue
        collection, index = entry
        try:
            given = data[collection][index]
        except (KeyError, IndexError, TypeError):
            given = None
        kind = ENTRY_KINDS[collection]
        label = (
            f"{kind} {given['name']}" if isinstance(given, dict) and "name" in given else f"{kind} number {index + 1}"
        )
        tag = given.get("type") if isinstance(given, dict) else None
        inner = [_inside_entry(e, tag) for e in errors]
        text = wording.validation_message(inner, valid=fields.get(collection), list_fields=False).strip()
        # A check of the whole entry already names it ("object 'ball': ..."); a field's problem does not.
        named = all(e["type"] == "value_error" and not e["loc"] for e in inner)
        sentences.append(text if named else f"{label}: {text}")
    more = len(by_entry) - MAX_ENTRIES
    return "; ".join(sentences) + (f"; and {more} more" if more > 0 else "")


# ---- compiler -------------------------------------------------------------------------


class CompiledWorld:
    def __init__(
        self,
        spec: mujoco.MjSpec,
        model: mujoco.MjModel,
        notes: list[str],
        doctor: dict[str, dict[str, Any]] | None = None,
        world_sha256: str | None = None,
    ):
        self.spec = spec
        self.model = model
        self.notes = notes
        self.world_sha256 = world_sha256
        self.doctor = doctor or {}  # robot name -> import doctor report, for imported files

    def to_xml(self) -> str:
        return self.spec.to_xml()


def compile_world(world: World, base_dir: Path | None = None) -> CompiledWorld:
    import hashlib

    world_sha256 = hashlib.sha256(world.model_dump_json().encode("utf-8")).hexdigest()
    spec = mujoco.MjSpec()
    spec.modelname = world.name
    spec.compiler.degree = False
    spec.option.timestep = 0.002  # robots built from parts are checked with the same (builder.assemble)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    _floor(spec, world.floor)
    # ---- terrain: patches on the floor, and the ground height everything else rests on
    notes: list[str] = terrain.add_terrain(spec, world.terrain, world.floor.style)
    ground = terrain.Ground.build(world.terrain, world.floor.style)

    doctor: dict[str, dict[str, Any]] = {}
    homes: list[tuple[str, dict[str, float], dict[str, float]]] = []
    frames: list[tuple[str, mujoco.MjsFrame]] = []  # robots that rest on the ground (no explicit z)
    for entry in world.robots:
        built = _robot_spec(entry, base_dir)
        if entry.sensors:  # any robot can carry them: placed on its own spec, before it is attached
            from robotforge.sensors.mount import mount

            mount(built.spec, entry.sensors, entry.name, built.home_qpos)
        z = entry.pos[2] if len(entry.pos) == 3 else ground.height(entry.pos[0], entry.pos[1]) + built.spawn_height
        frame = spec.worldbody.add_frame(pos=[entry.pos[0], entry.pos[1], z], quat=_yaw_quat(entry.yaw))
        _share_options(spec, built.spec)
        spec.attach(built.spec, prefix=f"{entry.name}/", frame=frame)
        homes.append((entry.name, built.home_qpos, built.home_ctrl))
        if len(entry.pos) == 2:
            frames.append((entry.name, frame))
        notes += [f"{entry.name}: {n}" for n in built.notes]
        if built.doctor:
            doctor[entry.name] = built.doctor

    for obj in world.objects:
        _object(spec, obj, ground)
        if obj.shape in mechanisms.MECHANISMS:  # a switch starts off, a conveyor running (the home keyframe)
            homes.append((obj.name, *mechanisms.home(obj.shape, obj.name, obj.speed)))
    if world.water is not None:
        water_physics.encode(spec, world.water.physics())
        notes.append(waters.describe(world.water))
        _moor_buoys(spec, world, ground)
        notes += _docks_under_water(world, ground)

    model = _compile(spec)
    if not ground.flat and frames and _settle_on_ground(model, homes, frames, ground):
        model = _compile(spec)
    if world.water is not None and _float_on_water(spec, model, world, homes, frames, ground):
        model = _compile(spec)
    if world.water is not None:
        notes += waters.stranded(model, world.water, [r.name for r in world.robots], ground.height)
    if world.water is not None:
        water_physics.add_surface(spec, world.water.physics(), model)
    _add_home_key(spec, model, homes)
    return CompiledWorld(spec, _compile(spec), notes, doctor, world_sha256)


def spawn_height(entry: RobotEntry, base_dir: Path | None = None) -> float:
    """How high a robot's base frame sits above the surface it stands on (to put arms on tables)."""
    return _robot_spec(entry, base_dir).spawn_height


def _settle_on_ground(model: mujoco.MjModel, homes, frames, ground: terrain.Ground) -> bool:
    """Raise or lower each robot so that, in its home pose, its lowest point rests on the terrain under it.

    The first placement only knew the ground height under the robot's center; on stairs,
    slopes or bumps a foot can be higher or lower than that. Returns True if anything moved.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = _home_qpos(model, homes)
    mujoco.mj_kinematics(model, data)
    moved = False
    for robot, frame in frames:
        lift = ground.lift_for(model, data, f"{robot}/")
        if abs(lift) > 5e-4:
            frame.pos = [frame.pos[0], frame.pos[1], frame.pos[2] + lift]
            moved = True
    return moved


def _moor_buoys(spec: mujoco.MjSpec, world: World, ground: terrain.Ground) -> None:
    """A buoy over water is tied to the bottom under it (see props.moor), with a little slack in its rope."""
    for obj in world.objects:
        if obj.shape != "buoy" or obj.fixed:
            continue
        x, y = obj.pos[0], obj.pos[1]
        bottom = ground.height(x, y)
        depth = world.water.level - bottom
        if depth > 0.2 and world.water.covers(x, y):
            props.moor(spec, obj.name, [x, y, bottom + 0.02], max(depth - 0.2 * obj.size[1], 0.15))


def _docks_under_water(world: World, ground: terrain.Ground) -> list[str]:
    """A dock's third size is its deck's height above the ground or bottom under it: an agent that read it as the
    deck's thickness put a deck under water, and only found out by looking. Say so at once, with a height that works."""
    notes = []
    for obj in world.objects:
        if obj.shape != "dock" or not world.water.covers(obj.pos[0], obj.pos[1]):
            continue
        base = (
            obj.pos[2]
            if len(obj.pos) == 3
            else ground.under(obj.pos[0], obj.pos[1], *props.footprint("dock", obj.size), obj.yaw)
        )
        deck = base + obj.size[2]
        if deck < world.water.level + 0.05:
            better = round(world.water.level - base + 0.35, 2)
            notes.append(
                f"{obj.name}: its deck is {world.water.level - deck:.2f} m under the water (deck height "
                f"{obj.size[2]:g} m above the bottom, water {world.water.level - base:.2f} m deep there); give it size "
                f"[{obj.size[0]:g}, {obj.size[1]:g}, {better:g}] for a deck 35 cm over the water"
            )
    return notes


def _float_on_water(spec: mujoco.MjSpec, model: mujoco.MjModel, world: World, homes, frames, ground) -> bool:
    """Robots and loose objects placed without a height over water start where they float (see design/water.py).
    Returns True if anything moved."""

    def lift(handle):
        def place(dz: float) -> None:
            handle.pos = [handle.pos[0], handle.pos[1], handle.pos[2] + dz]

        return place

    candidates = []
    for robot, frame in frames:
        root = waters.robot_root(model, robot)
        if root is not None:
            candidates.append((root, lift(frame)))
    free = {o.name for o in world.objects if len(o.pos) == 2 and not o.fixed and o.shape != "marker"}
    for body in spec.worldbody.bodies:  # not spec.body(): after an attach it can miss names
        if body.name in free:
            candidates.append((body.name, lift(body)))
    qpos = _home_qpos(model, homes)
    return bool(waters.float_up(spec, model, qpos, world.water, candidates, ground.height))


def _compile(spec: mujoco.MjSpec) -> mujoco.MjModel:
    try:
        return spec.compile()
    except ValueError as exc:
        raise ValueError(f"MuJoCo could not compile the world: {exc}") from exc


def _robot_spec(entry: RobotEntry, base_dir: Path | None) -> templates.BuiltRobot:
    """The robot as built from its source, holding with the end effector it was given, with the robots it carries
    mounted on it."""
    key = _placement_key(entry, base_dir) if entry.mounts else None
    later = entry.end_effector not in (None, end_effectors.FINGERS) and entry.mounts and not _has_tool_point(entry)
    if not later:
        return mounting.attach(_held_spec(entry, base_dir), entry.mounts, entry.name, entry.library, key)
    # Its arms are mounted ones (a PiPER on a rover): the cup or magnet goes on them once they are on it.
    built = mounting.attach(_source_spec(entry, base_dir), entry.mounts, entry.name, entry.library, key)
    try:
        built.notes += end_effectors.fit(built.spec, entry.end_effector, built.home_ctrl)
    except ValueError as exc:
        raise ValueError(f"robot {entry.name!r}: {exc}") from exc
    return built


def _has_tool_point(entry: RobotEntry) -> bool:
    """Whether the robot as its source builds it (without what it carries) has an arm's tool point."""
    if entry.template is not None and templates.canonical(entry.template) == "arm":
        return True
    if entry.library is not None:
        return bool(library.get(entry.library).tools)
    # these say so themselves (end_effectors.fit)
    return entry.file is not None or entry.build is not None or entry.catalogue is not None


def _placement_key(entry: RobotEntry, base_dir: Path | None) -> str:
    """What a robot and its mounts are built from, for keeping where its mounts go (mounting.PLACEMENTS): its entry
    but where it stands, the file it is read from as it is now, and where library robots come from."""
    made_of = entry.model_dump(mode="json", exclude={"name", "pos", "yaw"})
    stamp = None
    if entry.file is not None:
        path = Path(entry.file).expanduser()
        path = base_dir / path if not path.is_absolute() and base_dir is not None else path
        with contextlib.suppress(OSError):
            stat = path.stat()
            stamp = [str(path.resolve()), stat.st_mtime_ns, stat.st_size]
    menagerie_from = [os.environ.get(v, "") for v in ("ROBOTFORGE_MENAGERIE", "ROBOTFORGE_CACHE")]
    return json.dumps([made_of, stamp, menagerie_from, menagerie.COMMIT], sort_keys=True)


def _held_spec(entry: RobotEntry, base_dir: Path | None) -> templates.BuiltRobot:
    """The robot as its source builds it, holding with the end effector it was given: our arm template builds it
    in place of its fingers, any other arm gets it in front of them (see end_effectors.fit)."""
    tool = entry.end_effector
    if tool in end_effectors.KINDS and entry.template is not None and templates.canonical(entry.template) == "arm":
        native = entry.model_copy(update={"params": {**entry.params, "end_effector": tool}, "end_effector": None})
        return _source_spec(native, base_dir)
    built = _source_spec(entry, base_dir)
    if tool in (None, end_effectors.FINGERS):
        return built
    try:
        built.notes += end_effectors.fit(built.spec, tool, built.home_ctrl)
    except ValueError as exc:
        raise ValueError(f"robot {entry.name!r}: {exc}") from exc
    return built


def _source_spec(entry: RobotEntry, base_dir: Path | None) -> templates.BuiltRobot:
    if entry.template is not None:
        try:
            return templates.build(entry.template, entry.params)
        except ValueError as exc:  # includes pydantic ValidationError
            raise ValueError(f"robot {entry.name!r}: {exc}") from exc
    if entry.library is not None:
        try:
            return library.build(entry.library, entry.params)
        except ValueError as exc:
            raise ValueError(f"robot {entry.name!r}: {exc}") from exc
    if entry.build is not None:
        try:
            return builder.build_robot(entry.build)
        except ValueError as exc:
            raise ValueError(f"robot {entry.name!r}: {exc}") from exc
    if entry.catalogue is not None:
        return _catalogue_robot(entry)
    path = Path(entry.file).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        where = "" if Path(entry.file).expanduser() == path else f", looked in {path}"  # relative to the world folder
        raise ValueError(
            f"robot {entry.name!r}: file not found: {entry.file}{where} (give the path of {importer.formats_hint()})"
        )
    return _file_robot(entry.name, path, entry.params)


def _catalogue_robot(entry: RobotEntry) -> templates.BuiltRobot:
    """An open catalogue robot: its folder is fetched on first use, then its file is read like any other and
    catalogue_inspect.py makes it ready (an arm's tool point, its description)."""
    try:
        robot = catalogue.get(entry.catalogue)
        path = catalogue.model_path(robot.id)
        params = {"xacro_args": dict(robot.xacro_args), **entry.params} if robot.format == "xacro" else entry.params
        if robot.format == "xacro" and not params["xacro_args"]:
            del params["xacro_args"]
        problem = None
        try:
            built = _file_robot(entry.name, path, params, drive=True)
            wide = _lost_meshes(built)
        except ValueError as exc:  # a xacro that includes another package of the repository
            if "was not found" not in str(exc):
                raise
            wide, built, problem = True, None, exc
        if wide and catalogue.widen(robot.id):  # files next to the robot's folder: take the whole repository
            built = _file_robot(entry.name, path, params, drive=True)
        elif built is None:
            raise ValueError(f"{problem}; it needs a package from elsewhere in {robot.repo}") from problem
        return built  # (_file_robot has put the arm's tool point on it)
    except ValueError as exc:
        raise ValueError(f"robot {entry.name!r}: {exc}") from exc


def _lost_meshes(built: templates.BuiltRobot) -> bool:
    """Whether the import doctor could not find some of the meshes a URDF names."""
    findings = (built.doctor or {}).get("findings", [])
    return any(f["topic"] == "meshes" and f["status"] == "unfixed" and "not be found" in f["problem"] for f in findings)


log = logging.getLogger(__name__)


def _file_robot(name: str, path: Path, params: dict[str, Any], drive: bool = False) -> templates.BuiltRobot:
    """A robot from a file: MJCF as it is (with `drive`, its torque motors become position servos), URDF, xacro and
    meshes through the import doctor."""
    from robotforge.design import catalogue_inspect  # lazy: it needs the behaviors

    return catalogue_inspect.finish(_balanced(_read_file_robot(name, path, params, drive)), path.name)


def _balanced(built: templates.BuiltRobot) -> templates.BuiltRobot:
    """A two-legged robot whose file draws it with its weight on the edge of its feet starts in a balanced pose."""
    if not built.floating:
        return built
    try:
        moved = mjcf_stance.balance(built.spec, built.home_qpos, built.spawn_height)
    except (ValueError, RuntimeError) as exc:  # a courtesy: a file the check cannot read keeps its own pose
        log.debug("balance: %s", exc)
        return built
    if moved is not None:
        joints, servos, spawn = moved
        built.home_qpos.update(joints)
        built.home_ctrl.update(servos)
        built.spawn_height = spawn
        built.notes.append(
            f"its keyframe left the weight on the edge of the feet; moved {len(joints)} joints to balance"
        )
    return built


def _read_file_robot(name: str, path: Path, params: dict[str, Any], drive: bool = False) -> templates.BuiltRobot:
    """The robot of a file as it draws it (see _file_robot)."""
    if importer.is_importable(path):  # URDF, xacro and plain meshes go through the import doctor
        try:
            result = importer.import_file(path, params)
        except ValueError as exc:
            raise ValueError(f"robot {name!r}: {exc}") from exc
        return templates.BuiltRobot(
            spec=result.spec,
            home_qpos=result.home_qpos,
            home_ctrl=result.home_ctrl,
            spawn_height=result.spawn_height,
            floating=result.floating,
            notes=[f"imported {path.name}. {result.report.summary}"],
            doctor=result.report.to_dict(),
        )
    if path.is_dir():
        raise ValueError(f"robot {name!r}: {path.name} is a folder; give a file: {importer.formats_hint()}")
    if path.suffix.lower() not in (".xml", ".mjcf"):
        raise ValueError(f"robot {name!r}: {path.name}: expected {importer.formats_hint()}")
    return _mjcf_robot(name, path, drive)


def _mjcf_robot(name: str, path: Path, drive: bool = False) -> templates.BuiltRobot:
    """An MJCF file, ready to stand in a world as a library robot does: parts it left unnamed get names, and its
    home keyframe becomes its home pose (the world brings its own keyframe, so the file's is dropped)."""
    try:
        child, repaired = mjcf_doctor.read(path)
    except ValueError as exc:
        raise ValueError(f"robot {name!r}: MuJoCo could not read {path.name}: {exc}") from exc
    notes = [f"imported from {path.name}"]
    if repaired:
        notes.append(f"repaired the file to read it ({'; '.join(repaired)})")
    if drive:
        servos = mjcf_doctor.drive_motors(child)
        if servos:
            notes.append(f"{servos} torque motors driven as position servos (torque limits kept), so it holds its pose")
    motors = _name_unnamed(child)
    if motors:
        notes.append(f"named the {motors} motors the file left unnamed after their joints, so they can be driven")
    key = _home_key(child)
    key_qpos, key_ctrl = library.read_key(child, key)
    _strip_world_extras(child)
    if drive and key_qpos is None:  # no keyframe: a dog with its joints at zero lies down; stand it
        key_qpos = mjcf_stance.find(child)
        if key_qpos is not None:
            notes.append("the file has no standing pose; found one with the feet under the hips")
    if child.worldbody.first_body() is None:
        return templates.BuiltRobot(spec=child, notes=notes)
    home_qpos, home_ctrl, spawn, floating = library.stand_at_home(child, key_qpos, key_ctrl)
    if key is not None:
        notes.append(f"starts in its {key!r} keyframe")
    return templates.BuiltRobot(
        spec=child,
        home_qpos=home_qpos,
        home_ctrl=home_ctrl,
        # A floating robot stands on its lowest point; one fixed to the world stays where its file puts it.
        spawn_height=spawn if floating else 0.0,
        floating=floating,
        notes=notes,
    )


def _home_key(spec: mujoco.MjSpec) -> str | None:
    """The keyframe a file starts in: the one named home (Menagerie's convention), else its first (named home
    here if the file left it unnamed, so it can be read)."""
    keys = list(spec.keys)
    if not keys:
        return None
    if any(k.name == "home" for k in keys):
        return "home"
    if not keys[0].name:
        keys[0].name = "home"
    return keys[0].name


def _name_unnamed(spec: mujoco.MjSpec) -> int:
    """Name the bodies, joints and actuators a file left unnamed; returns how many actuators. A world finds each
    robot's parts by their names ("<robot>/..."), and attaching only prefixes names a file gave: a hand-written MJCF's
    unnamed motors belonged to no robot, so nothing could drive them. A motor is named after the joint it moves. Free
    joints stay as they are: nothing drives them, and a world finds a robot's base by its body."""

    def unique(base: str, taken: set[str]) -> str:
        name, k = base, 1
        while name in taken:
            k += 1
            name = f"{base}_{k}"
        taken.add(name)
        return name

    motors = 0
    for elements, kind in ((spec.bodies, "body"), (spec.joints, "joint"), (spec.actuators, "actuator")):
        taken = {e.name for e in elements if e.name}
        for i, element in enumerate(elements):
            if element.name or (kind == "joint" and element.type == mujoco.mjtJoint.mjJNT_FREE):
                continue
            on_joint = kind == "actuator" and element.trntype == mujoco.mjtTrn.mjTRN_JOINT and element.target
            element.name = unique(element.target if on_joint else f"{kind}{i}", taken)
            motors += kind == "actuator"
    return motors


def _share_options(world: mujoco.MjSpec, child: mujoco.MjSpec) -> None:
    """Adopt a robot's contact settings for the world, then align the robot's options to the world's.

    Physics options are global, so attach keeps the world's values; models like Menagerie's
    quadrupeds rely on elliptic friction cones with a high impratio to keep their feet from
    slipping. Note this is world-wide: adding a Go2 also makes every other robot and object in
    the world use elliptic cones (grip gets slightly stronger; all built-in behaviors still pass).
    Aligning the child afterwards also silences MuJoCo's attach-conflict warnings.
    """
    if child.option.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
        world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        world.option.impratio = max(world.option.impratio, child.option.impratio)
    for name in ("timestep", "integrator", "cone", "impratio", "iterations", "ls_iterations", "noslip_iterations"):
        setattr(child.option, name, getattr(world.option, name))
    # Menagerie's aircraft bring air (density, viscosity); the world has none: behaviors/air.py models drag.
    child.option.density, child.option.viscosity = world.option.density, world.option.viscosity


def _strip_world_extras(child: mujoco.MjSpec) -> None:
    """Imported files often bring their own floor and lights; the world provides those."""
    for geom in list(child.worldbody.geoms):
        child.delete(geom)
    for light in list(child.worldbody.lights):
        child.delete(light)
    for key in list(child.keys):
        child.delete(key)


def _floor(spec: mujoco.MjSpec, floor: Floor) -> None:
    terrain.add_floor(spec, floor.style, floor.friction)
    # A directional sun lights a 20 m park as evenly as a desk; a spot light would leave a lit circle.
    spec.worldbody.add_light(
        name="sun",
        pos=[0.5, -0.5, 3.0],
        dir=[-0.3, 0.4, -1.0],
        diffuse=[0.8, 0.8, 0.8],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        castshadow=True,
    )
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.quality.shadowsize = 4096
    # Same dark gradient as the browser viewer, so snapshots and the live view look alike.
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.17, 0.2, 0.24],
        rgb2=[0.1, 0.12, 0.15],
        width=512,
        height=3072,
    )


def _yaw_quat(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _plain(color: str | tuple[float, ...]) -> str | list[float]:
    return color if isinstance(color, str) else list(color)


PLASTIC_DENSITY = 600.0  # kg/m^3, light printed or molded parts
DENSITY = {"plastic": PLASTIC_DENSITY, "wood": 550.0, "steel": 7850.0}  # kg/m^3, by ObjectEntry.material


def _object(spec: mujoco.MjSpec, obj: ObjectEntry, ground: terrain.Ground) -> None:
    rgba = components.color(obj.color)
    G = mujoco.mjtGeom
    x, y = obj.pos[0], obj.pos[1]
    # Without an explicit z, objects rest on the ground (floor or terrain) under their footprint.
    ground_z = ground.under(x, y, *props.footprint(obj.shape, obj.size), obj.yaw)
    quat = _yaw_quat(obj.yaw)
    if obj.shape in mechanisms.SHAPES:
        # A rope with no height given hangs from a hook its length and 20 cm above the ground.
        hang = obj.size[0] + 0.2 if obj.shape == "rope" and obj.fixed else 0.0
        base = [x, y, obj.pos[2] if len(obj.pos) == 3 else ground_z + hang]
        mechanisms.build(spec, obj.name, obj.shape, obj.size, rgba, base, quat, obj.fixed, obj.resist)
        return
    if obj.shape in props.PROPS:
        base = [x, y, obj.pos[2] if len(obj.pos) == 3 else ground_z]
        props.build(spec, obj.name, obj.shape, obj.size, rgba, base, quat, obj.fixed, obj.mass, obj.seed, obj.params)
        if obj.shape in props.SOURCES:
            kind, default = props.SOURCES[obj.shape]
            at = props.source_lift(obj.shape, obj.size[0])
            source_fields.declare_source(spec, obj.name, kind, obj.strength or default, at)
        return
    if obj.shape == "table":
        _table(spec, obj, rgba, ground_z, quat)
        return
    if obj.shape == "bin":
        _bin(spec, obj, rgba, ground_z, quat)
        return
    if obj.shape == "box":
        half = [s / 2 for s in obj.size]
        geom = {"type": G.mjGEOM_BOX, "size": half}
        rest_z, volume = half[2], float(np.prod(obj.size))
    elif obj.shape == "sphere":
        r = obj.size[0] / 2
        geom = {"type": G.mjGEOM_SPHERE, "size": [r, 0, 0]}
        rest_z, volume = r, 4 / 3 * math.pi * r**3
    else:  # cylinder
        r, h = obj.size[0] / 2, obj.size[1] / 2
        geom = {"type": G.mjGEOM_CYLINDER, "size": [r, h, 0]}
        rest_z, volume = h, math.pi * r * r * obj.size[1]
    z = obj.pos[2] if len(obj.pos) == 3 else ground_z + rest_z
    mass = obj.mass if obj.mass is not None else volume * DENSITY[obj.material or "plastic"]
    if obj.material == "steel":  # an electromagnet's core touches steel only (end_effectors.MAGNET_BIT)
        geom["conaffinity"] = end_effectors.steel_affinity()
    if obj.fixed:
        g = spec.worldbody.add_geom(name=obj.name, pos=[x, y, z], quat=quat, rgba=rgba, **geom)
        looks.dress(
            spec,
            spec.worldbody,
            obj.shape,
            obj.size,
            rgba,
            obj.name,
            [g],
            pos=[x, y, z],
            quat=quat,
            material=obj.surface_material or obj.material,
        )
        return
    body = spec.worldbody.add_body(name=obj.name, pos=[x, y, z], quat=quat)
    body.add_freejoint(name=obj.name)
    # A ball needs rolling friction (condim 6) or it rolls on for ever once let go, off a plate, a counter and the
    # island: the kitchen's apple put on a plate ended on the floor.
    rolls = obj.shape == "sphere"
    friction = [1.0, 0.02, BALL_ROLLING if rolls else 0.002]
    g = body.add_geom(name=obj.name, rgba=rgba, mass=mass, friction=friction, condim=6 if rolls else 4, **geom)
    looks.dress(spec, body, obj.shape, obj.size, rgba, obj.name, [g], material=obj.surface_material or obj.material)


BALL_ROLLING = 0.005  # m: a ball's rolling friction coefficient (about a hard ball on a wooden top)


def _table(spec: mujoco.MjSpec, obj: ObjectEntry, rgba: list[float], ground_z: float, quat: list[float]) -> None:
    length, width, height = obj.size
    top = 0.025
    x, y = obj.pos[0], obj.pos[1]
    body = spec.worldbody.add_body(name=obj.name, pos=[x, y, obj.pos[2] if len(obj.pos) == 3 else ground_z], quat=quat)
    top_geom = body.add_geom(
        name=obj.name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length / 2, width / 2, top / 2],
        pos=[0, 0, height - top / 2],
        rgba=rgba,
    )
    leg = min(0.025, length / 10, width / 10)
    geoms = [top_geom]
    for sx in (1, -1):
        for sy in (1, -1):
            geoms.append(
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[leg, leg, (height - top) / 2],
                    pos=[sx * (length / 2 - 2 * leg), sy * (width / 2 - 2 * leg), (height - top) / 2],
                    rgba=[c * 0.7 for c in rgba[:3]] + [1],
                )
            )
    looks.dress(spec, body, "table", obj.size, rgba, obj.name, geoms)
    # a cloth hangs over its edges, not in under its top
    mechanisms.cloth_skirt(body, length, width, height - top, height)


def _bin(spec: mujoco.MjSpec, obj: ObjectEntry, rgba: list[float], ground_z: float, quat: list[float]) -> None:
    """Open-top container; walls are 5 mm thick."""
    length, width, height = obj.size
    t = 0.005
    x, y = obj.pos[0], obj.pos[1]
    z = obj.pos[2] if len(obj.pos) == 3 else ground_z
    parent = spec.worldbody
    if not obj.fixed:
        parent = spec.worldbody.add_body(name=obj.name, pos=[x, y, z], quat=quat)
        parent.add_freejoint(name=obj.name)
        x = y = z = 0.0
    elif obj.yaw:
        parent = spec.worldbody.add_body(name=obj.name, pos=[x, y, z], quat=quat)  # static, turned
        x = y = z = 0.0
    G = mujoco.mjtGeom.mjGEOM_BOX
    parts = [
        ([length / 2, width / 2, t / 2], [0, 0, t / 2]),
        ([t / 2, width / 2, height / 2], [length / 2 - t / 2, 0, height / 2]),
        ([t / 2, width / 2, height / 2], [-length / 2 + t / 2, 0, height / 2]),
        ([length / 2, t / 2, height / 2], [0, width / 2 - t / 2, height / 2]),
        ([length / 2, t / 2, height / 2], [0, -width / 2 + t / 2, height / 2]),
    ]
    mass = obj.mass if obj.mass is not None else 0.15
    geoms = []
    for i, (size, pos) in enumerate(parts):
        kwargs = {} if obj.fixed else {"mass": mass / len(parts)}
        g = parent.add_geom(
            name=obj.name if i == 0 else f"{obj.name}_wall{i}",
            type=G,
            size=size,
            pos=[x + pos[0], y + pos[1], z + pos[2]],
            rgba=rgba,
            **kwargs,
        )
        geoms.append(g)
    looks.dress(spec, parent, "bin", obj.size, rgba, obj.name, geoms, pos=[x, y, z])


def _home_qpos(model: mujoco.MjModel, homes) -> np.ndarray:
    """qpos with every robot's joints at their home values (free joints at their attachment frames)."""
    qpos = model.qpos0.copy()
    for robot, joint_values, _ in homes:
        for joint, value in joint_values.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}/{joint}")
            if j >= 0:
                qpos[model.jnt_qposadr[j]] = value
    return qpos


def _add_home_key(spec: mujoco.MjSpec, model: mujoco.MjModel, homes) -> None:
    """Bake every robot's home pose into a 'home' keyframe so reset returns there."""
    qpos = _home_qpos(model, homes)
    ctrl = np.zeros(model.nu)
    for robot, _, ctrl_values in homes:
        for act, value in ctrl_values.items():
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/{act}")
            if a >= 0:
                ctrl[a] = value
    if not any(_eased_gripper(model, a) for a in range(model.nu)):
        spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
        return
    # A hand's gripper eases toward its target (hands.py): it starts at rest there, open, not shut and opening.
    act = np.zeros(model.na)
    for a in range(model.nu):
        if _eased_gripper(model, a):
            act[model.actuator_actadr[a]] = ctrl[a]
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist(), act=act.tolist())


def _eased_gripper(model: mujoco.MjModel, a: int) -> bool:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
    eased = int(model.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_FILTEREXACT)
    return eased and name.endswith("gripper") and int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON)
