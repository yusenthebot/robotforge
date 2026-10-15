"""world.yaml: the single source of truth for a playground world, and its compiler to MuJoCo.

A world has a floor, terrain (hills, stairs, ramps, ... see terrain.py), robots (from templates,
the robot library, imported files or built from parts) and loose or fixed objects (including
furniture and architecture, see props.py). Every robot's names are prefixed with "<robot>/" so
several robots can share a world without clashing. Robots and objects without an explicit height
rest on the ground under them.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from robotforge import wording
from robotforge.design import builder, components, importer, library, props, templates, terrain
from robotforge.design.terrain import TerrainPatch
from robotforge.design.visuals import props as looks

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
# A hand-written `color: #9c6b3f` parses as no color at all (an unquoted # starts a YAML comment).
BARE_HEX_COLOR = re.compile(r"(?:^[ \t-]*|[{,][ \t]*)color:[ \t]*(#[0-9a-fA-F]+)(?![\w-])", re.MULTILINE)
ShapeName = Literal[tuple(props.SHAPES)]  # box, sphere, cylinder, table, bin, wall, shelf, ...


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class RobotEntry(_Strict):
    name: str
    template: str | None = None
    file: str | None = None  # MJCF or URDF path, relative to the world directory
    library: str | None = None  # a robot from the library, e.g. "go2" (see library.CATALOG)
    build: dict[str, Any] | None = None  # built from parts: {"base": {...}, "parts": [...]} (see builder)
    params: dict[str, Any] = Field(default_factory=dict)
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0])  # x, y (and optional z) in meters
    yaw: float = 0.0  # degrees, counter-clockwise seen from above

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @field_validator("template")
    @classmethod
    def _template(cls, v: str | None) -> str | None:
        return templates.canonical(v) if v is not None else None

    @field_validator("library")
    @classmethod
    def _library(cls, v: str | None) -> str | None:
        return library.canonical(v) if v is not None else None

    @model_validator(mode="after")
    def _one_source(self) -> RobotEntry:
        if [self.template, self.file, self.library, self.build].count(None) != 3:
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


class ObjectEntry(_Strict):
    name: str
    shape: ShapeName = "box"
    size: list[float] = Field(default_factory=lambda: [0.05, 0.05, 0.05])
    pos: list[float] = Field(default_factory=lambda: [0.3, 0.0])  # x, y (and optional z); z defaults to resting
    yaw: float = 0.0  # degrees, counter-clockwise seen from above
    color: str | list[float] = "red"
    mass: float | None = Field(None, gt=0, description="kg; default from size and a plastic density")
    fixed: bool = False

    @model_validator(mode="before")
    @classmethod
    def _shape_defaults(cls, data: Any) -> Any:
        """Fields left out take the shape's defaults: a wall is fixed and off-white, a box small and red."""
        if isinstance(data, dict):
            info = props.SHAPES.get(data.get("shape", "box"))
            if info is not None:
                data = {"size": list(info.size), "color": _plain(info.color), "fixed": info.fixed, **data}
        return data

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @model_validator(mode="after")
    def _sizes(self) -> ObjectEntry:
        info = props.SHAPES[self.shape]
        if len(self.size) not in info.size_lengths or any(s <= 0 for s in self.size):
            raise ValueError(f"object {self.name!r}: {self.shape} size is {info.size_hint} in meters")
        if len(self.pos) not in (2, 3):
            raise ValueError(f"object {self.name!r}: pos is [x, y] or [x, y, z]")
        components.color(self.color)
        return self


class World(_Strict):
    name: str = "my_world"
    floor: Floor = Field(default_factory=Floor)
    terrain: list[TerrainPatch] = Field(default_factory=list)
    robots: list[RobotEntry] = Field(default_factory=list)
    objects: list[ObjectEntry] = Field(default_factory=list)

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
    ):
        self.spec = spec
        self.model = model
        self.notes = notes
        self.doctor = doctor or {}  # robot name -> import doctor report, for imported files

    def to_xml(self) -> str:
        return self.spec.to_xml()


def compile_world(world: World, base_dir: Path | None = None) -> CompiledWorld:
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

    model = _compile(spec)
    if not ground.flat and frames and _settle_on_ground(model, homes, frames, ground):
        model = _compile(spec)
    _add_home_key(spec, model, homes)
    return CompiledWorld(spec, _compile(spec), notes, doctor)


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


def _compile(spec: mujoco.MjSpec) -> mujoco.MjModel:
    try:
        return spec.compile()
    except ValueError as exc:
        raise ValueError(f"MuJoCo could not compile the world: {exc}") from exc


def _robot_spec(entry: RobotEntry, base_dir: Path | None) -> templates.BuiltRobot:
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
    path = Path(entry.file).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        where = "" if Path(entry.file).expanduser() == path else f", looked in {path}"  # relative to the world folder
        raise ValueError(
            f"robot {entry.name!r}: file not found: {entry.file}{where} (give the path of {importer.formats_hint()})"
        )
    if importer.is_importable(path):  # URDF, xacro and plain meshes go through the import doctor
        try:
            result = importer.import_file(path, entry.params)
        except ValueError as exc:
            raise ValueError(f"robot {entry.name!r}: {exc}") from exc
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
        raise ValueError(f"robot {entry.name!r}: {path.name} is a folder; give a file: {importer.formats_hint()}")
    if path.suffix.lower() not in (".xml", ".mjcf"):
        raise ValueError(f"robot {entry.name!r}: {path.name}: expected {importer.formats_hint()}")
    return _mjcf_robot(entry.name, path)


def _mjcf_robot(name: str, path: Path) -> templates.BuiltRobot:
    """An MJCF file, ready to stand in a world as a library robot does: parts it left unnamed get names, and its
    home keyframe becomes its home pose (the world brings its own keyframe, so the file's is dropped)."""
    try:
        child = mujoco.MjSpec.from_file(str(path))
    except ValueError as exc:
        raise ValueError(f"robot {name!r}: MuJoCo could not read {path.name}: {exc}") from exc
    notes = [f"imported from {path.name}"]
    motors = _name_unnamed(child)
    if motors:
        notes.append(f"named the {motors} motors the file left unnamed after their joints, so they can be driven")
    key = _home_key(child)
    key_qpos, key_ctrl = library.read_key(child, key)
    _strip_world_extras(child)
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


def _object(spec: mujoco.MjSpec, obj: ObjectEntry, ground: terrain.Ground) -> None:
    rgba = components.color(obj.color)
    G = mujoco.mjtGeom
    x, y = obj.pos[0], obj.pos[1]
    # Without an explicit z, objects rest on the ground (floor or terrain) under their footprint.
    ground_z = ground.under(x, y, *props.footprint(obj.shape, obj.size), obj.yaw)
    quat = _yaw_quat(obj.yaw)
    if obj.shape in props.PROPS:
        base = [x, y, obj.pos[2] if len(obj.pos) == 3 else ground_z]
        props.build(spec, obj.name, obj.shape, obj.size, rgba, base, quat, obj.fixed, obj.mass)
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
    mass = obj.mass if obj.mass is not None else volume * PLASTIC_DENSITY
    if obj.fixed:
        g = spec.worldbody.add_geom(name=obj.name, pos=[x, y, z], quat=quat, rgba=rgba, **geom)
        looks.dress(spec, spec.worldbody, obj.shape, obj.size, rgba, obj.name, [g], pos=[x, y, z], quat=quat)
        return
    body = spec.worldbody.add_body(name=obj.name, pos=[x, y, z], quat=quat)
    body.add_freejoint(name=obj.name)
    g = body.add_geom(name=obj.name, rgba=rgba, mass=mass, friction=[1.0, 0.02, 0.002], condim=4, **geom)
    looks.dress(spec, body, obj.shape, obj.size, rgba, obj.name, [g])


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
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
