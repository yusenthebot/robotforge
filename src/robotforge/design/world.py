"""world.yaml: the single source of truth for a playground world, and its compiler to MuJoCo.

A world has a floor, robots (from templates, the robot library or imported files)
and loose or fixed objects. Every robot's names are prefixed with "<robot>/" so
several robots can share a world without clashing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robotforge.design import components, library, templates

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
FLOOR_STYLES = {
    "checker": [0.3, 0.33, 0.37, 1.0],
    "grass": [0.33, 0.52, 0.27, 1.0],
    "wood": [0.55, 0.42, 0.3, 1.0],
    "concrete": [0.5, 0.5, 0.5, 1.0],
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError("names use lowercase letters, digits and _, start with a letter, max 32 chars")
    return value


class Floor(_Strict):
    style: Literal["checker", "grass", "wood", "concrete"] = "checker"
    friction: float = Field(1.0, gt=0, le=3)


class RobotEntry(_Strict):
    name: str
    template: str | None = None
    file: str | None = None  # MJCF or URDF path, relative to the world directory
    library: str | None = None  # a robot from the library, e.g. "go2" (see library.CATALOG)
    params: dict[str, Any] = Field(default_factory=dict)
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0])  # x, y (and optional z) in meters
    yaw: float = 0.0  # degrees, counter-clockwise seen from above

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @model_validator(mode="after")
    def _one_source(self) -> RobotEntry:
        if [self.template, self.file, self.library].count(None) != 2:
            raise ValueError(f"robot {self.name!r}: give exactly one of 'template', 'library' or 'file'")
        if self.library is not None:
            library.get(self.library)
        if len(self.pos) not in (2, 3):
            raise ValueError(f"robot {self.name!r}: pos is [x, y] or [x, y, z]")
        return self


class ObjectEntry(_Strict):
    name: str
    shape: Literal["box", "sphere", "cylinder", "table", "bin"] = "box"
    size: list[float] = Field(default_factory=lambda: [0.05, 0.05, 0.05])
    pos: list[float] = Field(default_factory=lambda: [0.3, 0.0])  # x, y (and optional z); z defaults to resting
    color: str | list[float] = "red"
    mass: float | None = Field(None, gt=0, description="kg; default from size and a plastic density")
    fixed: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _check_name(v)

    @model_validator(mode="after")
    def _sizes(self) -> ObjectEntry:
        need = {"box": 3, "sphere": 1, "cylinder": 2, "table": 3, "bin": 3}[self.shape]
        if len(self.size) != need or any(s <= 0 for s in self.size):
            hint = {
                "box": "[length, width, height]",
                "sphere": "[diameter]",
                "cylinder": "[diameter, height]",
                "table": "[length, width, height]",
                "bin": "[length, width, height]",
            }[self.shape]
            raise ValueError(f"object {self.name!r}: {self.shape} size is {hint} in meters")
        if len(self.pos) not in (2, 3):
            raise ValueError(f"object {self.name!r}: pos is [x, y] or [x, y, z]")
        components.color(self.color)
        return self


class World(_Strict):
    name: str = "my_world"
    floor: Floor = Field(default_factory=Floor)
    robots: list[RobotEntry] = Field(default_factory=list)
    objects: list[ObjectEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> World:
        names = [r.name for r in self.robots] + [o.name for o in self.objects]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"names must be unique across robots and objects: {', '.join(dupes)}")
        return self

    # ---- persistence ----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, text: str) -> World:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("world.yaml must be a mapping")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_defaults=False)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


# ---- compiler -------------------------------------------------------------------------


class CompiledWorld:
    def __init__(self, spec: mujoco.MjSpec, model: mujoco.MjModel, notes: list[str]):
        self.spec = spec
        self.model = model
        self.notes = notes

    def to_xml(self) -> str:
        return self.spec.to_xml()


def compile_world(world: World, base_dir: Path | None = None) -> CompiledWorld:
    spec = mujoco.MjSpec()
    spec.modelname = world.name
    spec.compiler.degree = False
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    _floor(spec, world.floor)

    notes: list[str] = []
    homes: list[tuple[str, dict[str, float], dict[str, float]]] = []
    for entry in world.robots:
        built = _robot_spec(entry, base_dir)
        z = entry.pos[2] if len(entry.pos) == 3 else built.spawn_height
        frame = spec.worldbody.add_frame(pos=[entry.pos[0], entry.pos[1], z], quat=_yaw_quat(entry.yaw))
        _share_options(spec, built.spec)
        spec.attach(built.spec, prefix=f"{entry.name}/", frame=frame)
        homes.append((entry.name, built.home_qpos, built.home_ctrl))
        notes += [f"{entry.name}: {n}" for n in built.notes]

    for obj in world.objects:
        _object(spec, obj)

    model = _compile(spec)
    _add_home_key(spec, model, homes)
    return CompiledWorld(spec, _compile(spec), notes)


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
    path = Path(entry.file).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        raise ValueError(f"robot {entry.name!r}: file not found: {path}")
    try:
        child = mujoco.MjSpec.from_file(str(path))
    except ValueError as exc:
        raise ValueError(f"robot {entry.name!r}: MuJoCo could not read {path.name}: {exc}") from exc
    _strip_world_extras(child)
    return templates.BuiltRobot(spec=child, notes=[f"imported from {path.name}"])


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
    rgba = FLOOR_STYLES[floor.style]
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=rgba,
        friction=[floor.friction, 0.01, 0.001],
    )
    spec.worldbody.add_light(
        name="sun",
        pos=[0.5, -0.5, 3.0],
        dir=[-0.15, 0.15, -1.0],
        diffuse=[0.75, 0.75, 0.75],
        castshadow=True,
    )


def _yaw_quat(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


PLASTIC_DENSITY = 600.0  # kg/m^3, light printed or molded parts


def _object(spec: mujoco.MjSpec, obj: ObjectEntry) -> None:
    rgba = components.color(obj.color)
    G = mujoco.mjtGeom
    x, y = obj.pos[0], obj.pos[1]
    if obj.shape == "table":
        _table(spec, obj, rgba)
        return
    if obj.shape == "bin":
        _bin(spec, obj, rgba)
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
    z = obj.pos[2] if len(obj.pos) == 3 else rest_z
    mass = obj.mass if obj.mass is not None else volume * PLASTIC_DENSITY
    if obj.fixed:
        spec.worldbody.add_geom(name=obj.name, pos=[x, y, z], rgba=rgba, **geom)
        return
    body = spec.worldbody.add_body(name=obj.name, pos=[x, y, z])
    body.add_freejoint(name=obj.name)
    body.add_geom(name=obj.name, rgba=rgba, mass=mass, friction=[1.0, 0.02, 0.002], condim=4, **geom)


def _table(spec: mujoco.MjSpec, obj: ObjectEntry, rgba: list[float]) -> None:
    length, width, height = obj.size
    top = 0.025
    x, y = obj.pos[0], obj.pos[1]
    body = spec.worldbody.add_body(name=obj.name, pos=[x, y, 0])
    body.add_geom(
        name=obj.name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[length / 2, width / 2, top / 2],
        pos=[0, 0, height - top / 2],
        rgba=rgba,
    )
    leg = min(0.025, length / 10, width / 10)
    for sx in (1, -1):
        for sy in (1, -1):
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[leg, leg, (height - top) / 2],
                pos=[sx * (length / 2 - 2 * leg), sy * (width / 2 - 2 * leg), (height - top) / 2],
                rgba=[c * 0.7 for c in rgba[:3]] + [1],
            )


def _bin(spec: mujoco.MjSpec, obj: ObjectEntry, rgba: list[float]) -> None:
    """Open-top container; walls are 5 mm thick."""
    length, width, height = obj.size
    t = 0.005
    x, y = obj.pos[0], obj.pos[1]
    z = obj.pos[2] if len(obj.pos) == 3 else 0.0
    parent = spec.worldbody
    if not obj.fixed:
        parent = spec.worldbody.add_body(name=obj.name, pos=[x, y, z])
        parent.add_freejoint(name=obj.name)
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
    for i, (size, pos) in enumerate(parts):
        kwargs = {} if obj.fixed else {"mass": mass / len(parts)}
        parent.add_geom(
            name=obj.name if i == 0 else f"{obj.name}_wall{i}",
            type=G,
            size=size,
            pos=[x + pos[0], y + pos[1], z + pos[2]],
            rgba=rgba,
            **kwargs,
        )


def _add_home_key(spec: mujoco.MjSpec, model: mujoco.MjModel, homes) -> None:
    """Bake every robot's home pose into a 'home' keyframe so reset returns there."""
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    # Free joints start at their attachment frame; take that from a forward pass.
    mujoco.mj_forward(model, data)
    ctrl = np.zeros(model.nu)
    for robot, joint_values, ctrl_values in homes:
        for joint, value in joint_values.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot}/{joint}")
            if j >= 0:
                qpos[model.jnt_qposadr[j]] = value
        for act, value in ctrl_values.items():
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/{act}")
            if a >= 0:
                ctrl[a] = value
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
