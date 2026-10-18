"""The parts schema: a robot described as a base plus a tree of parts, the way you would build one.

Every field has a beginner-friendly default, units are meters, degrees and kilograms,
and validation errors say what to write instead. The compact form an agent writes is
what world.yaml keeps; defaults are filled in only when the robot is compiled.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robotforge.design import components

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
# Words an agent can use instead of axis letters. The robot's front is +x, its left +y, up +z.
AXIS_WORDS: dict[str, tuple[float, float, float]] = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
    "+x": (1, 0, 0),
    "+y": (0, 1, 0),
    "+z": (0, 0, 1),
    "-x": (-1, 0, 0),
    "-y": (0, -1, 0),
    "-z": (0, 0, -1),
    "forward": (1, 0, 0),
    "back": (-1, 0, 0),
    "left": (0, 1, 0),
    "right": (0, -1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}
# Anchor names usable in `at`: the center of a face of the parent's shape, its middle, the joint
# itself, or the tip of a link (see `direction`).
FACE_ANCHORS = ("top", "bottom", "front", "back", "left", "right")
ANCHORS = (*FACE_ANCHORS, "center", "start", "end")
SHAPES = ("box", "sphere", "capsule", "cylinder", "wheel", "caster", "rotor", "none")
SIZE_HINT = {
    "box": "[length, width, height] (along x, y, z)",
    "sphere": "[diameter]",
    "capsule": "[diameter, length] (length runs joint to end; the round ends stick out by the radius)",
    "cylinder": "[diameter, length]",
    "wheel": "[diameter, width]",
    "caster": "[diameter]",
    "rotor": "[diameter] (of the propeller)",
    "none": "[] (no shape: a bare joint)",
}
SIZE_COUNT = {"box": 3, "sphere": 1, "capsule": 2, "cylinder": 2, "wheel": 2, "caster": 1, "rotor": 1, "none": 0}
MAX_PARTS = 120  # after mirror and repeat; bigger robots get slow to check and to simulate
Material = Literal["pla", "aluminum", "carbon", "rubber"]
Vec3 = list[float]
Color = str | list[float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def check_name(value: str, what: str = "names") -> str:
    if not NAME_RE.match(value):
        raise ValueError(f"{what} use lowercase letters, digits and _, start with a letter, max 41 chars: {value!r}")
    return value


def axis_vector(value: str | list[float], what: str = "axis") -> tuple[float, float, float]:
    """A unit vector from an axis word ("z", "-y", "forward") or three numbers."""
    if isinstance(value, str):
        try:
            return AXIS_WORDS[value.strip().lower()]
        except KeyError:
            raise ValueError(
                f"{what} {value!r} is not an axis; use x, y, z, -x, -y, -z, forward, back, left, right, up, down "
                "or three numbers like [0, 1, 0]"
            ) from None
    if len(value) != 3:
        raise ValueError(f"{what} is three numbers [x, y, z], got {value!r}")
    norm = math.sqrt(sum(v * v for v in value))
    if norm < 1e-9:
        raise ValueError(f"{what} [0, 0, 0] has no direction; give e.g. [0, 0, 1]")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _vec3(value: list[float], what: str) -> list[float]:
    if len(value) != 3:
        raise ValueError(f"{what} is three numbers [x, y, z] in meters, got {value!r}")
    return value


# A part's color that follows the robot's main color: recolor the robot once and these parts change with it.
MAIN_COLOR = "main"


def _check_color(value: Color | None) -> Color | None:
    if value is not None and value != MAIN_COLOR:
        try:
            components.color(value)
        except ValueError as exc:
            raise ValueError(f'{exc}, or "{MAIN_COLOR}" for the robot\'s own color') from None
    return value


class SensorSpec(_Strict):
    """A sensor on the base or a part. Shorthand: just the type, e.g. "imu"."""

    type: Literal["imu", "touch", "camera", "rangefinder", "lidar"]
    name: str | None = Field(None, description="defaults to <part>_<type>, e.g. left_foot_touch")
    at: str | Vec3 = Field(
        "center", description="anchor (center, end, top, front...) or [x, y, z]; touch defaults to end"
    )
    direction: str | list[float] = Field("forward", description="where a camera or rangefinder looks")
    fov: float = Field(70.0, gt=5, lt=170, description="camera field of view, degrees")
    rays: int = Field(16, ge=3, le=64, description="lidar: rangefinders spread in a ring")
    range: float = Field(4.0, gt=0.05, le=30.0, description="rangefinder / lidar: farthest distance measured, m")
    # Realism, ideal when left out: the same knobs and meanings as add_sensor's (sensors/spec.py).
    rate_hz: float | None = Field(None, gt=0, le=2000)
    latency_s: float | None = Field(None, ge=0, le=2)
    noise: float | dict[str, float] | None = None
    bias: float | list[float] | dict[str, float | list[float]] | None = None
    drift: float | dict[str, float] | None = None
    dropout: float | None = Field(None, ge=0, lt=1)
    seed: int | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return None if v is None else check_name(v, "sensor names")

    @field_validator("at")
    @classmethod
    def _at(cls, v: str | Vec3) -> str | Vec3:
        return v if isinstance(v, str) else _vec3(v, "sensor at")

    @field_validator("direction")
    @classmethod
    def _direction(cls, v: str | list[float]) -> str | list[float]:
        axis_vector(v, "sensor direction")
        return v

    @model_validator(mode="after")
    def _knobs(self) -> SensorSpec:
        """The knobs a type takes, as add_sensor checks them (a camera has no dropout, an IMU no range noise)."""
        from robotforge.sensors.spec import SensorEntry

        knobs = {k: getattr(self, k) for k in ("rate_hz", "latency_s", "noise", "bias", "drift", "dropout", "seed")}
        given = {k: v for k, v in knobs.items() if v is not None}
        if given:
            SensorEntry.model_validate({"type": self.type, **given})
        return self

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            data = {"type": data}
        if isinstance(data, dict) and data.get("type") == "touch" and "at" not in data:
            data = {**data, "at": "end"}
        return data


class JointSpec(_Strict):
    """How a part moves relative to its parent. Shorthand: "hinge", "slide" or "fixed"."""

    type: Literal["hinge", "slide", "fixed"] = "hinge"
    axis: str | list[float] = Field("y", description="hinge: the axle it turns about; slide: the direction it moves")
    range: list[float] | None = Field(
        None, description="[low, high]: degrees for hinge (default [-90, 90]), meters for slide (required)"
    )
    continuous: bool = Field(False, description="turns forever, no range (wheels, propellers); needs a motor")
    home: float = Field(0.0, description="starting angle (degrees) or position (meters)")
    damping: float | None = Field(None, ge=0, le=1000, description="extra friction-like damping, N*m*s/rad")
    stiffness: float | None = Field(None, ge=0, le=10000, description="a spring back to home, N*m/rad (passive)")

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            if data == "free":
                raise ValueError(
                    "a part cannot have a free joint (only the whole robot floats); set base.fixed: false for a "
                    "robot that moves around, or use hinge, slide or fixed"
                )
            return {"type": data}
        return data

    @field_validator("axis")
    @classmethod
    def _axis(cls, v: str | list[float]) -> str | list[float]:
        axis_vector(v, "joint axis")
        return v

    @model_validator(mode="after")
    def _range(self) -> JointSpec:
        if self.type == "fixed":
            return self
        if self.range is not None:
            if self.continuous:
                raise ValueError("a continuous joint has no range; drop range or continuous")
            if len(self.range) != 2:
                raise ValueError(f"range is [low, high], got {self.range!r}")
            lo, hi = self.range
            if lo >= hi:
                raise ValueError(f"range {self.range!r} is backwards or empty; write [low, high], e.g. [-90, 90]")
            if self.type == "hinge" and (lo < -360 or hi > 360):
                raise ValueError(
                    f"hinge range {self.range!r} is in degrees and goes past a full turn; "
                    "use at most [-360, 360], or continuous: true for wheels"
                )
            if self.type == "slide" and (hi - lo > 3.0):
                raise ValueError(f"slide range {self.range!r} is in meters; that is a {hi - lo:.1f} m stroke")
            if not lo <= self.home <= hi:
                raise ValueError(f"home {self.home:g} is outside the range {self.range!r}")
        elif self.type == "slide" and not self.continuous:
            raise ValueError("a slide joint needs a range in meters, e.g. range: [0, 0.1]")
        elif self.type == "hinge" and not self.continuous and not -90 <= self.home <= 90:
            raise ValueError(f"home {self.home:g} is outside the default range [-90, 90]; give a range")
        return self


class ActuatorSpec(_Strict):
    """What drives a joint. Shorthand: a part name from the catalog ("sts3215", "jga25") or "none"."""

    model: str = Field(description="a servo, motor or linear actuator from the parts catalog")
    control: Literal["position", "velocity", "torque"] | None = Field(
        None, description="default: position for servos and linear actuators, velocity for motors"
    )
    gear: float = Field(1.0, ge=0.1, le=100.0, description="gearbox: multiplies torque, divides speed")

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        return {"model": data} if isinstance(data, str) else data

    @field_validator("model")
    @classmethod
    def _known(cls, v: str) -> str:
        components.actuator(v)
        return v


class RobotMount(_Strict):
    """A whole template or library robot mounted on a part (an arm on a rover, a gripper on an arm)."""

    template: str | None = None
    library: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    home: dict[str, float] = Field(
        default_factory=dict,
        description="its starting pose: joint name -> degrees (meters for a slide), e.g. an arm folded for driving",
    )

    @model_validator(mode="after")
    def _one(self) -> RobotMount:
        if (self.template is None) == (self.library is None):
            raise ValueError('robot is {"template": "arm", "params": {...}} or {"library": "so_arm100"}')
        if self.template is not None:
            from robotforge.design import templates

            templates.get(self.template)
        else:
            from robotforge.design import library

            library.get(self.library)
        return self


class RepeatSpec(_Strict):
    """Copies of a part (with everything attached to it): in a ring, in a row, or chained like a snake."""

    count: int = Field(ge=2, le=24)
    around: str | list[float] | None = Field(None, description="ring: rotate each copy about this parent axis")
    step: float | None = Field(None, description="ring: degrees between copies (default: 360 / count)")
    along: str | list[float] | None = Field(None, description="row: shift each copy along this parent direction")
    spacing: float | None = Field(None, gt=0, le=5.0, description="row: meters between copies")
    chain: bool = Field(False, description="snake: each copy hangs on the previous one, at the same `at`")

    @model_validator(mode="after")
    def _one_kind(self) -> RepeatSpec:
        kinds = [self.around is not None, self.along is not None, self.chain]
        if sum(kinds) != 1:
            raise ValueError(
                'repeat needs exactly one of: around (a ring, e.g. {"count": 6, "around": "z"}), along '
                '(a row, {"count": 3, "along": "x", "spacing": 0.1}) or chain ({"count": 8, "chain": true})'
            )
        if self.along is not None and self.spacing is None:
            raise ValueError("repeat along a direction needs spacing (meters between copies)")
        if self.around is not None:
            axis_vector(self.around, "repeat around")
        if self.along is not None:
            axis_vector(self.along, "repeat along")
        return self


class _Body(_Strict):
    """Fields shared by the base and the parts: a shape, its look and its mass."""

    size: list[float] = Field(default_factory=list, description="meters; see the shape for the meaning")
    color: Color | None = None
    material: Material | None = None
    mass: float | None = Field(None, gt=0, le=500, description="kg; default from the shape's volume and material")
    friction: float | None = Field(None, gt=0, le=3)
    offset: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="shift the shape, meters")
    mounts: dict[str, Vec3] = Field(default_factory=dict, description="named points other parts can attach at")
    sensors: list[SensorSpec] = Field(default_factory=list)

    @field_validator("color")
    @classmethod
    def _color(cls, v: Color | None) -> Color | None:
        return _check_color(v)

    @field_validator("offset")
    @classmethod
    def _offset(cls, v: Vec3) -> Vec3:
        return _vec3(v, "offset")

    @field_validator("mounts")
    @classmethod
    def _mounts(cls, v: dict[str, Vec3]) -> dict[str, Vec3]:
        for name, point in v.items():
            check_name(name, "mount names")
            if name in ANCHORS:
                raise ValueError(f"mount name {name!r} is taken by a built-in anchor; pick another")
            _vec3(point, f"mount {name!r}")
        return v

    def _check_size(self, shape: str) -> None:
        need = SIZE_COUNT[shape]
        if not self.size and need:
            raise ValueError(f"size is required: a {shape} has size {SIZE_HINT[shape]} in meters")
        if len(self.size) != need or any(s <= 0 for s in self.size):
            raise ValueError(f"a {shape} has size {SIZE_HINT[shape]} in meters, got {self.size!r}")
        if any(s > 5.0 for s in self.size):
            raise ValueError(f"size {self.size!r} is in meters; parts over 5 m are not supported")


class BaseSpec(_Body):
    """The main body everything else hangs on."""

    shape: Literal["box", "sphere", "capsule", "cylinder"] = "box"
    size: list[float] = Field(default_factory=lambda: [0.24, 0.16, 0.05])
    axis: str | list[float] = Field("z", description="capsule / cylinder base: which way its length runs")
    fixed: bool = Field(False, description="true: bolted to the floor (arms, turrets); false: moves freely")

    @field_validator("axis")
    @classmethod
    def _axis(cls, v: str | list[float]) -> str | list[float]:
        axis_vector(v, "base axis")
        return v

    @model_validator(mode="after")
    def _sizes(self) -> BaseSpec:
        self._check_size(self.shape)
        return self


class PartSpec(_Body):
    """One rigid piece of the robot, joined to its parent."""

    name: str
    parent: str = "base"
    at: str | Vec3 = Field("center", description="anchor on the parent (top, end, front, a mount...) or [x, y, z]")
    shift: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 0.0], description="added to `at`, meters")
    rotate: Vec3 = Field(
        default_factory=lambda: [0.0, 0.0, 0.0], description="[roll, pitch, yaw] degrees, relative to the parent"
    )
    joint: JointSpec = Field(default_factory=JointSpec)
    actuator: ActuatorSpec | None = Field(None, description='default: the robot servo / motor; "none" = passive')
    shape: Literal["box", "sphere", "capsule", "cylinder", "wheel", "caster", "rotor", "none"] = "capsule"
    direction: str | list[float] | None = Field(
        None, description="the link grows from its joint this way (e.g. down for a leg); unset = centered on it"
    )
    foot: bool = Field(False, description="add a rubber foot ball at the end")
    collide: bool = True
    mirror: bool = Field(False, description="add a mirror copy on the other side (left <-> right in its name)")
    repeat: RepeatSpec | None = None
    robot: RobotMount | None = Field(None, description="mount a template or library robot on this part")
    spin: Literal["ccw", "cw"] = Field(
        "ccw",
        description="rotor: which way its propeller turns seen from above; copies in a ring or a row alternate and "
        "a mirrored copy turns the other way, as multirotors need",
    )
    thrust: float | None = Field(None, gt=0, le=2000, description="rotor: the most it pushes, N (default from size)")
    passive: bool = Field(False, exclude=True)  # set from actuator: "none"

    @model_validator(mode="before")
    @classmethod
    def _defaults_from_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if data.get("actuator") in ("none", "passive", False):
            data["actuator"] = None
            data["passive"] = True
        shape = data.get("shape")
        if shape in ("wheel",) and "joint" not in data:
            data["joint"] = {"type": "hinge", "axis": "y", "continuous": True}
        if shape in ("caster", "rotor") and "joint" not in data:
            data["joint"] = "fixed"
        if data.get("robot") is not None:
            data.setdefault("shape", "none")
            data.setdefault("joint", "fixed")
        return data

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        check_name(v, "part names")
        if v in ("base", "world"):
            raise ValueError(f"{v!r} is reserved; name the part something else")
        return v

    @field_validator("at")
    @classmethod
    def _at(cls, v: str | Vec3) -> str | Vec3:
        return v if isinstance(v, str) else _vec3(v, "at")

    @field_validator("shift", "rotate")
    @classmethod
    def _three(cls, v: Vec3) -> Vec3:
        return _vec3(v, "shift / rotate")

    @field_validator("direction")
    @classmethod
    def _direction(cls, v: str | list[float] | None) -> str | list[float] | None:
        if v is not None:
            axis_vector(v, "direction")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> PartSpec:
        self._check_size(self.shape)
        if self.shape == "rotor":
            if self.joint.type != "fixed" or self.actuator is not None:
                raise ValueError(
                    "a rotor is bolted on (its joint is fixed) and drives itself: leave out joint and actuator, set "
                    "thrust (N) for a stronger one; to tilt it, hang it on a part with a hinge"
                )
        elif self.thrust is not None:
            raise ValueError('thrust is for rotor parts (shape: "rotor")')
        if self.joint.type == "fixed" and self.actuator is not None:
            raise ValueError("a fixed joint does not move, so it takes no actuator")
        if self.joint.continuous and self.actuator is not None:
            part = components.actuator(self.actuator.model)
            control = self.actuator.control or _default_control(part)
            if control == "position":
                raise ValueError(
                    f"a continuous joint cannot be position-controlled by {part.name}; "
                    "use a motor (e.g. tt_motor, jga25) or control: velocity"
                )
        if self.actuator is not None:
            part = components.actuator(self.actuator.model)
            if self.joint.type == "slide" and part.kind != "linear":
                linear = ", ".join(n for n, a in components.ACTUATORS.items() if a.kind == "linear")
                raise ValueError(f"{part.name} turns; a slide joint needs a linear actuator ({linear})")
            if self.joint.type == "hinge" and part.kind == "linear":
                raise ValueError(f"{part.name} is a linear actuator; a hinge needs a servo or a motor")
        if self.robot is not None and self.mirror and self.robot.library is not None:
            raise ValueError("library robots cannot be mirrored; add a second part instead")
        return self


def _default_control(part: components.Actuator) -> str:
    return "velocity" if part.kind == "motor" else "position"


class BuildSpec(_Strict):
    """A robot built from parts: world.yaml's `build:` source."""

    base: BaseSpec = Field(default_factory=BaseSpec)
    parts: list[PartSpec] = Field(default_factory=list)
    color: Color = Field("blue", description="the robot's main color (the base); parts default to white")
    material: Material = "pla"
    servo: str = Field("sts3215", description="default actuator for hinges")
    motor: str = Field("tt_motor", description="default actuator for wheels and continuous joints")
    linear: str = Field("l16", description="default actuator for slide joints")
    servo_caps: bool = Field(True, description="draw each joint's servo or motor, case and horn (it carries its mass)")

    @field_validator("color")
    @classmethod
    def _color(cls, v: Color) -> Color:
        components.color(v)
        return v

    @model_validator(mode="after")
    def _defaults(self) -> BuildSpec:
        for field_name, kinds in (("servo", ("servo",)), ("motor", ("motor",)), ("linear", ("linear",))):
            part = components.actuator(getattr(self, field_name))
            if part.kind not in kinds:
                options = ", ".join(n for n, a in components.ACTUATORS.items() if a.kind in kinds)
                raise ValueError(f"{field_name} must be one of: {options}")
        names = [p.name for p in self.parts]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"part names must be unique: {', '.join(dupes)}")
        return self

    def part(self, name: str) -> PartSpec | None:
        return next((p for p in self.parts if p.name == name), None)


def actuator_for(build: BuildSpec, part: PartSpec) -> ActuatorSpec | None:
    """The actuator a part ends up with: its own, the robot's default, or none."""
    if part.passive or part.joint.type == "fixed":
        return None
    if part.actuator is not None:
        return part.actuator
    if part.joint.type == "slide":
        return ActuatorSpec(model=build.linear)
    if part.joint.continuous:
        return ActuatorSpec(model=build.motor)
    return ActuatorSpec(model=build.servo)


def control_of(spec: ActuatorSpec) -> str:
    return spec.control or _default_control(components.actuator(spec.model))
