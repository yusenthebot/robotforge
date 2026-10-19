"""A sensor as a robot's world.yaml entry says it, checked as it is written.

world.yaml keeps the compact form an agent wrote (`{"type": "lidar3d", "rate_hz": 10}`); the defaults are
filled in only when the robot is compiled (`resolved`), so a better default reaches saved worlds without
rewriting them. Each type takes the options that mean something for it, and an option another type takes
is refused with the ones this type has: a silently ignored option is a sensor that is not what was asked.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from robotforge import wording

# A list, not a tuple: plugins add their types to it (robotforge.extend.sensor) and take them off again.
SENSOR_TYPES: list[str] = [
    "camera",
    "lidar",
    "lidar3d",
    "imu",
    "encoders",
    "force_torque",
    "touch",
    "rangefinder",
    "gps",
    "odometry",
    "wheel_odometry",
    "sonar",
    "skin",
    "temperature",
    "gas",
]
BUILT_IN_TYPES = tuple(SENSOR_TYPES)
PLUGIN_PARAMS: dict[str, dict[str, Any]] = {}  # a plugin sensor type -> its options and their defaults

# The words agents use for a type, and the options they imply ("depth camera" is a camera in depth mode).
ALIASES: dict[str, tuple[str, dict[str, Any]]] = {
    "depth_camera": ("camera", {"mode": "depth"}),
    "depth": ("camera", {"mode": "depth"}),
    "stereo": ("camera", {"mode": "stereo"}),
    "stereo_camera": ("camera", {"mode": "stereo"}),
    "stereo_pair": ("camera", {"mode": "stereo"}),
    "event": ("camera", {"mode": "event"}),
    "event_camera": ("camera", {"mode": "event"}),
    "dvs": ("camera", {"mode": "event"}),
    "neuromorphic": ("camera", {"mode": "event"}),
    "fisheye": ("camera", {"lens": "fisheye"}),
    "fisheye_camera": ("camera", {"lens": "fisheye"}),
    "wide_angle": ("camera", {"lens": "fisheye"}),
    "rgb_camera": ("camera", {"mode": "rgb"}),
    "rgbd": ("camera", {"mode": "depth"}),
    "segmentation_camera": ("camera", {"mode": "segmentation"}),
    "segmentation": ("camera", {"mode": "segmentation"}),
    "cam": ("camera", {}),
    "3d_lidar": ("lidar3d", {}),
    "lidar_3d": ("lidar3d", {}),
    "2d_lidar": ("lidar", {}),
    "lidar_2d": ("lidar", {}),
    "laser_scanner": ("lidar", {}),
    "ft": ("force_torque", {}),
    "force": ("force_torque", {}),
    "force_sensor": ("force_torque", {}),
    "wrench": ("force_torque", {}),
    "encoder": ("encoders", {}),
    "joint_encoders": ("encoders", {}),
    "range": ("rangefinder", {}),
    "range_finder": ("rangefinder", {}),
    "ultrasonic": ("sonar", {}),
    "ultrasound": ("sonar", {}),
    "wheel_odom": ("wheel_odometry", {}),
    "wheel_encoders": ("wheel_odometry", {}),
    "tactile": ("skin", {}),
    "tactile_skin": ("skin", {}),
    "taxels": ("skin", {}),
    "thermometer": ("temperature", {}),
    "temp": ("temperature", {}),
    "thermal": ("temperature", {}),
    "gas_sensor": ("gas", {}),
    "chemical": ("gas", {}),
    "gnss": ("gps", {}),
    "odom": ("odometry", {}),
    "contact": ("touch", {}),
    "bumper": ("touch", {}),
    "touch_array": ("touch", {"rows": 2, "cols": 2}),
    "accelerometer": ("imu", {}),
    "gyro": ("imu", {}),
    "gyroscope": ("imu", {}),
}

# The quantities a sensor measures, each in its own unit: a noise, bias or drift given as one number applies
# to every channel the type has (in that channel's unit), a mapping sets channels one by one.
CHANNELS: dict[str, tuple[str, ...]] = {
    "camera": ("depth", "rgb"),
    "lidar": ("range",),
    "lidar3d": ("range",),
    "rangefinder": ("range",),
    "imu": ("accel", "gyro", "orientation"),
    "encoders": ("position",),
    "force_torque": ("force", "torque"),
    "touch": ("force",),
    "gps": ("position",),
    "odometry": ("speed", "turn"),
    "wheel_odometry": ("speed", "turn"),
    "sonar": ("range",),
    "skin": ("force",),
    "temperature": ("temperature",),
    "gas": ("concentration",),
}
UNITS = {
    "depth": "m",
    "rgb": "levels of 255",
    "range": "m",
    "accel": "m/s^2",
    "gyro": "rad/s",
    "orientation": "deg",
    "position": "m (encoders: rad, or m for a sliding joint)",
    "force": "N",
    "torque": "N m",
    "speed": "m/s",
    "turn": "rad/s",
    "temperature": "deg C",
    "concentration": "ppm",
}
BIASED: dict[str, tuple[str, ...]] = {"imu": ("accel", "gyro"), "odometry": ("turn",), "wheel_odometry": ("turn",)}

PLACED = {"name", "on", "at", "direction"}
KNOBS = {"rate_hz", "latency_s", "noise", "seed"}
OPTIONS: dict[str, set[str]] = {
    "camera": PLACED | KNOBS | {"mode", "width", "height", "fov", "range", "lens", "baseline", "threshold", "window_s"},
    "lidar": PLACED | KNOBS | {"channels", "rays", "vfov", "hfov", "range", "dropout"},
    "lidar3d": PLACED | KNOBS | {"channels", "rays", "vfov", "hfov", "range", "dropout"},
    "rangefinder": PLACED | KNOBS | {"range", "dropout"},
    "imu": PLACED | KNOBS | {"bias", "drift"},
    "encoders": {"name"} | KNOBS | {"joints", "resolution"},
    "force_torque": PLACED | KNOBS | {"between"},
    "touch": {"name", "on", "at"} | KNOBS | {"rows", "cols", "size"},
    "gps": {"name", "on", "at"} | KNOBS,
    "odometry": {"name", "on"} | KNOBS | {"bias", "drift"},
    "wheel_odometry": {"name", "on"} | KNOBS | {"bias", "drift", "wheels", "radius", "slip"},
    "sonar": PLACED | KNOBS | {"range", "beam", "min_range", "dropout"},
    "skin": {"name", "on"} | KNOBS | {"rows", "cols", "faces"},
    "temperature": {"name", "on", "at"} | KNOBS | {"response_s"},
    "gas": {"name", "on", "at"} | KNOBS | {"response_s"},
}

# Defaults filled in at compile time. A 2D lidar is a hobby scanner (one ring, 1 degree, 12 m), a 3D lidar a
# 16-channel spinning one at 2 degrees across (2880 rays a scan: cheap enough to sample at 10 Hz).
DEFAULTS: dict[str, dict[str, Any]] = {
    "camera": {"mode": "rgb", "width": 320, "height": 240, "fov": 60.0, "range": 10.0, "at": "front"},
    "lidar": {"channels": 1, "rays": 360, "vfov": [0.0, 0.0], "hfov": 360.0, "range": 12.0, "at": "top"},
    "lidar3d": {"channels": 16, "rays": 180, "vfov": [-15.0, 15.0], "hfov": 360.0, "range": 30.0, "at": "top"},
    "rangefinder": {"range": 4.0, "at": "front"},
    "imu": {"at": "center"},
    "encoders": {"resolution": 0.0},
    "force_torque": {"at": "origin"},
    "touch": {"rows": 1, "cols": 1, "at": "bottom"},
    "gps": {"at": "center"},
    "odometry": {"at": "center"},
    "wheel_odometry": {"at": "center"},
    "sonar": {"range": 4.0, "beam": 30.0, "min_range": 0.15, "at": "front"},
    "skin": {"rows": 3, "cols": 3, "faces": ["front", "back", "left", "right", "top"]},
    "temperature": {"at": "top"},
    "gas": {"at": "top"},
}
ANCHORS = ("center", "origin", "top", "bottom", "front", "back", "left", "right")
MAX_RAYS = 8192  # rays in one scan: more slows every step a lidar samples at (multiRay is about 1 us a ray)
LIDAR_MAX_HZ = 50.0  # scans a second: real spinning lidars turn at 5 to 20 Hz
LIDAR_MAX_RAYS_S = 200_000  # rays a second (rate_hz x rays a scan): about a fifth of a second of work a second
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
BASE_NAMES = {
    "lidar3d": "lidar",
    "rangefinder": "range",
    "force_torque": "ft",
    "touch": "touch",
    "wheel_odometry": "wheel_odom",
    "temperature": "temp",
}


class SensorEntry(BaseModel):
    """One sensor on a robot. Every field but `type` is optional; see docs/SENSORS.md for what each means."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: str
    name: str | None = None
    on: str | None = Field(None, description="body, part, geom, site or joint it sits on; default the base")
    at: str | list[float] | None = Field(None, description="anchor (top, front, center, ...) or [x, y, z] m")
    direction: str | list[float] | None = Field(None, description="where it looks, in its body's frame")
    # camera
    mode: Literal["rgb", "depth", "segmentation", "event", "stereo"] | None = None
    width: int | None = Field(None, ge=16, le=1280)
    height: int | None = Field(None, ge=16, le=960)
    fov: float | None = Field(
        None, gt=5, le=200, description="camera vertical field of view, degrees (a fisheye's: across its circle)"
    )
    lens: Literal["pinhole", "fisheye"] | None = Field(
        None, description="camera: a fisheye lens sees up to 200 degrees"
    )
    baseline: float | None = Field(None, ge=0.02, le=0.5, description="stereo camera: distance between the lenses, m")
    threshold: float | None = Field(None, gt=0.01, le=1.0, description="event camera: log brightness change per event")
    window_s: float | None = Field(None, gt=0.001, le=0.5, description="event camera: time the two frames are apart, s")
    # lidar, rangefinder, depth camera
    channels: int | None = Field(None, ge=1, le=128, description="lidar rings")
    rays: int | None = Field(None, ge=3, le=4096, description="lidar rays per ring")
    vfov: float | list[float] | None = Field(None, description="lidar vertical field: degrees, or [low, high]")
    hfov: float | None = Field(None, gt=0, le=360, description="lidar horizontal field, degrees")
    range: float | None = Field(None, gt=0.05, le=200, description="farthest distance measured, m")
    # touch
    rows: int | None = Field(None, ge=1, le=8)
    cols: int | None = Field(None, ge=1, le=8)
    size: list[float] | None = Field(None, description="touch pad [length, width], m")
    # encoders
    joints: list[str] | None = None
    resolution: float | None = Field(None, ge=0, le=1, description="smallest step an encoder reads, rad")
    # force_torque
    between: list[str] | None = Field(None, description="two neighboring bodies, parent and child")
    # sonar
    beam: float | None = Field(None, gt=1, le=120, description="sonar: the cone's full angle, degrees")
    min_range: float | None = Field(None, ge=0, le=2, description="sonar: nearer than this it hears nothing, m")
    # wheel odometry
    wheels: list[str] | None = Field(None, description="wheel_odometry: the wheel joints, left and right ones")
    radius: float | None = Field(None, gt=0.005, le=2, description="wheel_odometry: wheel radius, m")
    slip: float | None = Field(None, ge=0, le=0.5, description="wheel_odometry: each wheel reads this share off")
    # skin
    faces: list[str] | None = Field(None, description="skin: which faces of the body carry taxels")
    # temperature, gas
    response_s: float | None = Field(None, gt=0, le=300, description="time constant of the sensor's response, s")
    # realism: ideal unless set
    rate_hz: float | None = Field(None, gt=0, le=2000)
    latency_s: float | None = Field(None, ge=0, le=2)
    noise: float | dict[str, float] | None = None
    bias: float | list[float] | dict[str, float | list[float]] | None = None
    drift: float | dict[str, float] | None = None
    dropout: float | None = Field(None, ge=0, lt=1)
    seed: int | None = None
    # a plugin sensor's own options (robotforge.extend.sensor); the type declares which
    params: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        """ "imu" alone is a sensor; a type an agent words otherwise ("depth camera", "3D lidar") is the type it
        means, with the options it implies."""
        if isinstance(data, str):
            data = {"type": data}
        if isinstance(data, dict) and isinstance(data.get("type"), str):
            kind = wording.squash(data["type"]).replace("-", "_")
            if kind not in SENSOR_TYPES and kind in ALIASES:
                kind, implied = ALIASES[kind]
                data = {**implied, **data, "type": kind}
            else:
                data = {**data, "type": kind if kind in SENSOR_TYPES else data["type"]}
            if data["type"] in PLUGIN_PARAMS:
                # A plugin sensor's own options are written beside the standard ones, whatever another type calls
                # them (a plugin's "radius" is not a wheel odometry's): all it takes are the placement and the knobs.
                allowed = OPTIONS[data["type"]] | {"type"}
                own = {k: v for k, v in data.items() if k not in allowed}
                if own:
                    given = data.get("params") if isinstance(data.get("params"), dict) else {}
                    data = {**{k: v for k, v in data.items() if k in allowed}, "params": {**given, **own}}
        return data

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v not in SENSOR_TYPES:
            plugin = [t for t in SENSOR_TYPES if t not in BUILT_IN_TYPES]
            more = f" (and from plugins: {', '.join(plugin)})" if plugin else "; plugins() lists the ones plugins add"
            raise ValueError(f"unknown sensor type {wording.short(v)}; the types are {', '.join(BUILT_IN_TYPES)}{more}")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        tidy = wording.squash(v)
        if not NAME_RE.match(tidy):
            raise ValueError(
                f"sensor name {wording.short(v)} must use lowercase letters, digits and _, start with a letter and "
                "have at most 32 characters, e.g. 'front_lidar'"
            )
        return tidy

    @field_validator("at")
    @classmethod
    def _at(cls, v: str | list[float] | None) -> str | list[float] | None:
        if isinstance(v, str):
            word = wording.squash(v)
            if word not in ANCHORS:
                raise ValueError(f"at {wording.short(v)} is not an anchor; use {', '.join(ANCHORS)} or [x, y, z] m")
            return word
        if v is not None and len(v) != 3:
            raise ValueError(f"at is an anchor word or three numbers [x, y, z] in meters, got {v!r}")
        return v

    @field_validator("direction")
    @classmethod
    def _direction(cls, v: str | list[float] | None) -> str | list[float] | None:
        if v is not None:
            axis_vector(v, "direction")
        return v

    @field_validator("vfov")
    @classmethod
    def _vfov(cls, v: float | list[float] | None) -> float | list[float] | None:
        if v is None:
            return v
        if isinstance(v, list) and len(v) != 2:
            raise ValueError(f"vfov is degrees (e.g. 30) or [low, high] (e.g. [-15, 15]), got {v!r}")
        low, high = (-v / 2, v / 2) if isinstance(v, int | float) else v
        if not (-90 <= low <= high <= 90):
            raise ValueError(f"vfov {v!r} must lie within -90 to 90 degrees, low first")
        return v

    @field_validator("size")
    @classmethod
    def _size(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and (len(v) != 2 or not all(0 < s <= 2 for s in v)):
            raise ValueError(f"touch size is [length, width] in meters, each up to 2, got {v!r}")
        return v

    @field_validator("faces")
    @classmethod
    def _faces(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        words = [wording.squash(f) for f in v]
        wrong = [w for w in words if w not in ("top", "bottom", "front", "back", "left", "right")]
        if wrong or not words:
            raise ValueError(f"faces are some of top, bottom, front, back, left, right, got {v!r}")
        return list(dict.fromkeys(words))

    @field_validator("between")
    @classmethod
    def _between(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) != 2:
            raise ValueError(f"between names two bodies, e.g. ['link7', 'hand'], got {v!r}")
        return v

    @model_validator(mode="after")
    def _fits_type(self) -> SensorEntry:
        given = {k for k in self.model_fields_set if getattr(self, k) is not None} - {"type"}
        wrong = sorted(given - OPTIONS[self.type])
        if wrong:
            own = sorted(OPTIONS[self.type] - {"name"})
            raise ValueError(
                f"{_a(self.type)} {self.type} sensor takes no {', '.join(map(repr, wrong))}; its options are "
                f"{', '.join(own)}"
            )
        self._camera_options()
        if self.type in PLUGIN_PARAMS:
            extra = sorted(set(self.params or {}) - set(PLUGIN_PARAMS[self.type]))
            if extra:
                have = ", ".join(PLUGIN_PARAMS[self.type]) or "none"
                raise ValueError(
                    f"the {self.type} sensor takes no option {', '.join(map(repr, extra))}; it takes {have}"
                )
        channels = CHANNELS[self.type]
        if isinstance(self.noise, dict):
            _channels(self.noise, channels, self.type, "noise")
        for knob in ("bias", "drift"):
            value = getattr(self, knob)
            if isinstance(value, dict):
                _channels(value, BIASED[self.type], self.type, knob)
        if isinstance(self.bias, list) and (self.type != "imu" or len(self.bias) != 3):
            raise ValueError("bias is one number, [x, y, z] for an IMU, or a mapping of channels")
        for knob in ("noise", "drift"):
            value = getattr(self, knob)
            numbers = value.values() if isinstance(value, dict) else [value] if value is not None else []
            if any(not isinstance(n, int | float) or n < 0 or not math.isfinite(n) for n in numbers):
                raise ValueError(f"{knob} is a standard deviation: zero or more, got {value!r}")
        rings = self.channels or DEFAULTS[self.type].get("channels", 1)
        per_ring = self.rays or DEFAULTS[self.type].get("rays", 1)
        if self.type.startswith("lidar") and rings * per_ring > MAX_RAYS:
            raise ValueError(
                f"{rings} channels x {per_ring} rays is {rings * per_ring} rays a scan; at most {MAX_RAYS} keep "
                "time running at speed"
            )
        if self.type.startswith("lidar") and self.rate_hz is not None:
            if self.rate_hz > LIDAR_MAX_HZ:
                raise ValueError(
                    f"a lidar's rate_hz is at most {LIDAR_MAX_HZ:g} (real ones spin at 5 to 20 Hz), "
                    f"got {self.rate_hz:g}"
                )
            if self.rate_hz * rings * per_ring > LIDAR_MAX_RAYS_S:
                most = LIDAR_MAX_RAYS_S / (rings * per_ring)
                raise ValueError(
                    f"{rings * per_ring} rays a scan at {self.rate_hz:g} Hz is {self.rate_hz * rings * per_ring:,.0f} "
                    f"rays a second; at most {LIDAR_MAX_RAYS_S:,} keep time running at speed: a rate of at most "
                    f"{most:.0f} Hz, or fewer rays"
                )
        return self

    def _camera_options(self) -> None:
        """The options only some cameras take, and what a lens can see."""
        if self.type != "camera":
            return
        mode = self.mode or "rgb"
        for option, owner in (("baseline", "stereo"), ("threshold", "event"), ("window_s", "event")):
            if getattr(self, option) is not None and mode != owner:
                raise ValueError(f"{option} is for a {owner} camera (mode={owner!r}); this one is {mode!r}")
        if self.lens == "fisheye" and mode != "rgb":
            raise ValueError(f"a fisheye lens gives an rgb picture, not {mode!r}")
        if self.fov is not None and self.fov >= 170 and self.lens != "fisheye":
            raise ValueError(
                f"fov {self.fov:g} is wider than a pinhole camera sees; lens='fisheye' reaches 200 degrees"
            )

    @model_serializer(mode="wrap")
    def _compact(self, handler: Any) -> dict[str, Any]:
        """Saved as written: the options left out stay out of world.yaml."""
        return {k: v for k, v in handler(self).items() if v is not None}


def register_type(name: str, channels: tuple[str, ...], at: str, params: dict[str, Any]) -> None:
    """Make `name` a sensor type (a plugin's): add_sensor takes it, world.yaml saves it, the tables below know it."""
    SENSOR_TYPES.append(name)
    CHANNELS[name] = channels
    OPTIONS[name] = PLACED | (KNOBS - {"dropout"}) | {"params"}
    DEFAULTS[name] = {"at": at}
    BIASED[name] = ()
    PLUGIN_PARAMS[name] = dict(params)


def unregister_type(name: str) -> None:
    if name in BUILT_IN_TYPES:
        raise ValueError(f"{name!r} is a built-in sensor type")
    if name in SENSOR_TYPES:
        SENSOR_TYPES.remove(name)
    for table in (CHANNELS, OPTIONS, DEFAULTS, BIASED, PLUGIN_PARAMS):
        table.pop(name, None)


def _a(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _channels(value: dict[str, Any], channels: tuple[str, ...], kind: str, knob: str) -> None:
    wrong = [k for k in value if k not in channels]
    if wrong:
        raise ValueError(
            f"{knob} of {_a(kind)} {kind} names its channels {', '.join(channels)} (or is one number for all), got "
            f"{', '.join(map(repr, wrong))}"
        )


# Directions an agent writes as words, in the body's own frame: its front is +x, its left +y, up +z.
AXIS_WORDS: dict[str, tuple[float, float, float]] = {
    "x": (1, 0, 0),
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "y": (0, 1, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "z": (0, 0, 1),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
    "forward": (1, 0, 0),
    "front": (1, 0, 0),
    "back": (-1, 0, 0),
    "backward": (-1, 0, 0),
    "left": (0, 1, 0),
    "right": (0, -1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}


def axis_vector(value: str | list[float], what: str = "direction") -> tuple[float, float, float]:
    """A unit vector from a word ("forward", "-z") or three numbers."""
    if isinstance(value, str):
        try:
            return AXIS_WORDS[wording.squash(value)]
        except KeyError:
            raise ValueError(
                f"{what} {wording.short(value)} is not an axis; use forward, back, left, right, up, down, x, -z ... "
                "or three numbers like [1, 0, -0.3]"
            ) from None
    if len(value) != 3:
        raise ValueError(f"{what} is three numbers [x, y, z], got {value!r}")
    norm = math.sqrt(sum(float(v) ** 2 for v in value))
    if norm < 1e-9:
        raise ValueError(f"{what} [0, 0, 0] has no direction; give e.g. [1, 0, 0]")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def resolved(entry: SensorEntry) -> dict[str, Any]:
    """Every option of the entry with the type's defaults filled in."""
    out = {**DEFAULTS[entry.type], **entry.model_dump()}
    out.setdefault("direction", "forward")
    if entry.type == "camera":
        if out.get("lens") == "fisheye" and entry.fov is None:
            out["fov"] = 180.0
        if out.get("mode") == "stereo":
            out.setdefault("baseline", 0.12)
        if out.get("mode") == "event":
            out.setdefault("threshold", 0.15)
            out.setdefault("window_s", 0.03)
    vfov = out.get("vfov")
    if isinstance(vfov, int | float):
        out["vfov"] = [-vfov / 2, vfov / 2]
    return out


def base_name(kind: str, mode: str | None = None, on: str | None = None, lens: str | None = None) -> str:
    """The name a sensor gets when none is given: "lidar", "depth_camera", "fl_touch", "hand_ft"."""
    word = BASE_NAMES.get(kind, kind)
    if kind == "camera" and mode not in (None, "rgb"):
        word = f"{mode}_camera"
    elif kind == "camera" and lens == "fisheye":
        word = "fisheye_camera"
    if on and kind in ("touch", "force_torque", "camera", "rangefinder", "imu", "sonar", "temperature", "gas"):
        word = f"{on}_{word}"
    word = re.sub(r"[^a-z0-9_]+", "_", wording.squash(word)).strip("_")
    if not word or not word[0].isalpha():
        word = f"s_{word}"
    return word[:32]


def unique_name(base: str, taken: set[str]) -> str:
    """`base`, or base_2, base_3, ... when the robot has a sensor of that name already."""
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base[:29]}_{n}"
    return name


def default_seed(full_name: str) -> int:
    """A sensor's noise is the same on every run of a world: its generator starts from its name."""
    return zlib.crc32(full_name.encode("utf-8"))


def without_sensors(robot: Any) -> Any:
    """A world.yaml robot entry with its sensors left out: adding or removing a sensor changes what a robot
    measures, not where it is, so an edit that only does that keeps the robot where it stands."""
    return robot.model_copy(update={"sensors": []}) if getattr(robot, "sensors", None) else robot
