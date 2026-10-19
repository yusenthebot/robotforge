"""Sensors that plugins define (robotforge.extend.sensor): the runtime side.

A plugin sensor is declared in the model like any other (a site on a body, a JSON definition with its options) and
read by one class, PluginSensor, that calls the plugin's function with a SensorContext. So it takes everything a
built-in sensor takes: a rate, noise and latency, a place on any body of any robot, the viewer's list and plots of
readings, `read_sensors`, `observe` and scripts. `install` puts a plugin's type into the tables the built-in types
live in, and returns what takes it out again.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import mount as mounting
from robotforge.sensors import spec
from robotforge.sensors.kinds import KINDS
from robotforge.sensors.runtime import Sensor

log = logging.getLogger(__name__)
LIST_LIMIT = 64  # numbers in one list of a reading: a plugin's reading is read by an agent, not a plotting library
SEEN_GROUPS = (1, 1, 0, 1, 1, 1)  # rays skip group 2, the drawn meshes: the shapes behind them are what is there


@dataclass(frozen=True)
class Look:
    """The module drawn where the sensor sits (it weighs nothing and collides with nothing)."""

    shape: str = "cylinder"  # cylinder [diameter, height], box [length, width, height] or sphere [diameter]
    size: Sequence[float] = (0.03, 0.012)
    color: Any = "orange"


class SensorContext:
    """What a plugin sensor's `read(ctx)` can look at. Everything is in the world's frame unless it says otherwise
    (x forward, y left, z up). Fields are fresh at every reading."""

    def __init__(self, sensor: PluginSensor, model: mujoco.MjModel, data: mujoco.MjData | None) -> None:
        self._sensor = sensor
        self.model = model
        self.data = data
        self.params: dict[str, Any] = sensor.params
        self.rng: np.random.Generator = sensor.rng
        self.robot: str = sensor.robot  # the robot it sits on
        self.name: str = sensor.local
        self.time = 0.0
        self.pos = np.zeros(3)  # the sensor's position, m
        self.rot = np.eye(3)  # its orientation: columns are its forward, left and up in the world

    def refresh(self) -> None:
        s, d = self._sensor, self.data
        self.time = float(d.time)
        if s.site >= 0:
            self.pos = np.array(d.site_xpos[s.site])
            self.rot = np.array(d.site_xmat[s.site]).reshape(3, 3)
        else:
            self.pos = np.array(d.xpos[s.body])
            self.rot = np.array(d.xmat[s.body]).reshape(3, 3)

    @property
    def forward(self) -> np.ndarray:
        return self.rot[:, 0]

    @property
    def left(self) -> np.ndarray:
        return self.rot[:, 1]

    @property
    def up(self) -> np.ndarray:
        return self.rot[:, 2]

    def position(self, name: str) -> np.ndarray:
        """Where a body, geom or site of the world is ("cube", "dog/trunk"), m."""
        for kind, table in (
            (mujoco.mjtObj.mjOBJ_BODY, self.data.xpos),
            (mujoco.mjtObj.mjOBJ_GEOM, self.data.geom_xpos),
            (mujoco.mjtObj.mjOBJ_SITE, self.data.site_xpos),
        ):
            found = mujoco.mj_name2id(self.model, kind, name)
            if found >= 0:
                return np.array(table[found])
        raise ValueError(f"the world has no body, geom or site named {name!r}")

    def distance(self, name: str) -> float:
        """Straight-line distance from the sensor to a body, geom or site, m."""
        return float(np.linalg.norm(self.position(name) - self.pos))

    def ray(self, direction: Sequence[float] = (1.0, 0.0, 0.0), max_range: float = 10.0) -> tuple[float | None, str]:
        """The first thing a ray meets from the sensor, `direction` in the sensor's own frame (forward is x):
        (distance m, the geom's name) or (None, "") when nothing is within `max_range`. Its own body is skipped."""
        vec = self.rot @ np.asarray(direction, float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            raise ValueError("a ray needs a direction, e.g. (1, 0, 0)")
        return self.world_ray(vec / norm, max_range)

    def world_ray(self, direction: Sequence[float], max_range: float = 10.0) -> tuple[float | None, str]:
        """Like ray, with `direction` in the world's frame."""
        geom = np.zeros(1, dtype=np.int32)
        group = np.array(SEEN_GROUPS, dtype=np.uint8)
        dist = mujoco.mj_ray(
            self.model, self.data, self.pos, np.asarray(direction, float), group, 1, self._sensor.body, geom
        )
        if dist < 0 or dist > max_range:
            return None, ""
        return float(dist), mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom[0])) or ""


class PluginSensor(Sensor):
    """Every sensor a plugin defines: the plugin's function does the measuring."""

    kind = "plugin"

    def __init__(self, model: mujoco.MjModel, decl: Any) -> None:
        super().__init__(model, decl)
        from robotforge.extend.registry import REGISTRY

        item = REGISTRY.get("sensor", self.type)
        if item is None:
            raise ValueError(
                f"the sensor type {self.type!r} comes from a plugin that is not loaded; plugins() lists the loaded ones"
            )
        self.item = item
        self.params = {**item.meta["params"], **decl.data.get("params", {})}
        self.context = SensorContext(self, model, None)
        obj = item.obj
        self.reader = obj
        self.instance: Any = None
        self.failure: str | None = None
        if inspect.isclass(obj):
            self.instance = self._call(lambda: obj(self.context), "was created")
            self.reader = getattr(self.instance, "read", None)

    def _call(self, fn: Callable[[], Any], doing: str) -> Any:
        try:
            return fn()
        except Exception as exc:  # the user's code: a mistake in it must not stop the simulation
            self.failure = f"{type(exc).__name__}: {exc}"
            if self.failure != getattr(self, "_logged", None):
                self._logged = self.failure
                log.warning("plugin sensor %s %s with %s", self.name, doing, self.failure)
            return None

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        ctx = self.context
        ctx.model, ctx.data = model, data
        ctx.refresh()
        if self.reader is None:
            return {"error": self.failure or "the sensor has no read(ctx)"}
        self.failure = None
        value = self._call(lambda: self.reader(ctx), "failed")
        if self.failure is not None:
            return {"error": self.failure}
        if not isinstance(value, dict):
            value = {"value": value}
        return clean(value)

    def corrupt(self, value: dict[str, Any], dt: float) -> dict[str, Any]:
        out = dict(value)
        for channel in self.knobs.noise:
            got = out.get(channel)
            if isinstance(got, bool) or got is None:
                continue
            if isinstance(got, int | float):
                out[channel] = float(got) + float(self.noise(channel))
            elif isinstance(got, list) and all(isinstance(v, int | float) for v in got):
                out[channel] = [float(v) + float(self.noise(channel)) for v in got]
        return out

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        return clean(value)  # after noise: rounded as a sensor would tell it

    def summary(self, value: dict[str, Any]) -> str:
        """The first few values of the reading, "level 0.667, distance_m 0.998"."""
        shown = [f"{key} {_short(got)}" for key, got in list(value.items())[:4]]
        return ", ".join(shown) or "no reading"

    def line(self, value: dict[str, Any]) -> str:
        return f"{self.type}: {self.summary(value)}"

    def viewer(self, value: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        return {"label": self.type.replace("_", " ").capitalize(), "line": self.summary(value), "reading": value}


def clean(value: Any, depth: int = 0) -> Any:
    """A reading the way an agent reads it: plain numbers, strings, lists and dicts, rounded to what a sensor could
    tell apart (numpy values from the plugin become Python ones, and a list is cut at LIST_LIMIT)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 5) if math.isfinite(value) else None
    if depth > 3:
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        out = [clean(v, depth + 1) for v in value[:LIST_LIMIT]]
        return out
    return str(value)


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, list):
        return "[" + ", ".join(_short(v) for v in value[:3]) + (", ..." if len(value) > 3 else "") + "]"
    return str(value)


# ---- putting a plugin's type among the built-in ones ----------------------------------------------------------


def install(
    name: str,
    channels: Sequence[str],
    at: str,
    params: dict[str, Any],
    look: Look,
) -> Callable[[], None]:
    """Make `name` a sensor type for add_sensor and world.yaml. Returns the function that takes it out again."""
    spec.register_type(name, tuple(channels), at, params)
    KINDS[name] = PluginSensor
    mounting.ADD[name] = _adder(look)

    def undo() -> None:
        mounting.ADD.pop(name, None)
        KINDS.pop(name, None)
        spec.unregister_type(name)

    return undo


def _adder(look: Look) -> Callable[..., dict[str, Any]]:
    def add(spec_, body, name, opts, place, vis, model, target) -> dict[str, Any]:
        from robotforge.design import components
        from robotforge.design.visuals import VISUAL_GROUP

        site = mounting._site(body, f"rf_{name}", place.point, place.frame)
        shapes = {
            "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, 2),
            "box": (mujoco.mjtGeom.mjGEOM_BOX, 3),
            "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, 1),
        }
        kind, count = shapes[look.shape]
        size = [float(s) for s in look.size][:count]
        half = [size[0] / 2, size[1] / 2, 0.0] if look.shape == "cylinder" else [s / 2 for s in size]
        half += [0.0] * (3 - len(half))
        lift = 0.0
        quat = [1.0, 0.0, 0.0, 0.0]
        if place.normal is not None:  # sits on a face: the module lies on it, its axis along the face's normal
            n = np.asarray(place.normal, float)
            quat_ = np.zeros(4)
            mujoco.mju_quatZ2Vec(quat_, n)
            quat = [float(v) for v in quat_]
            lift = half[1] if look.shape == "cylinder" else half[2] if look.shape == "box" else half[0]
        centre = np.asarray(place.point, float) + (np.asarray(place.normal, float) * lift if lift else 0.0)
        body.add_geom(
            type=kind,
            size=half,
            pos=[float(v) for v in centre],
            quat=quat,
            rgba=components.color(look.color),
            contype=0,
            conaffinity=0,
            group=VISUAL_GROUP,
            mass=0.0,
        )
        return {"site": site, "params": dict(opts.get("params") or {})}

    return add
