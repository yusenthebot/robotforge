"""Sampling sensors while time passes: rate, latency and the noise, bias and drift of each reading.

A suite is built from a compiled model alone (its `rf_sensor:` definitions, see declare.py). An ideal sensor
(no rate, no latency) is lazy: it is measured when someone reads it, so it costs nothing while the
simulation steps. A sensor with a rate is sampled during stepping at that rate and holds its last sample in
between; one with a latency keeps its recent samples in a ring buffer and is read `latency_s` behind. Each
sensor's noise comes from its own random generator, seeded from its name (or its `seed`), and restarted on
every reset, so the same run of a world reads the same numbers.

Samples taken after a MuJoCo step describe the state the step started from (MuJoCo computes positions,
contacts and sensor values before it integrates), so they are stamped one timestep back; encoders read the
joint angles the step ended on and are stamped with the clock.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.spec import CHANNELS, default_seed

log = logging.getLogger(__name__)

# Sampling rates of heavy sensors that are not ideal but were given no rate (light ones sample on every step).
LATENCY_RATES = {"lidar": 10.0, "lidar3d": 10.0, "camera": 30.0}
EPS = 1e-9


@dataclass
class Knobs:
    """A sensor's realism settings, resolved: zero means ideal."""

    rate_hz: float | None = None
    latency_s: float = 0.0
    noise: dict[str, float] = field(default_factory=dict)  # channel -> std
    bias: dict[str, np.ndarray] = field(default_factory=dict)  # channel -> constant offset (per axis)
    drift: dict[str, float] = field(default_factory=dict)  # channel -> random walk std per sqrt(s)
    dropout: float = 0.0
    seed: int = 0

    @classmethod
    def of(cls, kind: str, data: dict[str, Any], full_name: str) -> Knobs:
        channels = CHANNELS[kind]

        def per_channel(value: Any, names: Iterable[str]) -> dict[str, Any]:
            if value is None:
                return {}
            if isinstance(value, dict):
                return dict(value)
            return {c: value for c in names}

        noise = {k: float(v) for k, v in per_channel(data.get("noise"), channels).items() if v}
        biased = ("accel", "gyro") if kind == "imu" else ("turn",)
        bias = {
            k: np.broadcast_to(np.asarray(v, float), (3,)).copy()
            for k, v in per_channel(data.get("bias"), biased).items()
        }
        drift = {k: float(v) for k, v in per_channel(data.get("drift"), biased).items() if v}
        seed = data.get("seed")
        return cls(
            rate_hz=data.get("rate_hz"),
            latency_s=float(data.get("latency_s") or 0.0),
            noise=noise,
            bias={k: v for k, v in bias.items() if np.any(v)},
            drift=drift,
            dropout=float(data.get("dropout") or 0.0),
            seed=int(seed) if seed is not None else default_seed(full_name),
        )

    @property
    def ideal(self) -> bool:
        return not (self.rate_hz or self.latency_s or self.noise or self.bias or self.drift or self.dropout)


class Sensor:
    """One declared sensor bound to a compiled model. Subclasses say what it measures and how it reads."""

    kind: ClassVar[str]
    from_forward: ClassVar[bool] = True  # measures what MuJoCo's forward pass computed (stamped a step back)
    integrates: ClassVar[bool] = False  # must see every sample to keep its estimate (odometry)
    renders: ClassVar[bool] = False  # measured by drawing a picture (cameras): the hub does that

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        self.decl = decl
        self.name = decl.name
        self.local = decl.data["name"]
        self.robot = decl.robot
        self.type = decl.data["type"]
        self.knobs = Knobs.of(self.type, decl.data, decl.name)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, decl.full(decl.data.get("body", "")))
        self.body = body if body >= 0 else 0
        site = decl.data.get("site")
        self.site = _id(model, mujoco.mjtObj.mjOBJ_SITE, decl.full(site)) if site else -1
        # An ideal sensor is measured when read. Any other is sampled while time passes, so its noise does not
        # depend on who reads it how often (a viewer reads ten times a second): at its rate, or a heavy sensor
        # at a realistic one and a light one on every step.
        self.lazy = self.knobs.ideal and not self.integrates
        rate = self.knobs.rate_hz
        if rate is None and not self.lazy:
            rate = self.default_rate if self.integrates else LATENCY_RATES.get(self.type)
        self.period = 1.0 / rate if rate else None
        keep = math.ceil(self.knobs.latency_s / (self.period or model.opt.timestep)) + 3
        self.samples: deque[tuple[float, Any]] = deque(maxlen=max(keep, 2))
        self.rng = np.random.default_rng(self.knobs.seed)
        self.next_due = 0.0
        self.last_time: float | None = None  # when the last sample was taken (drift and integration)

    default_rate: ClassVar[float] = 50.0

    # ---- what subclasses provide ----------------------------------------------------------------------

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        """The ideal value now."""
        raise NotImplementedError

    def corrupt(self, value: Any, dt: float) -> Any:
        """The value as this sensor reports it: noise (and bias, drift, dropout where the type has them)."""
        return value

    def reading(self, value: Any, full: bool = False) -> dict[str, Any]:
        """The compact reading an agent gets."""
        raise NotImplementedError

    def line(self, value: Any) -> str:
        """One line for observe's sensor list."""
        raise NotImplementedError

    def viewer(self, value: Any, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        """What the viewer needs to draw it (besides its pose)."""
        return {}

    # ---- sampling -------------------------------------------------------------------------------------

    def noise(self, channel: str, shape: tuple[int, ...] | int = ()) -> np.ndarray | float:
        std = self.knobs.noise.get(channel, 0.0)
        if not std:
            return np.zeros(shape) if shape != () else 0.0
        return self.rng.normal(0.0, std, shape)

    def stamp(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        return float(data.time) - (model.opt.timestep if self.from_forward and data.time > 0 else 0.0)

    def sample(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, Any]:
        """Measure, corrupt and stamp one sample now."""
        when = self.stamp(model, data)
        dt = 0.0 if self.last_time is None else max(when - self.last_time, 0.0)
        self.last_time = when
        return when, self.corrupt(self.measure(model, data), dt)

    def take(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Keep a sample of the moment the data shows, and say when the next one is due (by sample time)."""
        sample = self.sample(model, data)
        self.samples.append(sample)
        self.next_due = sample[0] + (self.period or 0.0)

    def tick(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called after every step: take a sample when one is due. Due is judged by the moment a sample would
        show, so a run reads the same samples (and draws the same noise) however often it is read."""
        when = self.stamp(model, data)
        if self.samples:
            last = self.samples[-1][0]
            if when < last - EPS:  # the clock was turned back under it (data reset by hand): start again
                self.samples.clear()
                self.next_due = when
            elif when <= last + EPS:  # this moment is sampled already (the suite's first sample)
                return
        if when + EPS < self.next_due:
            return
        self.take(model, data)

    def current(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, Any] | None:
        """The (stamp, value) the sensor reports now: measured now when lazy, else its newest sample at least
        `latency_s` old, or None when it has none yet (right after a reset, with a latency)."""
        if self.lazy:
            return self.sample(model, data)
        if not self.samples:  # not primed (a suite used on its own): the moment now is its first sample
            self.take(model, data)
        cutoff = float(data.time) - self.knobs.latency_s + 1e-6
        for when, value in reversed(self.samples):
            if when <= cutoff:
                return when, value
        return None


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise ValueError(f"the model has no {kind.name.removeprefix('mjOBJ_').lower()} named {name!r}")
    return i


def native_slice(model: mujoco.MjModel, decl: declare.Declared, local: str) -> slice:
    """Where one of MuJoCo's own sensors keeps its value in sensordata."""
    s = _id(model, mujoco.mjtObj.mjOBJ_SENSOR, decl.full(local))
    adr, dim = int(model.sensor_adr[s]), int(model.sensor_dim[s])
    return slice(adr, adr + dim)


class Suite:
    """The sensors of one compiled model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        from robotforge.sensors.kinds import KINDS  # the sensor types, defined beside their readings

        self.model = model
        self.sensors: dict[str, Sensor] = {}
        self.broken: dict[str, str] = {}
        for decl in declare.read(model):
            cls = KINDS.get(decl.data["type"])
            if cls is None:
                self.broken[decl.name] = f"unknown sensor type {decl.data['type']!r}"
                continue
            try:
                self.sensors[decl.name] = cls(model, decl)
            except (ValueError, KeyError, TypeError) as exc:  # a broken definition must not break the world
                log.warning("sensor %s left out: %s", decl.name, exc)
                self.broken[decl.name] = str(exc)
        self.stepped = [s for s in self.sensors.values() if not s.lazy]
        # MuJoCo sensors some definition reads: the rest of a robot's MuJoCo sensors are its own (a G1's IMU).
        self.covered: set[str] = set()
        for s in self.sensors.values():
            native = s.decl.data.get("native", {})
            data = s.decl.data
            names = [*native.values(), *data.get("touch", data.get("cells", [])), *data.get("rays_native", [])]
            self.covered.update(s.decl.full(n) for n in names)

    def prime(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Take every sampled sensor's first sample from the state as it is (after a load, a reset or an edit
        that put robots back where they were): a run then draws the same noise whether or not anything reads
        before time first passes."""
        for sensor in self.stepped:
            sensor.samples.clear()
            sensor.take(model, data)

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for sensor in self.stepped:
            sensor.tick(model, data)

    def robots(self) -> list[str]:
        return sorted({s.robot for s in self.sensors.values()})

    def of(self, robot: str) -> list[Sensor]:
        return [s for s in self.sensors.values() if s.robot == robot]
