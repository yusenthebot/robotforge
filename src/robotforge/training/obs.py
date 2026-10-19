"""An observation assembled from the sensors a robot really carries.

A learner should see what the robot could sense, so the observation follows the robot: put an IMU, a lidar or a
depth camera on it (add_sensor) and its numbers join the observation; take one off and they leave. Each sensor
becomes a block of named channels:

- lidar and lidar3d: the ranges, the nearest hit in each of `lidar_bins` slices around the robot, as a share of
  the sensor's range (1 = nothing seen);
- camera in depth mode: the depth picture shrunk to `depth_size`, again as a share of the range;
- every other sensor, including a plugin's: the numbers of its own reading, the ones add_sensor / read_sensors
  show (an IMU's accel_m_s2, an encoder's rad by joint, a touch pad's total_n ...).

The layout is fixed when the observation is built (from a first reading) and then read by position, so the vector
never changes shape from step to step. Noise, latency and rate of each sensor work as in the live world, because
the same sensor objects measure here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from robotforge.control.tasks.base import TaskError

LIDAR_BINS = 16
DEPTH_SIZE = (8, 6)  # columns, rows
MAX_SENSOR_CHANNELS = 400  # a sensor that reads more numbers than this is shrunk or refused, not silently huge


@dataclass(frozen=True)
class Block:
    """The channels one source (the task, or one sensor) adds to the observation."""

    source: str  # "task" or the sensor's name, "imu"
    kind: str  # "task", or the sensor's type
    start: int
    labels: tuple[str, ...]

    @property
    def stop(self) -> int:
        return self.start + len(self.labels)

    def describe(self) -> dict[str, Any]:
        return {"source": self.source, "kind": self.kind, "slice": [self.start, self.stop], "labels": list(self.labels)}


def _flatten(value: Any, prefix: str, labels: list[str], out: list[float], depth: int = 0) -> None:
    """The numbers in a reading, in order, each named by where it sits ("accel_m_s2 0", "rad hip")."""
    if isinstance(value, bool):
        labels.append(prefix)
        out.append(float(value))
    elif isinstance(value, int | float | np.integer | np.floating):
        labels.append(prefix)
        out.append(0.0 if not math.isfinite(float(value)) else float(value))
    elif isinstance(value, dict) and depth < 4:
        for key, item in value.items():
            _flatten(item, f"{prefix} {key}".strip(), labels, out, depth + 1)
    elif isinstance(value, list | tuple | np.ndarray) and depth < 4:
        for i, item in enumerate(value):
            _flatten(item, f"{prefix} {i}".strip(), labels, out, depth + 1)
    # text, None and anything else is not a number: left out


def _reading_numbers(sensor: Any, value: Any) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    out: list[float] = []
    _flatten(sensor.reading(value), "", labels, out)
    return labels, np.asarray(out, dtype=np.float64)


class _Extractor:
    """Turns one sensor's sample into numbers, with a fixed list of labels."""

    labels: tuple[str, ...]

    def __init__(self, sensor: Any) -> None:
        self.sensor = sensor

    def numbers(self, value: Any) -> np.ndarray:
        raise NotImplementedError


class _Reading(_Extractor):
    """Any sensor: the numbers of its own reading, found once and then read by name."""

    def __init__(self, sensor: Any, value: Any) -> None:
        super().__init__(sensor)
        labels, numbers = _reading_numbers(sensor, value)
        if len(labels) > MAX_SENSOR_CHANNELS:
            raise TaskError(
                f"sensor {sensor.name!r} reads {len(labels)} numbers, more than {MAX_SENSOR_CHANNELS}; "
                "use fewer rays, joints or cells"
            )
        self.labels = tuple(labels)
        self._index = {label: i for i, label in enumerate(labels)}

    def numbers(self, value: Any) -> np.ndarray:
        labels, got = _reading_numbers(self.sensor, value)
        out = np.zeros(len(self.labels))
        for label, number in zip(labels, got, strict=True):
            i = self._index.get(label)
            if i is not None:
                out[i] = number
        return out


class _Lidar(_Extractor):
    def __init__(self, sensor: Any, bins: int) -> None:
        super().__init__(sensor)
        rays = int(sensor.rays)
        self.bins = max(1, min(int(bins), rays))
        azimuth = np.mod(np.asarray(sensor.azimuth, dtype=float), 2 * math.pi)
        self.slice_of = np.minimum((azimuth / (2 * math.pi) * self.bins).astype(int), self.bins - 1)
        self.range = float(sensor.range)
        self.labels = tuple(f"{sensor.local} range {round(360 * (b + 0.5) / self.bins)} deg" for b in range(self.bins))

    def numbers(self, value: Any) -> np.ndarray:
        dist = np.asarray(value["dist"], dtype=float)
        share = np.where(np.isfinite(dist), np.clip(dist / self.range, 0.0, 1.0), 1.0)
        out = np.ones(self.bins)
        np.minimum.at(out, self.slice_of, share)
        return out


class _Depth(_Extractor):
    """A depth camera, shrunk to a few columns and rows, each the nearest thing in its patch."""

    def __init__(self, sensor: Any, size: tuple[int, int]) -> None:
        super().__init__(sensor)
        self.cols, self.rows = max(1, int(size[0])), max(1, int(size[1]))
        self.range = float(sensor.range)
        self.labels = tuple(f"{sensor.local} depth r{r} c{c}" for r in range(self.rows) for c in range(self.cols))

    def numbers(self, value: Any) -> np.ndarray:
        depth = value.get("depth")
        if depth is None:  # no picture (rendering unavailable): as if nothing was seen
            return np.ones(self.rows * self.cols)
        depth = np.asarray(depth, dtype=float)
        h, w = depth.shape
        share = np.where(np.isfinite(depth), np.clip(depth / self.range, 0.0, 1.0), 1.0)
        rows = np.array_split(np.arange(h), self.rows)
        cols = np.array_split(np.arange(w), self.cols)
        return np.array([share[np.ix_(r, c)].min() for r in rows for c in cols])


class SensorObservation:
    """The observation block that a robot's sensors add. Owns a suite of its own sensors for one model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        *,
        lidar_bins: int = LIDAR_BINS,
        depth_size: tuple[int, int] = DEPTH_SIZE,
        start: int = 0,
    ) -> None:
        from robotforge.sensors.runtime import Suite

        self.model = model
        self.robot = robot
        self.lidar_bins = lidar_bins
        self.depth_size = depth_size
        self.suite = Suite(model)
        self.sensors = self.suite.of(robot)
        self._renderer: Any = None
        self._extractors: list[_Extractor] = []
        self.blocks: list[Block] = []
        self.skipped: list[dict[str, str]] = []  # sensors left out of the observation, and why
        self._build(start)

    # ---- layout ----

    def _build(self, start: int) -> None:
        scratch = mujoco.MjData(model := self.model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mujoco.mj_resetDataKeyframe(model, scratch, key)
        mujoco.mj_forward(model, scratch)
        self.suite.prime(model, scratch)
        at = start
        used = []
        for sensor in self.sensors:
            if sensor.kind == "camera" and sensor.mode not in ("depth", "stereo"):
                self.skipped.append(
                    {"sensor": sensor.local, "why": f"a {sensor.mode} camera is a picture; only depth cameras are used"}
                )
                continue
            used.append(sensor)
            value = self._sample(sensor, scratch)
            extractor = self._extractor(sensor, value)
            block = Block(sensor.local, sensor.type, at, extractor.labels)
            self._extractors.append(extractor)
            self.blocks.append(block)
            at = block.stop
        self.sensors = used
        self.size = at - start

    def _extractor(self, sensor: Any, value: Any) -> _Extractor:
        if sensor.kind == "lidar":
            return _Lidar(sensor, self.lidar_bins)
        if sensor.kind == "camera":
            return _Depth(sensor, self.depth_size)
        return _Reading(sensor, value)

    @property
    def labels(self) -> list[str]:
        return [f"{b.source}: {label}" for b in self.blocks for label in b.labels]

    def describe(self) -> list[dict[str, Any]]:
        return [b.describe() for b in self.blocks]

    # ---- running ----

    def _sample(self, sensor: Any, data: mujoco.MjData) -> Any:
        got = sensor.current(self.model, data)
        value = got[1] if got is not None else None
        if value is None:  # a sensor with a latency has nothing yet right after a reset: its first sample
            value = sensor.sample(self.model, data)[1]
        if sensor.renders:
            value = self._develop(sensor, value)
        return value

    def _develop(self, sensor: Any, moment: dict[str, Any]) -> dict[str, Any]:
        from robotforge.sim.render import RenderUnavailable, SnapshotRenderer

        if self._renderer is None:
            self._renderer = SnapshotRenderer()

        def render(at: dict[str, Any], **_: Any) -> Any:
            scratch = sensor.scene_data(self.model, at)
            return self._renderer.layers_of(
                self.model, scratch, sensor.view, sensor.width, sensor.height, id(self.model), geoms=True
            )

        try:
            return sensor.develop(render, moment)
        except RenderUnavailable as exc:
            raise TaskError(
                f"camera {sensor.local!r} needs offscreen OpenGL to draw its picture, which is not available here "
                f"({exc}); use a lidar or a rangefinder instead"
            ) from exc

    def prime(self, data: mujoco.MjData) -> None:
        """Take every sampled sensor's first sample from `data` (call after each reset)."""
        self.suite.prime(self.model, data)

    def vector(self, data: mujoco.MjData) -> np.ndarray:
        # Sensors with a rate, noise or latency sample as time passes; the observation is read once a control
        # step, so they are brought up to date here (no hook on the physics steps, which also lets a policy run
        # anywhere a model and its data are, such as the live world).
        self.suite.after_step(self.model, data)
        parts = [e.numbers(self._sample(e.sensor, data)) for e in self._extractors]
        return np.concatenate(parts) if parts else np.zeros(0)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def sensor_summary(model: mujoco.MjModel, robot: str) -> list[dict[str, str]]:
    """The sensors a robot carries, for explaining what an observation will hold."""
    from robotforge.sensors.runtime import Suite

    return [{"name": s.local, "type": s.type} for s in Suite(model).of(robot)]
