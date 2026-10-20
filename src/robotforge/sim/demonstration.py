"""Demonstrations: what a robot did, kept as data. A teleoperated motion, a script's run or a behavior's job is recorded
from the physics steps (joint positions, the actuator targets, the base pose, sensor readings) and the commands that
steered it (a key's move, an arm's target, the gripper), replayed on the same robot or another of the same kind, and
exported as JSON or NPZ for training.

    pg.demonstrations.start("dog", "walk_square")          # from now on, every physics step is watched
    ... keys, a gamepad, a script, move() ...
    demo = pg.demonstrations.stop("walk_square")            # a Demonstration, saved in the demonstrations folder
    pg.demonstrations.replay("walk_square", "go2")          # the commands again, on a Go2

Two ways to play one back:

- "commands" (the default) re-issues the recorded commands at their times: the robot's own controllers do the rest, so
  it works on any robot of the same kind (a Go2 replays a template dog's walk, a Panda an UR5e's reach). Arm targets are
  kept relative to the robot's base, so an arm standing elsewhere reaches the same place relative to itself.
- "joints" puts the robot in the demonstration's first pose and plays the recorded actuator targets back, as they were:
  exact, for the same robot (or one with the same actuators). Open loop, so for arms and hands, not for walking.

Nothing heavy is imported here: it runs beside the simulation's step.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robotforge import wording

FORMAT = "robotforge-demonstration"
VERSION = 1
RATE_HZ = 20.0  # frames kept per simulated second
MAX_RATE_HZ = 200.0
MAX_FRAMES = 60000  # a recording ends there (50 minutes at the default rate)
SENSOR_LEAVES = 64  # numbers kept from one sensor's reading
SKIPPED_SENSORS = ("camera", "lidar", "lidar3d", "depth")  # heavy: named on purpose, never by default
IGNORED_KEYS = {"type", "status", "t", "age_s", "stamp"}
SUFFIXES = (".json", ".npz")
FOLDER = "demonstrations"
NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Names Windows cannot make a file of, whatever the suffix
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


class DemonstrationError(ValueError):
    """A demonstration that cannot be recorded, played or written, with what to do instead."""


def clean_name(name: str) -> str:
    """A demonstration's name as a file name: lowercase letters, digits and underscores."""
    given = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if not NAME.match(given):
        raise DemonstrationError(
            f"{wording.short(str(name))!r} is not a name for a demonstration: use lowercase letters, digits and "
            'underscores, starting with a letter ("pick_cube")'
        )
    if given in RESERVED:
        raise DemonstrationError(f"{given} is a name Windows keeps for devices, so no file can have it: choose another")
    return given


@dataclass
class Demonstration:
    """The frames and commands of one robot, at `rate_hz` frames a simulated second."""

    name: str
    robot: str
    kind: str
    rate_hz: float
    created: str = ""
    world: str = ""
    joint_names: list[str] = field(default_factory=list)  # without the robot's prefix
    actuator_names: list[str] = field(default_factory=list)
    sensor_names: list[str] = field(default_factory=list)  # "imu.orientation_deg.roll"
    time: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) seconds from the first frame
    joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (T, J) rad or m
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (T, U) the actuator targets
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (T, 3) m, world frame
    base_quat: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # (T, 4) w x y z
    sensors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (T, S)
    commands: list[dict[str, Any]] = field(default_factory=list)  # {"t", "op", ...}, by t
    info: dict[str, Any] = field(default_factory=dict)  # max_speed, arms: what a replay on another robot checks

    @property
    def frames(self) -> int:
        return int(len(self.time))

    @property
    def seconds(self) -> float:
        return float(self.time[-1] - self.time[0]) if self.frames > 1 else 0.0

    def summary(self) -> dict[str, Any]:
        """What a tool says about it: the sizes, not the numbers."""
        out: dict[str, Any] = {
            "name": self.name,
            "robot": self.robot,
            "kind": self.kind,
            "seconds": round(self.seconds, 2),
            "frames": self.frames,
            "rate_hz": self.rate_hz,
            "joints": len(self.joint_names),
            "actuators": len(self.actuator_names),
            "sensor_values": len(self.sensor_names),
            "commands": len(self.commands),
        }
        ops: dict[str, int] = {}
        for command in self.commands:
            ops[command["op"]] = ops.get(command["op"], 0) + 1
        if ops:
            out["command_counts"] = ops
        return out

    # ---- files ----

    def meta(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "robot": self.robot,
            "kind": self.kind,
            "rate_hz": self.rate_hz,
            "created": self.created,
            "world": self.world,
            "info": self.info,
            "units": {
                "time": "s",
                "joints": "rad or m",
                "ctrl": "actuator units",
                "base_pos": "m",
                "base_quat": "w x y z",
            },
        }

    def to_json(self) -> dict[str, Any]:
        def rows(a: np.ndarray, digits: int = 5) -> list[list[float | None]]:
            # a reading that was missing in a frame is null: a bare NaN is not JSON and strict parsers refuse it
            return [[round(float(v), digits) if math.isfinite(v) else None for v in row] for row in a]

        return {
            **self.meta(),
            "joint_names": self.joint_names,
            "actuator_names": self.actuator_names,
            "sensor_names": self.sensor_names,
            "time": [round(float(t), 4) for t in self.time],
            "joints": rows(self.joints),
            "ctrl": rows(self.ctrl),
            "base_pos": rows(self.base_pos),
            "base_quat": rows(self.base_quat, 6),
            "sensors": rows(self.sensors),
            "commands": self.commands,
        }

    def save(self, path: Path) -> Path:
        """Write it as .json or .npz (by the suffix)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".json":
            text = json.dumps(self.to_json(), separators=(",", ":"), allow_nan=False)  # whole, before a file is touched
            with path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        elif suffix == ".npz":
            np.savez_compressed(
                path,
                time=self.time,
                joint_pos=self.joints,
                joint_names=np.array(self.joint_names, dtype=str),
                ctrl=self.ctrl,
                ctrl_names=np.array(self.actuator_names, dtype=str),
                base_pos=self.base_pos,
                base_quat=self.base_quat,
                sensors=self.sensors,
                sensor_names=np.array(self.sensor_names, dtype=str),
                command_t=np.array([c["t"] for c in self.commands], dtype=float),
                command_op=np.array([c["op"] for c in self.commands], dtype=str),
                commands=np.array(json.dumps(self.commands)),
                meta=np.array(json.dumps(self.meta())),
            )
        else:
            raise DemonstrationError(
                f"a demonstration is saved as {' or '.join(SUFFIXES)}, not {path.suffix or path.name}"
            )
        return path

    @classmethod
    def load(cls, path: Path) -> Demonstration:
        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                raw = json.loads(path.read_text(encoding="utf-8"))
                arrays = {k: raw[k] for k in ("time", "joints", "ctrl", "base_pos", "base_quat", "sensors")}
                names = (raw["joint_names"], raw["actuator_names"], raw["sensor_names"])
                meta, commands = raw, raw["commands"]
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as z:
                    meta = json.loads(str(z["meta"]))
                    arrays = {
                        "time": z["time"],
                        "joints": z["joint_pos"],
                        "ctrl": z["ctrl"],
                        "base_pos": z["base_pos"],
                        "base_quat": z["base_quat"],
                        "sensors": z["sensors"],
                    }
                    names = (
                        [str(n) for n in z["joint_names"]],
                        [str(n) for n in z["ctrl_names"]],
                        [str(n) for n in z["sensor_names"]],
                    )
                    commands = json.loads(str(z["commands"]))
            else:
                raise DemonstrationError(f"{path.name}: a demonstration is a {' or '.join(SUFFIXES)} file")
        except (OSError, KeyError, ValueError) as exc:
            if isinstance(exc, DemonstrationError):
                raise
            raise DemonstrationError(f"{path.name} is not a demonstration file ({type(exc).__name__}: {exc})") from exc
        if meta.get("format") != FORMAT:
            raise DemonstrationError(f"{path.name} is not a robotforge demonstration")
        count = len(arrays["time"])

        def table(key: str, width: int) -> np.ndarray:
            a = np.asarray(arrays[key], dtype=float)
            return a.reshape(count, width) if a.size or count == 0 else np.zeros((count, width))

        return cls(
            name=str(meta["name"]),
            robot=str(meta["robot"]),
            kind=str(meta["kind"]),
            rate_hz=float(meta["rate_hz"]),
            created=str(meta.get("created", "")),
            world=str(meta.get("world", "")),
            joint_names=list(names[0]),
            actuator_names=list(names[1]),
            sensor_names=list(names[2]),
            time=np.asarray(arrays["time"], dtype=float),
            joints=table("joints", len(names[0])),
            ctrl=table("ctrl", len(names[1])),
            base_pos=table("base_pos", 3),
            base_quat=table("base_quat", 4),
            sensors=table("sensors", len(names[2])),
            commands=list(commands),
            info=dict(meta.get("info", {})),
        )


def flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    """The numbers of a sensor's reading by dotted name: imu.orientation_deg.roll for {"orientation_deg": {...}}"""
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if math.isfinite(value):
            out[prefix] = float(value)
    elif isinstance(value, dict):
        for key, inner in value.items():
            if key not in IGNORED_KEYS:
                flatten(f"{prefix}.{key}", inner, out)
    elif isinstance(value, list | tuple) and len(value) <= SENSOR_LEAVES:
        for i, inner in enumerate(value):
            flatten(f"{prefix}.{i}", inner, out)
