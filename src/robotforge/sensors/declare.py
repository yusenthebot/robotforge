"""Sensor definitions kept in the model itself.

A robot's sensors are declared as custom texts named `rf_sensor:<name>` whose data is the definition as JSON:
the sensor's type, the model objects it reads (its site, camera, MuJoCo sensors, joints) and its resolved
options and realism knobs. MjSpec.attach prefixes a text's name with the robot's prefix like every other
name, so a definition written into a robot's own spec arrives in the world as `dog/rf_sensor:lidar` and its
object names resolve with the same `dog/` prefix. The world compiler, the builder and the templates write
definitions this way; the runtime reads them back from the compiled model alone, so a lesson's model, a model
saved as MJCF or a robot mounted on another robot all carry their sensors with them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import mujoco

TEXT_PREFIX = "rf_sensor:"
VERSION = 1


def declare(spec: mujoco.MjSpec, definition: dict[str, Any]) -> None:
    """Write a sensor's definition into `spec` (names inside it are the spec's own, unprefixed)."""
    body = {"v": VERSION, **definition}
    spec.add_text(name=f"{TEXT_PREFIX}{definition['name']}", data=json.dumps(body, separators=(",", ":")))


def declared_names(spec: mujoco.MjSpec) -> set[str]:
    """Names of the sensors `spec` declares at its own level (not those of robots mounted on it)."""
    return {t.name[len(TEXT_PREFIX) :] for t in spec.texts if t.name.startswith(TEXT_PREFIX)}


@dataclass(frozen=True)
class Declared:
    """A definition as the compiled model holds it."""

    prefix: str  # "dog/" (or "rover/arm/" for an arm mounted on a rover); "" for a model on its own
    data: dict[str, Any]

    @property
    def name(self) -> str:
        return f"{self.prefix}{self.data['name']}"

    @property
    def robot(self) -> str:
        return self.prefix.split("/", 1)[0] if self.prefix else ""

    def full(self, local: str) -> str:
        """A model object's name as the world knows it."""
        return f"{self.prefix}{local}"


def read(model: mujoco.MjModel) -> list[Declared]:
    """Every sensor definition in a compiled model, in the order they were declared."""
    out = []
    for i in range(model.ntext):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, i) or ""
        at = name.rfind(TEXT_PREFIX)
        if at < 0:
            continue
        start, size = int(model.text_adr[i]), int(model.text_size[i])
        raw = bytes(model.text_data[start : start + size]).rstrip(b"\0").decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue  # not ours after all: someone else's text that happens to have the name
        if isinstance(data, dict) and data.get("v") == VERSION and "type" in data and "name" in data:
            out.append(Declared(name[:at], data))
    return out
