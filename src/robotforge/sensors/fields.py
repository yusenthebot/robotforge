"""Sensors of quantities that spread through the scene from sources: heat and gas.

A source is a prop in the world (design/props.py: `heat_source`, `gas_source`, put there with add_object and a
strength) that the compiled model remembers as a custom text `rf_source:<body>` holding its kind and strength.
A temperature or gas sensor reads the field of every source of its kind at the place it sits:

- heat: the room's 20 degrees plus, for each source, its strength (the rise at the source itself, in degrees)
  over 1 + (d / 0.35 m)^2: warm within a meter or two, a fraction of a degree across the room;
- gas: each source's strength (ppm at the source) times exp(-d / 1.2 m): a plume that thins over a few meters.

Something solid between a source and the sensor shields it: heat falls to a fifth, gas to a half (it drifts
round). The field is a model of what a sensor would read, not a simulation of air: it does not move with wind,
and the sources do not run out. The sensor itself is slow, as real ones are: `response_s` is the time constant
(seconds) with which its reading follows the field; none means it reads the field as it is.
"""

from __future__ import annotations

import json
import math
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor

TEXT_PREFIX = "rf_source:"
AMBIENT_C = 20.0
HEAT_CORE_M = 0.35  # m: where a heat source's rise has fallen to a quarter
GAS_REACH_M = 1.2  # m: where a gas source's concentration has fallen to 1/e
SHIELD = {"heat": 0.2, "gas": 0.5}  # what gets past a solid thing between source and sensor
KINDS = {"temperature": "heat", "gas": "gas"}
UNITS = {"temperature": "deg C", "gas": "ppm"}
VISUAL_GROUP = 2
RAY_GROUPS = np.array([1 if g != VISUAL_GROUP else 0 for g in range(6)], dtype=np.uint8)
BLOCK_SLACK = 0.12  # m short of the sensor within which a hit is the sensor's own surroundings, not a shield


def declare_source(spec: mujoco.MjSpec, name: str, kind: str, strength: float, lift: float = 0.0) -> None:
    """Write a source's definition into the world's spec (named for the body that stands for it); `lift` is how far
    above the body's base the source itself is (the heater's element)."""
    spec.add_text(name=f"{TEXT_PREFIX}{name}", data=json.dumps({"kind": kind, "strength": strength, "lift": lift}))


def read_sources(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """Every source the model declares: {"name", "body", "kind", "strength"}."""
    out = []
    for i in range(model.ntext):
        text = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, i) or ""
        if not text.startswith(TEXT_PREFIX):
            continue
        start, size = int(model.text_adr[i]), int(model.text_size[i])
        try:
            data = json.loads(bytes(model.text_data[start : start + size]).rstrip(b"\0").decode("utf-8"))
        except json.JSONDecodeError:
            continue
        name = text[len(TEXT_PREFIX) :]
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body >= 0 and data.get("kind") in ("heat", "gas"):
            lift = float(data.get("lift", 0.0))
            out.append(
                {"name": name, "body": body, "kind": data["kind"], "strength": float(data["strength"]), "lift": lift}
            )
    return out


class Field(Sensor):
    """A temperature or gas sensor (see the module notes)."""

    kind = "temperature"
    default_rate = 20.0

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        self.response = float(decl.data.get("response_s") or 0.0)
        self.integrates = self.response > 0  # a slow sensor follows the field step by step: it is never read lazily
        super().__init__(model, decl)
        self.quantity = self.type
        self.field = KINDS[self.quantity]
        self.sources = [s for s in read_sources(model) if s["kind"] == self.field]
        self.state: float | None = None

    def carry(self) -> dict[str, Any]:
        return {**super().carry(), "state": self.state}

    def resume(self, state: dict[str, Any]) -> None:
        super().resume(state)
        self.state = state["state"]

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        point = np.array(data.site_xpos[self.site])
        total = AMBIENT_C if self.field == "heat" else 0.0
        for source in self.sources:
            at = data.xpos[source["body"]] + np.array([0.0, 0.0, source["lift"]])
            gap = point - at
            d = float(np.linalg.norm(gap))
            reach = 1.0 / (1.0 + (d / HEAT_CORE_M) ** 2) if self.field == "heat" else math.exp(-d / GAS_REACH_M)
            if d > 0.2 and self._shielded(model, data, at, gap / d, d, source["body"]):
                reach *= SHIELD[self.field]
            total += source["strength"] * reach
        return total

    def _shielded(
        self, model: mujoco.MjModel, data: mujoco.MjData, at: np.ndarray, direction: np.ndarray, d: float, body: int
    ) -> bool:
        """Whether a solid shape stands between the source and the sensor (not the source's own body, nor what
        the sensor sits on within a hand of it)."""
        geom = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(model, data, at + direction * 0.05, direction, RAY_GROUPS, 1, body, geom)
        return 0 <= hit < d - 0.05 - BLOCK_SLACK and int(model.body_rootid[model.geom_bodyid[geom[0]]]) != int(
            model.body_rootid[self.body]
        )

    def corrupt(self, value: float, dt: float) -> float:
        if self.response > 0:
            if self.state is None:
                self.state = value
            else:
                self.state += (value - self.state) * (1.0 - math.exp(-dt / self.response))
            value = self.state
        return value + float(self.noise(self.channel))

    @property
    def channel(self) -> str:
        return "temperature" if self.field == "heat" else "concentration"

    def reading(self, value: float, full: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {("temperature_c" if self.field == "heat" else "gas_ppm"): round(float(value), 2)}
        if not self.sources:
            out["note"] = f"there is no {self.field} source in the world: add_object(shape='{self.field}_source')"
        return out

    def line(self, value: float) -> str:
        return f"{self.quantity}: {value:.1f} {UNITS[self.quantity]}"

    def viewer(self, value: float, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        level = (value - AMBIENT_C) / 60.0 if self.field == "heat" else math.log10(1.0 + max(value, 0.0)) / 3.0
        return {
            "value": round(float(value), 2),
            "unit": UNITS[self.quantity],
            "level": round(float(min(max(level, 0.0), 1.0)), 3),
            "field": self.field,
        }
