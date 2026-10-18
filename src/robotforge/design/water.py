"""A world's water in world.yaml, and putting what floats on top of it when the world is compiled.

`water: {level: 1.2}` fills the world with water up to 1.2 m above the floor: the floor becomes the bottom of a
lake, a harbor or a pool, and terrain or furniture that rises above the level is dry land, a quay or a pool deck.
Waves, a current and the water's density are optional; `area` keeps the water to a rectangle (a pool in a hall,
a lake in a park) where it would otherwise reach everywhere. The physics is in sim/water.py.

Robots and objects placed without a height over water start where they float (a boat sits at its waterline,
a crate a little deeper, a buoy upright), or on the bottom if they sink (a steel block).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robotforge.sim import water as physics

MAX_CURRENT = 3.0  # m/s; a river in flood
FRESH, SEA = 1000.0, 1025.0


class WaterEntry(BaseModel):
    """The `water:` of a world.yaml."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    level: float = Field(gt=0, le=20, description="height of the still surface above the floor, m")
    waves: float = Field(0.0, ge=0, le=0.5, description="from the still level to a crest, m (0 for calm)")
    wavelength: float = Field(4.0, ge=0.3, le=60, description="crest to crest, m")
    heading: float = Field(30.0, description="the way the waves travel, degrees counter-clockwise from +x")
    current: list[float] = Field(default_factory=lambda: [0.0, 0.0], description="[x, y] m/s")
    density: float = Field(FRESH, ge=500, le=1500, description="kg/m^3: 1000 fresh water, 1025 the sea")
    area: list[float] | None = Field(None, description="[x0, y0, x1, y1] where there is water; left out: everywhere")

    @model_validator(mode="before")
    @classmethod
    def _number(cls, data: Any) -> Any:
        """A bare number is the level: `water: 1.2`."""
        if isinstance(data, bool):
            raise ValueError("water is a level in meters, e.g. 1.2, or {level: 1.2, waves: 0.03}")
        return {"level": data} if isinstance(data, int | float) else data

    @field_validator("current")
    @classmethod
    def _current(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError(f"current is [x, y] in m/s, got {v}")
        if math.hypot(*v) > MAX_CURRENT:
            raise ValueError(f"current {v} is faster than {MAX_CURRENT:g} m/s; it is in meters per second")
        return v

    @field_validator("area")
    @classmethod
    def _area(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and (len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]):
            raise ValueError(f"area is [x0, y0, x1, y1] with x0 < x1 and y0 < y1, in meters; got {v}")
        return v

    def physics(self) -> physics.Water:
        area = tuple(self.area) if self.area is not None else None
        return physics.Water(
            self.level, self.density, self.waves, self.wavelength, self.heading, tuple(self.current), area
        )

    def covers(self, x: float, y: float) -> bool:
        if self.area is None:
            return True
        x0, y0, x1, y1 = self.area
        return x0 <= x <= x1 and y0 <= y <= y1


def describe(entry: WaterEntry) -> str:
    """One line for the world's notes."""
    where = (
        "everywhere"
        if entry.area is None
        else f"over x {entry.area[0]:g}..{entry.area[2]:g}, y {entry.area[1]:g}..{entry.area[3]:g}"
    )
    waves = f", waves {entry.waves * 100:.0f} cm" if entry.waves else ", calm"
    return f"water up to {entry.level:g} m {where}{waves}: what is lower is under water"


Placer = Callable[[float], None]  # raises a robot's frame or an object's body by dz


def float_up(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    qpos: np.ndarray,
    entry: WaterEntry,
    candidates: list[tuple[str, Placer]],
    ground_height: Callable[[float, float], float],
) -> list[str]:
    """Raise (or lower) each candidate whose spot is under water to where it floats. `candidates` are the root
    bodies placed without a height, by name, each with a function that moves it. Returns the names moved."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    roots: dict[int, tuple[str, Placer]] = {}
    for name, place in candidates:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            continue
        x, y = float(data.xpos[b][0]), float(data.xpos[b][1])
        if entry.covers(x, y) and ground_height(x, y) < entry.level:
            roots[b] = (name, place)
    if not roots:
        return []
    shifts = physics.float_height(model, data, entry.physics(), list(roots))
    moved = []
    for b, dz in shifts.items():
        if abs(dz) > 5e-4:
            name, place = roots[b]
            place(dz)
            moved.append(name)
    return moved


def robot_root(model: mujoco.MjModel, robot: str) -> str | None:
    """The name of a robot's root body (the one on the world body), as attached with its prefix."""
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if model.body_parentid[b] == 0 and name.startswith(f"{robot}/"):
            return name
    return None
