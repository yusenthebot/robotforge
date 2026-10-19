"""An ultrasonic range sensor: a cone of sound, not a line.

A sonar pings and listens for the first echo. Its beam is wide (30 degrees across by default), so it hears the
nearest thing anywhere in the cone, not what is straight ahead; it is deaf to anything nearer than its blind
zone (the transducer is still ringing); and a flat surface turned far from square to the beam sends the sound
away from the sensor, so a wall seen at a grazing angle is not heard at all. The model casts a cone of 19 rays
(one on the axis and rings of 6 and 12), keeps the echoes that arrive within MAX_INCIDENCE degrees of square and
reports the nearest of them. Like a lidar it hears the shapes things collide with, not their visual meshes, and
not the robot it rides on.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor

MAX_INCIDENCE = 60.0  # degrees from square: a surface turned further than this sends its echo away
RINGS = ((0.5, 6), (1.0, 12))  # (share of the half angle, rays) round the axis
VISUAL_GROUP = 2
RAY_GROUPS = np.array([1 if g != VISUAL_GROUP else 0 for g in range(6)], dtype=np.uint8)


class Sonar(Sensor):
    kind = "sonar"

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        d = decl.data
        self.range = float(d["range"])
        self.beam = float(d["beam"])
        self.min_range = float(d.get("min_range") or 0.0)
        half = math.radians(self.beam) / 2
        cone = [(0.0, 0.0)]
        for share, count in RINGS:
            cone += [(share * half, 2 * math.pi * k / count) for k in range(count)]
        theta, phi = np.array(cone).T
        self.directions = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
        self.rays = len(self.directions)
        root = int(model.body_rootid[self.body])
        self.own = model.body_rootid[model.geom_bodyid] == root
        self._geom = np.zeros(self.rays, np.int32)
        self._dist = np.zeros(self.rays)
        self._normal = np.zeros(3 * self.rays)

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        origin = np.array(data.site_xpos[self.site])
        rot = data.site_xmat[self.site].reshape(3, 3).copy()
        vec = np.ascontiguousarray(self.directions @ rot.T)
        mujoco.mj_multiRay(
            model,
            data,
            origin,
            vec.ravel(),
            RAY_GROUPS,
            1,
            self.body,
            self._geom,
            self._dist,
            self._normal,
            self.rays,
            self.range,
        )
        geom = self._geom.copy()
        hit = (geom >= 0) & ~self.own[np.maximum(geom, 0)]
        dist = np.where(hit, self._dist, np.nan)
        square = -(vec * self._normal.reshape(-1, 3)).sum(axis=1)  # cosine of the angle off square
        echo = hit & (square >= math.cos(math.radians(MAX_INCIDENCE)))
        return {"origin": origin, "rot": rot, "dist": dist, "echo": echo, "hits": int(hit.sum())}

    def corrupt(self, value: dict[str, Any], dt: float) -> dict[str, Any]:
        k = self.knobs
        if not (k.noise or k.dropout):
            return value
        dist = value["dist"].copy()
        if k.noise.get("range"):
            dist = np.clip(dist + self.rng.normal(0.0, k.noise["range"], self.rays), 0.0, self.range)
        echo = value["echo"].copy()
        if k.dropout:
            echo &= self.rng.random(self.rays) >= k.dropout
        return {**value, "dist": dist, "echo": echo}

    def _heard(self, value: dict[str, Any]) -> tuple[float | None, bool]:
        """The nearest echo's distance (None when there is none) and whether something is in the blind zone."""
        dist = value["dist"][value["echo"]]
        seen = dist[np.isfinite(dist)]
        if not seen.size:
            return None, False
        nearest = float(seen.min())
        return (None, True) if nearest < self.min_range else (nearest, False)

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        nearest, blind = self._heard(value)
        out: dict[str, Any] = {"distance_m": None if nearest is None else round(nearest, 3)}
        out["echoes"] = int(value["echo"].sum())
        if blind:
            out["note"] = f"something is nearer than {self.min_range:g} m: the sonar is deaf that close"
        elif nearest is None and value["hits"]:
            out["note"] = "surfaces in the cone are turned too far from square to the beam: no echo"
        elif nearest is None:
            out["note"] = f"nothing within {self.range:g} m"
        return out

    def line(self, value: dict[str, Any]) -> str:
        nearest, blind = self._heard(value)
        if nearest is not None:
            return f"sonar: {nearest:.2f} m"
        return "sonar: too close to hear" if blind else f"sonar: nothing within {self.range:g} m"

    def viewer(self, value: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        nearest, _ = self._heard(value)
        rot = data.site_xmat[self.site].reshape(3, 3)
        return {
            "direction": [round(float(v), 4) for v in rot[:, 2]],
            "distance_m": None if nearest is None else round(nearest, 3),
            "range_m": self.range,
            "beam_deg": self.beam,
        }
