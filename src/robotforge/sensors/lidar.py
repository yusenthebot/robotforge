"""Lidars: one ring of laser rays (2D) or several rings over a vertical field of view (3D), cast together.

A scan is one call to mujoco.mj_multiRay from the sensor's origin, which tests every shape once for all the
rays. Rays see the shapes things collide with, not the visual meshes drawn over them: that is what a robot
can bump into, and it is about ten times cheaper (a Go2's and the warehouse's meshes have 280 thousand
triangles; their collision shapes are a few hundred boxes and cylinders). Hits on the robot itself are
filtered out as a real lidar's self mask does, and hits on the floor, terrain grids and flat ground markers
count as ground, not as obstacles.

A reading is small: the nearest obstacle with its bearing (0 is the sensor's forward, counter-clockwise
positive, so +90 is its left), the nearest one within 15 degrees of straight ahead, and the ranges reduced to
the nearest per 10 degree sector (36 numbers). A 3D scan also gives its point count and bounding box.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor
from robotforge.sim import lidar as sim_lidar

SECTORS = 36  # numbers a ring is reduced to in a reading
AHEAD_DEG = 15.0  # "ahead" is within this many degrees of the sensor's forward
VIEW_POINTS = 1200  # hit points sent to the viewer per lidar, at most
VISUAL_GROUP = 2
# Every geom group but the visual meshes' (see the module notes).
RAY_GROUPS = np.array([1 if g != VISUAL_GROUP else 0 for g in range(6)], dtype=np.uint8)
GROUND_TYPES = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))


class Lidar(Sensor):
    kind = "lidar"

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        d = decl.data
        self.channels = int(d.get("channels", 1))
        self.per_ring = int(d.get("rays", 360))
        self.range = float(d.get("range", 12.0))
        self.hfov = float(d.get("hfov", 360.0))
        low, high = d.get("vfov", [0.0, 0.0])
        self.elevations = np.radians(np.linspace(low, high, self.channels) if self.channels > 1 else [(low + high) / 2])
        if self.hfov >= 360.0:
            az = np.arange(self.per_ring) * (2 * math.pi / self.per_ring)
        else:
            half = math.radians(self.hfov) / 2
            az = np.linspace(-half, half, self.per_ring)
        el, az = np.meshgrid(self.elevations, az, indexing="ij")  # ring by ring
        self.azimuth = az.ravel()
        self.directions = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1).reshape(
            -1, 3
        )
        self.rays = len(self.directions)
        root = int(model.body_rootid[self.body])
        self.own = model.body_rootid[model.geom_bodyid] == root  # the robot's own shapes
        not_colliding = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
        self.ground = np.isin(model.geom_type, GROUND_TYPES) | not_colliding
        self.three_d = self.channels > 1
        self._geom = np.zeros(self.rays, np.int32)
        self._dist = np.zeros(self.rays)
        self.native: list[int] = []  # a built robot's lidar: the MuJoCo rangefinders that are its rays
        if d.get("rays_native"):
            self._ring_of_rangefinders(model, decl, d["rays_native"])

    def _ring_of_rangefinders(self, model: mujoco.MjModel, decl: declare.Declared, rays: list[str]) -> None:
        """A built robot's lidar: a ring of MuJoCo rangefinders, each on its own site looking along its +z. They
        are read where MuJoCo put them (it casts them on every step), past the robot's own parts, in the frame of the
        body they are on."""
        sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, decl.full(r)) for r in rays]
        if min(sites) < 0:
            raise ValueError(f"the lidar's rays {rays[0]}... are missing from the model")
        self.native = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, decl.full(r)) for r in rays]
        if min(self.native) < 0:
            raise ValueError(f"the lidar's rangefinders {rays[0]}... are missing from the model")
        self.site = sites[0]
        directions = []
        for site in sites:
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.site_quat[site])
            directions.append(rot.reshape(3, 3)[:, 2])
        self.directions = np.array(directions)
        self.azimuth = np.arctan2(self.directions[:, 1], self.directions[:, 0]) % (2 * math.pi)
        self.channels, self.per_ring, self.rays = 1, len(rays), len(rays)
        self.elevations = np.zeros(1)
        self.three_d = False

    # ---- measuring ---------------------------------------------------------------------------------------

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        origin = np.array(data.site_xpos[self.site])
        if self.native:  # MuJoCo has cast them; read past the robot's own parts as observe's lidars are (sim/lidar)
            dist = np.array([sim_lidar.seen(model, data, s) for s in self.native])
            reach = model.sensor_cutoff[self.native]  # MuJoCo gives a hit beyond the cutoff as the cutoff: none
            dist[(dist < 0) | ((reach > 0) & (dist >= reach))] = np.nan
            rot = data.xmat[self.body].reshape(3, 3).copy()
            return {"origin": origin, "rot": rot, "dist": dist, "ground": np.zeros(self.rays, bool)}
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
            None,
            self.rays,
            self.range,
        )
        geom = self._geom.copy()
        dist = np.where(geom >= 0, self._dist, np.nan)
        hit = geom >= 0
        own = np.zeros(self.rays, bool)
        own[hit] = self.own[geom[hit]]
        dist[own] = np.nan  # the robot's own legs and body: masked as a real lidar's self filter does
        ground = np.zeros(self.rays, bool)
        ground[hit] = self.ground[geom[hit]]
        # miss: rays that met nothing within range (not those the self filter masked), for mapping the open space
        return {"origin": origin, "rot": rot, "dist": dist, "ground": ground & ~own, "miss": ~hit}

    def corrupt(self, value: dict[str, Any], dt: float) -> dict[str, Any]:
        k = self.knobs
        if not (k.noise or k.dropout):
            return value
        dist = value["dist"].copy()
        if k.noise.get("range"):
            dist = np.clip(dist + self.rng.normal(0.0, k.noise["range"], self.rays), 0.0, self.range)
        if k.dropout:
            dist[self.rng.random(self.rays) < k.dropout] = np.nan
        return {**value, "dist": dist}

    # ---- reading -----------------------------------------------------------------------------------------

    def points(self, value: dict[str, Any]) -> np.ndarray:
        """World points of the rays that hit something (Nx3), in ray order."""
        hit = np.isfinite(value["dist"])
        return value["origin"] + (self.directions[hit] @ value["rot"].T) * value["dist"][hit, None]

    def cloud(self, value: dict[str, Any], limit: int) -> dict[str, Any]:
        """The scan's hit points in the world, thinned to `limit` (perception/cloud.py)."""
        from robotforge.perception import cloud

        return cloud.summary(self.points(value), limit)

    def _obstacles(self, value: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Horizontal distance, bearing (rad, in the sensor's frame) and ray index of every non-ground hit."""
        keep = np.isfinite(value["dist"]) & ~value["ground"]
        index = np.flatnonzero(keep)
        horizontal = value["dist"][keep] * np.cos(self.elevations.repeat(self.per_ring)[keep])
        return horizontal, self.azimuth[keep], index

    def _sectors(self, horizontal: np.ndarray, bearing: np.ndarray) -> tuple[list[float | None], list[float]]:
        """The nearest obstacle per sector, and the bearings (degrees) [first sector's middle, step]."""
        count = min(SECTORS, self.per_ring)
        if self.hfov >= 360.0:
            step = 360.0 / count
            k = np.floor((np.degrees(bearing) + step / 2) % 360.0 / step).astype(int) % count
            first = 0.0
        else:
            step = self.hfov / count
            k = np.clip(np.floor((np.degrees(bearing) + self.hfov / 2) / step).astype(int), 0, count - 1)
            first = -self.hfov / 2 + step / 2
        out: list[float | None] = [None] * count
        for sector in range(count):
            here = horizontal[k == sector]
            if here.size:
                out[sector] = round(float(here.min()), 3)
        return out, [round(first, 2), round(step, 2)]

    def _nearest(self, value: dict[str, Any], horizontal, bearing, index, ahead: bool = False) -> dict[str, Any] | None:
        deg = (np.degrees(bearing) + 180.0) % 360.0 - 180.0
        pick = np.abs(deg) <= AHEAD_DEG if ahead else np.ones(len(deg), bool)
        if not pick.any():
            return None
        i = int(np.argmin(np.where(pick, horizontal, np.inf)))
        out: dict[str, Any] = {"distance_m": round(self._settled(horizontal, deg, pick, i), 3)}
        out["bearing_deg"] = round(float(deg[i]), 1)
        if self.three_d:
            ray = int(index[i])
            point = value["origin"] + value["rot"] @ self.directions[ray] * value["dist"][ray]
            out["height_m"] = round(float(point[2]), 3)
            out["point_m"] = [round(float(v), 3) for v in point]
        return out

    def _settled(self, horizontal: np.ndarray, deg: np.ndarray, pick: np.ndarray, i: int) -> float:
        """The distance to the nearest obstacle. With range noise the single nearest ray is the one whose noise
        came out shortest (2 to 3 noise widths short against a wall of hundreds of rays), so a noisy lidar reports
        the median of the rays around it: within a few degrees, where a surface is as far, and within four noise
        widths of the nearest."""
        sigma = self.knobs.noise.get("range", 0.0)
        if not sigma:
            return float(horizontal[i])
        step = 360.0 / self.per_ring if self.hfov >= 360.0 else self.hfov / max(self.per_ring - 1, 1)
        near = (np.abs((deg - deg[i] + 180.0) % 360.0 - 180.0) <= max(3.0, 1.5 * step)) & pick
        near &= horizontal <= horizontal[i] + 4 * sigma
        return float(np.median(horizontal[near]))

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        horizontal, bearing, index = self._obstacles(value)
        sectors, bearings = self._sectors(horizontal, bearing)
        hits = int(np.isfinite(value["dist"]).sum())
        out: dict[str, Any] = {"rays": self.rays}
        if self.three_d:
            pts = self.points(value)
            out["channels"] = self.channels
            out["points"] = hits
            out["ground_points"] = int((np.isfinite(value["dist"]) & value["ground"]).sum())
            if len(pts):
                out["box_m"] = [
                    [round(float(v), 2) for v in pts.min(axis=0)],
                    [round(float(v), 2) for v in pts.max(axis=0)],
                ]
        else:
            out["hits"] = hits
        out["range_m"] = self.range
        out["nearest"] = self._nearest(value, horizontal, bearing, index)
        out["ahead"] = self._nearest(value, horizontal, bearing, index, ahead=True)
        out["sectors_m"] = sectors
        out["sector_bearings_deg"] = bearings
        if full:
            rings = value["dist"].reshape(self.channels, self.per_ring)
            out["ranges_m"] = [[None if not np.isfinite(v) else round(float(v), 4) for v in ring] for ring in rings]
            out["azimuth_deg"] = [round(float(a), 3) for a in np.degrees(self.azimuth[: self.per_ring])]
            out["elevation_deg"] = [round(float(e), 3) for e in np.degrees(self.elevations)]
            if self.three_d:
                out["points_m"] = [[round(float(v), 4) for v in p] for p in self.points(value)]
        return out

    def line(self, value: dict[str, Any]) -> str:
        horizontal, bearing, index = self._obstacles(value)
        what = f"3D lidar, {self.channels} rings" if self.three_d else "lidar"
        hits = int(np.isfinite(value["dist"]).sum())
        near = self._nearest(value, horizontal, bearing, index)
        nearest = f", nearest {near['distance_m']:.2f} m at {near['bearing_deg']:.0f} deg" if near else ", no obstacle"
        return f"{what}: {hits} of {self.rays} rays hit{nearest}"

    def viewer(
        self, value: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData, points: int = VIEW_POINTS
    ) -> dict[str, Any]:
        """`points`: the most hit points sent, spread evenly over the scan (the hub shares a total between lidars)."""
        pts = self.points(value)
        if len(pts) > points:
            pts = pts[np.linspace(0, len(pts) - 1, points).astype(int)]
        return {
            "origin": [round(float(v), 4) for v in value["origin"]],
            "points": [round(float(v), 3) for v in pts.ravel()],
            "rays": self.rays,
            "hits": int(np.isfinite(value["dist"]).sum()),
            "range_m": self.range,
            "channels": self.channels,
        }
