"""Cameras: a picture, a depth image or a segmentation image from a camera on the robot.

Pictures cost a render, so a camera is never drawn while time passes. Sampling a camera keeps the state of
the world at that moment (joint positions and mocap poses, a few hundred numbers), and reading it draws that
state on a scratch copy of the simulation: a camera with a rate or a latency shows the world as it was
when its frame was due, without rendering every frame nobody looks at.

A depth reading is the distance along the camera's view to what each pixel shows (MuJoCo's depth camera),
summarized as the distance at the center, the nearest and the median, and the nearest obstacle (the floor and
flat ground marks left out) with its bearing; a segmentation reading lists what the picture shows and how
much of it. Pictures come back only when asked for (image=true).
"""

from __future__ import annotations

import io
import math
import zlib
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor

SEEN_LIMIT = 6  # things a segmentation reading names, the biggest first
GROUND_TYPES = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))


class Camera(Sensor):
    kind = "camera"
    renders = True

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        d = decl.data
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, decl.full(d["camera"]))
        if self.camera < 0:
            raise ValueError(f"no camera {decl.full(d['camera'])!r}")
        self.view = decl.full(d["camera"])
        self.mode = d.get("mode", "rgb")
        self.width, self.height = int(d.get("width", 320)), int(d.get("height", 240))
        self.fov = float(d.get("fov", model.cam_fovy[self.camera]))
        self.range = float(d.get("range", 10.0))
        not_colliding = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
        self.ground = np.isin(model.geom_type, GROUND_TYPES) | (not_colliding & (model.geom_size[:, 2] < 0.01))

    def measure(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """The moment to draw: kept as the state, drawn when read (see the module notes)."""
        return {
            "qpos": data.qpos.copy(),
            "mocap_pos": data.mocap_pos.copy(),
            "mocap_quat": data.mocap_quat.copy(),
            "noise_seed": int(self.rng.integers(2**32)),  # the frame's noise is drawn when it is taken
        }

    def scene_data(self, model: mujoco.MjModel, moment: dict[str, np.ndarray]) -> mujoco.MjData:
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = moment["qpos"]
        scratch.mocap_pos[:] = moment["mocap_pos"]
        scratch.mocap_quat[:] = moment["mocap_quat"]
        mujoco.mj_forward(model, scratch)
        return scratch

    def develop(self, layers: Any, moment: dict[str, Any]) -> dict[str, Any]:
        """The camera's own reading of a rendered moment: its noise added, the sky beyond its range."""
        out: dict[str, Any] = {"layers": layers}
        rng = np.random.default_rng(moment["noise_seed"])
        if self.mode == "depth":
            depth = np.array(layers.depth, dtype=np.float64)
            far = depth >= min(self.range, layers.camera.far * 0.999)
            if self.knobs.noise.get("depth"):
                depth = depth + rng.normal(0.0, self.knobs.noise["depth"], depth.shape)
            depth[far] = np.nan
            out["depth"] = np.clip(depth, 0.0, self.range)
        elif self.mode == "rgb":
            rgb = layers.rgb
            if self.knobs.noise.get("rgb"):
                rgb = np.clip(rgb + rng.normal(0.0, self.knobs.noise["rgb"], rgb.shape), 0, 255).astype(np.uint8)
            out["rgb"] = rgb
        return out

    # ---- reading -----------------------------------------------------------------------------------------

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "size": [self.width, self.height], "fov_deg": round(self.fov, 1)}
        if value.get("note"):
            return {**out, "note": value["note"]}
        if self.mode == "depth":
            out.update(self._depth_summary(value))
        elif self.mode == "segmentation":
            out["seen"] = self._seen(value["layers"])
        return out

    def _depth_summary(self, value: dict[str, Any]) -> dict[str, Any]:
        depth = value["depth"]
        layers = value["layers"]
        h, w = depth.shape
        valid = np.isfinite(depth)
        out: dict[str, Any] = {}
        center = depth[h // 2, w // 2]
        out["center_m"] = round(float(center), 3) if np.isfinite(center) else None
        if valid.any():
            out["min_m"] = round(float(np.nanmin(depth)), 3)
            out["median_m"] = round(float(np.nanmedian(depth)), 3)
        out["beyond_range"] = round(float(1 - valid.mean()), 3)
        geoms = layers.geoms
        if geoms is not None:
            ground = np.zeros(depth.shape, bool)
            seen = geoms >= 0
            ground[seen] = self.ground[geoms[seen]]
            obstacle = valid & ~ground & seen
            if obstacle.any():
                masked = np.where(obstacle, depth, np.inf)
                v, u = np.unravel_index(int(np.argmin(masked)), depth.shape)
                cam = layers.camera
                ray = cam.ray(u + 0.5, v + 0.5)
                local = np.array([ray @ cam.right, ray @ cam.up, ray @ cam.forward])
                out["nearest"] = {
                    "distance_m": round(float(depth[v, u]), 3),
                    "bearing_deg": round(-math.degrees(math.atan2(local[0], local[2])), 1),
                    "elevation_deg": round(math.degrees(math.atan2(local[1], math.hypot(local[0], local[2]))), 1),
                }
        return out

    def _seen(self, layers: Any) -> list[dict[str, Any]]:
        geoms = layers.geoms
        if geoms is None:
            return []
        model = layers.model
        ids, counts = np.unique(geoms[geoms >= 0], return_counts=True)
        things: dict[str, int] = {}
        for g, n in zip(ids.tolist(), counts.tolist(), strict=True):
            name = thing_name(model, g)
            things[name] = things.get(name, 0) + n
        total = geoms.size
        ranked = sorted(things.items(), key=lambda kv: -kv[1])[:SEEN_LIMIT]
        return [{"name": name, "share": round(n / total, 3)} for name, n in ranked]

    def line(self, value: dict[str, Any]) -> str:
        what = {"rgb": "camera", "depth": "depth camera", "segmentation": "segmentation camera"}[self.mode]
        return f"{what}: {self.width}x{self.height}, {self.fov:.0f} deg; read_sensors(image=true) shows its picture"

    def viewer(self, value: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        rot = data.cam_xmat[self.camera].reshape(3, 3)
        return {
            "mode": self.mode,
            "fov_deg": round(self.fov, 1),
            "size": [self.width, self.height],
            "forward": [round(float(v), 4) for v in -rot[:, 2]],
            "up": [round(float(v), 4) for v in rot[:, 1]],
        }

    def pose(self, data: mujoco.MjData) -> dict[str, list[float]]:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.cam_xmat[self.camera])
        return {
            "pos": [round(float(v), 4) for v in data.cam_xpos[self.camera]],
            "quat": [round(float(v), 4) for v in quat],
        }

    def png(self, value: dict[str, Any]) -> bytes | None:
        """The picture as this camera's mode shows it."""
        layers = value.get("layers")
        if layers is None:
            return None
        if self.mode == "rgb":
            picture = value["rgb"]
        elif self.mode == "depth":
            picture = depth_picture(value["depth"], self.range)
        else:
            picture = segmentation_picture(layers.geoms, layers.model)
        buf = io.BytesIO()
        Image.fromarray(picture).save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def thing_name(model: mujoco.MjModel, geom: int) -> str:
    """What a geom belongs to as the world names it: a robot, a loose object, or a fixed one."""
    body = int(model.geom_bodyid[geom])
    root = int(model.body_rootid[body])
    if root == 0:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "floor"
        return name.split("/")[0]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or f"body{root}"
    return name.split("/")[0]


def depth_picture(depth: np.ndarray, far: float) -> np.ndarray:
    """Near is bright and warm, far is dark and blue, nothing within range is black."""
    t = np.clip(np.nan_to_num(depth, nan=far) / max(far, 1e-6), 0.0, 1.0)
    near = 1.0 - t
    rgb = np.stack([near**0.8, near**1.6 * 0.85, 0.25 + 0.55 * t * (1 - t) * 4 * 0.5 + 0.1 * near], axis=-1)
    rgb[~np.isfinite(depth)] = 0.0
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def segmentation_picture(geoms: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """One flat color per thing (a robot, an object), the same color on every read."""
    out = np.zeros((*geoms.shape, 3), np.uint8)
    for g in np.unique(geoms[geoms >= 0]).tolist():
        code = zlib.crc32(thing_name(model, g).encode("utf-8"))
        hue = (code % 360) / 360.0
        out[geoms == g] = _hsv(hue, 0.55 + (code >> 9) % 40 / 100, 0.75 + (code >> 17) % 25 / 100)
    return out


def _hsv(h: float, s: float, v: float) -> list[int]:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return [int(r * 255), int(g * 255), int(b * 255)]
