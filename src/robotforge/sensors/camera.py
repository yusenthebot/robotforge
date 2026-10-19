"""Cameras: a picture, a depth image or a segmentation image from a camera on the robot.

Pictures cost a render, so a camera is never drawn while time passes. Sampling a camera keeps the state of
the world at that moment (joint positions and mocap poses, a few hundred numbers), and reading it draws that
state on a scratch copy of the simulation: a camera with a rate or a latency shows the world as it was
when its frame was due, without rendering every frame nobody looks at.

A depth image holds each pixel's depth: how far what it shows is along the camera's axis (MuJoCo's depth
camera). A reading summarizes it as the depth at the center, the smallest and the median, and gives the nearest
obstacle as a distance: straight from the lens to the closest thing the picture shows, leaving out the floor,
flat ground marks and the camera's own robot (its legs, a lidar mounted beside the lens), as a lidar's self
filter does. Off to the side, depth is shorter than distance by the cosine of the angle (a pole 26 degrees off
the axis reads 10 percent near in depth). A segmentation reading lists what the picture shows and how much of
it. Pictures come back only when asked for (image=true).
"""

from __future__ import annotations

import io
import math
import zlib
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor

SEEN_LIMIT = 6  # things a segmentation reading names, the biggest first
TIE_M = 0.002  # m: pixels this close to the nearest distance tie, and the one nearest the picture's middle wins
SPREAD_DEG = 3.0  # with noise, the nearest is the median of the pixels within this angle of it (see _nearest)
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
        self.lens = d.get("lens", "pinhole")
        self.baseline = d.get("baseline")  # stereo
        self.window, self.threshold = d.get("window_s"), d.get("threshold")  # event
        not_colliding = (model.geom_contype == 0) & (model.geom_conaffinity == 0)
        flat = model.geom_aabb[:, 5] < 0.01  # half as thick as a centimetre in its own frame: a mark on the ground
        self.ground = np.isin(model.geom_type, GROUND_TYPES) | (not_colliding & flat)
        root = int(model.body_rootid[model.cam_bodyid[self.camera]])
        # The camera's own robot: its shapes are not obstacles. A camera on the world itself has no robot.
        self.own = (model.body_rootid[model.geom_bodyid] == root) if root > 0 else np.zeros(model.ngeom, bool)

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

    def develop(self, render: Any, moment: dict[str, Any]) -> dict[str, Any]:
        """The camera's own reading of a moment: `render(moment)` draws it (the hub's renderer), then the noise is
        added and the sky beyond the range left out."""
        layers = render(moment)
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
        if self.mode in ("depth", "stereo"):
            out.update(self._depth_summary(value))
        elif self.mode == "segmentation":
            out["seen"] = self._seen(value["layers"])
        return out

    def cloud(self, value: dict[str, Any], limit: int) -> dict[str, Any] | None:
        """The depth picture as world points, thinned to `limit` (perception/cloud.py); None for the other modes. The
        camera's own robot is left out, as in a reading's nearest obstacle."""
        from robotforge.perception import cloud

        if self.mode not in ("depth", "stereo") or "depth" not in value:
            return None
        depth, layers = value["depth"], value["layers"]
        if layers.geoms is not None:
            seen = layers.geoms >= 0
            own = seen & self.own[np.where(seen, layers.geoms, 0)]
            depth = np.where(own, np.nan, depth)
        return cloud.summary(cloud.depth_points(depth, layers.camera, stride=2), limit)

    def _depth_summary(self, value: dict[str, Any]) -> dict[str, Any]:
        """center_m, min_m and median_m are depths along the axis, as the image holds them; nearest is a distance."""
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
            seen = geoms >= 0
            index = np.where(seen, geoms, 0)
            own = seen & self.own[index]
            obstacle = valid & seen & ~self.ground[index] & ~own
            if own.any():
                out["own_share"] = round(float(own.mean()), 3)  # the picture's share showing its own robot
            if obstacle.any():
                out["nearest"] = self._nearest(depth, obstacle, layers.camera)
        return out

    def _nearest(self, depth: np.ndarray, obstacle: np.ndarray, cam: Any) -> dict[str, Any]:
        """The closest obstacle pixel by distance from the lens (depth times the length of the pixel's ray, which
        goes one meter along the axis), with its bearing (0 ahead, + left), elevation and world point. Pixels within
        TIE_M of it tie (a wall facing the camera has a ring of them), and the one nearest the picture's middle
        wins, so the bearing does not jump between reads. With depth noise the single nearest pixel is the one
        whose noise came out shortest (four noise widths short among thousands), so a noisy camera reports the
        median of the obstacle pixels within SPREAD_DEG of it that are not far behind it."""
        vs, us = np.nonzero(obstacle)
        rays = cam.ray(us + 0.5, vs + 0.5)
        lengths = np.linalg.norm(rays, axis=1)
        dist = depth[vs, us] * lengths
        sigma = self.knobs.noise.get("depth", 0.0)
        low = float(dist.min())
        close = np.flatnonzero(dist <= low + max(TIE_M, 4 * sigma))
        off_middle = (us[close] + 0.5 - cam.cx) ** 2 + (vs[close] + 0.5 - cam.cy) ** 2
        i = int(close[int(np.argmin(off_middle))])
        distance = low
        if sigma:
            units = rays / lengths[:, None]
            around = (units @ units[i] >= math.cos(math.radians(SPREAD_DEG))) & (dist <= low + 8 * sigma)
            distance = float(np.median(dist[around]))
        ray = rays[i]
        local = np.array([ray @ cam.right, ray @ cam.up, ray @ cam.forward])
        point = cam.pos + ray * depth[vs[i], us[i]]
        return {
            "distance_m": round(distance, 3),
            "bearing_deg": round(-math.degrees(math.atan2(local[0], local[2])), 1),
            "elevation_deg": round(math.degrees(math.atan2(local[1], math.hypot(local[0], local[2]))), 1),
            "point_m": [round(float(v), 3) for v in point],
        }

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
        what = {
            "rgb": "fisheye camera" if self.lens == "fisheye" else "camera",
            "depth": "depth camera",
            "segmentation": "segmentation camera",
            "stereo": "stereo camera",
            "event": "event camera",
        }[self.mode]
        return f"{what}: {self.width}x{self.height}, {self.fov:.0f} deg (its picture: read_sensors with image=true)"

    def viewer(self, value: dict[str, Any], model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        rot = data.cam_xmat[self.camera].reshape(3, 3)
        return {
            "mode": self.mode,
            "lens": self.lens,
            "fov_deg": round(self.fov, 1),
            "size": [self.width, self.height],
            "range_m": self.range,
            "forward": [round(float(v), 4) for v in -rot[:, 2]],
            "up": [round(float(v), 4) for v in rot[:, 1]],
            "baseline_m": self.baseline,
            "window_s": self.window,
            "threshold": self.threshold,
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
        from PIL import Image  # here: Pillow is heavy, and only a picture asked for needs it

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


# Google's Turbo colormap as polynomials in t (0 dark blue, 1 dark red): even steps of distance look even.
TURBO = (
    (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
    (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
    (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
)


def depth_picture(depth: np.ndarray, far: float) -> np.ndarray:
    """Near is red, far is blue across the distances in view (Turbo); nothing within range is black."""
    valid = np.isfinite(depth)
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    low, high = float(np.nanmin(depth)), float(np.nanmax(depth))
    near = 1.0 - np.clip((np.nan_to_num(depth, nan=high) - low) / max(high - low, 1e-6), 0.0, 1.0)
    t = 0.12 + 0.83 * near  # Turbo's two ends are nearly black; the far end stays blue, apart from "nothing"
    rgb = np.stack([sum(c * t**i for i, c in enumerate(channel)) for channel in TURBO], axis=-1)
    rgb[~valid] = 0.0
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
