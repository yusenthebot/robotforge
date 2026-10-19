"""Cameras with other optics: a stereo pair, a fisheye lens and an event camera.

All three are cameras (sensors/camera.py): the same mount, the same frustum in the viewer, the same way of
keeping the state of the world when a frame is due and drawing it when it is read. What differs is what a reading
is made of.

Stereo: a left camera and a right one `baseline` meters apart. Its depth is not the renderer's exact depth but what
a stereo matcher can report: the true geometry gives a disparity d = f B / Z pixels, a matcher finds it to a
fraction of a pixel (DISPARITY_STEP, plus noise), and Z is recomputed from it, so the error grows with the square
of the distance and nothing beyond f B / MIN_DISPARITY (or nearer than MIN_DEPTH) is measured. A picture is the
left and right images side by side.

Fisheye: the scene drawn through an equidistant lens (r = f theta) up to 200 degrees across. MuJoCo draws pinhole
views only, so five square views (front, left, right, up, down, 90 degrees each) are drawn and the lens's pixels
are looked up in them.

Event camera: each pixel reports only that its brightness changed. The frame at the moment a frame is due and the
one `window_s` earlier are compared in log brightness: a pixel whose log brightness rose by more than the
contrast threshold gave an ON event (one per threshold crossed), one that fell an OFF event. Noise in the picture
(rgb noise, levels of 255) gives the spurious events a real sensor has. A picture shows ON red and OFF blue.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.camera import Camera

DISPARITY_STEP = 1 / 8  # px a stereo matcher's subpixel estimate moves in
MIN_DISPARITY = 1.0  # px: a smaller disparity is lost in the matcher's noise, so the stereo range ends there
MAX_DISPARITY = 96.0  # px: the matcher's search window; anything nearer than f B / MAX_DISPARITY is lost
MIN_DEPTH = 0.15  # m
CUBE_SIZE = 256  # px: each of a fisheye's five views
EVENT_LEVELS = 4  # events one pixel can give in one window, at most
EVENT_FLOOR = 8.0  # levels of 255 added before the logarithm: a dark pixel does not blow up
ON_RGB, OFF_RGB, QUIET_RGB = (232, 64, 56), (56, 112, 236), (122, 122, 122)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


class Stereo(Camera):
    def __init__(self, model, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        d = decl.data
        self.right_view = decl.full(d["right_camera"])
        self.baseline = float(d["baseline"])
        self.focal_px = (self.height / 2) / math.tan(math.radians(self.fov) / 2)

    def stereo_depth(self, depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The depth a matcher would report for the true `depth` (NaN where it has none). The depth noise knob is the
        error in meters at 1 m: a constant error in disparity, which is a depth error that grows with Z squared."""
        with np.errstate(divide="ignore", invalid="ignore"):
            disparity = self.focal_px * self.baseline / depth
        if self.knobs.noise.get("depth"):
            disparity = disparity + rng.normal(
                0.0, self.knobs.noise["depth"] * self.focal_px * self.baseline, depth.shape
            )
        disparity = np.round(disparity / DISPARITY_STEP) * DISPARITY_STEP
        lost = ~np.isfinite(disparity) | (disparity < MIN_DISPARITY) | (disparity > MAX_DISPARITY)
        out = self.focal_px * self.baseline / np.where(lost, np.nan, disparity)
        out[(out < MIN_DEPTH) | (out > self.range)] = np.nan
        return out

    def develop(self, render, moment: dict[str, Any]) -> dict[str, Any]:
        layers = render(moment)
        rng = np.random.default_rng(moment["noise_seed"])
        true_depth = np.array(layers.depth, dtype=np.float64)
        true_depth[true_depth >= min(self.range, layers.camera.far * 0.999)] = np.nan
        depth = self.stereo_depth(true_depth, rng)
        return {"layers": layers, "depth": depth, "right": lambda: render(moment, view=self.right_view, geoms=False)}

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        out = super().reading(value, full)
        out.update(
            baseline_m=self.baseline,
            focal_px=round(self.focal_px, 1),
            reaches_m=round(self.focal_px * self.baseline / MIN_DISPARITY, 2),
            depth_error_m={
                f"{z:g}": round(z * z / (self.focal_px * self.baseline) * DISPARITY_STEP / 2, 4) for z in (1, 3)
            },
        )
        return out

    def png(self, value: dict[str, Any]) -> bytes | None:
        layers = value.get("layers")
        if layers is None:
            return None
        import io

        from PIL import Image

        left = layers.rgb
        right = value["right"]().rgb
        buf = io.BytesIO()
        Image.fromarray(np.concatenate([left, right], axis=1)).save(buf, format="PNG", optimize=True)
        return buf.getvalue()


class Fisheye(Camera):
    def __init__(self, model, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        self.cube = [decl.full(c) for c in decl.data["cube"]]
        self.lens_fov = float(decl.data.get("fov", self.fov))
        self._lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each pixel of the picture: which of the five views it falls in and where (row, column) in it."""
        if self._lookup is not None:
            return self._lookup
        w, h = self.width, self.height
        half = math.radians(self.lens_fov) / 2
        f = min(w, h) / 2 / half  # px per radian: the lens's circle (its diagonal field) fills the short side
        ys, xs = np.mgrid[0:h, 0:w]
        dx, dy = xs + 0.5 - w / 2, h / 2 - (ys + 0.5)
        r = np.hypot(dx, dy)
        theta = r / f
        phi = np.arctan2(dy, dx)
        # the ray in the camera's frame: x right, y up, z forward
        ray = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
        faces = np.full((h, w), -1, np.int8)
        rows = np.zeros((h, w), np.int32)
        cols = np.zeros((h, w), np.int32)
        # a face looks along its own axis with 90 degrees across: (axis, right, up) in the camera's frame
        views = [
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),  # front
            ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),  # left
            ((1, 0, 0), (0, 0, -1), (0, 1, 0)),  # right
            ((0, 1, 0), (1, 0, 0), (0, 0, -1)),  # up
            ((0, -1, 0), (1, 0, 0), (0, 0, 1)),  # down
        ]
        best = np.full((h, w), -np.inf)
        for k, (axis, right, up) in enumerate(views):
            ahead = ray @ np.array(axis, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (ray @ np.array(right, float)) / ahead
                v = (ray @ np.array(up, float)) / ahead
            fits = (ahead > 1e-6) & (np.abs(u) <= 1.0 + 1e-9) & (np.abs(v) <= 1.0 + 1e-9) & (ahead > best)
            faces[fits] = k
            best[fits] = ahead[fits]
            cols[fits] = np.clip(((u[fits] + 1) / 2 * CUBE_SIZE).astype(np.int32), 0, CUBE_SIZE - 1)
            rows[fits] = np.clip(((1 - v[fits]) / 2 * CUBE_SIZE).astype(np.int32), 0, CUBE_SIZE - 1)
        faces[theta > half] = -1  # outside the lens's circle
        self._lookup = (faces, rows, cols)
        return self._lookup

    def develop(self, render, moment: dict[str, Any]) -> dict[str, Any]:
        faces, rows, cols = self._faces()
        views = [render(moment, view=name, width=CUBE_SIZE, height=CUBE_SIZE, geoms=False).rgb for name in self.cube]
        picture = np.zeros((self.height, self.width, 3), np.uint8)
        for k, rgb in enumerate(views):
            pick = faces == k
            picture[pick] = rgb[rows[pick], cols[pick]]
        if self.knobs.noise.get("rgb"):
            rng = np.random.default_rng(moment["noise_seed"])
            noisy = picture.astype(np.float64) + rng.normal(0.0, self.knobs.noise["rgb"], picture.shape)
            picture = np.where((faces >= 0)[..., None], np.clip(noisy, 0, 255), 0).astype(np.uint8)
        return {"layers": None, "rgb": picture, "lens_share": float((faces >= 0).mean())}

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        out = {"mode": "rgb", "lens": "fisheye", "size": [self.width, self.height], "fov_deg": self.lens_fov}
        out["share_of_picture_in_the_circle"] = round(value.get("lens_share", 0.0), 3)
        return out

    def png(self, value: dict[str, Any]) -> bytes | None:
        picture = value.get("rgb")
        if picture is None:
            return None
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(picture).save(buf, format="PNG", optimize=True)
        return buf.getvalue()


class Event(Camera):
    integrates = True  # it samples as time passes, to know how the world was a window ago
    default_rate = 100.0

    def __init__(self, model, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        d = decl.data
        self.threshold = float(d.get("threshold") or 0.15)
        self.window = float(d.get("window_s") or 0.03)
        period = self.period or 0.01
        self.history: deque[tuple[float, dict[str, Any]]] = deque(maxlen=max(3, math.ceil(self.window / period) + 3))

    def measure(self, model, data) -> dict[str, Any]:
        moment = super().measure(model, data)
        now = float(data.time)
        while len(self.history) > 1 and now - self.history[1][0] >= self.window:
            self.history.popleft()  # the second oldest is still old enough: the oldest is not needed
        prior = self.history[0][1] if self.history else moment
        self.history.append((now, moment))
        return {**moment, "prior": prior, "span": now - (self.history[0][0] if self.history else now)}

    def develop(self, render, moment: dict[str, Any]) -> dict[str, Any]:
        before = render(moment["prior"], geoms=False)
        after = render(moment, geoms=False)
        rng = np.random.default_rng(moment["noise_seed"])
        a, b = luma(before.rgb), luma(after.rgb)
        sigma = self.knobs.noise.get("rgb", 0.0)
        if sigma:
            a = a + rng.normal(0.0, sigma, a.shape)
            b = b + rng.normal(0.0, sigma, b.shape)
        change = np.log(np.maximum(b, 0) + EVENT_FLOOR) - np.log(np.maximum(a, 0) + EVENT_FLOOR)
        count = np.clip(np.floor(np.abs(change) / self.threshold), 0, EVENT_LEVELS)
        on = (change > 0) * count
        off = (change < 0) * count
        return {"layers": after, "on": on, "off": off, "span": max(moment.get("span", 0.0), 1e-3)}

    def reading(self, value: dict[str, Any], full: bool = False) -> dict[str, Any]:
        on, off = value["on"], value["off"]
        active = (on + off) > 0
        out: dict[str, Any] = {
            "mode": "event",
            "size": [self.width, self.height],
            "events": int(on.sum() + off.sum()),
            "on": int(on.sum()),
            "off": int(off.sum()),
            "active_share": round(float(active.mean()), 4),
            "window_s": round(value["span"], 4),
        }
        out["rate_ev_s"] = round(out["events"] / value["span"])
        if active.any():
            vs, us = np.nonzero(active)
            out["centroid_px"] = [round(float(us.mean()), 1), round(float(vs.mean()), 1)]
            out["box_px"] = [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())]
        return out

    def png(self, value: dict[str, Any]) -> bytes | None:
        if "on" not in value:
            return None
        import io

        from PIL import Image

        picture = np.empty((*value["on"].shape, 3), np.uint8)
        picture[:] = QUIET_RGB
        picture[value["off"] > 0] = OFF_RGB
        picture[value["on"] > 0] = ON_RGB
        buf = io.BytesIO()
        Image.fromarray(picture).save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def make_camera(model: Any, decl: declare.Declared) -> Camera:
    """The camera a definition describes: a plain one, or a stereo pair, an event camera or a fisheye."""
    d = decl.data
    mode = d.get("mode", "rgb")
    if mode == "stereo":
        return Stereo(model, decl)
    if mode == "event":
        return Event(model, decl)
    if d.get("lens") == "fisheye":
        return Fisheye(model, decl)
    return Camera(model, decl)
