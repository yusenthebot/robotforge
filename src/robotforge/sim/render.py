"""Offscreen snapshots of a SimSession, so agents can look at what they built."""

from __future__ import annotations

import io
import math
import threading

import mujoco
import numpy as np
from PIL import Image

from robotforge.sim.session import SimSession

# Free-camera presets as (azimuth deg, elevation deg, distance in model extents).
# "default" matches the browser viewer's opening view.
VIEWS: dict[str, tuple[float, float, float]] = {
    "default": (127.6, -25.9, 2.0),
    "front": (180.0, -10.0, 1.8),
    "back": (0.0, -10.0, 1.8),
    "left": (270.0, -10.0, 1.8),
    "right": (90.0, -10.0, 1.8),
    "top": (90.0, -89.0, 2.0),
}

MAX_PIXELS = 1280 * 960


class SnapshotRenderer:
    """Caches one mujoco.Renderer per model version and size; safe to call from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._renderer: mujoco.Renderer | None = None
        self._key: tuple[int, int, int] | None = None

    def close(self) -> None:
        with self._lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
                self._key = None

    def png(self, session: SimSession, view: str = "default", width: int = 640, height: int = 480) -> bytes:
        rgb = self.rgb(session, view, width, height)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def rgb(self, session: SimSession, view: str = "default", width: int = 640, height: int = 480) -> np.ndarray:
        if width * height > MAX_PIXELS or width < 16 or height < 16:
            raise ValueError(f"image size {width}x{height} out of range (max {MAX_PIXELS} pixels)")
        with self._lock, session.lock:
            model, data = session.model, session.data
            width = min(width, model.vis.global_.offwidth)
            height = min(height, model.vis.global_.offheight)
            key = (session.version, width, height)
            if self._key != key:
                if self._renderer is not None:
                    self._renderer.close()
                self._renderer = mujoco.Renderer(model, height=height, width=width)
                self._key = key
            camera = _camera_for(model, view)
            self._renderer.update_scene(data, camera=camera)
            return self._renderer.render().copy()


def _camera_for(model: mujoco.MjModel, view: str) -> mujoco.MjvCamera | str:
    if view in VIEWS:
        azimuth, elevation, dist = VIEWS[view]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = model.stat.center
        cam.distance = dist * max(float(model.stat.extent), 0.1)
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view) >= 0:
        return view
    options = sorted(VIEWS) + [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
    raise ValueError(f"unknown view {view!r}; choose one of: {', '.join(options)}")


def look_direction(view: str) -> tuple[float, float, float]:
    """Unit vector the preset camera looks along (for tests and docs)."""
    azimuth, elevation, _ = VIEWS[view]
    a, e = math.radians(azimuth), math.radians(elevation)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
