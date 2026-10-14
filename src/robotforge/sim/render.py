"""Offscreen snapshots of a SimSession, so agents can look at what they built."""

from __future__ import annotations

import functools
import io
import math
import sys
import threading
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

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


class RenderUnavailable(RuntimeError):
    """Offscreen OpenGL rendering cannot start on this machine (no GPU, driver or display)."""


# An OpenGL context belongs to the thread that made it current (EGL refuses to move it, GLFW wants one
# owner), but MCP runs sync tools on whichever worker thread is free. Every GL call goes through this
# single thread so the context never changes hands.
_GL_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robotforge-gl")
_GL_HINTS = {
    "linux": "Install Mesa's EGL or OSMesa libraries and set MUJOCO_GL=egl or MUJOCO_GL=osmesa.",
    "win32": "Windows needs a graphics driver with OpenGL 3.3; headless machines and CI runners have none.",
    "darwin": "This is unexpected on macOS; check that MUJOCO_GL is unset or cgl.",
}


def _on_gl_thread[T](fn: Callable[..., T], *args: Any) -> T:
    return _GL_THREAD.submit(fn, *args).result()


@functools.cache
def _probe_gl() -> str | None:
    """Render one frame of an empty world. Only ever runs on the GL thread, once per process."""
    # GLFW reports why it failed ("the driver does not appear to support OpenGL") as a warning, not an exception.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            _render_empty_frame()
        except Exception as exc:
            # GL setup fails with RuntimeError, ImportError, OSError, AttributeError, GLFWError, FatalError
            # depending on the platform. The probe model is known good, so any failure is the machine's.
            reason = f"{type(exc).__name__}: {exc}"
            details = "; ".join(str(w.message) for w in caught)
            return " ".join((f"{reason} ({details})" if details else reason).split())[:300]
    return None


def _render_empty_frame() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    renderer = mujoco.Renderer(model, height=16, width=16)
    try:
        renderer.update_scene(mujoco.MjData(model))
        renderer.render()
    finally:
        renderer.close()


def render_unavailable_reason() -> str | None:
    """Why offscreen rendering cannot work here, or None when it can. Probed once, then cached."""
    return _on_gl_thread(_probe_gl)


class SnapshotRenderer:
    """Caches one mujoco.Renderer per model version and size; safe to call from any thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._renderer: mujoco.Renderer | None = None
        self._key: tuple[int, int, int] | None = None

    def close(self) -> None:
        with self._lock:
            if self._renderer is not None:
                _on_gl_thread(self._renderer.close)
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
            camera = _camera_for(model, view)  # bad arguments are reported the same with or without GL
            return _on_gl_thread(self._render, model, data, camera, (session.version, width, height))

    def _render(self, model: mujoco.MjModel, data: mujoco.MjData, camera, key: tuple[int, int, int]) -> np.ndarray:
        """Runs on the GL thread, with the renderer and session locks held by the caller."""
        reason = _probe_gl()
        if reason is not None:
            hint = _GL_HINTS.get(sys.platform, _GL_HINTS["linux"])
            raise RenderUnavailable(f"Offscreen OpenGL rendering is unavailable here ({reason}). {hint}")
        if self._key != key:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model, height=key[2], width=key[1])
            self._key = key
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
