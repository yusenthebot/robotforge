"""Offscreen snapshots of a SimSession, so agents can look at what they built."""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from robotforge import wording
from robotforge.sim import vision
from robotforge.sim.session import SimSession

# Free-camera presets as (azimuth deg, elevation deg, share of the image the content spans).
# "default" matches the browser viewer's opening direction; "overview" frames the whole world, terrain and
# furniture included, where the others frame what can move. Each is stepped back just far enough for its
# content to fill that share of the picture: a fixed multiple of the scene's size left a small arm and a
# cube a few pixels wide in the corner of a top view.
VIEWS: dict[str, tuple[float, float, float]] = {
    "default": (127.6, -25.9, 0.7),
    "overview": (122.0, -38.0, 0.85),
    "front": (180.0, -10.0, 0.75),
    "back": (0.0, -10.0, 0.75),
    "left": (270.0, -10.0, 0.75),
    "right": (90.0, -10.0, 0.75),
    "top": (90.0, -89.0, 0.75),
}
MIN_SPAN = np.array([0.8, 0.8, 0.3])  # m: a lone small robot is still shown with some ground around it

MAX_PIXELS = 1280 * 960
TOUCHING = 0.001  # m: geoms a contact holds closer than this touch (a contact with a margin can be further apart)
log = logging.getLogger(__name__)


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
        _check_size(width, height)
        with self._lock, session.lock:
            model, data = session.model, session.data
            width = min(width, model.vis.global_.offwidth)
            height = min(height, model.vis.global_.offheight)
            camera = _camera_for(model, view, data, width / height)  # bad views fail the same with or without GL
            return _on_gl_thread(self._render, model, data, camera, (session.version, width, height))

    def _render(self, model: mujoco.MjModel, data: mujoco.MjData, camera, key: tuple[int, int, int]) -> np.ndarray:
        """Runs on the GL thread, with the renderer and session locks held by the caller."""
        self._ready(model, key)
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render().copy()

    def _ready(self, model: mujoco.MjModel, key: tuple[int, int, int]) -> None:
        """A renderer for this model version and size, on the GL thread."""
        reason = _probe_gl()
        if reason is not None:
            hint = _GL_HINTS.get(sys.platform, _GL_HINTS["linux"])
            raise RenderUnavailable(f"Offscreen OpenGL rendering is unavailable here ({reason}). {hint}")
        # The model itself is checked too: layers_of renders models that are not a session's (a lesson's copy, a
        # measurement script's), whose version numbers say nothing about which model they are.
        if self._key != key or self._renderer is None or self._renderer.model is not model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(model, height=key[2], width=key[1])
            self._key = key

    def layers(
        self,
        session: SimSession,
        view: str = "default",
        width: int = 640,
        height: int = 480,
        geoms: bool = False,
        exposure: float = 1.0,
        shown: bool = False,
    ) -> Layers:
        """The picture, the depth and (with `geoms`) which geom each pixel shows, from one camera at one moment.
        `exposure` scales every light for this picture only, as a camera's exposure setting would; with `shown`
        the same moment is also drawn at the 3D view's brightness, the picture to show a person."""
        _check_size(width, height)
        with self._lock, session.lock:
            with _exposed(session.model, exposure):
                out = self._take_layers(session.model, session.data, view, width, height, session.version, geoms)
            if not shown:
                return out
            if exposure == 1.0:
                return replace(out, shown=out.rgb)
            h, w = out.rgb.shape[:2]
            camera = _camera_for(session.model, view, session.data, w / h)
            key = (session.version, w, h)
            picture = _on_gl_thread(self._camera_picture, session.model, session.data, camera, key)
            return replace(out, shown=picture)

    def layers_of(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        view: str,
        width: int,
        height: int,
        version: int,
        geoms: bool = False,
    ) -> Layers:
        """`layers` for a model and data the caller already holds the lock on (a lesson, or a scratch copy).

        The renderer's lock is taken after the caller's, the opposite order to `layers`: a renderer used this
        way (a lesson's own) must not also be used through `layers`."""
        _check_size(width, height)
        with self._lock:
            return self._take_layers(model, data, view, width, height, version, geoms)

    def _take_layers(
        self, model: mujoco.MjModel, data: mujoco.MjData, view: str, width: int, height: int, version: int, geoms: bool
    ) -> Layers:
        width = min(width, model.vis.global_.offwidth)
        height = min(height, model.vis.global_.offheight)
        camera = _camera_for(model, view, data, width / height)
        return _on_gl_thread(self._layers, model, data, camera, (version, width, height), geoms)

    def _camera_picture(self, model: mujoco.MjModel, data: mujoco.MjData, camera, key: tuple[int, int, int]):
        """What the camera sees, without the model's marker sites, as _layers draws it (on the GL thread)."""
        self._ready(model, key)
        self._renderer.update_scene(data, camera=camera, scene_option=_CAMERA_ONLY)
        return self._renderer.render().copy()

    def _layers(self, model: mujoco.MjModel, data: mujoco.MjData, camera, key: tuple[int, int, int], geoms: bool):
        self._ready(model, key)
        r = self._renderer
        r.update_scene(data, camera=camera, scene_option=_CAMERA_ONLY)
        rgb = r.render().copy()
        pinhole = vision.Pinhole.from_scene(r.scene, key[1], key[2])
        r.enable_depth_rendering()
        try:
            with _quiet_clip_control():
                depth = r.render().copy()
        finally:
            r.disable_depth_rendering()
        seen = None
        touching = None
        if geoms:
            r.enable_segmentation_rendering()
            try:
                seg = r.render()
            finally:
                r.disable_segmentation_rendering()
            seen = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.int32)
            contacts = data.contact
            touching = contacts.geom[contacts.dist <= TOUCHING].astype(np.int32)
        return Layers(rgb, depth, seen, pinhole, model, touching)


def _camera_only() -> mujoco.MjvOption:
    """What a real camera sees: the bodies, not the model's marker sites. The arm template's tool-centre marker,
    an orange ball between the fingers, filled the top of its own wrist camera's picture and was found as an
    orange thing that is not there."""
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    return option


_CAMERA_ONLY = _camera_only()


@dataclass(frozen=True)
class Layers:
    """One look through a camera: the picture, the depth camera's reading (m along the view direction) and,
    when asked, the geom each pixel shows (-1 for the sky), with the camera they were taken with and the model
    they show (a world swapped since has other geoms)."""

    rgb: np.ndarray
    depth: np.ndarray
    geoms: np.ndarray | None
    camera: vision.Pinhole
    model: mujoco.MjModel
    touching: np.ndarray | None = None  # with the geoms: the pairs of geoms in contact at that moment (Nx2)
    shown: np.ndarray | None = None  # asked for with `shown`: the picture at the 3D view's brightness


CLIP_CONTROL = "ARB_clip_control unavailable"  # macOS OpenGL 4.1 has no reverse-Z depth; MuJoCo says so


@contextlib.contextmanager
def _quiet_clip_control() -> Iterator[None]:
    """Drop MuJoCo's warning that the depth buffer is less precise without ARB_clip_control (macOS): MuJoCo writes
    it to stderr and appends it to a MUJOCO_LOG.TXT in whatever folder robotforge was started from. On a Mac the
    seeing lesson's depth estimates are still within 4 mm at half a metre. Other warnings meanwhile go to the log."""
    previous = mujoco.get_mju_user_warning()

    def handler(message: str) -> None:
        if CLIP_CONTROL in message:
            return
        if previous is not None:
            previous(message)
        else:
            log.warning("MuJoCo: %s", message)

    mujoco.set_mju_user_warning(handler)
    try:
        yield
    finally:
        mujoco.set_mju_user_warning(previous)


@contextlib.contextmanager
def _exposed(model: mujoco.MjModel, exposure: float) -> Iterator[None]:
    """Every light of the model (and the headlight) scaled by `exposure` until the block ends."""
    if exposure == 1.0:
        yield
        return
    saved = [
        (array, array.copy())
        for array in (
            model.light_diffuse,
            model.light_ambient,
            model.light_specular,
            model.vis.headlight.ambient,
            model.vis.headlight.diffuse,
            model.vis.headlight.specular,
        )
    ]
    try:
        for array, original in saved:
            array[:] = original * exposure
        yield
    finally:
        for array, original in saved:
            array[:] = original


def _check_size(width: int, height: int) -> None:
    if width * height > MAX_PIXELS or width < 16 or height < 16:
        raise ValueError(
            f"image size {width}x{height} is out of range: width and height are pixels, each at least 16, "
            f"and width*height is at most {MAX_PIXELS}"
        )


def _camera_for(
    model: mujoco.MjModel, view: str, data: mujoco.MjData | None = None, aspect: float = 4 / 3
) -> mujoco.MjvCamera | str:
    cameras = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
    if view not in VIEWS and view not in cameras:
        view = wording.match(view, [*VIEWS, *cameras]) or view  # "Front", "Dog/Front_Cam"
    if view in VIEWS:
        azimuth, elevation, fill = VIEWS[view]
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = azimuth
        cam.elevation = elevation
        box = _scene_box(model, data) if view == "overview" else _moving_box(model, data)
        if box is None:  # nothing posed yet: the model's own statistics
            cam.lookat[:] = model.stat.center
            cam.distance = 2.0 * max(float(model.stat.extent), 0.1)
            return cam
        low, high = box
        grow = np.maximum(MIN_SPAN - (high - low), 0.0) / 2
        low, high = low - grow, high + grow
        cam.lookat[:] = (low + high) / 2
        cam.distance = _fit_distance(low, high, azimuth, elevation, fill, float(model.vis.global_.fovy), aspect)
        return cam
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, view) >= 0:
        return view
    options = [*sorted(VIEWS), *cameras]
    close = wording.near(view, options)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise ValueError(f"unknown view {view!r}.{hint} Views: {', '.join(options)}")


def _fit_distance(
    low: np.ndarray, high: np.ndarray, azimuth: float, elevation: float, fill: float, fovy: float, aspect: float
) -> float:
    """How far back a camera looking at the box's center must stand for every corner to land inside `fill`
    of the image, given its direction (MuJoCo's azimuth and elevation) and field of view."""
    az, el = np.radians(azimuth), np.radians(elevation)
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_v = np.tan(np.radians(fovy) / 2) * fill
    tan_h = tan_v * aspect
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    rel = corners - (low + high) / 2
    across, upward, deeper = rel @ right, rel @ up, rel @ forward
    need = np.maximum(np.abs(across) / tan_h, np.abs(upward) / tan_v) - deeper
    return float(max(need.max(), 0.6 * np.linalg.norm(high - low)))


def _moving_box(model: mujoco.MjModel, data: mujoco.MjData | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds of what can move (robots, loose objects) as it is now, so a robot that walked away stays in the
    picture and a big terrain or a room full of furniture does not shrink it."""
    if data is None or model.nbody <= 1:
        return None
    moving = model.body_weldid[model.geom_bodyid] != 0
    return _geom_box(model, data, [g for g in range(model.ngeom) if moving[g] and model.geom_group[g] < 3])


def _geom_box(model: mujoco.MjModel, data: mujoco.MjData, geoms: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """World-aligned bounding box of some geoms in their current pose."""
    if not geoms:
        return None
    rot = data.geom_xmat[geoms].reshape(-1, 3, 3)
    aabb = model.geom_aabb[geoms]
    centers = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    half = np.einsum("nij,nj->ni", np.abs(rot), aabb[:, 3:])
    return (centers - half).min(axis=0), (centers + half).max(axis=0)


def _scene_box(model: mujoco.MjModel, data: mujoco.MjData | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds of everything drawn except infinite planes: terrain, furniture and robots."""
    if data is None:
        return None
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    box = _geom_box(
        model, data, [g for g in range(model.ngeom) if int(model.geom_type[g]) != plane and model.geom_group[g] < 3]
    )
    return box if box is not None else _moving_box(model, data)
