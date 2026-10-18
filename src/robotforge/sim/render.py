"""Offscreen snapshots of a SimSession, so agents can look at what they built."""

from __future__ import annotations

import contextlib
import functools
import io
import itertools
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

from robotforge import wording
from robotforge.design.visuals import HIDDEN_GROUP, VISUAL_GROUP
from robotforge.sim import vision
from robotforge.sim import water as water_model
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
FADED_ALPHA = 0.25  # opacity of what stands between the camera and the robot a video or a robot's view shows
OCCLUDER_LAYERS = 6  # geoms one ray goes through on its way to the robot
DRAWN_GROUPS = np.array([1, 1, 1, 0, 0, 0], np.uint8)  # geom groups a picture shows (3 and up are collision shapes)

MAX_PIXELS = 1280 * 960
TOUCHING = 0.001  # m: geoms a contact holds closer than this touch (a contact with a margin can be further apart)
log = logging.getLogger(__name__)
EMBEDDED = 0.001  # m: a robot camera inside its own part, or this close to its shape, sees out as through the bare part
_PRIMITIVES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
}


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
        from PIL import Image  # only when a picture is asked for: Pillow is slow to import

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
            name, _, robots = _named_view(model, view)
            faded = occluders(model, data, camera, robot_geoms(model, name)) if name in robots else frozenset()
            return _on_gl_thread(self._render, model, data, camera, (session.version, width, height), faded)

    def rgb_of(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: Any,
        width: int,
        height: int,
        version: int,
        faded: frozenset[int] = frozenset(),
    ) -> np.ndarray:
        """A picture of a model and data no session holds (a recording's frame, posed in a scratch MjData), from a
        camera already made (an MjvCamera, or a view or camera name); the `faded` geoms are drawn see-through."""
        _check_size(width, height)
        with self._lock:
            width = min(width, model.vis.global_.offwidth)
            height = min(height, model.vis.global_.offheight)
            if isinstance(camera, str):
                camera = _camera_for(model, camera, data, width / height)
            return _on_gl_thread(self._render, model, data, camera, (version, width, height), faded)

    def _render(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera,
        key: tuple[int, int, int],
        faded: frozenset[int] = frozenset(),
    ) -> np.ndarray:
        """Runs on the GL thread, with the renderer and session locks held by the caller."""
        self._ready(model, key)
        self._renderer.update_scene(data, camera=camera)
        _water_casts_no_shadow(model, self._renderer.scene)
        if isinstance(camera, str):
            see_out_of_own_part(model, data, self._renderer.scene, model.camera(camera).id)
        if faded:
            scene = self._renderer.scene
            for i in range(scene.ngeom):
                g = scene.geoms[i]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in faded:
                    g.rgba[3] = min(float(g.rgba[3]), FADED_ALPHA)
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
        _water_casts_no_shadow(model, self._renderer.scene)
        if isinstance(camera, str):
            see_out_of_own_part(model, data, self._renderer.scene, model.camera(camera).id)
        return self._renderer.render().copy()

    def _layers(self, model: mujoco.MjModel, data: mujoco.MjData, camera, key: tuple[int, int, int], geoms: bool):
        self._ready(model, key)
        r = self._renderer
        r.update_scene(data, camera=camera, scene_option=_CAMERA_ONLY)
        _water_casts_no_shadow(model, r.scene)
        if isinstance(camera, str):
            see_out_of_own_part(model, data, r.scene, model.camera(camera).id)
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


def see_out_of_own_part(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene, cam: int) -> bool:
    """A robot camera inside its own part (the default camera sits at the part's middle) or on its surface sees
    its part as the bare shape physics gives it, as a rangefinder's ray does: the dressing drawn over the shape
    (parting lines, modules, covers, servo cases) would fill its picture with the part's insides. Run after the
    scene is updated; returns True when it changed the scene."""
    body = int(model.cam_bodyid[cam])
    own = np.flatnonzero(model.geom_bodyid == body)
    covered = [int(g) for g in own if model.geom_group[g] == HIDDEN_GROUP and int(model.geom_type[g]) in _PRIMITIVES]
    dressing = {int(g) for g in own if model.geom_group[g] == VISUAL_GROUP and _is_dressing(model, int(g))}
    if not dressing:
        return False
    # The part's shape is what collides; the rest carry the mass of its hardware (the camera's own lens among them).
    shapes = [g for g in covered if model.geom_contype[g] or model.geom_conaffinity[g]]
    eye = data.cam_xpos[cam]
    if shapes:  # inside or on the part's shape
        embedded = any(_distance(model, data, g, eye) <= EMBEDDED for g in shapes)
    else:  # a part with no shape (a bare joint) or one that does not collide: inside its dressing
        embedded = any(_in_box(model, data, g, eye) for g in dressing)
    if not embedded:
        return False
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid in dressing:
            geom.rgba[3] = 0.0  # MuJoCo draws nothing of a geom with no opacity, not even its shadow
    for g in covered:
        rgba = model.geom_rgba[g]
        if rgba[3] <= 0 or scene.ngeom >= scene.maxgeom:
            continue
        added = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            added,
            int(model.geom_type[g]),
            model.geom_size[g].astype(np.float64),
            data.geom_xpos[g].astype(np.float64),
            data.geom_xmat[g].astype(np.float64),
            rgba.astype(np.float32),
        )
        # Who it is, for segmentation pictures (the seeing lesson's): a fresh geom's segid broke MuJoCo's lookup.
        added.objtype, added.objid, added.segid = int(mujoco.mjtObj.mjOBJ_GEOM), g, scene.ngeom
        scene.ngeom += 1
    return True


def _is_dressing(model: mujoco.MjModel, geom: int) -> bool:
    """A visual mesh added by design.visuals (its material is `<robot>/rf_<preset>`), not an imported model's own."""
    mat = int(model.geom_matid[geom])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else None
    return bool(name) and name.rsplit("/", 1)[-1].startswith("rf_")


def _in_box(model: mujoco.MjModel, data: mujoco.MjData, geom: int, point: np.ndarray) -> bool:
    """True when `point` lies inside the geom's bounding box (MuJoCo's geom_aabb: middle and half extents)."""
    p = data.geom_xmat[geom].reshape(3, 3).T @ (np.asarray(point, float) - data.geom_xpos[geom])
    middle, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    return bool(np.all(np.abs(p - middle) <= half))


def _distance(model: mujoco.MjModel, data: mujoco.MjData, geom: int, point: np.ndarray) -> float:
    """How far `point` lies outside a primitive geom (negative inside; a box's corners count as square)."""
    p = data.geom_xmat[geom].reshape(3, 3).T @ (np.asarray(point, float) - data.geom_xpos[geom])
    size, kind = model.geom_size[geom], int(model.geom_type[geom])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(np.linalg.norm(p) - size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(np.linalg.norm([p[0], p[1], max(abs(p[2]) - size[1], 0.0)]) - size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return float(max(np.hypot(p[0], p[1]) - size[0], abs(p[2]) - size[1]))
    return float(np.max(np.abs(p) - size))


def _named_view(model: mujoco.MjModel, view: str) -> tuple[str, list[str], list[str]]:
    """The view as the model spells it ("Front" is front, "Go2" the robot go2), with the model's cameras and
    robots."""
    cameras = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name]
    robots = _robot_names(model)
    if view not in VIEWS and view not in cameras and view not in robots:
        view = wording.match(view, [*VIEWS, *cameras, *robots]) or view
    return view, cameras, robots


def _camera_for(
    model: mujoco.MjModel, view: str, data: mujoco.MjData | None = None, aspect: float = 4 / 3
) -> mujoco.MjvCamera | str:
    view, cameras, robots = _named_view(model, view)
    if view in VIEWS or view in robots:
        # A robot's name: the default direction, framed on that robot alone. Asked to show a Go2 with a lidar in
        # the warehouse, an agent's first observe was view="go2", refused (agent eval).
        azimuth, elevation, fill = VIEWS.get(view, VIEWS["default"])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = azimuth
        cam.elevation = elevation
        if view in VIEWS:
            box = _scene_box(model, data) if view == "overview" else _moving_box(model, data)
        else:
            box = _robot_box(model, data, view)
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
    options = [*sorted(VIEWS), *cameras, *robots]
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


def _robot_names(model: mujoco.MjModel) -> list[str]:
    """The robots in a world, by name: a robot's bodies are named <robot>/<body> under a root body of its own
    (a carried robot's are <robot>/<part>/<body>, under the same root)."""
    names: dict[str, None] = {}
    for root in sorted({int(r) for r in model.body_rootid[1:]}):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) or ""
        if "/" in name:
            names[name.split("/", 1)[0]] = None
    return list(names)


def robot_geoms(model: mujoco.MjModel, robot: str) -> list[int]:
    """The geoms a picture draws of one robot: those under a root body named <robot>/..."""
    prefix = f"{robot}/"
    return [
        g
        for g in range(model.ngeom)
        if model.geom_group[g] < 3
        and (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[model.geom_bodyid[g]])) or ""
        ).startswith(prefix)
    ]


def _robot_box(model: mujoco.MjModel, data: mujoco.MjData | None, robot: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds of one robot's drawn geoms as it is now."""
    return None if data is None else _geom_box(model, data, robot_geoms(model, robot))


def occluders(model: mujoco.MjModel, data: mujoco.MjData, cam: Any, followed: list[int]) -> frozenset[int]:
    """The geoms standing between the camera and the robot, to be drawn see-through: a Go2 walking in the park
    went behind a bench, and a kitchen's counters would hide it as often (a video's frames and a robot's view).
    Rays from the camera to the middle and the box corners of each of the robot's shapes find what is in the way
    (a few rays at the robot's outline missed the bench's thin post; bounding spheres faded a rock beside the dog).
    The ground and terrain (on the world body) and the robot itself never count."""
    if not followed or isinstance(cam, str):
        return frozenset()
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    eye = np.array(cam.lookat) - cam.distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    aims = []
    for g in followed:
        rot, aabb = data.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
        aims.append(data.geom_xpos[g] + rot @ aabb[:3])
        aims.extend(data.geom_xpos[g] + (aabb[:3] + corners * aabb[3:]) @ rot.T)
    mine, found = set(followed), set()
    hit = np.zeros(1, np.int32)
    for aim in aims:
        start, toward = eye.copy(), aim - eye
        left = float(np.linalg.norm(toward))
        if left < 1e-6:
            continue
        toward /= left
        for _ in range(OCCLUDER_LAYERS):
            far = mujoco.mj_ray(model, data, start, toward, DRAWN_GROUPS, 1, -1, hit)
            g = int(hit[0])
            if far < 0 or far >= left - 1e-4 or g in mine:
                break
            if model.geom_bodyid[g] != 0:  # not the ground or terrain
                found.add(g)
            start, left = start + toward * (far + 1e-4), left - far - 1e-4
    return frozenset(found)


def _geom_box(model: mujoco.MjModel, data: mujoco.MjData, geoms: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """World-aligned bounding box of some geoms in their current pose."""
    if not geoms:
        return None
    rot = data.geom_xmat[geoms].reshape(-1, 3, 3)
    aabb = model.geom_aabb[geoms]
    centers = data.geom_xpos[geoms] + np.einsum("nij,nj->ni", rot, aabb[:, :3])
    half = np.einsum("nij,nj->ni", np.abs(rot), aabb[:, 3:])
    return (centers - half).min(axis=0), (centers + half).max(axis=0)


def _water_casts_no_shadow(model: mujoco.MjModel, scene: mujoco.MjvScene) -> None:
    """The water's see-through sheet shaded the bottom under all of it inside the sun's shadow map (a dark
    square on the sea in the overview): drawn as decoration, it casts no shadow."""
    sheet = water_model.surface_geom(model)
    if sheet < 0:
        return
    for i in range(scene.ngeom):
        geom = scene.geoms[i]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == sheet:
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            return


def _scene_box(model: mujoco.MjModel, data: mujoco.MjData | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Bounds of everything drawn except infinite planes: terrain, furniture and robots."""
    if data is None:
        return None
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    sheet = water_model.surface_geom(model)  # the water's surface reaches past the scene: it is not what to frame
    box = _geom_box(
        model,
        data,
        [g for g in range(model.ngeom) if int(model.geom_type[g]) != plane and model.geom_group[g] < 3 and g != sheet],
    )
    return box if box is not None else _moving_box(model, data)
