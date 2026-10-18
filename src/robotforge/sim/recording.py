"""Videos of what robots do, for an agent: move(record=...) and run_script(record=...) save a GIF or an animated WebP
(Pillow), or an MP4 where the user has turned that on (ffmpeg on PATH and ROBOTFORGE_FFMPEG=1: ffmpeg is not one of
robotforge's approved dependencies, so nothing calls it unless asked to).

The viewer's Record button records the browser's canvas, and an agent has no browser: asked for a video of a robot
from parts, one installed Playwright with npm, drove the viewer in headless Chromium and pressed the button for it
(198 s, 1.03M input tokens). Here the poses are kept at a fixed frame rate on every physics step, whoever steps (the
viewer's clock, a tool, a script), and drawn once the motion is over, offscreen, with a camera that follows the
robot smoothly from the viewer's opening direction.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import itertools
import math
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from robotforge.sim import render
from robotforge.sim.session import SimSession

FPS = 20  # frames kept per simulated second
WIDTH, HEIGHT = 640, 360
MAX_FRAMES = 300  # drawn: 15 s at FPS; a longer recording plays faster, every k-th frame
FORMATS = (".gif", ".webp", ".mp4")
FOLLOW_SECONDS = 0.5  # s either side the camera averages the robot's place over, so it does not shake
MIN_SPAN = np.array([0.7, 0.7, 0.25])  # m the picture keeps round a small robot: the floor sliding by shows it moving
VIEW = "default"  # the direction the camera looks from: the viewer's opening one
DRAWN = np.array([1, 1, 1, 0, 0, 0], np.uint8)  # geom groups a picture shows (3 and up are collision shapes)
MAX_LAYERS = 6  # geoms one ray goes through on its way to the robot
FFMPEG_SWITCH = "ROBOTFORGE_FFMPEG"  # set to 1 to let a name ending in .mp4 call ffmpeg


class RecordingError(ValueError):
    """A record argument that cannot be used (an unknown format, an MP4 without ffmpeg), or a video not written."""


@dataclass
class Recorder:
    """Keeps the poses of everything that moves, FPS times a simulated second, while it is attached to a session."""

    session: SimSession
    fps: float = FPS
    times: list[float] = field(default_factory=list)
    qpos: list[np.ndarray] = field(default_factory=list)
    mocap: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    stopped_by: str | None = None  # why it stopped early (the world was edited or reset)
    _model: mujoco.MjModel | None = None
    _version: int = 0
    _next: float = 0.0

    def start(self) -> Recorder:
        with self.session.lock:
            self._model, self._version = self.session.model, self.session.version
            self._next = float(self.session.data.time)
            self._grab()
            self.session.step_hooks.append(self._grab)
        return self

    def stop(self) -> None:
        with self.session.lock:
            if self._grab in self.session.step_hooks:
                self.session.step_hooks.remove(self._grab)

    def _grab(self) -> None:
        """Runs after every step, with the session's lock held."""
        s = self.session
        if s.version != self._version or s.model is not self._model:
            self.stopped_by = "the world changed (an edit or a reset), so the recording ends there"
            self.stop()
            return
        t = float(s.data.time)
        if t + 1e-9 < self._next:
            return
        if self.times and t < self.times[-1]:  # the clock went back: a reset
            self.stopped_by = "the simulation was reset, so the recording ends there"
            self.stop()
            return
        self.times.append(t)
        self.qpos.append(s.data.qpos.copy())
        self.mocap.append((s.data.mocap_pos.copy(), s.data.mocap_quat.copy()))
        self._next = t + 1.0 / self.fps

    @property
    def seconds(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0


def mp4_allowed() -> bool:
    """Whether a name ending in .mp4 may call ffmpeg: the user turned it on and ffmpeg is installed. An MP4 is far
    smaller (0.4 MB where a GIF of 6 s on the park's grass took 5.6), but ffmpeg is a program robotforge would run
    that is not among its approved dependencies."""
    return os.environ.get(FFMPEG_SWITCH) == "1" and shutil.which("ffmpeg") is not None


def target(record: bool | str, folder: Path, robot: str | None) -> Path:
    """Where a recording goes: record=true names it after the robot and the time, a GIF in `folder`; a name without
    a folder goes in `folder` too; a path ending in .gif, .webp or .mp4 is used as given (~ is the home folder)."""
    if record is True:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return folder / f"{(robot or 'world').replace('/', '_')}-{stamp}.gif"
    name = str(record).strip()
    if not name:
        raise RecordingError('record is true or a file name such as "dog_walk.gif"')
    path = Path(name).expanduser()
    if not path.suffix:
        path = path.with_suffix(".gif")
    if path.suffix.lower() not in FORMATS:
        raise RecordingError(f'record saves {", ".join(FORMATS)} files, not {path.suffix}; e.g. record="walk.gif"')
    if path.suffix.lower() == ".mp4" and not mp4_allowed():
        raise RecordingError(
            f"an .mp4 needs ffmpeg, which robotforge runs only where the user set {FFMPEG_SWITCH}=1 and ffmpeg is "
            'installed; .gif and .webp always work, e.g. record="walk.webp"'
        )
    return path if path.is_absolute() else folder / path


def _followed(model: mujoco.MjModel, robot: str | None) -> list[int]:
    """The geoms the camera follows: the robot's drawn ones; with no robot every robot's (a robot's bodies are
    named <robot>/<body>; a warehouse's loose crates would pull the camera far back), else everything that moves."""
    moving = model.body_weldid[model.geom_bodyid] != 0
    drawn = [g for g in range(model.ngeom) if model.geom_group[g] < 3]
    roots = {
        g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[model.geom_bodyid[g]])) or ""
        for g in drawn
    }
    if robot is not None:
        return [g for g in drawn if roots[g].startswith(f"{robot}/")]
    robots = [g for g in drawn if "/" in roots[g]]
    return robots or [g for g in drawn if moving[g]]


def _pose(model: mujoco.MjModel, data: mujoco.MjData, rec: Recorder, i: int) -> mujoco.MjData:
    """`data` posed as the recording kept frame i (one scratch MjData for all: a Go2's is megabytes)."""
    data.qpos[:] = rec.qpos[i]
    data.mocap_pos[:], data.mocap_quat[:] = rec.mocap[i]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    return data


def _cameras(model: mujoco.MjModel, boxes: list[Any], dt_s: float) -> list[Any]:
    """A camera for each frame: from the viewer's opening direction, as far back as the robot's biggest outline
    needs (the same all along), looking at the robot's middle averaged over FOLLOW_SECONDS either side. The whole
    motion is known before the first frame is drawn, so the average looks ahead as much as back: the camera does
    not shake with each step, and does not trail behind a robot going steadily."""
    azimuth, elevation, fill = render.VIEWS[VIEW]
    if not boxes or any(b is None for b in boxes):
        return [VIEW] * len(boxes)
    span = np.maximum(np.max([high - low for low, high in boxes], axis=0), MIN_SPAN)
    fovy = float(model.vis.global_.fovy)
    distance = render._fit_distance(-span / 2, span / 2, azimuth, elevation, fill, fovy, WIDTH / HEIGHT)
    middles = np.array([(low + high) / 2 for low, high in boxes])
    reach = max(0, round(FOLLOW_SECONDS / dt_s)) if dt_s > 0 else 0
    cams = []
    for k in range(len(middles)):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = middles[max(0, k - reach) : k + reach + 1].mean(axis=0)
        cams.append(cam)
    return cams


def occluders(model: mujoco.MjModel, data: mujoco.MjData, cam: Any, followed: list[int]) -> frozenset[int]:
    """The geoms standing between the camera and the robot, to be drawn see-through: a Go2 walking in the park
    went behind a bench, and a kitchen's counters would hide it as often. Rays from the camera to the middle and
    the box corners of each of the robot's shapes find what is in the way (a few rays at the robot's outline
    missed the bench's thin post; bounding spheres faded a rock beside the dog). The ground and terrain (on the
    world body) and the robot itself never count."""
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
        for _ in range(MAX_LAYERS):
            far = mujoco.mj_ray(model, data, start, toward, DRAWN, 1, -1, hit)
            g = int(hit[0])
            if far < 0 or far >= left - 1e-4 or g in mine:
                break
            if model.geom_bodyid[g] != 0:  # not the ground or terrain
                found.add(g)
            start, left = start + toward * (far + 1e-4), left - far - 1e-4
    return frozenset(found)


def save(
    rec: Recorder,
    renderer: render.SnapshotRenderer,
    path: Path,
    robot: str | None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict[str, Any]:
    """Draw the kept poses into a video at `path` and say what it is. Raises render.RenderUnavailable without
    OpenGL."""
    model = rec._model
    if model is None or len(rec.times) < 2:
        raise RecordingError("nothing was recorded: no time passed")
    every = max(1, math.ceil(len(rec.times) / MAX_FRAMES))
    picked = list(range(0, len(rec.times), every))
    scratch, geoms = mujoco.MjData(model), _followed(model, robot)
    boxes = [render._geom_box(model, _pose(model, scratch, rec, i), geoms) for i in picked]
    cams = _cameras(model, boxes, every / rec.fps)

    def draw(k: int) -> np.ndarray:
        posed = _pose(model, scratch, rec, picked[k])
        faded = occluders(model, posed, cams[k], geoms)
        return renderer.rgb_of(model, posed, cams[k], width, height, rec._version, faded=faded)

    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, draw, len(picked), 1000.0 / rec.fps)  # each drawn frame stands for `every` kept ones
    out = {
        "path": str(path),
        "seconds": round(rec.seconds, 2),
        "frames": len(picked),
        "size_mb": round(path.stat().st_size / 1e6, 2),
    }
    if every > 1:
        out["plays"] = f"{every}x faster (at most {MAX_FRAMES} frames)"
    if rec.stopped_by:
        out["note"] = rec.stopped_by
    return out


def _write(path: Path, draw: Callable[[int], np.ndarray], count: int, frame_ms: float) -> None:
    """Draw `count` frames one at a time into the file: an MP4 is fed to ffmpeg as they come, a GIF keeps each
    frame in 256 colors (0.2 MB at 640 x 360), a WebP keeps them whole until it is written."""
    kind = path.suffix.lower()
    if kind == ".mp4":
        first = draw(0)
        h, w = first.shape[:2]
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}"]
        cmd += ["-r", f"{1000.0 / frame_ms:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        cmd += ["-movflags", "+faststart", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        broken = False
        try:
            for k in range(count):
                proc.stdin.write((first if k == 0 else draw(k)).tobytes())
        except BrokenPipeError:  # ffmpeg stopped reading; what it wrote to stderr says why
            broken = True
        finally:
            with contextlib.suppress(BrokenPipeError):
                proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace").strip()
            proc.wait()
        if broken or proc.returncode != 0:
            raise RecordingError(f"ffmpeg could not write {path.name}: {err[-300:] or 'it stopped reading the frames'}")
        return
    if kind == ".webp":
        images = [Image.fromarray(draw(k)) for k in range(count)]
        images[0].save(path, save_all=True, append_images=images[1:], duration=round(frame_ms), loop=0, quality=80)
        return
    # One palette for every frame, from the first, middle and last: per-frame palettes make the floor shimmer.
    strip = Image.fromarray(np.concatenate([draw(0), draw(count // 2), draw(count - 1)], axis=0))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [Image.fromarray(draw(k)).quantize(palette=palette, dither=Image.Dither.NONE) for k in range(count)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(frame_ms), loop=0)
