"""Videos of what robots do, for an agent: move(record=...) and run_script(record=...) save a GIF, an animated WebP
or, with ffmpeg on PATH, an MP4.

The viewer's Record button records the browser's canvas, and an agent has no browser: asked for a video of a robot
from parts, one installed Playwright with npm, drove the viewer in headless Chromium and pressed the button for it
(198 s, 1.03M input tokens). Here the poses are kept at a fixed frame rate on every physics step, whoever steps (the
viewer's clock, a tool, a script), and drawn once the motion is over, offscreen, with a camera that follows the
robot smoothly from the viewer's opening direction.
"""

from __future__ import annotations

import datetime as dt
import math
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
FOLLOW_SECONDS = 0.6  # the camera eases toward the robot over about this long, so it does not shake with each step
MIN_SPAN = np.array([1.0, 1.0, 0.3])  # m of ground the picture keeps round a small robot, so its motion shows
VIEW = "default"  # the direction the camera looks from: the viewer's opening one


class RecordingError(ValueError):
    """A record argument that cannot be used (an unknown format, an MP4 without ffmpeg)."""


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


def target(record: bool | str, folder: Path, robot: str | None) -> Path:
    """Where a recording goes: record=true names it after the robot and the time, in `folder`; a name without a
    folder goes in `folder` too; a path ending in .gif, .webp or .mp4 is used as given (~ is the home folder)."""
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
    if path.suffix.lower() == ".mp4" and shutil.which("ffmpeg") is None:
        raise RecordingError("an .mp4 needs ffmpeg on PATH, which this computer does not have; .gif and .webp work")
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
    needs (the same all along), looking at the robot's middle eased over FOLLOW_SECONDS."""
    azimuth, elevation, fill = render.VIEWS[VIEW]
    if not boxes or any(b is None for b in boxes):
        return [VIEW] * len(boxes)
    span = np.maximum(np.max([high - low for low, high in boxes], axis=0), MIN_SPAN)
    fovy = float(model.vis.global_.fovy)
    distance = render._fit_distance(-span / 2, span / 2, azimuth, elevation, fill, fovy, WIDTH / HEIGHT)
    ease = 1.0 - math.exp(-dt_s / FOLLOW_SECONDS)
    look = (boxes[0][0] + boxes[0][1]) / 2
    cams = []
    for low, high in boxes:
        look = look + ((low + high) / 2 - look) * ease
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = look
        cams.append(cam)
    return cams


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
        return renderer.rgb_of(model, _pose(model, scratch, rec, picked[k]), cams[k], width, height, rec._version)

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
        try:
            for k in range(count):
                proc.stdin.write((first if k == 0 else draw(k)).tobytes())
        finally:
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            proc.wait()
        if proc.returncode != 0:
            raise RecordingError(f"ffmpeg could not write {path.name}: {err[-300:]}")
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
