"""Videos of what robots do, for an agent: move(record=...) and run_script(record=...) save a GIF or an animated WebP
(Pillow), in the world's videos folder, playing in real time.

The viewer's Record button records the browser's canvas, and an agent has no browser: asked for a video of a robot
from parts, one installed Playwright with npm, drove the viewer in headless Chromium and pressed the button for it
(198 s, 1.03M input tokens). Here the poses are kept at a fixed frame rate on every physics step, whoever steps (the
viewer's clock, a tool, a script), and drawn once the motion is over, offscreen, with a camera that follows the
robot smoothly from the viewer's opening direction.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.sim import render
from robotforge.sim.session import SimSession

FPS = 20  # frames kept per simulated second
WIDTH, HEIGHT = 640, 360
MAX_FRAMES = 300  # drawn: 15 s at FPS; a longer recording keeps every k-th frame, each shown k times as long
FORMATS = (".gif", ".webp")
# Video files Pillow does not write: a name ending in one is saved as a GIF, and the result says so. An MP4 needs
# ffmpeg, which is not one of robotforge's approved dependencies.
ELSEWHERE = (".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi")
FOLLOW_SECONDS = 0.5  # s either side the camera averages the robot's place over, so it does not shake
MIN_SPAN = np.array([0.7, 0.7, 0.25])  # m the picture keeps round a small robot: the floor sliding by shows it moving
VIEW = "default"  # the direction the camera looks from: the viewer's opening one


class RecordingError(ValueError):
    """A record argument that cannot be used (a format Pillow does not write, a folder outside the world's videos
    folder), or a video not written."""


@dataclass(frozen=True)
class Target:
    """Where a recording goes, and what the result says about the name it was given (an .mp4 saved as a GIF)."""

    path: Path
    note: str | None = None


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


def target(record: bool | str, folder: Path, robot: str | None) -> Target:
    """Where a recording goes: always in `folder`, the world's videos folder. record=true names it after the robot
    and the time, a GIF; a name is kept, a GIF without a suffix, and one ending in .mp4 (or another video type
    Pillow does not write) is saved as a GIF beside it. A full path, ~ or .. is refused, as write_script refuses
    "../evil": move(record="../../../escape.gif") wrote outside the world, so any GIF, WebP or MP4 on the disk could
    have been overwritten."""
    if record is True:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return Target(folder / f"{(robot or 'world').replace('/', '_')}-{stamp}.gif")
    name = str(record).strip()
    if not name:
        raise RecordingError('record is true or a file name such as "dog_walk.gif"')
    given = Path(name)
    inside = folder / given
    # The name read both ways, whatever this system is: on Windows /tmp/x.gif is rooted though not absolute and
    # ..\x.gif climbs out. Refused alike everywhere, a name means the same on every system.
    ways = (PurePosixPath(name), PureWindowsPath(name))
    if name.startswith("~") or any(way.anchor or ".." in way.parts for way in ways):
        raise RecordingError(
            f'record {wording.short(name)} is a file name, e.g. "walk.gif": videos are saved in the world\'s videos '
            "folder, and the result's video.path says where (copy it from there)"
        )
    if not inside.resolve().is_relative_to(folder.resolve()):  # a link in the videos folder pointing out of it
        raise RecordingError(f"record {wording.short(name)} leads out of the world's videos folder; give a file name")
    path, note = inside, None
    suffix = path.suffix.lower()
    if not suffix:
        path = path.with_suffix(".gif")
    elif suffix in ELSEWHERE:
        path = path.with_suffix(".gif")
        note = f"saved as {path.name}: robotforge writes GIF and WebP ({suffix} needs ffmpeg)"
    elif suffix not in FORMATS:
        raise RecordingError(f'record saves {", ".join(FORMATS)} files, not {path.suffix}; e.g. record="walk.gif"')
    return Target(path, note)


def _followed(model: mujoco.MjModel, robot: str | None) -> list[int]:
    """The geoms the camera follows: the robot's drawn ones; with no robot every robot's (a robot's bodies are
    named <robot>/<body>; a warehouse's loose crates would pull the camera far back), else everything that moves."""
    if robot is not None:
        return render.robot_geoms(model, robot)
    moving = model.body_weldid[model.geom_bodyid] != 0
    drawn = [g for g in range(model.ngeom) if model.geom_group[g] < 3]
    roots = {
        g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[model.geom_bodyid[g]])) or ""
        for g in drawn
    }
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


def save(
    rec: Recorder,
    renderer: render.SnapshotRenderer,
    path: Path,
    robot: str | None,
    width: int = WIDTH,
    height: int = HEIGHT,
    note: str | None = None,
) -> dict[str, Any]:
    """Draw the kept poses into a video at `path` and say what it is (with `note`, what was said about its name).
    It plays in real time: a recording longer than MAX_FRAMES keeps every k-th pose and shows each k times as long
    (each frame used to last 1/FPS whatever k was, and a 32 s trip to the warehouse dock played in 10.7 s; an agent
    slowed it down again with ffmpeg in the shell). Raises render.RenderUnavailable without OpenGL."""
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
        faded = render.occluders(model, posed, cams[k], geoms)
        return renderer.rgb_of(model, posed, cams[k], width, height, rec._version, faded=faded)

    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, draw, len(picked), every * 1000.0 / rec.fps)  # each drawn frame stands for `every` kept ones
    fps = round(rec.fps / every, 2)
    out = {
        "path": str(path),
        "seconds": round(rec.seconds, 2),
        "frames": len(picked),
        "fps": fps,
        "size_mb": round(path.stat().st_size / 1e6, 2),
        "plays": "in real time" if every == 1 else f"in real time, {fps:g} frames a second (at most {MAX_FRAMES})",
    }
    notes = [n for n in (note, rec.stopped_by) if n]
    if notes:
        out["note"] = "; ".join(notes)
    return out


def _write(path: Path, draw: Callable[[int], np.ndarray], count: int, frame_ms: float) -> None:
    """Draw `count` frames one at a time into the file: a GIF keeps each frame in 256 colors (0.2 MB at 640 x 360), a
    WebP keeps them whole until it is written."""
    from PIL import Image  # only when a video is written: Pillow is slow to import, and the MCP server imports this

    if path.suffix.lower() == ".webp":
        images = [Image.fromarray(draw(k)) for k in range(count)]
        images[0].save(path, save_all=True, append_images=images[1:], duration=round(frame_ms), loop=0, quality=80)
        return
    # One palette for every frame, from the first, middle and last: per-frame palettes make the floor shimmer.
    strip = Image.fromarray(np.concatenate([draw(0), draw(count // 2), draw(count - 1)], axis=0))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [Image.fromarray(draw(k)).quantize(palette=palette, dither=Image.Dither.NONE) for k in range(count)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(frame_ms), loop=0)
