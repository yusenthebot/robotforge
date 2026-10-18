"""Videos for an agent: move(record=...) and run_script(record=...) keep the poses at a fixed frame rate while time
passes and draw them into a GIF, a WebP or an MP4. Asked for a video of a robot from parts, an agent in the eval
installed Playwright with npm to press the viewer's Record button in headless Chromium."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from PIL import Image, ImageSequence

from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import recording
from robotforge.sim.session import SimSession

CAR = {
    "base": {"size": [0.3, 0.2, 0.08]},
    "motor": "jga25",
    "parts": [
        {"name": "front_left_wheel", "at": [0.1, 0.12, -0.03], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "rear_left_wheel", "at": [-0.1, 0.12, -0.03], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
    ],
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("garage")
    playground.add_robot("pip", build=CAR)
    return playground


def _gif(path: Path) -> list[np.ndarray]:
    with Image.open(path) as im:
        return [np.asarray(f.convert("RGB"), dtype=float) for f in ImageSequence.Iterator(im)]


def test_poses_are_kept_at_the_frame_rate_whoever_steps(pg):
    rec = recording.Recorder(pg.session).start()
    pg.behaviors.move("pip", 0.3, 0.0, None)
    pg.run_fast(lambda: False, 2.0)
    rec.stop()
    assert len(rec.times) == pytest.approx(2.0 * recording.FPS + 1, abs=1) and rec.seconds == pytest.approx(2.0, 0.05)
    assert rec.qpos[-1][0] - rec.qpos[0][0] > 0.3  # the base's x: it drove forward while the poses were kept
    pg.run_fast(lambda: False, 1.0)  # detached: nothing more is kept
    assert rec.seconds == pytest.approx(2.0, 0.05)


def test_an_edit_ends_the_recording_and_says_so(pg):
    rec = recording.Recorder(pg.session).start()
    pg.run_fast(lambda: False, 0.5)
    pg.add_object("cone", shape="cylinder", size=[0.2, 0.3], pos=[1.0, 0.0])  # a new model: the poses no longer fit
    pg.run_fast(lambda: False, 0.5)
    assert rec.seconds == pytest.approx(0.5, abs=0.06) and "world changed" in rec.stopped_by
    assert rec._grab not in pg.session.step_hooks


def test_where_a_recording_goes(tmp_path, monkeypatch):
    folder = tmp_path / "videos"
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ffmpeg")  # installed, and still not used unasked
    made = recording.target(True, folder, "pip")
    assert made.parent == folder and made.name.startswith("pip-") and made.suffix == ".gif"
    assert recording.target("lap", folder, "pip") == folder / "lap.gif"
    assert recording.target("lap.webp", folder, "pip") == folder / "lap.webp"
    assert recording.target(str(tmp_path / "share" / "lap.gif"), folder, "pip") == tmp_path / "share" / "lap.gif"
    with pytest.raises(recording.RecordingError, match="not .avi"):
        recording.target("lap.avi", folder, "pip")


def test_an_mp4_calls_ffmpeg_only_where_the_user_turned_it_on(tmp_path, monkeypatch):
    """ffmpeg is not one of robotforge's approved dependencies: record=true always makes a GIF, and a name ending in
    .mp4 is refused unless ROBOTFORGE_FFMPEG=1 is set and ffmpeg is installed."""
    folder = tmp_path / "videos"
    monkeypatch.delenv(recording.FFMPEG_SWITCH, raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ffmpeg")
    with pytest.raises(recording.RecordingError, match="ROBOTFORGE_FFMPEG=1"):
        recording.target("lap.mp4", folder, "pip")
    monkeypatch.setenv(recording.FFMPEG_SWITCH, "1")
    assert recording.target("lap.mp4", folder, "pip") == folder / "lap.mp4"
    assert recording.target(True, folder, "pip").suffix == ".gif"
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(recording.RecordingError, match="needs ffmpeg"):
        recording.target("lap.mp4", folder, "pip")


@pytest.mark.skipif(sys.platform == "win32", reason="the stand-in ffmpeg is a shell script")
def test_ffmpeg_that_stops_reading_says_why(tmp_path, monkeypatch):
    """A broken pipe used to be reported as "Broken pipe": what ffmpeg said on stderr was never read."""
    fake = tmp_path / "bin" / "ffmpeg"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho 'Unknown encoder libx264' >&2\nexit 1\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")
    frame = np.zeros((recording.HEIGHT, recording.WIDTH, 3), np.uint8)  # 0.7 MB: more than a pipe holds
    with pytest.raises(recording.RecordingError, match="Unknown encoder libx264"):
        recording._write(tmp_path / "lap.mp4", lambda k: frame, 40, 50.0)


def test_a_gif_shows_the_robot_moving(needs_gl, pg, tmp_path):
    rec = recording.Recorder(pg.session).start()
    pg.behaviors.move("pip", 0.3, 0.6, None)
    pg.run_fast(lambda: False, 3.0)
    rec.stop()
    saved = recording.save(rec, pg.renderer, tmp_path / "lap.gif", "pip")
    frames = _gif(tmp_path / "lap.gif")
    assert saved["frames"] == len(frames) == len(rec.times) and saved["seconds"] == pytest.approx(3.0, 0.05)
    assert frames[0].shape == (recording.HEIGHT, recording.WIDTH, 3)
    assert np.abs(frames[0] - frames[-1]).mean() > 1.0  # it turned and the floor slid by: the picture changed
    with Image.open(tmp_path / "lap.gif") as im:
        assert im.info["duration"] == 1000 / recording.FPS


def test_a_long_recording_plays_faster_within_the_frame_limit(needs_gl, pg, tmp_path, monkeypatch):
    monkeypatch.setattr(recording, "MAX_FRAMES", 20)
    rec = recording.Recorder(pg.session).start()
    pg.run_fast(lambda: False, 2.0)
    rec.stop()
    saved = recording.save(rec, pg.renderer, tmp_path / "long.gif", "pip")
    assert saved["frames"] <= 20 and saved["plays"].startswith("3x faster")


@pytest.mark.anyio
async def test_move_and_run_script_save_videos(needs_gl, pg):
    async with Client(build_server(pg, find_free_port(8790))) as client:
        moved = await client.call_tool("move", {"robot": "pip", "speed": 0.3, "seconds": 2, "record": True})
        out = json.loads(moved.content[0].text)
        video = Path(out["video"]["path"])
        assert video.parent == pg.store.dir / "videos" and video.exists() and out["video"]["frames"] >= 40
        code = "from robotforge.script import connect\nsim = connect()\nsim.move('pip', 0.3, 1.0)\nsim.step(2)\n"
        await client.call_tool("write_script", {"name": "spin", "code": code})
        ran = json.loads(
            (await client.call_tool("run_script", {"name": "spin", "record": "spin.webp"})).content[0].text
        )
        assert ran["ok"] and Path(ran["video"]["path"]).name == "spin.webp" and ran["video"]["seconds"] >= 2
        refused = await client.call_tool("move", {"robot": "pip", "seconds": None, "record": True})
        assert refused.is_error and "record needs seconds" in refused.content[0].text
        guessed = await client.call_tool("move", {"robot": "pip", "video": True})
        assert guessed.is_error and "did you mean record" in guessed.content[0].text


@pytest.mark.anyio
async def test_a_video_that_fails_to_draw_keeps_the_move_result(pg, monkeypatch):
    """Only some errors were caught while drawing: any other turned a finished move into a tool error."""

    def broken(*args, **kwargs):
        raise RuntimeError("the renderer fell over")

    monkeypatch.setattr(recording, "save", broken)
    async with Client(build_server(pg, find_free_port(8790))) as client:
        moved = await client.call_tool("move", {"robot": "pip", "speed": 0.3, "seconds": 1, "record": True})
    assert not moved.is_error
    out = json.loads(moved.content[0].text)
    assert out["stopped"] and "RuntimeError: the renderer fell over" in out["video"]["error"]


def test_what_stands_between_the_camera_and_the_robot_is_drawn_see_through(pg):
    """A Go2 walking in the park went behind a bench and out of sight in its video."""
    import mujoco

    from robotforge.sim import render
    from robotforge.sim.placement import owner

    # The camera looks from the viewer's opening direction: from +x, -y and above. A screen stands in its way, square
    # across the line of sight; a wall of the same size stands off to the side.
    pg.add_object("screen", shape="wall", size=[1.0, 0.05, 0.6], pos=[0.45, -0.55], yaw=38)
    pg.add_object("side", shape="wall", size=[1.0, 0.05, 0.6], pos=[-1.2, -0.6], yaw=38)
    model, data = pg.session.model, pg.session.data
    mujoco.mj_forward(model, data)
    geoms = recording._followed(model, "pip")
    cam = recording._cameras(model, [render._geom_box(model, data, geoms)], 0.05)[0]
    faded = recording.occluders(model, data, cam, geoms)
    assert {owner(model, g) for g in faded} == {"screen"}
    pg.update_object("screen", pos=[2.5, 2.5])  # out of the way: nothing to fade
    model, data = pg.session.model, pg.session.data
    mujoco.mj_forward(model, data)
    assert not recording.occluders(model, data, cam, recording._followed(model, "pip"))
