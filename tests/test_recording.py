"""Videos for an agent: move(record=...) and run_script(record=...) keep the poses at a fixed frame rate while time
passes and draw them into a GIF or a WebP in the world's videos folder, playing in real time. Asked for a video of a
robot from parts, an agent in the eval installed Playwright with npm to press the viewer's Record button in headless
Chromium."""

from __future__ import annotations

import json
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


def test_where_a_recording_goes(tmp_path):
    folder = tmp_path / "videos"
    made = recording.target(True, folder, "pip")
    assert made.path.parent == folder and made.path.name.startswith("pip-") and made.path.suffix == ".gif"
    assert recording.target("lap", folder, "pip") == recording.Target(folder / "lap.gif")
    assert recording.target("lap.webp", folder, "pip").path == folder / "lap.webp"
    assert recording.target("takes/lap.gif", folder, "pip").path == folder / "takes" / "lap.gif"
    with pytest.raises(recording.RecordingError, match="not .png"):
        recording.target("lap.png", folder, "pip")


@pytest.mark.parametrize("name", ["../../../escape.gif", "takes/../../escape.gif", "/tmp/escape.gif", "~/escape.gif"])
def test_a_recording_stays_in_the_videos_folder(tmp_path, name):
    """Regression: move(record="../../../escape.gif") wrote outside the world, and a full path could overwrite any
    GIF, WebP or MP4 on the disk. It is refused, as write_script refuses "../evil"."""
    with pytest.raises(recording.RecordingError, match="is a file name"):
        recording.target(name, tmp_path / "videos", "pip")


def test_a_link_out_of_the_videos_folder_is_refused(tmp_path):
    folder, away = tmp_path / "videos", tmp_path / "away"
    folder.mkdir()
    away.mkdir()
    (folder / "out").symlink_to(away, target_is_directory=True)
    with pytest.raises(recording.RecordingError, match="leads out of the world's videos folder"):
        recording.target("out/lap.gif", folder, "pip")


def test_an_mp4_is_saved_as_a_gif_and_the_result_says_so(tmp_path):
    """ffmpeg is not one of robotforge's approved dependencies, so robotforge never runs it: a name ending in .mp4
    (or another video type Pillow does not write) is saved as a GIF beside it, with a note, not refused."""
    made = recording.target("lap.mp4", tmp_path, "pip")
    assert made.path == tmp_path / "lap.gif" and "saved as lap.gif" in made.note and ".mp4 needs ffmpeg" in made.note
    assert recording.target("lap.MOV", tmp_path, "pip").path == tmp_path / "lap.gif"


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
    assert saved["fps"] == recording.FPS and saved["plays"] == "in real time"


def test_a_long_recording_plays_in_real_time_within_the_frame_limit(needs_gl, pg, tmp_path, monkeypatch):
    """Regression: every frame lasted 1/FPS however many poses it stood for, so a 32 s trip to the warehouse dock
    played in 10.7 s, and an agent slowed it down again with ffmpeg in the shell. Each frame now lasts as long as the
    poses it stands for."""
    monkeypatch.setattr(recording, "MAX_FRAMES", 20)
    rec = recording.Recorder(pg.session).start()
    pg.run_fast(lambda: False, 2.0)
    rec.stop()
    saved = recording.save(rec, pg.renderer, tmp_path / "long.gif", "pip")
    with Image.open(tmp_path / "long.gif") as im:
        plays = sum(f.info["duration"] for f in ImageSequence.Iterator(im)) / 1000
    assert saved["frames"] <= 20 and saved["fps"] == pytest.approx(recording.FPS / 3, abs=0.01)
    assert plays == pytest.approx(saved["seconds"], abs=3 / recording.FPS) and saved["plays"].startswith("in real time")


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
        before = pg.behaviors.pose("pip")["position"]
        away = {"robot": "pip", "speed": 0.3, "seconds": 1, "record": "../../escape.gif"}
        escaped = await client.call_tool("move", away)
        assert escaped.is_error and "is a file name" in escaped.content[0].text
        assert not (pg.store.dir.parent / "escape.gif").exists()
        assert pg.behaviors.pose("pip")["position"] == before  # refused before anything moved
        shared = json.loads((await client.call_tool("move", {**away, "record": "share.mp4"})).content[0].text)
        assert shared["video"]["path"].endswith("share.gif") and "needs ffmpeg" in shared["video"]["note"]


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
