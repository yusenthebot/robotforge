"""The cameras G12b added: a stereo pair, a fisheye lens and an event camera. They draw pictures, so the readings need
OpenGL; the options, the words and what the viewer is sent are checked without it."""

import io
import json

import numpy as np
import pytest
from mcp import Client
from PIL import Image

from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sensors.spec import SensorEntry, base_name, resolved

POST = {"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}}
WALL_X = 2.0  # the face of the wall the post looks at


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("optics")
    return playground


@pytest.fixture
def post(pg):
    pg.add_robot("post", build=POST)
    pg.add_object("wall", shape="wall", pos=[WALL_X + 0.05, 0], yaw=90, size=[3, 0.1, 1.0])
    return pg


def _read(pg, robot, name, **kwargs):
    reading, pictures = pg.read_sensors(robot, [name], **kwargs)
    return reading["sensors"][name], pictures


def _picture(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


# ---- the options and the words (no OpenGL) ---------------------------------------------------------------------


def test_the_words_for_the_new_cameras():
    for word, mode, lens in (
        ("event camera", "event", None),
        ("DVS", "event", None),
        ("stereo", "stereo", None),
        ("fisheye", "rgb", "fisheye"),
    ):
        entry = SensorEntry.model_validate(word)
        assert (entry.mode or "rgb", entry.lens) == (mode, lens), word
    assert resolved(SensorEntry.model_validate("fisheye"))["fov"] == 180.0
    assert resolved(SensorEntry.model_validate("stereo"))["baseline"] == 0.12
    event = resolved(SensorEntry.model_validate("event"))
    assert (event["threshold"], event["window_s"]) == (0.15, 0.03)
    assert base_name("camera", "stereo") == "stereo_camera"
    assert base_name("camera", None, None, "fisheye") == "fisheye_camera"


def test_a_camera_takes_the_options_of_its_kind_only():
    with pytest.raises(ValueError, match="baseline is for a stereo camera"):
        SensorEntry.model_validate({"type": "camera", "baseline": 0.1})
    with pytest.raises(ValueError, match="threshold is for a event camera|threshold is for an event camera"):
        SensorEntry.model_validate({"type": "camera", "mode": "depth", "threshold": 0.2})
    with pytest.raises(ValueError, match="wider than a pinhole"):
        SensorEntry.model_validate({"type": "camera", "fov": 180})
    with pytest.raises(ValueError, match="rgb picture"):
        SensorEntry.model_validate({"type": "camera", "mode": "depth", "lens": "fisheye"})
    assert SensorEntry.model_validate({"type": "camera", "lens": "fisheye", "fov": 200}).fov == 200


def test_the_viewer_gets_the_fields_the_new_cameras_draw_with(post):
    post.add_sensor("post", {"type": "stereo", "name": "pair"})
    post.add_sensor("post", {"type": "fisheye", "name": "wide"})
    post.add_sensor("post", {"type": "event", "name": "dvs"})
    message = post.sensors.viewer_message()
    by_name = {e["name"]: e for e in message["sensors"]}
    assert by_name["post/pair"]["baseline_m"] == 0.12 and by_name["post/pair"]["mode"] == "stereo"
    assert by_name["post/wide"]["lens"] == "fisheye" and by_name["post/wide"]["fov_deg"] == 180.0
    assert by_name["post/dvs"]["threshold"] == 0.15 and by_name["post/dvs"]["window_s"] == 0.03


# ---- stereo -------------------------------------------------------------------------------------------------


def test_a_stereo_camera_measures_depth_through_a_disparity(post, needs_gl):
    post.add_sensor("post", {"type": "stereo", "name": "pair", "baseline": 0.12})
    reading, _ = _read(post, "post", "pair")
    assert reading["mode"] == "stereo" and reading["baseline_m"] == 0.12
    # the wall from the lens: 1.9 m less the lens's little forward offset, as a disparity in steps of 1/8 pixel
    assert reading["center_m"] == pytest.approx(WALL_X - 0.1 - 0.02, abs=0.06)
    assert reading["reaches_m"] == pytest.approx(reading["focal_px"] * 0.12, rel=0.01)
    assert reading["depth_error_m"]["3"] > 8 * reading["depth_error_m"]["1"]  # the error grows with Z squared


def test_stereo_loses_depth_beyond_its_reach_and_its_noise_grows_with_distance(post, needs_gl):
    post.update_object("wall", pos=[8.05, 0.0])  # 7.9 m away
    post.add_sensor("post", {"type": "stereo", "name": "pair", "baseline": 0.03, "range": 10})
    far, _ = _read(post, "post", "pair")
    assert far["reaches_m"] < 7.0 and far["center_m"] is None  # a 3 cm baseline reaches about f B = 6 m
    post.remove("post/pair")
    post.add_sensor(
        "post", {"type": "stereo", "name": "pair", "baseline": 0.2, "range": 10, "noise": 0.005, "rate_hz": 100}
    )
    post.run_fast(lambda: False, 0.5)
    spread = {}
    for wall_x in (2.05, 5.05):
        post.update_object("wall", pos=[wall_x, 0.0])
        post.run_fast(lambda: False, 0.5)
        centers = []
        for _ in range(12):
            post.run_fast(lambda: False, 0.03)
            centers.append(_read(post, "post", "pair")[0]["center_m"])
        spread[wall_x] = float(np.std(centers))
    assert spread[5.05] > 3 * spread[2.05]  # about (4.9 / 1.9) squared, more than six times, with a spread of twelve


def test_a_stereo_picture_is_the_left_and_right_images_side_by_side(post, needs_gl):
    post.add_sensor("post", {"type": "stereo", "name": "pair", "width": 160, "height": 120})
    _, pictures = _read(post, "post", "pair", image=True)
    image = _picture(pictures[0][1])
    assert image.shape == (120, 320, 3)
    assert np.abs(image[:, :160].astype(int) - image[:, 160:].astype(int)).sum() > 0  # two views, not one twice


def test_a_stereo_camera_gives_a_point_cloud_like_a_depth_camera(post, needs_gl):
    post.add_sensor("post", {"type": "stereo", "name": "pair"})
    reading, _ = _read(post, "post", "pair", points=150)
    cloud = reading["cloud"]
    assert 0 < cloud["shown"] <= 150
    xs = [p[0] for p in cloud["points_m"]]
    assert sum(abs(x - WALL_X) < 0.1 for x in xs) > 20  # most of it is the wall's face


# ---- fisheye -----------------------------------------------------------------------------------------------


def test_a_fisheye_sees_what_a_pinhole_cannot(post, needs_gl):
    post.add_object("green", shape="box", size=[0.2, 0.2, 0.4], pos=[0.5, 1.6], color="green")  # 75 degrees to the left
    post.add_sensor("post", {"type": "camera", "name": "narrow", "width": 160, "height": 120, "fov": 60})
    post.add_sensor("post", {"type": "fisheye", "name": "wide", "width": 240, "height": 240})
    _, narrow = _read(post, "post", "narrow", image=True)
    reading, wide = _read(post, "post", "wide", image=True)
    assert reading["lens"] == "fisheye" and reading["fov_deg"] == 180.0

    def green(image):
        image = image.astype(int)
        return int((image[..., 1] - np.maximum(image[..., 0], image[..., 2]) > 30).sum())

    assert green(_picture(narrow[0][1])) < 20 and green(_picture(wide[0][1])) > 40
    picture = _picture(wide[0][1])
    assert picture[0, 0].sum() == 0 and picture[120, 120].sum() > 0  # black outside the circle, the scene inside it


# ---- the event camera --------------------------------------------------------------------------------------------


def test_an_event_camera_fires_where_things_move_and_is_silent_when_nothing_does(pg, needs_gl):
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    pg.add_object("pole", shape="pillar", pos=[1.2, 0.0], size=[0.2, 0.8], color="white")
    pg.add_sensor("rover", {"type": "event", "name": "dvs", "width": 160, "height": 120})
    pg.run_fast(lambda: False, 0.3)
    still, _ = _read(pg, "rover", "dvs")
    assert still["events"] == 0, still
    pg.behaviors.move("rover", 0.4, 0.8)
    pg.run_fast(lambda: False, 1.0)
    moving, pictures = _read(pg, "rover", "dvs", image=True)
    assert moving["events"] > 100 and moving["on"] > 0 and moving["off"] > 0
    assert moving["rate_ev_s"] > 1000 and "centroid_px" in moving
    image = _picture(pictures[0][1])
    assert (image[..., 0] > image[..., 2] + 60).any() and (image[..., 2] > image[..., 0] + 60).any()  # red and blue


def test_event_noise_gives_the_events_of_a_still_picture(pg, needs_gl):
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    pg.add_object("pole", shape="pillar", pos=[1.2, 0.0], size=[0.2, 0.8], color="white")
    pg.add_sensor("rover", {"type": "event", "name": "dvs", "width": 160, "height": 120, "noise": 6})
    pg.run_fast(lambda: False, 0.3)
    noisy, _ = _read(pg, "rover", "dvs")
    assert noisy["events"] > 20


# ---- the tool ------------------------------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_the_tool_adds_the_new_cameras_by_their_words(post):
    server = build_server(post, 0)
    async with Client(server) as client:
        for kind in ("stereo camera", "event camera", "fisheye"):
            added = await client.call_tool("add_sensor", {"robot": "post", "type": kind})
            assert not added.is_error, added.content[0].text
            assert json.loads(added.content[0].text)["type"] == "camera"
        assert {"stereo_camera", "event_camera", "fisheye_camera"} <= set(post.sensors.names("post"))
