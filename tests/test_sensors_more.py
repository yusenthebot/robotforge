"""The sensors G12b added: sonar, wheel odometry, a tactile skin, temperature and gas from sources in the world."""

import json
import math

import numpy as np
import pytest
from mcp import Client

from robotforge.design.world import World
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sensors.spec import SensorEntry

POST = {"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}}  # a fixed block, 20 cm a side
WALL_X = 2.0  # the face of the wall the post looks at


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("more")
    return playground


@pytest.fixture
def post(pg):
    pg.add_robot("post", build=POST)
    pg.add_object("wall", shape="wall", pos=[WALL_X + 0.05, 0], yaw=90, size=[3, 0.1, 1.0])
    return pg


def _read(pg, robot, name, **kwargs):
    reading, _ = pg.read_sensors(robot, [name], **kwargs)
    return reading["sensors"][name]


# ---- sonar --------------------------------------------------------------------------------------------------


def test_a_sonar_hears_the_wall_in_front_of_it(post):
    post.add_sensor("post", {"type": "sonar", "name": "ping"})
    heard = _read(post, "post", "ping")
    assert heard["distance_m"] == pytest.approx(WALL_X - 0.1, abs=0.03)  # from the post's front face
    assert heard["echoes"] >= 5


def test_ultrasonic_is_a_sonar_and_the_old_rangefinder_words_still_mean_what_they_did():
    assert SensorEntry.model_validate("ultrasonic").type == "sonar"
    assert SensorEntry.model_validate("sonar").type == "sonar"
    assert SensorEntry.model_validate("range_finder").type == "rangefinder"


def test_a_sonar_cone_hears_a_pole_a_line_misses(post):
    """A rangefinder sees along its beam; a sonar hears everything in a 30 degree cone."""
    post.add_object("pole", shape="pillar", pos=[1.0, 0.27], size=[0.12, 1.0])  # 15 degrees off the axis, at 1 m
    post.add_sensor("post", {"type": "sonar", "name": "ping"})
    post.add_sensor("post", {"type": "rangefinder", "name": "line"})
    assert _read(post, "post", "line")["distance_m"] == pytest.approx(WALL_X - 0.1, abs=0.03)
    heard = _read(post, "post", "ping")["distance_m"]
    assert heard == pytest.approx(0.9, abs=0.12)  # the pole, nearer than the wall


def test_a_sonar_is_deaf_up_close_and_to_a_wall_seen_edge_on(post):
    post.add_sensor("post", {"type": "sonar", "name": "ping", "min_range": 0.5})
    post.update_object("wall", pos=[0.45, 0.0])  # its face 0.4 m from the post's front: inside the blind zone
    close = _read(post, "post", "ping")
    assert close["distance_m"] is None and "deaf" in close["note"]
    post.remove("post/ping")
    post.add_sensor("post", {"type": "sonar", "name": "ping"})
    post.update_object("wall", pos=[2.5, 0.6], yaw=0, size=[6, 0.1, 1.0])  # along the beam, 0.55 m to the side
    edge_on = _read(post, "post", "ping")
    assert edge_on["distance_m"] is None and "too far from square" in edge_on["note"], edge_on


def test_sonar_noise_and_dropout_follow_their_knobs(post):
    post.add_sensor("post", {"type": "sonar", "name": "ping", "noise": 0.05, "rate_hz": 100})
    post.run_fast(lambda: False, 2.0)
    values = []
    for _ in range(30):
        post.run_fast(lambda: False, 0.05)
        values.append(_read(post, "post", "ping")["distance_m"])
    assert 0.005 < np.std(values) < 0.2 and np.mean(values) == pytest.approx(WALL_X - 0.1, abs=0.12)


# ---- wheel odometry -----------------------------------------------------------------------------------------


@pytest.fixture
def rover(pg):
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    return pg


def test_wheel_odometry_adds_up_the_wheels_as_the_rover_drives(rover):
    rover.add_sensor("rover", {"type": "wheel_odometry", "name": "wo"})
    rover.behaviors.move("rover", 0.3, 0.3)
    rover.run_fast(lambda: False, 6.0)
    read = _read(rover, "rover", "wo")
    truth = rover.behaviors.pose("rover")
    assert read["position_m"][0] == pytest.approx(truth["position"][0], abs=0.08)
    assert read["position_m"][1] == pytest.approx(truth["position"][1], abs=0.08)
    assert abs((read["heading_deg"] - truth["heading_deg"] + 180) % 360 - 180) < 3.0
    assert read["traveled_m"] > 1.0


def test_wheels_that_read_their_radius_off_bend_a_straight_run(rover):
    rover.add_sensor("rover", {"type": "wheel_odometry", "name": "wo", "slip": 0.08, "seed": 4})
    rover.add_sensor("rover", {"type": "wheel_odometry", "name": "good"})
    rover.behaviors.move("rover", 0.3, 0.0)
    rover.run_fast(lambda: False, 6.0)
    off, good = _read(rover, "rover", "wo"), _read(rover, "rover", "good")
    assert abs(good["heading_deg"]) < 1.0
    assert abs(off["heading_deg"]) > 1.5 * max(abs(good["heading_deg"]), 0.5)


def test_a_walking_robot_has_no_wheels_to_read(pg):
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    with pytest.raises(ValueError, match="no wheels"):
        pg.add_sensor("dog", {"type": "wheel_odometry"})


# ---- the tactile skin -----------------------------------------------------------------------------------------


def test_a_skin_feels_a_weight_on_its_top_and_says_where(post):
    post.add_sensor("post", {"type": "skin", "name": "skin", "rows": 2, "cols": 2, "faces": ["top", "front"]})
    idle = _read(post, "post", "skin")
    assert idle["contact"] is False and idle["taxels"] == 8 and "touched" not in idle
    post.add_object("weight", shape="box", size=[0.1, 0.1, 0.1], pos=[0.0, 0.0, 0.35], mass=1.0)
    post.run_fast(lambda: False, 1.5)
    felt = _read(post, "post", "skin")
    assert felt["contact"] and felt["touched"] == ["top"]
    assert felt["total_n"] == pytest.approx(9.81, rel=0.15)
    at = felt["loaded_at_m"][0]["at"]
    assert at[2] == pytest.approx(0.2, abs=0.03) and abs(at[0]) < 0.1  # on the post's top face
    assert np.array(felt["faces"]["top"]["cells_n"]).shape == (2, 2)


def test_a_skin_goes_into_world_yaml_and_back(post):
    post.add_sensor("post", {"type": "skin", "faces": ["Front", "left"], "rows": 3})
    entry = post.current_world().robots[0].sensors[-1]
    assert entry.model_dump() == {"type": "skin", "faces": ["front", "left"], "rows": 3, "name": "skin"}
    with pytest.raises(PlaygroundError, match="faces are some of"):
        post.add_sensor("post", {"type": "skin", "faces": ["sideways"]})


def test_the_viewer_gets_every_pad_of_a_skin(post):
    post.add_sensor("post", {"type": "skin", "rows": 2, "cols": 2, "faces": ["top", "front"]})
    message = post.sensors.viewer_message()
    entry = next(e for e in message["sensors"] if e["type"] == "skin")
    assert [p["face"] for p in entry["pads"]] == ["top", "front"]
    assert all(len(p["force_n"]) == 4 and "pad_quat" in p and "cells_local" in p for p in entry["pads"])


# ---- heat and gas ---------------------------------------------------------------------------------------------


def _heater(pg, at=(1.0, 0.0), strength=100):
    pg.add_object("heater", shape="heat_source", pos=list(at), strength=strength)


def test_a_thermometer_reads_the_room_and_rises_toward_a_heater(post):
    post.add_sensor("post", {"type": "temperature", "name": "t"})
    reading = _read(post, "post", "t")
    assert reading["temperature_c"] == pytest.approx(20.0) and "no heat source" in reading["note"]
    post.add_object("heater", shape="heat_source", pos=[-1.0, 0.0], strength=100)
    near = _read(post, "post", "t")["temperature_c"]
    post.update_object("heater", pos=[-3.0, 0.0])
    far = _read(post, "post", "t")["temperature_c"]
    assert 20.5 < far < near
    # the model: 20 + strength / (1 + (d / 0.35 m)^2), d from the heater's element to the sensor
    d = math.dist([-1.0, 0.0, 0.15], _sensor_at(post, "t"))
    assert near == pytest.approx(20 + 100 / (1 + (d / 0.35) ** 2), abs=0.5)


def _sensor_at(pg, name, robot="post"):
    import mujoco

    m, d = pg.session.model, pg.session.data
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/rf_{name}")]


def test_a_wall_between_a_heater_and_a_thermometer_shields_it(post):
    post.update_object("wall", pos=[WALL_X + 0.05, 0])
    post.add_object("heater", shape="heat_source", pos=[1.2, 0.0], strength=100)
    post.add_object("screen", shape="wall", pos=[0.6, 0.0], yaw=90, size=[1.5, 0.05, 1.0])
    post.add_sensor("post", {"type": "temperature", "name": "t"})
    shielded = _read(post, "post", "t")["temperature_c"]
    post.remove("screen")
    open_air = _read(post, "post", "t")["temperature_c"]
    assert shielded - 20 == pytest.approx(0.2 * (open_air - 20), rel=0.15)


def test_gas_thins_with_distance_and_the_sensor_is_slow_to_follow(rover):
    rover.add_object("vent", shape="gas_source", pos=[4.0, 0.0], strength=600)
    rover.add_sensor("rover", {"type": "gas", "name": "g", "response_s": 4.0, "on": "chassis", "at": "center"})
    rover.run_fast(lambda: False, 0.1)
    far = _read(rover, "rover", "g")["gas_ppm"]
    d = math.dist([4.0, 0.0, 0.12], _sensor_at(rover, "g", "rover"))
    assert far == pytest.approx(600 * math.exp(-d / 1.2), rel=0.1)
    rover.update_object("vent", pos=[0.6, 0.0])  # the plume comes close: a slow sensor does not jump to it
    rover.run_fast(lambda: False, 0.5)
    just_after = _read(rover, "rover", "g")["gas_ppm"]
    rover.run_fast(lambda: False, 24.0)
    settled = _read(rover, "rover", "g")["gas_ppm"]
    d2 = math.dist([0.6, 0.0, 0.12], _sensor_at(rover, "g", "rover"))
    assert settled == pytest.approx(600 * math.exp(-d2 / 1.2), rel=0.1)
    assert just_after < far + 0.4 * (settled - far)


def test_sources_are_props_that_save_only_what_they_use(post):
    post.add_object("vent", shape="gas_source", pos=[1.0, 1.0])
    post.add_object("heater", shape="heat_source", pos=[-1.0, 1.0], strength=55)
    text = post.current_world().to_yaml()
    assert text.count("strength") == 1 and "strength: 55" in text
    plain = World.model_validate({"objects": [{"name": "b", "shape": "box"}]}).to_yaml()
    assert "strength" not in plain
    with pytest.raises(PlaygroundError, match="strength is for a heat_source or gas_source"):
        post.add_object("crate", shape="box", strength=5)


def test_nothing_collides_with_a_source(pg):
    pg.add_object("heater", shape="heat_source", pos=[0.0, 0.0])
    pg.add_object("ball", shape="sphere", size=[0.1], pos=[0.0, 0.0, 0.5], mass=0.5)
    pg.run_fast(lambda: False, 1.5)
    z = pg.session.data.xpos[__import__("mujoco").mj_name2id(pg.session.model, 1, "ball")][2]
    assert z < 0.08  # it fell through the heater's casing to the floor


# ---- the tools -----------------------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_the_tools_add_and_read_the_new_sensors(post):
    server = build_server(post, 0)
    async with Client(server) as client:
        await client.call_tool(
            "add_object", {"name": "heater", "shape": "heat_source", "pos": [1.0, 1.0], "strength": 90}
        )
        for fields in (
            {"robot": "post", "type": "ultrasonic"},
            {"robot": "post", "type": "thermometer"},
            {"robot": "post", "type": "gas_sensor", "options": {"response_s": 2}},
            {"robot": "post", "type": "tactile_skin", "options": {"rows": 2, "cols": 2}},
        ):
            added = await client.call_tool("add_sensor", fields)
            assert not added.is_error, added.content[0].text
        read = await client.call_tool("read_sensors", {"robot": "post"})
        sensors = json.loads(read.content[0].text)["sensors"]
        assert sensors["sonar"]["distance_m"] > 1.5
        assert sensors["temp"]["temperature_c"] > 20.0
        assert sensors["gas"]["gas_ppm"] == pytest.approx(0.0, abs=1e-6) and "gas source" in sensors["gas"]["note"]
        assert sensors["skin"]["taxels"] == 4 * 5
        bad = await client.call_tool("add_sensor", {"robot": "post", "type": "sonar", "options": {"rows": 3}})
        assert bad.is_error and "sonar sensor takes no 'rows'" in bad.content[0].text
