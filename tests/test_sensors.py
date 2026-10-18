"""The sensor suite: every type against known geometry, the realism knobs, world.yaml, the tools and the viewer."""

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client
from starlette.testclient import TestClient

from robotforge.design.world import World
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sensors import declare
from robotforge.sensors.spec import SensorEntry
from robotforge.viewer.server import ViewerHub, create_app

sys.path.insert(0, str(Path(__file__).parent))
from built_robots import SCOUT  # noqa: E402

POST = {"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}}  # a fixed block, 20 cm a side
BOX = {"base": {"shape": "box", "size": [0.2, 0.2, 0.1], "mass": 2.0}}  # a loose 2 kg block
TRAY = {
    "base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True},
    "parts": [{"name": "plate", "at": "top", "joint": "fixed", "shape": "box", "size": [0.3, 0.3, 0.02]}],
}
WALL_X = 2.0  # the face of the wall the post looks at


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("sensing")
    return playground


@pytest.fixture
def post(pg):
    """A post 2 m from a wall: the wall's face is at x = 2.0, the post's middle at the origin."""
    pg.add_robot("post", build=POST)
    pg.add_object("wall", shape="wall", pos=[WALL_X + 0.05, 0], yaw=90, size=[3, 0.1, 1.0])
    return pg


def _site_x(pg, name: str) -> float:
    m, d = pg.session.model, pg.session.data
    return float(d.site_xpos[m.site(name).id][0])


def _read(pg, robot: str, name: str, **kwargs) -> dict:
    reading, _ = pg.read_sensors(robot, [name], **kwargs)
    return reading["sensors"][name]


# ---- the schema and world.yaml -----------------------------------------------------------------------


def test_sensors_round_trip_world_yaml_and_old_worlds_stay_as_they_were():
    world = World.model_validate(
        {
            "robots": [
                {"name": "dog", "template": "quadruped", "sensors": ["imu", {"type": "3D lidar", "rate_hz": 10}]},
                {"name": "arm", "template": "arm"},
            ]
        }
    )
    text = world.to_yaml()
    assert World.from_yaml(text) == world
    dog, arm = World.from_yaml(text).robots
    assert [s.type for s in dog.sensors] == ["imu", "lidar3d"] and dog.sensors[1].rate_hz == 10
    assert "sensors" not in text.split("- name: arm")[1]  # a robot without sensors is saved as before
    assert "channels" not in text  # defaults are not written: saved as given


def test_types_are_read_the_way_agents_word_them_and_wrong_options_are_refused():
    assert SensorEntry.model_validate({"type": "depth camera"}).mode == "depth"
    assert SensorEntry.model_validate({"type": "3D-lidar"}).type == "lidar3d"
    with pytest.raises(ValueError, match="an imu sensor takes no 'rays'"):
        SensorEntry.model_validate({"type": "imu", "rays": 16})
    with pytest.raises(ValueError, match="channels"):
        SensorEntry.model_validate({"type": "camera", "noise": {"accel": 0.1}})
    with pytest.raises(ValueError, match="rays a scan"):
        SensorEntry.model_validate({"type": "lidar3d", "channels": 64, "rays": 1024})


# ---- each type against known geometry ------------------------------------------------------------------


def test_2d_and_3d_lidars_measure_a_wall_at_a_known_distance(post):
    post.add_sensor("post", {"type": "lidar"})
    post.add_sensor("post", {"type": "lidar3d", "name": "scanner", "vfov": [-20, 10]})
    ring = _read(post, "post", "lidar")
    truth = WALL_X - _site_x(post, "post/rf_lidar")
    assert ring["ahead"]["distance_m"] == pytest.approx(truth, abs=0.002)
    assert ring["ahead"]["bearing_deg"] == pytest.approx(0, abs=0.6)
    assert len(ring["sectors_m"]) == 36 and ring["sectors_m"][18] is None  # nothing behind it
    scan = _read(post, "post", "scanner")
    assert scan["ahead"]["distance_m"] == pytest.approx(WALL_X - _site_x(post, "post/rf_scanner"), abs=0.002)
    assert scan["ground_points"] > 100  # rings that point down see the floor, which is not an obstacle
    low, high = scan["box_m"]
    assert low[2] == pytest.approx(0, abs=0.01) and high[2] <= 1.0 + 0.01  # from the floor up to the wall's top


def test_a_noisy_lidar_still_reads_the_wall_where_it_is(post):
    """The nearest of hundreds of noisy rays is the one whose noise came out shortest: with 2 cm of noise the
    nearest obstacle read 5 cm short of the wall. The reading now takes the median of the rays around it."""
    post.add_sensor("post", {"type": "lidar3d", "noise": 0.02, "rate_hz": 10})
    truth = WALL_X - _site_x(post, "post/rf_lidar")
    ahead = []
    for _ in range(10):
        post.simulate(0.1)
        ahead.append(_read(post, "post", "lidar")["ahead"]["distance_m"])
    assert np.mean(ahead) == pytest.approx(truth, abs=0.01)


def test_a_lidar_does_not_see_its_own_robot(pg):
    """A 3D lidar on a robot dog's back: its lower rings meet the dog's own head and legs, which real lidars
    mask out. Without the mask the nearest obstacle was the dog itself, 20 cm away."""
    pg.add_robot("dog", template="quadruped")
    pg.add_sensor("dog", {"type": "lidar3d", "vfov": [-30, 5]})
    scan = _read(pg, "dog", "lidar")
    assert scan["nearest"] is None or scan["nearest"]["distance_m"] > 1.0


def test_rangefinder_reads_the_distance_along_its_beam(post):
    post.add_sensor("post", {"type": "rangefinder"})
    truth = WALL_X - _site_x(post, "post/rf_range")
    # The wall's drawn plaster stands 6 mm proud of its collision box; a rangefinder sees what is drawn.
    assert _read(post, "post", "range")["distance_m"] == pytest.approx(truth, abs=0.008)


def test_depth_and_segmentation_cameras_see_the_wall(post, needs_gl):
    post.add_sensor("post", {"type": "depth camera"})
    post.add_sensor("post", {"type": "camera", "mode": "segmentation", "name": "seg"})
    m, d = post.session.model, post.session.data
    eye = float(d.cam_xpos[m.camera("post/depth_camera").id][0])
    reading, pictures = post.read_sensors("post", ["depth_camera", "seg"], image=True)
    depth = reading["sensors"]["depth_camera"]
    assert depth["center_m"] == pytest.approx(WALL_X - eye, abs=0.01)
    assert depth["nearest"]["distance_m"] == pytest.approx(WALL_X - eye, abs=0.02)  # the floor is not an obstacle
    seen = {s["name"]: s["share"] for s in reading["sensors"]["seg"]["seen"]}
    assert seen["wall"] > 0.3 and "floor" in seen
    assert [name for name, _ in pictures] == ["post/depth_camera", "post/seg"]
    assert all(png.startswith(b"\x89PNG") for _, png in pictures)


def test_a_depth_camera_gives_the_distance_to_an_obstacle_off_its_axis(pg, needs_gl):
    """A pole 26 degrees to the side: its depth along the axis is 10 percent short of its distance, and the
    nearest obstacle is the distance (the lidar beside the lens agrees). The lidar's puck stands next to the
    lens and fills part of the picture; it is the camera's own robot, not an obstacle."""
    pg.add_robot("post", build=POST)
    radius, bearing = 0.025, math.radians(26)
    middle = np.array([0.1 + 1.85 * math.cos(bearing), 1.85 * math.sin(bearing)])
    pg.add_object("pole", shape="cylinder", pos=middle.tolist(), size=[2 * radius, 1.0], fixed=True)
    pg.add_sensor("post", {"type": "depth camera"})
    pg.add_sensor("post", {"type": "lidar", "at": "front"})  # its puck stands beside the lens
    m, d = pg.session.model, pg.session.data
    eye = d.cam_xpos[m.camera("post/depth_camera").id][:2]
    origin = d.site_xpos[m.site("post/rf_lidar").id][:2]
    reading, _ = pg.read_sensors("post")
    depth, lidar = reading["sensors"]["depth_camera"], reading["sensors"]["lidar"]
    truth = float(np.linalg.norm(middle - eye)) - radius
    assert depth["nearest"]["distance_m"] == pytest.approx(truth, abs=0.01)
    assert depth["own_share"] > 0.01  # the puck is in the picture
    assert depth["nearest"]["bearing_deg"] == pytest.approx(26, abs=1.5)
    assert depth["nearest"]["distance_m"] - truth * math.cos(bearing) > 0.15  # not its depth along the axis
    lidar_truth = float(np.linalg.norm(middle - origin)) - radius
    assert lidar["nearest"]["distance_m"] == pytest.approx(lidar_truth, abs=0.01)
    # The same obstacle from both, once the 1.4 cm between the lens and the puck's middle is allowed for.
    between = depth["nearest"]["distance_m"] - lidar["nearest"]["distance_m"]
    assert between == pytest.approx(truth - lidar_truth, abs=0.01)
    again = [_read(pg, "post", "depth_camera")["nearest"] for _ in range(3)]
    assert all(n == depth["nearest"] for n in again)  # ties do not move it between reads


def test_a_noisy_depth_camera_reads_the_nearest_wall_where_it_is(post, needs_gl):
    post.add_sensor("post", {"type": "depth camera", "noise": 0.01, "rate_hz": 10})
    m, d = post.session.model, post.session.data
    truth = WALL_X - float(d.cam_xpos[m.camera("post/depth_camera").id][0])
    near = []
    for _ in range(10):
        post.session.step(50)
        near.append(_read(post, "post", "depth_camera")["nearest"]["distance_m"])
    # The single nearest of 76800 noisy pixels reads about four noise widths (4 cm) short; the median around it
    # does not. The wall's drawn plaster stands 6 mm proud of its collision face.
    assert float(np.mean(near)) == pytest.approx(truth - 0.003, abs=0.006)


def test_imu_on_a_dropped_box_reads_free_fall_then_gravity(pg):
    pg.add_robot("box", build=BOX, pos=[0, 0, 1.0])
    pg.add_sensor("box", {"type": "imu"})
    pg.simulate(0.2)
    falling = _read(pg, "box", "imu")
    assert np.linalg.norm(falling["accel_m_s2"]) < 0.05
    pg.simulate(1.5)
    resting = _read(pg, "box", "imu")
    assert resting["accel_m_s2"][2] == pytest.approx(9.81, abs=0.02)
    assert abs(resting["orientation_deg"]["roll"]) < 0.5


def test_a_touch_array_under_a_block_carries_its_weight_evenly(pg):
    pg.add_robot("box", build=BOX)
    pg.add_sensor("box", {"type": "touch", "rows": 2, "cols": 2})
    pg.simulate(1.0)
    pad = _read(pg, "box", "touch")
    assert pad["contact"] and pad["force_n"] == pytest.approx(2.0 * 9.81, rel=0.01)
    cells = np.array(pad["cells_n"])
    assert cells.shape == (2, 2) and cells.min() > 0.2 * cells.max()
    assert np.allclose(pad["center"], 0, atol=0.1)


def test_a_touch_pad_under_a_walking_foot_feels_what_the_foot_feels(pg):
    """Places are the robot's own: the bottom of a shin is the side that meets the floor as the dog stands, though
    the shin's frame is tilted. Pads laid on the shin's own -z missed most steps of a Go2's trot."""
    pg.add_robot("dog", template="quadruped")
    pg.add_sensor("dog", {"type": "touch", "on": "front_left_shin", "rows": 2, "cols": 2, "name": "pad"})
    pg.behaviors.move("dog", 0.2, 0.0)
    pg.simulate(1.0)
    m = pg.session.model
    shin = m.body("dog/front_left_shin").id
    agree = []
    for _ in range(150):
        pg.simulate(0.01)
        d = pg.session.data
        touching = any(shin in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) for c in d.contact[: d.ncon])
        agree.append(_read(pg, "dog", "pad")["contact"] == touching)
    assert np.mean(agree) > 0.95


def test_force_torque_weighs_what_a_tray_holds(pg):
    pg.add_robot("tray", build=TRAY)
    pg.add_sensor("tray", {"type": "force_torque", "on": "plate"})
    assert _read(pg, "tray", "plate_ft")["payload_kg"] == pytest.approx(0, abs=0.005)
    pg.add_object("weight", shape="box", size=[0.1, 0.1, 0.1], mass=0.5, pos=[0, 0, 0.35])
    pg.simulate(1.0)
    loaded = _read(pg, "tray", "plate_ft")
    assert loaded["payload_kg"] == pytest.approx(0.5, rel=0.02)
    assert loaded["force_world_n"][2] > 0  # the parent holds the plate up


def test_force_torque_between_two_bodies_must_name_neighbors(pg):
    pg.add_robot("arm", template="arm")
    with pytest.raises(ValueError, match="not neighbors"):
        pg.add_sensor("arm", {"type": "force_torque", "between": ["base", "hand"]})


def test_encoders_step_to_their_resolution(pg):
    pg.add_robot("arm", template="arm")
    pg.add_sensor("arm", {"type": "encoders", "joints": ["shoulder", "elbow"], "resolution": 0.01})
    pg.simulate(0.5, {"arm/shoulder": 0.333})
    angles = _read(pg, "arm", "encoders")["rad"]
    assert list(angles) == ["shoulder", "elbow"]
    for value in angles.values():
        assert value == pytest.approx(round(value / 0.01) * 0.01, abs=1e-9)
    m, d = pg.session.model, pg.session.data
    shoulder = float(d.qpos[m.jnt_qposadr[m.joint("arm/shoulder").id]])
    assert angles["shoulder"] == pytest.approx(shoulder, abs=0.005 + 1e-9)  # within half a step of the truth


def test_gps_and_odometry_follow_a_driving_rover(pg):
    pg.add_robot("rover", template="rover")
    pg.add_sensor("rover", {"type": "gps"})
    pg.add_sensor("rover", {"type": "odometry"})
    pg.add_sensor("rover", {"type": "odometry", "name": "cheap_odometry", "noise": {"speed": 0.05}, "drift": 0.05})
    pg.behaviors.move("rover", 0.3, 0.3)
    pg.simulate(4.0)
    truth = pg.behaviors.pose("rover")["position"]
    gps = _read(pg, "rover", "gps")["position_m"]
    good = _read(pg, "rover", "odometry")
    cheap = _read(pg, "rover", "cheap_odometry")
    assert np.allclose(gps[:2], truth[:2], atol=0.02)
    assert np.allclose(good["position_m"], gps[:2], atol=0.02)
    assert good["traveled_m"] > 0.8
    assert np.linalg.norm(np.subtract(cheap["position_m"], good["position_m"])) > 0.005  # it drifts


# ---- realism knobs -----------------------------------------------------------------------------------------


def test_measured_noise_matches_its_setting(post):
    post.add_sensor("post", {"type": "gps", "noise": 0.05, "rate_hz": 100})
    post.add_sensor("post", {"type": "imu", "noise": {"accel": 0.2}, "bias": {"accel": [0, 0, 0.5]}})
    gps, accel = [], []
    for _ in range(400):
        post.simulate(0.01)
        reading, _ = post.read_sensors("post", ["gps", "imu"])
        gps.append(reading["sensors"]["gps"]["position_m"])
        accel.append(reading["sensors"]["imu"]["accel_m_s2"])
    gps, accel = np.array(gps), np.array(accel)
    assert gps.std(axis=0) == pytest.approx([0.05] * 3, rel=0.15)
    assert accel.std(axis=0) == pytest.approx([0.2] * 3, rel=0.15)
    assert accel[:, 2].mean() == pytest.approx(9.81 + 0.5, abs=0.05)  # at rest on a post bolted down, plus the bias


def test_latency_delays_readings_by_its_setting(pg):
    """A dropped block with a GPS 0.1 s late: what it reads at t is where the block was at t - 0.1."""
    pg.add_robot("box", build=BOX, pos=[0, 0, 2.0])
    pg.add_sensor("box", {"type": "gps", "latency_s": 0.1})
    assert _read(pg, "box", "gps")["type"] == "gps"
    m = pg.session.model
    site = m.site("box/rf_gps").id
    truth = {}
    lags = []
    for _ in range(250):
        pg.session.step(1)
        d = pg.session.data
        stamp = round(float(d.time) - m.opt.timestep, 4)  # positions after a step are the step's start
        truth[stamp] = float(d.site_xpos[site][2])
        reading = _read(pg, "box", "gps")
        if "position_m" not in reading:
            assert d.time < 0.1 + 1e-6  # nothing until the first sample has waited out the latency
            continue
        z = reading["position_m"][2]
        match = min(truth, key=lambda t: abs(truth[t] - z))
        lags.append(round(float(d.time) - match, 4))
    assert np.median(lags) == pytest.approx(0.1 + m.opt.timestep, abs=m.opt.timestep + 1e-6)


def test_a_rate_holds_each_sample_until_the_next(pg):
    pg.add_robot("box", build=BOX, pos=[0, 0, 2.0])
    pg.add_sensor("box", {"type": "gps", "rate_hz": 10})
    seen = set()
    for _ in range(100):  # 0.2 s
        pg.session.step(1)
        seen.add(_read(pg, "box", "gps")["position_m"][2])
    assert len(seen) == pytest.approx(3, abs=1)  # samples at 0, 0.1 and 0.2 s


def test_lidar_dropout_loses_that_share_of_rays(post):
    post.add_sensor("post", {"type": "lidar"})
    post.add_sensor("post", {"type": "lidar", "name": "dusty", "dropout": 0.3})
    post.simulate(0.2)
    clean = _read(post, "post", "lidar")["hits"]
    dusty = _read(post, "post", "dusty")["hits"]
    assert dusty / clean == pytest.approx(0.7, abs=0.1)


def test_noise_is_repeatable_after_a_reset(pg):
    pg.add_robot("box", build=BOX)
    pg.add_sensor("box", {"type": "imu", "noise": 0.1, "drift": {"gyro": 0.01}})

    def run() -> list:
        out = []
        for _ in range(20):
            pg.simulate(0.05)
            out.append(_read(pg, "box", "imu")["gyro_rad_s"])
        return out

    first = run()
    pg.control("reset")
    assert run() == first


# ---- editing, names and the robots that carry sensors of their own ------------------------------------------


def test_adding_and_removing_a_sensor_leaves_the_robot_where_it_is(pg):
    pg.add_robot("rover", template="rover")
    pg.behaviors.move("rover", 0.3, 0.0)
    pg.simulate(2.0)
    pg.behaviors.stop("rover")
    before = pg.behaviors.pose("rover")["position"]
    added = pg.add_sensor("rover", {"type": "lidar"})
    assert added["sensor"] == "rover/lidar" and added["reading"]["rays"] == 360
    assert np.allclose(pg.behaviors.pose("rover")["position"][:2], before[:2], atol=0.01)
    assert "type: lidar" in pg.show_world()["world_yaml"]
    removed = pg.remove("rover/lidar")
    assert removed["removed"] == "rover/lidar"
    assert "lidar" not in pg.sensors.names("rover")
    assert "type: lidar" not in pg.show_world()["world_yaml"]
    assert np.allclose(pg.behaviors.pose("rover")["position"][:2], before[:2], atol=0.01)


def test_sensors_survive_reopening_the_world(pg):
    pg.add_robot("box", build=BOX)
    pg.add_sensor("box", {"type": "imu", "name": "body_imu"})
    reopened = Playground.with_demo()
    reopened.open_world("sensing")
    assert "body_imu" in reopened.sensors.names("box")


def test_mistakes_say_what_to_do(pg):
    pg.add_robot("go", template="quadruped")
    with pytest.raises(ValueError, match="no body named 'tail'"):
        pg.add_sensor("go", {"type": "touch", "on": "tail"})
    pg.add_sensor("go", {"type": "gps"})
    with pytest.raises(PlaygroundError, match="already has a sensor"):
        pg.add_sensor("go", {"type": "imu", "name": "gps"})
    with pytest.raises(PlaygroundError, match="comes with the robot itself"):
        pg.remove("go/front_cam")
    with pytest.raises(PlaygroundError, match="has no sensors yet"):
        pg.add_robot("bare", build=BOX, pos=[1, 1])
        pg.read_sensors("bare")


def test_builder_parts_and_templates_declare_their_sensors(pg):
    pg.add_robot(
        "scout",
        build={
            **SCOUT,
            "base": {**SCOUT["base"], "sensors": ["imu", {"type": "rangefinder", "at": "front", "noise": 0.01}]},
        },
    )
    pg.add_robot("dog", template="quadruped", pos=[1.5, 0])
    assert set(pg.sensors.names("scout")) >= {"imu", "base_range", "mast_lidar", "tilt_cam"}
    assert set(pg.sensors.names("dog")) == {
        "front_cam",
        "imu",
        "front_left_touch",
        "front_right_touch",
        "rear_left_touch",
        "rear_right_touch",
    }
    ring = _read(pg, "scout", "mast_lidar")
    assert ring["rays"] == 16
    assert not pg.sensors.suite.sensors["scout/base_range"].lazy  # its noise makes it sampled while time passes
    pg.simulate(1.0)
    feet = pg.read_sensors("dog")[0]["sensors"]
    assert all(feet[f"{leg}_touch"]["contact"] for leg in ("front_left", "front_right", "rear_left", "rear_right"))


MJCF_CART = """<mujoco><worldbody><body name="cart" pos="0 0 0.1"><freejoint/>
<geom type="box" size="0.15 0.1 0.05" mass="2"/></body></worldbody></mujoco>"""


def test_imported_urdf_and_mjcf_robots_carry_sensors(pg, tmp_path):
    from importer_fixtures import rover_urdf, write

    urdf = write(tmp_path / "rover.urdf", rover_urdf())
    mjcf = write(tmp_path / "cart.xml", MJCF_CART)
    pg.add_robot("urdf_rover", file=str(urdf))
    pg.add_robot("cart", file=str(mjcf), pos=[1.0, 1.0])
    pg.add_sensor("urdf_rover", {"type": "lidar"})
    pg.add_sensor("urdf_rover", {"type": "imu", "on": "chassis"})
    pg.add_sensor("cart", {"type": "touch", "rows": 2, "cols": 2})
    pg.simulate(1.0)
    assert _read(pg, "urdf_rover", "imu")["accel_m_s2"][2] == pytest.approx(9.81, abs=0.1)
    assert _read(pg, "urdf_rover", "lidar")["hits"] > 0  # it sees the cart
    assert _read(pg, "cart", "touch")["force_n"] == pytest.approx(2 * 9.81, rel=0.02)


def test_definitions_travel_with_the_model(pg):
    """Sensors are declared in the model itself, so whatever loads it (an MJCF saved from the world) has them."""
    pg.add_robot("box", build=BOX)
    pg.add_sensor("box", {"type": "lidar3d", "rate_hz": 10})
    xml = pg.store.compile().to_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    names = {d.name for d in declare.read(model)}
    assert names == {"box/lidar"}


# ---- the viewer, the tools, scripts ------------------------------------------------------------------------


def test_viewer_message_carries_poses_and_bounded_points(post):
    post.add_sensor("post", {"type": "lidar3d", "channels": 32, "rays": 256})
    post.add_sensor("post", {"type": "camera", "mode": "depth"})
    post.add_sensor("post", {"type": "touch"})
    msg = post.sensors.viewer_message()
    assert msg["type"] == "sensors"
    by_type = {e["type"]: e for e in msg["sensors"]}
    lidar = by_type["lidar3d"]
    assert len(lidar["pos"]) == 3 and len(lidar["quat"]) == 4
    assert 0 < len(lidar["points"]) <= 3 * 1200 and len(lidar["points"]) % 3 == 0
    assert by_type["camera"]["fov_deg"] == 60 and by_type["camera"]["mode"] == "depth"
    assert len(by_type["touch"]["cells_m"]) == 1
    assert len(json.dumps(msg)) < 60_000


def test_the_viewer_stream_sends_sensors_only_when_there_are_some(post):
    hub = ViewerHub(post.session)
    post.attach_viewer(hub)
    client = TestClient(create_app(hub), base_url="http://127.0.0.1:8737")
    post.add_sensor("post", {"type": "lidar"})
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        for _ in range(400):
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"])["type"] == "sensors":
                sensors = json.loads(msg["text"])["sensors"]
                break
        else:
            raise AssertionError("no sensors message")
    assert [s["name"] for s in sensors] == ["post/lidar"]
    post.remove("post/lidar")
    assert post.sensors.viewer_message() is None


@pytest.mark.anyio
async def test_tools_add_read_list_and_remove_a_sensor(post, gl_reason):
    server = build_server(post, 0)
    async with Client(server) as client:
        added = await client.call_tool(
            "add_sensor", {"robot": "post", "type": "depth_camera", "options": {"noise": 0.01, "rate_hz": 15}}
        )
        assert not added.is_error, added.content[0].text
        assert json.loads(added.content[0].text)["sensor"] == "post/depth_camera"
        await client.call_tool("add_sensor", {"robot": "post", "type": "lidar3d"})
        read = await client.call_tool("read_sensors", {"robot": "post", "image": True})
        body = json.loads(read.content[0].text)
        assert body["sensors"]["lidar"]["ahead"]["distance_m"] == pytest.approx(2.0, abs=0.05)
        if gl_reason is None:
            assert read.content[1].type == "image"
        assert len(read.content[0].text) < 3000  # small enough to read on every turn
        observed = await client.call_tool("observe", {"include_image": False})
        summary = json.loads(observed.content[0].text)["sensor_summary"]
        assert summary["post/lidar"].startswith("3D lidar")
        assert not any("rf_" in name for name in json.loads(observed.content[0].text)["sensors"])
        removed = await client.call_tool("remove", {"name": "post/lidar"})
        assert json.loads(removed.content[0].text)["removed"] == "post/lidar"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_a_script_reads_a_sensor_by_name(post):
    post.add_sensor("post", {"type": "lidar"})
    code = (
        "from robotforge.script import connect\n"
        "sim = connect()\n"
        "scan = sim.sensor('post', 'lidar', full=True)\n"
        "print('ahead', scan['ahead']['distance_m'], len(scan['ranges_m'][0]))\n"
    )
    server = build_server(post, 0)
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "look", "code": code})
        ran = json.loads((await client.call_tool("run_script", {"name": "look"})).content[0].text)
    assert ran["ok"], ran["output"]
    distance, count = ran["output"].split("ahead ")[1].split()
    assert float(distance) == pytest.approx(2.0, abs=0.05) and int(count) == 360
