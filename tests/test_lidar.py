"""A lidar read whole: observe's "lidars" gives every ray, the nearest hit and which way it is from the robot's front,
and a script's sensors are all of them, not observe's first 40."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design.builder.schema import SensorSpec
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.script_runner import ScriptRunner
from robotforge.sim import lidar
from robotforge.sim.session import SimSession

WALL_Y = 0.6  # the wall's middle, on the robot's left when it faces +x
WALL_THICK = 0.1


def _car(rays: int) -> dict:
    return {
        "base": {"size": [0.3, 0.2, 0.08], "sensors": ["imu", {"type": "lidar", "at": "top", "rays": rays}]},
        "motor": "jga25",
        "parts": [
            {
                "name": "front_left_wheel",
                "at": [0.1, 0.12, -0.03],
                "shape": "wheel",
                "size": [0.1, 0.03],
                "mirror": True,
            },
            {
                "name": "rear_left_wheel",
                "at": [-0.1, 0.12, -0.03],
                "shape": "wheel",
                "size": [0.1, 0.03],
                "mirror": True,
            },
        ],
    }


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("corridor")
    playground.add_object("wall", shape="wall", size=[3.0, WALL_THICK, 0.5], pos=[0.0, WALL_Y])
    playground.add_robot("scout", build=_car(64), pos=[0.0, 0.0])
    playground.run_fast(lambda: False, 0.3)  # settled on its wheels, the rangefinders computed
    return playground


def test_every_ray_is_read_and_the_nearest_hit_is_on_the_left(pg):
    """Regression: a 64-ray lidar is 64 sensors and observe lists 40; in the agent eval its last 27 rays never
    reached the agent."""
    state = pg.state()
    assert "..." in state["sensors"] and "lidars has every ray" in state["sensors"]["..."]
    ring = state["lidars"]["scout/base_lidar"]
    assert ring["rays"] == 64 and len(ring["ranges_m"]) == 64 and ring["range_m"] == 4.0
    assert ring["bearing_first_deg"] == 0.0 and ring["bearing_step_deg"] == pytest.approx(5.625)
    assert ring["nearest_side"] == "left" and ring["nearest_bearing_deg"] == pytest.approx(90.0, abs=0.5)
    assert ring["nearest_m"] == pytest.approx(WALL_Y - WALL_THICK / 2, abs=0.01)
    assert ring["nearest_by_side_m"]["left"] == ring["nearest_m"]
    assert ring["nearest_by_side_m"]["right"] is None  # nothing within 4 m on the other side: no hit, not -1
    assert ring["ranges_m"][16] == ring["nearest_m"]  # ray 16 points 90 degrees left
    assert sum(r is not None for r in ring["ranges_m"]) == ring["hits"]


def test_the_bearing_follows_the_robot_round(pg):
    pg.update_robot("scout", yaw=90)  # facing +y: the wall is ahead
    pg.run_fast(lambda: False, 0.3)
    ring = pg.state()["lidars"]["scout/base_lidar"]
    assert ring["nearest_side"] == "front" and ring["nearest_bearing_deg"] == pytest.approx(0.0, abs=0.5)
    pg.update_robot("scout", yaw=-90)  # facing -y: behind it
    pg.run_fast(lambda: False, 0.3)
    ring = pg.state()["lidars"]["scout/base_lidar"]
    assert ring["nearest_side"] == "back" and abs(ring["nearest_bearing_deg"]) == pytest.approx(180.0, abs=0.5)


def test_sides_split_at_45_degrees():
    assert [lidar.side(b) for b in (0, 45, 46, 135, 136, -180, -136, -135, -46, -45)] == [
        "front", "front", "left", "left", "back", "back", "back", "right", "right", "front",
    ]  # fmt: skip


def test_a_script_reads_all_the_sensors(pg):
    """The script API's sensors came from observe's list too: a script written to read the whole lidar got the
    same 37 rays."""
    assert len(pg.state()["sensors"]) == 41  # 40 and the "..." marker
    read = ScriptRunner(pg, watched=lambda: False)._op_sensors({"robot": "scout"})  # what sim.sensors("scout") gets
    assert sum(k.startswith("base_lidar_") for k in read) == 64 and "orientation" in read


def test_a_lidar_takes_any_fov_and_a_camera_is_still_held_to_its_limits():
    """fov is a camera's: a lidar given fov 360 (its rays go all the way round anyway) was refused with a camera's
    limit, below 170."""
    assert SensorSpec.model_validate({"type": "lidar", "fov": 360}).type == "lidar"
    with pytest.raises(ValueError, match="camera's fov must be between 5 and 170"):
        SensorSpec.model_validate({"type": "camera", "fov": 200})


@pytest.mark.anyio
async def test_observe_reports_the_lidar(pg):
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        result = await client.call_tool("observe", {"include_image": False})
    state = json.loads(result.content[0].text)
    assert state["lidars"]["scout/base_lidar"]["nearest_side"] == "left"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_the_mazes_starter_rover_can_be_rebuilt_with_a_lidar_where_it_stands(tmp_path, monkeypatch):
    """Both lidar sessions in the eval sent the maze's starter rover (a template) a build with a lidar, through
    update_robot and then add_part, were refused, and removed it to add the same robot again at the start mark."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.load_scene("maze")
    with pytest.raises(PlaygroundError, match=r'update_robot\("rover", build='):
        pg.add_part("rover", {"name": "puck", "at": "top", "sensors": ["lidar"]})
    done = pg.update_robot("rover", build=_car(16))
    assert any("rover: now built from parts (it was the rover template)" in n for n in done["notes"])
    rover = pg.current_world().robots[0]
    assert rover.build is not None and rover.template is None and rover.pos == [-2.0, -3.0] and rover.yaw == 90
    assert pg.state()["lidars"]["rover/base_lidar"]["rays"] == 16
    pg.undo()  # one edit: back to the template
    assert pg.current_world().robots[0].template == "rover"
    with pytest.raises(PlaygroundError, match="parts and sensors"):  # a few fields are no whole robot
        pg.update_robot("rover", build={"servo": "sts3215"})


MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"  # offline stand-ins for the real robots


def _shorthand_car() -> dict:
    car = _car(16)
    return {**car, "base": {**car["base"], "sensors": ["lidar"]}}  # at the base's middle, below its wheels' tops


@pytest.mark.parametrize("robot", [{"design": "lift_picker"}, {"design": "mobile_panda"}, {"build": _shorthand_car()}])
def test_a_lidar_does_not_see_its_own_robot(tmp_path, monkeypatch, robot):
    """Regression: a MuJoCo rangefinder leaves out only the body it is on. In an empty world the warehouse picker's
    lidar read 0.215 m ahead (its own lift column), the mobile Panda's 0.05 m behind and a car's 0.148 m (its own
    wheels), and an agent asked for the closest obstacle answered with the column."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("empty")
    pg.add_robot("r", **robot)
    pg.run_fast(lambda: False, 0.3)
    ring = pg.state()["lidars"]["r/base_lidar"]
    assert ring["hits"] == 0 and "nearest_m" not in ring, ring
    assert set(ring["ranges_m"]) == {None}
    rays = ScriptRunner(pg, watched=lambda: False)._op_sensors({"robot": "r"})  # a script's rays agree
    assert all(v < 0 or v >= ring["range_m"] for k, v in rays.items() if k.startswith("base_lidar_"))


def test_a_lidar_still_sees_past_its_robot_to_the_wall(tmp_path, monkeypatch):
    """The same car with its lidar down among its wheels reads the wall beyond them, as far as MuJoCo itself does
    from a lidar on top (the rays are level, so both measure the wall's face)."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("corridor")
    pg.add_object("wall", shape="wall", size=[3.0, WALL_THICK, 0.5], pos=[0.0, WALL_Y])
    pg.add_robot("scout", build=_shorthand_car(), pos=[0.0, 0.0])
    pg.run_fast(lambda: False, 0.3)
    ring = pg.state()["lidars"]["scout/base_lidar"]
    assert ring["nearest_side"] == "left" and ring["nearest_m"] == pytest.approx(WALL_Y - WALL_THICK / 2, abs=0.01)
    near = WALL_Y - WALL_THICK / 2
    assert ring["nearest_by_side_m"]["front"] == pytest.approx(near * 2**0.5, abs=0.01)  # the ray 45 degrees left
    assert ring["nearest_by_side_m"]["right"] is None and ring["hits"] == 7  # 45 to 135 degrees left: 7 rays


def test_a_rangefinder_on_the_world_reads_as_mujoco_has_it():
    """Only a robot's own parts are seen through: a rangefinder on the world body keeps MuJoCo's reading."""
    xml = """<mujoco><worldbody>
      <site name="eye" pos="0 0 0.2" zaxis="1 0 0"/>
      <body name="crate" pos="1 0 0.2"><geom type="box" size="0.05 0.5 0.5"/></body>
    </worldbody><sensor><rangefinder name="ahead" site="eye"/></sensor></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert lidar.seen(model, data, 0) == pytest.approx(0.95) == data.sensordata[0]


def test_a_ray_goes_through_as_many_of_its_robots_shapes_as_stand_in_the_way():
    """Regression: a ray gave up after 12 of its own robot's shapes and read nothing. A 64-ray lidar inside a real
    Go2's trunk on a plate goes through its meshes and collision shapes, and 8 rays saw no wall 0.85 to 1.45 m away.
    Here 16 thin plates of the robot's own (on a body of its own, which MuJoCo's rangefinder does not leave out)
    stand between the eye and a wall: in and out of each is 32 casts."""
    plates = "".join(
        f'<geom type="box" size="0.002 0.2 0.2" pos="{0.02 + 0.01 * k:.3f} 0 0" rgba="1 1 1 0.3"/>' for k in range(16)
    )
    xml = f"""<mujoco><worldbody>
      <body name="bot/base" pos="0 0 0.3"><freejoint/><site name="eye" zaxis="1 0 0"/>
        <geom type="sphere" size="0.01"/><body name="bot/hull">{plates}</body></body>
      <geom name="wall" type="box" size="0.05 1 1" pos="1 0 0.3"/>
    </worldbody><sensor><rangefinder name="bot/ahead" site="eye" cutoff="4"/></sensor></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.sensordata[0] == pytest.approx(0.018, abs=1e-3)  # MuJoCo's own reading: the first plate
    assert lidar.seen(model, data, 0) == pytest.approx(0.95, abs=1e-3)  # the wall's face, past every plate
