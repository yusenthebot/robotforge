"""A lidar read whole: observe's "lidars" gives every ray, the nearest hit and which way it is from the robot's front,
and a script's sensors are all of them, not observe's first 40."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design.builder.schema import SensorSpec
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
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
