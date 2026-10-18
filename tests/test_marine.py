"""Boats and underwater robots: floating level, speed, turning, righting after a push, depth hold, trips on and under
the water, scenes with water, robots built with thrusters, and the agent-facing pieces (set_floor water, observe)."""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import Kind, marine
from robotforge.behaviors.navigate import NoWay
from robotforge.cli import find_free_port
from robotforge.design import scenes
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.viewer import scene as viewer_scene

LEVEL = 1.0
RAFT = {
    "color": "teal",
    "base": {
        "shape": "box",
        "size": [0.8, 0.5, 0.16],
        "mass": 12.0,
        "thrusters": [
            {"at": [-0.42, 0.18, -0.06], "direction": "forward", "force": 15},
            {"at": [-0.42, -0.18, -0.06], "direction": "forward", "force": 15},
        ],
    },
}
SUB = {
    "color": "yellow",
    "base": {
        "shape": "box",
        "size": [0.5, 0.36, 0.22],
        "mass": 39.0,
        "thrusters": [
            {"at": [-0.27, 0.12, 0], "direction": "forward", "force": 25},
            {"at": [-0.27, -0.12, 0], "direction": "forward", "force": 25},
            {"at": [0, 0.2, 0.05], "direction": "up", "force": 25, "name": "left_lift"},
            {"at": [0, -0.2, 0.05], "direction": "up", "force": 25, "name": "right_lift"},
        ],
    },
}


@pytest.fixture
def pg(tmp_path, monkeypatch) -> Playground:
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("water")
    playground.set_floor("sand", water=LEVEL)
    return playground


def _root(pg: Playground, robot: str) -> int:
    return marine.root_body(pg.session.model, robot)


def _tilt_deg(pg: Playground, robot: str) -> float:
    up = pg.session.data.xmat[_root(pg, robot)].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))


def _speed(pg: Playground, robot: str) -> float:
    return float(np.linalg.norm(pg.session.data.cvel[_root(pg, robot)][3:5]))


# ---- the boat template ----------------------------------------------------------------------------------------


def test_the_boat_floats_level_where_it_was_put(pg):
    pg.add_robot("boat", template="boat", pos=[0, 0])
    start = pg.behaviors.pose("boat")["position"]
    pg.session.step(1500)
    now = pg.behaviors.pose("boat")["position"]
    assert np.allclose(start, now, atol=0.005)
    assert _tilt_deg(pg, "boat") < 0.5
    assert pg.behaviors.kind("boat") == Kind.BOAT
    report = {r["name"]: r for r in pg.state()["water"]["afloat"]}
    assert 0.15 < report["boat/hull"]["under_water"] < 0.4


def test_the_boat_holds_the_speed_it_is_asked(pg):
    pg.add_robot("boat", template="boat", pos=[0, 0])
    pg.behaviors.move("boat", 0.8, 0.0)
    pg.session.step(3000)  # 6 s
    assert _speed(pg, "boat") == pytest.approx(0.8, rel=0.03)


def test_the_boat_top_speed_turning_circle_and_stop(pg):
    pg.add_robot("boat", template="boat", pos=[0, 0])
    status = pg.behaviors.move("boat", 5.0, 0.0)
    top = status["max_speed"]
    assert status["notes"] and 0.8 < top < 2.0  # clamped to what its thrusters and its drag give
    pg.session.step(4000)
    assert _speed(pg, "boat") == pytest.approx(top, rel=0.03)
    pg.behaviors.move("boat", 0.8, 0.6)
    points = []
    for _ in range(40):
        pg.session.step(250)
        points.append(pg.behaviors.pose("boat")["position"][:2])
    circle = np.array(points[16:])
    diameter = 2 * float(np.mean(np.linalg.norm(circle - circle.mean(axis=0), axis=1)))
    assert diameter == pytest.approx(2 * 0.8 / 0.6, rel=0.1)
    pg.behaviors.stop("boat")
    pg.session.step(2000)
    assert _speed(pg, "boat") < 0.12


def test_the_boat_rights_itself_after_a_push(pg):
    pg.add_robot("boat", template="boat", pos=[0, 0])
    pg.session.step(100)
    dof = pg.session.model.body_dofadr[_root(pg, "boat")]
    pg.session.data.qvel[dof + 3] = 2.0  # a hard shove on its side: 2 rad/s of roll
    tilts = []
    for _ in range(300):
        pg.session.step(10)
        tilts.append(_tilt_deg(pg, "boat"))
    assert max(tilts) > 8.0, "the push did not heel it"
    assert max(tilts[-50:]) < 1.0, f"still heeled {max(tilts[-50:]):.1f} degrees after 6 s"


# ---- the ROV ----------------------------------------------------------------------------------------------------


def test_the_rov_dives_to_a_point_and_holds_depth_and_heading(pg):
    pg.add_robot("rov", template="rov", pos=[0, 0])
    assert pg.behaviors.kind("rov") == Kind.ROV
    trip = pg.behaviors.navigate("rov", [1.5, 0.5, 0.5])
    for _ in range(40):
        pg.session.step(500)
        if trip.done:
            break
    result = trip.result()
    assert result["arrived"], result
    assert result["error_m"] <= 0.1
    root = _root(pg, "rov")
    heading = pg.behaviors.pose("rov")["heading_deg"]
    zs = []
    for _ in range(250):
        pg.session.step(10)
        zs.append(float(pg.session.data.subtree_com[root][2]))
    assert max(abs(z - 0.5) for z in zs[100:]) < 0.03, f"depth wandered {max(abs(z - 0.5) for z in zs[100:]):.3f} m"
    assert abs((pg.behaviors.pose("rov")["heading_deg"] - heading + 180) % 360 - 180) < 3.0
    assert _tilt_deg(pg, "rov") < 2.0


def test_the_rov_is_kept_under_the_surface(pg):
    pg.add_robot("rov", template="rov", pos=[0, 0])
    trip = pg.behaviors.navigate("rov", [0.5, 0.0, 3.0])
    assert "surface" in trip.result()["note"]
    assert trip.target[2] < LEVEL


def test_without_power_the_rov_drifts_up(pg):
    pg.add_robot("rov", template="rov", pos=[0, 0])
    trip = pg.behaviors.navigate("rov", [0.0, 0.0, 0.5])
    for _ in range(30):
        pg.session.step(500)
        if trip.done:
            break
    pilot = pg.session.controllers["rov"]
    pg.session.set_controller("rov", None)  # the power off: it is trimmed a little lighter than water
    pg.session.data.ctrl[pilot.actuators] = 0.0
    z0 = float(pg.session.data.subtree_com[_root(pg, "rov")][2])
    pg.session.step(4000)  # 8 s: 2 N of lift against its heave drag, about 2 cm/s
    assert float(pg.session.data.subtree_com[_root(pg, "rov")][2]) > z0 + 0.1


def test_keys_take_the_rov_up_and_down(pg):
    pg.add_robot("rov", template="rov", pos=[0, 0])
    pg.behaviors.move("rov", 0.0, 0.0)
    pg.session.step(50)
    z0 = float(pg.session.data.subtree_com[_root(pg, "rov")][2])
    pg.behaviors.heave("rov", -0.3)
    pg.session.step(1500)
    assert float(pg.session.data.subtree_com[_root(pg, "rov")][2]) < z0 - 0.4


# ---- on the water: paths --------------------------------------------------------------------------------------


def test_the_boat_finds_its_way_round_a_line_of_buoys(pg):
    pg.add_robot("boat", template="boat", pos=[-3.0, 0.0])
    for i, y in enumerate((-1.2, 0.0, 1.2)):
        pg.add_object(f"buoy_{i + 1}", shape="buoy", pos=[0.0, y])
    buoys = [pg.session.model.body(f"buoy_{i + 1}").id for i in range(3)]
    trip = pg.behaviors.navigate("boat", [3.0, 0.0])
    assert trip.plan.length > 6.3  # not straight through the buoys
    nearest = math.inf
    for _ in range(60):
        pg.session.step(250)
        hull = pg.session.data.xpos[_root(pg, "boat")][:2]
        nearest = min(nearest, *(np.linalg.norm(hull - pg.session.data.xpos[b][:2]) for b in buoys))
        if trip.done:
            break
    assert trip.result()["arrived"], trip.result()
    assert nearest > 0.5, f"it came within {nearest:.2f} m of a buoy"


def test_the_boat_is_told_a_point_on_land_is_on_land(pg):
    pg.add_terrain("platform", name="island", pos=[4.0, 0.0], params={"size": [3, 3], "height": 1.4})
    pg.add_robot("boat", template="boat", pos=[0.0, 0.0])
    with pytest.raises(NoWay) as found:
        pg.behaviors.navigate("boat", [4.0, 0.0])
    assert found.value.cause == "on_land" and "land" in str(found.value)


def test_a_boat_without_water_is_told_so(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    dry = Playground.with_demo()
    dry.new_world("dry")
    dry.add_robot("boat", template="boat", pos=[0, 0])
    with pytest.raises(ValueError, match="no water"):
        dry.behaviors.navigate("boat", [2.0, 0.0])


# ---- scenes, built robots, the viewer and agents -----------------------------------------------------------


@pytest.mark.parametrize("preset", ["harbor", "lake", "pool"])
def test_water_scenes_start_their_robot_afloat(tmp_path, monkeypatch, preset):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    result = pg.load_scene(preset)
    robot = result["robots"][0]
    assert pg.session.water is not None
    pg.session.step(500)
    report = {r["name"]: r for r in pg.state()["water"]["afloat"]}
    root = mujoco.mj_id2name(pg.session.model, mujoco.mjtObj.mjOBJ_BODY, _root(pg, robot))
    assert 0.1 < report[root]["under_water"] < 0.999
    assert report["steel_block"]["under_water"] == pytest.approx(1.0)
    assert all(0.3 < r["under_water"] < 0.7 for name, r in report.items() if name.startswith("crate"))


def test_load_scene_puts_a_kept_boat_on_the_water(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("kept")
    pg.add_robot("boat", template="boat")
    pg.add_robot("dog", template="quadruped", pos=[1, 0])
    result = pg.load_scene("harbor")
    placed = " ".join(result["scene"]["placed"])
    slot = scenes.build("harbor").water_slots[0]
    assert f"boat placed at [{slot.pos[0]:g}, {slot.pos[1]:g}]" in placed
    assert pg.behaviors.pose("dog")["position"][2] > 1.5  # on the quay


def test_robots_built_with_thrusters_are_boats_and_rovs(pg):
    pg.add_robot("raft", build=RAFT, pos=[0, 0])
    pg.add_robot("sub", build=SUB, pos=[0, 2])
    kinds = {r["name"]: r["kind"] for r in pg.behaviors.robots()}
    assert kinds == {"raft": "boat", "sub": "rov"}
    pg.behaviors.move("raft", 0.5, 0.0)
    pg.session.step(2500)
    assert pg.behaviors.pose("raft")["position"][0] > 1.5


def test_the_viewer_scene_carries_the_water(pg):
    model = pg.session.model
    desc = viewer_scene.describe(model, 1, [])
    assert desc["water"]["level"] == LEVEL and desc["water"]["waves"] == []
    pg.set_floor(water={"level": 1.2, "waves": 0.03})
    desc = viewer_scene.describe(pg.session.model, 2, [])
    assert len(desc["water"]["waves"]) == 3 and desc["water"]["waves"][0][0] == pytest.approx(0.03)
    pg.set_floor(water=0)
    assert viewer_scene.describe(pg.session.model, 3, [])["water"] is None


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_set_floor_water_and_observe_over_mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_path(Playground.with_demo().session.source))
    server = build_server(playground, find_free_port(8840))
    async with Client(server) as client:
        await client.call_tool("new_world", {"name": "pond"})
        flooded = await client.call_tool("set_floor", {"water": {"level": 0.8, "waves": 0.02}})
        assert not flooded.is_error and "water up to 0.8 m" in flooded.content[0].text
        await client.call_tool("add_object", {"name": "crate", "shape": "box", "size": [0.4, 0.4, 0.3], "mass": 20})
        state = json.loads((await client.call_tool("observe", {"include_image": False})).content[0].text)
        assert state["water"]["level"] == 0.8
        assert state["water"]["afloat"][0]["name"] == "crate"
        bad = await client.call_tool("set_floor", {"water": {"level": 0.8, "current": [9, 0]}})
        assert bad.is_error and "current" in bad.content[0].text
        drained = json.loads((await client.call_tool("set_floor", {"water": 0})).content[0].text)
        assert not any("water up to" in n for n in drained["notes"])
