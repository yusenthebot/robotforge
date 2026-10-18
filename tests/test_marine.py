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


def test_in_deep_water_things_start_where_they_float(tmp_path, monkeypatch):
    """The search for the floating height covered 4 m: in water 12 m deep a crate and the boat started 8 m under."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("deep")
    playground.set_floor("sand", water=10.0)
    playground.add_object("float_box", shape="box", size=[0.6, 0.4, 0.2], pos=[2, 0], mass=40.0)
    playground.add_robot("boat", template="boat", pos=[0, 0])
    m, d = playground.session.model, playground.session.data
    box, hull = m.body("float_box").id, marine.root_body(m, "boat")
    assert d.xpos[box][2] == pytest.approx(10.0 - 40.0 / (1000 * 0.6 * 0.4) + 0.1, abs=0.02)  # Archimedes
    start = float(d.xpos[hull][2])
    playground.session.step(1500)
    assert abs(float(d.xpos[hull][2]) - start) < 0.02
    assert abs(float(d.xpos[box][2]) - (10.0 - 40.0 / 240 + 0.1)) < 0.02


def test_in_water_too_shallow_to_float_in_the_boat_rests_on_the_bottom(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("puddle")
    playground.add_robot("boat", template="boat", pos=[0, 0])
    dry = float(playground.session.data.xpos[marine.root_body(playground.session.model, "boat")][2])
    playground.set_floor("sand", water=0.04)
    m, d = playground.session.model, playground.session.data
    hull = marine.root_body(m, "boat")
    assert float(d.xpos[hull][2]) == pytest.approx(dry, abs=1e-3)  # not pushed into the ground
    playground.session.step(500)
    assert float(np.abs(d.qvel).max()) < 0.05


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


def test_a_dock_built_under_water_is_noted_with_a_size_that_works(pg):
    """An agent read the dock's third size as the deck's thickness (0.18) and put its deck under water."""
    result = pg.add_object("dock", shape="dock", pos=[4, 0], size=[2, 3, 0.18])
    note = next(n for n in result["notes"] if n.startswith("dock:"))
    assert "under the water" in note and "[2, 3, 1.35]" in note
    fixed = pg.update_object("dock", size=[2, 3, 1.35])
    assert not any(n.startswith("dock:") for n in fixed["notes"])


def test_the_rov_is_there_only_within_a_few_centimeters(pg):
    """Told it had arrived 6 cm off, an agent reported the depth it asked for rather than the one reached."""
    pg.add_robot("rov", template="rov", pos=[0, 0])
    trip = pg.behaviors.navigate("rov", [0.0, 0.0, 0.5])
    for _ in range(40):
        pg.session.step(250)
        if trip.done:
            break
    result = trip.result()
    assert result["arrived"] and result["error_m"] <= 0.04
    assert result["depth_m"] == pytest.approx(LEVEL - 0.5, abs=0.05)


# ---- where an ROV can dive -----------------------------------------------------------------------------------


def _dive(pg: Playground, to, seconds: float = 30.0) -> dict:
    trip = pg.behaviors.navigate("rov", to, None, None, seconds)
    for _ in range(round(seconds / 0.5) + 2):
        pg.session.step(250)
        if trip.done:
            break
    return trip.result()


def test_an_rov_sent_onto_the_pool_deck_is_told_it_is_outside_the_water(tmp_path, monkeypatch):
    """It ran the full minute into the pool wall and was told to call move again, which never could work."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("baths")
    pg.load_scene("pool")
    with pytest.raises(NoWay) as found:
        pg.behaviors.navigate("rov", [8.0, 0.0, 0.5])
    assert found.value.cause == "on_land" and "outside the water" in str(found.value)
    assert found.value.destination == {"point": [8.0, 0.0, 0.5]}


def test_an_rov_sent_under_the_lake_shore_is_told_it_is_on_land(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("park")
    pg.load_scene("lake")
    pg.add_robot("rov", template="rov")
    with pytest.raises(NoWay) as found:
        pg.behaviors.navigate("rov", [11.0, 0.0, 0.5])  # inside the water's rectangle, under the bank
    assert found.value.cause == "on_land" and "on land" in str(found.value)


def test_an_rov_sent_below_the_bottom_is_told_how_low_it_can_go(pg):
    pg.add_robot("rov", template="rov", pos=[0, 0])
    with pytest.raises(NoWay) as found:
        pg.behaviors.navigate("rov", [1.0, 0.0, 0.03])
    lowest = float(str(found.value).split("lowest its middle can go is z ")[1][:4])
    assert found.value.cause == "below_bottom" and 0.1 < lowest < 0.3
    result = _dive(pg, [1.0, 0.0, lowest])
    assert result["arrived"], result


def test_an_rov_goes_over_a_reef_on_its_way_and_down_to_the_point(pg):
    """The straight line runs into a reef standing on the bottom: it goes over it at a depth that clears it."""
    pg.add_robot("rov", template="rov", pos=[-2.0, 0.0])
    pg.add_object("reef", shape="box", size=[1.0, 2.5, 0.55], pos=[0.0, 0.0], fixed=True)
    trip = pg.behaviors.navigate("rov", [2.0, 0.0, 0.35], None, None, 40.0)
    assert len(trip.plan.points) > 2 and "goes over it" in trip.note
    root, lowest_over_reef = _root(pg, "rov"), math.inf
    for _ in range(162):
        pg.session.step(125)
        x, _, z = pg.session.data.subtree_com[root]
        if abs(x) < 0.6:
            lowest_over_reef = min(lowest_over_reef, float(z))
        if trip.done:
            break
    assert trip.result()["arrived"], trip.result()
    assert lowest_over_reef > 0.55 + 0.1  # its middle stayed well over the reef's top


def test_an_rov_that_cannot_get_closer_is_told_it_is_stuck(pg):
    """Something hangs in the water on its straight way (not on the bottom, so not on its map): it says so in
    seconds instead of pushing at it until the time runs out."""
    pg.add_robot("rov", template="rov", pos=[-1.5, 0.0])
    pg.add_object("net", shape="box", size=[0.1, 3.0, 0.6], pos=[0.0, 0.0, 0.5], fixed=True)
    result = _dive(pg, [1.5, 0.0, 0.5], seconds=40.0)
    assert not result["arrived"] and result["why"] == "stuck", result
    assert result["time_s"] < 25.0 and "in its way" in result["note"]


def test_the_rov_depth_sensor_reads_meters_under_the_surface(pg):
    """It was a position sensor: observe showed rov/depth as [x, y, z] with z above the floor, 0.168 for an ROV
    1.435 m under the surface, the mix-up that had an agent report the depth it asked for."""
    pg.add_robot("rov", template="rov", pos=[0, 0])
    pg.session.step(50)
    at_surface = pg.state()["sensors"]["rov/depth"]
    assert isinstance(at_surface, float) and at_surface < 0.15
    readings = []
    for z in (0.7, 0.4):
        assert _dive(pg, [0.0, 0.0, z])["arrived"]
        readings.append(pg.state()["sensors"]["rov/depth"])
    assert at_surface < readings[0] < readings[1]
    site = pg.session.data.site("rov/depth").xpos[2]
    assert readings[1] == pytest.approx(LEVEL - site, abs=1e-3)


# ---- land robots in shallow water ---------------------------------------------------------------------------


@pytest.fixture
def wading(tmp_path, monkeypatch) -> Playground:
    """A quadruped standing in water 12 cm deep, with small waves."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("shallows")
    playground.set_floor("sand", water={"level": 0.12, "waves": 0.03})
    playground.add_robot("dog", template="quadruped", pos=[0, 0])
    return playground


def test_a_dog_standing_in_shallow_water_is_not_held(wading, monkeypatch):
    """The gait reads qfrc_applied on its trunk as a hand dragging it in the viewer: with the water's forces written
    there it thought it was held on all but one of its updates, and skipped settling and its standing limits."""
    from robotforge.behaviors.quadruped import QuadrupedGait

    seen = []
    original = QuadrupedGait._recovering

    def spy(self, t, dt, measured, roll, moving, fallen, held):
        seen.append(held)
        return original(self, t, dt, measured, roll, moving, fallen, held)

    monkeypatch.setattr(QuadrupedGait, "_recovering", spy)
    wading.behaviors.move("dog", 0.0, 0.0)
    wading.session.step(1000)
    assert len(seen) > 100 and not any(seen)
    root = marine.root_body(wading.session.model, "dog")
    dof = int(wading.session.model.body_dofadr[root])
    assert not wading.session.data.qfrc_applied[dof : dof + 6].any()
    assert wading.state()["water"]["afloat"], "the water did reach its legs"


def test_a_dog_in_shallow_water_costs_about_what_a_crate_does(wading):
    """Its legs' wet links went through mj_applyFT one by one, so a wading quadruped cost four floating crates; now
    every body's wrench goes to MuJoCo at once."""
    import time

    def per_call(session) -> float:
        best = math.inf
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(100):
                session.water.apply(session.model, session.data)
            best = min(best, (time.perf_counter() - start) / 100)
        session.data.xfrc_applied[:] = 0
        return best

    wading.session.step(100)
    dog = per_call(wading.session)
    crate = SimSession.from_xml(
        "<mujoco><custom><numeric name='robotforge_water' data='1 1 1000 0 4 30 0 0 0 0 0 0 0'/></custom>"
        "<worldbody><body pos='0 0 1'><freejoint/><geom type='box' size='0.3 0.2 0.1' density='500'/></body>"
        "</worldbody></mujoco>"
    )
    assert crate.water is not None
    assert dog < 3.0 * per_call(crate), f"{dog * 1e6:.0f} us"
    assert dog < 1e-3


@pytest.mark.anyio
async def test_move_over_mcp_says_a_point_on_the_pool_deck_is_outside_the_water(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_path(Playground.with_demo().session.source))
    server = build_server(playground, find_free_port(8850))
    async with Client(server) as client:
        await client.call_tool("new_world", {"name": "baths"})
        await client.call_tool("load_scene", {"preset": "pool"})
        answer = json.loads((await client.call_tool("move", {"robot": "rov", "to": [8, 0, 0.5]})).content[0].text)
        assert answer["arrived"] is False and answer["why"] == "on_land"
        assert "outside the water" in answer["note"] and "call move" not in answer["note"]
