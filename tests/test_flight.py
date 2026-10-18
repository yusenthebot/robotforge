"""Aircraft: the rotor mixer for any layout, the flight controller, trips through the air and the drone template."""

import math

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import Kind, robot_kind
from robotforge.behaviors.airframe import AirframeError, Mixer, find_airframe
from robotforge.behaviors.flight import FlightController, Mission
from robotforge.design.world import World, compile_world
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


def _ring(n: int, arm: float = 0.2, k: float = 0.015, thrust: float = 8.0) -> Mixer:
    """The allocation of n rotors in an X ring with alternating spin, as the drone template lays them out."""
    cols = []
    for i in range(n):
        a = 2 * math.pi * (i + 0.5) / n
        spin = 1 if i % 2 == 0 else -1
        x, y = arm * math.cos(a), arm * math.sin(a)
        cols.append([1.0, y, -x, -spin * k])  # thrust up; roll = y * f, pitch = -x * f; reaction against the spin
    return Mixer(np.array(cols).T, np.zeros(n), np.full(n, thrust))


@pytest.mark.parametrize("n", [4, 6, 8])
def test_mixer_gives_what_is_asked_for_any_ring(n):
    mixer = _ring(n)
    want = np.array([12.0, 0.3, -0.2, 0.05])
    mix = mixer.mix(want[0], want[1:])
    assert not mix.saturated
    np.testing.assert_allclose(mixer.allocation @ mix.controls, want, atol=1e-9)
    hover = mixer.mix(12.0, np.zeros(3)).controls
    np.testing.assert_allclose(hover, 12.0 / n)  # level hover: every rotor carries the same share


@pytest.mark.parametrize("n", [4, 6, 8])
def test_mixer_keeps_roll_and_pitch_and_gives_up_height_then_yaw(n):
    mixer = _ring(n)
    near_full = 0.95 * 8.0 * n
    mix = mixer.mix(near_full, np.array([0.6, 0.0, 0.3]))
    got = mixer.allocation @ mix.controls
    assert got[1] == pytest.approx(0.6, rel=1e-6)  # the roll torque, in full
    assert got[0] < near_full  # less thrust, to make room for it
    assert mix.thrust_clipped and mix.yaw_share < 1.0
    assert np.all(mix.controls >= -1e-9) and np.all(mix.controls <= 8.0 + 1e-9)


def test_mixer_scales_an_impossible_roll_down_rather_than_breaking_the_ranges():
    mixer = _ring(4)
    mix = mixer.mix(10.0, np.array([50.0, 0.0, 0.0]))
    assert 0.0 < mix.attitude_share < 1.0
    assert np.all(mix.controls >= -1e-9) and np.all(mix.controls <= 8.0 + 1e-9)
    assert (mixer.allocation @ mix.controls)[1] > 0  # still rolls the right way, as hard as it can


def _drone_world(params: dict | None = None, pos=None) -> mujoco.MjModel:
    entry = {"name": "drone", "template": "drone", "params": params or {}}
    if pos is not None:
        entry["pos"] = pos
    return compile_world(World.model_validate({"robots": [entry]})).model


@pytest.mark.parametrize("rotors", [4, 6, 8])
def test_the_template_is_an_aircraft_whose_rotors_lift_it_and_steer_it(rotors):
    model = _drone_world({"rotors": rotors})
    assert robot_kind(model, "drone", "drone") == Kind.AIRCRAFT
    assert robot_kind(model, "drone") == Kind.AIRCRAFT  # found by its rotors, not only by its template's name
    frame = find_airframe(model, "drone")
    assert len(frame.rotors) == rotors
    assert frame.mass == pytest.approx(1.2, rel=0.01)
    assert np.linalg.matrix_rank(frame.mixer.allocation) == 4
    assert frame.mixer.max_thrust() / frame.weight == pytest.approx(2.4, rel=0.03)
    assert sorted({r.spin for r in frame.rotors}) == [-1, 1]  # half each way, or it could not hold its heading


def _session(model: mujoco.MjModel) -> tuple[SimSession, FlightController]:
    session = SimSession(model=model, data=mujoco.MjData(model))
    session.reset()
    fc = FlightController(model, "drone")
    session.set_controller("drone", fc)
    return session, fc


def test_it_takes_off_and_holds_a_hover_within_centimeters():
    session, fc = _session(_drone_world())
    session.step(250)
    assert fc.phase == "landed" and np.all(session.data.ctrl == 0.0)  # motors idle on the ground
    fc.fly(Mission([np.array([0.0, 0.0, 2.0])], False, 1.5))
    session.step(int(6 / 0.002))
    assert fc.mission.state == "arrived"
    errors = []
    for _ in range(int(10 / 0.002 / 10)):
        session.step(10)
        errors.append(np.linalg.norm(session.data.xpos[fc.frame.root] - [0.0, 0.0, 2.0]))
    assert max(errors) < 0.02


def test_a_gust_pushes_it_away_and_it_comes_back():
    session, fc = _session(_drone_world())
    fc.fly(Mission([np.array([0.0, 0.0, 2.0])], False, 1.5))
    session.step(int(6 / 0.002))
    fc.gusts([8.0, 0.0], 1.5, session.data.time)
    worst = 0.0
    for _ in range(int(2.5 / 0.002)):
        session.step(1)
        worst = max(worst, float(np.linalg.norm(session.data.xpos[fc.frame.root] - [0, 0, 2])))
    assert 0.05 < worst < 0.5  # it felt the gust, and held against it
    session.step(int(4 / 0.002))
    assert np.linalg.norm(session.data.xpos[fc.frame.root] - [0, 0, 2]) < 0.05
    assert fc.gust is None  # blown over: only the drag of its own (tiny) motion is left
    assert np.linalg.norm(session.data.xfrc_applied[fc.frame.root, :3]) < 0.01


def test_taking_the_controller_off_lets_go_of_the_air_it_pushed_with():
    session, fc = _session(_drone_world())
    fc.fly(Mission([np.array([0.0, 0.0, 2.0])], False, 1.5))
    session.step(int(1.5 / 0.002))
    assert np.any(session.data.xfrc_applied[fc.frame.root, :3] != 0.0)  # drag while climbing
    session.set_controller("drone", None)
    assert np.all(session.data.xfrc_applied[fc.frame.root, :3] == 0.0)


def test_a_drone_placed_in_the_air_hovers_there():
    session, fc = _session(_drone_world(pos=[0.0, 0.0, 1.5]))
    session.step(int(3 / 0.002))
    assert fc.phase == "flying"
    assert session.data.xpos[fc.frame.root][2] == pytest.approx(1.5, abs=0.05)


def test_rotors_that_all_spin_one_way_cannot_hold_a_heading():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="frame", pos=[0, 0, 1])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.02], mass=1.0)
    for i in range(4):
        a = math.pi / 4 + i * math.pi / 2
        body.add_site(name=f"r{i}", pos=[0.2 * math.cos(a), 0.2 * math.sin(a), 0])
        act = spec.add_actuator(name=f"r{i}", target=f"r{i}", trntype=mujoco.mjtTrn.mjTRN_SITE)
        act.gear = [0, 0, 1, 0, 0, 0.02]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [0, 6]
    world = mujoco.MjSpec()
    world.attach(spec, prefix="quad/", frame=world.worldbody.add_frame())
    model = world.compile()
    with pytest.raises(AirframeError, match="spin the other way"):
        find_airframe(model, "quad")


# ---- through the playground: move(to=...), stop, the keys ---------------------------------------------


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("air")
    playground.add_robot("drone", template="drone")
    return playground


def _trip(pg: Playground, to, seconds: float = 40.0) -> dict:
    trip = pg.behaviors.navigate("drone", to, None, None, seconds)
    pg.run_fast(lambda: trip.done, seconds + 1)
    return trip.result()


def test_move_to_a_point_in_the_air_flies_there_and_hovers(pg):
    result = _trip(pg, [1.5, -1.0, 2.0])
    assert result["arrived"] and result["hovering"]
    assert np.linalg.norm(np.subtract(result["position"], [1.5, -1.0, 2.0])) < 0.08
    robots = {r["name"]: r for r in pg.behaviors.robots()}
    assert robots["drone"]["kind"] == "aircraft"
    assert robots["drone"]["flight"]["state"] == "flying"


def test_a_list_of_points_is_flown_in_order_and_a_last_point_on_the_ground_lands(pg):
    result = _trip(pg, [[0, 0, 1.5], [2, 0, 1.5], [2, 2, 1.5], [2, 2, 0]])
    assert result["arrived"] and result["landed"]
    assert result["touchdown_m_s"] < 0.4
    assert np.hypot(result["position"][0] - 2, result["position"][1] - 2) < 0.1
    assert pg.behaviors.robots()[0]["flight"]["state"] == "landed"


def test_stop_hovers_and_says_how_to_land(pg):
    pg.behaviors.navigate("drone", [3.0, 0.0, 2.0], None, None, None)
    pg.run_fast(lambda: False, 3.0)
    out = pg.behaviors.stop("drone")
    assert out["hovering"] and "to=[x, y, 0]" in out["note"]
    here = np.array(pg.behaviors.pose("drone")["position"])
    pg.run_fast(lambda: False, 4.0)
    after = np.array(pg.behaviors.pose("drone")["position"])
    assert after[2] > 1.0  # still in the air
    assert np.linalg.norm(after - here) < 1.0  # braked and held


def test_the_keys_take_off_fly_forward_and_set_it_down(pg):
    pg.behaviors.move("drone", 1.0, 0.0)
    pg.run_fast(lambda: False, 3.0)
    pose = pg.behaviors.pose("drone")
    assert pose["position"][0] > 1.5 and pose["position"][2] > 0.8
    pg.behaviors.move("drone", 0.0, 0.0, climb=-0.8)  # down until it lands
    pg.run_fast(lambda: False, 6.0)
    assert pg.behaviors.robots()[0]["flight"]["state"] == "landed"


def test_a_tree_in_the_way_is_flown_over(pg):
    pg.add_object("tree", shape="tree", size=[1.2, 3.0], pos=[2.0, 0.0])
    result = _trip(pg, [4.0, 0.0, 1.5])
    assert result["arrived"]
    assert "over tree" in result["note"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_an_agent_flies_a_square_and_lands_through_the_mcp_tools(tmp_path, monkeypatch):
    import json

    from mcp import Client

    from robotforge.cli import find_free_port
    from robotforge.mcp_server import build_server

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    server = build_server(playground, find_free_port(8831))
    async with Client(server) as client:

        async def call(tool: str, args: dict) -> dict:
            return json.loads((await client.call_tool(tool, args)).content[0].text)

        await call("new_world", {"name": "sky"})
        await call("add_robot", {"name": "drone", "template": "drone"})
        up = await call("move", {"robot": "drone", "to": [0, 0, 2]})
        assert up["arrived"] and up["to"]["position"][2] == pytest.approx(2.0, abs=0.08)
        square = await call("move", {"robot": "drone", "to": "[[3, 0, 2], [3, 3, 2], [0, 3, 2], [0, 0, 2]]"})
        assert square["arrived"] and square["path_m"] == pytest.approx(12.0, abs=0.1)
        assert square["max_tracking_error_m"] < 0.15
        gust = await call("simulate", {"seconds": 1.5, "wind": [6, 0]})
        assert 0.02 < gust["wind"]["pushed_m"]["drone"] < 0.5
        stopped = await call("stop", {"robot": "drone"})
        assert stopped["hovering"]
        down = await call("move", {"robot": "drone", "to": [0, 0, 0]})
        assert down["arrived"] and down["landed"] and down["touchdown_m_s"] < 0.4
        robots = await call("robots", {})
        assert robots["robots"][0]["kind"] == "aircraft" and robots["robots"][0]["flight"]["state"] == "landed"
        described = await call("describe_model", {})
        assert described["aircraft"][0]["rotors"] == [f"drone/rotor_{k}" for k in range(1, 5)]


def test_wind_needs_an_aircraft_and_meters_per_second(pg):
    pg.remove("drone")
    pg.add_robot("rover", template="rover")
    with pytest.raises(ValueError, match="only pushes aircraft"):
        pg.simulate(1.0, wind=[5, 0])


def test_side_and_climb_are_for_aircraft(pg):
    pg.add_robot("rover", template="rover", pos=[0.0, 1.0])
    with pytest.raises(ValueError, match="only aircraft"):
        pg.behaviors.move("rover", 0.0, 0.0, climb=0.5)
