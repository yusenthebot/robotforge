"""Aircraft: the rotor mixer for any layout, the flight controller, trips through the air and the drone template."""

import math

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import Kind, robot_kind
from robotforge.behaviors.air import AIR, attach_air
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
    attach_air(session, "drone")
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
    wind = session.controllers[AIR + "drone"]
    wind.gusts([8.0, 0.0], 1.5, session.data.time)
    worst = 0.0
    for _ in range(int(2.5 / 0.002)):
        session.step(1)
        worst = max(worst, float(np.linalg.norm(session.data.xpos[fc.frame.root] - [0, 0, 2])))
    assert 0.05 < worst < 0.5  # it felt the gust, and held against it
    session.step(int(4 / 0.002))
    assert np.linalg.norm(session.data.xpos[fc.frame.root] - [0, 0, 2]) < 0.05
    assert wind.gust is None  # blown over: only the drag of its own (tiny) motion is left
    assert np.linalg.norm(session.data.xfrc_applied[fc.frame.root, :3]) < 0.01


def test_the_air_stays_whatever_flies_it_and_lets_go_when_taken_away():
    session, fc = _session(_drone_world())
    fc.fly(Mission([np.array([0.0, 0.0, 2.0])], False, 1.5))
    session.step(int(1.5 / 0.002))
    assert np.any(session.data.xfrc_applied[fc.frame.root, :3] != 0.0)  # drag while climbing
    session.set_controller("drone", None)  # another controller (LQR, MPC, a script) flies through the same air
    session.step(1)
    assert np.any(session.data.xfrc_applied[fc.frame.root, :3] != 0.0)
    session.set_controller(AIR + "drone", None)
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


def test_a_trunk_in_the_way_is_flown_round_and_a_long_wall_over(pg):
    pg.add_object("tree", shape="tree", size=[1.2, 3.0], pos=[2.0, 0.0])
    result = _trip(pg, [4.0, 0.0, 1.5])
    assert result["arrived"] and "round the" in result["note"] and "tree" in result["note"]  # not up to 3.6 m
    pg.add_object("wall", shape="wall", size=[12.0, 0.2, 2.0], pos=[6.0, 0.0], yaw=90)
    result = _trip(pg, [8.0, 0.0, 1.5])
    assert result["arrived"] and "over wall" in result["note"]  # shorter than 6 m round its end
    assert np.linalg.norm(np.subtract(result["position"], [8.0, 0.0, 1.5])) < 0.08


def _crown_world(pg):
    pg.add_object("tree", shape="tree", size=[2.0, 4.0], pos=[3.0, 0.0])
    pg.add_object("table", shape="table", size=[1.2, 0.8, 0.75], pos=[-2.0, 0.0], fixed=True)


def _touches(pg, robot: str = "drone") -> set[str]:
    """The names of what the aircraft touches now, but the floor."""
    m, d = pg.session.model, pg.session.data
    own = pg.session.controllers[robot].frame.descendants
    out = set()
    for c in d.contact[: d.ncon]:
        a, b = int(m.geom_bodyid[c.geom1]) in own, int(m.geom_bodyid[c.geom2]) in own
        if a != b:
            g = int(c.geom2 if a else c.geom1)
            out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"body {int(m.geom_bodyid[g])}")
    return out - {"floor"}


def _clean_trip(pg, to, seconds: float = 40.0) -> tuple[dict, set[str]]:
    trip = pg.behaviors.navigate("drone", to, None, None, seconds)
    touched: set[str] = set()
    while not trip.done and pg.session.data.time < 200:
        pg.run_fast(lambda: False, 0.05)
        touched |= _touches(pg)
    return trip.result(), touched


def test_a_point_under_a_crown_is_a_hover_there_not_a_landing_on_top(pg):
    _crown_world(pg)
    result, touched = _clean_trip(pg, [2.5, 0.0, 1.5])  # it once landed on the crown and fell through it
    assert result["arrived"] and result["hovering"] and "land" not in result["destination"]
    assert np.linalg.norm(np.subtract(result["position"], [2.5, 0.0, 1.5])) < 0.06
    assert result["clearance_m"] == pytest.approx(1.5 - 0.14, abs=0.05)
    assert not touched


def test_a_point_under_a_table_is_a_hover_under_it_not_a_landing_on_it(pg):
    _crown_world(pg)
    result, touched = _clean_trip(pg, [-2.0, 0.0, 0.4])  # it once landed on the tabletop at 0.89 m
    assert result["arrived"] and result["hovering"]
    assert result["position"][2] == pytest.approx(0.4, abs=0.03)
    assert not touched  # round the table's legs, not through one


def test_a_target_beside_a_trunk_is_reached_round_it_from_behind(pg):
    _crown_world(pg)
    _trip(pg, [5.0, 0.0])
    result, touched = _clean_trip(pg, [2.5, 0.0])  # over the tree meant down through its crown
    assert result["arrived"] and "round the" in result["note"]
    assert np.hypot(result["position"][0] - 2.5, result["position"][1]) < 0.06
    assert not touched


def test_it_comes_in_under_a_crown_to_land_beneath_it(pg):
    _crown_world(pg)
    _trip(pg, [2.5, 3.0, 5.5])
    result, touched = _clean_trip(pg, [2.5, 0.3, 0.0])
    assert result["arrived"] and result["landed"] and "under tree" in result["note"]
    assert np.hypot(result["position"][0] - 2.5, result["position"][1] - 0.3) < 0.1
    assert not touched


def test_a_point_inside_something_is_refused_and_z_is_a_world_height(pg):
    _crown_world(pg)
    with pytest.raises(ValueError, match="inside tree.*world's zero"):
        pg.behaviors.navigate("drone", [3.0, 0.0, 3.5], None, None, 10)
    result, _ = _clean_trip(pg, [-2.0, 0.0, 0.75])  # the tabletop's height: a landing on it
    assert result["arrived"] and result["landed"] and result["position"][2] > 0.75


def test_a_landing_that_comes_down_somewhere_else_did_not_arrive(pg):
    from robotforge.behaviors import airways

    _trip(pg, [0.0, 0.0, 1.5])
    trip = pg.behaviors.navigate("drone", [1.0, 0.0, 0.0], None, None, 30)
    trip.touchdown = trip.touchdown + np.array([0.0, 0.0, 1.0])  # as if it had settled on top of something
    pg.run_fast(lambda: trip.done, 31)
    result = trip.result()
    assert not result["arrived"] and result["why"] == "landed_elsewhere" and "not on the spot" in result["note"]
    assert isinstance(trip, airways.FlightTrip)


def _drag(pg, body: int, by, seconds: float = 1.2) -> np.ndarray:
    """Drag a body in the viewer's way (SimSession's spring on the cursor) by `by` metres, and let it go."""
    session = pg.session
    pick = session.data.xpos[body].copy()
    session.start_drag(body, pick)
    steps = int(seconds / 0.02)
    for k in range(steps):
        session.move_drag(pick + np.asarray(by, float) * min(1.0, (k + 1) / (0.7 * steps)))
        pg.run_fast(lambda: False, 0.02)
    session.end_drag()
    return session.data.xpos[body].copy()


def test_dragged_in_the_viewer_it_holds_where_it_is_let_go(pg):
    _trip(pg, [0.0, 0.0, 2.0])
    root = pg.session.controllers["drone"].frame.root
    let_go = _drag(pg, root, [1.5, 0.0, 0.0])
    assert let_go[0] > 1.3
    pg.run_fast(lambda: False, 3.0)  # it flew back to x 0.03
    here = pg.session.data.xpos[root]
    assert np.linalg.norm(here - let_go) < 0.1 and here[2] > 1.8


def test_lifted_off_the_ground_by_hand_it_hovers_where_it_is_let_go(pg):
    pg.run_fast(lambda: False, 0.5)
    fc = pg.session.controllers["drone"]
    assert fc.phase == "landed"
    let_go = _drag(pg, fc.frame.root, [0.0, 0.0, 1.0], seconds=1.5)
    pg.run_fast(lambda: False, 3.0)
    assert fc.phase == "flying" and np.linalg.norm(pg.session.data.xpos[fc.frame.root] - let_go) < 0.15


# ---- aircraft built from parts ------------------------------------------------------------------------


def test_a_hexacopter_built_from_parts_flies_with_its_own_rotor_layout(pg):
    from robotforge.design.builder.guide import EXAMPLE_HEXACOPTER

    pg.remove("drone")
    pg.add_robot("hexa", build=EXAMPLE_HEXACOPTER)
    robots = {r["name"]: r for r in pg.behaviors.robots()}
    assert robots["hexa"]["kind"] == "aircraft" and robots["hexa"]["flight"]["rotors"] == 6
    trip = pg.behaviors.navigate("hexa", [1.0, 1.0, 2.0], None, None, 30)
    pg.run_fast(lambda: trip.done, 31)
    assert trip.result()["arrived"]
    fc = pg.session.controllers["hexa"]
    assert [r.spin for r in fc.frame.rotors] == [1, -1, 1, -1, 1, -1]  # alternating round the ring by itself
    angles = sorted(round(math.degrees(math.atan2(r.position[1], r.position[0]))) % 360 for r in fc.frame.rotors)
    assert angles == [0, 60, 120, 180, 240, 300]  # where its builder put them
    errors = []
    for _ in range(50):
        pg.run_fast(lambda: False, 0.1)
        errors.append(np.linalg.norm(np.subtract(pg.behaviors.pose("hexa")["position"], [1.0, 1.0, 2.0])))
    assert max(errors) < 0.05


def test_a_mirrored_rotor_turns_the_other_way():
    build = {
        "base": {"size": [0.2, 0.2, 0.05], "mass": 0.6},
        "parts": [
            {"name": "left_front", "at": [0.15, 0.15, 0.03], "shape": "rotor", "size": [0.2], "mirror": True},
            {"name": "left_back", "at": [-0.15, 0.15, 0.03], "shape": "rotor", "size": [0.2], "spin": "cw"},
            {"name": "right_back", "at": [-0.15, -0.15, 0.03], "shape": "rotor", "size": [0.2]},
        ],
    }
    model = compile_world(World.model_validate({"robots": [{"name": "quad", "build": build}]})).model
    spins = {r.name: r.spin for r in find_airframe(model, "quad").rotors}
    assert spins == {"left_front": 1, "right_front": -1, "left_back": -1, "right_back": 1}


def test_rotor_parts_are_bolted_on_and_thrust_is_for_rotors():
    from robotforge.design import builder

    with pytest.raises(ValueError, match="a rotor is bolted on"):
        builder.parse({"parts": [{"name": "r", "shape": "rotor", "size": [0.2], "joint": "hinge"}]})
    with pytest.raises(ValueError, match="thrust is for rotor parts"):
        builder.parse({"parts": [{"name": "w", "shape": "wheel", "size": [0.1, 0.02], "thrust": 5}]})


# ---- library aircraft (offline stand-ins in tests/fixtures/menagerie) -------------------------------------


@pytest.fixture
def library_pg(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("air")
    return playground


def test_the_skydio_x2_flies_on_its_own_four_thrust_motors(library_pg):
    from robotforge.design import library

    assert library.get("Skydio X2").kind == "aircraft"
    out = library_pg.add_robot("x2", library="skydio_x2")
    assert any("Skydio X2 from MuJoCo Menagerie (Apache-2.0" in n for n in out["notes"])
    robot = library_pg.behaviors.robots()[0]
    assert robot["kind"] == "aircraft" and robot["model"] == "Skydio X2"
    assert robot["flight"]["rotors"] == 4 and robot["flight"]["thrust_to_weight"] == pytest.approx(4.0, rel=0.02)
    fc = FlightController(library_pg.session.model, "x2")
    assert [r.name for r in fc.frame.rotors] == ["thrust1", "thrust2", "thrust3", "thrust4"]
    assert [r.spin for r in fc.frame.rotors] == [1, -1, 1, -1]  # from the signs of the model's reaction torques
    trip = library_pg.behaviors.navigate("x2", [[0, 0, 1.5], [1.5, 1.0, 1.5]], None, None, 30)
    library_pg.run_fast(lambda: trip.done, 31)
    result = trip.result()
    assert result["arrived"], result
    assert np.linalg.norm(np.subtract(result["position"], [1.5, 1.0, 1.5])) < 0.08


def test_the_crazyflie_gets_its_real_rotors_and_flies(library_pg):
    out = library_pg.add_robot("cf", library="crazyflie")
    assert any("stand in for the model's body_thrust" in n for n in out["notes"])
    names = [mujoco.mj_id2name(library_pg.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(4)]
    assert names == ["cf/m1", "cf/m2", "cf/m3", "cf/m4"]
    trip = library_pg.behaviors.navigate("cf", [[0, 0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]], None, None, 30)
    library_pg.run_fast(lambda: trip.done, 31)
    result = trip.result()
    assert result["arrived"] and result["landed"], result


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


def test_wind_stepped_a_little_at_a_time_keeps_blowing(pg):
    """An agent watched a gust by calling simulate(0.05, wind) a hundred times: each call started the gust afresh
    from still air, so the drone was never pushed at all."""
    _trip(pg, [0.0, 0.0, 1.5])
    pg.run_fast(lambda: False, 2.0)
    start = np.array(pg.behaviors.pose("drone")["position"])
    furthest = 0.0
    for _ in range(40):
        pg.simulate(0.05, wind=[8.0, 0.0])
        furthest = max(furthest, float(np.linalg.norm(np.array(pg.behaviors.pose("drone")["position"]) - start)))
    assert furthest > 0.03
    pg.simulate(6.0)  # the wind dies down and it comes back
    assert np.linalg.norm(np.array(pg.behaviors.pose("drone")["position"]) - start) < 0.03


def test_wind_needs_an_aircraft_and_meters_per_second(pg):
    pg.remove("drone")
    pg.add_robot("rover", template="rover")
    with pytest.raises(ValueError, match="only pushes aircraft"):
        pg.simulate(1.0, wind=[5, 0])


def test_a_world_edit_while_it_hovers_leaves_it_hovering(pg):
    _trip(pg, [1.0, 0.0, 1.8])
    pg.add_object("ball", shape="sphere", size=[0.2], pos=[-1.0, 0.0])  # recompiles the world
    pg.run_fast(lambda: False, 3.0)
    pose = pg.behaviors.pose("drone")
    assert np.linalg.norm(np.subtract(pose["position"], [1.0, 0.0, 1.8])) < 0.05
    assert pg.behaviors.robots()[0]["flight"]["state"] == "flying"


def test_a_reset_puts_it_back_on_the_ground_with_its_motors_idle(pg):
    _trip(pg, [1.0, 0.0, 1.8])
    pg.control("reset")
    pg.run_fast(lambda: False, 1.0)
    assert pg.behaviors.pose("drone")["position"][2] < 0.2
    assert pg.behaviors.robots()[0]["flight"]["state"] == "landed"
    assert np.all(pg.session.data.ctrl == 0.0)


def test_flying_to_a_named_thing_hovers_over_it(pg):
    pg.add_object("table", shape="table", size=[0.8, 0.6, 0.75], pos=[2.0, 1.0], fixed=True)
    result = _trip(pg, "table")
    assert result["arrived"] and result["destination"]["object"] == "table"
    x, y, z = result["position"]
    assert np.hypot(x - 2.0, y - 1.0) < 0.05
    assert z > 0.75 + 0.9  # a metre over its top, not at the table's middle


def test_side_and_climb_are_for_aircraft(pg):
    pg.add_robot("rover", template="rover", pos=[0.0, 1.0])
    with pytest.raises(ValueError, match="only aircraft"):
        pg.behaviors.move("rover", 0.0, 0.0, climb=0.5)


# ---- the hover task: the flight controller, LQR and MPC on the same aircraft --------------------------------


def test_hover_compares_the_flight_controller_lqr_and_mpc_after_a_push(pg):
    import copy

    from robotforge.control.compare import compare_methods

    _trip(pg, [0.0, 0.0, 1.5])
    pg.run_fast(lambda: False, 1.0)
    table = compare_methods(
        "hover",
        ["cascade", "lqr", "mpc", "none"],
        5.0,
        model=pg.session.model,
        robot="drone",
        data=copy.copy(pg.session.data),
        seed=None,
        push_at=2.0,
    )
    results = {r.method: r for r in table.results}
    for method in ("cascade", "lqr"):
        assert results[method].success, results[method].report  # back on the spot after the shove
        assert results[method].report["position_error_m"] < 0.05
    assert not results["mpc"].failed
    assert results["none"].failed  # without control it drops out of the sky
    assert results["lqr"].reward > results["none"].reward


def test_after_lqr_flew_it_stop_hands_it_back_to_the_flight_controller(pg):
    from robotforge.control.methods import make_controller
    from robotforge.control.tasks import make_task

    _trip(pg, [0.5, 0.0, 1.2])
    session = pg.session
    task = make_task("hover", session.model, "drone")
    task.begin(session.data)
    session.set_controller("drone", make_controller("lqr", task, session.model, session.data))
    pg.run_fast(lambda: False, 2.0)
    assert np.linalg.norm(np.subtract(pg.behaviors.pose("drone")["position"], [0.5, 0.0, 1.2])) < 0.05
    out = pg.behaviors.stop("drone")
    assert "flight controller took over" in out["note"]
    pg.run_fast(lambda: False, 3.0)  # it holds its place instead of dropping with its rotors at rest
    assert np.linalg.norm(np.subtract(pg.behaviors.pose("drone")["position"], [0.5, 0.0, 1.2])) < 0.08
    assert pg.behaviors.robots()[0]["flight"]["state"] == "flying"


def test_robots_and_describe_report_what_really_flies_it_and_where(pg):
    from robotforge.control.methods import make_controller
    from robotforge.control.tasks import make_task

    _trip(pg, [0.5, 0.0, 2.0])
    session = pg.session
    task = make_task("hover", session.model, "drone")
    task.begin(session.data)
    session.set_controller("drone", make_controller("lqr", task, session.model, session.data))
    pg.run_fast(lambda: False, 2.0)
    info = pg.behaviors.robots()[0]["flight"]  # it said landed at 0 m, flown by the flight controller
    assert info["state"] == "flying" and info["controller"] == "lqr"
    assert info["altitude_m"] == pytest.approx(2.0 - 0.14, abs=0.05)
    assert "stop('drone')" in info["note"]
    (entry,) = pg.describe()["aircraft"]
    assert entry["controller"] == "lqr" and entry["state"] == "flying"
    session.set_controller("drone", make_controller("none", task, session.model, session.data))
    pg.run_fast(lambda: False, 3.0)
    info = pg.behaviors.robots()[0]["flight"]
    assert info["state"] == "landed" and info["controller"] == "none" and info["altitude_m"] == 0.0


def test_the_hover_task_refuses_a_robot_that_does_not_fly(pg):
    from robotforge.control.tasks import TaskError, make_task

    pg.add_robot("rover", template="rover", pos=[0.0, 1.0])
    with pytest.raises(TaskError, match="no rotors"):
        make_task("hover", pg.session.model, "rover")
