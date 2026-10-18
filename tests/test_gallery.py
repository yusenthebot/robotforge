"""The gallery: ready-made designs added by name, kept as their parts, and each one doing its job for real."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors.manager import BehaviorError
from robotforge.cli import find_free_port
from robotforge.design import gallery
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"  # offline stand-ins for the Go2 and the PiPER
NETWORK = os.environ.get("ROBOTFORGE_NETWORK_TESTS") == "1"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("gallery")
    return playground


def _run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def _tools(pg: Playground, robot: str) -> np.ndarray:
    m, d = pg.session.model, pg.session.data
    return np.array([d.site_xpos[m.site(f"{arm}/tcp").id] for arm in pg.behaviors.arms(robot)])


def _moved(pg: Playground, robot: str, speed: float, seconds: float, turn: float = 0.0) -> tuple[float, dict]:
    start = pg.behaviors.pose(robot)["position"]
    pg.behaviors.move(robot, speed, turn)
    _run(pg, seconds)
    pg.behaviors.stop(robot)
    _run(pg, 1.0)
    end = pg.behaviors.pose(robot)
    return float(end["position"][0] - start[0]), end


def _turned(pg: Playground, robot: str, turn: float, seconds: float) -> tuple[float, dict, float]:
    """Turn in place: (degrees turned, the pose after stopping, how far it wandered in meters)."""
    start = pg.behaviors.pose(robot)
    pg.behaviors.move(robot, 0.0, turn)
    total, last = 0.0, start["heading_deg"]
    for _ in range(round(seconds / 0.1)):
        _run(pg, 0.1)
        heading = pg.behaviors.pose(robot)["heading_deg"]
        total += (heading - last + 180) % 360 - 180
        last = heading
    pg.behaviors.stop(robot)
    _run(pg, 1.0)
    end = pg.behaviors.pose(robot)
    return total, end, float(np.linalg.norm(np.subtract(end["position"][:2], start["position"][:2])))


def _pick(pg: Playground, arm: str, obj: str, target) -> dict:
    motion, info = pg.behaviors.pick_and_place(arm, obj, target)
    pg.run_fast(lambda: motion.done, 40)
    _run(pg, 0.5)
    return pg.behaviors.placed(info["object"], info["region"])


# ---- adding a design ------------------------------------------------------------------------------------


def test_a_design_is_added_by_name_and_kept_as_its_parts(pg):
    pg.add_robot("hopper", design="Wheel Biped")
    entry = pg.current_world().robots[0]
    assert entry.build == gallery.build("wheel_biped") and entry.template is None
    # It is a robot built from parts now: change it part by part like any other.
    pg.update_part("hopper", "robot", {"color": "purple"})
    pg.update_part("hopper", "left_shin", {"size": [0.025, 0.14]})
    entry = pg.current_world().robots[0]
    assert entry.build["color"] == "purple" and entry.build["parts"][2]["size"] == [0.025, 0.14]
    assert "right_shin" in [pg.session.model.body(b).name.split("/")[-1] for b in range(pg.session.model.nbody)]


def test_swapping_a_designs_mounted_robot_drops_the_old_ones_pose(pg):
    """The folded PiPER's home pose stayed on the mount when another robot replaced it: an SO-100 has no joint2 and
    the edit failed, and a template named beside the old library robot was refused as two robots."""
    pg.add_robot("dog", design="go2_arm")
    pg.update_part("dog", "arm", {"robot": {"library": "so_arm100"}})
    assert pg.current_world().robots[0].build["parts"][-1]["robot"] == {"library": "so_arm100"}
    pg.update_part("dog", "arm", {"robot": {"template": "arm"}})
    assert pg.current_world().robots[0].build["parts"][-1]["robot"] == {"template": "arm"}
    pg.update_part("dog", "arm", {"robot": {"params": {"reach": 0.4}}})  # the same robot: changes merge
    assert pg.current_world().robots[0].build["parts"][-1]["robot"] == {"template": "arm", "params": {"reach": 0.4}}


def test_changing_one_setting_of_a_mounted_robot_keeps_the_others(pg):
    """Recoloring the lift picker's arm replaced all its params with the color: it was rebuilt as a 0.45 m arm on the
    hobby servos its design says sag 10 cm, and only a build note said so."""
    pg.add_robot("p", design="lift_picker")
    pg.update_part("p", "arm", {"robot": {"params": {"color": "red"}}})
    mount = next(p for p in pg.current_world().robots[0].build["parts"] if p["name"] == "arm")["robot"]
    assert mount["params"] == {"reach": 0.6, "base_height": 0.04, "servo": "bldc_joint", "color": "red"}
    assert mount["home"] == {"yaw": 180}
    pg.update_part("p", "arm", {"robot": {"params": {"base_height": None}, "home": {"shoulder": 20}}})
    mount = next(p for p in pg.current_world().robots[0].build["parts"] if p["name"] == "arm")["robot"]
    assert mount["params"] == {"reach": 0.6, "servo": "bldc_joint", "color": "red"}  # null takes one setting away
    assert mount["home"] == {"yaw": 180, "shoulder": 20}


def test_recoloring_a_design_recolors_its_accent_parts(pg):
    from robotforge.design import components

    pg.add_robot("scorpion", design="scorpion")
    pg.update_part("scorpion", "robot", {"color": "purple"})
    m = pg.session.model
    for part in ("base", "femur_4", "tail_2"):  # the body and the parts in its accent color
        body = m.body(f"scorpion/{part}").id
        rgba = [m.geom_rgba[g] for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        assert any(np.allclose(c, components.color("purple"), atol=1e-3) for c in rgba), part


def test_a_design_comes_alone_and_a_wrong_name_says_what_there_is(pg):
    with pytest.raises(PlaygroundError, match="leave out template.*update_part"):
        pg.add_robot("x", template="arm", design="centaur")
    with pytest.raises(PlaygroundError, match="Did you mean centaur"):
        pg.add_robot("x", design="centuar")
    with pytest.raises(PlaygroundError, match="quadruped is a template, not a design"):
        pg.add_robot("x", design="quadruped")
    with pytest.raises(ValueError, match=r"centaur is a ready-made design, not a template: .*design=.centaur."):
        pg.add_robot("x", template="centaur")
    assert pg.current_world().robots == []


def test_every_design_has_a_short_summary_that_names_how_to_drive_it():
    catalog = gallery.catalog()
    assert len(catalog) >= 8
    for entry in catalog:
        assert set(entry) == {"name", "summary"}
        assert len(entry["summary"]) < 200, entry
        assert any(tool in entry["summary"] for tool in ("move", "move_arm", "pick_and_place")), entry


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_agents_list_the_designs_and_add_one_in_one_call(tmp_path, monkeypatch):
    from mcp import Client

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_NO_BROWSER", "1")
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    server = build_server(playground, find_free_port(18800))
    async with Client(server) as client:
        listed = json.loads((await client.call_tool("list_templates", {})).content[0].text)
        assert {"centaur", "mars_rover", "wheel_biped"} <= {d["name"] for d in listed["designs"]}
        await client.call_tool("new_world", {"name": "zoo"})
        added = await client.call_tool("add_robot", {"name": "rover", "design": "mars_rover"})
        assert not added.is_error, added.content[0].text
        robots = json.loads((await client.call_tool("robots", {})).content[0].text)["robots"]
        assert robots[0]["kind"] == "rover" and robots[0]["arms"] == ["rover/hand"]
        world = json.loads((await client.call_tool("show_world", {})).content[0].text)
        assert "left_rocker" in json.dumps(world)  # the parts are in world.yaml, ready for update_part
    await asyncio.sleep(0)


# ---- every design stands, and does its job ----------------------------------------------------------------


@pytest.mark.parametrize("design", list(gallery.DESIGNS))
def test_every_design_compiles_and_stands_for_five_seconds(pg, design):
    pg.add_robot("r", design=design)
    _run(pg, 0.5)
    start, tools = pg.behaviors.pose("r"), _tools(pg, "r")
    _run(pg, 5.0)
    end = pg.behaviors.pose("r")
    assert end["upright"], design
    assert np.linalg.norm(np.subtract(end["position"], start["position"])) < 0.05, (design, start, end)
    if len(tools):  # the arms hold their pose on their servos
        assert np.abs(_tools(pg, "r") - tools).max() < 0.02, design


# The viewer's A and D keys turn rovers at 2 rad/s, balancers at 1.5 and walkers at 1 (TURN_RATE in app.js).
TURNS = {
    "wheel_biped": 1.5,
    "wheel_dog": 2.0,
    "mars_rover": 2.0,
    "lift_picker": 2.0,
    "mobile_panda": 2.0,
    "centaur": 1.0,
    "scorpion": 1.0,
    "go2_arm": 1.0,
}


@pytest.mark.parametrize("design", list(TURNS))
def test_every_mobile_design_turns_on_the_spot_at_full_key_and_stays_up(pg, design):
    """Nothing turned the designs before: the four-wheel Panda base could not skid-steer its 57 kg round (3 degrees in
    5 s), and the wheel dog rolled onto its side at 2 rad/s."""
    pg.add_robot("r", design=design)
    _run(pg, 0.5)
    turned, end, wandered = _turned(pg, "r", TURNS[design], 3.0)
    assert turned > 90 and end["upright"], (design, turned, end)
    assert wandered < 0.3, (design, wandered)


def test_the_wheel_dog_turns_in_its_tall_stance_too(pg):
    """Its summary invites changing the stance: standing tall the plain leg servos gave way and it tipped over."""
    pg.add_robot("dog", design="wheel_dog")
    tall = {"thigh": 15, "shin": -30}
    legs = {
        f"dog/{side}_{part}_{i}": np.radians(a)
        for side in ("left", "right")
        for i in (1, 2)
        for part, a in tall.items()
    }
    pg.set_controls(legs)
    _run(pg, 2.0)
    turned, end, _ = _turned(pg, "dog", 0.8, 3.0)
    assert turned > 60 and end["upright"], (turned, end)


def test_the_wheel_biped_balances_and_drives(pg):
    pg.add_robot("hopper", design="wheel_biped")
    assert pg.behaviors.robots()[0]["kind"] == "balancer"
    ahead, end = _moved(pg, "hopper", 0.4, 3.0)
    assert ahead > 0.9 and end["upright"]


def test_the_wheel_dog_rolls_on_its_wheel_feet(pg):
    pg.add_robot("dog", design="wheel_dog")
    assert pg.behaviors.robots()[0]["kind"] == "rover"
    ahead, end = _moved(pg, "dog", 0.4, 3.0)
    assert ahead > 1.0 and end["upright"]


def test_the_mars_rover_climbs_a_curb_and_samples_a_rock(pg):
    pg.add_robot("rover", design="mars_rover")
    pg.add_terrain("curb", pos=[0.7, 0.0], params={"height": 0.06})
    ahead, end = _moved(pg, "rover", 0.3, 6.0)
    assert ahead > 1.4 and end["upright"]  # over the 6 cm curb, a wheel-radius step
    x, y = end["position"][:2]
    pg.add_object("rock", shape="box", size=[0.04, 0.04, 0.04], pos=[x + 0.42, y - 0.08])
    pg.add_object("cache", shape="bin", size=[0.16, 0.16, 0.06], pos=[x + 0.42, y + 0.17], fixed=True)
    _run(pg, 0.5)
    assert _pick(pg, "rover/hand", "rock", "cache")["success"]


def test_the_lift_picker_raises_its_column_to_move_a_cube_to_a_higher_table(pg):
    """The arm starts folded backward, low over the base, with a table on either side of it: it rises before it swings
    around (swinging at the folded height it hit the higher table and spun the base around)."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("low", shape="table", size=[0.26, 0.26, 0.45], pos=[0.52, -0.14], fixed=True)
    pg.add_object("high", shape="table", size=[0.26, 0.26, 0.62], pos=[0.45, 0.26], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.52, -0.14, 0.47])
    _run(pg, 0.5)
    lift = pg.session.model.joint("picker/lift").qposadr[0]
    assert _pick(pg, "picker", "cube", "high")["success"]
    assert pg.session.data.qpos[lift] > 0.15  # it reached the higher table by raising the column
    assert pg.behaviors.pose("picker")["position"][:2] == pytest.approx([0.0, 0.0], abs=0.02)  # the base held still


def test_a_lift_arm_out_of_reach_is_told_to_bring_the_cube_nearer(pg):
    """Its reach was measured from the foot of the column with the lift and the arm stretched out straight (0.98 m), so
    a cube too far away was said to be within reach and to be moved further from the base."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("far", shape="table", size=[0.5, 0.3, 0.4], pos=[0.75, 0.15], fixed=True)
    pg.add_object("near", shape="table", size=[0.26, 0.26, 0.45], pos=[0.45, -0.25], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.75, 0.1, 0.42])
    _run(pg, 0.5)
    # The message says where the gripper points down at that height and sends the cube nearer, to a spot in it. (Told
    # to stay where it is: by default the picker drives over and fetches it, tests/test_fetch.py.)
    with pytest.raises(
        BehaviorError, match=r"it is 0.66 m from the arm's base.*points down only .* move the cube nearer"
    ):
        pg.behaviors.pick_and_place("picker", "cube", "near", go=False)
    pg.update_object("cube", pos=[0.6, 0.1, 0.42])  # just beyond its 0.49 m with the gripper down
    _run(pg, 0.5)
    with pytest.raises(BehaviorError, match="move the cube nearer") as refused:
        pg.behaviors.pick_and_place("picker", "cube", "near", go=False)
    assert "further" not in str(refused.value)


def test_the_mobile_panda_drives_then_lifts_a_cube_off_the_floor_onto_a_table(pg):
    pg.add_robot("mm", design="mobile_panda")
    info = pg.behaviors.robots()[0]
    assert info["kind"] == "rover" and info["arms"] == ["mm/arm"]
    ahead, end = _moved(pg, "mm", 0.4, 3.0)
    assert ahead > 1.0 and end["upright"]
    x, y = end["position"][:2]
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[x + 0.62, y + 0.22])
    pg.add_object("table", shape="table", size=[0.35, 0.35, 0.45], pos=[x + 0.75, y - 0.25], fixed=True)
    _run(pg, 0.5)
    assert _pick(pg, "mm/arm", "cube", "table")["success"]


def test_the_centaur_walks_and_picks_with_one_hand(pg):
    pg.add_robot("centaur", design="centaur")
    info = pg.behaviors.robots()[0]
    assert info["kind"] == "quadruped" and info["arms"] == ["centaur/left_hand", "centaur/right_hand"]
    ahead, end = _moved(pg, "centaur", 0.3, 4.0)
    assert ahead > 0.7 and end["upright"]
    x, y = end["position"][:2]
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[x + 0.39, y + 0.12])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[x + 0.31, y + 0.31], fixed=True)
    _run(pg, 0.5)
    assert _pick(pg, "centaur/left_hand", "cube", "bin")["success"]


def test_an_arm_built_from_parts_answers_to_the_parts_it_hangs_from(pg):
    pg.add_robot("centaur", design="centaur")
    assert pg.behaviors.resolve("centaur/left_arm", arm=True) == "centaur/left_hand"
    assert pg.behaviors.resolve("Centaur/Right_Fore", arm=True) == "centaur/right_hand"
    with pytest.raises(ValueError, match="not an arm of centaur; use centaur/left_hand or centaur/right_hand"):
        pg.behaviors.resolve("centaur/torso", arm=True)  # both arms hang from it: which one is not said
    target = [0.32, 0.2, 0.15]
    motion = pg.behaviors.move_arm("centaur/left_arm", target)
    pg.run_fast(lambda: motion.done, 10)
    _run(pg, 0.5)
    assert np.linalg.norm(pg.session.data.site_xpos[motion.site] - target) < 0.015


def test_a_robots_body_is_not_its_arm_and_other_tools_still_call_a_part_a_part(pg):
    """Looking an arm up by any part it hangs from also matched the robot's own base, for every tool: pose("rover/base")
    answered with the arm's pose (on the Mars rover: heading 180, not upright, as if it had fallen over)."""
    carrier = {
        "motor": "jga25",
        "base": {"shape": "box", "size": [0.34, 0.22, 0.06], "mass": 3.0},
        "parts": [
            {"name": "left_wheel", "at": [0, 0.13, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
            {"name": "front_caster", "at": [0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
            {"name": "rear_caster", "at": [-0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
            {"name": "arm", "at": "top", "shift": [0.06, 0, 0], "robot": {"template": "arm"}},
        ],
    }
    pg.add_robot("cart", build=carrier)
    pg.add_robot("rover", design="mars_rover", pos=[1.5, 0])
    for robot, arm in (("cart", "cart/arm"), ("rover", "rover/hand")):
        with pytest.raises(ValueError, match=f"'{robot}/base' is a part of {robot}, not a robot; use {robot}$"):
            pg.behaviors.pose(f"{robot}/base")
        with pytest.raises(ValueError, match=f"'{robot}/base' is a part of {robot}, not a robot"):
            pg.behaviors.move(f"{robot}/base", 0.2, 0.0)
        with pytest.raises(ValueError, match=f"'{robot}/base' is not an arm of {robot}; use {arm}$"):
            pg.behaviors.resolve(f"{robot}/base", arm=True)
    # The arm's own parts still find it, but only for the arm tools.
    assert pg.behaviors.resolve("rover/arm_upper", arm=True) == "rover/hand"
    with pytest.raises(ValueError, match="'rover/arm_upper' is a part of rover"):
        pg.behaviors.pose("rover/arm_upper")


def test_the_scorpion_walks_on_six_legs_and_picks_with_its_tail(pg):
    pg.add_robot("scorpion", design="scorpion")
    info = pg.behaviors.robots()[0]
    assert info["kind"] == "hexapod" and info["arms"] == ["scorpion/claw"]
    ahead, end = _moved(pg, "scorpion", 0.25, 4.0)
    assert ahead > 0.7 and end["upright"]
    x, y = end["position"][:2]
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[x + 0.3, y])
    _run(pg, 0.5)
    assert _pick(pg, "scorpion/claw", "cube", [x + 0.3, y - 0.14])["success"]


def _go2_moves_a_cube(pg: Playground, walk: bool) -> tuple[dict, list[str]]:
    """Have the Go2's arm move a cube between two tables (after a walk, or straight away); (placed, the steps of the
    motion that moved joint by joint)."""
    pg.add_robot("dog", design="go2_arm")
    info = pg.behaviors.robots()[0]
    assert info["kind"] == "quadruped" and info["arms"] == ["dog/arm"]
    x, y = pg.behaviors.pose("dog")["position"][:2]
    if walk:
        ahead, end = _moved(pg, "dog", 0.4, 3.0)
        assert ahead > 0.8 and end["upright"]
        x, y = end["position"][:2]
    pg.add_object("left_table", shape="table", size=[0.22, 0.22, 0.25], pos=[x + 0.42, y - 0.19], fixed=True)
    pg.add_object("right_table", shape="table", size=[0.22, 0.22, 0.25], pos=[x + 0.38, y + 0.25], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[x + 0.42, y - 0.19, 0.27])
    _run(pg, 0.5)
    motion, placed = pg.behaviors.pick_and_place("dog/arm", "cube", "right_table")
    joint_by_joint = []
    while not motion.done and pg.session.data.time < 60:
        _run(pg, 0.05)
        if motion._branch is not None and motion.current_label not in joint_by_joint:
            joint_by_joint.append(motion.current_label)
    _run(pg, 0.5)
    return pg.behaviors.placed("cube", placed["region"]), joint_by_joint


def test_the_go2_with_an_arm_walks_and_moves_a_cube_between_tables(pg):
    """The PiPER rides folded with its gripper pointing forward. Turning the wrist over along a straight path to its
    first goal left it sideways on the way down and it swept the real PiPER's cube off the table (the test stand-in's
    simpler shapes missed the cube); a tool starting that far from pointing down now gets there joint by joint."""
    placed, joint_by_joint = _go2_moves_a_cube(pg, walk=True)
    assert placed["success"]
    assert joint_by_joint[:1] == ["approach"]


# Floor picks where the summary says the mobile Panda works: in front and to the sides, 0.4 to 0.7 m from the arm and
# 10 cm or more clear of the base (drawn at random; 40 of 40 such picks landed with the real Panda).
MOBILE_PANDA_FLOOR = [
    ([0.177, 0.637], [0.395, -0.469]),  # front left to front right
    ([0.617, -0.017], [-0.188, -0.53]),  # straight ahead to beside the right wheel
    ([0.058, -0.641], [0.059, 0.612]),  # right side to left side, round the front
    ([0.604, 0.129], [-0.203, 0.452]),  # ahead to beside the left wheel
]


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize(("cube", "target"), MOBILE_PANDA_FLOOR)
def test_real_mobile_panda_works_the_floor_in_front_and_to_the_sides(tmp_path, monkeypatch, cube, target):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_MENAGERIE", raising=False)
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("real")
    playground.add_robot("r", design="mobile_panda")
    playground.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=cube)
    _run(playground, 0.5)
    placed = _pick(playground, "r/arm", "cube", target)
    assert placed["success"], placed


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_go2_arm_carries_a_floor_cube_from_one_side_to_the_other_over_the_dogs_head(tmp_path, monkeypatch):
    """Swung round in front at 10 cm the gripper could not point down without hitting the head and flung the cube; at
    30 cm the cube under the gripper struck the head and was knocked out of its fingers. (The offline stand-in's
    simpler shapes clear the head either way: tests/test_builder.py covers the same with a built robot.)"""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_MENAGERIE", raising=False)
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("real")
    playground.add_robot("dog", design="go2_arm")
    playground.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, 0.3])
    _run(playground, 0.5)
    placed = _pick(playground, "dog/arm", "cube", [0.0, -0.3])
    assert placed["success"], placed


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_go2_with_a_real_piper_moves_a_cube_between_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.delenv("ROBOTFORGE_MENAGERIE", raising=False)
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("real")
    placed, _ = _go2_moves_a_cube(playground, walk=False)  # standing as it was put down, it swept the cube away
    assert placed["success"]


CUBE = [0.04, 0.04, 0.04]
# A second job for every arm, away from the layout its own test was tuned on: (arm, props, object, target).
SECOND_JOBS = {
    "lift_picker": (  # from a coffee table on its left to a counter on its right
        "r",
        {
            "coffee": {"shape": "table", "size": [0.26, 0.26, 0.2], "pos": [0.1, 0.4], "fixed": True},
            "counter": {"shape": "table", "size": [0.3, 0.3, 0.7], "pos": [0.15, -0.42], "fixed": True},
            "cube": {"shape": "box", "size": CUBE, "pos": [0.1, 0.36, 0.22]},
        },
        "counter",
    ),
    "mobile_panda": (  # off the floor behind its left wheel, into a bin ahead on its right
        "r/arm",
        {
            "cube": {"shape": "box", "size": CUBE, "pos": [-0.2, 0.55]},
            "bin": {"shape": "bin", "size": [0.2, 0.2, 0.08], "pos": [0.65, -0.3], "fixed": True},
        },
        "bin",
    ),
    "mars_rover": ("r/hand", {"cube": {"shape": "box", "size": CUBE, "pos": [0.42, -0.12]}}, [0.36, 0.08]),
    "centaur": ("r/right_hand", {"cube": {"shape": "box", "size": CUBE, "pos": [0.4, -0.15]}}, [0.25, -0.4]),
    # From beside its front left leg to behind its right side: sweeping down on a slant from its raised pose, the claw
    # caught on that leg's femur and closed 3 cm above the cube.
    "scorpion": ("r/claw", {"cube": {"shape": "box", "size": CUBE, "pos": [0.2, 0.2]}}, [-0.4, -0.2]),
    "go2_arm": ("r/arm", {"cube": {"shape": "box", "size": CUBE, "pos": [0.0, 0.3]}}, [0.0, -0.3]),
    "twin_arm": (  # from the table in front onto a higher stand at its left
        "r/left_hand",
        {
            "table": {"shape": "table", "size": [0.3, 0.7, 0.35], "pos": [0.42, 0.0], "fixed": True},
            "stand": {"shape": "table", "size": [0.25, 0.25, 0.5], "pos": [0.0, 0.5], "fixed": True},
            "cube": {"shape": "box", "size": CUBE, "pos": [0.4, 0.2, 0.37]},
        },
        "stand",
    ),
    "tentacle": (  # from behind it on one side onto a step in front on the other
        "r",
        {
            "cube": {"shape": "box", "size": CUBE, "pos": [-0.25, 0.15]},
            "step": {"shape": "box", "size": [0.16, 0.16, 0.1], "pos": [0.25, -0.15], "fixed": True},
        },
        "step",
    ),
}


@pytest.mark.parametrize("design", list(SECOND_JOBS))
def test_every_arm_does_a_second_job_somewhere_else(pg, design):
    arm, props, target = SECOND_JOBS[design]
    pg.add_robot("r", design=design)
    for name, fields in props.items():
        pg.add_object(name, **fields)
    _run(pg, 0.5)
    placed = _pick(pg, arm, "cube", target)
    assert placed["success"], (design, placed)
    assert pg.behaviors.pose("r")["upright"]


# Floor picks that went wrong once the pick had been checked and accepted: (design, arm, cube, target).
CHECKED_WAYS = {
    # The carry was checked going 167 degrees one way round and went 193 the other, right over the body: thrown.
    "scorpion over its body": ("scorpion", "r/claw", [-0.129, -0.37], [0.006, 0.327]),
    # Checked going 239 degrees round, it went 121 the other way into the Panda's first joint limit and stalled.
    "mobile_panda round the front": ("mobile_panda", "r/arm", [-0.033, -0.36], [-0.175, 0.338]),
    # A 310 degree carry timed as the 0.52 m straight line between its ends: the hand flew at 1.1 m/s.
    "mobile_panda the long way": ("mobile_panda", "r/arm", [-0.497, 0.095], [-0.471, -0.42]),
    # Next to its base the tentacle's IK found the grasp in another pose: moving there joint by joint, the hand
    # swung in sideways and pushed the cube 4 cm.
    "tentacle beside its base": ("tentacle", "r", [-0.014, -0.167], [0.27, 0.153]),
    # The scorpion's tail reaches behind it by bending back over the top. Going round to a cube behind it, its base
    # joint ran into its limit and the claw stopped 30 cm short.
    "scorpion to a cube behind it": ("scorpion", "r/claw", [-0.5, 0.05], [0.05, 0.35]),
    # Behind it, 45 degrees round: no way round keeps the claw pointing down over the leg between, so the carry goes
    # joint by joint; planned from where the claw points, it was refused.
    "scorpion round behind it": ("scorpion", "r/claw", [-0.4, -0.1], [-0.2, -0.2]),
    # From its right to behind its left, across its back: refused the same way.
    "scorpion across its back": ("scorpion", "r/claw", [0.15, -0.35], [-0.25, 0.0]),
}


@pytest.mark.parametrize("case", list(CHECKED_WAYS))
def test_a_pick_goes_the_way_it_was_checked_and_lands(pg, case):
    design, arm, cube, target = CHECKED_WAYS[case]
    pg.add_robot("r", design=design)
    pg.add_object("cube", shape="box", size=CUBE, pos=cube)
    _run(pg, 0.5)
    placed = _pick(pg, arm, "cube", target)
    assert placed["success"], placed
    assert pg.behaviors.pose("r")["upright"]


def test_the_twin_arm_works_one_cube_with_each_hand_and_leaves_its_waist_alone(pg):
    pg.add_robot("twin", design="twin_arm")
    pg.add_object("table", shape="table", size=[0.3, 0.7, 0.35], pos=[0.42, 0.0], fixed=True)
    pg.add_object("red", shape="box", size=[0.04, 0.04, 0.04], pos=[0.35, -0.15, 0.37])
    pg.add_object("blue", shape="box", size=[0.04, 0.04, 0.04], pos=[0.35, 0.15, 0.37], color="blue")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.04], pos=[0.33, -0.27, 0.35], fixed=True)
    pg.add_object("cup", shape="bin", size=[0.14, 0.14, 0.04], pos=[0.33, 0.27, 0.35], fixed=True)
    _run(pg, 0.5)
    assert _pick(pg, "twin/right_hand", "red", "tray")["success"]
    assert _pick(pg, "twin/left_hand", "blue", "cup")["success"]
    waist = pg.session.model.joint("twin/waist").qposadr[0]
    assert abs(pg.session.data.qpos[waist]) < 0.01


def test_the_tentacle_curls_around_to_drop_a_cube_in_a_bin_behind_it(pg):
    pg.add_robot("tentacle", design="tentacle")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.3, 0.12])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[0.0, -0.3], fixed=True)
    _run(pg, 0.5)
    assert _pick(pg, "tentacle", "cube", "bin")["success"]
    assert mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, "tentacle/segment_8") >= 0
