import json
import math

import pytest
from mcp import Client

from robotforge.behaviors.manager import BehaviorError
from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "grass")
    return playground


def _run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def test_quadruped_walks_forward_and_stays_upright(pg):
    pg.add_robot("dog", template="quadruped")
    start = pg.behaviors.pose("dog")
    pg.behaviors.move("dog", 0.2, 0.0)
    _run(pg, 10.0)
    end = pg.behaviors.pose("dog")
    assert end["position"][0] - start["position"][0] > 1.0  # A1: more than 1 m in 10 s
    assert abs(end["position"][1] - start["position"][1]) < 0.25
    assert end["upright"]


def test_quadruped_turns_in_place(pg):
    pg.add_robot("dog", template="quadruped")
    pg.behaviors.move("dog", 0.0, 0.8)
    headings, last = [], pg.behaviors.pose("dog")["heading_deg"]
    total = 0.0
    for _ in range(40):
        _run(pg, 0.1)
        h = pg.behaviors.pose("dog")["heading_deg"]
        total += (h - last + 180) % 360 - 180
        last = h
        headings.append(h)
    assert total > 90  # 4 s at 0.8 rad/s after ramp-up is well over a quarter turn
    assert pg.behaviors.pose("dog")["upright"]


def test_stop_settles_a_walking_dog_into_a_stand(pg):
    pg.add_robot("dog", template="quadruped")
    pg.behaviors.move("dog", 0.3, 0.0)
    _run(pg, 2.0)
    pg.behaviors.stop("dog")
    _run(pg, 2.0)
    a = pg.behaviors.pose("dog")["position"]
    _run(pg, 1.0)
    b = pg.behaviors.pose("dog")["position"]
    assert math.dist(a, b) < 0.02
    assert pg.behaviors.pose("dog")["upright"]


def test_an_open_world_stands_its_legged_robots_on_their_gait_from_the_start(pg):
    """So a dog dragged before its first move steps along instead of rolling over (tests/test_push_recovery.py).
    Not a plain model file, nor a lesson: they run their robots their own way."""
    import mujoco

    from robotforge.behaviors.quadruped import QuadrupedGait

    pg.add_robot("dog", template="quadruped")
    assert isinstance(pg.session.controllers["dog"], QuadrupedGait)
    pg.add_object("crate", shape="box", pos=[-1.0, 1.0])  # an edit loads the world afresh
    assert isinstance(pg.session.controllers["dog"], QuadrupedGait)
    pg.session.reset()
    assert isinstance(pg.session.controllers["dog"], QuadrupedGait)
    model = pg.session.model
    plain = Playground(session=SimSession(model=model, data=mujoco.MjData(model)))
    assert "dog" not in plain.session.controllers


def test_a_leg_set_by_hand_moves_as_it_did_before_the_dog_had_a_gait(pg):
    """A viewer slider or set_controls on a dog still standing on the gait it opened with: the gait lets go (as
    'Joints by hand' does) instead of pulling the joint back every step. A dog that was driven keeps its gait."""
    from robotforge.viewer.server import ViewerHub

    pg.add_robot("dog", template="quadruped")
    m = pg.session.model
    knee = m.actuator("dog/front_left_knee").id
    ViewerHub(pg.session).handle({"type": "ctrl", "id": knee, "value": -1.2})  # the viewer's joint slider
    _run(pg, 1.0)
    assert "dog" not in pg.session.controllers
    assert pg.session.data.qpos[m.jnt_qposadr[m.actuator_trnid[knee, 0]]] == pytest.approx(-1.2, abs=0.1)
    pg.behaviors.move("dog", 0.0, 0.0)
    gait = pg.session.controllers["dog"]
    pg.set_controls({"dog/front_left_knee": -1.2})
    assert pg.session.controllers["dog"] is gait


def test_rover_drives_and_turns(pg):
    pg.add_robot("rover", template="rover")
    pg.behaviors.move("rover", 0.3, 0.0)
    _run(pg, 3.0)
    assert pg.behaviors.pose("rover")["position"][0] > 0.6
    pg.behaviors.move("rover", 0.0, 1.5)
    before = pg.behaviors.pose("rover")["heading_deg"]
    _run(pg, 1.0)
    assert abs(pg.behaviors.pose("rover")["heading_deg"] - before) > 30


def test_a_rover_with_big_wheels_rolls_on_its_caster(pg):
    """The template's caster ball took the floor's grip (MuJoCo uses the larger friction unless a geom has
    priority) and braked: on 7 cm wheels this 1.7 kg rover covered 2 % of its way, the eval's Mars rover none."""
    pg.add_robot("rover", template="rover", params={"wheel_radius": 0.07})
    start = pg.behaviors.pose("rover")["position"]
    pg.behaviors.move("rover", 0.3, 0.0)
    _run(pg, 3.0)
    end = pg.behaviors.pose("rover")["position"]
    assert end[0] - start[0] > 0.75 * 0.3 * 3.0


def test_arm_puts_the_red_cube_in_the_bin(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    arm, info = pg.behaviors.pick_and_place("arm", "cube", "tray")
    pg.run_fast(lambda: arm.done, 40.0)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", info["region"])["success"]  # A2


def test_unreachable_pick_is_refused_with_advice(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", pos=[0.9, 0.0])
    with pytest.raises(BehaviorError, match="cannot reach"):
        pg.behaviors.pick_and_place("arm", "cube", [0.2, 0.2])


def test_wrong_behavior_for_robot_kind_is_explained(pg):
    pg.add_robot("arm", template="arm")
    with pytest.raises(BehaviorError, match="only legged robots"):
        pg.behaviors.move("arm", 0.2, 0.0)
    with pytest.raises(BehaviorError, match="no robot named 'cat'"):
        pg.behaviors.move("cat", 0.2, 0.0)


@pytest.mark.anyio
async def test_move_and_pick_tools_report_results(pg):
    pg.add_robot("dog", template="quadruped", pos=[0, 0.8])
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        result = await client.call_tool("move", {"robot": "dog", "speed": 0.25, "seconds": 4})
        moved = json.loads(result.content[0].text)
        assert moved["distance_m"] > 0.6 and moved["shown_live"] is False
        placed = await client.call_tool("pick_and_place", {"robot": "arm", "object": "cube", "target": "tray"})
        assert json.loads(placed.content[0].text)["success"] is True
        listed = json.loads((await client.call_tool("robots", {})).content[0].text)
        assert {r["name"]: r["kind"] for r in listed["robots"]} == {"dog": "quadruped", "arm": "arm"}


@pytest.mark.anyio
async def test_a_timed_move_says_the_robot_stopped_and_leaves_its_wheels_at_rest(pg):
    """Agents in the eval called stop after nearly every timed move (the result showed the rover still driving at
    0.15 m/s), then saw the wheels still commanded in observe and zeroed them with set_controls and simulate."""
    pg.add_robot("rover", template="rover")
    pg.add_robot("dog", template="quadruped", pos=[0, 1.5])
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        driven = json.loads(
            (await client.call_tool("move", {"robot": "rover", "speed": 0.15, "seconds": 2})).content[0].text
        )
        assert driven["behavior"]["stopped"] is True and driven["behavior"]["speed"] == pytest.approx(0.15)
        assert "note" not in driven  # it covered the ground it was asked to
        state = json.loads((await client.call_tool("observe", {"include_image": False})).content[0].text)
        assert state["controls"]["rover/left_wheel"] == 0 and state["controls"]["rover/right_wheel"] == 0
        walked = json.loads(
            (await client.call_tool("move", {"robot": "dog", "speed": 0.3, "seconds": 2})).content[0].text
        )
        # The gait's speed ramps up from 0: at the start it read as a dog standing still.
        assert walked["behavior"]["stopped"] is True and walked["behavior"]["speed"] > 0.2


@pytest.mark.anyio
async def test_a_rover_too_heavy_for_its_motors_is_told_why_it_stood_still(pg):
    """A session built a 0.6 m solid aluminum rover on hobby gearmotors; it moved 2 mm in 6 s and the agent was
    told only distance_m 0.002, so it had to guess. (That one rolls now that its caster no longer brakes; this one,
    on the smallest motors, pushes with 1.3 N for 54 kg and cannot.)"""
    params = {"length": 0.6, "width": 0.42, "wheel_radius": 0.12, "motor": "tt_motor", "material": "aluminum"}
    pg.add_robot("mars_rover", template="rover", params=params)
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        result = await client.call_tool("move", {"robot": "mars_rover", "speed": 0.3, "seconds": 3})
        moved = json.loads(result.content[0].text)
    assert moved["distance_m"] < 0.05
    assert "wheels barely turned" in moved["note"] and "kg" in moved["note"] and "stronger motors" in moved["note"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_an_arm_whose_shapes_overlap_at_home_still_plans():
    """Regression: overlaps in the home pose (coarse imported shapes) were looked for before the arm knew
    which bodies move, crashing on any overlap, and the result was then discarded."""
    import mujoco

    from robotforge.behaviors.arm import ArmMotion

    xml = """
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="arm/base" pos="0 0 0.05">
          <geom type="cylinder" size="0.04 0.05"/>
          <body name="arm/link1" pos="0 0 0.06">
            <joint name="arm/j1" axis="0 0 1" range="-3 3"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
            <body name="arm/link2" pos="0 0 0.2">
              <joint name="arm/j2" axis="0 1 0" range="-2 2"/>
              <geom type="box" pos="0 0 -0.01" size="0.03 0.03 0.03"/>
              <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.015"/>
              <site name="arm/tcp" pos="0.2 0 0"/>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="arm/a1" joint="arm/j1" kp="20"/>
        <position name="arm/a2" joint="arm/j2" kp="20"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    arm = ArmMotion(model, "arm", model.qpos0.copy())
    assert arm._touching_at_home  # the elbow box overlaps the upper link at home...
    assert not arm._collides(arm.home)  # ...which is the model, not a collision


def test_robots_can_be_listed_while_the_policy_module_is_still_importing(pg, monkeypatch):
    """Regression: a lesson starting the G1 imports the policy module on a worker thread; mid-import the module
    is in sys.modules without PolicyWalker, and listing robots then raised and killed the viewer's stream."""
    import sys
    import types

    from robotforge.behaviors.manager import _is_policy_walker

    half = types.ModuleType("robotforge.bridge.onnx_policy")  # what another thread's import looks like early on
    monkeypatch.setitem(sys.modules, "robotforge.bridge.onnx_policy", half)
    pg.add_robot("dog", template="quadruped")
    assert not _is_policy_walker(object())
    assert [r["name"] for r in pg.behaviors.robots()] == ["dog"]


def test_a_robot_named_exactly_is_found_without_reading_world_yaml(pg, monkeypatch):
    """Scripts poll position and heading hundreds of times; reading and parsing world.yaml for each one made a
    loop in the warehouse scene about 9x slower. Only a name that needs matching ("Dog") reads the file."""
    pg.add_robot("dog", template="quadruped")

    def unreadable():
        raise AssertionError("world.yaml was read")

    monkeypatch.setattr(pg.behaviors, "_world", unreadable)
    assert pg.behaviors.pose("dog")["robot"] == "dog"
    assert pg.behaviors.stop("dog")["robot"] == "dog"
    with pytest.raises(AssertionError, match="world.yaml was read"):
        pg.behaviors.pose("Dog")
