"""Drawers, doors, buttons, switches, levers, valves, conveyors, ropes and cloths: built, watched and operated."""

import json
import math

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import operate
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import mechanisms as design
from robotforge.design.world import ObjectEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import mechanisms
from robotforge.sim.session import SimSession

# Where the template arm (on the floor at the origin) reaches each one with its gripper pointing down.
DRAWER = dict(shape="drawer", pos=[0.5, 0.0], yaw=-90)
CABINET = dict(shape="cabinet", size=[0.24, 0.2, 0.18], pos=[0.4, -0.05], yaw=-90)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "grass")
    return playground


def _operate(pg: Playground, target: str, action: str, amount: float | None = None, robot: str = "arm") -> dict:
    op, found, shape = operate.start(pg.behaviors, robot, target, action, amount)
    pg.run_fast(lambda: op.done, 90.0)
    pg.run_fast(lambda: False, 0.3)
    s = pg.session
    state = mechanisms.read(s.model, s.data, {found: shape}, s.controllers.get(mechanisms.KEY))
    return operate.result(op, s.model, s.data, state[found])


def _joint(pg: Playground, name: str) -> float:
    return float(pg.session.data.joint(name).qpos[0])


def test_every_mechanism_is_an_object_whose_state_observe_reports(pg):
    for name, shape in (
        ("chest", "drawer"),
        ("cupboard", "cabinet"),
        ("bell", "button"),
        ("light", "switch"),
        ("brake", "lever"),
        ("tap", "valve"),
        ("belt", "conveyor"),
        ("rope", "rope"),
        ("cloth", "cloth"),
    ):
        pg.add_object(name, shape=shape, pos=[2.0 * len(pg.store.load().objects), 3.0])
    state = pg.state()["mechanisms"]
    assert state["chest"] == {"type": "drawer", "open_m": 0.0, "of_m": pytest.approx(0.238, abs=0.001), "open": False}
    assert state["cupboard"]["angle_deg"] == 0.0 and not state["cupboard"]["open"]
    assert state["bell"] == {"type": "button", "pressed": False, "presses": 0}
    assert state["light"]["on"] is False  # a switch starts off
    assert state["brake"]["angle_deg"] == pytest.approx(-36.0, abs=0.5)
    assert state["tap"]["turns"] == 0.0 and state["tap"]["opening"] == 0.0
    assert state["belt"]["speed_m_s"] == design.DEFAULT_SPEED and state["belt"]["running"]
    assert state["rope"]["type"] == "rope" and state["cloth"]["type"] == "cloth"
    assert "mechanisms" not in json.dumps(pg.behaviors.status())  # the watcher is not a robot's behavior


def test_a_moving_part_does_not_touch_its_own_carcass():
    """MuJoCo leaves out contacts between a body and its parent, but not when the parent is welded to the world: a
    button's cap sat on its own collar and a 5 N push moved it 1 mm."""
    model = compile_world(World(objects=[ObjectEntry(name="bell", shape="button", pos=[0, 0])])).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.xfrc_applied[model.body("bell/cap").id] = [0, 0, -2.0, 0, 0, 0]
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert data.joint("bell/cap").qpos[0] < -design.PRESSED_AT * design.BUTTON_TRAVEL


def test_a_lever_stays_where_it_is_put(pg):
    """MuJoCo's joint friction gives way under a steady load: a lever crept 4 degrees in 10 s onto its stop."""
    pg.add_object("brake", shape="lever", pos=[0, 0])
    start = _joint(pg, "brake/lever")
    pg.run_fast(lambda: False, 10.0)
    assert abs(math.degrees(_joint(pg, "brake/lever") - start)) < 0.5


def test_things_on_a_conveyor_ride_at_its_speed_for_as_long_as_it_runs(pg):
    pg.add_object("belt", shape="conveyor", size=[2.0, 0.3, 0.6], pos=[0, 0], speed=0.2)
    pg.add_object("crate", shape="box", size=[0.08, 0.08, 0.08], pos=[-0.9, 0.0, 0.645], color="orange")
    pg.run_fast(lambda: False, 1.0)  # up to speed
    x0 = float(pg.session.data.body("crate").xpos[0])
    pg.run_fast(lambda: False, 6.0)  # 1.2 m: 80 wraps of the belt
    speed = (float(pg.session.data.body("crate").xpos[0]) - x0) / 6.0
    assert speed == pytest.approx(0.2, rel=0.03)
    assert 0.0 <= _joint(pg, "belt/belt") < design.BELT_WRAP  # put back every wrap
    pg.set_controls({"belt/speed": 0.0})  # stopped
    pg.run_fast(lambda: False, 0.5)
    x1 = float(pg.session.data.body("crate").xpos[0])
    pg.run_fast(lambda: False, 1.0)
    assert abs(float(pg.session.data.body("crate").xpos[0]) - x1) < 0.002
    assert pg.state()["mechanisms"]["belt"]["running"] is False
    pg.update_object("belt", speed=-0.1)  # and backwards, from the world file
    assert pg.state()["mechanisms"]["belt"]["speed_m_s"] == -0.1


def test_a_speed_is_only_for_conveyors(pg):
    with pytest.raises(ValueError, match="speed is a conveyor's belt speed"):
        pg.add_object("cube", shape="box", speed=0.2)


def test_presses_are_counted_as_they_happen(pg):
    pg.add_object("bell", shape="button", pos=[0, 0])
    cap = pg.session.model.body("bell/cap").id
    for _ in range(3):
        pg.session.data.xfrc_applied[cap] = [0, 0, -3.0, 0, 0, 0]
        pg.run_fast(lambda: False, 0.3)
        pg.session.data.xfrc_applied[cap] = 0.0
        pg.run_fast(lambda: False, 0.3)
    assert pg.state()["mechanisms"]["bell"] == {"type": "button", "pressed": False, "presses": 3}


@pytest.mark.parametrize("tool", ["gripper", "suction"])
def test_an_arm_opens_a_drawer_20_cm_and_closes_it(pg, tool):
    pg.add_robot("arm", template="arm", **({"end_effector": tool} if tool != "gripper" else {}))
    pg.add_object("chest", **DRAWER)
    opened = _operate(pg, "chest", "open", 0.2)
    assert opened["success"] and opened["moved"] == pytest.approx(0.2, abs=0.01)
    assert opened["state"]["open_m"] == pytest.approx(0.2, abs=0.01)
    closed = _operate(pg, "chest", "close")
    assert closed["success"] and closed["state"]["open_m"] < 0.02


def test_an_arm_opens_a_cabinet_door_60_degrees(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("cupboard", **CABINET)
    result = _operate(pg, "cupboard", "open", 60)
    assert result["success"] and result["moved"] == pytest.approx(60, abs=4)
    assert result["state"]["open"]


def test_an_arm_presses_a_button_flips_a_switch_and_opens_a_valve(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("bell", shape="button", pos=[0.25, 0.1])
    pg.add_object("light", shape="switch", pos=[0.25, -0.1])
    pg.add_object("tap", shape="valve", size=[0.16, 0.12], pos=[0.05, -0.27])
    pressed = _operate(pg, "bell", "press")
    assert pressed["pressed"] and pressed["state"]["presses"] == 1 and not pressed["state"]["pressed"]  # back up
    flipped = _operate(pg, "light", "press")
    assert flipped["success"] and flipped["state"]["on"]
    turned = _operate(pg, "tap", "open", 180)  # two holds of 90 degrees
    assert turned["success"] and turned["state"]["turns"] > 0.35


def test_operate_says_what_it_cannot_do(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("chest", shape="drawer", pos=[1.2, 0.0], yaw=-90)
    pg.add_object("cube", shape="box", pos=[0.25, 0.0])
    with pytest.raises(operate.OperateError, match="cannot take hold of chest's handle.*m from the arm's base"):
        operate.start(pg.behaviors, "arm", "chest", "open", None)
    with pytest.raises(operate.OperateError, match="chest is a drawer: it can open or close, not press"):
        operate.start(pg.behaviors, "arm", "chest", "press", None)
    with pytest.raises(operate.OperateError, match="cube is a box, nothing to operate.*pick_and_place moves it"):
        operate.start(pg.behaviors, "arm", "cube", "open", None)
    with pytest.raises(BehaviorError, match="chest is a drawer, built in place: operate can open or close it"):
        pg.behaviors.pick_and_place("arm", "chest", [0.2, 0.2])


def test_a_plate_goes_into_an_open_drawer_and_not_a_closed_one(pg):
    pg.add_robot("arm", template="arm", end_effector="suction")
    pg.add_object("chest", **DRAWER)
    pg.add_object("plate", shape="cylinder", size=[0.12, 0.012], pos=[0.05, 0.27], color="white")
    with pytest.raises(BehaviorError, match="chest is closed.*open it first with operate"):
        pg.behaviors.pick_and_place("arm", "plate", "chest")
    assert _operate(pg, "chest", "open", 0.2)["success"]
    arm, info = pg.behaviors.pick_and_place("arm", "plate", "chest")
    pg.run_fast(lambda: arm.done, 40.0)
    pg.run_fast(lambda: False, 0.5)
    assert pg.behaviors.placed("plate", info["region"])["success"]
    _operate(pg, "chest", "close")  # and it rides in with the drawer
    plate = pg.session.data.body("plate").xpos
    inside = pg.session.data.site("chest/inside").xpos
    assert abs(plate[0] - inside[0]) < 0.12 and plate[2] > inside[2] - 0.005


def test_a_rope_hangs_and_swings_and_a_cloth_drapes(pg):
    pg.add_object("rope", shape="rope", size=[0.5, 0.012], pos=[0.0, 0.0, 1.0])
    pg.add_object("table", shape="table", size=[0.4, 0.4, 0.3], pos=[1.0, 0.0], color="brown")
    pg.add_object("cloth", shape="cloth", size=[0.6, 0.6], pos=[1.0, 0.0, 0.45])
    ends = []
    for _ in range(40):
        pg.run_fast(lambda: False, 0.05)
        ends.append(pg.state()["mechanisms"]["rope"]["end"])
    xs = [e[0] for e in ends]
    assert max(xs) - min(xs) > 0.3  # it swings down from straight out, through below the hook
    assert min(e[2] for e in ends) < 1.0 - 0.4  # and hangs nearly its length below it
    pg.run_fast(lambda: False, 4.0)
    cloth = pg.state()["mechanisms"]["cloth"]
    low, high = cloth["height_m"]
    assert high == pytest.approx(0.3, abs=0.03)  # lying on the table top
    assert low < 0.25  # with its edges hanging down over the sides


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_operate_is_an_mcp_tool(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("chest", **DRAWER)
    server = build_server(pg, 0)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert "drawer: open / close" in tools["operate"].description
        reply = await client.call_tool("operate", {"robot": "arm", "target": "Chest", "action": "open", "amount": 0.15})
        result = json.loads(reply.content[0].text)
        assert result["target"] == "chest" and result["success"] and result["moved"] == pytest.approx(0.15, abs=0.01)
        refused = await client.call_tool("operate", {"robot": "arm", "target": "chest", "action": "turn"})
        assert refused.is_error and "it can open or close, not turn" in refused.content[0].text
        state = json.loads((await client.call_tool("observe", {"include_image": False})).content[0].text)
        assert state["mechanisms"]["chest"]["open_m"] == pytest.approx(0.15, abs=0.01)


def test_a_drawers_inside_moves_with_it():
    model = compile_world(World(objects=[ObjectEntry(name="chest", **DRAWER)])).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    before = data.site("chest/inside").xpos.copy()
    data.joint("chest/drawer").qpos[0] = 0.1
    mujoco.mj_forward(model, data)
    assert np.allclose(data.site("chest/inside").xpos - before, [-0.1, 0.0, 0.0], atol=1e-6)  # out toward -x (yaw -90)


def test_the_kitchen_arm_opens_its_drawer_and_puts_the_plate_in_it_by_suction(tmp_path, monkeypatch):
    """What the agent session was asked: 打开厨房的抽屉，再用吸盘把桌上的薄板放进抽屉. The plate tipped off the cup
    and hung 7 cm below it until the pad resisted peeling (end_effectors.PAD_FRICTION), and the template arm sagged
    5.6 cm carrying it until its servo targets held its weight (arm.servo_targets)."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.load_scene("kitchen")
    assert pg.state()["mechanisms"]["drawer"]["open_m"] == 0.0 and "cabinet" in pg.state()["mechanisms"]
    assert _operate(pg, "drawer", "open", 0.2)["success"]
    pg.update_robot("arm", end_effector="suction")
    assert pg.state()["mechanisms"]["drawer"]["open_m"] == pytest.approx(0.2, abs=0.01)  # the tool change kept it
    arm, info = pg.behaviors.pick_and_place("arm", "plate", "drawer")
    pg.run_fast(lambda: arm.done, 40.0)
    pg.run_fast(lambda: False, 0.5)
    placed = pg.behaviors.placed("plate", info["region"])
    assert placed["success"]
    floor = pg.session.data.site("drawer/inside").xpos[2]
    assert placed["object_position"][2] == pytest.approx(floor + 0.0075, abs=0.003)  # flat on the drawer's floor


def test_the_factory_belt_brings_a_box_to_the_stop_and_the_arm_sorts_it(tmp_path, monkeypatch):
    """The camera's part is in scripts/factory_bench.py (it needs OpenGL); here the belt, the stop and the reach."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    from robotforge.design import scenes

    pg = Playground.with_demo()
    result = pg.load_scene("factory", options={"boxes": 4})
    assert result["robots"] == ["arm"] and "factory" in {s["name"] for s in scenes.catalog()}
    world = pg.store.load()
    assert world.robots[0].end_effector == "suction"
    data = pg.session.data
    pg.run_fast(lambda: data.body("box_1").xpos[0] > scenes.PICK_X - 0.003, 10.0)
    pg.run_fast(lambda: False, 0.4)  # the belt slides on under it
    pg.set_controls({"conveyor/speed": 0.0})
    pg.run_fast(lambda: False, 1.0)
    assert data.body("box_1").xpos[0] == pytest.approx(scenes.PICK_X, abs=0.003)  # waiting against the stop
    assert data.body("box_2").xpos[0] < scenes.PICK_X - 0.05  # the next one still on its way, not pushed up to it
    color = next(o.color for o in world.objects if o.name == "box_1")
    arm, info = pg.behaviors.pick_and_place("arm", "box_1", f"{color}_bin")
    pg.run_fast(lambda: arm.done, 40.0)
    pg.run_fast(lambda: False, 0.5)
    assert pg.behaviors.placed("box_1", info["region"])["success"]
