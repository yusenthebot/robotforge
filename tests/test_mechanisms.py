"""Drawers, doors, buttons, switches, levers, valves, conveyors, ropes and cloths: built, watched and operated."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import operate
from robotforge.behaviors.arm import ArmMotion, servo_stiffness, servo_targets
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import mechanisms as design
from robotforge.design.world import ObjectEntry, RobotEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import mechanisms
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
# Where the template arm (on the floor at the origin) reaches each one with its gripper pointing down.
DRAWER = dict(shape="drawer", pos=[0.5, 0.0], yaw=-90)
CABINET = dict(shape="cabinet", size=[0.24, 0.2, 0.18], pos=[0.4, -0.05], yaw=-90)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
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
    from robotforge.viewer.server import ViewerHub

    assert all(b["robot"] != mechanisms.KEY for b in ViewerHub(pg.session).state_message()["behaviors"])
    # A rope's and a cloth's hundreds of unnamed joints stay out of observe's joints (they filled them with
    # "joint8" to "joint40"); the mechanisms section sums them up.
    assert not any(name.startswith("joint") for name in pg.state()["joints"])


def test_only_a_worlds_conveyors_and_buttons_are_watched(pg):
    """Found by the world's objects, not joint names: a robot built from parts with a slide called "belt" had it
    snapped back every 1.5 cm, as if it were a conveyor's."""
    pg.add_robot(
        "slider",
        build={
            "base": {"shape": "box", "size": [0.3, 0.2, 0.06]},
            "parts": [
                {
                    "name": "belt",
                    "at": "top",
                    "joint": {"type": "slide", "axis": "x", "range": [-0.05, 0.05]},
                    "shape": "box",
                    "size": [0.1, 0.1, 0.03],
                }
            ],
        },
    )
    joints = [mujoco.mj_id2name(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(pg.session.model.njnt)]
    assert "slider/belt" in joints
    assert mechanisms.KEY not in pg.session.controllers  # nothing in this world to watch
    pg.add_object("line", shape="conveyor", pos=[2.0, 0.0])
    watcher = pg.session.controllers[mechanisms.KEY]
    adr = int(pg.session.model.jnt_qposadr[pg.session.model.joint("line/belt").id])
    assert watcher.belts == [adr]


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


def test_a_box_rides_at_the_belt_speed_behind_a_queue_at_the_stop(pg):
    """Ten boxes held at a stop rub on the belt with their weight: with a soft drive (200 N per m/s) the belt, and a
    box still riding on it, went 37% slow."""
    pg.add_object("belt", shape="conveyor", size=[2.0, 0.12, 0.6], pos=[0, 0], speed=0.1)
    pg.add_object("stop", shape="box", size=[0.03, 0.14, 0.06], pos=[0.97, 0.0, 0.63], fixed=True, color="yellow")
    for i in range(10):
        pg.add_object(f"held{i}", shape="box", size=[0.05] * 3, pos=[0.927 - 0.0505 * i, 0.0, 0.626], color="red")
    pg.add_object("rider", shape="box", size=[0.05] * 3, pos=[-0.9, 0.0, 0.626], color="blue")
    pg.run_fast(lambda: False, 1.0)
    x0 = float(pg.session.data.body("rider").xpos[0])
    pg.run_fast(lambda: False, 3.0)
    speed = (float(pg.session.data.body("rider").xpos[0]) - x0) / 3.0
    assert speed == pytest.approx(0.1, rel=0.03)
    assert float(pg.session.data.body("held0").xpos[0]) == pytest.approx(0.927, abs=0.005)  # still at the stop
    # and stays there when the belt stops: stopped in one step, the stiff drive's kick threw the box at the stop of
    # the factory 25 mm back up the belt
    before = [float(pg.session.data.body(f"held{i}").xpos[0]) for i in range(10)]
    pg.set_controls({"belt/speed": 0.0})
    pg.run_fast(lambda: False, 1.0)
    after = [float(pg.session.data.body(f"held{i}").xpos[0]) for i in range(10)]
    assert np.max(np.abs(np.subtract(after, before))) < 0.005  # the queue only eases back a little


def test_too_small_a_mechanism_is_refused_with_the_smallest_it_can_be():
    """Built smaller, a part came out with no room and MuJoCo said only "size 2 must be positive in geom"."""
    with pytest.raises(ValueError, match="a conveyor needs height >= 0.06 m"):
        ObjectEntry(name="line", shape="conveyor", size=[1.0, 0.3, 0.04])
    with pytest.raises(ValueError, match="a valve needs wheel diameter >= 0.08 m, height >= 0.04 m"):
        ObjectEntry(name="tap", shape="valve", size=[0.02, 0.01])
    smallest = [
        ObjectEntry(name=f"m{i}", shape=shape, size=list(size), pos=[2.0 * i, 0.0])
        for i, (shape, size) in enumerate(design.MIN_SIZES.items())
    ]
    model = compile_world(World(objects=smallest)).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data, 250)
    assert np.isfinite(data.qpos).all()


def test_the_new_fields_are_saved_only_when_set():
    """A world that uses no end effector, belt speed or material saves exactly as it did before they existed (old
    world.yaml files and the tests that read them stay as they were); one that sets them keeps them."""
    plain = World(robots=[RobotEntry(name="arm", template="arm")], objects=[ObjectEntry(name="cube", shape="box")])
    text = plain.to_yaml()
    assert "end_effector" not in text and "speed" not in text and "material" not in text
    used = World(
        robots=[RobotEntry(name="arm", template="arm", end_effector="suction")],
        objects=[
            ObjectEntry(name="belt", shape="conveyor", speed=0.2),
            ObjectEntry(name="nut", shape="box", material="steel"),
        ],
    )
    again = World.from_yaml(used.to_yaml())
    assert again.robots[0].end_effector == "suction"
    assert again.objects[0].speed == 0.2 and again.objects[1].material == "steel"


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


@pytest.mark.parametrize("robot", ["template", "piper"])
def test_servo_targets_hold_an_arm_where_it_is_put(pg, robot):
    """The weight an arm's servos hold up, less what MuJoCo's gravity compensation already does: counted twice on a
    PiPER (gravcomp on 9 of its bodies), the hand stood 3.7 cm too high and pressed no button."""
    pg.add_robot("r", **({"template": "arm"} if robot == "template" else {"library": robot}))
    m, d = pg.session.model, pg.session.data
    arm = ArmMotion(m, "r", pg.behaviors.home_qpos())
    q, _, _ = arm._solve(np.array([arm.base_xy[0] + 0.3, arm.base_xy[1], 0.15]))
    want = arm.ik.point(arm.chain, q).copy()
    kp = servo_stiffness(m, arm.actuators)
    d.qpos[arm.chain.qpos_adr] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(int(3.0 / m.opt.timestep)):
        d.ctrl[arm.actuators] = servo_targets(d, arm.chain.dof_adr, kp, q)
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.site_xpos[arm.site] - want) < 0.005


def test_a_piper_presses_a_button(pg):
    pg.add_robot("piper", library="piper")
    pg.add_object("bell", shape="button", pos=[0.3, 0.0])
    pressed = _operate(pg, "bell", "press", robot="piper")
    assert pressed["success"] and pressed["state"]["presses"] == 1


def test_success_is_judged_against_how_far_it_can_go(pg):
    """A drawer opened fully when asked for more than it has, and anything already where it was asked to go, are
    successes: both read success false, and agents told to read success would have tried again."""
    pg.add_robot("arm", template="arm")
    pg.add_object("chest", **DRAWER)
    pg.add_object("tap", shape="valve", size=[0.16, 0.12], pos=[0.05, -0.27])
    full = _operate(pg, "chest", "open", 0.5)
    assert full["success"] and full["moved"] == pytest.approx(0.238, abs=0.008)
    assert "chest goes only 0.24 m that way" in full["notes"][0]
    again = _operate(pg, "chest", "open")  # within millimetres of its end already
    assert again["success"] and again["moved"] == 0.0 and "already there" in again["notes"][0]
    closed = _operate(pg, "tap", "close", 90)
    assert closed["success"] and closed["notes"] == ["tap is already at its end that way: 0 degrees"]


def test_a_valve_turns_as_far_as_it_was_asked(pg):
    """The wheel slips in the template's fingers, about 20 degrees a quarter turn: it is taken hold of again for what
    is left, planned from where it really is (a hold planned for where it would have been put a finger on a spoke)."""
    pg.add_robot("arm", template="arm")
    pg.add_object("tap", shape="valve", size=[0.16, 0.12], pos=[0.05, -0.27])
    turned = _operate(pg, "tap", "open")
    assert turned["success"] and turned["moved"] == pytest.approx(90, abs=5)


def test_a_panda_opens_a_drawer_and_a_door(pg):
    """Hold poses are checked against the mechanism: a Panda's best hold put a finger 2.8 cm into the chest's top,
    and for real the finger came down on it, the hand closed on nothing and the drawer moved 1 mm."""
    pg.add_robot("panda", library="panda")
    pg.add_object("chest", shape="drawer", pos=[0.7, 0.0], yaw=-90)
    opened = _operate(pg, "chest", "open", 0.2, robot="panda")
    assert opened["success"] and opened["moved"] == pytest.approx(0.2, abs=0.01)
    pg.update_object("chest", pos=[0.6, 0.0])  # where it had slipped: now as far as its plan says, not slipping
    pg.control("reset")
    part = _operate(pg, "chest", "open", 0.2, robot="panda")
    assert "follow the handle only" in part["notes"][0] and len(part["notes"]) == 1
    assert part["moved"] == pytest.approx(float(part["notes"][0].split("only ")[1].split(" m")[0]), abs=0.01)
    pg.remove("chest")
    pg.add_object("cupboard", shape="cabinet", size=[0.3, 0.24, 0.3], pos=[0.65, 0.0], yaw=-90)
    door = _operate(pg, "cupboard", "open", 60, robot="panda")
    assert door["success"] and door["moved"] == pytest.approx(60, abs=4)
    shut = _operate(pg, "cupboard", "close", robot="panda")
    assert shut["success"] and abs(shut["state"]["angle_deg"]) < 3


def test_a_panda_comes_down_onto_a_drawer_bar_from_above(pg):
    """Of its holds on the bar the one that comes straight down from highest: the one at a wrist limit came down 3 cm,
    swung in low over the drawer from afar and caught a finger on its top."""
    pg.add_robot("panda", library="panda")
    pg.add_object("chest", shape="drawer", pos=[0.55, 0.0], yaw=-90)
    opened = _operate(pg, "chest", "open", 0.2, robot="panda")
    assert opened["success"] and opened["moved"] > 0.16  # 0.2 with the real Panda's meshes
    assert not any("came down on" in note for note in opened.get("notes", []))


def test_a_piper_takes_hold_of_a_drawer_without_touching_its_top(pg):
    """A PiPER points down only low, so it cannot come down onto the bar from above: it comes in from the front. Its
    servos lag a moving plan unless led by their damping (arm.servo_targets), and a swing planned 3 mm over the top
    put a finger on it."""
    pg.add_robot("piper", library="piper")
    pg.add_object("chest", shape="drawer", pos=[0.45, 0.0], yaw=-90)
    op, found, shape = operate.start(pg.behaviors, "piper", "chest", "open", 0.2)
    touched = set()

    def watch() -> bool:
        s = pg.session
        parts = operate._parts(s.model, "chest")
        for con in s.data.contact[: s.data.ncon]:
            mine, theirs = operate._arm_and(op.arm, parts, int(con.geom1), int(con.geom2))
            if mine is not None and con.dist < 0 and op.current_label in ("approach", "reach"):
                touched.add(operate._geom_name(s.model, theirs))
        return op.done

    pg.run_fast(watch, 60.0)
    pg.run_fast(lambda: False, 0.3)
    s = pg.session
    result = operate.result(op, s.model, s.data, mechanisms.read(s.model, s.data, {found: shape})[found])
    assert not op.aborted and touched <= {"chest/handle"}
    assert result["moved"] > 0.1  # as far as it reaches, the note says (0.14 m)
    assert "follow the handle only 0.1" in result["notes"][0]


def test_a_hand_that_does_not_take_hold_stops_and_says_so(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("chest", **DRAWER)
    op, found, shape = operate.start(pg.behaviors, "arm", "chest", "open", 0.2)
    pg.run_fast(lambda: op.current_label == "reach", 20.0)
    pg.session.data.joint("chest/drawer").qpos[0] = 0.08  # pulled out from under the hand coming down for it
    pg.run_fast(lambda: op.done, 30.0)
    s = pg.session
    result = operate.result(op, s.model, s.data, mechanisms.read(s.model, s.data, {found: shape})[found])
    assert not result["success"] and op.aborted
    assert result["notes"] == [
        "arm did not take hold of chest's handle: the handle was not in its grip; move the arm or the chest a little "
        "and try again"
    ]  # not put down to slipping


def test_a_magnet_works_the_steel_bars_and_says_what_it_cannot_hold(pg):
    pg.add_robot("arm", template="arm", end_effector="magnet")
    pg.add_object("chest", **DRAWER)
    pg.add_object("light", shape="switch", pos=[0.0, -0.3])  # out of the drawer's way
    opened = _operate(pg, "chest", "open", 0.2)
    assert opened["success"] and opened["moved"] == pytest.approx(0.2, abs=0.01)
    with pytest.raises(operate.OperateError, match="electromagnet holds steel only, and light's knob is not steel"):
        operate.start(pg.behaviors, "arm", "light", "press", None)


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
    pg.add_object("rope", shape="rope", size=[0.3, 0.01], pos=[0.25, 0.2], fixed=False)
    with pytest.raises(BehaviorError, match="rope is a rope, soft: pick_and_place carries solid things"):
        pg.behaviors.pick_and_place("arm", "rope", [0.2, 0.2])


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
    ends = []
    for _ in range(40):
        pg.run_fast(lambda: False, 0.05)
        ends.append(pg.state()["mechanisms"]["rope"]["end"])
    xs = [e[0] for e in ends]
    assert max(xs) - min(xs) > 0.3  # it swings down from straight out, through below the hook
    assert min(e[2] for e in ends) < 1.0 - 0.4  # and hangs nearly its length below it
    # and keeps swinging, ever less, without blowing up: lightly damped, its end cracked like a whip and the world
    # went to NaN after 12.6 s
    later = []
    for _ in range(40):
        pg.run_fast(lambda: False, 0.5)
        later.append(pg.state()["mechanisms"]["rope"]["end"][0])
    assert np.isfinite(pg.session.data.qpos).all()
    assert 0.02 < max(later[-8:]) - min(later[-8:]) < max(xs) - min(xs)
    pg.add_object("table", shape="table", size=[0.4, 0.4, 0.3], pos=[1.0, 0.0], color="brown")
    pg.add_object("cloth", shape="cloth", size=[0.6, 0.6], pos=[1.0, 0.0, 0.45])
    pg.run_fast(lambda: False, 4.0)
    cloth = pg.state()["mechanisms"]["cloth"]
    low, high = cloth["height_m"]
    assert high == pytest.approx(0.3, abs=0.03)
    assert low < 0.25  # with its edges hanging down over the sides
    # and its middle on the top, not sunk through it: MuJoCo keeps 50 contacts a pair, and a finer cloth went
    # through the top held by its edges only
    points = pg.session.data.flexvert_xpos
    over = (np.abs(points[:, 0] - 1.0) < 0.15) & (np.abs(points[:, 1]) < 0.15)
    assert np.median(points[over, 2]) == pytest.approx(0.3, abs=0.01)
    # and what hangs over the edges hangs outside them: only its points touch things, and 110 of its 256 points
    # slipped round the top's lower edge and hung tucked in under it (design.cloth_skirt)
    under = (np.abs(points[:, 0] - 1.0) < 0.2) & (np.abs(points[:, 1]) < 0.2) & (points[:, 2] < 0.3 - 0.025)
    assert not under.any()


def test_a_cloth_folds_over_a_tables_edge_without_cutting_into_it(pg):
    """Drawn between its points, 4.6 cm apart, a cloth's sides cut up to 1 cm into a table top where they bent over
    its sharp edge, and the edge showed through the cloth in a long streak; the hidden hem rounds the fold."""
    pg.add_object("table", shape="table", size=[0.5, 0.4, 0.35], pos=[0.0, 0.0])
    pg.add_object("cloth", shape="cloth", size=[0.6, 0.5], pos=[0.0, 0.0, 0.5])
    m, d = pg.session.model, pg.session.data
    pg.run_fast(lambda: False, 5.0)
    points = d.flexvert_xpos[m.flex_vertadr[0] : m.flex_vertadr[0] + m.flex_vertnum[0]]
    edges = m.flex_edge[m.flex_edgeadr[0] : m.flex_edgeadr[0] + m.flex_edgenum[0]]
    s = np.linspace(0.0, 1.0, 21)[:, None]
    deepest = 0.0
    for a, b in edges:
        q = points[a] * (1 - s) + points[b] * s
        inside = np.minimum.reduce([0.25 - np.abs(q[:, 0]), 0.2 - np.abs(q[:, 1]), 0.35 - q[:, 2], q[:, 2] - 0.325])
        deepest = max(deepest, float(inside.max()))
    assert deepest < 0.002


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


@pytest.mark.anyio
async def test_the_shapes_go_by_the_words_people_use(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        for name, shape in (("line", "Conveyor Belt"), ("towel", "towel"), ("chest", "chest"), ("tap", "handwheel")):
            reply = await client.call_tool("add_object", {"name": name, "shape": shape, "pos": [len(name), 2.0]})
            assert not reply.is_error, reply.content[0].text
    shapes = {o.name: o.shape for o in pg.store.load().objects}
    assert shapes == {"line": "conveyor", "towel": "cloth", "chest": "drawer", "tap": "valve"}


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
