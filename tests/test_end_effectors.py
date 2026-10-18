"""Suction cups and electromagnets: built in place of fingers, fitted to any arm, and worked like a gripper."""

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest
from built_robots import CRANE

from robotforge.behaviors.arm import ArmMotion
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import end_effectors, templates
from robotforge.design.world import RobotEntry, World, compile_world
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"
TILE = [0.06, 0.06, 0.008]  # 8 mm: flatter than any gripper's fingertips reach under


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "grass")
    return playground


def _pick(pg: Playground, robot: str, obj: str, target: str) -> dict:
    arm, info = pg.behaviors.pick_and_place(robot, obj, target)
    pg.run_fast(lambda: arm.done, 40.0)
    pg.run_fast(lambda: False, 0.5)
    return pg.behaviors.placed(obj, info["region"])


def test_a_suction_cup_picks_a_tile_that_fingers_cannot_get_round(pg):
    pg.add_robot("arm", template="arm", end_effector="suction")
    pg.add_object("tile", shape="box", size=TILE, pos=[0.25, -0.12], color="orange")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    assert _pick(pg, "arm", "tile", "tray")["success"]

    pg.update_robot("arm", end_effector="gripper")  # the tool change back to fingers
    pg.update_object("tile", pos=[0.25, -0.12])
    with pytest.raises(BehaviorError, match="only 8 mm thick.*end_effector='suction'"):
        pg.behaviors.pick_and_place("arm", "tile", "tray")


def test_an_electromagnet_holds_steel_and_leaves_plastic(pg):
    pg.add_robot("arm", template="arm", end_effector="magnet")
    pg.add_object("steel_plate", shape="box", size=[0.05, 0.05, 0.004], pos=[0.25, -0.12], material="steel")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    assert pg.session.model.body("steel_plate").mass[0] == pytest.approx(7850 * 0.05 * 0.05 * 0.004)
    assert _pick(pg, "arm", "steel_plate", "tray")["success"]

    pg.remove("steel_plate")
    pg.add_object("plastic_plate", shape="box", size=[0.05, 0.05, 0.004], pos=[0.25, -0.12])
    placed = _pick(pg, "arm", "plastic_plate", "tray")
    assert not placed["success"]
    assert np.allclose(placed["object_position"][:2], [0.25, -0.12], atol=0.01)  # the magnet let it lie


def test_gripper_close_turns_a_cup_on_and_open_turns_it_off(pg):
    pg.add_robot("arm", template="arm", end_effector="suction")
    arm = pg.behaviors.set_gripper("arm", "close")
    pg.run_fast(lambda: arm.done, 2.0)
    assert pg.session.data.ctrl[arm.gripper] == 1.0
    arm = pg.behaviors.set_gripper("arm", "open")
    pg.run_fast(lambda: arm.done, 2.0)
    assert pg.session.data.ctrl[arm.gripper] == 0.0


def test_a_library_arm_gets_the_tool_in_front_of_its_fingers(pg):
    pg.add_robot("panda", library="panda")
    m = pg.session.model
    tcp_before = pg.session.data.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")].copy()
    result = pg.update_robot("panda", end_effector="suction")
    assert any("suction cup: holds anything" in n and "fingers stay open" in n for n in result["notes"])
    m, d = pg.session.model, pg.session.data
    names = [m.actuator(a).name for a in range(m.nu)]
    assert "panda/fingers" in names and "panda/gripper" in names
    assert end_effectors.tool_of(m, names.index("panda/gripper")) == "suction"
    tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "panda/tcp")]
    # Further out along the hand by the stand-off past the fingertips and the tool's own length.
    assert np.linalg.norm(tcp - tcp_before) == pytest.approx(
        end_effectors.AHEAD_OF_FINGERS + end_effectors.TOOLS["suction"].length, abs=1e-3
    )
    assert ArmMotion(m, "panda", pg.behaviors.home_qpos()).tool == "suction"


def test_an_arm_built_from_parts_takes_a_suction_cup_template_on_its_end_or_a_magnet_by_name():
    build = copy.deepcopy(CRANE)
    build["parts"][-1] = {"name": "hand", "parent": "wrist", "at": "end", "robot": {"template": "suction_cup"}}
    model = compile_world(World(robots=[RobotEntry(name="bot", build=build)])).model
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bot/hand/gripper")
    assert end_effectors.tool_of(model, gripper) == "suction"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "bot/hand/tcp") >= 0

    # The crane as it is, fingers and all, with end_effector: the magnet goes in front of its fingers.
    model = compile_world(World(robots=[RobotEntry(name="bot", build=CRANE, end_effector="magnet")])).model
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bot/hand/gripper")
    assert end_effectors.tool_of(model, gripper) == "magnet"
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "bot/hand/fingers") >= 0


def test_a_world_keeps_the_tool_and_old_worlds_still_load():
    entry = RobotEntry(name="arm", template="arm", end_effector="Suction Cup")
    assert entry.end_effector == "suction"
    text = World(robots=[entry]).to_yaml()
    assert "end_effector: suction" in text
    assert World.from_yaml(text).robots[0].end_effector == "suction"
    old = World.from_yaml("name: old\nrobots:\n- name: arm\n  template: arm\n")
    assert old.robots[0].end_effector is None
    with pytest.raises(ValueError, match="unknown end_effector 'hook'"):
        RobotEntry(name="arm", template="arm", end_effector="hook")


def test_end_effector_on_a_robot_without_an_arm_is_refused(pg):
    with pytest.raises(ValueError, match="dog.: it has no tool point to put a suction cup on"):
        pg.add_robot("dog", template="quadruped", end_effector="suction")


def test_the_arm_templates_tool_replaces_its_fingers():
    built = templates.build("arm", {"end_effector": "magnet"})
    model = built.spec.compile()
    names = [model.actuator(a).name for a in range(model.nu)]
    assert "gripper" in names and "fingers" not in names
    assert not any(model.joint(j).name.startswith("finger") for j in range(model.njnt))
    assert built.home_ctrl["gripper"] == 0.0
    core = model.geom(next(g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == "magnet_core"))
    assert core.contype == end_effectors.MAGNET_BIT and core.conaffinity == 0


def test_fingers_still_hold_by_the_middle(pg):
    """The parallel gripper's plan is what it always was: grasp at the object's middle, drop it half its height
    plus a centimetre above the place."""
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    arm, _ = pg.behaviors.pick_and_place("arm", "cube", "tray")
    assert arm.tool is None
    steps = {wp.label: wp for wp in arm.queue}
    cube = pg.session.data.body("cube").xpos
    assert np.allclose(steps["descend"].position, cube)
    assert steps["lower"].position[2] == pytest.approx(pg.session.data.body("tray").xpos[2] + 0.005 + 0.02 + 0.01)
    assert steps["carry"].load == pytest.approx(0.02)
