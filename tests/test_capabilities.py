"""One capability contract for the assembled robot and every interface."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from importer_fixtures import rover_urdf, write
from mcp import Client
from test_import_arm import ARM_URDF

from robotforge import capabilities, modes
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.robot import Robot
from robotforge.script import ScriptError
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("capabilities")
    return pg


def robot_info(pg, name):
    return next(r for r in pg.behaviors.robots() if r["name"] == name)


@pytest.mark.parametrize("source", ["template", "urdf", "library"])
def test_arm_capabilities_come_from_the_final_model(pg, tmp_path, source):
    args = {"template": "arm"} if source == "template" else {"library": "panda"}
    if source == "urdf":
        args = {"file": str(write(tmp_path / "arm.urdf", ARM_URDF))}
    pg.add_robot("a", **args)
    info = robot_info(pg, "a")
    caps = info["capabilities"]
    assert caps["schema_version"] == 1
    assert caps["source"]["kind"] == ("file" if source == "urdf" else source)
    assert caps["move_arm"]["endpoints"] == ["a"]
    assert caps["arms"][0]["tcp"] == "a/tcp"
    assert not caps["move"]["supported"]
    assert Robot(None, "a", info).has_arm
    assert modes.available(pg, info, False) == ["manual", "reach_ik", "reach_cartesian", "reach_mpc"]


def test_a_bare_arm_can_reach_without_claiming_a_gripper(pg):
    pg.add_robot("a", library="ur5e")
    info = robot_info(pg, "a")
    # A bare flange can mark a tool point without having an actuator that grips.
    assert info["kind"] == "arm"
    assert info["capabilities"]["move_arm"]["supported"]
    assert not info["capabilities"]["gripper"]["supported"]
    assert Robot(None, "a", info).has_arm
    with pytest.raises(ScriptError, match="cannot hold things"):
        _ = Robot(None, "a", info).gripper


def test_arm_classification_without_a_tcp_does_not_invent_an_endpoint():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    info = {"kind": "arm"}
    info["capabilities"] = capabilities.describe(capabilities.inspect_robot(model, "a", "arm"), info, {})
    assert not Robot(None, "a", info).has_arm
    assert not info["capabilities"]["tasks"]["reach"]["supported"]


def test_a_standalone_hand_exposes_raw_controls_without_an_arm(pg):
    pg.add_robot("hand", template="gripper")
    info = robot_info(pg, "hand")
    assert info["kind"] == "gripper"
    assert info["capabilities"]["manual"]["supported"]
    assert not info["capabilities"]["move_arm"]["supported"]
    assert not info["capabilities"]["gripper"]["supported"]
    assert not Robot(None, "hand", info).has_arm


def test_a_tool_point_without_power_is_not_an_arm():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="a/base"><site name="a/tcp"/></body></worldbody></mujoco>'
    )
    structure = capabilities.inspect_robot(model, "a", "arm")
    assert not structure.arms[0].move.supported
    assert "no powered joints" in structure.arms[0].move.reason


def test_composition_keeps_base_and_each_arm_endpoint(pg):
    pg.add_robot(
        "r",
        template="rover",
        mounts=[
            {"name": "left", "template": "arm", "shift": [0, 0.2, 0]},
            {"name": "right", "template": "arm", "shift": [0, -0.2, 0]},
        ],
    )
    info = robot_info(pg, "r")
    caps = info["capabilities"]
    assert caps["move"]["supported"]
    assert caps["move_arm"]["endpoints"] == ["r/left", "r/right"]
    assert caps["source"]["kind"] == "template" and len(caps["source"]["components"]) == 2
    sim = Mock()
    robot = Robot(sim, "r", info)
    assert robot.can_move and robot.has_arm and len(robot.arms) == 2
    robot.arms[1].gripper.close()
    sim.gripper.assert_called_once_with("r/right", "close")
    with pytest.raises(ScriptError, match="several grippers"):
        _ = robot.gripper


def test_imported_humanoid_cannot_claim_walking_from_its_kind(pg):
    source = Path(__file__).parent / "fixtures" / "menagerie" / "robotis_op3" / "op3.xml"
    pg.add_robot("h", file=str(source))
    info = robot_info(pg, "h")
    assert info["kind"] == "humanoid"
    assert not info["capabilities"]["move"]["supported"]
    assert "no walk" in info["capabilities"]["move"]["reason"]
    assert not Robot(None, "h", info).can_move
    assert modes.available(pg, info, False)[0] == "manual"


def test_cached_structure_is_read_only_and_invalidates_after_edits(pg, monkeypatch):
    pg.add_robot("r", template="rover")
    first = robot_info(pg, "r")
    state = pg.session.data.qpos.copy(), pg.session.data.qvel.copy(), pg.session.data.ctrl.copy(), pg.session.data.time
    controllers = dict(pg.session.controllers)
    inspect = Mock(wraps=capabilities.inspect_robot)
    monkeypatch.setattr(capabilities, "inspect_robot", inspect)
    first["capabilities"]["manual"]["actuators"].clear()
    assert robot_info(pg, "r")["capabilities"]["manual"]["actuators"]
    assert not inspect.called
    for actual, old in zip((pg.session.data.qpos, pg.session.data.qvel, pg.session.data.ctrl), state[:3], strict=True):
        np.testing.assert_array_equal(actual, old)
    assert pg.session.data.time == state[3] and pg.session.controllers == controllers
    pg.update_robot("r", mounts=[{"template": "arm"}])
    assert robot_info(pg, "r")["capabilities"]["move_arm"]["endpoints"] == ["r/arm"]
    assert inspect.called
    pg.update_robot("r", mounts=[])
    assert not robot_info(pg, "r")["capabilities"]["move_arm"]["supported"]


def test_old_server_robot_descriptions_still_work():
    assert Robot(None, "old", {"kind": "humanoid"}).can_move
    assert Robot(None, "old", {"kind": "arm"}).has_arm


def test_opening_a_model_drops_cached_world_provenance(pg, tmp_path):
    pg.add_robot("r", template="rover")
    assert robot_info(pg, "r")["capabilities"]["source"]["kind"] == "template"
    pg.open_model(str(write(tmp_path / "imported.urdf", rover_urdf())))
    assert all(r["capabilities"]["source"]["kind"] == "model" for r in pg.behaviors.robots())


def test_kept_walk_refreshes_script_snapshot_and_viewer_without_recompile(pg):
    from robotforge.behaviors.rhythm_walk import SavedWalk, joint_names, save_walk
    from robotforge.control.cpg import Body
    from robotforge.learn.walk_rl import STAR

    pg.add_robot("crab", build=STAR)
    before = robot_info(pg, "crab")
    assert not before["capabilities"]["move"]["supported"]
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    assert hub.robot_stream()["robots"][0]["capabilities"] == before["capabilities"]
    version = pg.session.version
    body = Body.find(pg.session.model, "crab")
    walk = SavedWalk(
        robot="crab",
        joints=joint_names(pg.session.model, body),
        freq=2,
        theta=[0.0] * (3 * body.joints),
        speed=0.2,
        distance=0.8,
        seconds=4,
    )
    path = save_walk(pg.store.dir, walk)
    assert pg.session.version == version
    sim = Mock()
    robot = Robot(sim, "crab", before)
    sim.robot.return_value = Robot(sim, "crab", robot_info(pg, "crab"))
    assert robot.refresh() is robot and robot.can_move
    # A joining viewer reads a fresh scene and a welcome state before the next broadcast.
    assert hub.scene_message()["robots"][0]["capabilities"] == robot.capabilities
    hub.state_message()
    updated = hub.robot_stream()["robots"][0]["capabilities"]
    assert updated == robot.capabilities and updated["move"]["supported"]
    wrong = walk.model_copy(update={"joints": ["missing", *walk.joints[1:]]})
    save_walk(pg.store.dir, wrong)
    broken = robot_info(pg, "crab")
    assert broken["max_speed"] == 0.2  # retain the legacy listing field
    assert not broken["capabilities"]["move"]["supported"]
    assert "learn the walk again" in broken["capabilities"]["move"]["reason"]
    path.unlink()
    assert not hub.robot_stream()["robots"][0]["capabilities"]["move"]["supported"]


def test_query_never_constructs_a_controller(pg, monkeypatch):
    from robotforge.behaviors.drive import DiffDrive

    pg.add_robot("r", template="rover")
    pg.session.controllers.clear()
    monkeypatch.setattr(DiffDrive, "__init__", Mock(side_effect=AssertionError("query constructed a controller")))
    monkeypatch.setattr(pg.behaviors, "_mobile_controller", Mock(side_effect=AssertionError("legacy probe called")))
    first = robot_info(pg, "r")["capabilities"]
    assert first["move"]["supported"] and not pg.session.controllers
    assert robot_info(pg, "r")["capabilities"] == first


def test_an_active_gait_limit_refreshes_without_recompiling(pg):
    pg.add_robot("dog", template="quadruped")
    old = robot_info(pg, "dog")["capabilities"]["move"]["max_speed"]
    version = pg.session.version
    gait = pg.session.controllers["dog"]
    gait.set_period(gait.period * 1.5)
    info = robot_info(pg, "dog")
    assert pg.session.version == version
    assert info["capabilities"]["move"]["max_speed"] == round(gait.max_speed, 3) < old


def test_modes_keep_the_explicit_learned_argument(pg):
    pg.add_robot("dog", template="quadruped")
    info = robot_info(pg, "dog")
    assert modes.available(pg, info, True)[:2] == ["drive", "learned"]
    assert modes.available(pg, {k: v for k, v in info.items() if k != "capabilities"}, True)[:2] == ["drive", "learned"]


@pytest.mark.anyio
async def test_mcp_script_and_viewer_return_identical_capabilities(pg, tmp_path):
    pg.add_robot("a", template="arm")
    pg.add_robot("imported", file=str(write(tmp_path / "rover.urdf", rover_urdf())), pos=[2, 0, 0.1])
    pg.add_robot("combined", template="rover", mounts=[{"template": "arm"}], pos=[-2, 0])
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    expected = {r["name"]: r["capabilities"] for r in pg.behaviors.robots()}
    scene = hub.scene_message()
    assert {r["name"]: r["capabilities"] for r in scene["robots"]} == expected
    code = (
        "from robotforge.script import connect\nimport json\ns = connect()\n"
        "print(json.dumps({n: s.robot(n).capabilities for n in s.robots()}))"
    )
    async with Client(build_server(pg, 0)) as client:
        out = await client.call_tool("robots", {})
        assert {r["name"]: r["capabilities"] for r in json.loads(out.content[0].text)["robots"]} == expected
        await client.call_tool("write_script", {"name": "caps", "code": code})
        result = await client.call_tool("run_script", {"name": "caps"})
        got = json.loads(result.content[0].text)
        assert got["ok"], got
        assert json.loads(got["output"]) == expected
