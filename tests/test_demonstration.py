"""Demonstrations: recorded from the physics steps and the commands, replayed on the same robot or another of its kind,
exported as JSON and NPZ."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from robotforge.demonstrations import Demonstrations
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.demonstration import Demonstration, DemonstrationError, clean_name
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("demonstrated")
    playground.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    playground.add_robot("pup", template="quadruped", pos=[0.0, 2.0], yaw=90)
    playground.add_robot("arm", template="arm", pos=[2.5, 0.0])
    playground.add_robot("arm2", template="arm", pos=[2.5, 1.5])
    return playground


def record_walk(pg, name="walk"):
    demos: Demonstrations = pg.demonstrations
    demos.start("dog", name)
    pg.behaviors.move("dog", 0.3, 0.0)
    pg.run_fast(lambda: False, 2.0)
    pg.behaviors.move("dog", 0.0, 0.8)
    pg.run_fast(lambda: False, 1.0)
    pg.behaviors.stop("dog")
    pg.run_fast(lambda: False, 0.4)
    return demos.stop(name)


def displacement(pg, robot):
    return np.array(pg.behaviors.pose(robot)["position"][:2])


def test_a_recording_keeps_frames_joints_targets_and_the_commands(pg):
    start = displacement(pg, "dog")
    out = record_walk(pg)
    assert out["name"] == "walk" and out["robot"] == "dog" and out["kind"] == "quadruped"
    assert 3.2 <= out["seconds"] <= 3.6
    assert out["frames"] == pytest.approx(out["seconds"] * 20 + 1, abs=2)
    assert out["joints"] == 12 and out["actuators"] == 12
    assert out["command_counts"] == {"move": 2, "stop": 1}
    demo = pg.demonstrations.get("walk")
    assert demo.joints.shape == (demo.frames, 12) and demo.ctrl.shape == (demo.frames, 12)
    assert [c["op"] for c in demo.commands] == ["move", "move", "stop"]
    assert demo.commands[0]["speed"] == pytest.approx(0.3) and demo.commands[1]["turn"] == pytest.approx(0.8)
    assert demo.commands[0]["t"] < demo.commands[1]["t"] < demo.commands[2]["t"]
    assert np.all(np.diff(demo.time) > 0)
    walked = np.linalg.norm(demo.base_pos[-1][:2] - demo.base_pos[0][:2])
    assert walked > 0.3 and np.linalg.norm(displacement(pg, "dog") - start) == pytest.approx(walked, abs=0.05)


def test_the_files_are_json_and_npz_that_load_back(pg):
    record_walk(pg)
    folder = pg.store.dir / "demonstrations"
    assert (folder / "walk.json").is_file()  # stopping saves it
    exported = pg.demonstrations.export("walk", "npz")
    assert exported["path"].endswith("walk.npz") and exported["format"] == "npz"
    with np.load(exported["path"], allow_pickle=False) as z:
        assert {"time", "joint_pos", "joint_names", "ctrl", "base_pos", "base_quat", "command_op"} <= set(z.files)
        assert z["joint_pos"].shape == (len(z["time"]), 12) and len(z["joint_names"]) == 12
        assert list(z["command_op"]) == ["move", "move", "stop"]
        assert json.loads(str(z["meta"]))["format"] == "robotforge-demonstration"
    raw = json.loads((folder / "walk.json").read_text(encoding="utf-8"))
    assert raw["format"] == "robotforge-demonstration" and raw["kind"] == "quadruped"
    assert len(raw["joints"]) == len(raw["time"]) and len(raw["joints"][0]) == 12
    first = Demonstration.load(folder / "walk.json")
    second = Demonstration.load(folder / "walk.npz")
    assert first.commands == second.commands and first.joint_names == second.joint_names
    assert np.allclose(first.joints, second.joints, atol=1e-4) and np.allclose(first.time, second.time, atol=1e-3)


def test_sensor_readings_are_kept_with_the_robots_cheap_sensors(pg):
    pg.add_sensor("dog", {"type": "imu", "name": "body_imu"})
    pg.demonstrations.start("dog", "sensed")
    pg.behaviors.move("dog", 0.2, 0.0)
    pg.run_fast(lambda: False, 1.0)
    pg.demonstrations.stop("sensed")
    demo = pg.demonstrations.get("sensed")
    assert any(n.startswith("body_imu.orientation_deg.") for n in demo.sensor_names)
    assert demo.sensors.shape == (demo.frames, len(demo.sensor_names))
    assert np.isfinite(demo.sensors).all()


def test_replaying_the_commands_on_another_robot_of_the_kind(pg):
    record_walk(pg)
    original = pg.demonstrations.get("walk")
    walked = np.linalg.norm(original.base_pos[-1][:2] - original.base_pos[0][:2])
    heading_before = pg.behaviors.pose("pup")["heading_deg"]
    start = displacement(pg, "pup")
    replay = pg.demonstrations.replay("walk", "pup")
    assert not replay.done
    pg.run_fast(lambda: replay.done, 6.0)
    assert replay.done and replay.issued == 3 and not replay.problems
    moved = np.linalg.norm(displacement(pg, "pup") - start)
    assert moved == pytest.approx(walked, rel=0.35)  # the same walk, from where the pup stood and faced
    turned = (pg.behaviors.pose("pup")["heading_deg"] - heading_before + 180) % 360 - 180
    assert 20 < turned < 90  # it turned too, left as the dog did
    assert pg.demonstrations.replays["pup"].status()["done"]


def test_a_replay_refuses_a_robot_of_another_kind_and_unknown_names(pg):
    record_walk(pg)
    with pytest.raises(DemonstrationError, match="a quadruped.s demonstration"):
        pg.demonstrations.replay("walk", "arm")
    with pytest.raises(DemonstrationError, match="no demonstration called nothing"):
        pg.demonstrations.replay("nothing", "dog")
    with pytest.raises(DemonstrationError, match="not a name"):
        clean_name("../escape")
    with pytest.raises(DemonstrationError, match="nothing is being recorded"):
        pg.demonstrations.stop()


def test_an_arms_targets_replay_on_another_arm_relative_to_its_base(pg):
    demos = pg.demonstrations
    demos.start("arm", "reach")
    pg.behaviors.move_arm("arm", [2.75, 0.1, 0.15], "open")
    pg.run_fast(lambda: pg.behaviors.arms("arm") and pg.session.controllers["arm"].done, 10.0)
    pg.behaviors.set_gripper("arm", "close")
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 5.0)
    demos.stop("reach")
    demo = demos.get("reach")
    assert [c["op"] for c in demo.commands] == ["move_arm", "gripper"]
    assert demo.commands[0]["position"][0] == pytest.approx(0.25, abs=0.02)  # in the arm's own frame
    replay = demos.replay("reach", "arm2")
    pg.run_fast(lambda: replay.done, 14.0)
    assert replay.issued == 2 and not replay.problems
    tool = pg.session.data.site_xpos[
        __import__("mujoco").mj_name2id(pg.session.model, __import__("mujoco").mjtObj.mjOBJ_SITE, "arm2/tcp")
    ]
    assert tool[0] == pytest.approx(2.75, abs=0.08) and tool[1] == pytest.approx(1.6, abs=0.08)


def test_joint_replay_plays_the_actuator_targets_back_on_the_same_arm(pg):
    demos = pg.demonstrations
    demos.start("arm", "swing")
    pg.behaviors.move_arm("arm", [2.75, -0.1, 0.2])
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 10.0)
    demos.stop("swing")
    demo = demos.get("swing")
    pg.session.reset()
    replay = demos.replay("swing", "arm", mode="joints")
    pg.run_fast(lambda: replay.done, demo.seconds + 2)
    assert replay.done
    joints = pg.state()["joints"]
    for name, value in zip(demo.joint_names, demo.joints[-1], strict=True):
        assert joints[f"arm/{name}"] == pytest.approx(float(value), abs=0.05), name
    with pytest.raises(DemonstrationError, match="an arm's demonstration"):
        demos.replay("swing", "dog", mode="joints")


def test_a_reset_ends_a_recording_and_says_so(pg):
    pg.demonstrations.start("dog", "cut")
    pg.behaviors.move("dog", 0.2, 0.0)
    pg.run_fast(lambda: False, 1.0)
    pg.session.reset()
    pg.run_fast(lambda: False, 0.2)
    out = pg.demonstrations.stop("cut")
    assert "reset" in out["note"] and out["seconds"] < 1.5


def test_a_second_recording_of_the_same_robot_or_name_is_refused(pg):
    pg.demonstrations.start("dog", "one")
    with pytest.raises(DemonstrationError, match="already being recorded"):
        pg.demonstrations.start("dog", "two")
    pg.run_fast(lambda: False, 0.5)
    pg.demonstrations.stop("one")
    with pytest.raises(DemonstrationError, match="exists"):
        pg.demonstrations.start("dog", "one")
    pg.demonstrations.start("dog", "one", overwrite=True)


def test_joint_replay_needs_the_same_actuators_and_commands_replay_needs_none(pg, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    pg.add_robot("panda", library="panda", pos=[4.0, 0.0])
    demos = pg.demonstrations
    demos.start("arm", "tiny")
    pg.behaviors.move_arm("arm", [2.75, 0.0, 0.2])
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 10.0)
    demos.stop("tiny")
    with pytest.raises(DemonstrationError, match="has no actuator"):
        demos.replay("tiny", "panda", mode="joints")
    replay = demos.replay("tiny", "panda")  # the commands work on any arm that reaches
    pg.run_fast(lambda: replay.done, 14.0)
    assert replay.done and replay.issued == 1 and not replay.problems


def test_a_trip_records_the_command_not_the_stop_it_makes_first(pg):
    """navigate stops the robot before it sets off: that is part of the trip, and replayed after it, a "stop" would
    cancel the trip the replay had just started."""
    pg.demonstrations.start("dog", "trip")
    pg.behaviors.navigate("dog", [1.2, 0.6])
    pg.run_fast(lambda: False, 0.5)
    pg.demonstrations.stop("trip")
    demo = pg.demonstrations.get("trip")
    assert [c["op"] for c in demo.commands] == ["go"]
    assert demo.commands[0]["to"] == [1.2, 0.6]
    replay = pg.demonstrations.replay("trip", "pup")
    pg.run_fast(lambda: replay.done, 3.0)
    assert replay.issued == 1 and not replay.problems


def test_what_a_controller_asks_of_the_behaviors_is_not_a_command(pg):
    """An errand drives the robot, stops it and works its arm from inside the simulation's step: its own steps."""

    class Errand:
        rate_hz = 50.0

        def update(self, model, data):
            pg.behaviors.stop("dog")

        def status(self):
            return {}

    pg.demonstrations.start("dog", "inner")
    pg.session.set_controller("errand", Errand())
    pg.run_fast(lambda: False, 0.5)
    pg.behaviors.move("dog", 0.1, 0.0)  # a command from outside, on the same thread, after the controller's calls
    pg.run_fast(lambda: False, 0.2)
    pg.demonstrations.stop("inner")
    assert [c["op"] for c in pg.demonstrations.get("inner").commands] == ["move"]


def test_a_jogged_arm_follows_the_stick_and_holds_when_it_is_released(pg):
    import mujoco

    site = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_SITE, "arm/tcp")
    pg.behaviors.move_arm("arm", [2.75, 0.0, 0.2])
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 10.0)
    start = pg.session.data.site_xpos[site].copy()
    pg.demonstrations.start("arm", "jog")
    for _ in range(8):  # a pad sends its stick every 0.1 s
        pg.behaviors.jog_arm("arm", [0.0, 0.1, 0.0])
        pg.run_fast(lambda: False, 0.1)
    pg.behaviors.jog_arm("arm", [0.0, 0.0, 0.0])
    pg.run_fast(lambda: False, 0.5)
    moved = pg.session.data.site_xpos[site] - start
    assert moved[1] == pytest.approx(0.08, abs=0.03) and abs(moved[0]) < 0.03 and abs(moved[2]) < 0.03
    held = pg.session.data.site_xpos[site].copy()
    pg.run_fast(lambda: False, 1.0)
    assert np.linalg.norm(pg.session.data.site_xpos[site] - held) < 0.01
    demo = pg.demonstrations.get("jog") if pg.demonstrations.stop("jog") else None
    assert {c["op"] for c in demo.commands} == {"jog"} and len(demo.commands) == 9
    with pytest.raises(Exception, match="velocity"):
        pg.behaviors.jog_arm("arm", [0.0, 0.1])


def test_a_pad_that_goes_quiet_does_not_leave_the_tool_moving(pg):
    import mujoco

    site = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_SITE, "arm/tcp")
    pg.behaviors.move_arm("arm", [2.75, 0.0, 0.2])
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 10.0)
    start = pg.session.data.site_xpos[site].copy()
    pg.behaviors.jog_arm("arm", [0.0, 0.1, 0.0])  # once, then silence
    pg.run_fast(lambda: False, 3.0)
    assert abs(pg.session.data.site_xpos[site][1] - start[1]) < 0.08  # it went on for the half second it was told


def test_the_viewers_pad_message_jogs_the_arm_and_the_state_says_what_is_recording(pg):
    import mujoco

    from robotforge.viewer.server import ViewerHub

    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    site = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_SITE, "arm/tcp")
    pg.behaviors.move_arm("arm", [2.75, 0.0, 0.2])
    pg.run_fast(lambda: pg.session.controllers["arm"].done, 10.0)
    start = pg.session.data.site_xpos[site].copy()
    pg.demonstrations.start("arm", "pad")
    for _ in range(5):
        hub.handle({"type": "teleop_arm", "robot": "arm", "velocity": [0.0, -0.1, 0.0], "gripper": "close"})
        pg.run_fast(lambda: False, 0.1)
    assert hub.state_message()["recording"] == [{"name": "pad", "robot": "arm", "seconds": pytest.approx(0.5, abs=0.2)}]
    assert pg.session.data.site_xpos[site][1] < start[1] - 0.02  # right on the pad is -y here: it went that way
    with pytest.raises(ValueError, match="velocity"):
        hub.handle({"type": "teleop_arm", "robot": "arm", "velocity": [0.0, 0.1]})
    pg.demonstrations.stop("pad")
    assert "recording" not in hub.state_message()


SCRIPT = """
from robotforge.script import connect

sim = connect()
dog = sim.robot("dog")
with dog.record("from_script") as demo:
    dog.move(speed=0.3, seconds=1.5)
print("recorded", demo.name, demo.stop()["command_counts"])
done = sim.robot("pup").replay("from_script")
print("replayed", done["robot"], done["done"], done["commands_issued"])
print("exported", sim.export_demo("from_script", "npz")["format"], [d["name"] for d in sim.demos()["demonstrations"]])
"""


@pytest.mark.anyio
async def test_a_script_records_replays_and_exports_with_the_robot_interface(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "demo_script", "code": SCRIPT})
        result = await client.call_tool("run_script", {"name": "demo_script"})
    out = json.loads(result.content[0].text)
    assert out["ok"], out["output"]
    assert "recorded from_script {'move': 1, 'stop': 1}" in out["output"]
    assert "replayed pup True 2" in out["output"]
    assert "exported npz ['from_script']" in out["output"]
    assert (pg.store.dir / "demonstrations" / "from_script.npz").is_file()


def test_exports_stay_inside_the_demonstrations_folder(pg):
    record_walk(pg)
    with pytest.raises(DemonstrationError, match="file name"):
        pg.demonstrations.export("walk", "json", to="../../escape.json")
    with pytest.raises(DemonstrationError, match="parquet"):
        pg.demonstrations.export("walk", "parquet")
    out = pg.demonstrations.export("walk", "json", to="copy")
    assert out["path"].endswith("copy.json")


pytestmark_anyio = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_the_tool_records_replays_and_exports(pg):
    server = build_server(pg, 0)

    async def call(**args):
        async with Client(server) as client:
            result = await client.call_tool("demonstration", args)
        return json.loads(result.content[0].text)

    started = await call(action="start", robot="dog", name="tool_walk")
    assert started["recording"] == "tool_walk"
    async with Client(server) as client:
        await client.call_tool("move", {"robot": "dog", "speed": 0.3, "seconds": 2.0})
    stopped = await call(action="stop")
    assert (
        stopped["name"] == "tool_walk" and stopped["command_counts"]["move"] >= 1 and stopped["path"].endswith(".json")
    )
    replayed = await call(action="replay", name="tool_walk", robot="pup")
    assert replayed["done"] and replayed["robot"] == "pup" and replayed["commands_issued"] >= 1
    exported = await call(action="export", name="tool_walk", format="npz")
    assert exported["format"] == "npz"
    listing = await call(action="list")
    assert [d["name"] for d in listing["demonstrations"]] == ["tool_walk"]
    assert math.isclose(listing["demonstrations"][0]["seconds"], stopped["seconds"])
