"""The robot interface in scripts (sim.robot): one script, a template dog and a Go2, a template arm and a Panda."""

import json
from pathlib import Path

import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"

WALK = """
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
ticks = []
robot.on_step(lambda r: ticks.append(round(r.time, 2)), every=0.5)
start = robot.position
robot.move(speed=0.25)
robot.run(2.0)
robot.stop()
end = robot.position
print("kind", robot.kind)
print("moved", round(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5, 2))
print("ticks", len(ticks), ticks[:2])
tilt = robot.sensors["body_imu"]["orientation_deg"]
print("upright", abs(tilt["roll"]) < 30 and abs(tilt["pitch"]) < 30)
print("named", "body_imu" in robot.sensors, robot.sensors.get("nothing_here", "none"))
print("arm", robot.has_arm)
"""

REACH = """
from robotforge.script import connect

sim = connect()
robot = sim.robot(sim.robots()[0])
x, y, z = robot.position
print("kind", robot.kind)
print("arms", [a.name for a in robot.arms])
tool = robot.arm.move_to([x + 0.2, y, 0.15], gripper="open")
print("reached", all(abs(a - b) < 0.05 for a, b in zip(tool, [x + 0.2, y, 0.15])))
robot.gripper.close()
print("closed")
print("can_move", robot.can_move)
"""

WRONG = """
from robotforge.script import connect, ScriptError

sim = connect()
robot = sim.robot(sim.robots()[0])
for what in (lambda: robot.arm, lambda: robot.gripper, lambda: robot.move(0.1)):
    try:
        what()
        print("no error")
    except ScriptError as exc:
        print("error:", exc)
"""


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def world(tmp_path, monkeypatch, **robot) -> Playground:
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot(**robot)
    return pg


async def run_script(pg: Playground, code: str, port: int) -> dict:
    server = build_server(pg, find_free_port(port))
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "t", "code": code})
        result = await client.call_tool("run_script", {"name": "t", "max_sim_seconds": 60})
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


def printed(out: dict) -> dict[str, str]:
    assert out["ok"], out["output"]
    lines = [line.split(" ", 1) for line in out["output"].strip().splitlines()]
    return {k: v for k, v in (line + [""] if len(line) == 1 else line for line in lines)}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("robot", "kind"),
    [({"name": "dog", "template": "quadruped"}, "quadruped"), ({"name": "go2", "library": "go2"}, "quadruped")],
)
async def test_one_script_walks_a_template_dog_and_a_go2(tmp_path, monkeypatch, robot, kind):
    pg = world(tmp_path, monkeypatch, **robot)
    pg.add_sensor(robot["name"], {"type": "imu", "name": "body_imu"})
    got = printed(await run_script(pg, WALK, 8841))
    assert got["kind"] == kind
    assert float(got["moved"]) > 0.15
    assert got["ticks"].startswith("4 ")  # every half second of the two it ran, the callback fired
    assert got["upright"] == "True" and got["named"] == "True none" and got["arm"] == "False"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "robot",
    [{"name": "arm", "template": "arm"}, {"name": "panda", "library": "panda"}],
)
async def test_one_script_reaches_with_a_template_arm_and_a_panda(tmp_path, monkeypatch, robot):
    pg = world(tmp_path, monkeypatch, **robot)
    got = printed(await run_script(pg, REACH, 8842))
    assert got["kind"] == "arm" and got["reached"] == "True" and got["closed"] == "" and got["can_move"] == "False"


@pytest.mark.anyio
async def test_a_dog_with_an_arm_does_both(tmp_path, monkeypatch):
    pg = world(tmp_path, monkeypatch, name="go2", library="go2", mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.3, 0.0])
    code = """
from robotforge.script import connect

sim = connect()
robot = sim.robot("go2")
print("arms", [a.name for a in robot.arms], robot.can_move)
robot.move(0.2, seconds=1.0)
robot.gripper.open()
print("done")
"""
    got = printed(await run_script(pg, code, 8843))
    assert got["arms"] == "['go2/arm'] True" and "done" in got


@pytest.mark.anyio
async def test_what_a_robot_cannot_do_says_what_it_can(tmp_path, monkeypatch):
    pg = world(tmp_path, monkeypatch, name="dog", template="quadruped")
    out = await run_script(pg, WRONG, 8844)
    lines = out["output"].strip().splitlines()
    assert lines[0].startswith("error: dog is a quadruped and cannot work an arm; it can walk or drive: use robot.move")
    assert lines[1].startswith("error: dog is a quadruped and cannot hold things")
    assert lines[2] == "no error"  # a dog moves
    pg2 = world(tmp_path / "again", monkeypatch, name="arm", template="arm")
    lines = (await run_script(pg2, WRONG, 8845))["output"].strip().splitlines()
    assert "cannot walk or drive; it has an arm: use robot.arm and robot.gripper" in lines[2]


@pytest.mark.anyio
async def test_a_callback_fires_while_a_script_steps_and_not_otherwise(tmp_path, monkeypatch):
    pg = world(tmp_path, monkeypatch, name="arm", template="arm")
    code = """
from robotforge.script import connect

sim = connect()
count = []
sim.on_step(lambda: count.append(sim.time), every=0.25)
sim.step(1.0)
print("fired", len(count))
sim.step(0.1)
print("after", len(count))
"""
    got = printed(await run_script(pg, code, 8846))
    assert got["fired"] == "4" and got["after"] == "4"
