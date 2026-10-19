"""Plugin controllers (inside the simulation) and behaviors (scripts written against the robot interface)."""

import json
import math
from pathlib import Path

import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.extend import REGISTRY, registry, routines
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

SWEEP = """
import math
from robotforge.extend import controller

@controller("sweep", summary="swings one joint to and fro", params={"joint": "shoulder", "amplitude": 0.4, "hz": 0.5})
def sweep(ctx):
    p = ctx.params
    ctx.set(p["joint"], p["amplitude"] * math.sin(2 * math.pi * p["hz"] * ctx.time))
"""

COUNTING = """
from robotforge.extend import controller

@controller("counting", rate_hz=100)
class Counting:
    def __init__(self, ctx):
        self.calls = 0

    def update(self, ctx):
        self.calls += 1
        if self.calls == 30:
            raise RuntimeError("thirty is enough")
"""

SQUARE = """
from robotforge.extend import behavior

@behavior("square", summary="walks a square", params={"side_s": 1.5, "speed": 0.2})
def square(robot, side_s, speed):
    start = robot.position
    for _ in range(4):
        robot.move(speed=speed, seconds=side_s)
        robot.move(speed=0.0, turn=1.2, seconds=1.0)
    end = robot.position
    return {"kind": robot.kind, "gap_m": round(((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5, 2)}
"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def folder(tmp_path, monkeypatch):
    user = tmp_path / "user_plugins"
    user.mkdir()
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(user))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        yield user


@pytest.fixture
def pg(folder):
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "checker")
    playground.add_robot("arm", template="arm")
    return playground


def plugin(folder: Path, name: str, code: str) -> None:
    (folder / name).write_text(code, encoding="utf-8", newline="\n")
    REGISTRY.reload()
    assert not REGISTRY.problems(), REGISTRY.problems()


def joint(pg: Playground, name: str) -> float:
    return pg.state()["joints"][f"arm/{name}"]


def test_a_controller_drives_a_robots_actuators_while_time_passes(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    info = routines.start_controller(pg.session, "arm", "sweep", {"amplitude": 0.5})
    assert info["options"] == {"joint": "shoulder", "amplitude": 0.5, "hz": 0.5} and info["rate_hz"] == 50.0
    seen = []
    for _ in range(40):
        pg.run_fast(lambda: False, 0.1)
        seen.append(joint(pg, "shoulder"))
    assert max(seen) > 0.3 and min(seen) < -0.3  # it swung both ways, about as far as asked


def test_stop_hands_the_robot_back(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    routines.start_controller(pg.session, "arm", "sweep", None)
    pg.run_fast(lambda: False, 0.5)
    assert isinstance(pg.session.controllers["arm"], routines.PluginController)
    assert pg.behaviors.stop("arm")["stopped"] is True
    assert "arm" not in pg.session.controllers


def test_an_option_the_controller_does_not_have_is_refused(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    with pytest.raises(registry.PluginError, match=r"takes no option 'speed'.*joint, amplitude, hz"):
        routines.start_controller(pg.session, "arm", "sweep", {"speed": 1})
    with pytest.raises(registry.PluginError, match="no controller named 'swep'"):
        routines.start_controller(pg.session, "arm", "swep", None)


def test_a_joint_that_is_not_there_names_the_ones_that_are(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    routines.start_controller(pg.session, "arm", "sweep", {"joint": "elbowz"})
    pg.run_fast(lambda: False, 0.2)
    status = pg.session.controllers["arm"].status()
    assert "no actuator 'elbowz'" in status["stopped_with_error"] and "shoulder" in status["stopped_with_error"]
    assert pg.session.data.time > 0.15  # the world went on


def test_a_controller_class_keeps_state_and_a_failure_stops_only_it(pg, folder):
    plugin(folder, "counting.py", COUNTING)
    routines.start_controller(pg.session, "arm", "counting", None)
    pg.run_fast(lambda: False, 1.0)
    status = pg.session.controllers["arm"].status()
    assert status["stopped_with_error"] == "RuntimeError: thirty is enough"
    assert status["updates"] == 30  # it ran 30 times and then not again


def test_it_replaces_what_drove_the_robot_and_a_move_replaces_it(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    routines.start_controller(pg.session, "arm", "sweep", None)
    first = pg.session.controllers["arm"]
    routines.start_controller(pg.session, "arm", "sweep", {"hz": 1.0})
    assert pg.session.controllers["arm"] is not first and pg.session.controllers["arm"].context.params["hz"] == 1.0


@pytest.mark.anyio
async def test_an_agent_runs_a_controller_and_stops_it(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    server = build_server(pg, find_free_port(8831))
    async with Client(server) as client:
        result = await client.call_tool(
            "plugins", {"action": "run", "name": "sweep", "robot": "arm", "options": {"amplitude": 0.6}, "seconds": 2}
        )
        assert not result.is_error, result.content[0].text
        out = json.loads(result.content[0].text)
        assert out["keeps_running"] is True and out["sim_seconds"] == pytest.approx(2.0, abs=0.1)
        assert out["status"]["updates"] > 80 and out["release_with"] == "stop(robot='arm')"
        stopped = await client.call_tool("stop", {"robot": "arm"})
        assert json.loads(stopped.content[0].text)["stopped"] is True
        wrong = await client.call_tool("plugins", {"action": "run", "name": "swepe", "robot": "arm"})
        assert wrong.is_error and "controller or behavior named 'swepe'" in wrong.content[0].text


@pytest.mark.anyio
async def test_an_agent_runs_a_behavior_on_a_dog(folder, tmp_path):
    plugin(folder, "square.py", SQUARE)
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("dog", template="quadruped")
    server = build_server(pg, find_free_port(8832))
    async with Client(server) as client:
        result = await client.call_tool(
            "plugins", {"action": "run", "name": "square", "robot": "dog", "options": {"side_s": 1.0}, "seconds": 30}
        )
        assert not result.is_error, result.content[0].text
        out = json.loads(result.content[0].text)
    assert out["ok"], out
    assert out["result"]["kind"] == "quadruped" and out["result"]["gap_m"] < 0.6  # it came round to near where it began
    assert math.hypot(*pg.behaviors.pose("dog")["position"][:2]) < 1.0
    assert "RESULT" not in out["output"]


INNER = """
from robotforge.extend import behavior

@behavior("inner", summary="a short step forward")
def inner(robot):
    robot.move(speed=0.1, seconds=0.3)
    return 7
"""

OUTER = """
from robotforge.extend import behavior, call

@behavior("outer", summary="runs inner from another file")
def outer(robot):
    return {"inner_said": call("inner", robot)}
"""


@pytest.mark.anyio
async def test_a_behavior_started_by_an_agent_can_call_a_behavior_from_another_file(folder):
    (folder / "a_inner.py").write_text(INNER, encoding="utf-8", newline="\n")
    plugin(folder, "b_outer.py", OUTER)
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("dog", template="quadruped")
    server = build_server(pg, find_free_port(8834))
    async with Client(server) as client:
        result = await client.call_tool("plugins", {"action": "run", "name": "outer", "robot": "dog", "seconds": 20})
        assert not result.is_error, result.content[0].text
        out = json.loads(result.content[0].text)
    assert out["ok"] and out["result"] == {"inner_said": 7}, out


def test_a_behavior_gets_a_wall_clock_limit_as_well_as_simulated_seconds():
    assert routines.wall_limit(1) == routines.MIN_WALL_SECONDS
    assert routines.wall_limit(60) == 180.0
    assert routines.wall_limit(300) == routines.MAX_WALL_SECONDS


@pytest.mark.anyio
async def test_a_behavior_that_does_not_exist_says_which_there_are(pg, folder):
    plugin(folder, "square.py", SQUARE)
    server = build_server(pg, find_free_port(8833))
    async with Client(server) as client:
        result = await client.call_tool("plugins", {"action": "run", "name": "squar", "robot": "arm"})
        assert result.is_error and "square" in result.content[0].text
        bad = await client.call_tool(
            "plugins", {"action": "run", "name": "square", "robot": "arm", "options": {"sides": 4}}
        )
        assert bad.is_error and "takes no option 'sides'" in bad.content[0].text


def test_the_list_shows_controllers_and_behaviors_with_their_options(pg, folder):
    plugin(folder, "sweep.py", SWEEP)
    plugin(folder, "square.py", SQUARE)
    report = REGISTRY.report()
    assert report["controllers"][0]["options"]["amplitude"] == 0.4 and report["controllers"][0]["rate_hz"] == 50.0
    assert report["behaviors"][0]["name"] == "square" and report["behaviors"][0]["options"] == {
        "side_s": 1.5,
        "speed": 0.2,
    }


def test_a_controller_and_a_behavior_cannot_share_a_name(folder):
    (folder / "a.py").write_text(SWEEP, encoding="utf-8")
    (folder / "b.py").write_text(SQUARE.replace('"square"', '"sweep"'), encoding="utf-8")
    REGISTRY.reload()
    assert "controller 'sweep' is already added by" in " ".join(REGISTRY.problems().values())
    assert list(REGISTRY.of("controller")) == ["sweep"] and not REGISTRY.of("behavior")
