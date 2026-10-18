import json
import textwrap
from pathlib import Path

import pytest
from mcp import Client

from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

pytestmark = pytest.mark.anyio

SCRIPT = textwrap.dedent(
    """
    # Walk the dog forward, then have the arm put the cube in the tray.
    from robotforge.script import connect

    sim = connect()
    start = sim.position("dog")
    sim.move("dog", speed=0.25)
    sim.step(3.0)
    sim.stop("dog")
    end = sim.position("dog")
    print(f"dog walked {end[0] - start[0]:.2f} m")

    sim.set_controls({"arm/wrist_roll": 0.3})
    sim.step(0.5)
    print("roll", round(sim.joints("arm")["wrist_roll"], 2))

    ok = sim.pick_and_place("arm", "cube", "tray")
    print("cube in tray:", ok, sim.position("cube"))
    """
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("scripted")
    playground.add_robot("dog", template="quadruped", pos=[0, 0.9])
    playground.add_robot("arm", template="arm")
    playground.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, -0.12], color="red")
    playground.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="blue")
    return playground


async def _call(client, tool, args):
    result = await client.call_tool(tool, args)
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    return result, text


async def test_agent_written_script_walks_and_places(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        _, saved = await _call(client, "write_script", {"name": "demo_task", "code": SCRIPT})
        assert Path(json.loads(saved)["saved"]).parts[-2:] == ("behaviors", "demo_task.py")  # separator-agnostic
        _, ran = await _call(client, "run_script", {"name": "demo_task"})
        out = json.loads(ran)
        assert out["ok"], out["output"]
        walked = float(out["output"].split("dog walked ")[1].split(" m")[0])
        assert walked > 0.4
        assert "roll 0.3" in out["output"]
        assert "cube in tray: True" in out["output"]  # A2 through an agent-written script


async def test_script_errors_come_back_as_tracebacks(pg):
    server = build_server(pg, 0)
    code = "from robotforge.script import connect\nsim = connect()\nsim.move('arm', 0.2)\n"
    async with Client(server) as client:
        await _call(client, "write_script", {"name": "bad", "code": code})
        _, ran = await _call(client, "run_script", {"name": "bad"})
        out = json.loads(ran)
        assert not out["ok"]
        assert "only legged robots" in out["output"]


async def test_script_time_budget_is_enforced(pg):
    server = build_server(pg, 0)
    code = "from robotforge.script import connect\nsim = connect()\nwhile True:\n    sim.step(1.0)\n"
    async with Client(server) as client:
        await _call(client, "write_script", {"name": "forever", "code": code})
        _, ran = await _call(client, "run_script", {"name": "forever", "max_sim_seconds": 3})
        out = json.loads(ran)
        assert not out["ok"]
        assert "time limit" in out["output"]
        assert out["sim_seconds"] == pytest.approx(3.0, abs=0.05)


async def test_script_names_are_validated(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        result, text = await _call(client, "write_script", {"name": "../escape", "code": "print(1)"})
        assert result.is_error and "lowercase" in text


async def test_script_output_survives_non_ascii_text(pg):
    # A child process writing to a pipe uses the ANSI code page on Windows unless told otherwise.
    code = "print('h\\u00e9llo \\u4f60\\u597d \\U0001f916')\n"
    server = build_server(pg, 0)
    async with Client(server) as client:
        await _call(client, "write_script", {"name": "unicode_out", "code": code})
        _, ran = await _call(client, "run_script", {"name": "unicode_out"})
        out = json.loads(ran)
        assert out["ok"], out["output"]
        assert "h\u00e9llo \u4f60\u597d \U0001f916" in out["output"]
