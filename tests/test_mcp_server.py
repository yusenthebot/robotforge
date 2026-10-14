import json
import sys
import urllib.request

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from robotforge.cli import find_free_port
from robotforge.mcp_server import VIEWER_UI_URI, build_server
from robotforge.playground import Playground

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _text(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def _data(result) -> dict:
    return json.loads(_text(result))


async def test_tools_drive_the_demo_arm():
    playground = Playground.with_demo()
    server = build_server(playground, find_free_port(8790))
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"open_model", "describe_model", "set_controls", "simulate", "observe", "open_viewer"} <= names

        desc = _data(await client.call_tool("describe_model", {}))
        assert [a["name"] for a in desc["actuators"]] == ["yaw", "shoulder", "elbow"]

        result = _data(await client.call_tool("simulate", {"seconds": 1.5, "controls": {"shoulder": 1.0}}))
        assert result["time"] == pytest.approx(1.5, abs=0.01)

        obs = await client.call_tool("observe", {"view": "front", "width": 320, "height": 240})
        state = json.loads(obs.content[0].text)
        assert state["joints"]["shoulder"] == pytest.approx(1.0, abs=0.05)
        assert obs.content[1].type == "image" and obs.content[1].mime_type == "image/png"


async def test_tool_errors_reach_the_agent_with_guidance():
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        bad = await client.call_tool("set_controls", {"controls": {"sholder": 1.0}})
        assert bad.is_error
        assert "Did you mean shoulder" in _text(bad)

        missing = await client.call_tool("open_model", {"path": "/nope/robot.urdf"})
        assert missing.is_error and "not found" in _text(missing)

        view = await client.call_tool("observe", {"view": "sideways"})
        assert view.is_error and "unknown view" in _text(view)


async def test_viewer_app_resource_points_at_the_local_viewer():
    port = find_free_port(8790)
    server = build_server(Playground.with_demo(), port)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["open_viewer"].meta["ui"]["resourceUri"] == VIEWER_UI_URI
        res = await client.read_resource(VIEWER_UI_URI)
        assert f"http://127.0.0.1:{port}/" in res.contents[0].text


async def test_stdio_server_keeps_stdout_clean_and_serves_the_viewer():
    port = find_free_port(8790)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", f"from robotforge.cli import main; main(['mcp', '--port', '{port}'])"],
        env={"ROBOTFORGE_NO_BROWSER": "1"},
    )
    async with Client(params, read_timeout_seconds=30) as client:
        opened = _data(await client.call_tool("open_viewer", {"launch_browser": False}))
        assert opened["viewer_url"] == f"http://127.0.0.1:{port}/"
        html = urllib.request.urlopen(opened["viewer_url"], timeout=5).read().decode()
        assert "robotforge" in html
