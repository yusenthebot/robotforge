import json
import sys
import urllib.request

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

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


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


async def test_tools_drive_the_demo_arm(gl_reason):
    playground = Playground.with_demo()
    server = build_server(playground, 0)
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
        if gl_reason is None:
            assert obs.content[1].type == "image" and obs.content[1].mime_type == "image/png"
        else:  # headless machine: the state still arrives, with a note instead of an image
            assert len(obs.content) == 1 and "No image" in state["note"]


async def test_tool_errors_reach_the_agent_with_guidance():
    server = build_server(Playground.with_demo(), 0)
    async with Client(server) as client:
        bad = await client.call_tool("set_controls", {"controls": {"sholder": 1.0}})
        assert bad.is_error
        assert "Did you mean shoulder" in _text(bad)

        missing = await client.call_tool("open_model", {"path": "/nope/robot.urdf"})
        assert missing.is_error and "not found" in _text(missing)

        view = await client.call_tool("observe", {"view": "sideways"})
        assert view.is_error and "unknown view" in _text(view)


async def test_a_crash_inside_a_tool_says_what_happened(monkeypatch):
    """The SDK answers crashes with a bare "Error executing tool"; a Codex session saw only that for observe and
    run_script and fell back to stepping the simulation by hand, 150 calls deep."""
    playground = Playground.with_demo()
    server = build_server(playground, 0)

    def broken_render(*_args, **_kwargs):
        raise RuntimeError("no GL context today")

    def broken_simulate(*_args, **_kwargs):
        raise KeyError("lost")

    monkeypatch.setattr(playground, "snapshot", broken_render)
    monkeypatch.setattr(playground, "simulate", broken_simulate)
    async with Client(server) as client:
        obs = await client.call_tool("observe", {})
        assert not obs.is_error  # the state is still worth having
        assert "rendering failed (RuntimeError: no GL context today)" in json.loads(_text(obs))["note"]
        crashed = await client.call_tool("simulate", {"seconds": 0.1})
        assert crashed.is_error and "KeyError: 'lost'" in _text(crashed)


async def test_viewer_app_resource_points_at_the_local_viewer():
    playground = Playground.with_demo()
    server = build_server(playground, 0)
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert tools["open_viewer"].meta["ui"]["resourceUri"] == VIEWER_UI_URI
        res = await client.read_resource(VIEWER_UI_URI)
        assert playground.viewer_url.startswith("http://127.0.0.1:") and playground.viewer_url in res.contents[0].text


async def test_servers_built_together_each_hold_their_own_viewer_port():
    """Tests took a port that was free (find_free_port) and the viewer bound it later; tests running side by side
    were given the same port, and the later viewer exited (SystemExit 3) in 2 of 24 runs. Port 0 is chosen by the
    system and held from the start."""
    first, second = Playground.with_demo(), Playground.with_demo()
    servers = build_server(first, 0), build_server(second, 0)
    assert first.viewer_url != second.viewer_url and ":0/" not in first.viewer_url
    async with Client(servers[0]), Client(servers[1]):
        for playground in (first, second):  # fetched from a thread: the viewers serve on this event loop
            page = await anyio.to_thread.run_sync(lambda url=playground.viewer_url: _fetch(url))
            assert "robotforge" in page


async def test_stdio_server_keeps_stdout_clean_and_serves_the_viewer():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from robotforge.cli import main; main(['mcp', '--port', '0'])"],
        env={"ROBOTFORGE_NO_BROWSER": "1"},
    )
    async with Client(params, read_timeout_seconds=30) as client:
        opened = _data(await client.call_tool("open_viewer", {"launch_browser": False}))
        assert opened["viewer_url"].startswith("http://127.0.0.1:") and ":0/" not in opened["viewer_url"]
        html = urllib.request.urlopen(opened["viewer_url"], timeout=5).read().decode()
        assert "robotforge" in html
