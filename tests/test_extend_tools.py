"""The plugin tools: an agent writes a plugin, uses what it adds, reads why one did not load, trusts a world's own."""

import json
from pathlib import Path

import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.extend import REGISTRY, registry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession

SIGNPOST = """
from robotforge.extend import part, Box, Cylinder

@part("signpost", size="[height]", default_size=[1.2], fixed=True, color="brown", summary="a post with a board")
def signpost(size, rgba):
    h = size[0]
    return [Cylinder(0.05, h, at=[0, 0, h / 2]), Box([0.4, 0.03, 0.2], at=[0, 0, h - 0.1], color=1.3)]
"""


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(tmp_path / "user_plugins"))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        playground = Playground(session=SimSession.from_xml("<mujoco/>"))
        playground.new_world("lab", "checker")
        yield playground


def write(folder: Path, name: str, code: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(code, encoding="utf-8", newline="\n")
    return path


async def call(client: Client, tool: str, args: dict | None = None) -> dict:
    result = await client.call_tool(tool, args or {})
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


@pytest.mark.anyio
async def test_an_agent_writes_a_plugin_and_uses_its_part(pg):
    server = build_server(pg, find_free_port(8811))
    async with Client(server) as client:
        saved = await call(client, "write_script", {"name": "sign", "code": SIGNPOST, "plugin": True})
        assert Path(saved["saved"]).parent.name == "plugins"
        assert saved["plugin"]["loaded"] and saved["plugin"]["adds"] == [{"kind": "part", "name": "signpost"}]
        added = await call(client, "add_object", {"name": "sign1", "shape": "signpost", "pos": [1, 0]})
        assert "sign1" in json.dumps(added)
        listed = await call(client, "plugins")
        assert listed["parts"][0]["name"] == "signpost" and "not_loaded" not in listed
        bad = await call(
            client, "write_script", {"name": "oops", "code": "raise RuntimeError('nope')\n", "plugin": True}
        )
        assert bad["plugin"]["loaded"] is False and "nope" in bad["plugin"]["error"]
        listed = await call(client, "plugins")
        assert any("oops.py" in key and "nope" in why for key, why in listed["not_loaded"].items())


@pytest.mark.anyio
async def test_opening_a_world_with_untrusted_plugins_says_so_and_trust_loads_them(pg):
    store = pg.store
    write(store.dir / "plugins", "sign.py", SIGNPOST)
    opened = pg.open_world(str(store.dir))
    assert any("not written on this computer" in n and "sign.py" in n for n in opened["notes"])
    server = build_server(pg, find_free_port(8812))
    async with Client(server) as client:
        report = await call(client, "plugins", {"action": "trust"})
    assert report["parts"][0]["name"] == "signpost"
    again = pg.open_world(str(store.dir))
    assert any("runs its own Python plugin sign.py" in n for n in again["notes"])


def test_a_world_that_uses_a_plugin_shape_it_may_not_load_says_why_it_will_not_open(pg):
    path = write(pg.store.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(pg.store.dir)
    pg.add_object("s", shape="signpost", pos=[1.0, 0.0])
    path.write_text(SIGNPOST + "\n# changed\n", encoding="utf-8")  # no longer the file that was trusted
    REGISTRY.reload()
    with pytest.raises(PlaygroundError, match=r"plugins\(action=\"trust\"\)"):
        pg.open_world(str(pg.store.dir))
