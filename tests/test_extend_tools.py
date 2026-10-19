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
async def test_opening_a_world_with_untrusted_plugins_says_so_and_an_agent_cannot_trust_them(pg):
    store = pg.store
    write(store.dir / "plugins", "sign.py", SIGNPOST)
    opened = pg.open_world(str(store.dir))
    assert any("not written on this computer" in n and "sign.py" in n for n in opened["notes"])
    assert any(f'robotforge plugins trust "{store.dir}"' in n for n in opened["notes"])
    server = build_server(pg, find_free_port(8812))
    async with Client(server) as client:
        refused = await client.call_tool("plugins", {"action": "trust"})
        assert refused.is_error and "the user's decision" in refused.content[0].text
        assert "robotforge plugins trust" in refused.content[0].text
        shown = await call(client, "plugins", {"action": "show"})
        assert shown["trusted"] is False and "def signpost" in shown["files"]["sign.py"]
        assert "signpost" not in REGISTRY.of("part")  # reading it did not load it
    REGISTRY.trust(store.dir)  # what the user's `robotforge plugins trust` does
    again = pg.open_world(str(store.dir))
    assert any("runs its own Python plugin sign.py" in n for n in again["notes"])
    later = pg.add_object("cube", shape="box", pos=[2.0, 0.0])
    assert not any("plugin" in n for n in later["notes"])  # said once, when the world opened


def test_a_world_that_uses_a_plugin_shape_it_may_not_load_says_why_it_will_not_open(pg):
    path = write(pg.store.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(pg.store.dir)
    pg.add_object("s", shape="signpost", pos=[1.0, 0.0])
    path.write_text(SIGNPOST + "\n# changed\n", encoding="utf-8")  # no longer the file that was trusted
    REGISTRY.reload()
    with pytest.raises(PlaygroundError) as why:
        pg.open_world(str(pg.store.dir))
    text = str(why.value)
    assert "unknown shape 'signpost'" in text and f"{pg.store.dir.name}/plugins/sign.py" in text
    assert 'plugins(action="show"' in text and f'robotforge plugins trust "{pg.store.dir}"' in text


def test_only_the_open_worlds_plugins_are_loaded_so_two_worlds_can_carry_the_same_file(pg, tmp_path):
    first = pg.store
    write(first.dir / "plugins", "sign.py", SIGNPOST)
    REGISTRY.trust(first.dir)
    other = pg.new_world("second", "checker")
    assert other["world"] == "second" and "signpost" not in REGISTRY.of(
        "part"
    )  # the first world's file is not loaded here
    write(pg.store.dir / "plugins", "sign.py", SIGNPOST)  # the same file, written into the second world
    pg.store.load()
    assert list(REGISTRY.problems()) == ["second/plugins/sign.py"]  # written by hand: waits to be trusted
    REGISTRY.trust(pg.store.dir)
    assert "signpost" in REGISTRY.of("part") and not REGISTRY.problems()


@pytest.mark.anyio
async def test_the_template_list_says_what_plugins_add_once_there_are_some(pg):
    server = build_server(pg, find_free_port(8813))
    async with Client(server) as client:
        before = await call(client, "list_templates")
        assert "plugins" not in before
        await call(client, "write_script", {"name": "sign", "code": SIGNPOST, "plugin": True})
        after = await call(client, "list_templates")
        assert after["plugins"]["parts"][0]["name"] == "signpost"
