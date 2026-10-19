"""What running a world's own plugin code needs: every file under its plugins/ folder reviewed as one set, trusted
only by the user (the command line), never by an agent, and the module machinery a plugin leans on."""

import json
import sys
from pathlib import Path

import pytest
from mcp import Client

from robotforge import cli
from robotforge.cli import find_free_port
from robotforge.design.store import WorldStore
from robotforge.extend import REGISTRY, registry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def home(tmp_path, monkeypatch):
    user = tmp_path / "user_plugins"
    user.mkdir()
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(user))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        yield tmp_path


def write(path: Path, code: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8", newline="\n")
    return path


def downloaded_world(tmp_path: Path, marker: Path) -> Path:
    """A world as somebody else sends it: a plugin that imports a helper and a module called colorsys, which sit
    beside it. Importing the helper leaves a mark on disk, so that code that ran can be told from code that did not."""
    world = tmp_path / "downloaded" / "dl"
    WorldStore.create("dl", parent=tmp_path / "downloaded")
    plugins = world / "plugins"
    write(
        plugins / "nice.py",
        "import colorsys\nimport _helper\nfrom robotforge.extend import part, Box\n"
        "@part('nice', size='[a]', default_size=[1])\n"
        "def nice(size, rgba):\n    return [Box([size[0], size[0], size[0]])]\n",
    )
    write(
        plugins / "_helper.py", f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran', encoding='utf-8')\n"
    )
    write(plugins / "colorsys.py", "SHADOW = True\n")
    return world


def test_a_helper_beside_a_plugin_is_part_of_what_is_trusted(home):
    marker = home / "helper_ran"
    world = downloaded_world(home, marker)
    WorldStore(world).load()
    assert not marker.exists() and "nice" not in REGISTRY.of("part")
    listed = REGISTRY.problems()
    assert {f"dl/plugins/{n}" for n in ("nice.py", "_helper.py", "colorsys.py")} <= set(listed)  # the helper too

    sys.modules.pop("colorsys", None)  # not imported yet, as in a fresh process
    REGISTRY.trust(world)
    assert marker.read_text(encoding="utf-8") == "ran" and "nice" in REGISTRY.of("part")
    assert not hasattr(sys.modules["colorsys"], "SHADOW")  # a world's file never stands in for a standard module

    marker.unlink()
    write(world / "plugins" / "_helper.py", "raise SystemExit('edited after it was trusted')\n")
    WorldStore(world).load()
    assert "nice" not in REGISTRY.of("part") and not marker.exists()
    assert any("_helper.py" in n for n in REGISTRY.world_notes(world))


def test_a_file_added_to_a_trusted_folder_untrusts_it_and_a_deleted_one_does_not(home):
    world = downloaded_world(home, home / "m")
    REGISTRY.trust(world)
    write(world / "plugins" / "extra.py", "x = 1\n")
    WorldStore(world).load()
    assert "nice" not in REGISTRY.of("part") and "dl/plugins/extra.py" in REGISTRY.problems()
    (world / "plugins" / "extra.py").unlink()
    WorldStore(world).load()
    assert "nice" in REGISTRY.of("part")


def test_a_package_in_a_subfolder_and_a_link_to_a_folder(home):
    world = home / "w"
    WorldStore.create("w", parent=home)
    write(
        world / "plugins" / "p.py",
        "from pkg import value\nfrom robotforge.extend import part, Box\n"
        "@part('pp', size='[a]', default_size=[1])\ndef f(size, rgba):\n    return [Box([value, 1, 1])]\n",
    )
    write(world / "plugins" / "pkg" / "__init__.py", "value = 2\n")
    tree = registry.read_tree(world / "plugins")
    assert set(tree.files) == {"p.py", "pkg/__init__.py"}
    REGISTRY.trust(world)
    assert "pp" in REGISTRY.of("part")
    outside = home / "outside"
    write(outside / "evil.py", "x = 1\n")
    try:
        (world / "plugins" / "linked").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this machine cannot make symbolic links")
    assert "link to a folder" in registry.read_tree(world / "plugins").flaws[0]
    with pytest.raises(registry.PluginError, match="cannot be trusted"):
        REGISTRY.trust(world)


def test_helpers_are_read_again_and_do_not_leak_between_worlds(home):
    first = home / "first"
    WorldStore.create("first", parent=home)
    entry = (
        "import _shared\nfrom robotforge.extend import part, Box\n"
        "@part({n!r}, size='[a]', default_size=[1])\n"
        "def f(size, rgba):\n    return [Box([_shared.SIZE, 1, 1])]\n"
    )
    write(first / "plugins" / "a.py", entry.format(n="from_first"))
    write(first / "plugins" / "_shared.py", "SIZE = 1\n")
    REGISTRY.trust(first)
    assert sys.modules["_shared"].SIZE == 1
    write(first / "plugins" / "_shared.py", "SIZE = 2\n")  # an edit that touches only the helper
    REGISTRY.trust(first)
    assert sys.modules["_shared"].SIZE == 2
    second = home / "second"
    WorldStore.create("second", parent=home)
    write(second / "plugins" / "b.py", entry.format(n="from_second"))
    write(second / "plugins" / "_shared.py", "SIZE = 3\n")
    REGISTRY.trust(second)
    assert sys.modules["_shared"].SIZE == 3 and "from_first" not in REGISTRY.of("part")
    REGISTRY.reload()
    REGISTRY._forget_folder((second / "plugins").resolve())
    assert "_shared" not in sys.modules


def test_a_dataclass_and_pickle_work_in_a_plugin(home):
    user = home / "user_plugins"
    write(
        user / "state.py",
        "from __future__ import annotations\nimport pickle\nfrom dataclasses import dataclass\n"
        "from robotforge.extend import controller\n\n"
        "@dataclass\nclass Tally:\n    n: int = 0\n\n"
        "assert pickle.loads(pickle.dumps(Tally(3))).n == 3\n\n"
        "@controller('tally', rate_hz=50)\nclass Tallier:\n"
        "    def __init__(self, ctx):\n        self.state = Tally()\n\n"
        "    def update(self, ctx):\n        self.state.n += 1\n",
    )
    REGISTRY.reload()
    assert not REGISTRY.problems(), REGISTRY.problems()
    assert "tally" in REGISTRY.of("controller")
    REGISTRY.reload()  # loading again after unloading works the same
    assert not REGISTRY.problems()
    (user / "state.py").unlink()
    REGISTRY.reload()
    assert not [m for m in sys.modules if m.startswith("robotforge_plugin_state")]


@pytest.mark.anyio
async def test_opening_a_copied_world_that_uses_a_plugin_shape_can_be_trusted_by_name_and_then_opened(home, capsys):
    marker = home / "helper_ran"
    world = downloaded_world(home, marker)
    store = WorldStore(world)
    with registry.isolated():  # the author's machine: the world is made with the plugin loaded
        REGISTRY.trust(world)
        from robotforge.design.world import World

        shape = World.model_validate({"objects": [{"name": "n", "shape": "nice"}]}).objects[0]
        store.apply(lambda w: w.objects.append(shape))
    marker.unlink()
    (home / "home" / registry.TRUST_FILE).unlink()  # this computer has never seen the world
    REGISTRY.reload()
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    with pytest.raises(PlaygroundError) as why:
        pg.open_world(str(world))
    assert "unknown shape 'nice'" in str(why.value) and "dl/plugins/_helper.py" in str(why.value)
    assert not marker.exists()

    server = build_server(pg, find_free_port(8871))
    async with Client(server) as client:
        shown = await client.call_tool("plugins", {"action": "show", "world": str(world)})  # no world is open yet
        assert not shown.is_error, shown.content[0].text
        files = json.loads(shown.content[0].text)["files"]
        assert set(files) == {"nice.py", "_helper.py", "colorsys.py"} and "import _helper" in files["nice.py"]
        refused = await client.call_tool("plugins", {"action": "trust", "world": str(world)})
        assert refused.is_error and not marker.exists()

    assert cli.main(["plugins", "trust", str(world)]) == 2  # no terminal to ask at, and no --yes
    assert "type yes at a terminal" in capsys.readouterr().err and not marker.exists()
    assert cli.main(["plugins", "show", str(world)]) == 0
    assert "SHADOW = True" in capsys.readouterr().out and not marker.exists()
    assert cli.main(["plugins", "trust", str(world), "--yes"]) == 0
    opened = pg.open_world(str(world))
    assert opened["objects"] == ["n"] and marker.read_text(encoding="utf-8") == "ran"


def test_the_terminal_asks_before_it_trusts(home, monkeypatch, capsys):
    world = downloaded_world(home, home / "m")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert cli.main(["plugins", "trust", str(world)]) == 1
    assert "not trusted" in capsys.readouterr().out and "nice" not in REGISTRY.of("part")
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert cli.main(["plugins", "trust", str(world)]) == 0
    assert "nice" in REGISTRY.of("part")


def test_a_script_written_here_waits_when_the_folder_holds_files_nobody_approved(home):
    world = downloaded_world(home, home / "m")
    saved = REGISTRY.written(world, write(world / "plugins" / "mine.py", "x = 1\n"))
    assert saved["loaded"] is False and "loads as one set" in saved["error"] and "_helper.py" in saved["error"]
    REGISTRY.trust(world)
    ok = REGISTRY.written(world, write(world / "plugins" / "more.py", "y = 2\n"))
    assert ok["loaded"] is True
