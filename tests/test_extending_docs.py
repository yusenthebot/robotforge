"""docs/EXTENDING.md: every example in it is run here, so the page cannot drift from the code."""

import json
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.extend import REGISTRY, registry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "EXTENDING.md"
FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
FENCE = re.compile(r'```python title="(?P<title>[^"]+)"(?: robots="(?P<robots>[^"]+)")?\n(?P<code>.*?)```', re.S)
ROBOTS = {
    "dog": {"name": "dog", "template": "quadruped"},
    "go2": {"name": "go2", "library": "go2"},
    "arm": {"name": "arm", "template": "arm"},
    "panda": {"name": "panda", "library": "panda"},
}


def examples() -> dict[str, tuple[str, list[str]]]:
    text = DOC.read_text(encoding="utf-8")
    return {
        m["title"]: (m["code"], (m["robots"] or "").split(",") if m["robots"] else []) for m in FENCE.finditer(text)
    }


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def plugins_loaded(tmp_path, monkeypatch):
    folder = tmp_path / "user_plugins"
    folder.mkdir()
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(folder))
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    with registry.isolated():
        REGISTRY.user_loaded = False
        for title, (code, _) in examples().items():
            if title.startswith("plugins/"):
                (folder / Path(title).name).write_text(code, encoding="utf-8", newline="\n")
        REGISTRY.reload()
        yield folder


def world(**robot) -> Playground:
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot(**robot)
    return pg


def test_the_page_has_all_the_examples_it_promises():
    titles = set(examples())
    assert {"plugins/signpost.py", "plugins/geiger.py", "plugins/sweep.py", "plugins/square.py"} <= titles
    assert {"behaviors/tour.py", "behaviors/reach.py", "behaviors/patrol.py", "check_export.py"} <= titles


def test_the_page_follows_the_house_style():
    text = DOC.read_text(encoding="utf-8")
    assert chr(0x2014) not in text and chr(0x2013) not in text  # no em or en dashes
    assert all(ord(c) < 0x2190 for c in text)  # no emoji, no pictographs


def test_every_plugin_example_loads(plugins_loaded):
    assert not REGISTRY.problems(), REGISTRY.problems()
    assert set(REGISTRY.of("part")) == {"signpost"} and set(REGISTRY.of("sensor")) == {"geiger"}
    assert set(REGISTRY.of("controller")) == {"sweep"} and set(REGISTRY.of("behavior")) == {"square"}


def test_the_part_example_places_and_stands(plugins_loaded):
    pg = world(name="arm", template="arm")
    pg.add_object("sign", shape="signpost", pos=[1.0, 0.5])
    pg.run_fast(lambda: False, 0.5)
    sign = pg.session.model
    assert mujoco.mj_name2id(sign, mujoco.mjtObj.mjOBJ_GEOM, "sign") >= 0
    assert "signpost" in pg.store.file.read_text(encoding="utf-8")


def test_the_sensor_example_reads_the_distance_to_its_source(plugins_loaded):
    pg = world(name="dog", template="quadruped")
    pg.add_object("hot", shape="box", size=[0.1] * 3, pos=[1.0, 0.0], color="red")
    pg.add_sensor("dog", {"type": "geiger", "radius": 3})
    pg.run_fast(lambda: False, 0.3)
    got = pg.read_sensors("dog", ["geiger"])[0]["sensors"]["geiger"]
    assert 0 < got["level"] < 1 and 0.8 < got["distance_m"] < 1.4 and "error" not in got


def test_the_controller_example_swings_the_joint(plugins_loaded):
    from robotforge.extend import routines

    pg = world(name="arm", template="arm")
    routines.start_controller(pg.session, "arm", "sweep", {"amplitude": 0.5})
    seen = []
    for _ in range(30):
        pg.run_fast(lambda: False, 0.1)
        seen.append(pg.state()["joints"]["arm/shoulder"])
    assert max(seen) > 0.3 and min(seen) < -0.3
    assert pg.session.controllers["arm"].failure is None


@pytest.mark.anyio
@pytest.mark.parametrize("robot", ["dog", "go2"])
async def test_the_behavior_example_runs_on_a_dog_and_a_go2(plugins_loaded, robot):
    pg = world(**ROBOTS[robot])
    server = build_server(pg, find_free_port(8861))
    async with Client(server) as client:
        result = await client.call_tool(
            "plugins", {"action": "run", "name": "square", "robot": robot, "options": {"side_s": 1.0}, "seconds": 40}
        )
        assert not result.is_error, result.content[0].text
        out = json.loads(result.content[0].text)
    assert out["ok"], out
    assert out["result"]["kind"] == "quadruped" and out["result"]["gap_m"] < 1.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("title", "robot"),
    [(t, r) for t, (_, robots) in examples().items() for r in robots],
)
async def test_the_script_examples_run_on_each_robot_they_name(plugins_loaded, tmp_path, monkeypatch, title, robot):
    pg = world(**ROBOTS[robot])
    server = build_server(pg, find_free_port(8862))
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "example", "code": examples()[title][0]})
        result = await client.call_tool("run_script", {"name": "example", "max_sim_seconds": 60})
        out = json.loads(result.content[0].text)
    assert out["ok"], out["output"]
    assert out["output"].strip()


@pytest.mark.anyio
async def test_the_export_example_opens_the_urdf_it_wrote(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = world(**ROBOTS["go2"])
    server = build_server(pg, find_free_port(8863))
    async with Client(server) as client:
        result = await client.call_tool("export_model", {"robot": "go2", "format": "urdf"})
        assert json.loads(result.content[0].text)["check"]["same"]
    script = tmp_path / "check_export.py"
    script.write_text(examples()["check_export.py"][0], encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(script)], cwd=pg.store.dir, capture_output=True, text=True, encoding="utf-8", timeout=60
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "bodies 14 joints 12"
