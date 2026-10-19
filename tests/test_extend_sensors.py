"""Plugin sensors: a type a user defines in Python, used like the built-in ones."""

import json
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design import components
from robotforge.extend import REGISTRY, registry
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sensors import spec
from robotforge.sim.session import SimSession

GEIGER = """
from robotforge.extend import sensor, look

@sensor(
    "geiger",
    summary="how close the hot source is: 1 on top of it, 0 at the radius",
    channels=("level",),
    params={"source": "hot", "radius": 2.0},
    at="top",
    appearance=look("cylinder", [0.05, 0.03], "yellow"),
)
def geiger(ctx):
    distance = ctx.distance(ctx.params["source"])
    return {"level": max(0.0, 1.0 - distance / ctx.params["radius"]), "distance_m": distance}
"""

COUNTER = """
from robotforge.extend import sensor

@sensor("bumps", summary="counts the readings it has made", channels=("count",))
class Bumps:
    def __init__(self, ctx):
        self.count = 0

    def read(self, ctx):
        self.count += 1
        return {"count": self.count}
"""

WHISKER = """
from robotforge.extend import sensor

@sensor("whisker", summary="a ray ahead", params={"reach": 3.0})
def whisker(ctx):
    distance, geom = ctx.ray((1, 0, 0), ctx.params["reach"])
    return {"touching": distance is not None, "distance_m": distance, "what": geom}
"""

BROKEN = """
from robotforge.extend import sensor

@sensor("flaky")
def flaky(ctx):
    raise RuntimeError("the sensor is broken")
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
    playground.add_robot("dog", template="quadruped")
    playground.add_object("hot", shape="box", size=[0.1] * 3, pos=[1.0, 0.0], color="red")
    return playground


def plugin(folder: Path, name: str, code: str) -> None:
    (folder / name).write_text(code, encoding="utf-8", newline="\n")
    REGISTRY.reload()
    assert not REGISTRY.problems(), REGISTRY.problems()


def test_a_plugin_sensor_is_a_type_like_the_built_in_ones(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    assert "geiger" in spec.SENSOR_TYPES and spec.SENSOR_TYPES[:3] == ["camera", "lidar", "lidar3d"]
    added = pg.add_sensor("dog", {"type": "geiger", "source": "hot", "radius": 3.0})
    assert added["sensor"] == "dog/geiger" and added["reading"]["type"] == "geiger"
    pg.run_fast(lambda: False, 0.3)
    reading, _ = pg.read_sensors("dog", ["geiger"])
    got = reading["sensors"]["geiger"]
    robot = pg.session.data.xpos[1]
    expected = float(np.linalg.norm(pg.session.data.xpos[pg.behaviors._body("hot")] - robot))
    assert got["distance_m"] == pytest.approx(expected, abs=0.35)  # from the sensor on its back, not the base
    assert 0 < got["level"] < 1
    assert "geiger" in pg.sensors.lines()["dog/geiger"]


def test_its_options_are_saved_with_the_robot_and_come_back(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    pg.add_sensor("dog", {"type": "geiger", "source": "hot", "radius": 3.0, "rate_hz": 20, "noise": 0.01})
    text = pg.store.file.read_text(encoding="utf-8")
    assert "type: geiger" in text and "radius: 3.0" in text and "noise: 0.01" in text
    pg.open_world(str(pg.store.dir))
    got = pg.read_sensors("dog", ["geiger"])[0]["sensors"]["geiger"]
    assert got["type"] == "geiger" and "level" in got


def test_an_option_it_does_not_have_is_refused_with_the_ones_it_has(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    with pytest.raises(PlaygroundError, match=r"takes no option 'sorce'.*source, radius"):
        pg.add_sensor("dog", {"type": "geiger", "sorce": "hot"})
    with pytest.raises(PlaygroundError, match="takes no 'bias'"):
        pg.add_sensor("dog", {"type": "geiger", "bias": 1.0})


def test_an_unknown_type_names_the_types_there_are(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    with pytest.raises(PlaygroundError, match=r"unknown sensor type 'gieger'.*from plugins: geiger"):
        pg.add_sensor("dog", {"type": "gieger"})


def test_noise_and_rate_work_like_any_sensors(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    pg.add_sensor("dog", {"type": "geiger", "name": "clean", "source": "hot"})
    pg.add_sensor("dog", {"type": "geiger", "name": "noisy", "source": "hot", "noise": 0.2, "rate_hz": 50})
    pg.run_fast(lambda: False, 1.0)
    clean = pg.read_sensors("dog", ["clean"])[0]["sensors"]["clean"]["level"]
    seen = set()
    for _ in range(6):
        pg.run_fast(lambda: False, 0.1)
        seen.add(pg.read_sensors("dog", ["noisy"])[0]["sensors"]["noisy"]["level"])
    assert len(seen) > 3 and max(abs(v - clean) for v in seen) > 0.01  # it errs, and differently each time
    assert max(abs(v - clean) for v in seen) < 1.5


def test_a_sensor_class_keeps_what_it_has_seen(pg, folder):
    plugin(folder, "bumps.py", COUNTER)
    pg.add_sensor("dog", {"type": "bumps", "rate_hz": 10})
    pg.run_fast(lambda: False, 1.0)
    first = pg.read_sensors("dog", ["bumps"])[0]["sensors"]["bumps"]["count"]
    pg.run_fast(lambda: False, 1.0)
    later = pg.read_sensors("dog", ["bumps"])[0]["sensors"]["bumps"]["count"]
    assert later > first >= 5  # one reading per sample at 10 Hz, kept between them


def test_a_ray_sees_what_is_ahead(pg, folder):
    plugin(folder, "whisker.py", WHISKER)
    pg.add_object("wall", shape="wall", pos=[1.5, 0.0], yaw=90, size=[2.0, 0.1, 1.0])
    pg.add_sensor("dog", {"type": "whisker", "at": "front", "reach": 4.0})
    pg.run_fast(lambda: False, 0.3)
    got = pg.read_sensors("dog", ["whisker"])[0]["sensors"]["whisker"]
    assert got["touching"] is True and got["what"] == "wall" and 1.0 < got["distance_m"] < 1.5


def test_a_sensor_that_raises_reads_as_an_error_and_the_world_goes_on(pg, folder):
    plugin(folder, "flaky.py", BROKEN)
    pg.add_sensor("dog", {"type": "flaky"})
    pg.run_fast(lambda: False, 0.2)
    got = pg.read_sensors("dog", ["flaky"])[0]["sensors"]["flaky"]
    assert got["error"] == "RuntimeError: the sensor is broken"
    assert pg.session.data.time > 0.15  # time kept passing


def test_the_viewer_gets_a_line_and_a_name_for_it(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    pg.add_sensor("dog", {"type": "geiger", "source": "hot"})
    entry = next(e for e in pg.sensors.viewer_message()["sensors"] if e["type"] == "geiger")
    assert entry["label"] == "Geiger" and entry["line"].startswith("level ") and "geiger" not in entry["line"]
    assert entry["added"] is True and len(entry["pos"]) == 3


def test_the_module_is_drawn_on_the_robot_and_weighs_nothing(pg, folder):
    before = pg.session.model.body_mass.sum()
    plugin(folder, "geiger.py", GEIGER)
    pg.add_sensor("dog", {"type": "geiger", "source": "hot"})
    model = pg.session.model
    assert model.body_mass.sum() == pytest.approx(before, rel=1e-9)
    yellow = [g for g in range(model.ngeom) if np.allclose(model.geom_rgba[g], components.color("yellow"))]
    assert yellow and all(model.geom_contype[g] == 0 and model.geom_group[g] == 2 for g in yellow)


def test_remove_takes_it_off_and_unloading_the_plugin_leaves_a_clear_message(pg, folder):
    path = folder / "geiger.py"
    plugin(folder, "geiger.py", GEIGER)
    pg.add_sensor("dog", {"type": "geiger", "source": "hot"})
    path.unlink()
    REGISTRY.reload()
    assert "geiger" not in spec.SENSOR_TYPES
    with pytest.raises(PlaygroundError, match="unknown sensor type"):
        pg.open_world(str(pg.store.dir))


def test_a_name_that_is_a_built_in_type_is_refused(folder):
    (folder / "bad.py").write_text(GEIGER.replace('"geiger"', '"lidar"'), encoding="utf-8")
    REGISTRY.reload()
    assert "lidar" not in REGISTRY.of("sensor")
    assert "already has a sensor type" in " ".join(REGISTRY.problems().values())


@pytest.mark.anyio
async def test_an_agent_adds_and_reads_a_plugin_sensor(pg, folder):
    plugin(folder, "geiger.py", GEIGER)
    server = build_server(pg, find_free_port(8821))
    async with Client(server) as client:
        added = await client.call_tool(
            "add_sensor", {"robot": "dog", "type": "geiger", "options": {"source": "hot", "radius": 4}}
        )
        assert not added.is_error, added.content[0].text
        read = await client.call_tool("read_sensors", {"robot": "dog", "names": ["geiger"]})
        text = json.loads(read.content[0].text)
        assert text["sensors"]["geiger"]["type"] == "geiger"
        listing = json.loads((await client.call_tool("plugins")).content[0].text)
        assert listing["sensors"][0]["options"] == {"source": "hot", "radius": 2.0}
