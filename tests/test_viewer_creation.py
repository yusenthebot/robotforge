"""Local starter placement validates input and commits against the actual current world."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest
from starlette.testclient import TestClient

from robotforge.design import world as world_module
from robotforge.design.world import RobotEntry, spawn_height
from robotforge.playground import Playground, PlaygroundError, viewer_catalog
from robotforge.sim.execution import ExecutionError, RunLimits
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub, create_app


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    result = Playground(session=SimSession.from_xml("<mujoco/>"))
    result.new_world("creation")
    return result


def place(pg, template="rover", pos=None, **extra):
    return pg.add_robot_for_play(
        {"template": template, "pos": [0.0, 0.0, 0.0] if pos is None else pos, "version": pg.session.version, **extra}
    )


@pytest.mark.parametrize("template", ["rover", "quadruped", "arm", "humanoid"])
@pytest.mark.parametrize("height", [0.0, 0.8])
def test_surface_height_preserves_each_templates_spawn_height(pg, template, height):
    expected = height + spawn_height(RobotEntry(name="test", template=template))
    out = place(pg, template, [0.0, 0.0, height])
    entry = pg.store.load().robots[0]
    assert entry.pos == ([0.0, 0.0] if height == 0 else [0.0, 0.0, expected])
    assert out == {"robot": entry.name, "version": pg.session.version, "notes": out["notes"], "undo": out["undo"]}
    assert pg.session.data.xpos[1, 2] == pytest.approx(expected)
    pg.session.step(20)
    assert np.all(np.isfinite(pg.session.data.qpos))


@pytest.mark.parametrize(
    "patch",
    [
        {"template": "go2"},
        {"template": "../rover"},
        {"file": "anything.xml"},
        {"params": {}},
        {"pos": [0]},
        {"pos": [0, 0, 0, 0]},
        {"pos": [float("nan"), 0]},
        {"pos": [float("inf"), 0]},
        {"pos": [101, 0]},
        {"pos": [True, 0]},
        {"pos": ["0", 0]},
        {"version": True},
        {"version": "2"},
        {"version": 0},
    ],
)
def test_invalid_input_never_changes_world_or_model(pg, patch):
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="invalid robot placement"):
        place(pg, **patch)
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_saved_and_live_occupied_spots_reject_without_resetting_others(pg):
    place(pg)
    with pg.session.lock:
        pg.session.data.qpos[0] = 2.0
        mujoco.mj_forward(pg.session.model, pg.session.data)
    before, live = pg.store.file.read_bytes(), pg.session.data.qpos.copy()
    for x in (0.0, 2.0):
        with pytest.raises(PlaygroundError, match="occupied by rover"):
            place(pg, "arm", [x, 0, 0])
        assert pg.store.file.read_bytes() == before
        np.testing.assert_array_equal(pg.session.data.qpos, live)
    place(pg, "arm", [4, 0, 0])
    assert pg.session.data.qpos[0] == 2.0


def test_movement_during_compile_is_rechecked_before_commit(pg, monkeypatch):
    place(pg)
    compile_world = world_module.compile_world

    def moving(*args, **kwargs):
        result = compile_world(*args, **kwargs)
        with pg.session.lock:
            pg.session.data.qpos[0] = 2.0
            mujoco.mj_forward(pg.session.model, pg.session.data)
        return result

    monkeypatch.setattr(world_module, "compile_world", moving)
    before = pg.store.file.read_bytes()
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        place(pg, "arm", [2, 0, 0])
    assert pg.store.file.read_bytes() == before and pg.session.data.qpos[0] == 2.0


@pytest.mark.parametrize("change", ["reset", "store", "lesson"])
def test_invalidation_during_preflight_does_not_commit(pg, monkeypatch, change):
    compile_world = world_module.compile_world
    store, before = pg.store, pg.store.file.read_bytes()

    def invalidating(*args, **kwargs):
        result = compile_world(*args, **kwargs)
        if change == "reset":
            pg.session.reset()
        elif change == "store":
            pg.store = None
        else:
            pg.lessons.start("pid_twins")
        return result

    monkeypatch.setattr(world_module, "compile_world", invalidating)
    with pytest.raises(PlaygroundError, match="simulation changed"):
        place(pg)
    assert store.file.read_bytes() == before


def test_plain_models_lessons_active_runs_and_stale_clients_are_refused(pg):
    with pytest.raises(PlaygroundError, match="scene changed"):
        place(pg, version=pg.session.version + 1)
    with pg.session.execution.run("test", RunLimits()), pytest.raises(ExecutionError):
        place(pg)
    pg.lessons.start("pid_twins")
    with pytest.raises(PlaygroundError, match="leave the lesson"):
        place(pg)
    pg.lessons.exit()
    pg.store = None
    with pytest.raises(PlaygroundError, match="no world"):
        place(pg)


def test_simultaneous_requests_never_overwrite_and_names_are_unique(pg):
    version = pg.session.version

    def request(x):
        try:
            return place(pg, "arm", [x, 0], version=version)
        except PlaygroundError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, [0.0, 2.0]))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert len(pg.store.load().robots) == 1
    out = place(pg, "arm", [4.0, 0])
    assert out["robot"] == "arm_2"


def test_catalog_and_real_websocket_dispatch(pg):
    assert [r["name"] for r in viewer_catalog()["robot_templates"]] == ["rover", "quadruped", "arm", "humanoid"]
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1:8737") as client,
        client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws,
    ):
        for ident, patch in [(1, {}), (2, {"file": "bad"})]:
            ws.send_text(
                json.dumps(
                    {
                        "id": ident,
                        "type": "world",
                        "action": "add_robot",
                        "template": "rover",
                        "pos": [ident * 2, 0, 0],
                        "version": pg.session.version,
                        **patch,
                    }
                )
            )
            for _ in range(400):
                msg = ws.receive()
                data = json.loads(msg["text"]) if msg.get("text") else {}
                if data.get("type") == "reply" and data["id"] == ident:
                    assert data["ok"] == (ident == 1)
                    if ident == 1:
                        assert data["data"]["robot"] == "rover"
                    break
            else:
                pytest.fail("missing reply")

    assert len(pg.store.load().robots) == 1


@pytest.mark.parametrize("template", ["rover", "quadruped", "arm", "humanoid"])
def test_actual_tabletop_support_is_not_treated_as_a_collision(pg, template):
    from robotforge.sim import placement

    pg.add_object("table", shape="table", size=[3, 3, 0.8], pos=[0, 0], fixed=True)
    out = place(pg, template, [0, 0, 0.8])
    assert not placement.inside(pg.session.model, pg.session.data, out["robot"])
    assert pg.store.load().robots[0].pos[2] >= 0.8


@pytest.mark.parametrize("feature", ["stairs", "ramp", "stones", "curb", "platform"])
def test_explicit_tabletop_position_also_accepts_feature_terrain(pg, feature):
    out = place(pg, "arm", [0, 0, 0.8])
    assert pg.terrain_for_play(feature, out["robot"])["terrain"] == [f"{feature}1"]


def test_ground_placement_follows_new_terrain_and_recognizes_its_surface(pg):
    from robotforge.design import terrain
    from robotforge.sim.session import fresh_data

    out = place(pg)
    assert len(pg.store.load().robots[0].pos) == 2
    pg.terrain_for_play("hills", out["robot"])
    world = pg.store.load()
    # A normal ground entry and a viewer-created entry use the same terrain settling path.
    normal = world_module.compile_world(world, pg.store.dir)
    np.testing.assert_array_equal(pg.session.data.qpos, fresh_data(normal.model).qpos)
    x, y = 3.0, 0.0
    height = terrain.Ground.build(world.terrain, world.floor.style).height(x, y)
    place(pg, "arm", [x, y, round(height, 4)])
    assert pg.store.load().robots[-1].pos == [x, y]
