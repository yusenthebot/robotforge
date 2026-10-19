"""The viewer's World panel: control modes, maps and terrain, and the hub's request/reply path."""

from __future__ import annotations

import json

import numpy as np
import pytest
from starlette.testclient import TestClient

from robotforge import modes
from robotforge.playground import Playground, viewer_catalog
from robotforge.viewer.server import ViewerHub, create_app


@pytest.fixture
def playground(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("modes")
    pg.add_robot("dog", template="quadruped")
    pg.add_robot("arm", template="arm", pos=[1.5, 0.0])
    pg.add_robot("rover", template="rover", pos=[0.0, 1.5])
    return pg


def _available(pg: Playground) -> dict[str, list[str]]:
    learned = pg.behaviors.learned_walkers()
    return {r["name"]: modes.available(pg, r, r["name"] in learned) for r in pg.behaviors.robots()}


def test_each_robot_gets_the_modes_that_fit_it(playground):
    offered = _available(playground)
    assert offered["dog"] == ["drive", "hold", "stand_mpc", "walk_mppi"]
    assert offered["arm"] == ["manual", "reach_ik", "reach_cartesian", "reach_mpc"]
    assert offered["rover"] == ["drive", "goto_mpc"]  # the stand task alone would accept a rover


def test_a_gripper_on_its_own_gets_sliders_and_does_not_break_the_scene(playground):
    """A gripper template with no arm under it is told as "gripper", which is no behavior kind: the viewer's scene
    message (which lists every robot's modes) failed, so the page never drew."""
    playground.add_robot("hand", template="gripper", pos=[0.0, -1.0])
    assert _available(playground)["hand"] == ["manual"]


def test_modes_attach_controllers_and_report_themselves(playground):
    session = playground.session
    out = modes.apply(playground, "dog", "stand_mpc")
    assert out == {"robot": "dog", "mode": "stand_mpc", "follows": False}
    assert modes.current(playground)["dog"] == "stand_mpc"
    session.step(100)  # the controller runs on the session's steps
    assert modes.apply(playground, "dog", "drive")["mode"] == "drive"
    assert modes.current(playground)["dog"] == "drive"
    modes.apply(playground, "arm", "reach_ik")
    assert modes.current(playground)["arm"] == "reach_ik"
    modes.apply(playground, "arm", "manual")
    assert "arm" not in modes.current(playground) and "arm" not in session.controllers


def test_reaching_aims_somewhere_new_each_time(playground):
    first = modes.apply(playground, "arm", "reach_ik")["marker"]
    second = modes.apply(playground, "arm", "reach_ik")["marker"]
    assert len(first) == 3 and first != second
    goal = modes.apply(playground, "rover", "goto_mpc")
    assert len(goal["marker"]) == 2 and goal["follows"] is True


def test_bad_modes_are_explained(playground):
    with pytest.raises(modes.ModeError, match="unknown mode"):
        modes.apply(playground, "dog", "fly")
    with pytest.raises(modes.ModeError, match="no robot named"):
        modes.apply(playground, "cat", "drive")


def test_features_go_in_front_of_the_robot_and_maps_clear(playground):
    result = playground.terrain_for_play("stairs", "rover")
    patch = playground.store.load().terrain[0]
    assert result["terrain"] == ["stairs1"]
    assert patch.pos[0] - patch.footprint()[0] / 2 == pytest.approx(0.6)  # starts 0.6 m ahead of the rover
    assert patch.pos[1] == 1.5 and patch.yaw == 0.0  # straight ahead: the rover faces +x
    assert playground.terrain_for_play("hills")["terrain"] == ["hills1"]
    playground.load_scene("park")
    cleared = playground.clear_scene()
    world = playground.store.load()
    assert cleared["robots"] == ["dog", "arm", "rover"]
    assert world.terrain == [] and world.objects == [] and world.floor.style == "checker"


def test_catalog_lists_maps_grounds_and_terrain():
    cat = viewer_catalog()
    assert {s["name"] for s in cat["scenes"]} >= {"park", "maze", "kitchen"}
    assert cat["grounds"]["grass"] == [0.33, 0.52, 0.27]
    assert cat["terrains"][0] == "flat" and "stairs" in cat["terrains"]


def _reply(ws, request_id):
    for _ in range(400):
        msg = ws.receive()
        if msg.get("text"):
            data = json.loads(msg["text"])
            if data["type"] == "reply" and data["id"] == request_id:
                return data
    raise AssertionError(f"no reply to request {request_id}")


def test_viewer_requests_change_the_world_and_get_answers(playground):
    hub = ViewerHub(playground.session)
    playground.attach_viewer(hub)
    scene = hub.scene_message()
    assert scene["world"] == "modes" and scene["floor"] == "checker"
    assert scene["modes"]["arm"][0] == "manual" and scene["catalog"]["terrains"][0] == "flat"
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1:8737") as client,
        client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws,
    ):
        ws.send_text(json.dumps({"type": "world", "action": "floor", "style": "grass", "id": 1}))
        assert _reply(ws, 1)["ok"] is True
        ws.send_text(json.dumps({"type": "world", "action": "floor", "style": "lava", "id": 2}))
        bad = _reply(ws, 2)
        assert bad["ok"] is False and "style" in bad["message"]
        ws.send_text(json.dumps({"type": "mode", "robot": "dog", "mode": "hold", "id": 3}))
        assert _reply(ws, 3)["data"]["mode"] == "hold"
    assert playground.store.load().floor.style == "grass"
    assert hub.state_message()["mode"]["dog"] == "hold"


def test_mppi_walking_stops_and_stands_after_a_while(playground):
    out = modes.apply(playground, "dog", "walk_mppi")
    assert out["follows"] is True
    session = playground.session
    trunk = session.model.body("dog/trunk").id
    start = session.data.xpos[trunk][:2].copy()
    session.step(round(4.0 / session.model.opt.timestep))
    assert modes.current(playground)["dog"] == "walk_mppi"
    session.step(round(5.0 / session.model.opt.timestep))  # past the 8 s walk
    assert modes.current(playground)["dog"] == "stand_mpc"
    stopped = session.data.xpos[trunk][:2].copy()
    session.step(round(2.0 / session.model.opt.timestep))
    assert np.linalg.norm(stopped - start) > 0.5  # it walked
    assert np.linalg.norm(session.data.xpos[trunk][:2] - stopped) < 0.15  # and now stands its ground
    assert session.data.xmat[trunk][8] > 0.9  # upright
