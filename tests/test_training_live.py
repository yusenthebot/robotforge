"""The viewer's Training panel: the stream of the run that changed last, and its play and stop buttons."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from starlette.testclient import TestClient

from robotforge.control import TaskError
from robotforge.control.rl import store
from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.training.envs import env_spec
from robotforge.training.live import TrainingFeed, install
from robotforge.training.runs import make_run_spec, run_inline
from robotforge.viewer.server import ViewerHub, create_app


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    return pg


def _train(pg, name="pole_ars", steps=6_000):
    world = pg.current_world()
    spec = env_spec("balance", world=world, robot="pole", world_dir=pg.store.dir).model_dump(mode="json")
    folder = store.policies_home(pg.store.dir) / name
    run_inline(make_run_spec(name, spec, algos=["ars", "cem"], steps=steps, parallel=0), folder)
    return folder


def test_no_run_no_message(lab):
    assert TrainingFeed(lab).message() is None


def test_the_message_carries_the_trials_and_is_sent_again_only_when_it_changes(lab):
    folder = _train(lab)
    feed = TrainingFeed(lab)
    first = feed.message()
    assert first["type"] == "training" and first["name"] == folder.name and first["state"] == "done"
    assert [t["algo"] for t in first["trials"]] == ["ars", "cem"] and all(t["playable"] for t in first["trials"])
    assert first["playable"] and first["leader"] in {"t01", "t02"}
    json.dumps(first)
    assert feed.message() is None  # nothing changed
    assert feed.message(force=True)["name"] == folder.name  # a viewer that just connected asks again


def test_a_run_of_another_age_is_not_offered(lab, monkeypatch):
    folder = _train(lab)
    old = time.time() - 13 * 3600
    import os

    os.utime(folder / store.STATUS_FILE, (old, old))
    assert TrainingFeed(lab).message() is None


def test_play_runs_the_best_policy_on_its_robot_and_a_trial_on_request(lab):
    folder = _train(lab)
    feed = TrainingFeed(lab)
    lab.session.paused = True
    out = feed.play({"run": folder.name})
    assert out["robot"] == "pole" and out["trial"] in {"t01", "t02"}
    assert lab.session.paused is False
    assert lab.session.controllers["pole"].status()["controller"] == "policy"
    lab.session.step(50)
    assert np.isfinite(lab.session.data.qpos).all()
    assert feed.play({"run": folder.name, "trial": "t02"})["trial"] == "t02"
    with pytest.raises(TaskError, match="no policy yet"):
        feed.play({"run": folder.name, "trial": "t99"})


def test_a_run_still_going_plays_the_trial_that_leads(lab):
    folder = _train(lab)
    (folder / "policy.npz").unlink()  # the run has no best policy yet: its trials do
    feed = TrainingFeed(lab)
    lead = json.loads((folder / "status.json").read_text(encoding="utf-8"))["leader"]
    assert feed.play({"run": folder.name})["trial"] == lead


def test_play_in_a_world_without_the_robot_says_which_world(lab):
    folder = _train(lab)
    lab.remove("pole")
    with pytest.raises(TaskError, match="not in this world"):
        TrainingFeed(lab).play({"run": folder.name})


def test_the_viewer_stream_sends_the_run_to_a_connecting_page_and_plays_on_request(lab):
    folder = _train(lab)
    hub = ViewerHub(lab.session)
    install(lab, hub)
    client = TestClient(create_app(hub), base_url="http://127.0.0.1:8737")
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        seen = {}
        for _ in range(200):
            msg = ws.receive()
            if msg.get("text"):
                data = json.loads(msg["text"])
                seen[data["type"]] = data
                if "training" in seen:
                    break
        assert seen["training"]["name"] == folder.name and len(seen["training"]["trials"]) == 2
        ws.send_text(json.dumps({"type": "training_play", "id": 7, "run": folder.name}))
        for _ in range(400):
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"]).get("type") == "reply":
                reply = json.loads(msg["text"])
                break
        assert reply["id"] == 7 and reply["ok"] is True and reply["data"]["robot"] == "pole"
