"""The agent's side of training: train_policy with several algorithms, a search and sensors, and training_status."""

from __future__ import annotations

import json

import pytest
from mcp import Client

from robotforge.control.rl import JOBS
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def no_leftover_training():
    yield
    for folder, proc in list(JOBS.procs.items()):
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        JOBS.procs.pop(folder, None)


def _call(client, name, args):
    return client.call_tool(name, args)


@pytest.mark.anyio
async def test_train_policy_compares_algorithms_on_a_robot_with_sensors_and_plays_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    pg.add_sensor("pole", {"type": "encoders"})
    async with Client(build_server(pg, 0)) as client:
        listed = json.loads((await _call(client, "list_algorithms", {})).content[0].text)
        names = {a["name"] for a in listed["training_algorithms"]["algorithms"]}
        assert {"ars", "cem", "ppo", "sac"} <= names

        started = await _call(
            client,
            "train_policy",
            {"task": "balance", "robot": "pole", "algo": ["ars", "cem"], "timesteps": 3000, "parallel": 2},
        )
        assert not started.is_error, started.content[0].text
        info = json.loads(started.content[0].text)
        assert [t["algo"] for t in info["trials"]] == ["ars", "cem"]
        assert "encoders" in [s["source"] for s in info["observation"]["from"]]  # the sensor joined the observation

        out = await _call(client, "training_status", {"name": info["started"], "wait_seconds": 180})
        status = json.loads(out.content[0].text)
        assert status["state"] == "done", status
        assert {t["algo"] for t in status["trials"]} == {"ars", "cem"} and status["leader"] in {"t01", "t02"}
        assert all("test_return" in t and t["settings"] for t in status["trials"])

        ran = await _call(
            client,
            "run_controller",
            {"robot": "pole", "task": "balance", "method": f"policy:{info['started']}", "seconds": 1},
        )
        assert not ran.is_error, ran.content[0].text
        assert json.loads(ran.content[0].text)["controller"]["controller"] == "policy"


@pytest.mark.anyio
async def test_a_search_and_its_mistakes_are_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("pole", template="cartpole")
    async with Client(build_server(pg, 0)) as client:
        bad = await _call(client, "train_policy", {"task": "balance", "robot": "pole", "algo": "arz"})
        assert bad.is_error and "ars" in bad.content[0].text
        bad = await _call(
            client,
            "train_policy",
            {"task": "balance", "robot": "pole", "algo": "ars", "search": {"space": {"banana": [1, 2]}}},
        )
        assert bad.is_error and "banana" in bad.content[0].text
        bad = await _call(
            client, "train_policy", {"task": "balance", "robot": "pole", "algo": "ars", "search": {"trials": 99}}
        )
        assert bad.is_error and "trials" in bad.content[0].text
        bad = await _call(
            client,
            "train_policy",
            {"task": "balance", "robot": "pole", "algo": "ars", "settings": {"ars": {"noise": 9}}},
        )
        assert bad.is_error and "noise" in bad.content[0].text
        bad = await _call(client, "train_policy", {"task": "balance", "robot": "ghost", "algo": "ars"})
        assert bad.is_error and "ghost" in bad.content[0].text


@pytest.mark.anyio
async def test_a_reach_target_named_after_an_object_trains_on_the_point_it_stands_at(tmp_path, monkeypatch):
    # scope 'robot' trains the arm alone, so an object named as the target used to be "not found" and the agent
    # had no way to fix it; the name now turns into the object's point, and the response says so
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab")
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04], pos=[0.25, 0.1, 0.2], fixed=True)
    async with Client(build_server(pg, 0)) as client:
        started = await _call(
            client,
            "train_policy",
            {
                "task": "reach",
                "robot": "arm",
                "algo": "ars",
                "timesteps": 400,
                "task_params": {"target": "cube", "jitter": 0.02},
                "parallel": 1,
            },
        )
        assert not started.is_error, started.content[0].text
        info = json.loads(started.content[0].text)
        assert any("'cube'" in n and "0.25" in n for n in info["notes"]), info["notes"]
        await _call(client, "stop_training", {"name": info["started"]})
