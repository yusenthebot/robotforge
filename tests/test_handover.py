"""Handing an object from one arm to another by way of a spot both reach (behaviors/relay.py), two Pandas offline."""

from __future__ import annotations

import json

import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import relay
from robotforge.behaviors.manager import BehaviorError
from robotforge.mcp_server import build_server
from robotforge.playground import Playground


@pytest.fixture
def anyio_backend():
    return "asyncio"


def two_arms(tmp_path, monkeypatch, apart: float = 0.55, cube=(-0.4, 0.2)) -> Playground:
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("handover", "wood")
    pg.add_robot("left", library="panda", pos=[-apart, 0.0], yaw=0)
    pg.add_robot("right", library="panda", pos=[apart, 0.0], yaw=180)
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], pos=list(cube), color="red", mass=0.1)
    pg.add_object("pad", shape="box", size=[0.2, 0.2, 0.01], pos=[apart * 0.7, 0.25], color="blue", fixed=True)
    pg.run_fast(lambda: False, 1.0)
    return pg


def test_one_arm_hands_the_cube_to_the_other_which_puts_it_on_the_pad(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch)
    handed, info = pg.behaviors.hand_over("left", "right", "cube", "pad")
    spot = np.array(info["spot"])
    assert np.linalg.norm(spot) < 0.1  # between them: neither could reach the other's side
    seen_at_spot = False
    t0 = pg.session.data.time
    while not handed.done and pg.session.data.time - t0 < handed.time_limit:
        pg.run_fast(lambda: False, 0.25)
        seen_at_spot |= handed.after is not None and bool(
            np.linalg.norm(pg.session.data.xpos[pg.session.model.body("cube").id][:2] - spot) < relay.ARRIVED
        )
    pg.run_fast(lambda: False, 0.5)
    assert seen_at_spot, "the cube was never set down at the spot before the second arm took it"
    assert pg.behaviors.placed("cube", info["region"])["success"]
    assert handed.result()["handed_over"]["from"] == "left" and handed.result()["handed_over"]["to"] == "right"
    assert handed.note is None


def test_the_giving_arm_waits_clear_of_the_receiving_one(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch)
    handed, _ = pg.behaviors.hand_over("left", "right", "cube", "pad")
    model, data = pg.session.model, pg.session.data
    t0 = data.time
    touched = 0.0
    while not handed.done and data.time - t0 < handed.time_limit:
        pg.run_fast(lambda: False, 0.1)
        for i in range(data.ncon):
            con = data.contact[i]
            names = {model.body(model.geom_bodyid[con.geom1]).name, model.body(model.geom_bodyid[con.geom2]).name}
            if any(n.startswith("left/") for n in names) and any(n.startswith("right/") for n in names):
                touched = max(touched, -float(con.dist))
    assert touched < 0.005, f"the two arms were pressed {touched * 1000:.0f} mm into each other"


def test_a_cube_the_giver_cannot_reach_is_refused_before_anything_moves(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch, cube=(0.9, 0.0))
    with pytest.raises(BehaviorError, match="left cannot reach"):
        pg.behaviors.hand_over("left", "right", "cube", "pad")
    assert not pg.session.controllers["left"].queue


def test_arms_too_far_apart_have_no_spot(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch, apart=1.4, cube=(-1.2, 0.0))
    with pytest.raises(BehaviorError, match="no spot"):
        pg.behaviors.hand_over("left", "right", "cube", "pad")


def test_an_arm_cannot_hand_to_itself(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch)
    with pytest.raises(BehaviorError, match="two different arms"):
        pg.behaviors.hand_over("left", "left", "cube", "pad")


def test_spots_lie_between_the_arms_and_keep_off_other_things(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch)
    give, take = pg.behaviors._arm_motion("left"), pg.behaviors._arm_motion("right")
    spots = relay.hand_off_spots(pg.session.data, give, take, [])
    assert spots and all(abs(s[0]) < 0.25 for s in spots)
    near = relay.hand_off_spots(pg.session.data, give, take, [np.array([0.0, 0.0, 0.0])])
    assert all(np.linalg.norm(s) >= relay.CLEAR for s in near)


@pytest.mark.anyio
async def test_pick_and_place_hand_to_runs_it_through_the_server(tmp_path, monkeypatch):
    pg = two_arms(tmp_path, monkeypatch)
    async with Client(build_server(pg, 0)) as client:
        reply = await client.call_tool(
            "pick_and_place", {"robot": "left", "object": "cube", "target": "pad", "hand_to": "right"}
        )
    out = json.loads("".join(c.text for c in reply.content if getattr(c, "type", "") == "text"))
    assert out["success"] is True, out
    assert out["handed_over"]["from"] == "left" and out["handed_over"]["to"] == "right"
