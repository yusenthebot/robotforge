"""Catching boxes off a running conveyor (behaviors/catch.py): the factory's arm with its suction cup, offline."""

from __future__ import annotations

import json

import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import catch
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import scenes
from robotforge.mcp_server import build_server
from robotforge.playground import Playground


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.load_scene("factory", options={"boxes": 6, "speed": 0.1})
    playground.run_fast(lambda: False, 1.0)
    return playground


def test_the_arm_catches_a_box_off_the_running_belt_and_sorts_it(pg):
    data = pg.session.data
    body = pg.session.model.body("box_3").id
    v = catch.velocity(pg.session.model, data, body)
    assert v is not None and v[0] == pytest.approx(0.1, abs=0.01)  # riding along +x
    motion, info = pg.behaviors.pick_and_place("arm", "box_3", "red_bin")
    assert isinstance(motion, catch.Catch) and info["region"] is not None
    pg.run_fast(lambda: motion.done, motion.time_limit)
    pg.run_fast(lambda: False, 0.5)
    result = motion.result()
    assert result["caught"], result
    assert result["caught_at"][0] < scenes.PICK_X - 0.06  # taken before it got to the stop: the belt never stopped
    assert pg.behaviors.placed("box_3", motion.region)["success"]
    assert pg.state()["mechanisms"]["conveyor"]["running"]


def test_a_box_that_queues_up_out_of_reach_is_refused_saying_so(pg):
    """box_4 comes to a stop behind the three ahead of it at the stop, beyond the arm's reach: the look ahead counts
    the boxes queued in front of it (passed through as if not there, the arm waited over a point it never got to)."""
    with pytest.raises(BehaviorError, match=r"box_4 is riding on at 0.10 m/s: arm reaches nowhere along the way"):
        pg.behaviors.pick_and_place("arm", "box_4", "red_bin")
    m, d = pg.session.model, pg.session.data
    stop = catch._clear_ahead(m, d, m.body("box_4").id, np.array([1.0, 0.0]))
    end = scenes.STOP_X - 0.015 - 3.5 * scenes.FACTORY_BOX  # 3 boxes and half its own before the stop's face
    assert d.body("box_4").xpos[0] + stop == pytest.approx(end, abs=0.01)


@pytest.mark.anyio
async def test_pick_and_place_catches_through_the_mcp_tool(pg):
    async with Client(build_server(pg, 0)) as client:
        reply = await client.call_tool("pick_and_place", {"robot": "arm", "object": "box_3", "target": "red_bin"})
    result = json.loads(reply.content[0].text)
    assert result["success"] and result["caught"], result
