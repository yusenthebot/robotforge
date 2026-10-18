"""Going over to work a mechanism: an arm on a robot that goes has the robot stand where the arm does the whole job
(behaviors/visit.py). Offline: our own lift picker."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import fetch, operate, visit
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import mechanisms
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
TABLE = 0.45  # m: a lift picker's arm reaches a drawer bar on a table this high, and not one on the floor


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("visit")
    return playground


def _drawer_world(pg: Playground) -> None:
    """A lift picker at the middle, a chest of drawers on a table 1.7 m away, its drawer pulling out toward -y."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("stand", shape="table", size=[0.6, 0.45, TABLE], pos=[1.6, 0.8], fixed=True)
    pg.add_object("chest", shape="drawer", pos=[1.6, 0.8, TABLE])
    pg.run_fast(lambda: False, 0.5)


def _run(pg: Playground, trip) -> dict:
    pg.run_fast(lambda: trip.done, visit.TIME_LIMIT)
    pg.run_fast(lambda: False, 0.3)
    s = pg.session
    state = mechanisms.read(s.model, s.data, {"chest": "drawer"}, s.controllers.get(mechanisms.KEY))["chest"]
    return trip.result(s.model, s.data, state)


def test_a_lift_picker_goes_over_to_a_drawer_out_of_reach_and_opens_it(pg):
    _drawer_world(pg)
    arm = pg.behaviors.resolve("picker", arm=True)
    trip, found, _ = operate.start(pg.behaviors, arm, "chest", "open", None)
    assert isinstance(trip, visit.Visit) and found == "chest"
    assert pg.behaviors.errand("picker") is trip
    result = _run(pg, trip)
    assert result["success"], result
    assert result["state"]["open_m"] == pytest.approx(result["state"]["of_m"], abs=0.02)
    (went,) = result["went"]
    assert went["arrived"] and went["distance_m"] > 0.6
    x, y, _ = result["stood_at"]
    assert y < 0.8 - 0.45 / 2  # in front of the chest, on the side its drawer pulls out to
    assert pg.behaviors.errand("picker") is None  # done and let go
    # Folded back over the robot for the way, the arm's own motion in charge of it again.
    arm_motion = pg.behaviors._arm_motion(trip.arm.robot)
    assert np.max(np.abs(arm_motion.q - trip.hold_q)) < 0.1


def test_a_robot_that_goes_is_kept_where_it_stands_with_go_false(pg):
    _drawer_world(pg)
    arm = pg.behaviors.resolve("picker", arm=True)
    with pytest.raises(operate.OperateError, match="cannot take hold of chest's handle"):
        operate.start(pg.behaviors, arm, "chest", "open", None, go=False)


def test_nowhere_to_stand_is_refused_before_it_moves(pg):
    """A drawer on the floor: the lift picker's arm points its gripper down no lower than its column lets it."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("chest", shape="drawer", pos=[1.6, 0.8])
    pg.run_fast(lambda: False, 0.5)
    arm = pg.behaviors.resolve("picker", arm=True)
    with pytest.raises(operate.OperateError, match="nowhere picker can stand does its arm reach chest's handle"):
        operate.start(pg.behaviors, arm, "chest", "open", None)
    assert pg.behaviors.errand("picker") is None


def test_a_spot_in_the_way_of_the_drawer_is_passed_over(pg):
    """Right in front of a chest on the floor the drawer, pulled out, would run into the robot: such a spot does not
    count (on a table it goes over the robot's chassis)."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("chest", shape="drawer", pos=[1.6, 0.8])
    pg.run_fast(lambda: False, 0.5)
    m, d = pg.session.model, pg.session.data
    arm = pg.behaviors._arm_motion(pg.behaviors._arm_name("picker"))
    _, hold_q = arm.hold_pose(0.0, 0.0)
    stand = fetch.Stand(m, d, "picker", arm, hold_q)
    job = visit._job(m, d, "chest", "drawer", "open", None)
    front = 0.8 - 0.3 / 2  # the chest is 0.3 m deep; its drawer pulls out 0.24 m
    assert stand.fits(np.array([1.6, front - 0.3]), math.pi / 2)  # it fits there with the drawer shut
    assert not visit._clear_of_way(stand, np.array([1.6, front - 0.3]), math.pi / 2, job)
    assert visit._clear_of_way(stand, np.array([1.6, front - 0.7]), math.pi / 2, job)


@pytest.mark.anyio
async def test_operate_goes_over_as_an_mcp_tool(pg):
    _drawer_world(pg)
    async with Client(build_server(pg, 0)) as client:
        reply = await client.call_tool(
            "operate", {"robot": "picker", "target": "chest", "action": "open", "amount": 0.15}
        )
    result = json.loads(reply.content[0].text)
    assert result["success"] and result["moved"] == pytest.approx(0.15, abs=0.015), result
    assert result["went"][0]["arrived"] and len(result["stood_at"]) == 3 and result["visit_s"] > 5
