"""Catching boxes off a running conveyor (behaviors/catch.py): the factory's arm with its suction cup, offline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import catch, catch_stand
from robotforge.behaviors.manager import BehaviorError
from robotforge.design import scenes
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


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
    stop = catch.clear_ahead(m, d, m.body("box_4").id, np.array([1.0, 0.0]))
    end = scenes.STOP_X - 0.015 - 3.5 * scenes.FACTORY_BOX  # 3 boxes and half its own before the stop's face
    assert d.body("box_4").xpos[0] + stop == pytest.approx(end, abs=0.01)


@pytest.mark.anyio
async def test_pick_and_place_catches_through_the_mcp_tool(pg):
    async with Client(build_server(pg, 0)) as client:
        reply = await client.call_tool("pick_and_place", {"robot": "arm", "object": "box_3", "target": "red_bin"})
    result = json.loads(reply.content[0].text)
    assert result["success"] and result["caught"], result


# ---- a robot that goes: catching and carrying it on, and driving beside the belt first ----------------------------


FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def factory(tmp_path, monkeypatch):
    """The factory with no robot of its own to speak of: a mobile Panda is added by the tests, at the place they say."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("belt")
    return playground


def _rover(pg: Playground, x: float, y: float, speed: float = 0.1, yaw: float = -90.0) -> None:
    pg.load_scene("factory", seed=0, starter=False, options={"boxes": 6, "speed": speed})
    pg.add_robot("mm", design="mobile_panda", pos=[x, y], yaw=yaw)
    pg.run_fast(lambda: False, 1.0)


def _run(pg: Playground, motion):
    pg.run_fast(lambda: motion.done, motion.time_limit)
    pg.run_fast(lambda: False, 0.5)
    return motion.result()


def test_a_catch_that_ends_in_a_carry_is_given_the_time_for_the_carry(factory):
    """box_6 is caught beside the belt and taken to the workbench, which the arm does not reach from there: the limit
    pick_and_place waits for is read before anything has happened, so it has to count the carry already."""
    _rover(factory, -0.9, 0.5)
    motion, info = factory.behaviors.pick_and_place("mm/arm", "box_6", "workbench")
    assert isinstance(motion, catch.Catch) and motion.carries
    assert motion.time_limit == catch.CATCH_S + catch.CARRY_S
    result = _run(factory, motion)
    assert result["caught"] and result["driven_m"] > 0.2, result
    assert factory.behaviors.placed("box_6", motion.region)["success"]
    assert factory.state()["mechanisms"]["conveyor"]["running"]


def test_a_catch_the_arm_puts_down_itself_needs_no_time_for_a_carry(pg):
    motion, _ = pg.behaviors.pick_and_place("arm", "box_3", "red_bin")
    assert isinstance(motion, catch.Catch) and not motion.carries
    assert motion.time_limit == catch.CATCH_S


def test_a_robot_whose_arm_does_not_reach_the_belt_drives_to_where_it_does_and_catches(factory):
    """A mobile Panda 0.9 m from the belt, upstream of the box's end stop: its arm does not reach the box's way from
    there. It goes beside the belt (base and arm planned together), takes the box off the running belt and carries
    it to the workbench."""
    _rover(factory, 0.2, 0.9)
    motion, info = factory.behaviors.pick_and_place("mm/arm", "box_6", "workbench")
    assert isinstance(motion, catch_stand.Approach) and motion.spot is not None
    assert math.dist(motion.spot.xy, (0.2, 0.9)) > 0.2
    result = _run(factory, motion)
    assert result["caught"], result
    assert math.dist(result["stood_at"][:2], motion.spot.xy) < 0.3  # it stopped where it was sent
    assert result["caught_at"][0] > -0.4  # off the belt while it still ran, not once it lay at the stop
    assert factory.behaviors.placed("box_6", motion.region)["success"]
    assert factory.state()["mechanisms"]["conveyor"]["running"]


def test_a_robot_that_cannot_be_beside_the_belt_in_time_fetches_the_box_where_it_lies(factory):
    """A mobile Panda 1.4 m from the belt: the box is at the end stop in 12 s, sooner than the robot can be beside the
    belt with its arm ready. It waits for the box to lie still there and sets out for it as for anything lying about
    (whether it then gets the box out of the queue at the stop is the pick's business, not this test's)."""
    _rover(factory, 0.2, 1.4)
    start = factory.behaviors.pose("mm")["position"][:2]
    motion, _ = factory.behaviors.pick_and_place("mm/arm", "box_6", "workbench")
    assert isinstance(motion, catch_stand.Approach) and motion.spot is None
    factory.run_fast(lambda: False, 5.0)
    assert factory.behaviors.pose("mm")["position"][:2] == pytest.approx(start, abs=0.05)  # it waits where it is
    result = _run(factory, motion)
    assert "caught_at" not in result  # nothing was caught off the belt
    assert result["went"][0]["to"] == "box_6"
    assert catching_velocity(factory, "box_6") is None  # and the box lay still when it set out


def catching_velocity(pg: Playground, name: str):
    m, d = pg.session.model, pg.session.data
    return catch.velocity(m, d, m.body(name).id)


def test_a_robot_that_cannot_reach_a_belt_at_all_is_refused_at_once_saying_so(factory):
    """No place beside the belt lets a Go2's arm point down at the box (0.62 m up)."""
    factory.load_scene("factory", seed=0, starter=False, options={"boxes": 6, "speed": 0.1})
    factory.add_robot("dog", design="go2_arm", pos=[-0.9, 0.5], yaw=-90.0)
    factory.run_fast(lambda: False, 1.0)
    with pytest.raises(BehaviorError, match=r"does not reach the height of the belt"):
        factory.behaviors.pick_and_place("dog/arm", "box_6", "workbench")


def test_a_fold_that_is_only_found_from_the_home_pose_still_carries_the_catch_on(factory, monkeypatch):
    """After a catch the arm stands folded some way, and from that pose the search for the pose to carry in found none
    (a mobile Panda on the real Menagerie model): the plan looks from the home pose too when the load is in its hand."""
    from robotforge.behaviors.arm import ArmMotion

    real = ArmMotion.hold_pose
    seen: list[bool] = []

    def hold_pose(self, half, radius):
        at_home = bool(np.allclose(self.q, self.home))
        seen.append(at_home)
        return real(self, half, radius) if at_home else None

    monkeypatch.setattr(ArmMotion, "hold_pose", hold_pose)
    _rover(factory, -0.9, 0.5)
    motion, _ = factory.behaviors.pick_and_place("mm/arm", "box_6", "workbench")
    result = _run(factory, motion)
    assert result["caught"] and "driven_m" in result, result
    assert factory.behaviors.placed("box_6", motion.region)["success"]
    assert False in seen  # the plan did meet an arm that was not at home
