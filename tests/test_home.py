"""The home scene and its swing door: a robot pushes the door open going through, a dog with an arm works the
kitchen's drawer from beside it (design/home_scenes.py, design/mechanisms.py, behaviors/nav_map.py, visit.py).
Offline: our own rover and the test stand-ins for Menagerie's Go2 and PiPER."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from robotforge.behaviors import operate, visit
from robotforge.design import home_scenes
from robotforge.playground import Playground
from robotforge.sim import mechanisms
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("home")
    return playground


def _door(pg: Playground) -> dict:
    s = pg.session
    return mechanisms.read(s.model, s.data, {"door": "door"}, None)["door"]


def _doorway(pg: Playground) -> None:
    """A wall across the floor with a door in a frame in the middle of it."""
    for name, y in (("wall_north", 1.55), ("wall_south", -1.55)):
        pg.add_object(name, shape="wall", pos=[0.0, y], size=[2.1, 0.1, 1.0], yaw=90)
    pg.add_object("frame", shape="door_frame", pos=[0.0, 0.0], size=[1.0, 0.12, 1.2], yaw=90)
    pg.add_object("door", shape="door", pos=[0.0, 0.0], size=[0.83, 0.035, 1.1], yaw=90)


def test_a_door_swings_either_way_when_pushed_and_closes_by_itself(pg):
    _doorway(pg)
    m, d = pg.session.model, pg.session.data
    leaf = m.body("door/leaf").id
    assert _door(pg) == {"type": "door", "angle_deg": 0.0, "open": False}
    for push in (10.0, -10.0):
        d.xfrc_applied[leaf] = [push, 0, 0, 0, 0, 0]
        pg.run_fast(lambda: False, 1.0)
        state = _door(pg)
        assert state["open"] and abs(state["angle_deg"]) > 60 and math.copysign(1, -state["angle_deg"]) == push / 10
        d.xfrc_applied[leaf] = 0.0
        pg.run_fast(lambda: False, 6.0)
        assert abs(_door(pg)["angle_deg"]) < 15  # swung back toward shut by its closer


def test_a_rover_drives_through_a_shut_door_pushing_it_open(pg):
    """Navigation counts the leaf as something to push, not to go round: with the leaf as an obstacle the wall
    across the floor had no way through it at all."""
    _doorway(pg)
    pg.add_robot("rover", template="rover", pos=[1.0, 0.1], yaw=180)
    pg.run_fast(lambda: False, 0.5)
    adr = int(pg.session.model.joint("door/leaf").qposadr[0])
    trip = pg.behaviors.navigate("rover", [-1.0, 0.0])
    peak = [0.0]

    def through() -> bool:
        peak[0] = max(peak[0], abs(math.degrees(float(pg.session.data.qpos[adr]))))
        return trip.done

    pg.run_fast(through, 40.0)
    result = trip.result()
    assert result["arrived"], result
    assert peak[0] > 30  # it pushed the door open on its way
    assert pg.behaviors.pose("rover")["position"][0] < -0.8


def test_the_home_has_two_rooms_a_door_and_things_a_dogs_arm_reaches(pg):
    out = pg.load_scene("home")
    assert out["robots"] == ["go2_arm"]
    names = set(out["objects"])
    assert {"kitchen", "living_room", "door", "door_frame", "apple", "chest", "block", "button"} <= names
    pg.run_fast(lambda: False, 1.0)
    state = pg.state()["mechanisms"]
    assert state["door"]["open"] is False and state["chest"]["open"] is False
    d = pg.session.data
    for thing in ("apple", "cup"):  # on the low kitchen table, near its front edge
        assert d.body(thing).xpos[2] < home_scenes.LOW_TABLE + 0.1
    chest = d.body("chest").xpos
    block = d.body("block").xpos
    assert abs(block[0] - chest[0]) < 0.12 and block[2] < 0.06  # in the drawer, on its floor


def test_a_dog_opens_the_drawer_from_beside_it(pg):
    """From in front of the chest the drawer pulled out runs into the dog; the spot is found beside its way."""
    pg.load_scene("home")
    pg.run_fast(lambda: False, 1.0)
    arm = pg.behaviors.resolve("go2_arm", arm=True)
    trip, _, _ = operate.start(pg.behaviors, arm, "chest", "open", None)
    assert isinstance(trip, visit.Visit)
    chest = pg.session.data.body("chest").xpos
    x, y = trip.spot.xy
    assert abs(x - chest[0]) > 0.3  # beside the drawer's way, not in it
    assert visit._clear_of_way(_stand(pg, trip), np.array(trip.spot.xy), trip.spot.heading, trip.job)


def _stand(pg: Playground, trip: visit.Visit) -> visit._ClearStand:
    m, d = pg.session.model, pg.session.data
    return visit._ClearStand(m, d, trip.robot, trip.arm, trip.hold_q)
