"""Where a robot starts. One added without a place after load_scene goes to the first free one of the scene's
robot_places; with none, one standing in something at [0, 0] (furniture, another robot, another robot's start) goes
to the nearest free place. A place is free in the world as it starts and as it is now, and another robot counts by
what it covers seen from above. One put somewhere on purpose stays, and the result says so and where is free."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import pytest

from robotforge.playground import Playground, PlaygroundError
from robotforge.sim import placement
from robotforge.sim.session import SimSession, fresh_data

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"  # offline stand-ins for the real robots
ISLAND = (-0.7, 0.7, -0.2, 0.6)  # the kitchen island seen from above: x low, x high, y low, y high
NEXT_TO = 0.5  # m: agent_eval's "next to" a thing, from the robot's outline


def _world(scene: bool) -> Playground:
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    if scene:
        playground.new_world("kitchen")
        playground.load_scene("kitchen", starter=False)  # its island stands on [0, 0]
    else:  # no scene loaded: robots without a place start round [0, 0]
        playground.new_world("yard")
    return playground


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))


@pytest.fixture
def pg():
    return _world(scene=True)


@pytest.fixture
def empty():
    return _world(scene=False)


def _saved(pg: Playground, name: str) -> list[float]:
    return next(r.pos for r in pg.current_world().robots if r.name == name)


def _inside(pg: Playground, name: str, start: bool = False) -> dict[str, float]:
    data = fresh_data(pg.session.model) if start else copy.copy(pg.session.data)
    return placement.inside(pg.session.model, data, name)


def _gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Meters between two boxes seen from above, 0 when they overlap."""
    dx = max(b[0] - a[1], a[0] - b[1], 0.0)
    dy = max(b[2] - a[3], a[2] - b[3], 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def _covers(pg: Playground, start: bool = True) -> dict[str, tuple[float, ...]]:
    data = fresh_data(pg.session.model) if start else copy.copy(pg.session.data)
    mujoco.mj_kinematics(pg.session.model, data)
    return placement.footprints(pg.session.model, data)


def test_a_robot_added_without_a_place_starts_on_the_first_robot_place(pg):
    """Regression: added without a place after load_scene, a Go2 carrying an arm stood at [0, 0], inside the kitchen's
    island, came out under the counter's edge and got stuck on every trip. Moved to the nearest free place, a dog was
    then 0.1 m from the island, so "walk over to the island" was done before it took a step. It starts where the
    scene puts robots, facing the way that place does, as the kitchen's tip says."""
    added = pg.add_robot("dog", template="quadruped")
    entry = next(r for r in pg.current_world().robots if r.name == "dog")
    assert entry.pos == [1.3, -1.0] and entry.yaw == 90 and not _inside(pg, "dog")
    assert "dog starts at [1.3, -1], facing 90 degrees, the first free one of robot_places" in added["notes"]
    assert _gap(_covers(pg)["dog"], ISLAND) > NEXT_TO  # a walk to the island is still to be made
    trip = pg.behaviors.navigate("dog", "island", seconds=40.0)
    pg.run_fast(lambda: trip.done, 41.0)
    assert trip.result()["arrived"] and trip.plan.length > 0.5, trip.result()
    pg.undo()  # one edit: undo takes the robot away, not back into the island
    assert pg.current_world().robots == []


def test_a_yaw_given_is_kept_on_the_robot_place(pg):
    pg.add_robot("dog", template="quadruped", yaw=180)
    entry = next(r for r in pg.current_world().robots if r.name == "dog")
    assert entry.pos == [1.3, -1.0] and entry.yaw == 180


def test_the_same_add_robot_again_is_still_a_no_op_after_the_robot_was_given_a_place(pg):
    """Regression: add_robot tells agents to send the same call again after a download timeout. Given a place, the
    first dog no longer matched the retry's [0, 0], and the retry failed with "'dog' already exists"."""
    pg.add_robot("dog", template="quadruped")
    again = pg.add_robot("dog", template="quadruped")
    assert again.get("unchanged") is True and again["robots"] == ["dog"]
    assert "already in the world exactly like this (at [1.3, -1])" in again["notes"][0]
    with pytest.raises(PlaygroundError, match="already exists"):  # a place asked for is another robot
        pg.add_robot("dog", template="quadruped", pos=[1.3, 1.5])


def test_robots_take_the_places_in_turn_and_then_the_nearest_free_one(pg):
    for name in ("a", "b", "c"):
        pg.add_robot(name, template="quadruped")
    assert [_saved(pg, n) for n in "abc"] == [[1.3, -1.0], [-0.4, -1.4], [1.3, 0.3]]
    fourth = pg.add_robot("d", template="quadruped")
    assert any("robot_places are all taken, so it starts at" in n for n in fourth["notes"]), fourth["notes"]
    covers = _covers(pg)
    for name in "abcd":
        assert not _inside(pg, name) and not _inside(pg, name, start=True)
        assert all(_gap(covers[name], covers[o]) > 0 for o in "abcd" if o != name), (name, covers)


def test_a_rover_added_after_a_go2_does_not_go_under_its_belly(pg):
    """Regression: a rover added after the go2_arm design went to the Go2's place, [0, -0.5], between its legs: it
    touched nothing there, and only touching counted. Another robot now counts by what it covers seen from above."""
    pg.add_robot("dog", design="go2_arm")
    pg.add_robot("rover", template="rover")
    covers = _covers(pg)
    assert _saved(pg, "rover") == [-0.4, -1.4] and _gap(covers["rover"], covers["dog"]) >= placement.CLEAR


def test_under_another_robot_counts_as_taken_without_a_scene(empty):
    """The same rover put under the Go2 by hand touches nothing: only the footprints say the place is taken."""
    empty.add_robot("dog", design="go2_arm", pos=[0.0, 0.0])
    empty.add_robot("probe", template="rover", pos=[0.0, 0.0])
    assert not _inside(empty, "probe")  # between the legs, under the belly: no contact at all
    empty.remove("probe")
    added = empty.add_robot("rover", template="rover")
    covers = _covers(empty)
    assert _saved(empty, "rover") != [0.0, 0.0] and _gap(covers["rover"], covers["dog"]) >= placement.CLEAR
    assert any("[0, 0] is too close to dog, so it starts at" in n for n in added["notes"]), added["notes"]


@pytest.mark.parametrize("scene", [True, False])
def test_a_robot_added_after_the_first_walked_off_does_not_take_its_start(scene):
    """Regression: a pup added once the dog had walked to the fridge was given the dog's saved start, free only in
    the live simulation; after reload_world they stood 6 cm inside each other, and 2 s later the pup was on the dog's
    back. A place has to be free in the world as it starts too."""
    world = _world(scene)
    world.add_robot("dog", template="quadruped")
    if scene:
        trip = world.behaviors.navigate("dog", "fridge", seconds=40.0)
        world.run_fast(lambda: trip.done, 41.0)
        assert trip.result()["arrived"], trip.result()
    else:
        world.behaviors.move("dog", 0.3, 0.0)
        world.run_fast(lambda: False, 4.0)
        world.behaviors.stop("dog")
        assert world.behaviors.pose("dog")["position"][0] > 0.6  # [0, 0] is free in the live simulation
    added = world.add_robot("pup", template="quadruped")
    assert _saved(world, "pup") != _saved(world, "dog")
    if not scene:
        assert any("[0, 0] is inside dog, so it starts at" in n for n in added["notes"]), added["notes"]
    world.reload_world()
    assert not _inside(world, "pup") and not _inside(world, "dog")
    world.run_fast(lambda: False, 2.0)
    dog, pup = world.behaviors.pose("dog"), world.behaviors.pose("pup")
    assert dog["upright"] and pup["upright"] and abs(pup["position"][2] - dog["position"][2]) < 0.03, (dog, pup)


def test_a_robot_put_inside_on_purpose_stays_and_is_told_where_is_free(pg):
    added = pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    assert _saved(pg, "dog") == [0.0, 0.0] and "island" in _inside(pg, "dog")
    note = next(n for n in added["notes"] if "starts inside island" in n)
    assert "pushed out or get stuck" in note and "is free" in note
    moved = pg.update_robot("dog", pos=[1.3, -1.0])  # where the scene puts robots: nothing to say
    assert not any("inside" in n for n in moved["notes"])
    back = pg.update_robot("dog", pos=[0.2, 0.2])
    assert any("starts inside island" in n for n in back["notes"])


def test_a_robot_put_on_another_robots_start_is_told(empty):
    empty.add_robot("dog", template="quadruped")
    empty.behaviors.move("dog", 0.3, 0.0)
    empty.run_fast(lambda: False, 4.0)
    added = empty.add_robot("pup", template="quadruped", pos=[0.0, 0.0])
    assert _saved(empty, "pup") == [0.0, 0.0]
    note = next(n for n in added["notes"] if "where dog starts too" in n)
    assert "after a reset or reload_world they stand inside each other" in note and "is free" in note


def test_an_arm_fixed_to_the_floor_is_only_told(pg):
    added = pg.add_robot("arm", template="arm")
    assert _saved(pg, "arm") == [0.0, 0.0]  # a fixed base is not moved
    assert any("arm starts inside island" in n and "is free" not in n for n in added["notes"]), added["notes"]


def test_a_scene_cleared_away_takes_its_places_with_it(pg):
    pg.clear_scene()
    pg.add_robot("dog", template="quadruped")
    assert _saved(pg, "dog") == [0.0, 0.0]
