"""Where a robot starts: one put inside something (furniture, another robot) for want of a place moves to the nearest
free one; one put there on purpose stays, and the result says so and where is free."""

from __future__ import annotations

import copy

import pytest

from robotforge.playground import Playground, PlaygroundError
from robotforge.sim import placement
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("kitchen")
    playground.load_scene("kitchen", starter=False)  # its island stands on [0, 0]
    return playground


def _saved(pg: Playground, name: str) -> list[float]:
    return next(r.pos for r in pg.current_world().robots if r.name == name)


def _inside(pg: Playground, name: str) -> dict[str, float]:
    return placement.inside(pg.session.model, copy.copy(pg.session.data), name)


def test_a_robot_added_without_a_place_starts_clear_of_the_furniture(pg):
    """Regression: added without a place after load_scene, a robot stood at [0, 0], inside the kitchen's island (the
    Go2 carrying an arm 32 cm deep); squeezed out under the counter's edge, it got stuck on every trip, and nothing
    said why."""
    added = pg.add_robot("dog", template="quadruped")
    where = _saved(pg, "dog")
    assert where != [0.0, 0.0] and not _inside(pg, "dog")
    assert any("[0, 0] is inside island" in n and "nearest free place" in n for n in added["notes"]), added["notes"]
    trip = pg.behaviors.navigate("dog", "fridge", seconds=40.0)
    pg.run_fast(lambda: trip.done, 41.0)
    assert trip.result()["arrived"], trip.result()
    pg.undo()  # one edit: undo takes the robot away, not back into the island
    assert pg.current_world().robots == []


def test_the_same_add_robot_again_is_still_a_no_op_after_the_robot_was_moved_clear(pg):
    """Regression: add_robot tells agents to send the same call again after a download timeout. Moved clear of the
    island to [0, -0.5], the first dog no longer matched the retry's [0, 0], and the retry failed with "'dog'
    already exists"."""
    pg.add_robot("dog", template="quadruped")
    again = pg.add_robot("dog", template="quadruped")
    assert again.get("unchanged") is True and again["robots"] == ["dog"]
    assert "already in the world exactly like this (at [0, -0.5])" in again["notes"][0]
    with pytest.raises(PlaygroundError, match="already exists"):  # a place asked for is another robot
        pg.add_robot("dog", template="quadruped", pos=[1.3, -1.0])


def test_a_robot_put_inside_on_purpose_stays_and_is_told_where_is_free(pg):
    added = pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])
    assert _saved(pg, "dog") == [0.0, 0.0] and "island" in _inside(pg, "dog")
    note = next(n for n in added["notes"] if "starts inside island" in n)
    assert "pushed out or get stuck" in note and "is free" in note
    moved = pg.update_robot("dog", pos=[1.3, -1.0])  # where the scene puts robots: nothing to say
    assert not any("inside" in n for n in moved["notes"])
    back = pg.update_robot("dog", pos=[0.2, 0.2])
    assert any("starts inside island" in n for n in back["notes"])


def test_a_second_robot_without_a_place_does_not_stand_in_the_first(pg):
    pg.update_object("island", pos=[3.0, 3.0])  # the middle of the kitchen free again
    first = pg.add_robot("a", template="quadruped")
    assert _saved(pg, "a") == [0.0, 0.0] and not any("inside" in n for n in first["notes"])
    second = pg.add_robot("b", template="quadruped")
    assert _saved(pg, "b") != [0.0, 0.0] and not _inside(pg, "b") and not _inside(pg, "a")
    assert any("[0, 0] is inside a" in n for n in second["notes"]), second["notes"]


def test_an_arm_fixed_to_the_floor_is_only_told(pg):
    added = pg.add_robot("arm", template="arm")
    assert _saved(pg, "arm") == [0.0, 0.0]  # a fixed base is not moved
    assert any("arm starts inside island" in n and "is free" not in n for n in added["notes"]), added["notes"]
