"""Right of way at a crossing: who goes first, who holds, and what is left to making way."""

from __future__ import annotations

import math

import numpy as np

from robotforge.behaviors.traffic import Mover, conflict, holds_at, must_wait
from robotforge.playground import Playground


def mover(name: str, points: list[tuple[float, float]], speed: float = 0.5, started: float = 0.0, **kw) -> Mover:
    return Mover(name, np.array(points, float), speed, 0.25, started, **kw)


def test_two_paths_that_cross_make_a_zone_on_each():
    a = mover("a", [(-3.0, 0.0), (3.0, 0.0)])
    b = mover("b", [(0.0, -3.0), (0.0, 3.0)])
    c = conflict(a, b)
    assert c is not None and c.other == "b"
    assert 2.0 < c.enters < 3.0 < c.leaves < 4.0  # the zone is the stretch within 0.65 m of the other's line
    assert abs(c.enters + c.leaves - 6.0) < 0.2  # centred on the crossing, 3 m along
    assert 2.0 < c.other_enters < 3.0 < c.other_leaves < 4.0


def test_paths_that_run_side_by_side_or_head_on_are_not_crossings():
    a = mover("a", [(-3.0, 0.0), (3.0, 0.0)])
    assert conflict(a, mover("b", [(-3.0, 0.4), (3.0, 0.4)])) is None  # side by side
    assert conflict(a, mover("c", [(3.0, 0.0), (-3.0, 0.0)])) is None  # head on: they step aside
    assert conflict(a, mover("d", [(-3.0, 5.0), (3.0, 5.0)])) is None  # far apart


def test_the_robot_that_reaches_the_crossing_first_goes_through():
    a = mover("a", [(-1.5, 0.0), (3.0, 0.0)])  # 1.5 m from the crossing
    b = mover("b", [(0.0, -3.0), (0.0, 3.0)])  # 3 m from it
    ca, cb = conflict(a, b), conflict(b, a)
    assert ca is not None and cb is not None
    assert not must_wait(a, b, ca)
    assert must_wait(b, a, cb)


def test_a_robot_already_in_the_zone_is_waited_for_whatever_the_times_say():
    a = mover("a", [(0.0, -0.2), (0.0, 3.0)], speed=0.1)  # slow, in the zone
    b = mover("b", [(-1.0, 0.0), (3.0, 0.0)], speed=1.0)  # quick, 1 m away
    assert must_wait(b, a, conflict(b, a))


def test_a_tie_goes_to_the_trip_that_began_first_then_to_the_name():
    a = mover("a", [(-2.0, 0.0), (2.0, 0.0)], started=5.0)
    b = mover("b", [(0.0, -2.0), (0.0, 2.0)], started=1.0)
    assert must_wait(a, b, conflict(a, b)) and not must_wait(b, a, conflict(b, a))
    a2, b2 = mover("a", [(-2.0, 0.0), (2.0, 0.0)]), mover("b", [(0.0, -2.0), (0.0, 2.0)])
    assert not must_wait(a2, b2, conflict(a2, b2)) and must_wait(b2, a2, conflict(b2, a2))


def test_nobody_waits_for_a_robot_that_is_itself_holding_or_parks_in_the_zone():
    a = mover("a", [(-2.0, 0.0), (2.0, 0.0)])
    b = mover("b", [(0.0, -2.0), (0.0, 2.0)])
    assert not must_wait(a, mover("b", [(0.0, -2.0), (0.0, 2.0)], holding=True), conflict(a, b))
    parks = mover("p", [(0.0, -2.0), (0.0, 0.0)])  # its path ends at the crossing
    c = conflict(a, parks)
    assert c is not None and c.other_ends_inside and not must_wait(a, parks, c)


def test_a_robot_holds_when_it_is_at_the_edge_of_the_zone():
    b = mover("b", [(0.0, -3.0), (0.0, 3.0)])
    far = mover("a", [(-3.5, 0.0), (3.0, 0.0)])
    near = mover("a", [(-1.0, 0.0), (3.0, 0.0)])
    assert not holds_at(far, conflict(far, b))
    assert holds_at(near, conflict(near, b))


def _crossing_rovers(tmp_path, monkeypatch, traffic: bool):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("cross", "wood")
    pg.add_robot("a", template="rover", pos=[-1.5, 0.0], yaw=0)
    pg.add_robot("b", template="rover", pos=[0.0, -1.5], yaw=90)
    pg.run_fast(lambda: False, 0.5)
    trips = [pg.behaviors.navigate("a", [1.6, 0.0], seconds=60), pg.behaviors.navigate("b", [0.0, 1.6], seconds=60)]
    if not traffic:
        for trip in trips:
            trip.crossings = None
    closest, t0 = 9.0, pg.session.data.time
    while not all(t.done for t in trips) and pg.session.data.time - t0 < 61:
        pg.run_fast(lambda: False, 0.1)
        closest = min(
            closest, math.dist(pg.behaviors.pose("a")["position"][:2], pg.behaviors.pose("b")["position"][:2])
        )
    return closest, [t.result() for t in trips]


def test_two_rovers_whose_paths_cross_pass_one_after_the_other(tmp_path, monkeypatch):
    """Both meet at the origin at the same moment: one holds short of the crossing until the other is through, and
    they never come near each other (without it they meet in the middle and both have to swerve)."""
    closest, (a, b) = _crossing_rovers(tmp_path, monkeypatch, traffic=True)
    assert a["arrived"] and b["arrived"], (a, b)
    assert closest > 0.5, closest
    waiting = [r for r in (a, b) if r.get("gave_way_to")]
    assert len(waiting) == 1 and waiting[0]["waited_s"] > 0.3, (a, b)
    assert "made_way" not in a and "made_way" not in b


def test_the_same_crossing_without_right_of_way_is_slower_and_closer(tmp_path, monkeypatch):
    near, (a, b) = _crossing_rovers(tmp_path, monkeypatch, traffic=False)
    far, (c, d) = _crossing_rovers(tmp_path, monkeypatch, traffic=True)
    assert near < far
    assert max(c["time_s"], d["time_s"]) < max(a["time_s"], b["time_s"])


def test_four_rovers_from_four_sides_all_arrive_and_do_not_touch(tmp_path, monkeypatch):
    """Two head-on pairs and a crossing in the middle: the right of way is no faster here than making way alone
    (docs/MULTI.md), but nobody may be pressed into anybody (a rover is 0.25 m across)."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("four", "wood")
    starts = {"a": (-1.5, 0.0, 0), "b": (1.5, 0.0, 180), "c": (0.0, -1.5, 90), "d": (0.0, 1.5, -90)}
    goals = {"a": [1.6, 0.0], "b": [-1.6, 0.0], "c": [0.0, 1.6], "d": [0.0, -1.6]}
    for name, (x, y, yaw) in starts.items():
        pg.add_robot(name, template="rover", pos=[x, y], yaw=yaw)
    pg.run_fast(lambda: False, 0.5)
    trips = [pg.behaviors.navigate(name, goal, seconds=60) for name, goal in goals.items()]
    closest, t0 = 9.0, pg.session.data.time
    while not all(t.done for t in trips) and pg.session.data.time - t0 < 61:
        pg.run_fast(lambda: False, 0.1)
        at = [pg.behaviors.pose(n)["position"][:2] for n in goals]
        closest = min(closest, *(math.dist(p, q) for i, p in enumerate(at) for q in at[i + 1 :]))
    results = [t.result() for t in trips]
    assert all(r["arrived"] for r in results), results
    assert closest > 0.27, closest
