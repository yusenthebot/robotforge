"""Navigation: the grid search on hand-built maps, maps of compiled scenes, trips in scenes for several kinds of
robot (move with to=...), and move without a destination behaving as it always did."""

import asyncio
import dataclasses
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client
from starlette.testclient import TestClient

from robotforge.behaviors import nav_grid, nav_map
from robotforge.behaviors.manager import TRIP, BehaviorError
from robotforge.behaviors.nav_grid import Grid
from robotforge.behaviors.navigate import NoWay
from robotforge.bridge.onnx_policy import PolicyWalker, onnxruntime_available
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub, create_app

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "concrete")
    return playground


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _trip(pg: Playground, robot: str, to, limit: float = 40.0, **kwargs):
    trip = pg.behaviors.navigate(robot, to, seconds=limit, **kwargs)
    pg.run_fast(lambda: trip.done, limit + 1.0)
    return trip


def _map(pg: Playground, robot: str) -> nav_map.NavMap:
    """The map the planner draws for `robot` (planned to where it stands)."""
    xy = pg.behaviors.pose(robot)["position"][:2]
    trip = pg.behaviors.navigate(robot, [xy[0] + 0.01, xy[1]])
    pg.behaviors.stop(robot)
    return trip.planner.last_map


def _lethal_at(nav: nav_map.NavMap, x: float, y: float) -> int:
    return int(nav.lethal[nav.grid.cell(x, y)])


# ---- the search on hand-built maps ---------------------------------------------------------------------------


def _goal_at(grid: Grid, cell: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    goal = np.zeros(grid.shape, bool)
    goal[cell] = True
    xs, ys = grid.centers()
    x, y = grid.center(*cell)
    return goal, np.hypot(xs - x, ys - y)


def test_the_search_goes_round_a_wall_the_shortest_way():
    grid = Grid(0.0, 0.0, 0.1, 30, 30)
    free = np.ones(grid.shape, bool)
    free[5:, 15] = False  # a wall up from row 5: the way round is below it
    goal, h = _goal_at(grid, (25, 25))
    path = nav_grid.astar(grid, free, np.zeros(grid.shape), (25, 5), goal, h)
    assert path[0] == (25, 5) and path[-1] == (25, 25)
    assert all(free[c] for c in path)
    assert min(r for r, _ in path) == 4  # just round the wall's end, no further
    # Down to the gap, through it and back up: two straight legs of about 2.24 m, in 45 and 90 degree steps.
    assert nav_grid.length([grid.center(*c) for c in path]) == pytest.approx(5.19, abs=0.15)


def test_the_search_never_cuts_a_blocked_corner():
    grid = Grid(0.0, 0.0, 0.1, 3, 3)
    free = np.ones(grid.shape, bool)
    free[0, 1] = free[1, 0] = False  # the diagonal step from (0, 0) to (1, 1) would scrape both
    goal, h = _goal_at(grid, (1, 1))
    assert nav_grid.astar(grid, free, np.zeros(grid.shape), (0, 0), goal, h) is None


def test_the_search_says_none_when_the_goal_is_closed_off():
    grid = Grid(0.0, 0.0, 0.1, 20, 20)
    free = np.ones(grid.shape, bool)
    free[8:13, 8] = free[8:13, 12] = free[8, 8:13] = free[12, 8:13] = False  # a closed box round (10, 10)
    goal, h = _goal_at(grid, (10, 10))
    assert nav_grid.astar(grid, free, np.zeros(grid.shape), (2, 2), goal, h) is None
    assert not nav_grid.reachable(grid, free, (2, 2))[10, 10]
    # Found from the small side alone: the box holds 9 cells. Not beyond the budget, nor when the box has a gap.
    assert nav_grid.enclosed(grid, free, goal, (2, 2))
    assert not nav_grid.enclosed(grid, free, goal, (2, 2), budget=5)
    free[8, 8] = True  # its corner open: only a diagonal step past two wall cells would get in, as in astar
    assert nav_grid.enclosed(grid, free, goal, (2, 2))
    free[10, 8] = True
    assert not nav_grid.enclosed(grid, free, goal, (2, 2))


def test_room_is_the_exact_distance_to_the_nearest_blocked_cell():
    rng = np.random.default_rng(3)
    blocked = rng.random((40, 50)) < 0.03
    room = nav_grid.clearance(blocked, 0.05, 1.0)
    rows, cols = np.nonzero(blocked)
    for r, c in [(0, 0), (17, 23), (39, 49), (20, 2)]:
        nearest = np.hypot(rows - r, cols - c).min() * 0.05
        assert room[r, c] == pytest.approx(min(max(nearest - 0.025, 0.0), 1.0), abs=1e-9)


def test_string_pulling_keeps_the_corners_and_the_room():
    grid = Grid(0.0, 0.0, 0.05, 60, 60)
    blocked = np.zeros(grid.shape, bool)
    blocked[:40, 30] = True  # walls make an L-shaped corridor bend
    room = nav_grid.clearance(blocked, grid.res, 1.0)
    free = room >= 0.1
    goal, h = _goal_at(grid, (5, 50))
    path = nav_grid.astar(grid, free, np.zeros(grid.shape), (5, 10), goal, h)
    pulled = nav_grid.pull_string(grid, path, room, 0.5)
    assert pulled[0] == path[0] and pulled[-1] == path[-1]
    assert len(pulled) <= 6 and len(path) > 40  # two long legs, and a few short ones round the wall's end
    least = min(room[c] for c in path)
    for a, b in zip(pulled, pulled[1:], strict=False):
        assert nav_grid._line_room(grid, a, b, room) >= min(0.5, least) - 1e-9


def test_a_cost_near_walls_keeps_paths_in_the_middle_of_a_corridor():
    grid = Grid(0.0, 0.0, 0.05, 20, 80)
    blocked = np.zeros(grid.shape, bool)
    blocked[0, :] = blocked[-1, :] = True  # a corridor 0.9 m wide
    room = nav_grid.clearance(blocked, grid.res, 0.6)
    penalty = 2.0 * np.clip((0.45 - room) / 0.4, 0, 1) ** 2
    goal, h = _goal_at(grid, (3, 75))
    path = nav_grid.astar(grid, room >= 0.05, penalty, (3, 4), goal, h)
    middle = [r for r, c in path if 20 <= c <= 60]
    assert min(middle) >= 8 and max(middle) <= 11  # away from both walls between the ends


def test_on_a_ramp_paths_go_straight_up_or_down_it():
    """Columns 30 to 49 are a ramp (and the ground around it a robot's length wide) whose way up is +x: there steps
    go only along x, and pulled strings keep within 8 degrees of x. Without that, the path from the bottom left to
    the top right crossed it at 45 degrees."""
    grid = Grid(0.0, 0.0, 0.05, 60, 80)
    free = np.ones(grid.shape, bool)
    aim = np.full(grid.shape, np.nan)
    aim[:, 30:50] = 0.0
    goal, h = _goal_at(grid, (55, 75))
    path = nav_grid.astar(grid, free, np.zeros(grid.shape), (5, 5), goal, h, nav_grid.along(aim))
    assert path is not None
    for (r0, c0), (r1, c1) in zip(path, path[1:], strict=False):
        if np.isfinite(aim[r0, c0]) or np.isfinite(aim[r1, c1]):
            assert r0 == r1, ((r0, c0), (r1, c1))
    room = np.full(grid.shape, 1.0)
    pulled = nav_grid.pull_string(grid, path, room, 0.5, aim, math.radians(4.0))
    assert len(pulled) <= 4, pulled  # in, along, out: no zigzag
    assert _worst_on_ramp(pulled, aim) <= math.radians(4.0)
    # A ramp turned 30 degrees still has a way through, its steps within 23 degrees of its way (the diagonals),
    # with no sharp corner on it (lines that go within 4 degrees of its way where the steps allow, 26.6 degrees).
    aim[np.isfinite(aim)] = math.radians(30.0)
    path = nav_grid.astar(grid, free, np.zeros(grid.shape), (5, 5), goal, h, nav_grid.along(aim))
    assert path is not None
    pulled = nav_grid.pull_string(grid, path, room, 0.5, aim, math.radians(4.0))
    assert _worst_on_ramp(pulled, aim) <= math.radians(23.0)
    on_ramp = [_heading(a, b) for a, b in zip(pulled, pulled[1:], strict=False) if _over(a, b, aim)]
    assert max(on_ramp) - min(on_ramp) < math.radians(20.0)


def _heading(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.atan2(b[0] - a[0], b[1] - a[1])  # rows run along y, columns along x


def _over(a: tuple[int, int], b: tuple[int, int], aim: np.ndarray) -> bool:
    rows, cols = nav_grid.line_cells(a, b)
    return bool(np.isfinite(aim[rows, cols]).any())


def _worst_on_ramp(pulled: list, aim: np.ndarray) -> float:
    """How far (rad) the lines of `pulled` over cells `aim` gives a way to go off it, at worst."""
    way = float(np.nanmax(aim))
    worst = [abs(math.asin(math.sin(_heading(a, b) - way))) for a, b in zip(pulled, pulled[1:], strict=False)]
    return max(w for w, (a, b) in zip(worst, zip(pulled, pulled[1:], strict=False), strict=True) if _over(a, b, aim))


# ---- maps of compiled scenes -----------------------------------------------------------------------------------


def _codes(nav: nav_map.NavMap, x: tuple[float, float], y: tuple[float, float]) -> set[int]:
    """The lethal causes in a rectangle (x from, to; y from, to)."""
    r0, c0 = nav.grid.cell(x[0], y[0])
    r1, c1 = nav.grid.cell(x[1], y[1])
    return {int(v) for v in np.unique(nav.lethal[r0 : r1 + 1, c0 : c1 + 1])}


def _fits(nav: nav_map.NavMap, x: float, y: float) -> bool:
    return bool(nav.free[nav.grid.cell(x, y)])


def test_walls_and_table_legs_block_but_a_high_table_top_does_not(pg):
    pg.add_robot("rover", template="rover")
    pg.add_object("wall", shape="wall", pos=[1.5, 0.0], size=[3.0, 0.1, 1.0], yaw=90)
    pg.add_object("table", shape="table", pos=[0.0, 1.5], size=[0.8, 0.6, 0.72])
    pg.add_object("bench", shape="bench", pos=[-1.5, 0.0])  # seat 0.45 m: nothing goes under it
    nav = _map(pg, "rover")
    assert nav_map.OBSTACLE in _codes(nav, (1.4, 1.6), (0.4, 0.6)) and not _fits(nav, 1.5, 0.5)
    assert _codes(nav, (-0.1, 0.1), (1.4, 1.6)) == {nav_map.FREE} and _fits(nav, 0.0, 1.5)  # under the top
    legs = 0.8 / 2 - 2 * 0.025
    assert nav_map.OBSTACLE in _codes(nav, (legs - 0.03, legs + 0.03), (1.72, 1.78))
    assert not _fits(nav, -1.5, 0.0)  # under the seat, or on it: its top is level, but there is no way up


def test_what_a_robot_climbs_decides_steps_and_slopes(pg):
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0])
    pg.add_robot("dog", template="quadruped", pos=[0.0, 3.0])
    pg.add_terrain("curb", name="curb", pos=[1.5, 1.5], params={"size": [0.3, 5.0], "height": 0.02})
    pg.add_terrain("ramp", name="gentle", pos=[-2.5, 0.0], params={"angle": 8, "length": 1.5, "direction": "up"})
    pg.add_terrain("ramp", name="steep", pos=[-2.5, 3.0], params={"angle": 20, "length": 1.0, "direction": "up"})
    rover, dog = _map(pg, "rover"), _map(pg, "dog")
    curb = ((1.3, 1.7), (1.0, 2.0))
    assert nav_map.STEP in _codes(rover, *curb)  # 2 cm: the rover climbs 1 cm...
    assert _codes(dog, *curb) == {nav_map.FREE}  # ...the dog 2.5 cm (its gait lifts its feet 4.3 cm)
    assert _codes(rover, (-3.2, -2.6), (-0.3, 0.3)) == {nav_map.FREE}  # 8 degrees: within the rover's 15
    assert _fits(rover, -3.0, 0.0) and rover.ground_at(-3.0, 0.0) > 0.03
    assert nav_map.SLOPE in _codes(rover, (-3.0, -2.8), (2.9, 3.1))  # 20 degrees


def test_a_ramp_as_steep_as_a_robot_climbs_is_free(pg):
    """Regression: a 10 degree ramp read 10.00 degrees, and half of it, a hair steeper in floating point, was too steep
    for a robot that climbs 10 (the rover climbs 15; here it is told 10)."""
    pg.add_robot("rover", template="rover", pos=[-2.0, 2.0])
    pg.add_terrain("ramp", name="ramp", pos=[1.5, 0.0], params={"angle": 10, "length": 1.5, "width": 3.0})
    nav = _map(pg, "rover")
    body = dataclasses.replace(nav.body, slope=math.radians(10.0))
    layer = pg.behaviors._layers(pg.session.version)["rover"]
    rover = nav_map.build(layer, body, [])
    assert _codes(rover, (0.0, 1.0), (-0.6, 0.6)) == {nav_map.FREE}  # up its slope, away from its sides


def test_gaps_narrower_than_a_cell_are_seen_between_stepping_stones(pg):
    """Regression: a cell's ground is the top of everything over any part of it, so the 4 cm gaps between the
    obstacle course's stepping stones vanished from the map: stones from 3.8 to 6.2 cm high, none more than a step
    above its neighbor, read as one platform the dog climbed onto from the lowest, and it put a foot in a gap and
    fell (scripts/nav_bench.py). The same here at the dog's present limit (2.5 cm): stones 2.3 to 4.7 cm high, half of
    the patch free without the gaps. The ground at the four quarter points of each cell shows them; a platform
    without gaps stays open to it."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0], yaw=0)
    params = {"size": [1.6, 1.4], "stone": 0.3, "gap": 0.04, "height": 0.035, "jitter": 0.012, "seed": 1}
    pg.add_terrain("stones", name="stones", pos=[2.0, 0.0], params=params)
    pg.add_terrain("platform", name="stage", pos=[2.0, 3.0], params={"size": [1.6, 1.4], "height": 0.02})
    dog = _map(pg, "dog")
    assert 0.02 < dog.body.step < 0.03
    assert nav_map.STEP in _codes(dog, (1.5, 2.5), (-0.5, 0.5))
    (r0, c0), (r1, c1) = dog.grid.cell(1.3, -0.6), dog.grid.cell(2.7, 0.6)
    assert dog.free[r0 : r1 + 1, c0 : c1 + 1].mean() < 0.1  # a few low stones in the middle, out of reach
    assert _codes(dog, (1.5, 2.5), (2.7, 3.3)) == {nav_map.FREE} and _fits(dog, 2.0, 3.0)
    with pytest.raises(NoWay) as caught:  # it is not sent onto them
        pg.behaviors.navigate("dog", [2.0, 0.0])
    assert caught.value.cause == "climb" and "(stones)" in str(caught.value), str(caught.value)


def test_on_hills_legged_robots_keep_to_gentle_slopes_and_a_humanoid_off_bumps(pg):
    """Regressions on the obstacle course (scripts/nav_bench.py). A path crosses a hill's slopes at any angle, and a
    gait slides down a slope to its side: the dog and the Go2 slid off their paths near the top of the bumps (up to
    10 degrees, their limit up a ramp) and gave up; on hills they now keep to 5. The humanoid fell twice on the rim
    of the rough patch, on 8 mm bumps where the slope over 20 cm was within its 3 degrees."""
    pg.add_robot("dog", template="quadruped")
    pg.add_robot("bot", template="humanoid")
    pg.load_scene("obstacle_course")
    with pytest.raises(NoWay) as caught:
        pg.behaviors.navigate("dog", [15.8, 0.1])  # near the top of the bumps
    assert caught.value.cause == "climb" and "5 on hills) (bumps)" in str(caught.value), str(caught.value)
    assert not _fits(_map(pg, "bot"), 9.15, -0.55)  # a bump on the rough patch's rim
    with pytest.raises(NoWay) as caught:
        pg.behaviors.navigate("bot", [8.3, 0.0])  # the middle of the rough patch
    assert "crosses rough, more uneven than it steps (0.5 cm)" in str(caught.value), str(caught.value)


def test_wheels_push_light_loose_objects_aside_and_legs_go_round_them(pg):
    pg.add_robot("rover", template="rover")
    pg.add_robot("dog", template="quadruped", pos=[0.0, -1.5])
    pg.add_object("ball", shape="sphere", pos=[1.0, 0.0], size=[0.15], mass=0.2)
    pg.add_object("crate", shape="box", pos=[0.0, 1.2], size=[0.4, 0.4, 0.3], mass=5.0)
    rover, dog = _map(pg, "rover"), _map(pg, "dog")
    assert _codes(rover, (0.9, 1.1), (-0.1, 0.1)) == {nav_map.FREE} and _fits(rover, 1.0, 0.0)  # 0.2 of 0.69 kg
    assert not _fits(dog, 1.0, 0.0)  # a dog would step on it
    for nav in (rover, dog):
        assert nav_map.OBSTACLE in _codes(nav, (-0.25, 0.25), (0.95, 1.45)) and not _fits(nav, 0.0, 1.2)
        assert not nav.free[nav.ground > 0.2].any()  # the crate's top has room, but no way up


def test_a_point_on_a_slope_too_steep_is_named_as_such(pg):
    """A point on the face of a 20 degree ramp is out of the rover's way because of the slope, not inside
    something: the trip says climb and names the slope."""
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    pg.add_terrain("ramp", name="steep", pos=[2.5, 0.0], params={"angle": 20, "length": 1.0, "direction": "up"})
    nav = _map(pg, "rover")
    rows, cols = np.nonzero(nav.lethal == nav_map.SLOPE)
    x, y = nav.grid.center(int(np.median(rows)), int(np.median(cols)))
    trip = pg.behaviors.navigate("rover", [round(x, 2), round(y, 2)])
    pg.behaviors.stop("rover")
    assert trip.plan.short == "climb" and "on a slope of steep steeper than it climbs" in trip.plan.note, trip.plan


def test_a_dog_walks_round_a_ball_in_its_way_instead_of_onto_it(pg):
    """Regression: loose objects up to half its weight were left off a dog's map as things it pushes aside. In the
    park the template dog walked straight into the 0.3 kg red ball (20 cm, nearly its own height) and fell over."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0], yaw=0)
    pg.add_object("ball", shape="sphere", pos=[1.2, 0.0], size=[0.2], mass=0.3, color="red")
    trip = _trip(pg, "dog", [2.6, 0.0])
    assert trip.result()["arrived"], trip.result()
    assert pg.behaviors.pose("dog")["upright"]
    assert max(abs(y) for _, y in trip.plan.points) > 0.15  # it planned round the ball


# ---- trips --------------------------------------------------------------------------------------------------------


def test_a_rover_finds_its_way_through_the_maze_without_touching_a_wall(pg):
    pg.add_robot("rover", template="rover")
    pg.load_scene("maze")
    nav = _map(pg, "rover")
    trip = pg.behaviors.navigate("rover", [0.0, 0.0], seconds=60)
    least = math.inf
    while not trip.done:
        pg.run_fast(lambda: False, 0.25)
        x, y = pg.behaviors.pose("rover")["position"][:2]
        least = min(least, float(nav.room[nav.grid.cell(x, y)]))
    result = trip.result()
    assert result["arrived"], result
    assert math.dist(pg.behaviors.pose("rover")["position"][:2], [0, 0]) < 0.1
    assert result["path_m"] > 5.0  # 3.6 m as the crow flies: it went round the hedges
    assert least > nav.body.radius  # its middle kept more than half its width from every hedge
    assert TRIP + "rover" not in pg.session.controllers  # done: the path is gone from the viewer


def test_a_dog_walks_to_the_fridge_and_stops_facing_it(pg):
    pg.add_robot("dog", template="quadruped")
    pg.load_scene("kitchen")
    trip = _trip(pg, "dog", "fridge")
    assert trip.result()["arrived"]
    pose = pg.behaviors.pose("dog")
    fridge = trip.planner.destination.point
    bearing = math.degrees(math.atan2(fridge[1] - pose["position"][1], fridge[0] - pose["position"][0]))
    assert abs((bearing - pose["heading_deg"] + 180) % 360 - 180) < 15
    assert pose["upright"]
    # In front of the fridge (its front at y = 1.27), its own front a hand's width from the door.
    assert 0.7 < pose["position"][1] < 1.1


def test_a_humanoid_and_a_go2_arrive_in_the_warehouse(pg):
    pg.add_robot("bot", template="humanoid")
    pg.add_robot("go2", library="go2")
    pg.load_scene("warehouse")
    go2 = _trip(pg, "go2", [-4.0, 0.5], limit=30)
    assert go2.result()["arrived"], go2.result()
    start = pg.behaviors.pose("bot")["position"][:2]
    bot = _trip(pg, "bot", [start[0] + 1.0, start[1] + 0.4], limit=40)
    assert bot.result()["arrived"], bot.result()
    assert pg.behaviors.pose("bot")["upright"]


def _ramp(pg: Playground, angle: float) -> None:
    """A ramp 3 m wide up to a landing, its foot at x = -0.25 and its landing from 1.25 to 3.25."""
    params = {"angle": angle, "length": 1.5, "landing": 2.0, "width": 3.0, "direction": "up"}
    pg.add_terrain("ramp", name="ramp", pos=[1.5, 0.0], params=params)


def test_the_foot_of_a_ramp_is_no_step_but_its_sides_are():
    """Regression: the slope's rise over one cell was allowed only between two cells of a ramp, so where the ramp
    starts on the floor its first cell (up to a cell's rise) read as a step: a rover with a 0.6 cm step had no way
    onto ramps it climbs, depending on where the grid's cells fell (scripts/nav_bench.py, a 15 degree ramp)."""
    ground = np.zeros((4, 10))
    ground[1:, 5:] = 0.009 * np.arange(1, 6)  # rows 1 to 3: a 10 degree ramp from column 5; row 0: floor beside it
    smooth = np.zeros(ground.shape, bool)
    smooth[1:, 5:] = True
    lethal = np.zeros(ground.shape, np.int8)
    body = nav_map.Body(0.1, 0.15, 0.1, 0.1, 1.0, step=0.006, slope=math.radians(15))
    nav_map._mark_steps(ground, smooth, body, 0.05, lethal)
    assert not lethal[2:].any()  # up the ramp from its foot (rows away from its side): fine
    assert (lethal[1, 7:] == nav_map.STEP).all()  # off its side, where it is 2.7 cm and more above the floor: a step
    assert not lethal[1, 5:7].any()  # near the foot its side is low enough to drive off


def test_a_dog_lines_up_with_a_ramp_on_the_floor_and_climbs_it_straight(pg):
    """Regression: paths crossed ramps at any angle and turned on them. The real Go2, sent up the warehouse dock
    (8 degrees) round its toe at 15 to 20 degrees off its slope, swung round, slid down sideways and fell (6 in 28
    trips, scripts/nav_bench.py --case dock). Now the path turns onto the ramp's way before the ramp, the robot lines
    up there on the spot, and climbs straight."""
    pg.add_robot("dog", template="quadruped", pos=[-1.2, 1.2], yaw=90)
    _ramp(pg, 8)
    trip = pg.behaviors.navigate("dog", "ramp", seconds=60)
    k = min(trip.plan.pivots)  # where it turns onto the ramp's way: on the floor, before the ramp's foot
    a, b = trip.plan.points[k], trip.plan.points[k + 1]
    assert a[0] < -0.25 - trip.planner.body.reach and abs(math.atan2(b[1] - a[1], b[0] - a[0])) < math.radians(4.0)
    m, d = pg.session.model, pg.session.data
    root = nav_map.root_body(m, "dog")
    off = []
    while not trip.done:
        pg.run_fast(lambda: trip.done, 0.1)
        if -0.25 < d.xpos[root][0] < 1.25:  # its middle over the slope
            off.append(abs(math.degrees(math.atan2(d.xmat[root][3], d.xmat[root][0]))))
    assert trip.result()["arrived"], trip.result()
    assert pg.behaviors.pose("dog")["position"][2] > 0.35 and pg.behaviors.pose("dog")["upright"]  # landing 0.21 m
    assert off and max(off) < 12.0, max(off)  # it was 21 degrees off, its path straight across the ramp's foot


def test_a_ramp_or_stairs_by_name_is_its_landing(pg):
    """Regression: the middle of a ramp patch is the top end of its slope, and a robot sent "to the dock" stopped
    standing on the slope. Now it goes onto the landing's middle."""
    pg.add_robot("dog", template="quadruped", pos=[-1.2, 1.2], yaw=90)
    _ramp(pg, 8)
    trip = pg.behaviors.navigate("dog", "ramp")
    pg.behaviors.stop("dog")
    assert np.allclose(trip.plan.points[-1], [2.25, 0.0], atol=0.03)  # the landing runs from 1.25 to 3.25
    assert trip.plan.face is None and trip.result()["destination"] == {"object": "ramp", "at": [2.25, 0.0]}


def test_a_rover_with_small_wheels_drives_up_a_ramp(pg):
    pg.add_robot("rover", template="rover", params={"wheel_radius": 0.02}, pos=[-1.0, 0.0], yaw=0)
    _ramp(pg, 10)
    trip = _trip(pg, "rover", [2.5, 0.0])
    assert trip.result()["arrived"], trip.result()
    assert pg.behaviors.pose("rover")["position"][2] > 0.26  # up on the landing, 0.26 m above the floor


def test_a_rover_standing_aslant_on_a_ramp_does_not_take_a_wheel_for_a_caster(pg):
    """Regression: leaning on a slope, the template rover's lower wheel stood a little ahead of the axle in its
    heading frame and was taken for a caster in front, whose slope limit is 3 degrees: standing aslant on a 10
    degree ramp (or on the obstacle course's bumps, in scripts/nav_bench.py) it had no room to move."""
    pg.add_robot("rover", template="rover", pos=[0.5, 0.0], yaw=45)
    _ramp(pg, 10)
    pg.run_fast(lambda: False, 1.0)
    trip = pg.behaviors.navigate("rover", [2.5, 0.0])
    pg.behaviors.stop("rover")
    assert math.degrees(trip.planner.body.slope) == pytest.approx(15.0)


def test_a_rover_with_a_caster_in_front_is_told_it_cannot_climb_a_ramp(pg):
    """Regression: a rover built from parts was sent up ramps a rover climbs and stuck at the foot of 5, 9 and 12
    degree ones, its front caster caught on the toe (scripts/nav_bench.py: twice in the park's hills)."""
    from test_builder import MOBILE_ARM

    pg.add_robot("mm", build=MOBILE_ARM, pos=[-1.0, 0.0], yaw=0)
    _ramp(pg, 9)
    with pytest.raises(NoWay) as caught:
        pg.behaviors.navigate("mm", [2.5, 0.0])
    assert caught.value.cause == "climb" and "slope" in str(caught.value), str(caught.value)


@pytest.mark.parametrize("build", ["SNAKE", "BALANCER"])
def test_a_snake_and_a_balancer_built_from_parts_go_round_a_wall(pg, build):
    """A snake cannot turn on the spot: it steers while it slithers on. A balancer drives while it balances."""
    import test_builder

    pg.add_robot("r", build=getattr(test_builder, build), pos=[0.0, 0.0], yaw=0)
    pg.add_object("wall", shape="wall", pos=[1.2, 0.0], size=[1.0, 0.1, 0.4], yaw=90)
    trip = _trip(pg, "r", [2.5, 0.0], limit=60)
    assert trip.result()["arrived"], trip.result()
    assert trip.turns_in_place == (build == "BALANCER")


@pytest.mark.skipif(not onnxruntime_available(), reason="needs the optional extra: policy")
def test_a_go2_finds_its_way_on_its_learned_walk(pg):
    """The follower steers whatever walks the robot: here the Go2's learned policy (walker='rl')."""
    pg.add_robot("go2", library="go2", pos=[0.0, 0.0], yaw=0)
    pg.add_object("wall", shape="wall", pos=[1.2, 0.0], size=[1.2, 0.1, 0.5], yaw=90)
    trip = _trip(pg, "go2", [2.6, 0.0], walker="rl")
    assert isinstance(trip.walker, PolicyWalker)
    assert trip.result()["arrived"], trip.result()
    assert pg.behaviors.pose("go2")["upright"]


def test_a_robot_built_from_parts_stops_where_its_arm_reaches_the_object(pg):
    from test_builder import MOBILE_ARM

    pg.add_robot("mm", build=MOBILE_ARM)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.8, 0.6])
    trip = _trip(pg, "mm", "cube")
    result = trip.result()
    assert result["arrived"] and result["within_reach"], result
    arm, reach = pg.behaviors._arm("mm/arm").base_and_reach(pg.session.data)
    cube = pg.session.data.xpos[pg.session.model.body("cube").id]
    assert np.linalg.norm(cube - arm) < 0.75 * reach + 0.02


def test_the_end_of_a_path_is_aimed_along_not_at(pg):
    """Regression: the path ended in a stub from the search's last cell center to the point (a few cm, pointing
    anywhere), and near the end the follower aimed at the end itself, whose bearing swings across as the robot
    draws near a little to one side. The real Go2 swung round on the warehouse ramp, slid off and fell
    (scripts/nav_bench.py)."""
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    trip = pg.behaviors.navigate("rover", [2.03, 0.31])  # not a cell center
    points = trip.plan.points
    assert np.allclose(points[-1], [2.03, 0.31])
    assert (np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.1).all()
    end = points[-1]
    along = (end - points[-2]) / np.linalg.norm(end - points[-2])
    xy = end - 0.2 * along + 0.1 * np.array([-along[1], along[0]])  # 20 cm before the end, 10 cm to its left
    trip._last_xy = xy
    s, _ = trip._project(xy)
    target, _ = trip._target(xy, s)
    assert (target - end) @ along > 0.1  # on the line, past the end


def test_the_follower_does_not_cut_a_corner_into_a_ramp_side(pg):
    """Regression: steering for a point a lookahead ahead, past two gentle corners, a rover cut across both where
    its path met the obstacle course's ramp at its toe, put a wheel on the ramp's side and stuck there
    (scripts/nav_bench.py). The lookahead now shrinks while the straight way to it is short of room."""
    pg.add_robot("rover", template="rover")
    pg.load_scene("obstacle_course")
    pg.update_robot("rover", pos=[7.3, -0.85], yaw=-17.9)
    trip = _trip(pg, "rover", "start", limit=60)
    assert trip.result()["arrived"], trip.result()


def test_pushed_off_its_path_it_plans_again_and_still_arrives(pg):
    pg.add_robot("rover", template="rover", pos=[-2.0, 0.0], yaw=0)
    pg.add_object("wall", shape="wall", pos=[0.0, 0.0], size=[2.0, 0.1, 0.5], yaw=90)
    trip = pg.behaviors.navigate("rover", [2.0, 0.0], seconds=40)
    pg.run_fast(lambda: False, 2.0)
    m, d = pg.session.model, pg.session.data
    root = nav_map.root_body(m, "rover")
    adr = m.jnt_qposadr[m.body_jntadr[root]]
    d.qpos[adr + 1] -= 0.7  # a shove sideways, well off the path
    mujoco.mj_forward(m, d)
    pg.run_fast(lambda: trip.done, 40)
    result = trip.result()
    assert result["arrived"] and result["replans"] >= 1, result


def test_a_robot_already_next_to_the_object_turns_to_face_it(pg):
    """Regression: next to the object already, the plan was one point, and the follower read a segment that was not
    there: IndexError at every step, and as the trip stayed attached, the simulation stood still for every robot.
    Here the rover faces away, and it is nudged before the first step, as one still slowing down would be."""
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=180)
    pg.add_object("crate", shape="box", pos=[0.32, 0.0], size=[0.12, 0.12, 0.12], mass=5.0)
    pg.run_fast(lambda: False, 0.5)
    trip = pg.behaviors.navigate("rover", "crate", seconds=20)
    assert len(trip.plan.points) == 2 and trip.plan.length < 0.05
    m, d = pg.session.model, pg.session.data
    adr = m.jnt_qposadr[m.body_jntadr[nav_map.root_body(m, "rover")]]
    d.qpos[adr + 1] += 0.12
    mujoco.mj_forward(m, d)
    pg.run_fast(lambda: trip.done, 21)
    result = trip.result()
    assert result["arrived"], result
    assert abs(pg.behaviors.pose("rover")["heading_deg"]) < 30  # it turned round to face the crate


def test_a_trip_that_breaks_ends_and_the_world_goes_on(pg, monkeypatch):
    """A bug in following a path ends that trip with why "error"; it must never stop the simulation."""
    pg.add_robot("rover", template="rover")
    trip = pg.behaviors.navigate("rover", [2.0, 0.0])

    def broken(*args):
        raise IndexError("a bug")

    monkeypatch.setattr(trip, "_follow", broken)
    pg.run_fast(lambda: False, 1.0)
    assert pg.session.data.time > 0.9
    assert trip.done and trip.result()["why"] == "error" and "a bug" in trip.result()["note"]
    assert pg.behaviors.trip("rover") is None


def test_in_a_corridor_too_narrow_to_turn_round_it_backs_up(pg):
    """Regression: sent back the way it came along a corridor 0.40 m wide (it needs 0.52 m to turn on the spot), the
    dog turned on the spot against both walls until its time ran out. It walks backwards now."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0], yaw=0)
    for side in (1, -1):
        pg.add_object(f"wall_{side + 1}", shape="wall", pos=[0.0, side * 0.25], size=[4.0, 0.1, 0.5])
    pg.add_object("end", shape="wall", pos=[2.05, 0.0], size=[0.6, 0.1, 0.5], yaw=90)
    pg.run_fast(lambda: False, 0.5)
    trip = _trip(pg, "dog", [-1.5, 0.0], limit=40)
    assert trip.result()["arrived"] and trip.result()["time_s"] < 20, trip.result()
    assert abs(pg.behaviors.pose("dog")["heading_deg"]) < 20  # backed all the way, facing as it was


def test_with_room_to_turn_it_turns_rather_than_backing_off_its_path(pg):
    """Regression: half turned, 120 degrees off its way, with a hair less room than it needs to turn, the Go2 backed
    up along its tail, sideways off its path, into the desk lab's desk, and got stuck there."""
    pg.add_robot("go2", library="go2")
    pg.load_scene("desk_lab")
    pg.update_robot("go2", pos=[-0.25, 0.6], yaw=14.0)
    pg.run_fast(lambda: False, 0.5)
    trip = _trip(pg, "go2", [-1.75, -0.15], limit=40)
    assert trip.result()["arrived"] and trip.result()["time_s"] < 20, trip.result()


def test_something_the_map_did_not_show_is_driven_round(pg):
    pg.add_robot("rover", template="rover", pos=[-1.5, 0.0], yaw=0)
    pg.add_object("crate", shape="box", pos=[0.0, 3.0], size=[0.5, 0.5, 0.3], mass=20.0)
    trip = pg.behaviors.navigate("rover", [1.5, 0.0], seconds=40)
    assert np.allclose(trip.plan.points[:, 1], 0.0)  # planned straight: the crate was out of the way
    m, d = pg.session.model, pg.session.data
    adr = m.jnt_qposadr[m.body_jntadr[m.body("crate").id]]
    d.qpos[adr : adr + 3] = [0.0, 0.0, 0.15]  # now it sits on the line
    mujoco.mj_forward(m, d)
    pg.run_fast(lambda: trip.done, 40)
    result = trip.result()
    assert result["arrived"] and result["replans"] >= 1, result


def _pen(pg: Playground, gap: float | None) -> None:
    """Walls round the square from (2, -1) to (4, 1); the west wall has a gap this wide, or none."""
    pg.add_object("north", shape="wall", pos=[3.0, 1.0], size=[2.1, 0.1, 0.6])
    pg.add_object("south", shape="wall", pos=[3.0, -1.0], size=[2.1, 0.1, 0.6])
    pg.add_object("east", shape="wall", pos=[4.0, 0.0], size=[2.1, 0.1, 0.6], yaw=90)
    if gap is None:
        pg.add_object("west", shape="wall", pos=[2.0, 0.0], size=[2.1, 0.1, 0.6], yaw=90)
        return
    half = (2.1 - gap) / 2
    for side in (1, -1):
        pg.add_object(
            f"west_{side + 1}", shape="wall", pos=[2.0, side * (gap + half) / 2], size=[half, 0.1, 0.6], yaw=90
        )


@pytest.mark.parametrize(
    ("build", "cause"),
    [
        (lambda pg: pg.add_object("block", shape="wall", pos=[3.0, 0.0], size=[3.0, 2.0, 1.0], yaw=90), "inside"),
        (lambda pg: _pen(pg, None), "closed_off"),
        (lambda pg: _pen(pg, 0.12), "too_narrow"),  # the rover is 0.18 m wide
        (
            lambda pg: pg.add_terrain(
                "platform", name="stage", pos=[3.0, 0.0], params={"size": [2.0, 2.0], "height": 0.03}
            ),
            "climb",
        ),  # 3 cm up; the rover climbs 1 cm
    ],
)
def test_no_way_says_why(pg, build, cause):
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0])
    build(pg)
    with pytest.raises(NoWay) as caught:
        pg.behaviors.navigate("rover", [3.0, 0.0])
    assert caught.value.cause == cause, str(caught.value)
    assert TRIP + "rover" not in pg.session.controllers


def test_too_narrow_gives_the_width_of_the_gap(pg):
    """Regression: the width came from the narrowest point of the shortest way, which hugged a side of the gap: a
    gap 0.25 m wide was "about 0.10 m wide", and so was a corridor of the maze 0.6 m wide for the Go2."""
    pg.add_robot("rover", template="rover", pos=[0.0, 1.0])
    _pen(pg, 0.25)
    with pytest.raises(NoWay) as caught:
        pg.behaviors.navigate("rover", [3.0, 0.0])
    width = float(re.search(r"about ([0-9.]+) m wide", str(caught.value)).group(1))
    assert caught.value.cause == "too_narrow" and 0.15 <= width <= 0.3, str(caught.value)


def test_a_robot_standing_beside_the_path_is_passed_not_waited_for(pg):
    """Regression: a robot standing next to the path counted as one that moved into the way (its reach, not its
    shape, was measured against the path), so the rover stopped, planned the same path again and gave up."""
    pg.add_robot("rover", template="rover", pos=[0.0, 0.0], yaw=0)
    for side in (1, -1):  # a corridor 0.9 m wide
        pg.add_object(f"wall_{side + 1}", shape="wall", pos=[1.5, 0.5 * side], size=[4.0, 0.1, 0.5])
    pg.add_robot("dog", template="quadruped", pos=[1.5, 0.26], yaw=0)  # standing along the north wall
    trip = _trip(pg, "rover", [3.0, 0.0])
    result = trip.result()
    assert result["arrived"] and "made_way" not in result, result


def test_two_robots_crossing_make_way_for_each_other(pg):
    """Head on along one line: the robot whose name comes first (dog) plans round the other after a moment; the
    rover waits, then goes on once the dog is off its path. Each goal is just past where the other one starts, so
    the rover's goal is taken when it sets off and free by the time it gets there."""
    pg.add_robot("dog", template="quadruped", pos=[-1.5, 0.0], yaw=0)
    pg.add_robot("rover", template="rover", pos=[1.5, 0.05], yaw=180)
    dog = pg.behaviors.navigate("dog", [1.6, 0.0], seconds=60)
    rover = pg.behaviors.navigate("rover", [-1.6, 0.0], seconds=60)
    pg.run_fast(lambda: dog.done and rover.done, 61)
    assert dog.result()["arrived"] and rover.result()["arrived"], (dog.result(), rover.result())
    assert dog.result()["made_way"] >= 1
    assert math.dist(pg.behaviors.pose("rover")["position"][:2], [-1.6, 0.0]) < 0.15
    assert pg.behaviors.pose("dog")["upright"] and pg.behaviors.pose("rover")["upright"]


def test_trips_share_one_map_of_the_furniture_per_robot(pg):
    """Regression: the map of what never moves was kept per robot and per measured shape, and a dog's shape
    differs a little every time its legs move, so each trip added another map to a store that only grew."""
    pg.add_robot("dog", template="quadruped")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.0, 0.1, 0.5], yaw=90)
    for x in (2.0, 0.0, 2.0):
        _trip(pg, "dog", [x, 0.6])
    assert len(pg.behaviors._layers(pg.session.version)) == 1


def test_a_far_destination_does_not_make_later_maps_coarser(pg):
    """Regression: a point far outside the scene stretches the map, and over MAX_CELLS cells a side it gets coarser.
    That map was kept for every later trip of the robot: after one trip to [60, 60] in the kitchen, a humanoid sent
    to the fridge stopped 0.45 m short of it on 9 cm cells."""
    pg.add_robot("rover", template="rover")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.0, 0.1, 0.5], yaw=90)
    far = pg.behaviors.navigate("rover", [60.0, 0.0])
    pg.behaviors.stop("rover")
    assert far.planner.last_map.grid.res > nav_map.RES
    near = pg.behaviors.navigate("rover", [2.0, 0.0])
    pg.behaviors.stop("rover")
    assert near.planner.last_map.grid.res == nav_map.RES


@pytest.mark.parametrize(("width", "arrived"), [(0.6, True), (2.4, False)])
def test_an_object_it_cannot_get_next_to_is_not_arrived_at(pg, width, arrived):
    """Regression: an object out of reach behind something (here in the middle of a block) let the goal ring widen
    up to 2.5 m, and the trip still said arrived. The dog stops where it can; next to a narrow block that is beside
    the cube (arrived, with a note); on a wide one it is a metre off: not arrived, and why."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0], yaw=0)
    pg.add_object("block", shape="box", pos=[2.5, 0.0, 0.25], size=[width, width, 0.5], fixed=True)
    pg.add_object("cube", shape="box", pos=[2.5, 0.0, 0.55], size=[0.1, 0.1, 0.1])
    trip = _trip(pg, "dog", "cube")
    result = trip.result()
    assert result["arrived"] is arrived and "block" in result["note"], result
    assert (result.get("why") == "closed_off") is not arrived, result
    x = pg.behaviors.pose("dog")["position"][0]
    assert 2.5 - width / 2 - 0.5 < x < 2.5 - width / 2  # it went up to the block either way


def test_a_robot_goes_onto_terrain_it_climbs_and_next_to_terrain_it_does_not(pg):
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0], yaw=0)
    pg.add_terrain("platform", name="low", pos=[2.5, 0.0], params={"size": [1.2, 1.2], "height": 0.02})
    pg.add_terrain("platform", name="high", pos=[0.0, 2.5], params={"size": [1.2, 1.2], "height": 0.1})
    onto = _trip(pg, "dog", "low")
    assert onto.result()["arrived"] and onto.plan.face is None, onto.result()
    assert math.dist(pg.behaviors.pose("dog")["position"][:2], [2.5, 0.0]) < 0.15
    beside = _trip(pg, "dog", "high")
    assert beside.result()["arrived"] and beside.plan.face is not None, beside.result()
    assert 1.3 < pg.behaviors.pose("dog")["position"][1] < 1.9  # on the floor in front of its edge at y = 1.9
    with pytest.raises(BehaviorError, match="Objects, robots and terrain patches in this world: low, high"):
        pg.behaviors.navigate("dog", "ramp")


def test_a_trip_sent_off_without_waiting_says_how_it_ended(pg):
    """move(to=..., seconds=null) and the viewer's double-click return at once; `robots` tells how it went after."""
    pg.add_robot("rover", template="rover")
    trip = pg.behaviors.navigate("rover", [1.5, 0.5])
    pg.run_fast(lambda: trip.done, 30)
    done = [s for s in pg.behaviors.status() if s.get("navigating") == "rover"]
    assert done and done[0]["last_trip"]["arrived"], pg.behaviors.status()
    pg.behaviors.move("rover", 0.0, 0.5)  # moved again: that trip is old news
    assert not [s for s in pg.behaviors.status() if s.get("navigating") == "rover"]


def test_a_mistyped_destination_leaves_a_walking_robot_walking(pg):
    """Regression: the robot was stood still (and its walker switched) before the destination was looked up, so a
    typo stopped a dog walking on its own and then came back as an error."""
    pg.add_robot("dog", template="quadruped")
    pg.add_object("fridge", shape="box", pos=[3.0, 0.0], size=[0.6, 0.6, 1.6], fixed=True)
    pg.behaviors.move("dog", 0.2, 0.0)
    pg.run_fast(lambda: False, 1.0)
    with pytest.raises(BehaviorError, match="fridge"):
        pg.behaviors.navigate("dog", "frige stove")
    before = pg.behaviors.pose("dog")["position"][0]
    pg.run_fast(lambda: False, 2.0)
    assert pg.behaviors.pose("dog")["position"][0] - before > 0.25  # still walking at about 0.2 m/s


def test_a_trip_needs_a_speed_that_gets_somewhere(pg):
    pg.add_robot("rover", template="rover")
    with pytest.raises(BehaviorError, match="leave speed out"):
        pg.behaviors.navigate("rover", [1.5, 0.0], speed=0.0)
    assert pg.behaviors.trip("rover") is None


def test_the_viewer_sends_a_robot_and_says_when_there_is_no_way(pg):
    """The viewer's double-click asks {"type": "go_to", "to": [x, y]}, planned off the hub's event loop (the stream
    stood still while it planned); the reply says how long the path is, or that there is no way and why."""
    pg.add_robot("rover", template="rover")
    _pen(pg, None)
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    assert "go_to" in hub.slow_handlers
    client = TestClient(create_app(hub), base_url="http://127.0.0.1:8737")
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        ws.send_text(json.dumps({"type": "go_to", "id": 1, "robot": "rover", "to": [3.0, 0.0]}))  # inside the pen
        reply = _next_message(ws, "reply")
        assert reply["id"] == 1 and reply["ok"] is False
        assert "no way to [3.00, 0.00]" in reply["message"] and "close it off" in reply["message"]
        ws.send_text(json.dumps({"type": "go_to", "id": 2, "robot": "rover", "to": [1.0, 1.5]}))
        reply = _next_message(ws, "reply")
        assert reply["id"] == 2 and reply["ok"] is True and reply["data"]["path_m"] > 1.5, reply
        assert pg.behaviors.trip("rover") is not None


def _next_message(ws, kind: str) -> dict:
    for _ in range(400):
        msg = ws.receive()
        if msg.get("text"):
            data = json.loads(msg["text"])
            if data["type"] == kind:
                return data
    raise AssertionError(f"no {kind} message")


def test_driving_by_hand_ends_a_trip(pg):
    pg.add_robot("rover", template="rover")
    trip = pg.behaviors.navigate("rover", [3.0, 0.0])
    pg.run_fast(lambda: False, 0.5)
    assert pg.behaviors.trip("rover") is trip
    pg.behaviors.move("rover", 0.0, 1.0)  # a key in the viewer, or move without a destination
    assert pg.behaviors.trip("rover") is None and TRIP + "rover" not in pg.session.controllers
    assert trip.done and trip.cause == "interrupted"
    before = pg.behaviors.pose("rover")["heading_deg"]
    pg.run_fast(lambda: False, 1.0)
    assert abs(pg.behaviors.pose("rover")["heading_deg"] - before) > 30  # it turns as told, nothing steers it back


# ---- the move tool ----------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_move_to_a_place_reports_the_trip(pg):
    pg.add_robot("dog", template="quadruped")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.5, 0.1, 0.6], yaw=90)
    server = build_server(pg, 0)
    async with Client(server) as client:
        result = json.loads((await client.call_tool("move", {"robot": "dog", "to": [2.0, 0.0]})).content[0].text)
        # Without seconds a trip is not cut off after a plain move's 3 s: this one takes about 10.
        assert result["arrived"] is True, result
        assert result["time_s"] > 3.0 and result["path_m"] > 2.0
        assert math.dist(result["to"]["position"][:2], [2.0, 0.0]) < 0.25
        nowhere = json.loads((await client.call_tool("move", {"robot": "dog", "to": "1, 0"})).content[0].text)
        assert nowhere["arrived"] is False and nowhere["why"] == "inside" and "wall" in nowhere["note"]


@pytest.mark.anyio
async def test_a_trip_out_of_time_goes_on_when_sent_again(pg):
    """A trip longer than one call (60 s at most; a humanoid walks 5 m in that) stops when the time is up, and the
    note says to call again; the next call goes on from where it stopped."""
    pg.add_robot("dog", template="quadruped")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.5, 0.1, 0.6], yaw=90)
    server = build_server(pg, 0)
    async with Client(server) as client:
        call = {"robot": "dog", "to": [2.0, 0.0], "seconds": 3}
        first = json.loads((await client.call_tool("move", call)).content[0].text)
        assert first["arrived"] is False and first["why"] == "out_of_time", first
        assert "call move with the same to again" in first["note"]
        call.pop("seconds")
        second = json.loads((await client.call_tool("move", call)).content[0].text)
        assert second["arrived"] is True and second["path_m"] < first["path_m"], (first, second)


@pytest.mark.anyio
@pytest.mark.parametrize("how", ["reset", "edit"])
async def test_a_reset_or_an_edit_on_the_way_ends_the_trip_and_says_so(pg, how):
    """Regression: a reset or a world edit cleared the trip without ending it; move waited out its whole time limit
    for nothing and returned time_s NaN (not JSON), and `robots` never told how a trip sent off with seconds=null
    ended."""
    pg.add_robot("dog", template="quadruped")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.5, 0.1, 0.6], yaw=90)
    server = build_server(pg, 0)

    async def disrupt():
        for _ in range(200_000):
            if pg.behaviors.trip("dog") is not None and pg.session.data.time > 2.0:
                break
            await asyncio.sleep(0)
        if how == "reset":
            pg.session.reset()
        else:
            pg.add_object("box", shape="box", pos=[-2.0, 2.0], size=[0.2, 0.2, 0.2])

    async with Client(server) as client:
        task = asyncio.create_task(disrupt())
        result = json.loads((await client.call_tool("move", {"robot": "dog", "to": [2.0, 0.0]})).content[0].text)
        await task
        assert result["arrived"] is False and result["why"] == "interrupted" and "reset" in result["note"], result
        assert 1.5 < result["time_s"] < 10.0  # when it was cut off, not the whole limit
        robots = json.loads((await client.call_tool("robots", {})).content[0].text)
    last = [b["last_trip"] for b in robots["behaviors"] if b.get("navigating") == "dog"]
    assert last and last[0]["why"] == "interrupted", robots["behaviors"]


@pytest.mark.anyio
async def test_move_without_a_destination_is_unchanged(pg):
    """speed left out is 0.2 m/s and seconds left out 3 s, exactly as before move took a destination."""
    pg.add_robot("dog", template="quadruped")
    server = build_server(pg, 0)
    async with Client(server) as client:
        given = json.loads((await client.call_tool("move", {"robot": "dog", "speed": 0.2})).content[0].text)
        pg.session.reset()
        left_out = json.loads((await client.call_tool("move", {"robot": "dog"})).content[0].text)
    assert given["sim_seconds"] == left_out["sim_seconds"] == 3.0
    assert given["distance_m"] == left_out["distance_m"] > 0.3
    schema = next(t for t in await _tools(server) if t.name == "move").input_schema["properties"]
    assert schema["speed"]["default"] == 0.2 and schema["seconds"]["default"] == 3.0  # as an agent reads them


async def _tools(server):
    async with Client(server) as client:
        return (await client.list_tools()).tools
