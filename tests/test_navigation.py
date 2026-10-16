"""Navigation: the grid search on hand-built maps, maps of compiled scenes, trips in scenes for several kinds of
robot (move with to=...), and move without a destination behaving as it always did."""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import nav_grid, nav_map
from robotforge.behaviors.manager import TRIP
from robotforge.behaviors.nav_grid import Grid
from robotforge.behaviors.navigate import NoWay
from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

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
    pg.add_terrain("curb", name="curb", pos=[1.5, 1.5], params={"size": [0.3, 5.0], "height": 0.03})
    pg.add_terrain("ramp", name="gentle", pos=[-2.5, 0.0], params={"angle": 8, "length": 1.5, "direction": "up"})
    pg.add_terrain("ramp", name="steep", pos=[-2.5, 3.0], params={"angle": 20, "length": 1.0, "direction": "up"})
    rover, dog = _map(pg, "rover"), _map(pg, "dog")
    curb = ((1.3, 1.7), (1.0, 2.0))
    assert nav_map.STEP in _codes(rover, *curb)  # 3 cm: the rover climbs 1 cm...
    assert _codes(dog, *curb) == {nav_map.FREE}  # ...the dog's gait lifts its feet 4.3 cm
    assert _codes(rover, (-3.2, -2.6), (-0.3, 0.3)) == {nav_map.FREE}  # 8 degrees: within the rover's 15
    assert _fits(rover, -3.0, 0.0) and rover.ground_at(-3.0, 0.0) > 0.03
    assert nav_map.SLOPE in _codes(rover, (-3.0, -2.8), (2.9, 3.1))  # 20 degrees


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


def test_a_rover_with_small_wheels_drives_up_a_ramp(pg):
    pg.add_robot("rover", template="rover", params={"wheel_radius": 0.02}, pos=[-1.0, 0.0], yaw=0)
    _ramp(pg, 10)
    trip = _trip(pg, "rover", [2.5, 0.0])
    assert trip.result()["arrived"], trip.result()
    assert pg.behaviors.pose("rover")["position"][2] > 0.26  # up on the landing, 0.26 m above the floor


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


def test_driving_by_hand_ends_a_trip(pg):
    pg.add_robot("rover", template="rover")
    trip = pg.behaviors.navigate("rover", [3.0, 0.0])
    pg.run_fast(lambda: False, 0.5)
    pg.behaviors.move("rover", 0.0, 1.0)  # a key in the viewer, or move without a destination
    assert TRIP + "rover" not in pg.session.controllers
    assert trip.done and trip.cause == "interrupted"
    before = pg.behaviors.pose("rover")["heading_deg"]
    pg.run_fast(lambda: False, 1.0)
    assert abs(pg.behaviors.pose("rover")["heading_deg"] - before) > 30  # it turns as told, nothing steers it back


# ---- the move tool ----------------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_move_to_a_place_reports_the_trip(pg):
    pg.add_robot("dog", template="quadruped")
    pg.add_object("wall", shape="wall", pos=[1.0, 0.0], size=[1.5, 0.1, 0.6], yaw=90)
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        result = json.loads((await client.call_tool("move", {"robot": "dog", "to": [2.0, 0.0]})).content[0].text)
        # Without seconds a trip is not cut off after a plain move's 3 s: this one takes about 10.
        assert result["arrived"] is True, result
        assert result["time_s"] > 3.0 and result["path_m"] > 2.0
        assert math.dist(result["to"]["position"][:2], [2.0, 0.0]) < 0.25
        nowhere = json.loads((await client.call_tool("move", {"robot": "dog", "to": "1, 0"})).content[0].text)
        assert nowhere["arrived"] is False and nowhere["why"] == "inside" and "wall" in nowhere["note"]


@pytest.mark.anyio
async def test_move_without_a_destination_is_unchanged(pg):
    """speed left out is 0.2 m/s and seconds left out 3 s, exactly as before move took a destination."""
    pg.add_robot("dog", template="quadruped")
    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        given = json.loads((await client.call_tool("move", {"robot": "dog", "speed": 0.2})).content[0].text)
        pg.session.reset()
        left_out = json.loads((await client.call_tool("move", {"robot": "dog"})).content[0].text)
    assert given["sim_seconds"] == left_out["sim_seconds"] == 3.0
    assert given["distance_m"] == left_out["distance_m"] > 0.3
