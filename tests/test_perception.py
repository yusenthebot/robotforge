"""Mapping and localization: the occupancy grid and the particle filter on hand-made scans, the point cloud, then a
robot in a real MuJoCo room that maps it, finds itself in the map and goes somewhere on its own map alone."""

import base64
import json
import math
import time
import zlib

import numpy as np
import pytest
from mcp import Client

from robotforge.mcp_server import build_server
from robotforge.perception import cloud
from robotforge.perception.explore import box_sum
from robotforge.perception.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from robotforge.perception.hub import PerceptionError
from robotforge.perception.particles import ParticleFilter, spread, wrap
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

# ---- the grid ----------------------------------------------------------------------------------------------


def _room_scan(
    origin: tuple[float, float], half: float = 3.0, rays: int = 360, half_y: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The ends of rays from `origin` in a room with walls at x = +-half and y = +-half_y (a square by default),
    and which hit (all of them)."""
    limits = (half, half if half_y is None else half_y)
    angle = np.linspace(0, 2 * math.pi, rays, endpoint=False)
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    reach = np.full(rays, np.inf)
    for axis in (0, 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            for wall in (-limits[axis], limits[axis]):
                t = (wall - origin[axis]) / direction[:, axis]
                other = origin[1 - axis] + t * direction[:, 1 - axis]
                ok = (t > 0) & (np.abs(other) <= limits[1 - axis] + 1e-9)
                reach = np.where(ok, np.minimum(reach, t), reach)
    return np.asarray(origin) + direction * reach[:, None], np.ones(rays, bool)


def test_a_scan_makes_walls_occupied_the_inside_free_and_the_outside_unknown():
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=4.0)
    ends, hit = _room_scan((0.0, 0.0))
    grid.integrate(np.zeros(2), ends, hit)
    state = grid.state()
    assert state[grid.cell(0.0, 0.0)] == FREE
    assert state[grid.cell(2.0, 1.0)] == FREE
    assert state[grid.cell(3.0, 0.0)] == OCCUPIED and state[grid.cell(0.0, -3.0)] == OCCUPIED
    assert state[grid.cell(3.8, 0.0)] == UNKNOWN  # behind the wall: never seen
    assert (state == OCCUPIED).sum() == pytest.approx(4 * 6.0 / 0.1, rel=0.15)  # the four walls' length


def test_the_grid_grows_to_hold_a_scan_that_reaches_past_its_edge():
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=1.0)
    ends, hit = _room_scan((0.0, 0.0))
    grid.integrate(np.zeros(2), ends, hit)
    assert grid.cell(3.0, 3.0)[0] < grid.rows and grid.x0 < -3.0
    assert grid.state()[grid.cell(-3.0, 1.0)] == OCCUPIED


def test_the_likelihood_field_made_in_parts_is_the_one_made_whole():
    """The field is made again only where walls came or went, and padded when the grid grows: it must equal a field
    made from scratch after every kind of change."""
    rng = np.random.default_rng(3)
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=1.5)  # small, so scans make it grow
    shape, grew = None, 0
    for i in range(14):
        origin = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5))
        ends, hit = _room_scan(origin, half=1.0 + 0.3 * i, rays=90, half_y=1.0 + 0.2 * i)
        hit &= rng.random(len(hit)) > 0.15  # some rays only clear, some walls are seen later
        grid.integrate(np.asarray(origin), ends, hit)
        incremental = grid.likelihood_field().copy()
        grew += int(shape is not None and shape != incremental.shape)
        shape = incremental.shape
        grid._field = None
        whole = grid.likelihood_field()
        assert incremental.shape == whole.shape
        assert np.abs(incremental - whole).max() < 1e-6, i
    assert grew > 0, "the grid never grew, so padding was not tried"


def test_the_navigation_score_is_made_when_the_robot_stands_in_a_cell_its_own_map_blocks():
    """It came back empty (no 'found') whenever the believed start cell was blocked on the robot's own map."""
    from types import SimpleNamespace

    from robotforge.behaviors.nav_grid import Grid
    from robotforge.perception import truth

    grid = Grid(0.0, 0.0, 0.1, 30, 30)
    free = np.ones(grid.shape, bool)
    scene = SimpleNamespace(grid=grid, free=free.copy())
    own_free = free.copy()
    own_free[14:17, 14:17] = False  # the cell the robot stands in and its neighbours, seen as a wall
    own = SimpleNamespace(grid=grid, free=own_free)
    score = truth.navigation_agreement(own, scene, (1.5, 1.5))
    assert score["found"] > 0.9 and score["through_blocked"] == 0.0 and 0 < score["start_moved_m"] < 0.3
    boxed = SimpleNamespace(grid=grid, free=np.zeros(grid.shape, bool))
    assert "unscored" in truth.navigation_agreement(boxed, scene, (1.5, 1.5))
    assert "unscored" in truth.navigation_agreement(own, scene, (50.0, 50.0))


def test_rays_over_a_low_crate_do_not_clear_it():
    """A 3D lidar's level ray passes over a crate; it must not undo what a steeper ray saw."""
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=4.0)
    crate = np.array([[2.0, 0.0]])
    grid.integrate(np.zeros(2), crate, np.array([True]))
    far = np.array([[3.5, 0.0]])
    for _ in range(10):  # a level ray at 0.4 m, which clears only below 0.2 m
        grid.integrate(np.zeros(2), far, np.array([False]), rise=np.zeros(1), height=0.4, clear_below=0.2)
    assert grid.state()[grid.cell(2.0, 0.0)] == OCCUPIED
    for _ in range(10):  # a ray that runs through the low air does clear
        grid.integrate(np.zeros(2), far, np.array([False]), rise=np.zeros(1), height=0.1, clear_below=0.2)
    assert grid.state()[grid.cell(1.0, 0.0)] == FREE


def test_the_frontier_is_where_free_space_meets_the_unknown():
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=4.0)
    ends = np.array([[2.0, 0.0], [2.0, 0.5]])
    grid.integrate(np.zeros(2), ends, np.array([True, True]))
    front = grid.frontier()
    assert front.any()
    state = grid.state()
    assert (state[front] == FREE).all()


def test_the_packed_map_is_a_deflated_byte_per_cell():
    grid = OccupancyGrid(0.1, (0.0, 0.0), reach=4.0)
    ends, hit = _room_scan((0.0, 0.0))
    grid.integrate(np.zeros(2), ends, hit)
    packed = grid.packed()
    cells = np.frombuffer(zlib.decompress(packed["cells"]), np.uint8).reshape(packed["rows"], packed["cols"])
    row, col = round((0.0 - packed["y0"]) / packed["res"]), round((3.0 - packed["x0"]) / packed["res"])
    assert cells[row, col] == OCCUPIED
    assert set(np.unique(cells).tolist()) <= {0, 1, 2}
    assert OccupancyGrid(0.1).packed() is None


def test_box_sum_counts_a_window_around_each_cell():
    values = np.zeros((7, 7), np.float32)
    values[3, 3] = 1.0
    summed = box_sum(values, 3)
    assert summed[3, 3] == 1.0 and summed[2, 2] == 1.0 and summed[1, 1] == 0.0 and summed.sum() == 9.0


# ---- the particle filter --------------------------------------------------------------------------------------


def _lopsided_room() -> OccupancyGrid:
    """A 6 x 4 m room with a pillar and a wall stub, so that no two places look alike."""
    grid = OccupancyGrid(0.05, (0.0, 0.0), reach=5.0)
    for origin in ((0.0, 0.0), (-1.5, 1.0), (1.5, -1.0), (1.0, 1.5), (-1.0, -1.5)):
        ends, hit = _room_scan(origin, 3.0, 720, half_y=2.0)
        grid.integrate(np.array(origin), ends, hit)
    for x, y in [(1.6 + 0.05 * i, -1.0 + 0.05 * j) for i in range(-4, 5) for j in range(-4, 5)]:  # a pillar
        r, c = grid.cell(x, y)
        grid.logodds[r, c] = 4.0
        grid.seen[r, c] = True
    for y in np.arange(-0.2, 2.0, 0.05):  # a long wall stub from the north wall
        r, c = grid.cell(-0.8, float(y))
        grid.logodds[r, c] = 4.0
        grid.seen[r, c] = True
    grid.version += 1
    return grid


def _scan_from(grid: OccupancyGrid, pose: np.ndarray, rays: int = 120) -> np.ndarray:
    """The ends a lidar at `pose` sees in the grid, in its own frame (cast through the grid's occupied cells)."""
    occupied = grid.occupied()
    angle = np.linspace(-math.pi, math.pi, rays, endpoint=False)
    out = []
    for a in angle:
        for d in np.arange(0.2, 9.0, 0.05):
            x, y = pose[0] + d * math.cos(pose[2] + a), pose[1] + d * math.sin(pose[2] + a)
            r, c = grid.cell(x, y)
            if 0 <= r < grid.rows and 0 <= c < grid.cols and occupied[r, c]:
                out.append((d * math.cos(a), d * math.sin(a)))
                break
    return np.array(out)


def test_a_filter_started_off_pulls_in_to_the_pose_the_scans_show():
    grid = _lopsided_room()
    truth = np.array([-0.5, 0.3, 0.4])
    pf = ParticleFilter(300, seed=1)
    pf.start_at(truth + np.array([0.25, -0.2, 0.12]), 0.15, 0.1)
    scan = _scan_from(grid, truth)
    for _ in range(12):
        pf._weighed_at = None  # the robot moved between these scans
        pf.weigh(grid, scan, np.zeros(3))
    est = pf.estimate()
    assert math.hypot(est.x - truth[0], est.y - truth[1]) < 0.12
    assert abs(wrap(est.heading - truth[2])) < math.radians(6)


def test_a_filter_with_no_idea_where_it_is_finds_out_as_it_moves():
    grid = _lopsided_room()
    pf = ParticleFilter(4000, seed=2)
    assert pf.start_anywhere(grid)
    pose = np.array([-2.0, -1.0, 0.3])
    for _ in range(30):
        scan = _scan_from(grid, pose, 90)
        pf._weighed_at = None
        pf.weigh(grid, scan, np.zeros(3))
        move = np.array([0.15 * math.cos(pose[2]), 0.15 * math.sin(pose[2]), 0.0])
        if _scan_from(grid, pose + move * 3, 12).shape[0] < 12:  # something close ahead: turn instead
            pose = pose + np.array([0, 0, 0.5])
            pf.predict(0.0, 0.0, 0.5)
        else:
            pose = pose + move
            pf.predict(0.15, 0.0, 0.0)
    est = pf.estimate()
    assert est.sigma_m < 0.25
    assert math.hypot(est.x - pose[0], est.y - pose[1]) < 0.25


def test_a_filter_refuses_to_match_a_map_without_walls():
    pf = ParticleFilter(50)
    assert not pf.weigh(OccupancyGrid(0.1), np.zeros((20, 2)), np.zeros(3))
    assert not pf.start_anywhere(OccupancyGrid(0.1))  # nothing free to start from


def test_a_filter_waits_for_the_robot_to_move_before_weighing_again():
    grid = _lopsided_room()
    pf = ParticleFilter(100)
    pf.start_at(np.array([0.0, 0.0, 0.0]))
    assert pf.due()
    pf.weigh(grid, _scan_from(grid, np.zeros(3)), np.zeros(3))
    assert not pf.due()
    pf.predict(0.1, 0.0, 0.0)
    pf.x += 0.1  # the estimate moves with the particles
    assert pf.due()


def test_spread_keeps_one_end_a_slice_the_farthest():
    angle = np.array([0.1, 0.12, 0.13, 2.0])
    reach = np.array([1.0, 3.0, 2.0, 1.5])
    ends = np.stack([reach * np.cos(angle), reach * np.sin(angle)], axis=1)
    kept = spread(ends, 12)
    assert len(kept) == 2
    assert any(np.allclose(k, ends[1]) for k in kept)


# ---- point clouds -----------------------------------------------------------------------------------------------


class _Pinhole:
    """A camera at the origin looking along +x: ray(u, v) goes one meter along it."""

    pos = np.zeros(3)

    def ray(self, us, vs):
        return np.stack([np.ones_like(us, float), -(us - 4.0) / 8.0, -(vs - 3.0) / 8.0], axis=1)


def test_a_depth_picture_becomes_points_at_their_depth():
    depth = np.full((6, 8), 2.0)
    depth[0, 0] = np.nan
    points = cloud.depth_points(depth, _Pinhole())
    assert len(points) == 47
    assert np.allclose(points[:, 0], 2.0)  # every one is two meters along the axis
    assert len(cloud.depth_points(depth, _Pinhole(), stride=2)) == 11


def test_a_cloud_is_thinned_to_the_number_asked_for():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 2, (5000, 3))
    thinned, voxel = cloud.thin(points, 200)
    assert 0 < len(thinned) <= 200 and voxel > 0.01
    summary = cloud.summary(points, 200)
    assert summary["total"] == 5000 and summary["shown"] == len(summary["points_m"])
    assert cloud.summary(np.zeros((0, 3)), 10)["shown"] == 0


@pytest.mark.parametrize("limit", [1, 2, 7, 50])
def test_a_cloud_across_the_origin_is_thinned_to_a_small_limit_at_once(limit):
    """Cubes anchored at the world origin kept -1 and 0 apart: points=1 took 20 s and answered an infinite voxel."""
    rng = np.random.default_rng(1)
    points = rng.uniform(-4, 4, (20000, 3))
    t0 = time.perf_counter()
    summary = cloud.summary(points, limit)
    assert time.perf_counter() - t0 < 10  # generous: it takes well under a second
    assert 1 <= summary["shown"] <= limit
    assert np.isfinite(summary["voxel_m"])
    json.dumps(summary, allow_nan=False)  # strict JSON, as the tool reply is
    if limit >= 7:
        assert summary["shown"] >= limit // 2  # not a handful when more were asked for


# ---- a robot in a real room -------------------------------------------------------------------------------------

ROOM = 3.0  # half the side of the room, m
LIDAR = {"type": "lidar", "rays": 240, "range": 8.0, "rate_hz": 10.0, "noise": 0.01}
ODOMETRY = {"type": "odometry", "noise": {"speed": 0.01, "turn": 0.005}, "bias": {"turn": 0.01}}


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("room", "concrete")
    for name, pos, yaw in (
        ("north", [0, ROOM + 0.05], 0),
        ("south", [0, -ROOM - 0.05], 0),
        ("east", [ROOM + 0.05, 0], 90),
        ("west", [-ROOM - 0.05, 0], 90),
    ):
        playground.add_object(name, shape="wall", pos=pos, yaw=yaw, size=[2 * ROOM + 0.2, 0.1, 1.0])
    playground.add_object("divider", shape="wall", pos=[0.5, 0.8], yaw=90, size=[3.2, 0.1, 1.0])
    playground.add_robot("rover", template="rover", pos=[-2.2, -2.0], yaw=0)
    return playground


def _drive(pg: Playground, points, limit: float = 40.0) -> None:
    for point in points:
        trip = pg.behaviors.navigate("rover", point, seconds=limit)
        pg.run_fast(lambda t=trip: t.done, limit + 1.0)


ROUND = [[2.2, -2.0], [2.2, 2.2], [-2.2, 2.2], [-2.2, -1.0]]


def test_mapping_adds_the_sensors_it_needs_and_reports_the_map(pg):
    started = pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    assert started["state"] == "mapping" and started["lidar"] == "map_lidar" and started["odometry"] == "map_odom"
    assert {"map_lidar", "map_odom"} <= set(pg.sensors.names("rover"))
    _drive(pg, ROUND)
    status = pg.perception.status("rover")
    assert status["scans"] > 100
    assert status["map"]["free_m2"] > 15 and status["map"]["occupied_m2"] > 1.0
    accuracy = status["accuracy"]
    assert accuracy["precision"] > 0.9 and accuracy["wall_error_m"]["mean"] < 0.06
    assert accuracy["recall"] > 0.5
    assert accuracy["navigation"]["through_blocked"] < 0.02  # almost nowhere it would plan through is blocked
    assert accuracy["navigation"]["found"] > 0.6  # and most of what the scene lets it reach it can reach on its map


def test_the_filter_keeps_the_pose_closer_to_the_truth_than_the_odometry_alone(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND + ROUND)
    errors = pg.perception.errors("rover")
    assert errors["odometry_m"].max() > 0.4  # the biased odometry has wandered off
    assert errors["estimate_m"].max() < 0.25
    assert errors["heading_deg"].max() < 6.0
    status = pg.perception.status("rover")
    assert status["pose_error"]["position_m"] < status["odometry_error"]["position_m"]


def test_mapping_from_the_odometry_alone_smears_the_map(pg):
    """What the filter buys: the same drive with the odometry's pose draws walls where there are none."""
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY, "pose": "odometry"})
    _drive(pg, ROUND + ROUND)
    smeared = pg.perception.status("rover")["accuracy"]
    pg2 = pg.perception.mapper("rover")
    assert pg2.pose_source == "odometry"
    assert smeared["precision"] < 0.8


def test_mapping_with_the_true_pose_is_exact_to_a_cell(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "pose": "truth"})
    _drive(pg, ROUND)
    accuracy = pg.perception.status("rover")["accuracy"]
    assert accuracy["precision"] > 0.97 and accuracy["wall_error_m"]["mean"] < 0.03


def test_mapping_runs_on_a_wheel_odometry_too(pg):
    """The rover's wheels, each reading its radius a few percent off, are the odometry the filter corrects."""
    pg.add_sensor("rover", {"type": "wheel_odometry", "name": "wheels", "slip": 0.01, "seed": 3})
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": "wheels"})
    _drive(pg, ROUND + ROUND)
    status = pg.perception.status("rover")
    assert status["odometry"] == "wheels"
    assert status["pose_error"]["position_m"] < 0.25
    assert status["accuracy"]["precision"] > 0.85


def test_a_robot_goes_to_a_point_on_its_own_map_and_pose(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND)
    mapper = pg.perception.mapper("rover")
    trip = pg.behaviors.navigate("rover", [1.8, 1.8], seconds=60, own_map=mapper)
    pg.run_fast(lambda: trip.done, 61)
    assert trip.result()["arrived"], trip.result()
    here = pg.behaviors.pose("rover")["position"]
    assert math.hypot(here[0] - 1.8, here[1] - 1.8) < 0.45
    check = pg.perception.pose_check("rover")
    assert check["error_position_m"] < 0.3


def test_an_unmapped_place_is_not_planned_through_unless_asked(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    pg.run_fast(lambda: False, 1.0)  # one look from the corner: most of the room is still unseen
    mapper = pg.perception.mapper("rover")
    from robotforge.behaviors.navigate import NoWay

    with pytest.raises(NoWay):
        pg.behaviors.navigate("rover", [2.2, 2.2], seconds=30, own_map=mapper)  # unseen behind the divider
    trip = pg.behaviors.navigate("rover", [2.2, 2.2], seconds=30, own_map=mapper, unknown_blocks=False)
    assert trip is not None


def test_a_trip_on_the_own_map_goes_round_a_wall_it_mapped_after_planning(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    mapper = pg.perception.mapper("rover")
    trip = pg.behaviors.navigate("rover", [2.2, 2.2], seconds=60, own_map=mapper, unknown_blocks=False)
    pg.run_fast(lambda: trip.done, 61)
    assert trip.result()["arrived"], trip.result()  # the divider was not on its map when it set off


def test_exploring_maps_the_room_and_says_when_nothing_reachable_is_unseen(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    trips = pg.perception.explore_trips("rover", 150.0)
    summary = None
    try:
        while True:
            trip, limit = next(trips)
            pg.run_fast(lambda t=trip: t.done, limit + 1.0)
    except StopIteration as done:
        summary = done.value
    assert summary["ended"] == "explored", summary
    status = pg.perception.status("rover")
    assert status["accuracy"]["explored"] > 0.9


def test_a_robot_that_is_not_mapping_is_told_how_to_start(pg):
    with pytest.raises(PerceptionError, match="action='start'"):
        pg.perception.mapper("rover")
    with pytest.raises(PerceptionError, match="resolution"):
        pg.perception.start("rover", {"resolution": 5.0})
    with pytest.raises(PerceptionError, match="not 'speed'"):
        pg.perception.start("rover", {"speed": 1})
    with pytest.raises(PerceptionError, match="pose is one of"):
        pg.perception.start("rover", {"pose": "magic"})


def test_a_world_edit_that_leaves_the_robot_keeps_its_map_and_a_reset_drops_it(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND[:1])
    scans = pg.perception.status("rover")["scans"]
    pg.add_object("crate", shape="box", pos=[0.0, -2.0], size=[0.3, 0.3, 0.3])
    assert pg.perception.status("rover", score=False)["scans"] >= scans
    pg.session.reset()
    with pytest.raises(PerceptionError):
        pg.perception.mapper("rover")


def test_localizing_from_nowhere_finds_the_robot_in_its_own_map(pg):
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY, "particles": 150})
    _drive(pg, ROUND)
    pg.perception.stop("rover")
    out = pg.perception.localize("rover", "global", 3000)
    assert out["state"].startswith("localizing")
    _drive(pg, [[2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [2.0, 2.0]])
    after = pg.perception.status("rover", score=False)
    assert after["pose"]["localized"], after["pose"]
    assert after["pose_error"]["position_m"] < 0.35


def test_the_viewer_message_carries_the_map_the_pose_and_the_particles(pg):
    assert pg.perception.viewer_message() is None
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND[:1])
    message = pg.perception.viewer_message()
    robot = message["robots"][0]
    assert message["type"] == "map" and robot["robot"] == "rover" and robot["mapping"]
    assert len(robot["pose"]) == 3 and len(robot["particles"]) % 2 == 0 and robot["sigma_m"] >= 0
    grid = robot["grid"]
    cells = np.frombuffer(zlib.decompress(base64.b64decode(grid["cells"])), np.uint8)
    assert cells.size == grid["rows"] * grid["cols"] and (cells == OCCUPIED).any()
    again = pg.perception.viewer_message()
    assert "grid" not in again["robots"][0]  # sent again only when it changed (or every few seconds)
    json.dumps(message)


def test_the_viewer_is_told_once_when_the_last_map_is_gone(pg):
    """The page drew a dead map for ever: with no mapper the stream sent nothing, and the page drops a map only when a
    later message leaves its robot out."""
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND[:1])
    assert pg.perception.viewer_message()["robots"]
    pg.perception.clear("rover")
    gone = pg.perception.viewer_message()
    assert gone["type"] == "map" and gone["robots"] == []
    assert pg.perception.viewer_message() is None  # said once
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    _drive(pg, ROUND[:1])
    assert "grid" in pg.perception.viewer_message()["robots"][0]  # a new map is sent whole
    pg.session.reset()
    assert pg.perception.viewer_message()["robots"] == []  # a reset drops every mapper too


def test_the_viewer_stream_sends_a_map_message_while_a_robot_maps(pg):
    from starlette.testclient import TestClient

    from robotforge.viewer.server import ViewerHub, create_app

    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    pg.perception.start("rover", {"lidar": LIDAR, "odometry": ODOMETRY})
    pg.run_fast(lambda: False, 1.0)
    client = TestClient(create_app(hub), base_url="http://127.0.0.1:8737")
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        for _ in range(600):
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"])["type"] == "map":
                robots = json.loads(msg["text"])["robots"]
                break
        else:
            raise AssertionError("no map message")
    assert robots[0]["robot"] == "rover" and robots[0]["grid"]["rows"] > 10 and robots[0]["pose"]


def test_the_page_reads_only_fields_the_server_sends():
    """mapview.js reads these names off the message: the server's docs and this list must agree."""
    from pathlib import Path

    page = (Path(__file__).parent.parent / "src" / "robotforge" / "viewer" / "static" / "mapview.js").read_text(
        encoding="utf-8"
    )
    for field in ("r.grid", "r.pose", "r.odometry", "r.particles", "grid.cells", "grid.res", "grid.x0", "grid.rows"):
        assert field in page
    server = (Path(__file__).parent.parent / "src" / "robotforge" / "viewer" / "server.py").read_text(encoding="utf-8")
    for name in ('"particles"', '"odometry"', '"grid"', '"cells"', '"sigma_m"'):
        assert name in server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_the_mapping_tool_explores_reports_and_move_goes_by_the_own_map(pg):
    server = build_server(pg, 0)
    async with Client(server) as client:
        refused = await client.call_tool("mapping", {"robot": "rover", "action": "explore"})
        assert refused.is_error and "action='start'" in refused.content[0].text
        started = await client.call_tool("mapping", {"robot": "rover", "action": "start", "options": {"lidar": LIDAR}})
        assert not started.is_error, started.content[0].text
        explored = await client.call_tool(
            "mapping", {"robot": "rover", "action": "explore", "options": {"seconds": 120}}
        )
        body = json.loads(explored.content[0].text)
        assert body["explored"]["ended"] == "explored" and body["accuracy"]["precision"] > 0.9
        status = await client.call_tool("mapping", {"robot": "rover", "action": "status", "options": {"image": True}})
        assert status.content[1].type == "image"
        moved = await client.call_tool("move", {"robot": "rover", "to": [1.8, 1.8], "map": "own"})
        result = json.loads(moved.content[0].text)
        assert result["arrived"] and result["on_its_own_map"]["error_position_m"] < 0.35
        bare = await client.call_tool("move", {"robot": "rover", "map": "own"})
        assert bare.is_error and "to=" in bare.content[0].text
        cleared = await client.call_tool("mapping", {"robot": "rover", "action": "clear"})
        assert json.loads(cleared.content[0].text)["cleared"]


@pytest.mark.anyio
async def test_read_sensors_gives_a_thinned_point_cloud_when_asked(pg):
    pg.add_sensor("rover", {"type": "lidar3d", "name": "scan"})
    server = build_server(pg, 0)
    async with Client(server) as client:
        got = await client.call_tool("read_sensors", {"robot": "rover", "names": ["scan"], "points": 150})
        scan = json.loads(got.content[0].text)["sensors"]["scan"]
        assert 0 < scan["cloud"]["shown"] <= 150 and scan["cloud"]["total"] >= scan["cloud"]["shown"]
        bad = await client.call_tool("read_sensors", {"robot": "rover", "points": 99999})
        assert bad.is_error and "1 to 2000" in bad.content[0].text
