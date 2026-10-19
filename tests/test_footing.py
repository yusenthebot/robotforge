"""Legged robots that look at the ground: ground_scan.py heights, footing.py footholds, and walks over terrain."""

import math
from pathlib import Path

import numpy as np
import pytest

from robotforge.behaviors import footing as footing_module
from robotforge.behaviors.ground_scan import GroundScan
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
STAIRS = {"steps": 3, "step_height": 0.08, "step_depth": 0.3, "landing": 0.8}


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.new_world("footing")
    return playground


def _trunk(pg: Playground, robot: str) -> int:
    return pg.behaviors._root_body(robot)


def test_a_flat_world_is_not_scanned(pg):
    """Walking on flat ground costs nothing extra: the gait has no footing at all there."""
    pg.add_robot("go2", library="go2")
    pg.behaviors.move("go2", 0.2, 0.0, walker="gait")
    assert pg.session.controllers["go2"].footing is None


def test_the_scan_sees_the_steps_and_not_the_robot(pg):
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=STAIRS)
    pg.add_robot("go2", library="go2")
    m, d = pg.session.model, pg.session.data
    scan = GroundScan(m, _trunk(pg, "go2"))
    assert scan.needed
    top = 1.0
    assert scan.height(m, d, 0.0, 0.0, top) == pytest.approx(0.0, abs=1e-6)  # under the trunk: the floor
    assert scan.height(m, d, 0.75, 0.0, top) == pytest.approx(0.08, abs=1e-6)  # the first tread
    assert scan.height(m, d, 1.05, 0.0, top) == pytest.approx(0.16, abs=1e-6)
    # Straight down through a foot to the floor under it.
    gait = pg.behaviors.move("go2", 0.0, 0.0, walker="gait") and pg.session.controllers["go2"]
    body, point = gait.legs[0].chain.body, gait.legs[0].chain.local_point
    foot = d.xpos[body] + d.xmat[body].reshape(3, 3) @ point
    assert scan.height(m, d, float(foot[0]), float(foot[1]), top) == pytest.approx(0.0, abs=1e-6)


def test_a_ray_that_passes_its_own_foot_finds_a_ramp_just_below_it(pg):
    """Regression: a ray started again 0.1 mm past each of the robot's own shapes; past a Go2's foot meshes at a
    ramp's top it started 0.03 mm inside the ramp, missed it and read the floor 32 cm below, and the Go2 fell off the
    ramp's crest. It now starts again a micrometre past."""
    pg.add_terrain("ramp", pos=[2.5, 0.0], params={"angle": 12, "length": 1.5, "landing": 0.8})  # from x 0.6
    pg.add_robot("go2", library="go2")
    m, d = pg.session.model, pg.session.data
    scan = GroundScan(m, _trunk(pg, "go2"))
    top_of_ramp = 1.5 * math.tan(math.radians(12.0))
    gait = pg.behaviors.move("go2", 0.0, 0.0, walker="gait") and pg.session.controllers["go2"]
    body = gait.legs[0].chain.body
    # Put the foot's body just over the ramp's top, then look down through it from just above.
    x = 0.6 + 1.5 + 0.1  # on the landing
    shift = np.array([x, 0.0, top_of_ramp + 0.03]) - d.xpos[body]
    d.xpos[body] += shift
    d.geom_xpos[m.geom_bodyid == body] += shift
    found = scan.height(m, d, x, 0.0, top_of_ramp + 0.06, body)
    assert found == pytest.approx(top_of_ramp, abs=1e-4)


def test_a_foot_does_not_land_on_a_stair_nosing(pg):
    """The foothold is moved off an edge to where the ground around the foot is flat."""
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=STAIRS)
    pg.add_robot("go2", library="go2")
    pg.behaviors.move("go2", 0.01, 0.0, walker="gait")
    pg.run_fast(lambda: False, 0.5)
    gait = pg.session.controllers["go2"]
    footing = gait.footing
    m, d = pg.session.model, pg.session.data
    edge_x = 0.6  # the first riser: the flight is 2 m long, centred 1.6 m ahead
    footing._heading = (1.0, 0.0)
    footing._look_ahead(m, d, 0, (edge_x + 0.005, 0.14), (1.0, 0.0), 1.0)
    landed = edge_x + 0.005 + footing.shift[0]
    assert abs(landed - edge_x) > footing_module.FOOTHOLD_WINDOW * footing.step - 1e-9
    assert footing.land[0] == pytest.approx(0.08 if landed > edge_x else 0.0, abs=1e-6)


def test_a_go2_walks_up_and_down_stairs_it_sees(pg):
    """8 cm stairs up and down at 0.2 m/s: on its old gait, which planned its feet as if on flat ground, a Go2 went
    round them or fell back off the second step (scripts/legged_bench.py)."""
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=STAIRS)
    pg.add_robot("go2", library="go2")
    pg.behaviors.move("go2", 0.2, 0.0, walker="gait")
    trunk = _trunk(pg, "go2")
    highest = 0.0
    for _ in range(30):
        pg.run_fast(lambda: False, 1.0)
        highest = max(highest, float(pg.session.data.xpos[trunk][2]))
        assert pg.behaviors.pose("go2")["upright"]
        if pg.session.data.xpos[trunk][0] > 2.8:
            break
    x, y = pg.session.data.xpos[trunk][:2]
    assert x > 2.8 and abs(y) < 0.5
    assert highest > 0.45  # over the landing, 24 cm up


def test_told_to_walk_straight_it_keeps_its_heading(pg):
    """A turn rate held at zero let slips add up (a Go2 ended up 55 degrees off on stairs); the heading is held."""
    pg.add_robot("go2", library="go2")
    gait = pg.behaviors.move("go2", 0.3, 0.0, walker="gait") and pg.session.controllers["go2"]
    pg.run_fast(lambda: False, 1.0)
    trunk = _trunk(pg, "go2")
    r = pg.session.data.xmat[trunk]
    start = math.atan2(r[3], r[0])
    pg.session.data.qvel[gait.root_dof + 5] += 1.0  # a twist
    pg.run_fast(lambda: False, 4.0)
    r = pg.session.data.xmat[trunk]
    turned = math.degrees((math.atan2(r[3], r[0]) - start + math.pi) % (2 * math.pi) - math.pi)
    assert abs(turned) < 4.0


def test_a_flat_world_with_other_robots_or_loose_things_is_not_scanned(pg):
    """Only what is fixed in the world is ground: another robot, an arm or a loose box a dog can push aside is not,
    and a world with only those costs no scans. A Go2 walking into a loose 0.15 m box was deflected off its path when
    it read the box as a step to climb."""
    pg.add_robot("go2", library="go2")
    pg.add_robot("second", library="anymal_c", pos=[3.0, 2.0])
    pg.add_object("crate", shape="box", size=[0.3, 0.3, 0.15], pos=[1.0, 0.0], mass=0.3)
    pg.behaviors.move("go2", 0.2, 0.0, walker="gait")
    assert pg.session.controllers["go2"].footing is None
    pg.add_object("curb", shape="box", size=[0.3, 0.3, 0.15], pos=[1.5, 1.0], fixed=True)
    pg.behaviors.move("go2", 0.2, 0.0, walker="gait")
    assert GroundScan(pg.session.model, _trunk(pg, "go2")).needed  # a fixed box is ground


def test_a_loose_box_is_not_ground_to_the_scan(pg):
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=STAIRS)
    pg.add_object("crate", shape="box", size=[0.3, 0.3, 0.2], pos=[0.3, 0.0], mass=0.3)
    pg.add_robot("go2", library="go2")
    m, d = pg.session.model, pg.session.data
    scan = GroundScan(m, _trunk(pg, "go2"))
    assert scan.height(m, d, 0.3, 0.0, 1.0) == pytest.approx(0.0, abs=1e-6)  # the floor under the crate
    assert scan.height(m, d, 0.75, 0.0, 1.0) == pytest.approx(0.08, abs=1e-6)  # the stairs still count


def test_a_robot_limited_in_slope_stands_before_stairs_it_cannot_climb(pg):
    """Spot falls backwards off 8 cm x 30 cm stairs (library max_slope); told to walk at them it stands upright
    before them, and backs off when told to."""
    steep = {"steps": 3, "step_height": 0.08, "step_depth": 0.3, "landing": 0.8}
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=steep)
    pg.add_robot("go2", library="go2")
    pg.behaviors.move("go2", 0.2, 0.0, walker="gait")
    gait = pg.session.controllers["go2"]
    gait.limit_slope(0.24)
    pg.run_fast(lambda: False, 12.0)
    trunk = _trunk(pg, "go2")
    assert gait.blocked and "steep" in pg.behaviors.pose("go2")["blocked"]
    assert pg.behaviors.pose("go2")["upright"]
    assert pg.session.data.xpos[trunk][0] < 0.6  # it did not walk up to the first step
    pg.behaviors.move("go2", -0.2, 0.0, walker="gait")  # backing off is always allowed
    pg.run_fast(lambda: False, 3.0)
    assert pg.session.data.xpos[trunk][0] < 0.0


def test_the_library_says_what_spot_cannot_climb():
    from robotforge.design import library

    assert library.get("spot").max_slope == pytest.approx(0.24)
    assert library.get("go2").max_slope is None


def test_tall_humanoids_turn_further_per_step_than_small_ones():
    """A Booster T1 (0.55 m legs) turned 10 degrees a second at the small robots' 0.2 rad a step."""
    from robotforge.behaviors.humanoid import MAX_TURN_PER_STEP, turn_per_step

    small = turn_per_step(math.sqrt(0.3 / 0.2), 0.3)
    tall = turn_per_step(math.sqrt(0.55 / 0.2), 0.55)
    assert small == 0.2
    assert 0.2 < tall <= MAX_TURN_PER_STEP


def test_told_to_run_at_stairs_a_go2_slows_for_them_and_stays_upright(pg):
    """At its top speed a Go2 toppled on 8 cm stairs (0 of 3); it now walks rough ground at 60 percent at most."""
    pg.add_terrain("stairs", pos=[1.6, 0.0], params=STAIRS)
    pg.add_robot("go2", library="go2")
    gait = pg.behaviors.move("go2", 5.0, 0.0, walker="gait") and pg.session.controllers["go2"]
    top = gait.max_speed
    trunk = _trunk(pg, "go2")
    fastest_on_steps = 0.0
    for _ in range(40):
        pg.run_fast(lambda: False, 0.5)
        assert pg.behaviors.pose("go2")["upright"]
        if 0.9 < pg.session.data.xpos[trunk][0] < 1.8:  # well onto the flight (it starts at 0.6 m, 2 m long)
            fastest_on_steps = max(fastest_on_steps, gait.speed)
        if pg.session.data.xpos[trunk][0] > 2.8:
            break
    assert pg.session.data.xpos[trunk][0] > 2.8
    assert 0.0 < fastest_on_steps <= 0.7 * top  # 60 percent, and a margin for how it blends in
