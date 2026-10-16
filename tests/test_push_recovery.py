"""Quadrupeds on their built-in gait take a push: a shove or a drag in the viewer steps them aside, never over.

Before push recovery, a standing template dog and a standing Go2 rolled onto their backs when dragged 0.4 m
sideways, and trotting ones did when grabbed by the back; scripts/push_bench.py measures the whole table.
Before they stood on their gait from the start, so did a dog or a Go2 dragged in a world just opened.
The Go2 is the offline stand-in from tests/fixtures/menagerie.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import quadruped
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
ROBOTS = {"dog": {"template": "quadruped"}, "go2": {"library": "go2"}}
TROT = 0.3  # m/s


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.new_world("push", "grass")
    return playground


def _on_its_gait(pg: Playground, robot: str, speed: float | None) -> tuple[int, int]:
    """Add `robot`, start its built-in gait at `speed` (None: never move it) and let it settle; its trunk body and
    first dof."""
    pg.add_robot(robot, **ROBOTS[robot])
    if speed is not None:
        pg.behaviors.move(robot, speed, 0.0, walker="gait")
    pg.run_fast(lambda: False, 2.0)
    m = pg.session.model
    for j in range(m.njnt):
        body = int(m.jnt_bodyid[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.body(body).name.startswith(f"{robot}/"):
            return body, int(m.jnt_dofadr[j])
    raise AssertionError(f"{robot} has no free joint")


def _tilt(pg: Playground, trunk: int) -> float:
    return math.degrees(math.acos(float(np.clip(pg.session.data.xmat[trunk][8], -1.0, 1.0))))


def _push(
    pg: Playground, robot: str, speed: float | None, push: str, distance: float = 0.4, seconds: float = 2.0
) -> dict[str, float]:
    """Push `robot` to its left while on its gait (speed None: never moved) and watch 5 s: the worst tilt, and
    where and how it ended. A shove is 0.5 m/s unless it says ("shove 0.7"); a drag pulls `distance` m over
    `seconds`."""
    trunk, dof = _on_its_gait(pg, robot, speed)
    session, data = pg.session, pg.session.data
    rot = data.xmat[trunk].reshape(3, 3).copy()
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    left, ahead = np.array([-math.sin(yaw), math.cos(yaw), 0.0]), np.array([math.cos(yaw), math.sin(yaw), 0.0])
    start = data.xpos[trunk].copy()
    worst = 0.0

    def run(seconds: float) -> None:
        nonlocal worst
        for _ in range(round(seconds / session.model.opt.timestep)):
            session.step()
            worst = max(worst, _tilt(pg, trunk))

    if push.startswith("shove"):
        data.qvel[dof : dof + 3] += float(push.removeprefix("shove") or 0.5) * left
    else:  # with the viewer's drag spring, by the trunk's origin or by its back
        grab = data.xpos[trunk] + rot @ [0.0, 0.0, 0.04 if push == "drag by the back" else 0.0]
        session.start_drag(trunk, grab)
        steps = round(20 * seconds)
        walked = ahead * (speed or 0.0) * seconds  # a hand that walks along
        for k in range(1, steps + 1):
            session.move_drag(grab + (left * distance + walked) * k / steps)
            run(0.05)
        session.end_drag()
    run(4.0)
    before = data.xpos[trunk].copy()
    run(1.0)
    return {
        "worst": worst,
        "tilt": _tilt(pg, trunk),
        "sideways": float((data.xpos[trunk] - start) @ left),
        "last_second": float(np.linalg.norm(data.xpos[trunk][:2] - before[:2])),
    }


@pytest.mark.parametrize("push", ["shove", "drag", "drag by the back"])
@pytest.mark.parametrize("speed", [0.0, TROT], ids=["standing", "trotting"])
@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_pushed_quadruped_steps_aside_and_stays_on_its_feet(pg, robot, speed, push):
    out = _push(pg, robot, speed, push)
    assert out["worst"] < 30, f"tilted {out['worst']:.0f} degrees"  # before: rolled onto its back when dragged
    assert out["tilt"] < 5  # level again
    if speed == 0.0:
        assert out["last_second"] < 0.03  # and standing still where the push left it
    else:
        assert out["last_second"] > 0.5 * TROT  # and trotting on
    if push != "shove":
        assert out["sideways"] > 0.25  # it went along with the hand instead of fighting it


@pytest.mark.parametrize("push", ["drag", "drag by the back", "shove", "shove 0.7"])
@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_quadruped_never_moved_takes_a_push_from_the_moment_the_world_opens(pg, robot, push):
    """Regression: a quadruped got its gait only with its first move, so a beginner who opened a world and dragged
    the dog or the Go2 rolled it onto its back (180 degrees). It stands on its gait from the start now."""
    out = _push(pg, robot, None, push)
    assert out["worst"] < 30, f"tilted {out['worst']:.0f} degrees"
    assert out["tilt"] < 5 and out["last_second"] < 0.03  # level and still again
    if push.startswith("drag"):
        assert out["sideways"] > 0.25  # it went along with the hand


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_quadruped_never_moved_stands_as_it_did_without_a_gait(pg, tmp_path, robot):
    """Its gait from the start stands it still: no steps, no drift, and the height and joint angles its servos
    alone gave it before (a world opened again, its gait taken away, is that stand)."""
    trunk, _ = _on_its_gait(pg, robot, None)
    gait = pg.session.controllers[robot]
    assert isinstance(gait, quadruped.QuadrupedGait)
    plain = Playground.with_demo()
    plain.open_world(str(pg.store.dir))
    plain.session.set_controller(robot, None)
    plain.run_fast(lambda: False, 2.0)
    start, heading = pg.session.data.xpos[trunk].copy(), pg.behaviors.pose(robot)["heading_deg"]
    stepped = 0.0
    for _ in range(500):  # 10 s
        pg.run_fast(lambda: False, 0.02)
        stepped = max(stepped, gait._amplitude)
    plain.run_fast(lambda: False, 10.0)
    assert stepped < 0.05
    assert np.linalg.norm(pg.session.data.xpos[trunk][:2] - start[:2]) < 0.002
    assert abs(pg.behaviors.pose(robot)["heading_deg"] - heading) < 0.2
    m = pg.session.model
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    legs = [
        int(m.jnt_qposadr[j])
        for j in range(m.njnt)
        if m.jnt_type[j] == hinge and m.joint(j).name.startswith(f"{robot}/")
    ]
    assert pg.session.data.xpos[trunk][2] == pytest.approx(plain.session.data.xpos[trunk][2], abs=0.001)
    assert np.degrees(np.abs(pg.session.data.qpos[legs] - plain.session.data.qpos[legs])).max() < 0.5


def test_a_quadruped_standing_still_on_its_gait_solves_no_ik(pg, monkeypatch):
    """Found in review: every quadruped in a world stands on its gait, and solving the same foot targets 100 times a
    second halved the sim's speed (one dog: real-time factor 173 -> 90). Standing still it solves nothing, also once a
    shove it stepped for has died down; while it steps it solves as before."""
    from robotforge.behaviors.ik import IKSolver

    solves = []
    solve = IKSolver.solve
    monkeypatch.setattr(IKSolver, "solve", lambda self, *a, **k: solves.append(1) or solve(self, *a, **k))
    _, dof = _on_its_gait(pg, "dog", None)
    solves.clear()
    pg.run_fast(lambda: False, 2.0)
    assert not solves
    pg.session.data.qvel[dof + 1] += 0.7  # a shove it steps for
    pg.run_fast(lambda: False, 1.0)
    assert len(solves) > 100
    pg.run_fast(lambda: False, 6.0)
    solves.clear()
    pg.run_fast(lambda: False, 2.0)
    assert not solves


@pytest.mark.parametrize(("robot", "angle"), [("dog", 15), ("go2", 15), ("go2", 18), ("go2", 20)])
def test_a_quadruped_put_down_on_a_slope_does_not_take_it_for_a_push(pg, robot, angle):
    """Regression found while measuring: on a 15 degree ramp a Go2 slid 12 cm down and back while it landed, and
    our dog creeps down at about 1 cm/s on its servos; both read as a push, and the robot stepped off the ramp
    (0.9 m) or about it for half a minute. It stands there as it would on its servos alone.
    Regression found in review: its gait leveled the trunk while it stood, and a Go2 on a 17 to 20 degree ramp that
    stood there on its servos went over backwards (180 degrees); it holds the stance its servos would now."""
    pg.add_terrain("ramp", pos=[0.0, 0.0], params={"angle": angle, "length": 1.5, "direction": "up"})
    pg.add_robot(robot, **ROBOTS[robot], pos=[-0.4, 0.0])
    gait = pg.session.controllers[robot]
    plain = Playground.with_demo()  # the same world, the robot on its servos alone
    plain.open_world(str(pg.store.dir))
    plain.session.set_controller(robot, None)
    trunk = pg.behaviors._root_body(robot)
    stepped = worst = 0.0
    for _ in range(400):  # 8 s
        pg.run_fast(lambda: False, 0.02)
        stepped = max(stepped, gait._amplitude)
        worst = max(worst, _tilt(pg, trunk))
    plain.run_fast(lambda: False, 8.0)
    assert stepped < 0.05
    assert worst < 90 and pg.behaviors.pose(robot)["upright"]
    assert _tilt(pg, trunk) == pytest.approx(_tilt(plain, trunk), abs=1.0)


@pytest.mark.parametrize(("robot", "angle", "tilted"), [("dog", 10, 8.0), ("go2", 15, 13.0)])
def test_a_quadruped_walked_on_a_slope_and_stopped_stands_leveled(pg, robot, angle, tilted):
    """Regression found in review: holding the servo stance is for robots nobody moved. A Go2 walked up a 15 degree
    ramp and stopped stood 10.4 degrees tilted on base 7a4aa92 and 18.7 with that stance, sliding 55 mm instead of
    12; our dog on 10 degrees 6.3 and 10.3. A robot that was moved stands leveled, as it always did."""
    pg.add_terrain("ramp", pos=[0.0, 0.0], params={"angle": angle, "length": 1.5, "direction": "up"})
    pg.add_robot(robot, **ROBOTS[robot], pos=[-0.4, 0.0])
    pg.run_fast(lambda: False, 1.0)
    pg.behaviors.move(robot, 0.1, 0.0, walker="gait")
    pg.run_fast(lambda: False, 1.5)
    pg.behaviors.stop(robot)
    stopped = np.array(pg.behaviors.pose(robot)["position"][:2])
    pg.run_fast(lambda: False, 10.0)
    trunk = pg.behaviors._root_body(robot)
    assert _tilt(pg, trunk) < tilted
    if robot == "go2":
        assert np.linalg.norm(np.array(pg.behaviors.pose(robot)["position"][:2]) - stopped) < 0.025


def test_a_go2_standing_on_a_steep_ramp_and_then_moved_settles_into_its_leveled_stand(pg):
    """Its first move (here at speed 0) turns the leveling on under a gait that had settled long before; the stance
    changing under it read as a push, and a Go2 on a 20 degree ramp stepped down it and off (worst 37 degrees). It
    settles anew, as the gait base 7a4aa92 started at that move did: worst 19.8, then 14 degrees on the ramp."""
    pg.add_terrain("ramp", pos=[0.0, 0.0], params={"angle": 20, "length": 1.5, "direction": "up"})
    pg.add_robot("go2", **ROBOTS["go2"], pos=[-0.4, 0.0])
    pg.run_fast(lambda: False, 3.0)
    pg.behaviors.move("go2", 0.0, 0.0, walker="gait")
    trunk = pg.behaviors._root_body("go2")
    worst = 0.0
    for _ in range(250):  # 5 s
        pg.run_fast(lambda: False, 0.02)
        worst = max(worst, _tilt(pg, trunk))
    assert worst < 25
    assert _tilt(pg, trunk) > 10  # still standing on the ramp, not walked off it


def _stepping(pg: Playground, robot: str, seconds: float) -> tuple[float, float]:
    """Run `seconds` and return how long `robot`'s gait stepped in all, and in its longest stretch."""
    gait = pg.session.controllers[robot]
    total = run = longest = 0.0
    for _ in range(round(seconds / 0.02)):
        pg.run_fast(lambda: False, 0.02)
        run = run + 0.02 if gait._amplitude > 0.05 else 0.0
        total, longest = total + (0.02 if run else 0.0), max(longest, run)
    return total, longest


def test_a_dog_never_moved_stands_still_on_stepping_stones(pg):
    """Regression found in review: put down on the stepping stones, our dog slid off a stone's edge at 3 cm/s while it
    landed; it settled while still sliding, that read as a push, and it stepped for 28 s, 0.5 m about the stones and
    55 degrees over (on its servos it stood: 11 mm). It settles once still now."""
    pg.add_terrain("stones", pos=[0.0, 0.0])
    pg.add_robot("dog", **ROBOTS["dog"])
    trunk = pg.behaviors._root_body("dog")
    pg.run_fast(lambda: False, 2.0)
    start = pg.session.data.xpos[trunk][:2].copy()
    total, _ = _stepping(pg, "dog", 20.0)
    assert total == 0.0
    assert np.linalg.norm(pg.session.data.xpos[trunk][:2] - start) < 0.03


def test_a_standing_recovery_that_never_calms_is_cut_off(pg, monkeypatch):
    """On stones stepping rocks the trunk itself, so a recovery started there may never calm; standing, it is cut off
    after STAND_RECOVER_MAX like a walking one. Settling as loosely as before brings back the false start above."""
    monkeypatch.setattr(quadruped, "SETTLE_CALM", 1.0)
    pg.add_terrain("stones", pos=[0.0, 0.0])
    pg.add_robot("dog", **ROBOTS["dog"])
    total, longest = _stepping(pg, "dog", 25.0)
    assert total > 1.0  # it did take the slide for a push
    assert longest < quadruped.STAND_RECOVER_MAX + 1.0 and total < 2 * quadruped.STAND_RECOVER_MAX
    assert pg.behaviors.pose("dog")["upright"]


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_a_quadruped_on_its_back_lies_still(pg, robot):
    """Regression found in review: a fresh dog flipped by a fast drag kicked its legs on its back for good (stepping
    29.9 s of 30); a robot that is down has nothing to step for. It lies still, as it did on its servos alone."""
    trunk, dof = _on_its_gait(pg, robot, None)
    data = pg.session.data
    adr = pg.session.model.jnt_qposadr[pg.session.model.body_jntadr[trunk]]
    data.qpos[adr + 2] += 0.15
    data.qpos[adr + 3 : adr + 7] = [0.0, 1.0, 0.0, 0.0]  # upside down, dropped onto its back
    data.qvel[dof : dof + 6] = 0.0
    _stepping(pg, robot, 3.0)
    assert _tilt(pg, trunk) > 90
    total, _ = _stepping(pg, robot, 10.0)
    assert total == 0.0


def test_a_dog_dragged_longer_than_a_recovery_may_last_keeps_stepping_along(pg):
    """The cut-off is for recoveries nobody is causing: a hand that pulls the robot along gets steps for as long as it
    pulls (here 1.6 m over 8 s; a cut-off at STAND_RECOVER_MAX would leave it a stiff table under the hand)."""
    trunk, _ = _on_its_gait(pg, "dog", None)
    session, data = pg.session, pg.session.data
    grab = data.xpos[trunk] + data.xmat[trunk].reshape(3, 3) @ [0.0, 0.0, 0.04]
    left = data.xmat[trunk].reshape(3, 3)[:, 1].copy()
    session.start_drag(trunk, grab)
    worst = 0.0
    for k in range(1, 161):
        session.move_drag(grab + 0.01 * k * left)
        pg.run_fast(lambda: False, 0.05)
        worst = max(worst, _tilt(pg, trunk))
    session.end_drag()
    assert pg.session.controllers["dog"]._recovery > 0.5  # still stepping with the hand after 8 s
    pg.run_fast(lambda: False, 4.0)
    assert worst < 30 and _tilt(pg, trunk) < 5
    assert float((data.xpos[trunk] - grab) @ left) > 1.2


def test_a_nudge_the_stance_soaks_up_takes_no_steps(pg):
    """Stepping lifts feet, so a robot that can simply stand through a push stays planted, as it always did."""
    trunk, dof = _on_its_gait(pg, "dog", 0.0)
    gait, data = pg.session.controllers["dog"], pg.session.data
    start = data.xpos[trunk][:2].copy()
    data.qvel[dof + 1] += 0.3
    stepped = 0.0
    for _ in range(100):
        pg.run_fast(lambda: False, 0.02)
        stepped = max(stepped, gait._amplitude)
    assert stepped < 0.05
    assert np.linalg.norm(data.xpos[trunk][:2] - start) < 0.02


@pytest.mark.parametrize("robot", list(ROBOTS))
def test_walking_undisturbed_never_counts_as_a_push(pg, robot):
    """Recovery places the feet differently, so a plain trot, a hard turn or a rocking pace must not set it off;
    the gait lesson's numbers depend on that."""
    _on_its_gait(pg, robot, TROT)
    gait = pg.session.controllers[robot]
    assert isinstance(gait, quadruped.QuadrupedGait)
    seen = 0.0
    for speed, turn, pattern in ((TROT, 0.0, "trot"), (gait.max_speed, 2.0, "trot"), (TROT, 0.0, "pace")):
        gait.set_pattern(quadruped.PATTERNS[pattern])
        gait.command(speed, turn)
        for _ in range(150):
            pg.run_fast(lambda: False, 0.02)
            seen = max(seen, gait._recovery)
    assert seen == 0.0
    assert pg.behaviors.pose(robot)["upright"]


def test_a_dog_walking_into_a_wall_is_not_kept_recovering(pg):
    """Regression found in review: blocked by a stair riser, a trotting dog leaned and wobbled like a pushed one,
    recovered without end (its strides following the wobble sideways) and never climbed; the obstacle course fell
    from 12 m to 4.6 m. Pinned against a wall it now recovers for at most WALK_RECOVER_MAX at a time (7.1 s before),
    and spends the rests on its plain gait."""
    pg.add_object("wall", shape="box", size=[0.6, 4.0, 0.3], pos=[1.0, 0.0])
    _on_its_gait(pg, "dog", TROT)
    gait = pg.session.controllers["dog"]
    run = longest = 0.0
    for _ in range(650):
        pg.run_fast(lambda: False, 0.02)
        run = run + 0.02 if gait._recovery > 0.5 else 0.0
        longest = max(longest, run)
    assert longest < quadruped.WALK_RECOVER_MAX + 0.5
    assert pg.behaviors.pose("dog")["upright"]


def test_a_rewind_mid_recovery_does_not_keep_it_stepping(pg):
    """Recovery times are sim-clock times; a reset of the data under a live gait used to leave it stepping in place
    until the old clock came round again (8 s in review)."""
    trunk, dof = _on_its_gait(pg, "dog", 0.0)
    gait, session = pg.session.controllers["dog"], pg.session
    pg.run_fast(lambda: False, 8.0)
    session.data.qvel[dof + 1] += 0.7  # a hard shove: it steps
    pg.run_fast(lambda: False, 0.3)
    assert gait._recovery > 0.5
    mujoco.mj_resetDataKeyframe(session.model, session.data, 0)  # time back to 0, the gait object kept
    pg.run_fast(lambda: False, quadruped.SETTLE_TIME + 1.5)
    assert gait._recovery < 0.05 and gait._amplitude < 0.05


@pytest.mark.parametrize("speed", [0.0, None], ids=["standing", "never moved"])
@pytest.mark.parametrize("name", ["DOG", "HEXAPOD"])
def test_robots_built_from_parts_go_along_with_a_drag(pg, monkeypatch, name, speed):
    import test_builder

    monkeypatch.setitem(ROBOTS, "built", {"build": getattr(test_builder, name)})
    out = _push(pg, "built", speed, "drag by the back")
    assert out["worst"] < 30, f"tilted {out['worst']:.0f} degrees"  # the built dog rolled over before
    assert out["tilt"] < 5 and out["sideways"] > 0.25


def test_a_curb_in_the_way_is_climbed_not_taken_for_a_push(pg):
    """The obstacle-course regression on a single 4 cm curb: taken for a push, the dog's strides followed the wobble
    along the curb instead of up it until it rolled over. With recovery cut off it climbs, and across the top
    rather than around it."""
    curb = {"steps": 1, "step_height": 0.04, "step_depth": 0.3, "direction": "up", "landing": 1.5}
    pg.add_terrain("stairs", pos=[1.35, 0.0], params=curb)  # 0.6 m ahead; the top is 1.5 m long and 1.2 m wide
    pg.add_robot("dog", **ROBOTS["dog"])
    top = next(r["max_speed"] for r in pg.behaviors.robots() if r["name"] == "dog")
    pg.behaviors.move("dog", 0.6 * top, 0.0, walker="gait")  # as in scripts/terrain_bench.py
    gait, data = pg.session.controllers["dog"], pg.session.data
    trunk = pg.session.model.body("dog/trunk").id
    height = float(data.xpos[trunk][2])
    worst = rise = aside = run = longest = 0.0
    while data.time < 25.0 and data.xpos[trunk][0] < 2.3:  # until past the far end of the top
        pg.run_fast(lambda: False, 0.05)
        worst = max(worst, _tilt(pg, trunk))
        rise = max(rise, float(data.xpos[trunk][2]) - height)
        aside = max(aside, abs(float(data.xpos[trunk][1])))
        run = run + 0.05 if gait._recovery > 0.5 else 0.0
        longest = max(longest, run)
    assert worst < 30, f"tilted {worst:.0f} degrees"  # before the cut-off: rolled over at the curb
    assert data.xpos[trunk][0] > 2.3, "stalled at the curb"
    assert rise > 0.025 and aside < 0.6  # up and across the top, not around it
    assert longest < quadruped.WALK_RECOVER_MAX + 0.5


def test_a_long_drag_keeps_a_trotting_dog_recovering(pg):
    """The cut-off is for a robot that is blocked, not for one a hand is still dragging: cut off regardless, a
    trotting dog dragged 0.8 m by its back over 6 s fought the hand once its recovery ran out and rolled over."""
    out = _push(pg, "dog", TROT, "drag by the back", distance=0.8, seconds=6.0)
    assert out["worst"] < 30, f"tilted {out['worst']:.0f} degrees"
    assert out["tilt"] < 5
    assert out["last_second"] > 0.5 * TROT
    assert out["sideways"] > 0.5


@pytest.mark.parametrize("robot", list(ROBOTS))
@pytest.mark.parametrize(("distance", "seconds"), [(0.4, 2.0), (0.6, 3.0)])
def test_a_never_moved_quadruped_pulled_forward_by_its_back_walks_along(pg, robot, distance, seconds):
    """Review of round 32: every drag in the bench pulled to the left. Pulled forwards by its back, a never-moved
    dog's strides braked against the hand, it pitched up to 61 degrees and went onto its back (0.6 m over 3 s)."""
    trunk, _ = _on_its_gait(pg, robot, None)
    session, data = pg.session, pg.session.data
    rot = data.xmat[trunk].reshape(3, 3).copy()
    ahead = np.array([rot[0, 0], rot[1, 0], 0.0]) / np.hypot(rot[0, 0], rot[1, 0])
    grab = data.xpos[trunk] + rot @ [0.0, 0.0, 0.04]
    start = data.xpos[trunk].copy()
    worst = 0.0
    session.start_drag(trunk, grab)
    steps = round(20 * seconds)
    for k in range(1, steps + 1):
        session.move_drag(grab + ahead * distance * k / steps)
        for _ in range(round(0.05 / session.model.opt.timestep)):
            session.step()
            worst = max(worst, _tilt(pg, trunk))
    session.end_drag()
    for _ in range(round(4.0 / session.model.opt.timestep)):
        session.step()
        worst = max(worst, _tilt(pg, trunk))
    assert worst < 30, f"tilted {worst:.0f} degrees"
    # it goes along (the heavier Go2 less far: the viewer's drag spring pulled it 0.18 m of 0.4)
    assert _tilt(pg, trunk) < 5 and float((data.xpos[trunk] - start) @ ahead) > 0.25 * distance
