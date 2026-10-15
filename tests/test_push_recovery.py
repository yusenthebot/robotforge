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
