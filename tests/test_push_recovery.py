"""Quadrupeds on their built-in gait take a push: a shove or a drag in the viewer steps them aside, never over.

Before push recovery, a standing template dog and a standing Go2 rolled onto their backs when dragged 0.4 m
sideways, and trotting ones did when grabbed by the back; scripts/push_bench.py measures the whole table.
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


def _on_its_gait(pg: Playground, robot: str, speed: float) -> tuple[int, int]:
    """Add `robot`, start its built-in gait at `speed` and let it settle; its trunk body and first dof."""
    pg.add_robot(robot, **ROBOTS[robot])
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


def _push(pg: Playground, robot: str, speed: float, push: str) -> dict[str, float]:
    """Push `robot` to its left while on its gait and watch 5 s: the worst tilt, and where and how it ended."""
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

    if push == "shove":
        data.qvel[dof : dof + 3] += 0.5 * left
    else:  # 0.4 m over 2 s with the viewer's drag spring, by the trunk's origin or by its back
        grab = data.xpos[trunk] + rot @ [0.0, 0.0, 0.04 if push == "drag by the back" else 0.0]
        session.start_drag(trunk, grab)
        for k in range(1, 41):
            session.move_drag(grab + 0.01 * k * left + 0.05 * k * speed * ahead)  # a hand that walks along
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


@pytest.mark.parametrize("name", ["DOG", "HEXAPOD"])
def test_robots_built_from_parts_go_along_with_a_drag(pg, monkeypatch, name):
    import test_builder

    monkeypatch.setitem(ROBOTS, "built", {"build": getattr(test_builder, name)})
    out = _push(pg, "built", 0.0, "drag by the back")
    assert out["worst"] < 30, f"tilted {out['worst']:.0f} degrees"  # the built dog rolled over before
    assert out["tilt"] < 5 and out["sideways"] > 0.25
