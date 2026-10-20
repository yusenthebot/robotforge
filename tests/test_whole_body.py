"""Whole-body balance: a robot of any build stands on its own feet, under a load and a push."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import Kind, robot_kind, stance_ik
from robotforge.behaviors.manager import BehaviorError
from robotforge.behaviors.whole_body import BalanceError, WholeBodyBalancer, _nnls
from robotforge.playground import Playground

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


def _leg(side: float, roll: bool = True) -> str:
    """One leg of the test biped: hip roll, hip pitch, knee, ankle pitch, ankle roll, then a flat foot."""
    s = "l" if side > 0 else "r"
    hip_roll = f'<joint name="{s}_hip_roll" axis="1 0 0" range="-0.6 0.6" damping="1"/>' if roll else ""
    return f"""
      <body name="{s}_hip" pos="0 {side * 0.1} -0.1">
        {hip_roll}
        <geom type="sphere" size="0.03" mass="0.5"/>
        <body name="{s}_thigh">
          <joint name="{s}_hip_pitch" axis="0 1 0" range="-1.5 1.5" damping="1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.035" mass="2.5"/>
          <body name="{s}_shin" pos="0 0 -0.3">
            <joint name="{s}_knee" axis="0 1 0" range="0 2.2" damping="1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="1.5"/>
            <body name="{s}_ankle" pos="0 0 -0.3">
              <joint name="{s}_ankle_pitch" axis="0 1 0" range="-0.8 0.8" damping="1"/>
              <joint name="{s}_ankle_roll" axis="1 0 0" range="-0.5 0.5" damping="1"/>
              <geom type="box" size="0.11 0.045 0.015" pos="0.04 0 -0.045" mass="0.6" friction="1"/>
            </body>
          </body>
        </body>
      </body>"""


def biped_xml(servo: bool = True, lean: float = 0.0, roll: bool = True) -> str:
    """A 15 kg biped. `servo`: position servos (else torque motors); `lean`: how far its weight starts behind the
    middle of its feet, as a keyframe drawn for looks leaves it (m); `roll`: legs with a hip roll joint (five joints,
    which the walking gait can drive) or without (four, which no gait fits)."""
    parts = (
        ("hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
        if roll
        else ("hip_pitch", "knee", "ankle_pitch", "ankle_roll")
    )
    joints = [f"{s}_{j}" for s in "lr" for j in parts]
    if servo:
        acts = "".join(
            f'<position name="{j}" joint="{j}" kp="250" kv="6" forcerange="-80 80" ctrlrange="-3 3"/>' for j in joints
        )
    else:
        acts = "".join(f'<motor name="{j}" joint="{j}" gear="1" ctrlrange="-80 80"/>' for j in joints)
    return f"""
<mujoco>
  <worldbody>
    <body name="pelvis" pos="{-lean} 0 0.78">
      <freejoint/>
      <geom type="box" size="0.06 0.12 0.05" mass="4"/>
      <body name="torso" pos="0 0 0.25">
        <geom type="box" size="0.07 0.14 0.2" mass="6"/>
      </body>{_leg(1.0, roll)}{_leg(-1.0, roll)}
    </body>
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground.with_demo()
    playground.new_world("lab")
    return playground


def _biped(pg: Playground, tmp_path: Path, servo: bool = True, lean: float = 0.0, roll: bool = True) -> str:
    path = tmp_path / "biped.xml"
    path.write_text(biped_xml(servo, lean, roll), encoding="utf-8")
    pg.add_robot("bot", file=str(path))
    return "bot"


def _trunk(pg: Playground, robot: str = "bot") -> int:
    return pg.session.model.body(f"{robot}/pelvis").id


def _tilt(pg: Playground, robot: str = "bot") -> float:
    d = pg.session.data
    return math.degrees(math.acos(float(d.xmat[_trunk(pg, robot)][8])))


def _shove(pg: Playground, speed: float, direction: tuple[float, float], robot: str = "bot") -> None:
    m = pg.session.model
    dof = int(m.jnt_dofadr[m.body_jntadr[_trunk(pg, robot)]])
    pg.session.data.qvel[dof : dof + 2] += speed * np.array(direction)


# ---- the kind of a robot comes from where it stands ---------------------------------------------------------


def test_a_robot_on_two_feet_is_a_humanoid_whatever_its_joints_are_called(pg, tmp_path):
    _biped(pg, tmp_path)
    assert robot_kind(pg.session.model, "bot") == Kind.HUMANOID
    info = next(r for r in pg.behaviors.robots() if r["name"] == "bot")
    assert info["kind"] == "humanoid"


def test_a_dog_is_still_a_dog_and_a_rover_a_rover(pg):
    pg.add_robot("dog", template="quadruped")
    pg.add_robot("rover", template="rover", pos=[2.0, 0.0])
    assert robot_kind(pg.session.model, "dog") == Kind.QUADRUPED
    assert robot_kind(pg.session.model, "rover") == Kind.ROVER


# ---- standing -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("servo", [True, False])
def test_a_biped_stands_on_servos_or_on_torque_motors(pg, tmp_path, servo):
    _biped(pg, tmp_path, servo=servo)
    assert isinstance(pg.session.controllers["bot"], WholeBodyBalancer)
    pg.run_fast(lambda: False, 6.0)
    d, m = pg.session.data, pg.session.model
    assert _tilt(pg) < 5.0
    assert d.xpos[_trunk(pg)][2] > 0.7  # 0.78 at the start
    assert np.isfinite(d.qpos).all()
    assert m.opt.timestep > 0  # the model was not touched


def test_a_biped_drawn_with_its_weight_on_its_heels_is_stood_up_before_it_starts(pg, tmp_path):
    """A keyframe with the weight behind the feet: the pose it starts in is moved until the weight is over them."""
    _biped(pg, tmp_path, lean=0.12)
    m, d = pg.session.model, pg.session.data
    mujoco.mj_forward(m, d)
    tree = np.isin(m.body_rootid, [_trunk(pg)])
    feet = stance_ik.plan_feet(m, d, tree)
    assert len(feet) == 2 and not stance_ik.needs_balancing(d.subtree_com[_trunk(pg)], feet, d)
    pg.run_fast(lambda: False, 6.0)
    assert _tilt(pg) < 5.0


def test_a_biped_takes_a_shove_and_a_load(pg, tmp_path):
    _biped(pg, tmp_path)
    pg.run_fast(lambda: False, 3.0)
    for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _shove(pg, 0.2, direction)
        pg.run_fast(lambda: False, 4.0)
        assert _tilt(pg) < 8.0, direction
        assert pg.session.data.xpos[_trunk(pg)][2] > 0.7
    m, d = pg.session.model, pg.session.data
    pelvis = _trunk(pg)
    mass = float(m.body_mass[pelvis])
    for step in range(1, 11):  # a backpack of a fifth of its weight, put on over a second
        m.body_mass[pelvis] = mass + 3.0 * step / 10
        mujoco.mj_setConst(m, d)
        pg.run_fast(lambda: False, 0.1)
    pg.run_fast(lambda: False, 4.0)
    assert _tilt(pg) < 8.0 and d.xpos[pelvis][2] > 0.7


def test_a_fallen_biped_stood_back_up_by_hand_is_held_again(pg, tmp_path):
    _biped(pg, tmp_path)
    ctrl = pg.session.controllers["bot"]
    m, d = pg.session.model, pg.session.data
    pg.run_fast(lambda: False, 3.0)
    _shove(pg, 3.0, (1, 0))  # far more than its feet can take
    pg.run_fast(lambda: False, 4.0)
    assert ctrl.fallen and ctrl.status()["behavior"] == "fallen"
    mujoco.mj_resetDataKeyframe(m, d, 0)  # stood up by hand: the controller is kept, the robot is put back
    mujoco.mj_forward(m, d)
    pg.run_fast(lambda: False, 4.0)
    assert not ctrl.fallen and _tilt(pg) < 8.0 and d.xpos[_trunk(pg)][2] > 0.7


def test_a_standing_humanoid_says_what_holds_it_up_and_that_it_has_no_gait(pg, tmp_path):
    _biped(pg, tmp_path, roll=False)
    ctrl = pg.session.controllers["bot"]
    assert ctrl.status()["controller"] == "whole-body balance"
    pg.run_fast(lambda: False, 1.0)
    assert pg.behaviors.pose("bot")["upright"]
    with pytest.raises(BehaviorError, match="stand"):
        pg.behaviors.move("bot", 0.2, 0.0)


def test_move_does_not_swap_the_balance_of_a_file_humanoid_for_a_gait_that_drops_it(pg, tmp_path):
    """The legs of this biped fit the gait search (hip roll, hip pitch, knee, ankle), but nobody has shown that gait to
    walk a robot of a file: every real one tried fell within seconds. So move refuses and the robot stays up."""
    _biped(pg, tmp_path, roll=True)
    assert isinstance(pg.session.controllers["bot"], WholeBodyBalancer)
    info = next(r for r in pg.behaviors.robots() if r["name"] == "bot")
    assert "max_speed" not in info and "move cannot walk" in info["walk_hint"]
    with pytest.raises(BehaviorError, match="whole-body balance"):
        pg.behaviors.move("bot", 0.2, 0.0)
    with pytest.raises(BehaviorError, match="whole-body balance"):
        pg.behaviors.strike_pose("bot", "wave")
    with pytest.raises(BehaviorError, match="whole-body balance"):
        pg.behaviors.navigate("bot", [1.0, 0.0])
    assert isinstance(pg.session.controllers["bot"], WholeBodyBalancer)
    pg.run_fast(lambda: False, 3.0)
    assert pg.behaviors.pose("bot")["upright"]


# ---- the parts -------------------------------------------------------------------------------------------


def test_nnls_meets_its_optimality_conditions():
    rng = np.random.default_rng(3)
    for _ in range(8):
        a, b = rng.normal(size=(18, 10)), rng.normal(size=18)
        x = _nnls(a, b)
        assert x.min() >= 0.0
        gradient = a.T @ (b - a @ x)
        assert gradient[x > 0].max(initial=0.0) < 1e-8 and abs(gradient[x > 0]).max(initial=0.0) < 1e-8
        assert gradient[x == 0].max(initial=-1.0) <= 1e-8


def test_the_centre_of_a_foot_print_is_its_area_not_its_points():
    # A foot with three points at its toe and one at its heel: the mean leans to the toe, the area does not.
    points = np.array([[0.0, 0.0, 0.0], [0.2, -0.05, 0.0], [0.2, 0.0, 0.0], [0.2, 0.05, 0.0], [0.0, 0.05, 0.0]])
    assert np.mean(points[:, 0]) > 0.12
    assert stance_ik.hull_centre(points)[0] == pytest.approx(0.1, abs=0.015)


def test_a_robot_with_nothing_to_stand_on_says_so(pg):
    pg.add_robot("arm", template="arm")
    with pytest.raises(BalanceError, match="floating"):
        WholeBodyBalancer(pg.session.model, "arm", pg.behaviors.home_qpos())


# ---- robots from the library and the standing humanoids' controllers ---------------------------------------


def test_the_tuned_humanoids_keep_their_walking_gait(pg):
    pg.add_robot("op3", library="op3", pos=[2.0, 0.0])
    ctrl = pg.session.controllers["op3"]
    assert not isinstance(ctrl, WholeBodyBalancer)


def test_a_learned_walker_stands_by_balance_under_a_load_and_walks_on_the_first_move(pg):
    """G1's policy falls under a fifth of its own mass; whole-body balance holds that, and the first move gives the
    robot back to its policy."""
    pg.add_robot("g1", library="g1")
    m, d = pg.session.model, pg.session.data
    assert isinstance(pg.session.controllers["g1"], WholeBodyBalancer)
    trunk = int(m.body("g1/pelvis").id)
    pg.run_fast(lambda: False, 2.0)
    was = float(m.body_mass[trunk])
    for step in range(1, 11):  # taken up over a second, as a hand puts a load on
        m.body_mass[trunk] = was + 0.2 * float(m.body_subtreemass[trunk]) * step / 10
        mujoco.mj_setConst(m, d)
        pg.run_fast(lambda: False, 0.1)
    pg.run_fast(lambda: False, 3.0)
    assert math.degrees(math.acos(float(d.xmat[trunk][8]))) < 15.0 and d.xpos[trunk][2] > 0.6
    m.body_mass[trunk] = was
    mujoco.mj_setConst(m, d)
    pg.behaviors.move("g1", 0.3, 0.0)
    assert not isinstance(pg.session.controllers["g1"], WholeBodyBalancer)


def test_a_learned_walker_with_an_arm_on_it_stays_on_its_policy(pg):
    """The arm's controller and the balance would both drive the joints of one tree."""
    pg.add_robot("g1", library="g1", mounts=[{"library": "piper"}])
    assert not isinstance(pg.session.controllers["g1"], WholeBodyBalancer)
