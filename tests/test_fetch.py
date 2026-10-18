"""Fetching: an arm on a robot that goes picks something up out of its reach, carries it and puts it down
(behaviors/fetch.py, behaviors/reach_in.py). Offline: our own lift picker and the test stand-ins for Menagerie."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import fetch, reach_in
from robotforge.behaviors.arm import ArmMotion, Waypoint
from robotforge.behaviors.manager import BehaviorError
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("fetch")
    return playground


def _wall_world(pg: Playground) -> None:
    """A lift picker at the middle; the cube on a table behind a 2 m wall; the goal table off to the other side."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("wall", shape="wall", size=[2.0, 0.1, 0.8], pos=[0.0, 0.8])
    pg.add_object("stand", shape="table", size=[0.4, 0.4, 0.45], pos=[0.0, 1.6], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, 1.45, 0.47])
    pg.add_object("goal", shape="table", size=[0.4, 0.4, 0.45], pos=[1.6, -0.6], fixed=True)
    pg.run_fast(lambda: False, 0.5)


def _start(pg: Playground, obj: str, target):
    return pg.behaviors.pick_and_place(pg.behaviors.resolve("picker", arm=True), obj, target)


def test_a_lift_picker_goes_round_a_wall_for_a_cube_and_puts_it_on_a_table(pg):
    _wall_world(pg)
    model = pg.session.model
    errand, info = _start(pg, "cube", "goal")
    assert isinstance(errand, fetch.Errand)
    gripper = errand.arm.gripper
    gains = model.actuator_gainprm[gripper].copy(), model.actuator_biasprm[gripper].copy()
    stiffer = []
    root = pg.behaviors.root("picker")
    beyond_wall = []  # where its middle was while the wall stood between it and the cube

    def step() -> bool:
        stiffer.append(model.actuator_gainprm[gripper, 0] > gains[0][0])
        x, y = pg.session.data.xpos[root][:2]
        beyond_wall.append(abs(x) < 1.0 and 0.75 < y < 0.85)
        return errand.done

    pg.run_fast(step, fetch.TIME_LIMIT)
    pg.run_fast(lambda: False, 0.5)
    result = errand.result()
    assert errand.state == "done", result
    assert pg.behaviors.placed(info["object"], info["region"])["success"]
    to_cube, to_goal = result["went"]
    assert to_cube["arrived"] and to_goal["arrived"]
    assert to_cube["path_m"] > 2.5  # 1.5 m away as the crow flies: round the end of the wall
    assert not any(beyond_wall)  # never through it
    assert result["driven_m"] > 5.0 and result["picked_up_at"] == pytest.approx([0.0, 1.45, 0.47], abs=0.01)
    # While it ran its gripper squeezed harder and the no-slip pass ran; afterwards the world is as it was.
    assert any(stiffer)
    assert model.opt.noslip_iterations == 0
    assert np.allclose(model.actuator_gainprm[gripper], gains[0]) and np.allclose(
        model.actuator_biasprm[gripper], gains[1]
    )
    assert not errand.arm.carry_weight
    assert pg.behaviors.errand("picker") is None


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (lambda pg: pg.update_object("cube", mass=4.0), r"weighs 4\.00 kg, more than picker/arm can hold"),
        (lambda pg: pg.update_object("cube", size=[0.1, 0.1, 0.1]), r"0\.10 m across .* opens only"),
        (
            lambda pg: [
                pg.add_object(
                    f"box_{i}_{j}", shape="box", size=[0.12, 0.12, 0.12], pos=[1.45 + 0.15 * i, -0.75 + 0.15 * j, 0.51]
                )
                for i in range(3)
                for j in range(3)
            ],
            r"the goal is full: .*box_",
        ),
        (
            lambda pg: [
                pg.add_object("wall_e", shape="wall", size=[2.0, 0.1, 0.8], pos=[1.1, 1.75], yaw=90),
                pg.add_object("wall_w", shape="wall", size=[2.0, 0.1, 0.8], pos=[-1.1, 1.75], yaw=90),
                pg.add_object("wall_n", shape="wall", size=[2.3, 0.1, 0.8], pos=[0.0, 2.75]),
            ],
            r"no way for picker to where its arm would reach the cube",
        ),
    ],
    ids=["too heavy", "too big", "place full", "no way there"],
)
def test_a_fetch_it_cannot_do_is_refused_before_the_robot_moves(pg, setup, message):
    _wall_world(pg)
    setup(pg)
    pg.run_fast(lambda: False, 0.5)
    before = pg.behaviors.pose("picker")["position"]
    with pytest.raises(BehaviorError, match=message):
        _start(pg, "cube", "goal")
    assert pg.behaviors.errand("picker") is None
    pg.run_fast(lambda: False, 1.0)
    assert pg.behaviors.pose("picker")["position"][:2] == pytest.approx(before[:2], abs=0.01)


def test_go_false_keeps_a_robot_where_it_stands_and_refuses_as_before(pg):
    _wall_world(pg)
    with pytest.raises(BehaviorError, match="move the cube nearer"):
        pg.behaviors.pick_and_place(pg.behaviors.resolve("picker", arm=True), "cube", "goal", go=False)
    assert pg.behaviors.errand("picker") is None


def test_a_fixed_arm_picks_and_places_the_way_its_own_planner_says(pg):
    """An arm bolted down never goes on an errand: the moves queued are those ArmMotion.pick_and_place plans, the
    simulation's options are left alone, and the cube lands."""
    pg.add_robot("arm", template="arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, 0.1, 0.02])
    pg.run_fast(lambda: False, 0.5)
    model, data = pg.session.model, pg.session.data
    fresh = ArmMotion(model, "arm", data.qpos.copy())
    fresh.sync(data)
    fresh.q = data.qpos[fresh.chain.qpos_adr].copy()
    body = model.body("cube").id
    fresh.pick_and_place(data.xpos[body].copy(), 0.02, np.array([0.2, -0.2, 0.0]))
    motion, info = pg.behaviors.pick_and_place("arm", "cube", [0.2, -0.2])
    assert isinstance(motion, ArmMotion) and not isinstance(motion, fetch.Errand)
    assert [w.label for w in motion.queue] == [w.label for w in fresh.queue]
    for planned, fixed in zip(motion.queue, fresh.queue, strict=True):
        if planned.position is not None:
            assert planned.position == pytest.approx(fixed.position, abs=1e-3)
    pg.run_fast(lambda: motion.done, 40)
    pg.run_fast(lambda: False, 0.5)
    assert pg.behaviors.placed(info["object"], info["region"])["success"]
    assert model.opt.noslip_iterations == 0 and not motion.carry_weight


PANEL_AND_LINK = """
<mujoco>
  <worldbody>
    <geom name="panel" type="box" size="0.98 0.01 1.0" pos="0 1.01 1.0" euler="0 0 180"/>
    <body name="arm/link" pos="0.366 1.014 0.64">
      <joint name="arm/turn" type="hinge" axis="0 0 1"/>
      <geom name="tube" type="capsule" size="0.021 0.105" zaxis="0.63 -0.63 0.45"/>
      <site name="arm/tcp" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator><position name="arm/turn" joint="arm/turn" kp="10"/></actuator>
</mujoco>
"""


def test_an_arm_link_right_through_a_thin_panel_counts_as_touching_it():
    """MuJoCo's capsule-box test misses a capsule that goes right through a thin box: a lift picker's upper arm
    through a rack's 2 cm back panel read as 3 cm clear (this one is that link and panel), and the real arm pushed on
    the panel instead of reaching the parcel behind it."""
    model = mujoco.MjModel.from_xml_string(PANEL_AND_LINK)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0  # what MuJoCo says
    arm = ArmMotion(model, "arm", data.qpos.copy())
    arm.ik.point(arm.chain, arm.q)
    assert reach_in._links_pass_through(arm, (arm._robot_root,))
    assert not reach_in.Hand(arm)._clear(arm.q)


def test_the_template_arm_turns_its_base_under_the_no_slip_pass(pg):
    """The template arm's shoulder drum sits 6 mm into its pedestal folded down (a lift picker at home). With
    MuJoCo's no-slip pass that contact held the base joint still against the servo's full torque."""
    pg.add_robot("picker", design="lift_picker")
    pg.run_fast(lambda: False, 0.5)
    model, data = pg.session.model, pg.session.data
    model.opt.noslip_iterations = 5
    arm = pg.behaviors._arm(pg.behaviors.resolve("picker", arm=True))
    pg.session.set_controller("picker/arm", arm)
    arm.sync(data)
    yaw = model.joint("picker/arm/yaw")
    assert abs(math.degrees(data.qpos[yaw.qposadr[0]])) > 170  # folded backward at home
    goal = arm.q.copy()
    goal[arm.joints.index(yaw.id)] = 0.0
    arm.queue.append(Waypoint(None, None, 3.0, "turn", goal))
    pg.run_fast(lambda: arm.done, 5.0)
    pg.run_fast(lambda: False, 0.5)
    assert abs(math.degrees(data.qpos[yaw.qposadr[0]])) < 10


def test_a_crate_beside_the_robot_blocks_the_hands_way_round_that_side(pg):
    """The empty hand's way round the base to over a load is checked pose by pose (Hand._way_clear): unchecked, a lift
    picker's hand swung round at shelf height into a crate beside it."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("crate", shape="box", size=[0.3, 0.3, 0.6], pos=[0.1, -0.55, 0.3], fixed=True)
    pg.run_fast(lambda: False, 0.5)
    arm = pg.behaviors._arm(pg.behaviors.resolve("picker", arm=True))
    arm.sync(pg.session.data)
    base = arm.base_xy
    behind, ahead = np.array([base[0] - 0.45, base[1], 0.5]), np.array([base[0] + 0.45, base[1], 0.5])
    hand = reach_in.Hand(arm)
    assert not hand._way_clear(behind, ahead, math.pi)  # round its right, where the crate is
    assert hand._way_clear(behind, ahead, -math.pi)  # round its left


def test_a_robot_pushed_off_its_spot_while_its_arm_reaches_plans_the_pick_again(pg):
    """The arm's moves are planned for where the robot stands. A lift picker's hand struck a crate on its way over a
    parcel, the knock turned the robot 120 degrees, and its hand went on to close on nothing 1 m off. Here the robot
    is moved 12 cm while its arm is on the way; it stops, plans the pick again from there and takes the cube."""
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("stand", shape="table", size=[0.4, 0.4, 0.45], pos=[1.6, 0.0], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.45, 0.0, 0.47])
    pg.add_object("goal", shape="table", size=[0.4, 0.4, 0.45], pos=[-1.6, 0.0], fixed=True)
    pg.run_fast(lambda: False, 0.5)
    errand, _ = _start(pg, "cube", "goal")
    assert isinstance(errand, fetch.Errand)
    pg.run_fast(lambda: errand.step == "picking" or errand.done, fetch.TIME_LIMIT)
    assert errand.step == "picking", errand.result()
    pg.run_fast(lambda: False, 0.5)  # the arm is on its way
    model, data = pg.session.model, pg.session.data
    free = model.body_jntadr[pg.behaviors.root("picker")]
    data.qpos[model.jnt_qposadr[free] + 1] += 0.12  # sideways, as a knock would
    mujoco.mj_forward(model, data)
    pg.run_fast(lambda: errand.step == "to_place" or errand.done, fetch.TIME_LIMIT)
    assert errand.step == "to_place" and errand.held, errand.result()
    assert errand._knocks == 1


def test_a_ball_held_on_an_errand_does_not_spin_between_the_fingers(pg):
    """The finger pads are made firm for an errand without taking the contact over: a load keeps its own torsional
    friction against them. Given priority, the Panda's pads (condim 3) let an apple held tilted, a point of contact a
    side, spin 6 rad/s about the line between them, and put down on the dining table it rolled off. (Held straight
    down as here, two points a side stop the spin anyway: the contacts are what is checked.)"""
    pg.add_robot("mm", design="mobile_panda")
    pg.add_object("stand", shape="table", size=[0.4, 0.4, 0.45], pos=[1.6, 0.0], fixed=True)
    pg.add_object("ball", shape="sphere", size=[0.06], pos=[1.45, 0.0, 0.48])
    pg.add_object("goal", shape="table", size=[0.4, 0.4, 0.45], pos=[-1.6, 0.0], fixed=True)
    pg.run_fast(lambda: False, 0.5)
    errand, _ = pg.behaviors.pick_and_place(pg.behaviors.resolve("mm", arm=True), "ball", "goal")
    assert isinstance(errand, fetch.Errand)
    pg.run_fast(lambda: errand.step == "to_place" or errand.done, fetch.TIME_LIMIT)
    assert errand.step == "to_place", errand.result()
    model, data = pg.session.model, pg.session.data
    pads, ball = set(fetch._finger_geoms(model, errand.arm).tolist()), errand.body
    held = [
        c
        for c in data.contact[: data.ncon]
        if {int(c.geom1), int(c.geom2)} & pads and ball in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
    ]
    assert held and all(c.dim >= 4 for c in held)  # the ball's own condim 4: torsional friction with it
    assert all(c.solref == pytest.approx(fetch.FIRM_SOLREF, abs=1e-3) for c in held)  # and the pads' firm contact


def test_a_path_move_gets_to_where_its_path_starts_joint_by_joint_first(pg):
    pg.add_robot("arm", template="arm")
    pg.run_fast(lambda: False, 0.3)
    data = pg.session.data
    arm = pg.behaviors._arm("arm")
    pg.session.set_controller("arm", arm)
    arm.sync(data)
    start = arm.q.copy()
    first = start + 0.3  # the path was worked out from another pose
    knots = np.array([first + 0.05 * k for k in range(5)])
    arm.queue.append(Waypoint(None, None, 1.0, "path", knots[-1], path=knots))
    labels, reached_first = [], []
    while not arm.done and data.time < 10:
        pg.run_fast(lambda: False, 0.05)
        labels.append(arm.current_label)
        reached_first.append(np.max(np.abs(arm.q - first)) < 1e-6)
    assert any(reached_first)  # it got to the path's first knot (by the joint move put in front) ...
    assert arm.q == pytest.approx(knots[-1])  # ... and followed the path to its end
    assert set(labels) - {"idle"} == {"path"}


def test_the_spots_tried_on_a_big_place_reach_its_edges():
    """On a 2 by 2.4 m loading dock the 16 spots nearest its middle were all beyond the reach of a robot that cannot
    drive up onto it."""
    xs, ys = np.arange(-1.0, 1.0 + 1e-9, 0.04), np.arange(-1.2, 1.2 + 1e-9, 0.04)
    spots = sorted((np.array([x, y, 0.3]) for x in xs for y in ys), key=lambda p: float(np.hypot(p[0], p[1])))
    taken = np.array([p[:2] for p in fetch._spread(spots, fetch.PLACE_SPOTS)])
    assert len(taken) == fetch.PLACE_SPOTS
    assert taken[:, 0].max() > 0.6 and taken[:, 0].min() < -0.6
    assert taken[:, 1].max() > 0.7 and taken[:, 1].min() < -0.7


def test_a_box_is_come_at_square_to_a_face_its_fingers_fit_across(pg):
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("cereal", shape="box", size=[0.06, 0.14, 0.2], pos=[0.5, 0.0, 0.1], yaw=30)
    pg.run_fast(lambda: False, 0.3)
    model, data = pg.session.model, pg.session.data
    arm = pg.behaviors._arm(pg.behaviors.resolve("picker", arm=True))
    arm.sync(data)
    body = model.body("cereal").id
    faces = fetch._box_faces(model, data, body, fetch.gripper_opening(model, arm))
    assert faces is not None
    hand = reach_in.Hand(arm, body, faces)
    way = hand.toward(data.xpos[body])
    long_axis = data.geom_xmat[next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body)].reshape(3, 3)[:2, 1]
    # The box is 6 cm thick and 14 cm wide: the fingers close across the 6 cm, so the hand comes along the 14.
    assert abs(float(way @ long_axis)) == pytest.approx(1.0, abs=1e-6)
