"""Grasping generated objects by their hints (behaviors/hints.py): where the arm takes hold, how firmly, where it puts
things (a rack's socket, a tray, the top of a box, a shelf's top board), offline with the Menagerie stand-ins."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge import objects
from robotforge.behaviors import hints
from robotforge.behaviors.manager import BehaviorError
from robotforge.objects import kit
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
TABLE = 0.4


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("grasps")
    playground.add_robot("arm", library="franka_emika_panda", pos=[0, 0, TABLE])
    playground.add_object("table", shape="table", size=[1.4, 1.2, TABLE], pos=[0.4, 0], fixed=True)
    playground.add_object("tray", shape="tray", size=[0.3, 0.26, 0.05], pos=[0.45, 0.28, TABLE])
    return playground


def _pick(pg: Playground, thing: str, target, grasp: str | None = None):
    arm = pg.behaviors.resolve("arm", arm=True)
    motion, info = pg.behaviors.pick_and_place(arm, thing, target, True, grasp)
    pg.run_fast(lambda: motion.done, 60)
    pg.run_fast(lambda: False, 1.0)
    return info, pg.behaviors.placed(thing, info["region"])


# ---- the geometry the hints are checked against -------------------------------------------------------------------


def test_the_highest_point_under_a_palm_counts_only_what_is_inside_its_rectangle():
    mug = objects.object_for("mug")
    reach = kit.top_under(mug, (0.0, 0.0), (1.0, 0.0), 0.05, 0.03)
    assert reach == pytest.approx(0.0974, abs=0.003)  # the rim over the mug's middle
    beside = kit.top_under(mug, (0.075, 0.0), (0.0, 1.0), 0.1, 0.02)
    assert beside < reach  # out by the handle's bar, clear of the rim
    far = kit.top_under(mug, (0.5, 0.5), (1.0, 0.0), 0.05, 0.03)
    assert far == -np.inf


def test_a_hint_is_lifted_until_the_palm_clears_the_top_and_dropped_when_it_cannot():
    can = objects.object_for("can")
    hint = can.grasps[0]
    short = hints.Palm(gap=0.03, tip=0.01, half_jaw=0.1, half_across=0.03)
    roomy = hints.Palm(gap=0.06, tip=0.01, half_jaw=0.05, half_across=0.05)
    z, why = hints._height(hint, can, short)
    assert z is not None and z >= can.height - short.gap + hints.PALM_MARGIN - 1e-9 and "palm clears" in why
    lower, _ = hints._height(hint, can, roomy)
    assert lower < z  # a hand with more room holds it lower
    tall = objects.object_for("bottle")
    body = next(g for g in tall.grasps if g.kind == "body")
    none, why = hints._height(body, tall, short)
    assert none is None and "palm would meet the top" in why


def test_the_squeeze_needed_grows_with_how_far_the_load_hangs_from_the_pinch():
    mug = objects.object_for("mug")
    rot = np.eye(3)
    base = np.zeros(3)
    handle = mug.grasps[0]
    faces = np.array([[0.0, 1.0, 0.0]] * 2)
    near = hints._squeeze_for(mug, handle, np.array([0.01, 0.0, 0.05]), base, rot, faces)
    far = hints._squeeze_for(mug, handle, np.array([0.08, 0.0, 0.05]), base, rot, faces)
    assert hints.FORCE_MIN <= near < far
    plate = objects.object_for("plate", material="ceramic")
    rim = plate.grasps[0]
    cannot = hints._squeeze_for(plate, rim, np.array([0.1, 0.0, 0.02]), base, rot, np.array([[1.0, 0.0, 0.0]] * 2))
    assert cannot > hints.FORCE_MAX * 0.9  # a china plate pinched on its rim does not stay in a Panda's hand


def test_an_object_that_is_tipped_over_has_no_hints_and_says_why(pg):
    pg.add_object("mug", shape="mug", pos=[0.45, -0.2, TABLE])
    pg.run_fast(lambda: False, 0.5)
    m, d = pg.session.model, pg.session.data
    body = m.body("mug").id
    d.qpos[m.jnt_qposadr[m.body_jntadr[body]] + 3 : m.jnt_qposadr[m.body_jntadr[body]] + 7] = [0.7071, 0.7071, 0, 0]
    mujoco.mj_forward(m, d)
    found, left = hints.choices(d, body, objects.object_for("mug"), 0.08, hints.DEFAULT_PALM)
    assert not found and "tipped over" in left[0]


# ---- picking and placing real objects ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "kind"),
    [("mug", "handle"), ("can", "body"), ("bottle", "neck"), ("hammer", "shaft"), ("tin", "body"), ("apple", "body")],
)
def test_each_kind_of_thing_is_taken_where_its_generator_says_and_ends_up_in_the_tray(pg, shape, kind):
    pg.add_object("thing", shape=shape, pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    info, result = _pick(pg, "thing", "tray")
    assert info["grasp"]["kind"] == kind
    assert result["success"] and result["tilt_deg"] < 30
    d = pg.session.data
    assert d.xpos[pg.session.model.body("thing").id][2] < TABLE + 0.09  # on the tray's floor, not held in the air


def test_a_grasp_can_be_asked_for_by_kind_and_a_kind_it_does_not_have_is_refused(pg):
    pg.add_object("bottle", shape="bottle", pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    arm = pg.behaviors.resolve("arm", arm=True)
    with pytest.raises(BehaviorError, match="asked for a handle grasp; it has body, neck"):
        pg.behaviors.pick_and_place(arm, "bottle", "tray", True, "handle")


def test_a_bottle_that_is_too_tall_for_the_palm_is_taken_by_the_neck_and_the_reason_is_in_the_hint(pg):
    pg.add_object("bottle", shape="bottle", pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    body = pg.session.model.body("bottle").id
    arm = pg.behaviors._arm_motion(pg.behaviors.resolve("arm", arm=True))
    palm = hints.palm_of(pg.session.model, arm)
    assert palm.gap > 0 and palm.half_jaw > palm.half_across
    found, left = hints.choices(pg.session.data, body, objects.object_for("bottle"), 0.08, palm)
    assert [c.hint.kind for c in found] == ["neck"]
    assert any(line.startswith("body: the hand's palm would meet the top") for line in left)


def test_a_heavy_thing_is_held_firmly_while_it_is_carried_and_the_gripper_is_put_back(pg):
    pg.add_object("can", shape="can", pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    m = pg.session.model
    arm = pg.behaviors.resolve("arm", arm=True)
    gripper = m.actuator("arm/gripper").id
    gain, noslip = m.actuator_gainprm[gripper].copy(), int(m.opt.noslip_iterations)
    hooks = len(pg.session.reset_hooks)
    motion, info = pg.behaviors.pick_and_place(arm, "can", "tray")
    assert m.opt.noslip_iterations > noslip  # firm while it holds the can
    assert motion.on_idle and motion.on_open
    pg.run_fast(lambda: motion.done, 60)
    pg.run_fast(lambda: False, 0.5)
    assert m.opt.noslip_iterations == noslip and np.allclose(m.actuator_gainprm[gripper], gain)
    assert len(pg.session.reset_hooks) == hooks


def test_a_reset_in_the_middle_of_a_firm_hold_puts_the_settings_back(pg):
    pg.add_object("can", shape="can", pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    m = pg.session.model
    noslip, hooks = int(m.opt.noslip_iterations), len(pg.session.reset_hooks)
    arm = pg.behaviors.resolve("arm", arm=True)
    pg.behaviors.pick_and_place(arm, "can", "tray")
    assert m.opt.noslip_iterations > noslip
    pg.control("reset")
    assert m.opt.noslip_iterations == noslip


def test_a_cylinder_piece_collides_as_a_prism_so_the_fingers_do_not_lock_on_it(pg):
    pg.add_object("can", shape="can", pos=[0.45, -0.22, TABLE])
    m = pg.session.model
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("can").id and m.geom_contype[g]]
    assert geoms and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in geoms)


# ---- where things are put ------------------------------------------------------------------------------------------


def test_the_first_free_socket_of_a_rack_is_taken_and_a_named_one_by_its_number(pg):
    pg.add_object("rack", shape="rack", pos=[0.5, 0.3, TABLE])
    pg.add_object("b1", shape="bottle", pos=[0.4, -0.22, TABLE])
    pg.add_object("b2", shape="bottle", pos=[0.55, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    info1, one = _pick(pg, "b1", "rack")
    info2, two = _pick(pg, "b2", "rack/3")
    assert one["success"] and two["success"]
    d, m = pg.session.data, pg.session.model
    x1, x2 = d.xpos[m.body("b1").id][0], d.xpos[m.body("b2").id][0]
    assert x2 - x1 == pytest.approx(0.2, abs=0.02)  # sockets 1 and 3 of a rack 30 cm long
    with pytest.raises(BehaviorError, match="has no place called 'seven'.*1, 2, 3"):
        pg.behaviors.pick_and_place(pg.behaviors.resolve("arm", arm=True), "b1", "rack/seven")


def test_things_put_in_one_tray_are_spread_apart_and_a_full_tray_says_so(pg):
    for i, x in enumerate((0.3, 0.38, 0.46)):
        pg.add_object(f"a{i}", shape="apple", pos=[x, -0.25, TABLE])
    pg.add_object("hold", shape="tray", size=[0.22, 0.17, 0.05], pos=[0.62, 0.22, TABLE])
    pg.add_object("tiny", shape="tray", size=[0.11, 0.09, 0.05], pos=[0.3, 0.3, TABLE])
    pg.run_fast(lambda: False, 0.8)
    for i in range(2):
        _pick(pg, f"a{i}", "hold")
    d, m = pg.session.data, pg.session.model
    gap = float(np.linalg.norm(d.xpos[m.body("a0").id][:2] - d.xpos[m.body("a1").id][:2]))
    assert gap >= hints.ROOM - 0.01
    _pick(pg, "a2", "tiny")  # room for one
    pg.add_object("a3", shape="apple", pos=[0.55, -0.25, TABLE])
    pg.run_fast(lambda: False, 0.5)
    with pytest.raises(BehaviorError, match="every place on tiny is taken"):
        pg.behaviors.pick_and_place(pg.behaviors.resolve("arm", arm=True), "a3", "tiny")


def test_a_flat_topped_thing_takes_things_stacked_on_it_and_a_mug_does_not(pg):
    pg.add_object("tin", shape="tin", pos=[0.5, 0.0, TABLE])
    pg.add_object("brick", shape="brick", pos=[0.4, -0.25, TABLE])
    pg.add_object("mug", shape="mug", pos=[0.6, -0.25, TABLE])
    pg.run_fast(lambda: False, 0.8)
    info, result = _pick(pg, "brick", "tin")
    assert result["success"] and result["tilt_deg"] < 15
    d, m = pg.session.data, pg.session.model
    assert d.xpos[m.body("brick").id][2] == pytest.approx(d.xpos[m.body("tin").id][2] + 0.082, abs=0.012)
    with pytest.raises(BehaviorError, match="nothing can be put in or on it"):
        pg.behaviors.pick_and_place(pg.behaviors.resolve("arm", arm=True), "brick", "mug")


def test_only_the_top_board_of_a_shelf_is_open_to_a_gripper_coming_from_above(pg):
    pg.add_object("shelf", shape="shelf", size=[0.4, 0.2, 0.2], pos=[0.5, 0.3, TABLE])
    pg.add_object("mug", shape="mug", pos=[0.45, -0.2, TABLE])
    pg.run_fast(lambda: False, 0.8)
    arm = pg.behaviors.resolve("arm", arm=True)
    with pytest.raises(BehaviorError, match="board 1 of shelf has another board over it"):
        pg.behaviors.pick_and_place(arm, "mug", "shelf/1")
    info, result = _pick(pg, "mug", "shelf")
    assert result["success"] and result["tilt_deg"] < 15


def test_the_carry_clears_what_stands_between_the_pick_and_the_place(pg):
    pg.add_object("tower", shape="package", pos=[0.5, 0.0, TABLE])  # 20 cm tall, between the mug and the tray
    pg.add_object("mug", shape="mug", pos=[0.45, -0.22, TABLE])
    pg.run_fast(lambda: False, 0.8)
    m, d = pg.session.model, pg.session.data
    tower = m.body("tower").id
    start = d.xpos[tower].copy()
    info, result = _pick(pg, "mug", "tray")
    assert result["success"] and np.linalg.norm(d.xpos[tower] - start) < 0.01  # carried over it, not through it
