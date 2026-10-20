"""What pick_and_place reports: the kitchen preset's apple, a mug 2 mm off its shelf slot, a pick that never lifts."""

from __future__ import annotations

import numpy as np
import pytest

from robotforge.behaviors.manager import BehaviorError, _why_not_placed
from robotforge.design.tasks import hard_bench
from robotforge.design.tasks.hard import TABLE_Z
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab")
    return playground


def _do(pg: Playground, robot: str, obj: str, target: str) -> dict:
    start = pg.behaviors.position_of(obj)
    motion, info = pg.behaviors.pick_and_place(robot, obj, target)
    pg.run_fast(lambda: motion.done, 60)
    pg.run_fast(lambda: False, 1.5)
    return pg.behaviors.placed(obj, info["region"], start)


@pytest.mark.parametrize("target", ["bowl", "plate", "red_cup"])
def test_the_kitchen_presets_own_arm_puts_its_apple_where_asked(pg, target):
    """Regression: the apple was dropped on the way (a plain squeeze lets a ball slip, the carry went through the cup
    beside the bowl) and rolled on for ever once let go: it ended on the floor 1.5 m away with no word why."""
    pg.load_scene(preset="kitchen")
    result = _do(pg, "arm", "apple", target)
    assert result["success"] is True, result
    assert "why" not in result


def test_a_ball_does_not_roll_on_for_ever(pg):
    pg.add_object("table", shape="table", size=[1.0, 1.0, 0.5], pos=[0, 0], fixed=True)
    pg.add_object("ball", shape="sphere", size=[0.06], pos=[0, 0, 0.53])
    data = pg.session.data
    adr = int(pg.session.model.jnt_dofadr[pg.session.model.joint("ball").id])
    data.qvel[adr] = 0.4
    pg.run_fast(lambda: False, 6.0)
    assert abs(float(data.qvel[adr])) < 0.02


def test_a_mug_2_mm_off_its_shelf_slot_counts_as_on_the_shelf(pg):
    """Regression: 3 of 12 mugs stood upright on the shelf with success false while the task check said done."""
    hard_bench.bench(pg, "ur5e")
    pg.add_object("shelf", shape="shelf", size=[0.4, 0.2, 0.2], pos=[0.52, 0.31, TABLE_Z], color="#b08a5a")
    pg.add_object("mug", shape="mug", seed=0, pos=[0.45, -0.2, TABLE_Z])
    result = _do(pg, "arm", "mug", "shelf")
    assert result["success"] is True, result


def test_a_pick_that_moves_nothing_says_so():
    where = np.array([0.45, -0.2, 0.4])
    why = _why_not_placed("mug", where + [0.01, 0, 0], np.array([0.5, 0.3, 0.6]), np.array([0.1, 0.06]), 0.59, where)
    assert "never lifted" in why
    dropped = _why_not_placed(
        "apple", np.array([-1.5, 0.8, 0.03]), np.array([-0.3, 0.1, 0.9]), np.array([0.08, 0.08]), 0.89, where
    )
    assert "dropped" in dropped
    beside = _why_not_placed(
        "cube", np.array([0.6, 0.3, 0.6]), np.array([0.5, 0.3, 0.6]), np.array([0.05, 0.05]), 0.59, None
    )
    assert "beside the target" in beside


def test_two_arms_whose_hands_meet_at_home_refuse_the_pick_before_moving(pg):
    """Regression: a second Panda at the far end of the table made the first arm's mug pick fail without a word."""
    hard_bench.bench(pg, "panda")
    pg.add_object("mug", shape="mug", seed=0, pos=[0.45, -0.2, TABLE_Z])
    pg.add_object("shelf", shape="shelf", size=[0.4, 0.2, 0.2], pos=[0.52, 0.31, TABLE_Z], color="#b08a5a")
    pg.add_robot("arm2", library="panda", pos=[1.0, 0, TABLE_Z], yaw=180)
    with pytest.raises(BehaviorError, match="touches another robot already"):
        pg.behaviors.pick_and_place("arm", "mug", "shelf")


def test_a_handover_between_arms_that_meet_in_the_middle_says_why_it_failed(pg):
    """Regression: two template arms 0.7 m apart handed a cube over, the receiver could not take it and the answer was
    success false with the cube on the floor between them and no word why."""
    pg.add_robot("left", template="arm", params={"reach": 0.6}, pos=[-0.35, 0], yaw=0)
    pg.add_robot("right", template="arm", params={"reach": 0.6}, pos=[0.35, 0], yaw=180)
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], pos=[-0.3, 0.1], mass=0.1)
    pg.add_object("pad", shape="box", size=[0.2, 0.2, 0.01], pos=[0.3, -0.1], fixed=True)
    start = pg.behaviors.position_of("cube")
    relay, info = pg.behaviors.hand_over("left", "right", "cube", "pad")
    pg.run_fast(lambda: relay.done, 120)
    pg.run_fast(lambda: False, 1.0)
    placed = pg.behaviors.placed("cube", info["region"], start)
    assert placed["success"] is False and placed["why"]
    assert "touches another robot" in relay.result()["note"] or "cannot" in relay.result()["note"]
