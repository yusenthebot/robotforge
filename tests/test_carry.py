"""Editing a world keeps what the edit did not touch where it is (sim/carry.py)."""

import numpy as np
import pytest

from robotforge.playground import Playground
from robotforge.sim.session import SimSession


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab")
    return playground


def _walk(pg: Playground, robot: str, seconds: float) -> list[float]:
    pg.behaviors.move(robot, 0.3, 0.0)
    pg.run_fast(lambda: False, seconds)
    pg.behaviors.stop(robot)
    pg.run_fast(lambda: False, 1.0)
    return pg.behaviors.pose(robot)["position"]


def test_adding_a_box_does_not_send_the_dog_back(pg):
    """Regression: every edit reloaded the world fresh; a Codex session had to walk its Go2 again."""
    pg.add_robot("dog", template="quadruped")
    walked = _walk(pg, "dog", 4.0)
    assert walked[0] > 0.6
    pg.add_object("crate", shape="box", pos=[-1.5, 1.5])
    kept = pg.behaviors.pose("dog")
    assert np.allclose(kept["position"][:2], walked[:2], atol=0.02) and kept["upright"]
    pg.run_fast(lambda: False, 1.0)  # and it stands there, not flung or sinking
    assert np.allclose(pg.behaviors.pose("dog")["position"][:2], walked[:2], atol=0.05)


def test_an_arm_keeps_its_pose_and_an_object_where_it_was_put(pg):
    pg.add_robot("arm", template="arm")
    pg.add_object("ball", shape="sphere", pos=[0.5, -0.5])
    pg.behaviors.move_arm("arm", [0.2, 0.15, 0.2])
    pg.run_fast(lambda: False, 2.0)
    m, d = pg.session.model, pg.session.data
    joints = {
        m.joint(j).name: float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.joint(j).name.startswith("arm/")
    }
    ball = d.xpos[m.body("ball").id].copy()
    d.qpos[m.jnt_qposadr[m.body("ball").jntadr[0]] : m.jnt_qposadr[m.body("ball").jntadr[0]] + 2] += [0.1, 0.0]
    pg.run_fast(lambda: False, 0.5)
    rolled = pg.session.data.xpos[m.body("ball").id].copy()
    assert np.linalg.norm(rolled[:2] - ball[:2]) > 0.05
    pg.add_object("crate", shape="box", pos=[-1.0, 1.0])
    m, d = pg.session.model, pg.session.data
    for name, angle in joints.items():
        assert abs(d.qpos[m.jnt_qposadr[m.joint(name).id]] - angle) < 0.02, name
    pg.run_fast(lambda: False, 1.0)  # the servos hold it there, not swing it home
    for name, angle in joints.items():
        assert abs(d.qpos[m.jnt_qposadr[m.joint(name).id]] - angle) < 0.05, name
    assert np.linalg.norm(d.xpos[m.body("ball").id][:2] - rolled[:2]) < 0.05


def test_what_the_edit_changed_or_buried_starts_fresh(pg):
    pg.add_robot("dog", template="quadruped")
    start = pg.behaviors.pose("dog")["position"]
    walked = _walk(pg, "dog", 4.0)
    pg.add_object("block", shape="box", size=[0.6, 0.6, 0.4], pos=walked[:2])  # dropped where the dog stands
    assert np.allclose(pg.behaviors.pose("dog")["position"][:2], start[:2], atol=0.05)

    walked = _walk(pg, "dog", 2.0)
    pg.update_robot("dog", params={"color": "blue"})  # the dog itself changed
    assert np.allclose(pg.behaviors.pose("dog")["position"][:2], start[:2], atol=0.05)

    walked = _walk(pg, "dog", 2.0)
    pg.add_terrain("hills", pos=[2.0, 2.0])  # new ground: heights no longer fit
    assert not np.allclose(pg.behaviors.pose("dog")["position"][:2], walked[:2], atol=0.05)


def test_leaving_a_lesson_brings_the_world_back_as_it_was(pg):
    pg.add_robot("dog", template="quadruped")
    walked = _walk(pg, "dog", 4.0)
    pg.lessons.start("pd_control")
    pg.lessons.exit()
    assert pg.store.load().name == "lab"
    assert np.allclose(pg.behaviors.pose("dog")["position"][:2], walked[:2], atol=0.02)


def test_a_go2_mid_stride_is_carried_past_a_crate_put_where_it_started(pg, monkeypatch):
    """Regression from a real Codex session: a Go2 walking on its learned policy sinks its feet 26 mm into the
    ground mid-stride; that read as buried, so the edit sent it back to its start, inside the new crate."""
    from pathlib import Path

    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    pg.add_robot("go2", library="go2", pos=[0, 0])
    pg.behaviors.move("go2", 0.4, 0.0, walker="rl")
    pg.run_fast(lambda: False, 3.0)
    walked = pg.behaviors.pose("go2")["position"]
    assert walked[0] > 0.8
    pg.add_object("crate", shape="box", size=[0.4, 0.4, 0.4], pos=[0.05, 0.04])  # where it started
    assert np.allclose(pg.behaviors.pose("go2")["position"][:2], walked[:2], atol=0.03)


def _knee(pg: Playground) -> float:
    m = pg.session.model
    return float(pg.session.data.qpos[m.jnt_qposadr[m.actuator_trnid[m.actuator("dog/front_left_knee").id, 0]]])


@pytest.mark.parametrize("trip", ["edit", "lesson", "joints by hand"])
def test_a_leg_set_by_hand_holds_its_angle_through_an_edit_or_a_lesson(pg, trip):
    """Regression found in review: a dog in a world stands on its gait from the start, and a leg set by hand takes
    it off; an edit or a lesson brought the world back with a fresh gait, which swung the knee home (22.9 degrees).
    The gait stays off, as for the arm above; a robot added in the same edit still gets its own."""
    from robotforge import modes

    pg.add_robot("dog", template="quadruped")
    pg.run_fast(lambda: False, 1.0)
    if trip == "joints by hand":
        modes.apply(pg, "dog", "manual")
    pg.set_controls({"dog/front_left_knee": -1.2})
    pg.run_fast(lambda: False, 1.5)
    posed = _knee(pg)
    assert posed == pytest.approx(-1.2, abs=0.05) and "dog" not in pg.session.controllers
    if trip == "lesson":
        pg.lessons.start("pd_control")
        pg.lessons.exit()
    else:
        pg.add_robot("pup", template="quadruped", pos=[1.5, 1.5])
        assert "pup" in pg.session.controllers
    pg.run_fast(lambda: False, 2.0)
    assert _knee(pg) == pytest.approx(posed, abs=0.02)
    assert "dog" not in pg.session.controllers
    pg.session.reset()  # a reset starts everything fresh, on its gait
    assert "dog" in pg.session.controllers
