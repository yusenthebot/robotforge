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
