"""Joint impedance, Cartesian impedance and operational-space control on real library arms (the Menagerie fixtures)."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.control import TaskError, compare_methods, make_controller, make_task
from robotforge.control.impedance import IMPEDANCE_METHODS, per_axis
from robotforge.control.tasks.tool import ToolTarget, pointing, rotation_error
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

MENAGERIE = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(MENAGERIE))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    return pg


def _reach_from_here(pg: Playground, robot: str, offset: list[float]) -> list[float]:
    m, d = pg.session.model, pg.session.data
    mujoco.mj_forward(m, d)
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/tcp")
    return (d.site_xpos[site] + offset).tolist()


def test_reach_drives_a_panda_by_its_arm_joints_not_by_its_tendon_gripper(lab):
    """The Panda's gripper is a tendon actuator whose tendon index (0) passed for joint 0, so `reach` drove the gripper
    in the first joint's place and the arm never moved."""
    lab.add_robot("panda", library="panda")
    model = lab.session.model
    task = make_task("reach", model, "panda", {"target": _reach_from_here(lab, "panda", [0.1, 0.1, -0.1])})
    names = [model.actuator(a).name for a in task.actuators]
    assert names == [f"panda/actuator{i}" for i in range(1, 8)]


@pytest.mark.parametrize("library", ["panda", "ur5e"])
def test_every_impedance_method_reaches_a_point_with_a_real_arm(lab, library):
    lab.add_robot("arm", library=library)
    model = lab.session.model
    target = _reach_from_here(lab, "arm", [0.1, 0.1, -0.12])
    cmp = compare_methods(
        "reach",
        list(IMPEDANCE_METHODS),
        3.0,
        task_params={"target": target, "tolerance": 0.01},
        model=model,
        robot="arm",
        data=copy.copy(lab.session.data),
        seed=None,
    )
    by = {r.method: r for r in cmp.results}
    assert set(by) == set(IMPEDANCE_METHODS)
    for name, result in by.items():
        assert result.success and not result.failed, (name, result.report)
        assert result.report["distance_m"] < 0.01
    # An impedance is a soft spring: the joint one has not settled as tightly as the operational-space one by 3 s.
    assert by["osc"].report["distance_m"] <= by["joint_impedance"].report["distance_m"] + 1e-3


def test_torque_control_uses_the_arm_joints_and_leaves_the_gripper_alone(lab):
    lab.add_robot("panda", library="panda")
    model, data = lab.session.model, copy.copy(lab.session.data)
    task = make_task("reach", model, "panda", {"target": _reach_from_here(lab, "panda", [0.1, 0.0, -0.1])})
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "panda/gripper")
    before = float(data.ctrl[gripper])
    session = SimSession(model=model, data=data)
    session.set_controller("panda", make_controller("osc", task, model, data))
    session.step(500)
    assert float(data.ctrl[gripper]) == before
    assert np.isfinite(data.qpos).all()


def test_a_mistyped_setting_names_the_method(lab):
    lab.add_robot("panda", library="panda")
    model, data = lab.session.model, lab.session.data
    task = make_task("reach", model, "panda", {"target": _reach_from_here(lab, "panda", [0.1, 0.0, -0.1])})
    with pytest.raises(TaskError, match="bad params"):
        make_controller("osc", task, model, data, {"stiffnes": 3})
    with pytest.raises(TaskError, match="stiffness takes one number or 3"):
        make_controller("cartesian_impedance", task, model, data, {"stiffness": [1, 2]})
    with pytest.raises(TaskError, match="omega must be a number between 0 and 200"):
        make_controller("joint_impedance", task, model, data, {"omega": -1})


def test_a_pose_gain_is_one_number_or_one_per_axis():
    assert per_axis("k", 5.0, 3).tolist() == [5.0, 5.0, 5.0]
    assert per_axis("k", [1, 2, 3], 3).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(TaskError):
        per_axis("k", [1.0, -2.0, 3.0], 3)


def test_rotation_error_and_pointing():
    ident = np.eye(3)
    about_z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 degrees about z
    err = rotation_error(ident, about_z)
    assert err[:2] == pytest.approx([0.0, 0.0], abs=1e-9) and err[2] == pytest.approx(1.0, abs=1e-9)
    down = pointing(np.array([0.0, 0.0, -1.0]), ident)
    assert down[:, 2] == pytest.approx([0.0, 0.0, -1.0]) and np.linalg.det(down) == pytest.approx(1.0)
    assert down[:, 0] == pytest.approx([1.0, 0.0, 0.0])  # the x axis already across the new z stays
    pointing(np.array([1.0, 0.0, 0.0]), ident)  # the old x along the new z: a perpendicular is found
    assert isinstance(ToolTarget(np.zeros(3)).vel, np.ndarray)
