"""Dexterous hands as end effectors (hands.py): one tendon closes the grasping joints as the arm's gripper.

Offline against the Menagerie stand-ins in tests/fixtures/menagerie; the test_real_ ones download the real hands.
"""

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from robotforge.behaviors.arm import ArmMotion
from robotforge.design import hands, library
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
NETWORK = os.environ.get("ROBOTFORGE_NETWORK_TESTS") == "1"
HANDS = ("leap_hand", "allegro_hand")


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch, request):
    if not request.node.name.startswith("test_real_"):
        monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "checker")
    return playground


def _id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, kind, name)


@pytest.mark.parametrize("name", HANDS)
def test_a_hand_from_the_library_closes_its_fingers_with_one_gripper(name):
    built = library.build(name)
    model = built.spec.compile()
    grip = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    assert grip >= 0 and model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_TENDON
    assert _id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp") >= 0
    low, high = model.actuator_ctrlrange[grip]
    assert built.home_ctrl["gripper"] == pytest.approx(high)  # it starts open
    hand = library.get(name).hand
    joints = [_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n, *_ in hand.flex]
    data = mujoco.MjData(model)
    for joint, value in built.home_qpos.items():
        data.qpos[model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = value
    for actuator, value in built.home_ctrl.items():
        data.ctrl[_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)] = value
    model.opt.gravity[:] = 0.0
    data.ctrl[grip] = low
    for _ in range(1000):
        mujoco.mj_step(model, data)
    # every grasping joint has moved most of the way from open to closed
    for j, (_, opened, closed) in zip(joints, hand.flex, strict=True):
        done = (data.qpos[model.jnt_qposadr[j]] - opened) / (closed - opened)
        assert done > 0.6, model.joint(j).name


def test_the_hand_mounts_with_its_grasp_point_along_the_flange():
    built = library.build("leap_hand")
    model = built.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    tcp = _id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    np.testing.assert_allclose(data.site_xpos[tcp], hands.LEAP.tcp, atol=1e-6)
    np.testing.assert_allclose(data.site_xmat[tcp].reshape(3, 3), np.eye(3), atol=1e-6)  # pointing along +z


@pytest.mark.parametrize("name", HANDS)
def test_a_hand_on_a_ur5e_is_the_arms_gripper(pg, name):
    pg.add_robot("arm", library="ur5e")
    out = pg.mount(name, "arm")
    model = pg.session.model
    grip = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm/gripper")
    assert grip >= 0 and model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_TENDON
    arm = ArmMotion(model, "arm", pg.session.data.qpos.copy())
    assert arm.closing is not None  # fingers and thumb close across the tool axis, as jaws do
    assert out["checks"]["arms"][0]["holds"]


def test_quick_change_robotiq_then_hands_then_suction(pg):
    pg.add_robot("arm", library="ur5e")
    pg.mount("robotiq_2f85", "arm")
    for tool in ("leap_hand", "allegro_hand", "panda_hand", "suction"):
        out = pg.mount(tool, "arm")
        assert "swapped" in out
        assert len(pg.current_world().robots[0].mounts) == 1
    model = pg.session.model
    grip = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm/gripper")
    assert model.actuator_trntype[grip] == mujoco.mjtTrn.mjTRN_BODY


def _pick(pg: Playground, arm: str, obj: str, target: str) -> bool:
    motion, info = pg.behaviors.pick_and_place(arm, obj, target, True)
    pg.run_fast(lambda: motion.done, 40.0)
    pg.run_fast(lambda: False, 1.0)
    return bool(pg.behaviors.placed(info["object"], info["region"])["success"])


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize(
    ("name", "shape", "size"), [("leap_hand", "box", [0.05] * 3), ("allegro_hand", "sphere", [0.05])]
)
def test_real_a_ur5e_with_a_hand_puts_a_thing_in_a_bin(pg, name, shape, size):
    pg.add_robot("arm", library="ur5e")
    pg.mount(name, "arm")
    pg.add_object("thing", shape=shape, size=size, pos=[0.5, 0.15])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.08], pos=[0.35, -0.3], fixed=True)
    assert _pick(pg, "arm", "thing", "bin")
