"""Repairs for MJCF files found in the wild: numbers written with commas, torque motors, no standing pose."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from importer_fixtures import write

from robotforge.design import mjcf_doctor, mjcf_stance


def _legs(hip_axis: str = "1 0 0", knee_range: str = "-2.7 -0.9", joints_per_leg: int = 3) -> str:
    """A dog of capsules in the way unitree_mujoco writes one: torque motors, ranges that exclude the zero pose of
    the knee, no keyframe."""
    body, motors = "", ""
    for name, (x, y) in {"FR": (0.18, -0.05), "FL": (0.18, 0.05), "RR": (-0.18, -0.05), "RL": (-0.18, 0.05)}.items():
        hip = f'<joint name="{name}_hip" axis="{hip_axis}" range="-0.8 0.8"/>' if joints_per_leg == 3 else ""
        body += (
            f'<body name="{name}_hipb" pos="{x} {y} 0">{hip}<geom type="sphere" size="0.02" mass="0.5"/>'
            f'<body name="{name}_thigh"><joint name="{name}_thigh_j" axis="0 1 0" range="-1 4"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015" mass="0.5"/>'
            f'<body name="{name}_calf" pos="0 0 -0.2"><joint name="{name}_calf_j" axis="0 1 0" range="{knee_range}"/>'
            '<geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.012" mass="0.2"/>'
            f'<geom type="sphere" pos="0 0 -0.2" size="0.02" mass="0.05" name="{name}_foot"/></body></body></body>'
        )
        for joint in (f"{name}_hip", f"{name}_thigh_j", f"{name}_calf_j"):
            if joints_per_leg == 3 or "hip" not in joint:
                motors += f'<motor name="{joint}_m" joint="{joint}" ctrlrange="-30 30"/>'
    return (
        '<mujoco model="dog"><compiler angle="radian"/><default><joint damping="0.01"/></default><worldbody>'
        '<body name="trunk" pos="0 0 0.4"><freejoint/><geom type="box" size="0.2 0.08 0.04" mass="8"/>'
        f"{body}</body></worldbody><actuator>{motors}</actuator></mujoco>"
    )


def test_numbers_written_with_commas_are_read(tmp_path):
    path = write(
        tmp_path / "robot.xml",
        '<mujoco><worldbody><body pos="0, 0, 1" quat="0.707107, 0, 0.707107, 0"><freejoint/>'
        '<geom size="0.1" mass="1"/></body></worldbody></mujoco>',
    )
    with pytest.raises(ValueError, match="bad format"):
        mujoco.MjSpec.from_file(str(path))
    spec, repaired = mjcf_doctor.read(path)
    assert repaired and "2 number lists" in repaired[0]
    model = spec.compile()
    assert np.allclose(model.body_quat[1], [0.707107, 0, 0.707107, 0])
    assert path.read_text(encoding="utf-8").count(",") > 0  # the file itself is left as it was


def test_an_include_with_commas_is_repaired_too(tmp_path):
    write(
        tmp_path / "body.xml",
        '<mujoco><worldbody><body quat="1, 0, 0, 0"><geom size="0.1"/></body></worldbody></mujoco>',
    )
    path = write(tmp_path / "robot.xml", '<mujoco><include file="body.xml"/></mujoco>')
    spec, repaired = mjcf_doctor.read(path)
    assert len(repaired) == 1 and "body.xml" in repaired[0]
    assert spec.compile().nbody == 2


def test_a_file_with_another_problem_is_not_hidden(tmp_path):
    path = write(tmp_path / "robot.xml", "<mujoco><worldbody><nonsense/></worldbody></mujoco>")
    with pytest.raises(ValueError, match="nonsense"):
        mjcf_doctor.read(path)


def test_torque_motors_on_limited_joints_become_servos(tmp_path):
    spec = mujoco.MjSpec.from_string(_legs())
    spec.compiler.degree = False
    assert mjcf_doctor.drive_motors(spec) == 12
    model = spec.compile()
    gains = {round(float(model.actuator_gainprm[a, 0]), 1) for a in range(model.nu)}
    assert gains == {120.0}  # 30 N*m at 0.25 rad
    assert np.allclose(model.actuator_forcerange, [-30, 30])
    knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_calf_j_m")
    assert np.allclose(model.actuator_ctrlrange[knee], [-2.7, -0.9])  # a target is an angle within the joint's range


def test_a_wheel_keeps_its_torque_motor():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body><freejoint/><geom size="0.1" mass="1"/><body pos="0 0.2 0">'
        '<joint name="w" axis="0 1 0"/><geom type="sphere" size="0.05" mass="0.1"/></body></body></worldbody>'
        '<actuator><motor name="drive" joint="w" ctrlrange="-5 5"/></actuator></mujoco>'
    )
    assert mjcf_doctor.drive_motors(spec) == 0
    assert spec.compile().actuator_biastype[0] == mujoco.mjtBias.mjBIAS_NONE


def test_a_dog_without_a_keyframe_gets_a_standing_pose():
    spec = mujoco.MjSpec.from_string(_legs())
    mjcf_doctor.drive_motors(spec)
    qpos = mjcf_stance.find(spec)
    assert qpos is not None
    model = spec.compile()
    knee = model.joint("FR_calf_j")
    thigh = model.joint("FR_thigh_j")
    assert -2.7 <= qpos[knee.qposadr[0]] <= -0.9  # inside its limit, which the zero pose is not
    assert qpos[thigh.qposadr[0]] > 0.3  # the thigh swung forward and the knee bent back under it
    # all four legs in the same pose
    assert len({round(float(qpos[model.joint(f"{n}_thigh_j").qposadr[0]]), 3) for n in ("FR", "FL", "RR", "RL")}) == 1


def test_a_stance_is_for_dogs_only():
    assert mjcf_stance.find(mujoco.MjSpec.from_string(_legs(joints_per_leg=2))) is None
    arm = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body><joint axis="0 1 0"/><geom size="0.1"/></body></worldbody></mujoco>'
    )
    assert mjcf_stance.find(arm) is None
