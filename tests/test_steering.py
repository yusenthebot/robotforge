"""Small kinematic contracts; physical motion is separately verified on the source robot."""

from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from robotforge import capabilities
from robotforge.behaviors import Kind, find_wheels, home_pose, robot_kind
from robotforge.behaviors.drive import DiffDrive, rover_drive
from robotforge.behaviors.manager import BehaviorError
from robotforge.behaviors.mobility import describe_mobility
from robotforge.behaviors.steering import SteeringDrive, has_steering, steering_binding
from robotforge.control import make_task, task_world
from robotforge.control.classic import GoToGoal
from robotforge.control.tasks.base import TaskError
from robotforge.design.importing.geometry import Geometry
from robotforge.design.importing.plan import PLAN_VERSION, RigPlan
from robotforge.design.importing.rig import SKID_STEER_FRICTION, _friction
from robotforge.design.importing.urdf import JointInfo, StagedUrdf
from robotforge.playground import Playground
from robotforge.sim.session import SimSession


def _robot(name="r", *, steering=True, count=4, flip=False, offset=0.0, yaw=0.0, home=0.0):
    bodies, actuators = [], []
    for i, (x, y) in enumerate(((0.2, 0.2), (0.2, -0.2), (-0.2, 0.2), (-0.2, -0.2))[:count]):
        wheel_axis = -1 if flip and i % 2 else 1
        steer_axis = -1 if flip and i >= 2 else 1
        prefix = f"{name}/{i}"
        carrier = (
            f'<joint name="{prefix}/steer" axis="0 0 {steer_axis}" range="-2.5 2.5" ref="{home}"/>' if steering else ""
        )
        bodies.append(f'''<body name="{prefix}/carrier" pos="{x} {y} 0">
          {carrier}<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>
          <body name="{prefix}/wheel" pos="0 {offset} -.04">
            <joint name="{prefix}/wheel" axis="0 {wheel_axis} 0"/>
            <geom type="cylinder" size=".05 .015" quat=".7071067812 .7071067812 0 0" mass=".1"/>
          </body></body>''')
        if steering:
            actuators.append(
                f'<position name="{prefix}/steer" joint="{prefix}/steer" kp="10" '
                'ctrlrange="-2.5 2.5" forcerange="-1 1"/>'
            )
        actuators.append(
            f'<velocity name="{prefix}/wheel" joint="{prefix}/wheel" kv="2" ctrlrange="-10 10" forcerange="-2 2"/>'
        )
    return (
        f'''<body name="{name}/base" pos="0 0 .15" euler="0 0 {yaw}"><freejoint/>
      <geom type="box" size=".12 .12 .02" mass="1"/>{"".join(bodies)}</body>''',
        "".join(actuators),
    )


def _model(**kwargs):
    body, actuators = _robot(**kwargs)
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{body}</worldbody><actuator>{actuators}</actuator></mujoco>'
    )


def _data(model):
    data = mujoco.MjData(model)
    data.qpos[:] = home_pose(model)
    mujoco.mj_forward(model, data)
    return data


def _aligned(drive, model, data):
    """Set a test fixture's joint observations to target; never a physical-motion assertion."""
    drive.update(model, data)
    for module in drive.modules:
        data.qpos[module.qpos] = data.ctrl[module.steering]
    mujoco.mj_forward(model, data)
    drive.update(model, data)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("command", [(0.2, 0.0), (-0.2, 0.0), (0.0, 0.5), (0.15, -0.4)])
def test_axis_signs_and_lateral_offsets_satisfy_rolling_constraints(flip, command):
    model = _model(flip=flip, offset=0.026)
    data = _data(model)
    drive = rover_drive(model, "r")
    assert isinstance(drive, SteeringDrive)
    drive.command(*command)
    _aligned(drive, model, data)
    for m in drive.modules:
        joint = int(model.actuator_trnid[m.wheel, 0])
        axle = data.xaxis[joint]
        rolling = np.cross(axle, [0, 0, 1])
        point = data.xanchor[joint] - data.xpos[model.body("r/base").id]
        velocity = np.array([command[0] - command[1] * point[1], command[1] * point[0], 0])
        assert velocity @ axle == pytest.approx(0, abs=1e-8)
        assert data.ctrl[m.wheel] * m.radius == pytest.approx(velocity @ rolling, abs=1e-8)


def test_steering_waits_for_every_module_and_stop_preserves_pose():
    model = _model()
    data = _data(model)
    drive = rover_drive(model, "r")
    drive.command(0, 1)
    before = data.qpos.copy()
    drive.update(model, data)
    np.testing.assert_array_equal(data.qpos, before)
    assert drive.aligning
    assert all(data.ctrl[m.wheel] == 0 for m in drive.modules)
    for m in drive.modules[:-1]:
        data.qpos[m.qpos] = data.ctrl[m.steering]
    drive.update(model, data)
    assert all(data.ctrl[m.wheel] == 0 for m in drive.modules)
    _aligned(drive, model, data)
    assert any(abs(data.ctrl[m.wheel]) > 0 for m in drive.modules)
    before, velocities = data.qpos.copy(), data.qvel.copy()
    drive.command()
    drive.update(model, data)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_array_equal(data.qvel, velocities)
    assert all(data.ctrl[m.wheel] == 0 for m in drive.modules)


def test_saturation_uses_every_asymmetric_wheel_range_and_keeps_twist():
    model = _model()
    modules = steering_binding(model, "r")
    slow = modules[-1].wheel
    model.actuator_ctrlrange[slow] = [-1, 2]
    data = _data(model)
    drive = rover_drive(model, "r")
    assert drive.max_speed == pytest.approx(0.05)
    drive.command(0.04, 4)
    _aligned(drive, model, data)
    scales = []
    for m in drive.modules:
        angle = m.heading + m.sign * (data.ctrl[m.steering] - m.home)
        expected = (
            (drive.speed - drive.turn * m.pivot[1]) * math.cos(angle) + drive.turn * m.pivot[0] * math.sin(angle)
        ) / m.radius
        scales.append(data.ctrl[m.wheel] / expected)
        assert m.wheel_lower <= data.ctrl[m.wheel] <= m.wheel_upper
        assert m.lower <= data.ctrl[m.steering] <= m.upper
    assert 0 < scales[0] < 1
    np.testing.assert_allclose(scales, scales[0])


def test_binding_uses_home_root_frame_and_description_does_not_mutate_live_state():
    model = _model(yaw=1.2, home=0.3)
    data = _data(model)
    drive = rover_drive(model, "r")
    assert all(m.home == pytest.approx(0.3) for m in drive.modules)
    np.testing.assert_allclose(drive.modules[0].pivot, [0.2, 0.2], atol=1e-8)
    before = data.qpos.copy(), data.ctrl.copy(), model.actuator_forcerange.copy()
    assert describe_mobility(model, "r", Kind.ROVER, home_pose(model), None, data) == drive.max_speed
    np.testing.assert_array_equal(data.ctrl, before[1])
    drive.command(0.1, 0)
    drive.update(model, data)
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(model.actuator_forcerange, before[2])
    assert all(data.ctrl[m.steering] == pytest.approx(0.3) for m in drive.modules)
    assert all(data.ctrl[m.wheel] == pytest.approx(2) for m in drive.modules)


def test_controller_scopes_robot_and_legacy_diff_drive_stays_available():
    first, a = _robot("first", flip=True)
    second, b = _robot("second", steering=False)
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><compiler angle="radian"/><worldbody>{first}{second}</worldbody><actuator>{a}{b}</actuator></mujoco>'
    )
    data = _data(model)
    assert robot_kind(model, "first") == Kind.ROVER
    assert find_wheels(model, "first") is None
    assert isinstance(rover_drive(model, "second"), DiffDrive)
    assert find_wheels(model, "second") is not None
    untouched = [i for i in range(model.nu) if model.actuator(i).name.startswith("second/")]
    data.ctrl[untouched] = 0.123
    drive = rover_drive(model, "first")
    drive.command(0.2, 0)
    drive.update(model, data)
    np.testing.assert_array_equal(data.ctrl[untouched], 0.123)
    with pytest.raises(ValueError, match="steering joints"):
        DiffDrive(model, "first")


@pytest.mark.parametrize(
    "case,reason",
    [
        ("two", "at least three"),
        ("range", "steering travel"),
        ("unlimited", "finite control"),
        ("gear", "unit-gear"),
        ("tilt", "horizontal wheel"),
        ("trail", "wheel trail"),
        ("wheel_range", "both forward and reverse"),
        ("wheel_joint", "continuously"),
        ("passive", "one position servo"),
    ],
)
def test_unsupported_geometry_is_refused_by_backend_and_capability_provider(case, reason):
    model = _model(count=2 if case == "two" else 4)
    steer = model.actuator("r/0/steer").id
    wheel = model.actuator("r/0/wheel").id
    if case == "range":
        model.actuator_ctrlrange[steer] = [-0.5, 0.5]
    elif case == "unlimited":
        model.actuator_ctrllimited[wheel] = 0
    elif case == "gear":
        model.actuator_gear[wheel, 0] = -1
    elif case == "tilt":
        model.jnt_axis[model.actuator_trnid[wheel, 0]] = [0, 0, 1]
    elif case == "trail":
        model.body_pos[model.body("r/0/wheel").id, 0] = 0.02
    elif case == "wheel_range":
        model.actuator_ctrlrange[wheel] = [0.1, 1]
    elif case == "wheel_joint":
        model.jnt_limited[model.actuator_trnid[wheel, 0]] = True
    elif case == "passive":
        model.actuator_biasprm[steer, 1] = 0
    data = _data(model)
    before = data.ctrl.copy()
    assert robot_kind(model, "r") == Kind.ROVER
    with pytest.raises(ValueError, match=reason):
        describe_mobility(model, "r", Kind.ROVER, home_pose(model), None, data)
    playground = Playground(session=SimSession(model, data))
    with pytest.raises(BehaviorError, match=reason):
        playground.behaviors.move("r", 0.1, 0.2)
    np.testing.assert_array_equal(data.ctrl, before)


def test_manager_stop_clears_wheel_controls_before_the_next_tick():
    model = _model()
    data = _data(model)
    playground = Playground(session=SimSession(model, data))
    playground.behaviors.move("r", 0.2, 0)
    drive = playground.session.controllers["r"]
    drive.update(model, data)
    assert any(data.ctrl[m.wheel] != 0 for m in drive.modules)
    playground.behaviors.stop("r")
    assert all(data.ctrl[m.wheel] == 0 for m in drive.modules)


def _staged_wheels(model):
    joints = []
    for i in range(4):
        joints += [
            JointInfo(f"r/{i}/steer", "revolute", "r/base", f"r/{i}/carrier", axis=(0, 0, 1)),
            JointInfo(f"r/{i}/wheel", "continuous", f"r/{i}/carrier", f"r/{i}/wheel", axis=(0, 1, 0)),
        ]
    return StagedUrdf("", "r/base", ["r/base"], [model.body(b).name for b in range(1, model.nbody)], joints, {}, {})


def test_steered_import_keeps_tire_friction_and_source_override():
    model = _model()
    staged, geo = _staged_wheels(model), Geometry(model, _data(model))
    assert _friction(staged, geo) == {}
    staged.link_friction = {"r/0/wheel": 0.9}
    assert _friction(staged, geo) == {"r/0/wheel": 0.9}
    fixed = replace(
        staged,
        joints=[replace(j, type="fixed") if j.name.endswith("/steer") else j for j in staged.joints],
        link_friction={},
    )
    assert _friction(fixed, geo) == {f"r/{i}/wheel": SKID_STEER_FRICTION for i in range(4)}
    with pytest.raises(ValueError, match="stale plan"):
        RigPlan.from_json({"version": PLAN_VERSION - 1})


@pytest.mark.parametrize("kind,axis", [("hinge", [1, 0, 0]), ("hinge", [0, 1, 0]), ("slide", [0, 0, 1])])
def test_nonsteering_suspension_keeps_legacy_drive_and_skid_friction(kind, axis):
    model = _model(yaw=1.2, home=0.3 if axis == [0, 1, 0] else 0)
    staged = _staged_wheels(model)
    for i in range(4):
        joint = model.joint(f"r/{i}/steer").id
        model.jnt_axis[joint] = axis
        model.jnt_type[joint] = mujoco.mjtJoint.mjJNT_HINGE if kind == "hinge" else mujoco.mjtJoint.mjJNT_SLIDE
        staged.joints[2 * i] = replace(
            staged.joints[2 * i], type="revolute" if kind == "hinge" else "prismatic", axis=tuple(axis)
        )
    data = _data(model)
    assert not has_steering(model, "r")
    assert steering_binding(model, "r") is None
    assert find_wheels(model, "r") is not None
    assert isinstance(rover_drive(model, "r"), DiffDrive)
    assert _friction(staged, Geometry(model, data)) == {f"r/{i}/wheel": SKID_STEER_FRICTION for i in range(4)}


@pytest.mark.parametrize("tilt", [0.1, 0.7])
def test_tilted_yaw_carriers_still_refuse_differential_fallback(tilt):
    model = _model(yaw=1.1)
    for i in range(4):
        model.jnt_axis[model.joint(f"r/{i}/steer").id] = [math.sin(tilt), 0, math.cos(tilt)]
    assert has_steering(model, "r")
    assert find_wheels(model, "r") is None
    with pytest.raises(ValueError, match="vertical steering axes"):
        rover_drive(model, "r")
    assert _friction(_staged_wheels(model), Geometry(model, _data(model))) == {}


def test_mixed_steering_and_pitch_suspension_is_not_differential_drive():
    model = _model()
    model.jnt_axis[model.joint("r/0/steer").id] = [0, 1, 0]
    assert has_steering(model, "r")
    with pytest.raises(ValueError, match="vertical steering axes"):
        rover_drive(model, "r")


def test_detection_and_import_friction_use_rotated_joint_frames():
    model = _model(yaw=1.1)
    for i in range(4):
        # Local X becomes root Z after this fixed joint-frame rotation; local metadata alone is wrong.
        model.body_quat[model.body(f"r/{i}/carrier").id] = [math.sqrt(0.5), 0, -math.sqrt(0.5), 0]
        model.jnt_axis[model.joint(f"r/{i}/steer").id] = [1, 0, 0]
    staged = _staged_wheels(model)
    staged.joints = [replace(j, axis=(1, 0, 0)) if j.name.endswith("/steer") else j for j in staged.joints]
    assert has_steering(model, "r")
    assert _friction(staged, Geometry(model, _data(model))) == {}


@pytest.mark.parametrize("speed,turn", [(math.nan, 0), (0, math.inf), (-math.inf, 0)])
def test_nonfinite_commands_do_not_replace_the_previous_command(speed, turn):
    model = _model()
    drive = rover_drive(model, "r")
    drive.command(0.1, 0.2)
    with pytest.raises(ValueError, match="finite"):
        drive.command(speed, turn)
    assert (drive.speed, drive.turn) == (0.1, 0.2)


def test_classic_template_driver_retains_two_wheel_action_space():
    model, robot = task_world("drive_to")
    task = make_task("drive_to", model, robot, {"goal": [1, 0]})
    assert task.actuators.tolist() == [task.left, task.right]
    assert len(task.actuators) == 2
    controller = GoToGoal(task, model)
    assert isinstance(controller.drive, DiffDrive)


def test_steering_task_reports_its_existing_action_space_limit():
    model = _model()
    with pytest.raises(TaskError, match="two-wheel action space"):
        make_task("drive_to", model, "r")
    structure = capabilities.inspect_robot(model, "r", Kind.ROVER)
    tasks = dict(structure.tasks)
    assert not tasks["drive_to"].supported
    assert "independent steering" in tasks["drive_to"].reason


def test_two_wheels_on_pitch_legs_reach_the_legacy_balancer_detector(monkeypatch):
    from robotforge.behaviors.balance import find_balancer

    model = _model(count=2)
    for i in range(2):
        model.jnt_axis[model.joint(f"r/{i}/steer").id] = [0, 1, 0]
    # This binding test does not retest the balancer's separate one-second physical settling check.
    monkeypatch.setattr("robotforge.behaviors.balance._stands_on_its_own", lambda *args: False)
    assert not has_steering(model, "r")
    assert steering_binding(model, "r") is None
    assert find_balancer(model, "r", home_pose(model)) is not None
