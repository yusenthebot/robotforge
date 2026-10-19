"""The import doctor end to end: synthetic URDFs and meshes in, working simulated robots out."""

from __future__ import annotations

import json
import re
import sys

import mujoco
import numpy as np
import pytest
import trimesh
from importer_fixtures import (
    arm_urdf,
    ascii_stl,
    box_link,
    box_stl,
    dae_text,
    inertial,
    revolute,
    robot,
    rover_urdf,
    write,
)

from robotforge.design import importer
from robotforge.design.importer import ImportOptions, compile_standalone, import_file
from robotforge.design.world import World, compile_world


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _world(spec_result, name="r"):
    """Attach an import result to a floor like the world compiler does and return (model, data)."""
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    frame = world.worldbody.add_frame(pos=[0, 0, spec_result.spawn_height])
    world.attach(spec_result.spec, prefix=f"{name}/", frame=frame)
    model = world.compile()
    data = mujoco.MjData(model)
    for joint, value in spec_result.home_qpos.items():
        data.qpos[model.jnt_qposadr[model.joint(f"{name}/{joint}").id]] = value
    for act, value in spec_result.home_ctrl.items():
        data.ctrl[model.actuator(f"{name}/{act}").id] = value
    mujoco.mj_forward(model, data)
    return model, data


def _run(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


def _topics(result, status=None):
    return {f.topic for f in result.report.findings if status is None or f.status == status}


# ---- arms -------------------------------------------------------------------------------------


def test_arm_urdf_opens_with_motors_stable_pose_and_split_geoms(tmp_path):
    result = import_file(write(tmp_path / "arm.urdf", arm_urdf()))
    assert result.kind == "robot" and not result.floating
    assert set(result.home_ctrl) == {"shoulder", "elbow"}

    model, data = _world(result)
    assert model.nu == 2
    for a in range(model.nu):
        assert model.actuator_ctrllimited[a] and model.actuator_forcelimited[a]
        assert model.actuator_forcerange[a][1] == pytest.approx(20.0)  # the URDF effort limit is respected
        assert tuple(model.actuator_ctrlrange[a]) == pytest.approx((-1.5, 1.5))

    visual = [g for g in range(model.ngeom) if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0]
    colliding = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    assert visual and colliding
    assert {int(model.geom_group[g]) for g in visual} == {2}  # visible, never collides
    assert {int(model.geom_group[g]) for g in colliding if model.geom_bodyid[g] > 1} == {3}  # hidden, like Menagerie

    shoulder = model.jnt_qposadr[model.joint("r/shoulder").id]
    _run(model, data, 2.0)
    assert abs(data.qpos[shoulder]) < 0.03  # holds its pose against gravity


def test_arm_sliders_move_the_joints(tmp_path):
    result = import_file(write(tmp_path / "arm.urdf", arm_urdf()))
    model, data = _world(result)
    data.ctrl[model.actuator("r/elbow").id] = 0.9
    _run(model, data, 1.5)
    assert data.qpos[model.jnt_qposadr[model.joint("r/elbow").id]] == pytest.approx(0.9, abs=0.05)


def test_report_lists_the_added_motors_and_the_base_choice(tmp_path):
    result = import_file(write(tmp_path / "arm.urdf", arm_urdf()))
    motors = next(f for f in result.report.findings if f.topic == "actuators")
    assert motors.status == "fixed" and set(motors.items) == {"shoulder", "elbow"}
    base = next(f for f in result.report.findings if f.topic == "base")
    assert "fixed base" in base.problem
    assert any(f.topic == "stability" and f.status == "info" for f in result.report.findings)
    assert result.report.stats["motors"] == 2


def test_base_override_forces_floating_or_fixed(tmp_path):
    path = write(tmp_path / "arm.urdf", arm_urdf())
    floating = import_file(path, {"base": "floating"})
    assert floating.floating
    model, _ = _world(floating)
    assert model.nq == 7 + 2  # free joint plus two hinges
    assert "you asked for it" in next(f.problem for f in floating.report.findings if f.topic == "base")
    assert not import_file(path, {"base": "fixed"}).floating


def test_floating_root_that_is_a_bare_frame_still_compiles(tmp_path):
    body = (
        '<link name="base_footprint"/><joint name="f" type="fixed"><parent link="base_footprint"/>'
        '<child link="base_link"/><origin xyz="0 0 0.05"/></joint>'
        + box_link("base_link", (0.2, 0.2, 0.1))
        + box_link("tip", (0.05, 0.05, 0.1))
        + revolute("j", "base_link", "tip", "0 0 0.1")
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)), {"base": "floating"})
    model, data = _world(result)
    _run(model, data, 0.5)


# ---- URDF repairs -----------------------------------------------------------------------------


def test_missing_zero_and_broken_inertia_are_repaired_from_shapes(tmp_path):
    zero = '<inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>'
    impossible = (
        '<inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
    )
    geom = '<geometry><box size="0.1 0.1 0.1"/></geometry>'
    body = (
        box_link("base_link", (0.2, 0.2, 0.1), 5.0)
        + f'<link name="no_inertial"><collision>{geom}</collision></link>'
        + f'<link name="zero_mass">{zero}<collision>{geom}</collision></link>'
        + f'<link name="bad_triangle">{impossible}<collision>{geom}</collision></link>'
        + '<link name="placeholder"/>'
        + revolute("j1", "base_link", "no_inertial")
        + revolute("j2", "no_inertial", "zero_mass")
        + revolute("j3", "zero_mass", "bad_triangle")
        + revolute("j4", "bad_triangle", "placeholder")
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    fixed = {f.problem: f for f in result.report.findings if f.topic == "inertia"}
    assert any("no mass or inertia" in p and "no_inertial" in f.items for p, f in fixed.items())
    assert any("zero, negative or tiny mass" in p and "zero_mass" in f.items for p, f in fixed.items())
    assert any("impossible" in p and "bad_triangle" in f.items for p, f in fixed.items())
    assert any("no shape and no mass" in p and "placeholder" in f.items for p, f in fixed.items())
    model, data = _world(result)  # compiles: every moving body has a valid mass and inertia
    assert (model.body_mass[2:] > 0).all()
    _run(model, data, 1.0)


def test_inertia_that_does_not_fit_the_link_is_rebuilt_but_the_mass_is_kept(tmp_path):
    huge = '<inertial><mass value="2"/><inertia ixx="500" ixy="0" ixz="0" iyy="500" iyz="0" izz="500"/></inertial>'
    geom = '<geometry><box size="0.1 0.1 0.1"/></geometry>'
    body = box_link("base_link") + f'<link name="arm">{huge}<collision>{geom}</collision></link>'
    body += revolute("j", "base_link", "arm")
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    model, _ = _world(result)
    arm = model.body("r/arm").id
    assert model.body_mass[arm] == pytest.approx(2.0)
    assert model.body_inertia[arm].max() < 1.0


def test_joint_limit_problems_are_repaired(tmp_path):
    body = (
        box_link("base_link")
        + "".join(box_link(n) for n in ("a", "b", "c", "d"))
        + revolute("swapped", "base_link", "a", limit='<limit lower="1" upper="-1" effort="5" velocity="1"/>')
        + revolute("unbounded", "a", "b", limit='<limit effort="5" velocity="1"/>')
        + revolute("zero_range", "b", "c", limit='<limit lower="0" upper="0" effort="5" velocity="1"/>')
        + revolute("no_axis_limit", "c", "d", axis="0 0 0", limit="")
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    text = " ".join(f.problem for f in result.report.findings if f.topic == "joints")
    assert "limits were missing or inconsistent" in text and "no usable rotation axis" in text
    model, _ = _world(result)
    assert model.jnt_range[model.joint("r/swapped").id].tolist() == [-1.0, 1.0]
    assert not model.jnt_limited[model.joint("r/unbounded").id]  # became continuous
    assert "r/zero_range" not in {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)}


def test_joint_range_not_containing_zero_starts_inside_the_range(tmp_path):
    body = (
        box_link("base_link")
        + box_link("arm")
        + revolute("j", "base_link", "arm", limit='<limit lower="0.5" upper="1.5" effort="5" velocity="1"/>')
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    assert result.home_qpos == {"j": 0.5} and result.home_ctrl == {"j": 0.5}
    model, data = _world(result)
    _run(model, data, 1.0)
    assert data.qpos[model.jnt_qposadr[model.joint("r/j").id]] == pytest.approx(0.5, abs=0.05)


def test_mimic_joints_follow_their_leader_and_get_no_motor(tmp_path):
    finger = (
        '<joint name="{n}" type="prismatic"><parent link="palm"/><child link="{c}"/>'
        '<axis xyz="1 0 0"/><limit lower="0" upper="0.04" effort="10" velocity="1"/>{m}</joint>'
    )
    body = (
        box_link("palm", (0.1, 0.05, 0.05))
        + box_link("f1", (0.01, 0.01, 0.04), 0.05)
        + box_link("f2", (0.01, 0.01, 0.04), 0.05)
        + finger.format(n="finger1", c="f1", m="")
        + finger.format(n="finger2", c="f2", m='<mimic joint="finger1" multiplier="-1" offset="0.04"/>')
    )
    result = import_file(write(tmp_path / "g.urdf", robot(body)))
    assert set(result.home_ctrl) == {"finger1"}
    model, data = _world(result)
    assert model.nu == 1 and model.neq >= 1


def test_closed_loops_and_dangling_joints_get_actionable_errors(tmp_path):
    loop = robot(box_link("a") + box_link("b") + box_link("c") + revolute("j1", "a", "c") + revolute("j2", "b", "c"))
    with pytest.raises(ValueError, match="closed loop"):
        import_file(write(tmp_path / "loop.urdf", loop))
    dangling = robot(box_link("a") + box_link("b") + revolute("j", "a", "b") + revolute("lost", "b", "ghost"))
    result = import_file(write(tmp_path / "dangling.urdf", dangling))
    assert any(
        f.topic == "structure" and "lead to links that do not exist" in f.problem for f in result.report.findings
    )
    _world(result)
    headless = robot(box_link("arm") + revolute("mount", "bracket", "arm"))  # parent link missing: made an empty frame
    created = import_file(write(tmp_path / "headless.urdf", headless))
    assert any(f.topic == "structure" and "never defines" in f.problem for f in created.report.findings)
    _world(created)
    implicit_world = robot(box_link("arm") + revolute("mount", "world", "arm"))
    assert not import_file(write(tmp_path / "world.urdf", implicit_world)).floating  # the usual sign of a fixed base
    with pytest.raises(ValueError, match="no <link>"):
        import_file(write(tmp_path / "empty.urdf", robot("")))
    with pytest.raises(ValueError, match="not valid XML"):
        import_file(write(tmp_path / "broken.urdf", "<robot><link name='a'></robot>"))
    with pytest.raises(ValueError, match="not a URDF"):
        import_file(write(tmp_path / "other.urdf", "<mujoco/>"))
    with pytest.raises(ValueError, match="file not found"):
        import_file(tmp_path / "missing.urdf")


def test_disconnected_pieces_are_reported_not_fatal(tmp_path):
    body = box_link("a") + box_link("b") + box_link("c") + revolute("j", "a", "b")
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    assert any(f.topic == "structure" and f.status == "unfixed" for f in result.report.findings)
    _world(result)


def test_xacro_flavoured_urdf_is_expanded(tmp_path):
    text = f"""<robot name="r" xmlns:xacro="http://ros.org/wiki/xacro">
    <xacro:arg name="ns" default="pre_"/>
    <xacro:property name="size" value="0.2"/>
    <link name="$(arg ns)base">{inertial()}<collision><geometry>
      <box size="${{size}} ${{size}} 0.1"/></geometry></collision></link>
    <link name="arm">{inertial()}<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
    <joint name="j" type="revolute"><parent link="$(arg ns)base"/><child link="arm"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="1"/></joint></robot>"""
    result = import_file(write(tmp_path / "r.urdf", text))
    assert "xacro" in _topics(result, "fixed")
    model, _ = _world(result)
    assert model.body("r/pre_base").id > 0
    renamed = import_file(write(tmp_path / "r.urdf", text), {"xacro_args": {"ns": "x_"}})
    assert _world(renamed)[0].body("r/x_base").id > 0


def test_xacro_file_with_python_helper_explains_what_to_do(tmp_path):
    text = '<robot name="r" xmlns:xacro="http://ros.org/wiki/xacro"><link name="${made_up_helper(1)}"/></robot>'
    with pytest.raises(ValueError, match="Python helper"):
        import_file(write(tmp_path / "bad.xacro", text))


# ---- meshes in URDFs ---------------------------------------------------------------------------


def _mesh_arm(tmp_path, mesh_uri: str, scale: str = "1 1 1"):
    """A two-link arm whose upper link is a mesh (visual and collision)."""
    geom = f'<geometry><mesh filename="{mesh_uri}" scale="{scale}"/></geometry>'
    body = (
        box_link("base_link", (0.2, 0.2, 0.1), 5.0, "0 0 0.05")
        + f'<link name="upper">{inertial(1.0, 0.01)}<visual>{geom}</visual><collision>{geom}</collision></link>'
        + revolute("shoulder", "base_link", "upper", "0 0 0.1")
    )
    return write(tmp_path / "pkg" / "urdf" / "arm.urdf", robot(body))


def test_package_paths_and_non_native_meshes_are_fixed_in_one_import(tmp_path):
    box_stl(tmp_path / "pkg" / "meshes" / "upper.stl", (0.05, 0.05, 0.3))
    urdf = _mesh_arm(tmp_path, "package://pkg/meshes/upper.stl")
    result = import_file(urdf)
    model, data = _world(result)
    assert model.nmesh >= 1 and model.ngeom >= 4
    _run(model, data, 1.0)
    assert not [f for f in result.report.findings if f.status == "unfixed"]


def test_dae_visuals_convert_and_keep_their_color(tmp_path):
    write(tmp_path / "pkg" / "meshes" / "upper.dae", dae_text(color=(0.0, 0.0, 1.0, 1.0), size=(0.05, 0.05, 0.3)))
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.dae"))
    assert any("DAE" in f.problem and f.status == "fixed" for f in result.report.findings)
    model, _ = _world(result)
    blue = [g for g in range(model.ngeom) if np.allclose(model.geom_rgba[g], [0, 0, 1, 1])]
    assert blue  # the visual picked up the color stored in the DAE


def test_ascii_stl_and_millimeter_meshes_are_fixed(tmp_path):
    ascii_stl(tmp_path / "pkg" / "meshes" / "upper.stl", (50.0, 50.0, 300.0))  # millimeters
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.stl"))
    texts = " ".join(f.problem + f.action for f in result.report.findings)
    assert "millimeters" in texts and "ASCII" in texts
    model, _ = _world(result)
    upper = model.body("r/upper").id
    sizes = [model.geom_rbound[g] for g in range(model.ngeom) if model.geom_bodyid[g] == upper]
    assert max(sizes) < 1.0  # about 0.3 m, not 300 m


def test_missing_mesh_is_dropped_and_reported_but_the_robot_still_opens(tmp_path):
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/gone.stl"))
    gone = next(f for f in result.report.findings if f.status == "unfixed" and f.topic == "meshes")
    assert "gone.stl" in " ".join(gone.items)
    model, data = _world(result)
    _run(model, data, 0.5)


def test_unreadable_mesh_is_dropped_and_reported(tmp_path):
    write(tmp_path / "pkg" / "meshes" / "upper.dae", "<html>not a collada file</html>")
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.dae"))
    bad = next(f for f in result.report.findings if f.status == "unfixed" and f.topic == "meshes")
    assert "could not be read" in bad.problem
    assert "pycollada" in bad.problem  # tells the user which package would widen what can be read
    _world(result)


def test_visual_only_urdf_gets_collision_so_it_does_not_fall_through_the_floor(tmp_path):
    geom = '<geometry><box size="0.1 0.1 0.1"/></geometry>'
    body = (
        f'<link name="base_link">{inertial(1.0)}<visual>{geom}</visual></link>'
        f'<link name="arm">{inertial()}<visual>{geom}</visual></link>' + revolute("j", "base_link", "arm")
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    assert "collision" in _topics(result, "fixed")
    model, _ = _world(result)
    assert any(model.geom_contype[g] for g in range(model.ngeom))


def test_heavy_collision_meshes_are_warned_about_and_capped(tmp_path):
    sphere = trimesh.creation.icosphere(subdivisions=5)  # 20k faces, 10k vertices
    (tmp_path / "pkg" / "meshes").mkdir(parents=True)
    sphere.export(tmp_path / "pkg" / "meshes" / "upper.stl")
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.stl"))
    heavy = next(f for f in result.report.findings if f.topic == "collision" and "thousands" in f.problem)
    assert "upper.stl" in heavy.items[0] and "decompose" in heavy.action
    model, _ = _world(result)
    assert any(f.topic == "collision" and "limited the collision outline" in f.problem for f in result.report.findings)
    assert any(model.geom_type[g] == 7 and model.geom_contype[g] for g in range(model.ngeom))


def test_concave_collision_meshes_can_be_decomposed_with_coacd(tmp_path):
    a = trimesh.creation.icosphere(subdivisions=3, radius=0.08)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.08)
    b.apply_translation([0.3, 0, 0])
    (tmp_path / "pkg" / "meshes").mkdir(parents=True)
    trimesh.util.concatenate([a, b]).export(tmp_path / "pkg" / "meshes" / "upper.stl")  # a dumbbell
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.stl"), {"decompose": True})
    assert any("concave" in f.problem and f.status == "fixed" for f in result.report.findings)
    model, _ = _world(result)
    colliding_meshes = [g for g in range(model.ngeom) if model.geom_type[g] == 7 and model.geom_contype[g]]
    assert len(colliding_meshes) >= 2


# ---- wheels and legs ---------------------------------------------------------------------------


def test_rover_gets_wheel_speed_motors_and_drives(tmp_path):
    result = import_file(write(tmp_path / "rover.urdf", rover_urdf()))
    assert result.floating and set(result.home_ctrl) == {"left_wheel_joint", "right_wheel_joint"}
    assert "base" in _topics(result, "info")
    model, data = _world(result)
    assert model.nu == 2  # the caster (a fixed joint) and nothing else
    for a in range(model.nu):
        assert model.actuator_biasprm[a][1] == 0.0 and model.actuator_biasprm[a][2] < 0  # speed motors, not position
    start = data.xpos[model.body("r/chassis").id].copy()
    data.ctrl[:] = 4.0
    _run(model, data, 3.0)
    travelled = data.xpos[model.body("r/chassis").id] - start
    assert travelled[0] > 0.5 and abs(travelled[1]) < 0.1  # 4 rad/s * 0.05 m = 0.2 m/s ... minus slip; drives straight
    assert travelled[2] == pytest.approx(0.0, abs=0.02)


def test_caster_skid_gets_low_friction_so_it_does_not_stall_the_robot(tmp_path):
    result = import_file(write(tmp_path / "rover.urdf", rover_urdf()))
    model, _ = _world(result)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("r/caster_link").id]
    assert any(model.geom_friction[g][0] == pytest.approx(0.05) for g in geoms)


def _walker_urdf(legs: int = 4) -> str:
    """A box body on four straight legs (hip joint + knee joint each): stands in its zero pose."""
    body = box_link("trunk", (0.4, 0.2, 0.08), 4.0, "0 0 0")
    for i in range(legs):
        x = 0.15 if i % 2 == 0 else -0.15
        y = 0.12 if i < 2 else -0.12
        body += box_link(f"thigh{i}", (0.04, 0.04, 0.2), 0.3, "0 0 -0.1")
        body += box_link(f"shin{i}", (0.03, 0.03, 0.2), 0.2, "0 0 -0.1")
        body += revolute(
            f"hip{i}", "trunk", f"thigh{i}", f"{x} {y} -0.04", limit='<limit lower="-1" upper="1" effort="30"/>'
        )
        body += revolute(
            f"knee{i}", f"thigh{i}", f"shin{i}", "0 0 -0.2", limit='<limit lower="-1" upper="1" effort="30"/>'
        )
    return robot(body, "walker")


def test_legged_robot_is_floating_and_stands_on_its_own(tmp_path):
    result = import_file(write(tmp_path / "walker.urdf", _walker_urdf()))
    assert result.floating
    base = next(f for f in result.report.findings if f.topic == "base")
    assert "limbs" in base.problem
    model, data = _world(result)
    root = model.body("r/trunk").id
    start_height = data.xpos[root][2]
    _run(model, data, 3.0)
    assert data.xpos[root][2] > 0.7 * start_height
    assert data.xmat[root].reshape(3, 3)[2, 2] > 0.9
    assert result.spawn_height > 0.3  # starts with its feet on the floor, not buried in it


def test_robot_that_cannot_hold_itself_up_is_reported_honestly(tmp_path):
    one_legged = _walker_urdf(legs=1)  # a body on one leg cannot balance
    result = import_file(write(tmp_path / "weak.urdf", one_legged), {"base": "floating"})
    stability = [f for f in result.report.findings if f.topic == "stability"]
    assert stability and stability[0].status == "unfixed"
    assert "does not hold" in stability[0].problem


def test_weak_effort_limits_are_relaxed_by_default_and_reported(tmp_path):
    weak = arm_urdf().replace('effort="20"', 'effort="0.05"')
    result = import_file(write(tmp_path / "arm.urdf", weak))
    relaxed = next(f for f in result.report.findings if f.topic == "actuators" and "too weak" in f.problem)
    assert (
        relaxed.status == "fixed"
        and "shoulder" in " ".join(relaxed.items)
        and "respect_effort_limits" in relaxed.action
    )
    assert not [f for f in result.report.findings if f.status == "unfixed"]
    model, data = _world(result)
    assert model.actuator_forcerange[model.actuator("r/shoulder").id][1] > 1.0
    _run(model, data, 1.5)
    assert abs(data.qpos[model.jnt_qposadr[model.joint("r/shoulder").id]]) < 0.05


def test_respecting_effort_limits_keeps_a_weak_robot_weak_and_says_so(tmp_path):
    weak = arm_urdf().replace('effort="20"', 'effort="0.05"')
    result = import_file(write(tmp_path / "arm.urdf", weak), {"respect_effort_limits": True})
    flagged = [f for f in result.report.findings if f.topic == "actuators" and f.status == "unfixed"]
    assert flagged and "shoulder" in " ".join(flagged[0].items) and "drop respect_effort_limits" in flagged[0].action
    model, _ = _world(result)
    assert model.actuator_forcerange[model.actuator("r/shoulder").id][1] == pytest.approx(0.05)


def test_adequate_effort_limits_are_kept(tmp_path):
    result = import_file(write(tmp_path / "arm.urdf", arm_urdf()))
    assert not any("too weak" in f.problem for f in result.report.findings)
    model, _ = _world(result)
    assert model.actuator_forcerange[0][1] == pytest.approx(20.0)


# ---- plain meshes -------------------------------------------------------------------------------


def test_stl_imports_as_a_fixed_object_resting_on_the_floor(tmp_path):
    result = import_file(box_stl(tmp_path / "block.stl", (0.1, 0.2, 0.3)))
    assert result.kind == "object" and not result.floating
    assert result.spawn_height == pytest.approx(0.15)  # half the height: the box sits on the floor
    model, data = _world(result, "block")
    _run(model, data, 0.5)
    assert model.nq == 0  # fixed in place
    assert any("fixed in place" in f.problem for f in result.report.findings)


def test_mesh_object_can_float_with_a_given_mass(tmp_path):
    result = import_file(box_stl(tmp_path / "block.stl", (0.1, 0.1, 0.1)), {"base": "floating", "mass": 2.5})
    assert result.floating
    model, data = _world(result, "block")
    assert model.body_mass[model.body("block/object").id] == pytest.approx(2.5)
    _run(model, data, 1.0)
    assert data.xpos[model.body("block/object").id][2] == pytest.approx(0.05, abs=0.01)  # resting on the floor


def test_mesh_object_scale_and_millimeter_detection(tmp_path):
    mm = import_file(box_stl(tmp_path / "part.stl", (50.0, 50.0, 100.0)))
    assert any(f.topic == "units" for f in mm.report.findings)
    assert mm.spawn_height == pytest.approx(0.05)
    scaled = import_file(box_stl(tmp_path / "part2.stl", (1.0, 1.0, 1.0)), {"scale": 0.1})
    assert scaled.spawn_height == pytest.approx(0.05)


def test_glb_obj_and_dae_objects_import(tmp_path):
    box = trimesh.creation.box((0.1, 0.1, 0.1))
    box.export(tmp_path / "a.glb")
    box.export(tmp_path / "b.obj")
    write(tmp_path / "c.dae", dae_text(size=(0.1, 0.1, 0.1)))
    for name in ("a.glb", "b.obj", "c.dae"):
        model, _ = _world(import_file(tmp_path / name), "o")
        assert model.nmesh >= 1


def test_object_decomposition_makes_several_collision_pieces(tmp_path):
    a = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    b.apply_translation([0.35, 0, 0])
    dumbbell = trimesh.util.concatenate([a, b])
    dumbbell.export(tmp_path / "bell.stl")
    result = import_file(tmp_path / "bell.stl", {"decompose": True})
    model, _ = _world(result, "o")
    hulls = [g for g in range(model.ngeom) if model.geom_contype[g] and model.geom_group[g] == 3]
    assert len(hulls) >= 2
    assert any(f.topic == "collision" and f.status == "fixed" for f in result.report.findings)


# ---- options, cache, world integration ------------------------------------------------------------


def test_unknown_options_are_rejected_with_the_list_of_valid_ones():
    with pytest.raises(ValueError, match="Import options are"):
        ImportOptions.parse({"bse": "fixed"})
    with pytest.raises(ValueError, match="base"):
        ImportOptions.parse({"base": "sideways"})


def test_unsupported_extension_is_explained(tmp_path):
    other = write(tmp_path / "robot.sdf", "<sdf/>")
    with pytest.raises(ValueError, match="imports .urdf"):
        import_file(other)


def test_results_are_cached_and_reused_until_a_file_changes(tmp_path, home):
    path = write(tmp_path / "arm.urdf", arm_urdf())
    first = import_file(path)
    entries = list((home / "cache" / "import").glob("*/meta.json"))
    assert len(entries) == 1
    again = import_file(path)
    assert again.report.to_dict() == first.report.to_dict()
    assert len(list((home / "cache" / "import").glob("*/meta.json"))) == 1
    write(path, arm_urdf().replace('effort="20"', 'effort="25"'))
    changed = import_file(path)
    assert len(list((home / "cache" / "import").glob("*/meta.json"))) == 2
    model, _ = _world(changed)
    assert model.actuator_forcerange[0][1] == pytest.approx(25.0)


def test_cache_notices_when_a_converted_mesh_changes_or_a_missing_one_appears(tmp_path):
    urdf = _mesh_arm(tmp_path, "package://pkg/meshes/upper.stl")
    missing = import_file(urdf)
    assert any(f.status == "unfixed" for f in missing.report.findings)
    box_stl(tmp_path / "pkg" / "meshes" / "upper.stl", (0.05, 0.05, 0.3))
    found = import_file(urdf)
    assert not [f for f in found.report.findings if f.status == "unfixed"]


def test_cache_is_pruned(tmp_path, home, monkeypatch):
    monkeypatch.setattr(importer, "CACHE_LIMIT", 2)
    for i in range(4):
        import_file(write(tmp_path / f"arm{i}.urdf", arm_urdf(f"arm{i}")))
    assert len(list((home / "cache" / "import").glob("*/meta.json"))) == 2


def test_import_fits_into_a_world_next_to_template_robots(tmp_path):
    arm = write(tmp_path / "arm.urdf", arm_urdf())
    world = World.model_validate(
        {
            "name": "mixed",
            "robots": [
                {"name": "mine", "file": str(arm), "pos": [0.5, 0.0]},
                {"name": "dog", "template": "quadruped", "pos": [-0.5, 0.0]},
            ],
            "objects": [{"name": "cube", "shape": "box", "pos": [0.5, 0.3]}],
        }
    )
    compiled = compile_world(world, tmp_path)
    assert set(compiled.doctor) == {"mine"}  # only imported files carry a report
    assert compiled.doctor["mine"]["fixed"] >= 1
    model = compiled.model
    assert model.body("mine/base_link").id > 0 and model.body("dog/trunk").id > 0
    assert model.nkey == 1
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(500):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


def test_relative_file_paths_in_world_yaml_resolve_against_the_world_folder(tmp_path):
    write(tmp_path / "robots" / "arm.urdf", arm_urdf())
    world = World.model_validate({"robots": [{"name": "mine", "file": "robots/arm.urdf"}]})
    assert compile_world(world, tmp_path).model.nu == 2
    with pytest.raises(ValueError, match="file not found"):
        compile_world(world, tmp_path / "elsewhere")


def test_bad_import_options_in_a_world_name_the_robot(tmp_path):
    arm = write(tmp_path / "arm.urdf", arm_urdf())
    world = World.model_validate({"robots": [{"name": "mine", "file": str(arm), "params": {"nope": 1}}]})
    with pytest.raises(ValueError, match="mine"):
        compile_world(world, tmp_path)


def test_compile_standalone_makes_a_one_robot_world_with_a_floor(tmp_path):
    compiled = compile_standalone(write(tmp_path / "07-my arm.urdf", arm_urdf()))
    model = compiled.model
    assert model.nu == 2
    assert any(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE for g in range(model.ngeom))
    assert model.body("robot_07_my_arm/base_link").id > 0


def test_doctor_report_is_json_serializable(tmp_path):
    result = import_file(write(tmp_path / "arm.urdf", arm_urdf()))
    data = json.loads(json.dumps(result.report.to_dict()))
    assert data["findings"] and data["summary"]


# ---- more real-world quirks ------------------------------------------------------------------


def test_friction_hints_from_gazebo_and_pybullet_tags_are_applied(tmp_path):
    body = (
        box_link("base_link")
        + box_link("foot_link").replace("</link>", '<contact><lateral_friction value="0.3"/></contact></link>')
        + box_link("pad")
        + revolute("j1", "base_link", "foot_link")
        + revolute("j2", "foot_link", "pad")
        + '<gazebo reference="pad"><mu1>0.7</mu1><mu2>0.7</mu2></gazebo>'
    )
    result = import_file(write(tmp_path / "r.urdf", robot(body)))
    model, _ = _world(result)
    friction = {
        n: model.geom_friction[g][0] for g in range(model.ngeom) for n in [model.body(model.geom_bodyid[g]).name]
    }
    assert friction["r/foot_link"] == pytest.approx(0.3) and friction["r/pad"] == pytest.approx(0.7)


def test_mesh_formats_robotforge_cannot_read_are_named_in_the_report(tmp_path):
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.blend"))
    bad = next(f for f in result.report.findings if f.status == "unfixed" and f.topic == "meshes")
    assert "not a mesh format robotforge reads" in " ".join(bad.items)


def test_a_utf8_bom_and_odd_numbers_do_not_break_the_import(tmp_path):
    text = "\ufeff" + arm_urdf().replace('xyz="0 0 0.1"', 'xyz="0 0 nan"', 1)
    path = tmp_path / "bom.urdf"
    path.write_text(text, encoding="utf-8")
    model, _ = _world(import_file(path))
    assert np.isfinite(model.body_pos).all()


def test_single_link_urdf_is_an_object_that_falls_to_the_floor_and_stays(tmp_path):
    result = import_file(write(tmp_path / "block.urdf", robot(box_link("block", (0.2, 0.2, 0.2)))))
    assert result.floating and result.home_ctrl == {}
    model, data = _world(result)
    _run(model, data, 1.0)
    assert data.xpos[model.body("r/block").id][2] == pytest.approx(0.1, abs=0.01)


def test_flipped_wheel_axes_still_drive_forward(tmp_path):
    from robotforge.behaviors import find_wheels
    from robotforge.behaviors.drive import DiffDrive

    flipped = rover_urdf().replace(
        '<origin xyz="0 -0.13 0"/><axis xyz="0 1 0"/>', '<origin xyz="0 -0.13 0"/><axis xyz="0 -1 0"/>'
    )
    model, data = _world(import_file(write(tmp_path / "r.urdf", flipped)), "r")
    wheels = find_wheels(model, "r")
    assert wheels is not None and wheels.left_signs == (1.0,) and wheels.right_signs == (-1.0,)
    assert wheels.radius == pytest.approx(0.05, abs=0.002) and wheels.track == pytest.approx(0.26, abs=0.01)
    drive = DiffDrive(model, "r")
    drive.command(0.2, 0.0)
    start = data.xpos[model.body("r/chassis").id].copy()
    for _ in range(1500):
        drive.update(model, data)
        mujoco.mj_step(model, data)
    assert data.xpos[model.body("r/chassis").id][0] - start[0] > 0.2


def test_undefined_xacro_macros_are_skipped_and_reported(tmp_path):
    text = (
        '<robot name="r" xmlns:xacro="http://ros.org/wiki/xacro">'
        '<link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>'
        '</inertial><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>'
        "<xacro:make_pelvis/></robot>"
    )
    result = import_file(write(tmp_path / "r.urdf", text))
    skipped = next(f for f in result.report.findings if f.topic == "xacro" and f.status == "unfixed")
    assert skipped.items == ("make_pelvis",)
    _world(result)


def test_meshes_listed_in_one_format_are_found_in_another(tmp_path):
    box_stl(tmp_path / "pkg" / "meshes" / "upper.stl", (0.05, 0.05, 0.3))
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.obj"))  # only the .stl exists
    guess = next(f for f in result.report.findings if f.topic == "meshes" and f.status == "fixed")
    assert "upper.stl" in " ".join(guess.items)
    assert not [f for f in result.report.findings if f.status == "unfixed"]


def test_empty_mesh_files_are_noted_not_failed(tmp_path):
    (tmp_path / "pkg" / "meshes").mkdir(parents=True)
    (tmp_path / "pkg" / "meshes" / "upper.stl").write_bytes(b"\x00" * 80 + b"\x00\x00\x00\x00")  # header, 0 triangles
    result = import_file(_mesh_arm(tmp_path, "package://pkg/meshes/upper.stl"))
    note = next(f for f in result.report.findings if f.topic == "meshes")
    assert note.status == "info" and "empty" in note.problem
    _world(result)


def test_axis_and_limit_leftovers_on_fixed_joints_do_not_break_loading(tmp_path):
    body = box_link("base_link") + box_link("plate")
    body += (
        '<joint name="mount" type="fixed"><parent link="base_link"/><child link="plate"/>'
        '<axis/><limit effort="1"/><dynamics damping="0.1"/></joint>'
    )
    _world(import_file(write(tmp_path / "r.urdf", robot(body))))


def test_a_robot_on_two_wheels_without_a_caster_is_reported_as_tipping(tmp_path):
    result = import_file(write(tmp_path / "r.urdf", rover_urdf(caster=False)))
    verdict = [f for f in result.report.findings if f.topic == "stability"]
    assert verdict  # either it balances on its body or the report says what is missing
    if verdict[0].status == "unfixed":
        assert "caster" in verdict[0].action


def test_deleted_cache_meshes_trigger_a_rebuild_and_the_mesh_cache_is_bounded(tmp_path, home, monkeypatch):
    write(tmp_path / "pkg" / "meshes" / "upper.dae", dae_text(size=(0.05, 0.05, 0.3)))
    urdf = _mesh_arm(tmp_path, "package://pkg/meshes/upper.dae")
    first = import_file(urdf)
    converted = list((home / "cache" / "import" / "meshes").glob("*.stl"))
    assert converted
    for f in converted:
        f.unlink()  # as if the mesh cache had been pruned
    again = import_file(urdf)  # must not hand MuJoCo a path that no longer exists
    _world(again)
    assert again.report.to_dict()["fixed"] == first.report.to_dict()["fixed"]
    monkeypatch.setattr(importer, "MESH_CACHE_BYTES", 1)
    importer._prune(home / "cache" / "import")
    assert not list((home / "cache" / "import" / "meshes").glob("*"))


def test_an_unwritable_cache_gives_an_actionable_error(tmp_path, monkeypatch):
    blocker = write(tmp_path / "not_a_folder", "x")
    monkeypatch.setenv("ROBOTFORGE_HOME", str(blocker))
    with pytest.raises(ValueError, match="ROBOTFORGE_HOME"):
        import_file(write(tmp_path / "arm.urdf", arm_urdf()))


def test_converted_meshes_do_not_make_a_cached_import_look_stale(tmp_path, monkeypatch):
    write(tmp_path / "pkg" / "meshes" / "upper.dae", dae_text(size=(0.05, 0.05, 0.3)))
    urdf = _mesh_arm(tmp_path, "package://pkg/meshes/upper.dae")
    import_file(urdf)

    def refuse(*args, **kwargs):
        raise AssertionError("the cached import was thrown away and redone")

    monkeypatch.setattr(importer, "_stage_and_plan", refuse)
    import_file(urdf)  # served from the cache, even though loading it touched the converted mesh files
    import_file(urdf)


# ---- time budgets and cache names ----------------------------------------------------------------


def _staged_walker(tmp_path):
    import xml.etree.ElementTree as ET

    from robotforge.design.importing import rig
    from robotforge.design.importing.report import DoctorReport
    from robotforge.design.importing.resolve import MeshResolver
    from robotforge.design.importing.urdf import stage_urdf

    path = write(tmp_path / "walker.urdf", _walker_urdf())
    staged = stage_urdf(
        ET.fromstring(path.read_text()), MeshResolver(path), tmp_path / "cache", DoctorReport(""), False
    )
    urdf = write(tmp_path / "staged.urdf", staged.text)
    return staged, urdf, rig.home_pose(staged)


def test_stance_search_respects_its_deadline(tmp_path):
    import time

    from robotforge.design.importing.stance import find_stances

    staged, urdf, home = _staged_walker(tmp_path)
    expired = find_stances(staged, urdf, home, deadline=time.monotonic() - 1.0)
    assert expired.stopped_early and expired.poses == []
    start = time.process_time()
    full = find_stances(staged, urdf, home)
    full_seconds = time.process_time() - start
    assert not full.stopped_early and full.poses  # unbounded, the walker has a stance

    start = time.process_time()
    short = find_stances(staged, urdf, home, deadline=time.monotonic() + 0.05)
    short_seconds = time.process_time() - start
    assert short.stopped_early
    assert short_seconds < full_seconds / 2, (
        f"{short_seconds:.2f} s with a deadline, {full_seconds:.2f} s without"
    )  # unbounded, the walker has a stance


def test_a_cut_short_stance_search_is_reported(tmp_path, monkeypatch):
    from robotforge.design.importing import rig
    from robotforge.design.importing.stance import StanceResult

    monkeypatch.setattr(rig, "find_stances", lambda *a, **k: StanceResult([], stopped_early=True))
    result = import_file(write(tmp_path / "one.urdf", _walker_urdf(legs=1)), {"base": "floating"})
    note = next(f for f in result.report.findings if f.topic == "stance")
    assert "stopped early" in note.problem and note.status == "info"


def test_cache_file_names_never_contain_characters_from_the_mesh_name(tmp_path, home):
    from robotforge.design.importing import collision, meshio

    a = trimesh.creation.icosphere(subdivisions=3, radius=0.08)
    b = trimesh.creation.icosphere(subdivisions=3, radius=0.08)
    b.apply_translation([0.3, 0, 0])
    dumbbell = trimesh.util.concatenate([a, b])
    out = home / "meshes"
    names = ("my robot; rm -rf $HOME`x`.stl", "..stl", "...stl", "\u00e9l\u00e9ment\n.stl", "a b'c\".stl")
    if sys.platform == "win32":  # Windows cannot even create files named with a newline or a double quote
        names = tuple(n for n in names if "\n" not in n and '"' not in n) + ("\u00e9l\u00e9ment.stl",)
    for name in names:
        src = tmp_path / name
        dumbbell.export(src)
        collision.decompose(meshio.prepare_mesh(src, out), out)
    written = sorted(out.glob("*"))
    assert written
    for path in written:
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", path.name) and not path.name.startswith("."), path.name
    outside = {p.name for p in tmp_path.iterdir()} - {"home", *names}
    assert not outside  # nothing was written next to the sources or anywhere else


def test_a_visual_with_several_materials_keeps_the_first(tmp_path):
    """Unitree's B2 exporter writes one material per mesh colour into the same visual, which MuJoCo refuses."""
    two = '<material name="a"><color rgba="1 0 0 1"/></material><material name="b"><color rgba="0 0 1 1"/></material>'
    body = box_link("base", mass=2.0).replace("</visual>", f"{two}</visual>")
    result = import_file(
        write(tmp_path / "bot.urdf", robot(body + box_link("arm", mass=1.0) + revolute("j", "base", "arm")))
    )
    notes = json.dumps(result.report.to_dict()) if hasattr(result.report, "to_dict") else str(result.report)
    assert "several materials" in notes
    model, _ = _world(result)
    base_visual = next(g for g in range(model.ngeom) if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0)
    assert tuple(model.geom_rgba[base_visual]) == pytest.approx((1, 0, 0, 1))
