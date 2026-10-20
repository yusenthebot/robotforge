"""A robot imported from a file is recognised for what it is: an arm gets its tool point like a catalogue arm."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotforge.playground import Playground
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"

LINK = (
    '<link name="{n}"><visual><origin xyz="0 0 {h}"/><geometry><box size="0.05 0.05 {l}"/></geometry></visual>'
    '<collision><origin xyz="0 0 {h}"/><geometry><box size="0.05 0.05 {l}"/></geometry></collision>'
    '<inertial><origin xyz="0 0 {h}"/><mass value="1"/>'
    '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>'
)
JOINT = (
    '<joint name="{n}" type="revolute"><parent link="{p}"/><child link="{c}"/><origin xyz="0 0 {z}"/>'
    '<axis xyz="{a}"/><limit lower="-2" upper="2" effort="20" velocity="2"/></joint>'
)
ARM_URDF = (
    '<?xml version="1.0"?><robot name="myarm">'
    + LINK.format(n="base", h=0.05, l=0.1)
    + LINK.format(n="l1", h=0.15, l=0.3)
    + LINK.format(n="l2", h=0.15, l=0.3)
    + LINK.format(n="l3", h=0.1, l=0.2)
    + JOINT.format(n="j1", p="base", c="l1", z=0.05, a="0 0 1")
    + JOINT.format(n="j2", p="l1", c="l2", z=0.3, a="0 1 0")
    + JOINT.format(n="j3", p="l2", c="l3", z=0.3, a="0 1 0")
    + "</robot>"
)


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab")
    return playground


def _kind(pg: Playground, name: str) -> str:
    return next(r["kind"] for r in pg.behaviors.robots() if r["name"] == name)


@pytest.mark.parametrize("source", ["urdf", "mjcf"])
def test_an_arm_imported_from_a_file_is_an_arm_that_move_arm_steers(pg, tmp_path, source):
    """Regression: a Menagerie UR5e MJCF and a handwritten URDF arm were kind 'other' and move_arm said 'robot of no
    known kind, not an arm', where the same UR5e from the library worked."""
    if source == "urdf":
        path = tmp_path / "myarm.urdf"
        path.write_text(ARM_URDF, encoding="utf-8")
    else:
        path = FIXTURES / "universal_robots_ur5e" / "ur5e.xml"
    result = pg.add_robot("a", file=str(path))
    assert _kind(pg, "a") == "arm"
    assert any("found its arm" in n for n in result["notes"])
    motion = pg.behaviors.move_arm("a", [0.3, 0.0, 0.4])
    assert motion is not None


WHEEL_LINK = (
    '<link name="{n}"><visual><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.08" length="0.04"/></geometry>'
    '</visual><collision><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.08" length="0.04"/></geometry>'
    '</collision><inertial><mass value="0.5"/>'
    '<inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>'
)
DRIVE_JOINT = (
    '<joint name="{n}" type="continuous"><parent link="chassis"/><child link="{c}"/><origin xyz="0 {y} -0.02"/>'
    '<axis xyz="0 1 0"/></joint>'
)
ROVER_URDF = (
    '<?xml version="1.0"?><robot name="rover"><link name="chassis"><visual><geometry><box size="0.4 0.25 0.1"/>'
    '</geometry></visual><collision><geometry><box size="0.4 0.25 0.1"/></geometry></collision>'
    '<inertial><mass value="4"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>'
    + WHEEL_LINK.format(n="left_link")
    + WHEEL_LINK.format(n="right_link")
    + DRIVE_JOINT.format(n="left_motor", c="left_link", y=0.15)
    + DRIVE_JOINT.format(n="right_motor", c="right_link", y=-0.15)
    + "</robot>"
)


def test_two_continuous_joints_that_turn_nothing_else_are_drive_wheels_whatever_they_are_called(pg, tmp_path):
    """Regression: a URDF rover whose wheel joints were not called 'wheel' was a 'humanoid' with position servos on
    its wheels, and move refused to walk it."""
    path = tmp_path / "rover.urdf"
    path.write_text(ROVER_URDF, encoding="utf-8")
    pg.add_robot("r", file=str(path), pos=[0, 0, 0.12])
    assert _kind(pg, "r") in ("rover", "balancer")
    model = pg.session.model
    for name in ("r/left_motor", "r/right_motor"):
        act = model.actuator(name)
        assert act.biasprm[1] == 0.0 and act.biasprm[2] < 0  # a speed servo, not a position servo
    pg.behaviors.move("r", 0.3, 0.0)
    pg.run_fast(lambda: False, 4.0)
    assert pg.behaviors.pose("r")["position"][0] > 0.3


def test_an_arm_with_continuous_joints_is_not_taken_for_wheels(pg, tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text(ARM_URDF.replace('name="j1" type="revolute"', 'name="j1" type="continuous"'), encoding="utf-8")
    pg.add_robot("a", file=str(path))
    assert _kind(pg, "a") == "arm"
