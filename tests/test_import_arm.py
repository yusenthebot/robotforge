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
