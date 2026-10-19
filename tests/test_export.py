"""Exporting a world or a robot for other simulators: MJCF and URDF with meshes, loaded back and compared."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.design import export
from robotforge.design.world import World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sensors import declare
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def anyio_backend():
    return "asyncio"


def compiled(tmp_path: Path, robots: list[dict], objects: list[dict] | None = None, **world):
    return compile_world(World.model_validate({"robots": robots, "objects": objects or [], **world}), tmp_path)


def test_a_world_exports_to_mjcf_that_loads_back_the_same(tmp_path):
    c = compiled(
        tmp_path,
        [
            {"name": "dog", "template": "quadruped", "sensors": ["imu", {"type": "lidar", "name": "scan"}]},
            {"name": "go2", "library": "go2", "pos": [2, 0]},
            {"name": "arm", "template": "arm", "pos": [0, 2]},
        ],
        [{"name": "table", "shape": "table", "pos": [2, 2]}, {"name": "cube", "shape": "box", "pos": [2, 2, 0.5]}],
        terrain=[{"type": "stairs", "pos": [-2, 0]}],
    )
    out = export.export_mjcf(c.spec, c.model, tmp_path / "out", "world")
    check = export.check_mjcf(out.main, c.model)
    assert check["same"], check
    assert check["vertex_error_m"] < 1e-5 and check["body_position_error_m"] < 1e-5
    meshes = sorted((tmp_path / "out" / "meshes").glob("*.obj"))
    assert len(meshes) == c.model.nmesh > 20
    text = out.main.read_text(encoding="utf-8")
    assert 'meshdir="meshes/"' in text and "<mesh " in text and "vertex=" not in text  # files, not inline numbers
    assert (tmp_path / "out" / "meshes" / meshes[0].name).read_text(encoding="utf-8").startswith("v ")


def test_the_exported_world_is_the_world_to_the_last_vertex_and_mass(tmp_path):
    c = compiled(tmp_path, [{"name": "go2", "library": "go2"}, {"name": "dog", "template": "quadruped", "pos": [1, 0]}])
    out = export.export_mjcf(c.spec, c.model, tmp_path / "out", "w")
    again = mujoco.MjModel.from_xml_path(str(out.main))
    assert again.nbody == c.model.nbody and again.njnt == c.model.njnt and again.nu == c.model.nu
    assert again.body_mass.sum() == pytest.approx(c.model.body_mass.sum(), rel=1e-9)
    assert np.allclose(c.model.jnt_range, again.jnt_range)
    assert [mujoco.mj_id2name(c.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(c.model.njnt)] == [
        mujoco.mj_id2name(again, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(again.njnt)
    ]


@pytest.mark.parametrize("what", ["mass", "vertices", "joint_ranges"])
def test_the_check_fails_when_the_export_differs_from_the_world(tmp_path, what):
    """'same' used to look only at names and positions: a world twice as heavy, with its meshes a centimetre off or
    its joints three times wider than the file said still came back same=True."""
    robots = [{"name": "go2", "library": "go2"}]
    table = [{"name": "table", "shape": "table"}]  # the meshes
    c = compiled(tmp_path, robots, table)
    out = export.export_mjcf(c.spec, c.model, tmp_path / "out", "w")
    assert export.check_mjcf(out.main, c.model)["same"]
    other = compiled(tmp_path / "other", robots, table).model
    if what == "mass":
        other.body_mass[:] *= 2
    elif what == "vertices":
        other.mesh_vert[:] += 0.01
    else:
        other.jnt_range[:] *= 3
    check = export.check_mjcf(out.main, other)
    assert check["loads"] and not check["same"], check


def test_a_sensor_is_still_declared_in_the_export(tmp_path):
    c = compiled(tmp_path, [{"name": "dog", "template": "quadruped", "sensors": [{"type": "lidar", "name": "scan"}]}])
    out = export.export_mjcf(c.spec, c.model, tmp_path / "out", "w")
    again = mujoco.MjModel.from_xml_path(str(out.main))
    assert [d.name for d in declare.read(again)] == [d.name for d in declare.read(c.model)] != []


def test_exporting_does_not_change_the_world_it_came_from(tmp_path):
    c = compiled(tmp_path, [{"name": "go2", "library": "go2"}])
    before = c.spec.to_xml()
    export.export_mjcf(c.spec, c.model, tmp_path / "out", "w")
    assert c.spec.to_xml() == before and len(c.spec.meshes) == c.model.nmesh


@pytest.mark.parametrize(
    "robot",
    [
        {"name": "dog", "template": "quadruped"},
        {"name": "arm", "template": "arm"},
        {"name": "go2", "library": "go2"},
        {"name": "panda", "library": "panda"},
        {"name": "kid", "template": "humanoid"},
    ],
)
def test_a_robot_exports_to_urdf_that_loads_back_with_the_same_joints(tmp_path, robot):
    c = compiled(tmp_path, [robot])
    out = export.export_urdf(c.model, robot["name"], tmp_path / "out")
    check = export.check_urdf(out.main, c.model, robot["name"])
    assert check["same"], check
    root = ET.parse(out.main).getroot()
    assert root.tag == "robot" and root.get("name") == robot["name"]
    moving = [j for j in root.iter("joint") if j.get("type") != "fixed"]
    assert len(moving) == check["joints"]["count"]
    for link in root.iter("link"):
        assert link.find("inertial") is not None  # every link has mass: a sim loads it without guessing
    for mesh in root.iter("mesh"):
        assert (tmp_path / "out" / mesh.get("filename")).is_file()


def test_what_a_urdf_cannot_say_is_listed(tmp_path):
    c = compiled(tmp_path, [{"name": "kid", "template": "humanoid"}, {"name": "panda", "library": "panda"}])
    kid = export.export_urdf(c.model, "kid", tmp_path / "kid")
    assert any("capsules" in n for n in kid.notes) and any("floats" in n for n in kid.notes)
    panda = export.export_urdf(c.model, "panda", tmp_path / "panda")
    assert any("equality constraints" in n for n in panda.notes)


def test_a_joint_limit_and_a_slide_survive_the_trip(tmp_path):
    c = compiled(tmp_path, [{"name": "arm", "template": "arm"}])
    out = export.export_urdf(c.model, "arm", tmp_path / "out")
    again = mujoco.MjModel.from_xml_path(str(out.main))
    for j in range(c.model.njnt):
        name = mujoco.mj_id2name(c.model, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix("arm/")
        k = mujoco.mj_name2id(again, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert k >= 0 and again.jnt_type[k] == c.model.jnt_type[j]
        if c.model.jnt_limited[j]:
            assert np.allclose(again.jnt_range[k], c.model.jnt_range[j], atol=1e-6)
        assert np.allclose(again.jnt_axis[k], c.model.jnt_axis[j], atol=1e-6)


def test_a_robot_that_is_not_there_is_refused(tmp_path):
    c = compiled(tmp_path, [{"name": "dog", "template": "quadruped"}])
    with pytest.raises(export.ExportError, match="no robot named 'cat'"):
        export.export_urdf(c.model, "cat", tmp_path / "out")


@pytest.mark.anyio
async def test_an_agent_exports_a_world_and_a_robot(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("go2", library="go2")
    pg.add_robot("arm", template="arm", pos=[1, 1])
    pg.add_object("cube", shape="box", pos=[0.5, 0.5])
    server = build_server(pg, find_free_port(8851))
    async with Client(server) as client:

        async def call(args: dict) -> dict:
            result = await client.call_tool("export_model", args)
            assert not result.is_error, result.content[0].text
            return json.loads(result.content[0].text)

        whole = await call({})
        assert whole["check"]["same"] and whole["format"] == "mjcf" and whole["meshes"] > 5
        assert Path(whole["file"]).is_file() and "mujoco.viewer" in whole["open_with"]
        urdf = await call({"robot": "go2", "format": "urdf"})
        assert urdf["check"]["same"] and urdf["format"] == "urdf"
        one = await call({"robot": "arm"})
        assert one["check"]["same"] and "arm" in Path(one["file"]).name
        again = await call({"robot": "arm"})
        assert again["files"] == one["files"]  # the same folder written again, nothing left over
        missing = await client.call_tool("export_model", {"format": "urdf"})
        assert missing.is_error and "give robot=NAME" in missing.content[0].text
        unknown = await client.call_tool("export_model", {"robot": "cat"})
        assert unknown.is_error and "no robot named 'cat'" in unknown.content[0].text


def test_a_height_field_and_water_come_through(tmp_path):
    c = compiled(
        tmp_path, [{"name": "dog", "template": "quadruped"}], terrain=[{"type": "hills"}], water={"level": 0.3}
    )
    assert c.model.nhfield == 1
    out = export.export_mjcf(c.spec, c.model, tmp_path / "out", "w")
    check = export.check_mjcf(out.main, c.model)
    assert check["same"] and check["height_field_error_m"] < 1e-6


def test_the_command_line_exports_without_an_agent(tmp_path, monkeypatch, capsys):
    from robotforge import cli

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("lab", "checker")
    pg.add_robot("arm", template="arm")
    assert cli.main(["export", "lab", "--robot", "arm", "--format", "urdf"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["check"]["same"] and Path(out["file"]).name == "arm.urdf" and Path(out["file"]).is_file()
    assert cli.main(["export", "lab", "--format", "urdf"]) == 2
    assert "give robot=NAME" in capsys.readouterr().err
    assert cli.main(["export", "lab", "--robot", "cat"]) == 2
    assert "no robot named 'cat'; its robots: arm" in capsys.readouterr().err
    assert cli.main(["export", "lab", "--to", str(tmp_path / "elsewhere")]) == 0
    assert (tmp_path / "elsewhere" / "lab.xml").is_file()


def test_the_command_line_lists_plugins_and_fails_when_one_did_not_load(tmp_path, monkeypatch, capsys):
    from robotforge import cli
    from robotforge.extend import REGISTRY, registry

    folder = tmp_path / "plugins"
    folder.mkdir()
    monkeypatch.setenv("ROBOTFORGE_PLUGINS", str(folder))
    with registry.isolated():
        REGISTRY.user_loaded = False
        assert cli.main(["plugins"]) == 0
        assert json.loads(capsys.readouterr().out)["folders"] == [str(folder)]
        (folder / "broken.py").write_text("x = (\n", encoding="utf-8")
        REGISTRY.reload()
        assert cli.main(["plugins"]) == 1
        assert "SyntaxError" in capsys.readouterr().out
