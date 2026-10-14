"""The doctor as agents and users meet it: open_model, add_robot, the MCP tools and the CLI."""

from __future__ import annotations

import asyncio
import json
import time

import mujoco
import numpy as np
import pytest
from importer_fixtures import arm_urdf, box_stl, rover_urdf, write
from mcp import Client

from robotforge import cli
from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def _text(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")


def _topics(doctor: dict, status: str) -> set[str]:
    return {f["topic"] for f in doctor["findings"] if f["status"] == status}


def test_open_model_repairs_a_urdf_and_reports_what_it_did(tmp_path):
    pg = Playground.with_demo()
    result = pg.open_model(str(write(tmp_path / "arm.urdf", arm_urdf())))
    assert [a["name"] for a in result["actuators"]] == ["arm/shoulder", "arm/elbow"]
    doctor = result["doctor"]
    assert "actuators" in _topics(doctor, "fixed") and doctor["summary"]
    # The loaded model is live: control it like any other.
    pg.set_controls({"arm/shoulder": 0.8})
    pg.simulate(1.5)
    assert pg.state()["joints"]["arm/shoulder"] == pytest.approx(0.8, abs=0.05)
    assert np.isfinite(pg.session.data.qpos).all()


def test_open_model_still_opens_mjcf_without_a_doctor():
    pg = Playground.with_demo()
    result = pg.open_model(str(cli.DEMO_MODEL))
    assert "doctor" not in result


def test_open_model_accepts_meshes_and_import_options(tmp_path):
    pg = Playground.with_demo()
    result = pg.open_model(str(box_stl(tmp_path / "block.stl", (0.1, 0.1, 0.1))), {"base": "floating", "mass": 1.0})
    assert result["free_bodies"] == ["block/object"]
    assert result["doctor"]["stats"]["triangles"] == 12


def test_open_model_errors_say_what_to_do(tmp_path):
    pg = Playground.with_demo()
    with pytest.raises(PlaygroundError, match="file not found"):
        pg.open_model(str(tmp_path / "nope.urdf"))
    with pytest.raises(PlaygroundError, match="expected an MJCF"):
        pg.open_model(str(write(tmp_path / "x.sdf", "<sdf/>")))
    with pytest.raises(PlaygroundError, match="Import options are"):
        pg.open_model(str(write(tmp_path / "arm.urdf", arm_urdf())), {"basee": "fixed"})
    with pytest.raises(PlaygroundError, match="closed loop"):
        loop = (
            '<robot name="r"><link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="j1" type="fixed"><parent link="a"/><child link="c"/></joint>'
            '<joint name="j2" type="fixed"><parent link="b"/><child link="c"/></joint></robot>'
        )
        pg.open_model(str(write(tmp_path / "loop.urdf", loop)))


def test_add_robot_from_a_file_returns_the_doctor_report_and_saves_options(tmp_path):
    pg = Playground.with_demo()
    pg.new_world("import test")
    urdf = write(tmp_path / "rover.urdf", rover_urdf())
    result = pg.add_robot("rover", file=str(urdf), params={"base": "floating"}, pos=[0.5, 0.0])
    assert result["robots"] == ["rover"]
    assert "actuators" in _topics(result["doctor"], "fixed")
    assert {a["name"] for a in result["actuators"]} == {"rover/left_wheel_joint", "rover/right_wheel_joint"}
    assert any("imported rover.urdf" in n for n in result["notes"])
    assert "base: floating" in pg.show_world()["world_yaml"]  # options live in world.yaml

    # Editing something else does not repeat the report, but the robot stays imported.
    other = pg.add_object("cube", shape="box")
    assert "doctor" not in other and "rover/left_wheel_joint" in {a["name"] for a in other["actuators"]}
    assert pg.undo()["objects"] == []


def test_add_robot_with_a_bad_file_does_not_change_the_world(tmp_path):
    pg = Playground.with_demo()
    pg.new_world("import fail")
    with pytest.raises(ValueError, match="file not found"):
        pg.add_robot("ghost", file=str(tmp_path / "ghost.urdf"))
    assert pg.show_world()["world_yaml"].count("ghost") == 0


def test_imported_robot_and_template_robot_share_a_world(tmp_path):
    pg = Playground.with_demo()
    pg.new_world("mix")
    pg.add_robot("dog", template="quadruped", pos=[-0.6, 0.0])
    result = pg.add_robot("arm", file=str(write(tmp_path / "arm.urdf", arm_urdf())), pos=[0.4, 0.0])
    names = {a["name"] for a in result["actuators"]}
    assert "dog/hip_fl" in names or any(n.startswith("dog/") for n in names)
    assert {"arm/shoulder", "arm/elbow"} <= names
    pg.simulate(1.0)
    assert np.isfinite(pg.session.data.qpos).all()


async def test_mcp_tools_return_the_doctor_field(tmp_path):
    urdf = write(tmp_path / "arm.urdf", arm_urdf())
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        opened = json.loads(_text(await client.call_tool("open_model", {"path": str(urdf)})))
        assert "actuators" in _topics(opened["doctor"], "fixed")

        await client.call_tool("new_world", {"name": "mcp import"})
        added = json.loads(
            _text(await client.call_tool("add_robot", {"name": "mine", "file": str(urdf), "params": {"base": "fixed"}}))
        )
        assert added["doctor"]["fixed"] >= 1 and added["robots"] == ["mine"]

        bad = await client.call_tool("add_robot", {"name": "x", "file": str(urdf), "params": {"nope": True}})
        assert bad.is_error and "Import options are" in _text(bad)

        meshed = await client.call_tool(
            "open_model", {"path": str(box_stl(tmp_path / "b.stl")), "options": {"scale": 2}}
        )
        assert not meshed.is_error and json.loads(_text(meshed))["doctor"]["stats"]["size_m"] == [0.2, 0.2, 0.2]


async def test_a_slow_import_does_not_freeze_the_event_loop(tmp_path, monkeypatch):
    """add_robot is an async tool; a cold import must run in a worker thread so the viewer keeps serving."""
    pg = Playground.with_demo()
    real = Playground.add_robot

    def slow_add_robot(self, *args, **kwargs):
        time.sleep(0.8)  # stands in for mesh conversion and the planning simulations
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Playground, "add_robot", slow_add_robot)
    urdf = write(tmp_path / "arm.urdf", arm_urdf())
    gaps: list[float] = []

    async def heartbeat() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    server = build_server(pg, find_free_port(8790))
    async with Client(server) as client:
        await client.call_tool("new_world", {"name": "slow import"})
        beat = asyncio.create_task(heartbeat())
        try:
            result = await client.call_tool("add_robot", {"name": "mine", "file": str(urdf)})
        finally:
            beat.cancel()
    assert not result.is_error and json.loads(_text(result))["robots"] == ["mine"]
    stall = max(gaps, default=float("inf"))
    assert len(gaps) > 10 and stall < 0.3, (
        f"the event loop was blocked during the import ({len(gaps)} beats, worst gap {stall:.2f} s)"
    )


async def test_two_imports_at_once_both_end_up_in_the_world(tmp_path, monkeypatch):
    """Tool calls run in worker threads; concurrent edits must queue instead of overwriting each other."""
    from robotforge.design import importer

    real = importer.import_file

    def slow_import(*args, **kwargs):
        time.sleep(0.3)
        return real(*args, **kwargs)

    monkeypatch.setattr(importer, "import_file", slow_import)
    urdf = write(tmp_path / "arm.urdf", arm_urdf())
    server = build_server(Playground.with_demo(), find_free_port(8790))
    async with Client(server) as client:
        await client.call_tool("new_world", {"name": "two at once"})
        results = await asyncio.gather(
            client.call_tool("add_robot", {"name": "left", "file": str(urdf), "pos": [-0.5, 0]}),
            client.call_tool("add_robot", {"name": "right", "file": str(urdf), "pos": [0.5, 0]}),
        )
        assert not any(r.is_error for r in results)
        final = json.loads(_text(await client.call_tool("robots", {})))
    assert sorted(r["name"] for r in final["robots"]) == ["left", "right"]


def test_cli_prints_the_doctor_report_for_imports(tmp_path, monkeypatch, capsys):
    urdf = write(tmp_path / "arm.urdf", arm_urdf())
    served = {}

    async def fake_serve(playground, host, port, open_browser):
        served["model"] = playground.session.model

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["open", str(urdf), "--no-browser", "--base", "fixed"]) == 0
    out = capsys.readouterr().out
    assert "Import doctor:" in out and "[fixed]" in out and "motor" in out
    assert served["model"].nu == 2
    data = mujoco.MjData(served["model"])
    mujoco.mj_resetDataKeyframe(served["model"], data, 0)


def test_cli_reports_import_failures_without_a_traceback(tmp_path, capsys):
    bad = write(tmp_path / "bad.urdf", "<robot><link name='a'></robot>")
    assert cli.main(["open", str(bad), "--no-browser"]) == 2
    assert "not valid XML" in capsys.readouterr().err


def test_imported_wheeled_robot_drives_with_the_rover_behaviors(tmp_path):
    pg = Playground.with_demo()
    pg.new_world("drive import")
    pg.add_robot("cart", file=str(write(tmp_path / "rover.urdf", rover_urdf())))
    robots = {r["name"]: r for r in pg.behaviors.robots()}
    assert robots["cart"]["kind"] == "rover" and robots["cart"]["max_speed"] > 0

    start = pg.behaviors.pose("cart")["position"]
    pg.behaviors.move("cart", 0.15, 0.0)  # the same call the keyboard sends
    pg.session.step(1500)
    end = pg.behaviors.pose("cart")
    assert end["position"][0] - start[0] > 0.3 and abs(end["position"][1] - start[1]) < 0.1
    pg.behaviors.move("cart", 0.0, 1.0)  # turn on the spot
    pg.session.step(500)
    assert abs(pg.behaviors.pose("cart")["heading_deg"]) > 10


def test_imported_arm_is_not_mistaken_for_a_rover(tmp_path):
    pg = Playground.with_demo()
    pg.new_world("arm import")
    pg.add_robot("mine", file=str(write(tmp_path / "arm.urdf", arm_urdf())))
    assert {r["name"]: r["kind"] for r in pg.behaviors.robots()}["mine"] == "other"


def test_cli_passes_import_options_through(tmp_path, monkeypatch, capsys):
    xacro = (
        '<robot name="r" xmlns:xacro="http://ros.org/wiki/xacro"><xacro:arg name="who" default="a"/>'
        '<link name="$(arg who)_base"><inertial><mass value="1"/>'
        '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
        '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link></robot>'
    )
    path = write(tmp_path / "r.xacro", xacro)
    served = {}

    async def fake_serve(playground, host, port, open_browser):
        served["model"] = playground.session.model

    monkeypatch.setattr(cli, "serve", fake_serve)
    assert cli.main(["open", str(path), "--no-browser", "--xacro-arg", "who=zed", "--respect-effort-limits"]) == 0
    assert mujoco.mj_name2id(served["model"], mujoco.mjtObj.mjOBJ_BODY, "r/zed_base") >= 0
    assert cli.main(["open", str(path), "--no-browser", "--xacro-arg", "oops"]) == 2
    assert "KEY=VALUE" in capsys.readouterr().err
