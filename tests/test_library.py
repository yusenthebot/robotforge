"""Robot library: catalog, world integration, walking library quadrupeds, MCP tools.

Most tests run offline against tests/fixtures/menagerie, small stand-ins that mirror Menagerie's
folder layout, names, kinematics and actuators (see the comment at the top of each file).
Downloading itself is tested in test_menagerie.py. The real robots are covered by opt-in tests:
set ROBOTFORGE_NETWORK_TESTS=1 to run them (the first run downloads each robot into the cache).
"""

import dataclasses
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge.behaviors import Kind, quadruped, robot_kind
from robotforge.cli import find_free_port
from robotforge.design import library, menagerie
from robotforge.design.world import RobotEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.actuators import is_position_servo
from robotforge.sim.session import SimSession

FIXTURES = Path(__file__).parent / "fixtures" / "menagerie"
NETWORK = os.environ.get("ROBOTFORGE_NETWORK_TESTS") == "1"


@pytest.fixture(autouse=True)
def offline_menagerie(monkeypatch, request):
    if not request.node.name.startswith("test_real_"):
        monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(FIXTURES))


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab", "grass")
    return playground


def _run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


# ---- catalog and validation ------------------------------------------------------------------


def test_catalog_describes_each_robot_with_its_license():
    entries = {e["name"]: e for e in library.catalog()}
    assert entries["go2"]["title"] == "Unitree Go2"
    assert entries["go2"]["kind"] == "quadruped"
    assert entries["go2"]["license"] == "BSD-3-Clause"
    assert entries["panda"]["license"] == "Apache-2.0"
    assert {e["kind"] for e in entries.values()} == {"quadruped", "arm", "humanoid"}
    assert entries["go2"]["downloaded"] is True  # the offline fixture counts as downloaded


def test_unknown_library_robot_suggests_close_names(pg):
    with pytest.raises(PlaygroundError, match="Did you mean go2"):
        pg.add_robot("dog", library="go3")


def test_a_robot_has_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one of 'template', 'library' or 'file'"):
        RobotEntry(name="dog", template="quadruped", library="go2")


def test_library_robots_take_no_params(pg):
    with pytest.raises(ValueError, match="take no params"):
        pg.add_robot("dog", library="go2", params={"leg_length": 0.3})


# ---- loading into a world --------------------------------------------------------------------


def test_go2_compiles_standing_on_the_floor_in_its_home_pose():
    compiled = compile_world(World.model_validate({"robots": [{"name": "go2", "library": "go2"}]}))
    m = compiled.model
    assert any("12 torque motors driven as position servos" in n for n in compiled.notes)
    # The file's own floor and light are gone; the world brings its own.
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "go2/their_floor") < 0
    # Its contact settings (elliptic cones, high impratio against foot slip) are adopted.
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.impratio == 100
    # Motors became servos with the same torque limits: ctrl is now a joint angle.
    knee = m.actuator("go2/FL_calf").id
    assert int(m.actuator_biastype[knee]) == mujoco.mjtBias.mjBIAS_AFFINE
    assert m.actuator_forcerange[knee][1] == pytest.approx(45.43)
    data = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, data, m.key("home").id)
    mujoco.mj_forward(m, data)
    assert data.joint("go2/FL_thigh_joint").qpos[0] == pytest.approx(0.9)
    assert data.ctrl[knee] == pytest.approx(-1.8)
    foot_bottom = data.geom("go2/FL").xpos[2] - m.geom("go2/FL").size[0]
    assert 0.0 < foot_bottom < 0.005  # just touching the floor


def test_fixed_base_arm_sits_on_the_floor_and_holds_its_home_pose():
    model = compile_world(World.model_validate({"robots": [{"name": "arm", "library": "so_arm100"}]})).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    base = model.geom(next(g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("arm/Base").id))
    assert data.geom_xpos[base.id][2] - base.size[2] == pytest.approx(0.0, abs=1e-6)  # lifted onto the floor
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert data.joint("arm/Elbow").qpos[0] == pytest.approx(0.6, abs=0.05)


# ---- walking -----------------------------------------------------------------------------------


def test_go2_stands_walks_turns_and_settles(pg):
    pg.add_robot("go2", library="go2")
    assert pg.behaviors.robots() == [
        {"name": "go2", "kind": "quadruped", "model": "Unitree Go2", "max_speed": pytest.approx(0.6, abs=0.03)}
    ]
    pg.behaviors.move("go2", 0.0, 0.0)  # gait running at zero command
    start = pg.behaviors.pose("go2")
    _run(pg, 5.0)
    standing = pg.behaviors.pose("go2")
    assert standing["upright"] and standing["position"][2] > 0.2
    assert math.dist(start["position"][:2], standing["position"][:2]) < 0.05

    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 10.0)
    walked = pg.behaviors.pose("go2")
    # A4 asks for 0.5 m; with velocity feedback the dog really makes its 0.3 m/s (about 2.9 m).
    assert walked["position"][0] - standing["position"][0] > 2.4
    assert abs(walked["position"][1] - standing["position"][1]) < 0.2
    assert walked["upright"]

    pg.behaviors.move("go2", 0.0, 0.8)
    turned, last = 0.0, walked["heading_deg"]
    for _ in range(40):
        _run(pg, 0.1)
        heading = pg.behaviors.pose("go2")["heading_deg"]
        turned += (heading - last + 180) % 360 - 180
        last = heading
    assert turned > 150  # 0.8 rad/s for 4 s, after ramping up
    assert math.dist(walked["position"][:2], pg.behaviors.pose("go2")["position"][:2]) < 0.15  # on the spot

    pg.behaviors.stop("go2")
    _run(pg, 2.0)
    a = pg.behaviors.pose("go2")
    _run(pg, 1.0)
    b = pg.behaviors.pose("go2")
    assert math.dist(a["position"], b["position"]) < 0.02 and b["upright"]


def test_torque_motor_quadruped_walks_with_pd_inside_the_gait(pg, monkeypatch):
    raw = dataclasses.replace(library.CATALOG["go2"], servo_kp=None)  # keep the model's torque motors
    monkeypatch.setitem(library.CATALOG, "go2", raw)
    pg.add_robot("go2", library="go2")
    m = pg.session.model
    assert int(m.actuator_biastype[m.actuator("go2/FL_calf").id]) == mujoco.mjtBias.mjBIAS_NONE
    pg.behaviors.move("go2", 0.3, 0.0)
    gait = pg.session.controllers["go2"]
    assert gait.motors is not None and gait.rate_hz == pytest.approx(1 / m.opt.timestep)
    _run(pg, 10.0)
    pose = pg.behaviors.pose("go2")
    assert pose["position"][0] > 2.0 and pose["upright"]
    assert np.all(np.abs(pg.session.data.ctrl) <= np.abs(m.actuator_ctrlrange).max(axis=1) + 1e-9)


def test_legs_are_found_from_the_kinematic_tree(pg):
    pg.add_robot("go2", library="go2")
    pg.add_robot("dog", template="quadruped", pos=[0, 1])
    m = pg.session.model
    root, legs = quadruped.discover_legs(m, "go2", pg.behaviors.home_qpos())
    assert m.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    names = {name: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints] for name, joints, *_ in legs}
    assert names["front_left"] == ["go2/FL_hip_joint", "go2/FL_thigh_joint", "go2/FL_calf_joint"]
    assert names["rear_right"] == ["go2/RR_hip_joint", "go2/RR_thigh_joint", "go2/RR_calf_joint"]
    # Our own template is recognized by its shape too, with the same leg names.
    _, dog_legs = quadruped.discover_legs(m, "dog", pg.behaviors.home_qpos())
    assert {leg[0] for leg in dog_legs} == set(quadruped.TEMPLATE_LEGS)


def test_arms_are_not_mistaken_for_quadrupeds(pg):
    pg.add_robot("arm", library="so_arm100")
    assert robot_kind(pg.session.model, "arm") == Kind.OTHER
    with pytest.raises(ValueError, match="only quadrupeds and rovers"):
        pg.behaviors.move("arm", 0.2, 0.0)


@pytest.mark.anyio
async def test_mcp_tools_list_the_library_and_add_a_go2(pg):
    server = build_server(pg, find_free_port(8791))
    async with Client(server) as client:

        async def call(tool: str, args: dict) -> dict:
            return json.loads((await client.call_tool(tool, args)).content[0].text)

        assert "go2" in {r["name"] for r in (await call("list_library", {}))["robots"]}
        assert (await call("add_robot", {"name": "go2", "library": "go2"}))["robots"] == ["go2"]
        moved = await call("move", {"robot": "go2", "speed": 0.3, "seconds": 6})
        assert moved["distance_m"] > 1.2 and moved["to"]["upright"]
        assert (await call("robots", {}))["robots"][0]["model"] == "Unitree Go2"


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---- library-level behavior around downloads ----------------------------------------------------


def test_every_catalog_robot_has_a_file_list():
    for robot in library.CATALOG.values():
        names = {f.path for f in menagerie.files(robot.folder)}
        assert robot.mjcf in names and "LICENSE" in names
        assert library.download_mb(robot.name) > 1


def test_motors_become_servos_in_radians_even_in_degree_models():
    spec = mujoco.MjSpec.from_string(
        """<mujoco><compiler angle="degree" autolimits="true"/><worldbody><body>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-90 45"/><geom size="0.05"/>
        <body pos="0.1 0 0"><joint name="spin" type="hinge" axis="1 0 0"/><geom size="0.05"/></body>
        </body></worldbody><actuator>
        <motor name="elbow" joint="elbow" ctrlrange="-5 5" gear="2"/><motor name="spin" joint="spin" ctrlrange="-1 1"/>
        <velocity name="spin_speed" joint="spin" kv="3"/></actuator></mujoco>"""
    )
    assert library._motors_to_servos(spec, kp=50.0) == 2
    model = spec.compile()
    elbow, spin, speed = (model.actuator(n).id for n in ("elbow", "spin", "spin_speed"))
    np.testing.assert_allclose(model.actuator_ctrlrange[elbow], np.radians([-90, 45]))
    np.testing.assert_allclose(model.actuator_forcerange[elbow], [-10, 10])  # torque limit times gear
    assert not model.actuator_ctrllimited[spin]  # a continuous joint takes any target
    assert is_position_servo(model, elbow) and is_position_servo(model, spin)
    assert not is_position_servo(model, speed)  # velocity actuators are not position servos


# ---- real downloads (opt-in) ----------------------------------------------------------------------


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("name", list(library.CATALOG))
def test_real_library_robot_loads_and_holds_still(name):
    model = compile_world(World.model_validate({"robots": [{"name": "r", "library": name}]})).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    root = model.body(next(b for b in range(1, model.nbody) if model.body_parentid[b] == 0)).id
    start = data.xpos[root].copy()
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert data.xmat[root].reshape(3, 3)[2, 2] > 0.95  # upright
    assert np.linalg.norm(data.xpos[root] - start) < 0.08  # settles where it was put


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_go2_walks(pg):
    pg.add_robot("go2", library="go2")
    pg.behaviors.move("go2", 0.0, 0.0)
    _run(pg, 5.0)
    assert pg.behaviors.pose("go2")["upright"]
    start = pg.behaviors.pose("go2")["position"]
    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 10.0)
    end = pg.behaviors.pose("go2")
    assert end["position"][0] - start[0] > 2.4 and end["upright"]
