"""Robot library: catalog, world integration, walking library quadrupeds, MCP tools.

Most tests run offline against tests/fixtures/menagerie, small stand-ins that mirror Menagerie's
folder layout, names, kinematics and actuators (see the comment at the top of each file).
Downloading itself is tested in test_menagerie.py. The real robots are covered by opt-in tests:
set ROBOTFORGE_NETWORK_TESTS=1 to run them (the first run downloads each robot into the cache).
"""

import asyncio
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge import mcp_server
from robotforge.behaviors import Kind, humanoid, quadruped, robot_kind
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
    assert {e["kind"] for e in entries.values()} == {"quadruped", "arm", "humanoid", "gripper"}
    assert entries["go2"]["downloaded"] is True  # the offline fixture counts as downloaded


def test_unknown_library_robot_suggests_close_names(pg):
    with pytest.raises(PlaygroundError, match="Did you mean go2"):
        pg.add_robot("dog", library="go3")


def test_a_robot_has_exactly_one_source():
    with pytest.raises(ValueError, match="exactly one of 'template', 'library', 'file' or 'build'"):
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


@pytest.mark.parametrize(
    ("cube", "target"),
    [
        ([-0.335, 0.445], [0.034, -0.536]),  # from behind on the left to in front on the right
        ([-0.409, -0.018], [0.105, 0.548]),  # from right behind it round to its left
    ],
)
def test_panda_moves_a_floor_cube_round_its_base_without_throwing_it(pg, cube, target):
    """Crossing high from its raised home pose and coming straight down, the Panda twisted its upper arm (its first
    and third joints line up while the shoulder is upright) and reached the cube that way. The carry then went the
    long way round into its first joint's limit, and the first cube here was thrown 2.4 m past the place as the
    lowering swung through 334 degrees (the real Panda threw it 2.3 m).
    """
    pg.add_robot("panda", library="panda")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=cube)
    _run(pg, 0.5)
    motion, info = pg.behaviors.pick_and_place("panda", "cube", target)
    pg.run_fast(lambda: motion.done, 40)
    _run(pg, 0.5)
    placed = pg.behaviors.placed(info["object"], info["region"])
    assert placed["success"], placed


def test_piper_carries_the_long_way_round_to_a_place_just_past_its_base_joint_limit(pg):
    """The place is 0.6 degrees past the PiPER's +-150 degree base joint limit, behind it on the right. Planning the
    short way round, through the back where the gripper cannot point down, refused the pick as one that would hit
    "the robot carrying the arm", though the PiPER stands alone and lands it the long way round."""
    pg.add_robot("piper", library="piper")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[-0.083, 0.244])
    _run(pg, 0.5)
    motion, info = pg.behaviors.pick_and_place("piper", "cube", [-0.224, -0.126])
    pg.run_fast(lambda: motion.done, 40)
    _run(pg, 0.5)
    placed = pg.behaviors.placed(info["object"], info["region"])
    assert placed["success"], placed


def test_an_arm_without_a_tool_point_says_so_rather_than_that_it_is_not_an_arm(pg, monkeypatch):
    """The SO-100's model marked no tool point: move_arm answered "arm is an arm, not an arm, and carries none". The
    library gives it one now (see Tool), so here a library arm is told it has none."""
    so100 = library.CATALOG["so_arm100"]
    monkeypatch.setitem(library.CATALOG, "so_arm100", dataclasses.replace(so100, tools=()))
    pg.add_robot("arm", library="so_arm100")
    with pytest.raises(ValueError, match="arm is an arm, but its model marks no tool point .*; set_controls moves"):
        pg.behaviors.move_arm("arm", [0.2, 0.0, 0.1])


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
    assert robot_kind(pg.session.model, "arm") == Kind.ARM  # by its tool point, not its legs
    with pytest.raises(ValueError, match="only legged robots"):
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


def test_adding_the_same_robot_twice_is_harmless(pg):
    pg.add_robot("go2", library="go2")
    _run(pg, 0.5)
    t = pg.session.data.time
    again = pg.add_robot("go2", library="go2")  # e.g. a retry after the client timed out
    assert again["unchanged"] is True and again["robots"] == ["go2"]
    assert pg.session.data.time == t  # the simulation was not reloaded
    with pytest.raises(PlaygroundError, match="unique"):
        pg.add_robot("go2", library="go2", pos=[1.0, 0.0])  # a different request under the same name


@pytest.mark.anyio
async def test_mcp_reports_download_progress(pg, monkeypatch):
    def slow_fetch(folder, progress=None):
        if progress is None:  # the build after the download: already "downloaded"
            return FIXTURES / folder
        for i in range(3):
            progress(i / 3, f"step {i}")
            time.sleep(0.6)
        return FIXTURES / folder

    monkeypatch.setattr(library, "is_downloaded", lambda _name: False)
    monkeypatch.setattr(menagerie, "fetch", slow_fetch)
    monkeypatch.setattr(mcp_server, "PROGRESS_INTERVAL", 0.2)
    updates = []

    async def on_progress(progress, total, message):
        updates.append((progress, total, message))

    server = build_server(pg, find_free_port(8792))
    async with Client(server) as client:
        result = await client.call_tool("add_robot", {"name": "go2", "library": "go2"}, progress_callback=on_progress)
    assert not result.is_error
    assert len(updates) >= 3 and all(total == 100 for _, total, _ in updates)
    assert [p for p, _, _ in updates] == sorted(p for p, _, _ in updates)  # never goes backwards
    assert any("Unitree Go2: step 1" in m for *_, m in updates)


@pytest.mark.anyio
async def test_a_retried_call_joins_the_running_download(pg, monkeypatch):
    downloads = []

    def slow_fetch(folder, progress=None):
        if progress is not None:
            downloads.append(folder)
            time.sleep(1.0)
        return FIXTURES / folder

    monkeypatch.setattr(library, "is_downloaded", lambda _name: False)
    monkeypatch.setattr(menagerie, "fetch", slow_fetch)
    server = build_server(pg, find_free_port(8794))
    async with Client(server) as client:
        first, retry = await asyncio.gather(
            client.call_tool("add_robot", {"name": "go2", "library": "go2"}),
            client.call_tool("add_robot", {"name": "go2", "library": "go2"}),
        )
    assert downloads == ["unitree_go2"]  # one download, both calls waited for it
    bodies = [json.loads(r.content[0].text) for r in (first, retry)]
    assert sorted(bool(b.get("unchanged")) for b in bodies) == [False, True]


@pytest.mark.anyio
async def test_a_failed_download_reaches_the_agent_as_a_tool_error(pg, monkeypatch):
    def broken(folder, progress=None):
        raise menagerie.MenagerieError("could not reach GitHub (MuJoCo Menagerie); check your internet connection")

    monkeypatch.setattr(library, "is_downloaded", lambda _name: False)
    monkeypatch.setattr(menagerie, "fetch", broken)
    server = build_server(pg, find_free_port(8793))
    async with Client(server) as client:
        result = await client.call_tool("add_robot", {"name": "go2", "library": "go2"})
    assert result.is_error
    assert "check your internet connection" in result.content[0].text
    assert pg.current_world().robots == []  # nothing half-added


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


# ---- the robots added for variety: dogs, humanoids, arms, a gripper ---------------------------------


def test_new_robots_are_listed_with_their_kind_and_licence():
    entries = {e["name"]: e for e in library.catalog()}
    expected = {
        "anymal_c": ("quadruped", "BSD-3-Clause"),
        "barkour": ("quadruped", "Apache-2.0"),
        "op3": ("humanoid", "Apache-2.0"),
        "t1": ("humanoid", "Apache-2.0"),
        "z1": ("arm", "BSD-3-Clause"),
        "ur5e": ("arm", "BSD-3-Clause"),
        "xarm7": ("arm", "BSD-3-Clause"),
        "aloha": ("arm", "BSD-3-Clause"),
        "robotiq_2f85": ("gripper", "BSD-2-Clause"),
    }
    for name, (kind, licence) in expected.items():
        assert (entries[name]["kind"], entries[name]["license"]) == (kind, licence)
        assert 1 < entries[name]["download_mb"] < 60


def test_a_robot_without_a_keyframe_stands_on_the_home_pose_it_is_given_facing_forward():
    built = library.build("anymal_c")
    # Its file turns it 180 degrees (a scene's choice); every robot here faces +x.
    assert built.spec.worldbody.first_body().quat == pytest.approx([1, 0, 0, 0])
    assert built.home_qpos["LF_HFE"] == pytest.approx(0.8) and built.home_qpos["LH_KFE"] == pytest.approx(1.4)
    model = compile_world(World.model_validate({"robots": [{"name": "dog", "library": "anymal_c"}]})).model
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    feet = [model.geom(g) for g in range(model.ngeom) if model.geom(g).name == "" and model.geom_priority[g] == 1]
    assert feet and min(data.geom_xpos[f.id][2] - model.geom_size[f.id][0] for f in feet) == pytest.approx(0, abs=3e-3)


def test_anymal_walks_on_the_gait(pg):
    pg.add_robot("dog", library="anymal_c")
    _run(pg, 1.0)
    start = pg.behaviors.pose("dog")["position"]
    pg.behaviors.move("dog", 0.3, 0.0)
    _run(pg, 8.0)
    end = pg.behaviors.pose("dog")
    assert end["position"][0] - start[0] > 1.8 and abs(end["position"][1] - start[1]) < 0.3 and end["upright"]


def _walk_and_watch(pg: Playground, robot: str, speed: float, turn: float, seconds: float) -> dict:
    """Walk a robot and watch it: how far it tilted (degrees), turned (degrees), and which of its bodies touched the
    floor away from its feet (where its gait puts them): a shin's tip beside its foot is fine, a knee is not."""
    m, d = pg.session.model, pg.session.data
    pg.behaviors.move(robot, speed, turn)
    gait = pg.session.controllers[robot]
    trunk = gait.trunk
    tilt, turned, last, touched = 0.0, 0.0, None, set()
    for _ in range(round(seconds / 0.05)):
        _run(pg, 0.05)
        rot = d.xmat[trunk].reshape(3, 3)
        tilt = max(tilt, math.degrees(math.acos(min(1.0, rot[2, 2]))))
        heading = math.atan2(rot[1, 0], rot[0, 0])
        turned += 0.0 if last is None else (heading - last + math.pi) % (2 * math.pi) - math.pi
        last = heading
        feet = [
            d.xpos[leg.chain.body] + d.xmat[leg.chain.body].reshape(3, 3) @ leg.chain.local_point for leg in gait.legs
        ]
        for c in d.contact[: d.ncon]:
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            own = g2 if m.geom_bodyid[g1] == 0 else g1 if m.geom_bodyid[g2] == 0 else None
            if own is not None and min(np.linalg.norm(c.pos - foot) for foot in feet) > 0.06:
                touched.add(m.body(m.geom_bodyid[own]).name)
    return {"tilt": tilt, "turned": math.degrees(turned), "touched": touched}


@pytest.mark.parametrize("turn", [1.0, -1.0])
def test_anymal_turns_at_keyboard_commands_without_rocking_onto_its_knees(pg, turn):
    """Regression: on Menagerie's soft servos (kp 100, damping 1) the 45 kg robot sagged and rocked. Walking and
    turning like the keyboard does (0.6 of its top speed, 1 rad/s), it tilted up to 34 degrees and its thighs and
    shins hit the floor a third of the time."""
    pg.add_robot("dog", library="anymal_c")
    _run(pg, 2.0)
    pg.behaviors.move("dog", 0.0, 0.0)
    walked = _walk_and_watch(pg, "dog", 0.6 * pg.session.controllers["dog"].max_speed, turn, 6.0)
    assert walked["tilt"] < 10 and not walked["touched"]
    assert abs(walked["turned"]) > 250 and walked["turned"] * turn > 0  # about 0.9 of the 344 degrees it was told


def test_op3_stands_and_walks_on_the_balance_gait_found_from_its_legs(pg):
    pg.add_robot("op3", library="op3")
    gait = pg.session.controllers["op3"]  # it balances from the start
    assert isinstance(gait, humanoid.HumanoidController)
    for roles in gait.roles:  # hip roll, ankle pitch and ankle roll of each leg, from the way their axes point
        assert roles.ankle_roll is not None and roles.hip_roll[0] == 1 and roles.ankle_pitch[0] == 4
    _run(pg, 2.0)
    start = pg.behaviors.pose("op3")["position"]
    pg.behaviors.move("op3", 0.1, 0.0)
    _run(pg, 6.0)
    end = pg.behaviors.pose("op3")
    assert end["position"][0] - start[0] > 0.45 and end["upright"]


def test_soft_joint_servos_are_stiffened_keeping_their_force_limits():
    model = library.build("anymal_c").spec.compile()
    legs = [a for a in range(model.nu) if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT]
    assert len(legs) == 12
    for a in legs:
        assert model.actuator_gainprm[a, 0] == 300 and tuple(model.actuator_biasprm[a, :3]) == (0, -300, -10)
        assert tuple(model.actuator_forcerange[a]) == (-80, 80)


def test_a_library_humanoid_walks_no_faster_than_its_known_top_speed(pg, monkeypatch):
    """Booster T1 fell within 2 s at the 0.17 m/s its gait allows, and walks at 0.12; the catalog's top speed binds
    humanoids as it does dogs."""
    monkeypatch.setitem(library.CATALOG, "op3", dataclasses.replace(library.CATALOG["op3"], max_speed=0.06))
    pg.add_robot("bot", library="op3")
    assert pg.behaviors.robots()[0]["max_speed"] == 0.06
    moved = pg.behaviors.move("bot", 0.2, 0.0)
    assert moved["notes"] == ["speed 0.2 m/s is above max_speed 0.06, so it moves at 0.06"]


def test_driving_and_listing_robots_read_world_yaml_once(pg, monkeypatch):
    """The keyboard drives a robot with a move every few frames and the viewer lists robots often; each read of
    world.yaml parses it. Regression: the checks added for library robots read it two to five times a call."""
    pg.add_robot("go2", library="go2")
    pg.add_robot("op3", library="op3", pos=[1, 1])
    reads = []
    load = pg.store.load
    monkeypatch.setattr(pg.store, "load", lambda: reads.append(1) or load())
    for call in (
        lambda: pg.behaviors.move("go2", 0.2, 0.1),
        lambda: pg.behaviors.move("op3", 0.05, 0.0),
        pg.behaviors.robots,
    ):
        reads.clear()
        call()
        assert len(reads) == 1


def test_arm_poses_are_refused_on_a_real_humanoid_whose_arms_hold_still(pg):
    pg.add_robot("op3", library="op3")
    with pytest.raises(ValueError, match="wave moves only the arms and head, which on op3 hold still; it can: squat"):
        pg.behaviors.strike_pose("op3", "wave")
    pg.behaviors.strike_pose("op3", "squat")  # drops the hips: that works on any legs


def test_an_arm_with_a_bare_flange_reaches(pg):
    pg.add_robot("ur", library="ur5e")
    m = pg.session.model
    tcp, flange, attachment = (m.site(f"ur/{s}").id for s in ("tcp", "flange", "attachment_site"))
    # Both sit where Menagerie's attachment site is, pointing the way it does.
    _run(pg, 0.1)
    d = pg.session.data
    np.testing.assert_allclose(d.site_xpos[tcp], d.site_xpos[attachment], atol=1e-9)
    np.testing.assert_allclose(d.site_xmat[flange], d.site_xmat[attachment], atol=1e-9)
    motion = pg.behaviors.move_arm("ur", [0.4, 0.1, 0.2])
    pg.run_fast(lambda: motion.done, 10.0)  # round its base, timed by the way it goes: about 4 s
    _run(pg, 0.5)
    assert np.linalg.norm(d.site_xpos[tcp] - [0.4, 0.1, 0.2]) < 0.012


def test_a_gripper_that_closes_high_is_turned_around_exactly():
    spec = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body><joint name="slide" type="slide" axis="1 0 0" range="0 0.05"/>
        <geom size="0.01" mass="0.1"/></body></worldbody><actuator><general name="fingers" joint="slide"
        ctrlrange="0 255" gainprm="0.3" biasprm="0 -100 -10" biastype="affine"/></actuator></mujoco>"""
    )
    before = spec.compile()
    low, high = library._turn_around(spec.actuators[0], library.CATALOG["robotiq_2f85"])
    after = spec.compile()
    assert (low, high) == (0, 255)
    for ctrl in (0.0, 80.0, 255.0):
        d1, d2 = mujoco.MjData(before), mujoco.MjData(after)
        d1.qpos[0] = d2.qpos[0] = 0.02
        d1.ctrl[0], d2.ctrl[0] = ctrl, low + high - ctrl
        mujoco.mj_forward(before, d1)
        mujoco.mj_forward(after, d2)
        assert d2.actuator_force[0] == pytest.approx(d1.actuator_force[0])


def test_two_armed_aloha_has_an_arm_on_each_side(pg):
    pg.add_robot("aloha", library="aloha")
    assert pg.behaviors.robots()[0]["arms"] == ["aloha/left", "aloha/right"]
    with pytest.raises(ValueError, match="aloha has 2 arms; name the one you mean: aloha/left, aloha/right"):
        pg.behaviors.move_arm("aloha", [0.0, 0.0, 0.2])
    arm = pg.behaviors.move_arm("aloha/right", [0.15, 0.0, 0.2])
    _run(pg, 2.5)
    assert np.linalg.norm(pg.session.data.site_xpos[arm.site] - [0.15, 0.0, 0.2]) < 0.02
    assert pg.session.model.actuator(arm.gripper).name == "aloha/right/gripper"


def test_a_piper_leans_over_a_bins_rim_to_put_a_cube_in(pg):
    """Regression: a PiPER's wrist cannot point its gripper down above about 13 cm at 30 cm out, so it refused to lift
    a cube over an 8 cm bin's rim. High above the grasp and the place its gripper may now lean; it still grasps and
    places pointing down, and move_arm still only points down."""
    pg.add_robot("arm", library="piper")
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.3, -0.3])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.08], pos=[0.3, 0.3], color="blue")
    _run(pg, 1.0)
    arm = pg.behaviors._arm("arm")
    _, error, _, axis = arm._solve_tilting(np.array([0.3, 0.3, 0.15]))
    assert error < 0.008 and np.degrees(np.arccos(-axis[2])) == pytest.approx(20, abs=1)  # it leans 20 degrees
    with pytest.raises(ValueError, match="cannot reach .* with its gripper pointing down"):
        pg.behaviors.move_arm("arm", [0.3, 0.3, 0.15])
    motion, info = pg.behaviors.pick_and_place("arm", "cube", "bin")
    for _ in range(80):
        _run(pg, 0.25)
        if motion.done:
            break
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]


def test_suggestions_for_a_misspelt_robot_start_like_it():
    with pytest.raises(library.LibraryError, match="Did you mean go2, go1, op3"):
        library.get("go3")


# ---- real downloads (opt-in) ----------------------------------------------------------------------


# Humanoids without a learned walk stand on the balance gait, not on their servos alone: see the test below them.
ON_SERVOS = [name for name, robot in library.CATALOG.items() if robot.kind != "humanoid" or name == "g1"]


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("name", ON_SERVOS)
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


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("push", [(0.3, 0.0), (-0.3, 0.0), (0.0, 0.3)])
@pytest.mark.parametrize("name", ["op3", "t1"])
def test_real_humanoids_on_the_balance_gait_take_a_shove(pg, name, push):
    """A T1 on Menagerie's soft servos (kp 75, kv 5) fell over at 0.4 m/s and swayed 12 to 20 degrees at 0.3."""
    pg.add_robot("bot", library=name)
    _run(pg, 2.0)
    m, d = pg.session.model, pg.session.data
    trunk = pg.behaviors._root_body("bot")
    dof = m.jnt_dofadr[m.body_jntadr[trunk]]
    d.qvel[dof : dof + 2] += push
    tilt = 0.0
    for _ in range(80):
        _run(pg, 0.05)
        tilt = max(tilt, math.degrees(math.acos(min(1.0, d.xmat[trunk].reshape(3, 3)[2, 2]))))
    assert tilt < 10 and pg.behaviors.pose("bot")["upright"]


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("name", ["op3", "t1"])
def test_real_humanoid_stands_ten_seconds_on_the_balance_gait(pg, name):
    pg.add_robot("bot", library=name)
    _run(pg, 0.5)
    start = pg.behaviors.pose("bot")["position"]
    _run(pg, 10.0)
    end = pg.behaviors.pose("bot")
    assert end["upright"] and math.dist(start[:2], end["position"][:2]) < 0.1


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("turn", [1.0, -1.0])
@pytest.mark.parametrize("name", [n for n, r in library.CATALOG.items() if r.kind == "quadruped"])
def test_real_dogs_turn_at_keyboard_commands_without_rocking_onto_their_knees(pg, name, turn):
    pg.add_robot("dog", library=name)
    _run(pg, 2.0)
    pg.behaviors.move("dog", 0.0, 0.0)
    walked = _walk_and_watch(pg, "dog", 0.6 * pg.session.controllers["dog"].max_speed, turn, 6.0)
    assert walked["tilt"] < 10 and not walked["touched"]
    assert abs(walked["turned"]) > 250 and walked["turned"] * turn > 0


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize(
    "name, speed, distance", [("anymal_c", 0.3, 2.4), ("barkour", 0.3, 2.4), ("op3", 0.1, 0.8), ("t1", 0.12, 0.9)]
)
def test_real_new_walkers_walk_ten_seconds(pg, name, speed, distance):
    pg.add_robot("bot", library=name)
    _run(pg, 2.0)
    start = pg.behaviors.pose("bot")["position"]
    pg.behaviors.move("bot", speed, 0.0)
    _run(pg, 10.0)
    end = pg.behaviors.pose("bot")
    assert end["position"][0] - start[0] > distance and abs(end["position"][1] - start[1]) < 0.5 and end["upright"]


@pytest.mark.parametrize("name", ["panda", "piper"])
def test_an_arm_grasps_a_turned_box_square_to_its_faces(pg, name):
    """Regression: pointing down left the turn of the jaws about the tool free, so they came down across a cube
    wherever the IK left them; beside an SO-100 its fixed jaw landed on the cube's edge and pushed it away."""
    pg.add_robot("bot", library=name)
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.15], yaw=30)
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.08], pos=[0.35, -0.25])
    _run(pg, 1.0)
    motion, info = pg.behaviors.pick_and_place("bot", "cube", "bin")
    while motion.current_label != "grasp":
        _run(pg, 0.05)
    closing = pg.session.data.xmat[motion.chain.body].reshape(3, 3) @ motion.closing
    off = (math.degrees(math.atan2(closing[1], closing[0])) - 30) % 90
    assert min(off, 90 - off) < 5  # closing across a face, whichever
    pg.run_fast(lambda: motion.done, 40)
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]


def test_an_arm_that_cannot_pick_yet_is_refused_and_says_what_works(pg, monkeypatch):
    """A Unitree Z1's one-jaw claw lets go of a cube as the arm swings; pick_and_place ran anyway and pushed the cube
    5 to 15 cm, taking the whole motion's time."""
    piper = library.CATALOG["piper"]
    tool = dataclasses.replace(piper.tools[0], cannot_pick="its claw lets go")
    monkeypatch.setitem(library.CATALOG, "piper", dataclasses.replace(piper, tools=(tool,)))
    pg.add_robot("bot", library="piper")
    pg.add_robot("dog", library="go2", pos=[2.0, 0.0], mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.15])
    for arm in ("bot", "dog/arm"):
        with pytest.raises(
            ValueError, match=rf"{arm} \(AgileX PiPER\) cannot pick and place yet: its claw lets go; mo"
        ):
            pg.behaviors.pick_and_place(arm, "cube", [0.3, -0.2])


# Cube and bin (x, y) around an arm's base, for an arm facing +x: in front, from the front to either side, and long
# carries across it. An xArm 7 gripping at its fingertips dropped the cube on four of these.
PICK_LAYOUTS = {
    "front": ([0.4, 0.15], [0.35, -0.25]),
    "to_left": ([0.35, 0.0], [0.0, 0.4]),
    "to_right": ([0.35, 0.0], [0.0, -0.4]),
    "carry_left": ([0.3, -0.3], [0.3, 0.3]),
    "carry_right": ([0.3, 0.3], [0.3, -0.3]),
    "diagonal": ([0.45, -0.1], [0.2, 0.35]),
}
# The arms with a gripper that pick from the floor: (robot, arm, mounts, base (x, y), 1 facing +x or -1 facing -x).
PICKERS = {
    "panda": ("panda", "bot", None, (0.0, 0.0), 1),
    "piper": ("piper", "bot", None, (0.0, 0.0), 1),
    "xarm7": ("xarm7", "bot", None, (0.0, 0.0), 1),
    "ur5e_2f85": ("ur5e", "bot", [{"library": "robotiq_2f85"}], (0.0, 0.0), 1),
    "aloha_left": ("aloha", "bot/left", None, (-0.469, -0.019), 1),
    "aloha_right": ("aloha", "bot/right", None, (0.469, -0.019), -1),
}


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("layout", PICK_LAYOUTS)
@pytest.mark.parametrize("picker", PICKERS)
def test_real_arms_pick_a_cube_into_a_bin(pg, picker, layout):
    name, arm, mounts, base, facing = PICKERS[picker]
    cube, bin_at = ([base[0] + facing * x, base[1] + y] for x, y in PICK_LAYOUTS[layout])
    pg.add_robot("bot", library=name, mounts=mounts)
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=cube)
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.08], pos=bin_at, color="blue")
    _run(pg, 1.0)
    motion, info = pg.behaviors.pick_and_place(arm, "cube", "bin")
    for _ in range(80):
        _run(pg, 0.25)
        if motion.done:
            break
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]


# The SO-100 picks in front of it (+x), within about 0.27 m: cube and tray (x, y) around its base at (0.045, 0).
SO100_LAYOUTS = {
    "front": ([0.15, 0.2], [0.28, 0.05]),
    "side": ([0.3, 0.0], [0.2, 0.2]),
    "across": ([0.1, 0.2], [0.1, -0.2]),
    "back_across": ([0.15, -0.2], [0.15, 0.2]),
    "near": ([0.18, 0.12], [0.2, -0.12]),
    # Beside it: its jaws came down 35 degrees across the cube and pushed it away, until they turned square to it.
    "beside": ([0.0, 0.22], [0.28, 0.05]),
    "beside_far": ([-0.05, 0.25], [0.3, 0.0]),
}


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.parametrize("layout", SO100_LAYOUTS)
def test_real_so100_picks_a_cube_from_the_floor_into_a_tray(pg, layout):
    """Regression: grasping in the middle of its long jaws would have put their tips into the floor, so it could not
    pick from the floor at all."""
    cube, tray = SO100_LAYOUTS[layout]
    pg.add_robot("bot", library="so_arm100")
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=cube)
    pg.add_object("tray", shape="bin", size=[0.16, 0.16, 0.04], pos=tray, color="blue")
    _run(pg, 1.0)
    motion, info = pg.behaviors.pick_and_place("bot", "cube", "tray")
    for _ in range(80):
        _run(pg, 0.25)
        if motion.done:
            break
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_z1_refuses_pick_and_place_but_moves_its_arm(pg):
    pg.add_robot("z1", library="z1")
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.45, 0.0])
    _run(pg, 1.0)
    with pytest.raises(ValueError, match="z1 \\(Unitree Z1\\) cannot pick and place yet: its one-jaw claw lets go"):
        pg.behaviors.pick_and_place("z1", "cube", [0.0, 0.4])
    motion = pg.behaviors.move_arm("z1", [0.4, 0.1, 0.1])
    pg.run_fast(lambda: motion.done, 10)
    assert np.linalg.norm(pg.session.data.site_xpos[motion.site] - [0.4, 0.1, 0.1]) < 0.02
