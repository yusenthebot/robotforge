"""Robots carrying robots (mounting.py): an arm on a robot dog's back, a gripper on an arm's flange.

Offline against the Menagerie stand-ins in tests/fixtures/menagerie, like test_library.py.
"""

import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
from mcp import Client

from robotforge import tool_input
from robotforge.behaviors import Kind
from robotforge.cli import find_free_port
from robotforge.design import library, mounting
from robotforge.design.templates.base import BuiltRobot
from robotforge.design.world import MountEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.geometry import lowest_point
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
    playground.new_world("lab", "checker")
    return playground


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def _arm_done(pg: Playground, arm: str, limit: float = 40.0) -> None:
    for _ in range(round(limit / 0.25)):
        _run(pg, 0.25)
        if pg.session.controllers[arm].done:
            return


# ---- an arm on a robot dog ----------------------------------------------------------------------------------------


def test_an_arm_rides_on_top_of_a_dogs_back(pg):
    result = pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    assert "go2: arm: AgileX PiPER mounted on top of base" in result["notes"]
    assert pg.behaviors.robots() == [
        {
            "name": "go2",
            "kind": "quadruped",
            "model": "Unitree Go2",
            "arms": ["go2/arm"],
            "max_speed": pytest.approx(0.6, abs=0.03),
        }
    ]
    m, d = pg.session.model, pg.session.data
    trunk, arm_base = m.body("go2/base").id, m.body("go2/arm/base_link").id
    assert m.body_parentid[arm_base] == trunk  # fixed to the trunk: no joint of its own
    assert m.body_jntnum[arm_base] == 0
    # It sits on the trunk's top, not inside it, and the dog still stands on its feet.
    mujoco.mj_forward(m, d)
    top = max(
        d.geom_xpos[g][2] + m.geom_size[g][2]
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == trunk and m.geom_type[g] == 6
    )
    assert d.xpos[arm_base][2] == pytest.approx(top, abs=0.002)
    assert pg.behaviors.pose("go2")["upright"]


def test_a_mount_rests_on_the_surface_under_it_not_on_a_shells_bounding_box():
    """Regression: the top came from the bounding boxes of the shapes over the middle, and the real Go2's shell is
    curved, so a PiPER floated 3 cm over its back. Rays dropped onto the shapes find the surface itself."""
    spec = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body name="trunk"><freejoint/>
        <geom type="box" size="0.2 0.1 0.04"/>
        <geom type="box" size="0.3 0.1 0.005" pos="0 0 0.1" euler="0 25 0" contype="0" conaffinity="0" group="2"/>
        </body></worldbody></mujoco>"""
    )
    _, top, _, _ = mounting._on_top(spec, {}, [], 0.03)
    # The slanted plate's bounding box reaches 0.23 m. Its top face passes 0.005 m above its centre (0.1 m) and drops
    # by tan(25 deg) per metre along x. Sloping the same everywhere, nowhere is flatter than over the centre of mass
    # (x = 0), and there the highest point under the base is at its back edge (x = -0.03 m), 12 cm high.
    a = np.radians(25)
    surface = 0.1 + 0.005 * np.cos(a) + (0.03 + 0.005 * np.sin(a)) * np.tan(a)
    assert top[2] == pytest.approx(surface, abs=1e-4) and top[2] < 0.13


def _carrier(xml: str) -> BuiltRobot:
    return BuiltRobot(spec=mujoco.MjSpec.from_string(xml), floating=True)


def _base_bottom(built: BuiltRobot, mount: str = "arm") -> float:
    """The lowest point of what a mount's base looks like, in its carrier's frame (the carrier stands at the origin)."""
    model = built.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    base = next(b for b in range(model.nbody) if model.body(b).name.startswith(f"{mount}/"))
    return min(float(lowest_point(model, data, g)[2]) for g in mounting._surface(model, [base]))


# Spot's back: what you see (its yellow shell), and a collision box 9 mm above it that nobody sees.
SHELL_AND_BOX = """<mujoco><worldbody><body name="body"><freejoint/>
    <geom type="box" size="0.3 0.1 0.04" contype="0" conaffinity="0" group="2"/>
    <geom type="box" size="0.3 0.1 0.049" group="3"/>
    </body></worldbody></mujoco>"""


def test_a_mount_rests_on_what_you_see_not_on_a_hidden_collision_box():
    """Regression: a Z1 on Spot rested on its invisible collision box and hovered 9 mm over the shell, and shifting it
    down to the shell was refused, measured against that box. A mount and the body it is welded to never touch in the
    simulation; only the looks matter."""
    piper = [MountEntry.model_validate({"library": "piper"})]
    built = mounting.attach(_carrier(SHELL_AND_BOX), piper, "spot")
    assert _base_bottom(built) == pytest.approx(0.04, abs=5e-4)
    sunk = [MountEntry.model_validate({"library": "piper", "shift": [0, 0, -0.009]})]
    with pytest.raises(ValueError, match=r"mount 'arm' sits 9 mm inside spot \(body\) at the home pose; shift "):
        mounting.attach(_carrier(SHELL_AND_BOX), sunk, "spot")


def test_a_narrow_knob_does_not_hold_a_mount_up():
    """Regression: resting on the highest point at the middle, a Z1 perched on the 13 mm payload knob of Spot's back.
    It stands beside the knob, flat on the back."""
    box = '<geom type="box" size="0.3 0.1 0.049" group="3"/>'
    knob = SHELL_AND_BOX.replace(box, '<geom type="box" size="0.015 0.02 0.0065" pos="0 0 0.0465"/>')
    built = mounting.attach(_carrier(knob), [MountEntry.model_validate({"library": "piper"})], "spot")
    assert _base_bottom(built) == pytest.approx(0.04, abs=5e-4)


@pytest.mark.parametrize(
    "carrier, mount, at",
    [("go2", "piper", "base"), ("op3", "so_arm100", "head_pan_link")],
)
def test_the_shift_an_overlap_refusal_suggests_lifts_the_mount_clear(carrier, mount, at):
    """Regression: the hint lifted a mount by how deep it sat from where it would be with no shift, so following it on a
    G1 left a PiPER stuck 117 mm inside the torso. The shift it suggests now is one that was tried."""
    entry = {"library": mount, "at": at}
    with pytest.raises(ValueError, match=r"shift \[([-\d., ]+)\] lifts it clear") as refused:
        mounting.attach(library.build(carrier), [MountEntry.model_validate(entry)], carrier)
    shift = [float(v) for v in re.search(r"shift \[([-\d., ]+)\]", str(refused.value)).group(1).split(",")]
    built = mounting.attach(library.build(carrier), [MountEntry.model_validate({**entry, "shift": shift})], carrier)
    assert any(f"mounted on {at}" in note for note in built.notes)


def test_a_mount_that_lifting_cannot_free_is_told_to_go_elsewhere():
    """Mounted at a humanoid's middle (inside its torso), lifting it only runs it into the head."""
    entry = MountEntry.model_validate({"library": "so_arm100", "at": "body_link"})
    with pytest.raises(ValueError, match="lifting it does not get it clear; mount it somewhere else with at"):
        mounting.attach(library.build("op3"), [entry], "op3")


def test_on_a_humanoid_top_is_its_head(pg):
    """Regression: on top of a humanoid's main body (its pelvis) is inside its torso, and a mount there was refused."""
    result = pg.add_robot("op3", library="op3", mounts=[{"library": "so_arm100"}])
    assert "op3: arm: SO-ARM100 mounted on top of head_pan_link" in result["notes"]


def test_the_dog_still_walks_with_the_arm_on_its_back(pg):
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    _run(pg, 1.0)
    start = pg.behaviors.pose("go2")["position"]
    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 8.0)
    end = pg.behaviors.pose("go2")
    assert end["position"][0] - start[0] > 2.0 and abs(end["position"][1] - start[1]) < 0.2 and end["upright"]
    # The arm holds its pose while the legs walk: only the legs belong to the gait.
    gait = pg.session.controllers["go2"]
    arm_joints = [j for j in range(pg.session.model.njnt) if pg.session.model.joint(j).name.startswith("go2/arm/")]
    assert arm_joints and not {int(pg.session.model.actuator_trnid[a, 0]) for a in gait._actuators} & set(arm_joints)


def test_a_dog_walks_to_a_cube_and_its_arm_puts_it_in_a_bin(pg):
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[1.05, 0.3])
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[1.05, -0.35], color="blue")
    _run(pg, 1.0)
    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 3.1)
    pg.behaviors.stop("go2")
    _run(pg, 2.0)
    stood = pg.behaviors.pose("go2")["position"]
    arm, info = pg.behaviors.pick_and_place("go2/arm", "cube", "bin")
    _arm_done(pg, "go2/arm")
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]
    after = pg.behaviors.pose("go2")
    assert after["upright"] and np.linalg.norm(np.subtract(after["position"][:2], stood[:2])) < 0.05


def test_an_arm_on_a_walking_dog_waits_until_the_dog_stands(pg):
    """Regression: the arm planned where the cube was while the dog walked on, so it reached at nothing (or the
    dog fell over as it reached out). Planning is refused while the carrier moves; stop_carrier stops it and says
    when it stands still."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[1.05, 0.3])
    _run(pg, 1.0)
    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 2.0)
    with pytest.raises(ValueError, match="go2 is moving, and go2/arm plans its motion where things are now; stop go2"):
        pg.behaviors.pick_and_place("go2/arm", "cube", [1.05, -0.3])
    with pytest.raises(ValueError, match="go2 is moving"):
        pg.behaviors.move_arm("go2", [1.0, 0.0, 0.4])
    still = pg.behaviors.stop_carrier("go2/arm")
    assert still is not None and not still()
    with pytest.raises(ValueError, match=r"go2 is still coming to a stand \(0\.\d\d m/s\); give it a second"):
        pg.behaviors.pick_and_place("go2/arm", "cube", [1.05, -0.3])
    pg.run_fast(still, 3.0)
    assert still() and pg.behaviors.stop_carrier("go2/arm") is None  # standing: nothing to wait for
    x, y, _ = pg.behaviors.pose("go2")["position"]
    pg.behaviors.move_arm("go2/arm", [x + 0.3, y, 0.45])


@pytest.mark.anyio
async def test_mcp_pick_and_place_stops_a_walking_carrier_first(pg):
    server = build_server(pg, find_free_port(8797))
    async with Client(server) as client:

        async def call(tool: str, args: dict) -> dict:
            result = await client.call_tool(tool, args)
            assert not result.is_error, result.content[0].text
            return json.loads(result.content[0].text)

        await call("add_robot", {"name": "go2", "library": "go2", "mounts": [{"library": "piper"}]})
        await call("add_object", {"name": "cube", "shape": "box", "size": [0.04] * 3, "pos": [1.05, 0.3]})
        await call("add_object", {"name": "bin", "shape": "bin", "size": [0.18, 0.18, 0.08], "pos": [1.05, -0.35]})
        _run(pg, 1.0)
        await call("move", {"robot": "go2", "speed": 0.3, "seconds": None})  # walks on until told to stop
        _run(pg, 3.1)
        picked = await call("pick_and_place", {"robot": "go2/arm", "object": "cube", "target": "bin"})
        assert picked["success"] and picked["carrier_stopped"].startswith("go2 stopped first and stood still after")
        assert pg.behaviors.pose("go2")["upright"]


WALK_AND_PICK = """
from robotforge.script import connect

sim = connect()
sim.move("go2", speed=0.3)
sim.step(3.1)
print("in the bin:", sim.pick_and_place("go2/arm", "cube", "bin"))
"""


@pytest.mark.anyio
async def test_a_script_picks_from_a_walking_dog_after_it_stops(pg):
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[1.05, 0.3])
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[1.05, -0.35])
    _run(pg, 1.0)
    server = build_server(pg, find_free_port(8799))
    async with Client(server) as client:
        await client.call_tool("write_script", {"name": "walk_and_pick", "code": WALK_AND_PICK})
        ran = json.loads((await client.call_tool("run_script", {"name": "walk_and_pick"})).content[0].text)
    assert ran["ok"] and "in the bin: True" in ran["output"], ran["output"]


def test_a_robot_imported_from_a_file_carries_a_library_arm(pg):
    pg.add_robot(
        "dog", file=str(FIXTURES / "unitree_go2" / "go2.xml"), pos=[0, 0, 0.5], mounts=[{"library": "so_arm100"}]
    )
    assert pg.behaviors.arms("dog") == ["dog/arm"]
    m = pg.session.model
    assert m.body_parentid[m.body("dog/arm/Base").id] == m.body("dog/base").id


def test_a_template_arm_can_ride_on_a_library_dog(pg):
    pg.add_robot("go2", library="go2", mounts=[{"template": "arm", "params": {"reach": 0.4}}])
    assert pg.behaviors.arms("go2") == ["go2/arm"]
    assert pg.behaviors.kind("go2/arm") == Kind.ARM


# ---- a gripper on an arm ------------------------------------------------------------------------------------------


def test_a_gripper_on_an_arms_flange_becomes_the_arms_own(pg):
    result = pg.add_robot("ur", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    assert "ur: gripper: Robotiq 2F-85 on the arm's flange; it is now the arm's gripper" in result["notes"]
    m = pg.session.model
    names = {m.site(s).name for s in range(m.nsite)}
    assert "ur/tcp" in names and "ur/flange" not in names and "ur/gripper/tcp" not in names
    assert m.actuator("ur/gripper").id >= 0  # the Robotiq's fingers, turned around: the top of the range opens
    assert pg.behaviors.arms("ur") == ["ur"]
    # The grasp point is the Robotiq's pinch point, 14.5 cm past the flange along the way the flange points.
    _run(pg, 0.1)
    d = pg.session.data
    tcp, flange = d.site_xpos[m.site("ur/tcp").id], d.site_xpos[m.site("ur/attachment_site").id]
    axis = d.site_xmat[m.site("ur/attachment_site").id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(tcp - flange, (0.007 + 0.0038 + 0.145) * axis, atol=2e-3)
    arm = pg.behaviors.move_arm("ur", [0.4, 0.1, 0.2], gripper="open")
    _arm_done(pg, "ur")
    _run(pg, 1.0)
    assert np.linalg.norm(d.site_xpos[arm.site] - [0.4, 0.1, 0.2]) < 0.012
    assert arm.gripper_open == 255.0 and arm.gripper_closed == 0.0


def test_an_arm_without_a_gripper_will_not_pick_and_says_how_to_mount_one(pg):
    pg.add_robot("ur", library="ur5e")
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.15])
    with pytest.raises(ValueError, match='has no gripper to pick with; mount one .*"robotiq_2f85"'):
        pg.behaviors.pick_and_place("ur", "cube", [0.3, -0.2])


@pytest.mark.parametrize("carrier", ["panda", "go2"])
def test_a_gripper_goes_only_on_a_bare_flange_unless_told_where(pg, carrier):
    """Regression: with no flange to go on, a gripper went on top: of a Panda's base, beside its own hand, or on a dog's
    back, where it is a dead weight."""
    with pytest.raises(ValueError, match=f"goes on the bare flange of an arm, and {carrier} has none"):
        pg.add_robot(carrier, library=carrier, mounts=[{"library": "robotiq_2f85"}])
    pg.add_robot(carrier, library=carrier, mounts=[{"library": "robotiq_2f85", "at": "top"}])  # said so: it goes


def test_at_tool_needs_an_arm_with_a_bare_flange(pg):
    with pytest.raises(ValueError, match="at 'tool' needs an arm with a bare flange"):
        pg.add_robot("go2", library="go2", mounts=[{"library": "robotiq_2f85", "at": "tool"}])


# ---- where it goes, and what is refused ---------------------------------------------------------------------------


def test_a_mount_inside_its_carrier_is_refused_with_the_overlap(pg):
    with pytest.raises(ValueError, match=r"mount 'arm' sits 57 mm inside go2 \(base\) at the home pose; shift \["):
        pg.add_robot("go2", library="go2", mounts=[{"library": "piper", "at": "base"}])
    assert pg.current_world().robots == []  # nothing half-added


def test_a_mount_on_a_body_that_does_not_exist_names_the_close_ones(pg):
    with pytest.raises(ValueError, match="no body 'bse' to mount on; did you mean base"):
        pg.add_robot("go2", library="go2", mounts=[{"library": "piper", "at": "bse"}])


def test_two_mounts_need_two_names(pg):
    with pytest.raises(ValueError, match="two mounts are called 'arm'; give each a name"):
        pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}, {"library": "so_arm100"}])


def test_shift_and_rotate_place_a_named_mount(pg):
    pg.add_robot(
        "go2",
        library="go2",
        mounts=[{"name": "left_arm", "library": "so_arm100", "shift": [0.1, 0.03, 0.0], "rotate": [0, 0, 90]}],
    )
    m, d = pg.session.model, pg.session.data
    mujoco.mj_forward(m, d)
    base = m.body("go2/left_arm/Base").id
    trunk = d.xmat[m.body("go2/base").id].reshape(3, 3)
    local = trunk.T @ (d.xpos[base] - d.xpos[m.body("go2/base").id])
    assert local[0] == pytest.approx(0.1, abs=0.03) and local[1] == pytest.approx(0.03, abs=0.03)
    yaw = (trunk.T @ d.xmat[base].reshape(3, 3))[:, 0]
    np.testing.assert_allclose(yaw, [0, 1, 0], atol=1e-6)  # turned 90 degrees to the left
    assert pg.behaviors.arms("go2") == ["go2/left_arm"]


def test_a_bad_mount_is_explained_by_world_yaml():
    with pytest.raises(ValueError, match="a mount is"):
        World.model_validate({"robots": [{"name": "go2", "library": "go2", "mounts": [{"name": "arm"}]}]})
    with pytest.raises(ValueError, match="shift is"):
        World.model_validate(
            {"robots": [{"name": "go2", "library": "go2", "mounts": [{"library": "piper", "shift": [1]}]}]}
        )


# ---- world.yaml and the tools -------------------------------------------------------------------------------------


def test_robots_without_mounts_are_written_as_before():
    plain = World.model_validate({"robots": [{"name": "dog", "template": "quadruped"}]})
    assert "mounts" not in plain.to_yaml()
    spliced = World.model_validate({"robots": [{"name": "go2", "library": "go2", "mounts": [{"library": "piper"}]}]})
    assert World.from_yaml(spliced.to_yaml()) == spliced
    assert len(compile_world(spliced).model.body("go2/arm/base_link").name) > 0


def test_update_robot_puts_on_and_takes_off_mounts(pg):
    pg.add_robot("go2", library="go2")
    pg.update_robot("go2", mounts=[{"library": "piper"}])
    assert pg.behaviors.arms("go2") == ["go2/arm"]
    pg.update_robot("go2", mounts=[])
    assert pg.behaviors.arms("go2") == []


def test_an_empty_mounts_list_filled_in_beside_another_change_keeps_the_arm():
    schema = {"properties": {"name": {}, "pos": {}, "mounts": {}}, "required": ["name"]}
    tidied = tool_input.tidy("update_robot", schema, {"name": "go2", "pos": [1, 0], "mounts": []})
    assert "mounts" not in tidied
    alone = tool_input.tidy("update_robot", schema, {"name": "go2", "mounts": []})
    assert alone["mounts"] == []  # said alone, it means it: take them off


@pytest.mark.anyio
async def test_mcp_add_robot_mounts_an_arm_and_works_it(pg):
    server = build_server(pg, find_free_port(8795))
    async with Client(server) as client:

        async def call(tool: str, args: dict) -> dict:
            result = await client.call_tool(tool, args)
            assert not result.is_error, result.content[0].text
            return json.loads(result.content[0].text)

        added = await call("add_robot", {"name": "go2", "library": "go2", "mounts": [{"library": "piper"}]})
        assert added["robots"] == ["go2"]
        assert (await call("robots", {}))["robots"][0]["arms"] == ["go2/arm"]
        moved = await call("move_arm", {"robot": "go2/arm", "position": [0.4, 0.0, 0.4]})
        assert np.linalg.norm(np.subtract(moved["tool_position"], [0.4, 0.0, 0.4])) < 0.02


# ---- real downloads (opt-in, see test_library.py) ----------------------------------------------------------------


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_a_go2_with_a_piper_walks_to_a_cube_and_bins_it(pg):
    test_a_dog_walks_to_a_cube_and_its_arm_puts_it_in_a_bin(pg)


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_a_ur5e_with_a_robotiq_gripper_picks_a_cube(pg):
    pg.add_robot("ur", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.15])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.08], pos=[0.35, -0.25], color="blue")
    _run(pg, 1.0)
    _, info = pg.behaviors.pick_and_place("ur", "cube", "bin")
    _arm_done(pg, "ur")
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]
