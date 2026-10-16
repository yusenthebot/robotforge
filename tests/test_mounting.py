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
from robotforge.behaviors.manager import CARRIER_WAIT, CARRIER_WAIT_SWAYING
from robotforge.cli import find_free_port
from robotforge.design import library, mounting
from robotforge.design.templates.base import BuiltRobot
from robotforge.design.world import MountEntry, World, compile_world
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim.geometry import highest_point, lowest_point
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


@pytest.mark.parametrize("humanoid, mount", [("op3", "so_arm100"), ("g1", "piper")])
def test_on_a_humanoid_top_is_its_upper_body_not_inside_its_torso(humanoid, mount):
    """Regression: on top of a humanoid's main body (its pelvis) is inside its torso, and a mount there was refused
    (G1 with a PiPER, T1 with an SO-100). It goes on what is on top: the head, or the shoulders of a head narrower
    than its base."""
    built = mounting.attach(library.build(humanoid), [MountEntry.model_validate({"library": mount})], "bot")
    m = built.spec.compile()  # an OP3 carries no SO-100 in a world (too heavy), but where it would go still matters
    d = mounting._posed(m, built.home_qpos)
    base = next(b for b in range(m.nbody) if m.body(b).name.startswith("arm/"))
    pelvis = 1
    assert m.body_parentid[base] != pelvis and d.xpos[base][2] > d.xpos[pelvis][2] + 0.1


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
    stop = pg.behaviors.stop_carrier("go2/arm")
    assert stop is not None and not stop.still() and stop.seconds == CARRIER_WAIT
    with pytest.raises(ValueError, match=r"go2 is still coming to a stand \(0\.\d\d m/s\); give it a second"):
        pg.behaviors.pick_and_place("go2/arm", "cube", [1.05, -0.3])
    pg.run_fast(stop.still, stop.seconds)
    assert stop.still() and pg.behaviors.stop_carrier("go2/arm") is None  # standing: nothing to wait for
    x, y, _ = pg.behaviors.pose("go2")["position"]
    pg.behaviors.move_arm("go2/arm", [x + 0.3, y, 0.45])


def test_an_arm_on_a_robot_that_keeps_stepping_is_refused_at_once(pg):
    """Regression: a G1 on its learned walk steps in place (0.2 to 0.35 m/s every couple of seconds), so after its
    first move each call waited 3 s for it to stand still and failed with "give it a second": the arm on it could never
    work again. It is refused at once and says where an arm does work, and nothing takes hold of the arm."""
    pg.add_robot("g1", library="g1", mounts=[{"library": "piper"}])
    stepping = r"g1 keeps stepping in place on its learned walk and never stands still, so g1/arm cannot work from it"
    for _ in range(2):
        with pytest.raises(ValueError, match=stepping + r"; .*a robot dog \(go2, spot, anymal_c\), a rover or a T1"):
            pg.behaviors.move_arm("g1/arm", [0.3, 0.0, 1.2])
        with pytest.raises(ValueError, match=stepping):
            pg.behaviors.stop_carrier("g1/arm")
        _run(pg, 1.0)
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.0])
    with pytest.raises(ValueError, match=stepping):
        pg.behaviors.pick_and_place("g1/arm", "cube", [0.4, 0.3])
    assert "g1/arm" not in pg.session.controllers


@pytest.mark.anyio
async def test_an_arm_on_a_humanoid_waits_longer_for_it_to_stand(pg):
    """A humanoid sways for a while after it stops (a T1 with an SO-100: 2.5 to 4.1 s), longer than the 3 s the tools
    waited for a robot dog, so the first move_arm after a walk was always refused."""
    pg.add_robot("h", template="humanoid", mounts=[{"template": "arm", "params": {"reach": 0.2}}])
    _run(pg, 1.0)
    pg.behaviors.move("h", 0.1, 0.0)
    _run(pg, 2.0)
    stop = pg.behaviors.stop_carrier("h/arm")
    assert stop is not None and stop.robot == "h" and stop.seconds == CARRIER_WAIT_SWAYING > CARRIER_WAIT
    server = build_server(pg, find_free_port(8796))
    pg.behaviors.move("h", 0.1, 0.0)
    _run(pg, 2.0)
    x, y, _ = pg.behaviors.pose("h")["position"]
    async with Client(server) as client:
        result = await client.call_tool("move_arm", {"robot": "h/arm", "position": [x + 0.13, y, 0.4]})
    assert not result.is_error, result.content[0].text
    assert json.loads(result.content[0].text)["carrier_stopped"].startswith("h stopped first and stood still after")
    assert pg.behaviors.pose("h")["upright"]


def test_a_carried_arm_takes_a_whole_turn_joint_the_near_way_round(pg):
    """Regression: on ANYmal's back an IK answer with the UR5e's base one turn (+-360 degrees allows it) from where it
    was swung the base through 376 degrees and the carry about it one and a half turns, until the joint hit its limit
    and flung the cube 3 m. The same pose is taken with the joint turned the near way."""
    pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "ur5e"}])
    arm = pg.behaviors._arm("anymal/arm")
    far = arm.q.copy()
    far[0] += 2 * np.pi + 0.29
    assert arm._unwound(far)[0] == pytest.approx(arm.q[0] + 0.29)
    assert arm._unwound(far)[1:] == pytest.approx(far[1:])


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


def test_an_arm_on_a_dog_carries_things_over_the_dogs_back(pg):
    """Regression: the carry height came from the bin's rim alone, and swinging round its base from one side of a Go2
    to the other, a PiPER swept the cube through the dog's head at 15 cm and lost it. Here the carry passes over the
    stand-in's hips."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.0, 0.3])
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[0.0, -0.35])
    _run(pg, 1.0)
    m, d = pg.session.model, pg.session.data
    hips = max(highest_point(m, d, g)[2] for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("_hip"))
    arm, placed = pg.behaviors.pick_and_place("go2/arm", "cube", "bin")
    carry = next(wp for wp in arm.queue if wp.label == "carry")
    assert carry.position[2] >= hips + 0.04 + 0.03 - 1e-3  # the cube's bottom clears the dog by 3 cm
    cube = m.body("cube").id
    dog = {b for b in range(m.nbody) if m.body(b).name.startswith("go2/") and "/arm/" not in m.body(b).name}
    touched = set()
    for _ in range(160):
        _run(pg, 0.1)
        for c in d.contact[: d.ncon]:
            bodies = {int(m.geom_bodyid[g]) for g in c.geom}
            if cube in bodies and bodies & dog:
                touched |= bodies & dog
        if arm.done:
            break
    _run(pg, 0.5)
    assert not touched
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_an_arm_on_a_dog_turns_its_gripper_down_gently(pg):
    """Regression: on a carrier the gripper snapped from its home pose to pointing down on the first ticks of a move
    (the gradual turn was for arms on the floor only); the jolt pushed a Go2 carrying a Z1 25 mm back, it took that
    for a shove and stepped off half a metre."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    _run(pg, 2.0)
    d = pg.session.data
    x, y, _ = pg.behaviors.pose("go2")["position"]
    arm = pg.behaviors.move_arm("go2/arm", [x + 0.1, y + 0.3, 0.25])

    def from_down() -> float:
        axis = d.xmat[arm.chain.body].reshape(3, 3) @ arm.tool_axis
        return float(np.degrees(np.arccos(np.clip(-axis[2], -1.0, 1.0))))

    start = from_down()
    _run(pg, 0.1)  # of a move of about 3.6 s
    assert start > 45 and abs(from_down() - start) < 1.5  # snapping, it had turned 4 degrees already
    _arm_done(pg, "go2/arm")
    assert from_down() < 3


def test_a_robot_imported_from_a_file_carries_a_library_arm(pg):
    pg.add_robot(
        "dog", file=str(FIXTURES / "unitree_go2" / "go2.xml"), pos=[0, 0, 0.5], mounts=[{"library": "so_arm100"}]
    )
    assert pg.behaviors.arms("dog") == ["dog/arm"]
    m = pg.session.model
    assert m.body_parentid[m.body("dog/arm/Base").id] == m.body("dog/base").id


def test_where_mounts_go_is_worked_out_once_per_robot_and_again_when_it_changes(pg, tmp_path, monkeypatch):
    """Regression: placing mounts (rays, several compiles) ran again on every world edit, so with a PiPER on its back a
    Go2 took 0.19 s per add_object instead of 0.06. It runs again when the mounts or the robot's file change."""
    looked = []
    for step in ("_place", "_check_overlaps"):
        original = getattr(mounting, step)
        monkeypatch.setattr(mounting, step, lambda *a, _o=original, _s=step, **k: looked.append(_s) or _o(*a, **k))
    mounting.PLACEMENTS.clear()
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    assert looked == ["_place", "_check_overlaps"]
    placed = pg.session.model.body("go2/arm/base_link").pos.copy()
    pg.add_object("box", shape="box", size=[0.05] * 3, pos=[1.0, 1.0])
    pg.add_object("ball", shape="sphere", size=[0.05], pos=[1.0, -1.0])
    assert looked == ["_place", "_check_overlaps"]  # the same place, not looked for again
    np.testing.assert_allclose(pg.session.model.body("go2/arm/base_link").pos, placed)
    pg.update_robot("go2", mounts=[{"library": "piper", "shift": [-0.05, 0, 0]}])
    assert len(looked) == 4
    source = (FIXTURES / "unitree_go2" / "go2.xml").read_text(encoding="utf-8")
    copy = tmp_path / "go2_copy" / "go2.xml"
    copy.parent.mkdir()
    for asset in (FIXTURES / "unitree_go2").iterdir():
        (copy.parent / asset.name).write_bytes(asset.read_bytes())
    pg.add_robot("dog", file=str(copy), pos=[2, 0, 0.5], mounts=[{"library": "so_arm100"}])
    seen = len(looked)
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[2.0, 1.0])
    assert len(looked) == seen
    copy.write_text(source.replace("<mujoco", "<mujoco ", 1), encoding="utf-8")  # the file changed
    os.utime(copy, ns=(copy.stat().st_atime_ns, copy.stat().st_mtime_ns + 10**9))
    pg.add_object("cube2", shape="box", size=[0.04] * 3, pos=[2.0, -1.0])
    assert len(looked) == seen + 2


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


def test_a_mounted_robotiq_starts_open(pg):
    """Regression: the 2F-85 has no keyframe, so its control started at 0, which after the turn-around closes it: a
    UR5e came up with its fingers shut."""
    pg.add_robot("ur", library="ur5e", mounts=[{"library": "robotiq_2f85"}])
    _run(pg, 1.0)
    m = pg.session.model
    gripper = m.actuator("ur/gripper").id
    assert pg.session.data.ctrl[gripper] == m.actuator_ctrlrange[gripper][1]  # the top of the range: open
    driver = m.joint("ur/gripper/right_driver_joint")
    assert pg.session.data.qpos[driver.qposadr[0]] < 0.25 * driver.range[1]  # shut, it is at 0.78 of 0.8


@pytest.mark.parametrize("source", [{"library": "robotiq_2f85"}, {"template": "gripper"}])
def test_a_gripper_on_its_own_says_to_mount_it_on_an_arm(pg, source):
    pg.add_robot("hand", **source)
    assert pg.behaviors.robots()[0]["kind"] == "gripper"
    for call in (
        lambda: pg.behaviors.move_arm("hand", [0.3, 0.0, 0.3]),
        lambda: pg.behaviors.set_gripper("hand", "close"),
    ):
        with pytest.raises(ValueError, match="hand is a gripper on its own .*mount it on an arm's bare flange"):
            call()


def test_an_arm_without_a_gripper_will_not_pick_and_says_how_to_mount_one(pg):
    pg.add_robot("ur", library="ur5e")
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.15])
    with pytest.raises(ValueError, match='has no gripper to pick with; mount one .*"robotiq_2f85"'):
        pg.behaviors.pick_and_place("ur", "cube", [0.3, -0.2])


@pytest.mark.parametrize("carrier", ["panda", "go2"])
def test_a_gripper_goes_only_on_a_bare_flange_unless_told_where(pg, carrier):
    """Regression: with no flange to go on, a gripper went on top: of a Panda's base, beside its own hand, or on a dog's
    back, where it is a dead weight."""
    bare = f"goes on the bare flange of an arm without a gripper, such as a ur5e, and {carrier} has none; carry one"
    with pytest.raises(ValueError, match=bare):
        pg.add_robot(carrier, library=carrier, mounts=[{"library": "robotiq_2f85"}])
    pg.add_robot(carrier, library=carrier, mounts=[{"library": "robotiq_2f85", "at": "top"}])  # said so: it goes


def test_at_tool_needs_an_arm_with_a_bare_flange(pg):
    with pytest.raises(ValueError, match="at 'tool' needs an arm with a bare flange"):
        pg.add_robot("go2", library="go2", mounts=[{"library": "robotiq_2f85", "at": "tool"}])


# ---- a gripper on an arm a robot carries ----------------------------------------------------------------------------


def test_a_dog_carries_an_arm_and_the_gripper_on_its_flange_picks_a_cube(pg):
    """Regression: a gripper could go only on the flange of an arm added on its own. Spot with a UR5e and a Robotiq was
    refused ("mount it on an arm such as ur5e"), and at the wrist link the gripper was a robot of its own, listed as an
    arm, while pick_and_place on the UR5e asked for the gripper just refused."""
    result = pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85"}])
    assert "anymal: gripper: Robotiq 2F-85 on anymal/arm's flange; it is now anymal/arm's gripper" in result["notes"]
    assert pg.behaviors.robots()[0]["arms"] == ["anymal/arm"]
    m = pg.session.model
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "anymal/arm/gripper") >= 0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "anymal/arm/flange") < 0  # taken by the gripper
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.0, 0.65])
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[0.0, -0.65])
    _run(pg, 1.5)
    arm, info = pg.behaviors.pick_and_place("anymal/arm", "cube", "bin")
    _arm_done(pg, "anymal/arm")
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]
    assert pg.behaviors.pose("anymal")["upright"]


def test_a_gripper_on_a_carried_arm_can_be_named_and_wrong_names_say_which_flanges_there_are(pg):
    mounts = [{"library": "ur5e"}, {"library": "robotiq_2f85", "at": "arm/tool"}]
    pg.add_robot("anymal", library="anymal_c", mounts=mounts)
    assert pg.behaviors.robots()[0]["arms"] == ["anymal/arm"]
    with pytest.raises(
        ValueError, match=r"at 'claw/tool' needs an arm with a bare flange on 'claw' .*give at 'arm/tool'"
    ):
        pg.update_robot("anymal", mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85", "at": "claw/tool"}])


def test_a_gripper_mounted_off_a_flange_is_no_arm(pg):
    """A gripper on a dog's body has no arm to move it: it is not listed among the arms, nor worked as one."""
    mounts = [{"library": "piper"}, {"library": "robotiq_2f85", "at": "base", "shift": [-0.25, 0.0, 0.1]}]
    pg.add_robot("anymal", library="anymal_c", mounts=mounts)
    assert pg.behaviors.robots()[0]["arms"] == ["anymal/arm"]
    with pytest.raises(ValueError, match="'anymal/gripper' is not an arm of anymal; use anymal/arm"):
        pg.behaviors.move_arm("anymal/gripper", [0.3, 0.0, 0.5])


def test_an_arm_given_our_gripper_template_compiles_and_picks(pg):
    """Regression: the gripper template measures its grasp point with a sensor, which still named the old site after
    the gripper became the arm's: "unrecognized name 'gripper/tcp' of sensorized object"."""
    pg.add_robot("ur", library="ur5e", mounts=[{"template": "gripper"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.4, 0.1])
    _run(pg, 0.5)
    arm, info = pg.behaviors.pick_and_place("ur", "cube", [0.35, -0.25])
    _arm_done(pg, "ur")
    _run(pg, 1.0)
    assert pg.behaviors.placed("cube", info["region"])["success"]


def test_an_arm_on_a_dog_without_a_gripper_says_how_to_give_it_one(pg):
    pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "ur5e"}])
    pg.add_object("cube", shape="box", size=[0.04] * 3, pos=[0.0, 0.65])
    expected = 'update_robot("anymal", mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85"}])'
    with pytest.raises(
        ValueError, match=re.escape(f"anymal/arm has no gripper to pick with; mount one on its flange: {expected}")
    ):
        pg.behaviors.pick_and_place("anymal/arm", "cube", [0.0, -0.65])


def test_a_robot_that_lost_its_arm_says_mounts_puts_one_back(pg):
    """Regression: after update_robot(mounts=[]), move_arm on its arm pointed to the builder (add_robot with build)."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}])
    pg.update_robot("go2", mounts=[])
    with pytest.raises(
        ValueError, match=re.escape('go2 carries no arm (update_robot("go2", mounts=[{"library": "piper"}])')
    ):
        pg.behaviors.move_arm("go2/arm", [0.3, 0.0, 0.4])


# ---- how much a robot carries ---------------------------------------------------------------------------------------


def test_a_dog_refuses_an_arm_heavier_than_it_carries_and_names_lighter_ones(pg):
    """Regression: a Go2 took a 21 kg UR5e and had fallen over before it was told to walk."""
    heavy = (
        r"Universal Robots UR5e weighs 21\.0 kg, and a Unitree Go2 carries at most 12\.1 kg \(up to 4\.7 kg at full "
        r"speed, 12\.1 kg at 0\.1 m/s\): try xarm7 \(12\.07 kg\), z1 \(4\.69 kg\), piper \(2\.37 kg\)"
    )
    with pytest.raises(ValueError, match=heavy):
        pg.add_robot("go2", library="go2", mounts=[{"library": "ur5e"}])
    with pytest.raises(ValueError, match="SO-ARM100 weighs 0.8 kg, and a Robotis OP3 carries at most 0.1 kg; no arm"):
        pg.add_robot("op3", library="op3", mounts=[{"library": "so_arm100"}])
    assert pg.current_world().robots == []


def test_a_dog_carrying_a_heavy_arm_walks_slower_and_says_so(pg):
    """Regression: robots kept offering a carrier's unloaded top speed, and a Go2 told to walk 0.3 m/s with a Panda
    on its back crawled 7 cm in 6 s."""
    result = pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "ur5e"}])
    assert any("carrying 21.0 kg, more than the 12.1 kg it walks with at full speed" in n for n in result["notes"])
    assert pg.behaviors.robots()[0]["max_speed"] == pytest.approx(0.2)
    moved = pg.behaviors.move("anymal", 0.5, 0.0)
    assert moved["notes"] == ["speed 0.5 m/s is above max_speed 0.2, so it moves at 0.2"]
    pg.update_robot("anymal", mounts=[{"library": "piper"}])
    assert pg.behaviors.robots()[0]["max_speed"] > 0.5  # a PiPER it carries at full speed


def test_a_g1_carrying_more_than_it_walks_with_at_full_speed_says_its_lower_top_speed(pg):
    """robots read a learned walker's top speed from its policy's card, so it offered 1 m/s whatever the G1 carried."""
    mounts = [{"library": "piper"}, {"library": "robotiq_2f85", "at": "pelvis", "shift": [0.25, 0, 0]}]
    pg.add_robot("g1", library="g1", mounts=mounts)  # 3.4 kg: more than the PiPER it walks with at full speed
    assert pg.behaviors.robots()[0]["max_speed"] == pytest.approx(0.3)


def test_the_library_lists_what_robots_weigh_and_carry():
    rows = {r["name"]: r for r in library.catalog()}
    assert rows["piper"]["mass_kg"] == 2.37 and "carries" not in rows["piper"]
    assert rows["go2"]["carries"] == "up to 4.7 kg at full speed, 12.1 kg at 0.1 m/s"
    assert rows["op3"]["carries"] == "up to 0.1 kg"


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


def test_two_arms_on_top_of_a_dog_stand_apart(pg):
    """Regression: two mounts on top landed on the same spot, 75 mm inside each other, and the refusal said to use
    'top', which they had."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}, {"name": "arm2", "library": "so_arm100"}])
    assert pg.behaviors.robots()[0]["arms"] == ["go2/arm", "go2/arm2"]
    m, d = pg.session.model, pg.session.data
    mujoco.mj_forward(m, d)
    apart = np.linalg.norm(d.xpos[m.body("go2/arm/base_link").id][:2] - d.xpos[m.body("go2/arm2/Base").id][:2])
    assert apart > 0.08


def test_a_second_arm_is_moved_clear_of_the_first_or_told_where_it_fits(pg):
    """Two arms reach out from their bases, so on a Go2 the second's arm ran into the first's whatever spot its base
    took. Put on top without being told exactly, it is moved clear along or across the dog; told to stand on top as it
    is, the refusal gives a shift that was tried, clears, and keeps its base on the dog."""
    arm = {"template": "arm", "params": {"reach": 0.3}}
    result = pg.add_robot("go2", library="go2", mounts=[arm, {**arm, "name": "arm2"}])
    assert "go2: arm2: moved 18 cm back to clear arm" in result["notes"]
    assert pg.behaviors.robots()[0]["arms"] == ["go2/arm", "go2/arm2"]
    told = [arm, {**arm, "name": "arm2", "at": "top"}]
    with pytest.raises(
        ValueError, match=r"mounts 'arm' and 'arm2' overlap by \d+ mm at the home pose; shift \["
    ) as hint:
        pg.update_robot("go2", mounts=told)
    shift = [
        float(v) for v in re.search(r"shift \[([-\d., ]+)\] moves 'arm2' clear", str(hint.value)).group(1).split(",")
    ]
    pg.update_robot("go2", mounts=[told[0], {**told[1], "shift": shift}])
    assert pg.behaviors.robots()[0]["arms"] == ["go2/arm", "go2/arm2"]


def test_two_arms_with_no_room_for_both_are_told_so(pg):
    with pytest.raises(ValueError, match="and there is no room for both side by side on top of anymal; mount 'arm2'"):
        pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "piper"}, {"name": "arm2", "library": "piper"}])


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
    front = trunk.T @ d.xmat[base].reshape(3, 3) @ [0, -1, 0]  # the SO-100's model faces its -y
    np.testing.assert_allclose(front, [0, 1, 0], atol=1e-6)  # turned 90 degrees to the left of the dog's front
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


def test_update_robot_says_which_mount_is_wrong(pg):
    """Regression: add_robot said "mounts[0]: ..." and update_robot only " must be an object"."""
    pg.add_robot("go2", library="go2")
    with pytest.raises(ValueError, match=r"^mounts\[0\] must be an object, got 'piper'"):
        pg.update_robot("go2", mounts=["piper"])
    with pytest.raises(ValueError, match=r"^mounts\[1\]: a mount's shift is \[x, y, z\]"):
        pg.update_robot("go2", mounts=[{"library": "piper"}, {"library": "z1", "shift": [1]}])
    assert pg.current_world().robots[0].mounts == []  # nothing half-changed


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


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_anymal_carrying_a_ur5e_with_a_robotiq_picks_across_its_back(pg):
    test_a_dog_carries_an_arm_and_the_gripper_on_its_flange_picks_a_cube(pg)


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
@pytest.mark.anyio
async def test_real_a_t1_carrying_an_so100_stops_without_falling_and_its_arm_moves(pg):
    """Regression: told to stop after a walk, a T1 with an SO-100 on its head swayed further and further sideways and
    fell over (its roll was damped as for a 0.2 m leg); and it sways for 3 to 4 s, so the tools gave up waiting."""
    pg.add_robot("t1", library="t1", mounts=[{"library": "so_arm100"}])
    _run(pg, 2.0)
    pg.behaviors.move("t1", 0.12, 0.0)
    _run(pg, 3.0)
    x, y, _ = pg.behaviors.pose("t1")["position"]
    server = build_server(pg, find_free_port(8795))
    async with Client(server) as client:
        result = await client.call_tool("move_arm", {"robot": "t1/arm", "position": [x + 0.25, y, 1.1]})
    assert not result.is_error, result.content[0].text
    assert "carrier_stopped" in json.loads(result.content[0].text)
    _run(pg, 3.0)
    assert pg.behaviors.pose("t1")["upright"]


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_a_t1_carrying_a_piper_walks_at_its_lower_top_speed_without_falling(pg):
    """Regression: robots offered a T1 with a PiPER its unloaded 0.12 m/s, at which it fell backwards within 4 s."""
    pg.add_robot("t1", library="t1", mounts=[{"library": "piper"}])
    assert pg.behaviors.robots()[0]["max_speed"] == pytest.approx(0.06)
    _run(pg, 2.0)
    start = pg.behaviors.pose("t1")["position"]
    pg.behaviors.move("t1", 0.12, 0.0)
    _run(pg, 10.0)
    end = pg.behaviors.pose("t1")
    assert end["upright"] and end["position"][0] - start[0] > 0.4


@pytest.mark.skipif(not NETWORK, reason="set ROBOTFORGE_NETWORK_TESTS=1 to download real robots from Menagerie")
def test_real_a_go2_carries_two_arms_side_by_side_and_walks(pg):
    result = pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}, {"name": "arm2", "library": "so_arm100"}])
    assert any(n.startswith("go2: arm2: moved") for n in result["notes"])
    _run(pg, 1.0)
    pg.behaviors.move("go2", 0.3, 0.0)
    _run(pg, 6.0)
    assert pg.behaviors.pose("go2")["upright"] and pg.behaviors.pose("go2")["position"][0] > 1.5
