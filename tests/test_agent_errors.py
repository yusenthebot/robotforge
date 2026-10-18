"""What an agent sees when a call is a little wrong.

A call with one plausible reading is accepted ("Ball" for a sphere, pos as "2, 0", a robot called "the dog");
any other mistake gets a sentence that names the argument and value and says what would have worked. Nothing
shows pydantic's report, a traceback or the SDK's "Error executing tool" prefix.
scripts/agent_errors.py runs a larger battery of these and grades it; this file pins the messages that matter.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from robotforge import tool_input, wording
from robotforge.mcp_server import _tool_errors, build_server
from robotforge.playground import Playground
from robotforge.sim.session import SimSession

pytestmark = pytest.mark.anyio

# What no answer to an agent may contain.
NEVER = re.compile(
    r"errors\.pydantic\.dev|validation error for|\[type=|Input should|Field required|Extra inputs|Traceback"
    r"|Error executing tool|\b(?:robots|objects|terrain)\.\d+"
)


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


class Lab:
    """The tools of a server on a world with a dog, an arm, a rover and a ball, called without a client."""

    def __init__(self, playground: Playground):
        self.playground = playground
        self.server = build_server(playground, 0)
        self.baseline = playground.store.file.read_text(encoding="utf-8")

    async def call(self, tool: str, **args) -> tuple[bool, str]:
        """(accepted, text): the result, or what the agent reads when the call is refused."""
        try:
            result = await self.server.call_tool(tool, args)
        except ToolError as exc:
            return False, str(exc)
        return True, "".join(b.text for b in result.content if getattr(b, "type", "") == "text")

    async def refused(self, tool: str, **args) -> str:
        accepted, text = await self.call(tool, **args)
        assert not accepted, f"{tool}({args}) was accepted: {text[:200]}"
        assert not NEVER.search(text), text
        return text

    async def accepted(self, tool: str, **args) -> str:
        accepted, text = await self.call(tool, **args)
        assert accepted, f"{tool}({args}) was refused: {text}"
        return text

    async def restore(self) -> None:
        """Back to the starting world after a call that changed it."""
        self.playground.store.file.write_text(self.baseline, encoding="utf-8")
        self.playground.reload_world()
        self.playground.session.paused = False


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    patch.setenv("ROBOTFORGE_HOME", str(tmp_path_factory.mktemp("home")))
    try:
        playground = Playground(session=SimSession.from_xml("<mujoco/>"))
        playground.new_world("lab")
        playground.add_robot("dog", template="quadruped", pos=[0.0, 1.2])
        playground.add_robot("arm", template="arm", pos=[0.0, 0.0])
        playground.add_robot("rover", template="rover", pos=[0.0, -1.0])
        playground.add_object("ball", shape="sphere", size=[0.04], pos=[0.25, -0.12])
        yield Lab(playground)
    finally:
        patch.undo()


# ---- a mistake with one reading is accepted --------------------------------------------------------


async def test_names_in_other_capitals_or_with_an_article_find_the_robot(lab):
    for spelled in ("Dog", "the dog", "DOG"):
        out = json.loads(await lab.accepted("move", robot=spelled, speed=0.2, seconds=0.5))
        assert out["from"]["robot"] == "dog"
    await lab.restore()


async def test_a_name_with_capitals_and_spaces_is_saved_tidy_and_found_again(lab):
    out = json.loads(await lab.accepted("add_object", name="Red Cube", pos="0.4, 0.1", shape="Cube"))
    assert "red_cube" in out["objects"] and any("saved as 'red_cube'" in n for n in out["notes"])
    await lab.accepted("update_object", name="red cube", color="#9c6b3f")
    await lab.accepted("remove", name="Red_Cube")
    await lab.restore()


@pytest.mark.parametrize(
    ("tool", "args", "yaml_has"),
    [
        ("add_object", {"name": "b2", "shape": "ball", "size": 0.05, "pos": [0.4, 0.1]}, "name: b2\n  shape: sphere"),
        ("add_object", {"name": "b2", "size": "0.05, 0.05, 0.05", "pos": {"x": 0.4, "y": 0.1}}, "name: b2"),
        ("add_object", {"name": "b2", "color": [255, 128, 0], "pos": [0.4, 0.1], "fixed": "yes"}, "fixed: true"),
        (
            "add_robot",
            {"name": "pup", "template": "Quadruped", "pos": "2, 0", "params": "{'leg_length': 0.3}"},
            "name: pup\n  template: quadruped",
        ),
        ("add_robot", {"name": "pup", "template": "arms", "pos": [2, 0]}, "name: pup\n  template: arm"),
        (
            "add_robot",
            {"name": "pup", "template": "quadruped", "library": "", "build": {}, "pos": [2, 0]},
            "name: pup\n  template: quadruped",
        ),
        ("add_robot", {"name": "pup", "build": {}, "pos": [2, 0]}, "name: pup"),  # build alone: a bare box
        ("add_terrain", {"type": "slope", "pos": "2, 0"}, "type: ramp"),
        ("set_floor", {"style": "Concrete"}, "style: concrete"),
    ],
)
async def test_forms_with_one_meaning_are_accepted(lab, tool, args, yaml_has):
    await lab.accepted(tool, **args)
    assert yaml_has in lab.playground.show_world()["world_yaml"]
    await lab.restore()


async def test_a_call_that_leaves_fields_null_or_empty_gets_their_defaults(lab):
    await lab.accepted("move", robot="dog", speed=None, turn=None, seconds=0.2, walker="")
    await lab.accepted("observe", view="", include_image=False)
    await lab.accepted("observe", include_image="null", width="None")  # the text null is a null, too
    await lab.accepted("add_object", name="b2", shape=None, size=[], pos=[0.4, 0.1], color="")
    await lab.restore()


async def test_actuators_resolve_without_the_robot_prefix_or_in_other_capitals(lab):
    for name in ("shoulder", "ARM/Shoulder"):
        out = json.loads(await lab.accepted("set_controls", controls={name: 0.3}))
        assert out["applied"] == {"arm/shoulder": 0.3}
    await lab.restore()


async def test_arm_and_pose_arguments_in_other_forms(lab):
    await lab.accepted("move_arm", robot="Arm", position={"x": 0.2, "y": 0.0, "z": 0.15}, gripper="Open")
    await lab.accepted("move_arm", robot="arm", position="0.2, 0.0, 0.15")
    await lab.accepted("gripper", robot="arm", state="grab")
    placed = json.loads(await lab.accepted("pick_and_place", robot="arm", object="Ball", target="0.2, 0.2"))
    assert placed["success"] and placed["object"] == "ball"
    await lab.restore()


async def test_tool_names_in_other_forms_find_the_tool(lab):
    await lab.accepted("Robots")
    await lab.accepted("robotforge.robots")
    await lab.accepted("sim_control", action=" Resume ")


async def test_a_yaw_that_looks_like_radians_and_a_one_number_box_are_explained(lab):
    robot = json.loads(await lab.accepted("add_robot", name="pup", template="rover", pos=[2, 0], yaw=1.57))
    assert any("yaw 1.57 is degrees" in n and "use 90" in n for n in robot["notes"])
    cube = json.loads(await lab.accepted("add_object", name="b2", shape="box", size=[0.1], pos=[0.4, 0.1]))
    assert any("cube of 0.1 m" in n for n in cube["notes"])
    assert "size:\n  - 0.1\n  - 0.1\n  - 0.1" in lab.playground.show_world()["world_yaml"]
    quiet = json.loads(await lab.accepted("update_object", name="b2", yaw=90))
    assert not any("yaw" in n for n in quiet["notes"])
    await lab.restore()


async def test_a_value_the_robot_cannot_do_is_clamped_and_said(lab):
    fast = json.loads(await lab.accepted("move", robot="dog", speed=50, seconds=0.2))
    assert "max_speed" in fast["behavior"]["notes"][0] and "moves at" in fast["behavior"]["notes"][0]
    turned = json.loads(await lab.accepted("move", robot="rover", turn=90, seconds=0.2))
    assert "rad/s" in turned["behavior"]["notes"][0] and "1.57" in turned["behavior"]["notes"][0]
    degrees = json.loads(await lab.accepted("set_controls", controls={"arm/shoulder": 90}))
    assert degrees["applied"] == {"arm/shoulder": 1.9} and "90 degrees is 1.57" in degrees["notes"][0]
    await lab.restore()


# ---- every other mistake is refused with what to do ------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args", "says"),
    [
        ("new_world", {"name": "w", "floor": "lava"}, ["floor", "'lava'", "not one of", "checker", "tile"]),
        ("new_world", {}, ["name is required", "new_world takes name", "optionally floor"]),
        (
            "new_world",
            {"world_name": "w"},
            ["no parameter 'world_name'", "did you mean name", "parameters: name, floor"],
        ),
        ("list_templates", {"kind": "arm"}, ["takes no parameters", "'kind'"]),
        (
            "add_robot",
            {"name": "p", "template": "", "library": "", "build": {}},
            ["give exactly one of", "'template'", "'build'"],
        ),
        ("add_robot", {"name": "p", "template": "dog"}, ["unknown template 'dog'", "Did you mean quadruped", "rover"]),
        (
            "add_robot",
            {"name": "p", "template": "rover", "params": {"wheel_radius": 3}},
            ["wheel_radius must be at most 0.3", "3 cm is 0.03"],
        ),
        (
            "add_robot",
            {"name": "p", "template": "quadruped", "params": {"colour": "red"}},
            ["'colour' is not a known field", "did you mean color"],
        ),
        (
            "add_robot",
            {"name": "p", "template": "quadruped", "leg_length": 0.3},
            ["no parameter 'leg_length'", "go in params"],
        ),
        ("add_robot", {"name": "1pup", "template": "rover"}, ["'1pup'", "lowercase", "start with a letter"]),
        ("add_robot", {"name": "dog", "template": "rover"}, ["'dog' already exists", "unique", "update_robot"]),
        ("add_robot", {"name": "p", "library": "go3"}, ["no robot 'go3' in the library", "Did you mean go2"]),
        ("add_robot", {"name": "p", "template": "rover", "pos": [150, 80]}, ["positions are meters", "[1.5, 0.8]"]),
        ("add_object", {"name": "b", "shape": "pyramid"}, ["shape", "'pyramid'", "box, sphere", "marker"]),
        (
            "add_object",
            {"name": "b", "pos": "centre"},
            ["pos must be a list [x, y] or [x, y, z] in meters", "'centre'"],
        ),
        ("add_object", {"name": "b", "size": {"l": 1}}, ["size must be a list of numbers in meters"]),
        (
            "add_object",
            {"name": "b", "shape": "table", "size": [120, 60, 75]},
            ["sizes are meters", "[1.2, 0.6, 0.75]"],
        ),
        ("add_object", {"name": "b", "mass": 0}, ["mass must be above 0"]),
        ("add_object", {"name": "b", "radius": 1}, ["no parameter 'radius'", "parameters: name, shape, size"]),
        ("update_object", {"name": "ball"}, ["nothing to change", "pos"]),
        ("update_object", {"name": "cube", "pos": [0, 0]}, ["no object named 'cube'", "Objects in this world: ball"]),
        (
            "update_robot",
            {"name": "dog1", "pos": [0, 0]},
            ["no robot named 'dog1'", "Did you mean dog", "dog, arm, rover"],
        ),
        (
            "update_robot",
            {"name": "dog", "params": {"leg_length": 2}},
            ["robot 'dog'", "leg_length must be at most 0.8"],
        ),
        ("remove", {"name": "table"}, ["no robot, object or terrain patch named 'table'", "dog, arm, rover, ball"]),
        ("load_scene", {"preset": "garage"}, ["preset", "'garage'", "desk_lab", "obstacle_course"]),
        (
            "set_terrain",
            {"type": "stairs", "params": {"step_height": 15}},
            ["step_height must be at most 0.5", "15 cm is 0.15"],
        ),
        ("add_terrain", {"type": "ramp", "pos": [2]}, ["ramp pos", "[x, y] in meters", "[2.0]"]),
        ("move", {"robot": "cat"}, ["no robot named 'cat'", "Robots in this world: dog, arm, rover"]),
        ("move", {"robot": "dog", "speed": "0.2 m/s"}, ["speed must be a plain number (no units)", "'0.2 m/s'"]),
        (
            "move",
            {"robot": "dog", "velocity": 0.2},
            ["no parameter 'velocity'", "did you mean speed", "robot, speed, turn"],
        ),
        ("move", {"robot": "dog", "distance": 2}, ["no parameter 'distance'", "speed (m/s) and seconds", "distance_m"]),
        (
            "move_arm",
            {"robot": "arm", "x": 0.2, "y": 0, "z": 0.1},
            ["no parameter 'x'", "position is one list [x, y, z]"],
        ),
        ("simulate", {"steps": 100}, ["no parameter 'steps'", "give seconds"]),
        ("move", {"robot": "dog", "seconds": 3000}, ["seconds must be in (0, 60]", "got 3000", "not milliseconds"]),
        ("move", {"robot": "arm"}, ["arm is an arm", "move_arm"]),
        (
            "move",
            {"robot": "dog", "walker": "sprint"},
            ["no policy named 'sprint'", "Installed: ", "; or use walker 'gait'"],
        ),
        ("stop", {"robot": "cat"}, ["no robot named 'cat'"]),
        ("pose", {"robot": "dog", "name": "wave"}, ["dog is a quadruped", "only humanoids", "move"]),
        (
            "move_arm",
            {"robot": "arm", "position": [20, 0, 15]},
            ["from the arm's base, which reaches about", "in cm that would be [0.2, 0.0, 0.15]"],
        ),
        ("move_arm", {"robot": "arm", "position": [0.2, 0.0, -0.5]}, ["below the floor"]),
        ("move_arm", {"robot": "arm", "position": [0.2, 0.0]}, ["position must be [x, y, z] in meters", "[0.2, 0.0]"]),
        (
            "move_arm",
            {"robot": "dog", "position": [0.2, 0.0, 0.1]},
            ["dog is a quadruped, not an arm", "moves with move"],
        ),
        (
            "move_arm",
            {"robot": "rover/arm", "position": [0.2, 0.0, 0.1]},
            ["'rover/arm' is not an arm", "rover carries no arm"],
        ),
        ("gripper", {"robot": "arm", "state": "half"}, ["state", "'half'", "open, close"]),
        (
            "pick_and_place",
            {"robot": "arm", "object": "cube", "target": [0.2, 0.2]},
            ["no object named 'cube'", "ball"],
        ),
        (
            "pick_and_place",
            {"robot": "arm", "object": "ball", "target": "bin"},
            ["no object named 'bin'", "or give a position [x, y]"],
        ),
        (
            "pick_and_place",
            {"robot": "arm", "object": "ball"},
            ["target is required", "pick_and_place takes robot, object, target"],
        ),
        (
            "pick_and_place",
            {"robot": "arm", "obj": "ball", "target": [0.2, 0.2]},
            ["no parameter 'obj'", "did you mean object"],
        ),
        ("set_controls", {"controls": {"sholder": 1.0}}, ["no actuator named 'sholder'", "Did you mean arm/shoulder"]),
        ("set_controls", {"controls": ["arm/shoulder", 0.3]}, ["controls must be an object like", "arm/shoulder"]),
        ("set_controls", {"controls": {"wing": 1}}, ["no actuator named 'wing'", "Names look like", "describe_model"]),
        ("simulate", {"seconds": 500}, ["seconds must be in (0, 60]", "got 500", "not milliseconds"]),
        ("simulate", {"seconds": 61}, ["seconds must be in (0, 60], got 61"]),
        ("observe", {"include_image": []}, ["include_image must be true or false", "got []"]),
        ("load_scene", {"preset": "maze", "keep_robots": {}}, ["keep_robots must be true or false", "got {}"]),
        ("update_robot", {"name": "dog", "build": {}}, ["build={} would remove every part", "leave build out"]),
        ("move", {"robot": "dog/trunk"}, ["'dog/trunk' is a part of dog, not a robot", "use dog"]),
        ("stop", {"robot": "Dog / Trunk"}, ["'dog/Trunk' is a part of dog, not a robot", "use dog"]),
        (
            "move_arm",
            {"robot": "dog/trunk", "position": [0.2, 0.0, 0.1]},
            ["'dog/trunk' is not an arm", "dog carries no arm"],
        ),
        ("update_terrain", {"name": "x", "yaw": 45}, ["no terrain patch named 'x'", "add_terrain makes one"]),
        ("remove", {"name": "floor"}, ["no robot, object or terrain patch named 'floor'", "set_floor"]),
        ("remove", {"name": "dogs"}, ["named 'dogs'", "Did you mean dog", "dog, arm, rover, ball"]),
        (
            "run_controller",
            {"robot": "arm", "task": "reach", "method": "PPO"},
            ["ppo is trained, not built on the spot", "train_policy"],
        ),
        (
            "pick_and_place",
            {"robot": "arm", "object": "ball", "target": [3.0, 3.0]},
            ["arm cannot reach the target", "it is 4.", "m from the arm's base, which reaches about", "nearer"],
        ),
        (  # within reach, but too close to point down: "nearer" would be the wrong way (review of the audit)
            "pick_and_place",
            {"robot": "arm", "object": "ball", "target": [0.02, 0.0]},
            ["arm cannot reach the target", "the gripper points down only", "move the target further from the base"],
        ),
        (
            "pick_and_place",
            {"robot": "arm", "object": "ball", "target": [30, 10]},
            ["arm cannot reach the target", "positions are meters", "in cm that would be [0.3, 0.1]"],
        ),
        ("simulate", {"seconds": 100}, ["seconds must be in (0, 60], got 100"]),
        ("sim_control", {"action": "stop"}, ["action", "'stop'", "play, pause, reset"]),
        ("observe", {"view": "sideways"}, ["unknown view 'sideways'", "Views:", "front", "overview"]),
        ("observe", {"view": "front", "width": 0}, ["image size 0x480 is out of range", "each at least 16"]),
        (
            "run_controller",
            {"robot": "arm", "task": "reach", "method": "pid"},
            ["unknown method 'pid'", "Did you mean pd"],
        ),
        ("run_controller", {"robot": "arm", "task": "grab", "method": "ik"}, ["unknown task 'grab'", "reach, balance"]),
        (
            "run_controller",
            {"robot": "arm", "task": "reach", "method": "ik", "task_params": {"target": [20, 0, 20]}},
            ["out of reach", "in cm that would be [0.2, 0.0, 0.2]"],
        ),
        (
            "run_controller",
            {"robot": "arm", "task": "reach", "method": "ik", "task_params": {"goal": 1}},
            ["'goal' is not a known field", "target, tolerance"],
        ),
        ("compare_controllers", {"robot": "arm", "task": "reach", "methods": ["ik"] * 7}, ["1 to 6 names, got 7"]),
        ("train_policy", {"task": "walk", "robot": "cat"}, ["no robot named 'cat'"]),
        ("training_status", {"name": "nope"}, ["no policy named 'nope'", "train_policy"]),
        ("lessons", {"number": "three"}, ["number must be a whole number", "'three'"]),
        # A template robot is carried on a plate for its parts, and a wrong part says what is wrong with it.
        ("add_part", {"robot": "dog", "part": {"name": "tail"}}, ["part 'tail'", "size is required"]),
        ("write_script", {"name": "../evil", "code": "print(1)"}, ["script name '../evil'", "pick_cube"]),
        (
            "write_script",
            {"name": "bad", "code": "for i in range(3)\n    print(i)\n"},
            ["syntax error", "line 1", "nothing was saved"],
        ),
        ("read_script", {"name": "nope"}, ["no script named 'nope'", "write_script saves one"]),
        ("run_script", {"name": "nope", "max_sim_seconds": 1000}, ["no script named 'nope'"]),
        ("open_model", {"path": "/nope/robot.urdf"}, ["file not found: /nope/robot.urdf", "MJCF", ".stl"]),
        ("move_robot", {"robot": "dog"}, ["no tool named 'move_robot'", "did you mean move"]),
        ("no_such_tool", {}, ["no tool named 'no_such_tool'", "tool list"]),
    ],
)
async def test_a_refusal_names_the_mistake_and_the_way_out(lab, tool, args, says):
    text = await lab.refused(tool, **args)
    for piece in says:
        assert piece in text, f"{piece!r} not in {text!r}"
    assert len(text) < 320, f"{len(text)} characters: {text}"


@pytest.mark.parametrize("odd", ["{[1]: 2}", "{1, [2]}", "{{}: 1}", "[{[1]: 2}]"])
@pytest.mark.parametrize(
    ("tool", "args", "param"),
    [
        ("add_robot", {"name": "p", "template": "rover"}, "params"),
        ("add_robot", {"name": "p", "template": "rover"}, "pos"),
        ("add_robot", {"name": "p"}, "build"),
        ("set_controls", {}, "controls"),
        ("move_arm", {"robot": "arm"}, "position"),
    ],
)
async def test_text_that_python_cannot_read_as_a_value_is_explained_not_a_crash(lab, tool, args, param, odd):
    """ast.literal_eval raises TypeError for an unhashable key, which escaped tidy as a bare "unhashable type"."""
    text = await lab.refused(tool, **args, **{param: odd})
    assert "unhashable" not in text and "unexpected error" not in text, text
    assert param in text, text


async def test_an_empty_build_beside_a_change_keeps_every_part_of_a_built_robot(lab):
    """A model that fills in every field sends build={}: update_robot took it as "no parts", and 12 servos were gone."""
    from robotforge.design.builder.guide import EXAMPLE_DOG

    await lab.accepted("add_robot", name="spider", build=EXAMPLE_DOG, pos=[-2.0, 2.0])

    def spider():
        return next(r for r in lab.playground.current_world().robots if r.name == "spider")

    parts, servos = len(spider().build["parts"]), lab.playground.session.model.nu
    out = json.loads(await lab.accepted("update_robot", name="spider", pos=[1, 1], params={}, yaw=None, build={}))
    assert spider().pos == [1, 1] and len(spider().build["parts"]) == parts
    assert lab.playground.session.model.nu == servos
    assert "(12 joints, 12 actuators)" in " ".join(out["notes"])  # not "built from 1 parts (0 joints, ...)"
    text = await lab.refused("update_robot", name="spider", build={})
    assert "build={} would remove every part" in text and len(spider().build["parts"]) == parts
    moved = json.loads(await lab.accepted("update_robot", name="dog", pos=[0.5, 1.2], build={}))  # a template robot
    assert moved["robots"][0] == "dog"
    await lab.restore()


async def test_remove_deletes_the_name_it_was_given_and_does_not_guess_a_plural(lab):
    await lab.accepted("add_robot", name="dog2", template="quadruped", pos=[3, 3])
    text = await lab.refused("remove", name="dogs")  # dog, dog2 or both: asked, not deleted
    assert "Did you mean dog, dog2" in text
    assert {"dog", "dog2"} <= {r.name for r in lab.playground.current_world().robots}
    await lab.accepted("remove", name="The Dog2")  # capitals and an article have one meaning
    assert "dog2" not in [r.name for r in lab.playground.current_world().robots]
    await lab.restore()


async def test_a_part_is_not_removed_under_its_plural_either(lab):
    from robotforge.design.builder.guide import EXAMPLE_DOG

    await lab.accepted("add_robot", name="spider", build=EXAMPLE_DOG, pos=[-2.0, 2.0])
    part = next(p["name"] for p in EXAMPLE_DOG["parts"] if p.get("name"))
    await lab.accepted(
        "update_part", robot="spider", part=part + "s", changes={"color": "red"}
    )  # editing may read a plural
    text = await lab.refused("remove_part", robot="spider", part=part + "s")
    assert f"no part named '{part}s'" in text and f"Did you mean {part}" in text
    await lab.restore()


async def test_a_world_file_edited_by_hand_is_explained_by_every_tool_that_reads_it(lab):
    file = lab.playground.store.file
    file.write_text(lab.baseline.replace("shape: sphere", "colour: red\n  shape: sphere", 1), encoding="utf-8")
    for tool, args in (("robots", {}), ("move", {"robot": "dog", "speed": 0.1}), ("add_object", {"name": "b2"})):
        text = await lab.refused(tool, **args)
        assert text == (
            "world.yaml is not valid, fix it in the file. object ball: 'colour' is not a known field; "
            "did you mean color?"
        )
    file.write_text(lab.baseline.replace("- name: dog\n", "- template: quadruped\n", 1), encoding="utf-8")
    assert "robot number 1: name is required" in await lab.refused("remove", name="ball")
    file.write_text("name: lab\nfloor: [1\n", encoding="utf-8")
    assert "world.yaml is not valid YAML at line" in await lab.refused("robots")
    await lab.restore()


async def test_a_world_that_does_not_open_says_where_its_file_is(lab):
    """An agent in the eval told only "unknown template 'quadrupd'" looked for the file in ~/.robotforge, read the
    troubleshooting page and searched the disk before it found and fixed it. The world open before stays open:
    the broken one used to become the one edits went to, while the simulation still ran the last one."""
    garden = lab.playground.store.dir.parent / "garden" / "world.yaml"
    garden.parent.mkdir()
    garden.write_text("name: garden\nrobots:\n- name: pup\n  template: quadrupd\n", encoding="utf-8")
    text = await lab.refused("open_world", name_or_path="garden")
    assert "unknown template 'quadrupd'. Did you mean quadruped?" in text
    assert f"the world is {garden}: fix it there, then open_world again" in text
    assert lab.playground.store.dir.name == "lab"
    listed = json.loads(await lab.accepted("robots"))
    assert "pup" not in {r["name"] for r in listed["robots"]}  # the broken file's robot is not in the session
    garden.write_text(garden.read_text(encoding="utf-8").replace("quadrupd", "quadruped"), encoding="utf-8")
    assert json.loads(await lab.accepted("open_world", name_or_path="garden"))["robots"] == ["pup"]
    lab.playground.store = type(lab.playground.store)(lab.playground.store.dir.parent / "lab")
    await lab.restore()


async def test_a_name_that_exists_reopens_the_world_and_a_new_name_in_chinese_is_a_new_world(lab):
    again = json.loads(await lab.accepted("new_world", name="Lab"))
    assert again["created"] is False and "already existed" in again["notes"][0]
    first = json.loads(await lab.accepted("new_world", name="机器人乐园"))
    second = json.loads(await lab.accepted("new_world", name="我的世界"))
    assert first["created"] is second["created"] is True and first["world"] != second["world"]
    assert f"saved as {first['world']!r}; open_world takes either name" in first["notes"][0]
    assert json.loads(await lab.accepted("open_world", name_or_path="机器人乐园"))["world"] == first["world"]
    lab.playground.store = type(lab.playground.store)(lab.playground.store.dir.parent / "lab")
    await lab.restore()


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("move", {"robot": "x" * 3000}),
        ("stop", {"robot": "x" * 3000}),
        ("remove", {"name": "x" * 3000}),
        ("update_robot", {"name": "x" * 3000, "pos": [1, 1]}),
        ("update_object", {"name": "x" * 3000, "pos": [1, 1]}),
        ("update_terrain", {"name": "x" * 3000, "yaw": 3}),
    ],
)
async def test_an_unknown_name_is_echoed_in_a_few_words_not_in_full(lab, tool, args):
    text = await lab.refused(tool, **args)
    assert "no " in text and "xxxxxxxx..." in text and len(text) < 300, len(text)


async def test_a_library_robot_asked_for_in_another_spelling_joins_the_download_in_progress(monkeypatch):
    import threading
    from pathlib import Path

    from robotforge import mcp_server
    from robotforge.design import library

    started, release = [], threading.Event()

    def slow(robot, progress):
        started.append(robot.name)
        release.wait(5)
        return Path(".")

    class Context:
        async def report_progress(self, *args):
            pass

    monkeypatch.setattr(library, "model_dir", slow)
    monkeypatch.setattr(library, "is_downloaded", lambda name: False)
    monkeypatch.setattr(mcp_server, "PROGRESS_INTERVAL", 0.05)
    monkeypatch.setattr(mcp_server, "_downloads", {})
    first = asyncio.ensure_future(mcp_server._download_with_progress("Go2", Context()))
    await asyncio.sleep(0.2)
    second = asyncio.ensure_future(mcp_server._download_with_progress("Unitree Go2", Context()))
    await asyncio.sleep(0.2)
    release.set()
    await asyncio.gather(first, second)
    assert started == ["go2"], started


def test_a_value_tidy_cannot_read_goes_on_as_given(monkeypatch, caplog):
    def broken(name, value, prop):
        raise RuntimeError("boom")

    monkeypatch.setattr(tool_input, "_tidy_value", broken)
    assert tool_input.tidy("t", SCHEMA, {"name": "x", "pos": "2, 0"}) == {"name": "x", "pos": "2, 0"}
    assert "could not read 'pos' forgivingly" in caplog.text


def test_an_unexpected_error_shows_a_short_text_not_the_whole_path_it_was_given():
    with pytest.raises(ToolError) as caught, _tool_errors():
        raise OSError(63, "File name too long", "/x" * 2000)
    text = str(caught.value)
    assert text.startswith("robotforge hit an unexpected error: OSError: [Errno 63] File name too long")
    assert len(text) < 400, len(text)


async def test_code_in_a_markdown_fence_is_saved_without_it(lab):
    out = json.loads(await lab.accepted("write_script", name="fenced", code="```python\nprint('hi')\n```\n"))
    assert "fence" in out["note"] and out["lines"] == 2
    assert await lab.accepted("read_script", name="fenced") == "print('hi')\n"
    await lab.restore()


@pytest.mark.parametrize(
    ("tool", "args", "unit_hint"),
    [
        ("simulate", {"seconds": 100}, False),  # a long run, not a count of milliseconds
        ("simulate", {"seconds": 500}, True),
        ("simulate", {"seconds": 60000}, True),
        ("simulate", {"seconds": 100000}, False),  # 100 s would still be over the limit
        ("move", {"robot": "dog", "seconds": 120}, False),
        ("move", {"robot": "dog", "seconds": 3000}, True),
        ("run_script", {"name": "idle", "max_sim_seconds": 400}, False),
        ("run_script", {"name": "idle", "max_sim_seconds": 1000}, False),
        ("run_script", {"name": "idle", "max_sim_seconds": 60000}, True),
    ],
)
async def test_a_run_too_long_says_the_limit_and_names_the_unit_only_when_milliseconds_would_fit(
    lab, tool, args, unit_hint
):
    await lab.accepted("write_script", name="idle", code="print('hi')\n")
    text = await lab.refused(tool, **args)
    assert "must be in (0, " in text and f"got {args.get('seconds', args.get('max_sim_seconds'))}" in text, text
    assert ("not milliseconds" in text) is unit_hint, text


async def test_a_script_that_does_not_parse_is_not_saved(lab):
    await lab.refused("write_script", name="broken", code="x = (\n")
    assert "no script named 'broken'" in await lab.refused("read_script", name="broken")


async def test_a_wrong_file_says_what_kind_it_wants(lab, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a robot", encoding="utf-8")
    for tool, args in (("open_model", {"path": str(tmp_path)}), ("open_model", {"path": str(notes)})):
        text = await lab.refused(tool, **args)
        assert "MJCF (.xml)" in text and "URDF" in text and "mesh (" in text
    text = await lab.refused("add_robot", name="x", file=str(notes))
    assert "notes.txt: expected an MJCF" in text
    text = await lab.refused("add_robot", name="x", file=str(tmp_path))
    assert "is a folder" in text


async def test_a_failing_script_shows_its_own_lines_only(lab):
    code = "from robotforge.script import connect\nconnect().move('cat', 1)\n"
    await lab.accepted("write_script", name="Boom", code=code)
    out = json.loads(await lab.accepted("run_script", name="boom"))
    assert not out["ok"]
    assert 'File "boom.py", line 2' in out["output"] and "no robot named 'cat'" in out["output"]
    assert "script.py" not in out["output"] and "robotforge/" not in out["output"]
    await lab.restore()


# ---- answers are compact -----------------------------------------------------------------------------


async def test_results_are_compact_json_with_whole_numbers_short(lab):
    text = await lab.accepted("robots")
    assert all(line[:1] in '{}"' for line in text.splitlines()), text  # a key per line, nothing indented
    assert json.loads(text)["robots"][0]["name"] == "dog"
    assert '"position":[0,1.2,' in text  # 0.0 is written 0
    state = await lab.accepted("observe", include_image=False)
    assert '"joints":{' in state and "\n " not in state


# ---- the pieces ---------------------------------------------------------------------------------------


def test_match_reads_one_meaning_and_refuses_a_guess():
    names = ["dog", "arm", "red_cube", "arm_2"]
    assert wording.match("Dog", names) == "dog"
    assert wording.match("the dog", names) == "dog"
    assert wording.match("Red Cube", names) == "red_cube"
    assert wording.match("arms", names) == "arm"
    assert wording.match("dog1", names) is None
    assert wording.match("A", ["a", "A"]) == "A"  # the exact spelling wins over the loose one
    assert wording.match("shoulder", ["a/shoulder", "b/shoulder"]) is None  # two readings: no guess


def test_unknown_lists_choices_and_closest_names():
    text = wording.unknown("robot", "dog1", ["dog", "arm", "rover"])
    assert text == "no robot named 'dog1'. Did you mean dog? Robots in this world: dog, arm, rover"
    assert wording.unknown("terrain patch", "x", []).endswith("There are no terrain patches in this world.")
    assert "and 3 more" in wording.choices([str(i) for i in range(15)])


def test_equally_close_names_that_start_like_the_given_one_come_first():
    """difflib alone ranks ties by the name itself: "go3" suggested op3 before go2."""
    assert wording.near("go3", ["go2", "op3", "go1"]) == ["go2", "go1", "op3"]
    assert wording.near("x", []) == []


def test_unit_hint_only_for_values_far_over_the_maximum():
    assert wording.unit_hint(30, 0.8) == " (meters, so 30 cm is 0.3)"
    assert wording.unit_hint(2, 0.8) == ""  # two meters is simply too long
    assert wording.unit_hint("30", 0.8) == ""


def test_unwrap_joins_prose_and_keeps_code():
    text = "one two\nthree\n\n    code a\n    code b\n\nfour\nfive"
    assert wording.unwrap(text) == "one two three\n\n    code a\n    code b\n\nfour five"


def test_dumps_is_compact_with_a_key_per_line():
    assert wording.dumps({"a": [0.0, 1.5, 2.0], "b": {"c": "é"}}) == '{\n"a":[0,1.5,2],\n"b":{"c":"é"}\n}'
    assert wording.dumps([1.0, 2]) == "[1,2]"
    assert wording.dumps({}) == "{}"


SCHEMA = {
    "properties": {
        "name": {"type": "string"},
        "pos": {"anyOf": [{"items": {"type": "number"}, "type": "array"}, {"type": "null"}]},
        "shape": {"default": "box", "enum": ["box", "sphere"], "type": "string"},
        "target": {"anyOf": [{"type": "string"}, {"items": {"type": "number"}, "type": "array"}]},
        "params": {"anyOf": [{"type": "object"}, {"type": "null"}]},
        "methods": {"anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}]},
        "speed": {"default": 0.2, "type": "number"},
        "seconds": {"anyOf": [{"type": "number"}, {"type": "null"}], "default": 3.0},
    },
    "required": ["name"],
}


@pytest.mark.parametrize(
    ("given", "becomes"),
    [
        ({"pos": "2, 0"}, {"pos": [2, 0]}),
        ({"pos": "[2, 0, 1]"}, {"pos": [2, 0, 1]}),
        ({"pos": "(2, 0)"}, {"pos": [2, 0]}),
        ({"pos": {"X": 1, "y": 2}}, {"pos": [1, 2]}),
        ({"pos": 0.5}, {"pos": [0.5]}),
        ({"pos": "centre"}, {"pos": "centre"}),  # left for the error to explain
        ({"pos": {"x": 1}}, {"pos": {"x": 1}}),
        ({"shape": "Ball"}, {"shape": "sphere"}),
        ({"target": "0.2, 0.2"}, {"target": [0.2, 0.2]}),
        ({"target": "bin"}, {"target": "bin"}),
        ({"params": "{'a': 1}"}, {"params": {"a": 1}}),
        ({"params": '{"a": 1}'}, {"params": {"a": 1}}),
        ({"methods": "ik, pd"}, {"methods": ["ik", "pd"]}),
        ({"speed": None, "shape": "", "pos": []}, {}),
        ({"seconds": None}, {"seconds": None}),  # null means "until stop" here
    ],
)
def test_tidy_reads_forms_with_one_meaning(given, becomes):
    assert tool_input.tidy("t", SCHEMA, {"name": "x", **given}) == {"name": "x", **becomes}


def test_tidy_never_drops_a_required_argument():
    assert tool_input.tidy("t", SCHEMA, {"name": ""}) == {"name": ""}


def test_tidy_matches_choices_loosely_with_the_parameters_aliases():
    shape = {"properties": {"shape": SCHEMA["properties"]["shape"]}}
    assert tool_input.tidy("add_object", shape, {"shape": "BOX"}) == {"shape": "box"}
    assert tool_input.tidy("add_object", shape, {"shape": "ball"}) == {"shape": "sphere"}
    assert tool_input.tidy("add_object", shape, {"shape": "pyramid"}) == {"shape": "pyramid"}  # left to the error


def test_tidy_refuses_a_parameter_the_tool_does_not_have():
    with pytest.raises(ToolError, match=r"t has no parameter 'velocity'; did you mean speed\? \(parameters: name,"):
        tool_input.tidy("t", SCHEMA, {"name": "x", "velocity": 1})
    with pytest.raises(ToolError, match="takes no parameters"):
        tool_input.tidy("t", {"properties": {}}, {"a": 1})
