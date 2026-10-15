"""What does an agent see when a call is a little wrong? Run a battery of plausible mistakes and grade the answers.

    uv run python scripts/agent_errors.py [--tool NAME] [--bad] [--json PATH]
    uv run python scripts/agent_errors.py --sizes [--json PATH]

The server is built in-process, as tests/test_mcp_server.py does, on a temporary ROBOTFORGE_HOME with the
Menagerie test fixtures. A world with a template dog, an arm, a rover and a ball is the starting point of
every case; each case calls one tool the way a hurried agent might (a near-miss name, a number as text,
degrees for radians, centimetres for metres, a missing argument, ...) and records whether the call was
accepted and, if not, the error text.

Each case says what it ought to do: "accept" when the intent is unambiguous and safe to act on (a name in the
wrong case), "reject" otherwise. A rejection is graded on four things an agent needs from the text alone:
  leak     a Python traceback, a pydantic URL or type tag, or an internal path
  vague    nothing in the text points at the argument or value that was wrong
  no_fix   no valid choices, form or range to correct it with
  long     over LONG_ERROR characters without listing choices (agents pay for every word)
"--sizes" prints the size in tokens (characters / 4) of every tool description and schema, the server
instructions and typical responses instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HOME = Path(tempfile.mkdtemp(prefix="rf_agent_errors_"))
# Set before robotforge reads them: the worlds live under HOME, library robots come from the offline fixtures.
os.environ["ROBOTFORGE_HOME"] = str(HOME)
os.environ["ROBOTFORGE_MENAGERIE"] = str(REPO / "tests" / "fixtures" / "menagerie")
os.environ["ROBOTFORGE_NO_BROWSER"] = "1"

from mcp import Client  # noqa: E402

from robotforge.cli import find_free_port  # noqa: E402
from robotforge.design.builder.guide import EXAMPLE_DOG  # noqa: E402
from robotforge.design.store import WorldStore, worlds_home  # noqa: E402
from robotforge.mcp_server import INSTRUCTIONS, build_server  # noqa: E402
from robotforge.playground import Playground  # noqa: E402

LONG_ERROR = 300  # characters; longer is fine only when the text lists the valid choices
CALL_TIMEOUT = 180.0  # s per call
ASSETS = HOME / "assets"
BASELINE = HOME / "baseline"

# ---- the battery ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    tool: str
    why: str  # the mistake, in a few words
    args: dict[str, Any]
    want: str = "reject"  # "accept" when the intent is unambiguous and safe to act on
    setup: tuple[tuple[str, dict[str, Any]], ...] = ()  # calls made first (not graded)
    no_world: bool = False  # run with no world open
    note: str = ""  # an accepted call must say this (a regex): it changed something the agent asked for


def c(tool: str, why: str, want: str = "reject", setup=(), no_world: bool = False, note: str = "", **args: Any) -> Case:
    return Case(tool, why, args, want, tuple(setup), no_world, note)


HUMANOID = ("add_robot", {"name": "hum", "template": "humanoid", "pos": [2.0, 2.0]})
BUILT = ("add_robot", {"name": "spider", "build": EXAMPLE_DOG, "pos": [-2.0, 2.0]})
RAMP = ("add_terrain", {"type": "ramp", "name": "ramp1", "pos": [3.0, 3.0]})
SCRIPT_OK = ("write_script", {"name": "hello", "code": "print('hi')\n"})

CASES: list[Case] = [
    # -- worlds
    c("new_world", "name with spaces and capitals", "accept", name="My Cool World"),
    c("new_world", "name with slashes", "accept", name="a/b/c"),
    c("new_world", "name that is only symbols", "accept", name="???"),
    c("new_world", "floor in capitals", "accept", name="w1", floor="Grass"),
    c("new_world", "unknown floor", name="w1", floor="lava"),
    c("new_world", "floor typo", name="w1", floor="grassy"),
    c("new_world", "name missing"),
    c("new_world", "name given as a number", name=7),
    c("new_world", "wrong parameter name", world_name="w1"),
    c("open_world", "world that does not exist", name_or_path="nope"),
    c("open_world", "path that does not exist", name_or_path="/nope/at/all"),
    c("open_world", "world name in capitals", "accept", name_or_path="Lab"),
    c("open_world", "name with a space", name_or_path="my lab"),
    c("open_world", "wrong parameter name", name="lab"),
    c("show_world", "no world open", no_world=True),
    c("undo", "unknown parameter", steps=2),
    # -- add_robot
    c("add_robot", "template in capitals", "accept", name="pup", template="Quadruped", pos=[2, 0]),
    c("add_robot", "name with spaces and capitals", "accept", name="My Pup", template="quadruped", pos=[2, 0]),
    c("add_robot", "name starts with a digit", name="1pup", template="quadruped"),
    c("add_robot", "name already used", name="dog", template="quadruped", pos=[2, 0]),
    c("add_robot", "template is a nickname", name="pup", template="dog"),
    c("add_robot", "template typo", name="pup", template="quadruped2"),
    c("add_robot", "template is a plural", "accept", name="pup", template="arms", pos=[2, 0]),
    c("add_robot", "no source at all", name="pup"),
    c("add_robot", "two sources", name="pup", template="arm", library="go2"),
    c("add_robot", "library name in capitals", "accept", name="go", library="Go2", pos=[3, 0]),
    c("add_robot", "unknown library robot", name="go", library="go3"),
    c("add_robot", "library robot with a long name", "accept", name="go", library="Unitree Go2", pos=[3, 0]),
    c("add_robot", "file that does not exist", name="x", file="/nope/robot.urdf"),
    c("add_robot", "file with the wrong extension", name="x", file="{assets}/notes.txt"),
    c("add_robot", "file that is a folder", name="x", file="{assets}"),
    c("add_robot", "file that is not valid XML", name="x", file="{assets}/broken.urdf"),
    c("add_robot", "params in centimetres", name="pup", template="quadruped", params={"leg_length": 30}),
    c(
        "add_robot",
        "param as text",
        "accept",
        name="pup",
        template="quadruped",
        pos=[2, 0],
        params={"leg_length": "0.3"},
    ),
    c("add_robot", "param the template does not have", name="pup", template="quadruped", params={"colour": "red"}),
    c("add_robot", "param is a bad color", name="pup", template="quadruped", params={"color": "reddish"}),
    c("add_robot", "params for a library robot", name="go", library="go2", params={"leg_length": 0.3}),
    c("add_robot", "params as Python text", "accept", name="pup", template="arm", pos=[2, 0], params="{'reach': 0.3}"),
    c("add_robot", "pos as text", "accept", name="pup", template="rover", pos="2, 0"),
    c("add_robot", "pos as a dict", "accept", name="pup", template="rover", pos={"x": 2, "y": 0}),
    c("add_robot", "pos with one number", name="pup", template="rover", pos=[2]),
    c("add_robot", "pos with four numbers", name="pup", template="rover", pos=[2, 0, 0, 0]),
    c("add_robot", "yaw as text", "accept", name="pup", template="rover", pos=[2, 0], yaw="90"),
    c("add_robot", "yaw with a unit", name="pup", template="rover", yaw="90 deg"),
    c("add_robot", "build with no parts", "accept", name="pup", build={}, pos=[2, 0]),
    c("add_robot", "build as nonsense", name="pup", build={"parts": "legs"}),
    c("add_robot", "nothing in the world", no_world=True, name="pup", template="rover"),
    c("add_robot", "everything missing"),
    c("update_robot", "robot name in capitals", "accept", name="Dog", pos=[0.5, 1.2]),
    c("update_robot", "robot that does not exist", name="cat", pos=[1, 1]),
    c("update_robot", "robot name with extra number", name="dog1", pos=[1, 1]),
    c("update_robot", "param out of range", name="dog", params={"leg_length": 2}),
    c("update_robot", "pos as text", "accept", name="dog", pos="0.5, 1.2"),
    c("update_robot", "build on a template robot", name="dog", build=EXAMPLE_DOG),
    # -- objects
    c("add_object", "shape called ball", "accept", name="b2", shape="ball", size=[0.05], pos=[0.4, 0.1]),
    c("add_object", "shape called cube", "accept", name="b2", shape="cube", pos=[0.4, 0.1]),
    c("add_object", "shape in capitals", "accept", name="b2", shape="Sphere", size=[0.05], pos=[0.4, 0.1]),
    c("add_object", "unknown shape", name="b2", shape="pyramid"),
    c("add_object", "name already used", name="ball", shape="sphere", size=[0.05]),
    c("add_object", "name with spaces and capitals", "accept", name="Red Cube", pos=[0.4, 0.1]),
    c("add_object", "size is one number", "accept", name="b2", shape="sphere", size=0.05, pos=[0.4, 0.1]),
    c("add_object", "size is text", "accept", name="b2", size="0.05, 0.05, 0.05", pos=[0.4, 0.1]),
    c("add_object", "box size with two numbers", name="b2", shape="box", size=[0.05, 0.05]),
    c("add_object", "size is negative", name="b2", shape="box", size=[0.05, -0.05, 0.05]),
    c("add_object", "size is a dict", name="b2", shape="box", size={"length": 1, "width": 1, "height": 1}),
    c("add_object", "color 0 to 255", "accept", name="b2", color=[255, 128, 0], pos=[0.4, 0.1]),
    c("add_object", "color not a color", name="b2", color="reddish"),
    c("add_object", "pos is a word", name="b2", pos="center"),
    c("add_object", "pos is a dict", "accept", name="b2", pos={"x": 0.4, "y": 0.1}),
    c("add_object", "pos is not a number", name="b2", pos=["left", "up"]),
    c("add_object", "mass is zero", name="b2", mass=0),
    c("add_object", "mass is a word", name="b2", mass="heavy"),
    c("add_object", "fixed as a word", "accept", name="b2", fixed="yes", pos=[0.4, 0.1]),
    c("add_object", "parameter the tool does not have", name="b2", radius=0.05),
    c("add_object", "everything missing"),
    c("update_object", "object name in capitals", "accept", name="Ball", pos=[0.3, -0.1]),
    c("update_object", "object that does not exist", name="cube", pos=[0.3, 0.1]),
    c("update_object", "nothing to change", name="ball"),
    c("update_object", "color not a color", name="ball", color="#12"),
    c("remove", "name in capitals", "accept", name="Ball"),
    c("remove", "nothing with that name", name="table"),
    c("remove", "the floor", name="floor"),
    c("set_floor", "style in capitals", "accept", style="Concrete"),
    c("set_floor", "unknown style", style="marble"),
    c("set_floor", "wrong parameter name", floor="grass"),
    # -- scenes and terrain
    c("load_scene", "preset in capitals", "accept", preset="Desk_Lab"),
    c("load_scene", "preset with a space", "accept", preset="desk lab"),
    c("load_scene", "unknown preset", preset="garage"),
    c("load_scene", "seed as text", "accept", preset="maze", seed="3"),
    c("load_scene", "seed with decimals", preset="maze", seed=1.5),
    c("load_scene", "option that does not exist", preset="maze", options={"size": 5}),
    c("load_scene", "option out of range", preset="maze", options={"rows": 70}),
    c("load_scene", "options as text", "accept", preset="maze", options="{'rows': 5, 'cols': 5}"),
    c("load_scene", "preset missing"),
    c("set_terrain", "type in capitals", "accept", type="Hills"),
    c("set_terrain", "unknown type", type="mountain"),
    c("set_terrain", "type is a nickname", "accept", type="slope"),
    c("set_terrain", "params in centimetres", type="stairs", params={"step_height": 15}),
    c("set_terrain", "param that does not exist", type="stairs", params={"stairs": 4}),
    c("set_terrain", "unknown style", type="hills", style="lava"),
    c("add_terrain", "type is a nickname", "accept", type="steps", pos=[2, 0]),
    c("add_terrain", "pos as text", "accept", type="ramp", pos="2, 0"),
    c("add_terrain", "pos with one number", type="ramp", pos=[2]),
    c("add_terrain", "angle that is steep", type="ramp", params={"angle": 80}),
    c("add_terrain", "name already used", type="ramp", name="dog"),
    c("update_terrain", "patch that does not exist", setup=[RAMP], name="ramp"),
    c("update_terrain", "patch name in capitals", "accept", setup=[RAMP], name="Ramp1", yaw=45),
    c("update_terrain", "no terrain at all", name="ramp1", yaw=45),
    # -- behaviors
    c("move", "robot name in capitals", "accept", robot="Dog", speed=0.2, seconds=1),
    c("move", "robot name with a number", robot="dog1", speed=0.2, seconds=1),
    c("move", "robot named with an article", "accept", robot="the dog", speed=0.2, seconds=1),
    c("move", "robot that does not exist", robot="cat"),
    c("move", "robot that cannot move", robot="arm", speed=0.2),
    c("move", "speed is a word", robot="dog", speed="fast"),
    c("move", "speed as text", "accept", robot="dog", speed="0.2", seconds=1),
    c("move", "speed with a unit", robot="dog", speed="0.2 m/s"),
    c("move", "speed far too high", "accept", note="max|limit|clamp", robot="dog", speed=50.0, seconds=1),
    c("move", "speed in cm per second", "accept", note="max|limit|clamp", robot="rover", speed=30.0, seconds=1),
    c("move", "turn in degrees per second", "accept", note="max|limit|clamp", robot="rover", turn=90.0, seconds=1),
    c("move", "seconds is zero", robot="dog", speed=0.2, seconds=0),
    c("move", "seconds too long", robot="dog", speed=0.2, seconds=600),
    c("move", "seconds in milliseconds", robot="dog", speed=0.2, seconds=3000),
    c("move", "walker in capitals", "accept", robot="dog", speed=0.2, seconds=1, walker="Gait"),
    c("move", "walker that does not exist", robot="dog", speed=0.2, walker="sprint"),
    c("move", "walker on a wheeled robot", robot="rover", speed=0.2, walker="rl"),
    c("move", "learned walker the dog does not have", robot="dog", speed=0.2, walker="rl"),
    c("move", "parameter the tool does not have", robot="dog", velocity=0.2),
    c("move", "everything missing"),
    c("stop", "robot name in capitals", "accept", robot="Rover"),
    c("stop", "robot that does not exist", robot="cat"),
    c("pose", "pose in capitals", "accept", setup=[HUMANOID], robot="hum", name="Wave"),
    c("pose", "pose that does not exist", setup=[HUMANOID], robot="hum", name="dance"),
    c("pose", "robot that is not a humanoid", robot="dog", name="wave"),
    c("move_arm", "robot name in capitals", "accept", robot="Arm", position=[0.2, 0.0, 0.15]),
    c("move_arm", "position as text", "accept", robot="arm", position="0.2, 0.0, 0.15"),
    c("move_arm", "position as a dict", "accept", robot="arm", position={"x": 0.2, "y": 0.0, "z": 0.15}),
    c("move_arm", "position with two numbers", robot="arm", position=[0.2, 0.0]),
    c("move_arm", "position in centimetres", robot="arm", position=[20, 0, 15]),
    c("move_arm", "position out of reach", robot="arm", position=[2.0, 0.0, 0.5]),
    c("move_arm", "position under the floor", robot="arm", position=[0.2, 0.0, -0.5]),
    c("move_arm", "gripper in capitals", "accept", robot="arm", position=[0.2, 0.0, 0.15], gripper="Open"),
    c("move_arm", "gripper word that does not exist", robot="arm", position=[0.2, 0.0, 0.15], gripper="half"),
    c("move_arm", "walking robot instead of an arm", robot="dog", position=[0.2, 0.0, 0.15]),
    c("move_arm", "arm mounted on a robot that has none", robot="rover/arm", position=[0.2, 0.0, 0.15]),
    c("gripper", "state in capitals", "accept", robot="arm", state="Close"),
    c("gripper", "state that does not exist", robot="arm", state="half"),
    c("gripper", "state is a synonym", "accept", robot="arm", state="grab"),
    c("gripper", "walking robot instead of an arm", robot="dog", state="open"),
    c("pick_and_place", "object name in capitals", "accept", robot="arm", object="Ball", target=[0.2, 0.2]),
    c("pick_and_place", "object that does not exist", robot="arm", object="cube", target=[0.2, 0.2]),
    c("pick_and_place", "target that does not exist", robot="arm", object="ball", target="bin"),
    c("pick_and_place", "target with one number", robot="arm", object="ball", target=[0.2]),
    c("pick_and_place", "target as text numbers", "accept", robot="arm", object="ball", target="0.2, 0.2"),
    c("pick_and_place", "target out of reach", robot="arm", object="ball", target=[3.0, 3.0]),
    c("pick_and_place", "the arm itself as object", robot="arm", object="arm", target=[0.2, 0.2]),
    c("pick_and_place", "target missing", robot="arm", object="ball"),
    c("pick_and_place", "parameter spelled differently", robot="arm", obj="ball", target=[0.2, 0.2]),
    c("pick_and_place", "walking robot instead of an arm", robot="dog", object="ball", target=[0.2, 0.2]),
    # -- scripts
    c("write_script", "name with spaces and capitals", "accept", name="Pick Cube", code="print(1)\n"),
    c("write_script", "name with .py", "accept", name="dance.py", code="print(1)\n"),
    c("write_script", "name that climbs out of the folder", name="../evil", code="print(1)\n"),
    c("write_script", "name with a slash", name="a/b", code="print(1)\n"),
    c("write_script", "syntax error in the code", name="broken", code="for i in range(3)\n    print(i)\n"),
    c("write_script", "code is not text", name="broken", code=["print(1)"]),
    c("write_script", "no world open", no_world=True, name="s", code="print(1)\n"),
    c("read_script", "script that does not exist", name="nope"),
    c("read_script", "name in capitals", "accept", setup=[SCRIPT_OK], name="Hello"),
    c("run_script", "script that does not exist", name="nope"),
    c("run_script", "name in capitals", "accept", setup=[SCRIPT_OK], name="Hello"),
    c("run_script", "max_sim_seconds is zero", setup=[SCRIPT_OK], name="hello", max_sim_seconds=0),
    c("run_script", "max_sim_seconds too long", setup=[SCRIPT_OK], name="hello", max_sim_seconds=1000),
    c("run_script", "max_sim_seconds as text", "accept", setup=[SCRIPT_OK], name="hello", max_sim_seconds="5"),
    c(
        "run_script",
        "script with a syntax error",
        "accept",
        setup=[("@write", {"path": "behaviors/bad.py", "text": "x = (\n"})],
        name="bad",
    ),
    c(
        "run_script",
        "script that raises",
        "accept",
        setup=[
            (
                "write_script",
                {"name": "boom", "code": "from robotforge.script import connect\nconnect().move('cat', 1)\n"},
            )
        ],
        name="boom",
    ),
    c(
        "run_script",
        "script that imports a module that is not there",
        "accept",
        setup=[("write_script", {"name": "imp", "code": "import numpyy\n"})],
        name="imp",
    ),
    # -- raw model access
    c("open_model", "file that does not exist", path="/nope/robot.urdf"),
    c("open_model", "file with the wrong extension", path="{assets}/notes.txt"),
    c("open_model", "file that is a folder", path="{assets}"),
    c("open_model", "file that is not valid XML", path="{assets}/broken.urdf"),
    c("open_model", "options that do not exist", path="{assets}/broken.urdf", options={"fixed": True}),
    c("open_model", "path missing"),
    c("set_controls", "actuator without the robot prefix", "accept", controls={"shoulder": 0.3}),
    c("set_controls", "actuator in capitals", "accept", controls={"ARM/Shoulder": 0.3}),
    c("set_controls", "actuator typo", controls={"arm/sholder": 0.3}),
    c(
        "set_controls",
        "actuator in two robots",
        controls={"shoulder": 0.3},
        setup=[("add_robot", {"name": "arm2", "template": "arm", "pos": [1, 0]})],
    ),
    c("set_controls", "degrees where radians are expected", "accept", note="clamp", controls={"arm/shoulder": 90}),
    c("set_controls", "value as text", "accept", controls={"arm/shoulder": "0.3"}),
    c("set_controls", "value is a word", controls={"arm/shoulder": "up"}),
    c("set_controls", "controls as a list", controls=["arm/shoulder", 0.3]),
    c("set_controls", "controls as text", controls="arm/shoulder=0.3"),
    c("set_controls", "controls missing"),
    c("simulate", "seconds is zero", seconds=0),
    c("simulate", "seconds too long", seconds=100),
    c("simulate", "seconds as text", "accept", seconds="0.2"),
    c("simulate", "seconds is a word", seconds="two"),
    c("simulate", "seconds in milliseconds", seconds=500),
    c("simulate", "controls with an unknown actuator", seconds=0.1, controls={"wing": 1}),
    c("sim_control", "action in capitals", "accept", action="Pause"),
    c("sim_control", "action synonym", "accept", action="resume"),
    c("sim_control", "action that does not exist", action="stop"),
    c("observe", "view in capitals", "accept", view="Front", width=160, height=120),
    c("observe", "view that does not exist", view="sideways"),
    c("observe", "view is a nickname", view="side"),
    c("observe", "width in the wrong form", view="front", width="wide"),
    c("observe", "width is zero", view="front", width=0),
    c("observe", "width far too large", view="front", width=100000),
    c("observe", "include_image as a word", "accept", include_image="no"),
    # -- algorithms and training
    c("run_controller", "method in capitals", "accept", robot="Arm", task="Reach", method="IK", seconds=1),
    c(
        "run_controller",
        "method spelled out",
        "accept",
        robot="arm",
        task="reach",
        method="inverse_kinematics",
        seconds=1,
    ),
    c("run_controller", "method that does not exist", robot="arm", task="reach", method="pid"),
    c("run_controller", "task that does not exist", robot="arm", task="grab", method="ik"),
    c("run_controller", "robot that does not exist", robot="cat", task="reach", method="ik"),
    c("run_controller", "method that does not fit the task", robot="arm", task="reach", method="lqr"),
    c("run_controller", "task that does not fit the robot", robot="dog", task="reach", method="ik"),
    c("run_controller", "seconds too long", robot="arm", task="reach", method="ik", seconds=1000),
    c("run_controller", "push after the end", robot="arm", task="reach", method="ik", seconds=2, push_at=10),
    c(
        "run_controller",
        "task param that does not exist",
        robot="arm",
        task="reach",
        method="ik",
        task_params={"goal": [0.2, 0, 0.2]},
    ),
    c(
        "run_controller",
        "target in centimetres",
        robot="arm",
        task="reach",
        method="ik",
        task_params={"target": [20, 0, 20]},
    ),
    c(
        "run_controller",
        "target as the name of an object",
        "accept",
        robot="arm",
        task="reach",
        method="ik",
        seconds=1,
        task_params={"target": "Ball"},
    ),
    c("run_controller", "policy that does not exist", robot="arm", task="reach", method="policy:nope"),
    c("compare_controllers", "no methods", robot="arm", task="reach", methods=[]),
    c(
        "compare_controllers",
        "methods as one text",
        "accept",
        robot="arm",
        task="reach",
        methods="ik,cartesian",
        seconds=1,
    ),
    c("compare_controllers", "too many methods", robot="arm", task="reach", methods=["ik"] * 7),
    c("compare_controllers", "method that does not exist", robot="arm", task="reach", methods=["ik", "magic"]),
    c("train_policy", "task that does not exist", task="jump", robot="dog"),
    c("train_policy", "robot that does not exist", task="walk", robot="cat"),
    c("train_policy", "minutes too many", task="walk", robot="dog", minutes=9999),
    c("train_policy", "algorithm that does not exist", task="walk", robot="dog", algo="ddpg"),
    c("train_policy", "robot missing", task="walk"),
    c("training_status", "run that does not exist", name="nope"),
    c("training_status", "waits too long", name="nope", wait_seconds=100000),
    c("stop_training", "run that does not exist", name="nope"),
    c("unilab_train", "task that does not exist", task="fly"),
    c("unilab_train", "iterations is zero", iterations=0),
    c("unilab_job", "job that does not exist", job="nope"),
    c("lessons", "lesson 0", number=0),
    c("lessons", "lesson 10", number=10),
    c("lessons", "lesson as text", "accept", number="3"),
    c("lessons", "lesson as a word", number="three"),
    # -- builder
    c(
        "add_part",
        "robot made from a template",
        robot="dog",
        part={"name": "tail", "at": "back", "shape": "sphere", "size": [0.03]},
    ),
    c("add_part", "robot that does not exist", robot="cat", part={"name": "tail"}),
    c(
        "add_part",
        "part given as text",
        "accept",
        setup=[BUILT],
        robot="spider",
        part='{"name": "tail", "at": [-0.2, 0, 0], "shape": "sphere", "size": [0.03]}',
    ),
    c(
        "add_part",
        "part with no name",
        setup=[BUILT],
        robot="spider",
        part={"at": [-0.2, 0, 0], "shape": "sphere", "size": [0.03]},
    ),
    c("add_part", "part with an unknown field", setup=[BUILT], robot="spider", part={"name": "tail", "colour": "red"}),
    c("update_part", "part that does not exist", setup=[BUILT], robot="spider", part="tail", changes={"size": [0.02]}),
    c(
        "update_part",
        "part name in capitals",
        "accept",
        setup=[BUILT],
        robot="spider",
        part="Left_Shin",
        changes={"size": [0.02, 0.11]},
    ),
    c(
        "update_part",
        "change that does not exist",
        setup=[BUILT],
        robot="spider",
        part="left_shin",
        changes={"colour": "red"},
    ),
    c("remove_part", "part that does not exist", setup=[BUILT], robot="spider", part="tail"),
    # -- tools and no-argument tools
    c("open_viewer", "launch_browser as a word", "accept", launch_browser="no"),
    c("list_templates", "filter the tool does not have", kind="arm"),
    c("robots", "robot given to a tool without arguments", name="dog"),
    c("describe_model", "unknown parameter", robot="dog"),
    c("no_such_tool", "tool name that does not exist"),
    c("move_robot", "tool name that is close to move", robot="dog", speed=0.2),
    c("Move", "tool name in capitals", "accept", robot="dog", speed=0.2, seconds=1),
    c("robotforge.move", "tool name with the server in front", "accept", robot="dog", speed=0.2, seconds=1),
    # -- numbers in centimetres, fields a strict model fills in, settings in the wrong place
    c("add_robot", "pos in centimetres", name="pup", template="rover", pos=[150, 80]),
    c("add_object", "pos in centimetres", name="b2", pos=[150, 80]),
    c("add_object", "size in centimetres", name="b2", shape="table", size=[120, 60, 75]),
    c("update_object", "pos in centimetres", name="ball", pos=[250, 0]),
    c("add_terrain", "pos in centimetres", type="ramp", pos=[300, 0]),
    c("move", "speed null", "accept", robot="dog", speed=None, seconds=1),
    c("new_world", "floor null", "accept", name="w1", floor=None),
    c(
        "add_robot",
        "template empty text with a library robot",
        "accept",
        name="go",
        template="",
        library="go2",
        pos=[3, 0],
    ),
    c("add_object", "size an empty list", "accept", name="b2", size=[], pos=[0.4, 0.1]),
    c("observe", "view empty text", "accept", view="", width=160, height=120),
    c("add_robot", "settings at the top level", name="pup", template="quadruped", leg_length=0.3),
    c("pick_and_place", "target as a dict", "accept", robot="arm", object="ball", target={"x": 0.2, "y": 0.2}),
    c("sim_control", "action with spaces around it", "accept", action=" Pause "),
    c(
        "add_object",
        "box size is one number",
        "accept",
        note="cube",
        name="b2",
        shape="box",
        size=[0.1],
        pos=[0.4, 0.1],
    ),
    c("add_robot", "yaw in radians", "accept", note="radians", name="pup", template="rover", pos=[2, 0], yaw=1.57),
    c("update_robot", "yaw in radians", "accept", note="radians", name="dog", yaw=3.14),
    c("add_object", "name that is not Latin letters", name="\u7403", pos=[0.4, 0.1]),
    c("add_robot", "name empty", name="", template="rover"),
    c("remove", "name empty", name=""),
    c("add_part", "part given as a list", setup=[BUILT], robot="spider", part=[{"name": "tail"}]),
]


# ---- running them -----------------------------------------------------------------------------------------


@dataclass
class Outcome:
    tool: str
    why: str
    args: str
    want: str
    accepted: bool
    text: str  # the error, or the start of the result
    length: int
    problems: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.accepted:
            return "good" if not self.problems else "bad"
        if self.want == "reject":
            return "accepted?"  # went through, though the call was wrong: the agent may think it did something else
        return self.problems[0] if self.problems else "accepted"  # "silent": changed its input without saying


def fill(value: Any) -> Any:
    """Paths to the files the harness made, spelled {assets} in the battery."""
    if isinstance(value, str):
        return value.replace("{assets}", str(ASSETS))
    if isinstance(value, list):
        return [fill(v) for v in value]
    if isinstance(value, dict):
        return {k: fill(v) for k, v in value.items()}
    return value


def leaves(value: Any) -> list[str]:
    """Every key and scalar of the arguments as text, and the file name of every path: what an error should point at."""
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in [str(k), *leaves(v)]]
    if isinstance(value, list):
        return [t for v in value for t in leaves(v)]
    if value is None:
        return []
    text = str(value)
    if not text:
        return ["''"]  # an empty text is pointed at by its quotes
    if "/" in text:
        return [text, Path(text).name]
    return [text, text.split(":")[-1]] if ":" in text else [text]  # "policy:nope" is pointed at by "nope"


# What an agent must never be shown: Python's or pydantic's own wording, a position inside robotforge's data
# ("robots.3:") and paths into its source or its libraries.
LEAK = re.compile(
    r"Traceback|errors\.pydantic\.dev|For further information|\[type=|validation errors? for|pydantic"
    r"|UnexpectedToolError|Input should|Extra inputs|Field required|Value error|\b(?:robots|objects|terrain)\.\d+\b"
)
INTERNAL = re.compile(r"[\\/]src[\\/]robotforge[\\/]|site-packages")
# Words that carry the way out: the valid choices, the right form, the allowed range.
FIX = re.compile(
    r"\b(use|choose|one of|did you mean|e\.g\.|for example|try|call|give|put|say|must be|should be|expected|valid"
    r"|available|instead|write it|see|list_|between|at most|at least|like|such as|first|in \(|in \[|is \[|are \[|leave"
    r"|takes)\b",
    re.IGNORECASE,
)
CHOICES = re.compile(
    r"one of|choose|available|did you mean|installed|existing|in this world|here:|"
    r"\b(?:parts|jobs|options|names|scenes|worlds|views|robots|objects|actuators|templates|methods|tasks|lessons"
    r"|scripts|patches|policies|walkers|arms|shapes|poses|joints|fields|parameters|views):",
    re.IGNORECASE,
)


def grade(case: Case, args: dict[str, Any], text: str, tool_names: set[str], params: set[str]) -> list[str]:
    problems = []
    if LEAK.search(text) or INTERNAL.search(text):
        problems.append("leak")
    tokens = {t for t in leaves(args) if t} | set(args) | params | {case.tool}
    pointed = any(re.search(rf"(?<![\w.]){re.escape(t)}(?!\w|\.\d)", text) for t in tokens)
    if not case.no_world and not pointed:
        problems.append("vague")
    if not (FIX.search(text) or CHOICES.search(text) or any(name in text for name in tool_names)):
        problems.append("no_fix")
    if len(text) > LONG_ERROR and not (text.count(",") >= 4 or CHOICES.search(text)):
        problems.append("long")
    return problems


class Bench:
    """One server on a world with a dog, an arm, a rover and a ball; restored after any case that could change it."""

    def __init__(self) -> None:
        ASSETS.mkdir(parents=True)
        (ASSETS / "notes.txt").write_text("not a robot\n", encoding="utf-8")
        (ASSETS / "broken.urdf").write_text("<robot name='x'><link", encoding="utf-8")
        self.playground = Playground.with_demo()
        self.server = build_server(self.playground, find_free_port(8790))
        self.client: Client
        self.schemas: dict[str, set[str]] = {}  # tool -> its parameter names

    async def call(self, tool: str, args: dict[str, Any]) -> tuple[bool, str]:
        try:
            result = await asyncio.wait_for(self.client.call_tool(tool, args), CALL_TIMEOUT)
        except Exception as exc:  # a protocol-level error, e.g. an unknown tool on some SDK versions
            return False, f"{type(exc).__name__}: {exc}"
        text = "".join(b.text for b in result.content if getattr(b, "type", "") == "text")
        return not result.is_error, text

    async def start(self) -> None:
        self.schemas = {
            t.name: set(t.input_schema.get("properties", {})) for t in (await self.client.list_tools()).tools
        }
        for tool, args in (
            ("new_world", {"name": "lab"}),
            ("add_robot", {"name": "dog", "template": "quadruped", "pos": [0.0, 1.2]}),
            ("add_robot", {"name": "arm", "template": "arm", "pos": [0.0, 0.0]}),
            ("add_robot", {"name": "rover", "template": "rover", "pos": [0.0, -1.0]}),
            ("add_object", {"name": "ball", "shape": "sphere", "size": [0.04], "pos": [0.25, -0.12]}),
        ):
            ok, text = await self.call(tool, args)
            if not ok:
                raise SystemExit(f"the starting world could not be built: {tool}: {text}")
        shutil.copytree(worlds_home(), BASELINE)

    def restore(self) -> None:
        shutil.rmtree(worlds_home())
        shutil.copytree(BASELINE, worlds_home())
        self.playground.store = WorldStore(worlds_home() / "lab")
        self.playground.reload_world()
        self.playground.session.paused = False

    async def run(self, case: Case) -> Outcome:
        args = fill(case.args)
        dirty = bool(case.setup) or case.no_world
        for tool, setup_args in case.setup:
            if tool == "@write":  # a file put in the world's folder directly, as a person could
                path = worlds_home() / "lab" / setup_args["path"]
                path.parent.mkdir(exist_ok=True)
                path.write_text(setup_args["text"], encoding="utf-8")
                continue
            ok, text = await self.call(tool, setup_args)
            if not ok:
                raise SystemExit(f"setup failed for {case.tool} / {case.why}: {tool}: {text}")
        if case.no_world:
            self.playground.store = None
        yaml_before = (worlds_home() / "lab" / "world.yaml").read_text(encoding="utf-8")
        accepted, text = await self.call(case.tool, args)
        yaml_after = (worlds_home() / "lab" / "world.yaml").read_text(encoding="utf-8")
        if accepted or dirty or yaml_after != yaml_before or len(list(worlds_home().iterdir())) != 1:
            self.restore()
        if accepted:
            problems = ["silent"] if case.note and not re.search(case.note, text, re.IGNORECASE) else []
            if INTERNAL.search(text):
                problems.append("noisy")  # a result that shows robotforge's own frames or paths
        else:
            problems = grade(case, args, text, set(self.schemas), self.schemas.get(case.tool, set()))
        shown = text if not accepted else " ".join(text.split())[:160]
        return Outcome(
            case.tool, case.why, json.dumps(args, ensure_ascii=False), case.want, accepted, shown, len(text), problems
        )


async def run_battery(only: str | None) -> list[Outcome]:
    bench = Bench()
    outcomes = []
    async with Client(bench.server) as client:
        bench.client = client
        await bench.start()
        for case in CASES:
            if only and only not in case.tool:
                continue
            outcomes.append(await bench.run(case))
    return outcomes


# ---- reporting --------------------------------------------------------------------------------------------


def summarize(outcomes: list[Outcome]) -> dict[str, float]:
    rejected = [o for o in outcomes if not o.accepted]
    return {
        "cases": len(outcomes),
        "accepted": sum(o.accepted for o in outcomes),
        "accepted_as_wanted": sum(o.verdict == "accepted" for o in outcomes),
        "accepted_but_silent_or_noisy": sum(o.verdict in ("silent", "noisy") for o in outcomes),
        "accepted_but_should_not": sum(o.verdict == "accepted?" for o in outcomes),
        "should_accept_but_rejected": sum(not o.accepted and o.want == "accept" for o in outcomes),
        "rejected": len(rejected),
        "good_error": sum(not o.problems for o in rejected),
        "bad_error": sum(bool(o.problems) for o in rejected),
        "mean_error_chars": round(sum(o.length for o in rejected) / max(len(rejected), 1)),
    }


def show(outcomes: list[Outcome], only_bad: bool) -> None:
    for o in outcomes:
        needs_work = o.verdict not in ("accepted", "good") or (not o.accepted and o.want == "accept")
        if only_bad and not needs_work:
            continue
        miss = " (wanted accept)" if not o.accepted and o.want == "accept" else ""
        flags = f" [{', '.join(o.problems)}]" if o.problems else ""
        print(f"{o.verdict:9s} {o.tool}: {o.why}{miss}{flags}")
        print(f"          {o.args[:150]}")
        print(f"          -> {o.length:4d} ch  {o.text[:400]!r}" if not o.accepted else f"          -> ok  {o.text}")


def print_summary(stats: dict[str, float]) -> None:
    print("\n".join(f"{k:28s}{v:5d}" for k, v in stats.items()))


# ---- sizes ------------------------------------------------------------------------------------------------


def tokens(text: str) -> int:
    return round(len(text) / 4)


async def measure_sizes() -> dict[str, Any]:
    bench = Bench()
    sizes: dict[str, Any] = {"tools": {}, "responses": {}}
    async with Client(bench.server) as client:
        bench.client = client
        await bench.start()
        for tool in (await client.list_tools()).tools:
            description = tokens(tool.description or "")
            schema = tokens(json.dumps(tool.input_schema, separators=(",", ":")))
            sizes["tools"][tool.name] = {"description": description, "schema": schema}
        sizes["instructions"] = tokens(INSTRUCTIONS)
        probes = {
            "robots": ("robots", {}),
            "observe (no image)": ("observe", {"include_image": False}),
            "list_templates": ("list_templates", {}),
            "show_world": ("show_world", {}),
            "add_robot": ("add_robot", {"name": "pup", "template": "quadruped", "pos": [2.0, 0.0]}),
            "move (1 s)": ("move", {"robot": "rover", "speed": 0.2, "seconds": 1}),
            "list_library": ("list_library", {}),
            "describe_model": ("describe_model", {}),
            "add_object": ("add_object", {"name": "b2", "pos": [0.4, 0.1]}),
        }
        for label, (tool, args) in probes.items():
            ok, text = await bench.call(tool, args)
            if not ok:
                raise SystemExit(f"{label}: {text}")
            sizes["responses"][label] = tokens(text)
            if tool in ("add_robot", "add_object", "move"):
                bench.restore()
    return sizes


def print_sizes(sizes: dict[str, Any]) -> None:
    tools = sizes["tools"]
    print(f"{'tool':24s}{'description':>12s}{'schema':>8s}")
    for name, t in tools.items():
        print(f"{name:24s}{t['description']:12d}{t['schema']:8d}")
    total_d = sum(t["description"] for t in tools.values())
    total_s = sum(t["schema"] for t in tools.values())
    print(f"{'TOTAL (' + str(len(tools)) + ' tools)':24s}{total_d:12d}{total_s:8d}")
    print(f"\nserver instructions: {sizes['instructions']} tokens")
    print("typical responses (tokens):")
    for label, n in sizes["responses"].items():
        print(f"  {label:22s}{n:6d}")
    print(f"\ndescriptions + schemas + instructions: {total_d + total_s + sizes['instructions']} tokens")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tool", help="only cases for tools whose name contains this")
    parser.add_argument("--bad", action="store_true", help="print only the cases that need work")
    parser.add_argument("--json", type=Path, help="also write the results here")
    parser.add_argument("--sizes", action="store_true", help="measure token sizes instead of running the battery")
    options = parser.parse_args(argv)
    try:
        if options.sizes:
            sizes = asyncio.run(measure_sizes())
            print_sizes(sizes)
            data: Any = sizes
        else:
            outcomes = asyncio.run(run_battery(options.tool))
            show(outcomes, options.bad)
            stats = summarize(outcomes)
            print()
            print_summary(stats)
            data = {"summary": stats, "cases": [asdict(o) | {"verdict": o.verdict} for o in outcomes]}
        if options.json:
            options.json.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    finally:
        shutil.rmtree(HOME, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
