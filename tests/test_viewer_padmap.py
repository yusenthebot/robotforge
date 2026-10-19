"""The gamepad's mapping (viewer/static/padmap.js) checked in Node: pure functions, so no page is needed.

The same table is what the real viewer applies to a Gamepad API pad (tests/test_viewer_gamepad.py runs that in Chrome).
Skipped without Node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "robotforge" / "viewer" / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

PROGRAM = """
import { pathToFileURL } from "node:url";
const m = await import(pathToFileURL(process.argv[1]).href);
const pad = (axes = [0, 0, 0, 0], pressed = []) => ({
  axes,
  buttons: Array.from({ length: 17 }, (_, i) => ({ pressed: pressed.includes(i), value: pressed.includes(i) ? 1 : 0 })),
});
const out = {};
const cases = JSON.parse(process.argv[2]);
for (const [name, c] of Object.entries(cases)) {
  const p = m.read(pad(c.axes, c.pressed));
  if (c.robot !== undefined) out[name] = m.commandFor(m.profileFor(c.robot), p, c.mode ?? "drive");
  else out[name] = c.fn === "shape" ? m.shape(c.value) : p;
}
console.log(JSON.stringify(out));
"""


def evaluate(cases: dict) -> dict:
    run = subprocess.run(
        [NODE, "--input-type=module", "-e", PROGRAM, str(STATIC / "padmap.js"), json.dumps(cases)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.strip().splitlines()[-1])


DOG = {"name": "dog", "kind": "quadruped", "max_speed": 0.4}
ARM = {"name": "arm", "kind": "arm"}
DOG_WITH_ARM = {"name": "go2", "kind": "quadruped", "max_speed": 0.6, "arms": ["go2/arm"]}
DRONE = {"name": "x2", "kind": "aircraft", "max_speed": 2.0}
ROV = {"name": "rov", "kind": "rov", "max_speed": 0.5}


def test_the_dead_zone_swallows_a_resting_stick_and_the_rest_is_stretched_to_the_full_range():
    got = evaluate(
        {
            "rest": {"fn": "shape", "value": 0.1},
            "edge": {"fn": "shape", "value": -0.15},
            "full": {"fn": "shape", "value": 1.0},
            "half": {"fn": "shape", "value": 0.575},
            "bad": {"fn": "shape", "value": "x"},
        }
    )
    assert got["rest"] == 0 and got["edge"] == 0 and got["full"] == 1
    assert got["half"] == pytest.approx(0.5, abs=1e-6)
    assert got["bad"] == 0


def test_a_robot_that_walks_or_drives_goes_with_the_left_stick_and_turns_with_either_stick():
    got = evaluate(
        {
            "rest": {"robot": DOG},
            "forward": {"robot": DOG, "axes": [0, -1, 0, 0]},
            "back": {"robot": DOG, "axes": [0, 1, 0, 0]},
            "left": {"robot": DOG, "axes": [0, 0, -1, 0]},
            "right_with_left_stick": {"robot": DOG, "axes": [1, 0, 0, 0]},
            "dpad_up": {"robot": DOG, "pressed": [12]},
            "arc": {"robot": DOG, "axes": [0, -0.6, -0.6, 0]},
        }
    )
    drive = lambda name: got[name]["drive"]  # noqa: E731
    assert drive("rest") == {"fwd": 0, "left": 0, "slide": 0, "up": 0}
    assert drive("forward")["fwd"] == 1 and drive("back")["fwd"] == -1
    assert drive("left")["left"] == 1  # right stick to the left turns left (positive)
    assert drive("right_with_left_stick")["left"] == -1
    assert drive("dpad_up")["fwd"] == 0.5  # half strength
    assert 0 < drive("arc")["fwd"] < 1 and 0 < drive("arc")["left"] < 1
    assert all(got[k]["gripper"] is None for k in got)


def test_an_aircraft_slides_climbs_and_an_rov_heaves():
    got = evaluate(
        {
            "slide": {"robot": DRONE, "axes": [-1, 0, 0, 0]},
            "climb": {"robot": DRONE, "axes": [0, 0, 0, -1]},
            "trigger_down": {"robot": DRONE, "axes": [0, 0, 0, 0], "pressed": [6]},
            "turn": {"robot": DRONE, "axes": [0, 0, -1, 0]},
            "heave": {"robot": ROV, "axes": [0, 0, 0, -1]},
        }
    )
    assert got["slide"]["drive"]["slide"] == 1  # left stick left: sideways to the left
    assert got["climb"]["drive"]["up"] == 1
    assert got["trigger_down"]["drive"]["up"] == -1
    assert got["turn"]["drive"]["left"] == 1
    assert got["heave"]["drive"]["up"] == 1 and got["heave"]["drive"]["slide"] == 0


def test_an_arm_is_jogged_with_the_sticks_and_its_gripper_closes_on_a_and_opens_on_b():
    got = evaluate(
        {
            "away": {"robot": ARM, "axes": [0, -1, 0, 0]},
            "right": {"robot": ARM, "axes": [1, 0, 0, 0]},
            "up": {"robot": ARM, "axes": [0, 0, 0, -1]},
            "close": {"robot": ARM, "pressed": [0]},
            "open": {"robot": ARM, "pressed": [1]},
            "both": {"robot": ARM, "pressed": [0, 1]},
            "trigger_close": {"robot": ARM, "pressed": [7]},
        }
    )
    assert got["away"]["arm"] == {"x": 0, "y": 1, "z": 0}  # up on the stick: away from the camera
    assert got["right"]["arm"]["x"] == 1 and got["right"]["arm"]["y"] == 0
    assert got["up"]["arm"]["z"] == 1
    assert got["away"]["drive"]["fwd"] == 0  # an arm does not drive
    assert got["close"]["gripper"] == "close" and got["open"]["gripper"] == "open"
    assert got["both"]["gripper"] is None and got["trigger_close"]["gripper"] == "close"


def test_a_robot_with_an_arm_drives_and_Y_switches_it_to_the_arm():
    got = evaluate(
        {
            "drive": {"robot": DOG_WITH_ARM, "axes": [0, -1, 0, 0], "mode": "drive"},
            "arm": {"robot": DOG_WITH_ARM, "axes": [0, -1, 0, 0], "mode": "arm"},
        }
    )
    assert got["drive"]["drive"]["fwd"] == 1 and got["drive"]["arm"]["y"] == 0
    assert got["arm"]["drive"]["fwd"] == 0 and got["arm"]["arm"]["y"] == 1


def test_something_that_neither_goes_nor_reaches_gets_no_command_and_values_come_in_steps_of_a_twentieth():
    got = evaluate(
        {
            "none": {"robot": {"name": "cartpole", "kind": "other"}, "axes": [0, -1, 0, 0]},
            "step": {"robot": DOG, "axes": [0, -0.6, 0, 0]},
        }
    )
    assert got["none"] == {
        "drive": {"fwd": 0, "left": 0, "slide": 0, "up": 0},
        "arm": {"x": 0, "y": 0, "z": 0},
        "gripper": None,
    }
    fwd = got["step"]["drive"]["fwd"]
    assert fwd * 20 == pytest.approx(round(fwd * 20))
