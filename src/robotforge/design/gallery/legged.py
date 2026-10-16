"""Gallery designs that walk on legs."""

from __future__ import annotations

from typing import Any

from robotforge.design.gallery.base import GRAPHITE, IVORY, Design, hanging_arm


def centaur() -> dict[str, Any]:
    """Four legs that walk on the quadruped gait, and an upper body with a head and two arms with grippers.

    The legs are the builder's dog legs, a little longer. The torso stands on the front of the body, over the front
    legs, with the shoulders low enough that the arms reach the ground in front of it.
    """
    return {
        "color": "#3a7ca5",
        # Shells, not solid blocks: printed solid, the body and torso weighed 2.8 kg and the front legs sagged.
        "base": {"shape": "box", "size": [0.34, 0.15, 0.07], "mass": 1.1, "sensors": ["imu"]},
        "parts": [
            {
                "name": "left_hip",
                "at": [0.13, 0.075, 0],
                "joint": {"axis": "x", "range": [-35, 35]},
                "shape": "none",
                "mirror": True,
                "repeat": {"count": 2, "along": "back", "spacing": 0.26},
            },
            {
                "name": "left_thigh",
                "parent": "left_hip",
                "at": [0, 0.035, 0],
                "joint": {"axis": "y", "range": [-70, 140], "home": 46},
                "size": [0.026, 0.13],
                "direction": "down",
                "color": GRAPHITE,
            },
            {
                "name": "left_shin",
                "parent": "left_thigh",
                "at": "end",
                "joint": {"axis": "y", "range": [-150, -15], "home": -92},
                "size": [0.02, 0.13],
                "direction": "down",
                "foot": True,
                "sensors": ["touch"],
            },
            {
                "name": "torso",
                "at": "top",
                "shift": [0.1, 0, 0],
                "joint": "fixed",
                "shape": "box",
                "size": [0.09, 0.13, 0.14],
                "mass": 0.35,
                "direction": "up",
                "color": IVORY,
            },
            {  # a chest plate in the robot's color
                "name": "chest",
                "parent": "torso",
                "at": [0.045, 0, 0.085],
                "joint": "fixed",
                "shape": "box",
                "size": [0.008, 0.09, 0.07],
                "direction": "forward",
                "color": "main",
            },
            {
                "name": "head",
                "parent": "torso",
                "at": "end",
                "shift": [0, 0, 0.012],
                "joint": {"axis": "z", "range": [-90, 90]},
                "actuator": "sg90",
                "shape": "sphere",
                "size": [0.075],
                "direction": "up",
                "color": IVORY,
                "sensors": [
                    {"type": "camera", "name": "left_eye", "at": [0.034, 0.016, 0.042]},
                    {"type": "camera", "name": "right_eye", "at": [0.034, -0.016, 0.042]},
                ],
            },
            *hanging_arm("torso", [0, 0.087, 0.118], upper=0.18, fore=0.18, color=IVORY, grip_color="#3a7ca5"),
        ],
    }


def scorpion() -> dict[str, Any]:
    """Six legs on the tripod gait and a tail that curls up over its back and ends in a claw (a gripper).

    The tail is an arm: it turns about an upright axis at the back of the body, then four segments bend it up, over
    and down in front of the head, where move_arm and pick_and_place use it. Its first two joints carry the rest of
    the tail stretched out, so they have a 2:1 gearbox.
    """
    tail = {"axis": "y", "range": [-120, 120]}
    return {
        "color": "#b4532a",
        "base": {"shape": "cylinder", "size": [0.26, 0.055], "sensors": ["imu", {"type": "camera", "at": "front"}]},
        "parts": [
            {
                "name": "coxa",
                "at": [0.1126, 0.065, 0],
                "rotate": [0, 0, 30],
                "joint": {"axis": "z", "range": [-35, 35]},
                "shape": "box",
                "size": [0.05, 0.03, 0.025],
                "direction": "forward",
                "color": GRAPHITE,
                "repeat": {"count": 6, "around": "z"},
            },
            {
                "name": "femur",
                "parent": "coxa",
                "at": "end",
                "joint": {"range": [-80, 80], "home": -20},
                "size": [0.022, 0.09],
                "direction": "forward",
                "color": "main",
            },
            {
                "name": "tibia",
                "parent": "femur",
                "at": "end",
                "joint": {"range": [0, 150], "home": 110},
                "size": [0.018, 0.13],
                "direction": "forward",
                "color": GRAPHITE,
                "foot": True,
            },
            {  # the tail's root stands on the back of the body, so the tail clears it when it bends forward
                "name": "tail",
                "at": "top",
                "shift": [-0.1, 0, 0],
                "joint": {"axis": "z", "range": [-120, 120]},
                "shape": "cylinder",
                "size": [0.05, 0.05],
                "direction": "up",
                "color": GRAPHITE,
            },
            {
                "name": "tail_1",
                "parent": "tail",
                "at": "end",
                "joint": {**tail, "home": -35},
                "actuator": {"model": "sts3215", "gear": 2},
                "size": [0.034, 0.15],
                "direction": "up",
                "color": "main",
            },
            {
                "name": "tail_2",
                "parent": "tail_1",
                "at": "end",
                "joint": {**tail, "home": 55},
                "actuator": {"model": "sts3215", "gear": 2},
                "size": [0.03, 0.15],
                "direction": "up",
                "color": "main",
            },
            {
                "name": "tail_3",
                "parent": "tail_2",
                "at": "end",
                "joint": {**tail, "home": 60},
                "size": [0.026, 0.13],
                "direction": "up",
                "color": "main",
            },
            {
                "name": "tail_4",
                "parent": "tail_3",
                "at": "end",
                "joint": {**tail, "home": 55},
                "size": [0.022, 0.07],
                "direction": "up",
                "color": GRAPHITE,
            },
            {
                "name": "claw",
                "parent": "tail_4",
                "at": "end",
                "robot": {"template": "gripper", "params": {"color": "#b4532a", "opening": 0.06}},
            },
        ],
    }


def go2_arm() -> dict[str, Any]:
    """Two real robots spliced together, as research labs do: a Unitree Go2 (Menagerie, BSD-3-Clause) carrying an
    AgileX PiPER arm (Menagerie, MIT) on a plate bolted to its back.

    The plate is the robot's base; the Go2 hangs under it (its trunk's top is 57 mm above its frame) and walks on the
    quadruped gait, the PiPER rides on top, folded back over the dog until it is used.
    """
    return {
        "color": GRAPHITE,
        "base": {"shape": "box", "size": [0.24, 0.13, 0.012], "mass": 0.4, "sensors": ["imu"]},
        "parts": [
            {"name": "dog", "at": [0, 0, -0.0645], "robot": {"library": "go2"}},
            {
                "name": "arm",
                "at": "top",
                "shift": [0.04, 0, 0],
                "robot": {"library": "piper", "home": {"joint2": 88, "joint3": -6, "joint5": -63}},
            },
        ],
    }


DESIGNS = [
    Design(
        "centaur",
        "four legs on the quadruped gait with a torso, a head and two arms with grippers: move walks it, move_arm "
        "and pick_and_place work centaur/left_hand and centaur/right_hand",
        centaur,
    ),
    Design(
        "scorpion",
        "six legs on the tripod gait and a tail that curls over its back to a claw: move walks it, move_arm and "
        "pick_and_place work scorpion/claw",
        scorpion,
    ),
    Design(
        "go2_arm",
        "a real Unitree Go2 carrying a real AgileX PiPER arm (both from MuJoCo Menagerie, downloaded on first use): "
        "move walks it, move_arm and pick_and_place work go2_arm/arm",
        go2_arm,
    ),
]
