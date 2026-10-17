"""What a gallery design is, and the few helpers and colors the designs share."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

# A shared palette, so parts of one robot (and robots side by side) look like they belong together.
GRAPHITE = "#2b3038"  # dark structure: chassis, legs, masts
STEEL = "#8c939c"  # brushed metal: tubes, sliders, struts
IVORY = "#e4e0d6"  # warm white body panels
GOLD = "#c8a043"  # foil and trim


@dataclass(frozen=True)
class Design:
    name: str
    summary: str  # one line for list_templates: what it is and which tools drive it
    make: Callable[[], dict[str, Any]]  # a fresh build description (the builder's input)


def hanging_arm(
    parent: str,
    at: Sequence[float],
    upper: float,
    fore: float,
    color: str,
    grip_color: str,
    home: Sequence[float] = (0, -30, -70, 100),
    mirror: bool = True,
) -> list[dict[str, Any]]:
    """Two arms built from parts that hang from shoulders on either side: a turn about the upright axis (so the arm
    swings toward a target), shoulder and elbow pitch, a wrist and a gripper. The left arm is written once at `at`
    (meters on the parent, y > 0) and mirrored, so both stay alike when one is changed. Parts are left_arm,
    left_upper, left_fore, left_wrist, left_hand and their right_ copies; the arms answer to "<robot>/left_hand" and
    "<robot>/right_hand". `home` is [turn, shoulder, elbow, wrist] in degrees: a negative shoulder raises the arm
    forward, a negative elbow bends the forearm up, and the default holds the hand in front, pointing down. Hung from
    a part that is mirrored itself (a shoulder), pass mirror=False: the parent's copy carries the other arm."""
    turn, shoulder, elbow, wrist = home
    return [
        {
            "name": "left_arm",
            "parent": parent,
            "at": [round(v, 4) for v in at],
            "joint": {"axis": "z", "range": [-150, 150], "home": turn},
            "shape": "cylinder",
            "size": [0.036, 0.022],
            "direction": "down",
            "color": GRAPHITE,
            "mirror": mirror,
        },
        {
            "name": "left_upper",
            "parent": "left_arm",
            "at": "end",
            "joint": {"axis": "y", "range": [-170, 60], "home": shoulder},
            "size": [0.026, upper],
            "direction": "down",
            "color": color,
        },
        {
            "name": "left_fore",
            "parent": "left_upper",
            "at": "end",
            "joint": {"axis": "y", "range": [-160, 10], "home": elbow},
            "size": [0.022, fore],
            "direction": "down",
            "color": color,
        },
        {
            "name": "left_wrist",
            "parent": "left_fore",
            "at": "end",
            "joint": {"axis": "y", "range": [-135, 135], "home": wrist},
            "size": [0.02, 0.03],
            "direction": "down",
            "color": GRAPHITE,
        },
        {
            "name": "left_hand",
            "parent": "left_wrist",
            "at": "end",
            "rotate": [180, 0, 0],  # the gripper's fingers point along its +z: turn them down along the wrist
            "robot": {"template": "gripper", "params": {"color": grip_color}},
        },
    ]


def tube(name: str, start: Sequence[float], end: Sequence[float], diameter: float, **fields: Any) -> dict[str, Any]:
    """A capsule part from `start` to `end` in its parent's frame, with its direction and length worked out (rounded
    to a tenth of a millimetre, so world.yaml stays readable)."""
    d = [e - s for s, e in zip(start, end, strict=True)]
    length = math.sqrt(sum(v * v for v in d))
    return {
        "name": name,
        "at": [round(v, 4) for v in start],
        "size": [diameter, round(length, 4)],
        "direction": [round(v / length, 4) for v in d],
        **fields,
    }
