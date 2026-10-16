"""Gallery designs that stand still and reach: a two-armed station and a tentacle."""

from __future__ import annotations

from typing import Any

from robotforge.design.gallery.base import GRAPHITE, IVORY, STEEL, Design, hanging_arm

TENTACLE_SEGMENTS = 8


def twin_arm() -> dict[str, Any]:
    """A torso with two arms and grippers on a turning waist, bolted to a pedestal: a two-handed workstation.

    Both arms ride on the waist, so neither arm's motion turns it (each arm reaches with its own joints); turn it by
    hand (set_controls on twin_arm/waist, or its slider) to face the arms another way.
    """
    accent = "#e07a2f"
    return {
        "color": accent,
        "base": {"shape": "cylinder", "size": [0.22, 0.36], "fixed": True, "color": GRAPHITE},
        "parts": [
            {
                "name": "waist",
                "at": "top",
                "joint": {"axis": "z", "range": [-170, 170]},
                "shape": "cylinder",
                "size": [0.2, 0.04],
                "direction": "up",
                "color": STEEL,
            },
            {
                "name": "torso",
                "parent": "waist",
                "at": "end",
                "joint": "fixed",
                "shape": "box",
                "size": [0.12, 0.2, 0.24],
                "mass": 0.6,  # a shell over the electronics, not a solid print
                "direction": "up",
                "color": IVORY,
            },
            {  # a chest plate in the robot's color
                "name": "chest",
                "parent": "torso",
                "at": [0.06, 0, 0.15],
                "joint": "fixed",
                "shape": "box",
                "size": [0.008, 0.14, 0.11],
                "direction": "forward",
                "color": "main",
            },
            {
                "name": "head",
                "parent": "torso",
                "at": "end",
                "shift": [0, 0, 0.012],
                "joint": {"axis": "z", "range": [-120, 120]},
                "actuator": "sg90",
                "shape": "box",
                "size": [0.09, 0.12, 0.07],
                "direction": "up",
                "color": "main",
                "sensors": [
                    {"type": "camera", "name": "left_eye", "at": [0.045, 0.028, 0.035]},
                    {"type": "camera", "name": "right_eye", "at": [0.045, -0.028, 0.035]},
                ],
            },
            # Hands held over a table at home, so a table in front never starts under them.
            *hanging_arm("torso", [0, 0.116, 0.2], 0.22, 0.22, IVORY, accent, home=(0, -60, -50, 110)),
        ],
    }


def tentacle() -> dict[str, Any]:
    """A soft-looking robot arm: eight short segments that bend alternately up-down and side to side, tapering toward a
    gripper, on a turret that turns it toward its target. Standing at home it curls forward like a question mark.

    A servo holds its angle only as stiffly as its gain, and eight of them add up: on hobby servos the tip sat 2 to 3 cm
    off where move_arm put it. The three lowest joints, which carry the most, are joint modules and the others have a
    2:1 gearbox, which brings it to about a centimetre, like the other arms.
    """
    parts: list[dict[str, Any]] = [
        {
            "name": "turret",
            "at": "top",
            "joint": {"axis": "z", "range": [-180, 180]},
            "shape": "cylinder",
            "size": [0.09, 0.03],
            "direction": "up",
            "color": GRAPHITE,
        }
    ]
    parent = "turret"
    curl = (0, 0, 30, 0, 40, 0, 50, 0)  # degrees at home: the up-down joints bend it forward, more toward the tip
    for i in range(TENTACLE_SEGMENTS):
        share = i / (TENTACLE_SEGMENTS - 1)
        joint = {"axis": "y" if i % 2 == 0 else "x", "range": [-75, 75], "home": curl[i]}
        segment: dict[str, Any] = {
            "name": f"segment_{i + 1}",
            "parent": parent,
            "at": "end",
            "joint": joint,
            "size": [round(0.054 - 0.028 * share, 3), round(0.078 - 0.026 * share, 3)],
            "mass": round(0.05 - 0.025 * share, 3),  # thin-walled shells
            "direction": "up",
            "color": "main" if i % 2 == 0 else "#a98ad8",
        }
        if i < 3:  # stiff joint modules where the most weight hangs on them
            segment["actuator"] = "bldc_joint"
        elif i < 7:
            segment["actuator"] = {"model": "sts3215", "gear": 2}
        parts.append(segment)
        parent = segment["name"]
    parts.append(
        {
            "name": "tip",
            "parent": parent,
            "at": "end",
            "robot": {"template": "gripper", "params": {"color": "#7a4fbf", "opening": 0.06}},
        }
    )
    return {
        "color": "#7a4fbf",
        "base": {"shape": "cylinder", "size": [0.2, 0.06], "fixed": True, "color": GRAPHITE},
        "parts": parts,
    }


DESIGNS = [
    Design(
        "twin_arm",
        "a torso with two arms and grippers on a turning waist, on a pedestal: pick_and_place with "
        "twin_arm/left_hand or twin_arm/right_hand works a table 0.15 to 0.6 m high in front",
        twin_arm,
    ),
    Design(
        "tentacle",
        "an eight-segment tentacle arm that bends every way, with a gripper at its tip: move_arm and pick_and_place "
        "work the floor around it, up to 0.2 m high",
        tentacle,
    ),
]
