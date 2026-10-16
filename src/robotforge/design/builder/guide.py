"""The builder cheat-sheet agents read before building (the builder_guide tool)."""

from __future__ import annotations

from typing import Any

from robotforge.design import components
from robotforge.design.builder.schema import ANCHORS, SIZE_HINT

EXAMPLE_DOG: dict[str, Any] = {
    "color": "orange",
    "base": {"shape": "box", "size": [0.3, 0.14, 0.06], "sensors": ["imu"]},
    "parts": [
        {
            "name": "left_hip",
            "at": [0.12, 0.07, 0],
            "joint": {"axis": "x", "range": [-35, 35]},
            "shape": "none",
            "mirror": True,
            "repeat": {"count": 2, "along": "back", "spacing": 0.24},
        },
        {
            "name": "left_thigh",
            "parent": "left_hip",
            "at": [0, 0.035, 0],
            "joint": {"axis": "y", "range": [-70, 140], "home": 46},
            "size": [0.022, 0.12],
            "direction": "down",
        },
        {
            "name": "left_shin",
            "parent": "left_thigh",
            "at": "end",
            "joint": {"axis": "y", "range": [-150, -15], "home": -92},
            "size": [0.018, 0.12],
            "direction": "down",
            "foot": True,
            "sensors": ["touch"],
        },
    ],
}


def guide() -> dict[str, Any]:
    parts = {
        n: {"kind": a.kind, "strength": a.stall_torque, "speed": a.no_load_speed, "note": a.note}
        for n, a in components.ACTUATORS.items()
    }
    return {
        "how": (
            "add_robot(name, build={...}) creates a robot from a base and a list of parts; add_part, update_part "
            "and remove_part change it later. Units: meters, kg, degrees. The robot's front is +x, left +y, up "
            "+z. Every part has a frame at its joint; its numbers are in its parent's frame."
        ),
        "robot": f"color ({components.COLOR_HINT}), material (pla, aluminum, carbon, rubber), servo / motor / linear "
        "(default actuators), servo_caps (draw servos), base, parts",
        "base": "shape box|sphere|capsule|cylinder, size, axis (capsule/cylinder), color, material, mass, "
        "fixed (true = bolted to the floor), mounts {name: [x,y,z]}, sensors",
        "part": {
            "name": "lowercase_with_underscores; also names its joint and actuator",
            "parent": "base (default) or another part's name",
            "at": f"where on the parent: [x, y, z] or an anchor: {', '.join(ANCHORS)}, or a mount name",
            "shift": "[x, y, z] added to at",
            "rotate": "[roll, pitch, yaw] degrees, the part frame relative to the parent",
            "joint": 'hinge (default) | slide | fixed, or {"type", "axis": x|y|z|-x|forward|left|up|[x,y,z], '
            '"range": [low, high] degrees (meters for slide), "home", "continuous", "damping", "stiffness"}',
            "actuator": 'catalog name (default: robot servo; wheels: robot motor), "none" for a passive joint, '
            'or {"model", "control": position|velocity|torque, "gear"}',
            "shape": f"{', '.join(SIZE_HINT)}; size per shape: {SIZE_HINT}",
            "direction": "the link grows from its joint this way (down for a leg); unset = centered on the joint",
            "offset, color, material, mass, friction, collide": 'look and physics of the shape; color "main" '
            "follows the robot's color, so recoloring the robot recolors the part",
            "foot": "true adds a rubber ball at the end",
            "sensors": 'imu, touch, camera, rangefinder, lidar, or {"type", "at", "direction", "fov", "rays", "range"}',
            "mirror": "true adds a copy on the other side of the parent (left <-> right in names and positions)",
            "repeat": '{"count", "around": axis} ring | {"count", "along": axis, "spacing"} row | '
            '{"count", "chain": true} each copy hangs on the previous one (snakes, tentacles); names get _1, _2...',
            "robot": '{"template": "arm"|"gripper"|..., "params": {...}} or {"library": "piper"}: mount a whole '
            'robot on this part (an arm on a rover, a gripper on an arm built from parts); "home": {joint: degrees} '
            "starts it in another pose (an arm folded for driving)",
        },
        "behaviors": (
            "Built robots get sliders and set_controls like any robot, and the built-in behaviors when their shape "
            "fits: 4 or 6 legs of 3 hinges walk (move, WASD); speed-driven wheels on both sides drive; exactly two "
            "wheels on one axle balance by themselves (LQR); a chain of 4+ side-to-side servos slithers; a mounted "
            "arm or gripper works with move_arm / pick_and_place (robot name 'rover' or 'rover/arm')."
        ),
        "designs": (
            "To start from something bigger: add_robot(name, design=...) adds a ready-made robot from the gallery "
            "(list_templates lists them: a wheel-legged biped, a Mars rover, a centaur, a Go2 with an arm...) as "
            "parts in this schema; show_world shows them, and add_part / update_part change them like any build."
        ),
        "checks": (
            "Each build is dropped onto a floor before it is saved: overlapping parts or parts in the floor are "
            "refused with a fix, too-weak servos get a warning in notes, the spawn height is automatic."
        ),
        "actuators": parts,
        "example": {"name": "dog", "build": EXAMPLE_DOG},
    }
