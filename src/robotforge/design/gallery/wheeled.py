"""Gallery designs that roll: a wheel-legged biped, a robot dog on wheels, rovers and mobile manipulators."""

from __future__ import annotations

from typing import Any

from robotforge.design.gallery.base import GOLD, GRAPHITE, IVORY, STEEL, Design, tube


def wheel_biped() -> dict[str, Any]:
    """Two legs that end in driven wheels: it balances on them like a Segway while its servo legs hold a crouch.

    The wheel axle sits right under the hips at the home crouch (thigh back 40 degrees, knee forward 80), so the
    body's weight is over the wheels; the balancer works out the rest. Each wheel hangs outboard of its shin.
    """
    return {
        "color": "orange",
        "motor": "jga25",
        # A drum lying across the robot, its hips at the two ends.
        "base": {"shape": "cylinder", "size": [0.13, 0.17], "axis": "y", "sensors": ["imu"]},
        "parts": [
            {  # a dark face plate with two cameras for eyes
                "name": "visor",
                "at": "front",
                "joint": "fixed",
                "shape": "box",
                "size": [0.012, 0.12, 0.05],
                "direction": "forward",
                "offset": [-0.004, 0, 0.008],
                "color": "#15191f",
                "sensors": [
                    {"type": "camera", "name": "left_eye", "at": [0.012, 0.03, 0.008]},
                    {"type": "camera", "name": "right_eye", "at": [0.012, -0.03, 0.008]},
                ],
            },
            {
                "name": "left_thigh",
                "at": [0, 0.105, -0.01],
                "joint": {"axis": "y", "range": [-30, 100], "home": 40},
                "size": [0.03, 0.12],
                "direction": "down",
                "color": GRAPHITE,
                "mirror": True,
            },
            {
                "name": "left_shin",
                "parent": "left_thigh",
                "at": "end",
                "joint": {"axis": "y", "range": [-150, 0], "home": -80},
                "size": [0.025, 0.12],
                "direction": "down",
                "color": GRAPHITE,
            },
            {
                "name": "left_wheel",
                "parent": "left_shin",
                "at": "end",
                "shift": [0, 0.032, 0],
                "shape": "wheel",
                "size": [0.13, 0.03],
            },
        ],
    }


def lift_picker() -> dict[str, Any]:
    """A warehouse picker: a driving base, a telescoping column that lifts, and an arm with a gripper on top.

    The column is a belt-driven slide inside a fixed sleeve; move_arm and pick_and_place raise and lower it together
    with the arm, so it reaches tables from coffee-table to counter height (not the floor: the arm cannot bend down
    past its own base). The arm sits at the front, where it reaches furthest ahead, and rides facing backward,
    folded over the base, so nothing sticks out in front while it drives. Its joints are the strong modules: on
    hobby servos the stretched-out arm sagged 10 cm with a cube in its hand.
    """
    return {
        "color": "#2f6fd6",
        "motor": "jga25",
        "linear": "belt_rail",
        "base": {
            "shape": "box",
            "size": [0.44, 0.3, 0.09],
            "mass": 8.0,  # batteries low in the chassis keep it on its wheels while the arm reaches out
            "sensors": [
                "imu",
                {"type": "lidar", "at": [-0.17, 0, 0.045], "rays": 16},
                {"type": "camera", "at": "front"},
            ],
        },
        "parts": [
            {
                "name": "left_wheel",
                "at": [0, 0.168, -0.025],
                "shape": "wheel",
                "size": [0.13, 0.034],
                # Geared down 3:1 the wheels hold the base still while the arm swings and still drive at 0.6 m/s.
                "actuator": {"model": "jga25", "gear": 3},
                "mirror": True,
            },
            {"name": "front_caster", "at": [0.17, 0, -0.07], "shape": "caster", "size": [0.04]},
            {"name": "rear_caster", "at": [-0.17, 0, -0.07], "shape": "caster", "size": [0.04]},
            {  # the outer tube of the column, for looks; the lift slides inside it
                "name": "sleeve",
                "at": "top",
                "shift": [0.1, 0, 0],
                "joint": "fixed",
                "shape": "box",
                "size": [0.11, 0.11, 0.3],
                "direction": "up",
                "color": GRAPHITE,
                "collide": False,
            },
            {
                "name": "lift",
                "at": "top",
                "shift": [0.1, 0, 0.002],
                "joint": {"type": "slide", "axis": "z", "range": [0, 0.26]},  # 4 cm stay in the sleeve
                # 2:1 gives 80 N for the 2 kg it lifts and 0.25 m/s, so it keeps up with the arm (at 3:1 it lagged
                # behind and the arm carried the cube low, into the table it was going to).
                "actuator": {"model": "belt_rail", "gear": 2},
                "shape": "box",
                "size": [0.085, 0.085, 0.31],
                "mass": 0.4,  # an aluminium tube, not a solid block
                "direction": "up",
                "color": STEEL,
            },
            {
                "name": "arm",
                "parent": "lift",
                "at": "end",
                "robot": {
                    "template": "arm",
                    "params": {"reach": 0.6, "base_height": 0.04, "servo": "bldc_joint", "color": "#2f6fd6"},
                    "home": {"yaw": 180},
                },
            },
        ],
    }


def mobile_panda() -> dict[str, Any]:
    """A research mobile manipulator: a real Franka Emika Panda (Menagerie, Apache-2.0) bolted to a heavy four-wheel
    base, as labs mount it on a Ridgeback or a Husky.

    The arm weighs about 17 kg and reaches 0.85 m, so the base is low and heavy (40 kg of batteries) and its geared
    wheels hold it still while the arm swings. The Panda starts in Menagerie's own ready pose, gripper down in front.
    """
    wheel = {"shape": "wheel", "size": [0.16, 0.06], "actuator": {"model": "jga25", "gear": 4}}
    return {
        "color": "#d4d7dc",
        "motor": "jga25",
        "base": {
            "shape": "box",
            "size": [0.66, 0.46, 0.16],
            "mass": 40.0,
            "sensors": [
                "imu",
                {"type": "lidar", "at": [0.3, 0, 0.08], "rays": 24},
                {"type": "camera", "at": "front"},
            ],
        },
        "parts": [
            {"name": "left_front_wheel", "at": [0.2, 0.27, -0.06], "mirror": True, **wheel},
            {"name": "left_rear_wheel", "at": [-0.2, 0.27, -0.06], "mirror": True, **wheel},
            {
                "name": "bumper",
                "at": "front",
                "joint": "fixed",
                "shape": "box",
                "size": [0.03, 0.4, 0.06],
                "direction": "forward",
                "offset": [0, 0, -0.03],
                "color": GRAPHITE,
            },
            {
                "name": "deck",
                "at": "top",
                "joint": "fixed",
                "shape": "box",
                "size": [0.5, 0.36, 0.012],
                "direction": "up",
                "color": GRAPHITE,
            },
            {"name": "arm", "parent": "deck", "at": "end", "shift": [0.1, 0, 0], "robot": {"library": "panda"}},
        ],
    }


def mars_rover() -> dict[str, Any]:
    """Six wheels on a rocker-bogie, like the Mars rovers: each side's rocker and bogie pivot freely, so all six wheels
    keep the ground over rocks and steps. A camera mast with a pan-tilt head rides on the deck, and a sampling arm
    on the front of the body reaches the ground.

    Points are where the parts meet, in the body's frame (meters): the rocker pivots at P, carries the front wheel at F
    and the bogie's pivot at B; the bogie carries the middle and rear wheels at M and R. A spring on each rocker holds
    the body level, standing in for the differential bar of a real rover.
    """
    y = 0.13  # the suspension's plane, just outside the body
    P, F, B = (0.02, y, -0.02), (0.17, y, -0.115), (-0.1, y, -0.06)
    M, R = (-0.02, y, -0.115), (-0.19, y, -0.115)
    hub = [0, 0.042, 0]  # wheels hang outboard of the tubes
    wheel = {"shape": "wheel", "size": [0.11, 0.05], "actuator": {"model": "jga25", "gear": 4}}

    def rel(point, origin):
        return [p - o for p, o in zip(point, origin, strict=True)]

    return {
        "color": GOLD,
        "motor": "jga25",
        "base": {
            "shape": "box",
            "size": [0.3, 0.2, 0.07],
            "color": IVORY,
            "mass": 2.5,
            "sensors": ["imu", {"type": "camera", "name": "hazcam", "at": "front", "direction": [1, 0, -0.5]}],
        },
        "parts": [
            # Suspension, left side; mirror makes the right one. Rocker and bogie are passive hinges.
            tube(
                "left_rocker",
                P,
                F,
                0.024,
                joint={"axis": "y", "range": [-30, 30], "stiffness": 6, "damping": 0.3},
                actuator="none",
                color=STEEL,
                mirror=True,
            ),
            tube("left_rocker_arm", [0, 0, 0], rel(B, P), 0.024, parent="left_rocker", joint="fixed", color=STEEL),
            {"name": "left_front_wheel", "parent": "left_rocker", "at": "end", "shift": hub, **wheel},
            tube(
                "left_bogie",
                rel(B, P),
                rel(M, P),
                0.02,
                parent="left_rocker",
                joint={"axis": "y", "range": [-35, 35], "damping": 0.1},
                actuator="none",
                color=STEEL,
            ),
            tube("left_bogie_arm", [0, 0, 0], rel(R, B), 0.02, parent="left_bogie", joint="fixed", color=STEEL),
            {"name": "left_middle_wheel", "parent": "left_bogie", "at": "end", "shift": hub, **wheel},
            {"name": "left_rear_wheel", "parent": "left_bogie_arm", "at": "end", "shift": hub, **wheel},
            # Deck: the camera mast with a pan-tilt head, a power unit at the back, the arm at the front.
            {
                "name": "mast",
                "at": "top",
                "shift": [0.1, 0.06, 0],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [0.028, 0.24],
                "direction": "up",
                "color": IVORY,
            },
            {
                "name": "mast_pan",
                "parent": "mast",
                "at": "end",
                "joint": {"axis": "z", "range": [-170, 170]},
                "actuator": "sg90",
                "shape": "cylinder",
                "size": [0.034, 0.02],
                "direction": "up",
                "color": GRAPHITE,
            },
            {
                "name": "mast_head",
                "parent": "mast_pan",
                "at": "end",
                "shift": [0, 0, 0.022],
                "joint": {"axis": "y", "range": [-60, 60]},
                "actuator": "sg90",
                "shape": "box",
                "size": [0.05, 0.11, 0.035],
                "color": IVORY,
                "sensors": [
                    {"type": "camera", "name": "left_navcam", "at": [0.025, 0.03, 0]},
                    {"type": "camera", "name": "right_navcam", "at": [0.025, -0.03, 0]},
                ],
            },
            {
                "name": "power_unit",
                "at": "back",
                "shift": [0.0, 0, 0.02],
                "rotate": [0, -40, 0],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [0.06, 0.13],
                "direction": "back",
                "color": GRAPHITE,
            },
            {
                "name": "antenna_post",
                "at": "top",
                "shift": [-0.07, -0.06, 0],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [0.012, 0.035],
                "direction": "up",
                "color": STEEL,
            },
            {  # the high-gain dish, tipped toward the sky
                "name": "antenna",
                "parent": "antenna_post",
                "at": "end",
                "rotate": [0, -25, 0],
                "joint": "fixed",
                "shape": "cylinder",
                "size": [0.075, 0.008],
                "direction": "up",
                "color": "main",
            },
            # The sampling arm, built from parts on the front of the body so it reaches down to the ground (a template
            # arm on the deck cannot bend below its own base). Stowed upright and folded between the front wheels.
            {
                "name": "arm",
                "at": "front",
                "shift": [0, -0.05, 0],
                "joint": {"axis": "z", "range": [-100, 100]},
                "shape": "cylinder",
                "size": [0.034, 0.035],
                "direction": "forward",
                "color": GRAPHITE,
            },
            {
                "name": "arm_upper",
                "parent": "arm",
                "at": "end",
                "joint": {"axis": "y", "range": [-100, 60], "home": -85},
                "size": [0.024, 0.18],
                "direction": "forward",
                "color": IVORY,
            },
            {
                "name": "arm_fore",
                "parent": "arm_upper",
                "at": "end",
                "joint": {"axis": "y", "range": [-10, 165], "home": 160},
                "size": [0.02, 0.16],
                "direction": "forward",
                "color": IVORY,
            },
            {
                "name": "arm_wrist",
                "parent": "arm_fore",
                "at": "end",
                "joint": {"axis": "y", "range": [-120, 120], "home": 15},
                "size": [0.02, 0.03],
                "direction": "forward",
                "color": GRAPHITE,
            },
            {
                "name": "hand",
                "parent": "arm_wrist",
                "at": "end",
                "rotate": [0, 90, 0],  # the gripper's fingers point along its +z: turn them along the wrist
                "robot": {"template": "gripper", "params": {"color": GOLD}},
            },
        ],
    }


def wheel_dog() -> dict[str, Any]:
    """A robot dog whose four legs end in driven wheels instead of feet, like a wheeled ANYmal: it rolls like a
    rover while its servo legs hold a stance, and set_controls on the legs makes it crouch, stand tall or lean.

    One leg is written and mirrored to the other side and repeated to the back; each wheel hangs outboard of its shin.
    """
    return {
        "color": "#f2b134",
        "motor": "jga25",
        "base": {
            "shape": "box",
            "size": [0.34, 0.16, 0.08],
            "mass": 1.6,
            "sensors": ["imu", {"type": "camera", "at": "front"}, {"type": "lidar", "at": "top", "rays": 16}],
        },
        "parts": [
            {
                "name": "left_hip",
                "at": [0.13, 0.08, 0],
                "joint": {"axis": "x", "range": [-30, 30]},
                "shape": "none",
                "mirror": True,
                "repeat": {"count": 2, "along": "back", "spacing": 0.26},
            },
            {
                "name": "left_thigh",
                "parent": "left_hip",
                "at": [0, 0.03, 0],
                "joint": {"axis": "y", "range": [-60, 100], "home": 35},
                "size": [0.03, 0.13],
                "direction": "down",
                "color": GRAPHITE,
            },
            {
                "name": "left_shin",
                "parent": "left_thigh",
                "at": "end",
                "joint": {"axis": "y", "range": [-140, 0], "home": -70},
                "size": [0.026, 0.13],
                "direction": "down",
                "color": GRAPHITE,
            },
            {
                "name": "left_wheel",
                "parent": "left_shin",
                "at": "end",
                "shift": [0, 0.034, 0],
                "shape": "wheel",
                "size": [0.1, 0.03],
                "actuator": {"model": "jga25", "gear": 2},
            },
        ],
    }


DESIGNS = [
    Design(
        "wheel_biped",
        "two legs ending in driven wheels: balances by itself and drives with move; set_controls on its "
        "thighs and shins changes the crouch",
        wheel_biped,
    ),
    Design(
        "lift_picker",
        "driving base with a lifting column and an arm with a gripper: move drives it; pick_and_place raises the "
        "column as it needs to, so it picks off the floor or a table",
        lift_picker,
    ),
    Design(
        "mobile_panda",
        "a real Franka Emika Panda arm (MuJoCo Menagerie, downloaded on first use) on a heavy four-wheel base: "
        "move drives it, move_arm and pick_and_place work mobile_panda/arm",
        mobile_panda,
    ),
    Design(
        "mars_rover",
        "six-wheel rocker-bogie rover with a camera mast and a small arm: move drives it over rough ground, "
        "move_arm and pick_and_place work its sampling arm (mars_rover/hand)",
        mars_rover,
    ),
    Design(
        "wheel_dog",
        "a robot dog with driven wheels for feet: move drives it like a rover; set_controls on its legs changes "
        "its stance",
        wheel_dog,
    ),
]
