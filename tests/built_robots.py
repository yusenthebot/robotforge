"""Robots built from parts that the look tests and the screenshot script share: the builder tests' robots
plus a few heterogeneous ones (sensor rover, fork lift, carbon spider, a dog carrying an arm and a lidar)."""

from __future__ import annotations

import copy

from test_builder import BALANCER, DOG, HEXAPOD, MOBILE_ARM, SNAKE

CRANE = {
    "base": {"shape": "cylinder", "size": [0.12, 0.04], "fixed": True},
    "parts": [
        {
            "name": "turret",
            "at": "top",
            "joint": {"axis": "z", "range": [-180, 180]},
            "shape": "cylinder",
            "size": [0.08, 0.04],
            "direction": "up",
        },
        {
            "name": "upper",
            "parent": "turret",
            "at": "end",
            "joint": {"range": [-100, 100], "home": 20},
            "size": [0.03, 0.18],
            "direction": "up",
        },
        {
            "name": "fore",
            "parent": "upper",
            "at": "end",
            "joint": {"range": [-150, 150], "home": 80},
            "size": [0.025, 0.16],
            "direction": "up",
        },
        {
            "name": "wrist",
            "parent": "fore",
            "at": "end",
            "joint": {"range": [-150, 150], "home": 80},
            "size": [0.022, 0.05],
            "direction": "up",
        },
        {"name": "hand", "parent": "wrist", "at": "end", "robot": {"template": "gripper"}},
    ],
}

SCOUT = {  # aluminium rover: four wheels, a lidar mast, a pan-tilt camera head and a range finder
    "color": "blue",
    "material": "aluminum",
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.36, 0.22, 0.08], "sensors": ["imu", {"type": "rangefinder", "at": "front"}]},
    "parts": [
        {
            "name": "left_wheel",
            "at": [0.11, 0.14, -0.02],
            "shape": "wheel",
            "size": [0.12, 0.04],
            "mirror": True,
            "repeat": {"count": 2, "along": "back", "spacing": 0.22},
        },
        {
            "name": "mast",
            "at": "top",
            "shift": [-0.12, 0, 0],
            "joint": "fixed",
            "shape": "cylinder",
            "size": [0.03, 0.16],
            "direction": "up",
            "color": "gray",
            "sensors": [{"type": "lidar", "at": "end"}],
        },
        {
            "name": "pan",
            "at": "top",
            "shift": [0.1, 0, 0],
            "joint": {"axis": "z", "range": [-90, 90]},
            "shape": "cylinder",
            "size": [0.05, 0.03],
            "direction": "up",
            "color": "white",
        },
        {
            "name": "tilt",
            "parent": "pan",
            "at": "end",
            "shift": [0, 0, 0.03],
            "joint": {"axis": "y", "range": [-45, 45]},
            "shape": "box",
            "size": [0.06, 0.1, 0.04],
            "sensors": [{"type": "camera", "at": "front"}],
        },
    ],
}

LIFT = {  # a fork lift: a carriage on a belt rail up an aluminium mast
    "color": "orange",
    "motor": "jga25",
    "linear": "belt_rail",
    "base": {"shape": "box", "size": [0.3, 0.2, 0.06], "mass": 2.0, "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [-0.06, 0.125, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "front_caster", "at": [0.12, 0, -0.04], "shape": "caster", "size": [0.04]},
        {
            "name": "mast",
            "at": "top",
            "shift": [0.1, 0, 0],
            "joint": "fixed",
            "shape": "box",
            "size": [0.04, 0.16, 0.3],
            "direction": "up",
            "color": "gray",
            "material": "aluminum",
        },
        {
            "name": "carriage",
            "parent": "mast",
            "at": "front",
            "shift": [0.025, 0, -0.08],
            "joint": {"type": "slide", "axis": "z", "range": [0, 0.15]},
            "shape": "box",
            "size": [0.02, 0.14, 0.08],
        },
        {
            "name": "left_fork",
            "parent": "carriage",
            "at": [0.01, 0.045, -0.035],
            "joint": "fixed",
            "shape": "box",
            "size": [0.14, 0.02, 0.01],
            "direction": "forward",
            "color": "black",
            "mirror": True,
        },
    ],
}

SPIDER = copy.deepcopy(HEXAPOD)  # the hexapod in carbon fibre, red coxae, aluminium femurs, Dynamixel servos
SPIDER.update({"color": "black", "material": "carbon", "servo": "xl430"})
SPIDER["base"]["sensors"] = ["imu", {"type": "camera", "at": "front"}]
SPIDER["parts"][0]["color"] = "red"
SPIDER["parts"][1].update({"material": "aluminum", "color": "gray"})

CENTAUR = {  # spliced: a dog's legs, an arm template on its back and a lidar mast
    "color": "purple",
    "servo": "xl430",
    "base": {"shape": "box", "size": [0.32, 0.15, 0.07], "sensors": ["imu", {"type": "camera", "at": "front"}]},
    "parts": [
        *copy.deepcopy(DOG["parts"]),
        {
            "name": "arm",
            "at": "top",
            "shift": [0.04, 0, 0],
            "robot": {"template": "arm", "params": {"dof": 4, "reach": 0.3, "color": "white"}},
        },
        {
            "name": "mast",
            "at": "top",
            "shift": [-0.12, 0, 0],
            "joint": "fixed",
            "shape": "capsule",
            "size": [0.02, 0.08],
            "direction": "up",
            "sensors": [{"type": "lidar", "at": "end", "rays": 8}],
        },
    ],
}

GALLERY: dict[str, dict] = {
    "dog": DOG,
    "hexapod": HEXAPOD,
    "mobile_arm": MOBILE_ARM,
    "balancer": BALANCER,
    "snake": SNAKE,
    "crane": CRANE,
    "scout": SCOUT,
    "lift": LIFT,
    "spider": SPIDER,
    "centaur": CENTAUR,
}
