"""Mobility discovery follows runtime bindings without constructing runtime controllers."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from robotforge.behaviors import (
    Kind,
    balance,
    drive,
    flight,
    home_pose,
    humanoid,
    marine,
    quadruped,
    rhythm_walk,
    snake,
)
from robotforge.behaviors.mobility import describe_mobility
from robotforge.design.world import World, compile_world
from robotforge.sim.water import hydro_for

BALANCER = {
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.07, 0.14, 0.22], "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [0, 0.095, -0.11], "shape": "wheel", "size": [0.12, 0.025], "mirror": True},
        {"name": "head", "at": "top", "joint": "fixed", "shape": "sphere", "size": [0.07], "direction": "up"},
    ],
}
SNAKE = {
    "servo": "mg996r",
    "base": {"shape": "box", "size": [0.09, 0.06, 0.045]},
    "parts": [
        {
            "name": "segment",
            "at": "back",
            "joint": {"axis": "z", "range": [-70, 70]},
            "shape": "box",
            "size": [0.08, 0.05, 0.04],
            "direction": "back",
            "repeat": {"count": 8, "chain": True},
        }
    ],
}


HEXAPOD = {
    "color": "teal",
    "base": {"shape": "cylinder", "size": [0.26, 0.05]},
    "parts": [
        {
            "name": "coxa",
            "at": [0.1126, 0.065, 0],
            "rotate": [0, 0, 30],
            "joint": {"axis": "z", "range": [-35, 35]},
            "shape": "box",
            "size": [0.05, 0.03, 0.025],
            "direction": "forward",
            "repeat": {"count": 6, "around": "z"},
        },
        {
            "name": "femur",
            "parent": "coxa",
            "at": "end",
            "joint": {"range": [-80, 80], "home": -20},
            "size": [0.02, 0.08],
            "direction": "forward",
        },
        {
            "name": "tibia",
            "parent": "femur",
            "at": "end",
            "joint": {"range": [0, 150], "home": 110},
            "size": [0.016, 0.12],
            "direction": "forward",
            "foot": True,
        },
    ],
}


def _world(kind: Kind, imported: bool = False, wet: bool = False, params: dict | None = None):
    robot = {"name": "bot"}
    if kind in (Kind.BALANCER, Kind.SNAKE, Kind.HEXAPOD):
        robot["build"] = {Kind.BALANCER: BALANCER, Kind.SNAKE: SNAKE, Kind.HEXAPOD: HEXAPOD}[kind]
    else:
        robot["template"] = "drone" if kind == Kind.AIRCRAFT else kind.value
        robot["params"] = params or {}
    compiled = compile_world(World.model_validate({"robots": [robot], **({"water": 1.0} if wet else {})}))
    if imported:
        # Remove only the template dispatch markers; retain the same kinematics and actuators.
        for site in compiled.spec.sites:
            if site.name in ("bot/front_left_foot", "bot/left_foot"):
                old_name = site.name
                site.name += "_imported"
                for sensor in compiled.spec.sensors:
                    if sensor.objname == old_name:
                        sensor.objname = site.name
        if kind == Kind.ROVER:
            for act in compiled.spec.actuators:
                if act.name in ("bot/left_wheel", "bot/right_wheel"):
                    act.name += "_imported"
        model = compiled.spec.compile()
    else:
        model = compiled.model
    home = home_pose(model)
    data = mujoco.MjData(model)
    data.qpos[:] = home
    mujoco.mj_forward(model, data)
    return model, home, data, hydro_for(model)


def _controller(kind, model, home, data, water, imported=False):
    if kind in (Kind.QUADRUPED, Kind.HEXAPOD):
        return quadruped.for_robot(model, "bot", home)
    if kind == Kind.HUMANOID:
        factory = humanoid.for_robot if imported else humanoid.for_template
        return factory(model, "bot", home)
    if kind == Kind.ROVER:
        return drive.DiffDrive(model, "bot")
    if kind == Kind.BALANCER:
        return balance.WheelBalancer(model, "bot", home)
    if kind == Kind.SNAKE:
        return snake.SnakeGait(model, "bot", home)
    if kind in (Kind.BOAT, Kind.ROV):
        return marine.pilot_for(kind, model, "bot", water, data)
    return flight.FlightController(model, "bot", home)


def _forbidden(*args, **kwargs):
    raise AssertionError("mobility discovery constructed a runtime controller or solver")


@pytest.mark.parametrize(
    "kind, imported, wet",
    [
        (Kind.QUADRUPED, False, False),
        (Kind.QUADRUPED, True, False),
        (Kind.HEXAPOD, False, False),
        (Kind.HUMANOID, False, False),
        (Kind.HUMANOID, True, False),
        (Kind.ROVER, False, False),
        (Kind.ROVER, True, False),
        (Kind.BALANCER, False, False),
        (Kind.SNAKE, False, False),
        (Kind.BOAT, False, False),
        (Kind.BOAT, False, True),
        (Kind.ROV, False, False),
        (Kind.ROV, False, True),
        (Kind.AIRCRAFT, False, False),
    ],
)
def test_description_matches_detached_controller_without_constructing_one(kind, imported, wet, monkeypatch):
    model, home, data, water = _world(kind, imported, wet)
    expected = _controller(kind, model, home, data, water, imported).max_speed
    for module, name in (
        (quadruped, "QuadrupedGait"),
        (quadruped, "IKSolver"),
        (humanoid, "HumanoidController"),
        (humanoid, "IKSolver"),
        (humanoid, "PreviewControl"),
        (drive, "DiffDrive"),
        (balance, "WheelBalancer"),
        (balance, "equilibrium"),
        (balance, "linearize"),
        (balance, "lqr_gains"),
        (snake, "SnakeGait"),
        (marine, "BoatPilot"),
        (marine, "RovPilot"),
        (marine, "_allocation"),
        (flight, "FlightController"),
        (rhythm_walk, "RhythmWalker"),
    ):
        monkeypatch.setattr(module, name, _forbidden)
    state = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl", "xpos", "xmat", "qacc")}
    original_home = home.copy()
    model_state = {name: getattr(model, name).copy() for name in ("qpos0", "actuator_gainprm", "actuator_ctrlrange")}
    for _ in range(3):
        assert describe_mobility(model, "bot", kind, home, water, data) == pytest.approx(expected, abs=1e-12)
    for name, value in state.items():
        np.testing.assert_array_equal(getattr(data, name), value)
    for name, value in model_state.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    np.testing.assert_array_equal(home, original_home)


def test_rover_preserves_left_upper_limit_even_when_other_limits_are_lower():
    model, home, data, water = _world(Kind.ROVER)
    left, right = model.actuator("bot/left_wheel").id, model.actuator("bot/right_wheel").id
    model.actuator_ctrlrange[left] = [-2.0, 11.0]
    model.actuator_ctrlrange[right] = [-1.0, 3.0]
    controller = drive.DiffDrive(model, "bot")
    assert describe_mobility(model, "bot", Kind.ROVER, home, water, data) == 11.0 * controller.radius


@pytest.mark.parametrize("failure", ["missing", "mixed", "unlimited"])
def test_quadruped_description_preserves_actuator_rejections(failure):
    model, home, data, water = _world(Kind.QUADRUPED)
    if failure == "missing":
        model.actuator_trnid[0, 0] = model.actuator_trnid[1, 0]
    else:
        count = 1 if failure == "mixed" else model.nu
        model.actuator_biastype[:count] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[:count, 0] = 1.0
        if failure == "unlimited":
            model.actuator_ctrllimited[:] = False
    with pytest.raises(ValueError) as runtime:
        quadruped.for_robot(model, "bot", home)
    with pytest.raises(ValueError, match=re.escape(str(runtime.value))):
        describe_mobility(model, "bot", Kind.QUADRUPED, home, water, data)


def test_saved_walk_binding_validates_joint_names_without_starting_walker(monkeypatch):
    model, home, _, _ = _world(Kind.QUADRUPED)
    body = rhythm_walk.Body.find(model, "bot")
    joints = rhythm_walk.joint_names(model, body)[::-1]
    walk = rhythm_walk.SavedWalk(
        robot="bot",
        joints=joints,
        freq=2.0,
        theta=[0.0] * (3 * len(joints)),
        speed=0.2,
        distance=0.8,
        seconds=4.0,
    )
    expected = rhythm_walk.RhythmWalker(model, "bot", walk, home)
    monkeypatch.setattr(rhythm_walk, "RhythmWalker", _forbidden)
    bound, order = rhythm_walk.saved_walk_binding(model, "bot", walk)
    np.testing.assert_array_equal(bound.actuators[order], expected.actuators)
    wrong = walk.model_copy(update={"joints": ["missing", *joints[1:]]})
    with pytest.raises(ValueError, match="learn the walk again"):
        rhythm_walk.saved_walk_binding(model, "bot", wrong)


@pytest.mark.parametrize("kind", [Kind.BOAT, Kind.ROV])
def test_marine_description_uses_current_pose_and_does_not_change_hydro(kind):
    model, home, data, water = _world(kind, wet=True)
    root = marine.root_body(model, "bot")
    z = model.jnt_qposadr[model.body_jntadr[root]] + 2
    hydro_arrays = {name: value.copy() for name, value in vars(water).items() if isinstance(value, np.ndarray)}
    speeds = []
    for height in (0.5, 1.0, 4.0):
        data.qpos[z] = height
        mujoco.mj_forward(model, data)
        expected = marine.pilot_for(kind, model, "bot", water, data).max_speed
        speeds.append(describe_mobility(model, "bot", kind, home, water, data))
        assert speeds[-1] == pytest.approx(expected, abs=1e-12)
    assert len(set(speeds)) > 1
    for name, value in hydro_arrays.items():
        np.testing.assert_array_equal(getattr(water, name), value)


@pytest.mark.parametrize("kind", [Kind.QUADRUPED, Kind.HUMANOID, Kind.ROVER, Kind.BOAT, Kind.ROV, Kind.AIRCRAFT])
def test_malformed_binding_is_rejected_without_creating_controller(kind):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError):
        describe_mobility(model, "missing", kind, model.qpos0, None, None)
