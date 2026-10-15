"""The free-form builder: robots from parts, their checks and errors, and built-in behaviors on them."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from robotforge.design import builder
from robotforge.design.world import World, compile_world
from robotforge.playground import Playground, PlaygroundError
from robotforge.sim.geometry import collides, lowest_point
from robotforge.sim.session import SimSession

DOG = {
    "color": "orange",
    "base": {"shape": "box", "size": [0.30, 0.14, 0.06], "sensors": ["imu"]},
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


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    playground.new_world("lab")
    return playground


def _run(pg: Playground, seconds: float) -> None:
    pg.run_fast(lambda: False, seconds)


def _model(build: dict) -> mujoco.MjModel:
    return compile_world(World.model_validate({"robots": [{"name": "r", "build": build}]})).model


# ---- building ------------------------------------------------------------------------------------


def test_mirror_and_repeat_unroll_into_named_parts():
    model = _model(DOG)
    bodies = {model.body(i).name for i in range(model.nbody)}
    for side in ("left", "right"):
        for i in (1, 2):
            assert {f"r/{side}_hip_{i}", f"r/{side}_thigh_{i}", f"r/{side}_shin_{i}"} <= bodies
    assert model.nu == 12
    assert model.sensor("r/left_shin_1_touch").type == mujoco.mjtSensor.mjSENS_TOUCH
    assert model.sensor("r/accel").type == mujoco.mjtSensor.mjSENS_ACCELEROMETER


def test_a_mirrored_leg_is_the_reflection_of_the_original():
    model = _model(DOG)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    for i in (1, 2):
        left = data.xpos[model.body(f"r/left_shin_{i}").id]
        right = data.xpos[model.body(f"r/right_shin_{i}").id]
        assert right == pytest.approx(left * [1, -1, 1], abs=1e-9)
    # Same command, mirrored motion: both hips roll outward for a positive angle.
    left_axis = data.xaxis[model.joint("r/left_hip_1").id]
    right_axis = data.xaxis[model.joint("r/right_hip_1").id]
    assert right_axis == pytest.approx(-left_axis, abs=1e-9)
    # Pitch joints keep their axis, so knees bend the same way on both sides.
    assert data.xaxis[model.joint("r/right_shin_1").id] == pytest.approx(
        data.xaxis[model.joint("r/left_shin_1").id], abs=1e-9
    )


def test_spawn_height_puts_the_feet_on_the_floor():
    robot = builder.build_robot(DOG)
    assert 0.1 < robot.spawn_height < 0.25
    model = _model(DOG)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    lowest = min(lowest_point(model, data, g)[2] for g in range(1, model.ngeom) if collides(model, g))
    assert 0.0 < lowest < 0.005


def test_ring_repeat_spaces_copies_evenly():
    model = _model(HEXAPOD)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    angles = sorted(np.degrees(np.arctan2(*data.xpos[model.body(f"r/coxa_{i}").id][[1, 0]])) % 360 for i in range(1, 7))
    assert np.diff(angles) == pytest.approx([60] * 5, abs=1e-6)
    assert angles[0] == pytest.approx(30, abs=0.01)


def test_chain_repeat_hangs_each_link_on_the_previous_one():
    snake = {
        "base": {"shape": "box", "size": [0.08, 0.06, 0.04]},
        "parts": [
            {
                "name": "segment",
                "at": "back",
                "joint": {"axis": "z", "range": [-60, 60]},
                "shape": "box",
                "size": [0.08, 0.05, 0.035],
                "direction": "back",
                "repeat": {"count": 5, "chain": True},
            }
        ],
    }
    model = _model(snake)
    parents = [model.body(model.body_parentid[model.body(f"r/segment_{i}").id]).name for i in range(1, 6)]
    assert parents == ["r/base", "r/segment_1", "r/segment_2", "r/segment_3", "r/segment_4"]


# ---- errors a beginner can act on ------------------------------------------------------------------


def _error(build: dict) -> str:
    with pytest.raises(ValueError) as info:
        builder.build_robot(build)
    return str(info.value)


def test_unknown_parent_suggests_the_closest_name():
    bad = copy.deepcopy(DOG)
    bad["parts"][1]["parent"] = "left_hipp"
    msg = _error(bad)
    assert "no parent named 'left_hipp'" in msg and "left_hip" in msg


def test_backwards_range_is_explained_with_the_part_name():
    bad = copy.deepcopy(DOG)
    bad["parts"][2]["joint"]["range"] = [-15, -150]
    msg = _error(bad)
    assert "part 'left_shin'" in msg and "backwards" in msg


def test_home_outside_range_and_bad_sizes_are_caught():
    bad = copy.deepcopy(DOG)
    bad["parts"][1]["joint"]["home"] = 170
    assert "outside the range" in _error(bad)
    bad = copy.deepcopy(DOG)
    bad["parts"][1]["size"] = [0.02]
    assert "capsule has size [diameter, length]" in _error(bad)


def test_overlapping_parts_are_refused_with_a_fix():
    bad = copy.deepcopy(DOG)
    bad["parts"][1]["at"] = [0, -0.03, 0]  # thigh pushed back into the body
    msg = _error(bad)
    assert "overlap" in msg and "base" in msg and "collide: false" in msg


def test_weak_servos_get_a_warning():
    heavy = copy.deepcopy(DOG)
    heavy["servo"] = "sg90"
    notes = builder.build_robot(heavy).notes
    assert any("too weak" in n for n in notes)


def test_slide_joints_need_a_linear_actuator():
    bad = {
        "base": {"fixed": True},
        "parts": [
            {
                "name": "lift",
                "joint": {"type": "slide", "range": [0, 0.1]},
                "actuator": "sts3215",
                "shape": "box",
                "size": [0.05, 0.05, 0.05],
            }
        ],
    }
    assert "linear actuator" in _error(bad)


# ---- behaviors on built robots -----------------------------------------------------------------------


def test_built_quadruped_walks_with_the_generic_gait(pg):
    pg.add_robot("dog", build=DOG)
    assert pg.behaviors.robots()[0]["kind"] == "quadruped"
    start = pg.behaviors.pose("dog")
    pg.behaviors.move("dog", 0.3, 0.0)
    _run(pg, 6.0)
    end = pg.behaviors.pose("dog")
    assert end["position"][0] - start["position"][0] > 1.0
    assert end["upright"]


def test_built_hexapod_walks_with_a_tripod_gait(pg):
    pg.add_robot("spider", build=HEXAPOD)
    assert pg.behaviors.robots()[0]["kind"] == "hexapod"
    start = pg.behaviors.pose("spider")
    pg.behaviors.move("spider", 0.2, 0.0)
    _run(pg, 5.0)
    end = pg.behaviors.pose("spider")
    assert end["position"][0] - start["position"][0] > 0.6
    assert end["upright"]


def test_bad_build_leaves_the_world_unchanged(pg):
    pg.add_robot("dog", build=DOG)
    bad = copy.deepcopy(DOG)
    bad["parts"][0]["joint"]["axis"] = "sideways"
    with pytest.raises(PlaygroundError, match="not an axis"):
        pg.add_robot("dog2", build=bad, pos=[1, 0])
    assert [r.name for r in pg.current_world().robots] == ["dog"]


MOBILE_ARM = {
    "color": "yellow",
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.34, 0.22, 0.06], "mass": 3.0, "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [0, 0.13, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "front_caster", "at": [0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "rear_caster", "at": [-0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "arm", "at": "top", "shift": [0.06, 0, 0], "robot": {"template": "arm", "params": {"reach": 0.45}}},
    ],
}


def test_mobile_manipulator_drives_then_picks_with_its_mounted_arm(pg):
    pg.add_robot("mm", build=MOBILE_ARM)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.2, 0.0])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[0.95, 0.32], fixed=True)
    info = pg.behaviors.robots()[0]
    assert info["kind"] == "rover" and info["arms"] == ["mm/arm"]
    pg.behaviors.move("mm", 0.3, 0.0)
    _run(pg, 3.0)
    pg.behaviors.stop("mm")
    _run(pg, 0.5)
    assert pg.behaviors.pose("mm")["position"][0] > 0.8  # casters glide instead of braking
    # The arm plans from where the rover is now, not from where the world started.
    arm, placed = pg.behaviors.pick_and_place("mm", "cube", "bin")
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_arm_commands_say_which_arm_when_a_robot_has_none(pg):
    pg.add_robot("dog", build=DOG)
    with pytest.raises(ValueError, match="not an arm, and carries none"):
        pg.behaviors.move_arm("dog", [0.3, 0, 0.1])


BALANCER = {
    "color": "red",
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.07, 0.14, 0.22], "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [0, 0.095, -0.11], "shape": "wheel", "size": [0.12, 0.025], "mirror": True},
        {"name": "head", "at": "top", "joint": "fixed", "shape": "sphere", "size": [0.07], "direction": "up"},
    ],
}


def test_two_wheeled_robot_balances_by_itself_and_drives(pg):
    pg.add_robot("seg", build=BALANCER)
    assert pg.behaviors.robots()[0]["kind"] == "balancer"
    _run(pg, 3.0)  # it starts balancing as soon as it appears
    assert pg.behaviors.pose("seg")["upright"]
    pg.behaviors.move("seg", 0.3, 0.0)
    _run(pg, 4.0)
    pg.behaviors.stop("seg")
    _run(pg, 2.0)
    end = pg.behaviors.pose("seg")
    assert end["upright"] and end["position"][0] > 0.8


def test_balance_task_compares_lqr_with_doing_nothing(pg):
    from robotforge.control.compare import compare_methods

    pg.add_robot("seg", build=BALANCER)
    s = pg.session
    comparison = compare_methods(
        "balance", ["lqr", "none"], 4.0, model=s.model, robot="seg", data=copy.copy(s.data), seed=0, push_at=1.5
    )
    lqr, none = comparison.results
    assert lqr.success and not lqr.failed
    assert none.failed  # without control it falls over within a second


SNAKE = {
    "color": "green",
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
        },
        {
            "name": "left_wheel",
            "parent": "segment",
            "at": [-0.04, 0.031, -0.004],
            "shape": "wheel",
            "size": [0.036, 0.008],
            "actuator": "none",
            "mirror": True,
        },
    ],
}


def test_snake_slithers_forward_and_steers(pg):
    pg.add_robot("snake", build=SNAKE)
    assert pg.behaviors.robots()[0]["kind"] == "snake"  # passive wheels do not make it a rover
    start = pg.behaviors.pose("snake")
    pg.behaviors.move("snake", 0.2, 0.0)
    _run(pg, 6.0)
    end = pg.behaviors.pose("snake")
    assert end["position"][0] - start["position"][0] > 0.6
    assert abs(end["position"][1] - start["position"][1]) < 0.3
    pg.behaviors.move("snake", 0.15, 0.4)
    before = pg.behaviors.pose("snake")["heading_deg"]
    _run(pg, 4.0)
    turned = (pg.behaviors.pose("snake")["heading_deg"] - before + 180) % 360 - 180
    assert turned > 30  # positive turn = left, like every other robot
