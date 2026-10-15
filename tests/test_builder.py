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


def test_mirrored_slides_move_as_mirror_images():
    """Review HIGH 3: a slide axis is a direction, not a rotation axis, so a mirror must not flip it like one.
    Two mirrored slides pushed the same way should move apart (y) and both forward (x)."""
    build = {
        "base": {"fixed": True, "shape": "box", "size": [0.2, 0.2, 0.05]},
        "parts": [
            {
                "name": "left_jaw",
                "at": [0, 0.06, 0.05],
                "joint": {"type": "slide", "axis": [0.6, 0.8, 0], "range": [0, 0.05]},
                "actuator": "l16",
                "shape": "box",
                "size": [0.02, 0.02, 0.02],
                "mirror": True,
            }
        ],
    }
    model = _model(build)
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.04
    for _ in range(1500):
        mujoco.mj_step(model, data)
    left, right = data.body("r/left_jaw").xpos, data.body("r/right_jaw").xpos
    assert data.qpos[model.joint("r/left_jaw").qposadr[0]] > 0.03
    assert left[0] == pytest.approx(right[0], abs=1e-3)  # both moved forward alike
    assert left[1] == pytest.approx(-right[1], abs=1e-3) and left[1] > 0.06  # and apart, mirrored


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


def test_an_arm_carried_at_a_heading_reaches_targets_on_either_side(pg):
    """Review HIGH 4: the IK seed did not wrap angles, so with the rover facing +y the place to its left
    (a world heading near 180 degrees) seeded at -268 degrees, clipped to the joint limit, and was refused."""
    pg.add_robot("mm", build=MOBILE_ARM, yaw=90)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, 0.30])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[-0.32, 0.05], fixed=True)
    _run(pg, 0.5)
    arm, placed = pg.behaviors.pick_and_place("mm", "cube", "bin")
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_an_arm_on_a_rover_that_turned_lowers_straight_into_the_bin(pg):
    """The release fling: after the rover drove a turn, the bin sat near the arm's +-180 degree base limit
    and a straight lowering was taken for a full circle about the base, which threw the cube out."""
    pg.add_robot("mm", build=MOBILE_ARM)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, 0.30])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[-0.32, 0.05], fixed=True)
    pg.behaviors.move("mm", 0.0, 1.0)
    _run(pg, 1.45)
    pg.behaviors.stop("mm")
    _run(pg, 1.0)
    heading = pg.behaviors.pose("mm")["heading_deg"]
    arm, placed = pg.behaviors.pick_and_place("mm", "cube", "bin")
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]
    assert abs(pg.behaviors.pose("mm")["heading_deg"] - heading) < 2  # nothing yanked the rover round


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


def test_two_wheels_and_a_caster_that_sits_a_little_high_is_a_rover(pg):
    """Review HIGH 1: a caster a few millimetres above the wheels' lowest point (common in imported URDFs;
    the robot tips onto it) made a plain rover look like a self-balancer, which then fought its own caster."""
    build = {
        "color": "blue",
        "motor": "jga25",
        "base": {"shape": "box", "size": [0.24, 0.16, 0.05]},
        "parts": [
            {"name": "left_wheel", "at": [0.06, 0.095, -0.01], "shape": "wheel", "size": [0.08, 0.025], "mirror": True},
            {"name": "caster", "at": [-0.1, 0, -0.031], "joint": "fixed", "shape": "sphere", "size": [0.012]},
        ],
    }
    pg.add_robot("cart", build=build)
    assert pg.behaviors.robots()[0]["kind"] == "rover"
    pg.behaviors.move("cart", 0.3, 0.0)
    _run(pg, 3.0)
    pg.behaviors.stop("cart")
    _run(pg, 1.0)
    end = pg.behaviors.pose("cart")
    assert end["upright"] and end["position"][0] > 0.5


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


# ---- MCP tools ------------------------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_agents_build_and_edit_robots_through_mcp(tmp_path, monkeypatch):
    import json

    from mcp import Client

    from robotforge.cli import find_free_port
    from robotforge.mcp_server import build_server

    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    monkeypatch.setenv("ROBOTFORGE_NO_BROWSER", "1")
    playground = Playground(session=SimSession.from_xml("<mujoco/>"))
    server = build_server(playground, find_free_port(8890))

    def text(result) -> str:
        return "".join(c.text for c in result.content if getattr(c, "type", "") == "text")

    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"builder_guide", "add_part", "update_part", "remove_part"} <= names
        guide = json.loads(text(await client.call_tool("builder_guide", {})))
        assert guide["example"]["build"]["parts"]

        await client.call_tool("new_world", {"name": "lab"})
        added = await client.call_tool("add_robot", {"name": "dog", "build": DOG})
        assert not added.is_error, text(added)
        head = {"name": "head", "at": "front", "shape": "sphere", "size": [0.05], "direction": "forward"}
        head["joint"] = {"axis": "z", "range": [-60, 60]}
        head["sensors"] = ["camera"]
        result = await client.call_tool("add_part", {"robot": "dog", "part": head})
        assert not result.is_error, text(result)
        assert "dog/head" in [a["name"] for a in json.loads(text(result))["actuators"]]

        result = await client.call_tool(
            "update_part", {"robot": "dog", "part": "left_thigh", "changes": {"joint": {"home": 40}}}
        )
        assert not result.is_error, text(result)
        world = playground.current_world().robots[0].build
        thigh = next(p for p in world["parts"] if p["name"] == "left_thigh")
        assert thigh["joint"] == {"axis": "y", "range": [-70, 140], "home": 40}  # merged, not replaced

        copy_edit = await client.call_tool("update_part", {"robot": "dog", "part": "right_thigh_2", "changes": {}})
        assert copy_edit.is_error
        bad = await client.call_tool(
            "update_part", {"robot": "dog", "part": "right_thigh_2", "changes": {"color": "red"}}
        )
        assert bad.is_error and "left_thigh" in text(bad)  # edit the part the copy comes from

        robot_wide = await client.call_tool(
            "update_part", {"robot": "dog", "part": "robot", "changes": {"servo": "xl430"}}
        )
        assert not robot_wide.is_error, text(robot_wide)

        overlap = await client.call_tool(
            "add_part",
            {
                "robot": "dog",
                "part": {
                    "name": "fender",
                    "at": [0.12, -0.105, 0],
                    "joint": "fixed",
                    "shape": "box",
                    "size": [0.04] * 3,
                },
            },
        )
        assert overlap.is_error and "overlap" in text(overlap) and "right_thigh_1" in text(overlap), text(overlap)
        assert "fender" not in [p["name"] for p in playground.current_world().robots[0].build["parts"]]

        removed = json.loads(text(await client.call_tool("remove_part", {"robot": "dog", "part": "left_thigh"})))
        assert removed["removed_parts"] == ["left_thigh", "left_shin"]
        # mirror copies come from the removed parts, so they go too
        model = playground.session.model
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dog/right_thigh_1") < 0


def test_arm_built_from_parts_with_a_mounted_gripper_picks_and_places(pg):
    crane = {
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
    pg.add_robot("crane", build=crane)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, 0.05])
    pg.add_object("bin", shape="bin", size=[0.14, 0.14, 0.06], pos=[0.0, 0.27], fixed=True)
    assert pg.behaviors.robots()[0]["kind"] == "arm"
    arm, placed = pg.behaviors.pick_and_place("crane", "cube", "bin")
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_range_sensors_compile_and_measure_distance():
    """A rangefinder and a lidar ring on a built robot compile (MuJoCo 3.15 needs their data spec set) and read
    the distance to a wall."""
    build = copy.deepcopy(DOG)
    build["base"]["sensors"] = [
        "imu",
        {"type": "rangefinder", "at": "front"},
        {"type": "lidar", "at": "top", "rays": 8},
    ]
    world = World.model_validate(
        {
            "robots": [{"name": "r", "build": build}],
            "objects": [{"name": "wall", "shape": "box", "size": [0.05, 1.0, 0.3], "pos": [1.0, 0.0]}],
        }
    )
    model = compile_world(world).model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ahead = data.sensor("r/base_range").data[0]
    assert 0.6 < ahead < 1.0  # the wall's face is about 0.85 m ahead of the body's front
    ring = [data.sensor(f"r/base_lidar_{i}").data[0] for i in range(8)]
    assert 0.6 < ring[0] < 1.0 and all(r == -1 or r > 0 for r in ring)
