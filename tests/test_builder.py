"""The free-form builder: robots from parts, their checks and errors, and built-in behaviors on them."""

from __future__ import annotations

import asyncio
import copy

import mujoco
import numpy as np
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from robotforge.behaviors.manager import BehaviorError
from robotforge.cli import find_free_port
from robotforge.design import builder, gallery
from robotforge.design.world import World, compile_world
from robotforge.mcp_server import build_server
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


def test_a_part_colored_main_follows_the_robot_color(pg):
    """Parts in a design's accent color had it written out, so recoloring the robot changed only its base and wheel
    rims; "main" makes a part follow the robot's color."""
    from robotforge.design import components

    build = copy.deepcopy(DOG)
    build["parts"][1]["color"] = "main"  # the thighs
    pg.add_robot("dog", build=build)

    def thigh_is(color: str) -> bool:
        m = pg.session.model
        body = m.body("dog/left_thigh_1").id
        rgba = [m.geom_rgba[g] for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        return any(np.allclose(c, components.color(color), atol=1e-3) for c in rgba)

    assert thigh_is("orange")
    pg.update_part("dog", "robot", {"color": "teal"})
    assert thigh_is("teal") and not thigh_is("orange")


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


def test_a_build_without_parts_changes_its_fields_and_keeps_the_legs(pg):
    """An agent in the eval sent update_robot(build={"servo": ..., "base": {...}}) to give a hexapod stronger servos
    and got a bare box of 1 part: a build without parts was taken as the whole robot."""
    pg.add_robot("spider", build=HEXAPOD)
    joints = pg.session.model.nu
    out = pg.update_robot("spider", build={"servo": "bldc_joint", "base": {"mass": 0.8}})
    built = pg.current_world().robots[0].build
    assert len(built["parts"]) == len(HEXAPOD["parts"]) and pg.session.model.nu == joints
    assert built["servo"] == "bldc_joint"
    assert built["base"] == {**HEXAPOD["base"], "mass": 0.8}  # the size and shape it had are kept
    # "kept its parts", not a count: the build has 3 entries, the robot 19 parts (review)
    assert any(n.startswith("spider: kept its parts and changed only servo, base") for n in out["notes"])
    # The same merge as update_part(robot, "robot" | "base"): null puts a field back to its default, and a field
    # that is neither robot-wide nor the base is refused with the list of robot-wide settings.
    pg.update_robot("spider", build={"servo": None})
    assert "servo" not in pg.current_world().robots[0].build
    with pytest.raises(ValueError, match="robot-wide settings are"):
        pg.update_robot("spider", build={"legs": 6})
    with pytest.raises(PlaygroundError, match=r"build \{\} changes nothing on spider"):  # it did nothing, and said so
        pg.update_robot("spider", build={})
    pg.update_robot("spider", build={**HEXAPOD, "parts": HEXAPOD["parts"][:3]})  # with parts: the whole robot
    assert len(pg.current_world().robots[0].build["parts"]) == 3


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


def test_a_mast_behind_a_rovers_arm_leaves_picks_in_front_of_it_alone(pg):
    """Regression: an arm on a moving robot carried things above the highest shape anywhere on that robot, so a 25 cm
    mast 19 cm behind the arm made this pick in front of it slow (15 cm mast) or refused (25 and 35 cm masts)."""
    build = copy.deepcopy(MOBILE_ARM)
    mast = {"name": "mast", "at": "top", "shift": [-0.13, 0, 0.175], "shape": "cylinder", "size": [0.03, 0.35]}
    build["parts"].append({**mast, "joint": "fixed"})
    pg.add_robot("mm", build=build)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.2, 0.0])
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[0.95, 0.32], fixed=True)
    pg.behaviors.move("mm", 0.3, 0.0)
    _run(pg, 3.0)
    pg.behaviors.stop("mm")
    _run(pg, 0.5)
    arm, placed = pg.behaviors.pick_and_place("mm", "cube", "bin")
    carry = next(wp for wp in arm.queue if wp.label == "carry")
    assert carry.position[2] < 0.25  # as without the mast: over the bin's rim, not over the mast's top (0.41 m)
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_a_carry_over_a_tall_part_of_the_rover_is_refused_naming_that_part(pg):
    """Too high to carry over, the refusal blamed the object and the target ("move them a little further from the
    base") instead of the part of the robot in the way."""
    build = copy.deepcopy(MOBILE_ARM)
    build["parts"] += [
        {"name": "boom", "at": [0.3, 0, 0.04], "shape": "box", "size": [0.2, 0.04, 0.02], "joint": "fixed"},
        {
            "name": "post",
            "parent": "boom",
            "at": [0.1, 0, 0],
            "shift": [0, 0, 0.15],
            "shape": "cylinder",
            "size": [0.03, 0.3],
            "joint": "fixed",
        },
    ]
    pg.add_robot("mm", build=build)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.3, 0.25])
    pg.add_object("bin", shape="bin", size=[0.12, 0.12, 0.05], pos=[0.3, -0.25], fixed=True)
    _run(pg, 0.5)
    with pytest.raises(ValueError, match=r"over mm's post, whose top is 0\.4\d m high, .*does not pass over post"):
        pg.behaviors.pick_and_place("mm", "cube", "bin")


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


def test_the_arm_on_a_rover_is_an_arm_to_move_and_pose_and_says_who_carries_it(pg):
    """move and pose on "mm/arm" took it for the rover and said "no driven wheels" or "mm/arm is a rover"."""
    pg.add_robot("mm", build=MOBILE_ARM)
    assert pg.behaviors.kind("mm/arm").value == "arm"
    assert pg.behaviors.kind("MM/Arm").value == "arm"  # other capitals still find it
    assert pg.behaviors.kind("mm / arm").value == "arm"  # and so do stray spaces
    with pytest.raises(ValueError, match=r"mm/arm is the arm mounted on mm: drive mm with move; .*move_arm"):
        pg.behaviors.move("mm/arm", 0.2, 0.0)
    with pytest.raises(ValueError, match=r"mm/arm is an arm, and only humanoids strike poses; .*move_arm"):
        pg.behaviors.strike_pose("mm/arm", "wave")
    with pytest.raises(ValueError, match=r"'mm/claw' is a part of mm, not a robot; use mm"):
        pg.behaviors.move("mm/claw", 0.2, 0.0)
    with pytest.raises(ValueError, match=r"'mm/claw' is not an arm of mm; use mm/arm"):
        pg.behaviors.move_arm("mm/claw", [0.3, 0, 0.2])


def test_the_tools_name_the_arms_a_rover_carries_and_say_what_fits_them(pg):
    """The move tool read the robot's pose before anything else, so "mm/claw" was answered with "no robot named"."""
    pg.add_robot("mm", build=MOBILE_ARM)
    server = build_server(pg, find_free_port(8790))

    def refused(tool, **args) -> str:
        async def call():
            with pytest.raises(ToolError) as caught:
                await server.call_tool(tool, args)
            return str(caught.value)

        return asyncio.run(call())

    assert "the arm mounted on mm: drive mm with move" in refused("move", robot="mm/arm", speed=0.2)
    assert "'mm/claw' is a part of mm, not a robot; use mm" in refused("move", robot="mm/claw", speed=0.2)
    assert "'mm/claw' is a part of mm" in refused("move", robot="mm/ claw", speed=0.2)  # echoed without the space
    assert "'mm/claw' is not an arm of mm; use mm/arm" in refused("move_arm", robot="mm/claw", position=[0.3, 0, 0.2])
    assert "mm/arm is an arm, and only humanoids strike poses" in refused("pose", robot="mm/arm", name="wave")


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


WHEEL_LEGS = {  # two legs that end in wheels: the knees are servos, which give a little under the body's weight
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.13, 0.17, 0.09]},
    "parts": [
        {
            "name": "left_thigh",
            "at": [0, 0.105, -0.02],
            "joint": {"axis": "y", "range": [-30, 100], "home": 40},
            "size": [0.03, 0.12],
            "direction": "down",
            "mirror": True,
        },
        {
            "name": "left_shin",
            "parent": "left_thigh",
            "at": "end",
            "joint": {"axis": "y", "range": [-150, 0], "home": -80},
            "size": [0.025, 0.12],
            "direction": "down",
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


def test_a_wheel_legged_robot_balances_in_place_and_stays_there_when_it_crouches(pg):
    """The balanced lean was worked out with the legs straightening (their controls were 0) and stiff, so it was a
    degree off and the robot rolled 0.2 m back holding the difference; a crouch set by hand moved the balance point
    and rolled it 0.34 m (standing taller) to 1.17 m (crouching deep)."""
    pg.add_robot("ascent", build=WHEEL_LEGS)
    assert pg.behaviors.robots()[0]["kind"] == "balancer"
    _run(pg, 0.5)
    start = pg.behaviors.pose("ascent")["position"]
    _run(pg, 4.0)
    assert abs(pg.behaviors.pose("ascent")["position"][0] - start[0]) < 0.03
    for thigh, shin in ((15, -30), (70, -140)):  # taller, then deep
        pg.set_controls(
            {
                f"ascent/{side}_{part}": np.radians(angle)
                for side in ("left", "right")
                for part, angle in (("thigh", thigh), ("shin", shin))
            }
        )
        _run(pg, 4.0)
        end = pg.behaviors.pose("ascent")
        assert end["upright"] and abs(end["position"][0] - start[0]) < 0.05, (thigh, end)
    assert end["position"][2] < start[2] - 0.08  # it did crouch


def test_a_wheel_legged_robot_whose_body_tips_back_as_it_crouches_keeps_balancing(pg):
    """Folding the thighs back slowly keeps the weight over the axle while the body tips back with them. The fall check
    read the body's pitch: 46 degrees back it called the robot fallen, cut the wheels, and it fell flat 0.36 m away."""
    pg.add_robot("ascent", build=WHEEL_LEGS)
    _run(pg, 1.0)
    start = pg.behaviors.pose("ascent")["position"]
    m, d = pg.session.model, pg.session.data
    goals = (("thigh", 90), ("shin", -60))
    legs = {f"ascent/{side}_{part}": goal for side in ("left", "right") for part, goal in goals}
    before = {name: float(d.ctrl[m.actuator(name).id]) for name in legs}
    for step in range(1, 41):  # over 2 s
        s = step / 40
        pg.set_controls({name: (1 - s) * before[name] + s * np.radians(goal) for name, goal in legs.items()})
        _run(pg, 0.05)
    _run(pg, 2.0)
    end = pg.behaviors.pose("ascent")
    assert not pg.session.controllers["ascent"].fallen
    assert end["upright"]  # balancing, though its body leans far back
    assert np.linalg.norm(np.subtract(end["position"][:2], start[:2])) < 0.05


def test_only_a_robot_with_nothing_to_hold_it_up_is_told_it_tips_over():
    """The wheel-legged biped tips over left alone, as every self-balancer does, and was told to get more feet, a wider
    stance or a caster, though the built-in balancer keeps it up; a robot that tips with nothing to hold it up still
    hears it."""
    tips = "does not stay upright by itself"
    assert not any(tips in n for n in builder.build_robot(gallery.build("wheel_biped")).notes)
    stilt = {  # a box on one stilt, off its middle
        "base": {"shape": "box", "size": [0.1, 0.1, 0.1]},
        "parts": [
            {"name": "stilt", "at": [0.04, 0, -0.05], "joint": "fixed", "size": [0.01, 0.2], "direction": "down"}
        ],
    }
    assert any(tips in n for n in builder.build_robot(stilt).notes)


def _cart_with_a_head(head_height: float) -> dict:
    """A cart with an arm at the back and a block (a head, a sensor box) at the front, in the way of the arm's swing
    from one side to the other."""
    return {
        "motor": "jga25",
        "base": {"shape": "box", "size": [0.4, 0.24, 0.06], "mass": 4.0},
        "parts": [
            {"name": "left_wheel", "at": [0, 0.14, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
            {"name": "front_caster", "at": [0.16, 0, -0.04], "shape": "caster", "size": [0.04]},
            {"name": "rear_caster", "at": [-0.16, 0, -0.04], "shape": "caster", "size": [0.04]},
            {
                "name": "head",
                "at": "top",
                "shift": [0.15, 0, 0],
                "joint": "fixed",
                "shape": "box",
                "size": [0.08, 0.16, head_height],
                "direction": "up",
            },
            {"name": "arm", "at": "top", "shift": [-0.12, 0, 0], "robot": {"template": "arm", "home": {"yaw": 90}}},
        ],
    }


def test_an_arm_carries_a_load_over_the_robot_it_rides_on_or_says_it_cannot(pg):
    """Moving a floor cube from the cart's left to its right swings it round in front, at 10 cm: the cube struck the
    block in the way, was dropped short and the pick still ran to the end. The load now goes over the block, and
    where nothing gets over it the pick is refused with the reason."""
    pg.add_robot("cart", build=_cart_with_a_head(0.08))
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[-0.12, 0.28])
    _run(pg, 0.5)
    motion, info = pg.behaviors.pick_and_place("cart/arm", "cube", [-0.12, -0.28])
    assert [round(float(w.position[2]), 2) for w in motion.queue if w.label == "carry"] == [0.25]
    pg.run_fast(lambda: motion.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed(info["object"], info["region"])["success"]
    pg.update_part("cart", "head", {"size": [0.08, 0.16, 0.16]})  # too tall for the arm to get a cube over it
    pg.add_object("another", shape="box", size=[0.04, 0.04, 0.04], pos=[-0.12, 0.28])
    _run(pg, 0.5)
    with pytest.raises(BehaviorError, match="cannot carry it across: .*put both on the same side of the robot"):
        pg.behaviors.pick_and_place("cart/arm", "another", [-0.12, -0.28])


def _wheel_motor(model: mujoco.MjModel, carrier: str) -> int:
    """The gear motor of the wheel on `carrier`: the cylinder on that body nothing collides with, in a gear motor's
    color (the knee's servo is one too, in the servo's dark case). Dressed robots draw a mesh over both and hide
    the cylinders, so the group says nothing here."""
    from robotforge.design.builder.assemble import MOTOR_YELLOW
    from robotforge.design.visuals import parts as P

    body = model.body(carrier).id
    cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    colors = [np.asarray(c, float) for c in (MOTOR_YELLOW, P.METAL)]
    found = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and model.geom_type[g] == cylinder
        and not model.geom_contype[g]
        and any(np.allclose(model.geom_rgba[g], c, atol=1e-3) for c in colors)
    ]
    assert len(found) == 1, found
    return found[0]


def test_a_wheel_hung_on_a_leg_gets_a_hub_motor_that_stays_outside_the_leg():
    """The motor was drawn as a 6 cm can inboard of the wheel, which on a leg stuck out through the leg."""
    from robotforge.design.builder.assemble import MOTOR_YELLOW
    from robotforge.design.visuals import parts as P

    model = _model(WHEEL_LEGS)
    motor = _wheel_motor(model, "r/left_shin")
    center, half = model.geom_pos[motor], model.geom_size[motor][1]
    assert center[1] - half >= 0.0 and center[1] + half <= 0.032 - 0.015  # between the leg's middle and the wheel
    assert list(model.geom_rgba[motor]) == pytest.approx(P.METAL)  # a JGA25 is a metal can
    tt = copy.deepcopy(WHEEL_LEGS)
    tt["motor"] = "tt_motor"
    toy = _model(tt)
    assert list(toy.geom_rgba[_wheel_motor(toy, "r/left_shin")]) == pytest.approx(MOTOR_YELLOW)  # a TT's yellow case


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


LIFT_ARM = {  # an arm on a lifting column: it reaches a high shelf only by raising itself
    "base": {"shape": "box", "size": [0.2, 0.2, 0.04], "fixed": True},
    "parts": [
        {
            "name": "lift",
            "at": "top",
            "joint": {"type": "slide", "axis": "z", "range": [0, 0.3]},
            "actuator": {"model": "belt_rail", "gear": 2},
            "shape": "box",
            "size": [0.06, 0.06, 0.2],
            "mass": 0.3,
            "direction": "up",
        },
        {
            "name": "arm",
            "parent": "lift",
            "at": "end",
            "robot": {"template": "arm", "params": {"reach": 0.45, "base_height": 0.03, "servo": "bldc_joint"}},
        },
    ],
}


def test_an_arm_on_a_lift_raises_it_to_reach_a_high_table(pg):
    """The arm's joints were its hinges only, so a cube on a table above the arm's reach from where the lift stood was
    refused as out of reach; the lift is a joint the arm rides on and is now steered with the others."""
    pg.add_robot("tower", build=LIFT_ARM)
    pg.add_object("high", shape="table", size=[0.2, 0.2, 0.55], pos=[0.0, -0.28], fixed=True)
    pg.add_object("low", shape="table", size=[0.2, 0.2, 0.3], pos=[0.0, 0.28], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, -0.28, 0.57])
    _run(pg, 0.5)
    arm, placed = pg.behaviors.pick_and_place("tower", "cube", "low")
    assert "tower/lift" in [mujoco.mj_id2name(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm.joints]
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


TWIN = {  # two arms on one turntable
    "base": {"shape": "cylinder", "size": [0.16, 0.06], "fixed": True},
    "parts": [
        {
            "name": "turntable",
            "at": "top",
            "joint": {"axis": "z", "range": [-180, 180]},
            "shape": "box",
            "size": [0.08, 0.38, 0.03],
            "direction": "up",
        },
        {
            "name": "left_arm",
            "parent": "turntable",
            "at": "end",
            "shift": [0, 0.14, 0],
            "robot": {"template": "arm", "params": {"reach": 0.35, "base_height": 0.03}},
        },
        {
            "name": "right_arm",
            "parent": "turntable",
            "at": "end",
            "shift": [0, -0.14, 0],
            "robot": {"template": "arm", "params": {"reach": 0.35, "base_height": 0.03}},
        },
    ],
}


def _tool_error(pg: Playground, arm, target: list[float]) -> float:
    return float(np.linalg.norm(pg.session.data.site_xpos[arm.site] - np.asarray(target)))


def test_two_arms_on_one_turntable_reach_without_turning_it_and_follow_it_when_it_turns(pg):
    """Both arms counted the turntable as their own first joint, so each turned it toward its target (and the other
    arm away from its own); and an arm's IK assumed the turntable stood where it started, so after it was turned by
    hand the arm missed its target by the turn."""
    pg.add_robot("twin", build=TWIN)
    assert pg.behaviors.robots()[0]["arms"] == ["twin/left_arm", "twin/right_arm"]
    turntable = pg.session.model.joint("twin/turntable").qposadr[0]
    target = [0.12, 0.3, 0.15]
    left = pg.behaviors.move_arm("twin/left_arm", target)
    pg.run_fast(lambda: left.done, 10)
    _run(pg, 0.5)
    assert _tool_error(pg, left, target) < 0.015  # small hobby servos sag about a centimetre
    assert abs(pg.session.data.qpos[turntable]) < 0.01  # it did not turn the turntable
    pg.set_controls({"twin/turntable": np.radians(40)})
    _run(pg, 1.5)
    target = [0.3, 0.0, 0.12]
    right = pg.behaviors.move_arm("twin/right_arm", target)
    pg.run_fast(lambda: right.done, 10)
    _run(pg, 0.5)
    assert _tool_error(pg, right, target) < 0.015
    assert pg.session.data.qpos[turntable] == pytest.approx(np.radians(40), abs=0.02)


def test_an_arm_folded_low_rises_before_it_swings_around_to_its_target(pg):
    """An arm folded backward, low, swung around toward the cube at that height and ran into the wall beside it; it
    now goes up to the carrying height first and swings around above the wall."""
    folded = copy.deepcopy(LIFT_ARM)
    folded["parts"][1]["robot"]["home"] = {"yaw": 180}
    pg.add_robot("tower", build=folded)
    pg.add_object("high", shape="table", size=[0.2, 0.2, 0.55], pos=[0.0, 0.28], fixed=True)
    pg.add_object("low", shape="table", size=[0.2, 0.2, 0.3], pos=[0.0, -0.28], fixed=True)
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.0, 0.28, 0.57])
    pg.add_object("wall", shape="box", size=[0.12, 0.12, 0.5], pos=[-0.18, 0.18], fixed=True)  # back left
    _run(pg, 0.5)
    arm, placed = pg.behaviors.pick_and_place("tower", "cube", "low")
    pg.run_fast(lambda: arm.done, 40)
    _run(pg, 0.5)
    assert pg.behaviors.placed("cube", placed["region"])["success"]


def test_a_mounted_panda_is_not_called_too_weak(monkeypatch):
    """The build check ran explicit Euler, a fresh spec's default, where the Panda's stiff servos (kp 4500) chatter at
    their 87 N*m limits holding the arm still, so every build carrying one was told its arm was too weak; worlds run
    the implicit integrator the check now uses too."""
    from pathlib import Path

    monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures" / "menagerie"))
    build = {
        "base": {"shape": "box", "size": [0.6, 0.45, 0.15], "mass": 40.0},
        "parts": [
            {"name": "left_wheel", "at": [0.2, 0.26, -0.05], "shape": "wheel", "size": [0.16, 0.06], "mirror": True},
            {"name": "left_rear", "at": [-0.2, 0.26, -0.05], "shape": "wheel", "size": [0.16, 0.06], "mirror": True},
            {"name": "arm", "at": "top", "robot": {"library": "panda"}},
        ],
    }
    notes = " ".join(builder.build_robot(build).notes)
    assert "Franka Emika Panda" in notes and "too weak" not in notes


def test_a_mounted_robot_starts_in_the_home_pose_its_mount_gives():
    folded = copy.deepcopy(LIFT_ARM)
    folded["parts"][1]["robot"]["home"] = {"yaw": 180, "shoulder": -30}
    model = _model(folded)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    assert data.joint("r/arm/yaw").qpos[0] == pytest.approx(np.pi)
    assert data.joint("r/arm/shoulder").qpos[0] == pytest.approx(np.radians(-30))
    assert data.actuator("r/arm/shoulder").ctrl[0] == pytest.approx(np.radians(-30))  # the servo holds it there
    assert data.joint("r/arm/elbow").qpos[0] == pytest.approx(1.45)  # the rest keep the template's own home
    folded["parts"][1]["robot"]["home"] = {"elbo": 10}
    with pytest.raises(ValueError, match="no joint 'elbo'.*Did you mean elbow"):
        builder.build_robot(folded)
    folded["parts"][1]["robot"]["home"] = {"elbow": 170}
    with pytest.raises(ValueError, match="outside its range"):
        builder.build_robot(folded)


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


# The agent eval's hexapod (ae450e3): legs that roll out at the hip and pitch at thigh and knee, like a dog's, on
# xl430 servos. Sent forwards at 0.12 m/s for 5 s it went 0.21 m backwards, told only that it was built.
SPIDER = {
    "material": "carbon",
    "servo": "xl430",
    "base": {"shape": "box", "size": [0.46, 0.18, 0.075], "mass": 0.7, "sensors": ["imu"]},
    "parts": [
        {
            "name": "left_hip",
            "at": [0.18, 0.09, 0],
            "joint": {"axis": "x", "range": [-45, 65], "home": 30},
            "shape": "none",
            "mirror": True,
            "repeat": {"count": 3, "along": "back", "spacing": 0.18},
        },
        {
            "name": "left_thigh",
            "parent": "left_hip",
            "at": [0, 0.045, 0],
            "joint": {"axis": "y", "range": [-70, 140], "home": 46},
            "shape": "capsule",
            "size": [0.025, 0.15],
            "direction": "down",
        },
        {
            "name": "left_shin",
            "parent": "left_thigh",
            "at": "end",
            "joint": {"axis": "y", "range": [-150, -15], "home": -92},
            "shape": "capsule",
            "size": [0.021, 0.16],
            "direction": "down",
            "foot": True,
        },
    ],
}


def test_a_walker_whose_servos_work_hard_standing_is_told_it_may_not_walk(pg):
    """Standing on six feet the agent's spider used 46 to 54 % of its xl430s' force, under the 60 % that got a note;
    walking holds it on three, and it slid backwards. On sts3215 servos (39 %) it walks."""
    out = pg.add_robot("spider", build=SPIDER)
    (note,) = [n for n in out["notes"] if "little headroom to walk" in n]
    assert "standing on all 6 feet" in note and "slide backwards" in note and "sts3215" in note
    stronger = pg.add_robot("strong", build={**SPIDER, "servo": "sts3215"}, pos=[0, 1.5])
    assert not [n for n in stronger["notes"] if n.startswith("strong:") and "headroom" in n], stronger["notes"]
