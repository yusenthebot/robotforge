"""RRT-Connect around an obstacle, shortcuts, and the clearance it keeps."""

import numpy as np

from robotforge.behaviors import motion_plan as mp
from robotforge.behaviors.arm import ArmMotion
from robotforge.design.world import World, compile_world


def _arm_and_fence():
    world = World.model_validate(
        {
            "name": "p",
            "robots": [{"name": "arm", "template": "arm", "params": {"dof": 6}}],
            "objects": [
                {"name": "fence", "shape": "box", "size": [0.24, 0.03, 0.18], "pos": [0.40, 0.0], "fixed": True}
            ],
        }
    )
    model = compile_world(world).model
    home = model.key("home").qpos.copy()
    arm = ArmMotion(model, "arm", home)
    space = mp.ArmSpace(model, arm.chain, home)
    start, _, _ = arm._solve(np.array([0.25, 0.2, 0.05]))
    arm.q = arm.home.copy()
    goal, _, _ = arm._solve(np.array([0.25, -0.2, 0.05]))
    return model, space, start, goal


def test_the_straight_sweep_hits_and_the_plan_goes_round():
    model, space, start, goal = _arm_and_fence()
    assert space.free(start) and space.free(goal)
    assert space.first_clash(start, goal) is not None  # turning every joint at once sweeps through the fence
    plan = mp.rrt_connect(space, start, goal, step=0.2, seed=1)
    assert plan.path is not None and plan.nodes > 2
    assert np.allclose(plan.path[0], start) and np.allclose(plan.path[-1], goal)
    assert all(space.segment_free(a, b) for a, b in zip(plan.path, plan.path[1:], strict=False))
    smooth = mp.shortcut(space, plan.path, seed=1)
    assert mp.joint_length(smooth) <= mp.joint_length(plan.path)
    assert all(space.segment_free(a, b) for a, b in zip(smooth, smooth[1:], strict=False))


def test_the_plan_keeps_its_distance_from_fixed_things():
    import mujoco

    model, space, start, goal = _arm_and_fence()
    fence = model.geom("fence").id
    path = mp.shortcut(space, mp.rrt_connect(space, start, goal, step=0.2, seed=2).path, seed=2)
    own = [g for g, o, _ in space.near if o == fence]
    closest = min(
        mujoco.mj_geomDistance(model, (space.pose(q), space.data)[1], g, fence, 0.3, None)
        for q in mp.resample(path, 0.02)
        for g in own
    )
    assert closest >= mp.CLEARANCE - 0.005  # checked every RESOLUTION rad, so a few millimetres can slip


def test_a_blocked_goal_is_reported_not_planned():
    model, space, start, goal = _arm_and_fence()
    inside = goal.copy()
    inside[1] = space.upper[1] + 1.0  # past its joint limit
    plan = mp.rrt_connect(space, start, inside, seed=0)
    assert plan.path is None
