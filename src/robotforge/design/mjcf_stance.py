"""A standing pose for a four-legged MJCF robot whose file has none.

Menagerie files come with a "home" keyframe; many others (unitree_mujoco's A1, hand-made models) leave every joint at
zero, which is a pose with the legs out of their limits or straight out. This looks for the pose a dog is photographed
in: the same hip, thigh and knee angles on all four legs, the feet under the hips, the legs bent rather than
stretched, found by forward kinematics alone.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

log = logging.getLogger(__name__)

SPAWN_CLEARANCE = 0.002  # m between the lowest point and the floor at the start (as in library.py)

GRID = 36  # angles tried per joint
MAX_SLIP = 0.04  # m the feet may be ahead of or behind their hips
STANCE_SHARE = 0.78  # of the tallest stance with the feet under the hips: the legs stay bent
LEVEL = 0.01  # m the four feet may differ in height


def find(child: mujoco.MjSpec) -> np.ndarray | None:
    """The qpos of a standing pose of `child` (a floating robot with four three-joint legs), or None when the
    robot is not such a robot or no pose has the feet under the hips."""
    from robotforge.behaviors.quadruped import discover_legs  # lazy: it imports the behaviors

    world = mujoco.MjSpec()
    frame = world.worldbody.add_frame()
    world.attach(child.copy(), prefix="r/", frame=frame)
    try:
        model = world.compile()
    except ValueError as exc:
        log.debug("stance: the robot does not compile on its own: %s", exc)
        return None
    found = discover_legs(model, "r", model.qpos0)
    if found is None or len(found[1]) != 4:
        return None
    root_joint, legs = found
    data = mujoco.MjData(model)
    trunk = int(model.jnt_bodyid[root_joint])
    adr = int(model.jnt_qposadr[root_joint])

    def pose(values: tuple[float, float, float]) -> np.ndarray:
        data.qpos[:] = model.qpos0
        data.qpos[adr : adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        for _name, js, *_ in legs:
            for j, v in zip(js, values, strict=True):
                lo, hi = _range(model, j)
                data.qpos[model.jnt_qposadr[j]] = float(np.clip(v, lo, hi))
        mujoco.mj_kinematics(model, data)
        return data.qpos.copy()

    ranges = [_range(model, j) for j in legs[0][1]]
    candidates = []
    for hip in (0.0,):  # the hips stay square: the same angle on both sides would splay one leg out and one in
        for thigh in np.linspace(*ranges[1], GRID):
            for knee in np.linspace(*ranges[2], GRID):
                pose((hip, thigh, knee))
                feet = [data.xpos[body] + data.xmat[body].reshape(3, 3) @ point for _n, _j, body, point, _p in legs]
                hips = [data.xanchor[js[1]] for _n, js, *_ in legs]  # the thigh joint: where the leg swings from
                heights = np.array([data.xpos[trunk][2] - f[2] for f in feet])
                slip = np.array([f[0] - h[0] for f, h in zip(feet, hips, strict=True)])
                if np.ptp(heights) > LEVEL or np.abs(slip).max() > MAX_SLIP or heights.min() <= 0:
                    continue
                candidates.append((float(heights.mean()), float(np.abs(slip).max()), (hip, thigh, knee)))
    if not candidates:
        return None
    tallest = max(c[0] for c in candidates)
    best = min(candidates, key=lambda c: (abs(c[0] - STANCE_SHARE * tallest), c[1]))
    return pose(best[2])


def _range(model: mujoco.MjModel, joint: int) -> tuple[float, float]:
    if model.jnt_limited[joint]:
        return float(model.jnt_range[joint][0]), float(model.jnt_range[joint][1])
    return -np.pi, np.pi


def balance(
    child: mujoco.MjSpec, home_qpos: dict[str, float], spawn_height: float
) -> tuple[dict[str, float], dict[str, float], float] | None:
    """A home pose for a two-legged robot whose keyframe leaves its feet off the floor or its weight on the edge of
    them, or None when it is no such robot or the pose stands balanced already: (the joint angles that changed, the
    servo targets that follow, the height of the base above the floor).

    Menagerie's H1 stands with its centre of mass 13 cm behind the middle of its feet, over the heels, and GR1's zero
    pose has its toes up; the pose is kept as near the keyframe as it can be with the feet flat and the centre of mass
    over them (behaviors/stance_ik.py)."""
    from robotforge.behaviors import stance, stance_ik  # lazy: they import the behaviors
    from robotforge.sim.actuators import is_position_servo
    from robotforge.sim.geometry import collides, lowest_point

    try:
        model = child.copy().compile()
    except ValueError as exc:
        log.debug("balance: the robot does not compile on its own: %s", exc)
        return None
    bodies = list(range(1, model.nbody))  # on its own, everything in the model is the robot
    if not bodies or not stance.is_floating(model, bodies) or stance.count_wheels(model, bodies) >= 2:
        return None
    data = mujoco.MjData(model)
    for name, value in home_qpos.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = value
    root = int(model.body_rootid[bodies[0]])
    free = next(
        (
            j
            for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ),
        None,
    )
    if free is None:
        return None
    adr = int(model.jnt_qposadr[free])
    data.qpos[adr : adr + 3] = [0.0, 0.0, spawn_height]
    data.qpos[adr + 3 : adr + 7] = model.qpos0[adr + 3 : adr + 7]
    mujoco.mj_forward(model, data)
    tree = np.isin(model.body_rootid, [root])
    feet = stance_ik.plan_feet(model, data, tree)
    if len(feet) < 2 or stance.count_feet(model, data, bodies) != 2:
        return None
    if not stance_ik.needs_balancing(data.subtree_com[root], feet, data):
        return None
    level = data.xmat[root].reshape(3, 3).copy()
    moved = stance_ik.balanced_pose(model, data.qpos, root, tree, feet, data.qpos.copy(), level)
    changed: dict[str, float] = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if tree[model.jnt_bodyid[j]] and name and int(model.jnt_type[j]) in stance_ik.JOINTS:
            value = float(moved[model.jnt_qposadr[j]])
            if abs(value - float(data.qpos[model.jnt_qposadr[j]])) > 1e-6:
                changed[name] = value
    servos: dict[str, float] = {}
    for a in range(model.nu):
        joint = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) or ""
        if (
            int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and is_position_servo(model, a)
            and name in changed
        ):
            servos[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""] = changed[name]
    # The base is where the feet are on the floor; the height that puts the new pose on it, as for any home pose.
    data.qpos[:] = moved
    data.qpos[adr : adr + 3] = 0.0
    mujoco.mj_kinematics(model, data)
    lowest = min(
        float(lowest_point(model, data, g)[2])
        for g in range(model.ngeom)
        if tree[model.geom_bodyid[g]] and collides(model, g)
    )
    return changed, servos, -lowest + SPAWN_CLEARANCE
