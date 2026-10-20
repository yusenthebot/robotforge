"""How a floating robot stands, read from its shape: on two feet (a humanoid), on four, or on wheels.

The catalogue brings in robots of every shape and no names to go by (their joints are called whatever their makers
called them), so the kind of a robot that no template or library entry vouches for comes from where its lowest points
are at home: two separate places under a body taller than they are far apart make a two-legged robot, however many
arms and fingers it also has. Wheels as feet make a wheeled robot, however many legs carry them.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

SOLE_GAP = 0.012  # m: soles closer than this belong to one foot (toe and heel); Atlas's two feet are 2 cm apart
UPRIGHT_RATIO = 2.0  # a body this many times taller than its footprint is long stands on two feet, not four
FULL_TURNS = 2  # a hinge whose range allows this many turns round is a wheel's, not a limb's
WHEEL_NAME = re.compile(r"wheel|tire|tyre|caster", re.IGNORECASE)
MOVING = frozenset({int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)})  # joints that move


def name_of(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or ""


def joints_of(model: mujoco.MjModel, body: int) -> range:
    return range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body]))


def bodies_of(model: mujoco.MjModel, robot: str) -> list[int]:
    prefix = f"{robot}/"
    return [b for b in range(1, model.nbody) if name_of(model, mujoco.mjtObj.mjOBJ_BODY, b).startswith(prefix)]


def colliders(model: mujoco.MjModel, bodies: list[int]) -> list[int]:
    wanted = set(bodies)
    return [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in wanted and model.geom_contype[g] + model.geom_conaffinity[g] > 0
    ]


def is_floating(model: mujoco.MjModel, bodies: list[int]) -> bool:
    return any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for b in bodies for j in joints_of(model, b))


def count_wheels(model: mujoco.MjModel, bodies: list[int]) -> int:
    """How many wheels the robot has: unlimited hinges at the end of a chain whose joint or body is named for one."""
    count = 0
    carriers = {int(model.body_parentid[c]) for c in range(1, model.nbody) if model.body_jntnum[c]}
    for b in bodies:
        if b in carriers:
            continue  # something with joints hangs off it: not the end of a chain
        for j in joints_of(model, b):
            turns_on = not model.jnt_limited[j] or float(np.ptp(model.jnt_range[j])) >= FULL_TURNS * 2 * np.pi
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) and turns_on:
                names = f"{name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j)} {name_of(model, mujoco.mjtObj.mjOBJ_BODY, b)}"
                count += bool(WHEEL_NAME.search(names))
    return count


def count_feet(model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]) -> int:
    """How many separate places the robot stands on at home: the end bodies of its limbs that reach the floor, as
    they stand (see stance_ik.end_bodies), the ones whose soles touch or lie within SOLE_GAP of each other counted as
    one (toe and heel of a foot, two shells of one foot)."""
    from robotforge.behaviors import stance_ik

    tree = np.zeros(model.nbody, dtype=bool)
    tree[bodies] = True
    ends = stance_ik.end_bodies(model, data, tree)
    if not ends:
        return 0
    soles = stance_ik.current_soles(ends)
    places = stance_ik.places(soles, SOLE_GAP)
    if len(places) == 4:
        centres = [np.mean(np.vstack([soles[i] for i in place]), axis=0) for place in places]
        spread = max(float(np.linalg.norm(a[:2] - b[:2])) for a in centres for b in centres)
        floor = min(float(s[:, 2].min()) for s in soles)
        height = max(float(data.xpos[b][2]) for b in bodies) - floor
        if height > UPRIGHT_RATIO * spread:  # toe and heel of two feet under a body taller than they are long
            return 2
    return len(places)


def stands_on_two_feet(model: mujoco.MjModel, robot: str, home_qpos: np.ndarray) -> bool:
    """Whether `robot` is a floating body that stands on two feet and not on wheels (see the module doc)."""
    bodies = bodies_of(model, robot)
    if not bodies or not is_floating(model, bodies) or count_wheels(model, bodies) >= 2:
        return False
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos
    mujoco.mj_kinematics(model, data)
    return count_feet(model, data, bodies) == 2
