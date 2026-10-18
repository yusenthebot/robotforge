"""Where a robot can stand at its start: not inside furniture, a wall, a ramp or another robot.

A robot added without a place starts at [0, 0]. After load_scene that can be inside the furniture: the kitchen's
island stands there, and a Go2 carrying an arm put at [0, 0] overlapped it by 32 cm, was squeezed out under the
counter's edge in the first second, and every trip it was sent on afterwards ended "stuck" where it came out.
Nothing said so. These queries find what a robot overlaps as it stands, and the nearest place it overlaps nothing.
"""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

DEPTH = 0.01  # m of overlap at the start that counts as standing inside something (touching does not)
CLEAR = 0.1  # m a free place keeps from anything, every way along the floor
STEP = 0.25  # m between the rings of places tried round the one asked for
REACH = 4.0  # m: the furthest ring


def owner(model: mujoco.MjModel, geom: int) -> str:
    """What a geom belongs to, by the name an agent knows: the robot or object, or the terrain patch."""
    body = int(model.geom_bodyid[geom])
    if body:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[body]))
    else:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    return (name or f"geom {geom}").split("/", 1)[0]


def _root(model: mujoco.MjModel, robot: str) -> int | None:
    for b in range(1, model.nbody):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/"):
            return int(model.body_rootid[b])
    return None


def inside(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> dict[str, float]:
    """What `robot` overlaps where `data` has it, by name, with how deep (m), deepest first. The floor does not
    count; another robot does. `data` is posed here (kinematics and collision), so pass a copy of a live one."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    prefix = f"{robot}/"
    found: dict[str, float] = {}
    for contact in data.contact[: data.ncon]:
        if contact.dist > -DEPTH:
            continue
        names = [owner(model, int(g)) for g in (contact.geom1, contact.geom2)]
        mine = [n == robot or n.startswith(prefix) for n in names]
        if mine[0] == mine[1]:
            continue
        other, g = (names[1], contact.geom2) if mine[0] else (names[0], contact.geom1)
        if int(model.geom_type[int(g)]) == mujoco.mjtGeom.mjGEOM_PLANE:  # the floor
            continue
        found[other] = max(found.get(other, 0.0), -float(contact.dist))
    return dict(sorted(found.items(), key=lambda kv: -kv[1]))


def move_to_free(
    model: mujoco.MjModel, data: mujoco.MjData, robot: str, step: float = STEP, reach: float = REACH
) -> tuple[float, float] | None:
    """How far to move `robot` along the floor, (dx, dy) in m, from where it stands in `data` to the nearest place
    where, moved there as it is, it overlaps nothing and keeps CLEAR from everything; places are tried on rings
    `step` apart out to `reach`. None for a robot fixed to the world (an arm on its base) or when no place that
    near is free."""
    root = _root(model, robot)
    if root is None:
        return None
    joint = int(model.body_jntadr[root])
    if joint < 0 or int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
        return None
    adr = int(model.jnt_qposadr[joint])
    trial = copy.copy(data)
    start = np.array(trial.qpos[adr : adr + 2], dtype=float)

    def free(x: float, y: float) -> bool:
        for dx, dy in ((0.0, 0.0), (CLEAR, 0.0), (-CLEAR, 0.0), (0.0, CLEAR), (0.0, -CLEAR)):
            trial.qpos[adr : adr + 2] = (x + dx, y + dy)
            if inside(model, trial, robot):
                return False
        return True

    for ring in range(1, int(reach / step) + 1):
        r = ring * step
        count = max(8, round(2 * math.pi * r / step))
        for k in range(count):  # round each ring from -y, always in the same order
            a = -math.pi / 2 + 2 * math.pi * k / count
            dx, dy = r * math.cos(a), r * math.sin(a)
            if free(start[0] + dx, start[1] + dy):
                return round(dx, 2), round(dy, 2)
    return None
