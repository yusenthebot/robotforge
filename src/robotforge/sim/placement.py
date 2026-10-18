"""Where a robot can stand at its start: not inside furniture, a wall, a ramp or another robot.

A robot added without a place starts at [0, 0]. After load_scene that can be inside the furniture: the kitchen's
island stands there, and a Go2 carrying an arm put at [0, 0] overlapped it by 32 cm, was squeezed out under the
counter's edge in the first second, and every trip it was sent on afterwards ended "stuck" where it came out.
Nothing said so. These queries find what a robot overlaps as it stands, and the first free place of those tried.

A place is free in the world as it starts (every robot at its saved place, as a reset or reload_world puts them) and
as it is now: a pup added once the dog had walked off to the fridge was given the dog's start, and after reload_world
it stood on the dog's back. Another robot counts by what it covers seen from above, not only by touching: a rover
fits between a Go2's legs touching nothing, and one added after the Go2 was put there, under its belly.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Iterator

import mujoco
import numpy as np

DEPTH = 0.01  # m of overlap at the start that counts as standing inside something (touching does not)
CLEAR = 0.1  # m a free place keeps from anything, every way along the floor, and from what other robots cover
STEP = 0.25  # m between the rings of places tried round the one asked for
REACH = 4.0  # m: the furthest ring
SHIFTS = ((CLEAR, 0.0), (-CLEAR, 0.0), (0.0, CLEAR), (0.0, -CLEAR))  # a free place keeps CLEAR from furniture

Box = tuple[float, float, float, float]  # seen from above: x low, x high, y low, y high (m)


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


def footprints(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Box]:
    """What each robot covers seen from above where `data` has it (posed already): the box round its shapes' boxes.
    A robot's bodies are named <robot>/<body>; objects and terrain are not robots."""
    out: dict[str, list[float]] = {}
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if not body or int(model.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        root = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[body])) or ""
        if "/" not in root:
            continue
        rot, box = data.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
        middle, half = data.geom_xpos[g] + rot @ box[:3], np.abs(rot) @ box[3:]
        low, high = middle - half, middle + half
        have = out.setdefault(root.split("/", 1)[0], [math.inf, -math.inf, math.inf, -math.inf])
        have[:] = [min(have[0], low[0]), max(have[1], high[0]), min(have[2], low[1]), max(have[3], high[1])]
    return {name: (b[0], b[1], b[2], b[3]) for name, b in out.items()}


def blocked(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> dict[str, float]:
    """What keeps `robot` from starting where `data` has it, by name, with how much (m), most first: what it stands
    inside (see `inside`), and every other robot whose footprint its own comes within CLEAR of."""
    found = inside(model, data, robot)
    covered = footprints(model, data)
    mine = covered.pop(robot, None)
    if mine is None:
        return found
    for other, box in covered.items():
        overlap = min(
            mine[1] + CLEAR - box[0], box[1] + CLEAR - mine[0], mine[3] + CLEAR - box[2], box[3] + CLEAR - mine[2]
        )
        if overlap > 0:
            found[other] = max(found.get(other, 0.0), round(float(overlap), 3))
    return dict(sorted(found.items(), key=lambda kv: -kv[1]))


def movable(model: mujoco.MjModel, robot: str) -> int | None:
    """Where `robot`'s free joint is in qpos, or None for a robot fixed to the world (an arm on its base)."""
    root = _root(model, robot)
    if root is None:
        return None
    joint = int(model.body_jntadr[root])
    if joint < 0 or int(model.jnt_type[joint]) != mujoco.mjtJoint.mjJNT_FREE:
        return None
    return int(model.jnt_qposadr[joint])


def rings(x: float, y: float, step: float = STEP, reach: float = REACH) -> Iterator[tuple[float, float]]:
    """Places round (x, y), ring by ring `step` apart out to `reach`, each ring from -y round, always in one order."""
    for ring in range(1, int(reach / step) + 1):
        r = ring * step
        count = max(8, round(2 * math.pi * r / step))
        for k in range(count):
            a = -math.pi / 2 + 2 * math.pi * k / count
            yield x + r * math.cos(a), y + r * math.sin(a)


def free_place(
    model: mujoco.MjModel, datas: list[mujoco.MjData], robot: str, spots: Iterable[tuple[float, float, float]]
) -> tuple[float, float, float] | None:
    """The first of `spots`, (x, y, turn in degrees), where `robot` moved there along the floor and turned that much
    is blocked by nothing (see `blocked`) and keeps CLEAR from the furniture, in every one of `datas` (pass the world
    as it starts and as it is now); x and y rounded to the cm. None for a robot fixed to the world, or when none is
    free."""
    adr = movable(model, robot)
    if adr is None:
        return None
    trials = [copy.copy(d) for d in datas]
    turned = [t.qpos[adr + 3 : adr + 7].copy() for t in trials]

    def fits(x: float, y: float, turn: float) -> bool:
        spin = np.array([math.cos(math.radians(turn) / 2), 0.0, 0.0, math.sin(math.radians(turn) / 2)])
        for trial, quat in zip(trials, turned, strict=True):
            mujoco.mju_mulQuat(trial.qpos[adr + 3 : adr + 7], spin, quat)
            trial.qpos[adr : adr + 2] = (x, y)
            if blocked(model, trial, robot):
                return False
            for dx, dy in SHIFTS:
                trial.qpos[adr : adr + 2] = (x + dx, y + dy)
                if inside(model, trial, robot):
                    return False
        return True

    for x, y, turn in spots:
        x, y = round(x, 2), round(y, 2)
        if fits(x, y, turn):
            return x, y, turn
    return None
