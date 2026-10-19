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
