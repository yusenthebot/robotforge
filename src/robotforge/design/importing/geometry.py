"""Pose-dependent geometry of a compiled robot: ground clearance, subtree loads, welded link groups."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.design.importing.plan import RigError
from robotforge.design.importing.urdf import StagedUrdf
from robotforge.sim.geometry import collides, lowest_point

GRAVITY = 9.81
TOUCH_TOLERANCE = 0.02  # m: a link this close to the lowest point counts as touching the ground


def pose_model(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    for name, value in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(model, data)


def compile_spec(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    try:
        model = spec.compile()
    except ValueError as exc:
        raise RigError(f"MuJoCo could not compile the repaired robot: {exc}") from exc
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


class Geometry:
    """Pose-dependent geometry of a compiled robot: ground clearance, loads, subtree masses."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.children: dict[int, list[int]] = {}
        for b in range(1, model.nbody):
            self.children.setdefault(int(model.body_parentid[b]), []).append(b)
        self.total_mass = float(model.body_mass.sum())
        self.collision_geoms = [g for g in range(model.ngeom) if collides(model, g)]
        self.low = self.lowest_by_body()

    def body_name(self, b: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    def joint_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def subtree(self, body: int) -> list[int]:
        out, stack = [], [body]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.children.get(b, []))
        return out

    def lowest_by_body(self) -> dict[int, float]:
        """Lowest world z of each body's colliding geoms (bodies without any are absent)."""
        low: dict[int, float] = {}
        for g in self.collision_geoms:
            b = int(self.model.geom_bodyid[g])
            low[b] = min(self.geom_low(g)[2], low.get(b, math.inf))
        return low

    def geom_low(self, g: int) -> tuple[float, float, float]:
        """Lowest world point of one geom (exact for primitives, vertex-based for meshes)."""
        p = lowest_point(self.model, self.data, g)
        return float(p[0]), float(p[1]), float(p[2])

    @property
    def lowest(self) -> float:
        return min(self.low.values()) if self.low else 0.0

    def touching(self) -> set[int]:
        floor = self.lowest
        return {b for b, z in self.low.items() if z <= floor + TOUCH_TOLERANCE}

    def joint_load(self, jid: int) -> tuple[float, float, float]:
        """(inertia about the joint, mass below it, worst-case gravity torque) for the subtree it moves."""
        m, d = self.model, self.data
        body = int(m.jnt_bodyid[jid])
        anchor, axis = d.xanchor[jid], d.xaxis[jid]
        slide = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        inertia = mass = torque = 0.0
        for b in self.subtree(body):
            mb = float(m.body_mass[b])
            if mb <= 0:
                continue
            r = d.xipos[b] - anchor
            perp = r - axis * float(r @ axis)
            mass += mb
            torque += mb * GRAVITY * (abs(float(axis[2])) if slide else float(np.linalg.norm(perp)))
            if not slide:
                rot = d.ximat[b].reshape(3, 3)
                tensor = rot @ np.diag(m.body_inertia[b]) @ rot.T
                inertia += float(axis @ tensor @ axis) + mb * float(perp @ perp)
        return inertia, mass, torque


def clusters(staged: StagedUrdf) -> dict[str, str]:
    """Group links that are welded together by fixed joints; the label is the group's top link."""
    label = {name: name for name in staged.links}

    def top(name: str) -> str:
        while label[name] != name:
            name = label[name]
        return name

    for j in staged.joints:
        if j.type == "fixed":
            label[j.child] = top(j.parent)
    return {name: top(name) for name in staged.links}
