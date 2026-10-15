"""Keep what an edit did not touch where it is.

Editing a world compiles it again and loads a fresh simulation, which put every robot back where the
world starts it: add a box, and the dog that had just walked a metre jumped back. Before an edit the
live state is read by name (each free body's pose, each joint's angle); after it, the robots and
objects whose description the edit left alone get theirs back. Anything that would then sit deep
inside something new (a box added where the robot stands) starts fresh instead.

Names are "<robot>/<part>" for robots and the object's own name for objects: the part before the
first "/" says whose it is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import mujoco
import numpy as np

DEEP = 0.01  # m of overlap that means a carried thing landed inside something new


def owner(name: str) -> str:
    return name.split("/", 1)[0]


@dataclass
class Snapshot:
    free: dict[str, np.ndarray] = field(default_factory=dict)  # body name -> its free joint's qpos (7)
    joints: dict[str, np.ndarray] = field(default_factory=dict)  # joint name -> qpos (hinge, slide, ball)
    controls: dict[str, float] = field(default_factory=dict)  # actuator name -> its target (or the arm swings home)


def take(model: mujoco.MjModel, data: mujoco.MjData) -> Snapshot:
    snap = Snapshot()
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            body = model.body(int(model.jnt_bodyid[j])).name
            if body:
                snap.free[body] = data.qpos[adr : adr + 7].copy()
        elif name := model.joint(j).name:
            width = 4 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL else 1
            snap.joints[name] = data.qpos[adr : adr + width].copy()
    for a in range(model.nu):
        if name := model.actuator(a).name:
            snap.controls[name] = float(data.ctrl[a])
    return snap


def restore(model: mujoco.MjModel, data: mujoco.MjData, snap: Snapshot, keep: Iterable[str]) -> list[str]:
    """Put back the poses and joint angles of the owners in `keep`, standing still. Owners that would then
    overlap something deeply are left fresh. Returns the owners carried."""
    keep = set(keep)
    fresh = take(model, data)
    _apply(model, data, snap, lambda name: owner(name) in keep)
    mujoco.mj_forward(model, data)
    stuck = _deep_overlaps(model, data) & keep
    if stuck:
        _apply(model, data, fresh, lambda name: owner(name) in stuck)
        mujoco.mj_forward(model, data)
    return sorted(keep - stuck)


def _apply(model: mujoco.MjModel, data: mujoco.MjData, snap: Snapshot, chosen: Callable[[str], bool]) -> None:
    for j in range(model.njnt):
        adr, dof = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            body = model.body(int(model.jnt_bodyid[j])).name
            if body in snap.free and chosen(body):
                data.qpos[adr : adr + 7] = snap.free[body]
                data.qvel[dof : dof + 6] = 0.0
            continue
        name = model.joint(j).name
        if name in snap.joints and chosen(name):
            value = snap.joints[name]
            data.qpos[adr : adr + len(value)] = value
            data.qvel[dof : dof + (3 if len(value) == 4 else 1)] = 0.0
    for a in range(model.nu):
        name = model.actuator(a).name
        if name in snap.controls and chosen(name):
            data.ctrl[a] = snap.controls[name]


GROUND = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))


def _deep_overlaps(model: mujoco.MjModel, data: mujoco.MjData) -> set[str]:
    """Owners with a geom pressed DEEP into something that is not their own. The ground does not count: a
    walking Go2's feet sink 26 mm into it mid-stride, which flagged it buried and sent it back to its start."""
    found = set()
    for con in data.contact[: data.ncon]:
        if con.dist > -DEEP or any(int(model.geom_type[g]) in GROUND for g in (con.geom1, con.geom2)):
            continue
        a, b = (owner(model.body(int(model.geom_bodyid[g])).name) for g in (con.geom1, con.geom2))
        if a != b:
            found.update(name for name in (a, b) if name and name != "world")
    return found
