"""The language between a task and an arm controller: where the tool point should be, and what it feels.

A task that moves an arm's tool (reach, press, wipe, insert, turn) hands out a `ToolTarget` every control step through a
`TargetSource`; the controllers in control/impedance.py turn it into joint torques. `ToolForce` is the sense of touch:
the force on the hand and what it holds, as a wrist force-torque sensor would read it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np


def min_jerk(s: float) -> tuple[float, float]:
    """Smooth 0 -> 1 profile (position, and its rate per unit s) with zero speed at both ends."""
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 - 60 * s**3 + 30 * s**4


def rotation_error(current: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The small rotation (world axes, radians) that turns the frame `current` into `wanted`."""
    return 0.5 * sum(np.cross(current[:, i], wanted[:, i]) for i in range(3))


def pointing(axis: np.ndarray, current: np.ndarray) -> np.ndarray:
    """The orientation closest to `current` (a 3 x 3 frame) whose z axis points along `axis`."""
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    x = current[:, 0] - z * float(z @ current[:, 0])
    if np.linalg.norm(x) < 1e-6:  # the old x axis is along the new z: any perpendicular will do
        x = np.cross(z, [0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


@dataclass
class ToolTarget:
    """What a task wants of the tool point at one instant."""

    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray | None = None  # the tool's frame, world axes; None leaves the orientation free
    normal: np.ndarray | None = None  # unit vector from a surface toward the tool, when the tool presses on it
    push: float = 0.0  # N the tool presses against the surface (along -normal)
    pressing: bool = False  # True: the normal axis is force controlled where the controller can (position is not)


class TargetSource(Protocol):
    """A task's plan for one run: asked every control step where the tool should be, given what it feels now."""

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        """`feel` is the force the surroundings exert on the tool, world axes, N (zeros when nothing touches)."""
        ...


class ToolForce:
    """The contact force on an arm's hand and whatever it holds, like a wrist force-torque sensor reads it.

    Contacts between the tool (the last arm link and everything below it, plus the held bodies) and anything that is
    neither the tool nor part of the same robot count; the hand against its own arm does not."""

    def __init__(self, model: mujoco.MjModel, tool_bodies: set[int], own_bodies: set[int]):
        bodies = model.geom_bodyid
        self.tool = np.isin(bodies, sorted(tool_bodies))
        self.own = np.isin(bodies, sorted(own_bodies)) & ~self.tool
        self._f6 = np.zeros(6)
        self._world = np.zeros(3)

    @classmethod
    def of_arm(cls, model: mujoco.MjModel, task: Any, held: set[int] | None = None) -> ToolForce:
        """For a task's arm: the tool is the body of the arm's last joint and everything below it."""
        body = int(model.jnt_bodyid[task.joints[-1]])
        tool = {b for b in range(model.nbody) if _below(model, b, body)} | set(held or ())
        prefix = task.robot.prefix
        own = {
            b
            for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
        }
        return cls(model, tool, own)

    def read(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """The force the surroundings exert on the tool now (world axes, N)."""
        n = data.ncon
        out = np.zeros(3)
        if n == 0:
            return out
        g1, g2 = data.contact.geom1[:n], data.contact.geom2[:n]
        t1, t2 = self.tool[g1], self.tool[g2]
        outside = (t1 & ~t2 & ~self.own[g2]) | (t2 & ~t1 & ~self.own[g1])
        f6 = self._f6
        for i in np.nonzero(outside)[0]:
            mujoco.mj_contactForce(model, data, int(i), f6)
            world = data.contact.frame[i].reshape(3, 3).T @ f6[:3]  # on geom2, from geom1
            out += -world if t1[i] else world
        return out


def _below(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return body == ancestor
