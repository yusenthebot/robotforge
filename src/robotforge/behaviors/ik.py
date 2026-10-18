"""Damped least-squares inverse kinematics on MuJoCo Jacobians.

IK runs on a private MjData so it never disturbs the live simulation. Works for
any kinematic chain, which is what lets one gait or reach routine drive both our
templates and imported robots.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np


def _fast_solve() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """numpy.linalg.solve without its Python wrapping (array checks, an error-state context): the same LAPACK
    routine, so the same numbers, in a quarter of the time on the 3 x 3 systems IK solves a thousand times a second
    for a walking robot. Its private home in numpy may move; then IK uses the public function."""
    try:
        from numpy.linalg import _umath_linalg

        def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # numpy.linalg.solve raises only when LAPACK finds the matrix singular; IK's are damped, so never.
            return _umath_linalg.solve1(a, b, signature="dd->d")

        a, b = np.array([[2.0, 0.5, 0.0], [0.5, 3.0, 0.1], [0.0, 0.1, 1.0]]), np.array([1.0, -2.0, 0.5])
        if np.array_equal(solve(a, b), np.linalg.solve(a, b)):
            return solve
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return np.linalg.solve


_solve = _fast_solve()


@dataclass(frozen=True)
class Chain:
    """The joints IK may move and the point it should place."""

    qpos_adr: np.ndarray  # int indices into qpos
    dof_adr: np.ndarray  # int indices into qvel / Jacobian columns
    lower: np.ndarray
    upper: np.ndarray
    body: int  # body carrying the end point
    local_point: np.ndarray  # end point in that body's frame

    @classmethod
    def from_joints(cls, model: mujoco.MjModel, joints: list[int], body: int, local_point) -> Chain:
        qadr = np.array([model.jnt_qposadr[j] for j in joints])
        dadr = np.array([model.jnt_dofadr[j] for j in joints])
        lower = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -np.pi for j in joints])
        upper = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else np.pi for j in joints])
        return cls(qadr, dadr, lower, upper, body, np.asarray(local_point, float))


class IKSolver:
    def __init__(self, model: mujoco.MjModel, base_qpos: np.ndarray | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_qpos = model.qpos0.copy() if base_qpos is None else base_qpos.copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._damped: dict[tuple[int, float], np.ndarray] = {}  # damping * identity, by size and damping

    def point(self, chain: Chain, q: np.ndarray) -> np.ndarray:
        self._set(chain, q)
        return self._point(chain)

    def solve(
        self,
        chain: Chain,
        target: np.ndarray,
        q_init: np.ndarray,
        iterations: int = 12,
        damping: float = 1e-3,
        tol: float = 1e-4,
        axis: tuple[np.ndarray, np.ndarray] | None = None,
        axis_weight: float = 0.3,
        rotation: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Joint values placing the chain's point at `target` (in the base frame of base_qpos).

        `axis` = (local_axis, desired_world_axis) also asks the end body's local axis to point
        along a direction, e.g. a gripper pointing straight down. `rotation` (3x3) instead asks
        for the end body's full orientation, e.g. a foot kept flat and turned to a heading.
        Returns (q, position error).
        """
        if axis is not None and rotation is not None:
            raise ValueError("give IK either an axis to point or a full rotation, not both")
        # Walking robots solve this four times a plan, a hundred plans a second, so what is the same every
        # iteration is done once and small vectors are measured with floats: the same numbers as before.
        q = np.clip(np.asarray(q_init, float), chain.lower, chain.upper)
        target = np.asarray(target, float)
        err = np.inf
        for _ in range(iterations):
            self._set(chain, q)
            p = self._point(chain)
            e = target - p
            ex, ey, ez = e.tolist()
            err = math.sqrt(ex * ex + ey * ey + ez * ez)
            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, p, chain.body)
            J = self._jacp[:, chain.dof_adr]
            if axis is not None:
                local_axis, want = axis
                rot = self.data.xmat[chain.body].reshape(3, 3)
                have = rot @ local_axis
                e_rot = np.cross(have, want) * axis_weight
                J = np.vstack([J, self._jacr[:, chain.dof_adr] * axis_weight])
                e = np.concatenate([e, e_rot])
            elif rotation is not None:
                e_rot = _rotation_error(self.data.xquat[chain.body], rotation) * axis_weight
                J = np.vstack([J, self._jacr[:, chain.dof_adr] * axis_weight])
                e = np.concatenate([e, e_rot])
            if err < tol and (len(e) == 3 or np.linalg.norm(e[3:]) < tol * 10):
                break
            n = J.shape[0]
            damped = self._damped.get((n, damping))
            if damped is None:
                damped = self._damped[n, damping] = damping * np.eye(n)
            JJt = J @ J.T + damped  # damped, so never singular
            q = np.minimum(np.maximum(q + J.T @ _solve(JJt, e), chain.lower), chain.upper)
        return q, err

    def _set(self, chain: Chain, q: np.ndarray) -> None:
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[chain.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # mj_jac needs the motion subspaces (cdof)

    def _point(self, chain: Chain) -> np.ndarray:
        rot = self.data.xmat[chain.body].reshape(3, 3)
        return self.data.xpos[chain.body] + rot @ chain.local_point


def _rotation_error(have_quat: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that turns orientation `have_quat` into rotation matrix `want`."""
    want_quat = np.zeros(4)
    mujoco.mju_mat2Quat(want_quat, np.ascontiguousarray(want, float).ravel())
    local = np.zeros(3)
    mujoco.mju_subQuat(local, want_quat, have_quat)  # have * quat(local) = want
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, have_quat)
    return rot.reshape(3, 3) @ local
