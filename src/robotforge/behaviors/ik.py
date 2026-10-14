"""Damped least-squares inverse kinematics on MuJoCo Jacobians.

IK runs on a private MjData so it never disturbs the live simulation. Works for
any kinematic chain, which is what lets one gait or reach routine drive both our
templates and imported robots.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


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
    ) -> tuple[np.ndarray, float]:
        """Joint values placing the chain's point at `target` (in the base frame of base_qpos).

        `axis` = (local_axis, desired_world_axis) also asks the end body's local axis to point
        along a direction, e.g. a gripper pointing straight down. Returns (q, position error).
        """
        q = np.clip(np.asarray(q_init, float).copy(), chain.lower, chain.upper)
        err = np.inf
        for _ in range(iterations):
            self._set(chain, q)
            p = self._point(chain)
            e = np.asarray(target, float) - p
            err = float(np.linalg.norm(e))
            mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, p, chain.body)
            J = self._jacp[:, chain.dof_adr]
            if axis is not None:
                local_axis, want = axis
                rot = self.data.xmat[chain.body].reshape(3, 3)
                have = rot @ local_axis
                e_rot = np.cross(have, want) * axis_weight
                J = np.vstack([J, self._jacr[:, chain.dof_adr] * axis_weight])
                e = np.concatenate([e, e_rot])
            if err < tol and (axis is None or np.linalg.norm(e[3:]) < tol * 10):
                break
            JJt = J @ J.T + damping * np.eye(J.shape[0])
            q = np.clip(q + J.T @ np.linalg.solve(JJt, e), chain.lower, chain.upper)
        return q, err

    def _set(self, chain: Chain, q: np.ndarray) -> None:
        self.data.qpos[:] = self.base_qpos
        self.data.qpos[chain.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # mj_jac needs the motion subspaces (cdof)

    def _point(self, chain: Chain) -> np.ndarray:
        rot = self.data.xmat[chain.body].reshape(3, 3)
        return self.data.xpos[chain.body] + rot @ chain.local_point
