"""Batched kinematics and IK for one Zeno Malo arm chain, in numpy (the cuRobo IK stand-in).

`Chain` evaluates ZenoBench's own URDF chain (`ArmKin.chain`: torso lift, waist and seven arm joints,
all URDF rpy zero) for many joint vectors at once: the fingertip TCP pose and its 6 x n Jacobian,
the same quantities `ArmKin.tcp` and `ArmKin.jac` give one at a time.  `solve` runs damped
Levenberg-Marquardt from every seed in parallel; `pick` is upstream's choice among the solutions
(zeno_skills/motion.py, `_Arm.ik`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

W_ROT = 0.35  # rotation error weight against metres (ArmKin's old DLS used the same)
STEP_MAX = 0.2  # rad: largest joint step per iteration (norm)
TOL_P, TOL_R = 5e-4, 5e-3  # converge well inside cuRobo's 4 mm / 0.04 rad success thresholds


class Chain:
    """`ArmKin`'s chain for a batch of joint vectors (rows of Q)."""

    def __init__(self, kin, tcp_depth: float):
        self.steps = [
            (typ, np.asarray(xyz, float), None if axis is None else np.asarray(axis, float))
            for _, typ, xyz, axis in kin.chain
        ]
        self.n = len(kin.names)
        self.prismatic = np.array([typ == "prismatic" for typ, _, _ in self.steps if typ != "fixed"])
        self.depth = float(tcp_depth)

    def tcp(self, Q, base_p, base_R):
        """(B, 3) TCP positions and (B, 3, 3) rotations."""
        p, R, _, _ = self._fk(np.atleast_2d(Q), base_p, base_R)
        return p - self.depth * R[:, :, 2], R

    def tcp_jac(self, Q, base_p, base_R):
        """TCP positions, rotations and (B, 6, n) Jacobians (linear rows first, like ArmKin.jac)."""
        p, R, origins, axes = self._fk(np.atleast_2d(Q), base_p, base_R)
        tcp = p - self.depth * R[:, :, 2]
        lin = np.cross(axes, tcp[:, None, :] - origins)
        pr = self.prismatic[None, :, None]
        J = np.empty((len(tcp), 6, self.n))
        J[:, :3, :] = np.where(pr, axes, lin).transpose(0, 2, 1)
        J[:, 3:, :] = np.where(pr, 0.0, axes).transpose(0, 2, 1)
        return tcp, R, J

    def _fk(self, Q, base_p, base_R):
        B = len(Q)
        p = np.repeat(np.asarray(base_p, float)[None], B, 0)
        R = np.repeat(np.asarray(base_R, float)[None], B, 0)
        origins = np.empty((B, self.n, 3))
        axes = np.empty((B, self.n, 3))
        i = 0
        for typ, xyz, axis in self.steps:
            p = p + R @ xyz
            if typ == "fixed":
                continue
            w = R @ axis
            origins[:, i], axes[:, i] = p, w
            if typ == "prismatic":
                p = p + w * Q[:, i : i + 1]
            else:
                R = R @ _axis_rot(axis, Q[:, i])
            i += 1
        return p, R, origins, axes


def _axis_rot(axis, a):
    """(B, 3, 3) rotations about a fixed unit axis by the angles a (Rodrigues)."""
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(a)[:, None, None], np.cos(a)[:, None, None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def rotvec(Rm):
    """Log map of a batch of rotation matrices (B, 3, 3) -> (B, 3)."""
    tr = np.clip((np.trace(Rm, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(tr)
    v = np.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], 1)
    out = 0.5 * v
    mid = (th > 1e-6) & (th < np.pi - 1e-4)
    out[mid] = (th[mid] / (2.0 * np.sin(th[mid])))[:, None] * v[mid]
    far = th >= np.pi - 1e-4
    if far.any():
        out[far] = Rotation.from_matrix(Rm[far]).as_rotvec()
    return out


def pose_error(chain, Q, base_p, base_R, p_t, R_t):
    """Per row: TCP position error (m) and orientation error (rad)."""
    tcp, R = chain.tcp(Q, base_p, base_R)
    ep = np.linalg.norm(np.asarray(p_t, float) - tcp, axis=1)
    er = np.linalg.norm(rotvec(np.asarray(R_t, float)[None] @ R.transpose(0, 2, 1)), axis=1)
    return ep, er


def solve(chain, base_p, base_R, p_t, R_t, seeds, lo, hi, retract=None, iters=120, null_gain=0.02):
    """Damped least squares on the TCP pose from every seed at once, joints kept inside [lo, hi].

    Returns (Q, pos_err, rot_err).  ``retract`` pulls the null space towards a posture (cuRobo's
    retract config: the first seed)."""
    Q = np.clip(np.array(seeds, float, ndmin=2), lo, hi)
    p_t, R_t = np.asarray(p_t, float), np.asarray(R_t, float)
    W = np.array([1.0, 1.0, 1.0, W_ROT, W_ROT, W_ROT])
    live = np.ones(len(Q), bool)
    eye6 = np.eye(6)
    eyen = np.eye(chain.n)
    ep = er = None
    for it in range(iters):
        tcp, R, J = chain.tcp_jac(Q, base_p, base_R)
        e = np.concatenate([p_t - tcp, rotvec(R_t[None] @ R.transpose(0, 2, 1))], 1)
        ep, er = np.linalg.norm(e[:, :3], axis=1), np.linalg.norm(e[:, 3:], axis=1)
        live &= ~((ep < TOL_P) & (er < TOL_R))
        if it == 60:
            live &= ep < 0.08  # hopeless seeds stop early
        if not live.any():
            break
        idx = np.nonzero(live)[0]
        Jw = W[None, :, None] * J[idx]
        ew = W[None, :] * e[idx]
        lam = 0.01 if it < iters // 2 else 0.002
        A = Jw @ Jw.transpose(0, 2, 1) + lam * eye6
        step = (Jw.transpose(0, 2, 1) @ np.linalg.solve(A, ew[..., None]))[..., 0]
        if retract is not None and null_gain > 0:
            N = eyen - Jw.transpose(0, 2, 1) @ np.linalg.solve(A, Jw)
            step += null_gain * (N @ (np.asarray(retract, float) - Q[idx])[..., None])[..., 0]
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, STEP_MAX / np.maximum(norm, 1e-12))
        Q[idx] = np.clip(Q[idx] + step, lo, hi)
    ep, er = pose_error(chain, Q, base_p, base_R, p_t, R_t)
    return Q, ep, er


def distinct(Q, tol=1e-3):
    """Rows of Q with near-duplicates (max joint difference below tol) removed, order kept."""
    keep = []
    for i, q in enumerate(Q):
        if all(np.max(np.abs(q - Q[j])) >= tol for j in keep):
            keep.append(i)
    return Q[keep]


def pick(kin, Q, seed0, lo, hi, local, limit_clear, manip_w, max_step):
    """Upstream's choice (zeno_skills/motion.py `_Arm.ik`): closest to the first seed, clear of the
    joint limits; ``local`` never trades continuity for the margin, otherwise away from singular
    postures.  Q holds successful solutions only; returns (q, ok)."""
    Q = np.atleast_2d(Q)
    if len(Q) == 0:
        return None, False
    inside = np.all((Q - lo >= -1e-4) & (hi - Q >= -1e-4), axis=1)
    if not inside.any():
        return Q[0], False
    Q = Q[inside]
    seed0 = np.asarray(seed0, float)
    near = np.max(np.abs(Q - seed0), axis=1)
    margin = np.min(np.minimum(Q - lo, hi - Q)[:, 2:], axis=1)
    score = near + 10.0 * np.maximum(0.0, limit_clear - margin)
    if local:
        score = np.where(near <= max_step, score, 1e3 + near)
    else:
        from zeno_skills.kinematics import manipulability

        m = np.array([manipulability(kin, q) for q in Q])
        score = score + manip_w * (1.0 - m / max(float(m.max()), 1e-9))
    return Q[int(np.argmin(score))], True


def halton(n: int, d: int) -> np.ndarray:
    """First n points of the d-dimensional Halton sequence in [0, 1) (deterministic IK seeds)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][:d]
    out = np.empty((n, d))
    for k, b in enumerate(primes):
        for i in range(n):
            f, r, x = 1.0, 0.0, i + 1
            while x > 0:
                f /= b
                r += f * (x % b)
                x //= b
            out[i, k] = r
    return out
