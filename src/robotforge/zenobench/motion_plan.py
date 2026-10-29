"""Joint-space planning with OMPL for the cuRobo stand-in (RRT-Connect, then OMPL's path simplifier).

cuRobo seeds a trajectory optimiser with a graph search; here a bidirectional RRT finds a collision-free
path in the arm's joint space and OMPL shortcuts and smooths it, every edit re-checked.  A constrained
plan (a held object's tilt kept level) runs in OMPL's projected state space: every sampled or
interpolated state is pulled back onto the constraint by Newton steps on its Jacobian.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable

import numpy as np

SEED = 7  # OMPL's global RNG seed: the same episode plans the same paths
_seeded = False


def _ompl():
    global _seeded
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    if not _seeded:
        ou.setLogLevel(ou.LogLevel.LOG_WARN)
        ou.RNG.setSeed(SEED)  # before any planner allocates its own RNG
        _seeded = True
    return ob, og


class Constraint:
    """F(q) = 0 for a constrained plan: ``f(q) -> (m,)`` and ``jac(q) -> (m, n)``."""

    def __init__(self, f: Callable, jac: Callable, m: int, tol: float = 1e-4):
        self.f, self.jac, self.m, self.tol = f, jac, m, tol

    def project(self, q, lo, hi, iters: int = 50) -> np.ndarray | None:
        """Newton (minimum-norm) steps onto F = 0 inside [lo, hi]; None when it does not converge."""
        q = np.clip(np.asarray(q, float).copy(), lo, hi)
        for _ in range(iters):
            e = self.f(q)
            if np.linalg.norm(e) < self.tol:
                return q
            J = self.jac(q)
            q = np.clip(q - np.linalg.pinv(J) @ e, lo, hi)
        return q if np.linalg.norm(self.f(q)) < self.tol else None


def plan(
    q0,
    q1,
    lo,
    hi,
    valid: Callable[[np.ndarray], bool],
    resolution: float,
    timeout: float,
    attempts: int,
    constraint: Constraint | None = None,
    simplify: float = 1.0,
):
    """Waypoints after q0 to q1 (q1 last), or (None, status).  ``valid`` judges one joint vector;
    edges are checked every ``resolution`` (joint-space distance)."""
    ob, og = _ompl()
    n = len(q0)
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i in range(n):
        bounds.setLow(i, float(lo[i]))
        bounds.setHigh(i, float(hi[i]))
    space.setBounds(bounds)
    cons = None
    if constraint is not None:
        cons = _OmplConstraint(ob, n, constraint)
        css = ob.ProjectedStateSpace(space, cons)
        si = ob.ConstrainedSpaceInformation(css)
        css.setDelta(resolution)
        work = css
    else:
        si = ob.SpaceInformation(space)
        work = space
    work.setLongestValidSegmentFraction(resolution / max(work.getMaximumExtent(), 1e-9))
    ss = og.SimpleSetup(si)
    ss.setStateValidityChecker(lambda s: bool(valid(np.fromiter((s[i] for i in range(n)), float, n))))
    start, goal = work.allocState(), work.allocState()
    for state, q in ((start, q0), (goal, q1)):
        if constraint is not None:
            state.copy([float(v) for v in q])  # constrained states take no item assignment
        else:
            for i, v in enumerate(q):
                state[i] = float(v)
    ss.setStartAndGoalStates(start, goal)
    ss.setPlanner(og.RRTConnect(si))
    status = "no plan"
    t_end = time.monotonic() + timeout
    try:
        for _ in range(max(1, attempts)):
            left = t_end - time.monotonic()
            if left <= 0:
                status = "timeout"
                break
            res = ss.solve(min(left, timeout / max(1, attempts) * 2))
            if ss.haveExactSolutionPath():
                break
            status = f"no exact solution ({res})"
            ss.clear()
        else:
            return None, status
        if not ss.haveExactSolutionPath():
            return None, status
        if simplify > 0:
            ss.simplifySolution(simplify)
        path = ss.getSolutionPath()
        if constraint is not None:
            path.interpolate()  # dense on the manifold: the executor interpolates linearly
        qs = [np.fromiter((path.getState(k)[i] for i in range(n)), float, n) for k in range(path.getStateCount())]
    finally:
        del ss, start, goal, si, work, space, bounds, cons
        gc.collect()
    out = qs[1:]
    if not out or np.max(np.abs(out[-1] - np.asarray(q1, float))) > 1e-6:
        out.append(np.asarray(q1, float))
    return out, None


def _ompl_constraint_class(ob):
    class _Wrapped(ob.Constraint):
        def __init__(self, n, c):
            super().__init__(n, c.m)
            self.setTolerance(c.tol)
            self.c = c

        def function(self, x, out):
            out[:] = self.c.f(np.asarray(x, float))

        def jacobian(self, x, out):
            out[:, :] = self.c.jac(np.asarray(x, float))

    return _Wrapped


def _OmplConstraint(ob, n, c):  # noqa: N802 - builds an ob.Constraint subclass instance
    return _ompl_constraint_class(ob)(n, c)
