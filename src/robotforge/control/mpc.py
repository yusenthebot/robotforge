"""Sampling-based model predictive control: predictive sampling and MPPI, as in MuJoCo MPC.

Every replan (e.g. 50 times a second) the planner:

1. shifts last plan forward in time (warm start),
2. makes many noisy variations of it,
3. simulates every variation a short time into the future with `mujoco.rollout`
   (multithreaded C, no Python per step),
4. scores each predicted future with the task's cost,
5. keeps the best one (predictive sampling) or a cost-weighted average (MPPI).

A plan is a few "knots" (control values at evenly spaced times over the horizon)
joined by a spline, so a 32-sample search over a 0.5 s future is only a few
dozen numbers per sample. No gradients and no learning: it works on any robot the
physics engine can simulate, which is why it is a good first "smart" controller.
"""

from __future__ import annotations

import atexit
import copy
import os
import threading
import time
from typing import Any, Literal, Protocol

import mujoco
import numpy as np
from mujoco import rollout
from pydantic import BaseModel, ConfigDict, Field

from robotforge.control.actuation import ActionScale
from robotforge.control.tasks import Batch, StateLayout, Task

FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS
_NOISE_SIZES = np.array([1.0, 0.3, 0.1])
_PREVIEW_DT = 0.02  # s between look-ahead samples of a base controller


class MPCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["predictive", "mppi"] = "predictive"
    horizon: float = Field(0.5, gt=0.02, le=5.0, description="how far ahead to predict, seconds")
    knots: int = Field(4, ge=2, le=20, description="control points along the horizon")
    samples: int = Field(32, ge=2, le=1024, description="candidate plans simulated per replan")
    noise: float = Field(0.1, gt=0, le=2.0, description="exploration noise, as a fraction of each control's range")
    plan_dt: float = Field(0.01, gt=0, le=0.05, description="physics step used inside predictions, seconds")
    replan_hz: float = Field(50.0, gt=0, le=1000.0, description="how often to replan, per simulated second")
    iterations: int = Field(1, ge=1, le=10, description="optimizer iterations per replan")
    temperature: float = Field(0.1, gt=0, le=10.0, description="MPPI only: lower = trust the best samples more")
    interpolation: Literal["zero", "linear", "cubic"] = "linear"
    threads: int | None = Field(None, ge=1, le=64, description="rollout threads; default: most CPU cores")
    seed: int = 0


# One thread pool shared by all planners (a pool per controller would leak threads).
_pool_lock = threading.Lock()
_pool: rollout.Rollout | None = None
_pool_threads = 0


def _rollout_pool(nthread: int) -> rollout.Rollout:
    global _pool, _pool_threads
    if _pool is None or _pool_threads != nthread:
        if _pool is not None:
            _pool.close()
        _pool, _pool_threads = rollout.Rollout(nthread=nthread), nthread
    return _pool


@atexit.register
def _close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def default_threads(samples: int) -> int:
    return max(1, min(samples, (os.cpu_count() or 2) - 2, 16))


def interpolate(knots: np.ndarray, times: np.ndarray, horizon: float, kind: str) -> np.ndarray:
    """Evaluate spline(s) through `knots` [..., K, m] at `times` (seconds from the first knot) -> [..., T, m]."""
    k = knots.shape[-2]
    u = np.clip(np.asarray(times, float) / horizon, 0.0, 1.0) * (k - 1)
    i = np.minimum(np.floor(u).astype(int), k - 2)
    f = (u - i)[:, None]
    p0, p1 = knots[..., i, :], knots[..., i + 1, :]
    if kind == "zero":
        return knots[..., np.minimum(np.floor(u).astype(int), k - 1), :]
    if kind == "linear":
        return p0 + f * (p1 - p0)
    # Cubic Hermite with finite-difference slopes (Catmull-Rom), clamped at the ends.
    slopes = np.empty_like(knots)
    slopes[..., 1:-1, :] = (knots[..., 2:, :] - knots[..., :-2, :]) / 2
    slopes[..., 0, :] = knots[..., 1, :] - knots[..., 0, :]
    slopes[..., -1, :] = knots[..., -1, :] - knots[..., -2, :]
    m0, m1 = slopes[..., i, :], slopes[..., i + 1, :]
    f2, f3 = f * f, f * f * f
    return (2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * m0 + (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * m1


class Base(Protocol):
    """A controller MPC can correct ("residual" MPC): it must also say what it will command next."""

    rate_hz: float

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def preview(self, model: mujoco.MjModel, data: mujoco.MjData, times: np.ndarray) -> np.ndarray:
        """Commands for the task's actuators at `times` seconds from now, shape [len(times), n]."""
        ...


class SamplingMPC:
    """A Controller that replans with sampled rollouts and applies the current best plan.

    With `base`, the plan is a correction added to what another controller commands (e.g. the
    trot gait): sampling only has to find small adjustments instead of inventing walking."""

    def __init__(
        self,
        task: Task,
        model: mujoco.MjModel,
        config: MPCConfig | dict[str, Any] | None = None,
        base: Base | None = None,
    ):
        self.task = task
        self.base = base
        self.base_ctrl: np.ndarray | None = None
        self._next_base = -np.inf
        self._preview: np.ndarray | None = None
        cfg = config if isinstance(config, MPCConfig) else MPCConfig.model_validate(config or {})
        self.cfg = cfg
        self.model = model
        if abs(model.opt.timestep - cfg.plan_dt) > 1e-12:
            # Predictions use a coarser step than the live physics: 5x fewer steps, close enough.
            self.plan_model = copy.copy(model)
            self.plan_model.opt.timestep = max(cfg.plan_dt, model.opt.timestep)
        else:
            self.plan_model = model
        self.dt = float(self.plan_model.opt.timestep)
        self.steps = max(2, round(cfg.horizon / self.dt))
        self.step_times = np.arange(self.steps) * self.dt
        self.scale = ActionScale(model, task.actuators)
        self.sigma = cfg.noise * self.scale.half
        self.layout = StateLayout.of(self.plan_model)
        self.threads = cfg.threads or default_threads(cfg.samples)
        self.datas = [mujoco.MjData(self.plan_model) for _ in range(self.threads)]
        self.rng = np.random.default_rng(cfg.seed)
        self.rate_hz = 1.0 / model.opt.timestep
        self.knot_times = np.linspace(0.0, cfg.horizon, cfg.knots)
        self.nominal: np.ndarray | None = None  # [knots, n_actuators]: ctrl values, or corrections with a base
        self.plan_start = 0.0
        self.next_plan = -np.inf
        self._state = np.zeros(self.layout.size)
        # Bookkeeping for the status report and the comparison table.
        self.plans = 0
        self.plan_seconds = 0.0
        self.last_plan_ms = 0.0
        self.best_cost = float("nan")

    # ---- Controller protocol ----

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.base is not None and data.time + 1e-9 >= self._next_base:
            self.base.update(model, data)
            self.base_ctrl = data.ctrl[self.task.actuators].copy()
            self._next_base = data.time + 1.0 / self.base.rate_hz
        if data.time + 1e-9 >= self.next_plan:
            self.plan(data)
            self.next_plan = data.time + 1.0 / self.cfg.replan_hz
        data.ctrl[self.task.actuators] = self.policy(data.time)

    def status(self) -> dict[str, Any]:
        return {
            "controller": "mpc" if self.cfg.method == "predictive" else "mppi",
            "samples": self.cfg.samples,
            "horizon_s": self.cfg.horizon,
            "plans": self.plans,
            "plan_ms": round(self.last_plan_ms, 2),
            "plan_hz": round(self.plan_hz, 1),
            "best_cost": round(self.best_cost, 4) if np.isfinite(self.best_cost) else None,
            **({"on_top_of": self.base.status()} if self.base is not None else {}),
        }

    @property
    def plan_hz(self) -> float:
        """How many replans per wall-clock second this machine can do (1 / mean planning time)."""
        return self.plans / self.plan_seconds if self.plan_seconds > 0 else 0.0

    # ---- planning ----

    def policy(self, t: float) -> np.ndarray:
        if self.nominal is None:
            return np.zeros(len(self.task.actuators))
        u = interpolate(self.nominal, np.array([t - self.plan_start]), self.cfg.horizon, self.cfg.interpolation)[0]
        if self.base_ctrl is not None:
            u = self.base_ctrl + u
        return self.scale.clip(u)

    def plan(self, data: mujoco.MjData) -> None:
        started = time.perf_counter()
        t = float(data.time)
        if self.nominal is None:
            start = np.zeros(len(self.task.actuators)) if self.base else self.task.nominal_ctrl(data)
            self.nominal = np.tile(start, (self.cfg.knots, 1))
        else:
            # Warm start: last plan, evaluated from now on (held at its final value past its end).
            shifted = interpolate(
                self.nominal, t - self.plan_start + self.knot_times, self.cfg.horizon, self.cfg.interpolation
            )
            self.nominal = shifted if self.base else self.scale.clip(shifted)
        self.plan_start = t
        self._preview = None
        if self.base is not None:
            # What the base will command over the horizon, sampled coarsely and interpolated.
            coarse = np.linspace(0.0, self.cfg.horizon, max(2, round(self.cfg.horizon / _PREVIEW_DT) + 1))
            ahead = self.base.preview(self.model, data, coarse)
            self._preview = np.stack(
                [np.interp(self.step_times, coarse, ahead[:, j]) for j in range(ahead.shape[1])], axis=1
            )
        mujoco.mj_getState(self.model, data, self._state, FULLPHYSICS)
        for _ in range(self.cfg.iterations):
            self._improve(data)
        self.last_plan_ms = (time.perf_counter() - started) * 1000
        self.plans += 1
        self.plan_seconds += self.last_plan_ms / 1000

    def _improve(self, data: mujoco.MjData) -> None:
        cfg, acts = self.cfg, self.task.actuators
        n, k = cfg.samples, cfg.knots
        # Three noise sizes: big steps to explore, small ones to fine-tune (precision near the goal).
        sizes = _NOISE_SIZES[np.arange(n - 1) % len(_NOISE_SIZES)][:, None, None]
        noise = self.rng.standard_normal((n - 1, k, len(acts))) * self.sigma * sizes
        if self._preview is None:
            candidates = np.concatenate([self.nominal[None], self.scale.clip(self.nominal[None] + noise)])
        else:
            candidates = np.concatenate([self.nominal[None], self.nominal[None] + noise])
        ctrl = np.repeat(np.asarray(data.ctrl, float)[None, None, :], self.steps, axis=1).repeat(n, axis=0)
        planned = interpolate(candidates, self.step_times, cfg.horizon, cfg.interpolation)
        ctrl[..., acts] = planned if self._preview is None else self.scale.clip(self._preview[None] + planned)
        with _pool_lock:
            pool = _rollout_pool(self.threads)
            state, sensordata = pool.rollout(
                self.plan_model, self.datas, np.tile(self._state, (n, 1)), ctrl, nstep=self.steps
            )
        batch = Batch.from_rollout(self.plan_model, self.layout, state, sensordata, ctrl)
        running = self.task.cost(batch)
        total = running.sum(axis=1) * self.dt + self.task.terminal_cost(batch)
        total = np.where(np.isfinite(total), total, np.inf)  # an exploding rollout is never chosen
        best = int(np.argmin(total))
        self.best_cost = float(total[best])
        if cfg.method == "predictive":
            self.nominal = candidates[best]
            return
        finite = np.isfinite(total)
        spread = float(np.mean(total[finite]) - total[best]) if finite.any() else 0.0
        lam = cfg.temperature * max(spread, 1e-9)
        weights = np.where(finite, np.exp(-(total - total[best]) / lam), 0.0)
        weights /= weights.sum()
        self.nominal = np.tensordot(weights, candidates, axes=1)
