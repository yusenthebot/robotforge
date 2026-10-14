"""A live MuJoCo simulation that the viewer, CLI and MCP tools share.

Physics is authoritative here; viewers only render the poses we publish.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

# Spring used when the user drags a body: natural frequency (rad/s) and damping ratio.
_DRAG_OMEGA = 12.0
_DRAG_ZETA = 1.0
# Never simulate more than this much wall time in one tick, so a stalled tab
# or a slow model degrades into slow motion instead of a burst of steps.
_MAX_TICK_SECONDS = 0.1


class ModelLoadError(RuntimeError):
    """Raised when MuJoCo cannot compile a model; carries the compiler message."""


@dataclass
class Drag:
    body: int
    local_point: np.ndarray  # anchor in body frame
    target: np.ndarray  # desired anchor position in world frame


class Controller(Protocol):
    """Something that drives actuators while time passes (a gait, an IK motion, a script)."""

    rate_hz: float

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...

    def status(self) -> dict: ...


@dataclass
class SimSession:
    """Owns one MjModel/MjData pair and advances it in (scaled) real time."""

    model: mujoco.MjModel
    data: mujoco.MjData
    source: str = "<memory>"
    paused: bool = False
    speed: float = 1.0
    version: int = 1
    drag: Drag | None = None
    controllers: dict[str, Controller] = field(default_factory=dict)
    clock_owner: str | None = None  # set while e.g. a script advances time itself
    # Run after a reset or a model swap clears the controllers, e.g. so humanoids keep their balance.
    reset_hooks: list[Callable[[], None]] = field(default_factory=list, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _next_update: dict[str, float] = field(default_factory=dict, repr=False)
    _sim_debt: float = 0.0
    _rtf_window: list[tuple[float, float]] = field(default_factory=list, repr=False)

    # ---- construction / loading -------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path) -> SimSession:
        model = _compile_path(Path(path))
        return cls(model=model, data=_fresh_data(model), source=str(path))

    @classmethod
    def from_xml(cls, xml: str, assets: dict[str, bytes] | None = None) -> SimSession:
        model = _compile_xml(xml, assets)
        return cls(model=model, data=_fresh_data(model))

    def load_path(self, path: str | Path) -> None:
        self._swap(_compile_path(Path(path)), str(path))

    def load_xml(self, xml: str, assets: dict[str, bytes] | None = None) -> None:
        self._swap(_compile_xml(xml, assets), "<memory>")

    def load_model(self, model: mujoco.MjModel, source: str) -> None:
        """Swap in an already compiled model (e.g. from a world.yaml)."""
        self._swap(model, source)

    def _swap(self, model: mujoco.MjModel, source: str) -> None:
        with self.lock:
            self.model = model
            self.data = _fresh_data(model)
            self.source = source
            self.drag = None
            self._sim_debt = 0.0
            self.controllers.clear()  # controllers hold ids from the old model
            self.version += 1
            self._run_reset_hooks()

    # ---- control ----------------------------------------------------------------

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            if self.model.nkey > 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            mujoco.mj_forward(self.model, self.data)
            self.drag = None
            self._sim_debt = 0.0
            self.controllers.clear()
            self._run_reset_hooks()

    def _run_reset_hooks(self) -> None:
        for hook in self.reset_hooks:
            hook()

    def set_ctrl(self, actuator: int, value: float) -> None:
        with self.lock:
            if not 0 <= actuator < self.model.nu:
                raise IndexError(f"actuator {actuator} out of range (nu={self.model.nu})")
            if self.model.actuator_ctrllimited[actuator]:
                lo, hi = self.model.actuator_ctrlrange[actuator]
                value = float(np.clip(value, lo, hi))
            self.data.ctrl[actuator] = value

    def start_drag(self, body: int, world_point: np.ndarray) -> None:
        with self.lock:
            if not 0 < body < self.model.nbody:
                raise IndexError(f"body {body} is not draggable")
            rot = self.data.xmat[body].reshape(3, 3)
            local = rot.T @ (np.asarray(world_point, float) - self.data.xpos[body])
            self.drag = Drag(body=body, local_point=local, target=np.asarray(world_point, float))

    def move_drag(self, world_target: np.ndarray) -> None:
        with self.lock:
            if self.drag is not None:
                self.drag.target = np.asarray(world_target, float)

    def end_drag(self) -> None:
        with self.lock:
            self.drag = None

    # ---- stepping ---------------------------------------------------------------

    def set_controller(self, name: str, controller: Controller | None) -> None:
        """Attach (or with None, remove) a controller; it runs on every future step at its rate."""
        with self.lock:
            if controller is None:
                self.controllers.pop(name, None)
                return
            self.controllers[name] = controller
            self._next_update[name] = self.data.time

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                self._run_controllers()
                self._apply_drag()
                mujoco.mj_step(self.model, self.data)

    def _run_controllers(self) -> None:
        t = self.data.time
        for name, controller in list(self.controllers.items()):
            if t + 1e-9 >= self._next_update.get(name, 0.0):
                controller.update(self.model, self.data)
                self._next_update[name] = t + 1.0 / controller.rate_hz

    def advance(self, wall_dt: float) -> int:
        """Advance by `wall_dt` seconds of wall time scaled by `speed`. Returns steps taken."""
        with self.lock:
            if self.paused:
                return 0
            self._sim_debt += min(wall_dt, _MAX_TICK_SECONDS) * self.speed
            n = int(self._sim_debt / self.model.opt.timestep)
            if n <= 0:
                return 0
            start_sim = self.data.time
            self.step(n)
            self._sim_debt -= n * self.model.opt.timestep
            self._record_rtf(self.data.time - start_sim)
            return n

    def _apply_drag(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        if self.drag is None:
            return
        m, d, body = self.model, self.data, self.drag.body
        rot = d.xmat[body].reshape(3, 3)
        anchor = d.xpos[body] + rot @ self.drag.local_point
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, vel6, 0)
        anchor_vel = vel6[3:] + np.cross(vel6[:3], anchor - d.xpos[body])
        mass = max(float(m.body_subtreemass[body]), 1e-6)
        force = mass * (_DRAG_OMEGA**2 * (self.drag.target - anchor) - 2 * _DRAG_ZETA * _DRAG_OMEGA * anchor_vel)
        mujoco.mj_applyFT(m, d, force, np.zeros(3), anchor, body, d.qfrc_applied)
        # Hold the dragged subtree's weight so it follows the cursor instead of sagging g/omega^2 below it.
        mujoco.mj_applyFT(m, d, -mass * m.opt.gravity, np.zeros(3), d.subtree_com[body], body, d.qfrc_applied)

    def _record_rtf(self, sim_dt: float) -> None:
        now = time.perf_counter()
        self._rtf_window.append((now, sim_dt))
        cutoff = now - 1.0
        while self._rtf_window and self._rtf_window[0][0] < cutoff:
            self._rtf_window.pop(0)

    @property
    def realtime_factor(self) -> float:
        """Simulated seconds per wall second over the last second."""
        if len(self._rtf_window) < 2:
            return 0.0
        span = self._rtf_window[-1][0] - self._rtf_window[0][0]
        if span <= 0:
            return 0.0
        return sum(dt for _, dt in self._rtf_window[1:]) / span

    # ---- snapshots ----------------------------------------------------------------

    def body_poses(self) -> np.ndarray:
        """Float32 array: [time, xpos (nbody*3), xquat (nbody*4)]."""
        with self.lock:
            return np.concatenate(([self.data.time], self.data.xpos.ravel(), self.data.xquat.ravel())).astype(
                np.float32
            )


def _fresh_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _compile_path(path: Path) -> mujoco.MjModel:
    if not path.exists():
        raise ModelLoadError(f"model file not found: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        raise ModelLoadError(f"MuJoCo could not load {path.name}: {exc}") from exc


def _compile_xml(xml: str, assets: dict[str, bytes] | None) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_string(xml, assets or {})
    except ValueError as exc:
        raise ModelLoadError(f"MuJoCo could not compile the model: {exc}") from exc
