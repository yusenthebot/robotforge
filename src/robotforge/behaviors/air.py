"""The air around an aircraft: drag, and wind that blows in gusts.

Air is part of the world an aircraft flies in, not of what flies it, so it is a session controller of its own
(key AIR + robot) that runs every physics step and pushes the airframe through MuJoCo's external force
(xfrc_applied): the flight controller, LQR, MPC or a script all fly through the same air. MuJoCo's own fluid
model (option density and viscosity) is world-wide and would slow every robot and ball in the world as well.

Drag is a linear rotor-drag term (a spinning propeller moved sideways pushes back in proportion to the speed) and
quadratic body drag; wind is the air's own velocity, so a gust pushes with the drag of the air moving past.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors.airframe import AirframeError, find_airframe

AIR = "__air__"  # session controller key of an aircraft's air: AIR + robot
# Rotor drag per kg of aircraft (N s/m per kg), and body drag as the drag coefficient times frontal area per square
# meter of span (a 0.65 m camera drone: CdA 0.025 m^2).
ROTOR_DRAG = 0.12
BODY_CD = 0.06
AIR_DENSITY = 1.225
RAMP = 0.3  # s a gust takes to rise and to die down
MAX_WIND = 30.0  # m/s; more is a storm no drone flies in (and a number in km/h by mistake)


class WindError(ValueError):
    """User-facing: says what to give instead."""


@dataclass
class Gust:
    """Wind (world frame, m/s) that turns from `before` to `wind` over RAMP s from `start`, blows until `end`,
    then dies down over RAMP s."""

    wind: np.ndarray
    start: float
    end: float
    before: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def at(self, t: float) -> np.ndarray:
        if t >= self.end + RAMP:
            return np.zeros(3)
        if t > self.end:
            return self.wind * ((self.end + RAMP - t) / RAMP)
        k = float(np.clip((t - self.start) / RAMP, 0.0, 1.0))
        return self.before + (self.wind - self.before) * k

    def over(self, t: float) -> bool:
        return t >= self.end + RAMP


class Air:
    """Drag and wind on one aircraft, every physics step (see the module notes)."""

    def __init__(self, model: mujoco.MjModel, robot: str, root: int, mass: float, radius: float) -> None:
        self.robot = robot
        self.model = model
        self.root = root
        joint = int(model.body_jntadr[root])
        self.dadr = int(model.jnt_dofadr[joint])
        self.rate_hz = 1.0 / model.opt.timestep
        span = 2.0 * radius
        self.drag_quad = 0.5 * AIR_DENSITY * BODY_CD * span * span
        self.drag_lin = ROTOR_DRAG * mass
        self.gust: Gust | None = None
        self.force = np.zeros(3)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        t = float(data.time)
        wind = self.gust.at(t) if self.gust is not None else np.zeros(3)
        if self.gust is not None and self.gust.over(t):
            self.gust = None
        rel = data.qvel[self.dadr : self.dadr + 3] - wind
        self.force = -(self.drag_lin + self.drag_quad * float(np.linalg.norm(rel))) * rel
        data.xfrc_applied[self.root, :3] = self.force

    def release(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Taken away (see SimSession.set_controller): the push it applied goes with it."""
        if self.model is model:
            data.xfrc_applied[self.root, :3] = 0.0

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"behavior": "air", "aircraft": self.robot}
        if self.gust is not None:
            out["wind_m_s"] = [round(float(v), 2) for v in self.gust.wind]
        return out

    def gusts(self, wind: list[float] | np.ndarray, seconds: float, now: float) -> None:
        """Wind that blows for `seconds` from `now` and dies down after. The same wind again while it still blows
        keeps it blowing: an agent stepped a gust 0.05 s at a time to watch it, and each call started it afresh
        from still air, so it never blew at all. Other wind turns from what blows now."""
        w = np.zeros(3)
        w[: len(wind)] = np.asarray(wind, float)[:3]
        g = self.gust
        if g is not None and now <= g.end and np.allclose(g.wind, w):
            g.end = max(g.end, now + seconds)
            return
        before = g.at(now) if g is not None else np.zeros(3)
        self.gust = Gust(w, now, now + seconds, before)


def attach_air(session: Any, robot: str) -> Air | None:
    """Give an aircraft its air (None for one that cannot fly, whose rotors could not lift it anyway)."""
    try:
        frame = find_airframe(session.model, robot)
    except AirframeError:
        return None
    air = Air(session.model, robot, frame.root, frame.mass, frame.radius)
    session.set_controller(AIR + robot, air)
    return air


def blow(session: Any, wind: list[float], seconds: float, steps: int) -> dict[str, Any]:
    """simulate(wind=...): wind on every aircraft for `seconds` (`steps` physics steps, taken here), and how far it
    pushed each one. Called with the session lock held."""
    if len(wind) not in (2, 3) or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in wind):
        raise WindError(f"wind is [x, y] or [x, y, z] in m/s, got {wind}")
    if float(np.linalg.norm(wind)) > MAX_WIND:
        raise WindError(f"wind {list(wind)} is over {MAX_WIND:g} m/s: it is in m/s (36 km/h is 10 m/s)")
    airs = {c.robot: c for c in session.controllers.values() if isinstance(c, Air)}
    if not airs:
        raise WindError("wind only pushes aircraft, and there is none here (add_robot template='drone')")
    now = float(session.data.time)
    start = {n: session.data.xpos[a.root].copy() for n, a in airs.items()}
    for a in airs.values():
        a.gusts(wind, seconds, now)
    worst = dict.fromkeys(airs, 0.0)
    done = 0
    while done < steps:
        chunk = min(10, steps - done)
        session.step(chunk)
        done += chunk
        for n, a in airs.items():
            worst[n] = max(worst[n], float(np.linalg.norm(session.data.xpos[a.root] - start[n])))
    end = {n: float(np.linalg.norm(session.data.xpos[a.root] - start[n])) for n, a in airs.items()}
    return {
        "m_s": [float(v) for v in wind],
        "pushed_m": {n: round(v, 3) for n, v in worst.items()},
        "off_by_m": {n: round(v, 3) for n, v in end.items()},
    }
