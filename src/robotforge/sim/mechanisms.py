"""The mechanisms of a world while time passes: what each one is doing, and what keeps them going.

A conveyor's belt is a plate that slides along with the things on it; every time it has gone one wrap
(design/mechanisms.BELT_WRAP) it is put back by exactly that. The plate is flat and the same all along, so the
things on it touch the same surface before and after, and MuJoCo's friction works on velocities, not on where a
contact started: nothing on the belt feels the put-back, and the belt runs for as long as time does. Things ride
on it by plain contact friction, so they speed up, slip, stop against a wall and fall off the end as they would on
a real belt. A button's presses are counted as they happen, so a quick press between two looks is not missed.

`Watcher` does both on every physics step; `attach` keeps one on the session through resets and model swaps.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from robotforge.design import mechanisms as design

KEY = "__mechanisms__"  # its key among the session's controllers


class Watcher:
    """Runs on every step: puts belts back by a wrap and counts button presses."""

    hidden = True  # not a behavior of any robot: left out of the lists agents and the viewer read

    def __init__(self, model: mujoco.MjModel, shapes: dict[str, str]) -> None:
        """Watch the conveyors and buttons among the world's objects (`shapes`: name -> shape). Found by the world's
        objects, not by joint names: a robot built from parts with a slide called "belt" would have been snapped
        back every 1.5 cm."""
        self.rate_hz = 1.0 / model.opt.timestep
        self.belts: list[int] = []  # qpos addresses of belt slides
        self.buttons: dict[str, int] = {}  # button name -> qpos address of its cap
        for name, shape in shapes.items():
            part = {"conveyor": "belt", "button": "cap"}.get(shape)
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/{part}") if part else -1
            if j < 0 or int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_SLIDE:
                continue
            if shape == "conveyor":
                self.belts.append(int(model.jnt_qposadr[j]))
            else:
                self.buttons[name] = int(model.jnt_qposadr[j])
        self.presses = dict.fromkeys(self.buttons, 0)
        self._down = dict.fromkeys(self.buttons, False)

    @property
    def needed(self) -> bool:
        return bool(self.belts or self.buttons)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for adr in self.belts:
            q = data.qpos[adr]
            if q >= design.BELT_WRAP or q < 0.0:
                data.qpos[adr] = q % design.BELT_WRAP
        for name, adr in self.buttons.items():
            down = data.qpos[adr] < -design.PRESSED_AT * design.BUTTON_TRAVEL
            if down and not self._down[name]:
                self.presses[name] += 1
            self._down[name] = bool(down)

    def status(self) -> dict[str, Any]:
        return {"behavior": "mechanisms", "presses": dict(self.presses)}


def install(session: Any, shapes: dict[str, str]) -> None:
    """Give the session's current model a watcher for the world objects `shapes` (name -> shape), or take it away
    when there is nothing to watch."""
    watcher = Watcher(session.model, shapes)
    session.set_controller(KEY, watcher if watcher.needed else None)


def attach(session: Any, shapes: Callable[[], dict[str, str]]) -> None:
    """Keep a watcher on the session from now on: after every reset and model swap, which clear controllers.
    `shapes` gives the world's objects as they are then (name -> shape)."""
    session.reset_hooks.append(lambda: install(session, shapes()))
    install(session, shapes())


def read(model: mujoco.MjModel, data: mujoco.MjData, shapes: dict[str, str], watcher: Any = None) -> dict:
    """What each mechanism and soft thing is doing, by name, in a few numbers each (for observe)."""
    out: dict[str, dict[str, Any]] = {}
    presses = getattr(watcher, "presses", {})
    for name, shape in shapes.items():
        if shape in design.MECHANISMS:
            part = design.MECHANISMS[shape].part
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/{part}")
            if j < 0:
                continue
            q = float(data.qpos[model.jnt_qposadr[j]])
            out[name] = {"type": shape, **_reading(model, data, shape, name, j, q, presses.get(name))}
        elif shape == "rope":
            ends = _bodies(model, f"{name}/")
            if ends:
                out[name] = {"type": "rope", "end": _rounded(data.xpos[ends[-1]])}
        elif shape == "cloth":
            flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, f"{name}/{name}")
            if flex >= 0:
                start, count = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
                points = data.flexvert_xpos[start : start + count]
                out[name] = {
                    "type": "cloth",
                    "center": _rounded(points.mean(axis=0)),
                    "height_m": [round(float(points[:, 2].min()), 3), round(float(points[:, 2].max()), 3)],
                }
    return out


def _reading(model, data, shape: str, name: str, joint: int, q: float, presses: int | None) -> dict[str, Any]:
    deg = round(math.degrees(q), 1)
    if shape == "drawer":
        travel = float(model.jnt_range[joint][1])
        return {"open_m": round(q, 3), "of_m": round(travel, 3), "open": q > 0.01}
    if shape == "cabinet":
        return {"angle_deg": deg, "open": deg > 5.0}
    if shape == "button":
        state: dict[str, Any] = {"pressed": q < -design.PRESSED_AT * design.BUTTON_TRAVEL}
        if presses is not None:
            state["presses"] = presses
        return state
    if shape == "switch":
        return {"on": q > 0.0, "angle_deg": deg}
    if shape == "lever":
        return {"angle_deg": deg, "toward": "-y" if q > 0 else "+y"}
    if shape == "valve":
        turns = q / (2 * math.pi)
        return {"turns": round(turns, 2), "opening": round(min(max(turns / design.VALVE_TURNS, 0.0), 1.0), 2)}
    if shape == "conveyor":
        act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}/speed")
        belt = float(data.qvel[model.jnt_dofadr[joint]])
        speed = float(data.ctrl[act]) if act >= 0 else belt
        return {"speed_m_s": round(speed, 3), "belt_m_s": round(belt, 3), "running": abs(speed) > 1e-4}
    return {"q": round(q, 4)}


def _bodies(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [
        b
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
    ]


def _rounded(v: np.ndarray) -> list[float]:
    return [round(float(x), 3) for x in v]
