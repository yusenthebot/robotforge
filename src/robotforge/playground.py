"""The playground: one simulation, one viewer, and the operations agents call.

Every MCP tool is a thin wrapper over a method here, so the logic is testable
without an MCP client. Methods return plain dicts that serialize to JSON.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.sim.render import VIEWS, SnapshotRenderer
from robotforge.sim.session import SimSession

DEMO_MODEL = Path(__file__).parent / "assets" / "demo.xml"
MAX_SIM_SECONDS = 60.0
_LIST_LIMIT = 40

_JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


class PlaygroundError(ValueError):
    """A user-facing error: the message tells the agent what to do instead."""


@dataclass
class Playground:
    session: SimSession
    viewer_url: str | None = None
    renderer: SnapshotRenderer = field(default_factory=SnapshotRenderer)

    @classmethod
    def with_demo(cls) -> Playground:
        return cls(session=SimSession.from_path(DEMO_MODEL))

    # ---- loading and describing ----------------------------------------------------

    def open_model(self, path: str) -> dict[str, Any]:
        p = Path(path).expanduser()
        if p.suffix.lower() not in {".xml", ".urdf", ".mjcf"}:
            raise PlaygroundError(f"{p.name}: expected an MJCF (.xml) or URDF (.urdf) file")
        self.session.load_path(p.resolve())
        return self.describe()

    def describe(self) -> dict[str, Any]:
        m = self.session.model
        with self.session.lock:
            joints = [
                {
                    "name": _name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                    "type": _JOINT_TYPES.get(int(m.jnt_type[j]), "unknown"),
                    "range": [round(float(x), 4) for x in m.jnt_range[j]] if m.jnt_limited[j] else None,
                }
                for j in range(m.njnt)
            ]
            actuators = [self._actuator_info(a) for a in range(m.nu)]
            sensors = [_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in range(m.nsensor)]
            cameras = [_name(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in range(m.ncam)]
            free_bodies = [_name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in _free_bodies(m)]
        return {
            "model": self.session.source,
            "counts": {"bodies": m.nbody - 1, "joints": m.njnt, "actuators": m.nu, "sensors": m.nsensor},
            "actuators": actuators,
            "joints": _limit(joints),
            "free_bodies": _limit(free_bodies),
            "sensors": _limit(sensors),
            "views": sorted(VIEWS) + [c for c in cameras if c],
            "timestep": m.opt.timestep,
            "viewer_url": self.viewer_url,
        }

    def _actuator_info(self, a: int) -> dict[str, Any]:
        m = self.session.model
        info: dict[str, Any] = {"name": _name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"}
        if m.actuator_ctrllimited[a]:
            info["range"] = [round(float(x), 4) for x in m.actuator_ctrlrange[a]]
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            info["joint"] = _name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0]))
        return info

    # ---- control and simulation ----------------------------------------------------

    def set_controls(self, controls: dict[str, float]) -> dict[str, Any]:
        applied = {}
        with self.session.lock:
            for name, value in controls.items():
                a = self._actuator_id(name)
                self.session.set_ctrl(a, float(value))
                applied[name] = round(float(self.session.data.ctrl[a]), 4)
        return {"applied": applied}

    def simulate(self, seconds: float, controls: dict[str, float] | None = None) -> dict[str, Any]:
        if not 0 < seconds <= MAX_SIM_SECONDS:
            raise PlaygroundError(f"seconds must be in (0, {MAX_SIM_SECONDS}]")
        with self.session.lock:
            if controls:
                self.set_controls(controls)
            steps = max(1, round(seconds / self.session.model.opt.timestep))
            before = self._free_body_positions()
            self.session.step(steps)
            after = self._free_body_positions()
            unstable = not np.isfinite(self.session.data.qpos).all()
        moved = {
            name: {"from": before[name], "to": pos}
            for name, pos in after.items()
            if np.linalg.norm(np.subtract(pos, before[name])) > 0.005
        }
        result: dict[str, Any] = {
            "time": round(float(self.session.data.time), 4),
            "steps": steps,
            "moved": moved,
        }
        if unstable:
            result["warning"] = "simulation became unstable (NaN state); reset and check the model"
        return result

    def control(self, action: str) -> dict[str, Any]:
        s = self.session
        if action == "play":
            s.paused = False
        elif action == "pause":
            s.paused = True
        elif action == "reset":
            s.reset()
        else:
            raise PlaygroundError(f"unknown action {action!r}; use play, pause or reset")
        return {"action": action, "paused": s.paused, "time": round(float(s.data.time), 4)}

    # ---- observation ----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        m, d = self.session.model, self.session.data
        with self.session.lock:
            joints = {}
            for j in range(m.njnt):
                jtype = int(m.jnt_type[j])
                if jtype in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    joints[_name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"] = round(
                        float(d.qpos[m.jnt_qposadr[j]]), 4
                    )
            ctrl = {self._actuator_info(a)["name"]: round(float(d.ctrl[a]), 4) for a in range(m.nu)}
            sensors = {}
            for s in range(m.nsensor):
                adr, dim = m.sensor_adr[s], m.sensor_dim[s]
                values = [round(float(v), 4) for v in d.sensordata[adr : adr + dim]]
                sensors[_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"] = (
                    values[0] if dim == 1 else values
                )
            return {
                "time": round(float(d.time), 4),
                "paused": self.session.paused,
                "joints": _limit_dict(joints),
                "controls": _limit_dict(ctrl),
                "free_bodies": _limit_dict(self._free_body_positions()),
                "sensors": _limit_dict(sensors),
                "contacts": int(d.ncon),
            }

    def snapshot(self, view: str = "default", width: int = 640, height: int = 480) -> bytes:
        try:
            return self.renderer.png(self.session, view, width, height)
        except ValueError as exc:
            raise PlaygroundError(str(exc)) from exc

    # ---- helpers ----------------------------------------------------------------------

    def _actuator_id(self, name: str) -> int:
        m = self.session.model
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a >= 0:
            return a
        names = [self._actuator_info(i)["name"] for i in range(m.nu)]
        close = difflib.get_close_matches(name, names, n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise PlaygroundError(f"no actuator named {name!r}.{hint} Available: {', '.join(names) or 'none'}")

    def _free_body_positions(self) -> dict[str, list[float]]:
        m, d = self.session.model, self.session.data
        return {
            _name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}": [round(float(x), 4) for x in d.xpos[b]]
            for b in _free_bodies(m)
        }


def _name(m: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(m, obj, idx) or ""


def _free_bodies(m: mujoco.MjModel) -> list[int]:
    """Bodies attached to the world by a free joint: robots' bases and loose objects."""
    return [
        int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]


def _limit(items: list) -> list:
    return (
        items if len(items) <= _LIST_LIMIT else [*items[:_LIST_LIMIT], f"... {len(items) - _LIST_LIMIT} more"]
    )


def _limit_dict(d: dict) -> dict:
    if len(d) <= _LIST_LIMIT:
        return d
    keys = list(d)[:_LIST_LIMIT]
    return {**{k: d[k] for k in keys}, "...": f"{len(d) - _LIST_LIMIT} more"}
