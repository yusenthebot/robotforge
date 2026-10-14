"""The playground: one simulation, one viewer, and the operations agents call.

Every MCP tool is a thin wrapper over a method here, so the logic is testable
without an MCP client. Methods return plain dicts that serialize to JSON.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from pydantic import ValidationError

from robotforge.behaviors.manager import BehaviorManager
from robotforge.design.store import WorldStore
from robotforge.design.world import CompiledWorld, ObjectEntry, RobotEntry, World
from robotforge.sim.render import VIEWS, SnapshotRenderer
from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub

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
    store: WorldStore | None = None
    behaviors: BehaviorManager = field(init=False)

    def __post_init__(self) -> None:
        self.behaviors = BehaviorManager(self.session, self.current_world)

    def attach_viewer(self, hub: ViewerHub) -> None:
        """Let the viewer drive robots from the keyboard and list them in its UI."""

        def teleop(msg: dict[str, Any]) -> None:
            walker = msg.get("walker")  # set only when the user switches between the gait and a learned policy
            speed, turn = float(msg.get("speed", 0.0)), float(msg.get("turn", 0.0))
            self.behaviors.move(str(msg["robot"]), speed, turn, str(walker) if walker else None)

        hub.handlers["teleop"] = teleop
        hub.scene_extras = lambda: {"robots": self.behaviors.robots(), "learned": self.behaviors.learned_walkers()}

    def current_world(self) -> World | None:
        return self.store.load() if self.store is not None else None

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

    # ---- worlds (design) ------------------------------------------------------------

    def new_world(self, name: str, floor: str = "checker") -> dict[str, Any]:
        self.store = WorldStore.create(name, floor)
        return self._load_world(self.store.compile(), created=True)

    def open_world(self, name_or_path: str) -> dict[str, Any]:
        self.store = WorldStore.open(name_or_path)
        return self._load_world(self.store.compile())

    def add_robot(
        self,
        name: str,
        template: str | None = None,
        file: str | None = None,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float = 0.0,
        library: str | None = None,
    ) -> dict[str, Any]:
        try:
            entry = RobotEntry(
                name=name,
                template=template,
                file=file,
                library=library,
                params=params or {},
                pos=pos or [0.0, 0.0],
                yaw=yaw,
            )
        except ValidationError as exc:
            raise PlaygroundError(_validation_message(exc)) from exc
        world = self.current_world()
        existing = next((r for r in world.robots if r.name == name), None) if world else None
        if existing is not None and existing == entry:
            # Typically a retry after the client timed out while the first call was still downloading.
            return self._unchanged(f"{name} is already in the world exactly like this; nothing to do")

        def change(w: World) -> None:
            w.robots.append(entry)

        return self._edit(change)

    def update_robot(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        pos: list[float] | None = None,
        yaw: float | None = None,
    ) -> dict[str, Any]:
        def change(w: World) -> None:
            robot = _find(w.robots, name, "robot")
            if params:
                robot.params = {**robot.params, **params}
            if pos is not None:
                robot.pos = pos
            if yaw is not None:
                robot.yaw = yaw

        return self._edit(change)

    def add_object(self, name: str, **fields: Any) -> dict[str, Any]:
        entry = ObjectEntry(name=name, **{k: v for k, v in fields.items() if v is not None})

        def change(w: World) -> None:
            w.objects.append(entry)

        return self._edit(change)

    def update_object(self, name: str, **fields: Any) -> dict[str, Any]:
        def change(w: World) -> None:
            obj = _find(w.objects, name, "object")
            for key, value in fields.items():
                if value is not None:
                    setattr(obj, key, value)

        return self._edit(change)

    def remove(self, name: str) -> dict[str, Any]:
        def change(w: World) -> None:
            before = len(w.robots) + len(w.objects)
            w.robots = [r for r in w.robots if r.name != name]
            w.objects = [o for o in w.objects if o.name != name]
            if len(w.robots) + len(w.objects) == before:
                raise PlaygroundError(f"nothing named {name!r} in this world")

        return self._edit(change)

    def set_floor(self, style: str) -> dict[str, Any]:
        def change(w: World) -> None:
            w.floor.style = style

        return self._edit(change)

    def undo(self) -> dict[str, Any]:
        store = self._require_store()
        store.undo()
        return self._load_world(store.compile())

    def show_world(self) -> dict[str, Any]:
        store = self._require_store()
        return {"world_dir": str(store.dir), "world_yaml": store.file.read_text(encoding="utf-8")}

    def _edit(self, change) -> dict[str, Any]:
        store = self._require_store()
        try:
            _, compiled = store.apply(change)
        except ValidationError as exc:
            raise PlaygroundError(_validation_message(exc)) from exc
        return self._load_world(compiled)

    def _unchanged(self, note: str) -> dict[str, Any]:
        """The world summary without recompiling: the simulation keeps running as it is."""
        store = self._require_store()
        world = store.load()
        return {
            "world": world.name,
            "world_dir": str(store.dir),
            "created": False,
            "unchanged": True,
            "robots": [r.name for r in world.robots],
            "objects": [o.name for o in world.objects],
            "notes": [note],
            "viewer_url": self.viewer_url,
        }

    def _require_store(self) -> WorldStore:
        if self.store is None:
            raise PlaygroundError("no world is open; call new_world first (or open_world for an existing one)")
        return self.store

    def _load_world(self, compiled: CompiledWorld, created: bool = False) -> dict[str, Any]:
        store = self._require_store()
        self.session.load_model(compiled.model, str(store.file))
        world = store.load()
        summary = self.describe()
        return {
            "world": world.name,
            "world_dir": str(store.dir),
            "created": created,
            "robots": [r.name for r in world.robots],
            "objects": [o.name for o in world.objects],
            "notes": compiled.notes,
            "actuators": summary["actuators"],
            "viewer_url": self.viewer_url,
        }

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

    def run_fast(self, until: Callable[[], bool], max_seconds: float) -> float:
        """Step as fast as possible until `until()` or `max_seconds` of sim time; returns sim seconds run."""
        with self.session.lock:
            start = self.session.data.time
            chunk = max(1, round(0.02 / self.session.model.opt.timestep))
            while not until() and self.session.data.time - start < max_seconds:
                self.session.step(chunk)
            return float(self.session.data.time - start)

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
                sensors[_name(m, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"sensor{s}"] = values[0] if dim == 1 else values
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


def _find(items: list, name: str, kind: str):
    for item in items:
        if item.name == name:
            return item
    names = ", ".join(i.name for i in items) or "none"
    raise PlaygroundError(f"no {kind} named {name!r}; {kind}s in this world: {names}")


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        msg = err["msg"].removeprefix("Value error, ")  # pydantic's prefix means nothing to a beginner
        where = ".".join(str(x) for x in err["loc"])
        parts.append(f"{where}: {msg}" if where else msg)
    return "; ".join(parts)


def _name(m: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(m, obj, idx) or ""


def _free_bodies(m: mujoco.MjModel) -> list[int]:
    """Bodies attached to the world by a free joint: robots' bases and loose objects."""
    return [int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]


def _limit(items: list) -> list:
    return items if len(items) <= _LIST_LIMIT else [*items[:_LIST_LIMIT], f"... {len(items) - _LIST_LIMIT} more"]


def _limit_dict(d: dict) -> dict:
    if len(d) <= _LIST_LIMIT:
        return d
    keys = list(d)[:_LIST_LIMIT]
    return {**{k: d[k] for k in keys}, "...": f"{len(d) - _LIST_LIMIT} more"}
