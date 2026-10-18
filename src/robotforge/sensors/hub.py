"""The live world's sensors: rebuilt with every model, sampled while it steps, read by tools, scripts and the viewer.

The hub hangs on the SimSession: a reset hook builds a fresh suite whenever the model changes or the world is
reset (so noise restarts from its seed), and the session's step hook samples the sensors that need it. A world
whose sensors are all ideal has no step hook at all.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.sensors import kinds
from robotforge.sensors.runtime import Sensor, Suite

log = logging.getLogger(__name__)

OTHER_LIMIT = 20  # a robot's own MuJoCo sensors listed beside its declared ones


class SensorError(ValueError):
    """A user-facing error: the message says what to do instead."""


class SensorHub:
    def __init__(self, session: Any) -> None:
        self.session = session
        self._renderer = None
        self._render_lock = threading.Lock()
        # Sensors an agent asked the viewer to draw (open_viewer(show=...)): the page applies each request once.
        self.viewer_show: dict[str, Any] | None = None
        session.reset_hooks.append(self.rebuild)
        self.rebuild()

    def rebuild(self) -> None:
        """A new model or a reset: sensors bound to the model as it is now, their noise from its seed again, and
        their first samples taken from the state as it is now."""
        self.suite = Suite(self.session.model)
        self.suite.prime(self.session.model, self.session.data)
        self._install()

    def _install(self) -> None:
        self.session.step_hook = self.suite.after_step if self.suite.stepped else None

    @property
    def renderer(self) -> Any:
        """Its own renderer: sharing observe's would rebuild its GL buffers at every change of picture size."""
        if self._renderer is None:
            from robotforge.sim.render import SnapshotRenderer

            self._renderer = SnapshotRenderer()
        return self._renderer

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()

    # ---- choosing sensors ---------------------------------------------------------------------------------

    def names(self, robot: str) -> list[str]:
        return [s.local for s in self.suite.of(robot)]

    def resolve_robot(self, robot: str, robots: list[str]) -> str:
        """The robot `robot` means, among the world's robots (and those with sensors on a loaded file)."""
        known = sorted({*robots, *self.suite.robots()} - {""})
        if robot in known:
            return robot
        found = wording.match(robot, known)
        if found is None:
            raise SensorError(wording.unknown("robot", robot, known, make="add_robot"))
        return found

    def chosen(self, robot: str, names: list[str] | None) -> list[Sensor]:
        own = self.suite.of(robot)
        if not own:
            raise SensorError(
                f"{robot} has no sensors yet; add one with add_sensor, e.g. add_sensor(robot={robot!r}, type='lidar3d')"
            )
        if not names:
            return own
        by_name = {s.local: s for s in own}
        out = []
        for given in names:
            local = given.split("/", 1)[1] if given.startswith(f"{robot}/") else given
            reader = self.suite.owner.get(f"{robot}/{local}")
            if local not in by_name and reader is not None and reader.robot == robot:
                out.append(reader)  # one of MuJoCo's own sensors behind it: "accel" is read by the imu
                continue
            found = local if local in by_name else wording.match(local, list(by_name))
            if found is None:
                raise SensorError(
                    wording.unknown("sensor", given, list(by_name), where=f"on {robot}", make="add_sensor")
                )
            out.append(by_name[found])
        return list(dict.fromkeys(out))

    def show(self, names: list[str], robots: list[str]) -> list[str]:
        """Ask the viewer to draw these sensors, "robot/sensor" or a robot's name for all of its sensors: a robot's
        own (a template's camera) are hidden until shown, and a camera's picture comes up beside the view."""
        out: list[str] = []
        for given in names:
            robot, _, local = given.partition("/")
            robot = self.resolve_robot(robot, robots)
            out += [s.name for s in self.chosen(robot, [local] if local else None)]
        shown = list(dict.fromkeys(out))
        seq = (self.viewer_show or {}).get("seq", 0) + 1
        self.viewer_show = {"seq": seq, "names": shown}
        return shown

    # ---- reading ---------------------------------------------------------------------------------------------

    def read(
        self, robot: str, names: list[str] | None = None, image: bool = False, full: bool = False
    ) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
        """Readings of a robot's sensors (all, or those named), and the cameras' pictures when `image`."""
        session = self.session
        out: dict[str, Any] = {}
        later: list[tuple[Sensor, float, Any]] = []
        with session.lock:
            model, data = session.model, session.data
            now = float(data.time)
            for sensor in self.chosen(robot, names):
                got = sensor.current(model, data)
                if got is None:
                    out[sensor.local] = {"type": sensor.type, "status": _waiting(sensor)}
                    continue
                stamp, value = got
                if sensor.renders:
                    later.append((sensor, stamp, value))
                    out[sensor.local] = None  # keeps the order
                    continue
                out[sensor.local] = {"type": sensor.type, **sensor.reading(value, full), **_age(sensor, stamp, now)}
            other = self._other(model, data, robot) if not names else {}
        pictures: list[tuple[str, bytes]] = []
        for sensor, stamp, moment in later:  # drawn with the session unlocked: the simulation keeps running
            value = self._develop(sensor, model, moment)
            out[sensor.local] = {"type": sensor.type, **sensor.reading(value, full), **_age(sensor, stamp, now)}
            if image and "layers" in value:
                pictures.append((sensor.name, sensor.png(value)))
        result: dict[str, Any] = {"robot": robot, "time": round(now, 4), "sensors": out}
        if other:
            result["other_sensors"] = other
        return result, pictures

    def _develop(self, sensor: Any, model: mujoco.MjModel, moment: dict[str, Any]) -> dict[str, Any]:
        from robotforge.sim.render import RenderUnavailable

        scratch = sensor.scene_data(model, moment)
        try:
            with self._render_lock:
                layers = self.renderer.layers_of(
                    model, scratch, sensor.view, sensor.width, sensor.height, id(model), geoms=sensor.mode != "rgb"
                )
        except RenderUnavailable as exc:
            return {"note": f"no picture: {exc}"}
        return sensor.develop(layers, moment)

    def _other(self, model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> dict[str, Any]:
        """The robot's MuJoCo sensors no definition reads (a library robot's own IMU), raw."""
        prefix = f"{robot}/"
        out: dict[str, Any] = {}
        for s in range(model.nsensor):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or ""
            if not name.startswith(prefix) or name in self.suite.covered or "/rf_" in name:
                continue
            if len(out) >= OTHER_LIMIT:
                out["..."] = "more"
                break
            adr, dim = int(model.sensor_adr[s]), int(model.sensor_dim[s])
            values = [round(float(v), 4) for v in data.sensordata[adr : adr + dim]]
            out[name[len(prefix) :]] = values[0] if dim == 1 else values
        return out

    def lines(self) -> dict[str, str]:
        """One line per sensor in the world, for observe."""
        out: dict[str, str] = {}
        session = self.session
        with session.lock:
            model, data = session.model, session.data
            for sensor in self.suite.sensors.values():
                if sensor.renders:
                    out[sensor.name] = sensor.line({})
                    continue
                got = sensor.current(model, data)
                out[sensor.name] = f"{sensor.type}: {_waiting(sensor)}" if got is None else sensor.line(got[1])
        return out

    def viewer_message(self) -> dict[str, Any] | None:
        """The "sensors" message for the viewer (see viewer/server.py): each sensor's pose in the world and what
        drawing it needs, None when the world has no sensors."""
        if not self.suite.sensors:
            return None
        session = self.session
        entries = []
        with session.lock:
            model, data = session.model, session.data
            for sensor in self.suite.sensors.values():
                entry: dict[str, Any] = {
                    "name": sensor.name,
                    "robot": sensor.robot,
                    "type": sensor.type,
                    "added": bool(sensor.decl.data.get("added")),
                }
                try:
                    entry.update(_mount(sensor, model))
                    if sensor.renders:
                        entries.append({**entry, **sensor.pose(data), **sensor.viewer({}, model, data)})
                        continue
                    entry.update(_pose(sensor, data))
                    got = sensor.current(model, data)
                    if got is None:
                        entries.append({**entry, "status": "waiting"})
                        continue
                    entries.append({**entry, "t": round(got[0], 4), **sensor.viewer(got[1], model, data)})
                except Exception:  # one sensor that cannot be drawn must not stop the others'
                    log.exception("sensor %s could not be described for the viewer", sensor.name)
            # The model's version: body ids mean that model's bodies, and the page draws on the one it has loaded.
            version = session.version
            msg = {"type": "sensors", "version": version, "time": round(float(data.time), 4), "sensors": entries}
            if self.viewer_show is not None:
                msg["show"] = self.viewer_show
            return msg


def _mount(sensor: Sensor, model: mujoco.MjModel) -> dict[str, Any]:
    """The body a sensor rides on and where it sits there (local_pos, local_quat in the body's frame): the viewer
    draws a camera's frustum, its picture and a force arrow from the body's pose in every frame it draws, so they
    stay on a walking robot instead of jumping to it ten times a second."""
    camera = getattr(sensor, "camera", -1) if sensor.renders else -1
    if camera >= 0:
        body, pos, quat = model.cam_bodyid[camera], model.cam_pos[camera], model.cam_quat[camera]
    elif sensor.site >= 0:
        body, pos, quat = model.site_bodyid[sensor.site], model.site_pos[sensor.site], model.site_quat[sensor.site]
    else:
        body, pos, quat = sensor.body, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "body": int(body),
        "local_pos": [round(float(v), 4) for v in pos],
        "local_quat": [round(float(v), 4) for v in quat],
    }


def _pose(sensor: Sensor, data: mujoco.MjData) -> dict[str, list[float]]:
    if sensor.site >= 0:
        return kinds.site_pose(data, sensor.site)
    quat = np.array(data.xquat[sensor.body])
    return {"pos": [round(float(v), 4) for v in data.xpos[sensor.body]], "quat": [round(float(v), 4) for v in quat]}


def _age(sensor: Sensor, stamp: float, now: float) -> dict[str, float]:
    """How old a sample is, for a sensor that is not read fresh (its rate or latency)."""
    if sensor.lazy:
        return {}
    return {"age_s": round(max(now - stamp, 0.0), 4)}


def _waiting(sensor: Sensor) -> str:
    return f"no reading yet: with a latency of {sensor.knobs.latency_s:g} s its first sample is still on its way"
