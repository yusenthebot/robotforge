"""The sensor types other than lidars and cameras: what each measures, how it errs and how it reads.

Every reading names its unit in its keys (accel_m_s2, force_n, heading_deg) and is rounded to what the
sensor could tell apart, so an agent reads the numbers without a manual and pays for no digits of noise.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.sensors import declare
from robotforge.sensors.runtime import Sensor, native_slice

LIST_LIMIT = 40  # joints an encoder reading lists; a humanoid has 29, a pair of arms a few more


def _r(values: Any, digits: int) -> Any:
    if isinstance(values, np.ndarray | list | tuple):
        return [round(float(v), digits) for v in values]
    return round(float(values), digits)


def rpy_deg(quat: np.ndarray) -> np.ndarray:
    """Roll, pitch and yaw in degrees (yaw about world z, then pitch, then roll) of a w-x-y-z quaternion."""
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def site_pose(data: mujoco.MjData, site: int) -> dict[str, list[float]]:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return {"pos": _r(data.site_xpos[site], 4), "quat": _r(quat, 4)}


class Imu(Sensor):
    """Accelerometer (proper acceleration: 9.81 up at rest, zero in free fall), gyro and orientation."""

    kind = "imu"

    def __init__(self, model: mujoco.MjModel, decl: declare.Declared) -> None:
        super().__init__(model, decl)
        native = decl.data["native"]
        self.slices = {role: native_slice(model, decl, native[role]) for role in ("accel", "gyro", "quat")}
        self.walk = {c: np.zeros(3) for c in ("accel", "gyro")}  # bias random walks
        self.yaw_error = 0.0  # deg: heading drifts with the gyro's bias, as an IMU's own estimate does
        # MuJoCo 3.15 leaves the accelerometer of a body welded to the world at zero; a real one bolted down feels
        # the floor holding it up against gravity, so it reads 9.81 up like any IMU at rest.
        self.static = int(model.body_weldid[self.body]) == 0 and self.site >= 0
        self.up_g = -np.array(model.opt.gravity, float)

    def measure(self, model, data):
        value = {role: np.array(data.sensordata[s]) for role, s in self.slices.items()}
        if self.static:
            value["accel"] = data.site_xmat[self.site].reshape(3, 3).T @ self.up_g
        return value

    def corrupt(self, value, dt):
        k = self.knobs
        out = {}
        for c in ("accel", "gyro"):
            if c in k.drift and dt > 0:
                self.walk[c] += self.rng.normal(0.0, k.drift[c] * math.sqrt(dt), 3)
            offset = k.bias.get(c, np.zeros(3)) + self.walk[c]
            out[c] = value[c] + offset + self.noise(c, 3)
        self.yaw_error += math.degrees(float(k.bias.get("gyro", np.zeros(3))[2] + self.walk["gyro"][2]) * dt)
        rpy = rpy_deg(value["quat"])
        rpy[2] = (rpy[2] + self.yaw_error + 180.0) % 360.0 - 180.0
        out["rpy"] = rpy + self.noise("orientation", 3)
        return out

    def reading(self, value, full=False):
        return {
            "accel_m_s2": _r(value["accel"], 3),
            "gyro_rad_s": _r(value["gyro"], 4),
            "orientation_deg": {
                "roll": round(float(value["rpy"][0]), 2),
                "pitch": round(float(value["rpy"][1]), 2),
                "yaw": round(float(value["rpy"][2]), 2),
            },
        }

    def line(self, value):
        roll, pitch, yaw = value["rpy"]
        g = float(np.linalg.norm(value["accel"]))
        return f"imu: accel {g:.2f} m/s2, tilt {math.hypot(roll, pitch):.1f} deg, yaw {yaw:.1f} deg"

    def viewer(self, value, model, data):
        return {
            "accel_m_s2": _r(value["accel"], 3),
            "gyro_rad_s": _r(value["gyro"], 4),
            "orientation_deg": _r(value["rpy"], 2),
        }


class Encoders(Sensor):
    """Joint angles (radians; meters for a sliding joint), stepped to the encoder's resolution."""

    kind = "encoders"
    from_forward = False

    def __init__(self, model, decl):
        super().__init__(model, decl)
        self.joints = decl.data["joints"]
        self.adr, self.slide = [], []
        for joint in self.joints:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, decl.full(joint))
            if j < 0:
                raise ValueError(f"no joint {decl.full(joint)!r}")
            self.adr.append(int(model.jnt_qposadr[j]))
            self.slide.append(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE))
        self.resolution = float(decl.data.get("resolution") or 0.0)

    def measure(self, model, data):
        return np.array(data.qpos[self.adr])

    def corrupt(self, value, dt):
        out = value + self.noise("position", len(value))
        if self.resolution:
            out = np.round(out / self.resolution) * self.resolution
        return out

    def _split(self, value) -> dict[str, dict[str, float]]:
        digits = 4 if not self.resolution else max(0, min(6, -int(math.floor(math.log10(self.resolution)))))
        out: dict[str, dict[str, float]] = {}
        for joint, slide, v in list(zip(self.joints, self.slide, value, strict=True))[:LIST_LIMIT]:
            out.setdefault("m" if slide else "rad", {})[joint] = round(float(v), digits)
        return out

    def reading(self, value, full=False):
        out = self._split(value)
        if len(self.joints) > LIST_LIMIT:
            out["more"] = len(self.joints) - LIST_LIMIT
        return out

    def line(self, value):
        return f"encoders: {len(self.joints)} joints" + (f", {self.resolution:g} rad steps" if self.resolution else "")

    def viewer(self, value, model, data):
        return self._split(value)


class ForceTorque(Sensor):
    """The force and torque between a body and the one it hangs from (a wrist, an ankle), as the parent pushes
    on the child, in the sensor's frame. payload_kg is what it carries beyond the weight hanging below it."""

    kind = "force_torque"

    def __init__(self, model, decl):
        super().__init__(model, decl)
        native = decl.data["native"]
        self.force = native_slice(model, decl, native["force"])
        self.torque = native_slice(model, decl, native["torque"])
        self.below = float(model.body_subtreemass[self.body])
        self.gravity = np.array(model.opt.gravity, float)

    def measure(self, model, data):
        rot = data.site_xmat[self.site].reshape(3, 3)
        return {
            "force": np.array(data.sensordata[self.force]),
            "torque": np.array(data.sensordata[self.torque]),
            "rot": rot.copy(),
        }

    def corrupt(self, value, dt):
        return {
            **value,
            "force": value["force"] + self.noise("force", 3),
            "torque": value["torque"] + self.noise("torque", 3),
        }

    def _payload(self, value) -> float | None:
        g = float(np.linalg.norm(self.gravity))
        if g < 1e-9:
            return None
        up = -self.gravity / g
        return float((value["rot"] @ value["force"]) @ up / g - self.below)

    def reading(self, value, full=False):
        world = value["rot"] @ value["force"]
        out = {
            "force_n": _r(value["force"], 3),
            "torque_nm": _r(value["torque"], 4),
            "force_world_n": _r(world, 3),
        }
        payload = self._payload(value)
        if payload is not None:
            out["payload_kg"] = round(payload, 3)
        return out

    def line(self, value):
        payload = self._payload(value)
        weight = f", carries {payload:.3f} kg beyond its own" if payload is not None else ""
        return (
            f"force_torque: {np.linalg.norm(value['force']):.2f} N, {np.linalg.norm(value['torque']):.3f} N m{weight}"
        )

    def viewer(self, value, model, data):
        rot = value["rot"]
        return {"force_n": _r(rot @ value["force"], 3), "torque_nm": _r(rot @ value["torque"], 4)}


class Touch(Sensor):
    """Normal contact force on a pad, one zone or a small array of them (rows run front to back)."""

    kind = "touch"

    def __init__(self, model, decl):
        super().__init__(model, decl)
        self.cells = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, decl.full(c)) for c in decl.data["cells"]]
        # The MuJoCo touch sensor of each zone: named after its site, unless the definition names them ("touch").
        self.slices = [native_slice(model, decl, c) for c in decl.data.get("touch", decl.data["cells"])]
        self.rows, self.cols = int(decl.data["rows"]), int(decl.data["cols"])

    def measure(self, model, data):
        return np.array([float(data.sensordata[s][0]) for s in self.slices])

    def corrupt(self, value, dt):
        return np.maximum(value + self.noise("force", len(value)), 0.0)

    def _threshold(self) -> float:
        return max(0.05, 3 * self.knobs.noise.get("force", 0.0))

    def reading(self, value, full=False):
        total = float(value.sum())
        out: dict[str, Any] = {"force_n": round(total, 2), "contact": bool(total > self._threshold())}
        if len(value) > 1:
            grid = value.reshape(self.rows, self.cols)
            out["cells_n"] = [[round(float(v), 2) for v in row] for row in grid]
            if out["contact"]:
                # Where the pressure is, from -1 to 1 across the pad: + is the first row (front) and first column.
                u = np.linspace(1, -1, self.rows) if self.rows > 1 else np.zeros(1)
                v = np.linspace(1, -1, self.cols) if self.cols > 1 else np.zeros(1)
                out["center"] = [
                    round(float(grid.sum(axis=1) @ u / total), 2),
                    round(float(grid.sum(axis=0) @ v / total), 2),
                ]
        return out

    def line(self, value):
        total = float(value.sum())
        state = "in contact" if total > self._threshold() else "no contact"
        cells = f", {len(value)} zones" if len(value) > 1 else ""
        return f"touch: {state}, {total:.1f} N{cells}"

    def viewer(self, value, model, data):
        return {
            "cells_m": [_r(data.site_xpos[c], 4) for c in self.cells],
            "force_n": _r(value, 2),
            "total_n": round(float(value.sum()), 2),
        }


class Rangefinder(Sensor):
    """Distance along one beam to the first thing it meets, up to its range."""

    kind = "rangefinder"

    def __init__(self, model, decl):
        super().__init__(model, decl)
        self.slice = native_slice(model, decl, decl.data["native"]["range"])
        self.range = float(decl.data.get("range") or 4.0)

    def measure(self, model, data):
        return float(data.sensordata[self.slice][0])

    def corrupt(self, value, dt):
        if value < 0:
            return value
        if self.knobs.dropout and self.rng.random() < self.knobs.dropout:
            return -1.0
        return float(np.clip(value + self.noise("range"), 0.0, self.range))

    def reading(self, value, full=False):
        if value < 0:
            return {"distance_m": None, "note": f"nothing within {self.range:g} m"}
        return {"distance_m": round(value, 3)}

    def line(self, value):
        return f"rangefinder: {value:.2f} m" if value >= 0 else f"rangefinder: nothing within {self.range:g} m"

    def viewer(self, value, model, data):
        rot = data.site_xmat[self.site].reshape(3, 3)
        return {
            "direction": _r(rot[:, 2], 4),
            "distance_m": round(value, 3) if value >= 0 else None,
            "range_m": self.range,
        }


class Gps(Sensor):
    """Where it is in the world, in meters (a GNSS fix in the world's frame)."""

    kind = "gps"

    def measure(self, model, data):
        return np.array(data.site_xpos[self.site])

    def corrupt(self, value, dt):
        return value + self.noise("position", 3)

    def reading(self, value, full=False):
        return {"position_m": _r(value, 3)}

    def line(self, value):
        x, y, z = value
        return f"gps: x {x:.2f}, y {y:.2f}, z {z:.2f} m"

    def viewer(self, value, model, data):
        return {"position_m": _r(value, 3)}


class Odometry(Sensor):
    """Dead reckoning: the body's own speed and turn rate added up at its rate, starting from where it was at
    the first sample. With noise, bias or drift the estimate wanders away from the truth, as a real one does."""

    kind = "odometry"
    integrates = True
    default_rate = 50.0

    def __init__(self, model, decl):
        super().__init__(model, decl)
        self.estimate: np.ndarray | None = None  # x, y, heading (rad)
        self.traveled = 0.0
        self.turn_walk = 0.0

    def measure(self, model, data):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.site, vel, 1)  # angular, linear
        rot = data.site_xmat[self.site].reshape(3, 3)
        return {
            "pos": np.array(data.site_xpos[self.site][:2]),
            "heading": math.atan2(rot[1, 0], rot[0, 0]),
            "speed": vel[3:5].copy(),  # forward and sideways, in its own frame
            "turn": float(vel[2]),
        }

    def corrupt(self, value, dt):
        k = self.knobs
        if self.estimate is None:
            self.estimate = np.array([*value["pos"], value["heading"]])
            self.last = (value["speed"], value["turn"])
            return self.estimate.copy()
        if "turn" in k.drift and dt > 0:
            self.turn_walk += float(self.rng.normal(0.0, k.drift["turn"] * math.sqrt(dt)))
        speed = value["speed"] + self.noise("speed", 2)
        turn = value["turn"] + float(k.bias.get("turn", np.zeros(3))[0]) + self.turn_walk + float(self.noise("turn"))
        last_speed, last_turn = self.last
        self.last = (speed, turn)
        heading = self.estimate[2] + (last_turn + turn) / 2 * dt  # trapezoid: turns integrate without lag
        mid = (self.estimate[2] + heading) / 2
        step = (last_speed + speed) / 2 * dt
        c, s = math.cos(mid), math.sin(mid)
        self.estimate = np.array(
            [self.estimate[0] + c * step[0] - s * step[1], self.estimate[1] + s * step[0] + c * step[1], heading]
        )
        self.traveled += float(np.linalg.norm(step))
        return self.estimate.copy()

    def reading(self, value, full=False):
        heading = (math.degrees(value[2]) + 180.0) % 360.0 - 180.0
        return {"position_m": _r(value[:2], 3), "heading_deg": round(heading, 2), "traveled_m": round(self.traveled, 3)}

    def line(self, value):
        heading = (math.degrees(value[2]) + 180.0) % 360.0 - 180.0
        return f"odometry: x {value[0]:.2f}, y {value[1]:.2f} m, heading {heading:.1f} deg"

    def viewer(self, value, model, data):
        return {"position_m": _r(value[:2], 3), "heading_deg": round(math.degrees(value[2]), 2)}


def _kinds() -> dict[str, type[Sensor]]:
    from robotforge.sensors.camera import Camera
    from robotforge.sensors.lidar import Lidar

    return {
        "imu": Imu,
        "encoders": Encoders,
        "force_torque": ForceTorque,
        "touch": Touch,
        "rangefinder": Rangefinder,
        "gps": Gps,
        "odometry": Odometry,
        "lidar": Lidar,
        "lidar3d": Lidar,
        "camera": Camera,
    }


KINDS = _kinds()
