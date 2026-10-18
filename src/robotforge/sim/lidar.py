"""What each lidar sees, as one reading: its nearest hit, which way that is from the robot, and every ray.

A lidar on a robot built from parts is a ring of rangefinders named <lidar>_0, <lidar>_1, ... (design/builder).
Read one ray at a time, a 64-ray lidar is 64 sensors, and observe lists at most 40: in the agent eval a lidar's
last 27 rays never reached the agent, which then wrote a script to read them and got the same 37 (the script's
sensors came from the same list). Here every ring is read whole, and each ray's bearing comes from where it really
points, measured from the robot's front (+ is to the left, as move's turn), so a lidar turned on its part or
mounted on a turret still says the right way.

A MuJoCo rangefinder leaves out only the body its site is on, so a lidar on a car's middle measured its own wheels
(0.15 m "ahead" in an empty world), and the warehouse picker's its own lift column: asked how far the nearest
obstacle was, an agent answered with the column. Every rangefinder on a robot is read past the robot's own parts
here (see `seen`), as robots filter themselves out of their scans.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

MIN_RAYS = 3  # a ring: fewer numbered rangefinders are separate sensors
RAY = re.compile(r"(.+)_(\d+)")
SIDES = (("front", 45.0), ("left", 135.0), ("back", 180.0))  # by |bearing|, with right for the negative half
PAST = 1e-4  # m a ray starts again beyond a shape of the robot's own


@dataclass(frozen=True)
class Ring:
    name: str  # "scout/base_lidar", as its rays are named without the number
    robot: str
    sensors: tuple[int, ...]  # sensor ids, ray 0 first
    reach: float  # m: the farthest a ray measures


def rings(model: mujoco.MjModel) -> list[Ring]:
    """The lidars in a model: rangefinders numbered 0, 1, 2, ... after a common name, at least MIN_RAYS of them."""
    found: dict[str, dict[int, int]] = {}
    for s in range(model.nsensor):
        if int(model.sensor_type[s]) != mujoco.mjtSensor.mjSENS_RANGEFINDER:
            continue
        ray = RAY.fullmatch(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or "")
        if ray:
            found.setdefault(ray[1], {})[int(ray[2])] = s
    out = []
    for name, rays in found.items():
        if len(rays) < MIN_RAYS or set(rays) != set(range(len(rays))):
            continue
        ids = tuple(rays[i] for i in range(len(rays)))
        reach = max(float(model.sensor_cutoff[s]) for s in ids)
        out.append(Ring(name, name.split("/", 1)[0], ids, reach))
    return out


def _front(model: mujoco.MjModel, data: mujoco.MjData, robot: str) -> np.ndarray | None:
    """The robot's front along the floor (its root body's +x), as a unit [x, y]; None when it points straight up."""
    for b in range(1, model.nbody):
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/"):
            x = data.xmat[int(model.body_rootid[b])].reshape(3, 3)[:2, 0]
            size = float(np.hypot(*x))
            return x / size if size > 1e-6 else None
    return None


def seen(model: mujoco.MjModel, data: mujoco.MjData, sensor: int) -> float:
    """What rangefinder `sensor` reads with its own robot seen through, in MuJoCo's terms: the distance (at most its
    cutoff), or -1 for nothing at all. The ray is cast again the way MuJoCo casts it (every geom group, static
    geoms too, leaving out the site's body); each time it meets a shape of the same robot (one tree of bodies under
    the world) it carries on from just beyond, until it meets something else or its cutoff. A rangefinder on the
    world itself reads as MuJoCo has it. It used to give up after 12 of the robot's own shapes: a lidar inside a
    Go2's trunk goes through its meshes and collision shapes, and 8 of 64 rays saw no wall 0.85 to 1.45 m away."""
    value = float(data.sensordata[model.sensor_adr[sensor]])
    site = int(model.sensor_objid[sensor])
    own = int(model.body_rootid[model.site_bodyid[site]])
    if value < 0 or own == 0:
        return value
    start = data.site_xpos[site].copy()
    look = data.site_xmat[site].reshape(3, 3)[:, 2].copy()  # a rangefinder looks along its +z
    cutoff = float(model.sensor_cutoff[sensor])
    gone, hit = 0.0, np.zeros(1, np.int32)
    # Each cast goes on past what it met, so the cutoff ends it; the count bounds one with none, or shapes that
    # touch: in and out of every shape twice over.
    for layer in range(4 * int(model.ngeom) + 8):
        far = mujoco.mj_ray(model, data, start, look, None, 1, int(model.site_bodyid[site]), hit)
        if far < 0:
            return -1.0
        if int(model.body_rootid[model.geom_bodyid[int(hit[0])]]) != own:
            if layer == 0:
                return value  # nothing of its own in the way: MuJoCo's own reading
            return min(gone + far, cutoff) if cutoff > 0 else gone + far
        start, gone = start + look * (far + PAST), gone + far + PAST
        if cutoff > 0 and gone >= cutoff:
            return cutoff  # MuJoCo's reading of a hit beyond the cutoff
    return -1.0


def side(bearing: float) -> str:
    """front, left, back or right of a bearing in degrees (+ left)."""
    turned = abs(bearing)
    name = next(n for n, limit in SIDES if turned <= limit)
    return "right" if bearing < 0 and name == "left" else name


def reading(model: mujoco.MjModel, data: mujoco.MjData, ring: Ring) -> dict[str, Any]:
    """One lidar's reading, from `data` as it is (sensors computed by the last step), past the robot's own parts."""
    front = _front(model, data, ring.robot)
    ranges: list[float | None] = []
    bearings: list[float | None] = []
    for s in ring.sensors:
        value = seen(model, data, s)
        hit = 0.0 <= value < ring.reach - 1e-6 if ring.reach > 0 else value >= 0.0
        ranges.append(round(value, 3) if hit else None)
        look = data.site_xmat[int(model.sensor_objid[s])].reshape(3, 3)[:2, 2]  # a rangefinder looks along its +z
        if front is None or float(np.hypot(*look)) < 1e-6:
            bearings.append(None)
            continue
        cross, dot = front[0] * look[1] - front[1] * look[0], float(front @ look)
        bearings.append(math.degrees(math.atan2(cross, dot)))
    hits = [(r, _deg(b)) for r, b in zip(ranges, bearings, strict=True) if r is not None]
    out: dict[str, Any] = {"rays": len(ranges), "range_m": round(ring.reach, 3), "hits": len(hits)}
    if hits:
        near, toward = min(hits, key=lambda h: h[0])
        out.update(nearest_m=near, nearest_bearing_deg=toward, nearest_side=None if toward is None else side(toward))
    by_side: dict[str, float | None] = {"front": None, "left": None, "back": None, "right": None}
    for r, b in hits:
        if b is not None and (by_side[side(b)] is None or r < by_side[side(b)]):
            by_side[side(b)] = r
    out["nearest_by_side_m"] = by_side
    if len(bearings) > 1 and None not in bearings[:2]:
        out["bearing_first_deg"] = _deg(bearings[0])
        out["bearing_step_deg"] = round((bearings[1] - bearings[0]) % 360, 3)
    out["ranges_m"] = ranges
    return out


def _deg(bearing: float | None) -> float | None:
    return None if bearing is None else round(bearing, 1) + 0.0  # + 0.0: never -0.0


def readings(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, dict[str, Any]]:
    """Every lidar's reading, by name."""
    return {ring.name: reading(model, data, ring) for ring in rings(model)}
