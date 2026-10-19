"""A robot that builds a map of where it is going and finds itself in it, from its own sensors.

The mapper listens to two of the robot's sensors: a lidar (the scans) and an odometry (the dead-reckoned pose,
which drifts the way a real one does). Each scan moves a particle filter by the odometry's step, weighs the
particles against the map built so far, and the filter's mean is the robot's pose estimate; the scan is then
added to the occupancy grid at that pose. So the filter pulls the odometry's drift back wherever the map has
walls to hold on to, and the map is drawn from the corrected poses (a particle-filter scan-matching mapper, the
idea of Gmapping with one map instead of one per particle). `pose="truth"` maps from the simulator's exact pose
(mapping with known poses, the best a map can be) and `pose="odometry"` from the odometry alone (the map smears
as it drifts): they exist to show what the filter buys.

The map is in the world's frame, which the robot is told at the start (its first odometry sample is its
pose there). Nothing about the scene is read from the simulator while mapping except two constants fixed when
mapping starts: where the lidar and the robot's root sit relative to the odometry, which the robot's own design
gives it. The ground truth is read only to score the map and the pose (perception/truth.py) and when asked to
(`pose="truth"`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robotforge.perception.grid import OccupancyGrid
from robotforge.perception.particles import ParticleFilter, wrap

MAX_SCAN_HZ = 10.0  # scans used a second, at most, whatever the lidar's rate
DEFAULT_SCAN_HZ = 5.0
# The filter matches the rays that run nearly level (within MATCH_LEVEL of the horizontal): a 3D lidar's steeper rays
# meet the same shelf at a different board on each pass, and matching them made the pose drift three times as far.
MATCH_LEVEL = math.radians(2.5)
LEVEL_RAY = math.radians(2.0)  # a ray that missed counts as open space only if it runs this level or lower
FLOOR_HIT = 0.05  # m above the floor: lower hits are the floor
OVER_HEAD = 0.1  # m above the lidar: higher hits are overhead (a shelf board above a robot that fits under)
TRACE_EVERY = 2  # scans between recorded poses
TRACE_LIMIT = 4000
POSES = ("estimate", "odometry", "truth")


def compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pose b (given in a's frame) in the frame a is given in."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def inverse(a: np.ndarray) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap(-a[2])])


def planar_pose(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """x, y and heading (about the world's z) of a frame given by its position and 3x3 rotation matrix."""
    return np.array([float(pos[0]), float(pos[1]), math.atan2(float(rot[1, 0]), float(rot[0, 0]))])


@dataclass
class Trace:
    """Poses at the scans, for the error plots: time, estimate, odometry alone and the truth (x, y, heading)."""

    t: list[float] = field(default_factory=list)
    estimate: list[np.ndarray] = field(default_factory=list)
    odometry: list[np.ndarray] = field(default_factory=list)
    truth: list[np.ndarray] = field(default_factory=list)

    def add(self, t: float, estimate: np.ndarray, odometry: np.ndarray, truth: np.ndarray) -> None:
        if len(self.t) >= TRACE_LIMIT:  # keep a long run's shape: every other point
            for column in (self.t, self.estimate, self.odometry, self.truth):
                del column[::2]
        self.t.append(t)
        self.estimate.append(estimate)
        self.odometry.append(odometry)
        self.truth.append(truth)

    def errors(self) -> dict[str, np.ndarray]:
        t = np.array(self.t)
        est, odo, tru = (np.array(v) for v in (self.estimate, self.odometry, self.truth))
        if not len(t):
            return {"t": t, "estimate_m": t, "odometry_m": t, "heading_deg": t, "odometry_heading_deg": t}
        return {
            "t": t,
            "estimate_m": np.hypot(*(est[:, :2] - tru[:, :2]).T),
            "odometry_m": np.hypot(*(odo[:, :2] - tru[:, :2]).T),
            "heading_deg": np.degrees(np.abs(wrap(est[:, 2] - tru[:, 2]))),
            "odometry_heading_deg": np.degrees(np.abs(wrap(odo[:, 2] - tru[:, 2]))),
        }


class Mapper:
    """Maps and localizes one robot. Built by perception.hub.PerceptionHub, which calls `tick` after every step."""

    def __init__(
        self,
        robot: str,
        lidar: str,
        odometry: str,
        *,
        resolution: float = 0.05,
        particles: int = 150,
        pose: str = "estimate",
        seed: int = 0,
    ) -> None:
        if pose not in POSES:
            raise ValueError(f"pose is one of {', '.join(POSES)}, got {pose!r}")
        self.robot, self.lidar_name, self.odometry_name = robot, lidar, odometry
        self.resolution, self.pose_source, self.seed = float(resolution), pose, seed
        self.filter = ParticleFilter(particles, seed)
        self.grid: OccupancyGrid | None = None
        self.mapping = True  # False: the map is kept as it is and the robot only localizes in it
        self.trace = Trace()
        self.started_at: float | None = None
        self.next_due = 0.0
        self.last_stamp = -math.inf
        self.previous: np.ndarray | None = None  # the odometry pose at the last scan
        self.lidar_in_odo: np.ndarray | None = None  # the lidar's pose in the odometry's frame (a constant)
        self.root_in_odo: np.ndarray | None = None  # ... and the robot's root's
        self.estimate_pose: np.ndarray | None = None  # the robot's root, as the filter puts it (x, y, heading)
        self.odometry_pose: np.ndarray | None = None  # ... as the odometry alone puts it
        self.scan_count = 0
        self.matched = 0
        self.height = 0.0  # m: where the lidar's plane is, averaged over the scans (to score a map against the scene)
        self.three_d = False
        self.last_ends: np.ndarray | None = None  # the last scan's obstacle ends in the world, for the viewer
        self.level_ends = np.zeros((0, 2))  # the ends of the rays that run nearly level: the filter matches those
        self.note = "starting"
        self.root = -1

    # ---- starting ------------------------------------------------------------------------------------------

    def start(self, suite: Any, model: mujoco.MjModel, data: mujoco.MjData, root: int) -> None:
        """Fix the constants from the robot's design and put the filter at the odometry's first pose."""
        lidar, odometry = self._sensors(suite)
        if lidar.native:
            raise ValueError(
                f"{self.lidar_name} is a ring of rangefinders on a built robot; add a lidar with add_sensor"
            )
        self.root = root
        truth_odo = self._truth_odometry(odometry, model, data)
        truth_lidar = planar_pose(data.site_xpos[lidar.site], data.site_xmat[lidar.site].reshape(3, 3))
        self.lidar_in_odo = compose(inverse(truth_odo), truth_lidar)
        self.root_in_odo = compose(inverse(truth_odo), self.root_truth(data))
        self.started_at = float(data.time)
        self.next_due = float(data.time)
        got = odometry.current(model, data)
        first = got[1][:3] if got is not None else truth_odo
        self.filter.start_at(np.array(first, float), 0.02, 0.01)
        self.previous = np.array(first, float)
        if self.grid is None:
            self.grid = OccupancyGrid(self.resolution, tuple(truth_lidar[:2]), reach=4.0)
        self.note = "mapping"

    def root_truth(self, data: mujoco.MjData) -> np.ndarray:
        """Where the robot's root really is: for scoring (and for pose="truth"), never for the estimate."""
        return planar_pose(data.xpos[self.root], data.xmat[self.root].reshape(3, 3))

    def _sensors(self, suite: Any) -> tuple[Any, Any]:
        lidar = suite.sensors.get(self.lidar_name)
        odometry = suite.sensors.get(self.odometry_name)
        if lidar is None or odometry is None:
            missing = self.lidar_name if lidar is None else self.odometry_name
            raise ValueError(f"the sensor {missing!r} is not on the robot (any more)")
        return lidar, odometry

    @staticmethod
    def _truth_odometry(odometry: Any, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        value = odometry.measure(model, data)
        return np.array([value["pos"][0], value["pos"][1], value["heading"]], float)

    # ---- one step ---------------------------------------------------------------------------------------------

    def tick(self, suite: Any, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called after every physics step: take a scan when one is due and use it."""
        t = float(data.time)
        if t + 1e-9 < self.next_due or self.grid is None:
            return
        lidar = suite.sensors.get(self.lidar_name)
        odometry = suite.sensors.get(self.odometry_name)
        if lidar is None or odometry is None:
            self.note = "waiting for its sensors"
            self.next_due = t + 0.5
            return
        period = 1.0 / min(MAX_SCAN_HZ, lidar.knobs.rate_hz or DEFAULT_SCAN_HZ)
        self.next_due = t + period
        got = lidar.current(model, data)
        if got is None or got[0] <= self.last_stamp + 1e-9:
            return
        stamp, scan = got
        self.last_stamp = stamp
        sample = self._odometry_at(odometry, stamp, model, data)
        if sample is None:
            return
        self.scan(lidar, scan, sample, model, data)

    @staticmethod
    def _odometry_at(odometry: Any, stamp: float, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        """The odometry's pose (x, y, heading) as of `stamp`: its newest sample not after it."""
        if not odometry.samples:
            got = odometry.current(model, data)
            return None if got is None else np.array(got[1][:3], float)
        for when, value in reversed(odometry.samples):
            if when <= stamp + 1e-6:
                return np.array(value[:3], float)
        return np.array(odometry.samples[0][1][:3], float)

    def scan(
        self, lidar: Any, scan: dict[str, Any], odo: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData
    ) -> None:
        """Use one scan with the odometry pose at that moment: move the filter, weigh it, add the scan to the map."""
        assert self.grid is not None and self.previous is not None and self.lidar_in_odo is not None
        origin, rot = scan["origin"], scan["rot"]
        self.height += (float(origin[2]) - self.height) / (self.scan_count + 1)
        self.three_d = lidar.three_d
        ends, ground_ends, open_ends, rise = self._ends(lidar, scan)
        if self.pose_source == "estimate" or self.pose_source == "odometry":
            step = odo - self.previous
            c, s = math.cos(self.previous[2]), math.sin(self.previous[2])
            forward, sideways = c * step[0] + s * step[1], -s * step[0] + c * step[1]
            self.filter.predict(forward, sideways, float(wrap(step[2])))
        self.previous = odo
        if (
            self.pose_source == "estimate"
            and self.filter.due()
            and self.filter.weigh(self.grid, self.level_ends, self.lidar_in_odo)
        ):
            self.matched += 1
        odo_pose = odo
        if self.pose_source == "truth":
            lidar_pose = planar_pose(origin, rot)
            root_pose = self.root_truth(data)
        else:
            body = odo_pose if self.pose_source == "odometry" else self.filter.estimate().pose
            lidar_pose = compose(body, self.lidar_in_odo)
            root_pose = compose(body, self.root_in_odo)
        self.estimate_pose = root_pose
        self.odometry_pose = compose(odo_pose, self.root_in_odo)
        if self.mapping:
            self._integrate(lidar_pose, float(origin[2]), ends, ground_ends, open_ends, rise)
        self.scan_count += 1
        if self.scan_count % TRACE_EVERY == 0:
            self.trace.add(float(data.time), root_pose, self.odometry_pose, self.root_truth(data))

    def _ends(self, lidar: Any, scan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The scan's rays as planar vectors in the lidar's own frame, and how steeply each climbs (m of height per m
        of the way): obstacle ends first (hits in the height the robot occupies), then floor ends (free up to them),
        then rays that left the range level (free to the range). Hits on the floor or overhead are no obstacle."""
        dist, rot, origin = scan["dist"], scan["rot"], scan["origin"]
        directions = lidar.directions @ rot.T  # world directions of the rays
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
        c, s = math.cos(-yaw), math.sin(-yaw)

        def planar(vectors: np.ndarray) -> np.ndarray:
            return np.stack([c * vectors[:, 0] - s * vectors[:, 1], s * vectors[:, 0] + c * vectors[:, 1]], axis=1)

        def rise(vectors: np.ndarray) -> np.ndarray:
            return vectors[:, 2] / np.maximum(np.hypot(vectors[:, 0], vectors[:, 1]), 1e-6)

        finite = np.isfinite(dist)
        vec = np.zeros((len(dist), 3))
        vec[finite] = directions[finite] * dist[finite, None]
        height = origin[2] + vec[:, 2]
        wall = finite & ~scan["ground"] & (height > FLOOR_HIT) & (height < origin[2] + OVER_HEAD)
        floor = finite & scan["ground"]
        elevation = np.repeat(lidar.elevations, lidar.per_ring)
        miss = scan.get("miss")
        level = np.zeros(len(dist), bool) if miss is None else (miss & (np.abs(elevation) <= LEVEL_RAY))
        far = directions[level] * (0.98 * lidar.range)
        near_level = wall & (np.abs(directions[:, 2]) <= math.sin(MATCH_LEVEL))
        self.level_ends = planar(vec[near_level]) if near_level.sum() >= 8 else planar(vec[wall])
        return (
            planar(vec[wall]),
            planar(vec[floor]),
            planar(far) if level.any() else np.zeros((0, 2)),
            np.concatenate([rise(vec[wall]), rise(vec[floor]), rise(far) if level.any() else np.zeros(0)]),
        )

    def _integrate(
        self,
        pose: np.ndarray,
        origin_z: float,
        ends: np.ndarray,
        ground_ends: np.ndarray,
        open_ends: np.ndarray,
        rise: np.ndarray,
    ) -> None:
        c, s = math.cos(pose[2]), math.sin(pose[2])

        def world(local: np.ndarray) -> np.ndarray:
            return np.stack(
                [pose[0] + c * local[:, 0] - s * local[:, 1], pose[1] + s * local[:, 0] + c * local[:, 1]], axis=1
            )

        parts = [world(ends), world(ground_ends), world(open_ends)]
        hit = np.concatenate([np.ones(len(ends), bool), np.zeros(len(ground_ends) + len(open_ends), bool)])
        all_ends = np.concatenate(parts) if len(hit) else np.zeros((0, 2))
        # A 3D lidar's rays clear only the low air (half its height): what stands in the robot's way is seen by the
        # steeper rays, and the level ones pass over it. A flat ring clears at its own height: it sees nothing else.
        clear_below = 0.5 * origin_z if self.three_d else origin_z + 0.05
        self.grid.integrate(pose[:2], all_ends, hit, rise, origin_z, clear_below)
        self.last_ends = parts[0]

    # ---- reading --------------------------------------------------------------------------------------------

    def pose(self) -> np.ndarray | None:
        """The robot's root pose (x, y, heading) as the robot itself believes it, or None before a scan."""
        return self.estimate_pose

    def estimate(self) -> Any:
        return self.filter.estimate()

    def localize_only(self, init: str, particles: int | None = None) -> bool:
        """Stop changing the map and find the robot in it: from where it believes it is (`init` "pose") or from
        nowhere ("global": the particles are spread over the free cells and gather as the robot moves)."""
        self.mapping = False
        self.pose_source = "estimate"
        if init == "global":
            ok = self.filter.start_anywhere(self.grid, particles or 1500)
            self.note = "localizing from nowhere" if ok else "the map has no free cell to start from"
            return ok
        if self.estimate_pose is not None and self.root_in_odo is not None:  # from what it believes now
            self.filter.start_at(compose(self.estimate_pose, inverse(self.root_in_odo)), 0.05, 0.03)
        self.note = "localizing"
        return True
