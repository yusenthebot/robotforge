"""The live world's maps: one mapper per robot, stepped with the simulation, read by tools and drawn by the viewer.

The hub hangs on the playground like the sensor hub does. A world without mapping has no step hook, so nothing
slows the headless step; with a mapper, every step costs one comparison and a scan once every 0.1 to 0.2 s of
simulated time (about 5 ms for a 360-ray scan on a 5 cm grid).
"""

from __future__ import annotations

import base64
import logging
import math
import time
from collections.abc import Generator
from typing import Any

import numpy as np

from robotforge.perception import truth
from robotforge.perception.grid import FREE, OCCUPIED, UNKNOWN
from robotforge.perception.mapper import POSES, Mapper

log = logging.getLogger(__name__)

LIDAR_NAME, ODOMETRY_NAME = "map_lidar", "map_odom"
# What mapping puts on a robot that has no lidar or odometry yet: a 16-ring lidar tilted to see the floor (a flat
# ring at a dog's back height passes over crates and pallets) and an odometry as a robot's
# wheels or legs give it: speed noise of 2 cm/s and a turn that reads 0.003 rad/s (0.17 deg/s) off, so after a
# minute it believes it has turned 10 degrees and walked a meter or two from where it is.
LIDAR_DEFAULTS = {
    "type": "lidar3d",
    "channels": 16,
    "rays": 270,
    "vfov": [-30.0, 6.0],
    "range": 15.0,
    "rate_hz": 8.0,
    "noise": 0.01,
}
ODOMETRY_DEFAULTS = {
    "type": "odometry",
    "noise": {"speed": 0.02, "turn": 0.01},
    "bias": {"turn": 0.003},
    "drift": {"turn": 0.0003},
}
VIEW_PARTICLES = 150  # particles sent to the viewer at most
GRID_EVERY = 3.0  # s: the map is sent again at least this often, so a tab that joins late gets it
FIRST_SCANS, FIRST_LOOK = 3, 1.0  # scans a robot takes standing still before it explores, and for how long
MAX_FAILED_GOALS = 8  # frontiers the robot could not get to before exploring gives up
SCORE_SLAB = 0.05  # m: a 2D lidar's map is scored against the scene's shapes within this far of its plane


class Pause:
    """Stands in for a trip while the robot only looks: done when `seconds` of simulated time have passed."""

    def __init__(self, session: Any, seconds: float) -> None:
        self.session, self.until = session, float(session.data.time) + seconds

    @property
    def done(self) -> bool:
        return float(self.session.data.time) >= self.until

    arrived = True


class PerceptionError(ValueError):
    """A user-facing error: the message says what to do instead."""


class PerceptionHub:
    def __init__(self, playground: Any) -> None:
        self.pg = playground
        self.mappers: dict[str, Mapper] = {}
        self._retired: dict[str, Mapper] = {}
        self._sent: tuple | None = None
        self._sent_at = 0.0
        playground.session.reset_hooks.append(self._reset)

    # ---- the simulation's side ------------------------------------------------------------------------------

    def _reset(self) -> None:
        """A new model or a reset: the maps go (an edit that leaves a robot where it stands gets its own back)."""
        self._retired = self.mappers
        self.mappers = {}
        self._install()

    def rebuild(self, kept: Any) -> None:
        for name in kept:
            if name in self._retired and name not in self.mappers:
                self.mappers[name] = self._retired[name]
        self._retired = {}
        self._install()

    def _install(self) -> None:
        hooks = self.pg.session.step_hooks
        if self._after_step in hooks:
            hooks.remove(self._after_step)
        if self.mappers:
            hooks.append(self._after_step)

    def _after_step(self) -> None:
        session = self.pg.sensors.session
        suite = self.pg.sensors.suite
        for name, mapper in list(self.mappers.items()):
            try:
                mapper.tick(suite, session.model, session.data)
            except Exception as exc:  # a mapper that breaks must not stop the simulation
                log.exception("mapping %s failed", name)
                mapper.note = f"stopped by an error: {type(exc).__name__}: {exc}"
                del self.mappers[name]
                self._install()

    # ---- tools ---------------------------------------------------------------------------------------------

    def resolve(self, robot: str) -> str:
        world = self.pg.current_world()
        robots = [r.name for r in world.robots] if world is not None else []
        try:
            return self.pg.sensors.resolve_robot(robot, robots)
        except ValueError as exc:
            raise PerceptionError(str(exc)) from exc

    def mapper(self, robot: str) -> Mapper:
        robot = self.resolve(robot)
        found = self.mappers.get(robot)
        if found is None:
            raise PerceptionError(
                f"{robot} is not mapping: mapping(robot={robot!r}, action='start') starts a map, then move it about "
                "(or action='explore')"
            )
        return found

    def start(self, robot: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Begin mapping: find the robot's lidar and odometry (add them when it has none), fix the mount constants
        and start the filter at the odometry's first pose. Mapping a robot that already maps carries on with its map."""
        opts = dict(options or {})
        robot = self.resolve(robot)
        pose = str(opts.pop("pose", "estimate"))
        resolution = float(opts.pop("resolution", 0.05))
        particles = int(opts.pop("particles", 150))
        wanted_lidar = opts.pop("lidar", None)
        wanted_odometry = opts.pop("odometry", None)
        if opts:
            raise PerceptionError(
                f"mapping options are pose ({', '.join(POSES)}), resolution (m), particles, lidar and odometry "
                f"(a sensor's name, or its options when it is to be added); not {', '.join(map(repr, opts))}"
            )
        if not 0.02 <= resolution <= 0.5:
            raise PerceptionError(f"resolution is the size of a map cell, 0.02 to 0.5 m, got {resolution}")
        if not 20 <= particles <= 5000:
            raise PerceptionError(f"particles is how many guesses the filter keeps, 20 to 5000, got {particles}")
        if pose not in POSES:
            raise PerceptionError(f"pose is one of {', '.join(POSES)}, got {pose!r}")
        if robot in self.mappers:
            existing = self.mappers[robot]
            existing.mapping = True
            existing.note = "mapping"
            return self.status(robot)
        lidar = self._sensor(robot, wanted_lidar, ("lidar", "lidar3d"), LIDAR_NAME, LIDAR_DEFAULTS)
        odometry = self._sensor(robot, wanted_odometry, ("odometry",), ODOMETRY_NAME, ODOMETRY_DEFAULTS)
        session = self.pg.session
        with session.lock:
            mapper = Mapper(
                robot, f"{robot}/{lidar}", f"{robot}/{odometry}", resolution=resolution, particles=particles, pose=pose
            )
            from robotforge.behaviors import nav_map

            try:
                mapper.start(
                    self.pg.sensors.suite, session.model, session.data, nav_map.root_body(session.model, robot)
                )
            except ValueError as exc:
                raise PerceptionError(str(exc)) from exc
            self.mappers[robot] = mapper
            self._install()
        return self.status(robot)

    def _sensor(self, robot: str, given: Any, types: tuple[str, ...], name: str, defaults: dict[str, Any]) -> str:
        """The local name of the sensor mapping reads: the one named, else the robot's first of the type, else a new
        one."""
        own = self.pg.sensors.suite.of(robot)
        if isinstance(given, str):
            local = given.split("/", 1)[1] if given.startswith(f"{robot}/") else given
            found = [s for s in own if s.local == local and s.type in types]
            if not found:
                raise PerceptionError(
                    f"{robot} has no {'/'.join(types)} sensor named {given!r}; "
                    f"it has {', '.join(s.local for s in own if s.type in types) or 'none'}"
                )
            return found[0].local
        if given is None:
            usable = [s for s in own if s.type in types and not getattr(s, "native", None)]
            if usable:
                return usable[0].local
        fields = {**defaults, **(given or {}), "name": (given or {}).get("name", name)}
        taken = {s.local for s in own}
        if fields["name"] in taken:
            fields["name"] = f"{fields['name']}_{len(taken)}"
        self.pg.add_sensor(robot, fields)
        return str(fields["name"])

    def stop(self, robot: str) -> dict[str, Any]:
        """Stop adding to the map; the map and the filter stay for localizing in it and for navigating."""
        mapper = self.mapper(robot)
        mapper.mapping = False
        mapper.note = "map kept, not growing"
        return self.status(robot)

    def clear(self, robot: str) -> dict[str, Any]:
        robot = self.resolve(robot)
        removed = self.mappers.pop(robot, None) is not None
        self._install()
        return {"robot": robot, "cleared": removed}

    def localize(self, robot: str, init: str = "pose", particles: int | None = None) -> dict[str, Any]:
        """Keep the map as it is and find the robot in it, from its believed pose or from nowhere."""
        if init not in ("pose", "global"):
            raise PerceptionError("init is 'pose' (from where the robot believes it is) or 'global' (from nowhere)")
        mapper = self.mapper(robot)
        with self.pg.session.lock:
            if not mapper.localize_only(init, particles):
                raise PerceptionError("the map has no free space yet: map first (action='start', then move the robot)")
        return self.status(mapper.robot)

    # ---- exploring -----------------------------------------------------------------------------------------

    def explore_trips(
        self, robot: str, seconds: float, area: list[float] | None = None, speed: float | None = None
    ) -> Generator[tuple[Any, float], None, dict[str, Any]]:
        """Explore for up to `seconds` of simulated time: yields (trip, seconds it may take) for each trip to the
        nearest frontier, for the caller to run the simulation until the trip is done (or the time is up), and
        returns how it ended: "explored" (no frontier is left that the robot can reach), "out_of_time", "stuck" or
        "blind"."""
        from robotforge.behaviors.navigate import NoWay
        from robotforge.perception.explore import Explorer

        mapper = self.mapper(robot)
        robot = mapper.robot
        session = self.pg.session
        explorer = Explorer(area=area)
        started = float(session.data.time)
        trips = failures = 0
        reason = "out_of_time"
        while True:
            left = seconds - (float(session.data.time) - started)
            if left <= 0.5:
                break
            if mapper.scan_count < FIRST_SCANS:  # nothing to go by yet: stand and look
                yield Pause(session, FIRST_LOOK), FIRST_LOOK + 1.0
                if mapper.scan_count < FIRST_SCANS and float(session.data.time) - started > 5.0:
                    reason = "blind"
                    break
                continue
            planner, data = self.pg.behaviors.own_map_planner(robot, mapper)
            goal = explorer.goal(planner, data)
            if goal is None:
                reason = "explored"
                break
            try:
                trip = self.pg.behaviors.navigate(
                    robot, [goal[0], goal[1]], speed, None, left, own_map=mapper, unknown_blocks=False
                )
            except NoWay:
                explorer.failed(goal)
                failures += 1
                if failures >= MAX_FAILED_GOALS:
                    reason = "stuck"
                    break
                continue
            trips += 1
            before = float(session.data.time)
            yield trip, left
            explorer.failed(
                goal
            )  # visited: a frontier that is still there (the map ends in the open) is not gone to again
            quick = float(session.data.time) - before < 0.5  # it was there already: no time passed, nothing learned
            if quick or (not trip.arrived and getattr(trip, "traveled", 0.0) < 0.3):
                failures += 1
                if failures >= MAX_FAILED_GOALS:
                    reason = "stuck"
                    break
        self.pg.behaviors.stop(robot)
        return {"ended": reason, "trips": trips, "failed_goals": failures}

    # ---- reading -------------------------------------------------------------------------------------------

    def status(self, robot: str, score: bool = True) -> dict[str, Any]:
        mapper = self.mapper(robot)
        session = self.pg.session
        with session.lock:
            out = self._summary(mapper, score)
        return out

    def _summary(self, mapper: Mapper, score: bool = True) -> dict[str, Any]:
        session = self.pg.session
        out: dict[str, Any] = {
            "robot": mapper.robot,
            "state": mapper.note,
            "mapping": mapper.mapping,
            "pose_source": mapper.pose_source,
            "lidar": mapper.lidar_name.split("/", 1)[1],
            "odometry": mapper.odometry_name.split("/", 1)[1],
            "scans": mapper.scan_count,
            "scan_matches": mapper.matched,
            "time": round(float(session.data.time), 2),
        }
        grid = mapper.grid
        if grid is None:
            return out
        state = grid.state()
        cell = grid.res**2
        extent = grid.extent()
        out["map"] = {
            "resolution_m": grid.res,
            "free_m2": round(float((state == FREE).sum() * cell), 1),
            "occupied_m2": round(float((state == OCCUPIED).sum() * cell), 2),
            "seen_box_m": None if extent is None else [round(v, 2) for v in extent],
        }
        pose = mapper.pose()
        if pose is not None:
            est = mapper.estimate()
            out["pose"] = {
                "x": round(float(pose[0]), 3),
                "y": round(float(pose[1]), 3),
                "heading_deg": round(math.degrees(float(pose[2])), 1),
                "localized": bool(est.sigma_m < 0.25),
                "sigma_m": round(est.sigma_m, 3),
            }
            real = mapper.root_truth(session.data)
            out["pose_error"] = truth.pose_errors(pose, real)
            if mapper.odometry_pose is not None:
                out["odometry_error"] = truth.pose_errors(mapper.odometry_pose, real)
        if score:
            out["accuracy"] = self.accuracy(mapper)
        return out

    def pose_check(self, robot: str) -> dict[str, Any]:
        """Where the robot believes it is, and how far that is from where it is (the truth is read to say)."""
        mapper = self.mapper(robot)
        with self.pg.session.lock:
            pose = mapper.pose()
            if pose is None:
                return {"believed": None}
            return {
                "believed": [round(float(pose[0]), 3), round(float(pose[1]), 3)],
                "heading_deg": round(math.degrees(float(pose[2])), 1),
                **{
                    f"error_{k}": v for k, v in truth.pose_errors(pose, mapper.root_truth(self.pg.session.data)).items()
                },
            }

    def picture(self, robot: str) -> bytes:
        """The map as a PNG (perception/picture.py)."""
        from robotforge.perception import picture

        mapper = self.mapper(robot)
        with self.pg.session.lock:
            try:
                return picture.map_png(mapper)
            except ValueError as exc:
                raise PerceptionError(str(exc)) from exc

    def accuracy(self, mapper: Mapper) -> dict[str, Any]:
        """The map against the scene (see perception/truth.py)."""
        session = self.pg.session
        if mapper.three_d:  # a 3D lidar sees every height up to its own: whatever stands in the robot's way
            low, high = 0.06, mapper.height + 0.15
        else:
            low, high = mapper.height - SCORE_SLAB, mapper.height + SCORE_SLAB
        wall = truth.scene_occupancy(session.model, session.data, mapper.robot, mapper.grid, low, high)
        body = truth.scene_occupancy(session.model, session.data, mapper.robot, mapper.grid, 0.06, mapper.height + 0.15)
        return truth.score(mapper.grid, wall, body)

    def errors(self, robot: str) -> dict[str, np.ndarray]:
        """Pose error over the run, per scan: time, the filter's, the odometry's alone."""
        return self.mapper(robot).trace.errors()

    # ---- the viewer ----------------------------------------------------------------------------------------

    def viewer_message(self) -> dict[str, Any] | None:
        """The "map" message: for each mapping robot the map's known part (deflated, sent when it changed or every
        GRID_EVERY s), its pose estimate and the odometry's alone, and its particles. None without a mapper."""
        if not self.mappers:
            return None
        session = self.pg.session
        now = time.monotonic()
        with session.lock:
            marker = tuple((n, m.grid.version if m.grid else -1, m.mapping) for n, m in self.mappers.items())
            with_grid = marker != self._sent or now - self._sent_at > GRID_EVERY
            robots = []
            for name, mapper in self.mappers.items():
                entry: dict[str, Any] = {"robot": name, "mapping": mapper.mapping, "scans": mapper.scan_count}
                if mapper.estimate_pose is not None:
                    entry["pose"] = [round(float(v), 4) for v in mapper.estimate_pose]
                if mapper.odometry_pose is not None:
                    entry["odometry"] = [round(float(v), 4) for v in mapper.odometry_pose]
                flt = mapper.filter
                if mapper.pose_source == "estimate":
                    step = max(1, flt.n // VIEW_PARTICLES)
                    entry["particles"] = [
                        round(float(v), 3) for v in np.stack([flt.x[::step], flt.y[::step]], axis=1).ravel()
                    ]
                    entry["sigma_m"] = round(mapper.estimate().sigma_m, 3)
                if with_grid and mapper.grid is not None:
                    packed = mapper.grid.packed()
                    if packed is not None:
                        packed["cells"] = base64.b64encode(packed["cells"]).decode("ascii")
                        entry["grid"] = packed
                robots.append(entry)
            if with_grid:
                self._sent, self._sent_at = marker, now
            return {"type": "map", "version": session.version, "robots": robots}


__all__ = ["PerceptionError", "PerceptionHub", "UNKNOWN"]
