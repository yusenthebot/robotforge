"""A walking robot looking down at the ground: heights of the scene under points around it, by ray casts.

Legged walkers place their feet with it (quadruped.py): where a swinging foot will land, what lies in its way, the
slope it stands on. It measures the live scene the way a depth camera or a height-scan lidar on the trunk would:
whatever is there counts as ground (terrain, a step, a crate it climbs onto), except the robot itself, which a real
robot filters out of its own scans, and shapes that only look (visual meshes, which nothing stands on).

A ray costs about 1.5 us against a world of 60 shapes; a physics step about 14. Scans are made only in worlds that
have something to step on besides the flat floor (see `needed`), so walking on flat ground costs nothing extra.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.sim.geometry import collides

# m a ray starts again beyond a shape it looks through. Small: a ray that starts again inside the ground cannot see
# it. At 0.1 mm, one passing a Go2's foot mesh started 0.03 mm under a ramp's top and read the floor under it.
PAST = 1e-6
# m above the trunk a ray starts. Ground a robot can step onto is below its hips; from higher up, a Go2 walking
# beside a desk took the desk's top for ground under its feet.
ABOVE = 0.02
# Shapes a ray may look through before it gives up: in and out of each of the robot's own (a Go2's leg is a dozen
# meshes and collision shapes over each other; with 12 a ray at a foot gave up and read the floor under a ramp).
MAX_LAYERS = 64


class GroundScan:
    """Ground heights at (x, y) points in the world for the robot whose root body is `root`."""

    def __init__(self, model: mujoco.MjModel, root: int) -> None:
        self.root = int(model.body_rootid[root])
        ngeom = int(model.ngeom)
        rootid = model.body_rootid[model.geom_bodyid[:ngeom]]
        hit = np.array([collides(model, g) for g in range(ngeom)], dtype=bool)
        # Shapes a ray goes through: the robot's own, and those nothing touches (they only look).
        self.through = (rootid == self.root) | ~hit
        self.needed = needed(model, self.through)
        self._hit = np.zeros(1, np.int32)
        self._down = np.array([0.0, 0.0, -1.0])
        self.rays = 0  # cast so far, for benchmarks

    def height(
        self, model: mujoco.MjModel, data: mujoco.MjData, x: float, y: float, top: float, skip: int = -1
    ) -> float:
        """The ground's height at (x, y), looking down from `top` (m), past body `skip` (a foot's) and the robot's
        own shapes: the floor's 0 when nothing at all is there, NaN when the robot's own shapes hide it."""
        start = np.array([x, y, top])
        gone = 0.0
        for layer in range(MAX_LAYERS):
            self.rays += 1
            far = mujoco.mj_ray(model, data, start, self._down, None, 1, skip, self._hit)
            if far < 0:  # nothing below: no floor, or it started again inside the ground
                return 0.0 if layer == 0 else math.nan
            if not self.through[int(self._hit[0])]:
                return top - gone - far
            gone += far + PAST
            start = np.array([x, y, top - gone])
        return math.nan

    def heights(self, model: mujoco.MjModel, data: mujoco.MjData, points: np.ndarray, top: float) -> np.ndarray:
        """Heights at each (x, y) row of `points`."""
        return np.array([self.height(model, data, float(p[0]), float(p[1]), top) for p in points])


def needed(model: mujoco.MjModel, through: np.ndarray) -> bool:
    """Whether the world has anything to step on but a flat floor: a heightfield, or a solid shape that is not a
    plane and not the robot's own (stairs, ramps, stones, furniture, objects lying about)."""
    for g in range(int(model.ngeom)):
        if through[g]:
            continue
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_HFIELD:
            return True
        if kind != mujoco.mjtGeom.mjGEOM_PLANE:
            return True
    return False
