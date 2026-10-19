"""Going somewhere on the robot's own map: the same planner and path follower as on land (navigate.py), with the
map the robot has drawn instead of the one the simulator could compute from the scene.

What the robot knows is its occupancy grid (perception/mapper.py): cells it has seen free, cells it has seen
occupied, and cells it has not seen. The planner treats occupied cells as walls, and unknown ones as walls too
unless asked otherwise (a robot that does not know what is there does not drive through it), so it plans only
through space it has seen to be open. It plans from where its own estimate puts it and the follower steers by
that estimate as well. What the simulator still gives it is the destination: a name ("rack_4") or a point is
the task it was given, as an address is given to a courier; the way there is its own.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import navigate as navigation
from robotforge.behaviors.nav_grid import Grid
from robotforge.behaviors.nav_map import OBSTACLE, Body, NavMap
from robotforge.perception.grid import OCCUPIED, UNKNOWN
from robotforge.perception.mapper import Mapper

OWNER = "the robot's own map"
PAD = 1.0  # m of unknown space added round the map and the trip's ends, so a destination just off the map is on it


class MapPlanner(navigation.Planner):
    """Plans a robot's way over its occupancy grid."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        body: Body,
        destination: navigation.Destination,
        mapper: Mapper,
        arm: navigation.Arm | None = None,
        carried: int = -1,
        unknown_blocks: bool = True,
    ) -> None:
        super().__init__(model, robot, body, destination, arm, None, carried)
        self.mapper = mapper
        self.unknown_blocks = unknown_blocks
        self.state: np.ndarray | None = None  # the map's cell states on the grid last built (UNKNOWN, FREE, OCCUPIED)

    def position(self, data: mujoco.MjData) -> tuple[float, float]:
        pose = self.mapper.pose()
        if pose is None:
            return super().position(data)
        return float(pose[0]), float(pose[1])

    def _inside(self, nav: NavMap) -> str:
        return (
            "too close to something on its map (or to space it has not seen)"
            if self.unknown_blocks
            else ("too close to something on its map")
        )

    def map(self, data: mujoco.MjData, blockers: list[tuple[float, float, float]] = ()) -> NavMap:
        """The map of where the robot's center may go on its own grid, as it stands now (kept as last_map)."""
        nav = self.build(self.position(data), blockers)
        self.last_map = nav
        nav.free_from(nav.grid.cell(*self.position(data)))
        return nav

    def build(self, start: tuple[float, float], blockers: list[tuple[float, float, float]] = ()) -> NavMap:
        own = self.mapper.grid
        assert own is not None
        grid, state = self._cover(own, [start, self.destination.point])
        self.state = state
        lethal = np.zeros(grid.shape, np.int8)
        lethal[state == OCCUPIED] = OBSTACLE
        if self.unknown_blocks:
            lethal[state == UNKNOWN] = OBSTACLE
        disks = navigation._disks(grid, blockers)
        if disks is not None:
            lethal[disks] = OBSTACLE
        owner = np.where(lethal != 0, 0, -1).astype(np.int32)
        zeros = np.zeros(grid.shape)
        return NavMap(grid, self.body, zeros, lethal, owner, [OWNER], zeros.copy(), None)

    @staticmethod
    def _cover(own: Any, points: list[tuple[float, float]]) -> tuple[Grid, np.ndarray]:
        """The map's grid grown with unknown cells to hold `points` and a margin, and its cell states there."""
        state = own.state()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        low = (min(xs) - PAD, min(ys) - PAD)
        high = (max(xs) + PAD, max(ys) + PAD)
        r_lo, c_lo = own.cell(low[0], low[1])
        r_hi, c_hi = own.cell(high[0], high[1])
        pad_r0, pad_c0 = max(-r_lo, 0), max(-c_lo, 0)
        pad_r1, pad_c1 = max(r_hi - own.rows + 1, 0), max(c_hi - own.cols + 1, 0)
        if pad_r0 or pad_c0 or pad_r1 or pad_c1:
            state = np.pad(state, ((pad_r0, pad_r1), (pad_c0, pad_c1)), constant_values=UNKNOWN)
        grid = Grid(own.x0 - pad_c0 * own.res, own.y0 - pad_r0 * own.res, own.res, state.shape[0], state.shape[1])
        return grid, state

    # ---- watching the map while the trip goes -----------------------------------------------------------

    def blocked_ahead(self, path: np.ndarray) -> bool:
        """Whether what is mapped now lies across `path` (a polyline, world x, y): an occupied cell closer to it
        than the robot's half width."""
        own = self.mapper.grid
        if own is None or len(path) < 2:
            return False
        seg = np.diff(path, axis=0)
        length = np.hypot(seg[:, 0], seg[:, 1])
        count = np.maximum(1, np.ceil(length / (own.res * 0.7)).astype(int))
        points = np.concatenate(
            [path[i] + seg[i] * (np.arange(count[i]) / count[i])[:, None] for i in range(len(seg))] + [path[-1:]]
        )
        reach = int(math.ceil(self.body.radius / own.res))
        offsets = np.array(
            [
                (dr, dc)
                for dr in range(-reach, reach + 1)
                for dc in range(-reach, reach + 1)
                if dr * dr + dc * dc <= reach * reach
            ]
        )
        r0 = np.rint((points[:, 1] - own.y0) / own.res).astype(int)
        c0 = np.rint((points[:, 0] - own.x0) / own.res).astype(int)
        rows = (r0[:, None] + offsets[None, :, 0]).ravel()
        cols = (c0[:, None] + offsets[None, :, 1]).ravel()
        ok = (rows >= 0) & (rows < own.rows) & (cols >= 0) & (cols < own.cols)
        return bool(own.occupied()[rows[ok], cols[ok]].any())
