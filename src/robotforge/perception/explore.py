"""Frontier exploration: go where the map ends.

A frontier is a cell the robot has seen free that has unknown space beside it (Yamauchi, 1997). Going to the
nearest one reveals what is behind it and moves the frontier on; when none is left that the robot can reach,
the part of the world it can get to is mapped. The nearest is by the cost of the robot's own path through
its own map (the planner's search to the whole frontier at once), so it never picks a frontier behind a wall it has
seen. Frontiers that are only a few stray cells (a gap between two scans' rays) are left out, and so are those
a trip to which has failed already.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robotforge.behaviors import nav_grid
from robotforge.behaviors import navigate as navigation
from robotforge.behaviors.map_nav import MapPlanner
from robotforge.perception.grid import FREE, UNKNOWN

WINDOW = 9  # cells: a frontier needs this much unknown space round it to count (0.45 m on 5 cm cells)
UNKNOWN_SHARE = 0.2  # of that window
AVOID_RADIUS = 0.6  # m round a frontier a trip to which failed


@dataclass
class Explorer:
    """Picks the next place to go, trip after trip. `area` ([x0, y0, x1, y1] m) keeps it to a part of the world."""

    area: list[float] | None = None
    avoid: list[tuple[float, float]] = field(default_factory=list)

    def goal(self, planner: MapPlanner, data) -> tuple[float, float] | None:
        """The nearest reachable frontier by the planner's path cost, or None when there is none."""
        nav = planner.map(data)
        state = planner.state
        grid = nav.grid
        unknown = (state == UNKNOWN).astype(np.float32)
        free = state == FREE
        beside = np.zeros_like(free)
        beside[1:, :] |= unknown[:-1, :] > 0
        beside[:-1, :] |= unknown[1:, :] > 0
        beside[:, 1:] |= unknown[:, :-1] > 0
        beside[:, :-1] |= unknown[:, 1:] > 0
        frontier = free & beside & (box_sum(unknown, WINDOW) >= UNKNOWN_SHARE * WINDOW * WINDOW)
        xs, ys = grid.centers()
        if self.area is not None:
            x0, y0, x1, y1 = self.area
            frontier &= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        for x, y in self.avoid:
            frontier &= np.hypot(xs - x, ys - y) > AVOID_RADIUS
        goal = frontier & nav.free
        if not goal.any():
            return None
        start_xy = planner.position(data)
        first, _ = planner._start(nav, grid.cell(*start_xy), start_xy)
        reached = nav_grid.reachable(grid, nav.free, first)
        goal &= reached
        if not goal.any():
            return None
        cells = nav_grid.astar(grid, nav.free, nav.penalty(), first, goal, navigation._heuristic(grid, goal))
        if cells is None:
            return None
        return grid.center(*cells[-1])

    def failed(self, point: tuple[float, float]) -> None:
        self.avoid.append((float(point[0]), float(point[1])))


def box_sum(values: np.ndarray, k: int) -> np.ndarray:
    """The sum of `values` over the k x k window centered on each cell (zero outside the array)."""
    half = k // 2
    padded = np.pad(values, half)
    total = np.cumsum(np.cumsum(np.pad(padded, ((1, 0), (1, 0))), axis=0), axis=1)
    rows, cols = values.shape
    return (
        total[k : k + rows, k : k + cols]
        - total[:rows, k : k + cols]
        - total[k : k + rows, :cols]
        + total[:rows, :cols]
    )
