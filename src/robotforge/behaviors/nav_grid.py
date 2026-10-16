"""Grid path search for robots that walk or drive: A* on an 8-connected grid, then string pulling.

The map (nav_map.py) marks the cells a robot's center may not be in. A* (Hart, Nilsson and Raphael, 1968)
finds the cheapest chain of neighboring free cells from the start to any cell of the goal region. A step
costs its length, more near obstacles (`penalty`), so paths keep to the middle of a corridor, where a robot
that sways as it walks or turns on the spot still has room. A diagonal step may not cut a blocked corner.
The heuristic is the straight-line distance to the goal region, which never overestimates, so the path
found is the cheapest one.

A path of cell centers zigzags in 45 degree steps. String pulling, as Theta* (Nash, Daniel, Koenig and
Felner, 2007) does during its search, keeps only the points where the robot has to turn: from each kept
point it goes straight to the furthest later point whose straight line keeps as much room around the
robot as the cells it replaces did.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Grid:
    """Square cells over the floor. Cell (row, col) is centered at (x0 + col * res, y0 + row * res)."""

    x0: float  # m, world x of column 0's center
    y0: float  # m, world y of row 0's center
    res: float  # m, cell side
    rows: int  # along y
    cols: int  # along x

    @classmethod
    def around(cls, low: tuple[float, float], high: tuple[float, float], res: float) -> Grid:
        """The grid that covers the rectangle from `low` to `high` (world x, y)."""
        cols = max(1, math.ceil((high[0] - low[0]) / res) + 1)
        rows = max(1, math.ceil((high[1] - low[1]) / res) + 1)
        return cls(float(low[0]), float(low[1]), float(res), rows, cols)

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.cols

    def cell(self, x: float, y: float) -> tuple[int, int]:
        """The (row, col) whose square holds the point; it may lie outside the grid."""
        return round((y - self.y0) / self.res), round((x - self.x0) / self.res)

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def center(self, row: int, col: int) -> tuple[float, float]:
        return self.x0 + col * self.res, self.y0 + row * self.res

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """World x and y of every cell center, each shaped like the grid."""
        xs = self.x0 + np.arange(self.cols) * self.res
        ys = self.y0 + np.arange(self.rows) * self.res
        return np.meshgrid(xs, ys)


def clearance(blocked: np.ndarray, res: float, cap: float) -> np.ndarray:
    """Room around each cell: meters from its center to the nearest blocked cell's edge, at most `cap`.

    Exact Euclidean distances, from the squared distance being separable: the nearest blocked cell along each
    row first, then the best of those down each column (Felzenszwalb and Huttenlocher's distance transform,
    here by shifted minimums over the `cap` window instead of their lower envelope)."""
    reach = math.ceil(cap / res) + 1
    big = float((2 * reach + 2) ** 2)
    f = np.where(blocked, 0.0, big)
    along_rows = _shifted_min(f, reach, axis=1, fill=big)
    squared = _shifted_min(along_rows, reach, axis=0, fill=big)
    room = np.sqrt(squared) * res - res / 2
    return np.clip(np.where(blocked, 0.0, room), 0.0, cap)


def _shifted_min(values: np.ndarray, reach: int, axis: int, fill: float) -> np.ndarray:
    """min over k in [-reach, reach] of values shifted by k along `axis`, plus k squared."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (reach, reach)
    padded = np.pad(values, pad, constant_values=fill)
    out = values.copy()
    n = values.shape[axis]
    for k in range(-reach, reach + 1):
        if k == 0:
            continue
        window = padded[reach + k : reach + k + n, :] if axis == 0 else padded[:, reach + k : reach + k + n]
        np.minimum(out, window + k * k, out=out)
    return out


def astar(
    grid: Grid,
    free: np.ndarray,
    penalty: np.ndarray,
    start: tuple[int, int],
    goal: np.ndarray,
    heuristic: np.ndarray,
) -> list[tuple[int, int]] | None:
    """The cheapest 8-connected path of free cells from `start` to any cell where `goal` is true, or None.

    `penalty` (>= 0) raises the cost of entering a cell: a step costs its length times 1 + the mean penalty
    of the two cells. `heuristic` is a lower bound of the distance (m) from each cell to the goal region."""
    rows, cols = grid.shape
    width = cols + 2  # a blocked border all round, so no step needs a bounds check
    ok = np.pad(free & np.isfinite(penalty), 1, constant_values=False).ravel().tolist()
    weight = np.pad(1.0 + np.nan_to_num(penalty), 1, constant_values=1.0).ravel().tolist()
    target = np.pad(goal & free, 1, constant_values=False).ravel().tolist()
    guess = np.pad(heuristic, 1, constant_values=0.0).ravel().tolist()
    first = (start[0] + 1) * width + start[1] + 1
    if not grid.inside(*start) or not ok[first]:
        return None
    res = grid.res
    straight = [(1, res), (-1, res), (width, res), (-width, res)]
    diagonal = [(width + 1, 1, width), (width - 1, -1, width), (-width + 1, 1, -width), (-width - 1, -1, -width)]
    cost = {first: 0.0}
    parent = {first: -1}
    done = set()
    heap = [(guess[first], first)]
    found = -1
    while heap:
        _, here = heapq.heappop(heap)
        if here in done:
            continue
        if target[here]:
            found = here
            break
        done.add(here)
        base = cost[here]
        w = weight[here]
        steps = [(here + d, length) for d, length in straight]
        # A diagonal step needs both cells beside it free: cutting a corner would scrape it.
        steps += [(here + d, res * SQRT2) for d, a, b in diagonal if ok[here + a] and ok[here + b]]
        for nxt, length in steps:
            if not ok[nxt] or nxt in done:
                continue
            g = base + length * 0.5 * (w + weight[nxt])
            if g < cost.get(nxt, math.inf):
                cost[nxt] = g
                parent[nxt] = here
                heapq.heappush(heap, (g + guess[nxt], nxt))
    if found < 0:
        return None
    path = []
    while found >= 0:
        path.append((found // width - 1, found % width - 1))
        found = parent[found]
    return path[::-1]


def enclosed(grid: Grid, free: np.ndarray, region: np.ndarray, start: tuple[int, int], budget: int = 20_000) -> bool:
    """Whether the free cells a walk reaches from `region` (8-connected, the same corner rule as astar) are fewer
    than `budget` and leave out `start`. Then no path joins the two, found at the cost of the small side alone:
    a point closed off in a pen took three searches through the whole park (1.3 s) to say there was no way."""
    width = grid.cols + 2
    ok = np.pad(free, 1, constant_values=False).ravel()
    seeds = np.flatnonzero(np.pad(region & free, 1, constant_values=False).ravel())
    if len(seeds) > budget:
        return False
    target = (start[0] + 1) * width + start[1] + 1
    seen = set(seeds.tolist())
    if target in seen:
        return False
    ok_list = ok.tolist()
    straight = (1, -1, width, -width)
    diagonal = ((width + 1, 1, width), (width - 1, -1, width), (-width + 1, 1, -width), (-width - 1, -1, -width))
    frontier = list(seen)
    while frontier:
        grown = []
        for here in frontier:
            steps = [here + d for d in straight]
            steps += [here + d for d, a, b in diagonal if ok_list[here + a] and ok_list[here + b]]
            for cell in steps:
                if not ok_list[cell] or cell in seen:
                    continue
                if cell == target:
                    return False
                seen.add(cell)
                grown.append(cell)
            if len(seen) > budget:
                return False
        frontier = grown
    return True


def reachable(grid: Grid, free: np.ndarray, start: tuple[int, int] | np.ndarray) -> np.ndarray:
    """The free cells an 8-connected walk reaches (same corner rule as astar) from `start`: a cell, or a mask
    of cells to start from all at once."""
    seen = np.zeros(grid.shape, bool)
    if isinstance(start, np.ndarray):
        seen = start & free
    elif grid.inside(*start) and free[start]:
        seen[start] = True
    frontier = seen.copy()
    padded_free = np.pad(free, 1, constant_values=False)
    while frontier.any():
        grown = np.zeros_like(frontier)
        p = np.pad(frontier, 1, constant_values=False)
        r, c = grid.shape
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                step = p[1 - dr : 1 - dr + r, 1 - dc : 1 - dc + c]
                if dr and dc:  # diagonal: both cells beside it must be free
                    side_a = padded_free[1 - dr : 1 - dr + r, 1 : 1 + c]
                    side_b = padded_free[1 : 1 + r, 1 - dc : 1 - dc + c]
                    step = step & side_a & side_b
                grown |= step
        frontier = grown & free & ~seen
        seen |= frontier
    return seen


def pull_string(grid: Grid, path: list[tuple[int, int]], room: np.ndarray, need: float) -> list[tuple[int, int]]:
    """The corners of `path`: a cell is skipped when the straight line past it keeps at least as much room as
    the path did there, up to `need` (m; a line with that much room is always fine)."""
    if len(path) <= 2:
        return list(path)
    along = np.array([room[c] for c in path])
    kept = [0]
    i = 0
    while i < len(path) - 1:
        j = i + 1
        while j + 1 < len(path) and _line_room(grid, path[i], path[j + 1], room) >= min(
            need, float(along[i : j + 2].min())
        ):
            j += 1
        kept.append(j)
        i = j
    return [path[k] for k in kept]


def _line_room(grid: Grid, a: tuple[int, int], b: tuple[int, int], room: np.ndarray) -> float:
    """The least room over the cells a straight line from cell a to cell b passes, sampled every third of a
    cell (both ends included)."""
    n = max(2, math.ceil(3 * math.hypot(b[0] - a[0], b[1] - a[1])) + 1)
    t = np.linspace(0.0, 1.0, n)
    rows = np.rint(a[0] + (b[0] - a[0]) * t).astype(int)
    cols = np.rint(a[1] + (b[1] - a[1]) * t).astype(int)
    return float(room[rows, cols].min())


def room_at(grid: Grid, room: np.ndarray, p: np.ndarray) -> float:
    """The room (m) at world point p, 0 off the grid."""
    r, c = grid.cell(float(p[0]), float(p[1]))
    return float(room[r, c]) if grid.inside(r, c) else 0.0


def room_along(grid: Grid, room: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """The least room (m) along the straight line between world points p and q, 0 if it leaves the grid."""
    a, b = grid.cell(float(p[0]), float(p[1])), grid.cell(float(q[0]), float(q[1]))
    if not (grid.inside(*a) and grid.inside(*b)):
        return 0.0
    return _line_room(grid, a, b, room)


def length(points: list[tuple[float, float]] | np.ndarray) -> float:
    """Length of a polyline, m."""
    p = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
