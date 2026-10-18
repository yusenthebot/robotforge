"""Room on the floor and on surfaces: footprints that must not overlap, and the floor a robot can get to.

A footprint is a rectangle seen from above, turned by a yaw. The generator keeps every piece of furniture's
footprint and every thing's footprint on its surface, and puts nothing where it would overlap another (with a gap).
`FloorMap` rasterizes what stands on the floor, grown by a robot's radius, and floods it from where the robot
starts: a place is reachable when free floor it can get to lies within an arm's reach of it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Footprint:
    """A rectangle seen from above: center (x, y), half extents (hx, hy) in its own frame, turned by yaw degrees."""

    x: float
    y: float
    hx: float
    hy: float
    yaw: float = 0.0

    def corners(self) -> list[tuple[float, float]]:
        a = math.radians(self.yaw)
        c, s = math.cos(a), math.sin(a)
        return [
            (self.x + c * dx - s * dy, self.y + s * dx + c * dy)
            for dx, dy in ((-self.hx, -self.hy), (self.hx, -self.hy), (self.hx, self.hy), (-self.hx, self.hy))
        ]

    def grown(self, by: float) -> Footprint:
        return Footprint(self.x, self.y, self.hx + by, self.hy + by, self.yaw)

    def overlaps(self, other: Footprint, gap: float = 0.0) -> bool:
        """Do the two come closer than `gap` (separating axis test on the rectangles grown by half the gap)?"""
        a, b = self.grown(gap / 2).corners(), other.grown(gap / 2).corners()
        for poly in (a, b):
            for i in range(4):
                (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
                nx, ny = y2 - y1, x1 - x2
                pa = [nx * px + ny * py for px, py in a]
                pb = [nx * px + ny * py for px, py in b]
                if max(pa) < min(pb) or max(pb) < min(pa):
                    return False
        return True

    def local(self, px: float, py: float) -> tuple[float, float]:
        a = math.radians(self.yaw)
        dx, dy = px - self.x, py - self.y
        return dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)

    def world(self, lx: float, ly: float) -> tuple[float, float]:
        a = math.radians(self.yaw)
        return self.x + lx * math.cos(a) - ly * math.sin(a), self.y + lx * math.sin(a) + ly * math.cos(a)

    def contains(self, px: float, py: float, margin: float = 0.0) -> bool:
        lx, ly = self.local(px, py)
        return abs(lx) <= self.hx - margin and abs(ly) <= self.hy - margin

    def gap_to(self, px: float, py: float) -> float:
        """How far the point is from the rectangle (0 inside)."""
        lx, ly = self.local(px, py)
        return math.hypot(max(abs(lx) - self.hx, 0.0), max(abs(ly) - self.hy, 0.0))

    def bounds(self) -> tuple[float, float, float, float]:
        xs, ys = zip(*self.corners(), strict=True)
        return min(xs), max(xs), min(ys), max(ys)


@dataclass
class Surface:
    """A flat top things can stand on: a table, a counter, a shelf's board, a pallet."""

    owner: str  # the object it is the top of
    area: Footprint  # where things may stand, already kept off the edges
    z: float  # m, the top's height above the ground
    clearance: float = 2.0  # m of free height over it (a shelf board has the next board above)
    room: str | None = None
    items: list[Footprint] = field(default_factory=list)
    kind: str = "table"  # table, counter, shelf, chest, bed, pallet or ground: an arm stands on tables and counters


CELL = 0.05  # m, the floor map's resolution


@dataclass
class FloorMap:
    """What stands on the floor, as a grid, grown by a robot's radius; and which cells a robot gets to from a start."""

    x0: float
    y0: float
    blocked: np.ndarray  # [rows (y), cols (x)] bool
    reach: np.ndarray | None = None  # cells the robot gets to

    @classmethod
    def of(cls, bounds: tuple[float, float, float, float], obstacles: list[Footprint], radius: float) -> FloorMap:
        x0, x1, y0, y1 = bounds
        cols, rows = int(math.ceil((x1 - x0) / CELL)) + 1, int(math.ceil((y1 - y0) / CELL)) + 1
        blocked = np.zeros((rows, cols), dtype=bool)
        # the outside of the bounds is wall: grow it by the radius too
        edge = int(math.ceil(radius / CELL))
        blocked[:edge, :] = blocked[-edge:, :] = True
        blocked[:, :edge] = blocked[:, -edge:] = True
        ys = y0 + np.arange(rows) * CELL
        xs = x0 + np.arange(cols) * CELL
        gx, gy = np.meshgrid(xs, ys)
        for fp in obstacles:
            bx0, bx1, by0, by1 = fp.grown(radius).bounds()
            c0, c1 = max(0, int((bx0 - x0) / CELL)), min(cols, int((bx1 - x0) / CELL) + 2)
            r0, r1 = max(0, int((by0 - y0) / CELL)), min(rows, int((by1 - y0) / CELL) + 2)
            if c0 >= c1 or r0 >= r1:
                continue
            a = math.radians(fp.yaw)
            dx, dy = gx[r0:r1, c0:c1] - fp.x, gy[r0:r1, c0:c1] - fp.y
            lx, ly = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
            ex, ey = np.maximum(np.abs(lx) - fp.hx, 0.0), np.maximum(np.abs(ly) - fp.hy, 0.0)
            blocked[r0:r1, c0:c1] |= ex * ex + ey * ey <= radius * radius
        return cls(x0, y0, blocked)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.y0) / CELL)), int(round((x - self.x0) / CELL))

    def free(self, x: float, y: float) -> bool:
        r, c = self.cell(x, y)
        rows, cols = self.blocked.shape
        return 0 <= r < rows and 0 <= c < cols and not self.blocked[r, c]

    def flood(self, x: float, y: float) -> int:
        """Mark the cells a robot starting at (x, y) gets to; returns how many."""
        rows, cols = self.blocked.shape
        reach = np.zeros_like(self.blocked)
        start = self.cell(x, y)
        if not self.free(x, y):
            self.reach = reach
            return 0
        queue = deque([start])
        reach[start] = True
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not reach[nr, nc] and not self.blocked[nr, nc]:
                    reach[nr, nc] = True
                    queue.append((nr, nc))
        self.reach = reach
        return int(reach.sum())

    def reachable_within(self, x: float, y: float, distance: float) -> bool:
        """Is there floor the robot gets to within `distance` of (x, y)? (flood first)"""
        if self.reach is None:
            raise RuntimeError("flood the map from the robot's start first")
        r, c = self.cell(x, y)
        n = int(math.ceil(distance / CELL))
        rows, cols = self.reach.shape
        r0, r1, c0, c1 = max(0, r - n), min(rows, r + n + 1), max(0, c - n), min(cols, c + n + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        window = self.reach[r0:r1, c0:c1]
        if not window.any():
            return False
        rr, cc = np.nonzero(window)
        d2 = ((rr + r0 - r) * CELL) ** 2 + ((cc + c0 - c) * CELL) ** 2
        return bool((d2 <= distance * distance).any())

    def reachable(self, x: float, y: float) -> bool:
        if self.reach is None:
            raise RuntimeError("flood the map from the robot's start first")
        r, c = self.cell(x, y)
        rows, cols = self.reach.shape
        return 0 <= r < rows and 0 <= c < cols and bool(self.reach[r, c])
