"""An occupancy grid grown from laser scans: log-odds per cell, unknown until a ray has passed or ended there.

Cell (row, col) is centered at (x0 + col * res, y0 + row * res), the convention of the navigation grid
(behaviors/nav_grid.py), so a robot's own map can be handed to the planner as it is. The grid grows when a scan
reaches past its edge. A scan is integrated the standard way (Moravec and Elfes, 1985; Thrun et al., 2005): every
cell a ray crosses is a little more free, the cell where it ends a little more occupied; a cell seen free from
many scans and occupied from few settles on free, so a person walking through a scan does not leave a wall.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from robotforge.behaviors.nav_grid import Grid, clearance

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
L_OCC, L_FREE = 0.9, -0.45  # log-odds added by one hit and by one ray passing through
L_MIN, L_MAX = -4.0, 4.0
OCC_AT, FREE_AT = 0.6, -0.3  # log-odds above which a cell is occupied, below which it is free
STEP_OF_RES = 0.7  # ray samples are this share of a cell apart: no cell on a ray is skipped
MAX_SAMPLES = 600_000  # ray samples in one scan, at most (a scan with more is thinned)
GROW_BY = 2.0  # m of room added past the point that made the grid grow


class OccupancyGrid:
    def __init__(self, res: float = 0.05, origin: tuple[float, float] = (0.0, 0.0), reach: float = 6.0) -> None:
        """A grid covering `reach` m round `origin`; it grows as scans reach past it."""
        self.res = float(res)
        cells = int(math.ceil(2 * reach / self.res)) + 1
        self.x0 = float(origin[0]) - reach
        self.y0 = float(origin[1]) - reach
        self.logodds = np.zeros((cells, cells), np.float32)
        self.seen = np.zeros((cells, cells), bool)
        self.scans = 0
        self.version = 0  # counts changes: the viewer and the likelihood field are redone only when it moves
        self._field: tuple[int, np.ndarray] | None = None

    # ---- geometry -----------------------------------------------------------------------------------------

    @property
    def rows(self) -> int:
        return self.logodds.shape[0]

    @property
    def cols(self) -> int:
        return self.logodds.shape[1]

    @property
    def grid(self) -> Grid:
        return Grid(self.x0, self.y0, self.res, self.rows, self.cols)

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return round((y - self.y0) / self.res), round((x - self.x0) / self.res)

    def center(self, row: int, col: int) -> tuple[float, float]:
        return self.x0 + col * self.res, self.y0 + row * self.res

    def ensure(self, low: np.ndarray, high: np.ndarray) -> None:
        """Make the grid cover the rectangle from `low` to `high` (world x, y), with room to spare."""
        lo_r, lo_c = self.cell(float(low[0]) - self.res, float(low[1]) - self.res)
        hi_r, hi_c = self.cell(float(high[0]) + self.res, float(high[1]) + self.res)
        before = (min(lo_r, 0), min(lo_c, 0), max(hi_r - self.rows + 1, 0), max(hi_c - self.cols + 1, 0))
        if not any(before):
            return
        extra = int(GROW_BY / self.res)
        pad_r0 = -before[0] + (extra if before[0] < 0 else 0)
        pad_c0 = -before[1] + (extra if before[1] < 0 else 0)
        pad_r1 = before[2] + (extra if before[2] > 0 else 0)
        pad_c1 = before[3] + (extra if before[3] > 0 else 0)
        pad = ((pad_r0, pad_r1), (pad_c0, pad_c1))
        self.logodds = np.pad(self.logodds, pad)
        self.seen = np.pad(self.seen, pad)
        self.x0 -= pad_c0 * self.res
        self.y0 -= pad_r0 * self.res
        self._field = None

    # ---- integrating a scan ---------------------------------------------------------------------------------

    def integrate(
        self,
        origin: np.ndarray,
        ends: np.ndarray,
        hit: np.ndarray,
        rise: np.ndarray | None = None,
        height: float = 0.0,
        clear_below: float = math.inf,
    ) -> None:
        """One scan from `origin` (x, y): rays to `ends` (N x 2). The cells a ray crosses become more free; where
        `hit` is true the end cell becomes more occupied (a ray that ends in the open, or on the floor, only clears).
        A ray that climbs or falls (`rise`: meters of height per meter of the way, from `height` m at the sensor)
        clears a cell only where it passes below `clear_below` m: a 3D lidar's level ray goes over a low crate and
        says nothing about the cells under it."""
        if len(ends) == 0:
            return
        self.ensure(np.minimum(ends.min(axis=0), origin), np.maximum(ends.max(axis=0), origin))
        delta = ends - origin
        length = np.hypot(delta[:, 0], delta[:, 1])
        count = np.maximum(1, np.ceil(length / (self.res * STEP_OF_RES)).astype(np.int64))
        if int(count.sum()) > MAX_SAMPLES:  # a huge scan on a fine grid: every other ray
            keep = np.arange(len(ends)) % 2 == 0
            ends, hit, delta, count, length = ends[keep], hit[keep], delta[keep], count[keep], length[keep]
            rise = rise[keep] if rise is not None else None
        total = int(count.sum())
        ray = np.repeat(np.arange(len(ends)), count)
        first = np.cumsum(count) - count
        t = (np.arange(total) - np.repeat(first, count)) / count[ray]  # 0 at the sensor, up to (not at) the end
        xy = origin + delta[ray] * t[:, None]
        rows = np.rint((xy[:, 1] - self.y0) / self.res).astype(np.int64)
        cols = np.rint((xy[:, 0] - self.x0) / self.res).astype(np.int64)
        free = np.zeros(self.logodds.shape, bool)
        inside = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        if rise is not None and math.isfinite(clear_below):
            inside &= height + rise[ray] * length[ray] * t <= clear_below
        free[rows[inside], cols[inside]] = True
        occupied = np.zeros(self.logodds.shape, bool)
        ends_hit = ends[hit]
        if len(ends_hit):
            er = np.rint((ends_hit[:, 1] - self.y0) / self.res).astype(np.int64)
            ec = np.rint((ends_hit[:, 0] - self.x0) / self.res).astype(np.int64)
            ok = (er >= 0) & (er < self.rows) & (ec >= 0) & (ec < self.cols)
            occupied[er[ok], ec[ok]] = True
        free &= ~occupied
        self.logodds[free] = np.maximum(self.logodds[free] + L_FREE, L_MIN)
        self.logodds[occupied] = np.minimum(self.logodds[occupied] + L_OCC, L_MAX)
        self.seen |= free | occupied
        self.scans += 1
        self.version += 1

    # ---- reading the map ------------------------------------------------------------------------------------

    def state(self) -> np.ndarray:
        """UNKNOWN, FREE or OCCUPIED per cell."""
        out = np.zeros(self.logodds.shape, np.int8)
        out[self.seen & (self.logodds <= FREE_AT)] = FREE
        out[self.logodds >= OCC_AT] = OCCUPIED
        return out

    def occupied(self) -> np.ndarray:
        return self.logodds >= OCC_AT

    def counts(self) -> dict[str, int]:
        state = self.state()
        return {
            "free": int((state == FREE).sum()),
            "occupied": int((state == OCCUPIED).sum()),
            "unknown_seen": int((self.seen & (state == UNKNOWN)).sum()),
        }

    def extent(self) -> tuple[float, float, float, float] | None:
        """The rectangle (x0, y0, x1, y1) in world meters that holds every cell a scan has touched."""
        rows, cols = np.nonzero(self.seen)
        if not len(rows):
            return None
        return (
            self.x0 + cols.min() * self.res,
            self.y0 + rows.min() * self.res,
            self.x0 + cols.max() * self.res,
            self.y0 + rows.max() * self.res,
        )

    def likelihood_field(self, sigma: float = 0.12, cap: float = 0.6) -> np.ndarray:
        """exp(-d^2 / 2 sigma^2) per cell, d the distance to the nearest occupied cell (at most `cap`): how well
        a scan end there agrees with the map. Kept until the map changes."""
        if self._field is not None and self._field[0] == self.version:
            return self._field[1]
        room = clearance(self.occupied(), self.res, cap)
        field = np.exp(-0.5 * (room / sigma) ** 2).astype(np.float32)
        self._field = (self.version, field)
        return field

    def frontier(self) -> np.ndarray:
        """Known free cells beside an unknown one: where the map can still grow."""
        state = self.state()
        free = state == FREE
        unknown = state == UNKNOWN
        beside = np.zeros_like(free)
        beside[1:, :] |= unknown[:-1, :]
        beside[:-1, :] |= unknown[1:, :]
        beside[:, 1:] |= unknown[:, :-1]
        beside[:, :-1] |= unknown[:, 1:]
        return free & beside

    # ---- the viewer and files -------------------------------------------------------------------------------

    def packed(self, margin: int = 4) -> dict | None:
        """The known part of the map for the viewer: the rectangle that holds it, one byte a cell (0 unknown,
        1 free, 2 occupied), deflated. None while nothing has been seen."""
        state = self.state()
        rows, cols = np.nonzero(self.seen)
        if not len(rows):
            return None
        r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, self.rows)
        c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, self.cols)
        crop = np.ascontiguousarray(state[r0:r1, c0:c1], np.uint8)
        return {
            "x0": round(self.x0 + c0 * self.res, 4),
            "y0": round(self.y0 + r0 * self.res, 4),
            "res": self.res,
            "rows": int(crop.shape[0]),
            "cols": int(crop.shape[1]),
            "cells": zlib.compress(crop.tobytes(), 3),
        }

    def copy(self) -> OccupancyGrid:
        other = OccupancyGrid.__new__(OccupancyGrid)
        other.res, other.x0, other.y0 = self.res, self.x0, self.y0
        other.logodds, other.seen = self.logodds.copy(), self.seen.copy()
        other.scans, other.version, other._field = self.scans, self.version, None
        return other
