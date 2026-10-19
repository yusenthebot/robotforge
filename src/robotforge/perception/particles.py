"""Particle-filter localization (Monte Carlo localization, Dellaert et al. 1999) in an occupancy grid.

Each particle is a guess of the robot's pose (x, y, heading). Odometry moves every guess, with the noise a real
odometry has; a laser scan then weighs each guess by how well the scan's end points fall on the map's occupied
cells (the likelihood field, Thrun et al. 2005); guesses that fit survive, the others are replaced. The estimate is
the weighted mean. Started at a known pose it corrects the odometry's drift against the map; started spread over
the map's free cells ("global") it finds out where it is as the robot moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotforge.perception.grid import FREE, OccupancyGrid

BEAMS = 60  # scan ends used to weigh a particle: more cost time and add little
Z_HIT, Z_RANDOM = 0.9, 0.1  # a scan end is on the map, or a stray one (a person, a crate that moved)
MOVE_TRANS, MOVE_ROT = 0.03, 0.02  # odometry noise: m per m moved, rad per rad turned
CROSS_TRANS, CROSS_ROT = 0.02, 0.01  # ... and m per rad turned, rad per m moved
FLOOR_TRANS, FLOOR_ROT = 0.002, 0.002  # noise even for a robot that barely moved (m, rad)
MIN_MOVE, MIN_TURN = 0.02, math.radians(1.5)  # a scan is weighed once the robot moved this much since the last
MIN_OCCUPIED = 30  # occupied cells the map needs before scans can be matched to it
# Looking for itself on the whole map, the filter is not yet sure: it softens the scans' say (a scan taken in a room
# like several others) and sprinkles new guesses over the map while the robot has traveled less than SEARCH_M.
SEARCH_M = 6.0
SEARCH_TEMPER = 0.25  # weight exponent at the start of a search, rising to 1 as it travels
SEARCH_INJECT = 0.03  # share of the particles replaced by new guesses at each resampling during a search


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def spread(ends: np.ndarray, count: int) -> np.ndarray:
    """`count` of the scan's ends spread evenly in bearing, the farthest of each slice (a far wall pins the heading
    down the most)."""
    bearing = np.arctan2(ends[:, 1], ends[:, 0])
    slot = np.minimum(((bearing + math.pi) / (2 * math.pi) * count).astype(int), count - 1)
    reach = np.hypot(ends[:, 0], ends[:, 1])
    order = np.lexsort((reach, slot))  # by slot, then by range
    last = np.ones(len(order), bool)
    last[:-1] = slot[order][1:] != slot[order][:-1]
    return ends[order[last]]


@dataclass
class Estimate:
    x: float
    y: float
    heading: float  # rad
    sigma_m: float  # spread of the particles in position
    sigma_deg: float  # ... and in heading

    @property
    def pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.heading])


class ParticleFilter:
    def __init__(self, particles: int = 150, seed: int = 0) -> None:
        self.n = int(particles)
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(self.n)
        self.y = np.zeros(self.n)
        self.th = np.zeros(self.n)
        self.w = np.full(self.n, 1.0 / self.n)
        self.converged = True
        self._weighed_at: np.ndarray | None = None  # the pose estimate when a scan was last weighed
        self.neff = float(self.n)
        self.searched = math.inf  # meters traveled since a search from nowhere began (inf: not searching)
        self.free_cells: np.ndarray | None = None
        self.cell_size = 0.05
        self.origin = (0.0, 0.0)

    # ---- starting ----------------------------------------------------------------------------------------

    def start_at(self, pose: np.ndarray, sigma_m: float = 0.05, sigma_rad: float = 0.03) -> None:
        self.x = pose[0] + self.rng.normal(0.0, sigma_m, self.n)
        self.y = pose[1] + self.rng.normal(0.0, sigma_m, self.n)
        self.th = pose[2] + self.rng.normal(0.0, sigma_rad, self.n)
        self.w[:] = 1.0 / self.n
        self.converged = True
        self._weighed_at = None
        self.searched = math.inf

    def start_anywhere(self, grid: OccupancyGrid, count: int | None = None) -> bool:
        """Spread over the map's free cells, any heading (the robot does not know where it is). False when the map
        has no free cell."""
        free = np.argwhere(grid.state() == FREE)
        if not len(free):
            return False
        if count is not None:
            self.resize(count)
        pick = self.rng.integers(0, len(free), self.n)
        self.y = grid.y0 + free[pick, 0] * grid.res + self.rng.uniform(-0.5, 0.5, self.n) * grid.res
        self.x = grid.x0 + free[pick, 1] * grid.res + self.rng.uniform(-0.5, 0.5, self.n) * grid.res
        self.th = self.rng.uniform(-math.pi, math.pi, self.n)
        self.w[:] = 1.0 / self.n
        self.converged = False
        self._weighed_at = None
        self.searched = 0.0
        self.free_cells, self.cell_size, self.origin = free, grid.res, (grid.x0, grid.y0)
        return True

    def resize(self, count: int) -> None:
        self.n = int(count)
        self.x, self.y, self.th = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)
        self.w = np.full(self.n, 1.0 / self.n)

    # ---- moving and weighing ---------------------------------------------------------------------------------

    def predict(self, forward: float, sideways: float, turn: float) -> None:
        """Move every particle by the odometry's step, in the robot's own frame, each with its own noise."""
        trans = math.hypot(forward, sideways)
        rot = abs(turn)
        if self.searched < SEARCH_M:
            self.searched += trans
        s_trans = MOVE_TRANS * trans + CROSS_TRANS * rot + FLOOR_TRANS
        s_rot = MOVE_ROT * rot + CROSS_ROT * trans + FLOOR_ROT
        f = forward + self.rng.normal(0.0, s_trans, self.n)
        s = sideways + self.rng.normal(0.0, s_trans, self.n)
        c, sn = np.cos(self.th), np.sin(self.th)
        self.x += c * f - sn * s
        self.y += sn * f + c * s
        self.th = wrap(self.th + turn + self.rng.normal(0.0, s_rot, self.n))

    def due(self) -> bool:
        """Whether the robot moved enough since the last scan was weighed to weigh another (a robot standing still
        would otherwise let the filter collapse on one scan's noise)."""
        if self._weighed_at is None:
            return True
        est = self.mean()
        moved = math.hypot(est[0] - self._weighed_at[0], est[1] - self._weighed_at[1])
        turned = abs(wrap(est[2] - self._weighed_at[2]))
        return moved >= MIN_MOVE or turned >= MIN_TURN

    def weigh(self, grid: OccupancyGrid, ends: np.ndarray, mount: np.ndarray) -> bool:
        """Weigh the particles with a scan: `ends` are the scan's obstacle end points in the sensor's own planar
        frame (N x 2), `mount` is the sensor's pose in the particle's frame (x, y, heading). False when the map is
        too bare to match, or no end is usable."""
        if len(ends) < 8 or int(grid.occupied().sum()) < MIN_OCCUPIED:
            return False
        if len(ends) > BEAMS:
            ends = spread(ends, BEAMS)
        field = grid.likelihood_field()
        cm, sm = math.cos(mount[2]), math.sin(mount[2])
        # the ends in the particle's frame, then in the world for each particle
        px = mount[0] + cm * ends[:, 0] - sm * ends[:, 1]
        py = mount[1] + sm * ends[:, 0] + cm * ends[:, 1]
        c, s = np.cos(self.th)[:, None], np.sin(self.th)[:, None]
        wx = self.x[:, None] + c * px[None, :] - s * py[None, :]
        wy = self.y[:, None] + s * px[None, :] + c * py[None, :]
        cols = np.rint((wx - grid.x0) / grid.res).astype(np.int64)
        rows = np.rint((wy - grid.y0) / grid.res).astype(np.int64)
        inside = (rows >= 0) & (rows < grid.rows) & (cols >= 0) & (cols < grid.cols)
        near = np.where(inside, field[np.clip(rows, 0, grid.rows - 1), np.clip(cols, 0, grid.cols - 1)], 0.0)
        log_like = np.log(Z_HIT * near + Z_RANDOM).sum(axis=1)
        log_like -= log_like.max()
        if self.searched < SEARCH_M:
            log_like *= SEARCH_TEMPER + (1.0 - SEARCH_TEMPER) * self.searched / SEARCH_M
        weight = self.w * np.exp(log_like)
        total = weight.sum()
        if not np.isfinite(total) or total <= 0:
            return False
        self.w = weight / total
        self.neff = float(1.0 / np.sum(self.w**2))
        self._weighed_at = self.mean()
        if self.neff < 0.5 * self.n:
            self.resample()
        return True

    def resample(self) -> None:
        """Low-variance resampling: the better guesses are copied, with a little jitter so copies can part."""
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        index = np.minimum(np.searchsorted(np.cumsum(self.w), positions), self.n - 1)
        self.x = self.x[index] + self.rng.normal(0.0, 0.004, self.n)
        self.y = self.y[index] + self.rng.normal(0.0, 0.004, self.n)
        self.th = wrap(self.th[index] + self.rng.normal(0.0, 0.003, self.n))
        self.w[:] = 1.0 / self.n
        if self.searched < SEARCH_M and self.free_cells is not None:  # new guesses, in case it is in none of these
            fresh = self.rng.random(self.n) < SEARCH_INJECT
            count = int(fresh.sum())
            pick = self.rng.integers(0, len(self.free_cells), count)
            self.y[fresh] = self.origin[1] + self.free_cells[pick, 0] * self.cell_size
            self.x[fresh] = self.origin[0] + self.free_cells[pick, 1] * self.cell_size
            self.th[fresh] = self.rng.uniform(-math.pi, math.pi, count)

    # ---- the estimate ----------------------------------------------------------------------------------------

    def mean(self) -> np.ndarray:
        x = float(self.w @ self.x)
        y = float(self.w @ self.y)
        th = math.atan2(float(self.w @ np.sin(self.th)), float(self.w @ np.cos(self.th)))
        return np.array([x, y, th])

    def estimate(self) -> Estimate:
        x, y, th = self.mean()
        sigma = math.sqrt(float(self.w @ ((self.x - x) ** 2 + (self.y - y) ** 2)))
        spread = float(np.abs(wrap(self.th - th)) @ self.w)
        self.converged = sigma < 0.25 and math.degrees(spread) < 15.0 and self.searched >= SEARCH_M
        return Estimate(x, y, th, sigma, math.degrees(spread))
