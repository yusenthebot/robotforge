"""A minimal Box space that looks like gymnasium.spaces.Box to code that only reads it.

gymnasium is not a dependency, so RobotEnv describes its observations and actions
with this class. It has the attributes RL libraries read (low, high, shape, dtype)
and the methods they call (sample, contains, seed), so converting it to a real
gymnasium Box is `gymnasium.spaces.Box(box.low, box.high, dtype=box.dtype)`.
"""

from __future__ import annotations

import numpy as np


class Box:
    """A box in R^n: each coordinate between low and high (which may be infinite)."""

    def __init__(self, low, high, shape: tuple[int, ...] | None = None, dtype=np.float32, seed: int | None = None):
        self.dtype = np.dtype(dtype)
        if shape is None:
            shape = np.broadcast(np.asarray(low), np.asarray(high)).shape
        self.shape = tuple(int(s) for s in shape)
        self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()
        if np.any(self.low > self.high):
            raise ValueError("Box: every low must be <= high")
        self._rng = np.random.default_rng(seed)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._rng = np.random.default_rng(seed)
        return [seed]

    def sample(self) -> np.ndarray:
        """Uniform inside bounded coordinates, normal where a side is unbounded (like gymnasium)."""
        low, high = self.low.astype(np.float64), self.high.astype(np.float64)
        bounded_lo, bounded_hi = np.isfinite(low), np.isfinite(high)
        out = self._rng.normal(size=self.shape)
        both = bounded_lo & bounded_hi
        out[both] = self._rng.uniform(low[both], high[both])
        only_lo = bounded_lo & ~bounded_hi
        out[only_lo] = low[only_lo] + self._rng.exponential(size=int(only_lo.sum()))
        only_hi = ~bounded_lo & bounded_hi
        out[only_hi] = high[only_hi] - self._rng.exponential(size=int(only_hi.sum()))
        return out.astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        if x.shape != self.shape or not np.can_cast(x.dtype, self.dtype, casting="same_kind"):
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __contains__(self, x) -> bool:
        return self.contains(x)

    def __repr__(self) -> str:
        return f"Box({self.low.min():.3g}, {self.high.max():.3g}, {self.shape}, {self.dtype})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and self.dtype == other.dtype
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    __hash__ = None  # mutable arrays inside; like numpy arrays, not hashable
