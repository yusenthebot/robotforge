"""Right of way between robots that go somewhere at the same time.

Each trip is planned on the map as if the robot were alone (other robots standing still are obstacles, see
nav_map.py); a robot that comes into its way later makes it stop and plan round (navigate.py, `_make_way`). That is
fine in the open, but not at a doorway or a crossing, where two robots that both keep going meet in the middle and one
has to back out, if there is room to.

This module is the reservation that avoids the meeting. Every moving robot publishes the path it has left, how fast it
keeps to it and how big it is. Two paths that cross make a conflict zone: the stretch of each path that comes within
the two robots' radii (and a margin) of the other. The robot that gets to the zone first goes through; the other holds
where it is, short of the zone, until the first has left it. Which gets there first is read off the times to the zone
(the path to it over the speed), a tie goes to the trip that began first, then to the name that sorts first, so two
robots never wait for each other.

Paths that run side by side or head on are not crossings and are left to making way (a hold there would be a long
wait where a step aside costs nothing); a path that ends inside the zone (the other robot is going to park there) is
left to it too: waiting for it would be for ever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MARGIN = 0.15  # m kept between two robots passing through a crossing, beyond their radii
HORIZON = 4.0  # m of its path that a robot looks at for others to cross
SAMPLE = 0.05  # m between the points a path is checked at
CROSSING = math.cos(math.radians(35.0))  # paths whose directions are closer than this to parallel are not crossings
TIE = 0.25  # s: times to the zone closer than this are a tie
BRAKE = 0.2  # m short of the zone's start that a robot holds at, plus BRAKE_TIME of travel
BRAKE_TIME = 0.4  # s


@dataclass(frozen=True)
class Mover:
    """A robot on its way: what the others need to know to give way to it, or to expect it to."""

    name: str
    path: np.ndarray  # (n, 2) the path it has left, from where it is
    speed: float  # m/s it keeps to it
    radius: float  # m from its middle that its body reaches
    started: float  # sim time its trip began
    holding: bool = False  # waiting for another right now: it will not be at the zone when its time says

    @property
    def length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))) if len(self.path) > 1 else 0.0


@dataclass(frozen=True)
class Conflict:
    """Where one path comes near another: metres along each path to the start and the end of the zone."""

    other: str
    enters: float
    leaves: float
    other_enters: float
    other_leaves: float
    other_ends_inside: bool  # the other path ends in the zone: that robot parks there


def _walk(path: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every SAMPLE m along `path` up to `limit` m, and the distance along it of each."""
    if len(path) < 2:
        return path[:1].copy(), np.zeros(1)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = min(float(seg.sum()), limit)
    at = np.arange(0.0, total + 1e-9, SAMPLE)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    x = np.interp(at, cum, path[:, 0])
    y = np.interp(at, cum, path[:, 1])
    return np.column_stack([x, y]), at


def conflict(mine: Mover, other: Mover, margin: float = MARGIN, horizon: float = HORIZON) -> Conflict | None:
    """The first stretch of `mine` (within `horizon` m) that crosses the path of `other`, or None when they do not
    cross there: far apart, side by side, one behind the other or head on."""
    if len(mine.path) < 2 or len(other.path) < 1:
        return None
    safe = mine.radius + other.radius + margin
    a, along_a = _walk(mine.path, horizon)
    b, along_b = _walk(other.path, math.inf)
    gaps = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    near = gaps.min(axis=1) < safe
    if not near.any():
        return None
    first = int(np.argmax(near))
    last = first
    while last + 1 < len(near) and near[last + 1]:
        last += 1
    zone = gaps[first : last + 1] < safe
    theirs = np.flatnonzero(zone.any(axis=0))
    mid = (first + last) // 2
    j = int(np.argmin(gaps[mid]))
    if not _crossing(a, mid, b, j):
        return None
    ends_inside = bool(along_b[theirs.max()] >= along_b[-1] - SAMPLE) if len(theirs) else False
    return Conflict(
        other.name,
        float(along_a[first]),
        float(along_a[last]),
        float(along_b[theirs.min()]),
        float(along_b[theirs.max()]),
        ends_inside,
    )


def _crossing(a: np.ndarray, i: int, b: np.ndarray, j: int) -> bool:
    """Whether the two paths cross at their nearest points, rather than run along each other."""
    ta = a[min(i + 2, len(a) - 1)] - a[max(i - 2, 0)]
    tb = b[min(j + 2, len(b) - 1)] - b[max(j - 2, 0)]
    na, nb = float(np.linalg.norm(ta)), float(np.linalg.norm(tb))
    if na < 1e-9 or nb < 1e-9:
        return True  # one of them stands there: a standing robot is a crossing for whoever comes
    return abs(float(ta @ tb) / (na * nb)) < CROSSING


def must_wait(mine: Mover, other: Mover, c: Conflict) -> bool:
    """Whether `mine` holds short of the zone `c` for `other`: it does when the other is in the zone or gets to it
    sooner, a tie going to the trip that began first. A robot that is itself holding is not waited for (it will not
    be there), which keeps two from waiting for each other."""
    if c.other_ends_inside or c.other_leaves <= 0.0 or other.holding:
        return False
    if c.other_enters <= 0.0:
        return True  # in the zone now
    mine_at = c.enters / max(mine.speed, 1e-3)
    other_at = c.other_enters / max(other.speed, 1e-3)
    if abs(mine_at - other_at) > TIE:
        return other_at < mine_at
    return (other.started, other.name) < (mine.started, mine.name)


def holds_at(mine: Mover, c: Conflict) -> bool:
    """Whether `mine` is at the zone's edge (inside the distance it brakes in): the moment to stop."""
    return c.enters <= BRAKE + BRAKE_TIME * mine.speed
