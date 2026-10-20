"""Handing an object from one arm to another by way of a place both reach.

Two arms that stand apart cannot each reach the other's side of the room: one picks the object up and sets it down at
a spot in reach of both, goes home out of the way, and the other picks it up from there and puts it where it is
wanted. `Relay` runs the second half when the first is done; `hand_off_spots` finds the spots.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .arm import ArmMotion

KEY = "__relay__"  # session controller key of a relay: KEY + the giving arm
GRID = 0.05  # m between the spots tried
SHARE = 0.8  # the spot lies within this share of each arm's reach
NEAR = 0.22  # m: nearer to an arm's base than this, it cannot lean over its own foot
CLEAR = 0.18  # m from any other loose object
ARRIVED = 0.08  # m: the object counts as set down at the spot within this
PARK_SHARE = 0.55  # of the way from the giving arm's base to the spot
PARK_HEIGHT = 0.45  # m, the giving arm's tool while it waits
LIMIT = 90.0  # s the whole hand-over may take


def hand_off_spots(
    data: mujoco.MjData, giver: ArmMotion, receiver: ArmMotion, avoid: list[np.ndarray], limit: int = 12
) -> list[np.ndarray]:
    """Floor points [x, y] both arms reach with the gripper pointing down, the easiest first: where the larger share
    of its reach either one needs is smallest. `avoid` are points (other loose things) to stay `CLEAR` of."""
    bases = []
    for arm in (giver, receiver):
        base, reach = arm.base_and_reach(data)
        bases.append((np.asarray(base, float), float(reach)))
    (g_base, g_reach), (r_base, r_reach) = bases
    lo = np.minimum(g_base[:2], r_base[:2]) - 0.1
    hi = np.maximum(g_base[:2], r_base[:2]) + 0.1
    found = []
    for x in np.arange(lo[0], hi[0] + 1e-9, GRID):
        for y in np.arange(lo[1], hi[1] + 1e-9, GRID):
            point = np.array([x, y])
            shares = []
            for base, reach in bases:
                out = float(np.linalg.norm(point - base[:2]))
                if out < NEAR or out > SHARE * reach:
                    break
                shares.append(out / reach)
            else:
                if all(float(np.linalg.norm(point - np.asarray(a)[:2])) >= CLEAR for a in avoid):
                    found.append((max(shares), point))
    found.sort(key=lambda s: s[0])
    return [p for _, p in found[:limit]]


def parking(base: np.ndarray, spot: np.ndarray) -> np.ndarray:
    """Where the giving arm waits, out of the receiver's way: on the line from its base to the spot, short of it,
    the tool pointing down at PARK_HEIGHT. At home an arm holds its hand over the middle of the room, which is where
    the spot is: the receiver's hand met its forearm."""
    out = float(np.linalg.norm(spot - base[:2]))
    toward = (spot - base[:2]) / max(out, 1e-6)
    near = min(max(PARK_SHARE * out, NEAR + 0.08), out)
    return np.array([base[0] + toward[0] * near, base[1] + toward[1] * near, PARK_HEIGHT])


@dataclass
class Relay:
    """Once the giving arm has set the object down and gone home, set the receiving arm going (`start`: its
    pick-and-place, or a note why there is none); done when that is. Installed as a hidden session controller."""

    giver: ArmMotion
    receiver: str
    obj: str
    spot: np.ndarray
    arrived: Callable[[], bool]
    start: Callable[[], Any]
    rate_hz: float = 20.0
    hidden: bool = True
    after: Any = None
    note: str | None = None
    on_done: Callable[[], None] | None = None  # takes the finished relay out of the session's controllers

    @property
    def time_limit(self) -> float:
        return LIMIT

    @property
    def done(self) -> bool:
        return self.note is not None or (self.after is not None and bool(self.after.done))

    def status(self) -> dict[str, Any]:
        return {
            "behavior": "hand_over",
            "robot": self.giver.robot,
            "step": "giving" if self.after is None else "receiving",
        }

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.done:
            if self.on_done is not None:  # a finished relay does not stay behind among the controllers
                on_done, self.on_done = self.on_done, None
                on_done()
            return
        if self.after is not None or self.note is not None or not self.giver.done:
            return
        if not self.arrived():
            self.note = f"{self.giver.robot} did not set {self.obj} down at the hand-off spot"
            return
        started = self.start()
        if isinstance(started, str):
            self.note = started
        else:
            self.after = started

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "handed_over": {
                "from": self.giver.robot,
                "to": self.receiver,
                "at": [round(float(v), 3) for v in self.spot],
            }
        }
        if self.note:
            out["note"] = self.note
        return out
