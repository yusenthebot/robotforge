"""Short viewer driving holds, independent of simulation time and controller type."""

from __future__ import annotations

import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robotforge.behaviors.manager import BehaviorManager
    from robotforge.sim.session import SimSession


@dataclass(frozen=True)
class HeldDrive:
    robot: str
    release: Callable[[], None]


@dataclass(frozen=True)
class _Hold:
    owner: Any
    deadline: float
    drive: HeldDrive


def hold_seconds(message: dict[str, Any], owner: Any) -> float | None:
    if "hold_ms" not in message:
        return None
    value = message["hold_ms"]
    if message.get("type") != "teleop" or owner is None:
        raise ValueError("hold_ms requires a connected viewer teleop request")
    if type(value) is not int or not 250 <= value <= 1500:
        raise ValueError("hold_ms must be an integer between 250 and 1500")
    return value / 1000


class DriveHolds:
    def __init__(self, session: SimSession) -> None:
        self.session = session
        self._holds: dict[str, _Hold] = {}
        self._bound: BehaviorManager | None = None

    def bind(self, manager: BehaviorManager) -> None:
        if self._bound is manager:
            return
        if self._bound is not None:
            raise ValueError("viewer driving holds are already bound")
        self._bound = manager
        ref = weakref.ref(self)

        def commanded(robot: str, _event: dict[str, Any]) -> None:
            registry = ref()
            # Only quick notifications share their command's lock. A delayed go
            # notification must not revoke a newer hold acquired during planning.
            if registry is not None and _event.get("op") in {"move", "heave", "stop"}:
                registry.revoke(robot)

        manager.command_hooks.append(commanded)
        weakref.finalize(self, manager.command_hooks.remove, commanded)

    def keep(self, owner: Any, drive: HeldDrive, seconds: float) -> None:
        with self.session.lock:
            self._holds[drive.robot] = _Hold(owner, time.perf_counter() + seconds, drive)

    def revoke(self, robot: str) -> None:
        # A newer command owns the robot. Revoke the old hold without stopping it.
        with self.session.lock:
            self._holds.pop(robot, None)

    def forget(self, owner: Any) -> None:
        with self.session.lock:
            for robot, hold in tuple(self._holds.items()):
                if hold.owner is owner:
                    del self._holds[robot]
                    hold.drive.release()

    def expire(self, now: float) -> None:
        if not self._holds:
            return
        with self.session.lock:
            for robot, hold in tuple(self._holds.items()):
                if now >= hold.deadline:
                    del self._holds[robot]
                    hold.drive.release()
