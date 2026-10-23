"""Rollback participants for authored-layout restoration, limited to built-in idle reset hooks.

No arbitrary callback side effects are claimed reversible. Refuse those cases before publication.
The session lock remains held throughout capture, installation and restoration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robotforge.design.layouts import LayoutError
from robotforge.experiments.live import LiveExperiments

if TYPE_CHECKING:
    from robotforge.playground import Playground


def check_supported_restore(pg: Playground) -> None:
    if tuple(pg.session.reset_hooks) != pg._layout_reset_hooks:
        raise LayoutError("runtime", "Stop custom reset hooks before restoring a layout")
    for hook in pg.session.controller_hooks:
        owner = getattr(hook, "__self__", None)
        if type(owner) is not LiveExperiments or getattr(hook, "__func__", None) is not LiveExperiments._changed:
            raise LayoutError("runtime", "Stop custom controller observers before restoring a layout")
        if owner.active:
            raise LayoutError("runtime", "Finish the active experiment before restoring a layout")
    if any(not trip.done for trip in pg.behaviors._live_trips.values()) or pg.behaviors._errands:
        raise LayoutError("runtime", "Stop navigation and errands before restoring a layout")
    if any(r.active for r in pg.demonstrations.recordings.values()) or any(
        not r.done for r in pg.demonstrations.replays.values()
    ):
        raise LayoutError("runtime", "Stop recording and replay before restoring a layout")


# Explicit owned fields; never copy MuJoCo objects, locks, controllers or callback owners.
SESSION = (
    "model",
    "data",
    "source",
    "version",
    "generation",
    "paused",
    "drag",
    "water",
    "_sim_debt",
    "controllers",
    "_next_update",
    "reset_hooks",
    "ctrl_hooks",
    "step_hooks",
    "controller_hooks",
)
SENSORS = ("suite", "_retired", "_suite_version", "uids", "viewer_show", "_scans", "_message")
PERCEPTION = ("mappers", "failed", "_retired", "_sent", "_announced", "_sent_at")
BEHAVIORS = (
    "_standing",
    "_last_trips",
    "_live_trips",
    "_errands",
    "_capability_model",
    "_capability_structures",
    "_mobility_limits",
    "_layer_version",
    "_layer_cache",
)
PLAYGROUND = (
    "_play_world_sha256",
    "_play_world_descriptor",
    "last_doctor",
    "_shapes",
    "_soft",
    "_probes",
    "_task_starts",
)
MISSING = object()


@dataclass(frozen=True)
class _Field:
    owner: Any
    name: str
    value: Any
    contents: Any


@dataclass(frozen=True)
class BuiltinLayoutState:
    fields: tuple[_Field, ...]

    @classmethod
    def capture(cls, pg: Playground) -> BuiltinLayoutState:
        fields = []
        for owner, names in (
            (pg.session, SESSION),
            (pg.sensors, SENSORS),
            (pg.perception, PERCEPTION),
            (pg.behaviors, BEHAVIORS),
            (pg, PLAYGROUND),
            (pg.flow, ("name", "nodes", "status", "notes", "rev")),
        ):
            for name in names:
                value = getattr(owner, name, MISSING)
                contents = value.copy() if isinstance(value, (dict, list, set)) else None
                fields.append(_Field(owner, name, value, contents))
        return cls(tuple(fields))

    def restore(self) -> None:
        for item in self.fields:
            if item.value is MISSING:
                if hasattr(item.owner, item.name):
                    delattr(item.owner, item.name)
                continue
            # Restore bound containers in place: reset_hooks includes pg._task_starts.clear.
            if isinstance(item.value, (dict, set)):
                item.value.clear()
                item.value.update(item.contents)
            elif isinstance(item.value, list):
                item.value[:] = item.contents
            setattr(item.owner, item.name, item.value)
