"""Plugin sensors: a new type for add_sensor, measured by the plugin's own function."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from robotforge.extend.registry import REGISTRY, Item, PluginError

Look = None  # replaced below: imported lazily, the sensor runtime pulls in mujoco


def look(shape: str = "cylinder", size: Sequence[float] = (0.03, 0.012), color: Any = "orange") -> Any:
    """What the sensor looks like on the robot: a small module drawn where it sits. shape is "cylinder" (size
    [diameter, height]), "box" ([length, width, height]) or "sphere" ([diameter]); it weighs nothing and nothing
    collides with it."""
    from robotforge.sensors.plugin import Look

    counts = {"cylinder": 2, "box": 3, "sphere": 1}
    if shape not in counts:
        raise PluginError(f"look shape {shape!r} must be cylinder, box or sphere")
    if len(size) != counts[shape] or any(s <= 0 for s in size):
        raise PluginError(f"look: a {shape} takes {counts[shape]} positive sizes in meters, got {list(size)!r}")
    return Look(shape, tuple(float(s) for s in size), color)


def sensor(
    name: str,
    *,
    summary: str = "",
    channels: Sequence[str] = ("value",),
    params: dict[str, Any] | None = None,
    at: str = "center",
    appearance: Any = None,
) -> Callable[[Any], Any]:
    """Register a new sensor type, `add_sensor(robot, type=name)`.

    The decorated function `read(ctx)` returns a dict of what the sensor measures (numbers, lists of numbers, text),
    e.g. {"count": 3, "nearest_m": 0.8}; ctx is a robotforge.sensors.plugin.SensorContext (its position and
    orientation, `ctx.params`, `ctx.position(name)`, `ctx.distance(name)`, `ctx.ray(direction)`, `ctx.rng`, `ctx.data`,
    `ctx.model`). A class works too: `__init__(self, ctx)` runs once, `read(self, ctx)` at every reading, so a sensor
    can keep what it has seen (a counter, a filter); an edit of the world starts it again.

    channels: the names in the reading that noise applies to (`noise=0.1` in add_sensor's options adds that much
    Gaussian noise to each); rate_hz and latency_s work as for any sensor.
    params: the options add_sensor takes for it, with their defaults, e.g. {"source": "cube", "radius": 2.0}; an
    option not listed is refused. at: where it sits on its body by default (top, front, center, ...).
    appearance: look(...), the module drawn on the robot (default a small orange puck).
    """

    def register(obj: Any) -> Any:
        from robotforge.sensors import plugin as installed
        from robotforge.sensors.spec import ALIASES, BUILT_IN_TYPES

        if name in BUILT_IN_TYPES or name in ALIASES:
            raise PluginError(f"sensor {name!r}: robotforge already has a sensor type of that name; pick another")
        taken = REGISTRY.get("sensor", name)
        if taken is not None:
            raise PluginError(f"sensor {name!r} is already added by {taken.plugin}; pick another name")
        if inspect.isclass(obj):
            if not callable(getattr(obj, "read", None)):
                raise PluginError(f"sensor {name!r}: the class needs a read(self, ctx) method")
        elif not callable(obj):
            raise PluginError(f"sensor {name!r}: decorate a function read(ctx) or a class with read(self, ctx)")
        if not channels or not all(isinstance(c, str) and c.isidentifier() for c in channels):
            raise PluginError(f"sensor {name!r}: channels are names like ('count', 'nearest_m')")
        from robotforge.sensors.spec import ANCHORS

        if at not in ANCHORS:
            raise PluginError(f"sensor {name!r}: at is one of {', '.join(ANCHORS)}, got {at!r}")
        shown = appearance if appearance is not None else installed.Look()
        defaults = dict(params or {})
        undo = installed.install(name, channels, at, defaults, shown)
        item = Item(
            "sensor",
            name,
            summary or f"a {name} sensor (from a plugin)",
            obj,
            meta={
                "params": defaults,
                "describe": {"channels": list(channels), "options": defaults, "at": at},
            },
            undo=undo,
        )
        try:
            REGISTRY.add(item)
        except PluginError:
            undo()
            raise
        return obj

    return register
