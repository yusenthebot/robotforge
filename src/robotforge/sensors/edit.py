"""Adding and removing sensors as world edits: saved on the robot's world.yaml entry, the world rebuilt.

A sensor edit changes what a robot measures, not where it is, so the robot carries on from where it stands
(Playground._edit compares robots without their sensors). The name is fixed when the sensor is added, so a
later edit that adds another sensor of the same type never renames this one.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from robotforge import wording
from robotforge.sensors.spec import SensorEntry, base_name, unique_name


def add(pg: Any, robot: str, fields: dict[str, Any]) -> dict[str, Any]:
    from robotforge.playground import PlaygroundError, _find

    world = pg._require_store().load()
    entry_robot = _find(world.robots, robot, "robot")
    try:
        entry = SensorEntry.model_validate(fields)
    except ValidationError as exc:
        raise PlaygroundError(wording.validation_message(exc)) from exc
    name = entry_robot.name
    taken = set(pg.sensors.names(name)) | {s.name for s in entry_robot.sensors if s.name}
    taken |= _cameras(pg, name)
    if entry.name is None:
        on = entry.between[1] if entry.between else entry.on
        label = on.rsplit("/", 1)[-1] if on else None
        entry = entry.model_copy(update={"name": unique_name(base_name(entry.type, entry.mode, label), taken)})
    elif entry.name in taken:
        raise PlaygroundError(
            f"{name} already has a sensor or camera named {entry.name!r}: give another name, or remove "
            f"{name}/{entry.name} first"
        )

    def change(w: Any) -> None:
        target = _find(w.robots, name, "robot")
        target.sensors = [*target.sensors, entry]

    result = pg._edit(change)
    out: dict[str, Any] = {"sensor": f"{name}/{entry.name}", "type": entry.type}
    sensor = pg.sensors.suite.sensors.get(f"{name}/{entry.name}")
    if sensor is not None:
        out["on"] = sensor.decl.data.get("body")
        out["position_m"] = _where(pg, sensor)
        try:
            reading, _ = pg.sensors.read(name, [entry.name])
            out["reading"] = reading["sensors"].get(entry.name)
        except Exception as exc:  # the sensor is added either way; say why there is no first reading
            out["reading"] = {"note": f"no first reading: {type(exc).__name__}: {exc}"}
    out["world"] = result.get("world")
    return out


def _cameras(pg: Any, robot: str) -> set[str]:
    """The robot's own camera names: a camera sensor must not take one."""
    import mujoco

    model = pg.session.model
    prefix = f"{robot}/"
    names = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c) or "" for c in range(model.ncam))
    return {n[len(prefix) :] for n in names if n.startswith(prefix)}


def _where(pg: Any, sensor: Any) -> list[float]:
    data = pg.session.data
    if sensor.renders:
        return [round(float(v), 3) for v in data.cam_xpos[sensor.camera]]
    if sensor.site >= 0:
        return [round(float(v), 3) for v in data.site_xpos[sensor.site]]
    return [round(float(v), 3) for v in data.xpos[sensor.body]]


def find(world: Any, name: str) -> tuple[str, str] | None:
    """The (robot, sensor) that `name` means for remove: "go2/lidar", or a sensor name only one robot has that is
    not also a robot, object or terrain patch. None when it means none."""
    things = [*(r.name for r in world.robots), *(o.name for o in world.objects), *(t.name for t in world.terrain)]
    if wording.match(name, things, plurals=False) is not None:
        return None
    if "/" in name:
        robot, sensor = name.split("/", 1)
        found = wording.match(robot, [r.name for r in world.robots], plurals=False)
        return (found, sensor) if found is not None else None
    owners = [(r.name, s.name) for r in world.robots for s in r.sensors if s.name]
    hits = [pair for pair in owners if pair[1] == wording.squash(name)]
    return hits[0] if len(hits) == 1 else None


def remove(pg: Any, robot: str, sensor: str) -> dict[str, Any]:
    from robotforge.playground import PlaygroundError, _find

    world = pg._require_store().load()
    entry = _find(world.robots, robot, "robot")
    names = [s.name for s in entry.sensors if s.name]
    found = wording.match(sensor, names, plurals=False)
    if found is None:
        built_in = set(pg.sensors.names(entry.name)) - set(names)
        if wording.match(sensor, sorted(built_in), plurals=False):
            raise PlaygroundError(
                f"{entry.name}/{sensor} comes with the robot itself (its template or its parts), not from add_sensor; "
                "a built robot's part sensors change with update_part"
            )
        raise PlaygroundError(
            wording.unknown("sensor", sensor, names, where=f"added to {entry.name}", make="add_sensor")
        )

    def change(w: Any) -> None:
        target = _find(w.robots, entry.name, "robot")
        target.sensors = [s for s in target.sensors if s.name != found]

    result = pg._edit(change)
    return {"removed": f"{entry.name}/{found}", "world": result.get("world"), "robots": result.get("robots")}
