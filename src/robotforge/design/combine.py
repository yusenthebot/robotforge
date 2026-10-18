"""Mount anything on anything, in one call: a robot, an end effector or a sensor on any body of a robot in the world.

    mount("ur5e", on="rover")                 an arm on top of a wheeled base, on an adapter plate
    mount("leap_hand", on="rover/arm")        a dexterous hand on that arm's flange
    mount("suction", on="rover/arm")          ... swapped for a suction cup (a quick change: the hand comes off)
    mount("lidar", on="rover/arm")            a lidar on the arm's wrist
    mount(on="rover", save_as="leap_rover")   the whole combination kept as a design, to add or mount again

What goes on is worked out from the word (see classify): a sensor type goes through add_sensor, the end effectors
(the library's grippers and hands, our gripper, suction cup and magnet) onto an arm's bare flange, any other library
robot or template on top or on the body named, and a saved design whole. Robots go on as world.yaml mounts (see
mounting.py), on an adapter plate unless told not to, and every mount is checked once it is on (mount_checks.py): a
robot that would not stand, or an arm that could not hold itself up, is refused and the world stays as it was.

Saved designs live in $ROBOTFORGE_HOME/designs/<name>.yaml: what the robot is built from and what it carries, without
where it stood.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from robotforge import wording
from robotforge.design import end_effectors, library, mounting, templates
from robotforge.design.templates.base import BuiltRobot
from robotforge.sensors.spec import ALIASES, SENSOR_TYPES

# Words for end effectors that are not a library or template name: (template or library name, what it is).
TOOL_WORDS = {
    "suction": "suction_cup",
    "suction_cup": "suction_cup",
    "vacuum": "suction_cup",
    "cup": "suction_cup",
    "magnet": "magnet",
    "electromagnet": "magnet",
    "gripper": "gripper",
    "parallel_gripper": "gripper",
    "two_finger_gripper": "gripper",
    "claw": "gripper",
}


def designs_home() -> Path:
    """Where saved designs live: $ROBOTFORGE_HOME/designs, default ~/robotforge/designs."""
    root = Path(os.environ.get("ROBOTFORGE_HOME", Path.home() / "robotforge")).expanduser()
    return root / "designs"


# ---- what goes on ------------------------------------------------------------------------------------------------


def classify(what: str) -> tuple[str, dict[str, Any]]:
    """("sensor", {"type": ...}), ("tool", mount fields), ("robot", mount fields) or ("design", mount fields) for a
    word: a sensor type, an end effector, a robot from the library or the templates, or a saved design. A ValueError
    names the close ones for any other."""
    word = wording.squash(what).replace("-", "_")
    if word in SENSOR_TYPES or word in ALIASES:
        return "sensor", {"type": word}
    if word in TOOL_WORDS:
        return "tool", {"template": TOOL_WORDS[word]}
    found = library.canonical(what)
    if found in library.CATALOG:
        kind = library.get(found).kind
        return ("tool" if kind == "gripper" else "robot"), {"library": found}
    if templates.canonical(what) in templates.TEMPLATES:
        name = templates.canonical(what)
        return ("tool" if name in mounting.TOOL_TEMPLATES else "robot"), {"template": name}
    saved = saved_design(word)
    if saved is not None:
        return "design", {"design": word}
    names = [*SENSOR_TYPES, *TOOL_WORDS, *library.CATALOG, *templates.TEMPLATES, *designs()]
    close = wording.near(what, names)
    hint = f"did you mean {', '.join(close)}? " if close else ""
    raise ValueError(
        f"nothing called {wording.short(what)} to mount; {hint}mount takes a robot from the library or the templates "
        f"(list_library, list_templates), an end effector (a gripper or hand from the library, gripper, suction, "
        f"magnet), a sensor type ({', '.join(SENSOR_TYPES)}) or a design saved with save_as"
    )


def split_on(on: str, robots: list[str]) -> tuple[str, str | None]:
    """The robot and the mount of it that `on` names: "rover" is (rover, None), "rover/arm" (rover, arm)."""
    from robotforge.playground import PlaygroundError

    head, _, rest = on.partition("/")
    found = wording.match(head, robots)
    if found is None:
        raise PlaygroundError(wording.unknown("robot", head, robots, make="add_robot"))
    return found, (wording.squash(rest) if rest else None)


# ---- the mount call -------------------------------------------------------------------------------------------------


def mount(
    pg: Any,
    what: str | None,
    on: str,
    at: str | None = None,
    shift: list[float] | None = None,
    rotate: list[float] | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    adapter: bool = True,
    save_as: str | None = None,
) -> dict[str, Any]:
    """See the module notes and the mount tool."""
    from robotforge.playground import PlaygroundError, _find

    store = pg._require_store()
    world = store.load()
    robot, part = split_on(on, [r.name for r in world.robots])
    if what is None:
        if save_as is None:
            raise PlaygroundError("give what to mount, or save_as to keep the robot as it is as a design")
        return {"saved": save(_find(world.robots, robot, "robot"), save_as, store.dir)}
    entry = _find(world.robots, robot, "robot")
    if part is not None and part not in [mounting.default_name(m) for m in entry.mounts]:
        mounted = [f"{robot}/{mounting.default_name(m)}" for m in entry.mounts]
        raise PlaygroundError(
            f"{robot} carries no {part!r}"
            + (f"; it carries {', '.join(mounted)}" if mounted else "; it carries nothing yet")
            + f'; mount on "{robot}" itself or on one of those'
        )
    try:
        sort, fields = classify(what)
    except ValueError as exc:
        raise PlaygroundError(str(exc)) from None
    if sort == "sensor":
        result = _sensor(pg, entry, part, at, fields, name, params)
    else:
        result = _robot_mount(pg, entry, part, sort, fields, at, shift, rotate, name, params, adapter)
    if save_as is not None:
        result["saved"] = save(_find(store.load().robots, robot, "robot"), save_as, store.dir)
    return result


def _sensor(
    pg: Any,
    entry: Any,
    part: str | None,
    at: str | None,
    fields: dict[str, Any],
    name: str | None,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """A sensor through add_sensor, on the body `at` names (of the mount `part`, if given), by default the wrist of
    that mount's arm or the robot's base."""
    body = None
    if part is not None:
        body = f"{part}/{at}" if at not in (None, "tool") else _wrist(pg, f"{entry.name}/{part}")
    elif at is not None:
        body = _wrist(pg, entry.name) if at == "tool" else at
    sensor = {**fields, **(params or {})}
    if body is not None:
        sensor["on"] = body
    if name is not None:
        sensor["name"] = name
    out = pg.add_sensor(entry.name, sensor)
    out["mounted"] = out.get("sensor")
    return out


def _wrist(pg: Any, arm: str) -> str:
    """The body an arm's tool point turns with, the last of its own links ("arm/wrist_3_link"), without the robot's
    name: where a sensor on an arm's wrist goes."""
    import mujoco

    from robotforge.playground import PlaygroundError

    model = pg.session.model
    robot = arm.split("/", 1)[0]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/tcp")
    if site < 0:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/flange")
    if site < 0:
        raise PlaygroundError(f"{arm} is not an arm (it has no tool point): give at, the body to put the sensor on")
    body = int(model.site_bodyid[site])
    prefix = f"{arm}/"
    while body > 0:  # up to the last link that has a joint of the arm's own (past its gripper's bodies)
        joints = range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body])
        if any(model.joint(j).name.startswith(prefix) for j in joints):
            break
        body = int(model.body_parentid[body])
    return model.body(body).name.removeprefix(f"{robot}/")


def _robot_mount(
    pg: Any,
    entry: Any,
    part: str | None,
    sort: str,
    fields: dict[str, Any],
    at: str | None,
    shift: list[float] | None,
    rotate: list[float] | None,
    name: str | None,
    params: dict[str, Any] | None,
    adapter: bool,
) -> dict[str, Any]:
    """A robot, an end effector or a saved design as a mount of `entry`, checked before the world keeps it."""
    from robotforge.design.world import MountEntry
    from robotforge.playground import PlaygroundError, _find, _validation_message

    robot = entry.name
    notes: list[str] = []
    if sort == "tool":
        place = at if at is not None else ("tool" if part is None else f"{part}/tool")
        if at is not None and part is not None and at != "tool":
            place = f"{part}/{at}"
        if place == "tool" and part is not None:
            place = f"{part}/tool"
    else:
        place = at if part is None else f"{part}/{at or 'top'}"
        if part is not None and at is None:
            raise PlaygroundError(
                f"{what_words(fields)} on {robot}/{part}: give at, the body of {part} to mount it on (or mount it on "
                f'"{robot}" itself, on top)'
            )
    mounts = list(entry.mounts)
    swapped = None
    if sort == "tool":
        old = _on_flange(mounts, place)
        if old is not None:  # a quick change: the new end effector takes the old one's place and name
            swapped = mounts.pop(old)
            name = name or swapped.name
            notes.append(f"swapped {mounting.what(swapped)} for {what_words(fields)} on {_flange_words(robot, place)}")
        elif _native_fingers(entry, part) and fields.get("template") in ("suction_cup", "magnet"):
            return _fit_tool(pg, entry, fields["template"])
    elif sort == "robot" and fields.get("template") is None and params:
        raise PlaygroundError("library robots take no params; params are for templates and sensors")
    data = {**fields, "at": place, "adapter": adapter}
    if name is not None:
        data["name"] = name
    if shift is not None:
        data["shift"] = shift
    if rotate is not None:
        data["rotate"] = rotate
    if params and fields.get("template") is not None:
        data["params"] = params
    try:
        new = MountEntry.model_validate(data)
    except ValidationError as exc:
        raise PlaygroundError(_validation_message(exc)) from exc
    taken = [mounting.default_name(m) for m in mounts]
    if mounting.default_name(new) in taken:
        if name is not None:
            raise PlaygroundError(f"{robot} already carries a mount called {name!r}: give another name")
        new = new.model_copy(update={"name": _free(mounting.default_name(new), taken)})
    mounts.append(new)
    if entry.end_effector in end_effectors.KINDS and sort == "tool" and part is None:
        notes.append(f"{robot}'s {entry.end_effector} comes off for it")
    checks = _checked(pg, robot, mounts, clear_tool=sort == "tool" and part is None)

    def change(w: Any) -> None:
        target = _find(w.robots, robot, "robot")
        target.mounts = mounts
        if sort == "tool" and part is None and target.end_effector in end_effectors.KINDS:
            target.end_effector = None

    result = pg._edit(change)
    result["mounted"] = f"{robot}/{mounting.default_name(new)}"
    if swapped is not None:
        result["swapped"] = f"{robot}/{mounting.default_name(swapped)}"
    result["checks"] = checks
    result["notes"] = [*notes, *result.get("notes", [])]
    return result


def what_words(fields: dict[str, Any]) -> str:
    if fields.get("library"):
        return library.get(fields["library"]).title
    if fields.get("design"):
        return f"the {fields['design']} design"
    return {"suction_cup": "a suction cup", "magnet": "an electromagnet", "gripper": "a gripper"}.get(
        fields["template"], f"the {fields['template']} template"
    )


def _flange_words(robot: str, place: str) -> str:
    return f"{robot}'s flange" if place == "tool" else f"{robot}/{place.removesuffix('/tool')}'s flange"


def _on_flange(mounts: list[Any], place: str) -> int | None:
    """The index of the end effector already on the flange `place` names ("tool", "arm/tool"), or None. A gripper
    mounted without at went on the only flange there was."""
    arms = [mounting.default_name(m) for m in mounts if not mounting.is_gripper(m)]
    for i, m in enumerate(mounts):
        if not mounting.is_gripper(m):
            continue
        there = m.at or "tool"
        if there == place or (there == "tool" and place.endswith("/tool") and len(arms) == 1):
            return i
        if place == "tool" and there.endswith("/tool") and len(arms) == 1:
            return i
    return None


def _native_fingers(entry: Any, part: str | None) -> bool:
    """Whether the arm `part` of `entry` (or `entry` itself) is a library arm with fingers of its own (a Panda, a
    PiPER): a cup or a magnet goes in front of them (end_effector) instead of on a flange."""
    source = next((m for m in entry.mounts if mounting.default_name(m) == part), None) if part else entry
    lib = getattr(source, "library", None)
    if lib is None:
        return part is None and entry.template == "arm"
    robot = library.get(lib)
    return robot.kind == "arm" and any(t.gripper is not None for t in robot.tools)


def _fit_tool(pg: Any, entry: Any, template: str) -> dict[str, Any]:
    kind = "suction" if template == "suction_cup" else "magnet"
    result = pg.update_robot(entry.name, end_effector=kind)
    result["mounted"] = f"{entry.name}: {end_effectors.TOOLS[kind].title}"
    result["checks"] = _report(pg, entry.name)
    return result


def _free(name: str, taken: list[str]) -> str:
    k = 2
    while f"{name}{k}" in taken:
        k += 1
    return f"{name}{k}"


def _checked(pg: Any, robot: str, mounts: list[Any], clear_tool: bool) -> dict[str, Any]:
    """The checks of the open world with `robot` carrying `mounts`, compiled but not kept; a PlaygroundError saying
    what to change for a robot that would not stand or hold its arm up."""
    from robotforge.design.world import World, compile_world
    from robotforge.playground import PlaygroundError

    store = pg._require_store()
    world = store.load().model_copy(deep=True)
    target = next(r for r in world.robots if r.name == robot)
    target.mounts = mounts
    if clear_tool and target.end_effector in end_effectors.KINDS:
        target.end_effector = None
    try:
        model = compile_world(World.model_validate(world.model_dump()), store.dir).model
    except (ValueError, ValidationError) as exc:
        text = str(exc) if not isinstance(exc, ValidationError) else wording.validation_message(exc)
        raise PlaygroundError(text) from None
    found = _measure(model, robot, mounts)
    if found.refusal:
        raise PlaygroundError(f"{found.refusal}. Nothing was mounted. Checks: {_short(found.report)}")
    return found.report


def _report(pg: Any, robot: str) -> dict[str, Any]:
    from robotforge.playground import _find

    entry = _find(pg._require_store().load().robots, robot, "robot")
    return _measure(pg.session.model, robot, entry.mounts).report


def _measure(model: Any, robot: str, mounts: list[Any]) -> Any:
    from robotforge.behaviors import robot_kind
    from robotforge.design import mount_checks

    carried = {}
    for m in mounts:
        mount_name = mounting.default_name(m)
        carried[mount_name] = mounting.carried_mass(model, robot, [m])
    return mount_checks.check(model, robot, robot_kind(model, robot).value, carried)


def _short(report: dict[str, Any]) -> str:
    import json

    return json.dumps({k: v for k, v in report.items() if k in ("mass_kg", "stands", "arms")})


# ---- taking one off -------------------------------------------------------------------------------------------------


def find_mount(world: Any, name: str) -> tuple[str, str] | None:
    """The (robot, mount) that `name` means for remove: "rover/arm"; None when it names no mount."""
    if "/" not in name:
        return None
    head, rest = name.split("/", 1)
    robot = wording.match(head, [r.name for r in world.robots], plurals=False)
    if robot is None:
        return None
    entry = next(r for r in world.robots if r.name == robot)
    names = [mounting.default_name(m) for m in entry.mounts]
    found = wording.match(rest, names, plurals=False) if "/" not in rest else None
    return (robot, found) if found is not None else None


def take_off(pg: Any, robot: str, name: str) -> dict[str, Any]:
    """Take mount `name` off `robot`, with what was mounted on it ("<name>/tool", "<name>/<body>") and the sensors
    put on its bodies."""
    from robotforge.playground import _find

    gone: list[str] = []

    def change(w: Any) -> None:
        target = _find(w.robots, robot, "robot")
        kept = []
        for m in target.mounts:
            mine = mounting.default_name(m) == name or (m.at or "").startswith(f"{name}/")
            if mine:
                gone.append(f"{robot}/{mounting.default_name(m)}")
            else:
                kept.append(m)
        target.mounts = kept
        sensors = [s for s in target.sensors if (s.on or "").startswith(f"{name}/")]
        gone.extend(f"{robot}/{s.name}" for s in sensors)
        target.sensors = [s for s in target.sensors if s not in sensors]

    result = pg._edit(change)
    return {
        "removed": gone,
        "world": result.get("world"),
        "robots": result.get("robots"),
        "notes": result.get("notes", []),
    }


def libraries(name: str) -> list[str]:
    """The library robots a saved design is made of (to download before it is built), its own designs' too."""
    fields = saved_design(name) or {}
    out = [fields["library"]] if fields.get("library") else []
    for m in fields.get("mounts", []):
        if m.get("library"):
            out.append(m["library"])
        elif m.get("design") and m["design"] != name:
            out += libraries(m["design"])
    return out


# ---- saved designs -----------------------------------------------------------------------------------------------


def designs() -> list[str]:
    home = designs_home()
    return sorted(p.stem for p in home.glob("*.yaml")) if home.is_dir() else []


def saved_design(name: str) -> dict[str, Any] | None:
    """A saved design's robot fields (what world.yaml keeps for a robot, without name and place), or None."""
    path = designs_home() / f"{name}.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    robot = data.get("robot") if isinstance(data, dict) else None
    if not isinstance(robot, dict):
        raise ValueError(f"the saved design {name!r} ({path}) is not valid: it has no robot")
    return robot


def save(entry: Any, name: str, world_dir: Path) -> dict[str, Any]:
    """Keep `entry` (a world.yaml robot) as the design `name`: what it is built from and what it carries."""
    from robotforge.design.world import NAME_RE
    from robotforge.playground import PlaygroundError

    tidy = wording.squash(name)
    if not NAME_RE.match(tidy):
        raise PlaygroundError(
            f"save_as {wording.short(name)}: a design's name uses lowercase letters, digits and _ and starts with a "
            "letter, e.g. 'leap_rover'"
        )
    if tidy in library.CATALOG or tidy in templates.TEMPLATES or tidy in TOOL_WORDS or tidy in SENSOR_TYPES:
        raise PlaygroundError(f"save_as {tidy!r} is taken by a robot of the library or a template: pick another name")
    from robotforge.design import gallery

    if gallery.canonical(tidy) == tidy:
        raise PlaygroundError(f"save_as {tidy!r} is a design of the gallery: pick another name")
    robot = entry.model_dump(mode="json", exclude={"name", "pos", "yaw"}, exclude_defaults=True)
    if robot.get("file"):  # a file next to the world: where it is, so the design opens from any world
        path = Path(robot["file"]).expanduser()
        robot["file"] = str(path if path.is_absolute() else (world_dir / path).resolve())
    home = designs_home()
    home.mkdir(parents=True, exist_ok=True)
    path = home / f"{tidy}.yaml"
    text = yaml.safe_dump({"design": tidy, "robot": robot}, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")
    return {"design": tidy, "file": str(path), "use": f'add_robot(name, design="{tidy}") or mount("{tidy}", on=...)'}


def entry_fields(name: str) -> dict[str, Any]:
    """The robot fields of a saved design, for add_robot(design=...)."""
    found = saved_design(name)
    if found is None:
        raise ValueError(f"no saved design {name!r}; saved designs: {', '.join(designs()) or 'none yet'}")
    return dict(found)


def build_design(name: str) -> BuiltRobot:
    """A saved design built as a world builds a robot: from its source, with its end effector, mounts and sensors."""
    from robotforge.design.world import RobotEntry, _robot_spec
    from robotforge.sensors.mount import mount as put_sensors

    fields = entry_fields(name)
    try:
        entry = RobotEntry.model_validate({"name": "design", **fields})
    except ValidationError as exc:
        raise ValueError(f"the saved design {name!r} is not valid: {wording.validation_message(exc)}") from None
    built = _robot_spec(entry, None)
    if entry.sensors:
        put_sensors(built.spec, entry.sensors, entry.name, built.home_qpos)
    return built


__all__ = ["build_design", "classify", "designs", "designs_home", "entry_fields", "mount", "save", "saved_design"]
