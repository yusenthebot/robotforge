"""Edits to a built robot's description, part by part, on the compact dicts world.yaml keeps.

These only change the description; the playground validates and compiles the whole world
afterwards and saves nothing if that fails, so a bad edit never breaks a robot.
"""

from __future__ import annotations

import copy
from typing import Any

from robotforge import wording

ROBOT_SETTINGS = ("color", "material", "servo", "motor", "linear", "servo_caps")


def part_names(build: dict[str, Any]) -> list[str]:
    return [p.get("name") for p in build.get("parts", []) if isinstance(p, dict)]


def add_part(build: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(part, dict) or not isinstance(part.get("name"), str):
        raise ValueError('a part is a mapping with at least a name, e.g. {"name": "head", "at": "top", ...}')
    if part["name"] in part_names(build):
        raise ValueError(f"there is already a part named {part['name']!r}; use update_part to change it")
    out = copy.deepcopy(build)
    out.setdefault("parts", []).append(copy.deepcopy(part))
    return out


def update_part(build: dict[str, Any], name: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Merge `changes` into a part ("base" is the base, "robot" the robot-wide settings).

    Nested mappings (joint, actuator, repeat...) merge one level deep; a value of null removes
    the field, so it goes back to its default.
    """
    if not isinstance(changes, dict) or not changes:
        raise ValueError('changes is a mapping of fields to new values, e.g. {"joint": {"home": 30}}')
    out = copy.deepcopy(build)
    name = name.lower() if name.lower() in ("robot", "base") else name
    if name == "robot":
        unknown = sorted(set(changes) - set(ROBOT_SETTINGS))
        if unknown:
            raise ValueError(f"robot-wide settings are {', '.join(ROBOT_SETTINGS)}; not {', '.join(unknown)}")
        _merge(out, changes)
        return out
    if name == "base":
        out["base"] = _merge(dict(out.get("base") or {}), changes)
        return out
    part = _find(out, name)
    name = part["name"]  # as the robot spells it
    if "name" in changes and changes["name"] != name:
        _rename(out, name, changes["name"])
    _merge(part, changes)
    return out


def remove_part(build: dict[str, Any], name: str) -> tuple[dict[str, Any], list[str]]:
    """Remove a part and everything that hangs on it; returns the new build and the removed names."""
    out = copy.deepcopy(build)
    name = _find(out, name, plurals=False)["name"]  # as the robot spells it; "legs" is not "leg" when deleting
    removed = {name}
    grew = True
    while grew:  # parts hanging on removed parts go too
        grew = False
        for p in out.get("parts", []):
            if p.get("name") not in removed and p.get("parent", "base") in removed:
                removed.add(p["name"])
                grew = True
    out["parts"] = [p for p in out.get("parts", []) if p.get("name") not in removed]
    return out, [p for p in part_names(build) if p in removed]


def libraries_in(build: dict[str, Any] | None) -> list[str]:
    """Library robots mounted on a build (they may need a download first)."""
    if not isinstance(build, dict):
        return []
    found = []
    for p in build.get("parts", []):
        mount = p.get("robot") if isinstance(p, dict) else None
        if isinstance(mount, dict) and isinstance(mount.get("library"), str):
            found.append(mount["library"])
    return found


def _find(build: dict[str, Any], name: str, plurals: bool = True) -> dict[str, Any]:
    for p in build.get("parts", []):
        if isinstance(p, dict) and p.get("name") == name:
            return p
    names = part_names(build)
    found = wording.match(name, [n for n in names if isinstance(n, str)], plurals=plurals)
    if found is not None:  # "Left_Shin" is left_shin
        return next(p for p in build["parts"] if isinstance(p, dict) and p.get("name") == found)
    source = _copied_from(build).get(name)
    if source is not None:
        raise ValueError(
            f"{name!r} is a copy made by mirror or repeat; edit the part it comes from, {source!r}, "
            "and every copy follows"
        )
    close = wording.near(name, names)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    raise ValueError(
        f"no part named {wording.short(name)}.{hint} Parts: {', '.join(names) or 'none'} (and base, robot)"
    )


def _copied_from(build: dict[str, Any]) -> dict[str, str]:
    """Final part names (after mirror and repeat) -> the part they were copied from."""
    from robotforge.design.builder import parse
    from robotforge.design.builder.expand import expand

    try:
        return {p.name: p.spec.name for p in expand(parse(build)) if p.name != p.spec.name}
    except ValueError:
        return {}


def _rename(build: dict[str, Any], old: str, new: str) -> None:
    if new in part_names(build):
        raise ValueError(f"there is already a part named {new!r}")
    for p in build.get("parts", []):
        if p.get("parent") == old:
            p["parent"] = new


def _merge(target: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    for key, value in changes.items():
        if value is None:
            target.pop(key, None)
        elif key == "robot" and _swaps_robot(target.get(key), value):
            # Another robot on the mount: the old one's params and home pose do not fit it (the PiPER's folded
            # joint2 on an SO-100 failed the edit, and a template kept beside a new library robot was refused).
            target[key] = {k: copy.deepcopy(v) for k, v in value.items() if v is not None}
        elif isinstance(value, dict) and isinstance(target.get(key), dict):
            merged = dict(target[key])
            for k, v in value.items():
                if v is None:
                    merged.pop(k, None)
                else:
                    merged[k] = v
            target[key] = merged
        else:
            target[key] = copy.deepcopy(value)
    return target


def _swaps_robot(old: Any, new: Any) -> bool:
    """True when a change to a part's `robot` names a different template or library robot than the one mounted."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    named = {k: new[k] for k in ("template", "library") if new.get(k) is not None}
    return bool(named) and any(old.get(k) != v for k, v in named.items())
