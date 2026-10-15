"""Free-form robot builder: a base and a tree of parts (`build:` in world.yaml), compiled with MjSpec.

    from robotforge.design import builder
    robot = builder.build_robot({"base": {...}, "parts": [...]})   # -> BuiltRobot

Templates are fixed designs with a few knobs; the builder is the Lego box they are
made from. Parts get real actuators from the parts catalog, mass from their
shape and material, and can carry sensors or a whole template robot (an arm on a
rover). mirror and repeat make symmetric and many-legged robots short to write.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from robotforge import wording
from robotforge.design.builder.assemble import assemble
from robotforge.design.builder.checks import check
from robotforge.design.builder.expand import expand
from robotforge.design.builder.schema import BuildSpec, PartSpec
from robotforge.design.templates.base import BuiltRobot

# Pydantic labels union branches in error locations; they mean nothing to a beginner.
_BRANCH_TAGS = ("str", "list[float]", "list[float]", "float", "int", "bool", "dict[str,any]", "none")


def parse(data: dict[str, Any] | BuildSpec) -> BuildSpec:
    """Validate a build description; errors name the part and say what to write instead."""
    if isinstance(data, BuildSpec):
        return data
    if not isinstance(data, dict):
        raise ValueError('build is a mapping like {"base": {...}, "parts": [...]}')
    try:
        return BuildSpec.model_validate(data)
    except ValidationError as exc:
        raise ValueError(readable(exc, data)) from None


def build_robot(data: dict[str, Any] | BuildSpec) -> BuiltRobot:
    """Compile a built robot: check it, measure it and return its spec with the home pose."""
    build = parse(data)
    placed = expand(build)
    report = check(assemble(build, placed))
    final = assemble(build, placed)  # a fresh spec: the checked one had a test floor added
    joints = sum(1 for p in placed if p.spec.joint.type != "fixed")
    summary = (
        f"built from {len(placed) + 1} parts ({joints} joints, {final.spec.compile().nu} actuators), "
        f"{report.mass:.2f} kg"
    )
    if final.mounted:
        summary += "; carries " + ", ".join(f"{what} on {part}" for part, what in final.mounted.items())
    return BuiltRobot(
        spec=final.spec,
        home_qpos=final.home_qpos,
        home_ctrl=final.home_ctrl,
        spawn_height=report.spawn_height,
        floating=final.floating,
        notes=[summary, *final.notes, *report.notes],
    )


def readable(exc: ValidationError, data: dict[str, Any]) -> str:
    """One line per problem, naming parts by their names instead of list positions."""
    parts = data.get("parts") if isinstance(data, dict) else None
    seen: dict[str, list[str]] = {}
    for err in exc.errors():
        loc = [x for x in err["loc"] if not (isinstance(x, str) and _is_tag(x))]
        where = []
        for i, key in enumerate(loc):
            if key == "parts" and i + 1 < len(loc) and isinstance(loc[i + 1], int):
                idx = loc[i + 1]
                name = parts[idx].get("name") if isinstance(parts, list) and idx < len(parts) else None
                where.append(f"part {name!r}" if isinstance(name, str) else f"parts[{idx}]")
            elif isinstance(key, int):
                if not (i > 0 and loc[i - 1] == "parts"):
                    where.append(f"[{key}]")
            else:
                where.append(str(key))
        label = " ".join(where).replace(" [", "[")
        if err["type"] == "extra_forbidden":
            msg = f"{label} is not a known field (check the spelling; see builder_guide)"
        else:
            msg = wording.validation_message([{**err, "loc": (label,)}])
        seen.setdefault(label, [])
        if msg not in seen[label]:
            seen[label].append(msg)
    lines = [" or ".join(msgs) for msgs in seen.values()]
    more = f" (and {len(lines) - 4} more)" if len(lines) > 4 else ""
    return "; ".join(lines[:4]) + more


def _is_tag(key: str) -> bool:
    k = key.lower().replace(" ", "")
    return (
        k in _BRANCH_TAGS
        or k.startswith(("function-", "literal[", "list[", "dict[", "tagged-union"))
        or (k[:1].isupper() and k.endswith("spec"))
    )


__all__ = ["BuildSpec", "PartSpec", "build_robot", "parse", "readable"]
