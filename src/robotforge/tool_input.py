"""The edge of the MCP server: forgiving about the form of a call, short about what is wrong with it.

An agent that sends `pos: "2, 0"`, `shape: "Ball"` or `velocity: 0.2` means something clear; refusing costs
it a round trip and refusing in pydantic's words (a type tag and a documentation link) leaves it guessing.
`tidy` reads the forms that have one meaning and rejects an argument no tool takes, since ignoring it would
let the agent believe it had an effect. `error_text` rewords what is left.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import ValidationError

from robotforge import wording

log = logging.getLogger(__name__)

# Words that name the same choice in a tool's own vocabulary, by parameter. Only readings with one meaning are
# here: the shape of a ball is a sphere, but "stop" could be pause or reset, so that one is left to the error.
ALIASES: dict[str, dict[str, str]] = {
    "shape": {
        "ball": "sphere",
        "cube": "box",
        "crate": "box",
        "block": "box",
        "can": "cylinder",
        "barrel": "cylinder",
        "desk": "table",
        "container": "bin",
        "boulder": "rock",
        "stone": "rock",
        "column": "pillar",
    },
    "type": {
        "slope": "ramp",
        "incline": "ramp",
        "staircase": "stairs",
        "steps": "stairs",
        "bumpy": "rough",
        "bumps": "rough",
        "uneven": "rough",
        "stepping_stones": "stones",
    },
    "action": {"resume": "play", "start": "play", "unpause": "play", "continue": "play", "restart": "reset"},
    "state": {"closed": "close", "grab": "close", "grasp": "close", "opened": "open", "release": "open"},
    "gripper": {"closed": "close", "grab": "close", "grasp": "close", "opened": "open", "release": "open"},
    "name": {
        "hi": "wave",
        "hello": "wave",
        "cheer": "arms_up",
        "hands_up": "arms_up",
        "crouch": "squat",
        "idle": "stand",
        "neutral": "stand",
    },
    "preset": {"desk": "desk_lab", "lab": "desk_lab", "obstacles": "obstacle_course", "course": "obstacle_course"},
}
ALIASES["style"] = ALIASES["floor"] = {
    "checkerboard": "checker",
    "checkered": "checker",
    "wooden": "wood",
    "cement": "concrete",
    "mud": "dirt",
    "soil": "dirt",
    "rocky": "rock",
}
# Parameter names agents reach for by habit, for "did you mean". Never accepted on their own: `position` is
# a real parameter of move_arm and a guess for `pos` elsewhere.
SYNONYMS = {
    "velocity": "speed",
    "vel": "speed",
    "angular": "turn",
    "yaw_rate": "turn",
    "angular_velocity": "turn",
    "duration": "seconds",
    "time": "seconds",
    "secs": "seconds",
    "robot_name": "robot",
    "robot_id": "robot",
    "position": "pos",
    "location": "pos",
    "coordinates": "pos",
    "world_name": "name",
    "world": "name",
    "object_name": "name",
    "obj": "object",
    "item": "object",
    "path": "file",
    "filename": "file",
    "colour": "color",
    "dimensions": "size",
    "dims": "size",
    "ground": "floor",
    "terrain": "type",
    "kind": "type",
    "script": "name",
    "script_name": "name",
    "content": "code",
    "source": "code",
    "joint_targets": "controls",
    "duration_s": "seconds",
}
# What a parameter takes, in words, for a value of the wrong shape.
FORMS = {
    "pos": "a list [x, y] or [x, y, z] in meters",
    "position": "a list [x, y, z] in meters",
    "size": "a list of numbers in meters, e.g. [0.1, 0.1, 0.1]",
    "target": "an object name or a list [x, y] in meters",
    "controls": 'an object like {"arm/shoulder": 0.5}',
    "methods": 'a list of names like ["ik", "pd"]',
}
NUMBERS = re.compile(r"[\[(]?\s*-?\d+(?:\.\d*)?(?:[\s,;]+-?\d+(?:\.\d*)?)*\s*[\])]?")
COORDINATES = ("x", "y", "z")
SETTINGS = ("params", "options", "task_params")  # the parameters that hold a thing's own settings
# What to say about a parameter an agent reaches for that a tool does not have, by (tool, parameter).
HINTS = {
    ("move", "distance"): "give speed (m/s) and seconds; the result reports distance_m",
    ("move", "direction"): "speed is negative for backwards and turn (rad/s) is positive for left",
    ("add_object", "radius"): "size is [diameter] for a sphere, [diameter, height] for a cylinder",
    ("add_object", "diameter"): "size is [diameter] for a sphere, [diameter, height] for a cylinder",
    ("add_object", "length"): "size is [length, width, height] in meters",
    ("add_object", "width"): "size is [length, width, height] in meters",
    ("add_object", "height"): "size is [length, width, height] in meters",
    ("move_arm", "x"): "position is one list [x, y, z]",
    ("move_arm", "y"): "position is one list [x, y, z]",
    ("move_arm", "z"): "position is one list [x, y, z]",
    ("simulate", "steps"): "give seconds (time passes in steps of the model's timestep)",
}
MAX_ERRORS = 3  # problems named in one answer; fix these and the rest follow


def _variants(prop: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The non-null alternatives of one parameter's JSON schema."""
    if "anyOf" in prop:
        return [v for sub in prop["anyOf"] for v in _variants(sub)]
    return [] if prop.get("type") == "null" else [prop]


def _literal(text: str) -> Any:
    """JSON, or Python's literal syntax (single quotes, tuples), which agents write as often. No code runs."""
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(text)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):  # TypeError: "{[1]: 2}"
            continue
    return None


def _as_list(text: str, item_type: str | None) -> list[Any] | None:
    parsed = _literal(text)
    if isinstance(parsed, list | tuple):
        return list(parsed)
    words = [w for w in re.split(r"[\s,;]+", text.strip().strip("[]()")) if w]
    if item_type in ("number", "integer"):
        try:
            return [float(w) for w in words] if words else None
        except ValueError:
            return None
    return words if item_type == "string" and words else None


def _coordinates(value: Mapping[str, Any]) -> list[Any] | None:
    """{"x": 1, "y": 2} as [1, 2]: the one reading of a position written as an object."""
    keys = {str(k).lower(): v for k, v in value.items()}
    if set(keys) <= set(COORDINATES) and {"x", "y"} <= set(keys):
        return [keys[k] for k in COORDINATES if k in keys]
    return None


def _tidy_value(name: str, value: Any, prop: Mapping[str, Any]) -> Any:
    variants = _variants(prop)
    kinds = {v.get("type") for v in variants}
    choices = next((v["enum"] for v in variants if "enum" in v), None)
    array = next((v for v in variants if v.get("type") == "array"), None)
    if isinstance(value, str):
        if choices is not None:
            return wording.match(value, choices, ALIASES.get(name)) or value
        if "string" in kinds:
            looks_like_numbers = array is not None and NUMBERS.fullmatch(value.strip())
            return (_as_list(value, "number") or value) if looks_like_numbers else value
        if array is not None:
            return _as_list(value, array.get("items", {}).get("type")) or value
        if "object" in kinds:
            parsed = _literal(value)
            return parsed if isinstance(parsed, dict) else value
    elif isinstance(value, dict) and array is not None and "object" not in kinds:
        return _coordinates(value) or value
    elif isinstance(value, int | float) and not isinstance(value, bool) and array is not None:
        if not kinds & {"number", "integer"}:
            return [value]  # size=0.05: one number is a one-item list
    return value


def _blank(value: Any, prop: Mapping[str, Any]) -> bool:
    """A value that says "not given": models that must fill in every field send null, "" or [] for the ones they
    do not mean to set. Only for a parameter that has a default, and only where the value could not mean anything."""
    variants = _variants(prop)
    nullable = len(variants) < len(prop.get("anyOf", variants))
    if value is None:
        return not nullable
    if value == "":
        return any(v.get("type") == "string" for v in variants)
    return value == [] and any(
        v.get("type") == "array" and v.get("items", {}).get("type") == "number" for v in variants
    )


def tidy(tool: str, schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments with every form that has one meaning turned into the form the tool takes.
    Raises ToolError for a parameter the tool does not have: ignoring it would make the agent think it applied."""
    props = schema.get("properties", {})
    extra = [k for k in arguments if k not in props]
    if extra:
        raise ToolError(unknown_parameters(tool, extra, list(props)))
    required = set(schema.get("required", []))
    out = {k: _read(k, v, props[k]) for k, v in arguments.items() if k in required or not _blank(v, props[k])}
    # add_robot takes its robot from exactly one of template, file, library or build. A model that fills in every
    # field sends build={} beside the source it means (or beside blank ones, when it means none): an empty build
    # says nothing, so it is not a second source. build={} alone is still a bare box.
    if tool == "add_robot" and out.get("build") == {} and any(k in arguments for k in ("template", "file", "library")):
        del out["build"]
    return out


def _read(name: str, value: Any, prop: Mapping[str, Any]) -> Any:
    """`_tidy_value`, which only ever improves a value: should it fail on some odd input, the value goes on as
    given and the tool's own check explains it. tidy runs before the server's error handling, so a crash here
    would reach the agent as a bare exception."""
    try:
        return _tidy_value(name, value, prop)
    except Exception:
        log.warning("could not read %r forgivingly; passed as given", name, exc_info=True)
        return value


def unknown_parameters(tool: str, extra: Sequence[str], params: Sequence[str]) -> str:
    named = ", ".join(repr(k) for k in extra)
    if not params:
        return f"{tool} takes no parameters (got {named})"
    close = [c[0] for k in extra if (c := wording.near(k, params, SYNONYMS, n=1))]
    text = f"{tool} has no parameter {named}"
    hint = next((HINTS[tool, k] for k in extra if (tool, k) in HINTS), None)
    if hint:
        text += f"; {hint}"
    elif close:
        text += f"; did you mean {', '.join(close)}?"
    elif settings := next((p for p in SETTINGS if p in params), None):
        text += f"; settings of the thing itself go in {settings}"
    return f"{text} (parameters: {', '.join(params)})"


def tool_name(name: str, tools: Sequence[str]) -> str | None:
    """The tool `name` means: in other capitals, or with the server's name in front ("robotforge.move")."""
    return wording.match(name, tools) or wording.match(re.split(r"[./:]|__", name)[-1], tools)


def unknown_tool(name: str, tools: Sequence[str]) -> str:
    """Tools whose name is a word of what was called ("move_robot" means move) come before look-alikes."""
    words = [w for w in name.lower().split("_") if w in tools]
    close = list(dict.fromkeys([*words, *wording.near(name, tools, n=3)]))[:3]
    return f"no tool named {name!r}" + (
        f"; did you mean {', '.join(close)}?" if close else "; use a name from the host's tool list"
    )


def error_text(tool: str, schema: Mapping[str, Any], exc: ToolError) -> str:
    """What an agent reads for a failed call: the SDK's "Error executing tool x:" is dropped (it is the tool
    the agent just called) and pydantic's report becomes one sentence per problem."""
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        return _argument_problems(tool, schema, cause)
    return str(exc).removeprefix(f"Error executing tool {tool}: ")


ARM_OF_UNION = re.compile(r"^(?:str|int|float|bool|list\[|dict\[|tuple\[|literal\[|none)", re.IGNORECASE)


def _signature(tool: str, schema: Mapping[str, Any]) -> str:
    """ "move takes robot, and optionally speed, turn": what a call that left something out should have had."""
    props = list(schema.get("properties", {}))
    required = [p for p in props if p in schema.get("required", [])]
    optional = [p for p in props if p not in required]
    text = f"{tool} takes {', '.join(required)}" if required else f"{tool} takes"
    if optional:
        text += f"{', and' if required else ''} optionally {', '.join(optional)}"
    return text


def _argument_problems(tool: str, schema: Mapping[str, Any], exc: ValidationError) -> str:
    """One sentence per rejected argument; a value that fits none of a union's forms is named once."""
    by_argument: dict[str, list[Mapping[str, Any]]] = {}
    for err in exc.errors():
        by_argument.setdefault(str(err["loc"][0]) if err["loc"] else "", []).append(err)
    sentences = []
    for name, errors in list(by_argument.items())[:MAX_ERRORS]:
        if len(errors) > 1 and all(len(e["loc"]) == 2 and ARM_OF_UNION.match(str(e["loc"][1])) for e in errors):
            form = FORMS.get(name, "one of the forms in the tool description")
            sentences.append(f"{name} must be {form}, got {wording.short(errors[0].get('input'))}")
        else:
            sentences.append(wording.validation_message(errors, forms=FORMS))
    extra = len(by_argument) - MAX_ERRORS
    text = "; ".join(sentences) + (f"; and {extra} more" if extra > 0 else "")
    if any(e["type"] == "missing" for e in exc.errors()):
        text += f". {_signature(tool, schema)}"
    return text
