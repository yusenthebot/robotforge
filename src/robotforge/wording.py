"""What agents read and write: names matched loosely, lists of choices, one-line validation errors, compact JSON.

Agents pay for every token of an answer and fix a mistake from its text alone, so an error names the
argument and the value that was wrong and says what would have worked. Names an agent mistypes in a way
that has only one reading (capitals, spaces for underscores, "the dog") are accepted instead of refused.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

LIST_LIMIT = 12  # names listed in an error before the rest are counted; a longer list costs more than it helps
ARTICLES = ("the_", "my_", "a_", "an_")  # "the dog", "my arm"
SHORT = 40  # characters of a rejected value shown back to the agent


def squash(text: str) -> str:
    """The loose spelling of a name: lowercase, spaces and hyphens as underscores ("The Dog" -> "the_dog")."""
    return re.sub(r"[\s_-]+", "_", text.strip().lower()).strip("_")


def match(given: str, names: Iterable[str], aliases: Mapping[str, str] | None = None) -> str | None:
    """The one name that `given` can only mean: spelled exactly, in other capitals or with spaces for
    underscores, with an article in front, in the singular or plural, or an alias. None if it fits none or
    several (a guess between two robots could act on the wrong one)."""
    names = list(names)
    if given in names:
        return given
    by_key: dict[str, list[str]] = {}
    for name in names:
        by_key.setdefault(squash(name), []).append(name)
    key = squash(given)
    candidates = [key, *(key[len(a) :] for a in ARTICLES if key.startswith(a) and len(key) > len(a))]
    for cand in candidates:
        for form in (cand, cand.removesuffix("s"), cand + "s"):
            found = by_key.get(form, [])
            if len(found) == 1:
                return found[0]
            target = aliases.get(form) if aliases else None
            if target in names:
                return target
    return None


def near(given: str, names: Iterable[str], aliases: Mapping[str, str] | None = None, n: int = 3) -> list[str]:
    """Names that look like `given`, closest first, for "did you mean"."""
    names = list(names)
    key = squash(given)
    out = [aliases[key]] if aliases and key in aliases and aliases[key] in names else []
    by_key = {squash(name): name for name in names}
    out += [by_key[k] for k in difflib.get_close_matches(key, list(by_key), n=n)]
    return list(dict.fromkeys(out))[:n]


def choices(names: Sequence[str], limit: int = LIST_LIMIT) -> str:
    """The names as a comma list, cut after `limit` with a count of the rest."""
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown}, and {len(names) - limit} more"


def plural(kind: str) -> str:
    return kind + ("es" if kind.endswith(("ch", "sh", "s", "x")) else "s")


def unknown(
    kind: str,
    given: str,
    names: Sequence[str],
    where: str = "in this world",
    aliases: Mapping[str, str] | None = None,
    many: str | None = None,
) -> str:
    """ "no robot named 'dog1'. Did you mean dog? Robots in this world: dog, arm": what failed, and the way out.
    `many` is the plural of `kind` when it is not just an s."""
    many = many or plural(kind)
    text = f"no {kind} named {given!r}."
    close = near(given, names, aliases)
    if close:
        text += f" Did you mean {', '.join(close)}?"
    if names:
        return f"{text} {many.capitalize()} {where}: {choices(list(names))}"
    return f"{text} There are no {many} {where}."


def short(value: Any) -> str:
    """A rejected value as the agent sent it, cut to a few words."""
    text = repr(value)
    return text if len(text) <= SHORT else text[: SHORT - 3] + "..."


TOO_BIG = 5  # a value this many times over its maximum is probably in the wrong unit, not just a little large


def unit_hint(value: Any, high: float) -> str:
    """Beginners write centimetres or millimetres where the tools take meters: say which number was meant."""
    if isinstance(value, bool) or not isinstance(value, int | float) or high <= 0 or value < TOO_BIG * high:
        return ""
    for scale, unit in ((100, "cm"), (1000, "mm")):
        if value / scale <= high:
            return f" (meters, so {value:g} {unit} is {value / scale:g})"
    return ""


def where(loc: Sequence[Any]) -> str:
    """A pydantic location as the agent wrote it: pos[0], params.leg_length."""
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else f".{part}" if out else str(part)
    return out


COLLECTIONS = ("robots", "objects", "terrain")  # the lists of a world
BOUNDS = {"greater_than": "gt", "greater_than_equal": "ge", "less_than": "lt", "less_than_equal": "le"}
BOUND_TEXT = {"gt": "above", "ge": "at least", "lt": "below", "le": "at most"}


def validation_message(
    exc: ValidationError | Iterable[Mapping[str, Any]],
    valid: Sequence[str] | None = None,
    forms: Mapping[str, str] | None = None,
) -> str:
    """A pydantic error as short sentences: which field, what was wrong with it, what is allowed.

    `valid` lists the field names the model has, for "not a known field"; `forms` says in words what a field
    takes ("pos": "[x, y] in meters") for values of the wrong shape."""
    parts = []
    for err in exc.errors() if isinstance(exc, ValidationError) else exc:
        loc = err["loc"]
        at = where(loc)
        kind, ctx, value = err["type"], err.get("ctx", {}), err.get("input")
        form = (forms or {}).get(str(loc[0]) if loc else "")
        if kind == "missing":
            parts.append(f"{at} is required")
        elif kind == "extra_forbidden":
            text = f"{at!r} is not a known field"
            close = near(str(loc[-1]), valid or [])
            if close:
                text += f"; did you mean {', '.join(close)}?"
            if valid:
                text += f" (fields: {choices(list(valid))})"
            parts.append(text)
        elif kind == "literal_error":
            options = re.findall(r"'([^']*)'", str(ctx.get("expected", "")))
            parts.append(f"{at} {short(value)} is not one of: {', '.join(options)}")
        elif kind in BOUNDS:
            bound = BOUNDS[kind]
            hint = unit_hint(value, ctx[bound]) if bound in ("le", "lt") else ""
            parts.append(f"{at} is {value!r} but must be {BOUND_TEXT[bound]} {ctx[bound]}{hint}")
        elif kind.endswith(("_parsing", "_type")) and kind.split("_")[0] in ("float", "int", "decimal"):
            parts.append(f"{at} must be a plain number (no units), got {short(value)}")
        elif kind == "int_from_float":
            parts.append(f"{at} must be a whole number, got {short(value)}")
        elif kind == "bool_parsing":
            parts.append(f"{at} must be true or false, got {short(value)}")
        elif kind in ("list_type", "dict_type", "tuple_type", "model_type"):
            what = form or ("a list" if kind in ("list_type", "tuple_type") else "an object")
            parts.append(f"{at} must be {what}, got {short(value)}")
        elif kind == "string_type":
            parts.append(f"{at} must be text, got {short(value)}")
        else:
            msg = err["msg"].removeprefix("Value error, ")
            if len(loc) >= 2 and loc[0] in COLLECTIONS and isinstance(loc[1], int):
                at = where(loc[2:])  # "robots.3" is a position in world.yaml; the message names the robot itself
            parts.append(f"{at}: {msg}" if at else msg)
    return "; ".join(parts)


def too_far(point: Sequence[float], distance: float, reach: float) -> str:
    """ "it is 25.0 m from the arm's base, which reaches about 0.45 m", and what the point would be in meters if
    its numbers were centimetres (the usual mistake)."""
    text = f"it is {distance:.2f} m from the arm's base, which reaches about {reach:.2f} m; give a nearer point"
    if distance >= TOO_BIG * reach and distance / 100 <= reach * 1.05:
        text += f" (positions are meters; in cm that would be {[round(float(v) / 100, 3) for v in point]})"
    return text


def dumps(data: object, default: Callable[[Any], Any] | None = None) -> str:
    """JSON for an agent: every top-level key on its own line, everything inside it compact.
    Indenting each list item and number (the default) roughly doubled the size of what the model reads."""
    flat = {"separators": (",", ":"), "ensure_ascii": False, "default": default}
    if not isinstance(data, dict) or not data:
        return json.dumps(data, **flat)
    lines = [f"{json.dumps(str(key), ensure_ascii=False)}:{json.dumps(value, **flat)}" for key, value in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}"
