"""From a description to an object: "a tall blue mug with a big round handle" -> shape, size, color, parameters.

The words come from what each generator declares (its keywords, the choices of its parameters, its `hints`), so a
generator a plugin adds is understood the same way. English and a few Chinese words. A description never guesses
quietly: `Described.notes` lists what was understood (and what was not), and add_object returns them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from robotforge.objects.registry import GENERATORS, Generator

# what the Chinese words say (the shape's name); the rest of a sentence is read as English words are
ZH = {
    "马克杯": "mug", "咖啡杯": "mug", "茶杯": "mug", "杯子": "cup", "杯": "cup", "水杯": "cup", "瓶子": "bottle",
    "瓶": "bottle", "水瓶": "bottle", "酒瓶": "bottle", "易拉罐": "can", "罐头": "can", "罐子": "can", "罐": "can",
    "碗": "bowl", "盘子": "plate", "盘": "plate", "苹果": "apple", "橙子": "orange", "橘子": "orange", "梨": "pear",
    "香蕉": "banana", "盒子": "package", "纸盒": "package", "麦片": "package", "书": "book", "锤子": "hammer",
    "螺丝刀": "screwdriver", "起子": "screwdriver", "扳手": "wrench", "芥末": "squeeze_bottle",
    "番茄酱": "squeeze_bottle", "海绵": "brick",
}  # fmt: skip
ZH_COLORS = {
    "红": "red", "蓝": "blue", "绿": "green", "黄": "yellow", "橙": "orange", "紫": "purple", "白": "white",
    "黑": "black", "灰": "gray", "棕": "brown", "粉": "pink",
}  # fmt: skip
CATEGORIES = {
    "fruit": ("apple", "orange", "pear", "banana"),
    "tool": ("screwdriver", "hammer", "wrench"),
    "drink": ("mug", "cup", "bottle", "can"),
    "drinkware": ("mug", "cup", "bottle", "can"),
    "tableware": ("mug", "cup", "bowl", "plate"),
    "dishware": ("mug", "cup", "bowl", "plate"),
    "container": ("mug", "cup", "bowl", "bottle", "can"),
    "food": ("apple", "orange", "pear", "banana", "can", "package"),
    "grocery": ("can", "package", "bottle", "tin"),
    "groceries": ("can", "package", "bottle", "tin"),
}
EXTRA_COLORS = {
    "silver": "#c0c4c9", "gold": "#d4a72c", "beige": "#d9c7a3", "navy": "#1f2f5c", "maroon": "#6d1f2b",
    "olive": "#6b7a2a", "cream": "#f1e6c9", "tan": "#c49a6c", "turquoise": "#2bb3b1", "lavender": "#a99bd6",
    "magenta": "#c2287f", "cyan": "#2bb4d8", "lime": "#9acd32", "crimson": "#b3202e", "mustard": "#d4a017",
    "burgundy": "#6e1423",
}  # fmt: skip
SIZE_WORDS = {  # word -> (share of the size it scales, which dimension: "all", "height", "length", "width")
    "tall": (1.3, "height"), "high": (1.25, "height"), "short": (0.75, "height"), "squat": (0.72, "height"),
    "low": (0.75, "height"), "big": (1.25, "all"), "large": (1.25, "all"), "huge": (1.5, "all"),
    "giant": (1.6, "all"), "small": (0.8, "all"), "tiny": (0.65, "all"), "little": (0.8, "all"),
    "mini": (0.7, "all"), "wide": (1.2, "width"), "broad": (1.2, "width"), "narrow": (0.8, "width"),
    "slim": (0.8, "width"), "thin": (0.8, "width"), "long": (1.25, "length"), "chunky": (1.15, "width"),
}  # fmt: skip
UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0}
DIM_WORDS = {
    "tall": "height", "high": "height", "long": "length", "wide": "width", "across": "width",
    "diameter": "width", "thick": "thickness",
}  # fmt: skip


@dataclass
class Described:
    shape: str
    size: list[float] | None = None
    color: Any = None
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def interpret(text: str, seed: int | None = None, default_shape: str | None = None) -> Described:
    """What `text` describes. Raises ValueError when it names nothing and there is no `default_shape`."""
    raw = text.strip()
    low = raw.lower()
    shape, matched = _shape(low, raw, seed)
    notes: list[str] = []
    if shape is None:
        if default_shape is None:
            known = ", ".join(sorted(GENERATORS))
            raise ValueError(f"no object in {raw!r}: say what it is, one of {known} (or a fruit, a tool, a drink)")
        shape = default_shape
    gen = GENERATORS.get(shape)
    if gen is None:
        return Described(shape, notes=notes)
    rest = low
    for word in matched:
        rest = rest.replace(word, " ", 1)
    out = Described(shape, notes=notes)
    out.color = _color(rest, raw, notes)
    out.params = _params(gen, low, notes)
    base = out.params.pop("_size", None)
    forced = out.params.pop("_color", None)
    if forced is not None and out.color is None:
        out.color = forced
    out.size = _size(gen, low, notes, base)
    said = [f"shape {shape}"]
    if out.color is not None:
        said.append(f"color {out.color}")
    if out.size is not None:
        said.append(f"size {[round(s, 3) for s in out.size]}")
    if out.params:
        said.append("parameters " + ", ".join(f"{k}={v}" for k, v in out.params.items()))
    notes.insert(0, "read " + repr(raw) + " as " + ", ".join(said))
    return out


def _shape(low: str, raw: str, seed: int | None) -> tuple[str | None, list[str]]:
    found: list[tuple[int, int, str, str]] = []  # (keyword length, -position, shape, keyword)
    for gen in GENERATORS.values():
        for key in {gen.name.replace("_", " "), *gen.keywords}:
            m = re.search(rf"\b{re.escape(key)}(?:s|es)?\b", low)
            if m:
                found.append((len(key), -m.start(), gen.name, key))
    for zh, shape in ZH.items():
        if zh in raw and shape in GENERATORS:
            found.append((len(zh) * 2, -raw.index(zh), shape, zh))
    if len(found) > 1:  # a color that is also a fruit ("an orange mug") is only the fruit when nothing else is named
        names = {s for _, _, s, _ in found}
        found = [f for f in found if not (f[3] in EXTRA_COLORS or f[3] in _basic_colors()) or len(names) == 1]
    if found:
        _, _, shape, key = max(found)
        return shape, [key]
    for cat, members in CATEGORIES.items():
        m = re.search(rf"\b{cat}\b", low)
        if m:
            pool = [s for s in members if s in GENERATORS]
            pick = random.Random(0 if seed is None else seed).choice(pool)
            return pick, [cat]
    return None, []


def _basic_colors() -> set[str]:
    from robotforge.design import components

    return set(components.COLORS)


def _color(rest: str, raw: str, notes: list[str]) -> Any:
    from robotforge.design import components

    for zh, name in ZH_COLORS.items():
        if zh in raw:
            return name
    words = re.findall(r"[a-z]+", rest)
    for i, w in enumerate(words):
        name = components.COLOR_SPELLINGS.get(w, w)
        base = components.COLORS.get(name)
        hexed = EXTRA_COLORS.get(name)
        if base is None and hexed is None:
            continue
        before = words[i - 1] if i else ""
        rgba = list(base) if base is not None else components.color(hexed)
        if before == "dark":
            rgba = [c * 0.6 for c in rgba[:3]] + [1.0]
        elif before in ("light", "pale", "pastel"):
            rgba = [c + (1 - c) * 0.45 for c in rgba[:3]] + [1.0]
        if before in ("dark", "light", "pale", "pastel") or hexed is not None:
            return [round(c, 3) for c in rgba[:3]]
        return name
    return None


def _params(gen: Generator, rest: str, notes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for phrase, setting in gen.hints.items():
        if re.search(rf"\b{re.escape(phrase)}\b", rest):
            out.update({k: v for k, v in setting.items() if k in gen.params or k in ("_size", "_color")})
    for key, spec in gen.params.items():
        if key in out:
            continue
        if spec.choices:
            for choice in spec.choices:
                if not isinstance(choice, str):
                    continue
                # a size word ("tall") only chooses when it stands next to what it chooses ("a tall handle")
                needs = (
                    rf"\b{re.escape(choice)}\s+(?:\w+\s+)?{re.escape(key)}"
                    if choice in SIZE_WORDS
                    else rf"\b{re.escape(choice)}\b"
                )
                if re.search(needs, rest):
                    out[key] = choice
                    break
        elif isinstance(spec.default, bool):
            word = key.replace("_", " ")
            if re.search(rf"\b(?:no|without)\s+(?:a\s+|an\s+)?{re.escape(word)}\w*", rest):
                out[key] = False
            elif re.search(rf"\b{re.escape(word)}\w*", rest):
                out[key] = True
    if "fill" in gen.params and "fill" not in out:
        for word, value in (("full", 1.0), ("half", 0.5), ("empty", 0.0), ("nearly empty", 0.1)):
            if re.search(rf"\b{word}\b", rest):
                out["fill"] = value
    for key in ("handle_size",):
        if key in gen.params and key not in out:
            for word, value in (("big", 1.2), ("large", 1.2), ("huge", 1.2), ("small", 0.85), ("tiny", 0.85)):
                if re.search(rf"\b{word}\s+(?:\w+\s+)?handle", rest):
                    out[key] = value
    extra = {k: out.pop(k) for k in ("_size", "_color") if k in out}
    checked = gen.check_params(out)
    if "_size" in extra:
        checked["_size"] = list(extra["_size"])
    if "_color" in extra:
        checked["_color"] = extra["_color"]
    return checked


def _dims(gen: Generator) -> list[str]:
    """What each number of the generator's size is: ["diameter", "height"]."""
    names = re.findall(r"[a-z ]+", gen.size_hint.lower())
    return [n.strip().replace("wheel ", "") for n in names if n.strip()]


def _size(gen: Generator, rest: str, notes: list[str], base: list[float] | None = None) -> list[float] | None:
    dims = _dims(gen)
    size = list(base) if base is not None else list(gen.default_size)
    changed = base is not None
    fixed: set[int] = set()  # dimensions a measurement set: a word like "tall" does not scale them again
    # explicit lengths: "25 cm tall", "a 9 cm wide mug", "70 mm diameter"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\b(?:\s*(tall|high|long|wide|across|diameter|thick))?", rest):
        meters = float(m.group(1)) * UNITS[m.group(2)]
        word = DIM_WORDS.get(m.group(3) or "")
        index = _index(dims, word) if word else None
        if index is None and len(size) == 1:
            index = 0
        if index is not None and 0.005 <= meters <= 1.5:
            size[index] = meters
            fixed.add(index)
            changed = True
    for word, (factor, which) in SIZE_WORDS.items():
        if not re.search(rf"\b{word}\b", rest) or re.search(rf"\b{word}\s+(?:\w+\s+)?handle", rest):
            continue
        if which == "all":
            size = [s if i in fixed else s * factor for i, s in enumerate(size)]
        else:
            index = _index(dims, which)
            if index is None and which == "width":
                index = _index(dims, "diameter")
            if index is None or index in fixed:
                continue
            size[index] *= factor
        changed = True
    return [round(s, 4) for s in size] if changed else None


def _index(dims: list[str], word: str) -> int | None:
    for i, d in enumerate(dims):
        if word in d or (word == "width" and "diameter" in d) or (word == "diameter" and "width" in d):
            return i
    return None


def resolve(fields: dict[str, Any], describe: str | None) -> tuple[dict[str, Any], list[str]]:
    """add_object's fields with a description and a seed worked out: the shape, size, color and parameters the
    description says (what the caller gave outright wins), and a seed's color from the generator's palette.
    Returns the fields to build the entry from and the notes to tell the caller."""
    out = {k: v for k, v in fields.items() if k != "describe"}
    notes: list[str] = []
    given_shape = out.get("shape")
    explicit = given_shape if given_shape not in (None, "box") else None
    if describe is not None:
        found = interpret(describe, out.get("seed"), explicit)
        notes += found.notes
        if explicit is not None and found.shape != explicit and found.shape in GENERATORS:
            notes.append(f"the description names a {found.shape} but shape={explicit!r} was given: kept {explicit}")
            found = Described(explicit)
        out["shape"] = found.shape
        if out.get("size") is None and found.size is not None:
            out["size"] = found.size
        if out.get("color") is None and found.color is not None:
            out["color"] = found.color
        out["params"] = {**found.params, **(out.get("params") or {})} or None
    gen = GENERATORS.get(out.get("shape") or "")
    if gen is not None and out.get("seed") is not None and out.get("color") is None and gen.palette:
        out["color"] = gen.palette[int(out["seed"]) % len(gen.palette)]
    if gen is not None and out.get("seed") is None and describe is not None and "random" in describe.lower():
        out["seed"] = random.Random(describe).randrange(1 << 20)
    return out, notes
