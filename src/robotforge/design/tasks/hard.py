"""The harder tasks: put a mug on a shelf upright, stack three different things, insert a bottle into a rack, sort a
cluttered bin by kind, hand a tool over, pour from one cup into another.

    add_task("put the mug on the shelf upright")
    add_task("stack the brick on the tin on the book")
    add_task("insert the bottle into the rack")
    add_task("sort the things in the bin into the trays by kind")

This module reads such a sentence (`plan`: it finds the things it names in the world, by what they are and their color),
builds a whole bench for each kind (`build`: a Panda or a UR5e with a Robotiq on a table with the fixtures and things
the task needs, seeded) and has a scripted expert do it (`solve`: the same pick_and_place calls an agent would make,
each reporting how it went). The success checks are `hard_check`.
"""

from __future__ import annotations

import math
import re
from typing import Any

from robotforge import objects
from robotforge.design.tasks.spec import DEFAULT_SECONDS, HARD_KINDS, Randomize, Start, TaskEntry

TABLE_Z = 0.4  # m: the bench's table top, over the floor the robot's base stands on
ARMS: dict[str, dict[str, Any]] = {
    "panda": {"library": "franka_emika_panda", "mounts": None},
    "ur5e": {"library": "ur5e", "mounts": [{"library": "robotiq_2f85"}]},
}
SORT_CLASS = {
    "apple": "fruit", "orange": "fruit", "pear": "fruit", "banana": "fruit",
    "can": "cans", "tin": "boxes", "brick": "boxes", "package": "boxes",
    "hammer": "tools", "screwdriver": "tools", "wrench": "tools", "spatula": "tools", "pen": "tools", "tape": "tools",
    "sponge": "boxes", "pitcher": "mugs",
    "bottle": "bottles", "squeeze_bottle": "bottles",
    "mug": "mugs", "cup": "mugs", "bowl": "dishes", "plate": "dishes", "book": "books",
}  # fmt: skip


class TaskProblem(ValueError):
    """The sentence names something the world does not have, or nothing can do it: the message says what to do."""


# ---- reading a sentence -----------------------------------------------------------------------------------------

WORDS = {
    "pour": r"\bpour\w*\b|\btip\b.*\binto\b|倒",
    "hand_over": r"\bhand(?:s|ed)?\b(?! ?(?:ful|le))|\bgive\b|\bpass\b|递给|交给",
    "sort": r"\bsort\w*\b|\borgani[sz]e\b|\bseparate\b|分类|整理",
    "stack": r"\bstack\w*\b|\bpile\b|堆叠|叠起",
    "insert": r"\binsert\w*\b|\bslot\b|\bplug\b|插入|插进",
    "upright": r"\bupright\b|\bright way up\b|\bright side up\b|\bstand(?:ing)? up\b|\bstand the\b|竖着|立着|正放",
}


def recognizes(sentence: str) -> str | None:
    """The kind of hard task a sentence asks for, or None (it is one of the older kinds)."""
    low = sentence.lower()
    for kind in ("pour", "sort", "stack", "insert", "hand_over", "upright"):
        if re.search(WORDS[kind], low):
            return kind
    return None


def _mentions(text: str) -> list[tuple[int, str, str | None]]:
    """The things a sentence names, in order, as (position, shape, keyword): by what the generators call themselves
    (and a few plain shapes). The longest keyword at a place wins ("wine bottle" over "bottle"), and a color that is
    also a fruit ("the orange mug") is not a thing when a noun follows it."""
    low = text.lower()
    spans: list[tuple[int, int, str, str]] = []
    for gen in objects.GENERATORS.values():
        for key in {gen.name.replace("_", " "), *gen.keywords}:
            spans += [(m.start(), m.end(), gen.name, key) for m in re.finditer(rf"\b{re.escape(key)}(?:s|es)?\b", low)]
    for shape, word in (
        ("shelf", "shelf"),
        ("shelf", "shelves"),
        ("table", "table"),
        ("bin", "bin"),
        ("bin", "basket"),
    ):
        spans += [(m.start(), m.end(), shape, word) for m in re.finditer(rf"\b{word}\b", low)]
    picked: list[tuple[int, int, str, str]] = []
    for span in sorted(spans, key=lambda s: -(s[1] - s[0])):
        if all(span[1] <= p[0] or span[0] >= p[1] for p in picked):
            picked.append(span)
    picked.sort()
    out = []
    for i, (start, end, shape, key) in enumerate(picked):
        follows = i + 1 < len(picked) and picked[i + 1][0] - end <= 2
        if key in ("orange", "lime") and follows:
            continue
        out.append((start, shape, key))
    return out


def _color_of(text: str, near: int) -> str | None:
    from robotforge.design import components

    words = re.findall(r"[a-z]+", text[max(0, near - 14) : near].lower())
    for w in reversed(words[-2:]):
        if w in components.COLORS or w in components.COLOR_SPELLINGS:
            return components.COLOR_SPELLINGS.get(w, w)
    return None


def _match(objs: list[dict[str, Any]], shape: str, color: str | None, used: set[str]) -> str | None:
    """The object in the world that is a `shape` (of that color when said), not yet taken by this sentence."""
    shapes = (shape, "tray") if shape == "bin" else (shape,)  # a tray is a bin to a sentence
    fits = [o for o in objs if o["shape"] in shapes and o["name"] not in used]
    fits.sort(key=lambda o: o["name"] != shape)  # one named for what it is first
    if color is not None:
        colored = [o for o in fits if _is_color(o, color)]
        fits = colored or fits
    return fits[0]["name"] if fits else None


def _is_color(o: dict[str, Any], color: str) -> bool:
    from robotforge.design import components

    if color in o["name"].split("_"):
        return True
    have = o.get("color")
    if isinstance(have, str):
        return have == color
    want = components.COLORS.get(color)
    return (
        want is not None
        and have is not None
        and all(abs(a - b) < 0.12 for a, b in zip(have[:3], want[:3], strict=False))
    )


def _objects(world: Any) -> list[dict[str, Any]]:
    return [o.model_dump(mode="json") for o in world.objects]


def plan(
    world: Any, sentence: str, robots: list[Any], robot: str | None, name: str | None, seed: int, kind: str
) -> Any:
    """The task `sentence` asks for in `world` (`kind` from `recognizes`), as a make.Plan."""
    from robotforge.design.tasks.make import Plan

    objs = _objects(world)
    mentions = _mentions(sentence)
    used: set[str] = set()
    names: list[str] = []
    for pos, shape, _ in mentions:
        found = _match(objs, shape, _color_of(sentence, pos), used)
        if found is None:
            have = ", ".join(sorted({o["shape"] for o in objs})) or "nothing"
            raise TaskProblem(
                f"no {shape.replace('_', ' ')} in this world (it has: {have}). Add one with add_object(name, "
                f'shape="{shape}") or describe="a {shape.replace("_", " ")}"'
            )
        used.add(found)
        names.append(found)
    arms = [r for r in robots if r.kind == "arm" or r.arms]
    chosen = robot or (arms[0].name if arms else None)
    if chosen is None:
        raise TaskProblem("no robot with an arm and a gripper to do it: add one (a Panda: add_robot(library='panda'))")
    task = _task(kind, sentence, names, objs, chosen, name or kind, world, arms)
    notes: list[str] = []
    entry = next((r for r in world.robots if r.name == chosen), None)
    task.start = Start(
        robot=[*map(float, entry.pos[:2]), float(entry.yaw)] if entry is not None else None,
        objects={n: [float(v) for v in next(o["pos"] for o in objs if o["name"] == n)] for n in _loose(task, objs)},
    )
    task.randomize = Randomize(robot_xy=0.0, robot_yaw=0.0, object_xy=0.02)
    task.seconds = DEFAULT_SECONDS[kind]
    out = Plan(task=task, notes=notes)
    return out


def _loose(task: TaskEntry, objs: list[dict[str, Any]]) -> list[str]:
    want = [n for n in [task.object, *(task.items or [])] if n]
    movable = {o["name"] for o in objs if not o.get("fixed")}
    return [n for n in dict.fromkeys(want) if n in movable and len(next(o["pos"] for o in objs if o["name"] == n)) == 3]


def _task(kind: str, sentence: str, names: list[str], objs: list[dict[str, Any]], robot: str, name: str,
          world: Any, arms: list[Any]) -> TaskEntry:  # fmt: skip
    by = {o["name"]: o for o in objs}
    low = sentence.lower()

    def need(n: int, what: str) -> None:
        if len(names) < n:
            raise TaskProblem(f"say {what}: {sentence!r} names {len(names)} thing(s) the world has, {kind} needs {n}")

    base = dict(name=_unique(name, world), sentence=sentence.strip(), kind=kind, robot=robot)
    if kind == "upright":
        need(2, "what to stand where, e.g. 'put the mug on the shelf upright'")
        return TaskEntry(**base, object=names[0], target=names[1])
    if kind == "insert":
        need(2, "what goes into what, e.g. 'insert the bottle into the rack'")
        target = names[1]
        slot = re.search(r"(?:socket|slot|hole|place|position)\s*(\d)|(\d)(?:st|nd|rd|th)\s+(?:socket|slot|hole)", low)
        ordinal = {"first": 1, "second": 2, "third": 3, "fourth": 4}
        word = next((v for k, v in ordinal.items() if re.search(rf"\b{k}\b", low)), None)
        number = int(next(g for g in slot.groups() if g)) if slot else word
        return TaskEntry(**base, object=names[0], target=f"{target}/{number}" if number else target)
    if kind == "stack":
        need(2, "what to stack, e.g. 'stack the brick on the tin on the book'")
        listed = re.search(r"\b(?:on|onto|on top of|over)\b", low)
        items = list(reversed(names)) if listed else list(names)  # "A on B on C" puts C at the bottom
        return TaskEntry(**base, object=items[-1], target=items[0], items=items)
    if kind == "sort":
        return _sort_task(base, names, objs, by, sentence)
    if kind == "pour":
        need(2, "what to pour into what, e.g. 'pour the cup into the bowl'")
        source, target = names[0], names[1]
        balls = _balls_in(by, source)
        if not balls:
            raise TaskProblem(f"there is nothing in {source} to pour: add balls (add_object shape sphere) inside it")
        return TaskEntry(**base, object=source, target=target, items=balls)
    if kind == "hand_over":
        need(1, "what to hand over, e.g. 'hand the hammer to the other arm'")
        others = [a.name for a in arms if a.name != robot]
        said = [a.name for a in arms if a.name != robot and a.name in low]
        receivers = said or others
        if not receivers:
            raise TaskProblem("a hand-over needs a second arm to hand it to: add one with add_robot")
        return TaskEntry(**base, object=names[0], target=receivers[0])
    raise TaskProblem(f"cannot plan a {kind} task")


def _sort_task(base: dict[str, Any], names: list[str], objs: list[dict[str, Any]], by: dict[str, Any],
               sentence: str) -> TaskEntry:  # fmt: skip
    trays = [o for o in objs if o["shape"] == "tray" and (o.get("params") or {}).get("tag") not in (None, "none")]
    if not trays:
        raise TaskProblem(
            "sorting needs trays with a name printed on each: add_object(shape='tray', params={'tag': 'fruit'})"
        )
    tags = {(t.get("params") or {})["tag"]: t["name"] for t in trays}
    bins = [o for o in objs if o["shape"] in ("tray", "bin") and o not in trays]
    loose = [o for o in objs if not o.get("fixed") and o["shape"] in SORT_CLASS]
    if bins:  # things in the bin the sentence names, else all those lying inside any bin
        keep = []
        for o in loose:
            for b in bins:
                hx, hy = b["size"][0] / 2, b["size"][1] / 2
                if abs(o["pos"][0] - b["pos"][0]) <= hx and abs(o["pos"][1] - b["pos"][1]) <= hy:
                    keep.append(o)
                    break
        loose = keep or loose
    into: dict[str, str] = {}
    for o in loose:
        klass = SORT_CLASS[o["shape"]]
        if klass in tags:
            into[o["name"]] = tags[klass]
    if not into:
        raise TaskProblem(
            f"none of the things fit a tray's name ({', '.join(tags)}): sort needs fruit, cans, boxes, ..."
        )
    first = next(iter(into.values()))
    return TaskEntry(**base, object=next(iter(into)), target=first, items=list(into), into=into)


def _balls_in(by: dict[str, Any], source: str) -> list[str]:
    cup = by[source]
    built = objects.built_of(cup)
    radius = float(built.facts.get("mouth_r", 0.04)) if built is not None else 0.04
    out = []
    for name, o in by.items():
        loose_ball = o["shape"] == "sphere" and not o.get("fixed") and len(o["pos"]) == 3
        if loose_ball and math.hypot(o["pos"][0] - cup["pos"][0], o["pos"][1] - cup["pos"][1]) <= radius + 0.01:
            out.append(name)
    return out


def _unique(name: str, world: Any) -> str:
    taken = {t.name for t in getattr(world, "tasks", [])}
    if name not in taken:
        return name
    k = 2
    while f"{name}_{k}" in taken:
        k += 1
    return f"{name}_{k}"


__all__ = ["ARMS", "HARD_KINDS", "TABLE_Z", "TaskProblem", "plan", "recognizes"]
