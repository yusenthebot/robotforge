"""Reading a task sentence: what to do (a verb), to what (the thing) and where (the place), in English or Chinese.

"Bring the red cup from the kitchen to the bedroom desk" is a bring: the thing is a red cup in the kitchen, the place
the desk in the bedroom. "把五个蓝色的方块放进篮子" is a put: blue cubes into the basket. The words for things,
colors and rooms are generate/words.py's, so a task can name whatever a generated world has.
"""

from __future__ import annotations

import re

from robotforge.design.generate import words
from robotforge.design.tasks.spec import Asked, Ref

# verb -> kind, longest phrases first when matched (English as whole words, Chinese as text)
VERBS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "switch_on": (("switch on", "turn on", "flip", "toggle"), ("打开开关", "拨开关", "拨", "开灯")),
    "press": (("press", "hit"), ("按下", "按一下", "按")),
    "open": (("open", "pull out", "pull open"), ("打开", "拉开", "开")),
    "close": (("close", "shut", "push in"), ("关上", "关闭", "合上", "推回", "关")),
    "pick_up": (("pick up", "lift", "grab", "raise", "pick"), ("捡起", "拿起", "举起", "抓起", "抓")),
    "push": (("push", "shove", "nudge", "kick"), ("推", "踢")),
    "bring": (
        ("bring", "fetch", "carry", "deliver", "take", "get"),
        ("拿到", "送到", "带到", "搬到", "取来", "拿来", "送", "带", "搬", "取", "拿"),
    ),
    "put": (
        ("put", "place", "drop", "set", "move", "store", "tidy", "sort", "stack"),
        ("放进", "放到", "放在", "放入", "装进", "收进", "放", "装"),
    ),
    "go_to": (
        ("go to", "walk to", "drive to", "navigate to", "head to", "move to", "go", "walk", "drive", "run",
         "navigate", "cross", "climb", "reach", "visit", "find your way"),
        ("走到", "开到", "去", "走", "开往", "到达", "穿过", "爬上", "跑到", "前往"),
    ),
}  # fmt: skip
# where the thing ends up: the words that start the place
EN_TO = r"\b(?:to|into|onto|in|on|inside|at|next to|beside|near|onto the top of|on top of|under)\b"
ZH_TO = r"(?:放进|放到|放在|放入|拿到|送到|带到|搬到|推到|开到|走到|装进|收进|到|进|在|去|往)"
EN_FROM = r"\b(?:from|in|on|out of|off)\b"
GOAL_WORDS = (("goal", "finish", "finish line", "end", "mark", "marker", "target"), ("终点", "目标", "标记", "红点"))
ROBOT_WORDS = words.ROBOTS


def _hits(text: str, en: tuple[str, ...], zh: tuple[str, ...]) -> list[tuple[int, int]]:
    out = []
    for p in sorted(en, key=len, reverse=True):
        out += [(m.start(), m.end()) for m in re.finditer(rf"\b{re.escape(p)}\b", text)]
    for p in sorted(zh, key=len, reverse=True):
        out += [(m.start(), m.end()) for m in re.finditer(re.escape(p), text)]
    return sorted(out)


def verb(text: str) -> tuple[str, int, int] | None:
    """The task's verb: its kind and where it is said (the first verb wins, longest phrase at a place)."""
    found: list[tuple[int, int, str]] = []
    for kind, (en, zh) in VERBS.items():
        found += [(a, -(b - a), kind) for a, b in _hits(text, en, zh)]
    if not found:
        return None
    a, neg_len, kind = min(found)
    return kind, a, a - neg_len


def ref(text: str) -> Ref | None:
    """A thing or place said in `text`: its concept, color and room."""
    t = text.strip()
    if not t:
        return None
    rooms = words.find(t, words.ROOMS)
    rest = words._masked(t, rooms)
    things = words.find(rest, words.THINGS)
    goal = _hits(rest, *GOAL_WORDS)
    named = re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", t)  # an object's own name, red_cup
    # the first thing said is the one meant; the rest say where it is ("the cup on the table")
    concept = things[0].concept if things else None
    if concept is None and goal:
        concept = "goal"
    if concept is None and rooms:
        concept = "room"
    color = None
    if things:
        color = words._color_before(rest, things[0].start)
    return Ref(
        text=t,
        concept=concept,
        color=color,
        room=rooms[-1].concept if rooms else None,
        name=named[0] if named else None,
    )


def read(sentence: str) -> Asked:
    """What a task sentence asks. Raises ValueError when it names nothing to do."""
    text = sentence.strip().lower()
    found = verb(text)
    if found is None:
        raise ValueError(
            "say what to do: go to, bring, put, pick up, push, open, close, press or switch on, e.g. 'bring the red "
            "cup to the bedroom desk' or '把苹果放进碗里'"
        )
    kind, _, end = found
    rest = text[end:]
    zh = bool(re.search(r"[一-鿿]", text))
    robots = words.find(text, ROBOT_WORDS)
    robot = robots[0].concept if robots else None
    before = text[: found[1]]
    if zh and "把" in before:  # 把苹果放进碗里: the thing comes before the verb
        thing_text = before[before.index("把") + 1 :]
        place_text = rest
    else:
        split = re.search(ZH_TO if zh else EN_TO, rest)
        if kind in ("go_to",):
            thing_text, place_text = "", rest
        elif split is None:
            thing_text, place_text = rest, ""
        else:
            thing_text, place_text = rest[: split.start()], rest[split.end() :]
            # "the cup on the table to the sink": the first "on" belongs to the thing, the place starts at "to"
            later = re.search(r"\b(?:to|into|onto)\b", rest[split.end() :]) if not zh else None
            if later is not None and split.group(0) not in ("to", "into", "onto"):
                thing_text = rest[: split.end() + later.start()]
                place_text = rest[split.end() + later.end() :]
    if kind in ("open", "close", "press", "switch_on", "pick_up") and not (zh and "把" in before):
        thing_text, place_text = rest, ""  # "open the drawer in the kitchen": the room says which drawer
    thing = ref(_strip_robot(thing_text))
    place = ref(_strip_robot(place_text))
    # bring and put with no place said: to the robot's start would be odd; a sentence like "pick up the cup" says it
    no_place = place is None  # nothing said after the thing: "bring me the apple"
    if kind in ("bring", "put") and no_place and thing is not None and thing.concept is not None:
        kind = "pick_up"
    if kind == "push" and thing is not None and thing.concept == "button":
        kind = "press"
    if kind in ("open",) and thing is not None and thing.concept == "switch":
        kind = "switch_on"
    return Asked(kind=kind, thing=thing, place=place, robot=robot)


def _strip_robot(text: str) -> str:
    hits = words.find(text, ROBOT_WORDS)
    return words._masked(text, hits) if hits else text
