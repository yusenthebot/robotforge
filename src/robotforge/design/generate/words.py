"""Reading a sentence for what world it asks for: the kind of place, its rooms, what is in it and how much.

No language model: a small dictionary of English and Chinese words for places, rooms, furniture, things, terrain,
grounds, colors, numbers and robots. What it does not know it leaves out, and the world says what it understood
(`Wish.summary`), so an agent can add the rest with add_object. The same sentence always reads the same way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Each concept: English phrases (matched as whole words, a plural s allowed) and Chinese ones (matched as text).
Lexicon = dict[str, tuple[tuple[str, ...], tuple[str, ...]]]

KINDS: Lexicon = {
    "warehouse": (
        ("warehouse", "storage hall", "depot", "distribution center", "logistics", "stockroom", "loading bay"),
        ("仓库", "库房", "物流", "货仓", "仓储"),
    ),
    "outdoor": (
        (
            "park",
            "forest",
            "wood",
            "woods",
            "garden",
            "yard",
            "backyard",
            "field",
            "meadow",
            "hillside",
            "hill",
            "mountain",
            "outdoor",
            "outdoors",
            "outside",
            "trail",
            "beach",
            "desert",
            "campsite",
            "countryside",
            "lawn",
        ),
        (
            "公园",
            "森林",
            "树林",
            "花园",
            "院子",
            "草地",
            "草坪",
            "山坡",
            "山丘",
            "户外",
            "野外",
            "沙滩",
            "沙漠",
            "雪地",
            "营地",
            "郊外",
            "田野",
            "小山",
            "山地",
            "室外",
        ),
    ),
    "house": (
        ("house", "apartment", "flat", "home", "cottage", "studio", "room", "rooms", "bungalow", "suite"),
        ("房子", "公寓", "家里", "屋", "房间", "住宅", "小屋", "套房", "居室", "室"),
    ),
}

ROOMS: Lexicon = {
    "kitchen": (("kitchen",), ("厨房",)),
    "living_room": (("living room", "lounge", "sitting room", "family room", "living area"), ("客厅", "起居室")),
    "bedroom": (("bedroom", "bed room", "nursery", "kids room"), ("卧室", "睡房", "儿童房", "主卧", "次卧")),
    "office": (("office", "study", "workroom", "workspace"), ("办公室", "书房", "工作室")),
    "dining_room": (("dining room", "dining area"), ("餐厅", "饭厅")),
    "bathroom": (("bathroom", "washroom", "toilet", "restroom"), ("卫生间", "浴室", "厕所", "洗手间")),
}

# What can be put in a world, by the name the generator builds it under.
THINGS: Lexicon = {
    # furniture
    "table": (("table", "dining table", "workbench"), ("桌子", "餐桌", "工作台")),
    "coffee_table": (("coffee table", "side table", "low table"), ("茶几", "边几")),
    "desk": (("desk", "workstation"), ("书桌", "办公桌", "写字台")),
    "shelf": (("shelf", "shelves", "bookshelf", "bookshelves", "bookcase", "shelving"), ("架子", "书架", "置物架")),
    "rack": (("rack", "racks", "rack row", "shelving row"), ("货架",)),
    "counter": (("counter", "countertop", "worktop", "kitchen counter"), ("台面", "灶台", "橱柜台", "操作台")),
    "sofa": (("sofa", "couch", "settee"), ("沙发",)),
    "bed": (("bed",), ("床",)),
    "chair": (("chair", "stool"), ("椅子", "凳子")),
    "cabinet": (("cabinet", "cupboard", "wardrobe", "closet"), ("柜子", "衣柜", "橱柜", "壁橱")),
    "drawer": (("drawer", "dresser", "nightstand", "chest of drawers", "bedside table"), ("抽屉", "床头柜", "斗柜")),
    "fridge": (("fridge", "refrigerator"), ("冰箱",)),
    "bench": (("bench",), ("长椅", "长凳")),
    "tree": (("tree",), ("树",)),
    "rock": (("rock", "boulder"), ("石头", "岩石", "大石")),
    "pallet": (("pallet",), ("托盘", "栈板", "垛板")),
    "conveyor": (("conveyor", "conveyor belt", "belt"), ("传送带", "输送带")),
    "bin": (("bin", "basket", "tray", "tote", "container"), ("篮子", "收纳盒", "垃圾桶", "筐", "容器")),
    "button": (("button",), ("按钮",)),
    "switch": (("switch", "light switch"), ("开关",)),
    "lever": (("lever",), ("拉杆", "把手杆")),
    "valve": (("valve",), ("阀门", "阀")),
    "pillar": (("pillar", "column"), ("柱子",)),
    "tv": (("tv", "television"), ("电视",)),
    # things to pick up
    "apple": (("apple",), ("苹果",)),
    "cup": (("cup", "mug", "glass"), ("杯子", "茶杯", "水杯", "杯")),
    "plate": (("plate", "dish"), ("盘子", "碟子")),
    "bowl": (("bowl",), ("碗",)),
    "ball": (("ball", "toy ball"), ("球", "皮球")),
    "book": (("book",), ("书本", "一本书", "书")),
    "can": (("can", "soda can", "tin can", "bottle", "jar", "tin"), ("罐子", "瓶子", "罐头", "易拉罐", "瓶")),
    "cube": (("cube", "block", "brick"), ("方块", "积木")),
    "box": (("box", "carton", "parcel", "package", "crate"), ("盒子", "箱子", "纸箱", "包裹", "快递")),
    "toy": (("toy", "teddy"), ("玩具", "玩偶")),
}
FURNITURE = (
    "table",
    "coffee_table",
    "desk",
    "shelf",
    "rack",
    "counter",
    "sofa",
    "bed",
    "chair",
    "cabinet",
    "drawer",
    "fridge",
    "bench",
    "tree",
    "rock",
    "pallet",
    "conveyor",
    "bin",
    "button",
    "switch",
    "lever",
    "valve",
    "pillar",
    "tv",
)
ITEMS = ("apple", "cup", "plate", "bowl", "ball", "book", "can", "cube", "box", "toy")

TERRAIN: Lexicon = {
    "hills": (
        ("hill", "hills", "hilly", "hillside", "rolling", "bumpy", "mound", "slopes"),
        ("山丘", "丘陵", "起伏", "小山", "土坡"),
    ),
    "rough": (("rocky", "rough", "gravel", "rubble", "uneven", "rugged"), ("碎石", "崎岖", "乱石", "坑洼", "不平")),
    "stairs": (("stairs", "steps", "staircase", "stairway"), ("台阶", "楼梯", "阶梯")),
    "ramp": (("ramp", "slope", "incline"), ("坡道", "斜坡")),
    "stones": (("stepping stones", "stepping stone", "stepping-stones"), ("踏脚石", "垫脚石", "汀步")),
    "dock": (("loading dock", "dock", "platform"), ("装货平台", "月台", "平台")),
}
GROUNDS: Lexicon = {
    "grass": (("grass", "grassy", "lawn", "meadow", "park", "garden", "field"), ("草",)),
    "sand": (("sand", "sandy", "beach", "desert", "dune"), ("沙",)),
    "snow": (("snow", "snowy", "winter", "icy"), ("雪", "冰面")),
    "dirt": (("dirt", "mud", "muddy", "farm", "trail"), ("泥", "土路", "田野")),
    "rock": (("stony", "mountain"), ("山地",)),
}
COLORS: Lexicon = {
    "red": (("red",), ("红",)),
    "green": (("green",), ("绿",)),
    "blue": (("blue",), ("蓝",)),
    "yellow": (("yellow",), ("黄",)),
    "orange": (("orange",), ("橙", "橘")),
    "purple": (("purple", "violet"), ("紫",)),
    "white": (("white",), ("白",)),
    "black": (("black",), ("黑",)),
    "gray": (("gray", "grey"), ("灰",)),
    "brown": (("brown", "wooden"), ("棕", "褐", "木头")),
    "teal": (("teal", "cyan"), ("青",)),
    "pink": (("pink",), ("粉",)),
}
# Robots a sentence may ask for, as the world's starter robot (`Starter` in __init__.py).
ROBOTS: Lexicon = {
    "picker": (
        ("mobile manipulator", "mobile arm", "robot with an arm", "lift picker", "picker", "fetch robot"),
        ("移动机械臂", "带机械臂的", "带臂", "取物机器人", "搬运机器人"),
    ),
    "go2": (("go2", "unitree"), ("宇树",)),
    "dog": (("dog", "quadruped", "puppy", "robot dog", "four-legged", "legged robot"), ("机器狗", "四足", "狗")),
    "rover": (("rover", "car", "cart", "vehicle", "wheeled robot", "agv", "forklift"), ("小车", "车", "叉车")),
    "humanoid": (("humanoid", "biped", "two-legged"), ("人形", "双足")),
    "arm": (("robot arm", "arm", "manipulator"), ("机械臂", "手臂")),
}
SIZES: Lexicon = {
    "small": (
        ("small", "tiny", "little", "compact", "cozy", "narrow"),
        ("小屋", "小公寓", "小房子", "小仓库", "小公园", "小型", "很小", "狭小"),
    ),
    "large": (
        ("large", "big", "huge", "spacious", "wide", "vast", "giant"),
        ("大型", "很大", "宽敞", "大房子", "大仓库", "大公园", "巨大"),
    ),
}

EN_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "pair": 2,
    "couple": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "few": 3,
    "several": 4,
    "some": 3,
    "many": 7,
    "lots": 7,
    "dozen": 12,
}
ZH_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "几": 3,
}
ZH_MANY = ("很多", "许多", "好多", "一堆", "大量")
ZH_MEASURE = "个张排棵块间台只把本座间条辆套盏扇面层堆"
MAX_COUNT = 12  # of any one thing a sentence can ask for


@dataclass
class Wish:
    """What a sentence asked for. Counts and colors are by concept (THINGS, ROOMS); 0 means "some, you choose"."""

    text: str
    kind: str = "house"  # house, warehouse or outdoor
    rooms: list[str] = field(default_factory=list)  # room types in the order said, repeated for "two bedrooms"
    room_count: int = 0  # "a three-room flat": rooms wanted in all, when their kinds are not all said
    counts: dict[str, int] = field(default_factory=dict)  # THINGS concept -> how many (0: some)
    colors: dict[str, str] = field(default_factory=dict)  # THINGS concept -> the color said with it
    terrain: list[str] = field(default_factory=list)
    ground: str | None = None
    size: float = 1.0
    robot: str | None = None
    understood: list[str] = field(default_factory=list)  # the words it acted on, for the summary
    rooms_of: dict[str, str] = field(default_factory=dict)  # THINGS concept -> the room said with it ("bedroom desk")
    on_of: dict[str, str] = field(default_factory=dict)  # THINGS concept -> the furniture it was said to be on

    def summary(self) -> str:
        parts = [self.kind]
        if self.rooms:
            parts.append("rooms: " + ", ".join(r.replace("_", " ") for r in self.rooms))
        if self.counts:
            things = [f"{n} {k}" if n else k for k, n in self.counts.items()]
            parts.append("with " + ", ".join(things))
        if self.terrain:
            parts.append("terrain: " + ", ".join(self.terrain))
        if self.ground:
            parts.append(f"ground: {self.ground}")
        if self.size != 1.0:
            parts.append("small" if self.size < 1 else "large")
        if self.robot:
            parts.append(f"robot: {self.robot}")
        return "; ".join(parts)


@dataclass(frozen=True)
class Hit:
    concept: str
    start: int
    end: int


MODAL_CAN = re.compile(r"\bcans?\b(?!\s+(?:you|i|we|they|it|he|she|be|make|build|do|have|get|put|give|add)\b)")


def _en_pattern(phrase: str) -> re.Pattern[str]:
    if phrase == "can":  # "a can of soda", not "can you make..."
        return MODAL_CAN
    words = r"[\s-]+".join(re.escape(w) for w in phrase.split())
    return re.compile(rf"\b{words}(?:e?s)?\b")


_PATTERNS: dict[int, list[tuple[str, re.Pattern[str] | str, bool]]] = {}


def find(text: str, lexicon: Lexicon) -> list[Hit]:
    """Where each concept of `lexicon` is said in `text` (lowercased), longest phrases first, never overlapping."""
    key = id(lexicon)
    if key not in _PATTERNS:
        entries: list[tuple[str, re.Pattern[str] | str, bool]] = []
        for concept, (en, zh) in lexicon.items():
            entries += [(concept, _en_pattern(p), True) for p in en]
            entries += [(concept, p, False) for p in zh]
        # Longest first, so "dining table" is a table and not "dining" plus a table, "robot dog" a dog.
        entries.sort(key=lambda e: -len(e[1].pattern if e[2] else e[1]))
        _PATTERNS[key] = entries
    taken: list[tuple[int, int]] = []
    hits: list[Hit] = []
    for concept, pattern, english in _PATTERNS[key]:
        spans = (
            [(m.start(), m.end()) for m in pattern.finditer(text)]
            if english
            else [(m.start(), m.start() + len(pattern)) for m in re.finditer(re.escape(pattern), text)]
        )
        for a, b in spans:
            if any(a < y and x < b for x, y in taken):
                continue
            taken.append((a, b))
            hits.append(Hit(concept, a, b))
    return sorted(hits, key=lambda h: h.start)


SKIPPABLE = (
    "of",
    "rows",
    "row",
    "stacks",
    "stack",
    "piles",
    "pile",
    "sets",
    "set",
    "kinds",
    "big",
    "small",
    "large",
    "little",
    "tall",
    "low",
    "round",
    "wooden",
    "more",
    "full",
)


def _count_before(text: str, at: int) -> int | None:
    """A number said just before position `at`: "three apples", "3 apples", "two rows of shelves", "three red cups",
    "三个苹果", "两排货架", "很多树"."""
    before = text[max(0, at - 24) : at]
    zh = re.search(rf"(\d+|[一两二三四五六七八九十几])\s*[{ZH_MEASURE}]?\s*(?:[\u4e00-\u9fff]{{0,3}}的)?\s*$", before)
    if zh and re.search(r"[\u4e00-\u9fff]", before[-3:] + text[at : at + 1]):
        token = zh[1]
        return int(token) if token.isdigit() else ZH_NUMBERS[token]
    if any(before.rstrip().rstrip("的").endswith(w) for w in ZH_MANY):
        return 7
    words = re.findall(r"[a-z]+|\d+", before)
    colors = {p for en, _ in COLORS.values() for p in en}
    for back, word in enumerate(reversed(words[-4:])):
        if word.isdigit():
            return int(word)
        if word in EN_NUMBERS:
            # "a" counts only right before the thing or a word or two that describe it ("a big red cup")
            return EN_NUMBERS[word]
        if back >= 2 or not (word in SKIPPABLE or word in colors):
            return None
    return None


def _room_count(text: str) -> int:
    """ "a three-room flat", "2 room apartment", "两室一厅", "三居室": rooms wanted in all."""
    en = re.search(r"\b(\d+|one|two|three|four|five|six)[\s-]+(?:room|bedroom)\b", text)
    if en:
        token = en[1]
        return int(token) if token.isdigit() else EN_NUMBERS[token]
    zh = re.search(r"([一两二三四五六])\s*(?:室|居室|个房间|间房)", text)
    if zh:
        n = ZH_NUMBERS[zh[1]]
        return n + (1 if "厅" in text else 0)
    return 0


def _masked(text: str, hits: list[Hit]) -> str:
    """`text` with what `hits` matched blanked out, so the next dictionary does not read it again ("书房" is a room,
    not a book; "沙发" a sofa, not sand)."""
    chars = list(text)
    for h in hits:
        chars[h.start : h.end] = " " * (h.end - h.start)
    return "".join(chars)


def read(text: str) -> Wish:
    """The world a sentence asks for. Never fails: a sentence with nothing it knows is a small house."""
    low = text.lower()
    wish = Wish(text=text)
    robots = find(low, ROBOTS)
    rest = _masked(low, robots)
    rooms = find(rest, ROOMS)
    rest = _masked(rest, rooms)
    terrain = find(rest, TERRAIN)
    rest = _masked(rest, terrain)
    things = find(rest, THINGS)
    kinds = find(_masked(rest, things), KINDS)
    if any(h.concept == "warehouse" for h in kinds):
        wish.kind = "warehouse"
    elif rooms or any(h.concept == "house" for h in kinds):
        wish.kind = "house"
    elif (
        any(h.concept == "outdoor" for h in kinds)
        or any(t.concept in ("hills", "rough", "stones") for t in terrain)
        or any(h.concept in ("tree", "rock", "bench") for h in things)
    ):
        wish.kind = "outdoor"
    elif any(h.concept in ("rack", "pallet", "conveyor") for h in things):
        wish.kind = "warehouse"
    wanted: dict[str, int] = {}  # a room said twice ("from the kitchen ... in the kitchen") is one room
    for hit in rooms:
        n = _count_before(low, hit.start) or 1
        wanted[hit.concept] = max(wanted.get(hit.concept, 0), min(n, 4))
        wish.understood.append(low[hit.start : hit.end])
    for room, n in wanted.items():
        wish.rooms += [room] * n
    wish.room_count = _room_count(low)
    for hit in things:
        room = _room_with(low, hit, rooms)
        if room is not None:
            wish.rooms_of.setdefault(hit.concept, room)
        n = _count_before(low, hit.start)
        wish.counts[hit.concept] = min(max(wish.counts.get(hit.concept, 0), n or 0), MAX_COUNT)
        color = _color_before(low, hit.start)
        if color:
            wish.colors.setdefault(hit.concept, color)
        furniture = _stands_on(low, hit, things, rooms)
        if furniture is not None:
            wish.on_of.setdefault(hit.concept, furniture)
        wish.understood.append(low[hit.start : hit.end])
    for hit in terrain:
        if hit.concept not in wish.terrain:
            wish.terrain.append(hit.concept)
        wish.understood.append(low[hit.start : hit.end])
    grounds = find(_masked(rest, things), GROUNDS)
    if grounds:
        wish.ground = grounds[0].concept
    sizes = find(low, SIZES)
    if sizes:
        wish.size = 0.8 if sizes[0].concept == "small" else 1.3
    if robots:
        wish.robot = robots[0].concept
    if wish.kind != "warehouse" and "dock" in wish.terrain:
        wish.terrain[wish.terrain.index("dock")] = "platform"  # a loading dock belongs to a warehouse
    return wish


SUPPORTS = ("table", "coffee_table", "desk", "shelf", "counter")  # furniture a small thing is said to be on
_ON = re.compile(r"\s*(?:on|onto|on top of)\s+(?:the|a|an|my)?\s*")


def _stands_on(text: str, thing: Hit, hits: list[Hit], rooms: list[Hit]) -> str | None:
    """The furniture a thing is said to be on: "a green apple on the coffee table", "a bowl on the bedroom desk" (a
    room name just before the furniture is allowed). Only English; None when it is not said."""
    for other in hits:
        if other.concept not in SUPPORTS or other.start < thing.end:
            continue
        between = text[thing.end : other.start]
        for r in rooms:  # the room said with the furniture is not a word between
            if thing.end <= r.start and r.end <= other.start:
                between = text[thing.end : r.start] + text[r.end : other.start]
        if _ON.fullmatch(between):
            return other.concept
    return None


def _room_with(text: str, thing: Hit, rooms: list[Hit]) -> str | None:
    """The room said with a thing: just before it ("the bedroom desk", "卧室的书桌", "卧室里的书桌") or just after
    ("the desk in the bedroom")."""
    for r in rooms:
        between = text[r.end : thing.start] if r.end <= thing.start else text[thing.end : r.start]
        before = r.end <= thing.start
        if before and re.fullmatch(r"\s*(?:'s\s*)?|[的里中上内]{0,2}", between):
            return r.concept
        if not before and re.fullmatch(r"\s*(?:in|from|of|inside)\s+(?:the|a|an|my)?\s*", between):
            return r.concept
    return None


def _color_before(text: str, at: int) -> str | None:
    """A color said just before a thing: "the red cup", "红色的杯子", "a big blue ball"."""
    before = text[max(0, at - 14) : at]
    found = find(before, COLORS)
    if not found:
        return None
    last = found[-1]
    between = before[last.end :]
    if re.search(r"[\u4e00-\u9fff]", before):
        return last.concept if len(between.strip(" 色的")) <= 1 else None
    return last.concept if len(re.findall(r"[a-z]+", between)) <= 1 else None
