"""Read a sentence into a plan, in English or Chinese: what body, how many legs, wheels and arms, how big, which
colors, materials and sensors.

The same reader makes a robot from scratch and changes one ("make it longer", "six legs instead of four", "add a
lidar", "another one"): sizes are factors, so "tall" in a first sentence and "taller" in a later one both multiply
the height. What it cannot build it refuses with what to say instead; what it builds differently (eight legs) it
builds and notes.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import replace
from typing import Any

from robotforge.design import components
from robotforge.design.generator.plan import NATURAL, Plan, clamp

NUMBERS = {
    "a": 1, "an": 1, "one": 1, "single": 1, "two": 2, "pair": 2, "dual": 2, "twin": 2, "both": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fourteen": 14, "sixteen": 16, "一": 1, "单": 1, "二": 2, "两": 2, "双": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十二": 12,
}  # fmt: skip
NUM = r"(\d+|" + "|".join(sorted((re.escape(k) for k in NUMBERS), key=len, reverse=True)) + r")"
COUNT = r"(\d+|" + "|".join(k for k in NUMBERS if k.isascii() and k not in ("a", "an", "single", "pair")) + r")\b"

# Colors beyond the builder's names, as hex: what people say about robots.
EXTRA_COLORS = {
    "gold": "#c8a043", "golden": "#c8a043", "silver": "#b8bec6", "navy": "#23395d", "olive": "#6b7a3a",
    "lime": "#9ccc3c", "mint": "#5cc8a8", "crimson": "#c8323c", "cyan": "#2bb7d6", "magenta": "#c2408f",
    "beige": "#d9c9a3", "ivory": "#e4e0d6", "khaki": "#b8a46a", "maroon": "#7a2430", "coral": "#ef6f5e",
    "turquoise": "#2fb5a5", "charcoal": "#2b3038", "graphite": "#2b3038", "sky": "#6fb3e6", "lavender": "#a58fd8",
    "红": "red", "橙": "orange", "黄": "yellow", "绿": "green", "蓝": "blue", "紫": "purple", "白": "white",
    "黑": "black", "灰": "gray", "棕": "brown", "青": "teal", "粉": "pink", "金": "#c8a043", "银": "#b8bec6",
}  # fmt: skip
MATERIALS = {
    "carbon": "carbon", "碳": "carbon", "aluminum": "aluminum", "aluminium": "aluminum", "metal": "aluminum",
    "alloy": "aluminum", "steel": "aluminum", "铝": "aluminum", "金属": "aluminum", "plastic": "pla", "pla": "pla",
    "printed": "pla", "塑料": "pla", "rubber": "rubber", "橡胶": "rubber",
}  # fmt: skip
# Styles name a color scheme (main, accent) when no color is said.
STYLES = {
    "military": ("#6b7a3a", "#c9b38a"), "army": ("#6b7a3a", "#c9b38a"), "space": ("#e4e0d6", "#c8a043"),
    "mars": ("#e4e0d6", "#c8a043"), "lunar": ("#e4e0d6", "#c8a043"), "industrial": ("#f2b134", "#1d2026"),
    "warehouse": ("#f2b134", "#1d2026"), "factory": ("#f2b134", "#1d2026"), "police": ("#2f6fd6", "#e9ecef"),
    "fire": ("#c8323c", "#8c939c"), "medical": ("#e9ecef", "#1f8a8a"), "军": ("#6b7a3a", "#c9b38a"),
    "太空": ("#e4e0d6", "#c8a043"), "工业": ("#f2b134", "#1d2026"),
}  # fmt: skip

# How a robot is drawn (assets styles): "a rugged robot dog", "a lab-grade arm", "a cobot", "a hobby servo robot".
# An English style word counts when it describes a robot ("lab robot", "rugged six-legged robot", "lab-grade"), not
# when it merely turns up in a use ("research papers", "a laboratory sample tray", "a drink maker").
_ROBOT = (
    r"(?:robots?|bots?|arms?|manipulators?|grippers?|hands?|cobots?|dogs?|quadrupeds?|hexapods?|humanoids?|rovers?|"
    r"walkers?|spiders?|snakes?|legged|machines?|vehicles?|build|design)"
)


def _describing(words: str) -> str:
    """The style word before a robot noun ("rugged six-legged robot"), as a grade ("lab-grade") or as the change
    asked for ("make it rugged")."""
    return (
        rf"\b(?:{words})(?:[ -]\w+){{0,2}}[ -]{_ROBOT}\b"
        rf"|\b(?:{words})[ -](?:grade|class|style|look|finish)\b"
        rf"|\b(?:make|turn|restyle|redo)\s+(?:it|them|this|the robot)\s+(?:more\s+|a bit more\s+)?(?:{words})\b"
    )


LOOK_STYLES = {
    "rugged": _describing(r"rugged|ruggedi[sz]ed|tough|all[ -]terrain|heavy[ -]duty|outdoor") + r"|越野|野外|坚固",
    "lab": _describing(r"lab|laboratory|research") + r"|实验室|科研",
    "hobby": _describing(r"hobby|hobbyist|diy") + r"|玩具|创客",
    "cobot": r"\b(cobots?|collaborative)\b|协作",
}

# Body words, strongest first: "a snake with a gripper" is a snake, "a dog with an arm" a walker.
NOT_BUILT = (
    (r"\b(fly|flying|flies|drone|quadcopter|copter|propellers?|plane|aircraft)\b|飞|无人机",
     'flying robots are the drone template: add_robot(name, template="drone")'),
    (r"\b(boat|ship|swim|swims|swimming|submarine|underwater|fish)\b|船|水下|游泳|鱼",
     'robots for water are the boat and rov templates: add_robot(name, template="boat") or template="rov"'),
)  # fmt: skip
WHEELED = (
    r"\b(wheels?|wheeled|rover|car|cart|truck|buggy|vehicle|courier|delivery|drives?|driving|tank|forklift|"
    r"mobile (manipulator|base|robot))\b|轮|车|快递|送货|配送|移动"
)
BODY_WORDS: tuple[tuple[str, str], ...] = (
    ("snake", r"\b(snake|serpent|worm|eel|slither\w*)\b|蛇"),
    ("balancer", r"\b(segway|self balancing|balancing|balances|balancer|unicycle)\b|平衡"),
    ("humanoid", r"\b(humanoid|android|biped|bipedal|person|man|woman|human)\b(?!\s+(sized?|scale|height|tall))"
     r"|人形|双足|人型"),
    ("legged", r"\b(legs?|legged|walk\w*|(?<!hot )dog|puppy|cat|horse|pony|cow|goat|lizard|quadruped|hexapod|spider|"
     r"insect|ant|beetle|bug|crab|mantis|centaur|scorpion|cockroach|critter|animal)\b|腿|足|狗|猫|马|蜘蛛|昆虫|蚂蚁|螃蟹|螳螂|"
     r"半人马|蝎子|甲虫"),
    ("wheeled", WHEELED),
    ("arm", r"\b(arms?|cobots?|manipulator|tentacle|gripper|claw)\b|机械臂|手臂|触手|夹爪"),
)  # fmt: skip
SIX_LEGS = (
    r"\b(hexapod|spider|insect|ant|beetle|bug|crab|mantis|scorpion|cockroach)\b|蜘蛛|昆虫|蚂蚁|螃蟹|螳螂|蝎子|甲虫|六足"
)
FOUR_LEGS = r"\b(dog|puppy|cat|horse|pony|cow|goat|lizard|quadruped|centaur)\b|狗|猫|马|四足|半人马"
NEGATION = (
    r"(\bno|\bwithout|\bremove|\btake off|\bdrop|\bminus|\bnot|\bnever|\bdon'?t|\bdoesn'?t|\bless|去掉|不要|没有|无|"
    r"拆掉)"
)
# What may stand between a negation and what it takes away: "does not have an arm", "without any lidar".
NEGATED = r"\s*(have |has |need |want |with |make it |paint it |be )?(the |its |any |all |an? |one )?"

# How much a word changes a factor; "very" and "extra" take it further.
LEGS = r"leg(s|ged)?\b"  # "long legs", "long legged" ("long-legged": _normal makes the hyphen a space)
SIZES: tuple[tuple[str, str, float], ...] = (
    ("height", rf"\blonger {LEGS}|\blegs? (\w+ ){{0,2}}longer\b|腿更长|腿长一点|腿长一些", 1.25),
    ("height", rf"\blong {LEGS}|长腿|腿长", 1.4),
    ("height", rf"\bshorter {LEGS}|\blegs? (\w+ ){{0,2}}shorter\b|腿更短|腿短一点|腿短一些", 0.83),
    ("height", rf"\b(short|stubby|stumpy) {LEGS}|短腿|腿短", 0.8),
    ("height", r"\btaller\b|更高|高一点|高一些", 1.25),
    ("height", r"\b(tall|towering|high)\b|高大|高个|很高|高的|高", 1.4),
    ("height", r"\b(squat|low|lower)\b|矮", 0.8),
    ("length", r"\blonger\b|更长|长一点|长一些|加长", 1.3),
    ("length", r"\b(long|elongated|stretched)\b|长", 1.4),
    ("length", r"\bshorter\b|短一点|短一些|更短", 0.8),
    ("length", r"\b(short|stubby|compact)\b|短", 0.8),
    ("width", r"\b(wider|broader)\b|更宽", 1.2),
    ("width", r"\b(wide|broad|chunky|fat|thick|stout)\b|宽|粗|胖", 1.3),
    ("width", r"\b(narrower|slimmer|thinner)\b|更窄|更细", 0.85),
    ("width", r"\b(narrow|slim|thin|slender|skinny)\b|窄|细|瘦", 0.8),
    ("size", r"\b(bigger|larger)\b|更大|大一点|大一些", 1.2),
    ("size", r"\b(big|large|huge|giant|heavy|massive)\b|大|巨", 1.35),
    ("size", r"\b(smaller|tinier)\b|更小|小一点|小一些", 0.83),
    ("size", r"\b(small|tiny|mini|little|baby|desktop)\b|小|迷你|微型", 0.75),
)
INTENSE = r"\b(very|extra|super|really)\s+$|非常$|特别$|超$"
SOFTENED = r"\b(not so|not too|not as|not that|less|no longer)\s+$|不那么$|不太$|没那么$"  # "make it not so long"
UNITS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0, "米": 1.0, "cm": 0.01, "厘米": 0.01,
         "mm": 0.001, "毫米": 0.001}  # fmt: skip
DIMENSION = {"long": "length", "长": "length", "tall": "height", "high": "height", "高": "height", "wide": "width",
             "宽": "width"}  # fmt: skip
# "again" is not here: "make it longer again" is the same robot, longer.
RESEED = r"\b(another|different|new one|variation|variant|reroll|re roll|shuffle)\b|换一个|再来一个|另一个|重新"
MAX_SENTENCES = 20  # a plan keeps the last ones it was made from


class SentenceError(ValueError):
    """A sentence that does not make a robot, with what to say instead."""


def seed_of(sentence: str) -> int:
    """The same sentence makes the same robot."""
    return zlib.crc32(" ".join(sentence.lower().split()).encode("utf-8")) % 100_000


def read(sentence: str, plan: Plan | None = None, seed: int | None = None) -> tuple[Plan, list[str]]:
    """The plan a sentence describes (plan None), or `plan` changed as the sentence says, with notes."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise SentenceError('describe the robot in a sentence, e.g. "a six-legged robot with two arms"')
    text = _normal(sentence)
    notes: list[str] = []
    for pattern, hint in NOT_BUILT:
        if re.search(pattern, text):
            raise SentenceError(f"generated robots walk, drive, slither or reach; {hint}")
    # "six legs instead of four", "six legs, not four": only the six counts. A bare "not" ("does not have an arm")
    # stays for the negations to read.
    text = re.sub(r"\b(instead of|rather than)\s+(a |an )?" + NUM + r"?\s*\w*", " ", text)
    text = re.sub(r"\bnot\s+" + COUNT + r"\s*\w*", " ", text)
    text = re.sub(r"(而不是|代替|替换)\S*", " ", text)
    body = _body(text)
    fresh = plan is None
    if fresh:
        if body is None:
            raise SentenceError(
                f"I could not tell what kind of robot {sentence.strip()!r} is: say legs (four or six), wheels, a "
                'snake, an arm, a humanoid or a self-balancing robot, e.g. "a six-legged robot with two arms" or "a '
                'tall wheeled courier with a lidar"'
            )
        plan = Plan(body, seed=seed_of(sentence) if seed is None else seed)
    assert plan is not None
    changes: dict[str, Any] = {}
    # An arm said in a change is one to add ("add an arm"), not a new body: the dog keeps its legs.
    if not fresh and body is not None and body != plan.body and body != "arm":
        changes["body"] = body
        notes.append(f"it is {_a(body)} {body} robot now")
    if not fresh and re.search(RESEED, text):
        changes["seed"] = plan.seed + 1 if seed is None else seed
    elif not fresh and seed is not None:
        changes["seed"] = seed
    target = changes.get("body", plan.body)
    changes |= _counts(text, plan, target, fresh, notes)
    changes |= _sizes(text, plan, target, notes, fresh)
    changes |= _looks(text)
    changes |= _equipment(text, plan, target)
    if fresh and target == "wheeled" and re.search(r"\b(courier|delivery|cargo|parcel|mail)\b|快递|送货|配送", text):
        changes.setdefault("cargo", True)
    if fresh and target == "arm":
        changes.setdefault("camera", False)  # a camera on its pedestal only when asked for
    if fresh and re.search(r"\b(spider|round|disc|disk|circular)\b|蜘蛛|圆", text):
        changes["round"] = True
    if target == "legged" and re.search(WHEELED, text):
        notes.append(
            "it walks on legs; for legs that end in driven wheels add the wheel_dog design (add_robot(name, "
            'design="wheel_dog")), or say "wheels instead of legs" for a rover'
        )
    if not fresh and not changes:
        raise SentenceError(
            f'I could not tell what to change from {sentence.strip()!r}: say e.g. "make it longer", "six legs '
            'instead of four", "add an arm", "add a lidar", "make it red", "bigger" or "another one" (a '
            "new variation)"
        )
    sentences = (*plan.sentences, sentence.strip())[-MAX_SENTENCES:]
    new, clamped = clamp(replace(plan, **changes, sentences=sentences))
    return new, notes + clamped


def _normal(sentence: str) -> str:
    text = sentence.lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"(\d)\s*(?=(legs?|wheels?|arms?|segments?|joints?|axis|dof)\b)", r"\1 ", text)
    return " " + re.sub(r"\s+", " ", text) + " "


def _a(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


def _said(text: str, pattern: str) -> bool:
    """The pattern is in the text, and not negated: "not a dog but a rover" does not say a dog."""
    return any(
        not re.search(NEGATION + NEGATED + r"$", text[max(0, m.start() - 24) : m.start()])
        for m in re.finditer(pattern, text)
    )


def _negated(text: str, pattern: str) -> bool:
    """The pattern is in the text only negated ("without legs")."""
    return re.search(pattern, text) is not None and not _said(text, pattern)


def _body(text: str) -> str | None:
    found = [body for body, pattern in BODY_WORDS if _said(text, pattern)]
    if not found:
        return None
    if "humanoid" in found and "wheeled" in found:
        return "balancer"  # legs that end in wheels balance like a Segway
    if (
        "wheeled" in found
        and re.search(r"\b(two|2) (wheels?|wheeled)\b|两轮|双轮", text)
        and not re.search(r"\bcasters?\b|万向", text)
        and "legged" not in found
    ):
        return "balancer"
    if "legged" in found and re.search(r"\b(two|2) (legs?|legged)\b|两条腿|两腿", text) and "wheeled" not in found:
        return "humanoid"
    return found[0]


def _number(word: str) -> int:
    return int(word) if word.isdigit() else NUMBERS[word]


ARTICLES = ("a", "an", "single")


def _count(text: str, noun: str, articles: bool = True) -> int | None:
    """The number said before `noun` ("six legs", "6 legged", "六条腿"), or None.

    A real count wins over an article before the same noun: "a wheeled robot with six wheels" is six, not one.
    """
    # ASCII only behind it: Chinese characters count as word characters, and a count mostly follows one (一辆六轮).
    found: list[tuple[int, bool]] = []  # (number, an article)
    for m in re.finditer(rf"(?<![0-9A-Za-z.]){NUM}\s*(?:条|只|个|节|根|台|辆)?\s*(?:{noun})", text):
        if re.search(NEGATION + r"\s*$", text[max(0, m.start() - 12) : m.start()]):
            found.append((0, m.group(1) in ARTICLES))
        else:
            found.append((_number(m.group(1)), m.group(1) in ARTICLES))
    if not articles:
        found = [f for f in found if not f[1]]  # "a legged robot" says nothing about how many legs
    if not found:
        return None
    return next((n for n, article in found if not article), found[0][0])


def _counts(text: str, plan: Plan, body: str, fresh: bool, notes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    legs = _count(text, r"legs?\b|legged\b|feet\b|footed\b|腿|足", articles=False)
    if legs is not None and legs > 2:
        out["legs"] = legs
    elif body == "legged" and (legs is not None or _negated(text, r"\blegs?\b|腿")):
        out["legs"] = 4
        said = "no legs" if not legs else f"{legs} {'leg' if legs == 1 else 'legs'}"
        notes.append(f"{said} became 4 legs: the built-in gaits walk four legs (a trot) or six (a tripod)")
    elif re.search(SIX_LEGS, text):
        out["legs"] = 6
    elif re.search(FOUR_LEGS, text):
        out["legs"] = 4
    elif not fresh and re.search(r"\bmore legs\b|更多腿", text):
        out["legs"] = 6
    elif not fresh and re.search(r"\b(fewer|less) legs\b|少点腿", text):
        out["legs"] = 4
    # "a wheeled courier" says nothing about how many wheels: counts of one are articles
    wheels = _count(text, r"wheels?\b|wheeled\b|轮")
    if wheels is not None and wheels > 1:
        out["wheels"] = wheels
    segments = _count(text, r"segments?\b|sections?\b|links?\b|节")
    if segments is not None and segments > 1:
        out["segments"] = segments
    elif not fresh and re.search(r"\bmore segments\b", text):
        out["segments"] = plan.segments + 2
    joints = _count(text, r"joints?\b|axis\b|axes\b|dof\b|自由度|轴")
    if joints is not None and joints > 1:
        out["joints"] = joints
    arms = _arms(text, plan, body, fresh)
    if arms is not None:
        out["arms"] = arms
    if re.search(r"\bcentaur\b|半人马|\bmantis\b|螳螂", text):
        out.setdefault("arms", 2)
    if re.search(r"\btentacle\b|触手", text):
        out["tentacle"] = True
    return out


def _arms(text: str, plan: Plan, body: str, fresh: bool) -> int | None:
    noun = (
        r"(?:robot |robotic |mechanical |gripper |grabbing )?"
        r"(?:arms?\b|hands?\b|grippers?\b|claws?\b|pincers?\b|手臂|臂|手|胳膊|夹爪)"
    )
    if not fresh and re.search(
        r"\b(remove|drop|take off|minus|lose)\s+(one|an|a|1) (arm|hand|gripper)\b|去掉一(个|条|只)?(手臂|臂|手)", text
    ):
        return max(0, plan.arms - 1)
    if re.search(NEGATION + NEGATED + r"(?:arms?|hands?|grippers?|claws?)\b|(去掉|不要|没有)(手|臂|夹爪)", text):
        return 0
    if not fresh and re.search(
        r"\b(add|another|extra|one more|second)\s+(an? )?(arm|hand|gripper)\b|再加.?(臂|手)", text
    ):
        return plan.arms + 1
    count = _count(text, noun)
    if count is not None:
        return count
    if re.search(r"\bdual arm|\btwo armed|\bdouble arm|双臂", text):
        return 2
    if re.search(
        r"\b(arm|hand|gripper|claw|pincer|jaws?|grabber|manipulator|suction|magnet)\b|机械臂|夹爪|爪|手|吸盘", text
    ):
        return max(1, plan.arms if not fresh else 1)
    if fresh and body == "arm":
        return 1
    return None


def _sizes(text: str, plan: Plan, body: str, notes: list[str], fresh: bool) -> dict[str, float]:
    """Size words first, then meters: "a small robot dog 50 cm long" is 50 cm long, whatever "small" did."""
    out: dict[str, float] = {}
    used: list[tuple[int, int]] = []
    metric: list[tuple[str, float]] = []
    unit = r"(mm|cm|m|meters?|metres?|米|厘米|毫米)"
    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*{unit}\s*(long|tall|high|wide|长|高|宽)", text):
        metric.append((DIMENSION[m.group(3)], float(m.group(1)) * UNITS[m.group(2)]))
        used.append(m.span())
    for m in re.finditer(rf"(long|tall|high|wide|长|高|宽)\s*(\d+(?:\.\d+)?)\s*{unit}", text):
        metric.append((DIMENSION[m.group(1)], float(m.group(2)) * UNITS[m.group(3)]))
        used.append(m.span())
    for what, pattern, factor in SIZES:
        for m in re.finditer(pattern, text):
            if any(a <= m.start() < b for a, b in used):
                continue  # "longer" in "longer legs" was read with the legs
            used.append(m.span())
            before = text[max(0, m.start() - 12) : m.start()]
            strong = factor ** (1.4 if re.search(INTENSE, before) else 1.0)
            if re.search(SOFTENED, before):
                if fresh:
                    continue  # "a not so tall robot": the usual height
                strong = 1 / strong  # "make it not so long": shorter
            # A humanoid or a balancer is all height: a short one is a low one.
            what_here = "height" if what == "length" and body in ("humanoid", "balancer") else what
            out[what_here] = round(out.get(what_here, getattr(plan, what_here)) * strong, 3)
    size = out.get("size", plan.size)
    for what, meters in metric:
        scale = 1.0 if what == "length" and body == "arm" else size  # an arm's reach is in meters already
        out[what] = round(meters / (NATURAL[body][what] * scale), 3)
    if "height" in out and body == "snake":
        notes.append("a snake lies flat: its height stays")
        out.pop("height")
    return out


def _color(word: str) -> str | None:
    """A builder color name or a hex code for a color word."""
    word = components.COLOR_SPELLINGS.get(word, word)
    return word if word in components.COLORS else EXTRA_COLORS.get(word)


def _looks(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    names = [*components.COLORS, *components.COLOR_SPELLINGS, *EXTRA_COLORS]
    pattern = "|".join(sorted((re.escape(n) for n in names), key=len, reverse=True))
    colors: list[str] = []
    for m in re.finditer(rf"(?<![a-z])({pattern})(?![a-z])", text):
        if m.group(1) in ("sky",) and not re.match(r"\s*blue", text[m.end() :]):
            continue
        if re.search(NEGATION + NEGATED + r"$", text[max(0, m.start() - 24) : m.start()]):
            continue  # "don't make it red", "not red"
        color = _color(m.group(1))
        if color is not None and color not in colors:
            colors.append(color)
    for m in re.finditer(r"#[0-9a-f]{6}\b|#[0-9a-f]{3}\b", text):
        colors.append(m.group(0))
    if colors:
        out["color"] = colors[0]
    if len(colors) > 1:
        out["accent"] = colors[1]
    if not colors:
        for word, (main, accent) in STYLES.items():
            if re.search(rf"\b{word}\b" if word.isascii() else word, text):
                out |= {"color": main, "accent": accent}
                break
    for word, material in MATERIALS.items():
        if re.search(rf"\b{word}\b" if word.isascii() else word, text):
            out["material"] = material
            break
    for style, pattern in LOOK_STYLES.items():
        if re.search(pattern, text) and not re.search(
            NEGATION + NEGATED + r"$", text[: re.search(pattern, text).start()][-24:]
        ):
            out["style"] = style
            break
    return out


def _has(text: str, pattern: str) -> bool | None:
    """True when the sentence asks for it, False when it says without it, None when it says nothing."""
    m = re.search(pattern, text)
    if m is None:
        return None
    return not re.search(NEGATION + NEGATED + r"$", text[max(0, m.start() - 24) : m.start()])


def _equipment(text: str, plan: Plan, body: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    lidar = _has(text, r"\b(lidar|laser|scanner|rangefinder)\b|激光|雷达")
    if lidar is not None:
        out["lidar"] = lidar
    camera = _has(text, r"\b(cameras?|eyes|vision|sees)\b|相机|摄像|视觉|眼睛")
    if camera is not None:
        out["camera"] = camera
    if re.search(r"\bblind\b", text):
        out["camera"] = False
    head = _has(text, r"\bhead\b|头")
    if re.search(r"\bheadless\b|无头", text):
        head = False
    if head is not None and not (body == "snake" and head):
        out["head"] = head
    if re.search(r"\b(suction|vacuum|sucker)\b|吸盘", text):
        out["hand"] = "suction"
    elif re.search(r"\b(magnet|magnetic|electromagnet)\b|磁", text):
        out["hand"] = "magnet"
    elif re.search(r"\b(gripper|claw|pincers?|fingers|jaws)\b|夹爪|爪", text):
        out["hand"] = "gripper"
    cargo = _has(text, r"\b(cargo|box|bin|parcel|courier|delivery)\b|货箱|快递")
    if cargo is not None and body == "wheeled":
        out["cargo"] = cargo
    return out
