"""A generated robot's plan: the few numbers a sentence decides, kept so a later sentence can change them.

A plan is parametric and seeded. The sentence picks the body (legs, wheels, a snake, an arm...), the counts (legs,
wheels, arms, segments) and the proportions as factors of that body's own default size; the seed picks what the
sentence left open: the color scheme, the materials and the small details, so another seed is another robot from
the same sentence and the same seed is the same robot again.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

Body = Literal["legged", "wheeled", "balancer", "snake", "arm", "humanoid"]
Hand = Literal["gripper", "suction", "magnet"]
BODIES: tuple[str, ...] = ("legged", "wheeled", "balancer", "snake", "arm", "humanoid")

# What each count and factor may be. Outside these the gait, the drive or the servos stop working, and the
# generator says so (see words.py) instead of building a robot that cannot move.
LEGS = (4, 6)
WHEELS = (2, 4, 6)
MAX_ARMS = 2
SEGMENTS = (8, 14)  # 4 or 5 segments slithered 23 to 46 % of the way, 6 or 7 41 to 63 %, 6 without jaws 51 % (median)
JAW_SEGMENTS = 12  # with jaws at most: 14 segments and jaws did under half the way in 30 of 106 runs, 12 in 7 of 108
JOINTS = (3, 6)  # a standing arm's joints, the gripper not counted
FACTOR = (0.6, 1.8)  # length, width and height, each times the body's own default
SIZE = (0.6, 1.6)  # the whole robot, times its default
# A balancer's size cubed times its length, width and height at most this: x1.6 with x1.4 long and tall, or bulkier,
# fell over driving off; x1.6 x1.4 x1.4 x1.0 drove (2026-10-19 sweep).
BALANCER_BULK = 8.0
TWO_ARMS_SIZE = 0.8  # a robot with two arms at least this size
# A walker's legs (size times height) at least this long: shorter, a hexapod's feet are not 5 cm under its body, which
# the gait needs to tell legs from the rest (a 0.6 size robot with 0.6 legs was not seen as a walker at all).
MIN_LEG = 0.6
SLIM = 0.55  # a quadruped's width factor at least this share of its length factor
ARMED_SMALL_LEGS = 1.4  # the longest legs (factor) of a quadruped smaller than x0.8 with an arm
MIN_LEG_TWO_ARMS = 0.7  # a two-armed dog's: on x0.6 legs they walked 17 to 71 % of the way, on x0.7 79 % and more
# A snake's spine (first joint to last) at most this long: 1.7 to 2.5 m snakes did under half the way in 5 s at 0.1 m/s
# and 0.03 m/s after 20 s; up to 1.6 m every one without jaws did (2026-10-19 sweep).
SNAKE_SPINE = 1.6  # m
SNAKE_SEGMENT = 0.08  # m, one segment at factor 1
SNAKE_CHAIN = 17  # segments times the length factor at most: 14 segments x1.4 long and x1.4 thick, x0.6 big, moved 9 %
SNAKE_CHAIN_MIN = 5.6  # and at least this: 8 segments x0.6 long did 47 to 49 % of the way, x0.7 and more all did more
SNAKE_GIRTH = 1.4  # a snake's size times its width at most this
HUMANOID_HEIGHT = 0.45  # m, the humanoid template's height at factor 1
# m: at 0.7 m some variations fell over walking at 0.15 m/s; a 0.3 m one walked 49 % of the way in 5 s, 0.32 m 53 %,
# 0.36 m 57 % (2026-10-19 sweep)
HUMANOID_RANGE = (0.36, 0.65)
# The default size of each body at factor 1 (meters), to turn "1 m long" into a factor.
NATURAL = {
    "legged": {"length": 0.3, "height": 0.26, "width": 0.14},
    "wheeled": {"length": 0.36, "height": 0.3, "width": 0.24},
    "balancer": {"length": 0.07, "height": 0.37, "width": 0.14},
    "snake": {"length": 0.73, "height": 0.05, "width": 0.06},
    "arm": {"length": 0.45, "height": 0.12, "width": 0.16},
    "humanoid": {"length": 0.1, "height": 0.45, "width": 0.2},
}


@dataclass(frozen=True)
class Plan:
    body: Body
    legs: int = 4
    wheels: int = 4
    arms: int = 0
    hand: Hand = "gripper"
    segments: int = 8
    joints: int = 5
    length: float = 1.0
    width: float = 1.0
    height: float = 1.0
    size: float = 1.0
    head: bool = True
    camera: bool = True
    lidar: bool = False
    cargo: bool = False  # a courier's cargo box
    round: bool = False  # a round body (a spider) instead of a long one (an insect)
    tentacle: bool = False  # a standing arm of many short segments
    color: str | None = None
    accent: str | None = None
    material: str | None = None
    seed: int = 0
    sentences: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """What world.yaml keeps: only what differs from the defaults, plus the body, the seed and the sentences."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("body", "seed") or value != f.default:
                out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"a generated robot's plan has unknown fields {unknown}; known: {sorted(known)}")
        if "body" not in data:
            raise ValueError("a generated robot's plan needs its body (legged, wheeled, ...)")
        values = dict(data)
        if "sentences" in values:
            values["sentences"] = tuple(values["sentences"])
        try:
            return clamp(cls(**values))[0]
        except TypeError as exc:  # a number written as a word, a list for a count
            raise ValueError(f"a generated robot's plan has a value of the wrong type: {exc}") from None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(plan: Plan) -> tuple[Plan, list[str]]:
    """The plan with every number brought inside what still builds and moves, and a note for each one changed."""
    notes: list[str] = []
    changes: dict[str, Any] = {}

    def nearest(name: str, value: int, allowed: tuple[int, ...], why: str) -> None:
        if value not in allowed:
            best = min(allowed, key=lambda a: (abs(a - value), -a))
            changes[name] = best
            notes.append(f"{value} {name} became {best}: {why}")

    def bounded(name: str, value: float, low: float, high: float, why: str) -> None:
        if not low <= value <= high:
            best = min(max(value, low), high)
            changes[name] = round(best, 3)
            notes.append(f"{name} x{value:.2f} became x{best:.2f}: {why}")

    def factor(name: str, value: float) -> None:
        """A length, width or height factor: the note names the body and the size in meters it stops at."""
        low, high = FACTOR
        if low <= value <= high:
            return
        best = min(max(value, low), high)
        changes[name] = round(best, 3)
        meters = NATURAL[plan.body][name] * min(max(plan.size, SIZE[0]), SIZE[1]) * best
        more = value > high
        fails = (
            "it crawls instead of slithering"
            if plan.body == "snake"
            else "its servos cannot hold it up or it tips over"
        )
        notes.append(
            f"{name} x{value:.2f} became x{best:.2f}: a {plan.body} robot this size is {meters:.2f} m {name} at "
            f"{'most' if more else 'least'}; {'more' if more else 'less'}, {fails}"
        )

    if plan.body not in BODIES:
        raise ValueError(f"unknown body {plan.body!r}; one of {', '.join(BODIES)}")
    if plan.body == "legged":
        nearest("legs", plan.legs, LEGS, "the built-in gaits walk four legs (a trot) or six (a tripod)")
    if plan.body == "wheeled":
        nearest("wheels", plan.wheels, WHEELS, "a rover drives on 2 (with casters), 4 or 6 wheels")
    if not 0 <= plan.arms <= MAX_ARMS:
        changes["arms"] = min(max(plan.arms, 0), MAX_ARMS)
        notes.append(f"{plan.arms} arms became {changes['arms']}: a generated robot has at most two arms")
    lo, hi = SEGMENTS
    if not lo <= plan.segments <= hi:
        changes["segments"] = min(max(plan.segments, lo), hi)
        notes.append(f"{plan.segments} segments became {changes['segments']}: a snake has {lo} to {hi}")
    lo, hi = JOINTS
    if not lo <= plan.joints <= hi:
        changes["joints"] = min(max(plan.joints, lo), hi)
        notes.append(f"{plan.joints} arm joints became {changes['joints']}: an arm has {lo} to {hi}")
    for name in ("length", "width", "height"):
        factor(name, getattr(plan, name))
    bounded("size", plan.size, *SIZE, "further and the servos cannot hold it up")
    arms = changes.get("arms", plan.arms)
    if arms >= 2 and changes.get("size", plan.size) < TWO_ARMS_SIZE:
        # Their grippers stay full size: on x0.6 robots they hit the base, or the robot tipped over (2026-10-19).
        was = changes.get("size", plan.size)
        changes["size"] = TWO_ARMS_SIZE
        notes.append(
            f"size x{was:.2f} became x{TWO_ARMS_SIZE:.2f}: two arms need a bigger robot (their grippers stay full size)"
        )
    size, height = changes.get("size", plan.size), changes.get("height", plan.height)
    shortest = MIN_LEG_TWO_ARMS if plan.body == "legged" and plan.legs == 4 and arms >= 2 else MIN_LEG
    if plan.body == "legged" and size * height < shortest:
        changes["height"] = round(shortest / size, 3)
        notes.append(f"legs x{height:.2f} became x{changes['height']:.2f}: shorter legs do not lift its feet to walk")
    dog = plan.body == "legged" and plan.legs == 4
    length, width = changes.get("length", plan.length), changes.get("width", plan.width)
    if dog and width < SLIM * length:
        # Long and narrow, a dog rolls from foot to foot: x1.4 long and x0.6 wide walked 34 to 45 % of the way.
        changes["width"] = width = round(SLIM * length, 3)
        notes.append(f"width x{plan.width:.2f} became x{width:.2f}: a dog this long and narrower rolls over its feet")
    if dog and arms and changes.get("size", plan.size) < 0.8 and changes.get("height", height) > ARMED_SMALL_LEGS:
        # A small dog on long legs with an arm on its back is top-heavy: x0.6 size, x1.8 legs did not walk at all.
        changes["height"] = ARMED_SMALL_LEGS
        notes.append(
            f"legs x{height:.2f} became x{ARMED_SMALL_LEGS:.2f}: a small dog with an arm on longer legs topples "
            "(a bigger dog can have them)"
        )
    wide = MIN_LEG + max(0.0, width - 1.0)  # a six-legged body x1.1 wide walked on legs x0.7, x1.4 wide not on x0.8
    if plan.body == "legged" and plan.legs == 6 and not plan.round and changes.get("height", height) < wide:
        changes["height"] = round(wide, 3)
        notes.append(f"legs x{height:.2f} became x{wide:.2f}: on a body x{width:.2f} wide shorter legs drag its belly")
    if plan.body == "snake" and changes.get("size", plan.size) * changes.get("width", plan.width) > SNAKE_GIRTH:
        # A snake's segments grow with its size and its girth together: x1.6 by x1.4 slithered under half the way in
        # 32 of 32 variations, x1.6 by x1.0 in 15 of 32, x1.4 in 4 of 32 (2026-10-19 sweep).
        size_now, width_now = changes.get("size", plan.size), changes.get("width", plan.width)
        width_new = max(FACTOR[0], math.floor(SNAKE_GIRTH / size_now * 1000) / 1000)
        size_new = min(size_now, math.floor(SNAKE_GIRTH / width_new * 1000) / 1000)
        changes |= {"width": width_new, "size": size_new}
        notes.append(
            f"a snake x{size_now:.2f} big and x{width_now:.2f} thick became x{size_new:.2f} and x{width_new:.2f}: "
            f"thicker than x{SNAKE_GIRTH} in all, it is too heavy to slither"
        )
    if plan.body == "snake":
        segments, length = changes.get("segments", plan.segments), changes.get("length", plan.length)
        per_length = (segments - 1) * SNAKE_SEGMENT * size * changes.get("width", plan.width)
        if per_length * length > SNAKE_SPINE:
            best = max(FACTOR[0], math.floor(SNAKE_SPINE / per_length * 1000) / 1000)  # stays under on a clamp again
            changes["length"] = best
            notes.append(
                f"length x{length:.2f} became x{best:.2f}: a snake longer than {SNAKE_SPINE} m crawls (fewer "
                "segments or a smaller snake can be longer)"
            )
    if plan.body == "snake":
        segments, length = changes.get("segments", plan.segments), changes.get("length", plan.length)
        if segments * length < SNAKE_CHAIN_MIN:
            best = math.ceil(SNAKE_CHAIN_MIN / segments * 1000) / 1000
            changes["length"] = best
            notes.append(
                f"length x{length:.2f} became x{best:.2f}: {segments} segments this short are a chain too stubby to "
                "slither (more segments can be shorter)"
            )
        if segments * length > SNAKE_CHAIN:
            best = math.floor(SNAKE_CHAIN / segments * 1000) / 1000
            changes["length"] = best
            notes.append(
                f"length x{length:.2f} became x{best:.2f}: {segments} segments this long are a chain too heavy to "
                "slither (fewer segments can be longer)"
            )
    if plan.body == "balancer":
        bulk = changes.get("size", plan.size) ** 3 * plan.length * plan.width * changes.get("height", plan.height)
        if bulk > BALANCER_BULK:
            was = changes.get("size", plan.size)
            changes["size"] = math.floor(was * (BALANCER_BULK / bulk) ** (1 / 3) * 1000) / 1000
            notes.append(
                f"size x{was:.2f} became x{changes['size']:.2f}: a balancer this big and long, wide or tall is too "
                "heavy for its wheel motors to catch"
            )
    _equipment(plan, changes, notes)
    tall = HUMANOID_HEIGHT * size * height
    if plan.body == "humanoid" and not HUMANOID_RANGE[0] <= tall <= HUMANOID_RANGE[1]:
        meters = min(max(tall, HUMANOID_RANGE[0]), HUMANOID_RANGE[1])
        changes["height"] = round(meters / (HUMANOID_HEIGHT * size), 3)
        why = " (taller ones fell over walking)" if tall > meters else " (shorter ones walk too slowly)"
        notes.append(
            f"a {tall:.2f} m humanoid became {meters:.2f} m: the humanoid walks from {HUMANOID_RANGE[0]} to "
            f"{HUMANOID_RANGE[1]} m tall{why}"
        )
    return replace(plan, **changes), notes


def _equipment(plan: Plan, changes: dict[str, Any], notes: list[str]) -> None:
    """What a body cannot carry comes off the plan with a note, so the plan says what the robot has."""
    if plan.body == "humanoid":
        if plan.lidar:
            changes["lidar"] = False
            notes.append('a humanoid carries no lidar yet; a wheeled or legged robot can ("a rover with a lidar")')
        if plan.hand != "gripper":
            changes["hand"] = "gripper"
            notes.append("a humanoid's hands hold nothing yet; ask for a wheeled or legged robot with an arm")
        if plan.camera != plan.head:
            changes["camera"] = plan.head
            notes.append(
                "a humanoid's camera is in its head: it has both or neither"
                + ("" if plan.head else "; this one has no head, so no camera")
            )
        if plan.material not in (None, "pla", "aluminum", "carbon"):
            changes["material"] = None
            notes.append(f"a humanoid is printed, aluminum or carbon, not {plan.material}")
    if plan.body == "balancer" and plan.arms:
        changes["arms"] = 0
        notes.append("a balancer carries no arms: it would topple reaching out; ask for a wheeled robot with an arm")
    if plan.body == "snake" and changes.get("arms", plan.arms) > 1:
        changes["arms"] = 1
        notes.append("a snake has one head: it got jaws (a gripper) for its arms")
    if plan.body == "snake" and changes.get("arms", plan.arms) and plan.segments > JAW_SEGMENTS:
        changes["segments"] = JAW_SEGMENTS
        notes.append(
            f"{plan.segments} segments became {JAW_SEGMENTS}: with jaws on its head a longer chain does not slither"
        )
