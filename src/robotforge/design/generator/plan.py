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
SEGMENTS = (4, 14)
JOINTS = (3, 6)  # a standing arm's joints, the gripper not counted
FACTOR = (0.6, 1.8)  # length, width and height, each times the body's own default
SIZE = (0.6, 1.6)  # the whole robot, times its default
# A walker's legs (size times height) at least this long: shorter, a hexapod's feet are not 5 cm under its body, which
# the gait needs to tell legs from the rest (a 0.6 size robot with 0.6 legs was not seen as a walker at all).
MIN_LEG = 0.6
# A snake's spine (first joint to last) at most this long: 1.7 to 2.5 m snakes did under half the way in 5 s at 0.1 m/s
# and 0.03 m/s after 20 s; up to 1.6 m every one without jaws did (2026-10-19 sweep).
SNAKE_SPINE = 1.6  # m
SNAKE_SEGMENT = 0.08  # m, one segment at factor 1
HUMANOID_HEIGHT = 0.45  # m, the humanoid template's height at factor 1
HUMANOID_RANGE = (0.3, 0.65)  # m: at 0.7 m some variations fell over walking at 0.15 m/s (2026-10-19 sweep)


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
        bounded(name, getattr(plan, name), *FACTOR, "further and the servos cannot hold it up or it tips over")
    bounded("size", plan.size, *SIZE, "further and the servos cannot hold it up")
    size, height = changes.get("size", plan.size), changes.get("height", plan.height)
    if plan.body == "legged" and size * height < MIN_LEG:
        changes["height"] = round(MIN_LEG / size, 3)
        notes.append(f"legs x{height:.2f} became x{changes['height']:.2f}: shorter legs do not lift its feet to walk")
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
    tall = HUMANOID_HEIGHT * size * height
    if plan.body == "humanoid" and not HUMANOID_RANGE[0] <= tall <= HUMANOID_RANGE[1]:
        meters = min(max(tall, HUMANOID_RANGE[0]), HUMANOID_RANGE[1])
        changes["height"] = round(meters / (HUMANOID_HEIGHT * size), 3)
        notes.append(
            f"a {tall:.2f} m humanoid became {meters:.2f} m: the humanoid walks from {HUMANOID_RANGE[0]} to "
            f"{HUMANOID_RANGE[1]} m tall (taller ones fell over walking)"
        )
    return replace(plan, **changes), notes
