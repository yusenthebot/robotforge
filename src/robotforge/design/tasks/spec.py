"""Tasks saved with a world: one sentence becomes a goal with a deterministic success check, a reward for learning,
a start and how far each attempt's start may vary.

    add_task("bring the red cup from the kitchen to the bedroom desk")

A task names a robot, the thing it works on (`object`), and where that has to end up or what has to happen to it
(`target`): go somewhere, bring or put a thing in or on something or in a room, pick it up, push it onto a mark,
open or close a drawer or a door, press a button, switch a switch on. world.yaml keeps it under `tasks:` (a world
without tasks saves as it always did). Whether it is done is judged from where things are (`check.py`): the same
rules on the live simulation, in a training environment and on the positions an agent's tool calls reported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from robotforge import wording

KINDS = ("go_to", "bring", "put", "pick_up", "push", "open", "close", "press", "switch_on")
# The harder ones (hard.py): things that have to end up a certain way, not just somewhere
HARD_KINDS = ("upright", "stack", "insert", "sort", "pour", "hand_over")
Kind = Literal[
    "go_to", "bring", "put", "pick_up", "push", "open", "close", "press", "switch_on",
    "upright", "stack", "insert", "sort", "pour", "hand_over",
]  # fmt: skip
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
DEFAULT_SECONDS = {"go_to": 60.0, "bring": 180.0, "put": 90.0, "pick_up": 60.0, "push": 90.0, "open": 60.0,
                   "close": 60.0, "press": 45.0, "switch_on": 45.0, "upright": 90.0, "stack": 240.0, "insert": 120.0,
                   "sort": 420.0, "pour": 150.0, "hand_over": 150.0}  # fmt: skip


def _unset(value: object) -> bool:
    """For fields added after worlds were first saved: left out of a world.yaml while unset."""
    return value is None


class Randomize(BaseModel):
    """How far each attempt's start may differ from the task's start, drawn uniformly: the robot's place (m each
    way along x and y) and heading (degrees each way), and the thing's place on its surface (m each way)."""

    model_config = ConfigDict(extra="forbid")
    robot_xy: float = Field(0.3, ge=0, le=3)
    robot_yaw: float = Field(30.0, ge=0, le=180)
    object_xy: float = Field(0.05, ge=0, le=1)


class Start(BaseModel):
    """Where an attempt starts: the robot at [x, y, yaw degrees] (None: where the world puts it) and things at
    [x, y, z]."""

    model_config = ConfigDict(extra="forbid")
    robot: list[float] | None = None
    objects: dict[str, list[float]] = Field(default_factory=dict)


class TaskEntry(BaseModel):
    """One task in world.yaml."""

    model_config = ConfigDict(extra="forbid")
    name: str
    sentence: str = ""  # what was asked
    kind: Kind
    robot: str
    object: str | None = None  # the thing it works on (go_to: none)
    target: str | None = None  # where it goes: an object, a room or a mark (pick_up, open, ...: none)
    radius: float = Field(0.5, gt=0, le=5, description="m: go_to counts within this of the target")
    seconds: float = Field(60.0, gt=0, le=600, description="time an attempt may take")
    # The harder kinds (hard.py): the things a stack is made of, bottom first, the things a sort puts away and the
    # balls a pour moves (`items`), where each of them goes (`into`), and how far from upright a thing may stand.
    items: list[str] | None = Field(None, exclude_if=_unset)
    into: dict[str, str] | None = Field(None, exclude_if=_unset)
    tilt: float | None = Field(None, gt=0, le=90, exclude_if=_unset)
    start: Start = Field(default_factory=Start)
    randomize: Randomize = Field(default_factory=Randomize)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError(f"task name {wording.short(v)} must be lowercase letters, digits and _")
        return v

    def describe(self) -> str:
        """The goal in a few words, for answers: "red_cup in bedroom_desk"."""
        verb = {"go_to": "robot at", "bring": "in or on", "put": "in or on", "push": "onto", "pick_up": "lifted",
                "open": "open", "close": "closed", "press": "pressed", "switch_on": "on", "upright": "upright on",
                "stack": "stacked on", "insert": "in", "sort": "put away", "pour": "poured into",
                "hand_over": "handed to"}[self.kind]  # fmt: skip
        if self.kind == "go_to":
            return f"{self.robot} at {self.target}"
        if self.kind == "stack" and self.items:
            return " on ".join(reversed(self.items))
        if self.kind == "sort" and self.items:
            return f"{len(self.items)} things sorted into {len(set((self.into or {}).values()))} trays"
        if self.target:
            return f"{self.object} {verb} {self.target}"
        return f"{self.object} {verb}"


@dataclass
class Asked:
    """What a task sentence asks, before it is matched to the world: a kind, and the words for the thing and the
    place (each a concept, a color and a room)."""

    kind: str
    thing: Ref | None = None
    place: Ref | None = None
    robot: str | None = None


@dataclass
class Ref:
    """A thing or a place as a sentence says it: "the red cup in the kitchen", "the bedroom desk", "the goal"."""

    text: str
    concept: str | None = None  # words.THINGS concept, or "room", "goal"
    color: str | None = None
    room: str | None = None  # room kind the sentence puts it in (words.ROOMS)
    name: str | None = None  # an object's exact name said in the sentence ("red_cup")
    extra: dict = field(default_factory=dict)
