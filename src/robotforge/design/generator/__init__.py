"""Generate a robot from one sentence: legs, arms, wheels, sizes, joints and sensors, built from the builder's parts.

    made = generator.generate("a six-legged robot with two arms")      # -> Generated
    made.source        # {"build": {...}}, or {"template": "humanoid", "params": {...}}
    again = generator.change(made.plan, "make it longer")              # same robot, longer

The sentence becomes a plan (words.py): a body and a few counts and factors, plus a seed for what the sentence left
open. The plan becomes parts (bodies.py) laid out like the gallery's proven robots, so every generated robot has a
shape the built-in behaviors know: it walks, drives, balances, slithers or reaches as soon as it is added. The world
keeps the plan with the robot, so a later sentence changes it ("six legs instead of four", "another one").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robotforge.design.generator import bodies
from robotforge.design.generator.plan import Plan, clamp
from robotforge.design.generator.words import SentenceError, read, seed_of

# What each body does once added, and the tools that drive it (for the add_robot result).
HOW = {
    "quadruped": "it walks on a trot: move and the WASD keys",
    "hexapod": "it walks on a tripod gait: move and the WASD keys",
    "rover": "it drives: move and the WASD keys",
    "balancer": "it balances by itself and drives: move and the WASD keys",
    "snake": "it slithers: move and the WASD keys",
    "arm": "it reaches: move_arm and pick_and_place",
    "humanoid": "it walks: move and the WASD keys; pose makes it wave, squat, bow or cheer",
}


@dataclass(frozen=True)
class Generated:
    plan: Plan
    source: dict[str, Any]  # the robot entry's source: {"build": ...} or {"template": ..., "params": ...}
    kind: str  # what the built-in behaviors will see it as
    notes: list[str] = field(default_factory=list)

    def describe(self, robot: str) -> dict[str, Any]:
        """For the add_robot result: what was made and how to drive it."""
        hands = self.hands(robot)
        how = HOW[self.kind]
        if self.plan.body == "snake" and hands:
            how += f"; gripper({hands[0]!r}, ...) opens and closes its jaws"
        elif self.kind == "arm":
            how += f" with {' or '.join(hands)}"
        elif hands:
            how += f"; pick_and_place, move_arm and gripper work {' and '.join(hands)}"
        out: dict[str, Any] = {"plan": self.plan.to_dict(), "kind": self.kind, "how": how}
        if self.notes:
            out["notes"] = self.notes
        return out

    def hands(self, robot: str) -> list[str]:
        """The names its arms (or jaws) answer to in move_arm, pick_and_place and gripper."""
        p = self.plan
        if p.body == "humanoid":
            return []
        if p.body == "snake":
            return [f"{robot}/jaws"] if p.arms else []
        if p.arms >= 2:
            return [f"{robot}/left_hand", f"{robot}/right_hand"]
        if p.body == "arm":
            return [f"{robot}/tip" if p.tentacle else f"{robot}/arm"]
        return [f"{robot}/arm"] if p.arms else []


def _kind(plan: Plan) -> str:
    if plan.body == "legged":
        return "quadruped" if plan.legs == 4 else "hexapod"
    return {"wheeled": "rover"}.get(plan.body, plan.body)


def realize(plan: Plan, notes: list[str] | None = None) -> Generated:
    """The robot a plan describes (its numbers first brought inside what builds and moves)."""
    plan, clamped = clamp(plan)
    made, more = bodies.make(plan)
    source = made if "template" in made else {"build": made}
    return Generated(plan, source, _kind(plan), [*(notes or []), *clamped, *more])


def generate(sentence: str, seed: int | None = None) -> Generated:
    """A robot from a sentence. The same sentence and seed make the same robot; another seed, a variation."""
    plan, notes = read(sentence, seed=seed)
    return realize(plan, notes)


def change(plan: Plan | dict[str, Any], sentence: str, seed: int | None = None) -> Generated:
    """The robot `plan` describes, changed as the sentence says ("make it longer", "add a lidar")."""
    if isinstance(plan, dict):
        plan = Plan.from_dict(plan)
    new, notes = read(sentence, plan, seed=seed)
    return realize(new, notes)


__all__ = ["Generated", "Plan", "SentenceError", "change", "generate", "read", "realize", "seed_of"]
