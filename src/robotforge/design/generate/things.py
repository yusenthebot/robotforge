"""The small things generated worlds are furnished with: what an apple, a cup or a book is made of in robotforge.

Each concept is one shape with a size and a mass a robot's gripper can take (7 cm across at most, under half a
kilogram), so every thing a generated task asks to fetch can be fetched. Colors come from the sentence when it says
one ("the red cup"), else from a palette in turn, and a thing that comes in colors is named by its color
("blue_cup"), so a task can tell them apart.
"""

from __future__ import annotations

import random

from robotforge.design.generate.draft import Item

PALETTE = ("red", "blue", "green", "yellow", "orange", "purple", "teal", "pink")
CARDBOARD = [0.74, 0.58, 0.38, 1]
COLORED = ("cup", "ball", "cube", "book", "can", "toy", "box")  # things named by their color


def make(concept: str, rng: random.Random, color: str | None = None) -> Item:
    """One `concept` (words.ITEMS) to put somewhere."""
    pick = color or rng.choice(PALETTE)
    named = f"{pick}_{concept}" if concept in COLORED else None
    if concept == "apple":
        return Item(
            "apple",
            "sphere",
            (0.065,),
            color or rng.choice(("red", "green")),
            0.15,
            named=f"{color}_apple" if color else None,
        )
    if concept == "cup":
        return Item("cup", "cylinder", (0.06, 0.085), pick, 0.12, named=named)
    if concept == "plate":
        return Item("plate", "cylinder", (0.16, 0.015), color or "white", 0.2)
    if concept == "bowl":
        return Item("bowl", "bin", (0.15, 0.15, 0.06), color or "white", 0.15)
    if concept == "ball":
        return Item("ball", "sphere", (0.06,), pick, 0.08, named=named)
    if concept == "book":
        return Item("book", "box", (0.15, 0.055, 0.2), pick, 0.3, named=named)
    if concept == "can":
        return Item("can", "cylinder", (0.055, 0.11), pick, 0.2, named=named)
    if concept == "cube":
        return Item("cube", "box", (0.045, 0.045, 0.045), pick, 0.06, named=named)
    if concept == "toy":
        return Item("toy", "box", (0.06, 0.05, 0.08), pick, 0.1, named=named)
    if concept == "box":
        if color:
            return Item("box", "box", (0.06, 0.06, 0.09), color, 0.25, named=named)
        return Item("box", "box", (0.06, 0.06, 0.09), CARDBOARD, 0.25, named="parcel")
    raise ValueError(f"no thing called {concept!r}")


def several(concept: str, count: int, rng: random.Random, color: str | None = None) -> list[Item]:
    """`count` of a concept: the first in the color said, the others in other colors (two red cups would be one
    name too many for "the red cup")."""
    out = []
    used = {color} if color else set()
    for k in range(count):
        if k == 0 and color:
            out.append(make(concept, rng, color))
            continue
        free = [c for c in PALETTE if c not in used] or list(PALETTE)
        pick = rng.choice(free)
        used.add(pick)
        out.append(make(concept, rng, pick if concept in COLORED else None))
    return out
