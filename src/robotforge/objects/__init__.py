"""Realistic objects for pick-and-place: mugs, bottles, cans, bowls, fruit, packages, books and tools.

Each one is a seeded, parametric generator: a good visual mesh, convex collision pieces, a mass and friction from what
it is made of, and grasp hints (a mug's handle, a bottle's body, a tool's shaft) that the arm planner uses.

    add_object("m", shape="mug", seed=3)                           # by name
    add_object("m", describe="a tall blue mug with a big handle")  # by description

Writing your own (a plugin file, see docs/EXTENDING.md):

    from robotforge.objects import generator, Range

    @generator("coaster", size="[diameter]", default_size=(0.09,), summary="a cork coaster")
    def coaster(g): ...
"""

from __future__ import annotations

from robotforge.objects.kit import Choice, Flag, Gen, Grasp, Range
from robotforge.objects.registry import GENERATORS, built_of, catalogue, generator, get, object_for

__all__ = [
    "GENERATORS",
    "Choice",
    "Flag",
    "Gen",
    "Grasp",
    "Range",
    "built_of",
    "catalogue",
    "generator",
    "get",
    "object_for",
]

from robotforge.design import props as _props  # noqa: E402,F401  (registers the shipped generators, see props.py)
