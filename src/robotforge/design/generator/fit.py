"""Choose a walker's leg servos by what the builder measures, not by a guess.

The builder's check stands the robot at home on a floor and reports how much of each servo's force holding it up
takes (its strain). A walk holds the robot on half its feet and swings the other half, so a leg needs room: the
generator goes up a ladder of joints, from the hobby servo to a geared joint module, and keeps the first whose legs
use at most WALK_STRAIN standing.
"""

from __future__ import annotations

import copy
from typing import Any

from robotforge.design import builder
from robotforge.design.builder.assemble import assemble
from robotforge.design.builder.checks import check
from robotforge.design.builder.expand import expand

# Lightest first. Measured on generated dogs (2026-10-19): a dog 1.4 times the size with legs 1.4 times as long used
# 50 % standing on plain joint modules and walked backwards; geared 2:1 it walked 1.27 m in 4 s. No geared hobby
# servo: it holds a big dog up but swings its legs at half speed, and big short-legged dogs on it walked 40 to 68 % of
# the way (0 of 3 at 1.6 times the size); on joint modules every one walked the whole way.
LADDER: tuple[Any, ...] = (
    None,  # the robot's own servo (sts3215)
    "bldc_joint",
    {"model": "bldc_joint", "gear": 2},
    {"model": "bldc_joint", "gear": 3},
)
WALK_STRAIN = 0.4  # the builder warns from 45 %; the guide's dog walks well at 38 % on its hobby servos
LEG_WORDS = ("hip", "thigh", "shin", "coxa", "femur", "tibia")


def is_leg(name: str) -> bool:
    return any(word in name for word in LEG_WORDS)


def strain(build: dict[str, Any], which=is_leg) -> float:
    """The largest share of force any servo `which` picks uses to hold the robot up at home."""
    spec = builder.parse(build)
    report = check(assemble(spec, expand(spec)))
    shares = [share for name, share in report.strain.items() if which(name)]
    return max(shares, default=0.0)


def fit_legs(build: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """`build` with the lightest leg servos that hold it with room to walk, and a note when even the strongest
    are short of it."""
    tried = None
    for servo in LADDER:
        tried = copy.deepcopy(build)
        for part in tried["parts"]:
            if is_leg(part["name"]):
                if servo is None:
                    part.pop("actuator", None)
                else:
                    part["actuator"] = servo
        used = strain(tried)
        if used <= WALK_STRAIN:
            return tried, []
    assert tried is not None
    return tried, [
        f"its legs use {used:.0%} of their strongest joints just to stand, so it may walk slowly: make it smaller, "
        "shorter-legged or lighter (fewer arms)"
    ]
