"""The gallery: ready-made robots built from parts, unusual bodies a beginner can add by name.

    add_robot("hopper", design="wheel_biped")

A design is only a starting point. What the world keeps is its build (a base and parts, see
the builder), written into world.yaml when the robot is added, so add_part, update_part and
remove_part change it part by part like any robot built from parts, and the built-in behaviors
recognize it by its shape (a balancer, a rover with an arm, a quadruped with two arms...).
"""

from __future__ import annotations

import copy
from typing import Any

from robotforge import wording
from robotforge.design.gallery import arms, legged, wheeled
from robotforge.design.gallery.base import Design

DESIGNS: dict[str, Design] = {d.name: d for d in (*wheeled.DESIGNS, *legged.DESIGNS, *arms.DESIGNS)}


def canonical(name: str) -> str | None:
    """The design `name` means ("Wheel Biped", "centaurs"), or None when there is none."""
    return wording.match(name, DESIGNS)


def get(name: str) -> Design:
    found = canonical(name)
    if found is None:
        from robotforge.design import templates

        template = wording.match(name, templates.TEMPLATES)
        if template is not None:
            raise ValueError(f"{template} is a template, not a design: add_robot(name, template={template!r}) adds it")
        close = wording.near(name, DESIGNS)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"unknown design {wording.short(name)}.{hint} Designs: {', '.join(DESIGNS)}")
    return DESIGNS[found]


def build(name: str) -> dict[str, Any]:
    """The design's build description, a fresh copy the caller may change."""
    return copy.deepcopy(get(name).make())


def catalog() -> list[dict[str, str]]:
    """Name and one-line summary of every design, for list_templates."""
    return [{"name": d.name, "summary": d.summary} for d in DESIGNS.values()]


__all__ = ["DESIGNS", "Design", "build", "canonical", "catalog", "get"]
