"""Registry of robot templates."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from robotforge import wording
from robotforge.design.templates.arm import Arm
from robotforge.design.templates.base import BuiltRobot, Template
from robotforge.design.templates.cartpole import Cartpole
from robotforge.design.templates.gripper import Gripper
from robotforge.design.templates.humanoid import Humanoid
from robotforge.design.templates.quadruped import Quadruped
from robotforge.design.templates.rover import Rover

TEMPLATES: dict[str, Template] = {t.name: t for t in (Arm(), Quadruped(), Humanoid(), Rover(), Cartpole(), Gripper())}


# What agents call a template that has another name here; offered as "did you mean", never taken for the template:
# a "dog" may also be the library's Go2.
NICKNAMES = {
    "dog": "quadruped",
    "legged": "quadruped",
    "walker": "quadruped",
    "car": "rover",
    "wheeled": "rover",
    "robot_arm": "arm",
    "biped": "humanoid",
    "human": "humanoid",
    "pendulum": "cartpole",
    "cart_pole": "cartpole",
    "claw": "gripper",
}


def get(name: str) -> Template:
    found = wording.match(name, TEMPLATES)
    if found is None:
        close = wording.near(name, TEMPLATES, NICKNAMES)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ValueError(f"unknown template {name!r}.{hint} Templates: {', '.join(TEMPLATES)}")
    return TEMPLATES[found]


def build(name: str, params: dict[str, Any]) -> BuiltRobot:
    template = get(name)
    try:
        checked = template.Params.model_validate(params)
    except ValidationError as exc:
        raise ValueError(wording.validation_message(exc, valid=list(template.Params.model_fields))) from None
    return template.build(checked)


def catalog() -> list[dict[str, Any]]:
    """Template names, summaries and parameter schemas for agents."""
    out = []
    for t in TEMPLATES.values():
        schema = t.Params.model_json_schema()
        params = {
            key: {
                k: v for k, v in prop.items() if k in ("type", "default", "description", "minimum", "maximum", "enum")
            }
            for key, prop in schema.get("properties", {}).items()
        }
        out.append({"name": t.name, "summary": t.summary, "params": params})
    return out


__all__ = ["TEMPLATES", "BuiltRobot", "Template", "build", "catalog", "get"]
