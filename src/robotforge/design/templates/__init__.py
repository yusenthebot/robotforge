"""Registry of robot templates."""

from __future__ import annotations

from typing import Any

from robotforge.design.templates.arm import Arm
from robotforge.design.templates.base import BuiltRobot, Template
from robotforge.design.templates.cartpole import Cartpole
from robotforge.design.templates.humanoid import Humanoid
from robotforge.design.templates.quadruped import Quadruped
from robotforge.design.templates.rover import Rover

TEMPLATES: dict[str, Template] = {t.name: t for t in (Arm(), Quadruped(), Humanoid(), Rover(), Cartpole())}


def get(name: str) -> Template:
    try:
        return TEMPLATES[name]
    except KeyError:
        raise ValueError(f"unknown template {name!r}; choose one of: {', '.join(TEMPLATES)}") from None


def build(name: str, params: dict[str, Any]) -> BuiltRobot:
    template = get(name)
    return template.build(template.Params.model_validate(params))


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
