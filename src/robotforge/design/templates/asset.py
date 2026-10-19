"""One generated part on a turntable: for looking at what a generator makes and for checking it.

add_robot("sample", template="asset", params={"generator": "joint_actuator", "args": {"radius": 0.04}, "style": "lab"})
shows the part in the viewer, in the style and for the seed asked; the `turn` slider spins it so every side shows.
Its collision shapes (what the generator gave it) are hidden and touch nothing. Every generator robotforge or a
plugin has registered works.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import Field

from robotforge import assets
from robotforge.assets.bridge import Palette, collide, draw
from robotforge.design import components
from robotforge.design.templates.base import BuiltRobot, Template, TemplateParams, new_spec, servo_joint
from robotforge.design.visuals import Visuals


class AssetParams(TemplateParams):
    generator: str = Field("joint_actuator", description="the generator's name (list_templates lists them)")
    args: dict[str, Any] = Field(default_factory=dict, description="the generator's parameters, by name")
    style: str | None = Field(None, description="cobot, lab, hobby, rugged or a plugin's style")
    seed: int = Field(0, ge=0, le=1_000_000, description="the generator's seed")
    accent: str | list[float] | None = Field(None, description="the accent color, where the generator has one")
    color: str | list[float] | None = Field(None, description="the body color; none takes the style's own")


class AssetPreview(Template):
    name = "asset"
    summary = "One generated part (a joint, a link, a wheel, a sensor, ...) on a turntable, to look at or to check."
    Params = AssetParams

    def build(self, p: AssetParams) -> BuiltRobot:
        try:
            asset = assets.generate(p.generator, style=p.style, seed=p.seed, **p.args)
        except assets.AssetError as exc:
            raise ValueError(str(exc)) from None
        spec = new_spec()
        vis = Visuals(spec)
        stand = spec.worldbody.add_body(name="stand")
        body = stand.add_body(name="part")
        servo_joint(spec, body, "turn", [0, 0, 1], (-math.pi, math.pi), components.actuator("sts3215"))
        palette = Palette.of(
            p.style,
            body=None if p.color is None else components.color(p.color),
            accent=None if p.accent is None else components.color(p.accent),
        )
        low, _ = asset.bounds()
        lift = float(-low[2]) if low[2] < 0 else 0.0  # the lowest point on the floor
        draw(vis, body, asset, palette)
        for geom in collide(spec, body, asset, 1000.0):
            geom.contype = geom.conaffinity = 0  # it only takes up room: nothing bumps into it
        vis.finish()
        return BuiltRobot(
            spec=spec,
            home_ctrl={"turn": 0.0},
            spawn_height=lift,
            floating=False,
            notes=[f"{asset.name} in the {asset.style} style, seed {asset.seed}: {asset.params}"],
        )
