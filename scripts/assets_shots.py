"""Shots of the asset engine's robots (G13): the same robots before and after a change to how parts look.

    uv run python scripts/screenshots.py --only assets_arm3 assets_arm6 ... --out <folder>

Imported by scripts/screenshots.py, which owns the shot machinery; the shots here only say what to build and where
the camera stands.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

HD = (1920, 1080)

# Five sentences, five bodies: a courier, a dog, a rover with an arm, a self-balancing robot and a humanoid.
DESCRIBED = [
    "a tall wheeled courier with a lidar",
    "a small orange robot dog",
    "a six-wheeled mars rover with an arm",
    "a self-balancing robot on two wheels",
    "a tall humanoid",
]


def _arm(dof: int) -> Callable[[Any], None]:
    def world(pg: Any) -> None:
        pg.add_robot("arm", template="arm", params={"dof": dof})

    return world


def _templates_row(pg: Any) -> None:
    """The templates that draw with the engine: a dog, a humanoid, a rover and a gripper."""
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.7], params={"color": "orange"})
    pg.add_robot("man", template="humanoid", pos=[0.0, 0.2], params={"color": "teal"})
    pg.add_robot("car", template="rover", pos=[0.0, -0.3], params={"color": "red"})


def _arm_styles(pg: Any) -> None:
    """The same 3-joint arm in each style, side by side."""
    colors = {"cobot": "blue", "lab": "orange", "hobby": "teal", "rugged": "yellow"}
    for i, style in enumerate(STYLES):
        params = {"dof": 3, "style": style, "color": colors[style]}
        pg.add_robot(f"arm_{style}", template="arm", params=params, pos=[0.0, 0.34 * (i - 1.5)])


def _describe(sentence: str) -> Callable[[Any], None]:
    def world(pg: Any) -> None:
        pg.add_robot("bot", describe=sentence)

    return world


def _described_row(pg: Any) -> None:
    for i, sentence in enumerate(DESCRIBED):
        pg.add_robot(f"d{i + 1}", describe=sentence, pos=[0.0, 1.0 - 0.5 * i])


STYLES = ("cobot", "lab", "hobby", "rugged")
# One part per column, one style per row: what the same generator makes in each style.
SHEET = [
    ("pedestal", {"radius": 0.06, "height": 0.07}),
    ("joint_actuator", {"radius": 0.035, "length": 0.07}),
    ("link_shell", {"length": 0.2, "radius": 0.028, "collar": 0.036}),
    ("chassis", {"length": 0.16, "width": 0.1, "height": 0.05}),
    ("wheel", {"radius": 0.045, "width": 0.035}),
    ("foot", {"radius": 0.025, "shape": "pad"}),
    ("sensor_housing", {"kind": "camera", "size": 0.04}),
    ("sensor_housing", {"kind": "lidar", "size": 0.06}),
    ("grip_pad", {"length": 0.07, "width": 0.03, "thickness": 0.012}),
]


def _sheet(pg: Any, columns: list[tuple[str, dict]] = SHEET) -> None:
    for r, style in enumerate(STYLES):
        for c, (name, args) in enumerate(columns):
            pos = [0.2 * (r - 1.5), 0.13 * (c - (len(columns) - 1) / 2)]
            pg.add_robot(
                f"{style}_{c}", template="asset", params={"generator": name, "args": args, "style": style}, pos=pos
            )


CLOSE = [
    ("joint_actuator", {"radius": 0.035, "length": 0.07}),
    ("link_shell", {"length": 0.12, "radius": 0.028, "collar": 0.036}),
    ("wheel", {"radius": 0.045, "width": 0.035}),
    ("sensor_housing", {"kind": "camera", "size": 0.04}),
]


def shots(Shot: Any, look: Callable, frame_robot: Callable, gallery_world: Callable) -> list[Any]:
    """The shots; the callables are the helpers of scripts/screenshots.py."""
    arm_view = "window.__robotforge.select(null);" + look((0.52, -0.44, 0.34), (0.1, 0.0, 0.2))
    return [
        Shot("assets_arm3", _arm(3), page=arm_view, size=HD, settle=2.5),
        Shot(
            "assets_arm3_close",
            _arm(3),
            page="window.__robotforge.select(null);" + look((0.3, -0.3, 0.27), (0.04, 0.0, 0.17)),
            size=HD,
            settle=2.5,
        ),
        Shot(
            "assets_templates",
            _templates_row,
            page="window.__robotforge.select(null);" + look((1.9, -0.4, 0.8), (0.0, 0.0, 0.2)),
            size=HD,
            settle=3.0,
        ),
        Shot(
            "assets_styles",
            _arm_styles,
            page="window.__robotforge.select(null);" + look((1.15, -0.04, 0.48), (0.05, -0.02, 0.2)),
            size=HD,
            settle=3.0,
        ),
        Shot(
            "assets_dog_close",
            lambda pg: pg.add_robot("dog", template="quadruped", params={"color": "orange"}),
            page="window.__robotforge.select(null);" + look((0.62, -0.5, 0.32), (0.0, 0.0, 0.14)),
            size=HD,
            settle=2.5,
        ),
        Shot(
            "assets_humanoid_close",
            lambda pg: pg.add_robot("man", template="humanoid", params={"color": "teal"}),
            page="window.__robotforge.select(null);" + look((0.9, -0.55, 0.45), (0.0, 0.0, 0.22)),
            size=HD,
            settle=2.5,
        ),
        Shot("assets_arm6", _arm(6), page=arm_view, size=HD, settle=2.5),
        Shot(
            "assets_hexapod",
            _describe("a six-legged robot with a camera"),
            page=frame_robot("bot", direction=(1.0, -0.85, 0.5), scale=1.9, least=0.6),
            size=HD,
            settle=2.5,
        ),
        Shot(
            "assets_sheet",
            _sheet,
            page="window.__robotforge.select(null);" + look((0.95, -0.02, 0.8), (0.0, 0.0, 0.03)),
            size=HD,
            settle=3.0,
        ),
        Shot(
            "assets_close",
            lambda pg: _sheet(pg, CLOSE),
            page="window.__robotforge.select(null);" + look((0.42, -0.02, 0.2), (0.0, 0.0, 0.03)),
            size=HD,
            settle=3.0,
        ),
        Shot("assets_gallery", gallery_world, page=look((1.7, -3.2, 1.7), (-0.7, 0.2, 0.3)), size=HD, settle=3.0),
        Shot(
            "assets_described",
            _described_row,
            page="window.__robotforge.select(null);" + look((2.0, -0.25, 0.6), (0.0, 0.0, 0.3)),
            size=HD,
            settle=3.0,
        ),
    ]
