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
        Shot("assets_arm6", _arm(6), page=arm_view, size=HD, settle=2.5),
        Shot(
            "assets_hexapod",
            _describe("a six-legged robot with a camera"),
            page=frame_robot("bot", direction=(1.0, -0.85, 0.5), scale=1.9, least=0.6),
            size=HD,
            settle=2.5,
        ),
        Shot("assets_sheet", _sheet, page=look((1.15, -0.05, 0.9), (0.0, 0.0, 0.03)), size=HD, settle=3.0),
        Shot(
            "assets_close",
            lambda pg: _sheet(pg, CLOSE),
            page="window.__robotforge.select(null);" + look((0.42, -0.02, 0.2), (0.0, 0.0, 0.03)),
            size=HD,
            settle=3.0,
        ),
        Shot("assets_gallery", gallery_world, page=look((3.5, -3.2, 1.9), (-0.3, 0.2, 0.3)), size=HD, settle=3.0),
        Shot("assets_described", _described_row, page=look((3.4, -0.2, 1.1), (0.0, 0.0, 0.35)), size=HD, settle=3.0),
    ]
