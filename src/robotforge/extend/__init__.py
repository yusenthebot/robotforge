"""Extending robotforge with your own code: parts, sensors, controllers and behaviors.

Drop a Python file into ~/.robotforge/plugins (or a world's plugins/ folder, see registry.py), import what you need
from here and register it:

    from robotforge.extend import part, Box, Cylinder

    @part("signpost", size="[height]", default_size=[1.2], fixed=True, color="brown")
    def signpost(size, rgba):
        h = size[0]
        return [Cylinder(0.05, h, at=[0, 0, h / 2]), Box([0.4, 0.03, 0.2], at=[0, 0, h - 0.1], color=1.3)]

After that add_object(shape="signpost") works like any shape. docs/EXTENDING.md has a worked example of each kind.
"""

from __future__ import annotations

from pathlib import Path

from robotforge.extend.algorithms import algorithm, task
from robotforge.extend.assets import generator, material, style
from robotforge.extend.parts import Box, Cylinder, Sphere, part, unknown_part
from robotforge.extend.registry import REGISTRY, PluginError
from robotforge.extend.routines import behavior, call, controller
from robotforge.extend.sensors import look, sensor
from robotforge.training.hyper import Hyper

__all__ = [
    "REGISTRY",
    "Box",
    "Cylinder",
    "Hyper",
    "PluginError",
    "Sphere",
    "algorithm",
    "behavior",
    "call",
    "controller",
    "ensure_user_plugins",
    "generator",
    "load_world_plugins",
    "look",
    "material",
    "part",
    "plugin_notes",
    "sensor",
    "style",
    "task",
    "unknown_part",
]


def __getattr__(name: str):
    # What a generator is written with, loaded when asked for (robotforge.assets imports mujoco).
    if name in ("Param", "Asset", "Piece", "Collider", "Style", "Material"):
        from robotforge import assets

        return getattr(assets, name)
    raise AttributeError(f"module 'robotforge.extend' has no attribute {name!r}")


def ensure_user_plugins() -> None:
    """Load the user's own plugin folders, once (a no-op afterwards and when there are none)."""
    REGISTRY.load_user()


def load_world_plugins(world_dir: Path) -> None:
    """Called whenever a world is read: the user's plugins, and the world's own when they are trusted."""
    REGISTRY.load_world(Path(world_dir))


def plugin_notes(world_dir: Path) -> list[str]:
    """Said when a world opens: that its folder holds plugin code, what of it runs and what was left alone."""
    return REGISTRY.world_notes(Path(world_dir))
