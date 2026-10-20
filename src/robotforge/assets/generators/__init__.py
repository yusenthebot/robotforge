"""The built-in generators. Importing this package registers them (robotforge.assets.registry does it on first use)."""

from robotforge.assets.generators import bases, joints, links, sensors, servos, tools, wheels

__all__ = ["bases", "joints", "links", "sensors", "servos", "tools", "wheels"]
