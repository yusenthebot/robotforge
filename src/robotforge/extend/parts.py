"""Plugin parts: new shapes for add_object, built from boxes, cylinders and spheres.

A part is a function from a size and a color to a list of primitives. Everything else (placing it on the ground or
a table, yaw, mass, fixed or loose, saving it in world.yaml, drawing it in the viewer) is what a built-in shape gets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from robotforge.extend.registry import REGISTRY, Item, PluginError

Number = float | int


@dataclass(frozen=True)
class Box:
    """A block `size` = [length, width, height] m, its middle at `at` = [x, y, z] in the part's own frame (x forward,
    y left, z up from the ground it stands on). `color` is the part's own color unless given (a name, a hex code,
    [r, g, b], or a number: that much lighter (above 1) or darker (below 1) than the part's color)."""

    size: Sequence[Number]
    at: Sequence[Number] = (0.0, 0.0, 0.0)
    color: Any = None
    rotate: Sequence[Number] = (0.0, 0.0, 0.0)  # degrees about x, y, z
    collide: bool = True


@dataclass(frozen=True)
class Cylinder:
    """A cylinder `diameter` x `height` m standing along z (see Box for the rest)."""

    diameter: Number
    height: Number
    at: Sequence[Number] = (0.0, 0.0, 0.0)
    color: Any = None
    rotate: Sequence[Number] = (0.0, 0.0, 0.0)
    collide: bool = True


@dataclass(frozen=True)
class Sphere:
    """A ball of `diameter` m (see Box for the rest)."""

    diameter: Number
    at: Sequence[Number] = (0.0, 0.0, 0.0)
    color: Any = None
    collide: bool = True


Primitive = Box | Cylinder | Sphere
MAX_PRIMITIVES = 60  # a part is a handful of shapes: more slows every step of every world that has one


def part(
    name: str,
    *,
    size: str,
    default_size: Sequence[Number],
    color: str | Sequence[float] = "gray",
    fixed: bool = False,
    mass: float | None = None,
    summary: str = "",
    sizes: Sequence[int] | None = None,
    footprint: Callable[[list[float]], tuple[float, float]] | None = None,
) -> Callable[[Callable[..., list[Primitive]]], Callable[..., list[Primitive]]]:
    """Register a new object shape, `add_object(shape=name)`.

    size: what the numbers mean, e.g. "[height]" or "[length, width, height]" (shown to agents).
    default_size: used when add_object gives none; the part is checked at this size when it is registered.
    sizes: how many numbers a size may have (default: as many as default_size).
    fixed: never moves (a signpost); otherwise it is loose, with `mass` kg (default 5).
    footprint: the half extents [x, y] of what it covers on the ground (default: measured from its pieces).

    The decorated function takes (size, rgba) and returns a list of Box, Cylinder and Sphere.
    """

    def register(fn: Callable[..., list[Primitive]]) -> Callable[..., list[Primitive]]:
        from robotforge.design import components, props

        counts = tuple(sizes) if sizes is not None else (len(default_size),)
        if not default_size or any(s <= 0 for s in default_size) or len(default_size) not in counts:
            raise PluginError(f"part {name!r}: default_size must be positive numbers, {counts[0]} of them")
        taken = REGISTRY.get("part", name)
        if taken is not None:
            raise PluginError(f"part {name!r} is already added by {taken.plugin}; pick another name")
        if name in props.SHAPES:
            raise PluginError(f"part {name!r}: robotforge already has a shape of that name; pick another")
        try:
            components.color(color)
        except ValueError as exc:
            raise PluginError(f"part {name!r}: color: {exc}") from None
        info = props.ShapeInfo(
            size_hint=size,
            size_lengths=counts,
            size=tuple(float(s) for s in default_size),
            color=color if isinstance(color, str) else tuple(float(c) for c in color),
            fixed=fixed,
            pickable=False,
            summary=summary or f"a {name} (from a plugin)",
        )

        def pieces(size_: list[float], rgba: list[float]) -> list[props.Part]:
            return _build(name, fn, size_, rgba)

        pieces(list(info.size), components.color(color))  # fails here, with the plugin's file named, not at add_object

        def measure(size_: list[float]) -> tuple[float, float]:
            return footprint(size_) if footprint is not None else _footprint_of(pieces(size_, [0.5, 0.5, 0.5, 1.0]))

        props.register_shape(name, info, pieces, measure, mass)
        REGISTRY.add(
            Item(
                "part",
                name,
                info.summary,
                fn,
                meta={"describe": {"size": size, "default_size": list(info.size), "fixed": fixed}},
                undo=lambda: props.unregister_shape(name),
            )
        )
        return fn

    return register


def _build(name: str, fn: Callable[..., list[Primitive]], size: list[float], rgba: list[float]):
    from robotforge.design import components, props

    try:
        made = fn(list(size), list(rgba))
    except Exception as exc:  # the plugin's code: say which part, not a bare traceback from deep in the compiler
        raise ValueError(f"the plugin part {name!r} failed for size {size}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(made, list | tuple) or not made:
        raise ValueError(f"the plugin part {name!r} must return a list of Box, Cylinder and Sphere, got {made!r}")
    if len(made) > MAX_PRIMITIVES:
        raise ValueError(f"the plugin part {name!r} returned {len(made)} shapes; at most {MAX_PRIMITIVES}")
    G = props.G
    out = []
    for p in made:
        if isinstance(p, Box):
            geom, half = G.mjGEOM_BOX, [float(s) / 2 for s in _three(p.size, "Box size")]
        elif isinstance(p, Cylinder):
            geom, half = G.mjGEOM_CYLINDER, [float(p.diameter) / 2, float(p.height) / 2, 0.0]
        elif isinstance(p, Sphere):
            geom, half = G.mjGEOM_SPHERE, [float(p.diameter) / 2, 0.0, 0.0]
        else:
            raise ValueError(f"the plugin part {name!r} returned {p!r}: use Box, Cylinder or Sphere")
        if any(h < 0 for h in half) or half[0] <= 0:
            raise ValueError(f"the plugin part {name!r} has a shape with no size: {p!r}")
        out.append(
            props.Part(
                geom,
                half,
                [float(v) for v in _three(p.at, "at")],
                _color(p.color, rgba, components),
                p.collide,
                _quat(getattr(p, "rotate", (0, 0, 0))),
            )
        )
    return out


def _three(values: Sequence[Number], what: str) -> Sequence[Number]:
    if len(values) != 3:
        raise ValueError(f"{what} is three numbers [x, y, z] in meters, got {list(values)!r}")
    return values


def _color(given: Any, rgba: list[float], components: Any) -> list[float]:
    if given is None:
        return list(rgba)
    if isinstance(given, int | float) and not isinstance(given, bool):
        return [min(1.0, c * float(given)) for c in rgba[:3]] + [rgba[3]]
    return components.color(given)


def _quat(rotate: Sequence[Number]) -> list[float]:
    """Degrees about x, then y, then z, as a w-x-y-z quaternion."""
    rx, ry, rz = (math.radians(float(a)) / 2 for a in _three(rotate, "rotate"))
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return [
        cz * cy * cx + sz * sy * sx,
        cz * cy * sx - sz * sy * cx,
        sz * cy * sx + cz * sy * cx,
        sz * cy * cx - cz * sy * sx,
    ]


def _footprint_of(pieces: list[Any]) -> tuple[float, float]:
    """Half extents (x, y) around the part's origin that cover its pieces; a turned piece counts by its reach."""
    from robotforge.design import props

    hx = hy = 0.0
    for p in pieces:
        turned = any(abs(v) > 1e-9 for v in p.quat[1:])
        if turned:
            ex = ey = math.sqrt(sum(float(v) ** 2 for v in p.size[:3]))
        elif p.type == props.G.mjGEOM_BOX:
            ex, ey = p.size[0], p.size[1]
        else:  # a cylinder or a ball is as wide as its radius
            ex = ey = p.size[0]
        hx, hy = max(hx, abs(p.pos[0]) + ex), max(hy, abs(p.pos[1]) + ey)
    return max(hx, 0.01), max(hy, 0.01)


def unknown_part(name: str, shapes: list[str]) -> str:
    """The message for a shape no one has added, when plugin files of the world were left unloaded: why."""
    left = sorted({f"{p.parent.parent.name}/{p.name}" for p in REGISTRY.untrusted})
    if left:
        return (
            f"unknown shape {name!r}: this world has plugin files that were not loaded ({', '.join(left)}) because "
            "they were not written on this computer or changed since. They are Python that would run with your "
            'rights: read them (read_script) and, if you trust them, plugins(action="trust")'
        )
    return f"unknown shape {name!r}; the shapes are {', '.join(shapes)}; plugins() lists the ones plugins add"
