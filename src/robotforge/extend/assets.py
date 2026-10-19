"""Plugin generators, materials and styles for the asset engine (robotforge.assets, docs/ASSETS.md).

    from robotforge.extend import generator, Param, Asset, Piece, Collider
    from robotforge.design.visuals import mesh as M

    @generator("standoff", summary="a hexagonal standoff", params=[Param("height", 0.02, 0.005, 0.1, "its height")])
    def standoff(a):
        body = M.cylinder(0.005, a.height, segments=6, z0=0.0)
        collider = Collider("cylinder", (0.005, a.height / 2), (0, 0, a.height / 2))
        return Asset("standoff", (Piece("trim", "metal", body),), (collider,))

A generator registered here is a generator everywhere: `add_robot(template="asset", params={"generator": "standoff"})`
shows it, `robotforge.assets.generate("standoff", height=0.03)` makes it for code, and the objects and the builder
can draw with it. Materials and styles work the same way: `material("velvet", roughness=0.9, texture="rubber")` and
`style("sleek", based_on="cobot", fillet=1.6)`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from robotforge.extend.registry import REGISTRY, Item, PluginError


def _params(given: Sequence[Any] | dict[str, Any], assets: Any) -> tuple[Any, ...]:
    """Parameters as `Param` objects: those given, or `{name: (default, low, high, doc)}` for floats."""
    if isinstance(given, dict):
        out = []
        for name, spec in given.items():
            if isinstance(spec, assets.Param):
                out.append(spec)
                continue
            default, low, high, *doc = spec
            out.append(assets.Param(name, default, low, high, doc[0] if doc else ""))
        return tuple(out)
    return tuple(given)


def generator(
    name: str,
    *,
    summary: str = "",
    params: Sequence[Any] | dict[str, Any] = (),
    category: str = "part",
    anchor: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a part generator, `assets.generate(name, **params)`.

    The decorated function takes `a` (its checked parameters as attributes, `a.style`, `a.seed`, `a.rng`) and returns an
    Asset: drawn pieces, collision shapes, and so a mass from density. params lists `Param(name, default, low, high,
    doc)` (floats are meters; `kind="int"`, "bool" or "choice" for the others) or a dict
    {name: (default, low, high, doc)}. It is built once with its defaults when registered, so a mistake shows here with
    the plugin's file named, not later in the viewer."""

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        from robotforge import assets

        try:
            gen = assets.Generator(
                name, summary or f"a {name} (from a plugin)", _params(params, assets), fn, category, anchor
            )
            assets.register_generator(gen)
        except (assets.AssetError, TypeError, ValueError) as exc:
            raise PluginError(f"generator {name!r}: {exc}") from None
        try:
            assets.generate(name)
            assets.generate(name, style="rugged", seed=1)
        except assets.AssetError as exc:
            assets.unregister_generator(name)
            raise PluginError(f"generator {name!r} does not make an asset with its defaults: {exc}") from None
        item = Item(
            "generator",
            name,
            gen.summary,
            fn,
            meta={"describe": {"category": category, "params": [p.describe() for p in gen.params]}},
            undo=lambda: assets.unregister_generator(name),
        )
        try:
            REGISTRY.add(item)
        except PluginError:
            assets.unregister_generator(name)
            raise
        return fn

    return register


def material(name: str, **surface: Any) -> Any:
    """Register a surface the viewer draws, `material(name, roughness=, metalness=, clearcoat=, sheen=, glow=,
    texture=, texture_scale=, texture_strength=)`; a generator's pieces name it. texture is one of none, brushed,
    weave, speckle, rubber, knurl."""
    from robotforge import assets

    summary = surface.pop("summary", "") or f"a {name} surface (from a plugin)"
    try:
        made = assets.Material(name, summary=summary, **surface)
        assets.register_material(made)
    except (TypeError, ValueError) as exc:
        raise PluginError(f"material {name!r}: {exc}") from None
    from robotforge.assets.materials import unregister_material

    try:
        REGISTRY.add(
            Item(
                "material",
                name,
                made.summary,
                made,
                meta={"describe": made.surface()},
                undo=lambda: unregister_material(name),
            )
        )
    except PluginError:
        unregister_material(name)
        raise
    return made


def style(name: str, *, based_on: str = "cobot", summary: str = "", **changes: Any) -> Any:
    """Register a style, `style(name, based_on="cobot", fillet=1.6, ...)`: the named style with some choices changed
    (any field of robotforge.assets.Style)."""
    from robotforge import assets

    try:
        base = assets.get_style(based_on)
        made = base.tuned(name=name, summary=summary or f"{name}, from {based_on} (a plugin)", **changes)
        assets.register_style(made)
    except (TypeError, ValueError) as exc:
        raise PluginError(f"style {name!r}: {exc}") from None
    try:
        REGISTRY.add(
            Item(
                "style",
                name,
                made.summary,
                made,
                meta={"describe": {"based_on": based_on, **changes}},
                undo=lambda: assets.unregister_style(name),
            )
        )
    except PluginError:
        assets.unregister_style(name)
        raise
    return made
