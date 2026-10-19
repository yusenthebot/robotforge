"""Object generators: a name, a size, a seed and parameters in; a finished object out.

    from robotforge.objects import generator, Range, Choice

    @generator("mug", size="[diameter, height]", default_size=(0.08, 0.095), color="white", material="ceramic",
               params={"handle": Choice("round", "d", "none"), "wall": Range(0.003, 0.006, 0.004)})
    def mug(g):
        ...   # g is the kit's Gen: g.part(...), g.decor(...), g.grasp(...), g.p("handle"), g.rng("x")

A registered generator is an `add_object` shape like any other (`add_object("m", shape="mug", seed=3)`), is saved in
world.yaml by name, size, seed and parameters, and builds the same object from the same four. The ones robotforge
ships (`library/`) register themselves when this package is imported; a plugin's register the same way and are listed
by `plugins()` and taken away again on a reload.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from robotforge.objects.kit import Built, Gen, Param

BuildFn = Callable[[Gen], None]
BUILT_IN: set[str] = set()


@dataclass(frozen=True)
class Generator:
    name: str
    fn: BuildFn
    size_hint: str
    default_size: tuple[float, ...]
    sizes: tuple[int, ...]
    color: str | tuple[float, ...]
    material: str
    summary: str
    params: dict[str, Param]
    keywords: tuple[str, ...] = ()
    fixed: bool = False
    kind: str = "thing"  # thing, container, tool, food, package: what describe() and sorting call it by
    default_mass: float | None = None
    palette: tuple[str, ...] = ()  # colors a seed picks from when the caller gave none
    hints: dict[str, dict[str, Any]] = field(default_factory=dict)  # a phrase in a description -> parameters it sets

    def check_params(self, given: dict[str, Any] | None) -> dict[str, Any]:
        """The caller's parameters, each checked against what the generator takes."""
        out: dict[str, Any] = {}
        for key, value in (given or {}).items():
            if key not in self.params:
                known = ", ".join(self.params) or "none"
                raise ValueError(f"{self.name} has no parameter {key!r}; its parameters are {known}")
            out[key] = self.params[key].check(key, value)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "size": self.size_hint,
            "default_size": list(self.default_size),
            "material": self.material,
            "summary": self.summary,
            "params": {
                k: _kind_of(p) | {"default": p.default}
                | ({"doc": p.doc} if p.doc else {})
                for k, p in self.params.items()
            },
        }


GENERATORS: dict[str, Generator] = {}


def _kind_of(p: Param) -> dict[str, Any]:
    if p.choices:
        return {"choices": list(p.choices)}
    if isinstance(p.default, bool):
        return {"flag": True}
    return {"range": [p.low, p.high]}


def generator(
    name: str,
    *,
    size: str,
    default_size: Sequence[float],
    color: str | Sequence[float] = "gray",
    material: str = "plastic",
    summary: str = "",
    params: dict[str, Param] | None = None,
    sizes: Sequence[int] | None = None,
    keywords: Sequence[str] = (),
    fixed: bool = False,
    kind: str = "thing",
    mass: float | None = None,
    palette: Sequence[str] = (),
    hints: dict[str, dict[str, Any]] | None = None,
    builtin: bool = False,
) -> Callable[[BuildFn], BuildFn]:
    """Register a function as the generator of the shape `name`; see the module notes."""

    def register(fn: BuildFn) -> BuildFn:
        from robotforge.design import components, props
        from robotforge.extend.registry import REGISTRY, Item, PluginError

        counts = tuple(sizes) if sizes is not None else (len(default_size),)
        if not default_size or any(s <= 0 for s in default_size) or len(default_size) not in counts:
            raise PluginError(f"object {name!r}: default_size must be {counts[0]} positive numbers")
        if name in props.SHAPES or name in GENERATORS:
            raise PluginError(f"object {name!r}: robotforge already has a shape of that name; pick another")
        if not builtin and REGISTRY.get("part", name) is not None:
            raise PluginError(f"object {name!r} is already added by {REGISTRY.get('part', name).plugin}")
        try:
            components.color(color)
        except ValueError as exc:
            raise PluginError(f"object {name!r}: color: {exc}") from None
        gen = Generator(
            name, fn, size, tuple(float(s) for s in default_size), counts,
            color if isinstance(color, str) else tuple(float(c) for c in color), material,
            summary or f"a {name} (from a plugin)", dict(params or {}), tuple(keywords), fixed, kind, mass,
            tuple(palette), dict(hints or {}),
        )  # fmt: skip
        GENERATORS[name] = gen
        if not builtin:  # a plugin's generator must work at its default size before anyone adds it to a world
            try:
                build(gen, gen.default_size, components.color(color), None, ())
            except Exception as exc:
                del GENERATORS[name]
                _build.cache_clear()
                raise PluginError(f"object {name!r} failed at its default size: {type(exc).__name__}: {exc}") from exc
        info = props.ShapeInfo(size, counts, gen.default_size, gen.color, fixed, True, gen.summary)

        def undo() -> None:
            GENERATORS.pop(name, None)
            _build.cache_clear()
            _footprint.cache_clear()
            props.unregister_shape(name)

        props.register_shape(
            name,
            info,
            None,
            lambda size_: footprint(gen, tuple(size_)),
            mass,
            builder=lambda **kw: _add(gen, **kw),
            builtin=builtin,
        )
        if builtin:
            BUILT_IN.add(name)
        else:
            REGISTRY.add(Item("part", name, gen.summary, fn, meta={"describe": gen.describe()}, undo=undo))
        return fn

    return register


def _add(gen: Generator, *, spec, body, owner, size, rgba, seed, params, mass, movable=True) -> Built:
    from robotforge.objects import compile as compiling

    built = build(gen, tuple(size), list(rgba), seed, _key(params))
    compiling.add_to_spec(spec, body, owner, built, mass=mass, movable=movable)
    return built


def _key(params: dict[str, Any] | None) -> tuple:
    return tuple(sorted((params or {}).items()))


def build(gen: Generator, size: tuple[float, ...], rgba: Sequence[float], seed: int | None, params_key: tuple) -> Built:
    """The finished object (cached: a world recompile redraws nothing it has drawn before)."""
    return _build(
        gen.name, tuple(round(float(s), 6) for s in size), tuple(round(float(c), 6) for c in rgba), seed, params_key
    )


@functools.lru_cache(maxsize=512)
def _build(name: str, size: tuple, rgba: tuple, seed: int | None, params_key: tuple) -> Built:
    gen = GENERATORS[name]
    explicit = gen.check_params(dict(params_key))
    g = Gen(name, size, list(rgba), seed, explicit, gen.params, gen.material)
    try:
        gen.fn(g)
        for key in gen.params:  # every parameter is resolved and stored, used or not
            g.p(key)
    except (ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} {list(size)}: {exc}") from exc
    return g.finish()


def object_for(
    name: str, size: Sequence[float] | None = None, color: Any = None, seed: int | None = None, **params: Any
) -> Built:
    """A generated object by shape name, for code and tests: `object_for("mug", seed=3, handle="d")`."""
    from robotforge.design import components

    gen = get(name)
    use = tuple(size) if size is not None else gen.default_size
    if len(use) not in gen.sizes:
        raise ValueError(f"{name} size is {gen.size_hint}")
    rgba = components.color(color if color is not None else gen.color)
    return build(gen, use, rgba, seed, _key(gen.check_params(params)))


def get(name: str) -> Generator:
    try:
        return GENERATORS[name]
    except KeyError:
        raise ValueError(
            f"no object generator called {name!r}; the generators are {', '.join(sorted(GENERATORS))}"
        ) from None


@functools.lru_cache(maxsize=256)
def _footprint(name: str, size: tuple) -> tuple[float, float]:
    gen = GENERATORS[name]
    from robotforge.design import components

    built = build(gen, size, components.color(gen.color), None, ())
    hx, hy = built.half
    # A handle or a spout is a little longer or shorter with the seed: leave room for the longest.
    return (hx * 1.15, hy * 1.15)


def footprint(gen: Generator, size: tuple[float, ...]) -> tuple[float, float]:
    return _footprint(gen.name, tuple(round(float(s), 6) for s in size))


def catalogue() -> list[dict[str, Any]]:
    """Every generator and what it takes, for agents and docs."""
    return [g.describe() for g in sorted(GENERATORS.values(), key=lambda g: (g.kind, g.name))]
