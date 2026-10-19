"""The generator registry: named, parametric recipes for parts, each with ranges, a seed and a style.

A generator is a function from checked parameters to an `Asset` (drawn meshes, collision shapes, mass from density).
It declares its parameters with ranges, so an agent or a UI can list them, ask for a value and be told what the
range is when it is outside; the same name, parameters, style and seed always make the same asset.

    from robotforge.assets import generate
    asset = generate("joint_actuator", radius=0.04, length=0.09, style="lab")

Plugins register their own with `robotforge.extend.generator` (docs/ASSETS.md); built-in ones live in
robotforge.assets.generators. Nothing here imports mujoco.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from robotforge.assets.asset import Asset
from robotforge.assets.styles import Style, get_style

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ParamKind = Literal["float", "int", "choice", "bool"]


class AssetError(ValueError):
    """A generator was asked for something it cannot make, or registered wrongly; the message says what to change."""


@dataclass(frozen=True)
class Param:
    """One parameter of a generator. Floats are meters unless `unit` says otherwise."""

    name: str
    default: Any
    low: float | None = None
    high: float | None = None
    doc: str = ""
    kind: ParamKind = "float"
    choices: tuple[str, ...] = ()
    unit: str = "m"

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise AssetError(f"parameter name {self.name!r} must be an identifier")
        if self.kind in ("float", "int"):
            if self.low is None or self.high is None or self.low > self.high:
                raise AssetError(f"parameter {self.name!r} needs low <= high")
            if self.default is not None and not self.low <= self.default <= self.high:
                raise AssetError(
                    f"parameter {self.name!r}: default {self.default!r} is outside {self.low}..{self.high}"
                )
        if self.kind == "choice" and (not self.choices or self.default not in self.choices):
            raise AssetError(f"parameter {self.name!r}: the default must be one of its choices")

    def check(self, value: Any, generator: str) -> Any:
        """The value as this parameter's type, or an error that names the range."""
        where = f"{generator}: {self.name}"
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise AssetError(f"{where} is true or false, got {value!r}")
            return value
        if self.kind == "choice":
            if value not in self.choices:
                raise AssetError(f"{where} is one of {', '.join(self.choices)}, got {value!r}")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
            raise AssetError(f"{where} is a number between {self.low} and {self.high} {self.unit}, got {value!r}")
        number = float(value)
        if not math.isfinite(number) or not self.low <= number <= self.high:  # type: ignore[operator]
            raise AssetError(f"{where} is between {self.low} and {self.high} {self.unit}, got {value!r}")
        if self.kind == "int":
            if number != int(number):
                raise AssetError(f"{where} is a whole number between {self.low} and {self.high}, got {value!r}")
            return int(number)
        return number

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind, "default": self.default, "doc": self.doc}
        if self.kind in ("float", "int"):
            out.update(low=self.low, high=self.high, unit=self.unit)
        if self.kind == "choice":
            out["choices"] = list(self.choices)
        return out


class Args:
    """What a generator receives: its checked parameters as attributes, the style and a seeded random generator."""

    def __init__(self, values: dict[str, Any], style: Style, seed: int) -> None:
        self.__dict__.update(values)
        self.style = style
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._values = values

    def values(self) -> dict[str, Any]:
        return dict(self._values)


Build = Callable[[Args], Asset]


@dataclass(frozen=True)
class Generator:
    name: str
    summary: str
    params: tuple[Param, ...]
    build: Build
    category: str = "part"  # part, link, joint, wheel, sensor, body, tool, object, ...
    anchor: str = ""  # where the origin and z axis are, in words (shown to agents)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "category": self.category,
            "anchor": self.anchor,
            "params": [p.describe() for p in self.params],
        }


GENERATORS: dict[str, Generator] = {}


def register_generator(gen: Generator, *, replace: bool = False) -> Generator:
    if not NAME_RE.match(gen.name):
        raise AssetError(
            f"generator name {gen.name!r} must use lowercase letters, digits and _, start with a letter and have at "
            "most 32 characters"
        )
    if gen.name in GENERATORS and not replace:
        raise AssetError(f"generator {gen.name!r} already exists; pick another name")
    seen = set()
    for p in gen.params:
        if p.name in seen or p.name in ("style", "seed"):
            raise AssetError(f"generator {gen.name!r}: parameter name {p.name!r} is taken or reserved")
        seen.add(p.name)
    GENERATORS[gen.name] = gen
    return gen


def unregister_generator(name: str) -> None:
    GENERATORS.pop(name, None)


def generator(
    name: str,
    *,
    summary: str,
    params: Sequence[Param] = (),
    category: str = "part",
    anchor: str = "",
) -> Callable[[Build], Build]:
    """Register `build(args) -> Asset` under `name`."""

    def register(fn: Build) -> Build:
        register_generator(Generator(name, summary, tuple(params), fn, category, anchor))
        return fn

    return register


def names() -> list[str]:
    _load_builtin()
    return sorted(GENERATORS)


def get(name: str) -> Generator:
    _load_builtin()
    found = GENERATORS.get(name)
    if found is None:
        raise AssetError(f"unknown generator {name!r}; the generators are {', '.join(sorted(GENERATORS))}")
    return found


def iter_generators() -> Iterator[Generator]:
    _load_builtin()
    yield from (GENERATORS[n] for n in sorted(GENERATORS))


def resolve(name: str, params: dict[str, Any]) -> tuple[Generator, dict[str, Any]]:
    """The generator and its parameters, each checked against its range and completed with defaults."""
    gen = get(name)
    known = {p.name: p for p in gen.params}
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise AssetError(f"{name}: unknown parameter {', '.join(unknown)}; it takes {', '.join(known) or 'none'}")
    values = {p.name: (p.check(params[p.name], name) if p.name in params else p.default) for p in gen.params}
    return gen, values


def clamped(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """`params` with each number pulled inside its parameter's range: for callers that compute sizes (a template at
    any size) and would rather draw a part at its limit than not draw it. Unknown names are still refused."""
    gen = get(name)
    known = {p.name: p for p in gen.params}
    out = {}
    for key, value in params.items():
        p = known.get(key)
        if p is not None and p.kind in ("float", "int") and isinstance(value, int | float):
            value = min(max(value, p.low), p.high)  # type: ignore[type-var]
            value = int(round(value)) if p.kind == "int" else float(value)
        out[key] = value
    return out


def generate(name: str, *, style: str | Style | None = None, seed: int = 0, **params: Any) -> Asset:
    """Make the asset `name` makes for these parameters, in `style` (a name; default `cobot`), for `seed`.

    The same arguments always give the same asset (a small cache makes a repeat free)."""
    gen, values = resolve(name, params)
    chosen = get_style(style)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise AssetError(f"{name}: seed is a whole number, got {seed!r}")
    return _make(gen, tuple(sorted(values.items())), chosen, seed)


@functools.lru_cache(maxsize=512)
def _make(gen: Generator, values: tuple, style: Style, seed: int) -> Asset:
    try:
        asset = gen.build(Args(dict(values), style, seed))
    except AssetError:
        raise
    except Exception as exc:  # a plugin's code: say which generator, not a bare traceback from deep inside it
        raise AssetError(f"generator {gen.name!r} failed for {dict(values)}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(asset, Asset):
        raise AssetError(f"generator {gen.name!r} must return an Asset, got {type(asset).__name__}")
    return Asset(asset.name or gen.name, asset.pieces, asset.colliders, dict(values), style.name, seed)


_loaded = False


def _load_builtin() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        import robotforge.assets.generators  # noqa: F401  (registers the built-in generators)
