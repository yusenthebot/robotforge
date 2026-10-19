"""A hyperparameter: its default, the range a search may draw from, and how to check a value.

Pure Python, so plugins can import it (`from robotforge.extend import Hyper`) without loading the simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class HyperError(ValueError):
    """A hyperparameter value that does not fit; the message says what does."""


@dataclass(frozen=True)
class Hyper:
    """One setting of an algorithm.

    default: the value used when none is given; its type (int, float, str, bool) is the setting's type.
    low, high: the range a search draws from (and the range a value must stay in); log=True draws evenly between
        the logarithms, for scales such as a learning rate.
    choices: instead of a range, the values a search picks from.
    help: one sentence for people and agents.
    search: False keeps it out of a search unless the search names it.
    minimum, maximum: the limits a value must stay within; low and high are only where a search looks first, so
        a search may be told to look wider (up to the limits).
    """

    default: Any
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] | None = None
    help: str = ""
    search: bool = True
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.choices is not None and self.default not in self.choices:
            raise HyperError(f"default {self.default!r} is not one of the choices {list(self.choices)}")
        if (self.low is None) != (self.high is None):
            raise HyperError("give both low and high, or neither")
        if self.low is not None and not self.low <= self.default <= self.high:  # type: ignore[operator]
            raise HyperError(f"default {self.default!r} is outside [{self.low}, {self.high}]")
        if self.log and (self.low is None or self.low <= 0):
            raise HyperError("a log scale needs a positive low")

    @property
    def searchable(self) -> bool:
        return self.search and (self.choices is not None or self.low is not None)

    @property
    def kind(self) -> str:
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, int):
            return "int"
        if isinstance(self.default, float):
            return "float"
        return "str"

    def sample(self, rng: Any) -> Any:
        """A random value from the range or the choices."""
        if self.choices is not None:
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.low is None:
            return self.default
        if self.log:
            value = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            value = rng.uniform(self.low, self.high)
        return int(round(value)) if self.kind == "int" else float(value)

    def check(self, name: str, value: Any) -> Any:
        """The value as this setting's type, or a HyperError saying what is allowed."""
        kind = self.kind
        try:
            if kind == "bool":
                if not isinstance(value, bool):
                    raise TypeError
                out: Any = value
            elif kind == "int":
                if isinstance(value, bool) or float(value) != int(value):
                    raise TypeError
                out = int(value)
            elif kind == "float":
                if isinstance(value, bool):
                    raise TypeError
                out = float(value)
                if not math.isfinite(out):
                    raise ValueError
            else:
                out = str(value) if not isinstance(value, str) else value
        except (TypeError, ValueError):
            raise HyperError(f"{name} must be a {kind} (default {self.default!r}), got {value!r}") from None
        if self.choices is not None and out not in self.choices:
            raise HyperError(f"{name} must be one of {list(self.choices)}, got {value!r}")
        if (self.minimum is not None and out < self.minimum) or (self.maximum is not None and out > self.maximum):
            lo = "-inf" if self.minimum is None else f"{self.minimum:g}"
            hi = "inf" if self.maximum is None else f"{self.maximum:g}"
            raise HyperError(f"{name} must be between {lo} and {hi}, got {value!r}")
        return out

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"default": self.default, "type": self.kind}
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.low is not None:
            out.update(low=self.low, high=self.high, **({"log": True} if self.log else {}))
        if self.minimum is not None or self.maximum is not None:
            out.update(limits=[self.minimum, self.maximum])
        if self.help:
            out["help"] = self.help
        return out
