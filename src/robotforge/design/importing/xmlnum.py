"""Small helpers for reading numbers out of URDF attributes."""

from __future__ import annotations

import math


def floats(text: str | None, count: int, default: float = 0.0) -> list[float]:
    try:
        vals = [float(p) for p in (text or "").split()]
    except ValueError:
        vals = []
    vals = [v if math.isfinite(v) else default for v in vals]
    return (vals + [default] * count)[:count]
