"""Reusable visual robot parts: actuator housings, horns, lenses, LEDs, wheels and feet.

Each function returns {preset: Mesh} in the part's own frame and is cached by its (rounded)
sizes, so recompiling a world reuses the same arrays. Callers place parts with `Visuals.add`.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from robotforge.design.visuals import mesh as M

Parts = dict[str, M.Mesh]

GLASS = [0.05, 0.07, 0.1, 1.0]  # nearly black, faintly blue: a lens or visor
METAL = [0.7, 0.72, 0.75, 1.0]
RUBBER = [0.08, 0.08, 0.09, 1.0]
LED_CYAN = [0.45, 0.9, 1.0, 1.0]
LED_GREEN = [0.35, 1.0, 0.55, 1.0]
LED_RED = [1.0, 0.25, 0.2, 1.0]
LED_WHITE = [1.0, 0.97, 0.88, 1.0]


def cached(fn):
    """lru_cache keyed on arguments with floats rounded to 0.01 mm, so tiny float noise still hits."""
    inner = functools.lru_cache(maxsize=1024)(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = tuple(round(a, 5) if isinstance(a, float) else a for a in args)
        kwargs = {k: round(v, 5) if isinstance(v, float) else v for k, v in kwargs.items()}
        return inner(*args, **kwargs)

    wrapper.cache_clear = inner.cache_clear  # type: ignore[attr-defined]
    return wrapper


def merge_parts(*groups: Parts) -> Parts:
    out: dict[str, list[M.Mesh]] = {}
    for g in groups:
        for k, m in g.items():
            out.setdefault(k, []).append(m)
    return {k: M.merge(v) for k, v in out.items()}


def moved(parts: Parts, pos=(0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> Parts:
    return {k: m.moved(pos, rot) for k, m in parts.items()}


@cached
def horn(r: float, t: float, screws: int = 4) -> Parts:
    """A round servo horn standing on the z = 0 plane: metal disc, dark center screw and (on horns big
    enough to see them) a screw circle."""
    segments = 24 if r > 0.01 else 18
    disc = M.cylinder(r, t, segments=segments, bevel=min(t * 0.4, r * 0.2), z0=0.0, arc=1)
    dark = [M.cylinder(r * 0.24, t * 0.45, segments=10, z0=t * 0.8)]
    for k in range(screws if r > 0.008 else 0):
        a = 2 * math.pi * k / screws + math.pi / 4
        head = M.cylinder(r * 0.1, t * 0.35, segments=6, z0=t * 0.85)
        dark.append(head.moved((r * 0.64 * math.cos(a), r * 0.64 * math.sin(a), 0)))
    return {"metal": disc, "dark": M.merge(dark)}


@cached
def servo_box(sx: float, sy: float, sz: float, horn_r: float) -> Parts:
    """A servo case of half sizes (sx, sy, sz) with its round horn on the +z face."""
    r = min(sx, sy, sz) * 0.3
    body = M.rounded_box(sx, sy, sz, r, segments=3)
    # A slightly proud collar under the horn, like the gear-train boss of a real servo.
    boss = M.cylinder(horn_r * 1.05, sz * 0.12, segments=24, bevel=sz * 0.03, z0=sz - 0.0006, arc=1)
    h = horn(horn_r, max(0.0016, sz * 0.09))
    return merge_parts({"dark": M.merge([body, boss])}, moved(h, (0, 0, sz + sz * 0.12 - 0.0012)))


@cached
def drum(r: float, half: float, cap_r: float, both: bool = True) -> Parts:
    """A cylindrical joint actuator along z (radius r, half length `half`) with metal end caps."""
    body = M.cylinder(r, 2 * half, segments=28, bevel=min(r * 0.16, half * 0.35), arc=2)
    caps = []
    for side in (1, -1) if both else (1,):
        h = horn(cap_r, max(0.0012, half * 0.16), screws=4)
        caps.append(moved(h, (0, 0, side * (half - 0.0006)), None if side > 0 else M.rot_x(math.pi)))
    return merge_parts({"dark": body}, *caps)


@cached
def lens(r: float, depth: float) -> Parts:
    """A camera lens looking along +z from the z = 0 plane: dark bezel, glass and a fine metal ring."""
    bezel = M.revolve(
        [
            (r * 0.66, -depth),
            (r, -depth),
            (r, depth * 0.15),
            (r * 0.9, depth * 0.4),
            (r * 0.7, depth * 0.4),
            (r * 0.66, 0),
        ],
        segments=28,
        sharp=(0, 1, 4, 5),
    )
    dome = [(r * 0.68 * math.cos(a), r * 0.2 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 5)]
    glass = M.lathe([(0, -depth * 0.6), (r * 0.68, -depth * 0.6), *dome], segments=28, sharp=(1,))
    ring = M.torus(r * 0.8, r * 0.045, segments=28, sides=6).moved((0, 0, depth * 0.4))
    return {"dark": bezel, "glass": glass, "metal": ring}


@cached
def led(r: float, depth: float = 0.0015) -> Parts:
    dome = [(r * math.cos(a), depth * math.sin(a)) for a in np.linspace(0, math.pi / 2, 4)]
    return {"led": M.lathe([(0, -depth), (r, -depth), *dome], segments=20, sharp=(1,))}


@cached
def ball_foot(r: float, shin_r: float) -> Parts:
    """A rubber ball foot of radius r at the origin with a metal collar where a shin of radius shin_r joins."""
    ball = M.sphere(r, rings=12, segments=24)
    collar = M.cylinder(shin_r * 1.12, r * 0.32, segments=20, bevel=r * 0.05, z0=r * 0.7)
    return {"rubber": ball, "metal": collar}


@cached
def tyre(r: float, half_w: float) -> Parts:
    """A wheel about z: tyre with tread blocks (tread on radius r, -half_w..half_w), spoked rim, hub."""
    shoulder = min(half_w * 0.45, r * 0.18)
    inner = r * 0.72
    prof = [(inner, -half_w * 0.9)]
    for a in np.linspace(0, math.pi / 2, 4):
        prof.append((r - shoulder + shoulder * math.sin(a), -half_w + shoulder - shoulder * math.cos(a)))
    for a in np.linspace(math.pi / 2, 0, 4):
        prof.append((r - shoulder + shoulder * math.sin(a), half_w - shoulder + shoulder * math.cos(a)))
    prof.append((inner, half_w * 0.9))
    n = 60
    rings = [M.at_z(M.circle(rr, n), z) for rr, z in prof]
    # Tread blocks: the rings across the tread step in and out a little, so the tyre shows grip.
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bump = 1.0 + 0.015 * np.sign(np.sin(a * 15 + 0.3))
    for k in range(3, len(rings) - 3):
        rings[k] = np.column_stack([rings[k][:, :2] * bump[:, None], rings[k][:, 2]])
    tyre_mesh = M.loft(rings, cap_start=False, cap_end=False)  # open inside: the rim fills it
    R = inner * 1.02
    rim = M.revolve(
        [
            (R * 0.66, -half_w * 0.8),
            (R, -half_w * 0.8),
            (R, half_w * 0.72),
            (R * 0.93, half_w * 0.86),
            (R * 0.8, half_w * 0.86),
            (R * 0.74, half_w * 0.5),
            (R * 0.66, half_w * 0.45),
        ],
        segments=48,
        sharp=(0, 1, 2, 5, 6),
    )
    spokes = []
    for k in range(5):
        spoke = M.rounded_box(R * 0.27, R * 0.075, half_w * 0.22, R * 0.05, segments=2)
        spokes.append(spoke.moved((R * 0.47, 0, half_w * 0.5)).moved(rot=M.rot_z(2 * math.pi * k / 5)))
    hub = M.cylinder(R * 0.26, half_w * 1.55, segments=28, bevel=half_w * 0.18, z0=-half_w * 0.75)
    back = M.cylinder(R * 0.7, half_w * 0.1, segments=36, z0=-half_w * 0.2)
    nut = bolt(R * 0.11, half_w * 0.25)
    return {
        "rubber": tyre_mesh,
        "shell": M.merge([rim, *spokes]),
        "dark": back,
        "metal": M.merge([hub, nut.moved((0, 0, half_w * 0.8))]),
    }


@cached
def bolt(r: float, h: float) -> M.Mesh:
    """A hex bolt head standing on the z = 0 plane."""
    return M.lathe([(0, 0), (r, 0), (r, h * 0.7), (r * 0.8, h), (0, h)], segments=6, sharp=(1, 2, 3))


def bolt_circle(radius: float, n: int, r: float, h: float, start: float = 0.0) -> M.Mesh:
    heads = []
    for k in range(n):
        a = start + 2 * math.pi * k / n
        heads.append(bolt(r, h).moved((radius * math.cos(a), radius * math.sin(a), 0)))
    return M.merge(heads)


@cached
def slats(hx: float, hz: float, count: int, depth: float) -> M.Mesh:
    """Horizontal vent slats in the x-z plane, `depth` thick along y."""
    out = []
    pitch = 2 * hz / count
    for k in range(count):
        z = -hz + pitch * (k + 0.5)
        out.append(M.rounded_box(hx, depth, pitch * 0.25, pitch * 0.12, segments=2).moved((0, 0, z)))
    return M.merge(out)
