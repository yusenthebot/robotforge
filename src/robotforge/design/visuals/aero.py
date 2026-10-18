"""Visual parts of aircraft: twisted propellers, brushless motors, a camera gimbal, landing skids.

Like parts.py, each function returns {preset: Mesh} (or one Mesh) in the part's own frame and is cached.
A propeller lies in the z = 0 plane with its hub on the z axis; it spins about z, counter-clockwise seen from
above when `spin` is +1 (its blades then lead with their raised edge that way, as a real CCW prop does).
"""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


def motor_size(diameter: float) -> tuple[float, float]:
    """(radius, height) of the brushless motor that turns a propeller this big (a 2212 for a 9 inch propeller:
    about 28 mm across and 30 mm tall with its mount)."""
    radius = float(np.clip(0.065 * diameter, 0.006, 0.045))
    return radius, 1.25 * radius


@M.cached
def blade(radius: float, hub: float, spin: int, pitch_ratio: float = 0.45) -> M.Mesh:
    """One blade along +x from the hub to `radius`: a thin airfoil-like section whose chord swells near the root
    and narrows to a round tip, twisted from steep at the root to flat at the tip (constant geometric pitch)."""
    stations = 14
    ring = 14
    chord_max = max(0.012, 0.22 * radius)
    pitch = pitch_ratio * 2 * radius
    rings = []
    for k in range(stations + 1):
        s = k / stations
        r = hub * 0.8 + (radius - hub * 0.8) * s
        # chord: widest at a third of the span, a rounded tip over the last tenth
        chord = chord_max * (0.55 + 0.75 * s - 0.95 * s * s)
        if s > 0.9:
            chord *= math.sqrt(max(1e-4, 1.0 - ((s - 0.9) / 0.1) ** 2))
        chord = max(chord, chord_max * 0.04)
        thick = max(0.0012, 0.11 * chord_max * (1.0 - 0.6 * s))
        angle = math.atan2(pitch, 2 * math.pi * max(r, 1e-3))
        a = np.linspace(0.0, 2 * math.pi, ring, endpoint=False)
        # Section in (y, z): flat underside, cambered top; leading edge toward +y for a CCW blade at +x.
        y = 0.5 * chord * np.cos(a)
        z = 0.5 * thick * np.sin(a) + 0.18 * thick * (1 - np.cos(a) ** 2)
        c, sn = math.cos(angle), math.sin(angle)
        yy, zz = y * c - z * sn, y * sn + z * c  # turned so the leading edge (+y) rises
        rings.append(np.column_stack([np.full(ring, r), spin * yy, zz]))
    return M.loft(rings)  # a mirrored (clockwise) blade too: loft turns a closed shape's faces outward itself


@M.cached
def propeller(diameter: float, spin: int, blades: int = 2) -> P.Parts:
    """A propeller in the z = 0 plane: "blades" (with their hub) and a domed "metal" spinner nut on top."""
    radius = diameter / 2
    hub = max(0.006, 0.09 * radius)
    parts = [blade(radius, hub, spin).moved(rot=M.rot_z(2 * math.pi * k / blades)) for k in range(blades)]
    boss = M.cylinder(hub, hub * 0.9, segments=24, bevel=hub * 0.2, z0=-hub * 0.45)
    dome = M.lathe(
        [(0.0, hub * 0.4), (hub * 0.75, hub * 0.4)]
        + [(hub * 0.75 * math.cos(a), hub * 0.4 + hub * 0.9 * math.sin(a)) for a in np.linspace(0.2, math.pi / 2, 5)],
        segments=20,
        sharp=(1,),
    )
    return {"blades": M.merge([*parts, boss]), "metal": dome}


@M.cached
def motor(radius: float, height: float) -> P.Parts:
    """A brushless outrunner below z = 0: a dark mount on the arm, the turning metal bell with cooling slots
    above it up to the propeller plane (z = 0)."""
    base_h = height * 0.4
    mount = M.cylinder(radius * 1.08, base_h, segments=28, bevel=base_h * 0.25, z0=-height)
    bell = M.cylinder(radius, height - base_h, segments=32, bevel=radius * 0.12, z0=-height + base_h)
    slots = []
    for k in range(10):  # dark slots round the bell
        a = 2 * math.pi * k / 10
        slot = M.box(radius * 0.06, radius * 0.16, (height - base_h) * 0.28)
        slots.append(slot.moved((radius * 0.97 * math.cos(a), radius * 0.97 * math.sin(a), -height * 0.32), M.rot_z(a)))
    cap = M.cylinder(radius * 0.55, height * 0.08, segments=24, z0=-height * 0.04)
    return {"dark": M.merge([mount, *slots]), "metal": M.merge([bell, cap])}


@M.cached
def bell(radius: float, height: float) -> P.Parts:
    """The part of a motor that turns with its propeller (the bell), from z = -height up to the prop plane."""
    body = M.cylinder(radius, height, segments=32, bevel=radius * 0.12, z0=-height)
    ribs = []
    for k in range(10):
        a = 2 * math.pi * k / 10
        rib = M.box(radius * 0.05, radius * 0.18, height * 0.3)
        ribs.append(rib.moved((radius * 0.98 * math.cos(a), radius * 0.98 * math.sin(a), -height * 0.5), M.rot_z(a)))
    return {"metal": body, "dark": M.merge(ribs)}


@M.cached
def stator_mount(radius: float, height: float) -> M.Mesh:
    """The fixed part of a motor on the end of an arm, below the bell (z from -height to 0)."""
    return M.cylinder(radius, height, segments=28, bevel=height * 0.25, z0=-height)


@M.cached
def gimbal(size: float) -> P.Parts:
    """A camera on a two-axis gimbal hanging below z = 0, looking along +x."""
    s = size
    yoke_top = M.rounded_box(s * 0.25, s * 0.45, s * 0.05, s * 0.04, segments=3).moved((0, 0, -s * 0.05))
    arms = [
        M.rounded_box(s * 0.12, s * 0.04, s * 0.28, s * 0.03, segments=2).moved((0, y, -s * 0.35))
        for y in (-s * 0.47, s * 0.47)
    ]
    head = M.rounded_box(s * 0.36, s * 0.38, s * 0.3, s * 0.12, segments=4).moved((s * 0.05, 0, -s * 0.55))
    lens = P.lens(s * 0.2, s * 0.06)
    lens = P.moved(lens, (s * 0.41, 0, -s * 0.55), M.Z_TO_X)
    return P.merge_parts({"dark": M.merge([yoke_top, *arms])}, {"shell": head}, lens)


@M.cached
def skid(length: float, half_width: float, top_width: float, drop: float, radius: float) -> M.Mesh:
    """One landing skid: two legs from (+-length/2 * 0.6, top_width, 0) down to a bar at (y = half_width,
    z = -drop) that runs along x, bent with rounded corners."""
    a = length / 2
    leg_x = a * 0.62
    path_front = [(leg_x, top_width, 0.0), (leg_x, half_width * 0.95, -drop * 0.9), (leg_x * 1.05, half_width, -drop)]
    bar = [(x, half_width, -drop) for x in np.linspace(a * 0.98, -a * 0.98, 9)]
    tip_f = [(a * 1.05, half_width, -drop * 0.88)]
    tip_b = [(-a * 1.05, half_width, -drop * 0.88)]
    bar_path = np.array([*tip_f, *bar, *tip_b], float)
    leg_f = np.array(path_front, float)
    leg_b = np.array([(-x, y, z) for x, y, z in path_front], float)
    tube = M.tube(bar_path, [radius] * len(bar_path), segments=12)
    legs = [M.tube(leg, [radius * 1.1, radius, radius], segments=12) for leg in (leg_f, leg_b)]
    return M.merge([tube, *legs])
