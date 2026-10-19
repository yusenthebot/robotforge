"""Things a robot rolls and stands on: wheels with real tread, and feet."""

from __future__ import annotations

import math

import numpy as np

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import Pieces, hollow_cylinder, screw_ring, segments
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M

TREADS = ("auto", "slick", "ribbed", "lugged", "knobby")


def _tread_for(a: Args) -> str:
    if a.tread != "auto":
        return a.tread
    return {"cobot": "ribbed", "lab": "slick", "hobby": "lugged", "rugged": "knobby"}.get(a.style.name, "ribbed")


@generator(
    "wheel",
    summary=(
        "a wheel: a tyre with a tread (slick, ribbed, lugged or knobby), a rim with spokes or a closed dish, "
        "a hub with a nut and bolts"
    ),
    params=[
        Param("radius", 0.06, 0.01, 0.6, "tyre radius"),
        Param("width", 0.04, 0.004, 0.4, "tyre width along the axle"),
        Param("tread", "auto", doc="the pattern on the tyre", kind="choice", choices=TREADS),
        Param("spokes", 5, 0, 12, "spokes in the rim; 0 gives a closed dish", kind="int", unit=""),
    ],
    category="wheel",
    anchor="the axle is z, the wheel centred on the origin",
)
def wheel(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    r, half_w = a.radius, a.width / 2
    tread = _tread_for(a)
    n = segments(r, 0.005, 24, 72)
    shoulder = min(half_w * 0.45, r * 0.2) * min(s.fillet, 1.3)
    inner = r * 0.74
    prof = [(inner, -half_w * 0.9)]
    for ang in np.linspace(0, math.pi / 2, 4):
        prof.append((r - shoulder + shoulder * math.sin(ang), -half_w + shoulder - shoulder * math.cos(ang)))
    for ang in np.linspace(math.pi / 2, 0, 4):
        prof.append((r - shoulder + shoulder * math.sin(ang), half_w - shoulder + shoulder * math.cos(ang)))
    prof.append((inner, half_w * 0.9))
    rings = [M.at_z(M.circle(rr, n), z) for rr, z in prof]
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    amp = {"slick": 0.0, "ribbed": 0.0, "lugged": 0.02, "knobby": 0.045}[tread]
    lugs = max(8, n // 4)
    for k in range(3, len(rings) - 3):
        across = (k - 3) / max(len(rings) - 7, 1)  # 0 to 1 over the tread
        scale = np.ones(n)
        if tread == "ribbed":
            scale -= 0.014 * (np.abs(across - 0.5) < 0.12)  # a centre groove
            scale -= 0.014 * ((np.abs(across - 0.2) < 0.06) | (np.abs(across - 0.8) < 0.06))
        elif amp:
            phase = 0.0 if tread == "lugged" else (math.pi if across > 0.5 else 0.0)
            scale += amp * np.sign(np.sin(ang * lugs + 0.3 + phase)) * (1.0 if tread == "knobby" else 0.7)
        rings[k] = np.column_stack([rings[k][:, :2] * scale[:, None], rings[k][:, 2]])
    pc.add("rubber", M.loft(rings, cap_start=False, cap_end=False))
    # rim, spokes, hub
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
        segments=max(24, n * 4 // 5),
        sharp=(0, 1, 2, 5, 6),
    )
    pc.add("body", rim)
    for k in range(a.spokes):
        spoke = M.rounded_box(R * 0.27, R * 0.075, half_w * 0.22, R * 0.05, segments=2)
        pc.add("body", spoke.moved((R * 0.47, 0, half_w * 0.5)).moved(rot=M.rot_z(2 * math.pi * k / a.spokes)))
    if a.spokes == 0:
        pc.add(
            "structure",
            M.cylinder(R * 0.76, half_w * 0.62, segments=max(20, n * 3 // 5), bevel=half_w * 0.2, z0=-half_w * 0.2),
            material="dark",
        )
    else:
        pc.add(
            "structure",
            M.cylinder(R * 0.7, half_w * 0.1, segments=max(20, n * 3 // 5), z0=-half_w * 0.2),
            material="dark",
        )
    pc.add(
        "metal",
        M.cylinder(R * 0.26, half_w * 1.55, segments=max(16, n * 7 // 15), bevel=half_w * 0.18, z0=-half_w * 0.75),
    )
    nb = 4 if R > 0.03 else 0
    screw_ring(
        pc, R * 0.17, nb, max(0.0008, R * 0.03), max(0.0006, half_w * 0.15), half_w * 0.8, socket=s.name == "lab"
    )
    return Asset("wheel", pc.done(), (Collider("cylinder", (r, half_w)),))


@generator(
    "foot",
    summary="a foot for a leg: a rubber ball or a flat pad with a tread ring, and a metal collar where the shin joins",
    params=[
        Param("radius", 0.02, 0.005, 0.12, "ball radius, or the pad's"),
        Param("shin", 0.011, 0.002, 0.06, "radius of the shin that joins it"),
        Param("shape", "ball", doc="a rounded ball or a flat pad", kind="choice", choices=("ball", "pad")),
    ],
    category="foot",
    anchor="the foot's centre at the origin, the shin coming from +z; the sole at z = -radius",
)
def foot(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    r, shin = a.radius, a.shin
    seg = 24 if r >= 0.02 else 18
    if a.shape == "ball":
        pc.add("rubber", M.sphere(r, rings=seg // 2, segments=seg))
        collider = Collider("sphere", (r,))
    else:
        h = 0.55 * r
        body = M.cylinder(r, h, segments=seg, bevel=0.35 * h, z0=-r, arc=3)
        pc.add("rubber", body)
        if s.bumpers:  # a dark ring on the sole's top edge, the guard
            pc.add(
                "structure", hollow_cylinder(0.8 * r, 0.62 * r, -r + h - 0.0002, -r + h + 0.0006, seg), material="dark"
            )
        collider = Collider("cylinder", (r, h / 2), (0.0, 0.0, -r + h / 2))
    collar_h = r * 0.32
    top = r * 0.7 if a.shape == "ball" else -r + 0.55 * r - 0.1 * r
    pc.add("metal", M.cylinder(max(shin * 1.12, 0.002), collar_h, segments=seg * 5 // 6, bevel=r * 0.05, z0=top))
    return Asset("foot", pc.done(), (collider,))
