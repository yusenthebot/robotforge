"""Hands: finger pads that grip and a gripper body to hang them on."""

from __future__ import annotations

import math

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import Pieces, screw_count, screw_ring
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


@generator(
    "grip_pad",
    summary="a finger with a rubber grip pad: a shaped finger in the part's color, a pad with ridges across it",
    params=[
        Param("length", 0.05, 0.01, 0.25, "the finger's length"),
        Param("width", 0.024, 0.006, 0.12, "its width across the grip"),
        Param("thickness", 0.01, 0.003, 0.05, "its thickness (the direction it closes in)"),
        Param("ridges", 6, 0, 14, "grip ridges across the pad", kind="int", unit=""),
    ],
    category="tool",
    anchor="the finger runs along z from its root (z = 0) to its tip; the gripping face looks toward +x",
)
def grip_pad(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    L, W, T = a.length, a.width, a.thickness
    hx, hy = T / 2, W / 2
    pad_t = min(0.3 * T, 0.003)
    body_hx = hx - pad_t / 2
    pc.add(
        "body",
        M.rounded_box(
            body_hx, hy, L / 2, min(0.4 * body_hx, 0.003 * s.fillet), segments=2, corner=min(0.3 * hy, 0.004)
        ).moved((-pad_t / 2, 0.0, L / 2)),
    )
    pad_len = L * 0.62
    pad = M.rounded_box(
        pad_t / 2, hy * 0.9, pad_len / 2, min(0.3 * pad_t, 0.0008), segments=2, corner=min(0.2 * hy, 0.003)
    )
    z0 = L - pad_len / 2 - 0.1 * L
    pc.add("rubber", pad.moved((hx - pad_t / 2, 0.0, z0)))
    ridge_pitch = pad_len / (a.ridges + 1) if a.ridges else 0.0
    ridges = []
    for k in range(a.ridges):
        z = z0 - pad_len / 2 + ridge_pitch * (k + 1)
        ridges.append(M.box(0.0008, hy * 0.82, min(0.25 * ridge_pitch, 0.0014)).moved((hx + 0.0004, 0.0, z)))
    pc.add("rubber", M.merge(ridges))
    if screw_count(s.fasteners, 1):
        r = max(0.0011, min(0.14 * hy, 0.0022))
        head = P.bolt(r, 0.7 * r).moved((-hx + 0.0005, 0.0, 0.14 * L), M.rot_y(-math.pi / 2))
        pc.add("metal", head)
    return Asset("grip_pad", pc.done(), (Collider("box", (hx, hy, L / 2), (0.0, 0.0, L / 2), friction=1.5),))


@generator(
    "gripper_body",
    summary="the palm of a parallel gripper: a housing with side covers, two guide rails and a flange with bolts",
    params=[
        Param("width", 0.07, 0.03, 0.25, "across the closing direction (y), fingers included when open"),
        Param("depth", 0.036, 0.02, 0.12, "front to back (x)"),
        Param("height", 0.04, 0.015, 0.12, "from the flange to the finger roots (z)"),
    ],
    category="tool",
    anchor="flange on z = 0 (the arm's tool plate), fingers' roots on the +z face; the closing direction is y",
)
def gripper_body(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    hx, hy, H = a.depth / 2, a.width / 2, a.height
    flange_t = min(0.2 * H, 0.008)
    fr = min(hx, hy) * 0.95
    pc.add("trim", M.cylinder(fr, flange_t, segments=32, bevel=0.3 * flange_t, z0=0.0, arc=1), material="metal")
    screw_ring(
        pc,
        fr * 0.72,
        screw_count(s.fasteners, 4),
        max(0.0012, fr * 0.08),
        0.0012,
        flange_t - 0.0002,
        socket=s.name == "lab",
        start=math.pi / 4,
    )
    body_h = H - flange_t
    pc.add(
        "structure",
        M.rounded_box(hx, hy, body_h / 2, min(0.2 * hx, 0.004 * s.fillet), segments=3, corner=0.005).moved(
            (0, 0, flange_t + body_h / 2)
        ),
        material="dark",
    )
    for side in (1, -1):
        pc.add(
            "body",
            M.rounded_box(0.0012, hy * 0.86, body_h * 0.32, 0.001, segments=2, corner=0.004).moved(
                (side * (hx + 0.0006), 0, flange_t + body_h * 0.5)
            ),
        )
    for x in (0.5 * hx, -0.5 * hx):
        pc.add(
            "trim",
            M.cylinder(0.0022, 2 * hy * 0.92, segments=12).moved((x, 0, H + 0.0008), M.rot_x(math.pi / 2)),
            material="metal",
        )
    return Asset("gripper_body", pc.done(), (Collider("box", (hx, hy, H / 2), (0.0, 0.0, H / 2)),))
