"""Joint housings: the barrel a rotary actuator is, drawn the way a real one is built."""

from __future__ import annotations

import math

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import (
    CYAN,
    Pieces,
    groove,
    hollow_cylinder,
    ring_led,
    screw_count,
    screw_ring,
    segments,
)
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M


@generator(
    "joint_actuator",
    summary=(
        "a rotary joint: a barrel housing with a rear cap, a dark output band, a flange with a bolt circle and a "
        "centre boss, a cable gland on the side and, in some styles, a light ring or cooling ribs"
    ),
    params=[
        Param("radius", 0.035, 0.004, 0.3, "housing radius"),
        Param("length", 0.07, 0.006, 0.4, "housing length along the axis"),
        Param("bolts", -1, -1, 16, "screws on the output flange; -1 lets the style decide", kind="int", unit=""),
        Param("gland", True, doc="a cable gland on the side of the rear cap", kind="bool"),
        Param("gland_angle", 0.0, -180.0, 180.0, "where the gland points round the axis, degrees from +x", unit="deg"),
    ],
    category="joint",
    anchor="axis z, the output flange on the +z face, the middle of the housing at the origin",
)
def joint_actuator(a: Args) -> Asset:
    s = a.style
    r, length = a.radius, a.length
    pc = Pieces(a)
    seg = segments(r, 0.005, 24, 56)
    fine = max(16, 4 * round(seg * 0.67 / 4))  # small details need fewer segments than the housing
    half = length / 2
    fillet = s.fillet
    cap = min(0.15 * length * s.wall, 0.5 * r, 0.6 * length)
    band = min(0.1 * length * s.wall, 0.3 * r)
    flange_t = min(0.07 * r, 0.25 * length)
    gap = 0.0007 * max(0.5, min(r / 0.03, 2.0))
    z_cap0, z_cap1 = -half, -half + cap
    z_body0, z_body1 = z_cap1 + gap, half - band - gap
    bevel = min(0.14 * r * fillet, 0.3 * (z_body1 - z_body0), 0.012)

    pc.add("body", M.cylinder(0.955 * r, cap, segments=seg, bevel=min(0.14 * r * fillet, 0.45 * cap), z0=z_cap0, arc=2))
    # a nameplate on the rear face: a dark recessed disc with a bright ring round it
    plate = max(0.0006, 0.012 * r)
    pc.add(
        "structure",
        M.cylinder(0.5 * r, plate, segments=fine, bevel=0.3 * plate, z0=z_cap0 - plate + 0.0002, arc=1),
        material="dark",
    )
    pc.add("trim", hollow_cylinder(0.58 * r, 0.5 * r, z_cap0 - plate * 0.6, z_cap0 + 0.0003, fine), material="metal")
    pc.add("body", M.cylinder(r, z_body1 - z_body0, segments=seg, bevel=bevel, z0=z_body0, arc=2))
    if s.panel_lines:
        pc.add("structure", groove(0.985 * r, z_cap1 + gap / 2, 2 * gap, seg=seg), material="dark")
        pc.add("structure", groove(0.985 * r, z_body1 + gap / 2, 2 * gap, seg=seg), material="dark")
    pc.add(
        "structure", M.cylinder(0.985 * r, band, segments=seg, bevel=min(0.06 * r * fillet, 0.4 * band), z0=half - band)
    )
    # the output: a bright flange standing a little proud of the band, a bolt circle, a dark ring, a centre boss
    flange_r = 0.8 * r
    pc.add("trim", M.cylinder(flange_r, flange_t, segments=seg, bevel=0.3 * flange_t, z0=half - 0.02 * length, arc=1))
    z_top = half - 0.02 * length + flange_t
    n = a.bolts if a.bolts >= 0 else screw_count(s.fasteners, 6)
    bolt_r = max(0.0009, min(0.07 * r, 0.004))
    screw_ring(
        pc,
        0.62 * r,
        n,
        bolt_r,
        0.55 * bolt_r + 0.0004,
        z_top - 0.0002,
        socket=s.name == "lab",
        start=math.pi / max(n, 1),
    )
    pc.add("structure", hollow_cylinder(0.42 * r, 0.33 * r, z_top - 0.0003, z_top + 0.0004, fine), material="dark")
    pc.add("trim", M.cylinder(0.3 * r, 0.04 * r + 0.0005, segments=fine, bevel=0.012 * r, z0=z_top - 0.0002, arc=1))
    if s.light_ring:
        pc.add("led", ring_led(0.985 * r, z_cap1 + gap / 2, max(0.0006, 0.018 * r), fine), color=CYAN)
    if a.gland:
        ang = math.radians(a.gland_angle)
        turn = M.rot_z(ang) @ M.rot_y(math.pi / 2)  # the gland's z becomes the radial direction
        zc = z_cap0 + cap * 0.55
        gr = max(0.0022, 0.17 * r)
        boss = M.cylinder(gr * 1.5, 0.5 * r, segments=18, bevel=gr * 0.25, z0=0.62 * r, arc=2)
        nut = M.cylinder(gr * 1.05, 0.28 * r, segments=6, z0=1.0 * r)
        pc.add("structure", boss.moved((0, 0, zc), turn))
        pc.add("metal", nut.moved((0, 0, zc), turn))
    ribs = int(round(3 * s.ribs))
    if ribs and z_body1 - z_body0 > 0.012:
        span = z_body1 - z_body0
        for k in range(ribs):
            z = z_body0 + span * (k + 1) / (ribs + 1)
            pc.add("body", M.cylinder(1.03 * r, min(0.03 * length, 0.004), segments=seg, bevel=0.0008, z0=z - 0.002))
    return Asset("joint_actuator", pc.done(), (Collider("cylinder", (r, half)),))
