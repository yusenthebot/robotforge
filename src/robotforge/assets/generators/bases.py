"""Bodies that things stand on or are carried in: a pedestal and a chassis shell."""

from __future__ import annotations

import math

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import (
    CYAN,
    GREEN,
    Pieces,
    groove,
    ring_led,
    screw_count,
    screw_ring,
    segments,
)
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


@generator(
    "pedestal",
    summary=(
        "a robot's base: a bolted mounting plate, a column that carries the first joint, a groove, a cable gland and, "
        "in some styles, a light ring"
    ),
    params=[
        Param("radius", 0.09, 0.02, 0.5, "mounting plate radius"),
        Param("height", 0.08, 0.01, 0.6, "from the mounting plane to the top of the column"),
        Param("column", 0.62, 0.3, 0.95, "the column's radius as a share of the plate's", unit="x"),
    ],
    category="body",
    anchor="z up from the mounting plane (z = 0) to the top; the first joint sits on the top face",
)
def pedestal(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    R, H = a.radius, a.height
    seg = segments(R, 0.006, 24, 64)
    plate_h = min(max(0.22 * H, 0.004), 0.03) * s.wall
    plate_h = min(plate_h, 0.6 * H)
    cr = R * a.column
    pc.add("structure", M.cylinder(R, plate_h, segments=seg, bevel=min(0.3 * plate_h, 0.006 * s.fillet), z0=0.0, arc=2))
    pc.add("trim", M.cylinder(R * 1.012, min(0.0016, 0.3 * plate_h), segments=seg, z0=0.0), material="metal")
    n = screw_count(s.fasteners, 6)
    bolt_r = max(0.0012, min(0.045 * R, 0.0045))
    screw_ring(
        pc, 0.84 * R, n, bolt_r, 0.6 * bolt_r + 0.0006, plate_h - 0.0002, socket=s.name == "lab", start=math.pi / 6
    )
    # column: slightly wider at the foot, the first joint's seat on top
    foot = cr * 1.06
    prof = [
        (0.0, plate_h),
        (foot, plate_h),
        (foot * 0.985, plate_h + 0.1 * (H - plate_h)),
        (cr, plate_h + 0.28 * (H - plate_h)),
        (cr, H - 0.01),
        (cr * 0.97, H),
        (0.0, H),
    ]
    pc.add("body", M.lathe(prof, seg, sharp=(1,)))
    if s.panel_lines:
        pc.add("structure", groove(cr * 0.995, plate_h + 0.5 * (H - plate_h), 0.0011, seg=seg), material="dark")
    if s.light_ring:
        pc.add("led", ring_led(foot * 0.995, plate_h + 0.0012, max(0.0007, 0.012 * cr)), color=CYAN)
    if s.cable_glands:
        gr = max(0.003, 0.1 * cr)
        turn = M.rot_y(math.pi / 2)
        z = plate_h + 0.4 * (H - plate_h)
        pc.add(
            "structure",
            M.cylinder(gr * 1.5, 0.3 * cr, segments=16, bevel=gr * 0.25, z0=0.84 * cr, arc=2).moved((0, 0, z), turn),
        )
        pc.add("metal", M.cylinder(gr * 1.05, 0.22 * cr, segments=6, z0=1.08 * cr).moved((0, 0, z), turn))
    colliders = (
        Collider("cylinder", (R, plate_h / 2), (0.0, 0.0, plate_h / 2)),
        Collider("cylinder", (cr, (H - plate_h) / 2), (0.0, 0.0, plate_h + (H - plate_h) / 2)),
    )
    return Asset("pedestal", pc.done(), colliders)


@generator(
    "chassis",
    summary=(
        "a robot's body shell: rounded edges, a parting line, a service hatch on top, vent slots at the back, corner "
        "screws and, in rugged styles, rubber corner guards"
    ),
    params=[
        Param("length", 0.3, 0.03, 1.5, "front to back (x)"),
        Param("width", 0.2, 0.03, 1.2, "side to side (y)"),
        Param("height", 0.08, 0.015, 0.8, "top to bottom (z)"),
        Param("hatch", True, doc="a service hatch panel on the top", kind="bool"),
        Param("vents", True, doc="vent slots on the back", kind="bool"),
    ],
    category="body",
    anchor="centred on the origin; x forward, y left, z up",
)
def chassis(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    hx, hy, hz = a.length / 2, a.width / 2, a.height / 2
    small = min(hx, hy, hz)
    rb = min(max(0.3 * small * s.fillet, 0.0012), 0.012)
    rc = min(max(0.35 * min(hx, hy) * s.fillet, rb), 0.03)
    per = 2 if rc < 0.004 else 3 if rc < 0.012 else 5
    pan = min(2 * hz * 0.34, 0.015)
    zs = -hz + pan
    gap = 0.0005
    finish = {"segments": 3, "per_corner": per, "fence": True}
    pc.add("body", M.slab(hx, hy, zs + gap, hz, 0.0005, rb, rc, **finish, bottom_cap=False))
    if s.panel_lines and small >= 0.008:
        inset = 0.0012
        pc.add(
            "structure",
            M.slab(hx - inset, hy - inset, -hz, zs - gap, rb, 0.0005, rc - inset, per_corner=per, top_cap=False),
            material="dark",
        )
        core = M.slab(
            hx - 0.0007,
            hy - 0.0007,
            zs - gap - 0.001,
            zs + gap + 0.001,
            0.0003,
            0.0003,
            rc - 0.0007,
            segments=1,
            per_corner=per,
            fence=False,
            bottom_cap=False,
            top_cap=False,
            rims=(inset - 0.0007, 0.0),
        )
        pc.add("structure", core, material="dark")
    else:
        pc.add("body", M.slab(hx, hy, -hz, zs - gap, rb, 0.0005, rc, **finish, top_cap=False))
    if a.hatch and hx > 0.04 and hy > 0.03:
        hw, hh = min(0.5 * hx, 0.12), min(0.55 * hy, 0.1)
        pc.add(
            "structure",
            M.rounded_box(hw, hh, 0.0006, 0.0006, segments=1, corner=min(0.01, hh * 0.3)).moved(
                (0.0, 0.0, hz + 0.0001)
            ),
            material="dark",
        )
        r = max(0.0011, min(0.05 * hh, 0.0026))
        head = P.bolt(r, 0.6 * r + 0.0004, underside=False)
        pc.add(
            "metal",
            M.merge(
                [
                    head.moved((sx * (hw - 2.2 * r), sy * (hh - 2.2 * r), hz + 0.0004))
                    for sx in (1, -1)
                    for sy in (1, -1)
                ]
            ),
        )
    if a.vents and hy >= 0.03 and (hz - zs) * 0.28 >= 0.005:
        grille_h = (hz - zs) * 0.28
        slats = P.slats(min(hy * 0.45, 0.05), grille_h, 4 if grille_h > 0.008 else 3, 0.0016, open_back=True)
        pc.add("structure", slats.moved((-hx - 0.0002, 0.0, (zs + hz) / 2), M.rot_z(math.pi / 2)), material="dark")
    n = screw_count(s.fasteners, 1)
    if n and hx >= 0.015 and hy >= 0.015:
        inset = float(min(max(min(hx, hy) * 0.22, 0.0045), 0.018))
        r = max(0.0012, min(min(hx, hy) * 0.07, 0.0028))
        head = P.bolt(r, 0.75 * r, underside=False)
        pc.add(
            "metal",
            M.merge(
                [head.moved((sx * (hx - inset), sy * (hy - inset), hz - 0.0002)) for sx in (1, -1) for sy in (1, -1)]
            ),
        )
    if s.bumpers and small >= 0.015:
        g = min(0.45 * small, 0.02)
        guard = M.rounded_box(g, g, g * 0.9, g * 0.5, segments=2)
        pc.add(
            "rubber",
            M.merge(
                [
                    guard.moved((sx * (hx - g * 0.55), sy * (hy - g * 0.55), -hz + g * 0.8))
                    for sx in (1, -1)
                    for sy in (1, -1)
                ]
            ),
        )
    if hx > 0.04:
        pc.add("led", P.led(0.0028, underside=False)["led"].moved((-0.5 * hx, 0.55 * hy, hz)), color=GREEN)
    return Asset("chassis", pc.done(), (Collider("box", (hx, hy, hz)),))
