"""Servos: the case of a hobby or bus servo, with the details that make it read as one."""

from __future__ import annotations

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import Pieces, screw_count
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P


@generator(
    "servo_case",
    summary=(
        "the case of a servo: a rounded housing split by a parting line, a label, a connector at the back, mounting "
        "ears with screws and a gear boss under the horn (the horn turns with the joint and is drawn apart)"
    ),
    params=[
        Param("length", 0.0452, 0.01, 0.2, "the case's long side (x)"),
        Param("width", 0.0247, 0.008, 0.1, "its short side (y)"),
        Param("thickness", 0.035, 0.008, 0.1, "its thickness along the output shaft (z)"),
        Param("shaft", -0.0108, -0.1, 0.1, "where the output shaft stands along x, from the case's middle"),
        Param("horn", 0.0, 0.0, 0.05, "radius of the horn that will sit on the boss; 0 takes the usual share"),
    ],
    category="joint",
    anchor="case centred on the origin, the output shaft on the +z face at x = shaft",
)
def servo_case(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    hx, hy, hz = a.length / 2, a.width / 2, a.thickness / 2
    horn_r = a.horn if a.horn > 0 else 0.78 * min(hx, hy)
    small = min(hx, hy, hz)
    r = small * 0.3 * s.fillet
    pc.add("structure", M.rounded_box(hx, hy, hz, min(r, small * 0.6), segments=3, corner=min(r * 1.3, hy * 0.6)))
    if s.panel_lines and small > 0.006:  # the seam between the two halves of the case
        z = hz * 0.28
        band = M.slab(
            hx + 0.0003,
            hy + 0.0003,
            z - 0.0005,
            z + 0.0005,
            0.00012,
            0.00012,
            min(r * 1.3, hy * 0.6),
            segments=1,
            per_corner=2,
            fence=False,
            bottom_cap=False,
            top_cap=False,
        )
        pc.add("trim", band, material="metal")
    # the gear boss under the horn, a little proud of the +z face
    shaft_x = a.shaft
    boss = M.cylinder(horn_r * 1.05, hz * 0.12, segments=20, bevel=hz * 0.03, z0=hz - 0.0006, arc=1)
    pc.add("trim", boss.moved((shaft_x, 0.0, 0.0)), material="metal")
    # a label on the top face, toward the end away from the shaft, and a connector on that end
    end = -1.0 if shaft_x > 0 else 1.0
    if hx > 0.012:
        label = M.rounded_box(0.3 * hx, 0.55 * hy, 0.0004, 0.0003, segments=1, corner=0.0015)
        pc.add(
            "trim", label.moved((end * 0.45 * hx, 0.0, hz + 0.0002)), material="powder", color=(0.86, 0.87, 0.88, 1.0)
        )
        plug = M.rounded_box(0.0016 + 0.08 * hx, 0.5 * hy, 0.35 * hz, 0.0006, segments=1, corner=0.001)
        pc.add("structure", plug.moved((end * (hx + 0.0008 + 0.04 * hx), 0.0, -0.1 * hz)), material="dark")
    # mounting ears at both ends, each with a screw
    ear_len = max(0.002, 0.32 * hx)
    ear_t = max(0.0006, hz * 0.055)
    ear_z = hz * 0.42
    n = screw_count(s.fasteners, 1)
    for side in (1, -1):
        ear = M.rounded_box(ear_len / 2 + 0.001, hy * 0.92, ear_t, ear_t * 0.6, segments=1, corner=hy * 0.3)
        pc.add("structure", ear.moved((side * (hx + ear_len / 2 - 0.001), 0.0, ear_z)))
        if n:
            head = P.bolt(min(ear_len * 0.28, hy * 0.3), ear_t * 1.4)
            pc.add("metal", head.moved((side * (hx + ear_len * 0.48), 0.0, ear_z + ear_t - 0.0002)))
    return Asset("servo_case", pc.done(), (Collider("box", (hx, hy, hz)),))
