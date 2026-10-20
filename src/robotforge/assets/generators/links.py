"""Links: the shell between two joints, tapered and filleted, in two halves with a parting line."""

from __future__ import annotations

import math

import numpy as np

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import Pieces, hollow_cylinder
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M


def _station_extents(a: Args, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half extents (x, y) of the link's section at each station t in 0..1: from the root to the tip, with the
    waist `a.waist` narrows in the middle."""
    tip = a.tip if a.tip > 0 else a.style.taper
    tip_y = a.tip_width / a.width if a.tip_width > 0 else tip  # the tip's width as a share of the root's
    waist = a.waist * np.sin(math.pi * t)
    hx = a.radius * (1.0 + (tip - 1.0) * t - waist)
    hy = a.radius * a.width * (1.0 + (tip_y - 1.0) * t - waist)
    return hx, hy


def _half_rings(hx: np.ndarray, hy: np.ndarray, corner: np.ndarray, per: int, z: np.ndarray, gap: float):
    """Open rings of the front (x > gap) and back (x < -gap) halves at every station."""
    front, back = [], []
    for k in range(len(z)):
        ring = M.rounded_rect(hx[k], hy[k], corner[k], per)  # counter-clockwise, 4 * (per + 1) points
        q = per + 1
        k0, k1, k2, k3 = ring[:q], ring[q : 2 * q], ring[2 * q : 3 * q], ring[3 * q :]
        f = np.vstack([[gap, -hy[k]], k3, k0, [gap, hy[k]]])
        b = np.vstack([[-gap, hy[k]], k1, k2, [-gap, -hy[k]]])
        front.append(M.at_z(f, z[k]))
        back.append(M.at_z(b, z[k]))
    return front, back


def _dome_rings(hx: float, hy: float, corner: float, per: int, z_end: float, depth: float, sign: float):
    """Rings from the end section (at z_end) out to a pole `depth` beyond it, `sign` -1 below the link, +1 above."""
    rings = []
    count = 4 * (per + 1)
    for ang in np.linspace(math.pi / 2, 0.0, 5):
        k = math.sin(ang)
        z = z_end + sign * depth * (1 - math.cos(ang))
        if k < 1e-9:
            rings.append(np.tile([0.0, 0.0, z], (count, 1)))
        else:
            rings.append(M.at_z(M.rounded_rect(hx * k, hy * k, corner * k, per), z))
    return rings


@generator(
    "link_shell",
    summary=(
        "the shell between two joints: a tapered, filleted tube in two halves with a parting line along its sides, "
        "collars where it meets the joint housings, a cable cover on the back in some styles"
    ),
    params=[
        Param("length", 0.25, 0.004, 2.0, "from the root joint's axis to the tip joint's axis"),
        Param("radius", 0.03, 0.003, 0.2, "half thickness at the root, front to back (x)"),
        Param("width", 1.0, 0.2, 3.0, "the section's half width along y as a multiple of its half thickness", unit="x"),
        Param("tip", 0.0, 0.0, 1.2, "the tip's size as a share of the root's; 0 lets the style decide", unit="x"),
        Param(
            "tip_width",
            0.0,
            0.0,
            3.0,
            "half width at the tip, as a multiple of the root's half thickness; 0 keeps the root's proportion",
            unit="x",
        ),
        Param(
            "ends",
            "plate",
            doc="a recessed plate (a joint housing hides it) or a dome (an exposed end)",
            kind="choice",
            choices=("plate", "dome"),
        ),
        Param(
            "seam",
            "split",
            doc="a parting line: a gap between two shell halves, or a dark strip on a one-piece shell (domed ends)",
            kind="choice",
            choices=("split", "strip"),
        ),
        Param(
            "roundness",
            0.0,
            0.0,
            1.0,
            "how round the section's corners are, 0 square to 1 round; 0 lets the style decide",
            unit="",
        ),
        Param("sides", 0, 0, 96, "points round the section; 0 picks by size", kind="int", unit=""),
        Param("waist", 0.0, 0.0, 0.3, "how much narrower the middle is than a straight taper", unit="x"),
        Param("collar", 0.0, 0.0, 0.08, "radius of the collars at the two ends where joint housings meet; 0 for none"),
    ],
    category="link",
    anchor="along z from the root (z = 0) to the tip (z = length); x front to back, y across",
)
def link_shell(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    length = a.length
    stations = 9 if length > 0.08 else 5
    t = np.linspace(0.0, 1.0, stations)
    hx, hy = _station_extents(a, t)
    strip = a.seam == "strip" and s.panel_lines  # one piece: the exact surface a limb has always had, plus a line
    dome = a.ends == "dome" or strip
    # a domed end reaches 0.9 of the end section's smaller half extent; the straight part lies between the two domes
    d0 = min(0.9 * float(min(hx[0], hy[0])), 0.3 * length) if dome else 0.0
    d1 = min(0.9 * float(min(hx[-1], hy[-1])), 0.3 * length) if dome else 0.0
    z = d0 + t * (length - d0 - d1)
    smallest = float(min(hx.min(), hy.min()))
    share = a.roundness if a.roundness > 0 else min(1.0, 0.18 * s.fillet + (1.0 - 0.18 * s.fillet) * s.roundness)
    corner = np.minimum(hx, hy) * share
    per = a.sides // 4 - 1 if a.sides >= 8 else (4 if smallest > 0.015 else 3)
    gap = 0.0005 if s.panel_lines else 0.0
    if strip:
        body = M.link(
            length,
            (float(hx[0]), float(hy[0])),
            (float(hx[-1]), float(hy[-1])),
            roundness=share,
            rounding=(0.9, 0.9),
            stations=5,
            per_corner=per,
            domes=int(np.clip(per + 1, 4, 10)),
        )
        pc.add("body", body)
        for sign in (1.0, -1.0):  # a dark line along the middle of each flank, a hair proud of the shell
            line = [
                M.at_z(
                    np.array(
                        [[-0.0005, sign * (hy[k] - 0.0002)], [0.0005, sign * (hy[k] - 0.0002)]]
                        + [[0.0005, sign * (hy[k] + 0.00025)], [-0.0005, sign * (hy[k] + 0.00025)]]
                    ),
                    z[k],
                )
                for k in range(stations)
            ]
            pc.add("structure", M.loft(line), material="dark")
    elif s.panel_lines:
        front, back = _half_rings(hx, hy, corner, per, z, gap)
        pc.add("body", M.loft(front, closed=False))
        pc.add("body", M.loft(back, closed=False))
        # the dark core in the parting line, a hair inside the shell so the groove reads dark
        inset = 0.0008
        core = [
            M.at_z(
                np.array(
                    [[-gap, -(hy[k] - inset)], [gap, -(hy[k] - inset)], [gap, hy[k] - inset], [-gap, hy[k] - inset]]
                ),
                z[k],
            )
            for k in range(stations)
        ]
        pc.add("structure", M.loft(core), material="dark")
    else:
        rings = [M.at_z(M.rounded_rect(hx[k], hy[k], corner[k], per), z[k]) for k in range(stations)]
        pc.add("body", M.loft(rings))
    if dome and not strip:
        for k, depth, sign in ((0, d0, -1.0), (stations - 1, d1, 1.0)):
            pc.add("body", M.loft(_dome_rings(hx[k], hy[k], corner[k], per, z[k], depth, sign), cap_start=False))
    elif not strip:
        # the two ends are closed with a thin plate each (a joint housing normally hides it)
        for k in (0, stations - 1):
            inward = 1.0 if k == 0 else -1.0  # recessed a little, so it never shares a plane with a collar
            ring = M.at_z(M.rounded_rect(hx[k], hy[k], corner[k], per), z[k] + inward * 0.0012)
            pc.add("body", M.loft([ring, ring + np.array([0.0, 0.0, inward * 0.0006])]))
    # collars
    if a.collar > 0:
        c = a.collar
        for z_end, sign in ((0.0, 1), (length, -1)):
            h = min(0.2 * c + 0.002, 0.25 * length)
            z0 = z_end if sign > 0 else z_end - h
            inner = float(min(hx[0] if sign > 0 else hx[-1], hy[0] if sign > 0 else hy[-1])) * 0.9
            pc.add("structure", hollow_cylinder(c, min(inner, 0.92 * c), z0, z0 + h, 32))
    # the cable cover along the back (-x): a thin raised strip that follows the taper
    if s.fasteners >= 1.5 and length > 0.06:
        t0, t1 = float(hx[0]), float(hx[-1])
        cover_w = float(min(hy[0], hy[-1]) * 0.34)
        strip = M.link(
            length * 0.78,
            (0.0017, cover_w),
            (0.0017, cover_w * 0.9),
            roundness=0.8,
            rounding=(0.8, 0.8),
            stations=3,
            per_corner=2,
            domes=3,
        )
        slope = math.atan2(t1 - t0, length)
        lift = M.rot_y(-slope)  # lean the strip along the tapered back
        x_mid = -(t0 + (t1 - t0) * 0.5) - 0.0008
        pc.add("trim", strip.moved((x_mid, 0.0, length * 0.11), lift))
    reach = float(max(hx.max(), hy.max()))
    return Asset(
        "link_shell",
        pc.done(),
        (Collider("capsule", (reach * 0.9, max(length / 2 - reach * 0.9, 0.001)), (0.0, 0.0, length / 2)),),
    )
