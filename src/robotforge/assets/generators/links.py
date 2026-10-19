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
    scale = 1.0 + (tip - 1.0) * t - a.waist * np.sin(math.pi * t)
    hx = a.radius * scale
    hy = a.radius * a.width * scale
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


@generator(
    "link_shell",
    summary=(
        "the shell between two joints: a tapered, filleted tube in two halves with a parting line along its sides, "
        "collars where it meets the joint housings, a cable cover on the back in some styles"
    ),
    params=[
        Param("length", 0.25, 0.004, 2.0, "from the root joint's axis to the tip joint's axis"),
        Param("radius", 0.03, 0.003, 0.2, "half thickness at the root, front to back (x)"),
        Param("width", 1.0, 0.5, 2.2, "the section's half width along y as a multiple of its half thickness", unit="x"),
        Param("tip", 0.0, 0.0, 1.2, "the tip's size as a share of the root's; 0 lets the style decide", unit="x"),
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
    z = t * length
    hx, hy = _station_extents(a, t)
    smallest = float(min(hx.min(), hy.min()))
    share = min(1.0, 0.18 * s.fillet + (1.0 - 0.18 * s.fillet) * s.roundness)  # of the smaller half extent
    corner = np.minimum(hx, hy) * share
    per = 4 if smallest > 0.015 else 3
    gap = 0.0005 if s.panel_lines else 0.0
    if s.panel_lines:
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
