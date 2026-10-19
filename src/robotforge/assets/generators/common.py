"""Shared pieces for the built-in generators: screws, grooves, rings and the way pieces are collected."""

from __future__ import annotations

import math

import numpy as np

from robotforge.assets.asset import Piece
from robotforge.assets.registry import Args
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

CYAN = (0.45, 0.9, 1.0, 1.0)
GREEN = (0.35, 1.0, 0.55, 1.0)
SEAM = 0.0005  # m: half the width of a parting line


class Pieces:
    """A list of pieces a generator fills, with the style's materials already looked up."""

    def __init__(self, args: Args) -> None:
        self.style = args.style
        self.items: list[Piece] = []

    def add(self, role: str, mesh: M.Mesh, material: str | None = None, color=None) -> None:
        if not mesh.triangles:
            return
        self.items.append(Piece(role, material or self.material(role), mesh, color))

    def material(self, role: str) -> str:
        s = self.style
        return {
            "body": s.body_material,
            "accent": s.accent_material,
            "structure": s.structure_material,
            "trim": "brushed" if s.body_material in ("brushed", "anodized") else "metal",
            "metal": s.metal_material,
            "rubber": s.rubber_material,
            "glass": "glass",
            "led": "led",
        }[role]

    def moved(self, mesh: M.Mesh, pos=(0.0, 0.0, 0.0), rot=None) -> M.Mesh:
        return mesh.moved(pos, rot) if (rot is not None or any(pos)) else mesh

    def done(self) -> tuple[Piece, ...]:
        return tuple(self.items)


def segments(r: float, step: float = 0.004, low: int = 20, high: int = 64) -> int:
    """Segments round a circle of radius r, about `step` apart, a multiple of 4."""
    return int(min(high, max(low, 4 * round(2 * math.pi * r / step / 4))))


def screw_count(style_fasteners: float, base: int) -> int:
    """How many screws a circle of `base` gets at this style's fastener level (0 for none)."""
    return int(round(base * style_fasteners)) if style_fasteners > 0.05 else 0


def screw(r: float, h: float, socket: bool) -> tuple[M.Mesh, M.Mesh | None]:
    """A screw head standing on z = 0: (head, dark socket or None). A hex bolt, or a round socket-head cap screw."""
    if not socket:
        return P.bolt(r, h), None
    head = M.cylinder(r, h, segments=12, bevel=min(h * 0.3, r * 0.2), z0=0.0, arc=1)
    hole = M.cylinder(r * 0.46, h * 0.15, segments=6, z0=h * 0.9)
    return head, hole


def screw_ring(
    pieces: Pieces,
    radius: float,
    count: int,
    r: float,
    h: float,
    z: float,
    socket: bool,
    start: float = 0.0,
    flip: bool = False,
) -> None:
    """`count` screws round a circle of `radius` about z, heads standing on the plane z (up, or down when flip)."""
    if count <= 0:
        return
    head, hole = screw(r, h, socket)
    heads, holes = [], []
    rot = M.rot_x(math.pi) if flip else None
    for k in range(count):
        a = start + 2 * math.pi * k / count
        at = (radius * math.cos(a), radius * math.sin(a), z)
        heads.append(head.moved(at, rot))
        if hole is not None:
            holes.append(hole.moved(at, rot))
    pieces.add("metal", M.merge(heads))
    if holes:
        pieces.add("structure", M.merge(holes), material="dark")


def groove(r: float, z: float, width: float = 2 * SEAM, depth: float = 0.0004, seg: int = 32) -> M.Mesh:
    """A dark parting line round a cylinder of radius r about z: a thin ring standing a hair proud."""
    rings = [M.at_z(M.circle(r + depth, seg), z - width / 2), M.at_z(M.circle(r + depth, seg), z + width / 2)]
    return M.loft(rings, cap_start=False, cap_end=False)  # open: a line is all it is


def ring_led(r: float, z: float, tube: float, seg: int = 28) -> M.Mesh:
    return M.torus(r, tube, segments=seg, sides=5).moved((0, 0, z))


def hollow_cylinder(r_out: float, r_in: float, z0: float, z1: float, seg: int = 32) -> M.Mesh:
    """A tube about z: outer wall, inner wall and flat ends (a closed surface)."""
    prof = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1), (r_in, z0)]
    rings = [M.at_z(M.circle(r, seg), z) for r, z in prof]
    return M.loft(rings, sharp=(1, 2, 3), cap_start=False, cap_end=False)


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)
