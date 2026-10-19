"""Printed labels as geometry: text, bars, a barcode and a logo as thin flat patches lying on a face of an object.

Nothing here is a texture: the ink is a few hundred flat quads a fraction of a millimetre over the surface, so it
renders the same in the viewer and in MuJoCo's renderer and costs a few thousand triangles. Text is a 5 x 7 pixel
font drawn as runs of rectangles. A `Sheet` is a flat face (an origin and two axes) and a `Wrap` a cylinder's
surface; both give `(rgba, mesh)` patches for `Gen.decor`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from robotforge.design.visuals import mesh as M

LIFT = 0.0006  # m: the first layer of ink sits this far over the surface (more would show as a ridge, less z-fights)
LAYER = 0.0004  # m: each later layer (text on a bar) sits this much over the one under it

_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "%": ("11001", "11010", "00010", "00100", "01000", "01011", "10011"),
    " ": ("00000",) * 7,
}
Patch = tuple[tuple[float, float, float, float], M.Mesh]


def text_width(text: str, height: float) -> float:
    """Width of `text` at a cap height of `height` (6 pixels of advance per letter)."""
    return len(text) * 6 * height / 7 - height / 7


def _quads(rects: np.ndarray, to_world) -> M.Mesh:
    """Flat quads from rectangles (x0, y0, x1, y1) in a face's 2D frame; `to_world(u, v)` maps to 3D points and
    returns also the face normal."""
    if len(rects) == 0:
        return M.empty()
    verts, faces, normals = [], [], []
    for x0, y0, x1, y1 in rects:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        pts, nrm = zip(*(to_world(u, v) for u, v in corners), strict=True)
        base = len(verts)
        verts.extend(pts)
        normals.extend(nrm)
        faces.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int32)
    n = np.asarray(normals, dtype=float)
    # faces must wind counter-clockwise seen from outside: flip those that wind against their normal
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    wrong = np.einsum("ij,ij->i", np.cross(b - a, c - a), n[f[:, 0]]) < 0
    f[wrong] = f[wrong][:, ::-1]
    return M.Mesh(v, n, f, np.arange(len(v)))


def _text_rects(
    text: str, x: float, y: float, height: float, align: str = "left"
) -> list[tuple[float, float, float, float]]:
    """Rectangles (u0, v0, u1, v1) of `text` with its baseline at v = y; `x` is its left, middle or right."""
    px = height / 7
    width = text_width(text, height)
    start = x - (width / 2 if align == "center" else width if align == "right" else 0.0)
    out = []
    for i, ch in enumerate(text.upper()):
        glyph = _FONT.get(ch, _FONT[" "])
        for row, bits in enumerate(glyph):
            col = 0
            while col < 5:
                if bits[col] == "1":
                    run = col
                    while run + 1 < 5 and bits[run + 1] == "1":
                        run += 1
                    u0 = start + (i * 6 + col) * px
                    u1 = start + (i * 6 + run + 1) * px
                    v1 = y + (6 - row + 1) * px
                    out.append((u0, v1 - px, u1, v1))
                    col = run + 1
                else:
                    col += 1
    return out


class Sheet:
    """A flat face: ink at (u, v) on the plane through `origin` spanned by the unit vectors `u_axis` and `v_axis`,
    lying `normal` outward (u cross v by default). Lengths are in meters."""

    def __init__(self, origin: Sequence[float], u_axis: Sequence[float], v_axis: Sequence[float]) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.u = np.asarray(u_axis, dtype=float)
        self.v = np.asarray(v_axis, dtype=float)
        self.normal = np.cross(self.u, self.v)
        self.patches: list[Patch] = []

    def _world(self, layer: int):
        """Maps (u, v) to a point on the face, `layer` layers of ink over it (text on a bar is layer 1 over it)."""
        lift = LIFT + LAYER * layer
        return lambda u, v: (self.origin + self.u * u + self.v * v + self.normal * lift, self.normal)

    def rect(self, rgba, u0: float, v0: float, u1: float, v1: float, layer: int = 0) -> None:
        self.patches.append((_rgba(rgba), _quads(np.array([[u0, v0, u1, v1]]), self._world(layer))))

    def text(self, rgba, text: str, x: float, y: float, height: float, align: str = "left", layer: int = 1) -> None:
        self.patches.append((_rgba(rgba), _quads(np.array(_text_rects(text, x, y, height, align)), self._world(layer))))

    def barcode(self, rgba, x: float, y: float, width: float, height: float, seed: int = 0, layer: int = 1) -> None:
        rng = np.random.default_rng(seed)
        unit, u, bars = width / 50, 0.0, []
        while True:
            w, gap = unit * int(rng.integers(1, 4)), unit * int(rng.integers(1, 3))
            if u + w > width:
                break
            bars.append((x + u, y, x + u + w, y + height))
            u += w + gap
        self.patches.append((_rgba(rgba), _quads(np.array(bars), self._world(layer))))

    def disc(self, rgba, x: float, y: float, radius: float, steps: int = 20, layer: int = 0) -> None:
        """A round logo, as a fan of thin strips (flat quads only)."""
        rects = []
        for k in range(steps):
            v = -radius + (k + 0.5) * 2 * radius / steps
            half = math.sqrt(max(radius * radius - v * v, 0.0))
            rects.append((x - half, y + v - radius / steps, x + half, y + v + radius / steps))
        self.patches.append((_rgba(rgba), _quads(np.array(rects), self._world(layer))))

    def stripes(self, rgba, x0: float, y0: float, x1: float, y1: float, count: int, layer: int = 1) -> None:
        h = (y1 - y0) / (2 * count - 1)
        rects = np.array([(x0, y0 + 2 * k * h, x1, y0 + (2 * k + 1) * h) for k in range(count)])
        self.patches.append((_rgba(rgba), _quads(rects, self._world(layer))))


class Wrap:
    """Ink on a cylinder of radius `radius` about the z axis: `u` runs round it in meters of arc from the angle
    `center` (rad), `v` is the height."""

    def __init__(self, radius: float, center: float = 0.0) -> None:
        self.radius, self.center = radius, center
        self.patches: list[Patch] = []

    def _world(self, layer: int):
        lift = LIFT + LAYER * layer

        def at(u: float, v: float):
            a = self.center + u / self.radius
            r = self.radius + lift
            return np.array([r * math.cos(a), r * math.sin(a), v]), np.array([math.cos(a), math.sin(a), 0.0])

        return at

    def _split(self, rects: np.ndarray, step: float = 0.004) -> np.ndarray:
        """Long rectangles cut into pieces a few millimetres wide, so they follow the curve."""
        out = []
        for x0, y0, x1, y1 in rects:
            n = max(1, math.ceil((x1 - x0) / step))
            xs = np.linspace(x0, x1, n + 1)
            out += [(a, y0, b, y1) for a, b in zip(xs[:-1], xs[1:], strict=True)]
        return np.array(out)

    def rect(self, rgba, u0: float, v0: float, u1: float, v1: float, layer: int = 0) -> None:
        self.patches.append((_rgba(rgba), _quads(self._split(np.array([[u0, v0, u1, v1]])), self._world(layer))))

    def band(self, rgba, v0: float, v1: float, layer: int = 0) -> None:
        """A ring all the way round."""
        arc = 2 * math.pi * self.radius
        self.rect(rgba, -arc / 2, v0, arc / 2, v1, layer)

    def text(self, rgba, text: str, x: float, y: float, height: float, align: str = "center", layer: int = 1) -> None:
        rects = np.array(_text_rects(text, x, y, height, align))
        self.patches.append((_rgba(rgba), _quads(self._split(rects, 0.0015), self._world(layer))))

    def disc(self, rgba, x: float, y: float, radius: float, steps: int = 16, layer: int = 1) -> None:
        rects = []
        for k in range(steps):
            v = -radius + (k + 0.5) * 2 * radius / steps
            half = math.sqrt(max(radius * radius - v * v, 0.0))
            rects.append((x - half, y + v - radius / steps, x + half, y + v + radius / steps))
        self.patches.append((_rgba(rgba), _quads(self._split(np.array(rects), 0.003), self._world(layer))))


def _rgba(color) -> tuple[float, float, float, float]:
    c = [float(v) for v in color]
    if len(c) == 3:
        c.append(1.0)
    return (c[0], c[1], c[2], c[3])


WORDS = (
    "OATS", "RICE", "TEA", "SALT", "PASTA", "COCOA", "SUGAR", "MILK", "HONEY", "BRAN", "CORN", "BEANS", "FLOUR",
    "MINT", "SOUP", "SODA", "JUICE", "COLA", "TUNA", "PEAS", "CHILI", "LIME", "PLUM", "COFFEE", "MUESLI",
)  # fmt: skip
