"""The modelling kit an object generator is written with: solids that carry a look, a collision shape and a mass.

A generator describes an object once, in the object's own frame (origin at the middle of its base, z up, the
object resting on the ground the way it is meant to stand or lie):

    g.part(mesh, "ceramic", pieces=[...])      # a visual mesh, the convex pieces that collide, what it is made of
    g.decor(mesh, rgba)                         # a visual only (a printed label, a stitched seam)
    g.grasp("handle", pos, jaw, opening)        # a place a gripper may take hold of it
    g.anchor("mouth", pos)                      # a named point other code can use (where to pour, where to insert)

The mass is the volume of each mesh times the density of its material (a hollow mug is its shell's volume), shared
out over the part's collision pieces by volume; the friction is the material's. Collision is always convex pieces:
boxes, cylinders, capsules, spheres, ellipsoids or the convex hull of a mesh, never a concave mesh. Hollow things
(a mug, a bowl, a bottle) are rings of box pieces round their wall (`vessel`), a handle is a chain of capsules.
`collision.decompose` splits any other mesh into convex pieces with CoACD, slowly and only once (the result is kept).
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robotforge.design.visuals import mesh as M
from robotforge.objects import materials

Vec3 = tuple[float, float, float]


# ---- small geometry ------------------------------------------------------------------------------------------


def volume(mesh: M.Mesh) -> float:
    """Volume enclosed by a closed mesh (a mesh with open parts gives only what its faces enclose)."""
    if mesh.faces.size == 0:
        return 0.0
    v, f = mesh.vertices, mesh.faces
    return abs(float(np.sum(v[f[:, 0]] * np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)


def mat_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
    """w-x-y-z quaternion of a rotation matrix."""
    m = np.asarray(rot, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((0.25 * s), (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    n = math.sqrt(sum(c * c for c in q))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def z_to(direction: Vec3 | np.ndarray) -> np.ndarray:
    """A rotation matrix taking the z axis to `direction`."""
    return M.z_to(direction)


def quat_z_to(direction: Vec3 | np.ndarray) -> tuple[float, float, float, float]:
    """The quaternion taking z to `direction` (what a capsule or cylinder along a segment needs)."""
    return mat_to_quat(z_to(direction))


def quat_z(angle: float) -> tuple[float, float, float, float]:
    """A turn of `angle` rad about z."""
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def rot_of(quat: tuple[float, float, float, float]) -> np.ndarray:
    return M.quat_to_mat(quat)


# ---- collision pieces ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """One convex collision shape. `size` is MuJoCo's: box half extents; cylinder (radius, half height); capsule
    (radius, half length of its straight part); sphere (radius); ellipsoid (radii). A hull has a closed convex mesh."""

    kind: str
    size: tuple[float, ...]
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    hull: M.Mesh | None = None
    material: str = "plastic"
    mass: float = 0.0
    solid: bool = True  # False: it adds mass but never collides (water in a bottle)

    @property
    def volume(self) -> float:
        s = self.size
        if self.kind == "box":
            return 8.0 * s[0] * s[1] * s[2]
        if self.kind == "cylinder":
            return math.pi * s[0] ** 2 * 2.0 * s[1]
        if self.kind == "capsule":
            return math.pi * s[0] ** 2 * 2.0 * s[1] + 4.0 / 3.0 * math.pi * s[0] ** 3
        if self.kind == "sphere":
            return 4.0 / 3.0 * math.pi * s[0] ** 3
        if self.kind == "ellipsoid":
            return 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
        return volume(self.hull) if self.hull is not None else 0.0

    def moved(self, pos: Vec3 = (0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> Piece:
        """The piece carried by a rotation and then a translation (mirroring and placing parts)."""
        r = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
        new_pos = tuple(float(v) for v in r @ np.asarray(self.pos) + np.asarray(pos))
        new_quat = mat_to_quat(r @ rot_of(self.quat))
        hull = self.hull.moved(pos, r) if self.hull is not None else None
        if hull is not None:  # a hull's vertices carry the placing already
            new_pos, new_quat = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        return Piece(self.kind, self.size, new_pos, new_quat, hull, self.material, self.mass, self.solid)


def box(hx: float, hy: float, hz: float, pos: Vec3 = (0, 0, 0), quat=(1.0, 0.0, 0.0, 0.0)) -> Piece:
    return Piece("box", (hx, hy, hz), _v(pos), quat)


def cylinder(radius: float, half_height: float, pos: Vec3 = (0, 0, 0), quat=(1.0, 0.0, 0.0, 0.0)) -> Piece:
    return Piece("cylinder", (radius, half_height), _v(pos), quat)


def capsule(radius: float, half_length: float, pos: Vec3 = (0, 0, 0), quat=(1.0, 0.0, 0.0, 0.0)) -> Piece:
    return Piece("capsule", (radius, half_length), _v(pos), quat)


def sphere(radius: float, pos: Vec3 = (0, 0, 0)) -> Piece:
    return Piece("sphere", (radius,), _v(pos))


def ellipsoid(radii: Vec3, pos: Vec3 = (0, 0, 0), quat=(1.0, 0.0, 0.0, 0.0)) -> Piece:
    return Piece("ellipsoid", _v(radii), _v(pos), quat)


def hull(mesh: M.Mesh) -> Piece:
    """The convex hull of a mesh (MuJoCo takes the hull of whatever mesh collides, so a non-convex one must be cut
    into pieces first)."""
    return Piece("hull", (), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), mesh)


def capsule_between(a: Vec3, b: Vec3, radius: float) -> Piece:
    """A capsule whose axis runs from `a` to `b` (its round ends reach `radius` beyond them)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    length = float(np.linalg.norm(d))
    mid = (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2
    return capsule(radius, length / 2, _v(mid), quat_z_to(d / length if length > 1e-9 else (0, 0, 1)))


def box_between(a: Vec3, b: Vec3, half_w: float, half_t: float, side: Vec3 = (0, 0, 1)) -> Piece:
    """A bar from `a` to `b` with half width `half_w` (along `side`) and half thickness `half_t`."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    length = float(np.linalg.norm(d))
    z = d / length
    x = np.cross(np.asarray(side, dtype=float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(np.array([1.0, 0, 0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mid = (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2
    return box(half_t, half_w, length / 2, _v(mid), mat_to_quat(np.column_stack([x, y, z])))


def _v(values) -> Vec3:
    return (float(values[0]), float(values[1]), float(values[2]))


def ring_of_boxes(
    r0: float, r1: float, z0: float, z1: float, thickness: float, n: int = 10, phase: float = 0.0
) -> list[Piece]:
    """A wall section as `n` boxes round the z axis: its outer surface runs from radius `r0` at height `z0` to `r1`
    at `z1`, `thickness` thick inward. A tilted wall gets tilted boxes."""
    d_r, d_z = r1 - r0, z1 - z0
    length = math.hypot(d_r, d_z)
    beta = math.atan2(d_r, d_z)  # lean of the section from vertical
    out = (math.cos(beta), -math.sin(beta))  # outward normal in the (radial, z) plane
    mid_r, mid_z = (r0 + r1) / 2 - out[0] * thickness / 2, (z0 + z1) / 2 - out[1] * thickness / 2
    inner = max(mid_r - thickness / 2, thickness)
    half_w = inner * math.tan(math.pi / n) + thickness * 0.25
    pieces = []
    for k in range(n):
        theta = phase + 2 * math.pi * k / n
        rot = M.rot_z(theta) @ M.rot_y(beta)
        pieces.append(
            box(
                thickness / 2,
                half_w,
                length / 2 + 0.0005,
                (mid_r * math.cos(theta), mid_r * math.sin(theta), mid_z),
                mat_to_quat(rot),
            )
        )
    return pieces


def vessel(outer: list[tuple[float, float]], wall: float, floor: float, n: int = 10, phase: float = 0.0) -> list[Piece]:
    """Collision for a hollow vessel: `outer` is its outside profile as (radius, height) from the bottom edge up to
    the rim, `wall` the wall's thickness, `floor` the bottom's. A floor disc, then a ring of boxes for each stretch
    of the wall."""
    pieces = [cylinder(max(outer[0][0], wall), floor / 2, (0, 0, floor / 2))]
    for (ra, za), (rb, zb) in zip(outer[:-1], outer[1:], strict=False):
        if zb - za < 1e-6:
            continue
        pieces += ring_of_boxes(ra, rb, za, zb, wall, n, phase)
    return pieces


def frustum(r0: float, r1: float, z0: float, z1: float, segments: int = 16) -> Piece:
    """A solid truncated cone as a convex hull (a bottle's shoulder, a cup that narrows toward its foot)."""
    return hull(M.lathe([(0.0, z0), (max(r0, 1e-4), z0), (max(r1, 1e-4), z1), (0.0, z1)], segments, sharp=(1, 2)))


# ---- grasps and results --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grasp:
    """A place a two-finger gripper may take hold of an object, with the object upright, in its own frame."""

    kind: str  # handle, rim, body, shaft, edge, top
    pos: Vec3  # where the tool point (between the finger pads) goes
    jaw: tuple[float, float]  # horizontal direction the jaws close along (unit)
    opening: float  # m the jaws must open to straddle the part (the width between the pads when closed on it)
    score: float = 1.0  # preference among the hints, higher first
    note: str = ""
    # The heights the tool point may be at (object frame): a hand whose palm would meet the top of the object holds it
    # higher up, within [lo, hi]; without them the hint is at `pos` exactly.
    lo: float | None = None
    hi: float | None = None
    side: bool = False  # taken with the gripper horizontal, from the side (a tall bottle's body), not from above

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pos": [round(v, 4) for v in self.pos],
            "jaw": [round(v, 3) for v in self.jaw],
            "opening": round(self.opening, 4),
            "score": self.score,
            "note": self.note,
            **({"side": True} if self.side else {}),
            **({"height_range": [round(self.lo, 4), round(self.hi, 4)]} if self.lo is not None else {}),
        }


@dataclass(frozen=True)
class Slot:
    """A named place something is put: a round socket, a flat surface or a tray, in the object's own frame."""

    name: str
    pos: Vec3  # where the base of what is put there rests (the middle of the opening, at its floor)
    kind: str  # socket, surface or tray
    radius: float | None = None  # a round opening
    half: tuple[float, float] | None = None  # a rectangular area, half extents along x and y
    rim: float = 0.0  # height (object frame) of the highest thing round it: what a carried load must clear
    depth: float = 0.0  # m from the rim down to the floor

    def as_dict(self) -> dict[str, Any]:
        out = {"name": self.name, "kind": self.kind, "pos": [round(v, 4) for v in self.pos], "rim": round(self.rim, 4)}
        if self.radius is not None:
            out["radius"] = round(self.radius, 4)
        if self.half is not None:
            out["half"] = [round(h, 4) for h in self.half]
        return out


@dataclass(frozen=True)
class Visual:
    preset: str
    rgba: tuple[float, float, float, float]
    mesh: M.Mesh


@dataclass
class Built:
    """An object as a generator made it: ready to be added to a MuJoCo spec (compile.py), or to be looked at."""

    shape: str
    size: tuple[float, ...]
    seed: int | None
    params: dict[str, Any]
    pieces: list[Piece]
    visuals: list[Visual]
    grasps: list[Grasp]
    anchors: dict[str, Vec3]
    mass: float
    extent: tuple[float, float, float, float, float, float]  # x0, x1, y0, y1, z0, z1 of the object (all parts)
    friction: tuple[float, float, float]
    facts: dict[str, Any] = field(default_factory=dict)  # what a task needs to know: mouth radius, slot depth, ...
    up: Vec3 = (0.0, 0.0, 1.0)  # the axis that is "up" for the object (a mug's opening faces it)
    slots: list[Slot] = field(default_factory=list)  # places things are put in or on it (a rack's sockets)

    @property
    def height(self) -> float:
        return self.extent[5]

    @property
    def half(self) -> tuple[float, float]:
        """Half extents (x, y) that cover the object about its origin."""
        x0, x1, y0, y1 = self.extent[:4]
        return (max(abs(x0), abs(x1)), max(abs(y0), abs(y1)))

    def describe(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "size": [round(s, 4) for s in self.size],
            "seed": self.seed,
            "params": self.params,
            "mass_kg": round(self.mass, 3),
            "height_m": round(self.height, 4),
            "pieces": len([p for p in self.pieces if p.solid]),
            "grasps": [g.as_dict() for g in self.grasps],
            "anchors": {k: [round(c, 4) for c in v] for k, v in self.anchors.items()},
            "slots": [s.as_dict() for s in self.slots],
            "facts": self.facts,
        }


def top_under(built: Built, center: tuple[float, float], jaw: tuple[float, float], half_jaw: float, half_across: float,
              above: float = -math.inf) -> float:  # fmt: skip
    """The highest point (object frame) of the object's solid pieces over a rectangle on the floor plane: centred at
    `center`, `half_jaw` long along `jaw` and `half_across` across it. Conservative: each piece counts by the box that
    bounds it. Only points over `above` (m) count, so a handle's own bar is not the thing that stops the palm."""
    u = np.array(jaw, dtype=float)
    u = u / (np.linalg.norm(u) or 1.0)
    v = np.array([-u[1], u[0]])
    top = -math.inf
    for p in built.pieces:
        if not p.solid:
            continue
        lo, hi = piece_bounds(p)
        if hi[2] <= above:
            continue
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) - np.asarray(center)
        along, across = corners @ u, corners @ v
        if (
            along.max() < -half_jaw
            or along.min() > half_jaw
            or across.max() < -half_across
            or across.min() > half_across
        ):
            continue
        top = max(top, float(hi[2]))
    return top


def center_of_mass(built: Built) -> np.ndarray:
    """Where the object's mass is, in its own frame (the pieces' middles weighted by their masses)."""
    total, weighted = 0.0, np.zeros(3)
    for p in built.pieces:
        if p.mass <= 0:
            continue
        lo, hi = piece_bounds(p)
        weighted += p.mass * (lo + hi) / 2
        total += p.mass
    return weighted / total if total > 0 else np.zeros(3)


def piece_bounds(p: Piece) -> tuple[np.ndarray, np.ndarray]:
    """The axis-aligned box (object frame) that bounds a collision piece."""
    if p.hull is not None:
        return p.hull.vertices.min(axis=0), p.hull.vertices.max(axis=0)
    c = np.asarray(p.pos, dtype=float)
    r = np.abs(rot_of(p.quat))
    if p.kind == "box":
        e = r @ np.asarray(p.size)
    elif p.kind == "cylinder":
        axis = np.abs(r[:, 2])
        e = p.size[0] * np.sqrt(np.maximum(1 - axis**2, 0.0)) + axis * p.size[1]
    elif p.kind == "capsule":
        e = r[:, 2] * p.size[1] + p.size[0]
    elif p.kind == "sphere":
        e = np.full(3, p.size[0])
    else:  # ellipsoid
        e = r @ np.asarray(p.size)
    return c - e, c + e


# ---- parameters ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One thing about an object that varies from one seed to the next, or that the caller may set."""

    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None
    doc: str = ""
    vary: bool = True  # False: a seed leaves it at the default (what pose a book lies in is not luck)

    def sample(self, rng: np.random.Generator) -> Any:
        if self.choices is not None:
            return self.choices[int(rng.integers(len(self.choices)))]
        if isinstance(self.default, bool):
            return bool(rng.integers(2))
        return float(rng.uniform(self.low, self.high))

    def check(self, name: str, value: Any) -> Any:
        """The value as the parameter takes it, or a ValueError saying what it may be."""
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(f"{name} is one of {', '.join(map(str, self.choices))}, not {value!r}")
            return value
        if isinstance(self.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{name} is true or false, not {value!r}")
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} is a number from {self.low:g} to {self.high:g}, not {value!r}") from None
        if not (self.low <= number <= self.high):
            raise ValueError(f"{name} is from {self.low:g} to {self.high:g}, not {number:g}")
        return number


def Range(low: float, high: float, default: float | None = None, doc: str = "", vary: bool = True) -> Param:  # noqa: N802
    return Param(float(default if default is not None else (low + high) / 2), float(low), float(high), None, doc, vary)


def Choice(*choices: Any, doc: str = "", vary: bool = True) -> Param:  # noqa: N802
    return Param(choices[0], None, None, tuple(choices), doc, vary)


def Flag(default: bool = False, doc: str = "", vary: bool = True) -> Param:  # noqa: N802
    return Param(bool(default), None, None, None, doc, vary)


# ---- the builder a generator gets ----------------------------------------------------------------------------


class Gen:
    """What a generator function receives: its size and color, its parameters and random numbers, and the methods
    that add solids, looks, grasp hints and anchors."""

    def __init__(
        self,
        shape: str,
        size: tuple[float, ...],
        rgba: list[float],
        seed: int | None,
        explicit: dict[str, Any],
        specs: dict[str, Param],
        material: str,
    ) -> None:
        self.shape, self.size, self.rgba, self.seed = shape, size, list(rgba), seed
        self.material = material  # the default material of the whole object; a generator may use others
        self._explicit, self._specs = explicit, specs
        self._pieces: list[Piece] = []
        self._visuals: list[Visual] = []
        self._grasps: list[Grasp] = []
        self._anchors: dict[str, Vec3] = {}
        self._slots: list[Slot] = []
        self._mass = 0.0
        self.facts: dict[str, Any] = {}
        self.up: Vec3 = (0.0, 0.0, 1.0)
        self._friction: list[tuple[float, float, float]] = []
        self.resolved: dict[str, Any] = {}
        self._extent = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf])

    # -- parameters and randomness --

    def rng(self, label: str = "") -> np.random.Generator:
        """A random stream for `label`, fixed by the seed: the same seed gives the same object, and adding a new
        label to a generator does not change the others."""
        return np.random.default_rng([0 if self.seed is None else self.seed, zlib.crc32(label.encode("utf-8"))])

    def p(self, name: str) -> Any:
        """A parameter: what the caller set, else (with a seed) a value drawn for this seed, else the default."""
        if name in self.resolved:
            return self.resolved[name]
        spec = self._specs[name]
        if name in self._explicit:
            value = spec.check(name, self._explicit[name])
        elif self.seed is not None and spec.vary:
            value = spec.sample(self.rng("param:" + name))
        else:
            value = spec.default
        self.resolved[name] = value
        return value

    # -- solids and looks --

    def part(
        self,
        mesh: M.Mesh | None,
        material: str | None,
        pieces: list[Piece],
        rgba: list[float] | tuple[float, ...] | None = None,
        preset: str | None = None,
        mass: float | None = None,
    ) -> None:
        """A solid: `mesh` is how it looks (None: not drawn), `pieces` how it collides, `material` what it is made
        of. Its mass is the mesh's volume times the material's density (or `mass`), shared over the pieces."""
        mat = materials.get(material or self.material)
        if mesh is not None:
            self._visuals.append(Visual(preset or mat.preset, _rgba(rgba or self.rgba), mesh))
            self._reach(mesh.vertices)
        total = mass if mass is not None else (volume(mesh) * mat.density if mesh is not None else 0.0)
        volumes = [max(p.volume, 1e-12) for p in pieces]
        for p, vol in zip(pieces, volumes, strict=True):
            share = total * vol / sum(volumes)
            self._pieces.append(Piece(p.kind, p.size, p.pos, p.quat, p.hull, mat.name, share, True))
            self._friction.append(mat.coefficients())
            if mesh is None:
                self._reach_piece(p)
        self._mass += total

    def decor(self, mesh: M.Mesh, rgba: list[float] | tuple[float, ...] | None = None, preset: str = "paint") -> None:
        """Something only drawn: it has no mass and no collision."""
        self._visuals.append(Visual(preset, _rgba(rgba or self.rgba), mesh))

    def contents(self, material: str, volume_m3: float, pos: Vec3, radius: float, half_height: float) -> None:
        """What a vessel holds (water in a bottle): adds its mass at `pos` and never collides."""
        mat = materials.get(material)
        mass = volume_m3 * mat.density
        self._pieces.append(
            Piece("cylinder", (radius, half_height), _v(pos), (1.0, 0.0, 0.0, 0.0), None, mat.name, mass, False)
        )
        self._mass += mass

    def grasp(
        self,
        kind: str,
        pos: Vec3,
        jaw: tuple[float, float],
        opening: float,
        score: float = 1.0,
        note: str = "",
        lo: float | None = None,
        hi: float | None = None,
        side: bool = False,
    ) -> None:
        """A place a two-finger gripper takes hold of: `lo` and `hi` (default: only `pos`'s height) are the heights the
        tool point may be at, so a hand whose palm would meet the object's top holds it higher up; `side` says it is
        taken with the gripper horizontal."""
        n = math.hypot(jaw[0], jaw[1]) or 1.0
        if (lo is None) != (hi is None):
            raise ValueError("a grasp needs both lo and hi, or neither")
        self._grasps.append(Grasp(kind, _v(pos), (jaw[0] / n, jaw[1] / n), opening, score, note, lo, hi, side))

    def anchor(self, name: str, pos: Vec3) -> None:
        self._anchors[name] = _v(pos)

    def slot(
        self,
        name: str,
        pos: Vec3,
        kind: str = "surface",
        radius: float | None = None,
        half: tuple[float, float] | None = None,
        rim: float = 0.0,
        depth: float = 0.0,
    ) -> None:
        """A place something is put (tasks and the arm planner read it): `pos` is where the base of what goes there
        rests, `rim` the height of the highest thing round it."""
        self._slots.append(Slot(name, _v(pos), kind, radius, half, rim, depth))

    def fact(self, **values: Any) -> None:
        self.facts.update(values)

    # -- the result --

    def _reach(self, vertices: np.ndarray) -> None:
        if len(vertices):
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            self._extent[[0, 2, 4]] = np.minimum(self._extent[[0, 2, 4]], lo)
            self._extent[[1, 3, 5]] = np.maximum(self._extent[[1, 3, 5]], hi)

    def _reach_piece(self, p: Piece) -> None:
        r = max(p.size) if p.size else 0.0
        if p.hull is not None:
            self._reach(p.hull.vertices)
            return
        if p.kind == "capsule":
            r = p.size[0] + p.size[1]
        elif p.kind == "cylinder":
            r = math.hypot(p.size[0], p.size[1])
        elif p.kind == "box":
            r = math.sqrt(sum(s * s for s in p.size))
        c = np.asarray(p.pos)
        self._reach(np.array([c - r, c + r]))

    def finish(self, mass: float | None = None, friction: float | None = None) -> Built:
        if not self._pieces:
            raise ValueError(f"the generator for {self.shape!r} made no solid: call g.part(...) at least once")
        if mass is not None and self._mass > 0:
            scale = mass / self._mass
            self._pieces = [
                Piece(p.kind, p.size, p.pos, p.quat, p.hull, p.material, p.mass * scale, p.solid) for p in self._pieces
            ]
            self._mass = mass
        fr = np.mean(self._friction, axis=0) if self._friction else np.array(materials.DEFAULT.coefficients())
        if friction is not None:
            fr = np.array([friction, fr[1], fr[2]])
        ext = self._extent.copy()
        if not np.all(np.isfinite(ext)):
            ext = np.zeros(6)
        return Built(
            shape=self.shape,
            size=self.size,
            seed=self.seed,
            params=dict(self.resolved),
            pieces=self._pieces,
            visuals=self._visuals,
            grasps=sorted(self._grasps, key=lambda g: -g.score),
            anchors=self._anchors,
            mass=self._mass,
            extent=tuple(float(v) for v in ext),  # type: ignore[arg-type]
            friction=(float(fr[0]), float(fr[1]), float(fr[2])),
            facts=self.facts,
            up=self.up,
            slots=self._slots,
        )


def _rgba(color) -> tuple[float, float, float, float]:
    c = [float(v) for v in color]
    if len(c) == 3:
        c.append(1.0)
    return (c[0], c[1], c[2], c[3])


def shade(rgba, k: float) -> list[float]:
    return [min(1.0, float(c) * k) for c in rgba[:3]] + [float(rgba[3]) if len(rgba) > 3 else 1.0]


def mix(a, b, t: float) -> list[float]:
    return [float(x) + (float(y) - float(x)) * t for x, y in zip(a[:3], b[:3], strict=True)] + [1.0]
