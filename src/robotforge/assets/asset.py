"""What a generator makes: drawn pieces, collision shapes and mass, in one part's own frame.

An `Asset` carries three things that must agree:

- `pieces`: the visual meshes, each with a role (which color it takes from a palette) and a material (how it
  reflects). Smooth, detailed, and free to be as fine as looks need: physics never reads them.
- `colliders`: what bodies touch. Boxes, cylinders, capsules and spheres where they fit, convex meshes where they do
  not (`hulls_of` splits a concave mesh into convex pieces with CoACD).
- `mass_properties(density)`: mass, centre of mass and inertia of those colliders, so a part weighs what its
  collision shape says, from one density, not from a guess.

Frames: z is the part's main axis, x and y across it, the origin where `Asset.anchor` says in the generator's
documentation (usually the middle of the part, or its mounting face).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from robotforge.design.visuals import mesh as M

ROLES = ("body", "accent", "structure", "trim", "metal", "rubber", "glass", "led")
ColliderKind = Literal["box", "cylinder", "capsule", "sphere", "mesh"]


@dataclass(frozen=True)
class Piece:
    """A drawn mesh. `color` overrides the palette's color for the role (a glowing ring, a lens)."""

    role: str
    material: str
    mesh: M.Mesh
    color: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"piece role {self.role!r} must be one of {', '.join(ROLES)}")


@dataclass(frozen=True)
class Collider:
    """A collision shape. Sizes follow MuJoCo: box half extents (x, y, z); cylinder (radius, half length);
    capsule (radius, half length of its straight part); sphere (radius). A `mesh` collider is touched as the convex
    hull of its mesh. The cylinder and capsule stand along their own z, turned by `rot` (a 3x3 rotation)."""

    kind: ColliderKind
    size: tuple[float, ...] = ()
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[tuple[float, ...], ...] | None = None
    mesh: M.Mesh | None = None
    friction: float | None = None  # sliding friction, when the part should be grippy or slick

    def __post_init__(self) -> None:
        need = {"box": 3, "cylinder": 2, "capsule": 2, "sphere": 1, "mesh": 0}[self.kind]
        if len(self.size) != need:
            raise ValueError(f"a {self.kind} collider takes {need} size numbers, got {list(self.size)!r}")
        if self.kind == "mesh" and (self.mesh is None or self.mesh.triangles == 0):
            raise ValueError("a mesh collider needs a mesh")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"collider sizes must be positive, got {list(self.size)!r}")

    def matrix(self) -> np.ndarray:
        return np.eye(3) if self.rot is None else np.asarray(self.rot, dtype=float)

    def moved(self, pos: Sequence[float] = (0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> Collider:
        r = np.eye(3) if rot is None else np.asarray(rot, dtype=float)
        new_pos = r @ np.asarray(self.pos) + np.asarray(pos, dtype=float)
        new_rot = r @ self.matrix()
        mesh = None if self.mesh is None else self.mesh.moved(pos, rot)
        # a mesh carries its place in its vertices
        return Collider(
            self.kind,
            self.size,
            (0.0, 0.0, 0.0) if self.kind == "mesh" else tuple(float(v) for v in new_pos),
            None if self.kind == "mesh" else tuple(tuple(float(v) for v in row) for row in new_rot),
            mesh,
            self.friction,
        )


@dataclass(frozen=True)
class MassProperties:
    mass: float  # kg
    volume: float  # m^3
    com: np.ndarray  # centre of mass, in the part's frame
    inertia: np.ndarray  # 3x3 about the centre of mass, in the part's axes (kg m^2)


@dataclass(frozen=True)
class Asset:
    name: str
    pieces: tuple[Piece, ...]
    colliders: tuple[Collider, ...] = ()
    params: dict = field(default_factory=dict)  # what the generator was given, checked and completed
    style: str = ""
    seed: int = 0

    @property
    def triangles(self) -> int:
        return sum(p.mesh.triangles for p in self.pieces)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(low, high) corners of the drawn meshes."""
        pts = [p.mesh.vertices for p in self.pieces if p.mesh.triangles]
        if not pts:
            return np.zeros(3), np.zeros(3)
        allv = np.vstack(pts)
        return allv.min(axis=0), allv.max(axis=0)

    def moved(self, pos: Sequence[float] = (0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> Asset:
        return Asset(
            self.name,
            tuple(Piece(p.role, p.material, p.mesh.moved(pos, rot), p.color) for p in self.pieces),
            tuple(c.moved(pos, rot) for c in self.colliders),
            self.params,
            self.style,
            self.seed,
        )

    def mass_properties(self, density: float) -> MassProperties:
        """Mass, centre of mass and inertia of the colliders at `density` kg/m^3 (overlaps count twice: a part's
        colliders are meant not to overlap)."""
        if density <= 0:
            raise ValueError(f"density must be positive, got {density!r}")
        if not self.colliders:
            raise ValueError(f"asset {self.name!r} has no colliders, so it has no mass")
        volume = 0.0
        moment = np.zeros(3)
        shapes = []
        for c in self.colliders:
            v, com, inertia_unit = _solid(c)  # inertia per unit density about its own centre, in its own axes
            r = c.matrix() if c.kind != "mesh" else np.eye(3)
            shapes.append((v, np.asarray(c.pos) + r @ com if c.kind != "mesh" else com, r @ inertia_unit @ r.T))
            volume += v
            moment += shapes[-1][0] * shapes[-1][1]
        com = moment / volume
        total = np.zeros((3, 3))
        for v, center, inertia_unit in shapes:
            d = center - com
            total += inertia_unit + v * (float(d @ d) * np.eye(3) - np.outer(d, d))
        return MassProperties(density * volume, volume, com, density * total)


# ---- solids --------------------------------------------------------------------------------------------


def _solid(c: Collider) -> tuple[float, np.ndarray, np.ndarray]:
    """(volume, centre of mass, inertia at unit density about the centre) of a collider in its own frame."""
    if c.kind == "box":
        hx, hy, hz = c.size
        v = 8 * hx * hy * hz
        return v, np.zeros(3), np.diag([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy]) * v / 3.0
    if c.kind == "sphere":
        (r,) = c.size
        v = 4.0 / 3.0 * math.pi * r**3
        return v, np.zeros(3), np.eye(3) * (0.4 * v * r * r)
    if c.kind == "cylinder":
        r, h = c.size
        v = 2 * math.pi * r * r * h
        ia = v * (3 * r * r + 4 * h * h) / 12.0
        return v, np.zeros(3), np.diag([ia, ia, v * r * r / 2.0])
    if c.kind == "capsule":
        r, h = c.size
        v_cyl, v_sph = 2 * math.pi * r * r * h, 4.0 / 3.0 * math.pi * r**3
        v = v_cyl + v_sph
        axial = v_cyl * r * r / 2.0 + v_sph * 0.4 * r * r
        # each hemisphere's centre of mass is 3r/8 beyond the end of the cylinder
        lateral_cyl = v_cyl * (3 * r * r + 4 * h * h) / 12.0
        hemi = v_sph / 2.0
        about_hemi = hemi * 0.4 * r * r - hemi * (3 * r / 8) ** 2  # about the hemisphere's own centre
        lateral_sph = 2 * (about_hemi + hemi * (h + 3 * r / 8) ** 2)
        lateral = lateral_cyl + lateral_sph
        return v, np.zeros(3), np.diag([lateral, lateral, axial])
    return mesh_solid(c.mesh)  # type: ignore[arg-type]


def mesh_solid(mesh: M.Mesh) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centre of mass and inertia (unit density, about the centre, in the mesh's axes) of a closed mesh,
    by summing the tetrahedra its triangles make with the origin."""
    tri = mesh.vertices[mesh.faces]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = float(vol.sum())
    if abs(volume) < 1e-15:
        raise ValueError("the mesh encloses no volume (is it closed?)")
    sign = 1.0 if volume > 0 else -1.0  # faces turned inward still give a positive solid
    vol = vol * sign
    volume = abs(volume)
    centroid = (vol[:, None] * (a + b + c) / 4.0).sum(axis=0) / volume
    s = a + b + c
    second = (
        np.einsum("i,ij,ik->jk", vol / 20.0, a, a)
        + np.einsum("i,ij,ik->jk", vol / 20.0, b, b)
        + np.einsum("i,ij,ik->jk", vol / 20.0, c, c)
        + np.einsum("i,ij,ik->jk", vol / 20.0, s, s)
    )  # the integral of x x^T over the solid
    second_com = second - volume * np.outer(centroid, centroid)
    inertia = np.trace(second_com) * np.eye(3) - second_com
    return volume, centroid, inertia


def hulls_of(mesh: M.Mesh, max_pieces: int = 8, threshold: float = 0.05) -> list[M.Mesh]:
    """Convex pieces that together cover a concave mesh, from CoACD (slow to import, and a second or more to run).
    Each piece is a closed mesh a `mesh` collider can use."""
    try:
        import coacd
    except ImportError as exc:  # approved, but absent from slim installs
        raise RuntimeError("splitting a concave mesh needs the 'coacd' package") from exc
    coacd.set_log_level("error")
    result = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=threshold,
        max_convex_hull=max_pieces,
        preprocess_mode="auto",
        mcts_iterations=60,
        seed=0,
    )
    out = []
    for verts, faces in result:
        verts = np.asarray(verts, dtype=float)
        faces = np.asarray(faces, dtype=np.int32)
        out.append(M.Mesh(verts, np.zeros_like(verts), faces, np.arange(len(verts), dtype=np.int64)))
    if not out:
        raise RuntimeError("CoACD returned no pieces for this mesh")
    return out
