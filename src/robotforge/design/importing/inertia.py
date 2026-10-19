"""Mass and inertia from shapes, and sanity checks for the values a URDF provides.

Everything here works on one link at a time, in the link's own frame. Inertia tensors are
3x3 matrices about the center of mass unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robotforge.design.importing.meshio import MeshAsset, solid_properties

DEFAULT_DENSITY = 1000.0  # kg/m^3, a light mix of plastic and aluminum
TINY_MASS = 0.01  # kg, for links that move but have neither shape nor mass
TINY_INERTIA = 1e-6  # kg m^2
MIN_VALID_MASS = 1e-6
# Smallest moment over the largest: below it the tensor is not positive once written with nine digits.
NEAR_SINGULAR = 1e-8
# Largest and smallest plausible principal moment, relative to mass * (radius of the link's shapes)^2.
MAX_INERTIA_RATIO = 25.0
MIN_INERTIA_RATIO = 1e-4
MIN_PLAUSIBLE_RADIUS = 0.03  # m: real parts are never smaller, and tiny marker shapes would trigger false alarms


@dataclass(frozen=True, eq=False)
class Shape:
    """One visual or collision shape of a link, placed in the link frame."""

    kind: str  # "box" | "cylinder" | "sphere" | "mesh"
    size: tuple[float, ...] = ()  # box: sx, sy, sz; cylinder: radius, length; sphere: radius
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    asset: MeshAsset | None = None
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True, eq=False)
class Inertial:
    mass: float
    com: np.ndarray  # link frame
    tensor: np.ndarray  # about com, link-frame axes


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rotation: fixed-axis roll, pitch, yaw, i.e. Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def shape_radius(shape: Shape) -> float:
    """Radius of a sphere around the shape's own center that contains it."""
    if shape.kind == "box":
        return 0.5 * float(np.linalg.norm(shape.size))
    if shape.kind == "cylinder":
        return math.hypot(shape.size[0], shape.size[1] / 2)
    if shape.kind == "sphere":
        return shape.size[0]
    assert shape.asset is not None
    return 0.5 * float(np.linalg.norm(shape.asset.mesh.extents * np.abs(shape.scale)))


def inertial_from_shapes(shapes: list[Shape], density: float = DEFAULT_DENSITY) -> Inertial | None:
    """Mass, center of mass and inertia of a link made of `shapes` at a uniform `density`."""
    parts = []
    for shape in shapes:
        volume, com, tensor = _unit_properties(shape)
        if volume <= 0:
            continue
        com_link = shape.rotation @ com + shape.position
        parts.append((volume * density, com_link, shape.rotation @ (tensor * density) @ shape.rotation.T))
    if not parts:
        return None
    mass = sum(m for m, _, _ in parts)
    com = sum(m * c for m, c, _ in parts) / mass
    tensor = np.zeros((3, 3))
    for m, c, i in parts:
        d = c - com
        tensor += i + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return Inertial(mass, com, tensor)


def shape_center(shape: Shape) -> np.ndarray:
    """Center of the shape's bounding box in the link frame."""
    if shape.kind != "mesh" or shape.asset is None:
        return shape.position
    local = np.asarray(shape.asset.bounds).mean(axis=0) * np.array(shape.scale)
    return shape.rotation @ local + shape.position


def shapes_radius(shapes: list[Shape], about: np.ndarray) -> float:
    """Radius of a sphere around `about` (link frame) that contains every shape."""
    if not shapes:
        return 0.0
    return max(float(np.linalg.norm(shape_center(s) - about)) + shape_radius(s) for s in shapes)


def inertia_problem(mass: float, tensor: np.ndarray) -> str:
    """What is wrong with this mass and inertia for the simulator, or '' when fine."""
    if not math.isfinite(mass) or mass < MIN_VALID_MASS:
        return "the mass is zero, negative or missing"
    if not np.isfinite(tensor).all():
        return "the inertia contains invalid numbers"
    eig = np.linalg.eigvalsh((tensor + tensor.T) / 2)
    if eig[0] <= NEAR_SINGULAR * max(eig[-1], 1e-12) or eig[0] <= 0:  # a hair above zero fails once written out
        return "the inertia is zero or not positive"
    if eig[2] > eig[0] + eig[1] + 1e-6 * eig[2]:
        return "the inertia is physically impossible (one axis is heavier than the other two together)"
    return ""


def repair_triangle(tensor: np.ndarray) -> np.ndarray:
    """Nudge principal moments so the largest is at most the sum of the other two."""
    vals, vecs = np.linalg.eigh((tensor + tensor.T) / 2)
    vals[2] = min(vals[2], vals[0] + vals[1])
    return vecs @ np.diag(vals) @ vecs.T


def inertia_implausible(mass: float, tensor: np.ndarray, radius: float) -> str:
    """Check the inertia against the size of the link's shapes; '' when it is believable."""
    if radius <= 0:
        return ""
    radius = max(radius, MIN_PLAUSIBLE_RADIUS)
    biggest = float(np.linalg.eigvalsh((tensor + tensor.T) / 2)[-1])
    ratio = biggest / (mass * radius * radius)
    if ratio > MAX_INERTIA_RATIO:
        return f"the inertia is far too large for a {mass:.3g} kg link about {radius * 100:.0f} cm across"
    if ratio < MIN_INERTIA_RATIO:
        return f"the inertia is far too small for a {mass:.3g} kg link about {radius * 100:.0f} cm across"
    return ""


def scaled_to_mass(shapes: list[Shape], mass: float) -> Inertial | None:
    """Shape-derived center and inertia, scaled so the total mass equals `mass`."""
    unit = inertial_from_shapes(shapes, density=1.0)
    if unit is None:
        return None
    return Inertial(mass, unit.com, unit.tensor * (mass / unit.mass))


def default_inertial() -> Inertial:
    return Inertial(TINY_MASS, np.zeros(3), np.eye(3) * TINY_INERTIA)


# ---- shape primitives ------------------------------------------------------------------------


def _unit_properties(shape: Shape) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, center of mass and inertia at density 1, in the shape's own frame."""
    sx, sy, sz = shape.scale
    if shape.kind == "box":
        x, y, z = (abs(a * s) for a, s in zip(shape.size, shape.scale, strict=True))
        volume = x * y * z
        return volume, np.zeros(3), np.diag([y * y + z * z, x * x + z * z, x * x + y * y]) * volume / 12.0
    if shape.kind == "cylinder":
        r, length = shape.size
        r, length = r * abs(sx), length * abs(sz)
        volume = math.pi * r * r * length
        side = volume * (3 * r * r + length * length) / 12.0
        return volume, np.zeros(3), np.diag([side, side, volume * r * r / 2.0])
    if shape.kind == "sphere":
        r = shape.size[0] * abs(sx)
        volume = 4.0 / 3.0 * math.pi * r**3
        return volume, np.zeros(3), np.eye(3) * (0.4 * volume * r * r)
    assert shape.asset is not None
    props = solid_properties(shape.asset.mesh, np.array(shape.scale))
    return props.volume, props.center, props.inertia
