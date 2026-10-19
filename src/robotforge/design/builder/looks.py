"""How robots built from parts look: visual-only meshes drawn over their collision shapes.

assemble.py makes the physics (shapes with mass from their material, actuators with their real
mass); this module draws a machine over it: shells with a parting line and screws, tapered
links, a servo with its horn at every powered hinge, gear motors behind wheels, rubber feet,
covers on unpowered hinges, flanges where parts are bolted together, an adapter plate under a
mounted robot and sensor heads you can recognise. Physics never sees any of it: every mesh goes
through `Visuals` (no contacts, no mass), and a primitive moves to the hidden group only when a
mesh covers it.

A part keeps its color; trims, housings and fasteners use neutral tones. The material says how
the surface reflects: printed PLA is the glossy shell plastic of the templates, aluminium is
anodised metal, carbon is clear-coated fibre and rubber is matte.

Positions here are in body frames at the home angle zero. A mirrored part's numbers come
reflected (`Placed.local`); meshes are always placed with proper rotations.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from robotforge.design import components
from robotforge.design.builder.expand import Placed, anchor, link_end, shape_box
from robotforge.design.builder.schema import BaseSpec, BuildSpec, PartSpec, actuator_for, axis_vector
from robotforge.design.templates.base import DARK
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

# Surface preset per material (see visuals.PRESETS). PLA parts look like the templates' shells.
PRESET_MATERIAL = {"shell": "pla", "aluminum": "aluminum", "carbon": "carbon", "rubber": "rubber"}  # the other way
MATERIAL_LOOK = {"pla": "shell", "aluminum": "aluminum", "carbon": "carbon", "rubber": "rubber"}
# A part with no color of its own shows its material: bare aluminium, black carbon fibre.
BARE_COLOR = {"aluminum": [0.76, 0.78, 0.8, 1.0], "carbon": [0.15, 0.155, 0.165, 1.0]}
MOTOR_YELLOW = [0.95, 0.8, 0.2, 1.0]
YELLOW_GEARBOX = ("tt_motor",)  # catalog motors whose real gearbox is yellow plastic

SEAM_GAP = 0.0005  # m, half the width of a shell's parting line
SEAM_EDGE = 0.0005  # m, edge radius on either side of a parting line: crisp, but not a knife edge
SEAM_INSET = 0.0007  # m the dark core under a parting line sits inside the shell, so the groove reads dark
PAN_INSET = 0.0012  # m a base's dark lower pan steps in from its colored upper shell
BASE_SEAM = 0.36  # share of a base's height taken by its dark pan (the templates' trunk uses about a third)
PAN_MAX = 0.015  # m: a tall base keeps its color; only its bottom 15 mm is the dark pan
# m of flat rim round the bottom of a base's open seam core: with the core's rounded edge it reaches over the
# pan's step, so looking up into the parting line from below never finds a way through the hollow shell
SEAM_RIM = PAN_INSET - SEAM_INSET
LINK_ROUNDNESS = 0.85  # corner radius of a hinged link's section, as a share of its smaller half extent
LINK_SHRINK = 0.25  # a hinged link's drawn shell lies at most this share of its radius inside its capsule (LinkShape)
ACROSS = 0.3  # a hinge whose share across a link is no more than this runs along it: the link spins, it does not swing
NECK = 0.42  # a ball is held by a neck this share of its radius (a head on its neck, a bob on its rod)
LID_SEAM = 0.74  # a box part's lid line, as a share of its height from the bottom
KEEPOUT_MARGIN = 0.003  # m kept clear around whatever is attached to a face before a screw goes there
SENSOR_KEEPOUT = 0.016  # m around a sensor's mounting point
BRACKET_REACH = 0.08  # m: a lead screw further than this from its parent stands on its own
SHAFT_OFFSET = 0.24  # share of a servo's length from its middle to its output shaft (sts3215: 11 of 45 mm)
TREAD_STEP = 0.005  # m of tread between two of a wheel's segments; small wheels get fewer segments
CURVE_STEP = 0.004  # m between segments round a shell's curved edges
PLATE_MIN_FOOTPRINT = 0.03  # m: a mounted robot narrower than this (a gripper) brings its own flange
CAMERA_HALF = 0.011  # m, half the width of a board camera's housing
CAMERA_DEPTH = 0.009  # m the housing reaches back from its front face
CAMERA_PROUD = 0.002  # m the housing stands in front of the camera point, which stays inside it (see camera_module)
CAMERA_BARREL = 0.003  # m the lens barrel stands in front of the housing
CAMERA_SEAT = 0.0005  # m a camera raised onto a face sinks into it, so no gap shows under the housing
LIDAR_RADIUS, LIDAR_HEIGHT = 0.022, 0.024  # m, as the lidar puck assemble.py gives its mass to
IMU_HALF = 0.009  # m, half the width of an IMU breakout board
RANGE_HALF = 0.012  # m, half the width of an ultrasonic range finder board
RANGE_ASPECT = 0.42  # its height as a share of its width (P.range_finder)
FACE_NORMALS = {"top": "up", "bottom": "down", "front": "forward", "back": "back", "left": "left", "right": "right"}


def look_of(material: str) -> str:
    return MATERIAL_LOOK[material]


def part_color(spec: BaseSpec | PartSpec, material: str, default: list[float]) -> list[float]:
    """The color a part is drawn in: its own, else its bare material (aluminium, carbon), else the default."""
    if spec.color is not None:
        return components.color(spec.color)
    if isinstance(spec, PartSpec) and spec.shape not in ("wheel", "caster") and material in BARE_COLOR:
        return list(BARE_COLOR[material])
    return default


def _unit(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, float) / (np.linalg.norm(v) or 1.0)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cross product of two 3-vectors (np.cross's per-call overhead dominated compiling a hexapod)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return v - (v @ axis) * axis


def _frame(z: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
    """A proper rotation whose z column is `z` and whose x column leans toward `hint`."""
    z = _unit(z)
    if hint is not None:
        x = _perp(np.asarray(hint, float), z)
        if np.linalg.norm(x) > 1e-6:
            x = _unit(x)
            return np.column_stack([x, _cross(z, x), z])
    return M.z_to(z)


def _segments(r: float, step: float = CURVE_STEP, low: int = 20, high: int = 64) -> int:
    """Segments round a circle of radius r: about `step` apart, a multiple of 4."""
    return int(min(high, max(low, 4 * round(2 * math.pi * r / step / 4))))


def _add(vis: Visuals, body: mujoco.MjsBody, parts: P.Parts, colors: dict[str, Sequence[float]], pos, rot) -> None:
    for preset, mesh in parts.items():
        vis.add(body, preset, colors[preset], mesh, pos, rot)


NEUTRAL = {"dark": DARK, "metal": P.METAL, "glass": P.GLASS, "rubber": P.RUBBER}


# ---- where things are ---------------------------------------------------------------------------------


@dataclass
class Layout:
    """What the dressing needs to know about the whole robot: where every body sits, the middle of every
    shape, and the spots on each body where something is attached (no screw or vent goes there)."""

    frames: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # body -> (rot, pos) in base
    centers: dict[str, np.ndarray | None] = field(default_factory=dict)  # body -> shape middle, own frame
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # body -> shape bounds, own frame
    solids: dict[str, tuple] = field(default_factory=dict)  # body -> its shape, own frame (see `inside`)
    links: dict[str, LinkShape] = field(default_factory=dict)  # body -> how its capsule is drawn, when not round
    keepouts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    middle: np.ndarray = field(default_factory=lambda: np.zeros(3))  # the robot's middle (the base shape's)
    poses: dict[str, tuple[np.ndarray, np.ndarray, int]] = field(default_factory=dict)  # part -> its servo case

    @classmethod
    def of(cls, build: BuildSpec, placed: list[Placed], footprints: dict[str, float]) -> Layout:
        out = cls()
        out.frames["base"] = (np.eye(3), np.zeros(3))
        out.centers["base"] = _center(build.base, np.eye(3))
        out._bounds("base", build.base, np.eye(3), None)
        out.middle = out.centers["base"] if out.centers["base"] is not None else np.zeros(3)
        out._sensor_keepouts("base", build.base, np.eye(3))
        for part in placed:
            rot_p, pos_p = out.frames[part.parent]
            out.frames[part.name] = (rot_p @ part.rot, rot_p @ part.pos + pos_p)
            out.centers[part.name] = _center(part.spec, part.local)
            out._bounds(part.name, part.spec, part.local, hinge_axis(part))
            out._sensor_keepouts(part.name, part.spec, part.local)
        for part in placed:
            out._child_keepout(build, part, footprints.get(part.name))
        return out

    def _bounds(self, body: str, spec: BaseSpec | PartSpec, local: np.ndarray, hinge: np.ndarray | None) -> None:
        if spec.shape == "none":
            return
        low, high = shape_box(spec)
        mid, half = local @ ((low + high) / 2), np.abs(local) @ ((high - low) / 2)
        self.boxes[body] = (mid - half, mid + half)
        if spec.shape == "box":
            self.solids[body] = ("box", mid, half)
        elif spec.shape in ("sphere", "caster"):
            self.solids[body] = ("capsule", mid, mid, spec.size[0] / 2)
        elif spec.shape == "wheel":
            axis = local @ np.array([0.0, spec.size[1] / 2, 0.0])
            self.solids[body] = ("cylinder", mid - axis, mid + axis, spec.size[0] / 2)
        elif spec.shape == "rotor":
            return  # a propeller hides nothing (builder/rotor.py draws it)
        else:
            start, end = segment(spec)
            self.solids[body] = (spec.shape, local @ start, local @ end, spec.size[0] / 2)
            shape = LinkShape.of(local @ start, local @ end, spec.size[0] / 2, hinge)
            if spec.shape == "capsule" and shape is not None:
                self.links[body] = shape

    def shell_offset(self, body: str, point: np.ndarray, normal: np.ndarray) -> float:
        """How far beyond `point`, a point on the body's collision shape, its drawn shell's surface lies along
        `normal` (negative: inside). Only a hinged link is drawn other than its shape: tapered and flattened."""
        link = self.links.get(body)
        return 0.0 if link is None else link.offset(np.asarray(point, float), _unit(normal))

    def inside(self, body: str, points: np.ndarray, margin: float = 0.0005) -> bool:
        """True when every point lies inside the body's own shape, `margin` deep: what is there cannot be seen."""
        solid = self.solids.get(body)
        if solid is None:
            return False
        pts = np.atleast_2d(points)
        if solid[0] == "box":
            _, mid, half = solid
            return bool(np.all(np.abs(pts - mid) <= half - margin))
        kind, a, b, r = solid
        d = b - a
        length = float(np.linalg.norm(d))
        t = (pts - a) @ (d / length) if length > 1e-9 else np.zeros(len(pts))
        if kind == "cylinder":
            radial = np.linalg.norm(pts - a - np.outer(t, d / length), axis=1)
            return bool(np.all((t >= margin) & (t <= length - margin) & (radial <= r - margin)))
        nearest = a + np.outer(np.clip(t, 0.0, length), d / length if length > 1e-9 else np.zeros(3))
        return bool(np.all(np.linalg.norm(pts - nearest, axis=1) <= r - margin))

    def hidden(self, body: str, center: np.ndarray, rot: np.ndarray, half: Sequence[float]) -> bool:
        """True when a box (its middle, rotation and half extents) lies inside the body's drawn shell, where
        nobody sees it: not from outside, and not a camera inside (it would only hide the view)."""
        solid = self.solids.get(body)
        if solid is None:
            return False
        margin = 0.0005 + (LINK_SHRINK * solid[3] if body in self.links else 0.0)
        return self.inside(body, _box_corners(np.asarray(center, float), rot, half), margin)

    def _keep(self, body: str, low: np.ndarray, high: np.ndarray) -> None:
        self.keepouts.setdefault(body, []).append((low - KEEPOUT_MARGIN, high + KEEPOUT_MARGIN))

    def _sensor_keepouts(self, body: str, spec: BaseSpec | PartSpec, local: np.ndarray) -> None:
        for sensor in spec.sensors:
            try:
                point = local @ anchor(spec, sensor.at, body)
            except ValueError:
                continue  # assemble reports the bad anchor
            self._keep(body, point - SENSOR_KEEPOUT, point + SENSOR_KEEPOUT)

    def _child_keepout(self, build: BuildSpec, part: Placed, footprint: float | None) -> None:
        """The child's shape (and the servo case that drives it) as boxes in its parent's frame."""
        if part.spec.shape == "none":
            reach = footprint or 0.012
            self._keep(part.parent, part.pos - reach, part.pos + reach)
        else:
            low, high = shape_box(part.spec)
            corners = np.array(
                [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
            )
            pts = part.pos + (part.rot @ (part.local @ corners.T)).T
            self._keep(part.parent, pts.min(axis=0), pts.max(axis=0))
        act = actuator_for(build, part.spec)
        if act is None or part.spec.shape == "wheel" or not build.servo_caps:
            return
        actuator = components.actuator(act.model)
        if is_servo_case(actuator) and part.spec.joint.type == "hinge":
            center, rot, _ = self.servo_pose(part, actuator, part.axis(axis_vector(part.spec.joint.axis)))
            sx, sy, sz = actuator.size
            half = np.array([sx * 0.66, sy / 2, sz * 0.62])  # the case with its ears, boss and horn
            corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
            pts = center + corners @ rot.T
            self._keep(part.parent, pts.min(axis=0), pts.max(axis=0))
        else:
            reach = 0.5 * float(np.linalg.norm(actuator.size))
            self._keep(part.parent, part.pos - reach, part.pos + reach)

    def free(self, body: str, point: np.ndarray, radius: float | Sequence[float]) -> bool:
        """True when a decoration reaching `radius` (or half extents x, y, z) from `point` touches nothing attached."""
        for low, high in self.keepouts.get(body, ()):
            if np.all(point + radius > low) and np.all(point - radius < high):
                return False
        return True

    def surface(self, body: str, point: np.ndarray, way: np.ndarray) -> float | None:
        """How far from `point` along `way` the body's shape begins (None: the ray misses it)."""
        if body not in self.boxes:
            return None
        low, high = self.boxes[body]
        enter, leave = -math.inf, math.inf
        for k in range(3):
            if abs(way[k]) < 1e-9:
                if not low[k] <= point[k] <= high[k]:
                    return None
                continue
            t0, t1 = (low[k] - point[k]) / way[k], (high[k] - point[k]) / way[k]
            enter, leave = max(enter, min(t0, t1)), min(leave, max(t0, t1))
        if enter > leave or leave < 0:
            return None
        return max(enter, 0.0)

    def servo_pose(
        self, part: Placed, actuator: components.Actuator, axis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Where a servo case sits in its parent's frame: its middle, its rotation (x along the case toward
        the parent, z out through the horn) and which way along the hinge axis the horn faces (+1 or -1)."""
        if part.name in self.poses:
            return self.poses[part.name]
        axis_parent = _unit(part.rot @ axis)
        side = self.outward(part, axis_parent)
        u = self.case_direction(part, axis_parent, direction_of(part.spec, part.local))
        z = side * axis_parent
        rot = np.column_stack([u, _cross(z, u), z])
        self.poses[part.name] = (part.pos + u * SHAFT_OFFSET * actuator.size[0], rot, side)
        return self.poses[part.name]

    def outward(self, part: Placed, axis_parent: np.ndarray) -> int:
        """+1 when the joint axis points away from the robot's middle (servo horns face out), else -1."""
        rot_p, pos_p = self.frames[part.parent]
        where = rot_p @ part.pos + pos_p - self.middle
        side = float((rot_p @ axis_parent) @ where)
        return -1 if side < -1e-3 else 1

    def case_direction(self, part: Placed, axis_parent: np.ndarray, direction: np.ndarray | None) -> np.ndarray:
        """Which way a servo case runs from its shaft (perpendicular to the axis, in the parent frame): into the
        parent's shape, else toward the robot's middle (a shoulder servo lies along the body), else back along
        the child link, else level."""
        candidates = []
        center = self.centers.get(part.parent)
        if center is not None:
            candidates.append(center - part.pos)
        rot_p, pos_p = self.frames[part.parent]
        candidates.append(rot_p.T @ (self.middle - pos_p) - part.pos)
        if direction is not None:
            candidates.append(-(part.rot @ direction))
        candidates += [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        for v in candidates:
            flat = _perp(v, axis_parent)
            n = float(np.linalg.norm(flat))
            if n > 1e-6 and n > 0.3 * float(np.linalg.norm(v)):
                return flat / n
        return _unit(_perp(np.array([1.0, 0.0, 0.0]), axis_parent))


@dataclass(frozen=True)
class LinkShape:
    """A hinged capsule drawn as a moulded limb, in its body frame: from `a` to `b` (the capsule's end points),
    wider at its joint and flattened along the hinge it swings on. Its section is a rounded rectangle of half
    extents `first` at `a` going linearly to `last` at `b`, in the x and y columns of `rot` (z runs a -> b);
    the ends are domes reaching 0.9 times the end section's smaller half extent beyond a and b."""

    a: np.ndarray
    b: np.ndarray
    rot: np.ndarray
    first: tuple[float, float]
    last: tuple[float, float]

    @classmethod
    def of(cls, a: np.ndarray, b: np.ndarray, r: float, hinge: np.ndarray | None) -> LinkShape | None:
        """How a capsule of radius r from a to b is drawn, or None when it is drawn as it is (round, untapered):
        it has no hinge across it (a base, a fixed part, a wheel-like spinner)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        length = float(np.linalg.norm(b - a))
        if hinge is None or length < 1e-6:
            return None
        d = (b - a) / length
        across = _perp(_unit(hinge), d)
        if np.linalg.norm(across) <= ACROSS:  # the hinge runs along the link: it spins, it does not swing
            return None
        y = _unit(across)
        return cls(a, b, np.column_stack([_cross(y, d), y, d]), (r * 1.12, r * 0.9), (r * 0.92, r * 0.86))

    @property
    def ends(self) -> tuple[float, float]:
        """How far the domes reach beyond a and b."""
        return 0.9 * min(self.first), 0.9 * min(self.last)

    def offset(self, point: np.ndarray, normal: np.ndarray) -> float:
        d = self.rot[:, 2]
        length = float(np.linalg.norm(self.b - self.a))
        along = float(normal @ d)
        if abs(along) > 0.7:  # on an end: the dome's tip
            tip = self.b + d * self.ends[1] if along > 0 else self.a - d * self.ends[0]
            return float((tip - point) @ normal)
        s = float(np.clip((point - self.a) @ d, 0.0, length))
        u = _unit(_perp(normal, d))
        t = s / length
        hx = self.first[0] + (self.last[0] - self.first[0]) * t
        hy = self.first[1] + (self.last[1] - self.first[1]) * t
        ux, uy = abs(float(u @ self.rot[:, 0])), abs(float(u @ self.rot[:, 1]))
        reach = _reach(hx, hy, LINK_ROUNDNESS * min(hx, hy), ux, uy)
        return reach - float((point - self.a - s * d) @ u)


def _reach(hx: float, hy: float, rr: float, ux: float, uy: float) -> float:
    """How far from its middle a rounded rectangle (half extents hx, hy, corner radius rr) ends along (ux, uy)."""
    cx, cy = hx - rr, hy - rr
    if ux > 1e-9 and hx / ux * uy <= cy:
        return hx / ux
    if uy > 1e-9 and hy / uy * ux <= cx:
        return hy / uy
    uc = ux * cx + uy * cy
    return uc + math.sqrt(max(uc * uc - cx * cx - cy * cy + rr * rr, 0.0))


def spins(spec: BaseSpec | PartSpec, local: np.ndarray, hinge: np.ndarray) -> bool:
    """True when a cylinder or capsule turns about its own axis (a wheel-like disc, a spinning rod)."""
    if spec.shape not in ("cylinder", "capsule"):
        return False
    start, end = segment(spec)
    d = local @ (end - start)
    if np.linalg.norm(d) < 1e-9:
        return False
    return float(np.linalg.norm(_perp(_unit(hinge), _unit(d)))) <= ACROSS


def hinge_axis(part: Placed | None) -> np.ndarray | None:
    """The hinge a part swings on, in its body frame (its link is drawn flattened along it)."""
    if part is None or part.spec.joint.type != "hinge" or part.spec.joint.continuous:
        return None
    return part.axis(axis_vector(part.spec.joint.axis))


def face_normal(spec: BaseSpec | PartSpec, at: str | Sequence[float], local: np.ndarray) -> np.ndarray | None:
    """The outward normal, in the body frame, of the face a named anchor such as "front" sits on (None for
    the center, the ends, a mount or numbers)."""
    if not isinstance(at, str) or at in spec.mounts or at not in FACE_NORMALS:
        return None
    return local @ np.asarray(axis_vector(FACE_NORMALS[at]), float)


def _center(spec: BaseSpec | PartSpec, local: np.ndarray) -> np.ndarray | None:
    if spec.shape == "none":
        return None
    low, high = shape_box(spec)
    return local @ ((low + high) / 2)


def segment(spec: BaseSpec | PartSpec) -> tuple[np.ndarray, np.ndarray]:
    """A capsule's or cylinder's end points, in the part's own (author's) coordinates."""
    offset = np.asarray(spec.offset, float)
    length = spec.size[1]
    direction = getattr(spec, "direction", None)
    if direction is not None:
        return offset, offset + np.asarray(axis_vector(direction), float) * length
    u = np.asarray(axis_vector(getattr(spec, "axis", "z")), float)
    return offset - u * length / 2, offset + u * length / 2


def _box_corners(center: np.ndarray, rot: np.ndarray, half: Sequence[float]) -> np.ndarray:
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return center + (signs * np.asarray(half, float)) @ rot.T


def is_servo_case(actuator: components.Actuator) -> bool:
    """A hobby servo (a long case with an off-centre horn); round joint modules and motors are drawn otherwise."""
    sx, sy, _ = actuator.size
    return actuator.kind == "servo" and abs(sx - sy) >= 0.15 * max(sx, sy)


def direction_of(spec: PartSpec, local: np.ndarray) -> np.ndarray | None:
    """The way a link grows from its joint, in its body frame (None when it is centred on the joint)."""
    if not isinstance(spec, PartSpec) or spec.direction is None:
        return None
    return _unit(local @ np.asarray(axis_vector(spec.direction), float))


def footprint(spec: mujoco.MjSpec) -> float:
    """How far a mounted robot's root body reaches from its origin on the mounting plane (0: unknown)."""
    root = spec.worldbody.first_body()
    if root is None:
        return 0.0
    G = mujoco.mjtGeom
    reach = 0.0
    for geom in root.geoms:
        pos, size = np.asarray(geom.pos, float), np.asarray(geom.size, float)
        if geom.type == G.mjGEOM_CYLINDER:
            r, bottom = size[0], pos[2] - size[1]
        elif geom.type == G.mjGEOM_BOX:
            r, bottom = math.hypot(size[0], size[1]), pos[2] - size[2]
        else:
            continue
        if bottom < 0.01:  # it stands on the plate
            reach = max(reach, r + math.hypot(pos[0], pos[1]))
    return reach


# ---- shells -------------------------------------------------------------------------------------------


def _corner_points(corner: float) -> int:
    """Points per rounded corner of a shell: enough that a big radius does not look faceted."""
    return 2 if corner < 0.004 else 3 if corner < 0.012 else 5


def box(
    vis: Visuals,
    body: mujoco.MjsBody,
    name: str,
    half: Sequence[float],
    center: np.ndarray,
    rgba: Sequence[float],
    preset: str,
    layout: Layout,
    base: bool,
) -> None:
    """A box as a moulded shell: rounded edges, a parting line (a base gets a dark lower pan, like the
    templates' trunks), screws in the corners of the top and, on a base, a vent and a status light."""
    hx, hy, hz = (float(h) for h in half)
    small = min(hx, hy, hz)
    rb = float(np.clip(small * 0.3, 0.0012, 0.007))
    rc = float(np.clip(min(hx, hy) * 0.35, rb, 0.02))
    per = _corner_points(rc)
    c = np.asarray(center, float)
    if small < 0.006 or hz < 0.008:  # plates, bars and forks: one rounded block, too thin for a seam
        vis.add(body, preset, rgba, M.rounded_box(hx, hy, hz, rb, segments=2, corner=rc), c)
        return
    zs = -hz + (min(2 * hz * BASE_SEAM, PAN_MAX) if base else 2 * hz * LID_SEAM)
    # A base is big on screen: smoother edges and fenced flat faces. A part makes do with less.
    seg, fence = (3, True) if base else (2, False)
    # The two halves are open toward each other and the dark core behind the parting line is a bare band:
    # from outside the core closes the gap, and a camera inside the part sees out as through one shell.
    # Under a base's core a narrow rim closes the step above the pan.
    finish = {"segments": seg, "per_corner": per, "fence": fence}
    upper = M.slab(hx, hy, zs + SEAM_GAP, hz, SEAM_EDGE, rb, rc, **finish, bottom_cap=False)
    vis.add(body, preset, rgba, upper, c)
    if base:
        pan_half = (hx - PAN_INSET, hy - PAN_INSET)
        pan = M.slab(*pan_half, -hz, zs - SEAM_GAP, rb, SEAM_EDGE, rc - PAN_INSET, per_corner=per, top_cap=False)
        vis.add(body, "dark", DARK, pan, c)
    else:
        lower = M.slab(hx, hy, -hz, zs - SEAM_GAP, rb, SEAM_EDGE, rc, **finish, top_cap=False)
        vis.add(body, preset, rgba, lower, c)
    depth = SEAM_GAP + 0.0012
    inner = (hx - SEAM_INSET, hy - SEAM_INSET, zs - depth, zs + depth, 0.0003, 0.0003, rc - SEAM_INSET)
    rims = (SEAM_RIM if base else 0.0, 0.0)
    core = M.slab(*inner, segments=1, per_corner=per, fence=False, bottom_cap=False, top_cap=False, rims=rims)
    vis.add(body, "dark", DARK, core, c)
    if base or (hx >= 0.015 and hy >= 0.015):
        _corner_screws(vis, body, name, hx, hy, hz, c, layout)
    if base:
        _base_details(vis, body, name, hx, hy, hz, zs, c, layout)


def _corner_screws(vis: Visuals, body, name: str, hx: float, hy: float, hz: float, c: np.ndarray, layout: Layout):
    inset = float(np.clip(min(hx, hy) * 0.22, 0.0045, 0.018))
    r = float(np.clip(min(hx, hy) * 0.07, 0.0012, 0.0028))
    head = P.bolt(r, r * 0.75, underside=False)
    for sx in (1, -1):
        for sy in (1, -1):
            # Into the corner, or along the long side when something is attached at the corner.
            for dx, dy in ((inset, inset), (min(inset * 2.5, hx * 0.5), inset)):
                at = c + np.array([sx * (hx - dx), sy * (hy - dy), hz - 0.0002])
                if layout.free(name, at, r + 0.002):
                    vis.add(body, "metal", P.METAL, head, at)
                    break


def _base_details(vis: Visuals, body, name: str, hx: float, hy: float, hz: float, zs: float, c, layout: Layout):
    """A vent grille on the back face and a status light near the back of the top, where nothing is attached."""
    upper_mid = (zs + hz) / 2
    grille_w, grille_h = min(hy * 0.45, 0.05), (hz - zs) * 0.28
    if grille_h >= 0.005 and hy >= 0.03:
        at = c + np.array([-hx - 0.0002, 0.0, upper_mid])
        if layout.free(name, at, (0.002, grille_w, grille_h)):
            slats = P.slats(grille_w, grille_h, 4 if grille_h > 0.008 else 3, 0.0016, open_back=True)
            vis.add(body, "dark", DARK, slats, at, M.rot_z(math.pi / 2))
    if hx < 0.03:
        return
    for fx, fy in ((-0.78, 0.55), (-0.78, -0.55), (-0.5, 0.0), (0.0, 0.55)):
        light = c + np.array([hx * fx, hy * fy, hz])
        if layout.free(name, light, 0.006):
            vis.add(body, "led", P.LED_GREEN, P.led(0.0028, underside=False)["led"], light)
            return


@M.cached
def _disc(
    r: float,
    z0: float,
    z1: float,
    bottom: float,
    top: float,
    segments: int,
    lids: tuple[bool, bool] = (True, True),
    rims: tuple[float, float] = (0.0, 0.0),
) -> M.Mesh:
    """A solid cylinder along z with its own edge radius at each end (a slab, but round). An end without its
    lid is open, keeping a flat rim `rims` wide round it (see M.slab)."""
    h = (z1 - z0) / 2
    rb, rt = min(bottom, r, h), min(top, r, h)
    prof = [(r - rb + rb * math.sin(t), z0 + rb - rb * math.cos(t)) for t in np.linspace(0, math.pi / 2, 3)]
    prof += [(r - rt + rt * math.sin(t), z1 - rt + rt * math.cos(t)) for t in np.linspace(math.pi / 2, 0, 3)]
    creases = []
    if not lids[0] and rims[0] > 0:
        prof.insert(0, (max(r - rb - rims[0], 0.2 * r), z0))
        creases.append(1)
    if not lids[1] and rims[1] > 0:
        prof.append((max(r - rt - rims[1], 0.2 * r), z1))
        creases.append(len(prof) - 2)
    rings = [M.at_z(M.circle(rr, segments), z) for rr, z in prof]
    return M.loft(rings, sharp=creases, cap_start=lids[0], cap_end=lids[1])


def cylinder(
    vis: Visuals,
    body: mujoco.MjsBody,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    r: float,
    rgba: Sequence[float],
    preset: str,
    layout: Layout,
    base: bool,
) -> None:
    """A cylinder as a puck (a base gets a dark lower ring and a bolt circle), or as a tube with dark end
    collars when it is long and thin, like a mast."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    d = (b - a) / length
    rot = M.z_to(d)
    n = _segments(r)
    rim = min(0.006, r * 0.2, length * 0.2)
    if length >= 2.5 * r and not base:
        vis.add(body, preset, rgba, M.cylinder(r, length, segments=n, bevel=min(0.0015, r * 0.15), z0=0.0), a, rot)
        h = float(np.clip(r * 0.35, 0.002, 0.01))
        collar = M.cylinder(r * 1.07, h, segments=n, bevel=h * 0.25, z0=0.0, arc=2)
        vis.add(body, "dark", DARK, collar, a - d * 0.0003, rot)
        vis.add(body, "dark", DARK, collar, b - d * (h - 0.0003), rot)
        return
    if base and length >= 0.016:  # open toward the parting line, as box() explains
        zs = min(length * BASE_SEAM, PAN_MAX)
        upper = _disc(r, zs + SEAM_GAP, length, SEAM_EDGE, rim, n, (False, True))
        vis.add(body, preset, rgba, upper, a, rot)
        vis.add(body, "dark", DARK, _disc(r - PAN_INSET, 0.0, zs - SEAM_GAP, rim, SEAM_EDGE, n, (True, False)), a, rot)
        depth = SEAM_GAP + 0.0012
        core = _disc(r - SEAM_INSET, zs - depth, zs + depth, 0.0003, 0.0003, n, (False, False), (SEAM_RIM, 0.0))
        vis.add(body, "dark", DARK, core, a, rot)
    else:
        vis.add(body, preset, rgba, M.cylinder(r, length, segments=n, bevel=rim, z0=0.0), a, rot)
    if r >= 0.02:  # a bolt circle on the far face, skipping bolts under anything attached there
        count = 8 if r > 0.06 else 6
        bolt_r = float(np.clip(r * 0.05, 0.0014, 0.003))
        head = P.bolt(bolt_r, bolt_r * 0.75, underside=False)
        ring = r * 0.8 - rim * 0.5
        # Of the two ways to turn the circle, the one that leaves more bolts clear of attached parts.
        layouts = []
        for phase in (0.5, 0.0):
            spots = []
            for k in range(count):
                angle = 2 * math.pi * (k + phase) / count
                at = a + rot @ np.array([ring * math.cos(angle), ring * math.sin(angle), length - 0.0002])
                if layout.free(name, at, bolt_r + 0.002):
                    spots.append(at)
            layouts.append(spots)
        for at in max(layouts, key=len):
            vis.add(body, "metal", P.METAL, head, at, rot)
    if base and r >= 0.04:
        light = a + rot @ np.array([-r * 0.62, 0.0, length])
        if layout.free(name, light, 0.006):
            vis.add(body, "led", P.LED_GREEN, P.led(0.0028, underside=False)["led"], light, rot)


def sphere(
    vis: Visuals, body: mujoco.MjsBody, center: np.ndarray, r: float, up: np.ndarray, rgba: Sequence[float], preset: str
) -> None:
    """A sphere as two hemispheres with a parting line round its middle (a robot head, a ball body)."""
    n = _segments(r, low=24, high=48)
    rot = M.z_to(up)
    for half in (1, -1):
        vis.add(body, preset, rgba, _hemisphere(r, n, half), center, rot)
    vis.add(body, "dark", DARK, M.sphere(r - SEAM_INSET, rings=12, segments=n), center, rot)


@M.cached
def _hemisphere(r: float, segments: int, half: int) -> M.Mesh:
    """The upper (half = 1) or lower (-1) half of a sphere, its rim turned in by SEAM_EDGE so the parting line
    shows the dark core rather than a bright lip. It is open at the rim: a camera inside the ball sees out."""
    gap = min(SEAM_GAP / r, 0.2)
    edge = min(SEAM_EDGE / r, 0.2)
    prof = [(r * math.cos(gap) - SEAM_EDGE, r * math.sin(gap))]
    prof += [(r * math.cos(t), r * math.sin(t)) for t in np.linspace(gap + edge, math.pi / 2, 9)]
    rings = [M.at_z(M.circle(rr, segments), half * z) for rr, z in prof]
    return M.loft(rings, sharp=(1,), cap_start=False)


def link(
    vis: Visuals,
    body: mujoco.MjsBody,
    start: np.ndarray,
    end: np.ndarray,
    r: float,
    rgba: Sequence[float],
    preset: str,
    hinge: np.ndarray | None,
) -> None:
    """A capsule as a moulded limb: domed ends where the capsule's round ends are, wider at its joint and
    flattened along the hinge it swings on (see LinkShape). With no hinge across it (a base, a fixed part)
    it is drawn as the capsule it is. Either way a thick one gets as many points round as a cylinder."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b - a))
    n = _segments(r, low=16, high=64)
    if length < 1e-6:
        vis.add(body, preset, rgba, M.sphere(r, rings=max(10, n // 4), segments=n), a)
        return
    shape = LinkShape.of(a, b, r, hinge)
    if shape is None:
        vis.add(body, preset, rgba, M.capsule(r, length, rings=max(6, n // 8), segments=n), a, M.z_to(b - a))
        return
    per = n // 4 - 1  # points per rounded corner, so the section has n points like a cylinder of radius r
    e0, e1 = shape.ends
    if vis.dress is not None:  # a two-part shell with a parting line, in the robot's style
        d = vis.dress.with_material(PRESET_MATERIAL.get(preset, "pla"))
        d.limb(
            vis,
            body,
            length + e0 + e1,
            shape.first,
            shape.last,
            replace(d.shell, body=tuple(rgba)),
            pos=a - shape.rot[:, 2] * e0,
            rot=shape.rot,
            seam="strip",  # the shell stays the exact surface sensors and mounts are placed on
            roundness=LINK_ROUNDNESS,
            sides=n,
        )
        return
    mesh = M.link(
        length + e0 + e1,
        shape.first,
        shape.last,
        roundness=LINK_ROUNDNESS,
        rounding=(0.9, 0.9),
        stations=5,
        per_corner=per,
        domes=int(np.clip(per + 1, 4, 10)),
    )
    vis.add(body, preset, rgba, mesh, a - shape.rot[:, 2] * e0, shape.rot)


def caster(vis: Visuals, body: mujoco.MjsBody, center: np.ndarray, r: float, up: np.ndarray, ball, preset: str) -> None:
    parts = P.caster(r, r * 1.35)
    vis.add(body, preset, ball, parts["metal"], center, M.z_to(up))
    vis.add(body, "dark", DARK, parts["dark"], center, M.z_to(up))


def wheel(
    vis: Visuals, body: mujoco.MjsBody, center: np.ndarray, r: float, half_w: float, face: np.ndarray, rim
) -> None:
    """A tyre with tread on a spoked rim (a dish on small wheels); detail scales with the wheel."""
    if vis.dress is not None:
        d = vis.dress
        spokes = 5 if r >= 0.025 else 0
        d.draw(
            vis,
            body,
            "wheel",
            replace(d.shell, body=tuple(rim)),
            center,
            M.z_to(face),
            radius=r,
            width=2 * half_w,
            spokes=spokes,
        )
        return
    n = int(min(60, max(24, 4 * round(2 * math.pi * r / TREAD_STEP / 4))))
    colors = {"rubber": P.RUBBER, "shell": rim, "dark": DARK, "metal": P.METAL}
    _add(vis, body, P.tyre(r, half_w, segments=n, spokes=5 if r >= 0.025 else 0), colors, center, M.z_to(face))


def foot(
    vis: Visuals, body: mujoco.MjsBody, center: np.ndarray, r: float, toward_joint: np.ndarray, shin_r: float
) -> None:
    """A rubber ball foot with a metal collar on the side its leg comes from."""
    if vis.dress is not None:
        vis.dress.draw(vis, body, "foot", None, center, M.z_to(toward_joint), radius=r, shin=shin_r)
        return
    colors = {"rubber": P.RUBBER, "metal": P.METAL}
    segments = 24 if r >= 0.02 else 18  # a small foot is a few pixels across
    _add(vis, body, P.ball_foot(r, shin_r, segments), colors, center, M.z_to(toward_joint))


# ---- joints -------------------------------------------------------------------------------------------


def servo(
    vis: Visuals,
    parent: mujoco.MjsBody,
    child: mujoco.MjsBody,
    part: Placed,
    actuator: components.Actuator,
    axis: np.ndarray,
    layout: Layout,
) -> None:
    """The servo at a powered hinge: its case bolted to the parent, its horn turning with the part.

    A round joint module (square footprint, like bldc_joint) is a drum about the axis instead, and a
    gear motor on a hinge that is not a wheel sits behind its output flange.
    """
    sx, sy, sz = actuator.size
    axis = _unit(axis)
    axis_parent = part.rot @ axis
    side = layout.outward(part, axis_parent)
    face = side * axis  # the horn's side, in the child frame
    if actuator.kind == "motor":
        r, length = max(sy, sz) / 2, min(sx, 0.06)
        colors = {"shell": MOTOR_YELLOW, "metal": P.METAL, "dark": DARK}
        motor = P.gear_motor(r, length, actuator.name in YELLOW_GEARBOX)
        _add(vis, parent, motor, colors, part.pos - side * axis_parent * 0.002, M.z_to(-side * axis_parent))
        t = max(0.0015, r * 0.12)
        if not _buried(layout, part.name, np.zeros(3), face, r * 0.6, t):
            _add(vis, child, P.horn(r * 0.6, t), NEUTRAL, (0.0, 0.0, 0.0), M.z_to(face))
        return
    if not is_servo_case(actuator):  # a round joint module
        r, half = min(sx, sy) / 2 + 0.002, sz / 2
        _add(vis, parent, P.joint_module(r, half, r * 0.72), NEUTRAL, part.pos, M.z_to(side * axis_parent))
        t = max(0.0012, half * 0.16)
        if not _buried(layout, part.name, face * (half - 0.0006), face, r * 0.72, t):
            _add(vis, child, P.horn(r * 0.72, t, screws=4), NEUTRAL, face * (half - 0.0006), M.z_to(face))
        return
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    center, rot, _ = layout.servo_pose(part, actuator, axis)
    horn_r = 0.78 * min(hx, hy)
    outline = (hx * 1.32, hy, hz)  # the case with its ears
    # A hinge whose axis comes straight out of its parent's face (a turntable, a pan joint) has its servo
    # mounted inside the parent, horn flush with the face: nothing of it shows but the horn.
    sunk = center - rot[:, 2] * (hz * 1.12 + 0.001)
    flush = layout.inside(part.parent, _box_corners(sunk, rot, outline))
    if not flush and not layout.inside(part.parent, _box_corners(center, rot, outline)):
        if vis.dress is not None:
            vis.dress.draw(
                vis,
                parent,
                "servo_case",
                None,
                center,
                rot,
                length=sx,
                width=sy,
                thickness=sz,
                shaft=-SHAFT_OFFSET * sx,
                horn=horn_r,
            )
        else:
            _add(vis, parent, P.servo_case(hx, hy, hz, horn_r, -SHAFT_OFFSET * sx), NEUTRAL, center, rot)
    # The horn and the idler turn with the part; inside a part thicker than the servo they cannot be seen.
    t = max(0.0016, hz * 0.09)
    at = face * (0.0 if flush else hz * 1.12 - 0.0012)
    if not _buried(layout, part.name, at, face, horn_r, t):
        _add(vis, child, P.horn(horn_r, t), NEUTRAL, at, M.z_to(face))
    t = max(0.0012, hz * 0.06)
    if not flush and not _buried(layout, part.name, -face * (hz - 0.0004), -face, horn_r * 0.5, t):
        cap = M.cylinder(horn_r * 0.5, t, segments=12, bevel=0.0004, z0=0.0, arc=1)
        vis.add(child, "metal", P.METAL, cap, -face * (hz - 0.0004), M.z_to(-face))  # the idler on the far side


def _buried(layout: Layout, body: str, at: np.ndarray, z: np.ndarray, r: float, t: float) -> bool:
    """True when a disc of radius r and thickness t standing on `at` along z lies inside the body's shape."""
    rot = _frame(z)
    return layout.inside(body, _box_corners(np.asarray(at, float) + rot[:, 2] * t / 2, rot, (r, r, t / 2)))


def wheel_motor(
    vis: Visuals,
    parent: mujoco.MjsBody,
    parent_name: str,
    center: np.ndarray,
    inboard: np.ndarray,
    r: float,
    length: float,
    model: str,
    layout: Layout,
) -> None:
    """A wheel's gear motor on the chassis, from the wheel inward (where assemble.py puts its mass); not drawn
    when it is inside the chassis anyway."""
    if layout.inside(parent_name, _box_corners(center, _frame(inboard), (r, r, length / 2))):
        return
    colors = {"shell": MOTOR_YELLOW, "metal": P.METAL, "dark": DARK}
    start = np.asarray(center, float) - inboard * length / 2
    _add(vis, parent, P.gear_motor(r, length, model in YELLOW_GEARBOX), colors, start, M.z_to(inboard))


def linear(
    vis: Visuals,
    parent: mujoco.MjsBody,
    child: mujoco.MjsBody,
    part: Placed,
    actuator: components.Actuator,
    axis: np.ndarray,
    joint_range: tuple[float, float],
    layout: Layout,
) -> None:
    """A linear actuator drawn as a lead screw along the whole stroke: the motor below the start of the
    travel, a bearing block past its end and a nut that rides with the part. With nothing of the parent near
    the far end to hold a bearing (a lifter rising off a base's top) the screw ends free where the nut stops."""
    sx, sy, sz = actuator.size
    a = _unit(part.rot @ axis)
    lo, hi = joint_range
    rod_r = float(np.clip(0.2 * min(sy, sz), 0.002, 0.008))
    nut_h = max(0.004, rod_r * 1.4)  # m, half the nut's height
    center = layout.centers.get(part.parent)
    rot = _frame(a, None if center is None else center - part.pos)
    back = rot[:, 0]  # toward the parent's middle, across the screw
    bottom, top = lo - nut_h - 0.004, hi + nut_h + 0.004
    ends = [(bottom - sx / 2, sy / 2)]  # where a bracket holds the screw, and its half width
    gap = layout.surface(part.parent, part.pos + a * (top + rod_r * 1.3), back)
    if gap is not None and gap <= BRACKET_REACH:
        ends.insert(0, (top + rod_r * 1.3, rod_r * 2.2))
    else:
        top = hi + nut_h
    motor = M.rounded_box(sy / 2, sz / 2, sx / 2, min(sy, sz) * 0.12, segments=2)
    vis.add(parent, "dark", DARK, motor, part.pos + a * (bottom - sx / 2), rot)
    for end in (1, -1):
        cap = M.rounded_box(sy / 2 + 0.0004, sz / 2 + 0.0004, sx * 0.07, 0.0008, segments=1)
        vis.add(parent, "metal", P.METAL, cap, part.pos + a * (bottom - sx / 2 + end * sx * 0.42), rot)
    vis.add(parent, "metal", P.METAL, P.lead_screw(rod_r, top - bottom), part.pos + a * bottom, rot)
    # The screw's ends are held by brackets that reach back to the parent's surface, so nothing floats.
    for at, half_w in ends:
        start = part.pos + a * at
        gap = layout.surface(part.parent, start, back)
        reach = rod_r * 2.2 if gap is None or gap > BRACKET_REACH else max(gap + 0.002, rod_r * 2.2)
        thick = rod_r * 1.3 if half_w < sy / 2 else sx * 0.3
        bracket = M.rounded_box(reach / 2 + rod_r * 1.1, half_w, thick, min(thick, half_w) * 0.35, segments=2)
        vis.add(parent, "dark", DARK, bracket, start + back * (reach / 2 - rod_r * 1.1), rot)
    nut = M.rounded_box(rod_r * 2.1, rod_r * 2.1, nut_h, rod_r * 0.5, segments=2)
    vis.add(child, "dark", DARK, nut, (0.0, 0.0, 0.0), part.rot.T @ rot)


def passive(vis: Visuals, body: mujoco.MjsBody, part: Placed, axis: np.ndarray, r_link: float, layout: Layout) -> None:
    """An unpowered hinge: a dark sleeve about the axis with a bearing cap on each side, sized to the link (or
    neck) that starts there. Not drawn where nothing pokes through the parent: a disc or rod spinning about its
    own axis, or a hinge deep inside its part (a ball turning about its middle)."""
    if spins(part.spec, part.local, axis):
        return
    r = float(np.clip(1.3 * r_link, 0.006, 0.05))
    half = float(np.clip(1.1 * r_link, 0.005, 0.04))
    if layout.hidden(part.name, np.zeros(3), _frame(axis), (r, r, half)):
        return
    _add(vis, body, P.knuckle(r, half), NEUTRAL, (0.0, 0.0, 0.0), M.z_to(axis))


def fixed(vis: Visuals, body: mujoco.MjsBody, direction: np.ndarray, r_link: float) -> None:
    """A part bolted on where it starts: a round flange with bolts (a neck under a head, the foot of a mast)."""
    r = float(np.clip(1.45 * r_link, 0.008, 0.06))
    t = float(np.clip(0.14 * r, 0.0015, 0.005))
    _add(vis, body, P.flange(r, t, bolts=4 if r < 0.025 else 6), NEUTRAL, -direction * 0.0004, M.z_to(direction))


def mount_plate(vis: Visuals, body: mujoco.MjsBody, reach: float) -> None:
    """The aluminium plate a mounted robot stands on (its base sits a few millimetres into it)."""
    if reach < PLATE_MIN_FOOTPRINT:
        return
    r = reach * 1.12
    t = float(np.clip(r * 0.07, 0.003, 0.006))
    _add(vis, body, P.plate(r, t), NEUTRAL, (0.0, 0.0, -0.0008), None)


# ---- sensors ------------------------------------------------------------------------------------------


def camera_frame(look: np.ndarray, up: np.ndarray) -> np.ndarray:
    """A board camera's housing frame: z where it looks, y up (so its LED sits top right)."""
    z = _unit(look)
    y = _perp(np.asarray(up, float), z)
    y = _unit(y) if np.linalg.norm(y) > 1e-6 else M.z_to(z)[:, 1]
    return np.column_stack([_cross(y, z), y, z])


def camera_box(where: np.ndarray, look: np.ndarray) -> tuple[np.ndarray, Sequence[float]]:
    """The middle and half extents (in camera_frame) of the board camera drawn round `where`."""
    back, front = CAMERA_DEPTH - CAMERA_PROUD, CAMERA_PROUD + CAMERA_BARREL * 1.5  # the glass dome stands proud
    return np.asarray(where, float) + _unit(look) * (front - back) / 2, (CAMERA_HALF, CAMERA_HALF, (front + back) / 2)


def camera(vis: Visuals, body: mujoco.MjsBody, where: np.ndarray, look: np.ndarray, up: np.ndarray) -> None:
    """A board camera around the camera point, its lens looking where the camera looks."""
    front = np.asarray(where, float) + _unit(look) * CAMERA_PROUD
    parts = P.camera_module(CAMERA_HALF, CAMERA_DEPTH, CAMERA_BARREL)
    _add(vis, body, parts, {**NEUTRAL, "led": P.LED_RED}, front, camera_frame(look, up))


def lidar(vis: Visuals, body: mujoco.MjsBody, where: np.ndarray, up: np.ndarray) -> None:
    _add(vis, body, P.lidar_head(LIDAR_RADIUS, LIDAR_HEIGHT), NEUTRAL, where, M.z_to(up))


IMU_BOX = (IMU_HALF, IMU_HALF, 0.0013)  # m, half extents of the IMU board with its chip (2.6 mm tall, P.imu_board)


def imu(vis: Visuals, body: mujoco.MjsBody, where: np.ndarray, normal: np.ndarray, along: np.ndarray | None) -> None:
    """An IMU board lying flat on the face whose outward normal is `normal`, its long side `along` that way."""
    colors = {**NEUTRAL, "shell": P.PCB, "led": P.LED_RED}
    rot = _frame(normal, along)
    _add(vis, body, P.imu_board(IMU_HALF), colors, np.asarray(where, float) - _unit(normal) * 0.0003, rot)


def rangefinder_frame(look: np.ndarray, up: np.ndarray) -> np.ndarray:
    """A range finder board's frame: z where it looks, x (its long side, the two cans) level."""
    side = _cross(np.asarray(up, float), _unit(look))
    return _frame(look, side if np.linalg.norm(side) > 1e-6 else None)


RANGE_BOX = (RANGE_HALF * 0.9, RANGE_HALF * RANGE_ASPECT, 0.003)  # m, half extents of the board and its cans


def rangefinder(vis: Visuals, body: mujoco.MjsBody, where: np.ndarray, rot: np.ndarray) -> None:
    """An ultrasonic range finder board on `where`, facing the z column of `rot`."""
    colors = {**NEUTRAL, "shell": P.PCB}
    _add(vis, body, P.range_finder(RANGE_HALF), colors, np.asarray(where, float) - rot[:, 2] * 0.0006, rot)


def link_end_point(spec: PartSpec, local: np.ndarray) -> np.ndarray:
    return local @ link_end(spec)
