"""Turn a BuildSpec's part tree into a flat list of placed parts: mirror and repeat unrolled, anchors resolved.

Coordinates: every part's numbers (`at`, `rotate`, axis, direction, sensors) are written in its
parent's frame as the author pictured it. A mirrored copy is the author's picture reflected,
so each placed part carries `local`, the reflection to apply to its own numbers. A ring copy
only turns where the part is attached. Keeping the two apart is what makes a mirrored knee bend
the same way as the original, and a chain of snake segments line up.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from robotforge.design.builder.schema import (
    ANCHORS,
    MAX_PARTS,
    BaseSpec,
    BuildSpec,
    PartSpec,
    axis_vector,
)

MIRROR_Y = np.diag([1.0, -1.0, 1.0])  # reflect across the parent's x-z plane: left <-> right


@dataclass(frozen=True)
class Placed:
    """One part, ready to be built: where its frame sits in its parent's frame, and its own numbers' reflection."""

    name: str
    spec: PartSpec
    parent: str  # "base" or another placed part's name
    pos: np.ndarray  # attachment point in the parent body frame
    rot: np.ndarray  # 3x3: the part frame in the parent body frame
    local: np.ndarray  # 3x3 orthogonal: maps the part's own numbers into its body frame (I or a reflection)
    # -1 on every other copy of a ring or a row (and on what hangs on it), so rotors on them turn the other way
    flip: int = 1

    def vec(self, v) -> np.ndarray:
        """A direction or point the author wrote for this part, in its body frame."""
        return self.local @ np.asarray(v, float)

    def axis(self, v) -> np.ndarray:
        """A rotation axis (a pseudo-vector: a mirror flips it so the motion mirrors too)."""
        return np.linalg.det(self.local) * (self.local @ np.asarray(v, float))


def euler(rpy_deg: list[float]) -> np.ndarray:
    """Rotation from [roll, pitch, yaw] in degrees: roll about x, then pitch about y, then yaw about z."""
    r, p, y = (math.radians(a) for a in rpy_deg)
    cx, sx, cy, sy, cz, sz = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


# ---- shapes and anchors (in the author's coordinates of one part) ----------------------------------


def shape_box(spec: BaseSpec | PartSpec) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds (low, high) of a part's main shape in its own author coordinates."""
    shape, size = spec.shape, spec.size
    offset = np.asarray(spec.offset, float)
    direction = getattr(spec, "direction", None)
    d = np.asarray(axis_vector(direction), float) if direction is not None else None
    if shape == "none":
        return offset.copy(), offset.copy()
    if shape == "box":
        half = np.asarray(size, float) / 2
        center = offset + (d * float(np.abs(d) @ half) if d is not None else 0.0)
        return center - half, center + half
    if shape in ("sphere", "caster"):
        r = size[0] / 2
        center = offset + (d * r if d is not None else 0.0)
        return center - r, center + r
    if shape == "wheel":
        r, w = size[0] / 2, size[1] / 2
        half = np.array([r, w, r])
        return offset - half, offset + half
    if shape == "rotor":  # a motor standing on z = 0 with its propeller on top (builder/rotor.py)
        from robotforge.design.builder.rotor import height

        r = size[0] / 2
        return offset + np.array([-r, -r, 0.0]), offset + np.array([r, r, height(size[0])])
    # capsule and cylinder: a segment with a radius
    r, length = size[0] / 2, size[1]
    u = d if d is not None else np.asarray(axis_vector(getattr(spec, "axis", "z")), float)
    start = offset if d is not None else offset - u * length / 2
    end = start + u * length
    # A capsule's round ends reach r in every direction; a flat-ended cylinder only sideways from its axis.
    pad = np.full(3, r) if shape == "capsule" else r * np.sqrt(np.clip(1.0 - u * u, 0.0, 1.0))
    return np.minimum(start, end) - pad, np.maximum(start, end) + pad


def link_end(spec: BaseSpec | PartSpec) -> np.ndarray:
    """The tip of a link: where the next joint of a chain naturally goes."""
    shape, size = spec.shape, spec.size
    offset = np.asarray(spec.offset, float)
    direction = getattr(spec, "direction", None)
    if direction is None:
        low, high = shape_box(spec)
        return np.array([(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]])  # top
    d = np.asarray(axis_vector(direction), float)
    if shape in ("capsule", "cylinder"):
        return offset + d * size[1]
    if shape in ("sphere", "caster"):
        return offset + d * size[0]
    if shape == "box":
        return offset + d * 2 * float(np.abs(d) @ (np.asarray(size, float) / 2))
    return offset.copy()


def anchor(spec: BaseSpec | PartSpec, where: str | list[float], owner: str) -> np.ndarray:
    """A point on a part, in its author coordinates: a named anchor, a mount, or numbers."""
    if not isinstance(where, str):
        return np.asarray(where, float)
    if where in spec.mounts:
        return np.asarray(spec.mounts[where], float)
    if where == "start":
        return np.zeros(3)
    if where == "end":
        return link_end(spec)
    low, high = shape_box(spec)
    center = (low + high) / 2
    faces = {
        "center": center,
        "top": [center[0], center[1], high[2]],
        "bottom": [center[0], center[1], low[2]],
        "front": [high[0], center[1], center[2]],
        "back": [low[0], center[1], center[2]],
        "left": [center[0], high[1], center[2]],
        "right": [center[0], low[1], center[2]],
    }
    if where in faces:
        return np.asarray(faces[where], float)
    options = [*ANCHORS, *spec.mounts]
    close = difflib.get_close_matches(where, options, n=2)
    hint = f" Did you mean {' or '.join(close)}?" if close else ""
    raise ValueError(
        f"{owner} has no anchor {where!r}.{hint} Anchors: {', '.join(options)}, or give [x, y, z] in meters"
    )


# ---- names ------------------------------------------------------------------------------------------


def mirror_name(name: str) -> str:
    """left <-> right; names without a side get "_mirror"."""
    swapped = name.replace("left", "\0").replace("right", "left").replace("\0", "right")
    return swapped if swapped != name else f"{name}_mirror"


# ---- expansion ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Copy:
    pos: np.ndarray  # attachment point in the parent's author coordinates
    rot: np.ndarray
    reflect: np.ndarray  # I or MIRROR_Y: applied to the copy's own numbers on top of the parent's
    rename: Callable[[str], str]
    chained: bool  # a chain link after the first: it hangs on the previous link of the same side
    flip: int = 1  # -1 on every other copy of a ring or a row


def expand(build: BuildSpec) -> list[Placed]:
    """All parts in build order (every parent before its children)."""
    by_name = {p.name: p for p in build.parts}
    children: dict[str, list[PartSpec]] = {}
    for part in build.parts:
        children.setdefault(part.parent, []).append(part)
    _check_cycles(by_name)

    placed: list[Placed] = []
    specs: dict[str, BaseSpec | PartSpec] = {"base": build.base}
    locals_: dict[str, np.ndarray] = {"base": np.eye(3)}
    flips: dict[str, int] = {"base": 1}

    def place(part: PartSpec, parent_final: str, parent_spec: BaseSpec | PartSpec, rename, flip: int = 1) -> None:
        previous = parent_final
        for copy in _copies(part, parent_spec, rename):
            parent_name = previous if copy.chained else parent_final
            parent_local = locals_[parent_name]
            local = parent_local @ copy.reflect
            name = copy.rename(part.name)
            if name in specs:
                raise ValueError(f"two parts end up named {name!r} after mirror/repeat; rename one of them")
            # Conjugating by the reflection keeps the frame a proper rotation; `local` reflects the contents.
            rot = parent_local @ copy.rot @ copy.reflect @ parent_local.T
            placed.append(Placed(name, part, parent_name, parent_local @ copy.pos, rot, local, flip * copy.flip))
            specs[name], locals_[name], flips[name] = part, local, flip * copy.flip
            previous = name
            if len(placed) > MAX_PARTS:
                raise ValueError(f"the robot has more than {MAX_PARTS} parts after mirror and repeat; make it smaller")
            for child in children.get(part.name, []):
                place(child, name, part, copy.rename, flip * copy.flip)

    for part in children.get("base", []):
        place(part, "base", build.base, lambda n: n)

    # Parts hung on a copy by its final name, e.g. parent: leg_2 (only that copy gets them).
    pending = [p for p in build.parts if p.parent != "base" and p.parent not in by_name]
    while pending:
        ready = [p for p in pending if p.parent in specs]
        if not ready:
            part = pending[0]
            known = sorted(specs)
            close = difflib.get_close_matches(part.parent, known, n=3)
            hint = f" Did you mean {', '.join(close)}?" if close else ""
            raise ValueError(f"part {part.name!r}: no parent named {part.parent!r}.{hint} Parents: {', '.join(known)}")
        for part in ready:
            place(part, part.parent, specs[part.parent], lambda n: n, flips[part.parent])
            pending.remove(part)
    return placed


def _copies(part: PartSpec, parent_spec: BaseSpec | PartSpec, rename) -> list[_Copy]:
    """Every copy of `part` that mirror and repeat ask for, side by side, in chain order."""
    owner = "the base" if isinstance(parent_spec, BaseSpec) else f"part {parent_spec.name!r}"
    try:
        at = anchor(parent_spec, part.at, owner) + np.asarray(part.shift, float)
        on_self = anchor(part, part.at, f"part {part.name!r}") + np.asarray(part.shift, float)
    except ValueError as exc:
        raise ValueError(f"part {part.name!r}: {exc}") from None
    rot = euler(part.rotate)
    rep = part.repeat
    # (position, rotation, name suffix, chained, flip) in the parent's author coordinates
    placements: list[tuple[np.ndarray, np.ndarray, str, bool, int]] = []
    if rep is None:
        placements.append((at, rot, "", False, 1))
    elif rep.chain:
        # Later links use the same anchor, measured on the previous link (which has this part's shape).
        placements += [(at if i == 0 else on_self, rot, f"_{i + 1}", i > 0, 1) for i in range(rep.count)]
    elif rep.around is not None:
        axis = np.asarray(axis_vector(rep.around), float)
        step = math.radians(rep.step if rep.step is not None else 360.0 / rep.count)
        for i in range(rep.count):
            turn = axis_rotation(axis, i * step)
            placements.append((turn @ at, turn @ rot, f"_{i + 1}", False, (-1) ** i))
    else:
        along = np.asarray(axis_vector(rep.along), float)
        placements += [(at + along * rep.spacing * i, rot, f"_{i + 1}", False, (-1) ** i) for i in range(rep.count)]

    variants: list[tuple[np.ndarray, Callable[[str], str]]] = [(np.eye(3), lambda n: n)]
    if part.mirror:
        variants.append((MIRROR_Y, mirror_name))
    out = []
    for reflect, side in variants:
        for pos, r, suffix, chained, flip in placements:

            def named(n: str, side=side, suffix=suffix) -> str:
                return rename(side(n) + suffix)

            if chained:  # the previous link already carries the reflection
                out.append(_Copy(pos, r, np.eye(3), named, True))
            else:
                out.append(_Copy(reflect @ pos, reflect @ r, reflect, named, False, flip))
    return out


def _check_cycles(by_name: dict[str, PartSpec]) -> None:
    for start in by_name:
        seen = []
        node = start
        while node in by_name:
            if node in seen:
                loop = " -> ".join([*seen[seen.index(node) :], node])
                raise ValueError(f"parts hang on each other in a loop: {loop}; one of them should hang on base")
            seen.append(node)
            node = by_name[node].parent
