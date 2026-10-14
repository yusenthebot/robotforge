"""Ground: the floor's look and grip, plus terrain patches that make it uneven.

A world's `terrain` is a list of patches laid on the floor. Hills and rough ground
are generated heightfields (MuJoCo hfields); stairs, ramps, stepping stones, curbs
and platforms are crisp boxes, which give feet clean edges and several contact
points. Every patch has a center `pos` [x, y] in meters, a heading `yaw` in degrees
and a `style` (look and grip). Stairs and ramps climb along the patch's own +x.

`Ground` answers "how high is the ground here?", so robots and objects can be
placed on top of whatever terrain is under them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robotforge.sim.geometry import collides, lowest_point

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")  # same rule as robot and object names


@dataclass(frozen=True)
class Style:
    """How a ground surface looks (color and texture pattern) and grips (sliding friction)."""

    rgba: tuple[float, float, float, float]
    friction: float
    pattern: Literal["checker", "speckle", "tiles", "planks"]
    tile: float  # meters per texture repeat


STYLES: dict[str, Style] = {
    "checker": Style((0.3, 0.33, 0.37, 1.0), 1.0, "checker", 1.0),
    "grass": Style((0.33, 0.52, 0.27, 1.0), 1.0, "speckle", 0.6),
    "wood": Style((0.55, 0.42, 0.3, 1.0), 0.8, "planks", 1.2),
    "concrete": Style((0.5, 0.5, 0.5, 1.0), 1.0, "speckle", 1.0),
    "dirt": Style((0.46, 0.35, 0.25, 1.0), 0.9, "speckle", 0.6),
    "sand": Style((0.78, 0.68, 0.5, 1.0), 0.7, "speckle", 0.6),
    "rock": Style((0.48, 0.47, 0.46, 1.0), 1.0, "speckle", 0.8),
    "snow": Style((0.88, 0.9, 0.94, 1.0), 0.35, "speckle", 0.8),
    "tile": Style((0.8, 0.79, 0.76, 1.0), 0.8, "tiles", 0.6),
}
NATURAL = ("grass", "dirt", "sand", "rock", "snow")
StyleName = Literal["checker", "grass", "wood", "concrete", "dirt", "sand", "rock", "snow", "tile"]

HFIELD_SINK = 0.002  # m: heightfields sit this far down so their flat rims hide under the floor (no lip, no flicker)
HFIELD_BASE = 0.05  # m of solid ground modeled under a heightfield
MAX_GRID = 256  # heightfield samples per side; bigger patches get coarser cells
SPAWN_CLEARANCE = 0.002  # m between a robot's lowest point and the ground at spawn
_RAY_TOP = 50.0  # m: height queries cast rays down from here


# ---- schema --------------------------------------------------------------------------------


class _Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str  # each patch type narrows this to its own name; declared here so YAML lists it first
    name: str | None = None  # filled in by the world ("hills1", ...) when left out
    pos: list[float] = Field(default_factory=lambda: [0.0, 0.0])  # center [x, y] in meters
    yaw: float = 0.0  # degrees, counter-clockwise seen from above
    style: StyleName | None = None  # look and grip; None picks one that suits the type
    friction: float | None = Field(None, gt=0, le=3)  # overrides the style's grip

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        if v is not None and not _NAME_RE.match(v):
            raise ValueError("names use lowercase letters, digits and _, start with a letter, max 32 chars")
        return v

    @field_validator("pos")
    @classmethod
    def _pos(cls, v: list[float]) -> list[float]:
        if len(v) != 2:
            raise ValueError("terrain pos is the patch center [x, y] in meters")
        return v

    @model_validator(mode="after")
    def _size(self) -> _Patch:
        size = getattr(self, "size", None)
        if size is not None:
            if len(size) != 2 or any(v <= 0 for v in size):
                raise ValueError(f"{self.type} size is [x, y] in meters, both positive")
            if max(size) > 100:
                raise ValueError(f"{self.type} size is at most 100 m per side")
        return self

    def default_style(self, floor_style: str) -> str:  # pragma: no cover - every type overrides
        return "concrete"

    def footprint(self) -> tuple[float, float]:  # pragma: no cover - every type overrides
        """Full extent along the patch's own x (the climbing direction for stairs and ramps) and y, m."""
        raise NotImplementedError


class Hills(_Patch):
    """Smooth rolling hills, generated from seeded noise; the rim blends into the floor."""

    type: Literal["hills"]
    size: list[float] = Field(default_factory=lambda: [6.0, 6.0])
    height: float = Field(0.25, gt=0, le=5, description="highest hilltop above the floor, m")
    scale: float = Field(1.5, ge=0.2, le=30, description="typical distance between hilltops, m")
    seed: int = 0

    def default_style(self, floor_style: str) -> str:
        return floor_style if floor_style in NATURAL else "grass"

    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


class Rough(_Patch):
    """Bumpy ground: many small seeded bumps, like a dirt track or a lawn full of roots."""

    type: Literal["rough"]
    size: list[float] = Field(default_factory=lambda: [3.0, 3.0])
    height: float = Field(0.03, gt=0, le=0.5, description="tallest bump, m")
    scale: float = Field(0.25, ge=0.05, le=3, description="typical bump width, m")
    seed: int = 0

    def default_style(self, floor_style: str) -> str:
        return floor_style if floor_style in NATURAL else "dirt"

    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


class Stairs(_Patch):
    """A flight of stairs: `steps` risers up to a landing (and down again for up_down)."""

    type: Literal["stairs"]
    steps: int = Field(4, ge=1, le=40, description="risers to climb from the floor to the landing")
    step_height: float = Field(0.05, gt=0, le=0.5)
    step_depth: float = Field(0.3, ge=0.05, le=3)
    width: float = Field(1.2, gt=0, le=30)
    direction: Literal["up", "down", "up_down"] = "up_down"  # seen walking along +x
    landing: float = Field(0.8, ge=0.2, le=30, description="flat top, m along x")

    def default_style(self, floor_style: str) -> str:
        return "concrete"

    def footprint(self) -> tuple[float, float]:
        flights = 2 if self.direction == "up_down" else 1
        return flights * (self.steps - 1) * self.step_depth + self.landing, self.width


class Ramp(_Patch):
    """A slope of `angle` degrees up to a landing (and down again for up_down)."""

    type: Literal["ramp"]
    angle: float = Field(10.0, gt=0, le=45, description="slope in degrees")
    length: float = Field(1.5, gt=0.1, le=50, description="horizontal run of each slope, m")
    width: float = Field(1.2, gt=0, le=30)
    direction: Literal["up", "down", "up_down"] = "up_down"
    landing: float = Field(0.8, ge=0.2, le=30)

    def default_style(self, floor_style: str) -> str:
        return "concrete"

    @property
    def rise(self) -> float:
        return self.length * math.tan(math.radians(self.angle))

    def footprint(self) -> tuple[float, float]:
        flights = 2 if self.direction == "up_down" else 1
        return flights * self.length + self.landing, self.width


class Stones(_Patch):
    """Stepping stones: a grid of square blocks with gaps, tops at slightly random heights."""

    type: Literal["stones"]
    size: list[float] = Field(default_factory=lambda: [2.4, 1.2])
    stone: float = Field(0.25, ge=0.05, le=2, description="side of each stone, m")
    gap: float = Field(0.08, ge=0.0, le=2)
    height: float = Field(0.08, gt=0, le=2)
    jitter: float = Field(0.02, ge=0, le=0.5, description="random +- change of each stone's height, m")
    seed: int = 0

    @model_validator(mode="after")
    def _fits(self) -> Stones:
        if self.stone > min(self.size):
            raise ValueError(f"stones: a stone ({self.stone} m) does not fit in size {self.size}")
        return self

    def default_style(self, floor_style: str) -> str:
        return "rock"

    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


class Curb(_Patch):
    """A low bar across the way, like a sidewalk curb or a door threshold."""

    type: Literal["curb"]
    size: list[float] = Field(default_factory=lambda: [0.15, 2.0])
    height: float = Field(0.06, gt=0, le=1)

    def default_style(self, floor_style: str) -> str:
        return "concrete"

    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


class Platform(_Patch):
    """A raised flat block: a stage, a loading dock, a step up."""

    type: Literal["platform"]
    size: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    height: float = Field(0.1, gt=0, le=5)

    def default_style(self, floor_style: str) -> str:
        return "wood"

    def footprint(self) -> tuple[float, float]:
        return self.size[0], self.size[1]


TerrainPatch = Annotated[Hills | Rough | Stairs | Ramp | Stones | Curb | Platform, Field(discriminator="type")]
TYPES: dict[str, type[_Patch]] = {
    "hills": Hills,
    "rough": Rough,
    "stairs": Stairs,
    "ramp": Ramp,
    "stones": Stones,
    "curb": Curb,
    "platform": Platform,
}


def assign_names(patches: list[_Patch], taken: set[str]) -> None:
    """Give unnamed patches a name like hills1, ramp2 that no robot, object or patch uses."""
    used = set(taken) | {p.name for p in patches if p.name}
    for patch in patches:
        if patch.name:
            continue
        n = 1
        while f"{patch.type}{n}" in used:
            n += 1
        patch.name = f"{patch.type}{n}"
        used.add(patch.name)


def catalog() -> dict[str, dict]:
    """Every terrain type with its summary and parameters, for agents and docs."""
    out = {}
    for name, cls in TYPES.items():
        schema = cls.model_json_schema()
        params = {}
        for key, prop in schema["properties"].items():
            if key in ("type", "name"):
                continue
            entry = {k: v for k, v in prop.items() if k in ("default", "description", "enum", "minimum", "maximum")}
            if "anyOf" in prop:
                enums = [e for alt in prop["anyOf"] for e in alt.get("enum", [])]
                if enums:
                    entry["enum"] = enums
            params[key] = entry
        out[name] = {"summary": (cls.__doc__ or "").strip(), "params": params}
    return out


# ---- materials -----------------------------------------------------------------------------


def material_name(style: str) -> str:
    """The MuJoCo material of a ground style; the viewer reads the style back from this name."""
    return f"ground_{style}"


def _ensure_material(spec: mujoco.MjSpec, style: str) -> str:
    name = material_name(style)
    if any(m.name == name for m in spec.materials):  # not spec.material(): it can miss names after an attach
        return name
    look = STYLES[style]
    T, B, M = mujoco.mjtTexture, mujoco.mjtBuiltin, mujoco.mjtMark
    # Textures are grayscale; the material color tints them, so the viewer and MuJoCo agree on the color.
    if look.pattern == "checker":
        tex = {"builtin": B.mjBUILTIN_CHECKER, "rgb1": [1.0] * 3, "rgb2": [0.9] * 3, "mark": M.mjMARK_NONE}
        repeat = [1.0 / look.tile] * 2
    elif look.pattern == "tiles":
        tex = {"builtin": B.mjBUILTIN_CHECKER, "rgb1": [1.0] * 3, "rgb2": [0.96] * 3, "mark": M.mjMARK_EDGE}
        tex["markrgb"] = [0.72] * 3
        repeat = [0.5 / look.tile] * 2  # one checker texture holds 2 x 2 tiles
    elif look.pattern == "planks":
        tex = {"builtin": B.mjBUILTIN_FLAT, "rgb1": [1.0] * 3, "mark": M.mjMARK_EDGE, "markrgb": [0.7] * 3}
        repeat = [1.0 / look.tile, 1.0 / 0.16]  # long planks along x
    else:  # speckle: fine random grain, like grass, gravel or concrete seen from standing height
        tex = {"builtin": B.mjBUILTIN_FLAT, "rgb1": [1.0] * 3, "mark": M.mjMARK_RANDOM, "random": 0.35}
        tex["markrgb"] = [0.84] * 3
        repeat = [1.0 / look.tile] * 2
    spec.add_texture(name=f"{name}_tex", type=T.mjTEXTURE_2D, width=256, height=256, **tex)
    mat = spec.add_material(name=name, rgba=list(look.rgba), texrepeat=repeat, texuniform=True, reflectance=0.0)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{name}_tex"
    return name


def add_floor(spec: mujoco.MjSpec, style: str, friction: float) -> None:
    """The infinite floor plane at z = 0."""
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        material=_ensure_material(spec, style),
        friction=[friction, 0.01, 0.001],
    )


# ---- building patches ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Box:
    """A box in the patch frame: center, half sizes, pitch about y (radians, for ramp slabs)."""

    center: tuple[float, float, float]
    half: tuple[float, float, float]
    pitch: float = 0.0


def add_terrain(spec: mujoco.MjSpec, patches: list[_Patch], floor_style: str) -> list[str]:
    """Add every patch to the world body. Returns one plain-language note per patch."""
    notes = []
    for patch in patches:
        style = patch.style or patch.default_style(floor_style)
        material = _ensure_material(spec, style)
        friction = patch.friction if patch.friction is not None else STYLES[style].friction
        if isinstance(patch, Hills | Rough):
            notes.append(_add_heightfield(spec, patch, material, friction))
        else:
            boxes = _boxes(patch)
            _add_boxes(spec, patch, boxes, material, friction)
            notes.append(_describe_blocks(patch, boxes))
    return notes


def _yaw_frame(patch: _Patch) -> tuple[np.ndarray, float, float]:
    """(center [x, y], cos yaw, sin yaw) of a patch."""
    a = math.radians(patch.yaw)
    return np.array(patch.pos[:2], float), math.cos(a), math.sin(a)


def _add_boxes(spec: mujoco.MjSpec, patch: _Patch, boxes: list[_Box], material: str, friction: float) -> None:
    center, c, s = _yaw_frame(patch)
    yaw = math.radians(patch.yaw)
    for i, box in enumerate(boxes):
        lx, ly, z = box.center
        pos = [center[0] + c * lx - s * ly, center[1] + s * lx + c * ly, z]
        quat = _quat_yaw_pitch(yaw, box.pitch)
        spec.worldbody.add_geom(
            name=patch.name if i == 0 else f"{patch.name}/{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(box.half),
            pos=pos,
            quat=quat,
            material=material,
            friction=[friction, 0.01, 0.001],
        )


def _quat_yaw_pitch(yaw: float, pitch: float) -> list[float]:
    """Rotation by yaw about z, then (in the turned frame) by pitch about y."""
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, qy)
    return out.tolist()


def _boxes(patch: _Patch) -> list[_Box]:
    """The patch as boxes in its own frame (x from -length/2 to +length/2, z = 0 at the floor)."""
    if isinstance(patch, Stairs):
        boxes = _stairs(patch)
    elif isinstance(patch, Ramp):
        boxes = _ramp(patch)
    elif isinstance(patch, Stones):
        return _stones(patch)
    elif isinstance(patch, Curb | Platform):
        return [_Box((0.0, 0.0, patch.height / 2), (patch.size[0] / 2, patch.size[1] / 2, patch.height / 2))]
    else:  # pragma: no cover - heightfields are handled elsewhere
        raise TypeError(type(patch).__name__)
    shift = patch.footprint()[0] / 2  # built from x = 0; center it
    return [_Box((b.center[0] - shift, b.center[1], b.center[2]), b.half, b.pitch) for b in boxes]


def _block(x0: float, x1: float, top: float, width: float) -> _Box:
    """A solid block from the floor up to `top`, spanning x0..x1."""
    return _Box(((x0 + x1) / 2, 0.0, top / 2), ((x1 - x0) / 2, width / 2, top / 2))


def _stairs(p: Stairs) -> list[_Box]:
    top = p.steps * p.step_height
    treads = [(i * p.step_depth, (i + 1) * p.step_depth, (i + 1) * p.step_height) for i in range(p.steps - 1)]
    run = (p.steps - 1) * p.step_depth
    boxes: list[_Box] = []
    if p.direction in ("up", "up_down"):
        boxes += [_block(x0, x1, h, p.width) for x0, x1, h in treads]
        start = run
    else:
        start = 0.0
    boxes.append(_block(start, start + p.landing, top, p.width))
    if p.direction in ("down", "up_down"):
        end = start + p.landing
        boxes += [_block(end + run - x1, end + run - x0, h, p.width) for x0, x1, h in treads]
    return boxes


def _ramp(p: Ramp) -> list[_Box]:
    """Slopes are tilted slabs whose top face runs exactly from the floor to the landing.

    The slab's underside dips below the floor at the low end and into the landing at the
    high end, both hidden, so the walking surface has no lip. Boxes (not a wedge mesh) give
    flat feet several contact points.
    """
    a = math.radians(p.angle)
    rise = p.rise
    thick = max(0.02, min(0.1, rise))
    slope_len = p.length / math.cos(a)
    normal = np.array([-math.sin(a), math.cos(a)])  # of the top face, in (x, z)

    def slab(x0: float, sign: float) -> _Box:
        # sign +1 rises along +x from (x0, 0); -1 falls along +x to (x0 + length, 0)
        mid_top = np.array([x0 + p.length / 2, rise / 2])
        n = normal * np.array([sign, 1.0])
        c = mid_top - n * thick / 2
        return _Box((float(c[0]), 0.0, float(c[1])), (slope_len / 2, p.width / 2, thick / 2), -sign * a)

    boxes: list[_Box] = []
    start = 0.0
    if p.direction in ("up", "up_down"):
        boxes.append(slab(0.0, 1.0))
        start = p.length
    boxes.append(_block(start, start + p.landing, rise, p.width))
    if p.direction in ("down", "up_down"):
        boxes.append(slab(start + p.landing, -1.0))
    return boxes


def _stones(p: Stones) -> list[_Box]:
    rng = np.random.default_rng(p.seed)
    pitch = p.stone + p.gap
    nx = max(1, int((p.size[0] + p.gap) // pitch))
    ny = max(1, int((p.size[1] + p.gap) // pitch))
    boxes = []
    for ix in range(nx):
        for iy in range(ny):
            x = (ix - (nx - 1) / 2) * pitch
            y = (iy - (ny - 1) / 2) * pitch
            top = max(0.01, p.height + rng.uniform(-p.jitter, p.jitter))
            boxes.append(_Box((x, y, top / 2), (p.stone / 2, p.stone / 2, top / 2)))
    return boxes


def _describe_blocks(patch: _Patch, boxes: list[_Box]) -> str:
    x0, x1, y0, y1 = _world_bounds(patch)
    where = f"x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f}"
    if isinstance(patch, Stairs):
        top = patch.steps * patch.step_height
        return f"{patch.name}: {patch.steps} steps of {patch.step_height * 100:.0f} cm up to {top:.2f} m ({where})"
    if isinstance(patch, Ramp):
        return f"{patch.name}: {patch.angle:g} degree ramp up to {patch.rise:.2f} m ({where})"
    if isinstance(patch, Stones):
        return f"{patch.name}: {len(boxes)} stepping stones about {patch.height * 100:.0f} cm high ({where})"
    return f"{patch.name}: {patch.type} {patch.height * 100:.0f} cm high ({where})"  # type: ignore[attr-defined]


def _world_bounds(patch: _Patch) -> tuple[float, float, float, float]:
    """Axis-aligned world bounds of a patch's footprint: x0, x1, y0, y1."""
    center, c, s = _yaw_frame(patch)
    hx, hy = patch.footprint()[0] / 2, patch.footprint()[1] / 2
    xs = [center[0] + c * dx - s * dy for dx in (-hx, hx) for dy in (-hy, hy)]
    ys = [center[1] + s * dx + c * dy for dx in (-hx, hx) for dy in (-hy, hy)]
    return min(xs), max(xs), min(ys), max(ys)


# ---- heightfields ---------------------------------------------------------------------------


def heights(patch: Hills | Rough) -> tuple[np.ndarray, float]:
    """Elevation grid in meters (rows along y from -y to +y, columns along x) and its cell size."""
    lx, ly = patch.size
    if isinstance(patch, Hills):
        cell = min(0.1, patch.scale / 12)
        octaves = [(1.0, 1.0), (0.35, 0.5), (0.12, 0.25)]
        margin = min(max(1.0, patch.scale), lx / 4, ly / 4)
    else:
        cell = min(0.04, patch.scale / 6)
        octaves = [(1.0, 1.0), (0.5, 0.5)]
        margin = min(0.3, lx / 4, ly / 4)
    cell = max(cell, max(lx, ly) / (MAX_GRID - 1))
    ncol = max(3, round(lx / cell) + 1)
    nrow = max(3, round(ly / cell) + 1)
    rng = np.random.default_rng(patch.seed)
    noise = sum(amp * _value_noise(nrow, ncol, lx, ly, patch.scale * f, rng) for amp, f in octaves)
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-9)
    # Blend into the floor at the rim, so the patch has no cliff and its edge rows are exactly 0.
    x = np.linspace(0.0, lx, ncol)
    y = np.linspace(0.0, ly, nrow)
    wx = _smoothstep(np.minimum(x, lx - x) / margin)
    wy = _smoothstep(np.minimum(y, ly - y) / margin)
    h = patch.height * noise * wy[:, None] * wx[None, :]
    return h * (patch.height / max(float(h.max()), 1e-9)), lx / (ncol - 1)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _value_noise(nrow: int, ncol: int, lx: float, ly: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random values on a lattice `scale` meters apart, sampled on the grid."""
    nx, ny = int(math.ceil(lx / scale)) + 2, int(math.ceil(ly / scale)) + 2
    lattice = rng.random((ny, nx))
    ox, oy = rng.random(2)  # shift the lattice too, so seeds differ in more than values
    u = np.linspace(0.0, lx / scale, ncol) + ox
    v = np.linspace(0.0, ly / scale, nrow) + oy
    iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = _fade(u - iu)[None, :], _fade(v - iv)[:, None]
    iu, iv = iu[None, :], iv[:, None]
    a, b = lattice[iv, iu], lattice[iv, iu + 1]
    c, d = lattice[iv + 1, iu], lattice[iv + 1, iu + 1]
    return (a + (b - a) * fu) * (1 - fv) + (c + (d - c) * fu) * fv


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6 - 15) + 10)  # smooth in value and slope


def _add_heightfield(spec: mujoco.MjSpec, patch: Hills | Rough, material: str, friction: float) -> str:
    h, cell = heights(patch)
    top = float(h.max())
    nrow, ncol = h.shape
    spec.add_hfield(
        name=patch.name,
        size=[patch.size[0] / 2, patch.size[1] / 2, top, HFIELD_BASE],
        nrow=nrow,
        ncol=ncol,
        userdata=(h / top).ravel().tolist(),  # MuJoCo rescales to 0..1 anyway; the edges are exactly 0
    )
    yaw = math.radians(patch.yaw)
    spec.worldbody.add_geom(
        name=patch.name,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=patch.name,
        pos=[patch.pos[0], patch.pos[1], -HFIELD_SINK],
        quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
        material=material,
        friction=[friction, 0.01, 0.001],
    )
    gy, gx = np.gradient(h, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    x0, x1, y0, y1 = _world_bounds(patch)
    return (
        f"{patch.name}: {patch.type} up to {top * 100:.0f} cm, steepest slope {slope.max():.0f} degrees "
        f"(x {x0:.2f}..{x1:.2f}, y {y0:.2f}..{y1:.2f})"
    )


# ---- ground height queries -------------------------------------------------------------------


@dataclass
class Ground:
    """Height of the floor plus terrain at any (x, y), from MuJoCo's own ray casts.

    Built from a small model that holds only the floor and the terrain, so robots and
    objects never count as ground. A world without terrain is flat at z = 0.
    """

    model: mujoco.MjModel | None = None
    data: mujoco.MjData | None = field(default=None, repr=False)

    @classmethod
    def build(cls, patches: list[_Patch], floor_style: str) -> Ground:
        if not patches:
            return cls()
        spec = mujoco.MjSpec()
        add_floor(spec, floor_style, 1.0)
        add_terrain(spec, patches, floor_style)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        return cls(model, data)

    @property
    def flat(self) -> bool:
        return self.model is None

    def height(self, x: float, y: float) -> float:
        if self.model is None:
            return 0.0
        geom = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(
            self.model, self.data, np.array([x, y, _RAY_TOP]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geom
        )
        return 0.0 if dist < 0 else max(0.0, _RAY_TOP - float(dist))

    def under(self, x: float, y: float, half_x: float, half_y: float, yaw_deg: float = 0.0) -> float:
        """Highest ground under a rectangle's center, corners and edge midpoints."""
        if self.model is None:
            return 0.0
        a = math.radians(yaw_deg)
        c, s = math.cos(a), math.sin(a)
        best = 0.0
        for fx in (-1.0, 0.0, 1.0):
            for fy in (-1.0, 0.0, 1.0):
                dx, dy = fx * half_x, fy * half_y
                best = max(best, self.height(x + c * dx - s * dy, y + s * dx + c * dy))
        return best

    def lift_for(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> float:
        """How far to raise (or lower) a posed robot so its lowest collision geom rests on the ground."""
        need = -math.inf
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if body == 0 or not name.startswith(prefix) or not collides(model, g):
                continue
            low = lowest_point(model, data, g)
            need = max(need, self.height(float(low[0]), float(low[1])) + SPAWN_CLEARANCE - float(low[2]))
        return 0.0 if need == -math.inf else need
