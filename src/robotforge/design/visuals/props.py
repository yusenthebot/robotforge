"""Looks for scene props and objects: trees, benches, rocks, tables, trays, shelves, walls, hedges, ...

Each `_<shape>` function draws one prop in its own body frame (base point on the ground at the
center of its footprint, `length` along x) on top of the collision primitives that props.py and
world.py build. They follow those primitives closely, so what a robot bumps into is what it sees.
Organic shapes (canopies, rocks, hedges) are seeded by the object's name: the same world always
looks the same, and two trees never look alike.
"""

from __future__ import annotations

import functools
import math
import zlib
from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from robotforge.design.visuals import Visuals, material_name, mix, shade
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

BARK = [0.36, 0.26, 0.19, 1.0]
IRON = [0.17, 0.18, 0.19, 1.0]
STONE = [0.55, 0.54, 0.51, 1.0]
CONCRETE = [0.66, 0.65, 0.62, 1.0]
STEEL_LIGHT = [0.74, 0.76, 0.78, 1.0]
WATER_TINT = [0.22, 0.42, 0.55, 1.0]

Look = Callable[["Placed", list[float], list[float], str], None]


def seed_of(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def woody(rgba: Sequence[float]) -> bool:
    r, g, b = rgba[:3]
    return r > g > b and r - b > 0.12 and (r - b) / max(r, 1e-6) < 0.62


def leafy(rgba: Sequence[float]) -> bool:
    r, g, b = rgba[:3]
    return g - max(r, b) > 0.08


def dress(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    shape: str,
    size: Sequence[float],
    rgba: Sequence[float],
    name: str,
    covered: Sequence[mujoco.MjsGeom],
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat: Sequence[float] | None = None,
) -> None:
    """Draw `shape` on `body` and hide the primitives it covers. `pos` and `quat` place the prop's
    base frame in the body (for fixed objects that are plain geoms on the world body)."""
    look = look_for(shape, name)
    if look is None:
        return
    vis = Visuals(spec, owner=name)
    if not vis.on:
        return
    for geom in covered:
        vis.hide(geom)
    for preset, color, mesh in _drawn(shape, M._key(size), M._key(rgba), name, M._key(pos), M._key(quat)):
        vis.add(body, preset, color, mesh)
    vis.finish()


@functools.lru_cache(maxsize=1024)
def _drawn(shape: str, size: tuple, rgba: tuple, name: str, pos: tuple, quat: tuple | None) -> tuple:
    """A prop's parts as (preset, color, mesh), merged per material and color; cached, so a world
    recompile (a robot moved, an object added) redraws nothing it has drawn before."""
    look = look_for(shape, name)
    rot = None if quat is None else M.quat_to_mat(quat)
    placed = Placed(pos, rot)
    look(placed, list(size), list(rgba), name)
    return tuple((preset, list(color), M.merge(meshes)) for (preset, color), meshes in placed.parts.items())


class Placed:
    """Collects a prop's parts, each moved by a fixed frame (the prop's base point and heading)."""

    def __init__(self, pos: Sequence[float], rot: np.ndarray | None) -> None:
        self.pos, self.rot = np.asarray(pos, dtype=float), rot
        self.parts: dict[tuple[str, tuple[float, ...]], list[M.Mesh]] = {}

    def add(self, preset: str, rgba, mesh: M.Mesh, pos=(0.0, 0.0, 0.0), rot=None) -> None:
        material_name(preset)  # fail early on a typo
        mesh = mesh.moved(pos, rot)
        if self.rot is not None or self.pos.any():
            mesh = mesh.moved(self.pos, self.rot)
        color = tuple(round(float(c), 4) for c in rgba)
        self.parts.setdefault((preset, color), []).append(mesh)


def look_for(shape: str, name: str) -> Look | None:
    """The look for a prop shape, or for a plain box/sphere/cylinder the look its name suggests."""
    if shape in _LOOKS:
        return _LOOKS[shape]
    lowered = name.lower()
    for words, look in _NAMED.get(shape, ()):
        if any(w in lowered for w in words):
            return look
    return _PLAIN.get(shape)


# ---- park ----------------------------------------------------------------------------------------


def _tree(vis: Placed, size, rgba, name) -> None:
    from robotforge.design.props import CROWN  # the collision crown: the canopy follows it

    crown, height = size
    r = crown / 2
    trunk_r = max(0.04, crown * 0.06)
    center = height - 0.55 * r
    trunk_h = max(0.3, center - 0.2 * r)
    rng = np.random.default_rng(seed_of(name))
    top = trunk_h + 0.3 * r
    zs = np.array([0.0, 0.03, 0.12, 0.35, 0.65, 1.0]) * top
    radii = trunk_r * np.array([1.55, 1.18, 1.02, 0.94, 0.8, 0.55])
    lean = rng.normal(0, 0.18 * trunk_r, size=len(zs)).cumsum() * np.linspace(0, 1, len(zs))
    trunk = _trunk(radii, zs, lean, rng)
    branches = []
    for k in range(3):
        a = rng.uniform(0, 2 * math.pi) + k * 2.1
        z0 = trunk_h * rng.uniform(0.72, 0.92)
        reach = r * rng.uniform(0.35, 0.55)
        end = np.array([math.cos(a) * reach, math.sin(a) * reach, z0 + r * rng.uniform(0.35, 0.6)])
        mid = np.array([end[0] * 0.45, end[1] * 0.45, z0 + (end[2] - z0) * 0.35])
        path = np.array([[0.0, 0.0, z0 - trunk_r], mid, end])
        branches.append(M.tube(path, [trunk_r * 0.5, trunk_r * 0.34, trunk_r * 0.16], segments=8))
    vis.add("bark", BARK, M.merge([trunk, *branches]))
    # Canopy: a lumpy blob over each collision ball, darker below and lighter on top, plus a few
    # small tufts on the rim that break the outline.
    layers: dict[str, list[M.Mesh]] = {"low": [], "mid": [], "high": []}
    for i, (dx, dy, dz, k, _) in enumerate(CROWN):
        rad = r * k
        lump = M.blob(
            (rad * 1.04, rad * 1.04, rad * 0.94), seed_of(name) + i, amplitude=0.16, lumps=8, rings=9, segments=16
        )
        layer = "low" if dz < -0.05 else "high" if dz > 0.15 else "mid"
        layers[layer].append(lump.moved((dx * r, dy * r, center + dz * r), M.rot_z(rng.uniform(0, 2 * math.pi))))
    for t in range(4):
        a = 2 * math.pi * t / 4 + rng.uniform(-0.4, 0.4)
        rad = r * rng.uniform(0.28, 0.36)
        pos = (math.cos(a) * r * 0.78, math.sin(a) * r * 0.78, center + r * rng.uniform(-0.25, 0.1))
        layers["mid"].append(
            M.blob((rad, rad, rad * 0.85), seed_of(name) + 50 + t, amplitude=0.2, rings=7, segments=12).moved(pos)
        )
    for layer, k in (("low", 0.82), ("mid", 0.95), ("high", 1.1)):
        if layers[layer]:
            # Flat-shaded facets read as clusters of leaves; smooth blobs read as cotton wool.
            vis.add("foliage", shade(rgba, k), M.merge(layers[layer]).faceted())


def _trunk(radii: np.ndarray, zs: np.ndarray, lean: np.ndarray, rng: np.random.Generator) -> M.Mesh:
    """A tapered, slightly crooked trunk with shallow bark ridges running up it."""
    n = 27
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    phase = rng.uniform(0, 2 * math.pi)
    rings = []
    for k, (rr, z, dx) in enumerate(zip(radii, zs, lean, strict=True)):
        ridge = 1.0 + 0.05 * np.sin(9 * a + phase + 0.8 * k) + 0.03 * np.sin(4 * a - phase)
        if k == 0:  # the root flare spreads into a few roots
            ridge = ridge + 0.12 * np.maximum(np.sin(5 * a + phase), 0) ** 2
        ring = np.column_stack([rr * ridge * np.cos(a) + dx, rr * ridge * np.sin(a), np.full(n, z)])
        rings.append(ring)
    return M.loft(rings)


def _bench(vis: Placed, size, rgba, name) -> None:
    length, depth, seat = size
    slab = 0.03
    leg_x = length / 2 - min(0.15, length / 6)
    wood = "wood" if woody(rgba) else "paint"
    # Seat: slats with small gaps across the depth.
    n = max(3, round(depth / 0.1))
    gap = 0.01
    w = (depth - (n - 1) * gap) / n
    slat = M.rounded_box(length / 2, w / 2, slab / 2, 0.006, segments=2, corner=0.008)
    slats = [slat.moved((0, -depth / 2 + w / 2 + i * (w + gap), seat - slab / 2)) for i in range(n)]
    # Backrest: two slats where the collision board is.
    back_y = depth / 2 - 0.02
    board = M.rounded_box(length / 2, 0.015, 0.04, 0.006, segments=2, corner=0.008)
    slats += [board.moved((0, back_y, seat + 0.27 + dz)) for dz in (-0.046, 0.046)]
    vis.add(wood, rgba, M.merge(slats))
    # Cast iron ends: a rounded side panel, a post up the back, and bolt heads on the slats.
    side = _panel(depth / 2 - 0.03, (seat - slab) / 2, 0.025, corner=0.035)
    post = M.rounded_box(0.025, 0.02, 0.2, 0.008, segments=2)
    frame, bolts = [], []
    for sx in (1, -1):
        frame.append(side.moved((sx * leg_x, 0, (seat - slab) / 2)))
        frame.append(post.moved((sx * leg_x, back_y, seat + 0.2)))
        for i in range(n):
            y = -depth / 2 + w / 2 + i * (w + gap)
            bolts.append(P.bolt(0.006, 0.003).moved((sx * leg_x, y, seat - 0.0005)))
    vis.add("stone", CONCRETE, M.merge(frame[0::2]))
    vis.add("paint", IRON, M.merge(frame[1::2]))
    vis.add("metal", P.METAL, M.merge(bolts))


def _panel(half_w: float, half_h: float, half_t: float, corner: float) -> M.Mesh:
    """A flat panel in the y-z plane (thin along x) with rounded corners and edges."""
    m = M.rounded_box(half_w, half_h, half_t, min(half_t * 0.6, 0.01), segments=2, corner=corner)
    return m.moved(rot=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def _rock(vis: Placed, size, rgba, name) -> None:
    from robotforge.design.props import ROCK_LUMPS

    d, h = size
    r = d / 2
    rng = np.random.default_rng(seed_of(name))
    for i, (dx, dy, kxy, kz, k) in enumerate(ROCK_LUMPS):
        half = (r * kxy * 1.03, r * kxy * 0.85 * 1.03, h * kz * 0.6 * 1.05)
        lump = M.blob(half, seed_of(name) + i, amplitude=0.22, lumps=7, sharpness=3.0, rings=7, segments=11)
        lump = lump.moved((dx * r, dy * r, h * kz * 0.4), M.rot_z(rng.uniform(-0.3, 0.3)))
        vis.add("stone", shade(rgba, k), lump.faceted())


def _pond(vis: Placed, size, rgba, name) -> None:
    """A water mark gets a glossy surface over a darker bed and a low pebble rim."""
    from robotforge.design.props import MARK

    rr = size[0] / 2
    vis.add("paint", shade(rgba, 0.7), M.cylinder(rr, MARK, segments=64, z0=0.0))
    vis.add("glass", mix(rgba, WATER_TINT, 0.35), M.cylinder(rr * 0.97, 0.002, segments=64, z0=MARK))
    rim = M.revolve([(rr * 0.93, 0.0), (rr * 1.06, 0.0), (rr * 1.04, MARK + 0.012), (rr * 0.95, MARK + 0.01)], 64)
    vis.add("stone", STONE, rim)


# ---- furniture -----------------------------------------------------------------------------------


def _table(vis: Placed, size, rgba, name) -> None:
    length, width, height = size
    top = 0.025
    leg = min(0.025, length / 10, width / 10)
    wood = "wood" if woody(rgba) else "paint"
    slab = M.rounded_box(length / 2, width / 2, top / 2, min(0.008, top * 0.4), segments=3, corner=min(0.03, width / 8))
    vis.add(wood, rgba, slab, (0, 0, height - top / 2))
    legs, apron = [], []
    leg_h = height - top
    lx, ly = length / 2 - 2 * leg, width / 2 - 2 * leg
    taper = M.link(leg_h, (leg * 0.78, leg * 0.78), (leg, leg), roundness=0.35, rounding=(0.05, 0.05))
    for sx in (1, -1):
        for sy in (1, -1):
            legs.append(taper.moved((sx * lx, sy * ly, 0.0)))
    skirt = min(0.08, leg_h * 0.12)
    t = max(0.008, leg * 0.45)
    for sy in (1, -1):
        apron.append(M.rounded_box(lx, t, skirt / 2, t * 0.4, segments=2).moved((0, sy * ly, leg_h - skirt / 2)))
    for sx in (1, -1):
        apron.append(M.rounded_box(t, ly, skirt / 2, t * 0.4, segments=2).moved((sx * lx, 0, leg_h - skirt / 2)))
    vis.add(wood, shade(rgba, 0.82), M.merge(legs + apron))
    feet = [
        M.cylinder(leg * 0.7, 0.006, segments=12, z0=0.0).moved((sx * lx, sy * ly, 0))
        for sx in (1, -1)
        for sy in (1, -1)
    ]
    vis.add("rubber", P.RUBBER, M.merge(feet))


def _bin(vis: Placed, size, rgba, name) -> None:
    """An open tray with rounded corners and a rolled rim (walls as thick as the collision walls)."""
    length, width, height = size
    t = 0.005
    hx, hy = length / 2, width / 2
    corner = min(hx, hy) * 0.28
    lip = min(0.003, height * 0.08)
    rings = [
        (0.0, hx - 0.003, hy - 0.003, corner - 0.003),
        (0.003, hx, hy, corner),
        (height - lip, hx, hy, corner),
        (height, hx - lip * 0.4, hy - lip * 0.4, corner - lip * 0.4),
        (height, hx - t, hy - t, corner - t),
        (t + 0.004, hx - t, hy - t, corner - t),
        (t, hx - t - 0.004, hy - t - 0.004, max(corner - t - 0.004, 0.002)),
    ]
    loft = M.loft([M.at_z(M.rounded_rect(a, b, c, 5), z) for z, a, b, c in rings], sharp=(1, 2, 4, 5))
    lowered = name.lower()
    preset = "metal" if "sink" in lowered else "ceramic" if any(w in lowered for w in ("bowl", "dish")) else "shell"
    vis.add(preset, rgba, loft)


def _shelf(vis: Placed, size, rgba, name) -> None:
    from robotforge.design.props import PANEL, shelf_levels

    length, depth, height = size
    t = PANEL
    is_wood = woody(rgba)
    frame = "wood" if is_wood else "paint"
    side = _panel(depth / 2, height / 2, t / 2, corner=0.012)
    sides = [side.moved((sx * (length / 2 - t / 2), 0, height / 2)) for sx in (1, -1)]
    vis.add(frame, rgba, M.merge(sides))
    back = M.box(length / 2 - t, t / 2 * 0.6, height / 2 - 0.002).moved((0, depth / 2 - t * 0.3, height / 2))
    vis.add(frame, shade(rgba, 0.78), back)
    boards = []
    for level in shelf_levels(size):
        board = M.rounded_box(length / 2 - t, depth / 2 - t / 2, t / 2, 0.004, segments=2, corner=0.003)
        boards.append(board.moved((0, -t / 2, level - t / 2)))
    if is_wood:
        vis.add("wood", shade(rgba, 1.06), M.merge(boards))
    else:  # steel racking: galvanized decks and a bright safety edge on each beam
        vis.add("metal", STEEL_LIGHT, M.merge(boards))
        beams = [
            M.rounded_box(length / 2 - t, 0.006, 0.022, 0.003, segments=2).moved((0, -depth / 2 + 0.006, lv - 0.022))
            for lv in shelf_levels(size)[1:]
        ]
        vis.add("paint", [0.95, 0.55, 0.15, 1.0], M.merge(beams))


def _wall(vis: Placed, size, rgba, name) -> None:
    if leafy(rgba):
        _hedge(vis, size, rgba, name)
        return
    length, thick, height = size
    slab = M.rounded_box(length / 2, thick / 2, height / 2, min(0.006, thick * 0.1), segments=2, corner=0.004)
    vis.add("paint", rgba, slab, (0, 0, height / 2))
    base_h = min(0.08, height * 0.08)
    skirting = M.rounded_box(length / 2 - 0.001, thick / 2 + 0.006, base_h / 2, 0.003, segments=2, corner=0.004)
    vis.add("paint", shade(rgba, 0.62), skirting, (0, 0, base_h / 2))
    cap = M.rounded_box(length / 2 + 0.004, thick / 2 + 0.008, 0.012, 0.005, segments=2, corner=0.006)
    vis.add("stone", shade(rgba, 0.88), cap, (0, 0, height - 0.006))


def _hedge(vis: Placed, size, rgba, name) -> None:
    """A clipped hedge: a leafy block with gentle bulges, a little fuller than its collision wall."""
    length, thick, height = size
    hx, hy, hz = length / 2, thick / 2 + 0.018, height / 2 + 0.008
    block = M.dense_rounded_box(hx, hy, hz, min(0.04, hy * 0.7), step=0.05)
    leafy_block = M.noise_displaced(block, seed_of(name), amplitude=min(0.016, hy * 0.25), wavelength=0.35)
    vis.add("foliage", rgba, leafy_block.faceted(), (0, 0, hz - 0.01))


def _pillar(vis: Placed, size, rgba, name) -> None:
    d, h = size
    r = d / 2
    plinth = min(0.08, h * 0.05)
    prof = [
        (0.0, 0.0),
        (r * 1.12, 0.0),
        (r * 1.12, plinth * 0.7),
        (r * 1.04, plinth),
        (r, plinth * 1.3),
        (r * 0.97, h * 0.55),
        (r, h - plinth * 1.3),
        (r * 1.04, h - plinth),
        (r * 1.1, h - plinth * 0.7),
        (r * 1.1, h),
        (0.0, h),
    ]
    vis.add("stone" if not woody(rgba) else "wood", rgba, M.lathe(prof, 40, sharp=(1, 2, 3, 7, 8, 9)))


def _door_frame(vis: Placed, size, rgba, name) -> None:
    width, depth, height = size
    post = min(0.08, width / 4)
    lintel = M.rounded_box(width / 2, depth / 2, post / 2, 0.006, segments=2, corner=0.005)
    side = M.rounded_box(post / 2, depth / 2, (height - post) / 2, 0.006, segments=2, corner=0.005)
    parts_ = [lintel.moved((0, 0, height - post / 2))]
    parts_ += [side.moved((sx * (width / 2 - post / 2), 0, (height - post) / 2)) for sx in (1, -1)]
    vis.add("wood" if woody(rgba) else "paint", rgba, M.merge(parts_))
    trim = []
    for sy in (1, -1):  # a fine molding around the opening on both faces
        trim.append(
            M.rounded_box(width / 2 - post + 0.012, 0.004, 0.012, 0.003, segments=2).moved(
                (0, sy * (depth / 2 + 0.003), height - post - 0.006)
            )
        )
        for sx in (1, -1):
            x = sx * (width / 2 - post - 0.006)
            trim.append(
                M.rounded_box(0.012, 0.004, (height - post) / 2, 0.003, segments=2).moved(
                    (x, sy * (depth / 2 + 0.003), (height - post) / 2)
                )
            )
    vis.add("wood" if woody(rgba) else "paint", shade(rgba, 0.85), M.merge(trim))


def _counter(vis: Placed, size, rgba, name) -> None:
    from robotforge.design.props import STONE_TOP

    length, depth, height = size
    top, inset = 0.04, 0.03
    body_h = height - top
    slab = M.rounded_box(length / 2, depth / 2, top / 2, 0.008, segments=3, corner=0.01)
    vis.add("stone", STONE_TOP, slab, (0, 0, height - top / 2))
    cabinet = M.rounded_box(length / 2 - 0.01, depth / 2 - inset / 2, body_h / 2, 0.004, segments=2, corner=0.004)
    vis.add("paint", rgba, cabinet, (0, inset / 2, body_h / 2))
    # Doors (about 50 cm each) with bar handles, over a dark toe kick.
    kick = 0.08
    n = max(1, round(length / 0.5))
    w = (length - 0.02) / n
    doors, handles = [], []
    front = -depth / 2 + inset
    for i in range(n):
        x = -length / 2 + 0.01 + w * (i + 0.5)
        doors.append(
            M.rounded_box(w / 2 - 0.004, 0.006, (body_h - kick) / 2 - 0.012, 0.004, segments=2, corner=0.006).moved(
                (x, front - 0.004, kick + (body_h - kick) / 2)
            )
        )
        hx = x + (w / 2 - 0.05) * (1 if i % 2 == 0 else -1)
        handles.append(M.rounded_box(0.006, 0.006, 0.07, 0.004, segments=2).moved((hx, front - 0.016, body_h - 0.12)))
    vis.add("paint", shade(rgba, 1.08), M.merge(doors))
    vis.add("metal", P.METAL, M.merge(handles))
    toe = M.box(length / 2 - 0.02, 0.004, kick / 2 - 0.002).moved((0, front + 0.02, kick / 2))
    vis.add("dark", IRON, toe)


def _pallet(vis: Placed, size, rgba, name) -> None:
    length, width, height = size
    deck = 0.022
    boards = 5
    board_w = min(0.1, width / (boards * 1.3))
    top = []
    for i in range(boards):
        y = -width / 2 + board_w / 2 + i * (width - board_w) / (boards - 1)
        top.append(M.box(length / 2, board_w / 2, deck / 2, chamfer=0.003).moved((0, y, height - deck / 2)))
    vis.add("wood", rgba, M.merge(top))
    runner = min(0.1, width / 6)
    runners = []
    for y in (-width / 2 + runner / 2, 0.0, width / 2 - runner / 2):
        runners.append(
            M.box(length / 2, runner / 2, (height - deck) / 2, chamfer=0.004).moved((0, y, (height - deck) / 2))
        )
    vis.add("wood", shade(rgba, 0.8), M.merge(runners))


# ---- loose objects (plain shapes, styled by name) -----------------------------------------------


def _block(vis: Placed, size, rgba, name) -> None:
    hx, hy, hz = (s / 2 for s in size)
    r = min(0.006, min(hx, hy, hz) * 0.18)
    vis.add("shell", rgba, M.rounded_box(hx, hy, hz, r, segments=3))


def _crate(vis: Placed, size, rgba, name) -> None:
    """A slatted wooden crate: planks on every side over a darker core."""
    hx, hy, hz = (s / 2 for s in size)
    vis.add("wood", shade(rgba, 0.62), M.box(hx - 0.006, hy - 0.006, hz - 0.006))
    planks = []
    t = 0.006
    rows = max(2, round(2 * hz / 0.1))
    gap = 0.008
    ph = (2 * hz - (rows - 1) * gap) / rows
    for i in range(rows):
        z = -hz + ph / 2 + i * (ph + gap)
        for sy in (1, -1):
            planks.append(M.box(hx - t, t / 2, ph / 2, chamfer=0.0015).moved((0, sy * (hy - t / 2), z)))
        for sx in (1, -1):
            planks.append(M.box(t / 2, hy - t, ph / 2, chamfer=0.0015).moved((sx * (hx - t / 2), 0, z)))
    for sx in (1, -1):  # corner posts
        for sy in (1, -1):
            planks.append(M.box(t, t, hz, chamfer=0.0015).moved((sx * (hx - t), sy * (hy - t), 0)))
    cols = max(2, round(2 * hy / 0.1))
    pw = (2 * hy - (cols - 1) * gap) / cols
    for j in range(cols):
        y = -hy + pw / 2 + j * (pw + gap)
        for sz in (1, -1):
            planks.append(M.box(hx, pw / 2, t / 2, chamfer=0.0015).moved((0, y, sz * (hz - t / 2))))
    vis.add("wood", rgba, M.merge(planks))


def _fridge(vis: Placed, size, rgba, name) -> None:
    hx, hy, hz = (s / 2 for s in size)
    vis.add("paint", rgba, M.rounded_box(hx, hy, hz, 0.02, segments=3, corner=0.025))
    # The door split (freezer on top) as a fine dark line on the front (-y) face, and a dark plinth.
    split = M.box(hx * 0.96, 0.002, 0.003).moved((0, -hy, hz * 0.35))
    plinth = M.box(hx * 0.9, 0.002, 0.03).moved((0, -hy, -hz + 0.035))
    vis.add("dark", IRON, M.merge([split, plinth]))


def _handle(vis: Placed, size, rgba, name) -> None:
    hx, hy, hz = (s / 2 for s in size)
    vis.add("metal", P.METAL, M.rounded_box(hx, hy, hz, min(hx, hy) * 0.9, segments=3))


def _ball(vis: Placed, size, rgba, name) -> None:
    r = size[0] / 2
    vis.add("shell", rgba, M.sphere(r, rings=16, segments=32))
    seam = M.torus(r * 0.995, r * 0.03, segments=48, sides=6)
    vis.add("shell", shade(rgba, 0.6), M.merge([seam, seam.moved(rot=M.rot_x(math.pi / 2))]))


def _apple(vis: Placed, size, rgba, name) -> None:
    r = size[0] / 2
    prof = []
    for t in np.linspace(0, math.pi, 17):
        rad = r * math.sin(t) * (1.0 + 0.06 * math.sin(t) ** 2)
        z = -r * math.cos(t) * 0.98
        dip = 0.16 * r * math.exp(-(((t - math.pi) / 0.35) ** 2)) + 0.08 * r * math.exp(-((t / 0.35) ** 2))
        prof.append((rad, z - dip if t > math.pi / 2 else z + dip))
    vis.add("shell", rgba, M.lathe(prof, 28))
    stem = M.tube(
        np.array([[0, 0, r * 0.7], [0.0, 0.004, r * 1.0], [0.004, 0.008, r * 1.15]]), [0.0028, 0.0024, 0.002], 6
    )
    vis.add("bark", BARK, stem)
    leaf = M.blob((r * 0.32, r * 0.14, r * 0.03), 7, amplitude=0.0, rings=6, segments=10)
    vis.add("foliage", [0.3, 0.6, 0.25, 1.0], leaf, (r * 0.22, 0.006, r * 1.0))


def _citrus(vis: Placed, size, rgba, name) -> None:
    r = size[0] / 2
    vis.add(
        "shell",
        rgba,
        M.blob((r, r, r * 0.96), seed_of(name), amplitude=0.02, lumps=30, sharpness=12, rings=14, segments=24),
    )


def _plain_sphere(vis: Placed, size, rgba, name) -> None:
    vis.add("shell", rgba, M.sphere(size[0] / 2, rings=16, segments=32))


def _cup(vis: Placed, size, rgba, name) -> None:
    """A tumbler: open top with a visible inside, slightly narrower at the foot."""
    r, h = size[0] / 2, size[1]
    t = min(0.003, r * 0.12)
    prof = [
        (r * 0.86, -h / 2),
        (r * 0.92, -h / 2 + 0.002),
        (r, h / 2 - 0.0015),
        (r - t * 0.5, h / 2),
        (r - t, h / 2 - 0.001),
        (r * 0.84 - t, -h / 2 + 0.012),
        (r * 0.75 - t, -h / 2 + 0.008),
        (0.0, -h / 2 + 0.008),
    ]
    rings = [M.at_z(M.circle(rr, 32), z) for rr, z in [(0.0, -h / 2), *prof]]
    vis.add("ceramic", rgba, M.loft(rings, sharp=(1, 7)))


def _plate(vis: Placed, size, rgba, name) -> None:
    r, h = size[0] / 2, size[1]
    prof = [
        (0.0, -h / 2),
        (r * 0.55, -h / 2),
        (r * 0.62, -h / 2 + h * 0.2),
        (r * 0.62, -h / 2 + h * 0.45),
        (r, h / 2 - h * 0.15),
        (r * 0.98, h / 2),
        (r * 0.93, h / 2 - h * 0.1),
        (r * 0.6, -h / 2 + h * 0.62),
        (0.0, -h / 2 + h * 0.62),
    ]
    vis.add("ceramic", rgba, M.lathe(prof, 48, sharp=(1, 7)))


def _jar(vis: Placed, size, rgba, name) -> None:
    r, h = size[0] / 2, size[1]
    lid_h = min(0.018, h * 0.18)
    jar = M.lathe(
        [
            (0.0, -h / 2),
            (r * 0.92, -h / 2),
            (r, -h / 2 + 0.006),
            (r, h / 2 - lid_h - 0.006),
            (r * 0.82, h / 2 - lid_h),
            (0.0, h / 2 - lid_h),
        ],
        32,
        sharp=(1,),
    )
    vis.add("ceramic", rgba, jar)
    lid = M.cylinder(r * 0.9, lid_h, segments=32, bevel=lid_h * 0.3, z0=h / 2 - lid_h)
    knob = M.sphere(r * 0.2, rings=6, segments=12).moved((0, 0, h / 2 + r * 0.05))
    vis.add("wood", [0.7, 0.55, 0.38, 1.0], M.merge([lid, knob]))


def _can(vis: Placed, size, rgba, name) -> None:
    r, h = size[0] / 2, size[1]
    neck = min(0.006, h * 0.08)
    prof = [
        (0.0, -h / 2 + 0.002),
        (r * 0.8, -h / 2),
        (r, -h / 2 + neck),
        (r, h / 2 - neck),
        (r * 0.86, h / 2),
        (0.0, h / 2),
    ]
    vis.add("shell", rgba, M.lathe(prof, 32))
    rims = [
        M.torus(r * 0.86, 0.0012, 32, 6).moved((0, 0, h / 2)),
        M.torus(r * 0.8, 0.001, 32, 6).moved((0, 0, -h / 2 + 0.001)),
    ]
    tab = M.rounded_box(r * 0.25, r * 0.12, 0.0006, 0.0005, segments=1).moved((r * 0.3, 0, h / 2 + 0.0006))
    vis.add("metal", P.METAL, M.merge([*rims, tab]))


def _burner(vis: Placed, size, rgba, name) -> None:
    r, h = size[0] / 2, size[1]
    plate = M.cylinder(r, h, segments=40, bevel=h * 0.4, z0=-h / 2)
    vis.add("dark", IRON, plate)
    rings = [M.torus(r * k, h * 0.18, 40, 6).moved((0, 0, h / 2)) for k in (0.35, 0.6, 0.85)]
    vis.add("metal", [0.4, 0.4, 0.42, 1.0], M.merge(rings))


def _plain_cylinder(vis: Placed, size, rgba, name) -> None:
    r, h = size[0] / 2, size[1]
    vis.add("shell", rgba, M.cylinder(r, h, segments=32, bevel=min(0.004, r * 0.15, h * 0.15)))


_LOOKS: dict[str, Look] = {
    "tree": _tree,
    "bench": _bench,
    "rock": _rock,
    "table": _table,
    "bin": _bin,
    "shelf": _shelf,
    "wall": _wall,
    "pillar": _pillar,
    "door_frame": _door_frame,
    "counter": _counter,
    "pallet": _pallet,
}
_NAMED: dict[str, tuple[tuple[tuple[str, ...], Look], ...]] = {
    "box": ((("crate",), _crate), (("fridge",), _fridge), (("handle",), _handle)),
    "sphere": ((("apple",), _apple), (("orange", "fruit", "lemon"), _citrus), (("ball",), _ball)),
    "cylinder": (
        (("cup", "mug", "glass"), _cup),
        (("plate", "dish"), _plate),
        (("jar",), _jar),
        (("can", "tin"), _can),
        (("burner", "stove"), _burner),
    ),
    "marker": ((("pond", "water", "lake", "pool"), _pond),),
}
_PLAIN: dict[str, Look] = {"box": _block, "sphere": _plain_sphere, "cylinder": _plain_cylinder}
