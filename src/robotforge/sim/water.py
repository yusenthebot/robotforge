"""Water: buoyancy, drag and marine thrusters below a water surface (a lake, a harbor, a pool).

MuJoCo's own fluid model (option density and viscosity) is global: the whole world, air included, would be under
water. Here the water has a surface, a still level with gentle waves on it, and only what is below the surface
feels it. Every colliding shape of a body that moves is cut, once, into a few dozen small cells in its own frame;
each physics step all cells of all bodies are evaluated at once with numpy:

- buoyancy: the share of each cell below the surface (exact for spheres and ellipsoids at any tilt, for boxes that
  float level and for upright or lying cylinders; close for everything in between) times its volume, pushed up at
  the centroid of its wet part, so a boat that heels is pushed back upright by the shift of its wet volume;
- drag: on the wet share of each shape, per axis of the shape (a hull slides forward easily and resists going
  sideways, like a keel), linear (calm water, wave making) plus quadratic (turbulence) in the velocity relative to
  the water (current plus the orbital motion of the waves), and the same for turning;
- thrusters: site actuators whose site is named "...thruster..." push like any MuJoCo site actuator (a rotor),
  but a propeller out of the water pushes nothing, so their thrust fades as they leave it.

The forces are worked out from where things are in this step: the session runs mj_step1 (which poses the model),
then `apply`, then mj_step2. Worked out before mj_step, from the pose of the step before, they lagged a step, and
a light plate slapping the water gained energy from the lag until it flew off. They reach MuJoCo as one wrench per
body in xfrc_applied, so MuJoCo maps them onto the joints in C (a jointed robot costs no more than a crate), and
the session puts back what was there after mj_step2: qfrc_applied keeps meaning a hand in the viewer to the
controllers that read it, and pushes written into xfrc_applied stay as they were written.

The water a world has rides along in the compiled model as a custom numeric field (`robotforge_water`), so every
way a model is loaded (a world, a saved MJCF, the viewer, a lesson's copy) knows whether it has water.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

G = mujoco.mjtGeom
GRAVITY = 9.81
NUMERIC = "robotforge_water"  # the custom numeric field that carries a model's water (see encode / decode)
SURFACE_GEOM = "water:surface"  # the visual surface for MuJoCo's renderer (the browser viewer draws its own)
ENCODING = 1  # version of the numeric layout below
CELLS_PER_GEOM = 32  # aim for about this many box cells per shape (see _box_cells)
MAX_CELLS_ALONG = 6
# Drag: linear coefficient in m/s. Floating things lose their heave, roll and pitch to the waves they make, so a
# 40 cm crate settles in a few bobs (damping ratio about 0.2); going along or turning flat on the water makes far
# fewer waves, so moving horizontally (and turning about the vertical) gets HORIZONTAL_SHARE of it. Quadratic drag
# coefficients by shape.
LINEAR_DRAG = 0.5
HORIZONTAL_SHARE = 0.2
DRAG_COEFFICIENT = {
    int(G.mjGEOM_BOX): 1.05,
    int(G.mjGEOM_MESH): 1.05,
    int(G.mjGEOM_SPHERE): 0.47,
    int(G.mjGEOM_ELLIPSOID): 0.47,
    int(G.mjGEOM_CYLINDER): 0.82,
    int(G.mjGEOM_CAPSULE): 0.6,
}
# A drag force integrated explicitly can overshoot on a very light shape (a thin board): it is capped to take at
# most this share of the body's velocity per step.
DRAG_CAP = 0.5
# Wave components as (share of the amplitude, share of the wavelength, degrees off the main heading): one swell
# looks like corrugated iron; three crossing ones look like a lake on a breezy day.
WAVE_MIX = ((1.0, 1.0, 0.0), (0.45, 0.61, 37.0), (0.25, 0.37, -58.0))
KIND_BOX, KIND_ELLIPSOID, KIND_DISC = 0, 1, 2
DRY = -1e6  # m: the "surface" where a world's water does not reach (a finite number keeps the arithmetic plain)


@dataclass(frozen=True)
class Water:
    """The water of a world: a still level (m above the floor, which is its bottom where it is lower), its density,
    waves (amplitude m, wavelength m, the heading they travel, degrees), a current (m/s) and where there is water
    (x0, y0, x1, y1; None for everywhere)."""

    level: float
    density: float = 1000.0
    waves: float = 0.0
    wavelength: float = 4.0
    heading: float = 30.0
    current: tuple[float, float] = (0.0, 0.0)
    area: tuple[float, float, float, float] | None = None

    def components(self) -> list[tuple[float, float, float, float, float]]:
        """The waves as (amplitude, kx, ky, omega, phase): deep-water waves, omega^2 = g k."""
        if self.waves <= 0:
            return []
        out = []
        for i, (share, length_share, turn) in enumerate(WAVE_MIX):
            k = 2 * math.pi / (self.wavelength * length_share)
            a = math.radians(self.heading + turn)
            out.append((self.waves * share, k * math.cos(a), k * math.sin(a), math.sqrt(GRAVITY * k), 1.7 * i))
        return out

    def describe(self) -> dict:
        """For the viewer, which animates the same waves the physics feels."""
        return {
            "level": self.level,
            "density": self.density,
            "area": list(self.area) if self.area is not None else None,
            "waves": [[round(v, 6) for v in c] for c in self.components()],
            "current": list(self.current),
        }


def encode(spec: mujoco.MjSpec, water: Water) -> None:
    """Store `water` in the model, as a custom numeric field."""
    area = water.area if water.area is not None else (0.0, 0.0, 0.0, 0.0)
    data = [
        ENCODING,
        water.level,
        water.density,
        water.waves,
        water.wavelength,
        water.heading,
        *water.current,
        float(water.area is not None),
        *area,
    ]
    spec.add_numeric(name=NUMERIC, data=[float(v) for v in data])


def add_surface(spec: mujoco.MjSpec, water: Water, model: mujoco.MjModel) -> None:
    """A see-through sheet at the water level for MuJoCo's own renderer (observe's pictures); the browser viewer
    draws its own. `model` is the world compiled without it: open water gets a sheet twice the scene's size round
    its middle, and the model's statistics are pinned to that model's, so the sheet moves neither the cameras that
    frame the scene nor the near clipping plane (a 120 m sheet had pushed it out of a boat's own camera)."""
    if water.area is not None:
        x0, y0, x1, y1 = water.area
        center, half = [(x0 + x1) / 2, (y0 + y1) / 2], [(x1 - x0) / 2, (y1 - y0) / 2]
    else:
        reach = max(20.0, 2.0 * float(model.stat.extent))
        center, half = [float(model.stat.center[0]), float(model.stat.center[1])], [reach, reach]
    spec.stat.extent = float(model.stat.extent)
    spec.stat.center = [float(v) for v in model.stat.center]
    # Thin, not a plane: a plane is the floor to the viewer and to the navigation map. It collides with nothing,
    # and its material is fully transparent to rays, so rangefinders and lidars look through it.
    if not any(m.name == "water_rays" for m in spec.materials):
        spec.add_material(name="water_rays", rgba=[1.0, 1.0, 1.0, 0.0], specular=0.9, shininess=0.9, reflectance=0.0)
    spec.worldbody.add_geom(
        name=SURFACE_GEOM,
        type=G.mjGEOM_BOX,
        size=[half[0], half[1], 0.0005],
        pos=[center[0], center[1], water.level - 0.0005],
        rgba=[0.2, 0.45, 0.6, 0.45],
        material="water_rays",
        contype=0,
        conaffinity=0,
        group=2,
        mass=0.0,
    )


def decode(model: mujoco.MjModel) -> Water | None:
    """The water stored in `model`, or None for a model without water."""
    n = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, NUMERIC)
    if n < 0:
        return None
    adr, size = int(model.numeric_adr[n]), int(model.numeric_size[n])
    v = [float(x) for x in model.numeric_data[adr : adr + size]]
    if len(v) < 13 or int(v[0]) != ENCODING:
        return None
    area = (v[9], v[10], v[11], v[12]) if v[8] > 0.5 else None
    return Water(v[1], v[2], v[3], v[4], v[5], (v[6], v[7]), area)


def surface_geom(model: mujoco.MjModel) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SURFACE_GEOM)


# ---- cutting shapes into cells -------------------------------------------------------------------------------


@dataclass
class _Cells:
    geom: list[int] = field(default_factory=list)
    center: list[np.ndarray] = field(default_factory=list)  # in the geom frame
    half: list[np.ndarray] = field(default_factory=list)
    volume: list[float] = field(default_factory=list)
    kind: list[int] = field(default_factory=list)

    def add(self, g: int, center, half, volume: float, kind: int) -> None:
        self.geom.append(g)
        self.center.append(np.asarray(center, float))
        self.half.append(np.asarray(half, float))
        self.volume.append(float(volume))
        self.kind.append(kind)


def _box_cells(cells: _Cells, g: int, center: np.ndarray, half: np.ndarray) -> None:
    """A grid of small boxes, roughly cubes, about CELLS_PER_GEOM of them: level boxes float exactly (each layer is
    a slab, so a thin plate needs only one), and five cells across a hull give its righting moment within a few
    percent."""
    dims = 2 * half
    volume = float(np.prod(dims))
    side = (volume / CELLS_PER_GEOM) ** (1 / 3)
    counts = np.clip(np.round(dims / max(side, 1e-9)), 1, MAX_CELLS_ALONG).astype(int)
    step = dims / counts
    axes = [np.linspace(-h + s / 2, h - s / 2, n) for h, s, n in zip(half, step, counts, strict=True)]
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                cells.add(g, center + [x, y, z], step / 2, volume / counts.prod(), KIND_BOX)


def _disc_cells(cells: _Cells, g: int, r: float, h: float, center: np.ndarray | None = None) -> None:
    """A cylinder (radius r, half length h along its own z) as slabs across its axis."""
    center = np.zeros(3) if center is None else center
    n = int(np.clip(math.ceil(2 * h / max(r, 1e-9)), 2, 8))
    t = h / n
    for i in range(n):
        z = -h + t * (2 * i + 1)
        cells.add(g, center + [0, 0, z], [r, r, t], math.pi * r * r * 2 * t, KIND_DISC)


def _cut(model: mujoco.MjModel, g: int, cells: _Cells) -> bool:
    """Add the cells of geom `g`; False for shapes water ignores (planes, heightfields)."""
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    if kind == G.mjGEOM_BOX:
        _box_cells(cells, g, np.zeros(3), np.asarray(size, float))
    elif kind == G.mjGEOM_MESH:  # by its bounding box, in the mesh's own frame
        aabb = model.geom_aabb[g]
        _box_cells(cells, g, np.asarray(aabb[:3], float), np.asarray(aabb[3:], float))
    elif kind == G.mjGEOM_SPHERE:
        r = float(size[0])
        cells.add(g, [0, 0, 0], [r, r, r], 4 / 3 * math.pi * r**3, KIND_ELLIPSOID)
    elif kind == G.mjGEOM_ELLIPSOID:
        cells.add(g, [0, 0, 0], size[:3], 4 / 3 * math.pi * float(np.prod(size[:3])), KIND_ELLIPSOID)
    elif kind == G.mjGEOM_CYLINDER:
        _disc_cells(cells, g, float(size[0]), float(size[1]))
    elif kind == G.mjGEOM_CAPSULE:
        r, h = float(size[0]), float(size[1])
        _disc_cells(cells, g, r, h)
        for end in (1, -1):  # the two round ends: half a ball each (exact while the capsule lies level)
            cells.add(g, [0, 0, end * h], [r, r, r], 2 / 3 * math.pi * r**3, KIND_ELLIPSOID)
    else:
        return False
    return True


def _axis_areas(model: mujoco.MjModel, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Projected area (m^2) seen along each of the geom's own axes, and its half extents along them."""
    kind, size = int(model.geom_type[g]), model.geom_size[g]
    if kind in (G.mjGEOM_SPHERE, G.mjGEOM_ELLIPSOID):
        r = np.full(3, size[0]) if kind == G.mjGEOM_SPHERE else np.asarray(size[:3], float)
        return math.pi * np.array([r[1] * r[2], r[0] * r[2], r[0] * r[1]]), r
    if kind in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
        r, h = float(size[0]), float(size[1])
        cap = math.pi * r * r if kind == G.mjGEOM_CAPSULE else 0.0
        side = 4 * r * h + cap
        reach = h + (r if kind == G.mjGEOM_CAPSULE else 0.0)
        return np.array([side, side, math.pi * r * r]), np.array([r, r, reach])
    half = np.asarray(size[:3] if kind == G.mjGEOM_BOX else model.geom_aabb[g][3:], float)
    return 4 * np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]]), half


# ---- the occupancy of one cell -------------------------------------------------------------------------------


def _cap(t: np.ndarray) -> np.ndarray:
    """Share of a ball below a plane t radii above its center (t clipped to -1..1)."""
    return (1 + t) ** 2 * (2 - t) / 4


def _cap_centroid(t: np.ndarray) -> np.ndarray:
    """Height of the centroid of that wet part above the center, in radii (-1 at the bottom, 0 when all wet)."""
    return -3 * (1 - t) ** 2 / (4 * (2 - t))


def _segment(t: np.ndarray) -> np.ndarray:
    """Share of a disc below a line t radii above its center."""
    return (math.pi - np.arccos(t) + t * np.sqrt(np.maximum(1 - t * t, 0.0))) / math.pi


def _linear(t: np.ndarray) -> np.ndarray:
    return np.clip((t + 1) / 2, 0.0, 1.0)


# ---- the model in a running simulation -----------------------------------------------------------------------


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product: np.cross costs more in argument handling than in arithmetic on a few rows."""
    out = np.empty((max(len(a), len(b)), 3))
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


class Hydro:
    """The water forces for one compiled model; `apply` adds them to xfrc_applied in each step.

    The step is a fixed number of whole-array numpy operations whatever the number of bodies (sums go through
    small dense matrices, cells are sorted by kind so each kind's formula runs on one slice), so a harbor of
    twenty floating things costs about what one does."""

    def __init__(self, model: mujoco.MjModel, water: Water) -> None:
        self.model, self.water = model, water
        self.rho = water.density
        waves = np.array(water.components(), float).reshape(-1, 5)
        self._wave_k = waves[:, 1:3].T.copy()  # (2, n): phase = x kx + y ky - omega t + phase0
        self._wave_amp, self._wave_omega, self._wave_phase = waves[:, 0], waves[:, 3], waves[:, 4]
        self._wave_number = np.hypot(waves[:, 1], waves[:, 2])
        cells = _Cells()
        geoms = []
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            if body == 0 or int(model.body_weldid[body]) == 0:
                continue  # welded to the world: water cannot move it
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue  # visual only: the colliding shapes under it carry the volume
            if _cut(model, g, cells):
                geoms.append(g)
        self.geoms = np.array(geoms, int)
        index = {g: i for i, g in enumerate(geoms)}
        kind = np.array(cells.kind, int)
        order = np.argsort(kind, kind="stable")
        self.cell_geom = np.array([index[g] for g in cells.geom], int)[order] if cells.geom else np.zeros(0, int)
        self.cell_center = np.array(cells.center, float).reshape(-1, 3)[order]
        self.cell_half = np.array(cells.half, float).reshape(-1, 3)[order]
        self.cell_volume = np.array(cells.volume, float)[order]
        self.cell_kind = kind[order]
        self._cell_model_geom = np.array(geoms, int)[self.cell_geom] if geoms else np.zeros(0, int)
        self._blocks = [(k, np.nonzero(self.cell_kind == k)[0]) for k in (KIND_BOX, KIND_ELLIPSOID, KIND_DISC)]
        self._blocks = [(k, slice(int(ix[0]), int(ix[-1]) + 1)) for k, ix in self._blocks if len(ix)]
        n_geoms, n_cells = len(geoms), len(self.cell_geom)
        # Sums over each shape's cells: the cells of one shape lie in at most two runs (a capsule's slabs and its
        # ends sit in different kinds' slices), summed with one reduceat and then a small run-to-shape matrix.
        self._run_start = np.flatnonzero(np.r_[True, self.cell_geom[1:] != self.cell_geom[:-1]]) if n_cells else None
        runs = self.cell_geom[self._run_start] if n_cells else np.zeros(0, int)
        self._runs_to_geoms = np.zeros((n_geoms, len(runs)))
        self._runs_to_geoms[runs, np.arange(len(runs))] = 1.0
        self.geom_volume = self.per_geom(self.cell_volume)
        bodies = model.geom_bodyid[self.geoms] if n_geoms else np.zeros(0, int)
        self.bodies, self.geom_body = np.unique(bodies, return_inverse=True)  # geom -> index into self.bodies
        self._geoms_to_bodies = np.zeros((len(self.bodies), n_geoms))
        self._geoms_to_bodies[self.geom_body, np.arange(n_geoms)] = 1.0
        self.body_volume = self._geoms_to_bodies @ self.geom_volume
        areas = [_axis_areas(model, g) for g in geoms]
        self.area = np.array([a for a, _ in areas], float).reshape(-1, 3)
        reach = np.array([r for _, r in areas], float).reshape(-1, 3)
        # Turning drag about each axis i: the faces along the other two axes sweep through the water.
        j, k = [1, 2, 0], [2, 0, 1]
        self.turn_linear = (self.area[:, j] * reach[:, k] ** 2 + self.area[:, k] * reach[:, j] ** 2) / 3
        self.turn_quadratic = (self.area[:, j] * reach[:, k] ** 3 + self.area[:, k] * reach[:, j] ** 3) / 4
        self.drag = np.array([DRAG_COEFFICIENT.get(int(model.geom_type[g]), 1.0) for g in geoms], float)[:, None]
        self._geom_bodyid = model.geom_bodyid[self.geoms]
        self._geom_rootid = model.body_rootid[self._geom_bodyid]
        # For the step that has nothing in the water (a robot on the shore of a lake): each shape's bounding sphere
        # and the highest crest the waves reach.
        self._geom_rbound = model.geom_rbound[self.geoms] if n_geoms else np.zeros(0)
        self._crest = water.level + float(np.abs(self._wave_amp).sum())
        dt = model.opt.timestep
        # The cap on explicit drag (DRAG_CAP), per geom from its body's mass and smallest principal inertia.
        self._force_cap = (DRAG_CAP * np.maximum(model.body_mass[self._geom_bodyid], 1e-6) / dt)[:, None]
        inertia = np.maximum(model.body_inertia[self._geom_bodyid].min(axis=1), 1e-9) if n_geoms else np.zeros(0)
        self._turn_cap = (DRAG_CAP * inertia / dt)[:, None]
        self._thrusters()
        self.ever_wet = np.zeros(len(self.bodies), bool)
        self.last_wet: np.ndarray | None = None  # per water body, m^3 below the surface at the last step

    # ---- set up -------------------------------------------------------------------------------------------

    def _thrusters(self) -> None:
        """Marine thrusters: site actuators on a site named "...thruster...", with the propeller's radius."""
        m = self.model
        acts, sites = [], []
        for a in range(m.nu):
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_SITE:
                continue
            site = int(m.actuator_trnid[a, 0])
            if "thruster" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, site) or "").lower():
                acts.append(a)
                sites.append(site)
        self.thruster_actuators = np.array(acts, int)
        self.thruster_sites = np.array(sites, int)
        self._prop_radius = np.maximum(m.site_size[self.thruster_sites, 0], 0.01) if sites else np.zeros(0)

    def per_geom(self, values: np.ndarray) -> np.ndarray:
        """Sums of per-cell `values` (cells, ...) over each shape's cells."""
        if self._run_start is None:
            return np.zeros((0, *np.shape(values)[1:]))
        return self._runs_to_geoms @ np.add.reduceat(values, self._run_start, axis=0)

    # ---- the surface ----------------------------------------------------------------------------------------

    def surface(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """Height of the water surface at points (x, y) at time t; DRY (far below anything) where there is none."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        h = np.full(x.shape, self.water.level)
        if len(self._wave_amp):
            phase = np.stack([x, y], axis=-1) @ self._wave_k
            h = h + np.sin(phase - (self._wave_omega * t - self._wave_phase)) @ self._wave_amp
        if self.water.area is not None:
            x0, y0, x1, y1 = self.water.area
            h = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1), h, DRY)
        return h

    def flow(self, p: np.ndarray, t: float) -> np.ndarray:
        """Water velocity at points p (n, 3): the current plus the orbital motion of the waves, which fades with
        depth as exp(k z)."""
        v = np.empty_like(p)
        v[:, 0], v[:, 1], v[:, 2] = self.water.current[0], self.water.current[1], 0.0
        if not len(self._wave_amp):
            return v
        theta = p[:, :2] @ self._wave_k - self._wave_omega * t + self._wave_phase  # (n, waves)
        depth = np.minimum(p[:, 2] - self.water.level, 0.0)[:, None]
        fade = self._wave_amp * self._wave_omega * np.exp(self._wave_number * depth)
        along = fade * np.sin(theta) / self._wave_number
        v[:, :2] += along @ self._wave_k.T
        v[:, 2] -= (fade * np.cos(theta)).sum(axis=1)
        return v

    # ---- each step ------------------------------------------------------------------------------------------

    def cells(self, data: mujoco.MjData, lift: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Each cell's wet volume (m^3) and the world centroid of that wet part. `lift` (per water body, m) raises
        bodies for the floating height search at spawn (see float_height)."""
        rot = data.geom_xmat[self._cell_model_geom].reshape(-1, 3, 3)
        center = data.geom_xpos[self._cell_model_geom] + np.einsum("nij,nj->ni", rot, self.cell_center)
        if lift is not None:
            center[:, 2] += lift[self.geom_body[self.cell_geom]]
        up = rot[:, 2, :]  # the world z of each local axis
        ue = up * self.cell_half
        reach = np.empty(len(center))
        for kind, s in self._blocks:
            if kind == KIND_BOX:
                reach[s] = np.abs(ue[s]).sum(axis=1)
            elif kind == KIND_ELLIPSOID:
                reach[s] = np.sqrt((ue[s] * ue[s]).sum(axis=1))
            else:
                reach[s] = np.abs(ue[s, 2]) + self.cell_half[s, 0] * np.sqrt(np.maximum(1 - up[s, 2] ** 2, 0.0))
        np.maximum(reach, 1e-9, out=reach)
        top = self.surface(center[:, 0], center[:, 1], float(data.time))
        t = np.clip((top - center[:, 2]) / reach, -1.0, 1.0)
        share = np.empty_like(t)
        centroid = center
        for kind, s in self._blocks:
            ts = t[s]
            if kind == KIND_ELLIPSOID:
                share[s] = _cap(ts)
                # The wet part's centroid lies below the center along the conjugate of up (exact for a ball).
                conj = np.einsum("nij,nj->ni", rot[s], self.cell_half[s] * ue[s]) / reach[s, None]
                centroid[s] += _cap_centroid(ts)[:, None] * conj
                continue
            lin = np.clip((ts + 1) / 2, 0.0, 1.0)
            if kind == KIND_DISC:  # a leaning disc wets like a circle segment, an upright one like a slab
                lean = self.cell_half[s, 0] * np.sqrt(np.maximum(1 - up[s, 2] ** 2, 0.0)) / reach[s]
                lin = lean * _segment(ts) + (1 - lean) * lin
            share[s] = lin
            centroid[s, 2] -= reach[s] * (1 - lin)  # halfway between its bottom and the waterline
        return share * self.cell_volume, centroid

    def wrenches(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Force and moment about the world origin on every water body (world frame), (n, 3) each."""
        nb = len(self.bodies)
        if not len(self.geoms) or not (data.geom_xpos[self.geoms, 2] - self._geom_rbound < self._crest).any():
            self.last_wet = np.zeros(nb)  # all of it above the highest crest
            return np.zeros((nb, 3)), np.zeros((nb, 3))
        wet, centroid = self.cells(data)
        sums = self.per_geom(np.column_stack([wet, wet[:, None] * centroid]))  # (geoms, 4)
        geom_wet, moment = sums[:, 0], sums[:, 1:]
        weight = self.rho * GRAVITY
        per_geom = np.zeros((len(self.geoms), 7))
        per_geom[:, 2] = weight * geom_wet  # buoyancy, up
        per_geom[:, 3] = weight * moment[:, 1]  # its moment about the origin: p x (0, 0, f)
        per_geom[:, 4] = -weight * moment[:, 0]
        per_geom[:, 6] = geom_wet
        if geom_wet.any():
            where = moment / np.maximum(geom_wet, 1e-12)[:, None]
            share = geom_wet / np.maximum(self.geom_volume, 1e-12)
            f, tau = self._drag(data, share[:, None], where)
            per_geom[:, :3] += f
            per_geom[:, 3:6] += _cross(where, f) + tau
        per_body = self._geoms_to_bodies @ per_geom
        self.last_wet = per_body[:, 6]
        self.ever_wet |= self.last_wet > 0
        return per_body[:, :3], per_body[:, 3:6]

    def _drag(self, data: mujoco.MjData, share: np.ndarray, where: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linear and quadratic drag on every shape's wet share, per axis of the shape, in the velocity relative to
        the water: force, and torque about `where` (the centroid of its wet part)."""
        rot = data.geom_xmat[self.geoms].reshape(-1, 3, 3)
        cvel = data.cvel[self._geom_bodyid]  # [angular, linear at the subtree center of mass of the root]
        spin = cvel[:, :3]
        v = cvel[:, 3:] + _cross(spin, where - data.subtree_com[self._geom_rootid])
        rel = v - self.flow(where, float(data.time))
        local = np.einsum("nji,nj->ni", rot, rel)  # in the shape's own axes
        speed = np.sqrt((rel * rel).sum(axis=1))[:, None]
        up = np.abs(rot[:, 2, :])  # how vertical each of the shape's axes is
        linear = LINEAR_DRAG * (HORIZONTAL_SHARE + (1 - HORIZONTAL_SHARE) * up)
        coef = (self.rho * share) * self.area * (linear + 0.5 * self.drag * speed)
        f_local = -np.minimum(coef, self._force_cap) * local
        w_local = np.einsum("nji,nj->ni", rot, spin)
        turning = LINEAR_DRAG * (1 - (1 - HORIZONTAL_SHARE) * up)  # rolling and pitching make waves, yawing few
        turn = (self.rho * share) * (
            turning * self.turn_linear + 0.5 * self.drag * self.turn_quadratic * np.abs(w_local)
        )
        t_local = -np.minimum(turn, self._turn_cap) * w_local
        return np.einsum("nij,nj->ni", rot, f_local), np.einsum("nij,nj->ni", rot, t_local)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Add this step's water forces to data.xfrc_applied, as a force and a torque about each body's center of
        mass. `data` is posed for this step (mj_step1); the session takes the forces out again after mj_step2."""
        if len(self.bodies):
            force, moment = self.wrenches(data)
            torque = moment - _cross(data.xipos[self.bodies], force)  # about each body's center of mass
            data.xfrc_applied[self.bodies] += np.column_stack([force, torque])  # self.bodies has no repeats
        if len(self.thruster_actuators):
            self._thrust(model, data)

    def _thrust(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Take away the thrust of propellers out of the water, in proportion to how far out they are."""
        push = data.actuator_force[self.thruster_actuators]
        if not push.any():
            return
        p = data.site_xpos[self.thruster_sites]
        top = self.surface(p[:, 0], p[:, 1], float(data.time))
        wet = np.clip(((top - p[:, 2]) / self._prop_radius + 1) / 2, 0.0, 1.0)
        out = np.nonzero((wet < 1.0) & (push != 0))[0]
        if not len(out):
            return
        sites = self.thruster_sites[out]
        gear = model.actuator_gear[self.thruster_actuators[out]]
        rot = data.site_xmat[sites].reshape(-1, 3, 3)
        scale = (-(1 - wet[out]) * push[out])[:, None]
        f = scale * np.einsum("nij,nj->ni", rot, gear[:, :3])
        bodies = model.site_bodyid[sites]
        tau = scale * np.einsum("nij,nj->ni", rot, gear[:, 3:]) + _cross(p[out] - data.xipos[bodies], f)
        np.add.at(data.xfrc_applied, bodies, np.column_stack([f, tau]))  # two thrusters can share a body

    # ---- for pilots -----------------------------------------------------------------------------------------

    def resistance(
        self,
        data: mujoco.MjData,
        root: int,
        along: np.ndarray | None = None,
        about: np.ndarray | None = None,
        lift: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """How the water holds back the robot whose root body is `root` as it is now (its wet share), moving along
        the world direction `along` (force c1 v + c2 v |v|, N) or turning about the world axis `about` (torque,
        N m): what a pilot needs to know its top speed and to feed forward the thrust that drag takes. `lift` (per
        water body, m) moves bodies first, as in cells (an ROV asks with itself pushed under)."""
        mine = self.model.body_rootid[self._geom_bodyid] == root
        if not mine.any():
            return 0.0, 0.0
        wet, _ = self.cells(data, lift=lift)
        share = self.per_geom(wet) / np.maximum(self.geom_volume, 1e-12)
        rot = data.geom_xmat[self.geoms].reshape(-1, 3, 3)[mine]
        up = np.abs(rot[:, 2, :])
        weight = (self.rho * share[mine])[:, None]
        drag = self.drag[mine]
        if along is not None:
            l2 = np.einsum("nji,j->ni", rot, np.asarray(along, float)) ** 2
            linear = LINEAR_DRAG * (HORIZONTAL_SHARE + (1 - HORIZONTAL_SHARE) * up)
            area = self.area[mine]
            return float((weight * area * linear * l2).sum()), float((weight * 0.5 * drag * area * l2).sum())
        axis = np.einsum("nji,j->ni", rot, np.asarray(about, float))  # the turning axis in each shape's own frame
        turning = LINEAR_DRAG * (1 - (1 - HORIZONTAL_SHARE) * up)
        c1 = (weight * turning * self.turn_linear[mine] * axis * axis).sum()
        c2 = (weight * 0.5 * drag * self.turn_quadratic[mine] * np.abs(axis) ** 3).sum()
        return float(c1), float(c2)

    def buoyancy(self, root: int) -> float:
        """N of lift the robot with root body `root` gets fully under water."""
        mine = self.model.body_rootid[self.bodies] == root
        return float(self.body_volume[mine].sum()) * self.rho * GRAVITY

    # ---- for agents -----------------------------------------------------------------------------------------

    def report(self, data: mujoco.MjData, limit: int = 12) -> list[dict]:
        """How things sit in the water, for the free bodies water has touched (at most `limit`): the share of
        them below the surface, and how deep their center is (negative above the surface)."""
        if self.last_wet is None:
            self.wrenches(data)
        out = []
        for i in np.nonzero(self.ever_wet)[0]:
            b = int(self.bodies[i])
            if int(self.model.body_rootid[b]) != b:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
            share = float(self.last_wet[i] / max(self.body_volume[i], 1e-12))
            p = data.xipos[b]
            top = float(self.surface(np.array([p[0]]), np.array([p[1]]), float(data.time))[0])
            depth = top - float(p[2]) if top > DRY / 2 else None
            out.append(
                {"name": name, "under_water": round(share, 3), "depth_m": None if depth is None else round(depth, 3)}
            )
            if len(out) >= limit:
                break
        return out


def float_height(model: mujoco.MjModel, data: mujoco.MjData, water: Water, roots: list[int]) -> dict[int, float]:
    """How far to raise (+) or lower each free root body in `roots` (with all it carries) so it floats in still
    water, its weight held by its buoyancy at its present tilt. Bodies that sink are left out. `data` is posed
    (mj_kinematics). The search runs from where its top is just under the surface to where its bottom is just
    over it, however deep the water (a fixed 4 m had a crate start 8 m under in water 12 m deep)."""
    still = Water(water.level, water.density, 0.0, area=water.area)
    hydro = Hydro(model, still)
    owner = model.body_rootid[hydro.bodies] if len(hydro.bodies) else np.zeros(0, int)
    roots = [r for r in roots if r in set(owner.tolist())]
    if not roots:
        return {}
    index = {r: k for k, r in enumerate(roots)}
    pick = np.array([index.get(int(o), -1) for o in owner])  # water body -> its root's place in `roots`
    chosen = pick >= 0
    need = np.array([float(model.body_subtreemass[r]) * GRAVITY for r in roots])

    def buoyancy(shift: np.ndarray) -> np.ndarray:
        wet, _ = hydro.cells(data, lift=np.where(chosen, shift[np.maximum(pick, 0)], 0.0))
        per_body = hydro._geoms_to_bodies @ hydro.per_geom(wet)
        return np.bincount(pick[chosen], per_body[chosen], minlength=len(roots)) * still.density * GRAVITY

    mine = pick[hydro.geom_body]  # water geom -> its root's place in `roots`
    shapes = mine >= 0
    z, r = data.geom_xpos[hydro.geoms[shapes], 2], hydro._geom_rbound[shapes]
    top, bottom = np.full(len(roots), -np.inf), np.full(len(roots), np.inf)
    np.maximum.at(top, mine[shapes], z + r)
    np.minimum.at(bottom, mine[shapes], z - r)
    low, high = water.level - top - 0.01, water.level - bottom + 0.01  # all under; all out
    floats = buoyancy(low) >= need  # pushed all under, it lifts its weight
    for _ in range(40):
        mid = (low + high) / 2
        up = buoyancy(mid) >= need  # more than its weight at this height: it floats higher
        low, high = np.where(up, mid, low), np.where(up, high, mid)
    return {r: float((low[k] + high[k]) / 2) for k, r in enumerate(roots) if floats[k]}


def hydro_for(model: mujoco.MjModel) -> Hydro | None:
    """The water forces for a model that has water, or None."""
    water = decode(model)
    return Hydro(model, water) if water is not None else None
