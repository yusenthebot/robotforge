"""A small robot boat (an uncrewed surface vessel): a hull, a console with a camera, two stern thrusters.

It steers like a tank, by thrusting harder on one side (differential thrust), so it can turn on the spot.
The hull the water and the dock feel is three boxes stepping in toward the bow, so it floats level at the draft
its weight gives; the hull you see is a lofted mesh over them. Its weight sits low (the battery in the bilge),
so it rolls back upright after a push. Thrusters are site actuators named "..._thruster": they push like rotors,
and only while their propeller is in the water (sim/water.py).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from pydantic import Field

from robotforge.design import components
from robotforge.design.templates.base import DARK, GEOM, SHELL, BuiltRobot, Template, TemplateParams, imu, new_spec
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

WATER_DENSITY = 1000.0
LOAD = 0.3  # the boat's mass as a share of the water its hull boxes displace when awash: a 30 percent draft
SPIN_UP = 0.15  # s for a thruster to reach a new thrust (its motor and propeller speeding up)
DECK = [0.86, 0.86, 0.83, 1.0]
BOTTOM = [0.16, 0.2, 0.26, 1.0]


class BoatParams(TemplateParams):
    color: str | list[float] = "orange"
    length: float = Field(1.2, ge=0.4, le=4.0, description="hull length in meters")
    width: float | None = Field(None, gt=0.1, le=1.6, description="beam in meters (default 0.4 x length)")
    thrust: float | None = Field(None, gt=0, le=2000, description="newtons per thruster (default from the size)")


class Boat(Template):
    name = "boat"
    summary = (
        "Robot boat: hull, console with a front camera and an IMU, two stern thrusters (differential thrust, "
        "turns on the spot); floats in a world with water; move drives it, move(to=...) finds its way round the shore."
    )
    Params = BoatParams

    def build(self, p: BoatParams) -> BuiltRobot:
        L = p.length
        B = p.width if p.width is not None else 0.4 * L
        H = max(0.12, 0.25 * L)
        accent = components.color(p.color)
        spec = new_spec()
        vis = Visuals(spec)
        hull = spec.worldbody.add_body(name="hull")
        hull.add_freejoint(name="root")

        # Hull boxes from the stern: the full-width main hull, then two narrower steps that make the bow. The steps
        # start above the waterline: at rest the bow is out of the water (no buoyancy, no drag), as on a real boat;
        # it is pushed back up when it dips into a wave, and it is there for the dock to bump into.
        boxes = [  # x0, x1 (along L), share of the beam, bottom (share of H)
            (-0.5, 0.2, 1.0, 0.0),
            (0.2, 0.35, 0.68, LOAD + 0.04),
            (0.35, 0.5, 0.32, LOAD + 0.12),
        ]
        main = boxes[0]
        mass = LOAD * WATER_DENSITY * (main[1] - main[0]) * L * B * H  # floats with LOAD of the main hull under
        volume = sum((x1 - x0) * w * (1 - z0) for x0, x1, w, z0 in boxes)
        shell = 0.5 * mass  # the hull itself; the battery and the rest of the gear make up the remainder
        weights: list[tuple[float, float]] = []  # (mass, x) of everything, to trim the boat level
        for x0, x1, w, z0 in boxes:
            half = [(x1 - x0) * L / 2, w * B / 2, (1 - z0) * H / 2]
            share = (x1 - x0) * w * (1 - z0) / volume
            x = (x0 + x1) / 2 * L
            geom = hull.add_geom(
                type=GEOM.mjGEOM_BOX,
                size=half,
                pos=[x, 0, z0 * H + half[2]],
                rgba=accent,
                mass=shell * share,
                friction=[0.6, 0.01, 0.001],
            )
            vis.hide(geom)
            weights.append((shell * share, x))
        console_x = -0.12 * L
        console = [0.11 * L, 0.32 * B, 0.17 * H]
        _gear(hull, [console_x, 0, H + console[2]], console, 0.08 * mass)
        weights.append((0.08 * mass, console_x))
        pod_r = max(0.018, 0.035 * L)
        pod_z = -1.6 * pod_r
        pod_y = 0.27 * B
        pod_x = -0.5 * L + 2.2 * pod_r
        weights.append((0.07 * mass, pod_x))
        # The battery, heavy and low, goes where it puts the center of mass over the center of the wet hull, so the
        # boat floats level; low, it keeps the boat upright as ballast does.
        battery = mass - sum(m for m, _ in weights)
        center = (main[0] + main[1]) / 2 * L
        battery_x = float(np.clip((mass * center - sum(m * x for m, x in weights)) / battery, -0.4 * L, 0.15 * L))
        _gear(hull, [battery_x, 0, 0.08 * H], [0.12 * L, 0.3 * B, 0.05 * H], battery)
        thrust = p.thrust if p.thrust is not None else round(0.6 * mass, 1)  # about 0.6 g of push per thruster pair
        pod_r = max(0.018, 0.035 * L)
        pod_z = -1.6 * pod_r
        pod_y = 0.27 * B
        for side, sign in (("left", 1), ("right", -1)):
            x = pod_x
            _gear(hull, [x, sign * pod_y, pod_z], [2.2 * pod_r, pod_r, pod_r], 0.035 * mass)
            hull.add_site(name=f"{side}_thruster", pos=[x - 1.6 * pod_r, sign * pod_y, pod_z], size=[pod_r, 0, 0])
            act = spec.add_actuator(
                name=f"{side}_thruster", target=f"{side}_thruster", trntype=mujoco.mjtTrn.mjTRN_SITE
            )
            act.gear = [1, 0, 0, 0, 0, 0]
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            act.ctrlrange = [-1, 1]
            act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
            act.dynprm[0] = SPIN_UP
            act.gainprm[0] = thrust
        cam_z = H + 2 * console[2] + 0.02 * L
        hull.add_camera(name="front_cam", pos=[console_x + console[0], 0, cam_z], xyaxes=[0, -1, 0, 0, 0, 1], fovy=70)
        hull.add_site(name="gps", pos=[console_x - 0.3 * console[0], 0, cam_z + 0.12 * L], size=[0.01, 0, 0])
        imu(spec, hull)
        _dress(vis, hull, L, B, H, accent, console_x, console, pod_r, pod_y, pod_z)
        vis.finish()
        return BuiltRobot(
            spec=spec,
            spawn_height=0.002,
            floating=True,
            notes=[
                f"boat, {L:.2f} m long, {mass:.0f} kg, two stern thrusters of {thrust:g} N (left_thruster, "
                "right_thruster, -1 to 1: thrust shares, negative is astern); it floats in a world with water "
                "(set_floor water=... or load_scene lake, harbor or pool)"
            ],
        )


def _gear(body: mujoco.MjsBody, pos: list[float], half: list[float], mass: float) -> None:
    """Mass inside the boat that touches nothing (the water only sees the hull boxes)."""
    body.add_geom(
        type=GEOM.mjGEOM_BOX, size=half, pos=pos, mass=mass, contype=0, conaffinity=0, group=3, rgba=[0.3] * 3 + [1]
    )


# ---- looks: visual-only meshes over the hull boxes ---------------------------------------------------------


def hull_rings(L: float, B: float, H: float, stations: int = 18) -> list[np.ndarray]:
    """Cross sections of a planing hull from the transom to the stem: a soft V bottom, flared topsides and a
    cambered deck. The plan narrows ahead of 15 percent of the length, and the keel sweeps up to the stem."""
    xs = -L / 2 + L * (1 - np.cos(np.linspace(0, math.pi / 2, stations))) ** 0.8  # closer at the bow
    rings = []
    for x in xs:
        u = (x - 0.15 * L) / (0.35 * L)
        w = B / 2 * (math.sqrt(max(1 - u * u, 0.0)) if u > 0 else 1.0)
        w = max(w, 0.006 * L)
        bow = max(0.0, (x - 0.1 * L) / (0.4 * L))
        keel = 0.42 * H * bow**1.6
        sheer = H * (1 + 0.07 * max(0.0, (x + 0.1 * L) / (0.6 * L)) ** 2)
        dead = 0.12 * H * (1 + 1.2 * bow)
        camber = 0.03 * w
        half = [
            (0.5 * 0.88 * w, keel + 0.42 * dead),
            (0.88 * w, keel + dead),
            (0.95 * w, keel + dead + 0.4 * (sheer - keel - dead)),
            (w, sheer),
            (0.55 * w, sheer + 0.75 * camber),
        ]
        ring = [(0.0, keel), *half, (0.0, sheer + camber), *[(-y, z) for y, z in reversed(half)]]
        rings.append(np.array([[x, y, z] for y, z in ring]))
    return rings


def _dress(vis: Visuals, hull, L, B, H, accent, console_x, console, pod_r, pod_y, pod_z) -> None:
    rings = hull_rings(L, B, H)
    vis.add(hull, "shell", accent, M.loft(rings))
    # Antifouling below the waterline: the same hull, a hair larger, cut off at a third of its height.
    low = [r.copy() for r in rings]
    for r in low:
        r[:, 1] *= 1.004
        r[:, 2] = np.minimum(r[:, 2], 0.3 * H) - 0.001
    vis.add(hull, "paint", BOTTOM, M.loft(low))
    # A deck inset inside the gunwale, and a rub rail along it.
    deck_rings = []
    for r in rings[1:-3]:
        top = r[r[:, 2] >= r[:, 2].max() - 1e-9][0, 2]
        w = np.abs(r[:, 1]).max() * 0.86
        deck_rings.append(np.array([[r[0, 0], y, top + 0.004 + 0.02 * (w - abs(y))] for y in np.linspace(w, -w, 9)]))
    deck = M.loft(deck_rings, closed=False)
    vis.add(hull, "paint", DECK, deck)
    gunwale = np.array([[r[0, 0], np.abs(r[:, 1]).max() * 1.01, r[:, 2].max() - 0.01 * H] for r in rings])
    for sign in (1, -1):
        path = gunwale * np.array([1, sign, 1])
        vis.add(hull, "rubber", P.RUBBER, M.tube(path, [0.012 * L] * len(path), segments=10))
    # The console: a rounded box with a dark windscreen, a camera, a mast with the lights and a GPS dome.
    cx, cw, ch = console
    cz = H + ch
    vis.add(
        hull, "shell", SHELL, M.rounded_box(cx, cw, ch, 0.25 * ch, segments=3, corner=0.35 * cw), (console_x, 0, cz)
    )
    screen = M.rounded_box(0.004, 0.8 * cw, 0.55 * ch, 0.003, segments=2, corner=0.1 * cw)
    vis.add(hull, "glass", P.GLASS, screen, (console_x + cx * 0.98, 0, cz + 0.25 * ch), M.rot_y(-0.35))
    lens = max(0.008, 0.012 * L)
    for preset, mesh in P.lens(lens, 0.6 * lens).items():
        color = {"dark": DARK, "glass": P.GLASS, "metal": P.METAL}[preset]
        vis.add(hull, preset, color, mesh, (console_x + cx, 0, cz + ch + 0.02 * L - lens * 0.2), M.Z_TO_X)
    mast_x = console_x - 0.3 * cx
    mast_h = 0.12 * L
    vis.add(hull, "metal", P.METAL, M.cylinder(0.006 * L, mast_h, segments=16, z0=0.0), (mast_x, 0, cz + ch))
    top = cz + ch + mast_h
    vis.add(hull, "shell", SHELL, M.sphere(0.03 * L, rings=10, segments=20).moved(scale=(1, 1, 0.6)), (mast_x, 0, top))
    vis.add(hull, "led", P.LED_WHITE, M.sphere(0.008 * L, rings=6, segments=12), (mast_x, 0, top + 0.02 * L))
    # Navigation lights at the bow: red to port (left), green to starboard.
    for sign, led in ((1, P.LED_RED), (-1, P.LED_GREEN)):
        vis.add(hull, "led", led, M.sphere(0.009 * L, rings=6, segments=12), (0.3 * L, sign * 0.3 * B, H * 1.04))
    # Thruster pods under the stern: a torpedo motor in a duct, on a fin.
    for sign in (1, -1):
        y = sign * pod_y
        x = -0.5 * L + 2.2 * pod_r
        body = M.lathe(
            [(0.0, -2.0), (0.5, -1.8), (0.95, -1.0), (1.0, 0.0), (0.9, 1.2), (0.4, 2.0), (0.0, 2.2)], segments=24
        ).moved(scale=pod_r)
        vis.add(hull, "dark", DARK, body, (x, y, pod_z), M.Z_TO_X)
        duct = M.torus(1.25 * pod_r, 0.18 * pod_r, segments=28, sides=8)
        vis.add(hull, "dark", DARK, duct, (x - 1.7 * pod_r, y, pod_z), M.Z_TO_X)
        fin = M.rounded_box(1.2 * pod_r, 0.12 * pod_r, (0 - pod_z) / 2 + 0.01, 0.1 * pod_r, segments=2)
        vis.add(hull, "dark", DARK, fin, (x, y, pod_z / 2 + 0.005))
        blades = M.merge(
            M.rounded_box(0.08 * pod_r, 1.05 * pod_r, 0.22 * pod_r, 0.04 * pod_r, segments=1).moved(
                (0, 0, 0), M.rot_x(k * math.pi / 3)
            )
            for k in range(3)
        )
        vis.add(hull, "metal", P.METAL, blades, (x - 1.7 * pod_r, y, pod_z))
