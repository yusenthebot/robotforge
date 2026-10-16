"""Reusable visual robot parts: actuator housings, horns, lenses, LEDs, wheels and feet.

Each function returns {preset: Mesh} in the part's own frame and is cached by its (rounded)
sizes, so recompiling a world reuses the same arrays. Callers place parts with `Visuals.add`.
"""

from __future__ import annotations

import math

import numpy as np

from robotforge.design.visuals import mesh as M

Parts = dict[str, M.Mesh]

GLASS = [0.05, 0.07, 0.1, 1.0]  # nearly black, faintly blue: a lens or visor
METAL = [0.7, 0.72, 0.75, 1.0]
RUBBER = [0.08, 0.08, 0.09, 1.0]
LED_CYAN = [0.45, 0.9, 1.0, 1.0]
LED_GREEN = [0.35, 1.0, 0.55, 1.0]
LED_RED = [1.0, 0.25, 0.2, 1.0]
LED_WHITE = [1.0, 0.97, 0.88, 1.0]


def merge_parts(*groups: Parts) -> Parts:
    out: dict[str, list[M.Mesh]] = {}
    for g in groups:
        for k, m in g.items():
            out.setdefault(k, []).append(m)
    return {k: M.merge(v) for k, v in out.items()}


def moved(parts: Parts, pos=(0.0, 0.0, 0.0), rot: np.ndarray | None = None) -> Parts:
    return {k: m.moved(pos, rot) for k, m in parts.items()}


@M.cached
def horn(r: float, t: float, screws: int = 4) -> Parts:
    """A round servo horn standing on the z = 0 plane: metal disc, dark center screw and (on horns big
    enough to see them) a screw circle."""
    segments = 24 if r > 0.01 else 18
    disc = M.cylinder(r, t, segments=segments, bevel=min(t * 0.4, r * 0.2), z0=0.0, arc=1)
    dark = [M.cylinder(r * 0.24, t * 0.45, segments=10, z0=t * 0.8)]
    for k in range(screws if r > 0.008 else 0):
        a = 2 * math.pi * k / screws + math.pi / 4
        head = M.cylinder(r * 0.1, t * 0.35, segments=6, z0=t * 0.85)
        dark.append(head.moved((r * 0.64 * math.cos(a), r * 0.64 * math.sin(a), 0)))
    return {"metal": disc, "dark": M.merge(dark)}


@M.cached
def servo_box(sx: float, sy: float, sz: float, horn_r: float) -> Parts:
    """A servo case of half sizes (sx, sy, sz) with its round horn on the +z face."""
    r = min(sx, sy, sz) * 0.3
    body = M.rounded_box(sx, sy, sz, r, segments=3)
    # A slightly proud collar under the horn, like the gear-train boss of a real servo.
    boss = M.cylinder(horn_r * 1.05, sz * 0.12, segments=24, bevel=sz * 0.03, z0=sz - 0.0006, arc=1)
    h = horn(horn_r, max(0.0016, sz * 0.09))
    return merge_parts({"dark": M.merge([body, boss])}, moved(h, (0, 0, sz + sz * 0.12 - 0.0012)))


@M.cached
def drum(r: float, half: float, cap_r: float, both: bool = True) -> Parts:
    """A cylindrical joint actuator along z (radius r, half length `half`) with metal end caps."""
    body = M.cylinder(r, 2 * half, segments=28, bevel=min(r * 0.16, half * 0.35), arc=2)
    caps = []
    for side in (1, -1) if both else (1,):
        h = horn(cap_r, max(0.0012, half * 0.16), screws=4)
        caps.append(moved(h, (0, 0, side * (half - 0.0006)), None if side > 0 else M.rot_x(math.pi)))
    return merge_parts({"dark": body}, *caps)


@M.cached
def lens(r: float, depth: float) -> Parts:
    """A camera lens looking along +z from the z = 0 plane: dark bezel, glass and a fine metal ring."""
    bezel = M.revolve(
        [
            (r * 0.66, -depth),
            (r, -depth),
            (r, depth * 0.15),
            (r * 0.9, depth * 0.4),
            (r * 0.7, depth * 0.4),
            (r * 0.66, 0),
        ],
        segments=28,
        sharp=(0, 1, 4, 5),
    )
    dome = [(r * 0.68 * math.cos(a), r * 0.2 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 5)]
    glass = M.lathe([(0, -depth * 0.6), (r * 0.68, -depth * 0.6), *dome], segments=28, sharp=(1,))
    ring = M.torus(r * 0.8, r * 0.045, segments=28, sides=6).moved((0, 0, depth * 0.4))
    return {"dark": bezel, "glass": glass, "metal": ring}


@M.cached
def led(r: float, depth: float = 0.0015) -> Parts:
    dome = [(r * math.cos(a), depth * math.sin(a)) for a in np.linspace(0, math.pi / 2, 4)]
    return {"led": M.lathe([(0, -depth), (r, -depth), *dome], segments=20, sharp=(1,))}


@M.cached
def ball_foot(r: float, shin_r: float, segments: int = 24) -> Parts:
    """A rubber ball foot of radius r at the origin with a metal collar where a shin of radius shin_r joins."""
    ball = M.sphere(r, rings=segments // 2, segments=segments)
    collar = M.cylinder(shin_r * 1.12, r * 0.32, segments=segments * 5 // 6, bevel=r * 0.05, z0=r * 0.7)
    return {"rubber": ball, "metal": collar}


@M.cached
def tyre(r: float, half_w: float, segments: int = 60, spokes: int = 5) -> Parts:
    """A wheel about z: tyre with tread blocks (tread on radius r, -half_w..half_w), spoked rim, hub.

    `segments` around the tread (a multiple of 4) sets the detail: small wheels need fewer. With
    `spokes` 0 the rim is a closed dish, which is what a wheel a few centimetres across shows anyway.
    """
    shoulder = min(half_w * 0.45, r * 0.18)
    inner = r * 0.72
    prof = [(inner, -half_w * 0.9)]
    for a in np.linspace(0, math.pi / 2, 4):
        prof.append((r - shoulder + shoulder * math.sin(a), -half_w + shoulder - shoulder * math.cos(a)))
    for a in np.linspace(math.pi / 2, 0, 4):
        prof.append((r - shoulder + shoulder * math.sin(a), half_w - shoulder + shoulder * math.cos(a)))
    prof.append((inner, half_w * 0.9))
    n = segments
    rings = [M.at_z(M.circle(rr, n), z) for rr, z in prof]
    # Tread blocks: the rings across the tread step in and out a little, so the tyre shows grip.
    # Four samples per block keep each one square at any segment count.
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bump = 1.0 + 0.015 * np.sign(np.sin(a * (n // 4) + 0.3))
    for k in range(3, len(rings) - 3):
        rings[k] = np.column_stack([rings[k][:, :2] * bump[:, None], rings[k][:, 2]])
    tyre_mesh = M.loft(rings, cap_start=False, cap_end=False)  # open inside: the rim fills it
    R = inner * 1.02
    rim = M.revolve(
        [
            (R * 0.66, -half_w * 0.8),
            (R, -half_w * 0.8),
            (R, half_w * 0.72),
            (R * 0.93, half_w * 0.86),
            (R * 0.8, half_w * 0.86),
            (R * 0.74, half_w * 0.5),
            (R * 0.66, half_w * 0.45),
        ],
        segments=max(24, n * 4 // 5),
        sharp=(0, 1, 2, 5, 6),
    )
    arms = []
    for k in range(spokes):
        spoke = M.rounded_box(R * 0.27, R * 0.075, half_w * 0.22, R * 0.05, segments=2)
        arms.append(spoke.moved((R * 0.47, 0, half_w * 0.5)).moved(rot=M.rot_z(2 * math.pi * k / spokes)))
    hub = M.cylinder(R * 0.26, half_w * 1.55, segments=max(16, n * 7 // 15), bevel=half_w * 0.18, z0=-half_w * 0.75)
    back = M.cylinder(R * 0.7, half_w * 0.1, segments=max(20, n * 3 // 5), z0=-half_w * 0.2)
    if not spokes:  # a closed dish sitting a little proud of the back plate
        back = M.cylinder(R * 0.76, half_w * 0.62, segments=max(20, n * 3 // 5), bevel=half_w * 0.2, z0=-half_w * 0.2)
    nut = bolt(R * 0.11, half_w * 0.25)
    return {
        "rubber": tyre_mesh,
        "shell": M.merge([rim, *arms]),
        "dark": back,
        "metal": M.merge([hub, nut.moved((0, 0, half_w * 0.8))]),
    }


@M.cached
def bolt(r: float, h: float) -> M.Mesh:
    """A hex bolt head standing on the z = 0 plane."""
    return M.lathe([(0, 0), (r, 0), (r, h * 0.7), (r * 0.8, h), (0, h)], segments=6, sharp=(1, 2, 3))


def bolt_circle(radius: float, n: int, r: float, h: float, start: float = 0.0) -> M.Mesh:
    heads = []
    for k in range(n):
        a = start + 2 * math.pi * k / n
        heads.append(bolt(r, h).moved((radius * math.cos(a), radius * math.sin(a), 0)))
    return M.merge(heads)


@M.cached
def slats(hx: float, hz: float, count: int, depth: float) -> M.Mesh:
    """Horizontal vent slats in the x-z plane, `depth` thick along y."""
    out = []
    pitch = 2 * hz / count
    for k in range(count):
        z = -hz + pitch * (k + 0.5)
        out.append(M.rounded_box(hx, depth, pitch * 0.25, pitch * 0.12, segments=2).moved((0, 0, z)))
    return M.merge(out)


# ---- parts for robots built from parts (design/builder) ---------------------------------------------

PCB = [0.1, 0.32, 0.24, 1.0]  # solder-mask green: a bare sensor board reads as electronics at a glance


@M.cached
def servo_case(hx: float, hy: float, hz: float, horn_r: float, shaft_x: float) -> Parts:
    """A hobby servo case of half sizes (hx, hy, hz) without its horn (the horn turns with the joint).

    The output boss stands on the +z face at x = shaft_x, off-centre as on a real servo, and the
    mounting ears stick out of both ends a little below the top, each with a bolt.
    """
    body = M.rounded_box(hx, hy, hz, min(hx, hy, hz) * 0.28, segments=2)
    boss = M.cylinder(horn_r * 1.05, hz * 0.12, segments=16, bevel=hz * 0.03, z0=hz - 0.0006, arc=1)
    ear_len = max(0.002, 0.32 * hx)
    ear_t = max(0.0006, hz * 0.055)
    ear_z = hz * 0.42
    ears, bolts = [], []
    for side in (1, -1):
        ear = M.rounded_box(ear_len / 2 + 0.001, hy * 0.92, ear_t, ear_t * 0.6, segments=1, corner=hy * 0.3)
        ears.append(ear.moved((side * (hx + ear_len / 2 - 0.001), 0, ear_z)))
        head = bolt(min(ear_len * 0.28, hy * 0.3), ear_t * 1.4)
        bolts.append(head.moved((side * (hx + ear_len * 0.48), 0, ear_z + ear_t - 0.0002)))
    return {"dark": M.merge([body, boss.moved((shaft_x, 0, 0)), *ears]), "metal": M.merge(bolts)}


@M.cached
def joint_module(r: float, half: float, cap_r: float) -> Parts:
    """The fixed half of a round joint module (a quadruped-style motor) about z: the drum and a cap on its
    -z face. Its output flange is a `horn` on the +z face that turns with the joint."""
    body = M.cylinder(r, 2 * half, segments=32, bevel=min(r * 0.16, half * 0.35), arc=2)
    cap = horn(cap_r, max(0.0012, half * 0.16), screws=4)
    return merge_parts({"dark": body}, moved(cap, (0, 0, -(half - 0.0006)), M.rot_x(math.pi)))


@M.cached
def knuckle(r: float, half: float) -> Parts:
    """An unpowered hinge seen from outside: a short dark sleeve about z with a bearing cap on each face."""
    sleeve = M.cylinder(r, 2 * half, segments=28, bevel=min(r * 0.2, half * 0.3), arc=2)
    out: list[Parts] = [{"dark": sleeve}]
    for side in (1, -1):
        cap = horn(r * 0.58, max(0.001, half * 0.14), screws=3 if r > 0.012 else 0)
        out.append(moved(cap, (0, 0, side * (half - 0.0005)), None if side > 0 else M.rot_x(math.pi)))
    return merge_parts(*out)


@M.cached
def flange(r: float, t: float, bolts: int = 4) -> Parts:
    """A round mounting flange on the z = 0 plane, `t` thick, with a bolt circle on top."""
    disc = M.cylinder(r, t, segments=32, bevel=t * 0.35, z0=0.0, arc=1)
    heads = bolt_circle(r * 0.72, bolts, max(0.0012, r * 0.1), max(0.0008, t * 0.55), start=math.pi / bolts)
    return {"dark": disc, "metal": heads.moved((0, 0, t - 0.0002))}


@M.cached
def plate(r: float, t: float, bolts: int = 8) -> Parts:
    """An aluminium adapter plate (a robot mounted on another stands on one), bolted round its rim."""
    disc = M.cylinder(r, t, segments=48, bevel=t * 0.3, z0=0.0, arc=1)
    heads = bolt_circle(r * 0.87, bolts, max(0.0015, r * 0.045), max(0.001, t * 0.5), start=math.pi / bolts)
    return {"metal": disc, "dark": heads.moved((0, 0, t - 0.0002))}


@M.cached
def gear_motor(r: float, length: float, yellow_box: bool) -> Parts:
    """A wheel's gear motor along z from 0 (the wheel side) to `length`, radius r.

    A TT motor is a yellow plastic gearbox with a silver can behind it; a 25 mm motor is a metal
    gearbox and can with a dark ring between them and a dark end cap.
    """
    if yellow_box:
        box = M.rounded_box(r, r * 0.82, length * 0.3, r * 0.25, segments=2).moved((0, 0, length * 0.3))
        can = M.cylinder(r * 0.82, length * 0.36, segments=24, bevel=r * 0.08, z0=length * 0.6)
        cap = M.cylinder(r * 0.5, length * 0.04, segments=20, z0=length * 0.96)
        return {"shell": box, "metal": can, "dark": cap}
    gearbox = M.cylinder(r, length * 0.42, segments=20, bevel=r * 0.06, z0=0.0, arc=1)
    ring = M.cylinder(r * 1.02, length * 0.05, segments=20, z0=length * 0.42)
    can = M.cylinder(r, length * 0.42, segments=20, bevel=r * 0.04, z0=length * 0.47, arc=1)
    cap = M.cylinder(r * 0.92, length * 0.11, segments=20, bevel=r * 0.12, z0=length * 0.89, arc=2)
    return {"metal": M.merge([gearbox, can]), "dark": M.merge([ring, cap])}


@M.cached
def caster(r: float, height: float) -> Parts:
    """A ball caster: a chrome ball of radius r at the origin in a dark socket that reaches `height`
    up the +z axis, with a flange where it is screwed on."""
    ball = M.sphere(r, rings=12, segments=24)
    top = max(height, r * 1.2)
    socket = M.lathe(
        [
            (0.0, r * 0.2),
            (r * 1.08, r * 0.2),
            (r * 1.08, top - r * 0.18),
            (r * 1.3, top - r * 0.16),
            (r * 1.3, top),
            (0.0, top),
        ],
        segments=32,
        sharp=(1, 2, 3, 4),
    )
    return {"metal": ball, "dark": socket}


@M.cached
def camera_module(s: float, depth: float, barrel: float) -> Parts:
    """A small board camera looking along +z: a square housing (half width s, `depth` deep) behind the
    z = 0 plane, a lens barrel `barrel` long in front of it with a metal rim and a glass dome, and a
    status LED in a corner (+x, +y).

    The camera itself sits inside the housing, so everything in front of it is drawn open toward it:
    the barrel, rim and dome have no back faces, and from the camera's point of view (backface culled)
    they are invisible. A robot's own camera never sees its housing.
    """
    housing = M.rounded_box(s, s, depth / 2, min(s, depth) * 0.3, segments=3, corner=s * 0.35)
    r = s * 0.62
    n = 28

    def rings(profile):
        return [M.at_z(M.circle(rr, n), z) for rr, z in profile]

    tube = M.loft(rings([(r, -0.0004), (r, barrel * 0.8), (r * 0.92, barrel)]), cap_start=False)
    rim = M.loft(
        rings([(r * 1.07, barrel * 0.5), (r * 1.07, barrel * 0.92), (r * 0.98, barrel * 1.02)]),
        cap_start=False,
        cap_end=False,
    )
    dome = [(r * 0.8 * math.cos(a), barrel + r * 0.22 * math.sin(a)) for a in np.linspace(0, math.pi / 2, 5)]
    glass = M.loft(rings(dome), cap_start=False)
    dot = moved(led(s * 0.09, 0.0008), (s * 0.68, s * 0.68, 0.0))
    return merge_parts({"dark": M.merge([housing.moved((0, 0, -depth / 2)), tube]), "metal": rim, "glass": glass}, dot)


@M.cached
def lidar_head(r: float, h: float) -> Parts:
    """A spinning lidar on the z = 0 plane: dark base, a dark glass window all round, a cap with a metal ring."""
    base = M.cylinder(r, h * 0.36, segments=32, bevel=h * 0.05, z0=0.0, arc=2)
    window = M.cylinder(r * 0.955, h * 0.44, segments=32, z0=h * 0.33)
    cap = M.cylinder(r * 0.98, h * 0.27, segments=32, bevel=h * 0.11, z0=h * 0.73, arc=3)
    ring = M.torus(r * 0.985, r * 0.025, segments=32, sides=5).moved((0, 0, h * 0.355))
    return {"dark": M.merge([base, cap]), "glass": window, "metal": ring}


@M.cached
def imu_board(s: float) -> Parts:
    """An IMU breakout on the z = 0 plane: a green board, the sensor chip and a power LED."""
    t = max(0.0006, s * 0.09)
    board = M.rounded_box(s, s * 0.78, t, t * 0.6, segments=2, corner=s * 0.15).moved((0, 0, t))
    chip = M.rounded_box(s * 0.32, s * 0.32, t * 0.6, t * 0.2, segments=1).moved((0, 0, 2 * t + t * 0.5))
    dot = moved(led(s * 0.08, t), (s * 0.68, s * 0.5, 2 * t))
    holes = bolt_circle(s * 0.95, 4, s * 0.1, t * 0.5, start=math.pi / 4).moved((0, 0, 2 * t - 0.0001))
    return merge_parts({"shell": board, "dark": chip, "metal": holes}, dot)


@M.cached
def range_finder(w: float) -> Parts:
    """An ultrasonic range finder facing +z from the z = 0 plane: a board with two transducer cans."""
    t = max(0.0006, w * 0.05)
    board = M.rounded_box(w, w * 0.42, t, t * 0.6, segments=2, corner=w * 0.08).moved((0, 0, t))
    cans, faces = [], []
    for side in (1, -1):
        can = M.cylinder(w * 0.36, w * 0.38, segments=16, bevel=w * 0.03, z0=2 * t)
        cans.append(can.moved((side * w * 0.52, 0, 0)))
        face = M.cylinder(w * 0.3, w * 0.02, segments=16, z0=2 * t + w * 0.38 - 0.0002)
        faces.append(face.moved((side * w * 0.52, 0, 0)))
    return {"shell": board, "metal": M.merge(cans), "dark": M.merge(faces)}


@M.cached
def lead_screw(r: float, length: float) -> M.Mesh:
    """A lead screw along z from 0 to `length`, drawn as a polished rod: ridges fine enough to be a thread
    shimmer when the view moves, and coarse ones read as a twisted cable."""
    return M.cylinder(r, length, segments=16, bevel=r * 0.25, z0=0.0, arc=2)
