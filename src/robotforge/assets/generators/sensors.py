"""Sensor housings: a camera, a depth camera, a lidar, an IMU board and an ultrasonic range finder you can recognise."""

from __future__ import annotations

import math

import numpy as np

from robotforge.assets.asset import Asset, Collider
from robotforge.assets.generators.common import CYAN, GREEN, Pieces, hollow_cylinder, screw_ring, segments
from robotforge.assets.registry import Args, Param, generator
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

KINDS = ("camera", "depth_camera", "lidar", "imu", "ultrasonic")
PCB = (0.1, 0.32, 0.24, 1.0)


def _lens(pc: Pieces, r: float, z: float, at=(0.0, 0.0), barrel: float = 0.0) -> None:
    """A lens looking along +z from the plane z: a dark barrel with fine rings, a metal rim, a glass dome."""
    x, y = at
    n = segments(r, 0.002, 20, 40)
    tube = M.cylinder(r, barrel + 0.0004, segments=n, bevel=min(0.12 * r, 0.0008), z0=z - 0.0004)
    pc.add("structure", tube.moved((x, y, 0.0)), material="dark")
    rim = M.torus(r * 0.97, r * 0.06, segments=n, sides=6).moved((x, y, z + barrel))
    pc.add("trim", rim, material="metal")
    ring = M.torus(r * 0.78, r * 0.04, segments=n, sides=6).moved((x, y, z + barrel))
    pc.add("structure", ring, material="dark")
    dome = [(r * 0.72 * math.cos(t), z + barrel + r * 0.28 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 6)]
    pc.add(
        "glass",
        M.lathe([(0.0, z + barrel * 0.9), (r * 0.72, z + barrel * 0.9), *dome], n, sharp=(1,)).moved((x, y, 0.0)),
    )


@generator(
    "sensor_housing",
    summary=(
        "a sensor you can recognise: a camera with a lens barrel and glass, a depth camera with two lenses and an "
        "emitter, a spinning lidar, an IMU board or an ultrasonic range finder"
    ),
    params=[
        Param("kind", "camera", doc="which sensor", kind="choice", choices=KINDS),
        Param("size", 0.03, 0.008, 0.2, "its main dimension: a camera's width, a lidar's diameter, a board's width"),
    ],
    category="sensor",
    anchor="mounted on the z = 0 plane; the sensor looks along +z",
)
def sensor_housing(a: Args) -> Asset:
    s = a.style
    pc = Pieces(a)
    w = a.size
    kind = a.kind
    if kind == "camera":
        hx, depth = w / 2, 0.8 * w
        pc.add(
            "structure",
            M.rounded_box(hx, hx, depth / 2, min(hx, depth) * 0.3 * s.fillet, segments=3, corner=hx * 0.35).moved(
                (0, 0, depth / 2)
            ),
            material="dark",
        )
        pc.add(
            "trim",
            M.rounded_box(hx * 1.02, hx * 1.02, 0.03 * w, 0.02 * w, segments=2, corner=hx * 0.35).moved((0, 0, depth)),
            material="metal",
        )
        _lens(pc, 0.3 * w, depth, barrel=0.18 * w)
        pc.add("led", P.led(0.045 * w, 0.0008)["led"].moved((0.36 * w, 0.36 * w, depth + 0.03 * w)), color=GREEN)
        col = Collider("box", (hx, hx, depth / 2), (0.0, 0.0, depth / 2))
    elif kind == "depth_camera":
        hx, hy, depth = 1.15 * w, 0.32 * w, 0.4 * w
        pc.add(
            "body",
            M.rounded_box(hx, hy, depth / 2, hy * 0.28 * s.fillet, segments=3, corner=hy * 0.6).moved(
                (0, 0, depth / 2)
            ),
        )
        pc.add(
            "structure",
            M.rounded_box(hx * 0.94, hy * 0.86, 0.02 * w, 0.01 * w, segments=2, corner=hy * 0.5).moved(
                (0, 0, depth + 0.002)
            ),
            material="dark",
        )
        for x in (-0.62 * hx, 0.18 * hx):
            _lens(pc, 0.2 * w, depth + 0.001, at=(x, 0.0), barrel=0.06 * w)
        pc.add("glass", M.cylinder(0.09 * w, 0.01 * w, segments=16, z0=depth + 0.001).moved((0.62 * hx, 0, 0)))
        pc.add("led", P.led(0.03 * w, 0.0008)["led"].moved((0.86 * hx, 0.0, depth + 0.002)), color=CYAN)
        col = Collider("box", (hx, hy, depth / 2), (0.0, 0.0, depth / 2))
    elif kind == "lidar":
        r, h = w / 2, 0.85 * w
        seg = segments(r, 0.003, 32, 64)
        pc.add("structure", M.cylinder(r, h * 0.3, segments=seg, bevel=h * 0.04, z0=0.0, arc=2), material="dark")
        pc.add("glass", M.cylinder(r * 0.96, h * 0.4, segments=seg, z0=h * 0.28))
        pc.add("body", M.cylinder(r * 0.99, h * 0.3, segments=seg, bevel=h * 0.1, z0=h * 0.7, arc=3))
        pc.add("trim", M.torus(r * 0.985, r * 0.025, segments=seg, sides=5).moved((0, 0, h * 0.295)), material="metal")
        pc.add("led", ring_led_z(r * 0.9, h * 0.995), color=CYAN)
        col = Collider("cylinder", (r, h / 2), (0.0, 0.0, h / 2))
    elif kind == "imu":
        t = max(0.0006, w * 0.09)
        pc.add(
            "structure",
            M.rounded_box(w / 2, w * 0.39, t, t * 0.6, segments=2, corner=w * 0.08).moved((0, 0, t)),
            color=PCB,
            material="powder",
        )
        pc.add(
            "structure",
            M.rounded_box(w * 0.16, w * 0.16, t * 0.6, t * 0.2, segments=1).moved((0, 0, 2 * t + t * 0.5)),
            material="dark",
        )
        screw_ring(pc, w * 0.45, 4, w * 0.04, t * 0.5, 2 * t - 0.0001, socket=False, start=math.pi / 4)
        pc.add("led", P.led(w * 0.04, t)["led"].moved((w * 0.34, w * 0.25, 2 * t)), color=GREEN)
        col = Collider("box", (w / 2, w * 0.39, 1.5 * t), (0.0, 0.0, 1.5 * t))
    else:  # ultrasonic
        t = max(0.0006, w * 0.05)
        pc.add(
            "structure",
            M.rounded_box(w / 2, w * 0.21, t, t * 0.6, segments=2, corner=w * 0.04).moved((0, 0, t)),
            color=PCB,
            material="powder",
        )
        for side in (1, -1):
            x = side * w * 0.26
            pc.add(
                "trim",
                M.cylinder(w * 0.18, w * 0.19, segments=20, bevel=w * 0.015, z0=2 * t).moved((x, 0, 0)),
                material="metal",
            )
            pc.add(
                "structure",
                hollow_cylinder(w * 0.14, w * 0.04, 2 * t + w * 0.19 - 0.0002, 2 * t + w * 0.19 + 0.0003, 20).moved(
                    (x, 0, 0)
                ),
                material="dark",
            )
        col = Collider("box", (w / 2, w * 0.21, 0.5 * (w * 0.19 + 2 * t)), (0.0, 0.0, 0.5 * (w * 0.19 + 2 * t)))
    return Asset("sensor_housing", pc.done(), (col,))


def ring_led_z(r: float, z: float) -> M.Mesh:
    return M.torus(r, max(0.0004, r * 0.03), segments=32, sides=6).moved((0, 0, z))
