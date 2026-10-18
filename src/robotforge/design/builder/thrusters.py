"""Thrusters on robots built from parts: a propeller pod that pushes the part it sits on, like a rotor, but only
while it is in water (sim/water.py takes its push away out of the water).

Each `thrusters` entry of the base or a part adds a site named "<part>_thruster" (or its own name) and a site
actuator on it: control -1 to 1, pushing along `direction` with up to `force` newtons (negative pushes back). A
robot with thrusters that all push flat is a boat to `move`; one with some that push up or down is an ROV.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotforge.design.builder.schema import BaseSpec, PartSpec, ThrusterSpec, axis_vector
from robotforge.design.visuals import Visuals
from robotforge.design.visuals import mesh as M
from robotforge.design.visuals import parts as P

SPIN_UP = 0.1  # s for a thruster to reach a new push
KG_PER_NEWTON = 0.007  # pod mass: a 50 N hobby thruster weighs about 350 g
DARK = [0.12, 0.13, 0.15, 1.0]


def add(
    spec: mujoco.MjSpec,
    body: mujoco.MjsBody,
    owner: str,
    part: BaseSpec | PartSpec,
    local: np.ndarray,
    vis: Visuals,
) -> list[str]:
    """Add the thrusters of `part` (on `body`); returns their names."""
    from robotforge.design.builder.expand import anchor

    names: list[str] = []
    for k, thruster in enumerate(part.thrusters):
        name = thruster.name or (f"{owner}_thruster" if k == 0 else f"{owner}_thruster_{k + 1}")
        if "thruster" not in name:
            name = f"{name}_thruster"  # the water model knows a marine thruster by its name
        where = local @ anchor(part, thruster.at, "the base" if owner == "base" else f"part {owner!r}")
        push = local @ np.asarray(axis_vector(thruster.direction), float)
        radius = thruster.size / 2
        body.add_site(name=name, pos=where.tolist(), size=[radius, 0, 0], rgba=[0, 0, 0, 0])
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_SITE)
        act.gear = [*push.tolist(), 0.0, 0.0, 0.0]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-1.0, 1.0]
        act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        act.dynprm[0] = SPIN_UP
        act.gainprm[0] = thruster.force
        # The pod's weight, carried by the part (it touches nothing: the part's own shape meets the world).
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius * 0.6, 0, 0],
            pos=where.tolist(),
            mass=KG_PER_NEWTON * thruster.force,
            contype=0,
            conaffinity=0,
            group=3,
        )
        _draw(vis, body, where, push, radius)
        names.append(name)
    return names


def _draw(vis: Visuals, body: mujoco.MjsBody, where: np.ndarray, push: np.ndarray, radius: float) -> None:
    """A motor pod in a duct with three blades, its axis along the way it pushes (the propeller at the back)."""
    rot = M.z_to(push)
    motor = M.lathe([(0.0, -0.9), (0.42, -0.8), (0.5, 0.0), (0.42, 0.8), (0.0, 0.95)], segments=20).moved(scale=radius)
    duct = M.lathe(
        [(1.05, -0.5), (1.2, -0.45), (1.24, 0.0), (1.18, 0.5), (1.05, 0.46), (1.02, 0.0)], segments=28
    ).moved(scale=radius)
    blades = M.merge(
        M.rounded_box(0.95 * radius, 0.12 * radius, 0.04 * radius, 0.02 * radius, segments=1).moved(
            (0, 0, -0.3 * radius), M.rot_z(k * 2 * np.pi / 3) @ M.rot_x(0.45)
        )
        for k in range(3)
    )
    vis.add(body, "dark", DARK, motor, where, rot)
    vis.add(body, "dark", DARK, duct, where, rot)
    vis.add(body, "metal", P.METAL, blades, where, rot)


__all__ = ["ThrusterSpec", "add"]
