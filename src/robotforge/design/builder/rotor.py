"""Rotor parts (shape "rotor"): a brushless motor with a propeller on top, for aircraft built from parts.

A rotor is a thrust actuator, like the drone template's: a MuJoCo motor on a site at the propeller hub that pushes
along the part's up axis (its own z, so a rotor turned on its side pushes sideways) and twists its body the other
way the propeller spins. Its joint is fixed: the propeller's turning is drawn by the viewer, not simulated, which
would need a time step a hundred times smaller. Mount a rotor on a hinged part to tilt it.

Defaults come from the propeller's diameter: thrust from a typical disc loading of hobby multirotors (a 10 inch
propeller on a 2212 motor lifts about 8 N), mass from that thrust (motor and propeller), and the spin-up lag and
reaction torque from its size.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from robotforge.design.builder.expand import Placed
from robotforge.design.templates.base import DARK, GEOM
from robotforge.design.visuals import Visuals, aero
from robotforge.design.visuals import parts as P

DISC_LOADING = 160.0  # N of thrust per m^2 of propeller disc, full throttle
MASS_PER_NEWTON = 0.006  # kg of motor per N of thrust, plus PROP_MASS
PROP_MASS = 0.008  # kg
TORQUE_PER_THRUST = 0.065  # reaction torque per newton of thrust, per meter of propeller diameter
SPIN_UP = (0.015, 0.06)  # s: lag = a + b * diameter
PROP_COLOR = [0.13, 0.14, 0.16, 1.0]
BELL_COLOR = [0.5, 0.52, 0.56, 1.0]
CLEARANCE = 0.006  # m between the top of the motor and the propeller's plane


def motor_size(diameter: float) -> tuple[float, float]:
    """(radius, height) of the motor that turns a propeller this big."""
    radius = float(np.clip(0.11 * diameter, 0.008, 0.05))
    return radius, 1.05 * radius


def height(diameter: float) -> float:
    """From the part's frame (the motor's foot) up to just above the propeller."""
    return motor_size(diameter)[1] + CLEARANCE + 0.004


def default_thrust(diameter: float) -> float:
    return DISC_LOADING * math.pi * (diameter / 2) ** 2


def build(spec: mujoco.MjSpec, body: mujoco.MjsBody, part: Placed, vis: Visuals) -> None:
    """The rotor's site, thrust actuator, propeller disc (it collides and carries the mass) and looks."""
    p = part.spec
    diameter = p.size[0]
    thrust = p.thrust if p.thrust is not None else default_thrust(diameter)
    mass = p.mass if p.mass is not None else MASS_PER_NEWTON * thrust + PROP_MASS
    mirrored = np.linalg.det(part.local) < 0
    spin = (1 if p.spin == "ccw" else -1) * part.flip * (-1 if mirrored else 1)
    radius, motor_h = motor_size(diameter)
    foot = part.local @ np.asarray(p.offset, float)
    axis = part.local @ np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    hub = foot + axis * (motor_h + CLEARANCE)
    turn = _z_to(axis)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(turn, float))
    rot = rot.reshape(3, 3)
    body.add_site(name=part.name, pos=hub.tolist(), quat=turn, size=[0.004, 0, 0], rgba=[0, 0, 0, 0])
    disc = body.add_geom(
        type=GEOM.mjGEOM_CYLINDER,
        size=[diameter / 2, 0.003, 0],
        pos=hub.tolist(),
        quat=turn,
        mass=mass,
        rgba=[0.2, 0.2, 0.2, 0.3],
    )
    vis.hide(disc)
    vis.add(body, "dark", DARK, aero.stator_mount(radius * 1.08, motor_h * 0.4), foot + axis * motor_h * 0.4, rot)
    for preset, mesh in aero.bell(radius, motor_h * 0.6).items():
        color = BELL_COLOR if preset == "metal" else DARK
        vis.add(body, preset, color, mesh, foot + axis * motor_h, rot)
    for preset, mesh in aero.propeller(diameter, spin).items():
        if preset == "blades":
            vis.add(body, "shell", PROP_COLOR, mesh, hub, rot)
        else:  # the spinner nut: silver on counter-clockwise propellers, black on clockwise ones
            vis.add(body, "metal", P.METAL if spin > 0 else DARK, mesh, hub, rot)
    act = spec.add_actuator(name=part.name, target=part.name, trntype=mujoco.mjtTrn.mjTRN_SITE)
    act.set_to_motor()
    # The propeller turning one way twists the body the other way: the reaction torque (gear is in the site frame).
    act.gear = [0, 0, 1, 0, 0, -spin * TORQUE_PER_THRUST * diameter]
    act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
    act.dynprm = [SPIN_UP[0] + SPIN_UP[1] * diameter] + [0.0] * 9
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [0.0, float(thrust)]


def _z_to(direction: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_quatZ2Vec(q, np.asarray(direction, float))
    return q.tolist()
