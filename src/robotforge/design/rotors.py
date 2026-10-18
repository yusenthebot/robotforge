"""Real rotors for a library aircraft whose model drives its body some other way.

Menagerie's Crazyflie 2 pushes and turns its body with one thrust actuator and three moment actuators whose
limits its README calls arbitrary: 0.00001 N*m of roll, about a thousandth of what its real rotors make, so it
hovers but tumbles out of the sky as soon as it moves (measured: the first leg of a 2 m square). Given here
instead are the four rotors the real one has, with the numbers Bitcraze's own firmware flies it with, and the
flight controller works them out like any other layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class RotorLayout:
    names: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]  # m, propeller hubs in the root body frame
    spins: tuple[int, ...]  # +1 counter-clockwise seen from above, -1 clockwise
    thrust: float  # N each at full throttle
    torque_per_thrust: float  # m: reaction torque per newton of thrust
    lag: float  # s, the motors' spin-up time constant
    source: str  # where the numbers come from, for the note


# crazyflie-firmware power_distribution_quadrotor.c: arm length 0.046 m (diagonal), thrust to torque 0.005964552 m,
# full-throttle thrust per motor pwmToThrustA + pwmToThrustB = 0.0915 + 0.0677 N; motor 1 front right. Hubs where
# the model's propellers are drawn (12.9 mm above its body frame).
CRAZYFLIE = RotorLayout(
    names=("m1", "m2", "m3", "m4"),
    positions=(
        (0.0325, -0.0325, 0.0129),
        (-0.0325, -0.0325, 0.0129),
        (-0.0325, 0.0325, 0.0129),
        (0.0325, 0.0325, 0.0129),
    ),
    spins=(1, -1, 1, -1),
    thrust=0.159,
    torque_per_thrust=0.005964552,
    lag=0.02,
    source="Bitcraze's firmware: 46 mm arms, 0.16 N per motor",
)


def fit(spec: mujoco.MjSpec, layout: RotorLayout) -> str:
    """Replace the model's actuators with the layout's rotors; returns a note saying so."""
    root = spec.worldbody.first_body()
    replaced = [a.name for a in spec.actuators]
    for act in list(spec.actuators):
        spec.delete(act)
    for name, pos, spin in zip(layout.names, layout.positions, layout.spins, strict=True):
        root.add_site(name=name, pos=list(pos), size=[0.003, 0, 0], rgba=[0, 0, 0, 0])
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_SITE)
        act.set_to_motor()
        act.gear = [0, 0, 1, 0, 0, -spin * layout.torque_per_thrust]
        act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        act.dynprm = [layout.lag] + [0.0] * 9
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [0.0, layout.thrust]
    return (
        f"its {len(layout.names)} rotors ({', '.join(layout.names)}) stand in for the model's {', '.join(replaced)} "
        f"actuators, whose limits are placeholders ({layout.source})"
    )
