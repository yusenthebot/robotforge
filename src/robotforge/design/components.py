"""Real-world parts that templates build from, so simulated robots behave like buildable ones.

Actuator numbers are approximate public datasheet values at the listed voltage.
They set torque limits, speed-related damping and mass; they are not a guarantee
that a printed robot will match the simulation exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Actuator:
    name: str
    kind: str  # "servo" (position control) or "motor" (velocity control, e.g. wheels)
    stall_torque: float  # N*m
    no_load_speed: float  # rad/s
    mass: float  # kg
    size: tuple[float, float, float]  # case dimensions in meters (x, y, z)
    armature: float  # reflected rotor inertia, kg*m^2
    note: str

    @property
    def kp(self) -> float:
        """Position gain: reaches stall torque at ~8.6 degrees of error, like a stiff hobby servo."""
        return self.stall_torque / 0.15

    @property
    def damping(self) -> float:
        """Joint damping so peak speed under stall torque roughly matches the no-load speed."""
        return self.stall_torque / self.no_load_speed


ACTUATORS: dict[str, Actuator] = {
    a.name: a
    for a in [
        Actuator("sg90", "servo", 0.18, 10.5, 0.009, (0.0222, 0.0118, 0.031), 0.0002, "micro servo, 4.8 V"),
        Actuator("mg996r", "servo", 1.08, 7.5, 0.055, (0.0407, 0.0197, 0.0429), 0.002, "metal-gear servo, 6 V"),
        Actuator(
            "sts3215",
            "servo",
            1.91,
            4.7,
            0.055,
            (0.0452, 0.0247, 0.035),
            0.004,
            "Feetech bus servo, 7.4 V (LeRobot SO-100 arm)",
        ),
        Actuator("xl430", "servo", 1.5, 6.0, 0.057, (0.0285, 0.0465, 0.034), 0.003, "Dynamixel XL430-W250, 12 V"),
        Actuator(
            "bldc_joint",
            "servo",
            12.0,
            30.0,
            0.35,
            (0.06, 0.06, 0.04),
            0.01,
            "small quadruped joint module (Go1/Go2 class, lower bound)",
        ),
        Actuator("tt_motor", "motor", 0.08, 20.0, 0.03, (0.07, 0.022, 0.018), 0.0001, "yellow TT gear motor, 6 V"),
        Actuator("jga25", "motor", 0.5, 29.0, 0.09, (0.07, 0.025, 0.025), 0.0005, "25 mm metal gear motor, 12 V"),
    ]
}

# kg/m^3 for printed and machined parts
MATERIALS: dict[str, float] = {
    "pla": 1240.0 * 0.4,  # FDM print at ~40 % infill
    "aluminum": 2700.0,
    "carbon": 1600.0,
    "rubber": 1100.0,
}

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.92, 0.3, 0.28, 1),
    "green": (0.3, 0.8, 0.45, 1),
    "blue": (0.3, 0.55, 0.95, 1),
    "yellow": (0.98, 0.8, 0.25, 1),
    "orange": (0.98, 0.55, 0.2, 1),
    "purple": (0.6, 0.4, 0.9, 1),
    "white": (0.92, 0.93, 0.95, 1),
    "black": (0.12, 0.13, 0.15, 1),
    "gray": (0.55, 0.57, 0.6, 1),
    "brown": (0.55, 0.38, 0.25, 1),
    "teal": (0.2, 0.7, 0.7, 1),
    "pink": (0.95, 0.5, 0.7, 1),
}


def actuator(name: str) -> Actuator:
    try:
        return ACTUATORS[name]
    except KeyError:
        raise ValueError(f"unknown actuator {name!r}; choose one of: {', '.join(ACTUATORS)}") from None


def color(value: str | list[float] | tuple[float, ...]) -> list[float]:
    """Accept a color name or an [r, g, b(, a)] list in 0..1."""
    if isinstance(value, str):
        try:
            return list(COLORS[value.lower()])
        except KeyError:
            raise ValueError(f"unknown color {value!r}; use one of {', '.join(COLORS)} or [r, g, b]") from None
    rgba = [float(v) for v in value]
    if len(rgba) == 3:
        rgba.append(1.0)
    if len(rgba) != 4 or not all(0.0 <= v <= 1.0 for v in rgba):
        raise ValueError(f"color must be 3 or 4 numbers in 0..1, got {value!r}")
    return rgba
