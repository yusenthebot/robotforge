"""What an actuator of a compiled model does with its control signal."""

from __future__ import annotations

import mujoco
import numpy as np


def drives_joint(model: mujoco.MjModel, a: int) -> bool:
    return int(model.actuator_trntype[a]) == mujoco.mjtTrn.mjTRN_JOINT


def is_torque_motor(model: mujoco.MjModel, a: int) -> bool:
    """A plain motor: the control is a force or torque, with no feedback inside the actuator."""
    return (
        int(model.actuator_biastype[a]) == mujoco.mjtBias.mjBIAS_NONE
        and int(model.actuator_gaintype[a]) == mujoco.mjtGain.mjGAIN_FIXED
        and int(model.actuator_dyntype[a]) == mujoco.mjtDyn.mjDYN_NONE
    )


def is_position_servo(model: mujoco.MjModel, a: int) -> bool:
    """A position servo: the control is a target position (force = kp * (ctrl - q) - kv * qdot).

    Velocity actuators also have an affine bias, but without the -kp * q term.
    """
    kp = float(model.actuator_gainprm[a, 0])
    return (
        int(model.actuator_gaintype[a]) == mujoco.mjtGain.mjGAIN_FIXED
        and int(model.actuator_biastype[a]) == mujoco.mjtBias.mjBIAS_AFFINE
        and kp > 0
        and bool(np.isclose(model.actuator_biasprm[a, 1], -kp))
    )


def is_velocity_servo(model: mujoco.MjModel, a: int) -> bool:
    """A velocity servo: the control is a target speed (force = kv * (ctrl - qdot)), as on driven wheels."""
    kv = float(model.actuator_gainprm[a, 0])
    return (
        int(model.actuator_gaintype[a]) == mujoco.mjtGain.mjGAIN_FIXED
        and int(model.actuator_biastype[a]) == mujoco.mjtBias.mjBIAS_AFFINE
        and kv > 0
        and bool(np.isclose(model.actuator_biasprm[a, 2], -kv))
        and model.actuator_biasprm[a, 1] == 0
    )
