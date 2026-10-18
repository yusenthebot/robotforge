"""Turning what an algorithm wants into actuator commands.

Two translations every controller needs:

- `ActionScale`: learning agents and planners like actions in [-1, 1]; actuators
  want values in their own ctrlrange (a servo angle, a wheel speed, a force).
- `TorqueMap`: classic controllers (PD with gravity compensation, LQR, operational
  space) compute joint torques, but our robots mostly have position servos. Each
  MuJoCo actuator is affine, force = gain * ctrl + bias(length, velocity), so the
  ctrl that produces a wanted torque can be solved exactly. For a position servo
  that is ctrl = q + torque / kp; for a motor it is ctrl = torque / gear.
"""

from __future__ import annotations

import mujoco
import numpy as np


class ActionScale:
    """Maps [-1, 1] to each actuator's ctrlrange (actuators without a range use +-1)."""

    def __init__(self, model: mujoco.MjModel, actuators: np.ndarray):
        self.actuators = np.asarray(actuators, dtype=int)
        limited = model.actuator_ctrllimited[self.actuators].astype(bool)
        rng = model.actuator_ctrlrange[self.actuators]
        self.low = np.where(limited, rng[:, 0], -1.0)
        self.high = np.where(limited, rng[:, 1], 1.0)
        self.mid = (self.low + self.high) / 2
        self.half = (self.high - self.low) / 2

    def to_ctrl(self, action: np.ndarray) -> np.ndarray:
        return self.mid + np.clip(action, -1.0, 1.0) * self.half

    def clip(self, ctrl: np.ndarray) -> np.ndarray:
        return np.clip(ctrl, self.low, self.high)


class TorqueMap:
    """Joint torques -> ctrl, for actuators that drive one hinge or slide joint directly."""

    def __init__(self, model: mujoco.MjModel, actuators: np.ndarray):
        self.actuators = np.asarray(actuators, dtype=int)
        one_dof = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        bad = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)
            for a in self.actuators
            if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT)
            or int(model.actuator_dyntype[a]) != int(mujoco.mjtDyn.mjDYN_NONE)
            or int(model.jnt_type[model.actuator_trnid[a, 0]]) not in one_dof
        ]
        if bad:
            raise ValueError(
                f"torque control needs actuators that drive a single joint directly; these do not: {', '.join(bad)}"
            )
        joints = model.actuator_trnid[self.actuators, 0]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.gear = model.actuator_gear[self.actuators, 0].copy()
        self.gain = model.actuator_gainprm[self.actuators, :3].copy()
        self.affine_gain = model.actuator_gaintype[self.actuators] == mujoco.mjtGain.mjGAIN_AFFINE
        self.bias = model.actuator_biasprm[self.actuators, :3].copy()
        self.has_bias = model.actuator_biastype[self.actuators] == mujoco.mjtBias.mjBIAS_AFFINE
        self.scale = ActionScale(model, self.actuators)
        self.force_limited = model.actuator_forcelimited[self.actuators].astype(bool)
        self.force_range = model.actuator_forcerange[self.actuators].copy()
        # Learned walkers solve this every physics step (2 ms), where each small numpy call counts. With fixed gains
        # and every actuator biased or none (servos, motors), the parts that never change are worked out once; the
        # numbers are the same, computed in the same order.
        self._fixed = not self.affine_gain.any() and (self.has_bias.all() or not self.has_bias.any())
        gain = self.gain[:, 0] + 0.0
        self._safe_gain = np.where(np.abs(gain) > 1e-9, gain, 1e-9)
        self._biased = bool(self.has_bias.any())
        self._b0, self._b1, self._b2 = (self.bias[:, k].copy() for k in range(3))

    def ctrl(self, data: mujoco.MjData, torque: np.ndarray) -> np.ndarray:
        """The ctrl values that make each actuator apply `torque` to its joint right now (clipped)."""
        if self._fixed:
            force = np.asarray(torque, float) / self.gear
            if self._biased:
                length = self.gear * data.qpos[self.qadr]
                velocity = self.gear * data.qvel[self.dadr]
                force = force - (self._b0 + self._b1 * length + self._b2 * velocity)
            return np.minimum(np.maximum(force / self._safe_gain, self.scale.low), self.scale.high)
        length = self.gear * data.qpos[self.qadr]
        velocity = self.gear * data.qvel[self.dadr]
        gain = self.gain[:, 0] + np.where(self.affine_gain, self.gain[:, 1] * length + self.gain[:, 2] * velocity, 0)
        bias = np.where(self.has_bias, self.bias[:, 0] + self.bias[:, 1] * length + self.bias[:, 2] * velocity, 0.0)
        force = np.asarray(torque, float) / self.gear
        safe_gain = np.where(np.abs(gain) > 1e-9, gain, 1e-9)
        return self.scale.clip((force - bias) / safe_gain)
