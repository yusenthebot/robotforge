"""Rebuild a policy's observation vector from the live MuJoCo state.

The policy card names each joint the way the training robot did ("FL_hip_joint").
`JointBinding` finds those joints on a robotforge robot ("go2/FL_hip_joint"), through an
explicit rename map when the names differ, and refuses to bind when anything does not
line up (a missing joint, a joint without an actuator, a default pose outside the limits):
a silently mis-ordered joint makes a policy flail, so it must never happen quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.bridge.spec import ObsTerm, PolicySpec, SpecError
from robotforge.sim.actuators import drives_joint

GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class JointBinding:
    """Policy joint i -> this model's joint, its qpos/dof address and the actuator that drives it."""

    names: tuple[str, ...]  # model joint names, in the card's order
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuators: np.ndarray
    root_body: int
    root_qpos: int
    root_dof: int

    @classmethod
    def bind(
        cls, model: mujoco.MjModel, robot: str, spec: PolicySpec, joint_map: dict[str, str] | None = None
    ) -> JointBinding:
        joint_map = joint_map or {}
        unknown = sorted(set(joint_map) - set(spec.joints))
        if unknown:
            raise SpecError(f"joint_map renames joints the policy does not use: {', '.join(unknown)}")
        prefix = f"{robot}/" if robot else ""
        names, missing, hinges_only = [], [], []
        for joint in spec.joints:
            name = prefix + joint_map.get(joint, joint)
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                missing.append(name)
            elif int(model.jnt_type[j]) != mujoco.mjtJoint.mjJNT_HINGE:
                hinges_only.append(name)
            names.append(name)
        if missing:
            raise SpecError(
                f"policy {spec.name!r} needs joints {robot!r} does not have: {', '.join(missing)}. "
                "Pass joint_map to rename them if the robot calls them something else."
            )
        if hinges_only:
            raise SpecError(f"policy {spec.name!r} drives hinge joints, but these are not hinges: {hinges_only}")
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
        by_joint = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}
        undriven = [n for n, j in zip(names, ids, strict=True) if int(j) not in by_joint]
        if undriven:
            raise SpecError(f"these joints have no actuator, so the policy cannot move them: {', '.join(undriven)}")
        roots = {_root_of(model, int(model.jnt_bodyid[j])) for j in ids}
        if len(roots) != 1:
            raise SpecError("the policy's joints belong to more than one robot")
        root = roots.pop()
        free = [
            j
            for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])
            if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE
        ]
        if not free:
            raise SpecError("this policy is for a free-moving robot (a walker), but the robot's base is fixed")
        _check_default_pose(model, ids, names, spec)
        return cls(
            names=tuple(names),
            qpos_adr=model.jnt_qposadr[ids].copy(),
            dof_adr=model.jnt_dofadr[ids].copy(),
            actuators=np.array([by_joint[int(j)] for j in ids]),
            root_body=root,
            root_qpos=int(model.jnt_qposadr[free[0]]),
            root_dof=int(model.jnt_dofadr[free[0]]),
        )


def _root_of(model: mujoco.MjModel, body: int) -> int:
    while int(model.body_parentid[body]) != 0:
        body = int(model.body_parentid[body])
    return body


def _check_default_pose(model: mujoco.MjModel, ids: np.ndarray, names: list[str], spec: PolicySpec) -> None:
    """The policy's default pose must be reachable, or the joint order or units are wrong."""
    bad = []
    for j, name, q in zip(ids, names, spec.default_joint_pos, strict=True):
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            if not lo - 1e-6 <= q <= hi + 1e-6:
                bad.append(f"{name}={q:.3f} (limits {lo:.3f}..{hi:.3f})")
    if bad:
        raise SpecError(
            f"policy {spec.name!r}: its default pose is outside these joints' limits: {'; '.join(bad)}. "
            "The joint order or the joint_map is probably wrong."
        )


class Observer:
    """Computes the observation vector of a policy card for one robot."""

    def __init__(self, model: mujoco.MjModel, spec: PolicySpec, binding: JointBinding, robot: str):
        self.spec = spec
        self.binding = binding
        self.default_pos = np.asarray(spec.default_joint_pos, float)
        self.site = -1
        if spec.base_site:
            prefix = f"{robot}/" if robot else ""
            self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + spec.base_site)
            if self.site < 0:
                raise SpecError(
                    f"policy {spec.name!r} measures the base at site {spec.base_site!r}, which {robot!r} does not "
                    "have; add the site or set base_site to null to use the root body"
                )
        self._history: list[np.ndarray | None] = [None] * len(spec.observations)
        self.phase = 0.0

    def reset(self) -> None:
        self._history = [None] * len(self.spec.observations)
        self.phase = 0.0

    def advance_phase(self, moving: bool) -> None:
        """Move the gait clock one policy step forward (after each observation)."""
        for term in self.spec.observations:
            if term.term == "gait_phase" and (moving or term.always_advance):
                self.phase = (self.phase + self.spec.policy_dt * float(term.frequency or 0.0)) % 1.0
                return

    def build(
        self, model: mujoco.MjModel, data: mujoco.MjData, last_action: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        """The observation for the current state, as float32 of shape (obs_dim,)."""
        parts = []
        for i, term in enumerate(self.spec.observations):
            value = self._term(model, data, term, last_action, command)
            value = value * np.asarray(term.scale, float)
            if term.clip is not None:
                value = np.clip(value, -term.clip, term.clip)
            if term.history > 1:
                past = self._history[i]
                if past is None:  # first step: the history is the present repeated, as at a reset
                    past = np.tile(value, (term.history, 1))
                past = np.vstack([past[1:], value])
                self._history[i] = past
                value = past.reshape(-1)
            parts.append(value)
        return np.concatenate(parts).astype(np.float32)

    def _term(
        self, model: mujoco.MjModel, data: mujoco.MjData, term: ObsTerm, last_action: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        b = self.binding
        name = term.term
        if name == "joint_pos_rel":
            return data.qpos[b.qpos_adr] - self.default_pos
        if name == "joint_vel":
            return data.qvel[b.dof_adr].copy()
        if name == "last_action":
            return np.asarray(last_action, float).copy()
        if name == "velocity_command":
            return np.asarray(command, float).copy()
        if name == "gait_phase":
            cycles = np.mod(self.phase + np.asarray(term.offsets, float), 1.0)
            return cycles * (2 * np.pi if term.unit == "radian" else 1.0)
        rot = self._base_rotation(data)
        if name == "projected_gravity":
            return rot.T @ GRAVITY_DIR
        if name == "neg_up_axis_world":
            return -rot[:, 2]
        vel = np.zeros(6)  # [angular, linear] in the base frame
        if self.site >= 0:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, self.site, vel, 1)
        else:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b.root_body, vel, 1)
        if name == "base_ang_vel":
            return vel[:3]
        if name == "base_lin_vel":
            return vel[3:]
        raise SpecError(f"observation term {name!r} is not supported")  # pragma: no cover (pydantic guards it)

    def _base_rotation(self, data: mujoco.MjData) -> np.ndarray:
        if self.site >= 0:
            return data.site_xmat[self.site].reshape(3, 3)
        return data.xmat[self.binding.root_body].reshape(3, 3)
