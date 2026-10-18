"""Run a learned walking policy (ONNX) as a robotforge controller.

The policy runs at its own rate (e.g. 50 Hz); between its steps a joint PD loop runs at
physics rate with the gains the policy was trained with. The robot's own actuators may be
different (robotforge's Go2 has stiffer position servos than UniLab's), so the PD output is
turned into whatever control signal the actuators take (`TorqueMap`): the policy feels the
joints it was trained on, whatever motors the robot has.

Small policies (MLPs, like the shipped Go2 walk) run with numpy alone (`onnx_numpy`); onnxruntime,
the optional `policy` extra, is used when it is installed and needed for anything bigger.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.bridge.observations import JointBinding, Observer
from robotforge.bridge.spec import PolicySpec, SpecError
from robotforge.control.actuation import TorqueMap

log = logging.getLogger(__name__)
INSTALL_HINT = "install the optional extra: uv sync --extra policy (or pip install 'robotforge[policy]')"
# A walker that tips past this (cosine of the tilt from upright) has fallen; the policy is then
# out of its training distribution, so it is paused rather than left to thrash.
FALLEN_UPRIGHTNESS = 0.3
# Going straight, a learned policy's small turning error adds up (the 1-hour G1 policy drifts ~6 deg/s at
# 0.5 m/s with no turn asked). Like Isaac Lab's heading commands, a turn of this many rad/s per radian
# of heading error is asked instead, so "straight" stays straight.
HEADING_GAIN = 1.5
# Standing still: a learned policy told "speed 0" stands, but not perfectly; the G1's drifted 6.5 cm/s and walked
# into the next lane within half a minute. An idle walker remembers where it stood and walks back once it has
# drifted HOLD_START away, faster than the policy's stand threshold (or it would just stand), until HOLD_DONE.
HOLD_START = 0.2  # m
HOLD_DONE = 0.05  # m
HOLD_GAIN = 1.0  # m/s per m of drift
HOLD_TOP = 0.35  # m/s


class PolicyUnavailable(RuntimeError):
    """The model cannot be run (e.g. it needs onnxruntime, which is missing); the message says what to do."""


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


class OnnxPolicy:
    """A callable obs -> action: onnxruntime when installed (CPU, one thread), otherwise numpy."""

    def __init__(self, path: Path, input_name: str = "obs", output_name: str | None = None):
        try:
            import onnxruntime as ort
        except ImportError:
            self._numpy(path, input_name, output_name)
            return
        self.runtime = "onnxruntime"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3  # errors only: warnings would go to stderr of the MCP server
        try:
            self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime raises its own exception types for unreadable models
            raise PolicyUnavailable(f"onnxruntime could not load {path}: {exc}") from exc
        inputs = {i.name: i for i in self.session.get_inputs()}
        if input_name not in inputs:
            raise PolicyUnavailable(f"{path.name} has no input {input_name!r}; its inputs: {', '.join(inputs)}")
        outputs = [o.name for o in self.session.get_outputs()]
        if output_name is not None and output_name not in outputs:
            raise PolicyUnavailable(f"{path.name} has no output {output_name!r}; its outputs: {', '.join(outputs)}")
        self.input_name = input_name
        self.output_name = output_name or outputs[0]
        self.input_dim = _last_dim(inputs[input_name].shape)
        self.output_dim = _last_dim(next(o for o in self.session.get_outputs() if o.name == self.output_name).shape)

    def _numpy(self, path: Path, input_name: str, output_name: str | None) -> None:
        from robotforge.bridge.onnx_numpy import NumpyOnnx, UnsupportedModel

        try:
            model = NumpyOnnx(path)
        except UnsupportedModel as exc:
            raise PolicyUnavailable(f"{path.name}: {exc}; {INSTALL_HINT}") from None
        except OSError as exc:
            raise PolicyUnavailable(f"could not read {path}: {exc}") from exc
        if input_name not in model.inputs:
            raise PolicyUnavailable(f"{path.name} has no input {input_name!r}; its inputs: {', '.join(model.inputs)}")
        if output_name is not None and output_name not in model.outputs:
            outputs = ", ".join(model.outputs)
            raise PolicyUnavailable(f"{path.name} has no output {output_name!r}; its outputs: {outputs}")
        self.runtime = "numpy"
        self.model = model
        self.input_name = input_name
        self.output_name = output_name or model.outputs[0]
        self.input_dim = _last_dim(model.shapes.get(input_name, []))
        self.output_dim = _last_dim(model.shapes.get(self.output_name, []))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs, np.float32).reshape(1, -1)
        if self.runtime == "numpy":
            return np.asarray(self.model(batch), float).reshape(-1)
        return np.asarray(self.session.run([self.output_name], {self.input_name: batch})[0], float).reshape(-1)


def _last_dim(shape: list[Any]) -> int | None:
    return shape[-1] if shape and isinstance(shape[-1], int) else None


def load_onnx(spec: PolicySpec, directory: Path) -> OnnxPolicy:
    """Load the card's model and check its width against the card."""
    policy = OnnxPolicy(directory / spec.onnx, spec.input_name, spec.output_name)
    if policy.input_dim is not None and policy.input_dim != spec.obs_dim:
        raise SpecError(
            f"{spec.name}: the model reads {policy.input_dim} observation values but the card describes "
            f"{spec.obs_dim}; the card does not match this model"
        )
    if policy.output_dim is not None and policy.output_dim != len(spec.joints):
        raise SpecError(f"{spec.name}: the model outputs {policy.output_dim} actions for {len(spec.joints)} joints")
    return policy


class PolicyWalker:
    """Walks a robot with a learned velocity-tracking policy. Same interface as the trot gait:
    `command(speed, turn)`, `max_speed`, `status()`, so move/stop/WASD work unchanged."""

    def __init__(
        self,
        model: mujoco.MjModel,
        robot: str,
        spec: PolicySpec,
        policy: Callable[[np.ndarray], np.ndarray],
        joint_map: dict[str, str] | None = None,
    ):
        self.robot = robot
        self.spec = spec
        self.policy = policy
        self.binding = JointBinding.bind(model, robot, spec, joint_map)
        self.observer = Observer(model, spec, self.binding, robot)
        self.torque = TorqueMap(model, self.binding.actuators)
        n = len(spec.joints)
        self.default_pos = np.asarray(spec.default_joint_pos, float)
        self.scale = np.broadcast_to(np.asarray(spec.action_scale, float), (n,)).copy()
        self.kp = np.asarray(spec.pd.kp, float)
        self.kd = np.asarray(spec.pd.kd, float)
        # The model's joints may be damped differently from the training robot's; the PD loop makes
        # up the difference so the policy feels the damping it learned with.
        model_damping = model.dof_damping[self.binding.dof_adr]
        trained = model_damping if spec.pd.joint_damping is None else np.asarray(spec.pd.joint_damping, float)
        self.extra_damping = trained - model_damping
        self.torque_limit = None if spec.pd.torque_limit is None else np.asarray(spec.pd.torque_limit, float)
        self._extra = bool(self.extra_damping.any())  # none to add: the PD loop skips that step (it runs every 2 ms)
        self.rate_hz = 1.0 / model.opt.timestep  # the PD loop runs every physics step
        ranges = spec.commands
        self.max_speed = float(ranges.lin_vel_x[1])
        self.max_reverse = float(-ranges.lin_vel_x[0])
        self.max_turn = float(max(abs(ranges.ang_vel_z[0]), abs(ranges.ang_vel_z[1])))
        self.command_speed = 0.0
        self.command_turn = 0.0
        self.heading: float | None = None  # held while no turn is asked (set at the next policy step)
        self.anchor: np.ndarray | None = None  # where it stood when it was last told to stop (set at the next step)
        self.returning = False  # walking back to the anchor
        root_dof = int(model.body_dofadr[self.binding.root_body])
        self.root_dofs = slice(root_dof, root_dof + 6) if root_dof >= 0 else slice(0, 0)
        self.last_action = np.zeros(n)
        self.targets = self.default_pos.copy()
        self._next_step = -math.inf
        self.steps = 0
        self.paused_reason: str | None = None

    @property
    def fallen(self) -> bool:
        return bool(self.paused_reason and self.paused_reason.startswith("fallen"))

    def limit_speed(self, top: float) -> None:
        """Lower the fastest accepted forward speed (m/s); kept for parity with the gait."""
        self.max_speed = min(self.max_speed, top)
        self.command(self.command_speed, self.command_turn)

    def command(self, speed: float = 0.0, turn: float = 0.0) -> None:
        """speed in m/s (negative = backwards), turn in rad/s (positive = left), clipped to what it learned."""
        r = self.spec.commands
        self.command_speed = float(np.clip(speed, max(r.lin_vel_x[0], -self.max_reverse), self.max_speed))
        self.command_turn = float(np.clip(turn, r.ang_vel_z[0], r.ang_vel_z[1]))
        if self.command_turn != 0.0:
            self.heading = None  # turning on purpose; a new heading is taken when the turn stops
        if self.command_speed != 0.0 or self.command_turn != 0.0:
            self.anchor = None  # moving on purpose; where it stops next becomes the place to hold
            self.returning = False

    def status(self) -> dict[str, Any]:
        out = {
            "behavior": "walk",
            "walker": "rl",
            "policy": self.spec.name,
            "robot": self.robot,
            "speed": round(self.command_speed, 3),
            "turn": round(self.command_turn, 3),
            "max_speed": round(self.max_speed, 3),
        }
        if self.paused_reason:
            out["paused"] = self.paused_reason
        return out

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        time_went_back = data.time < self._next_step - 2 * self.spec.policy_dt  # e.g. the world was reset
        if time_went_back:
            self._restart()
        if data.time + 1e-9 >= self._next_step:
            self._policy_step(model, data)
            self._next_step = (data.time if self._next_step == -math.inf else self._next_step) + self.spec.policy_dt
        self._apply_pd(data)

    def _restart(self) -> None:
        self.heading = None
        self.anchor = None
        self.returning = False
        self.observer.reset()
        self.last_action = np.zeros_like(self.last_action)
        self.targets = self.default_pos.copy()
        self._next_step = -math.inf
        self.paused_reason = None

    def _policy_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._fallen(data):
            # Out of distribution: hold the default pose and wait to be picked up (dragged upright or reset).
            self.paused_reason = "fallen over; drag it upright or reset the world to walk again"
            self.targets = self.default_pos.copy()
            return
        if self.paused_reason:
            self._restart()
        command = np.array([*self._planar(data), self._turn(data)])
        if math.hypot(command[0], command[1]) <= self.spec.commands.planar_dead_zone:  # as trained: too slow, stand
            command[:2] = 0.0
        obs = self.observer.build(model, data, self.last_action, command)
        try:
            action = np.asarray(self.policy(obs), float).reshape(-1)
        except Exception as exc:  # onnxruntime raises its own types; the sim loop must survive any of them
            log.warning("policy %s failed: %s", self.spec.name, exc)
            self.paused_reason = f"the policy failed to run ({type(exc).__name__}); holding the default pose"
            self.targets = self.default_pos.copy()
            return
        if action.shape != self.last_action.shape or not np.all(np.isfinite(action)):
            self.paused_reason = "the policy returned an invalid action; holding the default pose"
            self.targets = self.default_pos.copy()
            return
        self.last_action = action
        clip = self.spec.action_clip
        clipped = action if clip is None else np.clip(action, -clip, clip)
        self.targets = self.default_pos + self.scale * clipped
        self.observer.advance_phase(moving=bool(np.any(np.abs(command) > 1e-3)))
        self.steps += 1

    def _planar(self, data: mujoco.MjData) -> tuple[float, float]:
        """Forward and sideways speed to ask for: the commanded walk, or idle, a walk back to where it stood."""
        if self.command_speed != 0.0 or self.command_turn != 0.0:
            return self.command_speed, 0.0
        here = data.xpos[self.binding.root_body][:2].copy()
        if np.any(data.qfrc_applied[self.root_dofs]):  # a hand in the viewer is moving it: hold where it is let go
            self.anchor = None
            self.returning = False
        if self.anchor is None:
            self.anchor = here
            return 0.0, 0.0
        off = self.anchor - here
        dist = float(np.linalg.norm(off))
        if self.returning and dist < HOLD_DONE:
            self.returning = False
        elif not self.returning and dist > HOLD_START:
            self.returning = True
        if not self.returning:
            return 0.0, 0.0
        rot = data.xmat[self.binding.root_body]
        yaw = math.atan2(float(rot[3]), float(rot[0]))
        c, s = math.cos(yaw), math.sin(yaw)
        ahead, left = c * off[0] + s * off[1], -s * off[0] + c * off[1]  # the way back, in the robot's frame
        r = self.spec.commands
        speed = float(np.clip(HOLD_GAIN * dist, r.planar_dead_zone + 0.05, HOLD_TOP))
        vx = float(np.clip(ahead / dist * speed, r.lin_vel_x[0], r.lin_vel_x[1]))
        vy = float(np.clip(left / dist * speed, r.lin_vel_y[0], r.lin_vel_y[1]))
        return vx, vy

    def _turn(self, data: mujoco.MjData) -> float:
        """The turn rate to ask the policy for: the commanded one, or with none, a hold on the heading."""
        if self.command_turn != 0.0:
            return self.command_turn
        rot = data.xmat[self.binding.root_body]
        yaw = math.atan2(float(rot[3]), float(rot[0]))
        if self.heading is None:
            self.heading = yaw
        error = math.atan2(math.sin(self.heading - yaw), math.cos(self.heading - yaw))
        r = self.spec.commands
        return float(np.clip(HEADING_GAIN * error, r.ang_vel_z[0], r.ang_vel_z[1]))

    def _fallen(self, data: mujoco.MjData) -> bool:
        return float(data.xmat[self.binding.root_body][8]) < FALLEN_UPRIGHTNESS

    def _apply_pd(self, data: mujoco.MjData) -> None:
        q = data.qpos[self.binding.qpos_adr]
        qd = data.qvel[self.binding.dof_adr]
        tau = self.kp * (self.targets - q) - self.kd * qd
        if self.torque_limit is not None:  # the training motors saturate here, before the passive damping
            tau = np.minimum(np.maximum(tau, -self.torque_limit), self.torque_limit)
        if self._extra:
            tau = tau - self.extra_damping * qd
        data.ctrl[self.binding.actuators] = self.torque.ctrl(data, tau)


def walker_for(
    model: mujoco.MjModel,
    robot: str,
    library_robot: str | None,
    choice: str = "rl",
    joint_map: dict[str, str] | None = None,
) -> PolicyWalker:
    """A PolicyWalker for `robot`: choice "rl" picks the policy made for its library model, any
    other value names an installed policy (see `robotforge.bridge.policies`)."""
    from robotforge.bridge import policies

    if choice == "rl":
        found = policies.for_robot(library_robot)
        if found is None:
            names = ", ".join(sorted(policies.all_policies())) or "none"
            what = f"the library robot {library_robot!r}" if library_robot else "robots that are not from the library"
            raise SpecError(
                f"no learned walk fits {what} (installed policies: {names}); walk with the built-in gait "
                "(walker='gait') or train one with unilab_train"
            )
    else:
        found = policies.load(choice)
    policy = load_onnx(found.spec, found.directory)
    return PolicyWalker(model, robot, found.spec, policy, joint_map)
