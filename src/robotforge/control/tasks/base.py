"""What a task is: an objective for one robot in a world, written once and used by every algorithm.

A task says what the robot should achieve (a cost, lower is better), what it can
sense (an observation vector), when an attempt has failed or succeeded, and how
to start a fresh attempt. The same task object feeds the Gym-style environment
(reward = alive bonus - cost), the sampling MPC planners (cost summed over
predicted futures) and the comparison table.

Costs are vectorized: they read a `Batch` of states shaped [samples, time, ...],
so MPC can score hundreds of predicted futures in one numpy call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


class TaskError(ValueError):
    """User-facing: the message says what to do instead."""


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def readable(exc: ValidationError, what: str) -> str:
    """A pydantic error as one line a beginner (or an agent) can act on."""
    parts = [f"{'.'.join(str(x) for x in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors()]
    return f"{what}: {'; '.join(parts)}"


# ---- states in bulk ------------------------------------------------------------------------


@dataclass(frozen=True)
class StateLayout:
    """Where qpos and qvel sit inside a mjSTATE_FULLPHYSICS vector (what mujoco.rollout returns)."""

    size: int
    time: int
    qpos: slice
    qvel: slice

    @classmethod
    def of(cls, model: mujoco.MjModel) -> StateLayout:
        def offset(bit: int) -> int:
            # mj_getState writes components in increasing bit order.
            return mujoco.mj_stateSize(model, int(FULLPHYSICS) & (bit - 1))

        time_at = offset(int(mujoco.mjtState.mjSTATE_TIME))
        qpos_at = offset(int(mujoco.mjtState.mjSTATE_QPOS))
        qvel_at = offset(int(mujoco.mjtState.mjSTATE_QVEL))
        return cls(
            size=mujoco.mj_stateSize(model, FULLPHYSICS),
            time=time_at,
            qpos=slice(qpos_at, qpos_at + model.nq),
            qvel=slice(qvel_at, qvel_at + model.nv),
        )


@dataclass
class Batch:
    """Many states at once, arrays shaped [samples, time, dim]. One live state is a [1, 1] batch."""

    model: mujoco.MjModel
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    sensordata: np.ndarray

    @classmethod
    def from_data(cls, model: mujoco.MjModel, data: mujoco.MjData) -> Batch:
        def one(x: np.ndarray) -> np.ndarray:
            return np.asarray(x, dtype=float).reshape(1, 1, -1)

        return cls(model, one(data.qpos), one(data.qvel), one(data.ctrl), one(data.sensordata))

    @classmethod
    def from_rollout(
        cls, model: mujoco.MjModel, layout: StateLayout, state: np.ndarray, sensordata: np.ndarray, ctrl: np.ndarray
    ) -> Batch:
        return cls(model, state[..., layout.qpos], state[..., layout.qvel], ctrl, sensordata)

    @property
    def shape(self) -> tuple[int, int]:
        return self.qpos.shape[0], self.qpos.shape[1]

    def last(self) -> Batch:
        """Only the final state of each sample."""
        return Batch(self.model, self.qpos[:, -1:], self.qvel[:, -1:], self.ctrl[:, -1:], self.sensordata[:, -1:])

    def sensor(self, sensor_id: int) -> np.ndarray:
        adr, dim = self.model.sensor_adr[sensor_id], self.model.sensor_dim[sensor_id]
        return self.sensordata[..., adr : adr + dim]

    def site_pos(self, site_id: int, framepos_sensor: int = -1, scratch: mujoco.MjData | None = None) -> np.ndarray:
        """World position of a site in every state. Reads a framepos sensor when the model has one
        (free: MuJoCo computed it during the rollout), otherwise runs forward kinematics per state
        on `scratch` (pass a reusable MjData to avoid allocating one per call)."""
        if framepos_sensor >= 0:
            return self.sensor(framepos_sensor)
        data = scratch if scratch is not None else mujoco.MjData(self.model)
        b, t = self.shape
        out = np.empty((b, t, 3))
        for i in range(b):
            for j in range(t):
                data.qpos[:] = self.qpos[i, j]
                mujoco.mj_kinematics(self.model, data)
                out[i, j] = data.site_xpos[site_id]
        return out


# ---- finding a robot's parts --------------------------------------------------------------


class Robot:
    """Looks up one robot's joints, actuators, sites and sensors (names are prefixed '<robot>/')."""

    def __init__(self, model: mujoco.MjModel, name: str):
        self.model = model
        self.name = name
        self.prefix = f"{name}/"
        if not any(self._names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)):
            robots = sorted({n.split("/", 1)[0] for n in self._all_names(mujoco.mjtObj.mjOBJ_BODY, model.nbody)})
            hint = f" Robots in this world: {', '.join(robots)}." if robots else ""
            raise TaskError(f"no robot named {name!r}.{hint}")

    def _all_names(self, obj: mujoco.mjtObj, count: int) -> list[str]:
        return [n for i in range(count) if "/" in (n := mujoco.mj_id2name(self.model, obj, i) or "")]

    def _names(self, obj: mujoco.mjtObj, count: int) -> list[int]:
        return [i for i in range(count) if (mujoco.mj_id2name(self.model, obj, i) or "").startswith(self.prefix)]

    def find(self, obj: mujoco.mjtObj, name: str) -> int:
        return mujoco.mj_name2id(self.model, obj, self.prefix + name)

    def need(self, obj: mujoco.mjtObj, name: str, what: str) -> int:
        idx = self.find(obj, name)
        if idx < 0:
            raise TaskError(f"robot {self.name!r} has no {name!r}; this task needs {what}")
        return idx

    @property
    def actuators(self) -> np.ndarray:
        return np.array(self._names(mujoco.mjtObj.mjOBJ_ACTUATOR, self.model.nu), dtype=int)

    @property
    def joints(self) -> list[int]:
        """Joints on the robot's bodies. Matched by body, not joint name: imported models often leave
        their free joint unnamed, and an unnamed joint gets no '<robot>/' prefix."""
        bodies = set(self._names(mujoco.mjtObj.mjOBJ_BODY, self.model.nbody))
        return [j for j in range(self.model.njnt) if int(self.model.jnt_bodyid[j]) in bodies]

    def hinge_joints(self) -> list[int]:
        return [j for j in self.joints if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]

    def free_joint(self) -> int:
        for j in self.joints:
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return j
        return -1

    def framepos_sensor(self, site: int) -> int:
        """A framepos sensor attached to `site`, or -1."""
        m = self.model
        for s in range(m.nsensor):
            if (
                m.sensor_type[s] == mujoco.mjtSensor.mjSENS_FRAMEPOS
                and m.sensor_objtype[s] == mujoco.mjtObj.mjOBJ_SITE
                and m.sensor_objid[s] == site
                and m.sensor_reftype[s] == mujoco.mjtObj.mjOBJ_UNKNOWN
            ):
                return s
        return -1

    def dofs(self) -> np.ndarray:
        """Velocity indices (columns of Jacobians, entries of qvel) of all the robot's joints."""
        out: list[int] = []
        for j in self.joints:
            n = {0: 6, 1: 3}.get(int(self.model.jnt_type[j]), 1)
            out.extend(range(self.model.jnt_dofadr[j], self.model.jnt_dofadr[j] + n))
        return np.array(out, dtype=int)


_DIVERGENCE = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)


def divergences(data: mujoco.MjData) -> int:
    """How many times the simulation has blown up so far. MuJoCo resets the state when that
    happens, so the state itself can look fine afterwards; only these counters remember."""
    return sum(int(data.warning[int(w)].number) for w in _DIVERGENCE)


def home_qpos(model: mujoco.MjModel) -> np.ndarray:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_qpos[key].copy() if key >= 0 else model.qpos0.copy()


def home_ctrl(model: mujoco.MjModel) -> np.ndarray:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    return model.key_ctrl[key].copy() if key >= 0 else np.zeros(model.nu)


def smooth_abs(x: np.ndarray, eps: float) -> np.ndarray:
    """|x| with a rounded bottom: steep far away, smooth near zero, so planners can home in precisely."""
    return np.sqrt(x * x + eps * eps) - eps


# ---- the task interface ---------------------------------------------------------------------


class Task(ABC):
    """An objective for one robot. Subclasses fill in the class constants and the abstract methods."""

    name: ClassVar[str]
    summary: ClassVar[str]
    robot_hint: ClassVar[str]  # which robots fit, said for a beginner
    methods: ClassVar[tuple[str, ...]]  # algorithms that fit, in teaching order
    Params: ClassVar[type[TaskParams]]
    default_seconds: ClassVar[float] = 5.0
    control_hz: ClassVar[float] = 50.0  # default decision rate for the environment
    alive_bonus: ClassVar[float] = 0.0  # reward for each second not failed (only for tasks that can fail)
    mpc_defaults: ClassVar[dict[str, Any]] = {}

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        try:
            self.params = self.Params.model_validate(params or {})
        except ValidationError as exc:
            raise TaskError(readable(exc, f"task {self.name!r}")) from exc
        self.model = model
        self.robot = Robot(model, robot)
        self.actuators: np.ndarray = self.robot.actuators
        if self.actuators.size == 0:
            raise TaskError(f"robot {robot!r} has no actuators to control")

    # ---- what the algorithms read ----

    @abstractmethod
    def observation(self, data: mujoco.MjData) -> np.ndarray:
        """What a learning agent sees, as a float32 vector."""

    @abstractmethod
    def observation_labels(self) -> list[str]:
        """One short name per observation entry, for explanations."""

    @abstractmethod
    def cost(self, batch: Batch) -> np.ndarray:
        """Cost of every state in the batch, shape [samples, time]; lower is better."""

    def terminal_cost(self, batch: Batch) -> np.ndarray:
        """Extra cost on the last predicted state, shape [samples]."""
        return self.cost(batch.last())[:, 0]

    def step_cost(self, data: mujoco.MjData) -> float:
        return float(self.cost(Batch.from_data(self.model, data))[0, 0])

    def failed(self, data: mujoco.MjData) -> bool:
        """True once the attempt cannot recover (the robot fell, the pole dropped...)."""
        return False

    @abstractmethod
    def success(self, data: mujoco.MjData) -> bool:
        """True while the goal is met."""

    @abstractmethod
    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        """A few readable numbers about how it is going, e.g. {"distance_m": 0.004}."""

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Randomize the start of an episode (data is already at the home pose). Environments only."""
        return None

    def begin(self, data: mujoco.MjData) -> None:
        """Called whenever an attempt starts from `data` (episode, live run, comparison), e.g. to
        remember where the robot started or where a target object is right now."""
        return None

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        """Disturb the robot (a sudden shove) to test recovery; returns what was done."""
        raise TaskError(f"task {self.name!r} has no push test")

    def nominal_ctrl(self, data: mujoco.MjData) -> np.ndarray:
        """A sensible control to start planning from: what the actuators are commanded now."""
        return data.ctrl[self.actuators].copy()

    def describe(self) -> dict[str, Any]:
        return {"task": self.name, "robot": self.robot.name, "params": self.params.model_dump()}
