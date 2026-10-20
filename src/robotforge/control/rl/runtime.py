"""Run a trained policy with numpy only: no PyTorch, no Stable-Baselines3.

Training exports the network that picks actions (a few small matrices) and the observation
normalizer into policy.npz. Running it is a handful of matrix products, so a trained policy
drives the live world inside the MCP server without loading PyTorch, and works on a core
install too (e.g. a policy folder someone shared).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.control.actuation import ActionScale
from robotforge.control.ars import PolicyController
from robotforge.control.rl import store
from robotforge.control.tasks import Task, TaskError

ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
    "identity": lambda x: x,
}


@dataclass
class MlpPolicy:
    """action = squash(W_n act(... act(W_1 normalize(obs) + b_1) ...) + b_n).

    `squash` is "clip" for PPO (its deterministic action is the Gaussian mean, clipped to [-1, 1])
    and "tanh" for SAC (its actions are squashed through tanh)."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    activation: str
    squash: str
    obs_mean: np.ndarray
    obs_var: np.ndarray
    clip_obs: float = 10.0
    epsilon: float = 1e-8
    _act: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise TaskError(f"unknown activation {self.activation!r} in the policy file")
        if self.squash not in ("clip", "tanh"):
            raise TaskError(f"unknown output squash {self.squash!r} in the policy file")
        self._check_shapes()
        self._act = ACTIVATIONS[self.activation]
        self._scale = 1.0 / np.sqrt(self.obs_var + self.epsilon)

    def _check_shapes(self) -> None:
        """A shared or damaged file must fail here, readably, not later inside the physics loop."""
        broken = "the policy file has a broken network"
        if len(self.weights) != len(self.biases) or not self.weights:
            raise TaskError(f"{broken} (layers and biases do not match)")
        width = None
        for w, b in zip(self.weights, self.biases, strict=True):
            if w.ndim != 2 or b.shape != (w.shape[0],) or (width is not None and w.shape[1] != width):
                raise TaskError(f"{broken} (layer sizes do not chain)")
            width = w.shape[0]
        n = self.weights[0].shape[1]
        if self.obs_mean.shape != (n,) or self.obs_var.shape != (n,) or np.any(self.obs_var < 0):
            raise TaskError(f"{broken} (the observation normalizer does not fit the first layer)")
        values = [*self.weights, *self.biases, self.obs_mean, self.obs_var]
        if not all(np.isfinite(v).all() for v in values):
            raise TaskError(f"{broken} (it contains NaN or infinity)")

    @property
    def obs_dim(self) -> int:
        return int(self.weights[0].shape[1])

    @property
    def act_dim(self) -> int:
        return int(self.weights[-1].shape[0])

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(obs, float) - self.obs_mean) * self._scale, -self.clip_obs, self.clip_obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = self.normalize(obs)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = w @ x + b
            if i < last:
                x = self._act(x)
        out = np.tanh(x) if self.squash == "tanh" else np.clip(x, -1.0, 1.0)
        return out.astype(np.float32)

    def save(self, path: Path) -> None:
        """Write atomically: the server may load the file while training saves a better one."""
        arrays: dict[str, np.ndarray] = {"obs_mean": self.obs_mean, "obs_var": self.obs_var}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        info = {"activation": self.activation, "squash": self.squash, "clip_obs": self.clip_obs, "eps": self.epsilon}
        arrays["info"] = np.frombuffer(json.dumps(info).encode("utf-8"), dtype=np.uint8)
        path = Path(path)
        tmp = store.temp_path(path)
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
        store.replace_file(tmp, path)

    @classmethod
    def load(cls, path: Path) -> MlpPolicy:
        try:
            with np.load(path, allow_pickle=False) as f:  # never unpickle a file someone may have shared
                info = json.loads(bytes(f["info"]).decode("utf-8"))
                n = sum(1 for k in f.files if k.startswith("w"))
                weights = [np.asarray(f[f"w{i}"], float) for i in range(n)]
                biases = [np.asarray(f[f"b{i}"], float) for i in range(n)]
                mean, var = np.asarray(f["obs_mean"], float), np.asarray(f["obs_var"], float)
            clip_obs, eps = float(info["clip_obs"]), float(info["eps"])
            return cls(weights, biases, info["activation"], info["squash"], mean, var, clip_obs, eps)
        except TaskError:
            raise
        except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile) as exc:
            raise TaskError(f"could not read the trained network in {Path(path)}: {exc}") from exc


@dataclass
class TrainedPolicy:
    """A policy folder, loaded: its metadata (policy.json) and its network."""

    folder: Path
    meta: dict[str, Any]
    net: MlpPolicy

    @classmethod
    def load(cls, folder: Path) -> TrainedPolicy:
        folder = Path(folder)
        meta = store.read_json(folder / store.POLICY_FILE)
        if not meta:
            raise TaskError(f"{folder} is not a policy folder (no {store.POLICY_FILE})")
        weights = folder / store.WEIGHTS_FILE
        if not weights.exists():
            state = store.read_json(folder / store.STATUS_FILE).get("state", "unknown")
            raise TaskError(
                f"policy {folder.name!r} has no trained network yet (training is {state}); "
                "wait for it with training_status, or train again"
            )
        return cls(folder, meta, MlpPolicy.load(weights))

    @property
    def name(self) -> str:
        return self.folder.name

    def check_fits(self, task: Task) -> None:
        """Refuse a task or robot the policy was not trained for, saying what it was trained for."""
        if self.meta.get("env"):  # trained by robotforge.training: judged on the observation it was trained on
            from robotforge.training.play import fit

            task = fit(self.meta, task, task.model)
        meta = self.meta
        if meta.get("task") != task.name:
            raise TaskError(f"policy {self.name!r} was trained for the {meta.get('task')} task, not {task.name}")
        want = meta.get("actuators") or []
        have = actuator_names(task)
        if want and want != have:
            raise TaskError(
                f"policy {self.name!r} was trained on a robot whose motors are {', '.join(want)}; "
                f"{task.robot.name!r} has {', '.join(have)}. Train a policy for this robot instead."
            )
        obs = task.observation_labels()
        if len(obs) != self.net.obs_dim or len(have) != self.net.act_dim:
            raise TaskError(
                f"policy {self.name!r} expects {self.net.obs_dim} observations and {self.net.act_dim} actions; "
                f"this task on {task.robot.name!r} gives {len(obs)} and {len(have)}"
            )

    def controller(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData) -> LearnedPolicyController:
        if self.meta.get("env"):  # trained by robotforge.training: on the sensors it was trained on
            from robotforge.training.play import fit

            task = fit(self.meta, task, model)
        self.check_fits(task)
        return LearnedPolicyController(task, model, data, self)


def actuator_names(task: Task) -> list[str]:
    """The task's actuators without the '<robot>/' prefix, so robots built alike match by name."""
    names = (mujoco.mj_id2name(task.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) or "" for a in task.actuators)
    return [n.removeprefix(task.robot.prefix) for n in names]


class LearnedPolicyController(PolicyController):
    """Runs a trained network on the live world (or in a comparison), exactly like during training.

    Direct mode: the network's action is the motor command, as in RobotEnv(action_mode="direct").
    Residual mode: a classic controller (the trot gait, holding the pose, IK...) runs underneath at
    its own rate and the network adds a correction, as in RobotEnv(action_mode="residual")."""

    def __init__(self, task: Task, model: mujoco.MjModel, data: mujoco.MjData, policy: TrainedPolicy):
        meta = policy.meta
        algo = str(meta.get("algo", "rl")).upper()
        info = {
            "learned": f"{algo} neural network ({meta.get('action_mode', 'direct')} mode)",
            "policy": policy.name,
            "train_seconds": meta.get("train_seconds"),
            "train_timesteps": meta.get("timesteps"),
        }
        super().__init__(task, model, policy.net, {k: v for k, v in info.items() if v is not None})
        self.policy_hz = 1.0 / float(meta.get("control_dt") or 1.0 / task.control_hz)
        self.residual = meta.get("action_mode") == "residual"
        self.base = None
        if self.residual:
            from robotforge.control.methods import make_controller

            self.base_method = meta.get("base_method") or task.methods[0]
            if self.base_method not in task.methods:  # never another policy: that could loop forever
                raise TaskError(f"policy {policy.name!r} names an unknown base controller {self.base_method!r}")
            self.base = make_controller(self.base_method, task, model, data)
            self.residual_scale = float(meta.get("residual_scale", 0.2))
        # Run at the physics rate and schedule the network and the base controller inside, so both
        # tick at exactly the rates they had during training.
        self.rate_hz = 1.0 / model.opt.timestep if self.residual else self.policy_hz
        self.scale = ActionScale(model, task.actuators)
        self.offset = np.zeros(len(task.actuators))
        self.base_ctrl: np.ndarray | None = None
        self._next_policy = self._next_base = -np.inf

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if not self.residual:
            super().update(model, data)
            return
        t = float(data.time)
        if t + 1e-9 >= self._next_policy:
            action = np.asarray(self.policy(self.task.observation(data)), float)
            self.offset = action * self.residual_scale * self.scale.half
            self._next_policy = t + 1.0 / self.policy_hz
        if t + 1e-9 >= self._next_base:
            self.base.update(model, data)
            self.base_ctrl = data.ctrl[self.task.actuators].copy()
            self._next_base = t + 1.0 / self.base.rate_hz
        if self.base_ctrl is not None:
            data.ctrl[self.task.actuators] = self.scale.clip(self.base_ctrl + self.offset)

    def status(self) -> dict[str, Any]:
        out = super().status()
        if self.base is not None:
            out["base"] = self.base_method
        return out


def policy_controller(
    name: str, task: Task, model: mujoco.MjModel, data: mujoco.MjData, params: dict[str, Any] | None = None
) -> LearnedPolicyController:
    """The controller for method "policy:<name>". params may give the folder: {"path": "..."}; otherwise the
    name is looked up in $ROBOTFORGE_HOME/policies (the MCP tools pass the open world's folder)."""
    params = dict(params or {})
    path = params.pop("path", None)
    world_dir = params.pop("world_dir", None)
    if params:
        raise TaskError(f"policy: unknown settings {sorted(params)}; a trained policy has nothing to tune")
    folder = Path(path) if path else store.find_policy(name, Path(world_dir) if world_dir else None)
    return TrainedPolicy.load(folder).controller(task, model, data)


def policy_params(
    method: str, params: dict[str, Any] | None, world_dir: Path | None, task: Task | None = None
) -> dict[str, Any] | None:
    """For a "policy:<name>" method: find its folder (the open world's first) and, given the task, check that
    the policy fits it before anything runs. Other methods' params come back unchanged."""
    from robotforge.control.methods import POLICY_PREFIX

    if not method.startswith(POLICY_PREFIX):
        return params
    folder = store.find_policy(method.removeprefix(POLICY_PREFIX), world_dir)
    if task is not None:
        TrainedPolicy.load(folder).check_fits(task)
    return {**(params or {}), "path": str(folder)}


def attach_policy(
    session, robot: str, name: str, world_dir: Path | None = None, task_params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Let a trained policy drive `robot` in the live world until something else takes over (scripts' run_policy)."""
    from robotforge.control.tasks import make_task

    policy = TrainedPolicy.load(store.find_policy(name, world_dir))
    params = {**(policy.meta.get("task_params") or {}), **(task_params or {})}
    with session.lock:
        task = make_task(policy.meta["task"], session.model, robot, params)
        task.begin(session.data)
        controller = policy.controller(task, session.model, session.data)
        session.set_controller(robot, controller)
    return {"robot": robot, "policy": name, "task": task.describe(), **controller.status()}
