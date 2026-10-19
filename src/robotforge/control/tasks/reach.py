"""reach: move an arm's tool point (its "tcp" site) to a target point in space."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import Field

from robotforge import wording
from robotforge.control.tasks.arm import ArmTask
from robotforge.control.tasks.base import Batch, TaskError, TaskParams, home_qpos, smooth_abs
from robotforge.control.tasks.tool import ToolTarget, min_jerk

MAX_REACH_RATIO = 2.0  # a target farther than this many arm lengths away is a wrong unit, not a hard reach


class ReachParams(TaskParams):
    target: list[float] | str | None = Field(
        None, description="[x, y, z] in meters, an object name, or empty for a random reachable point"
    )
    tolerance: float = Field(0.01, gt=0, le=0.2, description="success when the tool is this close, meters")
    jitter: float = Field(
        0.0,
        ge=0,
        le=0.3,
        description="with a target, each training episode moves it by up to this much (m) in every direction: "
        "a whole small region to learn, not one point",
    )


class Reach(ArmTask):
    name = "reach"
    summary = "Move an arm's tool to a target point and hold it there."
    robot_hint = "an arm with a tool point named 'tcp' (any robot from the arm template)"
    methods = ("ik", "pd", "cartesian", "joint_impedance", "cartesian_impedance", "osc", "mpc", "mppi")
    Params = ReachParams
    default_seconds = 4.0
    control_hz = 50.0
    mpc_defaults = {"horizon": 0.4, "knots": 3, "samples": 32, "noise": 0.06}
    goal_prefixes = (
        "target",
        "to target",
    )  # observation entries that are the goal (kept when only sensors give the rest)

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: ReachParams
        self._find_arm(model, robot)
        self._fk = mujoco.MjData(model)  # scratch for forward kinematics, reused to avoid allocations
        self.target = np.zeros(3)
        self._anchor = np.zeros(3)  # the target as the task states it; episodes with a jitter move around it
        self._offset = np.zeros(3)  # the current episode's jitter
        self._fresh_target = False  # set by reset(): an episode already drew its random target
        self._resolve_target(home_qpos(model), np.random.default_rng(0))

    # ---- target ----

    def _resolve_target(self, qpos: np.ndarray, rng: np.random.Generator) -> None:
        t = self.params.target
        if isinstance(t, list):
            if len(t) != 3:
                raise TaskError(f"reach target must be [x, y, z] in meters, got {t}")
            self.target = np.asarray(t, float)
            self._check_distance(qpos)
        elif isinstance(t, str):
            self.target = self._object_position(t, qpos)
        else:
            self.target = self.random_target(qpos, rng)
        self._anchor = self.target.copy()

    def _check_distance(self, qpos: np.ndarray) -> None:
        """Refuse a target so far off that the numbers must be in the wrong unit (centimetres for meters)."""
        d = self._fk
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        points = [*(d.xanchor[j] for j in self.joints), d.site_xpos[self.site]]
        reach = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:], strict=False))
        distance = float(np.linalg.norm(self.target - points[0]))
        if distance > MAX_REACH_RATIO * reach:
            where = [round(float(v), 3) for v in self.target]
            raise TaskError(f"reach target {where} is out of reach: {wording.too_far(where, distance, reach)}")

    def _object_position(self, name: str, qpos: np.ndarray) -> np.ndarray:
        m = self.model
        self._fk.qpos[:] = qpos
        mujoco.mj_kinematics(m, self._fk)
        kinds = ((mujoco.mjtObj.mjOBJ_BODY, range(1, m.nbody)), (mujoco.mjtObj.mjOBJ_GEOM, range(m.ngeom)))
        found = (mujoco.mj_id2name(m, obj, i) for obj, ids in kinds for i in ids)
        names = list(dict.fromkeys(n for n in found if n and n != "floor"))  # a fixed object is a geom and a body
        name = wording.match(name, names) or name  # "Ball" is ball
        lookups = (
            (mujoco.mjtObj.mjOBJ_BODY, self._fk.xpos),
            (mujoco.mjtObj.mjOBJ_GEOM, self._fk.geom_xpos),  # fixed objects are plain geoms
            (mujoco.mjtObj.mjOBJ_SITE, self._fk.site_xpos),
        )
        for obj, table in lookups:
            idx = mujoco.mj_name2id(m, obj, name)
            if idx >= 0:
                return table[idx].copy()
        objects = [n for n in names if "/" not in n]
        raise TaskError(f"{wording.unknown('object', name, objects, many='objects')}. Or give [x, y, z] in meters")

    def random_target(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """A point the tool can certainly reach: the tool position of a random joint configuration."""
        m, d = self.model, self._fk
        limited = m.jnt_limited[self.joints].astype(bool)
        lo = np.where(limited, np.maximum(m.jnt_range[self.joints, 0], -np.pi), -np.pi)
        hi = np.where(limited, np.minimum(m.jnt_range[self.joints, 1], np.pi), np.pi)
        for _ in range(200):
            d.qpos[:] = qpos
            d.qpos[self.qadr] = rng.uniform(lo, hi)
            mujoco.mj_kinematics(m, d)
            p = d.site_xpos[self.site].copy()
            base = d.xpos[m.jnt_bodyid[self.joints[0]]]
            if p[2] > base[2] + 0.03 and np.linalg.norm(p[:2] - base[:2]) > 0.1:
                return p
        return d.site_xpos[self.site].copy()

    def tool(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site].copy()

    def distance(self, data: mujoco.MjData) -> float:
        return float(np.linalg.norm(self.tool(data) - self.target))

    def target_source(self, model: mujoco.MjModel, data: mujoco.MjData) -> ReachPath:
        """The path the impedance controllers follow: a smooth line from the tool to the target."""
        return ReachPath(self, data)

    # ---- task interface ----

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        tool = self.tool(data)
        return np.concatenate(
            [data.qpos[self.qadr], data.qvel[self.dadr], tool, self.target, self.target - tool]
        ).astype(np.float32)

    def observation_labels(self) -> list[str]:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        return (
            [f"{n} angle" for n in names]
            + [f"{n} speed" for n in names]
            + [f"tool {a}" for a in "xyz"]
            + [f"target {a}" for a in "xyz"]
            + [f"to target {a}" for a in "xyz"]
        )

    def cost(self, batch: Batch) -> np.ndarray:
        return self._cost(batch.site_pos(self.site, self.site_sensor, self._fk), batch.qvel)

    def _cost(self, tool: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(tool - self.target, axis=-1)
        speed = np.sum(qvel[..., self.dadr] ** 2, axis=-1)
        return 10.0 * smooth_abs(dist, 0.02) + 0.02 * speed

    def terminal_cost(self, batch: Batch) -> np.ndarray:
        # Sensors in a rollout lag the state by one step, so place the final tool exactly with kinematics.
        final_q = batch.qpos[:, -1]
        tool = np.empty((len(final_q), 3))
        for i, q in enumerate(final_q):
            self._fk.qpos[:] = q
            mujoco.mj_kinematics(self.model, self._fk)
            tool[i] = self._fk.site_xpos[self.site]
        return 5.0 * self._cost(tool, batch.qvel[:, -1])

    def success(self, data: mujoco.MjData) -> bool:
        return self.distance(data) < self.params.tolerance

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        return {
            "distance_m": round(self.distance(data), 4),
            "tool": [round(float(v), 3) for v in self.tool(data)],
            "target": [round(float(v), 3) for v in self.target],
        }

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        if self.params.target is None:
            self.target = self.random_target(data.qpos.copy(), rng)
            self._fresh_target = True
        elif self.params.jitter > 0:
            self._offset = rng.uniform(-self.params.jitter, self.params.jitter, 3)
            self.target = self._anchor + self._offset

    def begin(self, data: mujoco.MjData) -> None:
        if isinstance(self.params.target, str):  # the object may have moved since the task was made
            self._anchor = self._object_position(self.params.target, data.qpos.copy())
            self.target = self._anchor + self._offset
        elif self.params.target is None and not self._fresh_target:
            # A live run without a target: pick one reachable from where the arm is now.
            self.target = self.random_target(data.qpos.copy(), np.random.default_rng(0))
        self._fresh_target = False

    def push(self, data: mujoco.MjData, strength: float = 1.0) -> str:
        data.qvel[self.dadr[: min(3, len(self.dadr))]] += 2.0 * strength
        return f"knocked the first joints with {2.0 * strength:.1f} rad/s"

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "target": [round(float(v), 3) for v in self.target]}


class ReachPath:
    """A min-jerk line from where the tool is to the reach target, at about 25 cm/s; the orientation is left free."""

    SPEED = 0.25  # m/s average tool speed along the path

    def __init__(self, task: Reach, data: mujoco.MjData):
        self.start = data.site_xpos[task.site].copy()
        self.goal = task.target.copy()
        self.duration = float(np.clip(np.linalg.norm(self.goal - self.start) / self.SPEED, 0.6, 2.5))
        self.t0 = float(data.time)

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        s, ds = min_jerk((data.time - self.t0) / self.duration)
        delta = self.goal - self.start
        return ToolTarget(self.start + s * delta, ds * delta / self.duration)
