"""A task on a running simulation: reading its `State` from MuJoCo, and starting an attempt where its randomization
says.

`Probe` finds the robot's middle, its hand (an arm's tool point), the task's thing and place and its mechanism once
per model, then reads them from any MjData in microseconds: the live world (observe), each step of a training
environment, a planner's predicted states.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from robotforge.design import mechanisms as mechanism_design
from robotforge.design.tasks.check import State, Verdict, judge
from robotforge.design.tasks.spec import TaskEntry

UPRIGHT_COS = 0.7  # the robot's up axis within about 45 degrees of vertical
OVERLAP = 0.01  # m: a randomized start overlapping something deeper than this is drawn again
DRAWS = 12


def _bodies(model: mujoco.MjModel, prefix: str) -> list[int]:
    return [b for b in range(1, model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            .startswith(prefix)]  # fmt: skip


class Probe:
    """Reads a task's state from MjData. `world` is the world.yaml the model was compiled from."""

    def __init__(self, model: mujoco.MjModel, task: TaskEntry, world: Any) -> None:
        self.model, self.task, self.world = model, task, world
        mine = _bodies(model, f"{task.robot}/")
        self.root = int(model.body_rootid[mine[0]]) if mine else -1
        self.floating = self.root > 0 and any(
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            for j in range(model.njnt)
            if model.jnt_bodyid[j] == self.root
        )
        self.hand = -1
        for s in range(model.nsite):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or ""
            if name.startswith(f"{task.robot}/") and (name.endswith("/tcp")):
                self.hand = s
                break
        self.things: dict[str, int] = {}
        for name in (task.object, task.target):
            if name is None:
                continue
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if (
                b > 0
                and model.body_jntnum[b] > 0
                and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
            ):
                self.things[name] = b
        self.shapes = {o.name: o.shape for o in world.objects if o.shape in mechanism_design.MECHANISMS}
        self.mechanism = task.object if task.object in self.shapes else None

    def state(self, data: mujoco.MjData, watcher: Any = None) -> State:
        from robotforge.sim import mechanisms

        robot = upright = hand = None
        if self.root > 0:
            robot = tuple(float(v) for v in data.xpos[self.root])
            upright = bool(data.xmat[self.root][8] > UPRIGHT_COS) if self.floating else None
        if self.hand >= 0:
            hand = tuple(float(v) for v in data.site_xpos[self.hand])
        things = {n: tuple(float(v) for v in data.xpos[b]) for n, b in self.things.items()}
        moving = {}
        if self.mechanism is not None:
            moving = mechanisms.read(self.model, data, {self.mechanism: self.shapes[self.mechanism]}, watcher)
        return State(robot, upright, hand, things, moving)

    def verdict(self, data: mujoco.MjData, start: State | None = None, watcher: Any = None) -> Verdict:
        return judge(self.task, self.world, self.state(data, watcher), start)


def randomize(model: mujoco.MjModel, data: mujoco.MjData, task: TaskEntry, rng: np.random.Generator) -> list[str]:
    """Move the robot and the thing within the task's ranges from where `data` has them (its start), drawing again
    while either overlaps something. Returns what was moved. `data` must be at the start (forward run)."""
    moved = []
    r = task.randomize
    probe = Probe(model, task, _NoWorld())
    plans: list[tuple[int, float, float]] = []  # (body, xy range, yaw range in rad)
    if probe.floating and (r.robot_xy > 0 or r.robot_yaw > 0):
        plans.append((probe.root, r.robot_xy, math.radians(r.robot_yaw)))
    for name, b in probe.things.items():
        if name == task.object and r.object_xy > 0:
            plans.append((b, r.object_xy, 0.0))
    for body, span, turn in plans:
        j = int(model.body_jntadr[body])
        adr = int(model.jnt_qposadr[j])
        home = data.qpos[adr : adr + 7].copy()
        for _ in range(DRAWS):
            q = home.copy()
            q[0] += rng.uniform(-span, span)
            q[1] += rng.uniform(-span, span)
            if turn > 0:
                a = rng.uniform(-turn, turn)
                spin = np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
                out = np.empty(4)
                mujoco.mju_mulQuat(out, spin, q[3:7])
                q[3:7] = out
            data.qpos[adr : adr + 7] = q
            mujoco.mj_forward(model, data)
            if not _overlaps(model, data, body):
                moved.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or str(body))
                break
        else:
            data.qpos[adr : adr + 7] = home
            mujoco.mj_forward(model, data)
    return moved


def _overlaps(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> bool:
    root = int(model.body_rootid[body])
    for c in data.contact[: data.ncon]:
        if c.dist > -OVERLAP:
            continue
        mine = [int(model.body_rootid[model.geom_bodyid[g]]) == root for g in (c.geom1, c.geom2)]
        if any(mine) and not all(mine):
            other = c.geom2 if mine[0] else c.geom1
            if model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            return True
    return False


class _NoWorld:
    objects: list = []
