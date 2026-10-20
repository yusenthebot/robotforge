"""Tasks where an arm's tool touches a surface and the force matters: press (stop or hold at a force) and wipe.

Both are about touch in the loop. The tool comes down onto a surface, feels it through the contact force on the hand
(control/tasks/tool.py: ToolForce), and then either stops at a force it was given, holds one, or keeps one while it
slides along the surface. The numbers a run reports are the ones a person asks of a force controller: how close the
force stayed to the wanted one, how hard it hit first, how well the tool kept to its path, and whether the job (a path
wiped, a force held) got done.

The plan every controller gets is the same: where the tool goes, and how hard it presses. The position methods (ik, pd)
can only go where they are told, so their force is whatever the arm's stiffness makes of the few millimetres of
`depth` they are told to press in; the impedance methods add the wanted force on top; osc also closes a loop on it.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Literal, NamedTuple

import mujoco
import numpy as np
from pydantic import Field

from robotforge import wording
from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.control.tasks.arm import ArmTask
from robotforge.control.tasks.base import Batch, TaskError, TaskParams
from robotforge.control.tasks.tool import ToolForce, ToolTarget, min_jerk

SENSE_HZ = 40.0  # cut-off of the low-pass filter on the force a run is judged by
CONTACT_N = 1.5  # N: the tool counts as having touched at this much force
DESCENT_SPEED = 0.02  # m/s the tool comes down onto a surface, in the last SLOW_ZONE
FAST_DESCENT = 0.15  # m/s above that
MAX_LAG = 0.005  # m the aim of a descent may run ahead of the tool
GUARD_LAG = 0.05  # m: a guarded move keeps going until the force it stops at shows, however soft the arm is
SLOW_ZONE = 0.05  # m above the surface (the tool point's height) where the slow descent begins
TRAVEL_SPEED = 0.25  # m/s average tool speed moving between places
SETTLE_S = 0.5  # s the force takes to build up to its value once the tool has touched
LIFT_S = 1.0  # s the tool takes to lift off at the end
LIFT = 0.06  # m
CORNER = 0.03  # m before and after a corner of the path where the tool slows down
CORNER_SPEED = 0.35  # of the path speed, at a corner
BIN = 0.01  # m of path: a wiped path is judged in bins this long
TRACE_S = 0.01  # s between the points of a force trace
LOST = 0.3  # of the wanted force: below this the tool has lost touch
JITTER_XY = 0.03  # m: a seeded start puts the tool up to this far to either side of where it is (reset)
JITTER_Z = 0.04  # m: and up to this much higher or lower
WIPED = 0.5  # of the wanted force: a bin counts as wiped if the tool pressed at least this hard
CONTROL_METHODS = ("ik", "pd", "joint_impedance", "cartesian_impedance", "osc")
# Settings that suit sliding on a surface, per method: stiff along the surface (friction drags the tool, and the path
# should still be followed), soft along the normal, where the force is.
SLIDING_GAINS: dict[str, dict[str, Any]] = {
    "joint_impedance": {"omega": 12.0},
    "cartesian_impedance": {"stiffness": [2000.0, 2000.0, 400.0], "rot_stiffness": 150.0},
    "osc": {"omega": 25.0, "rot_omega": 40.0},
}


class Surface(NamedTuple):
    point: np.ndarray  # where the ray from above met it
    normal: np.ndarray  # unit, pointing out of the surface
    name: str  # the object it belongs to


def _ray_solid(
    model: mujoco.MjModel, data: mujoco.MjData, g: int, pnt: np.ndarray, vec: np.ndarray, normal: np.ndarray
):
    """Distance along the ray to collision shape g (None if it misses). mj_ray skips the invisible shapes a world's
    collisions are made of, so each shape is tried with mju_rayGeom; a mesh stands in as its bounding box."""
    pos, mat = data.geom_xpos[g], data.geom_xmat[g]
    gtype, size = int(model.geom_type[g]), model.geom_size[g]
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        pos = pos + mat.reshape(3, 3) @ centre
        gtype, size = int(mujoco.mjtGeom.mjGEOM_BOX), half
    dist = mujoco.mju_rayGeom(pos, mat, size, pnt, vec, gtype, normal)
    return dist if dist >= 0 else None


def find_surface(model: mujoco.MjModel, data: mujoco.MjData, above: np.ndarray, prefix: str) -> Surface:
    """What is below `above`: the nearest shape that collides, not the robot's own links."""
    pnt = np.asarray(above, float).copy()
    vec = np.array([0.0, 0.0, -1.0])
    best: tuple[float, int, np.ndarray] | None = None
    normal = np.zeros(3)
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if name.startswith(prefix):
            continue
        dist = _ray_solid(model, data, g, pnt, vec, normal)
        if dist is not None and (best is None or dist < best[0]):
            best = (dist, g, normal.copy())
    if best is None:
        raise TaskError("nothing below the tool to touch; move the arm over a table, or give the surface's name")
    dist, g, n = best
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
    return Surface(pnt + vec * dist, n / np.linalg.norm(n), body)


class PathPlane:
    """A surface's plane with a frame on it: u along the surface where world x points, v across, n out of it."""

    def __init__(self, origin: np.ndarray, normal: np.ndarray):
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        u = np.array([1.0, 0.0, 0.0]) - n * float(n[0])
        if np.linalg.norm(u) < 1e-6:
            u = np.array([0.0, 1.0, 0.0]) - n * float(n[1])
        self.u = u / np.linalg.norm(u)
        self.n = n
        self.v = np.cross(self.n, self.u)
        self.origin = np.asarray(origin, float)

    def point(self, u: float, v: float, up: float = 0.0) -> np.ndarray:
        return self.origin + self.u * u + self.v * v + self.n * up

    def coords(self, p: np.ndarray) -> np.ndarray:
        """(u, v, height above the plane) of a world point."""
        d = np.asarray(p, float) - self.origin
        return np.array([d @ self.u, d @ self.v, d @ self.n])


class Tally:
    """What a run measures while the tool presses on a surface: the force it presses with, sample by sample."""

    def __init__(self) -> None:
        self.force: list[float] = []  # through the sensor's low-pass filter: what the run is judged by
        self.raw: list[float] = []  # the same steps, unfiltered: what the wrist sensor reads step by step
        self.trace: list[tuple[float, float]] = []  # (seconds into the run, newtons) at about 100 per second

    def add(self, newtons: float, seconds: float = 0.0, raw: float | None = None) -> None:
        self.force.append(newtons)
        self.raw.append(newtons if raw is None else raw)
        if not self.trace or seconds - self.trace[-1][0] >= TRACE_S:
            self.trace.append((seconds, newtons))

    def summary(self, wanted: float) -> dict[str, float]:
        if not self.force:
            return {}
        f = np.asarray(self.force)
        return {
            "force_mean_n": float(f.mean()),
            "force_std_n": float(f.std()),
            "force_rms_error_n": float(np.sqrt(np.mean((f - wanted) ** 2))),
            "force_peak_n": float(f.max()),
            "force_p95_n": float(np.percentile(f, 95)),
            "lost_contact_pct": float(100.0 * np.mean(f < LOST * wanted)),
            # the same count on the unfiltered sensor: a pad that skips off the table for a step or two shows here
            "raw_dropout_pct": float(100.0 * np.mean(np.asarray(self.raw) < LOST * wanted)),
        }


class ContactTask(ArmTask):
    """Base of the tasks that press on a surface: finds the surface, owns the force sense and the bookkeeping."""

    contact: ClassVar[bool] = True
    jitter_start: ClassVar[bool] = True  # a seeded start moves the tool a little (reset)
    methods: ClassVar[tuple[str, ...]] = CONTROL_METHODS
    control_hz: ClassVar[float] = 50.0
    gains: ClassVar[dict[str, dict[str, Any]]] = SLIDING_GAINS
    robot_hint = "an arm with a tool point named 'tcp' (the arm template, or a library arm: panda, ur5e...)"

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self._find_arm(model, robot)
        self.plane: PathPlane | None = None
        self.surface: Surface | None = None
        self.tally = Tally()
        self.plan: Any = None
        self.sensed = 0.0
        self.raw_now = 0.0
        self._alpha = 1.0 - math.exp(-2 * math.pi * SENSE_HZ * model.opt.timestep)
        self._sense = ToolForce.of_arm(model, self)
        self._chain = Chain.from_joints(
            model, self.joints, int(model.site_bodyid[self.site]), model.site_pos[self.site].copy()
        )
        self._ik = IKSolver(model, model.qpos0.copy())

    # ---- what a controller needs of a task ----

    def target_source(self, model: mujoco.MjModel, data: mujoco.MjData) -> Any:
        """What a controller follows: whatever plan the task has now. A run that starts the task again (begin) after
        its controller was made, as run_controller does on the live world, swaps the plan under it."""
        if self.plan is None:
            self.begin(data)
        return _Current(self)

    def begin(self, data: mujoco.MjData) -> None:
        self.tally = Tally()
        self.sensed = 0.0
        self._close_gripper(data)
        self.surface = self._look(data)
        self.plane = PathPlane(self.surface.point, self.surface.normal)
        self.plan = self._plan(data, self.plane)
        self.default_seconds = self._duration()  # type: ignore[misc]

    def _duration(self) -> float:
        """How long a run needs from where the tool is now: the class default unless the task knows better."""
        return type(self).default_seconds

    def _descent_seconds(self, height: float) -> float:
        """Time to come down `height` m onto the surface: fast, then slowly over the last SLOW_ZONE."""
        height = max(0.0, height)
        return max(0.0, height - SLOW_ZONE) / FAST_DESCENT + min(height, SLOW_ZONE) / DESCENT_SPEED

    def reset(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """A start that differs from run to run (a seeded comparison): the tool a few cm to the side and higher or
        lower, found by inverse kinematics from where the arm is. The job is the same; only where it begins varies."""
        if not self.jitter_start:
            return
        jitter = rng.uniform(-1.0, 1.0, 3) * np.array([JITTER_XY, JITTER_XY, JITTER_Z])
        q, err = self._ik.solve(
            self._chain, data.site_xpos[self.site] + jitter, data.qpos[self.qadr], iterations=200, damping=1e-3
        )
        if err < 0.003:  # a start the arm cannot take is left as it was
            data.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, data)

    def surface_report(self) -> dict[str, Any]:
        """Which surface the job is on and how high it is: a tool over the floor wipes the floor, and says so."""
        if self.surface is None:
            return {}
        floor = self.surface.name in ("", "world")
        out: dict[str, Any] = {
            "surface": "floor" if floor else self.surface.name,
            "surface_z_m": round(float(self.surface.point[2]), 3),
        }
        if floor:
            out["note"] = (
                "working on the floor, not on a table: the arm is not over a table, or the table is out of the way; "
                "give surface='<object>' (it checks the tool is over it) or move the arm over the table first"
            )
        return out

    def _close_gripper(self, data: mujoco.MjData) -> None:
        """Fingers shut, so that the two finger pads press as one flat tool: apart, the hand rocks from pad to pad
        on the surface and the force flickers. A suction cup or a magnet has nothing to close."""
        m = self.model
        act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.robot.name}/gripper")
        if act < 0 or int(m.actuator_trntype[act]) not in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_TENDON),
        ):
            return
        if m.actuator_ctrllimited[act]:
            data.ctrl[act] = m.actuator_ctrlrange[act, 0]

    def _look(self, data: mujoco.MjData) -> Surface:
        """The surface to work on: the thing named by the `surface` parameter, or what is under the tool."""
        tool = data.site_xpos[self.site]
        name = getattr(self.params, "surface", None)
        found = find_surface(self.model, data, np.array([tool[0], tool[1], tool[2] + 0.02]), self.robot.prefix)
        if name and found.name != name and not found.name.startswith(name):
            known = [
                n
                for b in range(1, self.model.nbody)
                if (n := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b)) and "/" not in n
            ]
            if name not in known:
                raise TaskError(wording.unknown("surface", name, known, many="objects"))
            raise TaskError(f"the tool is not over {name}: below it is {found.name or 'the floor'}; move the arm there")
        return found

    def _plan(self, data: mujoco.MjData, plane: PathPlane) -> Any:
        raise NotImplementedError

    def _reachable(self, data: mujoco.MjData, points: list[np.ndarray]) -> None:
        """Refuse a job whose places the arm cannot reach, saying which one."""
        q = data.qpos.copy()
        for i, p in enumerate(points):
            _, err = self._ik.solve(self._chain, p, q[self.qadr], iterations=200, damping=1e-3)
            if err > 0.01:
                where = [round(float(v), 3) for v in p]
                raise TaskError(
                    f"{self.robot.name} cannot reach {where} ({'its start' if i == 0 else 'a place on the path'}; "
                    f"{err * 100:.0f} cm short): move the surface nearer the robot or the robot nearer the surface"
                )

    # ---- measuring ----

    def tool(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site].copy()

    def pressing(self, data: mujoco.MjData) -> float:
        """The force the tool presses on the surface with now, N (zero in the air)."""
        if self.plane is None:
            return 0.0
        return max(0.0, float(self._sense.read(self.model, data) @ self.plane.n))

    def feel_step(self, data: mujoco.MjData) -> float:
        """What a force sensor reads now: the pressing force through the same low-pass filter the controllers use
        (a sensor does not report the 2 ms spike of an impact), updated once per physics step."""
        self.raw_now = self.pressing(data)
        self.sensed += self._alpha * (self.raw_now - self.sensed)
        return self.sensed

    def observation(self, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            [data.qpos[self.qadr], data.qvel[self.dadr], self.tool(data), [self.pressing(data)]]
        ).astype(np.float32)

    def observation_labels(self) -> list[str]:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.joints]
        return [f"{n} angle" for n in names] + [f"{n} speed" for n in names] + [f"tool {a}" for a in "xyz"] + ["force"]

    def cost(self, batch: Batch) -> np.ndarray:
        raise TaskError(f"{self.name} is not planned by sampling: use one of {', '.join(self.methods)}")


class _Current:
    """A controller's view of a contact task's plan: always the task's latest."""

    def __init__(self, task: ContactTask):
        self.task = task

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        return self.task.plan.target(data, feel)


class PressParams(TaskParams):
    force: float = Field(5.0, gt=0, le=100, description="N the tool presses with")
    mode: Literal["hold", "stop"] = Field(
        "hold", description="hold: keep pressing at the force; stop: come down and stop the moment the force is reached"
    )
    depth: float = Field(
        0.002,
        ge=0,
        le=0.02,
        description="m past the first touch the tool is told to go (position control turns it into a force)",
    )
    surface: str | None = Field(None, description="the object to press on; empty for what is under the tool")


class Press(ContactTask):
    name = "press"
    summary = "Press the tool on a surface: stop the moment a force is reached, or hold that force."
    Params = PressParams
    default_seconds = 8.0
    # ---- plan ----

    def _plan(self, data: mujoco.MjData, plane: PathPlane) -> PressPlan:
        return PressPlan(self, data, plane)

    def _duration(self) -> float:
        height = float(self.plane.coords(self.plan.top)[2]) if self.plane is not None else 0.0
        return round(max(type(self).default_seconds, self._descent_seconds(height) + SETTLE_S + 4.0), 1)

    # ---- the task interface ----

    def step_cost(self, data: mujoco.MjData) -> float:
        plan = self.plan
        if plan is None or plan.tripped_at is None:
            return 0.0
        error = (self.sensed - self.params.force) / self.params.force
        return float(min(error * error, 4.0))

    def observe(self, data: mujoco.MjData) -> None:
        felt = self.feel_step(data)
        plan = self.plan
        if plan is not None and plan.tripped_at is not None:
            self.tally.add(felt, float(data.time) - self.plan.t0, self.raw_now)

    def success(self, data: mujoco.MjData) -> bool:
        wanted, f = self.params.force, self.tally.force
        if self.plan is None or self.plan.tripped_at is None or len(f) < 50:
            return False
        last = float(np.mean(f[-250:]))  # about the last half second
        peak = float(np.max(f))
        if self.params.mode == "stop":
            return peak <= 1.5 * wanted
        return abs(last - wanted) <= 0.2 * wanted and peak <= 2.0 * wanted

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        wanted, f = self.params.force, self.tally.force
        out: dict[str, Any] = {"wanted_n": wanted, "pressing_n": round(self.sensed, 2), **self.surface_report()}
        if self.plan is not None and self.plan.tripped_at is not None and f:
            peak = float(np.max(f))
            out |= {
                "touched_at_s": round(self.plan.tripped_at, 2),
                "settled_n": round(float(np.mean(f[-250:])), 2),
                "peak_n": round(peak, 2),
                "overshoot_n": round(max(0.0, peak - wanted), 2),
            }
        else:
            out["touched"] = False
        return out


class PressPlan:
    """Come down on the surface over where the tool is, and when the force shows, stop or press."""

    def __init__(self, task: Press, data: mujoco.MjData, plane: PathPlane):
        self.task, self.plane = task, plane
        self.t0 = float(data.time)
        self.top = data.site_xpos[task.site].copy()
        self.ref = self.top.copy()
        self.travel = 0.0  # m the aim has gone down from the top
        self.axis = -plane.n
        self.last_t = self.t0
        self.tripped_at: float | None = None
        self.frozen: np.ndarray | None = None

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        p, n = self.task.params, self.plane.n
        t = float(data.time)
        dt, self.last_t = t - self.last_t, t
        pushing = max(0.0, float(feel @ n))
        if self.tripped_at is None:
            trip = p.force if p.mode == "stop" else min(CONTACT_N, p.force)
            if pushing >= trip:
                self.tripped_at = t - self.t0
                self.frozen = data.site_xpos[self.task.site].copy()
            else:
                here = data.site_xpos[self.task.site]
                speed = DESCENT_SPEED if float(self.plane.coords(here)[2]) < SLOW_ZONE else FAST_DESCENT
                # The aim runs ahead at the descent speed but never more than MAX_LAG past the tool: when it meets the
                # surface the tool is not pulling against however far the aim has run away.
                gone = float((self.top - here) @ n)
                lag = MAX_LAG if p.mode == "hold" else GUARD_LAG
                self.travel = min(self.travel + speed * dt, gone + lag)
                self.ref = self.top - n * self.travel
                return ToolTarget(self.ref.copy(), -n * speed, axis=self.axis, normal=n)
        assert self.frozen is not None
        if p.mode == "stop":
            return ToolTarget(self.frozen.copy(), np.zeros(3), axis=self.axis, normal=n)
        ramp = min(1.0, (t - self.t0 - self.tripped_at) / SETTLE_S)
        pos = self.frozen - n * p.depth
        return ToolTarget(pos, np.zeros(3), axis=self.axis, normal=n, push=p.force * ramp, pressing=True, depth=p.depth)


class WipeParams(TaskParams):
    force: float = Field(8.0, gt=0, le=100, description="N the tool presses on the surface with while it wipes")
    length: float = Field(0.2, gt=0.02, le=1.0, description="m, how long each pass is (along the surface's x)")
    width: float = Field(0.1, ge=0, le=1.0, description="m across, from the first pass to the last")
    passes: int = Field(3, ge=1, le=8, description="how many passes back and forth")
    speed: float = Field(0.08, gt=0, le=0.4, description="m/s along the surface")
    depth: float = Field(
        0.003, ge=0, le=0.02, description="m past the first touch the tool is told to go (position control: a force)"
    )
    center: list[float] | None = Field(None, description="[x, y] of the middle of the area, empty for under the tool")
    surface: str | None = Field(None, description="the object to wipe; empty for what is under the tool")


class Wipe(ContactTask):
    name = "wipe"
    summary = "Wipe a surface with the tool along back-and-forth passes while pressing with a constant force."
    Params = WipeParams

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: WipeParams
        p = self.params
        self.path_m = p.length * p.passes + p.width * (1 if p.passes > 1 else 0)
        self.default_seconds = round(5.0 + self.path_m / p.speed + 2.5, 1)  # type: ignore[misc]
        self.lateral = 0.0
        self.lateral_n = 0
        self.wiped = np.zeros(1, dtype=bool)

    def _duration(self) -> float:
        """Over the start, down from where the tool is, settle, the passes (corners slow them), lift, some to spare."""
        plan = self.plan
        work = self.path_m / self.params.speed * 1.15
        total = plan.above_s + self._descent_seconds(plan.height) + SETTLE_S + work + LIFT_S + 2.0
        return round(max(total, 5.0 + self.path_m / self.params.speed + 2.5), 1)

    def _plan(self, data: mujoco.MjData, plane: PathPlane) -> WipePlan:
        plan = WipePlan(self, data, plane)
        self.wiped = np.zeros(plan.bins, dtype=bool)
        self.lateral, self.lateral_n = 0.0, 0
        self._reachable(data, [plane.point(u, v, 0.02) for u, v in plan.corners_uv])
        return plan

    def observe(self, data: mujoco.MjData) -> None:
        f = self.feel_step(data)
        plan = self.plan
        if plan is None or plan.phase != "work":
            return
        self.tally.add(f, float(data.time) - plan.t0, self.raw_now)
        b = min(plan.bins - 1, int(plan.s / BIN))
        if f >= WIPED * self.params.force:
            self.wiped[b] = True
        tool = self.plane.coords(data.site_xpos[self.site]) if self.plane is not None else np.zeros(3)
        self.lateral += float(np.sum((tool[:2] - plan.ref_uv) ** 2))
        self.lateral_n += 1

    def step_cost(self, data: mujoco.MjData) -> float:
        plan = self.plan
        if plan is None or plan.phase != "work":
            return 0.0
        error = (self.sensed - self.params.force) / self.params.force
        return float(min(error * error, 4.0))

    def wiped_pct(self) -> float:
        return float(100.0 * self.wiped.mean()) if self.wiped.size else 0.0

    def success(self, data: mujoco.MjData) -> bool:
        wanted = self.params.force
        s = self.tally.summary(wanted)
        if not s or self.plan is None or self.plan.phase == "work":
            return False
        return (
            self.wiped_pct() >= 90.0
            and abs(s["force_mean_n"] - wanted) <= 0.25 * wanted
            and s["force_p95_n"] <= 2.0 * wanted
        )

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        wanted = self.params.force
        out: dict[str, Any] = {"wanted_n": wanted, "pressing_n": round(self.sensed, 2), **self.surface_report()}
        out |= {k: round(v, 2) for k, v in self.tally.summary(wanted).items()}
        if self.lateral_n:
            out["track_rms_mm"] = round(1000.0 * math.sqrt(self.lateral / self.lateral_n), 2)
        out["path_wiped_pct"] = round(self.wiped_pct(), 1)
        out["phase"] = self.plan.phase if self.plan is not None else "idle"
        return out


class WipePlan:
    """Over the start of the path, down onto the surface, settle the force, back and forth along the passes, lift."""

    def __init__(self, task: Wipe, data: mujoco.MjData, plane: PathPlane):
        self.task, self.plane = task, plane
        p = task.params
        start = plane.coords(data.site_xpos[task.site])
        cu, cv = (start[0], start[1]) if p.center is None else self._center(p.center, plane)
        self.axis = -plane.n
        rows = [(-p.length / 2, p.length / 2) if i % 2 == 0 else (p.length / 2, -p.length / 2) for i in range(p.passes)]
        offsets = [0.0] if p.passes == 1 else [-p.width / 2 + i * p.width / (p.passes - 1) for i in range(p.passes)]
        pts: list[tuple[float, float]] = []
        for (a, b), off in zip(rows, offsets, strict=True):
            pts += [(cu + a, cv + off), (cu + b, cv + off)]
        self.corners_uv = pts
        self.path = np.array(pts)
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])
        self.bins = max(1, int(math.ceil(self.length / BIN)))
        self.corner_at = self.cum[1:-1]
        self.phase = "above"
        self.t0 = float(data.time)
        self.t_phase = self.t0
        self.s = 0.0
        self.ref_uv = np.array(pts[0])
        self.start_world = data.site_xpos[task.site].copy()
        self.hover_uv = np.array(pts[0])
        self.height = float(start[2])
        self.ref_height = self.height
        self.touch_height: float | None = None
        self.last_t = self.t0
        self.above_s = float(np.clip(np.linalg.norm(self.hover_uv - start[:2]) / TRAVEL_SPEED, 1.0, 3.0))
        self.start_uv = start[:2].copy()

    @staticmethod
    def _center(center: list[float], plane: PathPlane) -> tuple[float, float]:
        if len(center) != 2:
            raise TaskError(f"center is [x, y] in meters, got {center}")
        c = plane.coords(np.array([center[0], center[1], plane.origin[2]]))
        return float(c[0]), float(c[1])

    def _pos(self, uv: np.ndarray, height: float) -> np.ndarray:
        return self.plane.point(float(uv[0]), float(uv[1]), height)

    def _speed(self) -> float:
        """Path speed at arc length s: slower near a corner, so the target's velocity does not jump."""
        v = self.task.params.speed
        if len(self.corner_at) == 0:
            return v
        near = float(np.min(np.abs(self.corner_at - self.s)))
        k = min(1.0, near / CORNER)
        return v * (CORNER_SPEED + (1 - CORNER_SPEED) * k * k * (3 - 2 * k))

    def _on_path(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        i = min(int(np.searchsorted(self.cum, s, side="right")) - 1, len(self.cum) - 2)
        seg = self.path[i + 1] - self.path[i]
        length = float(np.linalg.norm(seg))
        direction = seg / length if length > 0 else np.zeros(2)
        return self.path[i] + direction * (s - self.cum[i]), direction

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        p, n = self.task.params, self.plane.n
        t = float(data.time)
        dt, self.last_t = t - self.last_t, t
        pushing = max(0.0, float(feel @ n))
        here = self.plane.coords(data.site_xpos[self.task.site])
        if self.phase == "above":
            s, ds = min_jerk((t - self.t_phase) / self.above_s)
            uv = self.start_uv + s * (self.hover_uv - self.start_uv)
            vel = ds * (self.hover_uv - self.start_uv) / self.above_s
            self.ref_uv = uv
            if t - self.t_phase >= self.above_s:
                self._enter("descend", t)
            return ToolTarget(
                self._pos(uv, self.height), self.plane.u * vel[0] + self.plane.v * vel[1], axis=self.axis, normal=n
            )
        if self.phase == "descend":
            speed = DESCENT_SPEED if float(here[2]) < SLOW_ZONE else FAST_DESCENT
            self.ref_height = max(self.ref_height - speed * dt, float(here[2]) - MAX_LAG)
            if pushing >= min(CONTACT_N, p.force):
                self.touch_height = float(here[2])
                self._enter("settle", t)
            return ToolTarget(self._pos(self.hover_uv, self.ref_height), -n * speed, axis=self.axis, normal=n)
        assert self.touch_height is not None
        work_height = self.touch_height - p.depth
        if self.phase == "settle":
            k = min(1.0, (t - self.t_phase) / SETTLE_S)
            if k >= 1.0:
                self._enter("work", t)
            height = self.touch_height - p.depth * k
            return ToolTarget(
                self._pos(self.hover_uv, height),
                np.zeros(3),
                axis=self.axis,
                normal=n,
                push=p.force * k,
                pressing=True,
                depth=p.depth * k,
            )
        if self.phase == "work":
            self.s = min(self.length, self.s + self._speed() * dt)
            uv, direction = self._on_path(self.s)
            self.ref_uv = uv
            vel = self._speed() * direction
            if self.s >= self.length:
                self._enter("lift", t)
            return ToolTarget(
                self._pos(uv, work_height),
                self.plane.u * vel[0] + self.plane.v * vel[1],
                axis=self.axis,
                normal=n,
                push=p.force,
                pressing=True,
                depth=p.depth,
            )
        k, dk = min_jerk((t - self.t_phase) / LIFT_S)
        if self.phase == "lift" and t - self.t_phase >= LIFT_S:
            self._enter("done", t)
        height = work_height + LIFT * k
        return ToolTarget(self._pos(self.ref_uv, height), n * LIFT * dk / LIFT_S, axis=self.axis, normal=n)

    def _enter(self, phase: str, t: float) -> None:
        self.phase, self.t_phase = phase, t
