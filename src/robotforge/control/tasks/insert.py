"""insert: put a peg the gripper holds into a hole it is not exactly told the place of.

The peg is held, a socket (add_object shape="socket") is somewhere near, and the robot is told where the hole is
only to a few millimetres: `error` is how far off the hole it thinks it is. A hole chamfered 6 mm and cleared 1.5 mm
takes a peg that arrives within a few millimetres, if the arm gives way to what the chamfer asks of the peg; a stiff
arm sets the peg's corner on the lip and pushes. So the plan is the same for every controller (above the hole, down
until the peg touches, then push down with a force) and the controllers that give way sideways go in.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import mujoco
import numpy as np
from pydantic import Field

from robotforge.control.tasks.base import TaskError, TaskParams
from robotforge.control.tasks.contact import (
    CONTACT_N,
    DESCENT_SPEED,
    FAST_DESCENT,
    MAX_LAG,
    SETTLE_S,
    SLOW_ZONE,
    TRAVEL_SPEED,
    ContactTask,
    PathPlane,
    Surface,
)
from robotforge.control.tasks.tool import ToolForce, ToolTarget, blend_rotation, min_jerk, pointing
from robotforge.sim.geometry import collides, highest_point, lowest_point

HOVER_ABOVE = 0.03  # m the peg's tip waits over the hole's mouth
INSERTED = 0.004  # m: the peg counts as in when its tip is this near the hole's floor
TILTED = math.radians(8)  # a peg this far off upright is not in
# Settings that suit a peg going into a hole: soft sideways, firm down, so that the chamfer can steer the peg.
INSERT_GAINS: dict[str, dict[str, Any]] = {
    "joint_impedance": {"omega": 8.0},
    "cartesian_impedance": {
        "stiffness": [900.0, 900.0, 900.0],
        "soft_stiffness": [300.0, 300.0, 500.0],
        "rot_stiffness": 60.0,
    },
    "osc": {"omega": 14.0, "soft_omega": 9.0, "rot_omega": 15.0, "normal_omega": 12.0},
}


class InsertParams(TaskParams):
    hole: str = Field(description="the socket to put the peg in (add_object shape='socket')")
    peg: str | None = Field(None, description="the object the gripper holds; empty for the thing between its fingers")
    force: float = Field(8.0, gt=0, le=100, description="N the peg is pushed into the hole with")
    limit: float = Field(
        40.0, gt=0, le=1000, description="N the part takes: pressed harder than this, down or sideways, it is damaged"
    )
    error: list[float] = Field(
        [0.004, 0.0], min_length=2, max_length=2, description="[x, y] m the robot thinks the hole is off its real place"
    )
    depth: float | None = Field(
        None, ge=0, le=0.1, description="m past the hole's mouth the peg is told to go (the hole's depth if empty)"
    )


class Insert(ContactTask):
    name = "insert"
    summary = "Push a peg the gripper holds into a socket whose place the robot knows only to a few millimetres."
    Params = InsertParams
    gains: ClassVar[dict[str, dict[str, Any]]] = INSERT_GAINS
    jitter_start: ClassVar[bool] = False  # the peg is already in the hand over the hole: a moved hand would not match
    default_seconds = 14.0

    def __init__(self, model: mujoco.MjModel, robot: str, params: dict[str, Any] | None = None):
        super().__init__(model, robot, params)
        self.params: InsertParams
        self.hole_body = self._body(self.params.hole, "socket")
        self.peg_body = -1
        self.held_bodies: set[int] = set()
        self.floor_z = 0.0
        self.mouth = np.zeros(3)
        self.tilt = 0.0
        self.depth_reached = 0.0
        self.peak_down = 0.0
        self.peak_side = 0.0
        self.inserted_at: float | None = None

    def _body(self, name: str, what: str) -> int:
        b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            from robotforge import wording

            names = [
                n
                for i in range(1, self.model.nbody)
                if (n := mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i))
            ]
            raise TaskError(wording.unknown(what, name, [n for n in names if "/" not in n], many="objects"))
        return b

    # ---- the world the task works in ----

    def begin(self, data: mujoco.MjData) -> None:
        from robotforge.control.tasks.contact import Tally

        self.tally = Tally()
        self.sensed = 0.0
        self.peak_down = self.peak_side = 0.0
        self.depth_reached, self.tilt, self.inserted_at = 0.0, 0.0, None
        self.peg_body = self._find_peg(data)
        self._tool_force = None  # made again with the peg in it
        self._held(data)
        self._sense = ToolForce.of_arm(self.model, self, self.held_bodies)
        socket_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] == self.hole_body and collides(self.model, g)
        ]
        top = max(float(highest_point(self.model, data, g)[2]) for g in socket_geoms)
        base = data.xpos[self.hole_body]
        self.mouth = np.array([base[0], base[1], top])
        self.floor_z = float(base[2]) + 0.008
        self.surface = Surface(self.mouth.copy(), np.array([0.0, 0.0, 1.0]), self.params.hole)
        self.plane = PathPlane(self.mouth, self.surface.normal)
        self.plan = InsertPlan(self, data)
        self._reachable(data, [self.mouth + [0.0, 0.0, HOVER_ABOVE + self.tip_below(data) + 0.01]])

    def _find_peg(self, data: mujoco.MjData) -> int:
        if self.params.peg is not None:
            return self._body(self.params.peg, "object")
        held = self._between_fingers(data)
        if held < 0:
            raise TaskError(
                f"{self.robot.name} holds nothing: pick the peg up first (move_arm to it with the gripper open, "
                "gripper close, move_arm up), or name it with peg"
            )
        return held

    def _between_fingers(self, data: mujoco.MjData) -> int:
        """The loose body that touches the hand most, if any."""
        sense = ToolForce.of_arm(self.model, self)
        counts: dict[int, int] = {}
        for i in range(data.ncon):
            c = data.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                body = int(self.model.geom_bodyid[b])
                if sense.tool[a] and not sense.own[b] and not sense.tool[b] and self.model.body_dofnum[body] == 6:
                    counts[body] = counts.get(body, 0) + 1
        return max(counts, key=counts.get) if counts else -1

    def _held(self, data: mujoco.MjData) -> None:
        self.held_bodies = {self.peg_body}

    def tip_below(self, data: mujoco.MjData) -> float:
        """How far the peg's lowest point is below the tool point."""
        tip = self.tip(data)
        return float(data.site_xpos[self.site][2] - tip[2])

    def tip(self, data: mujoco.MjData) -> np.ndarray:
        """The middle of the peg's lowest face: under its centre of mass, at its lowest point."""
        geoms = [
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.peg_body and collides(self.model, g)
        ]
        low = min(float(lowest_point(self.model, data, g)[2]) for g in geoms)
        axis = data.xmat[self.peg_body].reshape(3, 3)[:, 2]
        centre = data.xipos[self.peg_body]
        # slide the centre of mass along the peg's axis to the lowest level (a tilted peg's tip is not under its middle)
        t = (low - centre[2]) / axis[2] if abs(float(axis[2])) > 0.2 else 0.0
        return centre + axis * t * (1.0 if axis[2] > 0 else 1.0)

    # ---- measuring ----

    def observe(self, data: mujoco.MjData) -> None:
        f = self.feel_step(data)
        plan = self.plan
        if plan is None or plan.phase in ("above", "descend"):
            return
        self.tally.add(f, float(data.time) - plan.t0)
        force = self._sense.read(self.model, data)
        self.peak_down = max(self.peak_down, float(force[2]))
        self.peak_side = max(self.peak_side, float(np.hypot(force[0], force[1])))
        tip = self.tip(data)
        self.depth_reached = max(self.depth_reached, float(self.mouth[2] - tip[2]))
        axis = data.xmat[self.peg_body].reshape(3, 3)[:, 2]
        self.tilt = math.acos(min(1.0, abs(float(axis[2]))))
        if self.inserted_at is None and self.is_in(data):
            self.inserted_at = float(data.time) - plan.t0

    def is_in(self, data: mujoco.MjData) -> bool:
        tip = self.tip(data)
        inside = float(np.hypot(*(tip[:2] - self.mouth[:2]))) < 0.012
        return bool(inside and tip[2] <= self.floor_z + INSERTED and self.tilt < TILTED)

    def step_cost(self, data: mujoco.MjData) -> float:
        plan = self.plan
        if plan is None or plan.phase in ("above", "descend"):
            return 0.0
        return float(np.hypot(*(self.tip(data)[:2] - self.mouth[:2])) / 0.01)

    def success(self, data: mujoco.MjData) -> bool:
        """In the hole, and never pressed harder than the part takes."""
        gentle = max(self.peak_down, self.peak_side) <= self.params.limit
        return self.inserted_at is not None and self.is_in(data) and gentle

    def report(self, data: mujoco.MjData) -> dict[str, Any]:
        out: dict[str, Any] = {
            "depth_mm": round(1000 * self.depth_reached, 1),
            "of_mm": round(1000 * (self.mouth[2] - self.floor_z), 1),
            "tilt_deg": round(math.degrees(self.tilt), 1),
            "peak_down_n": round(self.peak_down, 1),
            "peak_sideways_n": round(self.peak_side, 1),
            "damaged": bool(max(self.peak_down, self.peak_side) > self.params.limit),
            "phase": self.plan.phase if self.plan is not None else "idle",
        }
        if self.inserted_at is not None:
            out["in_at_s"] = round(self.inserted_at, 2)
        return out


class InsertPlan:
    """Over the hole where the robot thinks it is, down until the peg touches, then pushing it in."""

    def __init__(self, task: Insert, data: mujoco.MjData):
        self.task = task
        p = task.params
        self.t0 = float(data.time)
        self.t_phase = self.t0
        self.phase = "above"
        self.n = np.array([0.0, 0.0, 1.0])
        self.axis = -self.n
        self.believed = task.mouth + np.array([p.error[0], p.error[1], 0.0])
        self.start = data.site_xpos[task.site].copy()
        offset = data.site_xpos[task.site] - task.tip(data)  # tool point minus the peg's tip
        self.tip_off = offset
        want = np.array([self.believed[0], self.believed[1], task.mouth[2] + HOVER_ABOVE]) + offset
        self.hover = want
        self.above_s = float(np.clip(np.linalg.norm(want - self.start) / TRAVEL_SPEED, 1.0, 4.0))
        self.travel = 0.0
        self.last_t = self.t0
        self.frozen: np.ndarray | None = None
        # The peg's square meets the hole's only turned the right way: the hand turns the peg to the nearest of the
        # hole's four ways (a peg gripped across a corner would meet the funnel with its diagonal).
        hand = data.site_xmat[task.site].reshape(3, 3).copy()
        flat = pointing(self.axis, hand)
        peg_yaw = _yaw(data.xmat[task.peg_body].reshape(3, 3))
        hole_yaw = _yaw(data.xmat[task.hole_body].reshape(3, 3))
        quarter = math.pi / 2
        turn = (hole_yaw - peg_yaw + quarter / 2) % quarter - quarter / 2
        c, s = math.cos(turn), math.sin(turn)
        self.start_rot = hand
        self.end_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ flat
        self.rot = hand

    def target(self, data: mujoco.MjData, feel: np.ndarray) -> ToolTarget:
        t = float(data.time)
        dt, self.last_t = t - self.last_t, t
        p = self.task.params
        n = self.n
        pushing = max(0.0, float(feel @ n))
        if self.phase == "above":
            s, ds = min_jerk((t - self.t_phase) / self.above_s)
            delta = self.hover - self.start
            self.rot = blend_rotation(self.start_rot, self.end_rot, s)
            if t - self.t_phase >= self.above_s:
                self._enter("descend", t)
            return ToolTarget(self.start + s * delta, ds * delta / self.above_s, rot=self.rot, normal=n)
        tcp = data.site_xpos[self.task.site]
        tip = tcp - self.tip_off
        # the tool point that puts the peg's tip over the believed hole, from where the peg really is in the hand
        xy = self.believed[:2] + (tcp[:2] - tip[:2])
        if self.phase == "descend":
            height = float(tip[2] - self.task.mouth[2])
            speed = DESCENT_SPEED if height < SLOW_ZONE else FAST_DESCENT
            self.travel = min(self.travel + speed * dt, float(self.hover[2] - tcp[2]) + MAX_LAG)
            z = self.hover[2] - self.travel
            if pushing >= min(CONTACT_N, p.force):
                self.frozen = tcp.copy()
                self._enter("insert", t)
            return ToolTarget(np.array([xy[0], xy[1], z]), -n * speed, rot=self.rot, normal=n)
        assert self.frozen is not None
        k = min(1.0, (t - self.t_phase) / SETTLE_S)
        # aimed `depth` below the mouth: a position controller pushes with its stiffness times how far that is
        depth = p.depth if p.depth is not None else self.task.mouth[2] - self.task.floor_z + 0.002
        z = self.task.mouth[2] - depth + self.tip_off[2]
        pos = np.array([xy[0], xy[1], z])
        if self.phase == "insert" and (self.task.is_in(data) or t - self.t_phase > 9.0):
            self._enter("done", t)
        if self.phase == "done":
            return ToolTarget(
                pos,
                np.zeros(3),
                rot=self.rot,
                normal=n,
                push=p.force,
                pressing=True,
                depth=depth,
                anchored=False,
                compliance=1.0,
            )
        return ToolTarget(
            pos,
            np.zeros(3),
            rot=self.rot,
            normal=n,
            push=p.force * k,
            pressing=True,
            depth=depth,
            anchored=False,
            compliance=k,
        )

    def _enter(self, phase: str, t: float) -> None:
        self.phase, self.t_phase = phase, t


def _yaw(frame: np.ndarray) -> float:
    """The heading of a frame's x axis about the vertical, rad."""
    return math.atan2(float(frame[1, 0]), float(frame[0, 0]))
