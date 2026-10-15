"""Arm motions: move the gripper to points in space and pick-and-place objects.

A motion is a queue of Cartesian waypoints. Each control tick interpolates the
tool point along a smooth path and solves IK with the gripper pointing down.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from robotforge.behaviors.ik import Chain, IKSolver
from robotforge.sim.actuators import drives_joint

DOWN = np.array([0.0, 0.0, -1.0])
SPEED = 0.18  # m/s tool speed
APPROACH = 0.07  # m above a grasp or place point


@dataclass
class Waypoint:
    position: np.ndarray | None  # tool target in world frame; None = stay
    gripper: float | None  # gripper ctrl value; None = unchanged
    duration: float
    label: str


class ArmMotion:
    rate_hz = 100.0

    def __init__(self, model: mujoco.MjModel, robot: str, home_qpos: np.ndarray):
        """`robot` names the arm: a robot ("arm") or an arm mounted on one ("rover/arm")."""
        self.robot = robot
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/tcp")
        if site < 0:
            raise ValueError(f"robot {robot!r} has no tool point; is it an arm?")
        self.site = site
        tool_body = int(model.site_bodyid[site])
        joint_to_act = {int(model.actuator_trnid[a, 0]): a for a in range(model.nu) if drives_joint(model, a)}
        # The arm is the powered hinges between the robot's root and the tool: that leaves out wheels,
        # legs or a second arm of the same robot, and works for arms built from parts too.
        joints = _path_hinges(model, tool_body, joint_to_act)
        if not joints:
            raise ValueError(f"{robot!r} has a tool point but no powered joints that move it")
        self.chain = Chain.from_joints(model, joints, tool_body, model.site_pos[site].copy())
        self.actuators = np.array([joint_to_act[j] for j in joints])
        # The tool points along the tcp site's +z (for the templates that is the hand's +z).
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.site_quat[site])
        self.tool_axis = rot.reshape(3, 3)[:, 2].copy()
        self.gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{robot}/gripper")
        self.gripper_open = float(model.actuator_ctrlrange[self.gripper][1]) if self.gripper >= 0 else 0.0
        self.ik = IKSolver(model, home_qpos)
        self.q = home_qpos[self.chain.qpos_adr].copy()
        # An arm on a moving robot: IK must start from where that robot is now (see sync).
        root = _root_body(model, tool_body)
        free = [j for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root])]
        free = [j for j in free if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_adr = int(model.jnt_qposadr[free[0]]) if free else None
        # Paths sweep around the base axis so the tool never cuts through the arm itself.
        self.base_body = int(model.jnt_bodyid[joints[0]])
        self._base_parent = int(model.body_parentid[self.base_body])
        self.ik.point(self.chain, self.q)
        self._read_base(self.ik.data)
        self.queue: deque[Waypoint] = deque()
        self._segment: tuple[np.ndarray, Waypoint, float] | None = None  # (start, waypoint, t0)
        self.current_label = "idle"
        self.log: list[str] = []

    def _read_base(self, data: mujoco.MjData) -> None:
        self.base_xy = data.xpos[self.base_body][:2].copy()
        # Heading of the arm's zero yaw in the world (the arm may be placed or carried with a yaw).
        parent_rot = data.xmat[self._base_parent].reshape(3, 3)
        self.base_yaw = float(np.arctan2(parent_rot[1, 0], parent_rot[0, 0]))

    def sync(self, data: mujoco.MjData) -> None:
        """Follow the robot carrying the arm: IK and path planning start from where it is now."""
        if self.free_adr is not None:
            self.ik.base_qpos[self.free_adr : self.free_adr + 7] = data.qpos[self.free_adr : self.free_adr + 7]
        self._read_base(data)

    # ---- planning ------------------------------------------------------------------

    def reach_error(self, target: np.ndarray) -> float:
        _, err = self.ik.solve(self.chain, target, self._seed(target), iterations=60, axis=(self.tool_axis, DOWN))
        return err

    def _seed(self, target: np.ndarray) -> np.ndarray:
        """Start IK with the base already turned toward the target; DLS alone can stall on big turns."""
        q = self.q.copy()
        d = np.asarray(target)[:2] - self.base_xy
        if np.linalg.norm(d) > 1e-3:
            lo, hi = self.chain.lower[0], self.chain.upper[0]
            want = float((np.arctan2(d[1], d[0]) - self.base_yaw + np.pi) % (2 * np.pi) - np.pi)
            # The same heading, one turn either way: take the one inside the limits closest to now.
            inside = [a for a in (want, want - 2 * np.pi, want + 2 * np.pi) if lo - 1e-9 <= a <= hi + 1e-9]
            q[0] = min(inside, key=lambda a: abs(a - q[0])) if inside else np.clip(want, lo, hi)
        return q

    def move_to(self, target, gripper: str | None = None, from_point: np.ndarray | None = None) -> None:
        target = np.asarray(target, float)
        start = from_point if from_point is not None else self.tool_target_end()
        dist = float(np.linalg.norm(target - start)) if start is not None else 0.2
        self.queue.append(Waypoint(target, self._grip(gripper), max(0.5, dist / SPEED), "move"))

    def set_gripper(self, state: str, wait: float = 0.5) -> None:
        self.queue.append(Waypoint(None, self._grip(state), wait, f"gripper {state}"))

    def pick_and_place(
        self, obj_pos: np.ndarray, obj_half_height: float, place: np.ndarray, place_top: float | None = None
    ) -> None:
        """Queue a full pick-and-place. Raises ValueError if any key point is out of reach.

        `place` is where the object's bottom should end up; `place_top` is the highest point
        around it (e.g. a bin's rim), which the carried object must clear.
        """
        grasp = obj_pos.copy()
        drop = place + [0, 0, obj_half_height + 0.01]
        rim = max(place_top if place_top is not None else place[2], grasp[2] - obj_half_height)
        travel_z = max(grasp[2], drop[2]) + APPROACH
        travel_z = max(travel_z, rim + 2 * obj_half_height + 0.03)
        above = np.array([grasp[0], grasp[1], travel_z])
        above_drop = np.array([drop[0], drop[1], travel_z])
        bad = self.unreachable({"object": grasp, "above object": above, "place": drop, "above place": above_drop})
        if bad:
            raise ValueError(
                f"{self.robot} cannot reach the {' and the '.join(bad)} with its gripper pointing down; "
                "move the object/target closer to the arm or give the arm a longer reach"
            )
        start = self.tool_target_end()
        self.queue.append(Waypoint(above, self._grip("open"), self._time(start, above), "approach"))
        self.queue.append(Waypoint(grasp, None, self._time(above, grasp) + 0.3, "descend"))
        self.queue.append(Waypoint(None, self._grip("close"), 0.6, "grasp"))
        self.queue.append(Waypoint(above, None, self._time(grasp, above) + 0.3, "lift"))
        self.queue.append(Waypoint(above_drop, None, self._time(above, above_drop), "carry"))
        self.queue.append(Waypoint(drop, None, self._time(above_drop, drop) + 0.3, "lower"))
        self.queue.append(Waypoint(None, self._grip("open"), 0.5, "release"))
        self.queue.append(Waypoint(above_drop, None, self._time(drop, above_drop), "retreat"))

    def tool_target_end(self) -> np.ndarray:
        for wp in reversed(self.queue):
            if wp.position is not None:
                return wp.position
        if self._segment is not None and self._segment[1].position is not None:
            return self._segment[1].position
        return self.ik.point(self.chain, self.q)

    @property
    def done(self) -> bool:
        return not self.queue and self._segment is None

    def status(self) -> dict:
        return {
            "behavior": "arm",
            "robot": self.robot,
            "step": self.current_label,
            "remaining_steps": len(self.queue),
            "done": self.done,
        }

    # ---- control -------------------------------------------------------------------

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.free_adr is not None:
            self.sync(data)
        t = data.time
        if self._segment is None and self.queue:
            wp = self.queue.popleft()
            start = data.site_xpos[self.site].copy()
            self._segment = (start, wp, t)
            self.current_label = wp.label
            if wp.gripper is not None and self.gripper >= 0:
                data.ctrl[self.gripper] = wp.gripper
        if self._segment is None:
            self.current_label = "idle"
            data.ctrl[self.actuators] = self.q
            return
        start, wp, t0 = self._segment
        u = min(1.0, (t - t0) / wp.duration)
        if wp.position is not None:
            s = u * u * (3 - 2 * u)  # smoothstep
            target = self._arc_point(start, wp.position, s)
            self.q, _ = self.ik.solve(self.chain, target, self.q, iterations=6, axis=(self.tool_axis, DOWN))
        data.ctrl[self.actuators] = self.q
        if u >= 1.0:
            self._segment = None

    def _arc_point(self, a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
        """Interpolate in cylindrical coordinates (radius, angle, height) about the base axis."""
        da, db = a[:2] - self.base_xy, b[:2] - self.base_xy
        ra, rb = np.linalg.norm(da), np.linalg.norm(db)
        if min(ra, rb) < 1e-3:
            return a + (b - a) * s
        ta = np.arctan2(da[1], da[0])
        r, t = ra + (rb - ra) * s, ta + self._sweep(ta, np.arctan2(db[1], db[0])) * s
        return np.array([*(self.base_xy + r * np.array([np.cos(t), np.sin(t)])), a[2] + (b[2] - a[2]) * s])

    def _sweep(self, from_angle: float, to_angle: float) -> float:
        """Signed turn from one world heading to another that keeps the base joint inside its limits."""
        lo, hi = self.chain.lower[0], self.chain.upper[0]
        start = np.clip(from_angle - self.base_yaw, lo, hi)
        goal = to_angle - self.base_yaw
        options = [
            g
            for g in (goal, goal - 2 * np.pi, goal + 2 * np.pi, goal - 4 * np.pi, goal + 4 * np.pi)
            if lo - 1e-6 <= g <= hi + 1e-6
        ]
        if not options:
            return float((to_angle - from_angle + np.pi) % (2 * np.pi) - np.pi)
        return float(min(options, key=lambda g: abs(g - start)) - start)

    def unreachable(self, points: dict[str, np.ndarray], tol: float = 0.008) -> list[str]:
        """Names of points the tool cannot reach while pointing down."""
        return [name for name, p in points.items() if self.reach_error(p) > tol]

    def _grip(self, state: str | None) -> float | None:
        if state is None or self.gripper < 0:
            return None
        return self.gripper_open if state == "open" else 0.0

    @staticmethod
    def _time(a: np.ndarray, b: np.ndarray) -> float:
        return max(0.5, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / SPEED)


def _path_hinges(model: mujoco.MjModel, body: int, powered: dict[int, int]) -> list[int]:
    """Powered hinge joints from the root of `body`'s robot down to `body`, in that order."""
    joints: list[int] = []
    while body > 0:
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        own = [
            j for j in range(start, start + count) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j in powered
        ]
        joints = own + joints
        body = int(model.body_parentid[body])
    return joints


def _root_body(model: mujoco.MjModel, body: int) -> int:
    while model.body_parentid[body] > 0:
        body = int(model.body_parentid[body])
    return body
