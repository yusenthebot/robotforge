"""Start, stop and run behaviors on the robots of the live session."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import Kind, quadruped, robot_kind
from robotforge.behaviors.arm import ArmMotion
from robotforge.behaviors.drive import DiffDrive
from robotforge.design import library
from robotforge.design.world import ObjectEntry, World
from robotforge.sim.session import SimSession

MAX_RUN_SECONDS = 60.0


class BehaviorError(ValueError):
    """User-facing: says what to do instead."""


class BehaviorManager:
    def __init__(self, session: SimSession, world: Callable[[], World | None]):
        self.session = session
        self._world = world

    # ---- robots in the session -------------------------------------------------------

    def robots(self) -> list[dict[str, Any]]:
        m = self.session.model
        world = self._world()
        names = [r.name for r in world.robots] if world else sorted(self._prefixes())
        templates = {r.name: r.template for r in world.robots} if world else {}
        models = {r.name: library.get(r.library).title for r in world.robots if r.library} if world else {}
        out = []
        for name in names:
            kind = robot_kind(m, name, templates.get(name))
            info: dict[str, Any] = {"name": name, "kind": kind.value}
            if name in models:
                info["model"] = models[name]
            if kind in (Kind.QUADRUPED, Kind.ROVER):
                with contextlib.suppress(BehaviorError):
                    info["max_speed"] = round(self._mobile_controller(name, kind, create=False).max_speed, 3)
            out.append(info)
        return out

    def _prefixes(self) -> set[str]:
        m = self.session.model
        names = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "" for a in range(m.nu))
        return {n.split("/", 1)[0] for n in names if "/" in n}

    def kind(self, robot: str) -> Kind:
        world = self._world()
        if world is not None:
            entry = next((r for r in world.robots if r.name == robot), None)
            if entry is None:
                names = ", ".join(r.name for r in world.robots) or "none"
                raise BehaviorError(f"no robot named {robot!r}; robots in this world: {names}")
            return robot_kind(self.session.model, robot, entry.template)
        if robot not in self._prefixes():
            raise BehaviorError(f"no robot named {robot!r}")
        return robot_kind(self.session.model, robot)

    def home_qpos(self) -> np.ndarray:
        m = self.session.model
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        return m.key_qpos[key].copy() if key >= 0 else m.qpos0.copy()

    # ---- moving robots (legs and wheels) ---------------------------------------------

    def move(self, robot: str, speed: float, turn: float) -> dict[str, Any]:
        kind = self.kind(robot)
        if kind not in (Kind.QUADRUPED, Kind.ROVER):
            raise BehaviorError(f"{robot} is a {kind.value}; only quadrupeds and rovers can move around")
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, create=True)
            ctrl.command(speed, turn)
        return ctrl.status()

    def _mobile_controller(self, robot: str, kind: Kind, create: bool):
        existing = self.session.controllers.get(robot)
        if isinstance(existing, quadruped.QuadrupedGait | DiffDrive):
            return existing
        m = self.session.model
        try:
            ctrl = quadruped.for_robot(m, robot, self.home_qpos()) if kind == Kind.QUADRUPED else DiffDrive(m, robot)
        except (ValueError, KeyError) as exc:
            raise BehaviorError(str(exc)) from exc
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def stop(self, robot: str) -> dict[str, Any]:
        with self.session.lock:
            ctrl = self.session.controllers.get(robot)
            if isinstance(ctrl, quadruped.QuadrupedGait | DiffDrive):
                ctrl.command(0.0, 0.0)  # keep the gait so a dog settles into a stand instead of collapsing
                return {"robot": robot, "stopped": True}
            if ctrl is not None:
                self.session.set_controller(robot, None)
            return {"robot": robot, "stopped": ctrl is not None}

    def pose(self, robot: str) -> dict[str, Any]:
        """Where a robot is and which way it faces (degrees, counter-clockwise from +x)."""
        d = self.session.data
        root = self._root_body(robot)
        rot = d.xmat[root].reshape(3, 3)
        return {
            "robot": robot,
            "position": [round(float(v), 3) for v in d.xpos[root]],
            "heading_deg": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 1),
            "upright": bool(rot[2, 2] > 0.7),
        }

    def _root_body(self, robot: str) -> int:
        m = self.session.model
        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith(f"{robot}/"):
                return b
        raise BehaviorError(f"no robot named {robot!r}")

    # ---- arms ---------------------------------------------------------------------------

    def _arm(self, robot: str) -> ArmMotion:
        kind = self.kind(robot)
        if kind != Kind.ARM:
            raise BehaviorError(f"{robot} is a {kind.value}, not an arm")
        existing = self.session.controllers.get(robot)
        if isinstance(existing, ArmMotion):
            return existing
        arm = ArmMotion(self.session.model, robot, self.home_qpos())
        # Continue from where the arm is now, not from its home pose.
        arm.q = self.session.data.qpos[arm.chain.qpos_adr].copy()
        self.session.set_controller(robot, arm)
        return arm

    def move_arm(self, robot: str, position: list[float], gripper: str | None = None) -> ArmMotion:
        target = np.asarray(position, float)
        if target.shape != (3,):
            raise BehaviorError("position is [x, y, z] in meters (world frame)")
        with self.session.lock:
            arm = self._arm(robot)
            if arm.reach_error(target) > 0.008:
                raise BehaviorError(
                    f"{robot} cannot reach {position} with its gripper pointing down; try a point closer to its base"
                )
            arm.move_to(target, gripper)
        return arm

    def set_gripper(self, robot: str, state: str) -> ArmMotion:
        with self.session.lock:
            arm = self._arm(robot)
            if arm.gripper < 0:
                raise BehaviorError(f"{robot} has no gripper")
            arm.set_gripper(state)
        return arm

    def pick_and_place(self, robot: str, obj: str, target: str | list[float]) -> tuple[ArmMotion, dict[str, Any]]:
        with self.session.lock:
            arm = self._arm(robot)
            pick = self._object_geometry(obj)
            place, place_top, region = self._place_target(target)
            try:
                arm.pick_and_place(pick["center"], pick["half_height"], place, place_top)
            except ValueError as exc:
                raise BehaviorError(str(exc)) from exc
        return arm, {"object": obj, "target": target, "region": region}

    def placed(self, obj: str, region: dict[str, Any]) -> dict[str, Any]:
        """Check whether `obj` ended up inside the target region."""
        body = self._body(obj)
        pos = self.session.data.xpos[body]
        center = np.asarray(region["center"])
        half = np.asarray(region["half_xy"])
        inside = bool(np.all(np.abs(pos[:2] - center[:2]) <= half) and pos[2] >= region["min_z"])
        return {"success": inside, "object_position": [round(float(v), 3) for v in pos]}

    def _body(self, name: str) -> int:
        b = mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise BehaviorError(f"no movable object named {name!r}")
        return b

    def _object_entry(self, name: str) -> ObjectEntry | None:
        world = self._world()
        if world is None:
            return None
        return next((o for o in world.objects if o.name == name), None)

    def _object_geometry(self, name: str) -> dict[str, Any]:
        entry = self._object_entry(name)
        body = self._body(name)
        if entry is not None and entry.fixed:
            raise BehaviorError(f"{name} is fixed in place and cannot be picked up")
        if entry is not None and entry.shape in ("table", "bin"):
            raise BehaviorError(f"{name} is a {entry.shape}; pick up something small like a box or ball")
        center = self.session.data.xpos[body].copy()
        if entry is not None:
            half = {"box": entry.size[-1] / 2, "sphere": entry.size[0] / 2, "cylinder": entry.size[-1] / 2}[entry.shape]
        else:
            g = next(i for i in range(self.session.model.ngeom) if self.session.model.geom_bodyid[i] == body)
            half = float(self.session.model.geom_rbound[g])
        return {"center": center, "half_height": half}

    def _place_target(self, target: str | list[float]) -> tuple[np.ndarray, float, dict[str, Any]]:
        """(place point for the object's bottom, highest point around it, success region)."""
        if isinstance(target, list):
            if len(target) not in (2, 3):
                raise BehaviorError("target is an object name or a position [x, y]")
            p = np.array([target[0], target[1], target[2] if len(target) == 3 else 0.0])
            return p, p[2], {"center": p.tolist(), "half_xy": [0.04, 0.04], "min_z": p[2] - 0.01}
        entry = self._object_entry(target)
        if entry is None:
            raise BehaviorError(f"no object named {target!r} to place onto")
        m, d = self.session.model, self.session.data
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
        if b >= 0:
            base = d.xpos[b].copy()
        else:  # fixed objects are plain geoms
            base = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, target)].copy()
        length, width = entry.size[0], entry.size[1] if len(entry.size) > 1 else entry.size[0]
        if entry.shape == "bin":
            floor_z = base[2] + 0.005
            region = {"center": base.tolist(), "half_xy": [length / 2, width / 2], "min_z": floor_z - 0.01}
            return np.array([base[0], base[1], floor_z]), base[2] + entry.size[2], region
        if entry.shape == "table":
            top = entry.size[2]
            region = {"center": [base[0], base[1], top], "half_xy": [length / 2, width / 2], "min_z": top - 0.01}
            return np.array([base[0], base[1], top]), top, region
        top = base[2] + (entry.size[-1] / 2 if entry.shape != "sphere" else entry.size[0] / 2)
        region = {"center": [base[0], base[1], top], "half_xy": [length / 2, width / 2], "min_z": top - 0.01}
        return np.array([base[0], base[1], top]), top, region

    # ---- status --------------------------------------------------------------------------

    def status(self) -> list[dict[str, Any]]:
        return [c.status() for c in self.session.controllers.values()]
