"""Start, stop and run behaviors on the robots of the live session."""

from __future__ import annotations

import contextlib
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge.behaviors import Kind, humanoid, quadruped, robot_kind
from robotforge.behaviors.arm import REACH_TOL, ArmMotion
from robotforge.behaviors.balance import WheelBalancer
from robotforge.behaviors.drive import DiffDrive
from robotforge.behaviors.rhythm_walk import RhythmWalker, SavedWalk, load_walk
from robotforge.behaviors.snake import SnakeGait
from robotforge.design import library
from robotforge.design.world import ObjectEntry, World
from robotforge.sim.session import SimSession

MOBILE = (Kind.QUADRUPED, Kind.HEXAPOD, Kind.ROVER, Kind.HUMANOID, Kind.BALANCER, Kind.SNAKE)
WALKERS = (quadruped.QuadrupedGait, DiffDrive, humanoid.HumanoidController, WheelBalancer, SnakeGait)
SELF_BALANCING = (Kind.HUMANOID, Kind.BALANCER)  # they fall over without a controller
LEGGED = (Kind.QUADRUPED, Kind.HUMANOID)  # can walk with a learned policy (move walker=...)
RHYTHM = "rhythm"  # walker: the walk a robot learned for itself in the RL II lesson and kept with its world
LEARN_A_WALK = (
    "it can learn one: open the viewer, press Learn, pick 'RL II: any body can learn to walk', choose your "
    "robot, press Learn, then 'Keep this walk for my robot'"
)


class BehaviorError(ValueError):
    """User-facing: says what to do instead."""


class BehaviorManager:
    def __init__(
        self,
        session: SimSession,
        world: Callable[[], World | None],
        world_dir: Callable[[], Path | None] = lambda: None,
    ):
        self.session = session
        self._world = world
        self._world_dir = world_dir  # where the open world keeps its files (walks/ among them); None in lessons
        session.reset_hooks.append(self.start_balancing)
        self.start_balancing()

    def start_balancing(self) -> None:
        """Humanoids and two-wheeled robots keep their balance from the moment they appear and after every
        reset, so a beginner never sees one hang limp from its servos or topple at the first nudge."""
        world = self._world()
        names = [r.name for r in world.robots] if world else sorted(self._prefixes())
        told = {r.name: _told_kind(r) for r in world.robots} if world else {}
        with self.session.lock:
            for name in names:
                kind = robot_kind(self.session.model, name, told.get(name))
                if kind in SELF_BALANCING:
                    with contextlib.suppress(BehaviorError):
                        self._mobile_controller(name, kind, create=True)

    # ---- robots in the session -------------------------------------------------------

    def robots(self) -> list[dict[str, Any]]:
        m = self.session.model
        world = self._world()
        names = [r.name for r in world.robots] if world else sorted(self._prefixes())
        told = {r.name: _told_kind(r) for r in world.robots} if world else {}
        models = {r.name: library.get(r.library).title for r in world.robots if r.library} if world else {}
        out = []
        for name in names:
            kind = robot_kind(m, name, told.get(name))
            info: dict[str, Any] = {"name": name, "kind": kind.value}
            if name in models:
                info["model"] = models[name]
            arms = self.arms(name)
            if arms and arms != [name]:  # e.g. an arm riding on a rover: drive the robot, work the arm
                info["arms"] = arms
            if kind in MOBILE and self._real_humanoid(name):  # its policy's card says, without loading it
                learned = self.learned_walkers().get(name)
                if learned is not None:
                    info["max_speed"] = round(learned["max_speed"], 3)
            elif kind in MOBILE:
                with contextlib.suppress(BehaviorError):
                    info["max_speed"] = round(self._mobile_controller(name, kind, create=False).max_speed, 3)
            walk = self._kept_walk_or_none(name)
            if walk is not None:
                info["learned_walk"] = walk.describe()
                info.setdefault("max_speed", round(walk.speed, 3))  # robots no gait code knows drive with it
            elif kind not in MOBILE and self._can_learn_walk(name):
                info["walk_hint"] = f"no gait code knows this body, so move cannot walk it yet; {LEARN_A_WALK}"
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
            return robot_kind(self.session.model, robot, _told_kind(entry))
        if robot not in self._prefixes():
            raise BehaviorError(f"no robot named {robot!r}")
        return robot_kind(self.session.model, robot)

    def home_qpos(self) -> np.ndarray:
        m = self.session.model
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        return m.key_qpos[key].copy() if key >= 0 else m.qpos0.copy()

    # ---- moving robots (legs and wheels) ---------------------------------------------

    def move(self, robot: str, speed: float, turn: float, walker: str | None = None) -> dict[str, Any]:
        """walker (legged robots): "gait" = the built-in walking controller, "rl" = the learned policy
        installed for this robot, or a policy name; None keeps the robot's current walker."""
        if not (math.isfinite(speed) and math.isfinite(turn)):
            raise BehaviorError(f"speed and turn must be finite numbers, got speed={speed} turn={turn}")
        kind = self.kind(robot)
        if kind not in MOBILE and walker in (None, RHYTHM):
            if self.kept_walk(robot) is None:
                raise BehaviorError(self._cannot_move(robot, kind))
        elif kind not in MOBILE:
            raise BehaviorError(self._cannot_move(robot, kind))
        if walker == "gait" and kind == Kind.HEXAPOD:
            walker = None  # its built-in gait is the only one it has
        if walker not in (None, RHYTHM) and kind not in LEGGED:
            raise BehaviorError(
                f"{robot} is a {kind.value}; walker {walker!r} needs a quadruped or a humanoid, leave it out"
            )
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, create=True, walker=walker)
            _check_standing(ctrl)
            ctrl.command(speed, turn)
        return ctrl.status()

    def _cannot_move(self, robot: str, kind: Kind) -> str:
        text = (
            f"{robot} is a {kind.value} robot; only legged robots (quadrupeds, hexapods, humanoids), wheeled "
            "robots (rovers, balancers), snakes and robots that learned a walk of their own can move around"
        )
        if self._can_learn_walk(robot):
            text += f". {robot} has no walk yet, but {LEARN_A_WALK}"
        return text

    def walk_notes(self) -> list[str]:
        """For a world's load result: robots that walk only with a walk they learned, or could learn one."""
        world = self._world()
        if world is None:
            return []
        out = []
        for entry in world.robots:
            if robot_kind(self.session.model, entry.name, _told_kind(entry)) in MOBILE:
                continue
            walk = self._kept_walk_or_none(entry.name)
            if walk is not None:
                turning = "and turns on the spot" if walk.turns else "without turning (no turns learned yet)"
                out.append(
                    f"{entry.name} walks with the walk it learned: move drives it forward at its own pace "
                    f"(about {walk.speed:.2f} m/s) {turning}"
                )
            elif self._can_learn_walk(entry.name):
                out.append(f"{entry.name}: no gait code knows this body, so move cannot walk it yet; {LEARN_A_WALK}")
        return out

    def _can_learn_walk(self, robot: str) -> bool:
        from robotforge.control.cpg import Body, CPGError

        try:
            Body.find(self.session.model, robot)
        except CPGError:
            return False
        return True

    def kept_walk(self, robot: str) -> SavedWalk | None:
        """The walk `robot` learned in the RL II lesson and kept in this world's folder, if any."""
        directory = self._world_dir()
        if directory is None:
            return None
        try:
            return load_walk(directory, robot)
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc

    def _kept_walk_or_none(self, robot: str) -> SavedWalk | None:
        try:
            return self.kept_walk(robot)
        except BehaviorError:
            return None  # listing robots should not fail on a damaged file; moving it says what is wrong

    def _rhythm_walker(self, robot: str, existing: Any, create: bool) -> RhythmWalker:
        walk = self.kept_walk(robot)
        if walk is None:
            raise BehaviorError(f"{robot} has not learned a walk of its own; {LEARN_A_WALK}")
        if isinstance(existing, RhythmWalker) and existing.walk == walk:
            return existing
        try:
            ctrl = RhythmWalker(self.session.model, robot, walk, self.home_qpos())
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def _mobile_controller(self, robot: str, kind: Kind, create: bool, walker: str | None = None):
        existing = self.session.controllers.get(robot)
        if walker is None and _is_mover(existing):  # keep whatever walks it now, e.g. for WASD driving
            return existing
        if walker == RHYTHM or kind not in MOBILE:
            return self._rhythm_walker(robot, existing, create)
        if walker not in (None, "gait") and kind in LEGGED:
            return self._policy_walker(robot, walker, existing, create)
        if isinstance(existing, WALKERS):
            return existing
        m = self.session.model
        try:
            if kind == Kind.HUMANOID and self._real_humanoid(robot):
                # A real humanoid (not built from our template) walks with its learned policy only.
                return self._policy_walker(robot, "rl", existing, create)
            if kind == Kind.HUMANOID:
                ctrl = humanoid.for_template(m, robot, self.home_qpos())
            elif kind in (Kind.QUADRUPED, Kind.HEXAPOD):
                ctrl = quadruped.for_robot(m, robot, self.home_qpos())
            elif kind == Kind.BALANCER:
                ctrl = WheelBalancer(m, robot, self.home_qpos())
            elif kind == Kind.SNAKE:
                ctrl = SnakeGait(m, robot, self.home_qpos())
            else:
                ctrl = DiffDrive(m, robot)
        except (ValueError, KeyError) as exc:
            if walker is None and self._kept_walk_or_none(robot) is not None:  # a body the gait code cannot walk
                return self._rhythm_walker(robot, existing, create)
            raise BehaviorError(str(exc)) from exc
        top = self._library_top_speed(robot)
        if top is not None and isinstance(ctrl, quadruped.QuadrupedGait):
            ctrl.limit_speed(top)
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def _real_humanoid(self, robot: str) -> bool:
        """A humanoid from the library (like G1), as opposed to one built from our template."""
        world = self._world()
        entry = next((r for r in world.robots if r.name == robot), None) if world else None
        return entry is not None and entry.library is not None and library.get(entry.library).kind == "humanoid"

    def _policy_walker(self, robot: str, walker: str, existing: Any, create: bool):
        """A learned walking policy (ONNX, optional extra `policy`) in place of the gait."""
        from robotforge.bridge.onnx_policy import PolicyUnavailable, PolicyWalker, walker_for

        if isinstance(existing, PolicyWalker) and walker in ("rl", existing.spec.name):
            return existing
        world = self._world()
        entry = next((r for r in world.robots if r.name == robot), None) if world else None
        try:
            ctrl = walker_for(self.session.model, robot, entry.library if entry else None, walker)
        except (PolicyUnavailable, ValueError) as exc:
            raise BehaviorError(str(exc)) from exc
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def learned_walkers(self) -> dict[str, dict[str, Any]]:
        """Legged robots that have a learned walking policy: {robot: {policy, max_speed, active}}."""
        from robotforge.bridge import policies  # light: cards only, no onnxruntime

        world = self._world()
        if world is None:
            return {}
        out = {}
        for entry in world.robots:
            found = policies.for_robot(entry.library)
            if found is None or robot_kind(self.session.model, entry.name, _told_kind(entry)) not in LEGGED:
                continue
            active = self.session.controllers.get(entry.name)
            out[entry.name] = {
                "policy": found.spec.name,
                "max_speed": found.spec.commands.lin_vel_x[1],
                "active": _is_policy_walker(active),
            }
        return out

    def _library_top_speed(self, robot: str) -> float | None:
        world = self._world()
        entry = next((r for r in world.robots if r.name == robot), None) if world else None
        return library.get(entry.library).max_speed if entry is not None and entry.library else None

    def stop(self, robot: str) -> dict[str, Any]:
        with self.session.lock:
            ctrl = self.session.controllers.get(robot)
            if _is_mover(ctrl):
                ctrl.command(0.0, 0.0)  # keep the gait so a dog settles into a stand instead of collapsing
                return {"robot": robot, "stopped": True}
            if ctrl is not None:
                self.session.set_controller(robot, None)
            return {"robot": robot, "stopped": ctrl is not None}

    def settled(self, robot: str) -> Callable[[], bool] | None:
        """For robots that keep stepping for a moment after `stop`: a check for when they stand still."""
        ctrl = self.session.controllers.get(robot)
        if isinstance(ctrl, humanoid.HumanoidController):
            return lambda: ctrl.done
        return None

    def strike_pose(self, robot: str, name: str) -> humanoid.HumanoidController:
        """Start a named humanoid pose (wave, squat, ...); a walking humanoid stops first."""
        kind = self.kind(robot)
        if kind != Kind.HUMANOID:
            raise BehaviorError(f"{robot} is a {kind.value}; only humanoids strike poses")
        if name not in humanoid.POSES:
            raise BehaviorError(f"unknown pose {name!r}; choose one of: {', '.join(humanoid.POSES)}")
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, create=True, walker="gait")  # poses need the built-in one
            _check_standing(ctrl)
            ctrl.play(name)
        return ctrl

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

    def arms(self, robot: str) -> list[str]:
        """The arms of a robot: the robot itself for an arm, "<robot>/<part>" for each arm mounted on it."""
        m = self.session.model
        found = []
        for s in range(m.nsite):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ""
            if name == f"{robot}/tcp":
                found.append(robot)
            elif name.startswith(f"{robot}/") and name.endswith("/tcp"):
                found.append(name.removesuffix("/tcp"))
        return found

    def _arm_name(self, robot: str) -> str:
        """Which arm a command means: "rover/arm" names one; a robot with a single arm stands for it."""
        if "/" in robot:
            owner = robot.split("/", 1)[0]
            self.kind(owner)  # unknown robots get the usual message
            arms = self.arms(owner)
            if robot not in arms:
                raise BehaviorError(f"{robot!r} is not an arm; arms on {owner}: {', '.join(arms) or 'none'}")
            return robot
        kind = self.kind(robot)
        arms = self.arms(robot)
        if not arms:
            raise BehaviorError(f"{robot} is a {kind.value}, not an arm, and carries none")
        if len(arms) > 1:
            raise BehaviorError(f"{robot} has {len(arms)} arms; name the one you mean: {', '.join(arms)}")
        return arms[0]

    def _arm(self, robot: str) -> ArmMotion:
        name = self._arm_name(robot)
        existing = self.session.controllers.get(name)
        if isinstance(existing, ArmMotion):
            existing.sync(self.session.data)
            return existing
        try:
            arm = ArmMotion(self.session.model, name, self.home_qpos())
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        # Continue from where the arm (and whatever carries it) is now, not from the home pose.
        arm.sync(self.session.data)
        arm.q = self.session.data.qpos[arm.chain.qpos_adr].copy()
        self.session.set_controller(name, arm)
        return arm

    def move_arm(self, robot: str, position: list[float], gripper: str | None = None) -> ArmMotion:
        target = np.asarray(position, float)
        if target.shape != (3,):
            raise BehaviorError("position is [x, y, z] in meters (world frame)")
        with self.session.lock:
            arm = self._arm(robot)
            if arm.reach_error(target) > REACH_TOL:
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


def _told_kind(entry: Any) -> str | None:
    """The kind a world entry declares: its template, or for a library robot what the catalog says
    (G1's joints are not named like our template's, so its shape alone does not say humanoid)."""
    return entry.template or (library.get(entry.library).kind if entry.library else None)


def _is_mover(ctrl: Any) -> bool:
    """A controller that walks or drives on speed and turn commands (gaits, wheels, learned walks)."""
    return isinstance(ctrl, (*WALKERS, RhythmWalker)) or _is_policy_walker(ctrl)


def _is_policy_walker(ctrl: Any) -> bool:
    # No import here: the bridge pulls in the control layer. Without the module loaded, no walker exists; nor
    # while another thread is still importing it (a lesson with the G1 starting): the module is listed already
    # but has no PolicyWalker yet, and asking for it then killed the viewer's stream.
    walker = getattr(sys.modules.get("robotforge.bridge.onnx_policy"), "PolicyWalker", None)
    return walker is not None and isinstance(ctrl, walker)


def _check_standing(ctrl) -> None:
    if getattr(ctrl, "fallen", False):
        raise BehaviorError(
            f"{ctrl.robot} has fallen over and cannot get up by itself; reset the simulation (sim_control reset)"
        )
