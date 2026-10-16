"""Start, stop and run behaviors on the robots of the live session."""

from __future__ import annotations

import contextlib
import math
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.behaviors import Kind, humanoid, quadruped, robot_kind
from robotforge.behaviors.arm import REACH_TOL, ArmMotion, OutOfReach
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
# Legged walkers stand on their servos alone, but dragged in the viewer before their first move they rolled onto
# their backs (dog, Go2 and a dog built from parts: 180 degrees; scripts/push_bench.py --states fresh). Their gait
# steps along with a push (quadruped.PUSH_DRIFT) and stands exactly as the servos do, so in an open world they
# stand on it from the start.
STEPPING = (Kind.QUADRUPED, Kind.HEXAPOD)
LEGGED = (Kind.QUADRUPED, Kind.HUMANOID)  # can walk with a learned policy (move walker=...)
RHYTHM = "rhythm"  # walker: the walk a robot learned for itself in the RL II lesson and kept with its world
LEARN_A_WALK = (
    "it can learn one: open the viewer, press Learn, pick 'RL II: any body can learn to walk', choose your "
    "robot, press Learn, then 'Keep this walk for my robot'"
)


GAIT_WALKERS = ("gait", "rl", RHYTHM)  # the walker names that are not a policy's own name
NUMBER_LIST = re.compile(r"\s*[\[(]?\s*-?\d+(?:\.\d*)?(?:[\s,;]+-?\d+(?:\.\d*)?)+\s*[\])]?\s*")


class BehaviorError(ValueError):
    """User-facing: says what to do instead."""


def _a(kind: Kind) -> str:
    """ "a quadruped", "an arm"; a robot no behavior code recognizes is "a robot of no known kind"."""
    if kind == Kind.OTHER:
        return "a robot of no known kind"
    return f"an {kind.value}" if kind.value[0] in "aeiou" else f"a {kind.value}"


def _how_to_move(kind: Kind) -> str:
    """What to call for a robot of this kind when the agent used the wrong tool for it."""
    if kind == Kind.ARM:
        return "it works with move_arm, gripper and pick_and_place"
    if kind in MOBILE:
        return "it moves with move" + (" and strikes poses with pose" if kind == Kind.HUMANOID else "")
    return "robots lists what each robot can do"


def _clamp_notes(ctrl: Any, speed: float, turn: float) -> list[str]:
    """Say so when a robot was asked for more than it can do: the controllers clip speed and turn silently."""
    top = getattr(ctrl, "max_speed", None)
    got_speed = getattr(ctrl, "command_speed", getattr(ctrl, "speed", speed))
    got_turn = getattr(ctrl, "command_turn", getattr(ctrl, "turn", turn))
    notes = []
    if top is not None and abs(speed) > top + 1e-9 and abs(got_speed - speed) > 1e-9:
        notes.append(f"speed {speed:g} m/s is above max_speed {top:.3g}, so it moves at {got_speed:.3g}")
    if abs(got_turn - turn) > 1e-9:
        note = f"turn {turn:g} rad/s is above the limit for this robot, so it turns at {got_turn:g}"
        if abs(turn) > 2 * math.pi:
            note += f" (turn is in rad/s; {turn:g} degrees/s is {math.radians(turn):.2f})"
        notes.append(note)
    return notes


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
        self._standing: dict[str, Any] = {}  # robot -> the gait start_balancing stood it on, until it is moved
        session.reset_hooks.append(self.start_balancing)
        session.ctrl_hooks.append(self._set_by_hand)
        self.start_balancing()

    def start_balancing(self) -> None:
        """Humanoids and two-wheeled robots keep their balance from the moment they appear and after every
        reset, so a beginner never sees one hang limp from its servos or topple at the first nudge.
        In an open world, legged walkers stand on their gait from the start too (see STEPPING). Not in lessons,
        which drive their robots themselves, nor in a plain model file."""
        world = self._world()
        names = self._names(world)
        told = {r.name: _told_kind(r) for r in world.robots} if world else {}
        in_world = self._world_dir() is not None
        with self.session.lock:
            self._standing = {}
            for name in names:
                kind = robot_kind(self.session.model, name, told.get(name))
                if kind in SELF_BALANCING:
                    with contextlib.suppress(BehaviorError):
                        self._mobile_controller(name, kind, create=True)
                elif kind in STEPPING and in_world:
                    with contextlib.suppress(BehaviorError):
                        gait = self._mobile_controller(name, kind, create=True, walker="gait")
                        gait.level_standing = False  # it stands as on its servos until moved (see QuadrupedGait)
                        self._standing[name] = gait

    def _set_by_hand(self, actuator: int) -> None:
        """A leg joint set by hand (a viewer slider, set_controls) on a robot still on the gait start_balancing
        gave it: the gait lets go, so the joint moves as it did before that gait existed. Like the "Joints by
        hand" mode. A gait the robot was moved with keeps its legs, as it always did."""
        name = mujoco.mj_id2name(self.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) or ""
        robot = name.split("/", 1)[0]
        gait = self._standing.get(robot)
        if gait is not None and any(actuator in leg.actuators for leg in gait.legs):  # not an arm mounted on it
            self.let_go([robot])

    def hands_off(self) -> set[str]:
        """The robots no controller runs right now: legs set by hand, or the "Joints by hand" mode. A world edit
        or a lesson that brings them back stands them on a fresh gait; let_go takes it away again."""
        return set(self.robot_names()) - set(self.session.controllers)

    def let_go(self, robots: Iterable[str]) -> None:
        """Take away the gait start_balancing stood these robots on, so their joints stay where they were set:
        carried through a world edit, a leg posed by hand swung back home under a fresh gait (22.9 degrees)."""
        with self.session.lock:
            for robot in robots:
                gait = self._standing.pop(robot, None)
                if gait is not None and self.session.controllers.get(robot) is gait:
                    self.session.set_controller(robot, None)

    # ---- robots in the session -------------------------------------------------------

    def robots(self) -> list[dict[str, Any]]:
        m = self.session.model
        world = self._world()
        names = self._names(world)
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

    def _names(self, world: World | None) -> list[str]:
        return [r.name for r in world.robots] if world else sorted(self._prefixes())

    def robot_names(self) -> list[str]:
        """The robots by name: the open world's, else those a loaded model names with a prefix."""
        return self._names(self._world())

    def resolve(self, robot: str, arm: bool = False) -> str:
        """The robot `robot` means: spelled exactly, in other capitals or as "the dog"; "Rover/Arm" for an arm
        mounted on a robot. Raises BehaviorError listing the robots when there is no such one. `arm` is for the
        tools that work an arm: another part of a robot (a leg, "dog/trunk") is then refused as "not an arm"
        and the arms it has are named, where any other tool is told that it is a part and to use the robot."""
        # Scripts poll pose() hundreds of times a second. An exact name that the compiled model knows is
        # enough, so only a name that needs matching pays for reading and parsing world.yaml.
        if robot in self._prefixes():
            return robot
        return self._resolve(robot, self._world(), arm)

    def _resolve(self, robot: str, world: World | None, arm: bool = False) -> str:
        names = self._names(world)
        owner, slash, part = robot.partition("/")
        found = wording.match(owner, names)
        if found is None:
            raise BehaviorError(wording.unknown("robot", robot, names))
        if not slash:
            return found
        named = f"{found}/{part.strip()}"
        arms = self.arms(found)
        mounted = wording.match(named, arms) or self._arm_hanging_from(found, part.strip(), arms)
        if mounted is not None:
            return mounted
        if not arm:
            raise BehaviorError(f"{named!r} is a part of {found}, not a robot; use {found}")
        if arms:
            raise BehaviorError(f"{named!r} is not an arm of {found}; use {' or '.join(arms)}")
        raise BehaviorError(
            f"{named!r} is not an arm: {found} carries no arm (add_robot with build can mount one: see builder_guide)"
        )

    def _arm_hanging_from(self, robot: str, part: str, arms: list[str]) -> str | None:
        """The one arm that hangs from a part of `robot`: an arm built from parts answers to the part its gripper is
        mounted on ("centaur/left_hand"), and its shoulder ("centaur/left_arm") or forearm should find it too."""
        m = self.session.model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{robot}/{wording.squash(part)}")
        if body < 0:
            return None
        hanging = []
        for name in arms:
            b = int(m.site_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}/tcp")])
            while b > 0 and b != body:
                b = int(m.body_parentid[b])
            if b == body:
                hanging.append(name)
        return hanging[0] if len(hanging) == 1 else None

    def kind(self, robot: str) -> Kind:
        world = self._world()  # read once: it comes from the file
        robot = self._resolve(robot, world)
        if "/" in robot:  # "rover/arm": an arm mounted on a robot is an arm, whatever carries it
            return Kind.ARM
        if world is not None:
            entry = next(r for r in world.robots if r.name == robot)
            return robot_kind(self.session.model, robot, _told_kind(entry))
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
        robot = self.resolve(robot)
        if walker is not None:
            walker = wording.match(walker, GAIT_WALKERS) or walker
        kind = self.kind(robot)
        if "/" in robot:
            owner = robot.split("/", 1)[0]
            raise BehaviorError(
                f"{robot} is the arm mounted on {owner}: drive {owner} with move; the arm works with "
                "move_arm, gripper and pick_and_place"
            )
        if kind not in MOBILE and walker in (None, RHYTHM):
            if self.kept_walk(robot) is None:
                raise BehaviorError(self._cannot_move(robot, kind))
        elif kind not in MOBILE:
            raise BehaviorError(self._cannot_move(robot, kind))
        if walker == "gait" and kind == Kind.HEXAPOD:
            walker = None  # its built-in gait is the only one it has
        if walker not in (None, RHYTHM) and kind not in LEGGED:
            raise BehaviorError(
                f"{robot} is {_a(kind)}; walker {walker!r} is for quadrupeds and humanoids, so leave walker out"
            )
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, create=True, walker=walker)
            _check_standing(ctrl)
            ctrl.command(speed, turn)
            standing = self._standing.pop(robot, None)
            if standing is not None:  # moved: its gait is its own now, and stands as any gait it was moved with
                standing.stand_leveled()
        notes = _clamp_notes(ctrl, speed, turn)
        return {**ctrl.status(), "notes": notes} if notes else ctrl.status()

    def _cannot_move(self, robot: str, kind: Kind) -> str:
        if kind == Kind.ARM:
            return (
                f"{robot} is an arm, and only legged robots, wheeled robots and snakes move around; "
                f"{_how_to_move(kind)}"
            )
        text = (
            f"{robot} is {_a(kind)}, which has no built-in way to move: only legged robots, wheeled robots, "
            "snakes and robots that learned a walk of their own do"
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
            other = "; or use walker 'gait' (built in), 'rl' or 'rhythm'" if "no policy named" in str(exc) else ""
            raise BehaviorError(f"{exc}{other}") from exc
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
        robot = self.resolve(robot)
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
        robot = self.resolve(robot)
        kind = self.kind(robot)
        if kind != Kind.HUMANOID:
            raise BehaviorError(f"{robot} is {_a(kind)}, and only humanoids strike poses; {_how_to_move(kind)}")
        if name not in humanoid.POSES:
            raise BehaviorError(f"unknown pose {name!r}; choose one of: {', '.join(humanoid.POSES)}")
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, create=True, walker="gait")  # poses need the built-in one
            _check_standing(ctrl)
            ctrl.play(name)
        return ctrl

    def pose(self, robot: str) -> dict[str, Any]:
        """Where a robot is and which way it faces (degrees, counter-clockwise from +x)."""
        robot = self.resolve(robot)
        d = self.session.data
        root = self._root_body(robot)
        rot = d.xmat[root].reshape(3, 3)
        ctrl = self.session.controllers.get(robot)
        # A balancer knows best: a wheel-legged robot that folds its legs back tips its body far over, balanced.
        upright = not ctrl.fallen if isinstance(ctrl, WheelBalancer) else rot[2, 2] > 0.7
        return {
            "robot": robot,
            "position": [round(float(v), 3) for v in d.xpos[root]],
            "heading_deg": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 1),
            "upright": bool(upright),
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
        robot = self.resolve(robot, arm=True)  # unknown robots get the usual message
        if "/" in robot:
            return robot
        kind = self.kind(robot)
        arms = self.arms(robot)
        if not arms:
            raise BehaviorError(f"{robot} is {_a(kind)}, not an arm, and carries none; {_how_to_move(kind)}")
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
            raise BehaviorError(f"position must be [x, y, z] in meters (world frame), got {list(position)}")
        with self.session.lock:
            arm = self._arm(robot)
            if arm.reach_error(target) > REACH_TOL:
                raise BehaviorError(self._out_of_reach(arm, target))
            arm.move_to(target, gripper)
        return arm

    def _out_of_reach(self, arm: ArmMotion, target: np.ndarray) -> str:
        """Why a point cannot be reached, in numbers: how far it is, how far the arm goes, and whether the
        numbers look like centimetres (the usual mistake)."""
        point = [round(float(v), 3) for v in target]
        base, reach = arm.base_and_reach(self.session.data)
        distance = float(np.linalg.norm(target - base))
        head = f"{arm.robot} cannot reach {point}"
        if target[2] < 0:
            return f"{head}: z is below the floor; give z of 0 or more"
        if distance > reach * 1.05:
            return f"{head}: {wording.too_far(point, distance, reach)}"
        return f"{head} with its gripper pointing down, though it is within the arm's {reach:.2f} m reach; try nearer"

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
            obj = self._object_name(obj)
            pick = self._object_geometry(obj)
            place, place_top, region = self._place_target(target)
            try:
                arm.pick_and_place(pick["center"], pick["half_height"], place, place_top)
            except OutOfReach as exc:
                raise BehaviorError(self._pick_out_of_reach(arm, exc, pick["center"], place, target)) from exc
            except ValueError as exc:
                raise BehaviorError(str(exc)) from exc
        return arm, {"object": obj, "target": target, "region": region}

    def _pick_out_of_reach(
        self, arm: ArmMotion, exc: OutOfReach, object_at: np.ndarray, target_at: np.ndarray, target: str | list[float]
    ) -> str:
        """What move_arm says about a point, for the two a pick-and-place has: how far each one that is out of reach
        is, how far the arm goes, and the target in meters if its numbers look like centimetres."""
        base, reach = arm.base_and_reach(self.session.data)
        points = {"object": object_at, "place": target_at}
        far = {("target" if b.split()[-1] == "place" else "object"): points[b.split()[-1]] for b in exc.bad}
        away = " and ".join(f"{np.linalg.norm(p - base):.2f}" for p in far.values())
        one = len(far) == 1
        # Within its reach the gripper can still fail to point down there (too close to the base, too low or too
        # high): "move it nearer" would then send the agent the wrong way.
        beyond = all(np.linalg.norm(p - base) > 0.97 * reach for p in far.values())
        fix = (
            f"move {'it' if one else 'them'} nearer"
            if beyond
            else "that is within its reach, but not with the gripper pointing down there: move "
            f"{'it' if one else 'them'} a little further from the base or nearer the height of the arm's shoulder"
        )
        text = (
            f"{arm.robot} cannot reach the {' and the '.join(far)} with its gripper pointing down: "
            f"{'it is' if one else 'they are'} {away} m from the arm's base, which reaches about {reach:.2f} m; {fix}"
        )
        if "target" in far and (not isinstance(target, str) or NUMBER_LIST.fullmatch(target)):  # not a named object
            text += wording.in_meters(target_at[:2], float(np.linalg.norm(target_at - base)), reach)
        return text

    def placed(self, obj: str, region: dict[str, Any]) -> dict[str, Any]:
        """Check whether `obj` ended up inside the target region."""
        body = self._body(obj)
        pos = self.session.data.xpos[body]
        center = np.asarray(region["center"])
        half = np.asarray(region["half_xy"])
        inside = bool(np.all(np.abs(pos[:2] - center[:2]) <= half) and pos[2] >= region["min_z"])
        return {"success": inside, "object_position": [round(float(v), 3) for v in pos]}

    def _movable(self) -> list[str]:
        """Names of the loose objects: bodies with a free joint that are not part of a robot."""
        m = self.session.model
        found = []
        for b in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name and "/" not in name and m.body_dofnum[b]:
                found.append(name)
        return found

    def _object_name(self, name: str) -> str:
        """The object `name` means (other capitals too), or an error listing the objects."""
        world = self._world()
        names = [o.name for o in world.objects] if world else self._movable()
        found = wording.match(name, names)
        if found is None:
            raise BehaviorError(wording.unknown("object", name, names))
        return found

    def _body(self, name: str) -> int:
        b = mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise BehaviorError(wording.unknown("movable object", name, self._movable(), many="movable objects"))
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
        if isinstance(target, str) and NUMBER_LIST.fullmatch(target):  # "0.2, 0.2": a position, not an object
            target = [float(w) for w in re.split(r"[^\d.+-]+", target.strip(" []()")) if w]
        if isinstance(target, list):
            if len(target) not in (2, 3):
                raise BehaviorError(f"target must be an object name or a position [x, y] in meters, got {target}")
            p = np.array([target[0], target[1], target[2] if len(target) == 3 else 0.0])
            return p, p[2], {"center": p.tolist(), "half_xy": [0.04, 0.04], "min_z": p[2] - 0.01}
        world = self._world()
        names = [o.name for o in world.objects] if world else []
        found = wording.match(target, names)
        entry = self._object_entry(found) if found else None
        if entry is None:
            raise BehaviorError(wording.unknown("object", target, names) + " (or give a position [x, y])")
        target = found
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
