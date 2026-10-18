"""Start, stop and run behaviors on the robots of the live session."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.behaviors import Kind, air, airways, fetch, flight, humanoid, nav_map, quadruped, robot_kind
from robotforge.behaviors import navigate as navigation
from robotforge.behaviors.airframe import find_airframe
from robotforge.behaviors.arm import REACH_TOL, ArmMotion, OutOfReach, arm_bodies
from robotforge.behaviors.balance import WheelBalancer
from robotforge.behaviors.drive import DiffDrive
from robotforge.behaviors.rhythm_walk import RhythmWalker, SavedWalk, load_walk
from robotforge.behaviors.snake import SnakeGait
from robotforge.design import library, mounting
from robotforge.design.world import ObjectEntry, RobotEntry, World
from robotforge.sim.session import SimSession

MOBILE = (Kind.QUADRUPED, Kind.HEXAPOD, Kind.ROVER, Kind.HUMANOID, Kind.BALANCER, Kind.SNAKE, Kind.AIRCRAFT)
WALKERS = (
    quadruped.QuadrupedGait,
    DiffDrive,
    humanoid.HumanoidController,
    WheelBalancer,
    SnakeGait,
    flight.FlightController,
)
# They fall over without a controller; an aircraft also keeps its motors idle on the ground and hovers in the air.
SELF_BALANCING = (Kind.HUMANOID, Kind.BALANCER, Kind.AIRCRAFT)
TRIPS = (navigation.Navigator, airways.FlightTrip)  # a trip to a destination over the ground or through the air
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


# An arm riding on a robot plans its motion where things are when it starts, so its carrier must stand still: a Go2
# walking on at 0.3 m/s carried its PiPER past the cube and fell over as the arm reached out. Still is this slow
# (m/s, rad/s) for CARRIER_CALM s. Told to stop after walking 0.3 or 0.5 m/s, a Go2 with a PiPER is that still within
# 0.9 s; the tools wait up to CARRIER_WAIT s for it. A robot that balances sways for longer (a T1 with an SO-100 on its
# head, stopped after walking 0.1 to 0.12 m/s, stood still after 2.5 to 4.1 s): for those, CARRIER_WAIT_SWAYING s.
CARRIER_STILL, CARRIER_STILL_TURN = 0.03, 0.05
CARRIER_CALM = 0.2
CARRIER_WAIT = 3.0
CARRIER_WAIT_SWAYING = 6.0

GAIT_WALKERS = ("gait", "rl", RHYTHM)  # the walker names that are not a policy's own name
# A timed move that ends less than this fraction of its ideal reach from its start says why (shortfall). On flat
# ground the robots of the agent eval's sessions covered 72 to 108 percent of speed x seconds on straight moves:
# dog 94-102, Go2 86, hexapod 100, rovers 80-98, template humanoid 73-108 (0.1 m/s for 1.5 to 3 s), a two-wheeled
# balancer 72. A move is judged by how far it got along the way it was sent: for a move that turns, along its arc's
# chord (ideal_shift), or a full circle reads as stuck; a robot that went backwards gets the note too.
SHORTFALL_FRACTION = 0.25
SHORTFALL_MIN_ASKED = (
    0.1  # m: a move whose ideal reach is shorter (a nudge, a turn on the spot, a circle) is not judged
)
# s of every timed move left out of its ideal reach: getting going from a stand. Short moves on flat ground (0.3 to
# 3 s, 11 robots, two speeds each) needed up to 0.5 s of it to stay clear of the note (Spot at 0.3 m/s for 0.8 s
# covered 0.03 m); without it a snake told 0.2 m/s for 1 s (0.03 m) and a Go2 for 0.5 s were told "held back".
SHORTFALL_STARTUP = 1.0
NUMBER_LIST = re.compile(r"\s*[\[(]?\s*-?\d+(?:\.\d*)?(?:[\s,;]+-?\d+(?:\.\d*)?)+\s*[\])]?\s*")
TRIP = "__navigate__"  # session controller key of a robot's trip to a destination: TRIP + robot (see navigate)
MAX_DESTINATION = 100.0  # m; the largest scene is 20 m across (see playground.MAX_POSITION)
MIN_TRIP_SPEED = 0.02  # m/s; slower, a trip would not get anywhere (a humanoid walks at 0.09)


class BehaviorError(ValueError):
    """User-facing: says what to do instead."""


def _height_out_of_reach(point: np.ndarray, base: np.ndarray) -> str:
    """ "is too high at z 1.20 m for the arm, whose shoulder is at 0.47 m" for a height the gripper cannot get to."""
    where = "high" if point[2] > base[2] else "low"
    return (
        f"is too {where} at z {point[2]:.2f} m for the arm, whose shoulder is at {base[2]:.2f} m; "
        "give a point nearer that height"
    )


@dataclass(frozen=True)
class CarrierStop:
    """A robot carrying an arm, told to stop so the arm can work from it: `still` turns true once it has stood still
    long enough (see CARRIER_STILL); wait for that at most `seconds` of simulated time."""

    robot: str
    still: Callable[[], bool]
    seconds: float


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


DOWN_BAND_STEP = 0.02  # m between the distances from an arm's base tried for where its gripper points down
CLEARANCE = 0.01  # m kept between a thing the advice moves and the arm's base or the other thing of the pick
TURN_STEP = 15.0  # degrees between the directions around the base tried when nothing fits on the thing's own line


@dataclass(frozen=True)
class Room:
    """What a thing the advice moves must keep clear of, seen from above."""

    radius: float  # of the thing moved, all ways round (m)
    boxes: list[tuple[float, float, float, float]]  # the arm's base: (x min, y min, x max, y max)
    others: list[tuple[np.ndarray, float]]  # the other thing of the pick: (x y, radius)
    names: str  # what those are, for the message

    def clear(self, xy: np.ndarray) -> bool:
        for x0, y0, x1, y1 in self.boxes:
            if math.hypot(max(x0 - xy[0], 0.0, xy[0] - x1), max(y0 - xy[1], 0.0, xy[1] - y1)) < self.radius + CLEARANCE:
                return False
        return all(math.dist(xy, at) >= self.radius + r + CLEARANCE for at, r in self.others)


def footprint_radius(entry: ObjectEntry | None, fallback: float = 0.03) -> float:
    """Half the width of an object seen from above, all ways round (m): half the diagonal of a box, bin or table,
    half the diameter of a ball, can or rock (their size starts with it)."""
    if entry is None:
        return fallback
    return math.hypot(entry.size[0], entry.size[1]) / 2 if len(entry.size) >= 3 else entry.size[0] / 2


def _turned(direction: np.ndarray, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])


def _where_it_works(
    arm: ArmMotion, base: np.ndarray, reach: float, what: str, at: np.ndarray, heights: dict[str, float], room: Room
) -> str:
    """Advice for a pick-and-place point the gripper cannot point down at: the distances from the base, along the
    same line and at the same heights (`heights`: the key points that move with the thing and their z, see
    ArmMotion.key_point_error), where it can and the thing fits, and a position there; another side of the base when
    nothing fits on that line. "Within reach, so move it further" sent an agent the wrong way three times: the
    template arm gets over a bin only within about 0.36 m of its base, though it reaches 0.52 m. A bin sent onto the
    arm's stand was pushed off beside the cube and caught nothing."""
    toward = np.asarray(at[:2], float) - base[:2]
    now = float(np.linalg.norm(toward))
    if now > 1e-6:
        own = toward / now
    else:  # on the base itself: start on the side away from the other thing
        away = base[:2] - room.others[0][0] if room.others else np.zeros(2)
        own = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0.0])
    radii = np.arange(DOWN_BAND_STEP, reach + 1e-9, DOWN_BAND_STEP)

    def down(xy: np.ndarray) -> bool:
        # The points the gripper must point down at first, as checking a point where it may lean costs up to seven
        # times as much (see arm.TILTS): a refused pick on a Go2's PiPER took 31 s to advise, leaning first. Then the
        # highest first: it fails most often, and a failure there settles it.
        ordered = sorted(heights.items(), key=lambda point: (point[0].startswith("above"), -point[1]))
        return all(arm.key_point_error(name, np.array([*xy, z])) <= REACH_TOL for name, z in ordered)

    pointing = [float(r) for r in radii if down(base[:2] + r * own)]
    if not pointing:
        return (
            f"no point on that side of the base at {'that height' if len(heights) == 1 else 'those heights'} works "
            f"with the gripper pointing down: put the {what} lower, nearer the height of the arm's shoulder"
        )
    # The thing's own line first, then the other sides of the base nearest the other thing first: the shortest
    # carry (a mobile arm told to swing a cube over its rover to a bin behind it dropped it there). A side where
    # only the edge of the band fits is a last resort: a bin put there was reached, but the cube fell short of it.
    beside = room.others[0][0] - base[:2] if room.others else own
    beside = beside / np.linalg.norm(beside) if np.linalg.norm(beside) > 1e-6 else own
    lines = [own, *(_turned(beside, s * k * TURN_STEP) for k in range(int(180 / TURN_STEP) + 1) for s in (1, -1))]
    found: list[tuple[np.ndarray, list[float]]] = []
    for i, line in enumerate(lines):
        # Clear first (cheap); the arm points down at about the same distances all round its base.
        fits = [r for r in pointing if room.clear(base[:2] + r * line) and (i == 0 or down(base[:2] + r * line))]
        if fits:
            found.append((line, fits))
            if _inside(fits):
                break
    if not found:
        return (
            f"the gripper points down only {pointing[0]:.2f} to {pointing[-1]:.2f} m from the base, and nowhere "
            f"there does the {what} clear {room.names}: make room around the arm"
        )
    line, fits = next(((u, f) for u, f in found if _inside(f)), found[0])
    best = _nearest_inside(fits, now)
    xy = [round(float(v), 2) for v in base[:2] + best * line]
    span = f"the gripper points down only {pointing[0]:.2f} to {pointing[-1]:.2f} m from the base"
    if not np.allclose(line, own):
        return (
            f"along that line nothing works ({span}, where the {what} would hit {room.names}); move the {what} to "
            f"about {xy} ({best:.2f} m from the base, on another side)"
        )
    if best != _nearest_inside(pointing, now):  # only when it changes the advice: the message stays short
        clear = f"{fits[0]:.2f} to {fits[-1]:.2f} m" if fits[-1] > fits[0] else f"{fits[0]:.2f} m"
        span += f" ({clear} clear of {room.names})"
    way = "nearer" if best < now else "further from the base"
    return f"there {span}, so move the {what} {way}: to about {xy} ({best:.2f} m)"


def _inside(radii: list[float]) -> list[float]:
    """The distances from the base whose neighbours on both sides work too (the samples are DOWN_BAND_STEP apart):
    a spot on the edge of where it works is reached in the plan and missed in the carry."""
    steps = {round(r / DOWN_BAND_STEP) for r in radii}
    return [r for r in radii if {round(r / DOWN_BAND_STEP) - 1, round(r / DOWN_BAND_STEP) + 1} <= steps]


def _nearest_inside(radii: list[float], now: float) -> float:
    """Of the distances from the base where it works, the one nearest to `now` away from the edge if there is one."""
    return min(_inside(radii) or radii, key=lambda r: abs(r - now))


def ideal_reach(speed: float, turn: float, seconds: float) -> float:
    """How far from its start a robot ends that moves at `speed` m/s turning `turn` rad/s for `seconds`, if it does
    exactly that: speed x seconds in a straight line, the chord of the arc when it turns (0 after a full circle).
    Lesson 2's challenge, a rover driving a full circle back to its start, was told it was stuck."""
    angle = abs(turn) * seconds
    if angle < 1e-6:
        return abs(speed) * seconds
    return 2 * abs(speed) / abs(turn) * abs(math.sin(angle / 2))


def ideal_shift(speed: float, turn: float, seconds: float) -> np.ndarray:
    """Where a robot that moves at `speed` m/s turning `turn` rad/s (positive = left) for `seconds` ends if it does
    exactly that, in the frame it started in (front +x, left +y); its length is ideal_reach."""
    angle = turn * seconds
    if abs(angle) < 1e-6:
        return np.array([speed * seconds, 0.0])
    radius = speed / turn
    return np.array([radius * math.sin(angle), radius * (1.0 - math.cos(angle))])


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
        self._last_trips: dict[str, dict[str, Any]] = {}  # robot -> how its last trip ended, until it is moved again
        self._live_trips: dict[str, navigation.Navigator] = {}  # robot -> the trip it is on
        self._errands: dict[str, fetch.Errand] = {}  # robot -> the errand it is on (pick_and_place that goes over)
        session.reset_hooks.append(self._trips_after_reset)
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
                        self._mobile_controller(name, kind, world, create=True)
                    if kind == Kind.AIRCRAFT:
                        air.attach_air(self.session, name)  # drag and wind, whatever flies it (behaviors/air.py)
                elif kind in STEPPING and in_world:
                    with contextlib.suppress(BehaviorError):
                        gait = self._mobile_controller(name, kind, world, create=True, walker="gait")
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
            # A gripper on its own has a tool point like an arm, but nothing to move it (see _arm_name).
            info: dict[str, Any] = {"name": name, "kind": "gripper" if told.get(name) == "gripper" else kind.value}
            if name in models:
                info["model"] = models[name]
            arms = self.arms(name)
            if arms and arms != [name]:  # e.g. an arm riding on a rover: drive the robot, work the arm
                info["arms"] = arms
            entry = _entry(name, world)
            if kind in MOBILE and _learned_humanoid(entry):  # its policy's card says, without loading it
                learned = self._learned_walkers(world).get(name)
                if learned is not None:
                    loaded = self._loaded_speed(name, entry)  # slower when it carries much (see library.Payload)
                    top = learned["max_speed"] if loaded is None else min(learned["max_speed"], loaded)
                    info["max_speed"] = round(top, 3)
            elif kind in MOBILE:
                with contextlib.suppress(BehaviorError):
                    info["max_speed"] = round(self._mobile_controller(name, kind, world, create=False).max_speed, 3)
            if kind == Kind.AIRCRAFT:
                info["flight"] = self.flight_info(name)
            walk = self._kept_walk_or_none(name)
            if walk is not None:
                info["learned_walk"] = walk.describe()
                info.setdefault("max_speed", round(walk.speed, 3))  # robots no gait code knows drive with it
            elif kind not in MOBILE and self._can_learn_walk(name):
                info["walk_hint"] = f"no gait code knows this body, so move cannot walk it yet; {LEARN_A_WALK}"
            out.append(info)
        return out

    def flight_info(self, robot: str) -> dict[str, Any]:
        """An aircraft's rotors, how hard they lift, what flies it and what it is doing (or why it cannot fly).
        Flown by LQR or MPC it was reported landed at 0 m with the flight controller in charge, from a fresh one."""
        m, d = self.session.model, self.session.data
        ctrl = self.session.controllers.get(robot)
        if isinstance(ctrl, flight.FlightController):
            state = ctrl.status()
            return {
                **ctrl.frame.describe(),
                "controller": flight.CONTROLLER,
                "state": state["state"],
                "altitude_m": state["altitude_m"],
            }
        try:
            frame = find_airframe(m, robot, self.home_qpos())
        except ValueError as exc:
            return {"cannot_fly": str(exc)}
        state, altitude = flight.seen_from_outside(m, d, frame)
        out = {
            **frame.describe(),
            "controller": _flown_by(ctrl),
            "state": state,
            "altitude_m": round(max(altitude, 0.0), 2),
        }
        if ctrl is not None:
            out["note"] = f"stop({robot!r}) hands it back to its flight controller, where it is"
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
        mounted = wording.match(named, arms)
        if mounted is None and arm:
            mounted = self._arm_hanging_from(found, part.strip(), arms)
        if mounted is not None:
            return mounted
        if not arm:
            raise BehaviorError(f"{named!r} is a part of {found}, not a robot; use {found}")
        if arms:
            raise BehaviorError(f"{named!r} is not an arm of {found}; use {' or '.join(arms)}")
        entry = _entry(found, world)
        how = (
            "add_robot with build can mount one: see builder_guide"
            if entry is None or entry.build is not None
            else f'update_robot("{found}", mounts=[{{"library": "piper"}}]) puts one on it'
        )
        raise BehaviorError(f"{named!r} is not an arm: {found} carries no arm ({how})")

    def _arm_hanging_from(self, robot: str, part: str, arms: list[str]) -> str | None:
        """The arm a part of `robot` belongs to, for the tools that work an arm: an arm built from parts answers to
        the part its gripper is mounted on ("centaur/left_hand"), and its shoulder ("centaur/left_arm") or forearm
        should find it too. Only the arm's own moving parts count: the robot's body, its wheels or a waist two arms
        share are not an arm."""
        m = self.session.model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{robot}/{wording.squash(part)}")
        if body < 0:
            return None
        sites = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}/tcp") for name in arms}
        return next((name for name, site in sites.items() if body in arm_bodies(m, site)), None)

    def kind(self, robot: str) -> Kind:
        return self._kind(robot, self._world())  # read once: it comes from the file

    def _kind(self, robot: str, world: World | None) -> Kind:
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

    def move(
        self, robot: str, speed: float, turn: float, walker: str | None = None, side: float = 0.0, climb: float = 0.0
    ) -> dict[str, Any]:
        """walker (legged robots): "gait" = the built-in walking controller, "rl" = the learned policy
        installed for this robot, or a policy name; None keeps the robot's current walker. side (m/s to the left)
        and climb (m/s up) are for aircraft, which also fly sideways and up and down."""
        if not all(math.isfinite(v) for v in (speed, turn, side, climb)):
            raise BehaviorError(f"speed and turn must be finite numbers, got speed={speed} turn={turn}")
        robot = self.resolve(robot)
        if walker is not None:
            walker = wording.match(walker, GAIT_WALKERS) or walker
        world = self._world()  # read once: a keyboard drives the robot with a move every few frames
        kind = self._kind(robot, world)
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
        if (side or climb) and kind != Kind.AIRCRAFT:
            raise BehaviorError(f"{robot} is {_a(kind)}: only aircraft fly sideways, up and down")
        with self.session.lock:
            ctrl = self._mobile_controller(robot, kind, world, create=True, walker=walker)
            _check_standing(ctrl)
            self._end_trip(robot)  # driven by hand (a key, a move) it stops following a path
            self._last_trips.pop(robot, None)
            if isinstance(ctrl, flight.FlightController):
                ctrl.command(speed, turn, side=side, climb=climb)
            else:
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

    def _mobile_controller(self, robot: str, kind: Kind, world: World | None, create: bool, walker: str | None = None):
        """The controller that walks or drives a robot: the one it has, or a new one (see MOBILE). `world` is the
        open world as the caller read it."""
        existing = self.session.controllers.get(robot)
        if walker is None and _is_mover(existing):  # keep whatever walks it now, e.g. for WASD driving
            return existing
        if walker == RHYTHM or kind not in MOBILE:
            return self._rhythm_walker(robot, existing, create)
        entry = _entry(robot, world)
        if walker not in (None, "gait") and kind in LEGGED:
            return self._policy_walker(robot, walker, existing, create, entry)
        if isinstance(existing, WALKERS):
            return existing
        m = self.session.model
        try:
            if kind == Kind.HUMANOID and _learned_humanoid(entry):
                # A real humanoid with a learned walk (G1) walks with that policy only.
                return self._policy_walker(robot, "rl", existing, create, entry)
            if kind == Kind.HUMANOID and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/left_foot") >= 0:
                ctrl = humanoid.for_template(m, robot, self.home_qpos())
            elif kind == Kind.HUMANOID:  # a real one (OP3, T1): legs found in its kinematic tree
                ctrl = humanoid.for_robot(m, robot, self.home_qpos())
            elif kind in (Kind.QUADRUPED, Kind.HEXAPOD):
                ctrl = quadruped.for_robot(m, robot, self.home_qpos())
            elif kind == Kind.BALANCER:
                ctrl = WheelBalancer(m, robot, self.home_qpos())
            elif kind == Kind.SNAKE:
                ctrl = SnakeGait(m, robot, self.home_qpos())
            elif kind == Kind.AIRCRAFT:
                ctrl = flight.FlightController(m, robot, self.home_qpos())
            else:
                ctrl = DiffDrive(m, robot)
        except (ValueError, KeyError) as exc:
            if walker is None and self._kept_walk_or_none(robot) is not None:  # a body the gait code cannot walk
                return self._rhythm_walker(robot, existing, create)
            raise BehaviorError(str(exc)) from exc
        top = library.get(entry.library).max_speed if entry is not None and entry.library else None
        loaded = self._loaded_speed(robot, entry)
        if loaded is not None:  # it carries more than it walks with at full speed (see library.Payload)
            top = loaded if top is None else min(top, loaded)
        if top is not None and isinstance(ctrl, (quadruped.QuadrupedGait, humanoid.HumanoidController)):
            ctrl.limit_speed(top)
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def _loaded_speed(self, robot: str, entry: RobotEntry | None) -> float | None:
        """The top speed of a library robot carrying more than it walks with at full speed, or None."""
        if entry is None or not entry.mounts:
            return None
        return mounting.loaded_speed(entry.library, mounting.carried_mass(self.session.model, robot, entry.mounts))

    def _policy_walker(self, robot: str, walker: str, existing: Any, create: bool, entry: RobotEntry | None):
        """A learned walking policy (ONNX, optional extra `policy`) in place of the gait."""
        from robotforge.bridge.onnx_policy import PolicyUnavailable, PolicyWalker, walker_for

        if isinstance(existing, PolicyWalker) and walker in ("rl", existing.spec.name):
            return existing
        try:
            ctrl = walker_for(self.session.model, robot, entry.library if entry else None, walker)
        except (PolicyUnavailable, ValueError) as exc:
            other = "; or use walker 'gait' (built in), 'rl' or 'rhythm'" if "no policy named" in str(exc) else ""
            raise BehaviorError(f"{exc}{other}") from exc
        loaded = self._loaded_speed(robot, entry)
        if loaded is not None:
            ctrl.limit_speed(loaded)
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def learned_walkers(self) -> dict[str, dict[str, Any]]:
        """Legged robots that have a learned walking policy: {robot: {policy, max_speed, active}}."""
        return self._learned_walkers(self._world())

    def _learned_walkers(self, world: World | None) -> dict[str, dict[str, Any]]:
        from robotforge.bridge import policies  # light: cards only, no onnxruntime

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

    def stop(self, robot: str) -> dict[str, Any]:
        robot = self.resolve(robot)
        with self.session.lock:
            self._end_trip(robot)
            ctrl = self.session.controllers.get(robot)
            if isinstance(ctrl, flight.FlightController):
                # Hovers, it does not land: cutting the rotors drops it, and landing wants a clear spot under it,
                # which is the caller's to choose (move with to=[x, y, 0]).
                ctrl.hover()
                if not ctrl.airborne:
                    return {"robot": robot, "stopped": True, "state": ctrl.status()["state"]}
                return {"robot": robot, "stopped": True, "hovering": True, "note": flight.STOP_NOTE}
            if _is_mover(ctrl):
                ctrl.command(0.0, 0.0)  # keep the gait so a dog settles into a stand instead of collapsing
                if isinstance(ctrl, DiffDrive):
                    # Wheel speeds at rest now, not at the controller's next tick: an observe right after a stop
                    # showed the wheels still commanded, and agents zeroed them by hand (set_controls, simulate).
                    ctrl.update(self.session.model, self.session.data)
                return {"robot": robot, "stopped": True}
            if ctrl is not None and self.kind(robot) == Kind.AIRCRAFT:
                # LQR, MPC or a policy was flying it: the flight controller takes over where it is. Resting its
                # rotors like a stopped wheel's motor would drop it out of the sky.
                with contextlib.suppress(BehaviorError):
                    self._mobile_controller(robot, Kind.AIRCRAFT, self._world(), create=True)
                    return {
                        "robot": robot,
                        "stopped": True,
                        "note": f"the flight controller took over; {flight.STOP_NOTE}",
                    }
            if ctrl is not None:
                self.session.set_controller(robot, None)
                self._rest_speed_and_force_actuators(robot)
            return {"robot": robot, "stopped": ctrl is not None}

    def _rest_speed_and_force_actuators(self, robot: str) -> None:
        """Zero the speed and force commands a removed controller left behind; position servos keep holding.
        A rover stopped after run_controller(drive_to) kept its wheels at -2.1 and 5.0 rad/s and went on
        spinning: 0.11 m and a quarter turn in the next 2 s, until the agent zeroed them by hand."""
        m, d = self.session.model, self.session.data
        prefix = f"{robot}/"
        for a in range(m.nu):
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "").startswith(prefix):
                continue
            plain = m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE  # a motor: ctrl is force or torque
            speed = m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and m.actuator_biasprm[a][1] == 0
            if plain or speed:  # a speed servo pulls toward ctrl as a joint speed, with no position term
                d.ctrl[a] = 0.0

    def shortfall(self, robot: str, seconds: float, start: list[float], heading_deg: float = 0.0) -> str | None:
        """Why a timed move that is still running got far less far along its way than it should, or went the
        wrong way, or None. A rover too heavy for its motors stood still for 6 s and the agent was told only
        "distance_m 0.002"; a hexapod on weak servos went 0.21 m backwards when asked forwards and was told
        nothing, since it had moved. `start` and `heading_deg` are where it was and faced when the move began."""
        ctrl = self.session.controllers.get(robot)
        if isinstance(ctrl, flight.FlightController):  # nothing holds an aircraft back; a crash shows in its pose
            return None
        speed = float(getattr(ctrl, "command_speed", getattr(ctrl, "speed", 0.0)) or 0.0)
        turn = float(getattr(ctrl, "command_turn", getattr(ctrl, "turn", 0.0)) or 0.0)
        going = max(0.0, seconds - SHORTFALL_STARTUP)
        # The start-up second shortens the way, not the arc: a dog's full circle in 6.3 s, judged as 5.3 s of
        # turning, was a 0.28 m chord it had not made.
        shift = ideal_shift(speed, turn, seconds) * (going / seconds)
        asked = float(np.linalg.norm(shift))
        now = self.pose(robot)
        if asked < SHORTFALL_MIN_ASKED or not now["upright"]:
            return None
        h = math.radians(heading_deg)
        way = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]]) @ (shift / asked)
        along = float(np.dot(np.subtract(now["position"][:2], start[:2]), way))  # m along the way it was sent
        if along >= SHORTFALL_FRACTION * asked:
            return None
        motion = f"{speed:g} m/s for {seconds:g} s, less {SHORTFALL_STARTUP:g} s to get going"
        if abs(turn) * seconds >= 1e-6:
            motion = f"{speed:g} m/s turning {turn:g} rad/s for {seconds:g} s, less {SHORTFALL_STARTUP:g} s to get "
            motion += "going: the chord of that arc"
        if along < 0:
            way_back = "backwards" if abs(turn) * seconds < 1e-6 and speed > 0 else "the wrong way"
            text = f"{robot} went {-along:.2f} m {way_back} instead of the {asked:.2f} m asked ({motion})"
        else:
            text = f"{robot} got {along:.2f} m along its way, of the {asked:.2f} m asked ({motion})"
        if isinstance(ctrl, quadruped.QuadrupedGait):
            return (
                f"{text}: its legs may be too weak for its weight (they give way and the feet slide back; "
                "add_robot's notes say how hard its servos work standing, and stronger servos help), or something "
                "held it back (a wall or an object in the way, a slope); look with observe"
            )
        if not isinstance(ctrl, DiffDrive):
            return f"{text}: something held it back (a wall or an object in the way, a slope); look with observe"
        m, d = self.session.model, self.session.data
        if ctrl.spin(m, d) > SHORTFALL_FRACTION:
            return f"{text}: its wheels turned but it did not get anywhere (slipping or stuck); look with observe"
        push = ctrl.push_limit(m)
        weight = f"it weighs {float(m.body_subtreemass[self._root_body(robot)]):.1f} kg"
        strength = f" and its wheel motors push with at most {push:.1f} N in all" if push is not None else ""
        return (
            f"{text}: its wheels barely turned, {weight}{strength}. Give it stronger motors (list_templates lists "
            "them), smaller wheels or a lighter body (material pla)"
        )

    def settled(self, robot: str) -> Callable[[], bool] | None:
        """For robots that keep stepping for a moment after `stop`: a check for when they stand still."""
        ctrl = self.session.controllers.get(robot)
        if isinstance(ctrl, humanoid.HumanoidController):
            return lambda: ctrl.done
        return None

    # ---- going somewhere ------------------------------------------------------------------

    def navigate(
        self,
        robot: str,
        to: str | list[float],
        speed: float | None = None,
        walker: str | None = None,
        seconds: float | None = None,
        *,
        face: tuple[float, float] | None = None,
        carrying: str | None = None,
    ) -> navigation.Navigator:
        """Send a robot that moves to `to`: a point [x, y] (m) or the name of an object or another robot. It plans
        a path around what is in the way and follows it with its own walker (`walker` as for move) at up to
        `speed` m/s, for at most `seconds` of sim time (None: until it gets there). Raises navigation.NoWay, saying
        why, when there is no way; the Navigator returned says how the trip goes (`done`, `result()`).
        A trip to a point can end turned to `face` a point; `carrying` names a loose object the robot holds in its
        gripper, which goes with it: not in its way, and part of its footprint (see fetch.Errand)."""
        if speed is not None and not math.isfinite(speed):
            raise BehaviorError(f"speed must be a finite number, got {speed}")
        if speed is not None and abs(speed) < MIN_TRIP_SPEED:
            raise BehaviorError(
                f"speed {speed:g} would not get it anywhere: with to, speed is its top speed on the way (at least "
                f"{MIN_TRIP_SPEED:g} m/s); leave speed out for the robot's own pace"
            )
        robot = self.resolve(robot)
        named = to if isinstance(to, str) and not NUMBER_LIST.fullmatch(to) else None
        if named is not None and wording.match(named, [robot]) == robot:
            raise BehaviorError(f"{robot} cannot go to itself; give a point [x, y] or another object or robot")
        if "/" not in robot and self.kind(robot) == Kind.AIRCRAFT:
            if walker is not None:
                raise BehaviorError(f"{robot} is an aircraft; walker is for quadrupeds and humanoids, so leave it out")
            return self._fly(robot, to, speed, seconds)
        with self.session.lock:  # a destination it does not know is a mistake in the call: refused before it stops
            self._destination(robot, to, self.session.model, self.session.data)
        # The walker it goes with, standing. This also refuses what cannot move (an arm, "rover/arm") the way
        # move does, and ends any trip it was on.
        self.move(robot, 0.0, 0.0, walker)
        kind = self.kind(robot)
        with self.session.lock:
            model, version = self.session.model, self.session.version
            data = copy.copy(self.session.data)
            ctrl = self.session.controllers[robot]
        if isinstance(ctrl, RhythmWalker) and not {"left", "right"} <= set(ctrl.rhythms):
            raise BehaviorError(
                f"{robot} walks only straight ahead (its learned walk has no turns yet), so it cannot follow a path; "
                "teach it turns in the RL II lesson, or drive it with move and turn"
            )
        destination = self._destination(robot, to, model, data)
        if face is not None and destination.name is None:
            destination = dataclasses.replace(destination, face=(float(face[0]), float(face[1])))
        try:
            body = nav_map.measure_body(model, data, robot, kind, ctrl)
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        load = -1
        if carrying is not None:
            load = self._body(carrying)
            body = nav_map.with_load(model, data, robot, body, load)
        arm = self._carried_arm(robot, data)
        planner = navigation.Planner(model, robot, body, destination, arm, self._layers(version), carried=load)
        try:
            plan = planner.plan(data)
        except navigation.NoWay as exc:
            exc.destination = destination.describe()
            raise
        top = float(ctrl.max_speed)
        cruise = navigation.default_cruise(top) if speed is None else min(abs(speed), top)
        key = TRIP + robot

        def finished(trip: navigation.Navigator) -> None:  # runs inside a sim step, with the session lock held
            if self.session.controllers.get(key) is trip:
                self.session.set_controller(key, None)
                self._last_trips[robot] = trip.result()  # for `robots`, after a trip sent off with seconds=null
            if self._live_trips.get(robot) is trip:
                del self._live_trips[robot]

        trip = navigation.Navigator(
            robot,
            ctrl,
            planner,
            plan,
            cruise=cruise,
            turn_rate=navigation.turn_rate_for(kind.value, ctrl),
            turns_in_place=not isinstance(ctrl, SnakeGait),
            backs_up=not isinstance(ctrl, (SnakeGait, RhythmWalker)) and getattr(ctrl, "max_reverse", 1.0) > 0,
            start_time=float(data.time),
            deadline=float(data.time) + seconds if seconds is not None else None,
            still_mine=lambda: self.session.controllers.get(robot) is ctrl,
            on_finish=finished,
            others=[
                navigation.Other(name, nav_map.root_body(model, name), nav_map.footprint_radius(model, data, name))
                for name in self._names(self._world())
                if name != robot and "/" not in name
            ],
        )
        with self.session.lock:
            if self.session.version != version or self.session.controllers.get(robot) is not ctrl:
                raise BehaviorError(f"{robot} was changed while its path was planned; send it again")
            trip.start_time = float(self.session.data.time)
            if seconds is not None:
                trip.deadline = trip.start_time + seconds
            trip.note_others(self.session.data)
            self.session.set_controller(key, trip)
            self._live_trips[robot] = trip
        return trip

    def _fly(self, robot: str, to: str | list[Any], speed: float | None, seconds: float | None) -> airways.FlightTrip:
        """navigate for an aircraft: through the air, straight to the point or through the points, up over what is
        in the way; a last point on the ground (z 0) lands there."""
        with self.session.lock:
            fc = self._mobile_controller(robot, Kind.AIRCRAFT, self._world(), create=True)
            _check_standing(fc)
            model, data = self.session.model, self.session.data
            named = None
            if isinstance(to, str) and to.strip().startswith("[") and airways.parse_points(to) is not None:
                to = airways.parse_points(to)  # "[[0, 0, 2], [3, 0, 2]]" as text
            elif isinstance(to, str) and not NUMBER_LIST.fullmatch(to):
                named = self._destination(robot, to, model, data)
            elif isinstance(to, str):
                to = [float(w) for w in re.split(r"[^\d.+-]+", to.strip(" []()")) if w]
            try:
                grounds = airways.Grounds.of(self._world())  # which shapes are the ground, which are trees
                goals, land, described = airways.goals_for(model, data, fc, to, named, grounds)
                plan = airways.plan_flight(model, data, fc, goals, land, grounds)
            except flight.FlightError as exc:
                raise BehaviorError(str(exc)) from exc
            except navigation.NoWay as exc:
                exc.destination = described
                raise
            self._end_trip(robot)
            self._last_trips.pop(robot, None)
            key = TRIP + robot

            def finished(trip: airways.FlightTrip) -> None:  # inside a sim step, with the session lock held
                if self.session.controllers.get(key) is trip:
                    self.session.set_controller(key, None)
                    self._last_trips[robot] = trip.result()
                if self._live_trips.get(robot) is trip:
                    del self._live_trips[robot]

            cruise = flight.CRUISE if speed is None else min(abs(speed), fc.max_speed)
            trip = airways.FlightTrip.start(fc, plan, described, cruise, float(data.time), seconds, finished)
            self.session.set_controller(key, trip)
            self._live_trips[robot] = trip
        return trip

    def trip(self, robot: str) -> navigation.Navigator | airways.FlightTrip | None:
        """The robot's trip to a destination, while it is on one."""
        found = self.session.controllers.get(TRIP + robot)
        return found if isinstance(found, TRIPS) else None

    def _end_trip(self, robot: str) -> None:
        """Called with the session lock held: the robot is driven some other way now."""
        self._live_trips.pop(robot, None)
        trip = self.session.controllers.get(TRIP + robot)
        if isinstance(trip, airways.FlightTrip):
            self.session.set_controller(TRIP + robot, None)
            trip.interrupt("it was flown another way (keys, move or stop) on the way")
            self._last_trips[robot] = trip.result()
        elif isinstance(trip, navigation.Navigator):
            self.session.set_controller(TRIP + robot, None)
            if not trip.done:
                trip.state, trip.cause, trip.note = "stopped", "interrupted", "it was moved or stopped on the way"
                trip.end_time = float(self.session.data.time)
                self._last_trips[robot] = trip.result()

    def _trips_after_reset(self) -> None:
        """A reset or a change to the world clears every controller, trips among them, before this runs. A trip under
        way ends there, and says so (to move(to=...) waiting for it, and in `robots` after seconds=null): it used to
        vanish, and move waited out its whole time limit and returned time_s NaN. Trips that ended before are
        forgotten, as the robots were put back."""
        live, self._live_trips = self._live_trips, {}
        self._last_trips.clear()
        for robot, trip in live.items():
            if not trip.done:
                trip.interrupt("the world was reset or changed while it was on the way; send it again")
                self._last_trips[robot] = trip.result()
        errands, self._errands = self._errands, {}
        for errand in errands.values():
            errand.interrupt("the world was reset or changed while it was on its errand; ask again", errand.started)

    def _layers(self, version: int) -> dict:
        """The static map layers of this compiled world, shared by all trips in it (see navigation.Planner)."""
        if getattr(self, "_layer_version", None) != version:
            self._layer_version, self._layer_cache = version, {}
        return self._layer_cache

    def _destination(self, robot: str, to: str | list[float], model, data) -> navigation.Destination:
        if isinstance(to, str) and NUMBER_LIST.fullmatch(to):
            to = [float(w) for w in re.split(r"[^\d.+-]+", to.strip(" []()")) if w]
        if isinstance(to, (list, tuple)):
            if len(to) not in (2, 3) or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in to):
                raise BehaviorError(f"to must be a position [x, y] in meters or the name of an object, got {to}")
            if max(abs(to[0]), abs(to[1])) > MAX_DESTINATION:
                raise BehaviorError(f"to {list(to)} is more than {MAX_DESTINATION:g} m away: positions are in meters")
            return navigation.Destination((float(to[0]), float(to[1])))
        world = self._world()
        objects = [o.name for o in world.objects] if world else self._movable()
        robots = [r for r in self._names(world) if r != robot]
        terrain = [t.name for t in world.terrain] if world else []  # "go up the ramp", "to the stairs"
        found = wording.match(to, [*objects, *robots, *terrain])
        if found is None:
            known = [*objects, *robots, *terrain]
            many = "objects, robots and terrain patches"
            raise BehaviorError(
                wording.unknown("object, robot or terrain patch", to, known, many=many) + " (or give a point [x, y])"
            )
        if found in robots:
            return navigation.robot_destination(model, data, found)
        destination = navigation.find_destination(model, data, found)
        if destination is None:
            raise BehaviorError(f"{found} is not in the simulation; reload the world (load_world)")
        if found not in terrain:
            return destination
        destination = dataclasses.replace(destination, terrain=True)
        if next(t.type for t in world.terrain if t.name == found) in ("ramp", "stairs"):
            top = navigation.landing(model, data, destination.geoms)
            destination = dataclasses.replace(destination, point=top) if top else destination
        return destination

    def _carried_arm(self, robot: str, data: mujoco.MjData) -> navigation.Arm | None:
        """The arm a moving robot carries (the first, if it has several), for stopping within its reach."""
        arms = [a for a in self.arms(robot) if a != robot]
        if not arms:
            return None
        try:
            base, reach = ArmMotion(self.session.model, arms[0], self.home_qpos()).base_and_reach(data)
        except ValueError:
            return None
        root = nav_map.root_body(self.session.model, robot)
        rot = data.xmat[root].reshape(3, 3)
        heading = np.array([rot[0, 0], rot[1, 0]]) / max(math.hypot(rot[0, 0], rot[1, 0]), 1e-9)
        forward = float((base[:2] - data.xpos[root][:2]) @ heading)
        return navigation.Arm(forward, float(base[2]), reach)

    def strike_pose(self, robot: str, name: str) -> humanoid.HumanoidController:
        """Start a named humanoid pose (wave, squat, ...); a walking humanoid stops first."""
        robot = self.resolve(robot)
        world = self._world()
        kind = self._kind(robot, world)
        if kind != Kind.HUMANOID:
            raise BehaviorError(f"{robot} is {_a(kind)}, and only humanoids strike poses; {_how_to_move(kind)}")
        if name not in humanoid.POSES:
            raise BehaviorError(f"unknown pose {name!r}; choose one of: {', '.join(humanoid.POSES)}")
        with self.session.lock:
            # Poses need the built-in controller.
            ctrl = self._mobile_controller(robot, kind, world, create=True, walker="gait")
            if not isinstance(ctrl, humanoid.HumanoidController):
                raise BehaviorError(f"{robot} walks with a learned policy, which has no poses")
            body_moves = any(k.drop or k.lean for k in humanoid.POSES[name][1])
            if not ctrl.upper_adr and not body_moves:
                # A real humanoid's arms hold their home pose: no one has taught the controller which way they turn.
                usable = [p for p, (_, frames) in humanoid.POSES.items() if any(k.drop or k.lean for k in frames)]
                raise BehaviorError(
                    f"{name} moves only the arms and head, which on {robot} hold still; it can: {', '.join(usable)}"
                )
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
        world = self._world()
        entry = _entry(robot, world)
        if entry is not None and (entry.template == "gripper" or library.kind_of(entry.library) == "gripper"):
            what = library.get(entry.library).title if entry.library else "a gripper template"
            raise BehaviorError(
                f"{robot} is a gripper on its own ({what}), with no arm to move it; mount it on an arm's bare flange, "
                f'e.g. add_robot with library ur5e and mounts=[{{"library": "{entry.library or "robotiq_2f85"}"}}]'
            )
        kind = self._kind(robot, world)
        arms = self.arms(robot)
        if not arms and kind == Kind.ARM:  # an arm whose model marks no tool point (a library arm without a Tool)
            raise BehaviorError(
                f"{robot} is an arm, but its model marks no tool point for move_arm, gripper and pick_and_place to "
                "steer; set_controls moves its joints"
            )
        if not arms:
            raise BehaviorError(f"{robot} is {_a(kind)}, not an arm, and carries none; {_how_to_move(kind)}")
        if len(arms) > 1:
            raise BehaviorError(f"{robot} has {len(arms)} arms; name the one you mean: {', '.join(arms)}")
        return arms[0]

    def _arm(self, robot: str) -> ArmMotion:
        return self._arm_motion(self._arm_name(robot))

    def _arm_motion(self, name: str) -> ArmMotion:
        """The motion controller of the arm `name` (as _arm_name gives it), made the first time it is needed."""
        existing = self.session.controllers.get(name)
        if isinstance(existing, ArmMotion):
            existing.sync(self.session.data)
            return existing
        robot, _ = self._library_tool(name)
        try:
            gravity = robot is not None and robot.holds_against_gravity
            arm = ArmMotion(self.session.model, name, self.home_qpos(), gravity=gravity)
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        # Continue from where the arm (and whatever carries it) is now, not from the home pose.
        arm.sync(self.session.data)
        arm.q = self.session.data.qpos[arm.chain.qpos_adr].copy()
        self.session.set_controller(name, arm)
        return arm

    def stop_carrier(self, robot: str) -> CarrierStop | None:
        """For an arm riding on a robot that moves (see CARRIER_STILL): stop the robot, and say when it has stood still
        long enough for the arm to start. None when there is nothing to wait for."""
        with self.session.lock:
            arm = self._arm_name(robot)
            owner = _carrier(arm)
            if owner is None:
                return None
            self._refuse_stepping_carrier(owner, arm)
            if not self._carrier_moving(owner):
                return None
            self.stop(owner)
            swaying = isinstance(self.session.controllers.get(owner), (humanoid.HumanoidController, WheelBalancer))
        calm_since: list[float] = []

        def still() -> bool:
            now = float(self.session.data.time)
            if self._carrier_moving(owner):
                calm_since.clear()
            elif not calm_since:
                calm_since.append(now)
            return bool(calm_since) and now - calm_since[0] >= CARRIER_CALM

        return CarrierStop(owner, still, CARRIER_WAIT_SWAYING if swaying else CARRIER_WAIT)

    def _refuse_stepping_carrier(self, owner: str, arm: str) -> None:
        """A robot on a learned walk (G1) never stands still: its policy keeps stepping in place, at 0.2 to 0.35 m/s in
        every couple of seconds, so an arm on it could never start (see CARRIER_STILL)."""
        if _is_policy_walker(self.session.controllers.get(owner)):
            raise BehaviorError(
                f"{owner} keeps stepping in place on its learned walk and never stands still, so {arm} cannot work "
                "from it; an arm works on a robot that stands still: a robot dog (go2, spot, anymal_c), a rover or a T1"
            )

    def _carrier_motion(self, owner: str) -> tuple[float, float]:
        """How fast a robot's main body moves over the floor (m/s) and turns (rad/s)."""
        velocity = np.zeros(6)
        body = self._root_body(owner)
        mujoco.mj_objectVelocity(self.session.model, self.session.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return float(np.linalg.norm(velocity[3:5])), float(abs(velocity[2]))

    def _carrier_moving(self, owner: str) -> bool:
        speed, turn = self._carrier_motion(owner)
        told = _told_to_move(self.session.controllers.get(owner))
        return told or speed > CARRIER_STILL or turn > CARRIER_STILL_TURN

    def _check_carrier(self, arm: str) -> None:
        """Refuse to plan an arm's motion while the robot carrying it moves (see CARRIER_STILL)."""
        owner = _carrier(arm)
        if owner is None:
            return
        self._refuse_stepping_carrier(owner, arm)
        if _told_to_move(self.session.controllers.get(owner)):
            raise BehaviorError(
                f"{owner} is moving, and {arm} plans its motion where things are now; stop {owner} first, then "
                "use its arm"
            )
        speed, turn = self._carrier_motion(owner)
        if speed > CARRIER_STILL or turn > CARRIER_STILL_TURN:
            raise BehaviorError(
                f"{owner} is still coming to a stand ({speed:.2f} m/s); give it a second, then use {arm}"
            )

    def move_arm(self, robot: str, position: list[float], gripper: str | None = None) -> ArmMotion:
        target = np.asarray(position, float)
        if target.shape != (3,):
            raise BehaviorError(f"position must be [x, y, z] in meters (world frame), got {list(position)}")
        with self.session.lock:
            name = self._arm_name(robot)
            self._check_carrier(name)  # before its motion takes hold of the arm
            arm = self._arm_motion(name)
            if arm.reach_error(target) > REACH_TOL:
                raise BehaviorError(self._out_of_reach(arm, target))
            try:
                arm.move_to(target, gripper)
            except ValueError as exc:  # a riding arm with no way there that keeps off its carrier
                raise BehaviorError(str(exc)) from exc
        return arm

    def _out_of_reach(self, arm: ArmMotion, target: np.ndarray) -> str:
        """Why a point cannot be reached, in numbers: how far out it is, how far the arm reaches at that height
        with its gripper pointing down, and whether the numbers look like centimetres (the usual mistake)."""
        point = [round(float(v), 3) for v in target]
        head = f"{arm.robot} cannot reach {point}"
        if target[2] < 0:
            return f"{head}: z is below the floor; give z of 0 or more"
        at = arm.down_reach(self.session.data, target)
        if at.reach == 0.0:
            distance = float(np.linalg.norm(target - at.base))
            if wording.in_meters(point, distance, at.longest):  # far out of reach because it is in centimetres
                return f"{head}: {wording.too_far(point, distance, at.longest)}"
            return f"{head}: {_height_out_of_reach(target, at.base)}"
        if at.out > at.reach * 1.05:
            return f"{head}: {wording.too_far(point, at.out, at.reach)}"
        way = "further from its base" if at.out < 0.5 * at.reach else "nearer, or nearer the height of its shoulder"
        return (
            f"{head} with its gripper pointing down, though it is within the arm's {at.reach:.2f} m reach at that "
            f"height; try a point a little {way}"
        )

    def set_gripper(self, robot: str, state: str) -> ArmMotion:
        with self.session.lock:
            arm = self._arm(robot)
            if arm.gripper < 0:
                raise BehaviorError(f"{robot} has no gripper")
            arm.set_gripper(state)
        return arm

    def pick_and_place(
        self, robot: str, obj: str, target: str | list[float], go: bool = True
    ) -> tuple[ArmMotion | fetch.Errand, dict[str, Any]]:
        """Pick `obj` up and put it on or in `target`. An arm that rides on a robot that drives or walks, told to fetch
        something out of its reach from where the robot stands, has the robot go over (fetch.Errand): to the object,
        then with it to the place. `go=False` keeps the robot where it is and refuses instead, as before. Returns what
        does it (an ArmMotion or an Errand: wait until `done`) and {object, target, region}."""
        with self.session.lock:
            name = self._arm_name(robot)
            if mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}/gripper") < 0:
                raise BehaviorError(f"{name} has no gripper to pick with; {self._gripper_hint(name)}")
            model, tool = self._library_tool(name)
            if tool is not None and tool.cannot_pick:
                raise BehaviorError(
                    f"{name} ({model.title}) cannot pick and place yet: {tool.cannot_pick}; move_arm and gripper work"
                )
            self._check_carrier(name)  # before its motion takes hold of the arm
            arm = self._arm_motion(name)
            obj = self._object_name(obj)
            pick = self._object_geometry(obj)
            goes = go and self._goes(arm.robot)
            try:
                place, place_top, region = self._place_target(target)
            except BehaviorError:
                if not goes or self._place_kind(target) != "terrain":
                    raise
                place = None  # a terrain patch: only a robot that goes over puts things on one
            if place is not None and not (goes and fetch.holding(self.session.data, arm, self._body(obj))):
                try:
                    arm.pick_and_place(pick["center"], pick["half_height"], place, place_top, pick["faces"])
                except OutOfReach as exc:
                    # Out of reach only for a part of its own robot in the way of the carry (exc.over): that part
                    # goes with it wherever it drives, and the refusal names it.
                    if not goes or exc.over is not None:
                        raise BehaviorError(
                            self._pick_out_of_reach(arm, exc, pick["center"], place, target, obj)
                        ) from exc
                except ValueError as exc:
                    raise BehaviorError(str(exc)) from exc
                else:
                    return arm, {"object": obj, "target": target, "region": region}
            errand = self._fetch(arm, obj, target)
        return errand, {"object": obj, "target": target, "region": errand.place.region}

    def _goes(self, arm: str) -> bool:
        """Whether the arm rides on a robot that drives or walks itself somewhere."""
        if "/" not in arm:
            return False
        owner = arm.split("/", 1)[0]
        return self.kind(owner) in MOBILE

    def _place_kind(self, target: str | list[float]) -> str | None:
        """What a target is, for fetch.place_for: an object's shape, "terrain" for a terrain patch, None for a point."""
        if not isinstance(target, str) or NUMBER_LIST.fullmatch(target):
            return None
        world = self._world()
        if world is None:
            return None
        found = wording.match(target, [o.name for o in world.objects])
        if found is not None:
            return next(o.shape for o in world.objects if o.name == found)
        return "terrain" if wording.match(target, [t.name for t in world.terrain]) is not None else None

    def _fetch(self, arm: ArmMotion, obj: str, target: str | list[float]) -> fetch.Errand:
        """Start an errand for `arm`'s robot to bring `obj` to `target` (session lock held)."""
        robot = arm.robot.split("/", 1)[0]
        _check_standing(self._mobile_controller(robot, self.kind(robot), self._world(), create=True))
        kind = self._place_kind(target)
        named = target
        if isinstance(target, str) and NUMBER_LIST.fullmatch(target):
            named = [float(w) for w in re.split(r"[^\d.+-]+", target.strip(" []()")) if w]
        elif isinstance(target, str):
            world = self._world()
            names = [*(o.name for o in world.objects), *(t.name for t in world.terrain)] if world else []
            named = wording.match(target, names) or target
        try:
            errand = fetch.plan(self, arm, obj, named, kind)
        except fetch.FetchError as exc:
            raise BehaviorError(str(exc)) from exc
        self._end_trip(robot)
        self._last_trips.pop(robot, None)
        self.session.set_controller(fetch.KEY + robot, errand)
        self._errands[robot] = errand
        return errand

    def errand(self, robot: str) -> fetch.Errand | None:
        """The robot's errand (pick_and_place that goes over), while it is on one."""
        found = self.session.controllers.get(fetch.KEY + robot)
        return found if isinstance(found, fetch.Errand) else None

    def errand_ended(self, errand: fetch.Errand) -> None:
        """Called by an errand as it ends (inside a sim step, the session lock held)."""
        if self.session.controllers.get(fetch.KEY + errand.robot) is errand:
            self.session.set_controller(fetch.KEY + errand.robot, None)
        if self._errands.get(errand.robot) is errand:
            del self._errands[errand.robot]

    def root(self, robot: str) -> int:
        """The body a robot's position and heading are read from."""
        return self._root_body(robot)

    def floor_map(self, robot: str, data: mujoco.MjData, carrying: str | None = None) -> nav_map.NavMap:
        """Where the robot's middle may go on the floor, as a trip would plan it now, with the robot as posed in
        `data` (its arm tucked in, a load in its gripper): for where to stand to work (fetch.py)."""
        return self._fetch_planner(robot, data, carrying, None).map(data)

    def way_from(
        self, robot: str, data: mujoco.MjData, to: str | list[float], carrying: str | None = None
    ) -> tuple[navigation.Plan, nav_map.NavMap]:
        """The trip to `to` from where the robot stands in `data` (posed elsewhere, its arm tucked, a load in its
        gripper) and the map it is planned on; raises navigation.NoWay, saying why, if there is none: for a spot to
        pick something up from that the robot can then leave with it (fetch.py)."""
        planner = self._fetch_planner(robot, data, carrying, to)
        plan = planner.plan(data)
        return plan, planner.last_map

    def _fetch_planner(
        self, robot: str, data: mujoco.MjData, carrying: str | None, to: str | list[float] | None
    ) -> navigation.Planner:
        model = self.session.model
        kind = self.kind(robot)
        ctrl = self._mobile_controller(robot, kind, self._world(), create=False)
        try:
            body = nav_map.measure_body(model, data, robot, kind, ctrl)
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        load = -1
        if carrying is not None:
            load = self._body(carrying)
            body = nav_map.with_load(model, data, robot, body, load)
        if to is None:
            x, y = data.xpos[self._root_body(robot)][:2]
            destination = navigation.Destination((float(x), float(y)))
        else:
            destination = self._destination(robot, to, model, data)
        layers = self._layers(self.session.version)
        return navigation.Planner(model, robot, body, destination, None, layers, carried=load)

    def _library_tool(self, arm: str) -> tuple[library.LibraryRobot | None, library.Tool | None]:
        """The library robot an arm is ("z1", "go2/arm" carrying a PiPER, "aloha/left") and its Tool, if it is one."""
        owner, _, rest = arm.partition("/")
        entry = _entry(owner, self._world())
        if entry is None:
            return None, None
        first = rest.split("/", 1)[0]
        mount = next((m for m in entry.mounts if mounting.default_name(m) == first), None) if rest else None
        source, rest = (mount.library, rest[len(first) + 1 :]) if mount is not None else (entry.library, rest)
        if source is None:
            return None, None
        robot = library.get(source)
        return robot, next((t for t in robot.tools if t.arm.rstrip("/") == rest), None)

    def _gripper_hint(self, arm: str) -> str:
        """How to give an arm without a gripper one: on its flange, as the next of the robot's mounts."""
        owner = arm.split("/", 1)[0]
        entry = _entry(owner, self._world())
        mounts = [m.model_dump(exclude_defaults=True) for m in entry.mounts] if entry is not None else []
        listed = json.dumps([*mounts, {"library": "robotiq_2f85"}])
        return f'mount one on its flange: update_robot("{owner}", mounts={listed})'

    def _pick_out_of_reach(
        self,
        arm: ArmMotion,
        exc: OutOfReach,
        object_at: np.ndarray,
        target_at: np.ndarray,
        target: str | list[float],
        obj: str,
    ) -> str:
        """What move_arm says about a point, for the two a pick-and-place has: how far each one that is out of reach
        is, how far the arm goes, where along the same line from the base it would work, and the target in meters
        if its numbers look like centimetres."""
        base, reach = arm.base_and_reach(self.session.data)
        points = {"object": object_at, "place": target_at}
        far = {("target" if b.split()[-1] == "place" else "object"): points[b.split()[-1]] for b in exc.bad}
        if exc.over is not None:  # it reaches both, but not high enough above them to carry over its carrier
            body, top = exc.over
            owner, _, part = body.partition("/")
            part = part.split("/", 1)[0] or owner
            return (
                f"{arm.robot} carries things from the object to the target over {owner}'s {part}, whose top is "
                f"{top:.2f} m high, and cannot lift them that high above the {' and the '.join(far)}; put the object "
                f"and the target where the carry between them does not pass over {part} (on the same side of {owner}, "
                "say)"
            )
        away = " and ".join(f"{np.linalg.norm(p - base):.2f}" for p in far.values())
        one = len(far) == 1
        named = isinstance(target, str) and not NUMBER_LIST.fullmatch(target)
        names = {"object": obj, "target": target if named else "target"}
        size = footprint_radius(self._object_entry(obj))
        # How wide each thing is from above: a named target is itself (a bin); at a position, the object lands there.
        widths = {"object": size, "target": footprint_radius(self._object_entry(target)) if named else size}
        boxes = arm.footprint(self.session.model, self.session.data)
        fixes = []
        for what, at in far.items():
            # Both heights the tool works at over it (on it and above it) move with the thing.
            suffix = "place" if what == "target" else "object"
            heights = {name: float(p[2]) for name, p in exc.points.items() if name.split()[-1] == suffix}
            heights = heights or {suffix: float(at[2])}
            # Clear of the arm's base and, when both are things on the floor, of the other one.
            other = "target" if what == "object" else "object"
            keep = [({"object": object_at, "target": target_at}[other][:2], widths[other])] if named else []
            room = Room(widths[what], boxes, keep, "the arm's base" + (f" and the {names[other]}" if keep else ""))
            # Out to where the tool gets at all at those heights (see ArmMotion.farthest), not the arm's length.
            data, xy = self.session.data, np.asarray(at[:2], float)
            out = min(arm.farthest(data, np.array([*xy, z]), name.startswith("above")) for name, z in heights.items())
            fixes.append(_where_it_works(arm, base, min(reach, out + DOWN_BAND_STEP), names[what], at, heights, room))
        text = (
            f"{arm.robot} cannot reach the {' and the '.join(far)} with its gripper pointing down: "
            f"{'it is' if one else 'they are'} {away} m from the arm's base, which reaches about {reach:.2f} m; "
            + "; ".join(fixes)
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
        faces = None  # a box's sides, for jaws closing square to them (see arm.SQUARE_TOL)
        if entry is not None and entry.shape == "box":
            sides = self.session.data.xmat[body].reshape(3, 3)[:, :2].T.copy()
            sides[:, 2] = 0.0
            if np.all(np.linalg.norm(sides, axis=1) > 0.5):  # standing upright, not on an edge
                faces = sides / np.linalg.norm(sides, axis=1, keepdims=True)
        return {"center": center, "half_height": half, "faces": faces}

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
        done = [
            {"behavior": "navigate", "navigating": robot, "state": "done", "last_trip": result}
            for robot, result in self._last_trips.items()
        ]
        return [c.status() for c in self.session.controllers.values()] + done


def _told_kind(entry: Any) -> str | None:
    """The kind a world entry declares: its template, or for a library robot what the catalog says
    (G1's joints are not named like our template's, so its shape alone does not say humanoid)."""
    return entry.template or (library.get(entry.library).kind if entry.library else None)


def _entry(robot: str, world: World | None) -> RobotEntry | None:
    return next((r for r in world.robots if r.name == robot), None) if world else None


def _learned_humanoid(entry: RobotEntry | None) -> bool:
    """A humanoid from the library with a learned walking policy (like G1); others (OP3, T1, our template's) stand and
    walk on the built-in balance gait."""
    from robotforge.bridge import policies  # light: cards only, no onnxruntime

    if entry is None or entry.library is None or library.get(entry.library).kind != "humanoid":
        return False
    return policies.for_robot(entry.library) is not None


def _carrier(arm: str) -> str | None:
    """The robot an arm rides on ("go2" for "go2/arm"), or None for an arm standing on its own."""
    return arm.split("/", 1)[0] if "/" in arm else None


def _told_to_move(ctrl: Any) -> bool:
    """Whether a robot's walking or driving controller has been told to move (gaits keep the command they were given,
    wheels their speed)."""
    if isinstance(ctrl, RhythmWalker):
        return ctrl.walking
    if not _is_mover(ctrl):
        return False
    return bool(
        getattr(ctrl, "command_speed", getattr(ctrl, "speed", 0.0))
        or getattr(ctrl, "command_turn", getattr(ctrl, "turn", 0.0))
    )


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


def _flown_by(ctrl: Any) -> str:
    """The name of what flies an aircraft other than its flight controller: lqr, mpc, a policy, a script."""
    if ctrl is None:
        return "nothing yet (move, the keys or stop start its flight controller)"
    status = ctrl.status() if callable(getattr(ctrl, "status", None)) else {}
    name = status.get("controller") if isinstance(status, dict) else None
    return name if isinstance(name, str) else type(ctrl).__name__
