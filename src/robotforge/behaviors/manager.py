"""Start, stop and run behaviors on the robots of the live session."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import inspect
import json
import math
import re
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robotforge import wording
from robotforge.behaviors import (
    Kind,
    air,
    airways,
    catch_stand,
    fetch,
    flight,
    grasping,
    hints,
    humanoid,
    marine,
    nav_map,
    quadruped,
    relay,
    robot_kind,
    visit,
    water_nav,
    whole_body,
)
from robotforge.behaviors import catch as catching
from robotforge.behaviors import navigate as navigation
from robotforge.behaviors.airframe import find_airframe
from robotforge.behaviors.arm import REACH_TOL, ArmMotion, OutOfReach, arm_bodies
from robotforge.behaviors.balance import WheelBalancer
from robotforge.behaviors.drive import DiffDrive
from robotforge.behaviors.reach_in import LOADED_TILT, Hand
from robotforge.behaviors.rhythm_walk import RhythmWalker, SavedWalk, load_walk
from robotforge.behaviors.snake import SnakeGait
from robotforge.design import end_effectors, library, mechanisms, mounting
from robotforge.design.world import ObjectEntry, RobotEntry, World
from robotforge.sim.geometry import collides
from robotforge.sim.session import SimSession

MOBILE = (
    Kind.QUADRUPED,
    Kind.HEXAPOD,
    Kind.ROVER,
    Kind.HUMANOID,
    Kind.BALANCER,
    Kind.SNAKE,
    Kind.BOAT,
    Kind.ROV,
    Kind.AIRCRAFT,
)
WALKERS = (
    quadruped.QuadrupedGait,
    DiffDrive,
    humanoid.HumanoidController,
    WheelBalancer,
    SnakeGait,
    marine.BoatPilot,
    marine.RovPilot,
    flight.FlightController,
)
# They fall over without a controller; an aircraft also keeps its motors idle on the ground and hovers in the air.
SELF_BALANCING = (Kind.HUMANOID, Kind.BALANCER, Kind.AIRCRAFT)
# A trip to a destination over the ground, under water or through the air.
TRIPS = (navigation.Navigator, marine.DiveTrip, airways.FlightTrip)
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
# A robot sent somewhere folds an idle arm over itself first when that makes it this much narrower or less far round
# as it turns (m of footprint radius or reach: navigation plans for the folded robot), unless something lies
# within HOLDING_NEAR (m) of its tool point.
FOLD_GAIN = 0.03
HOLDING_NEAR = 0.08
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


SPREAD_MAX = 0.04  # m: the most a thing's own width keeps its neighbours in a tray away
HOLD_LIFT = 0.04  # m a held thing must be above where it lay for the hold to count


def _firm(built: Any) -> bool:
    """Whether a generated thing is held with the firm hold: a heavy one, or a light one that slips out of a plain
    squeeze (a pen, thin and round)."""
    return bool(built.mass >= HEAVY or built.facts.get("firm"))


def _why_not_placed(
    obj: str, pos: np.ndarray, center: np.ndarray, half: np.ndarray, min_z: float, start: np.ndarray | None
) -> str:
    """Why `obj` at `pos` is not in the region: never taken, dropped on the way, or put down beside or below it."""
    if start is not None and float(np.linalg.norm(pos - start)) < 0.08:
        return (
            f"{obj} was never lifted: it is within {100 * float(np.linalg.norm(pos - start)):.0f} cm of where it "
            "started (the arm did not get hold of it, or only knocked it); look at the arm's pose and what stands near"
        )
    miss = float(np.hypot(*np.maximum(np.abs(pos[:2] - center[:2]) - half, 0.0)))
    if pos[2] < min_z - 0.15 or (pos[2] < 0.1 and center[2] > 0.2):
        return (
            f"{obj} was dropped or fell: it ended at height {pos[2] * 100:.0f} cm, {miss * 100:.0f} cm sideways from "
            "the target (slipped out of the grip, knocked by something on the way, or rolled off after it was let go)"
        )
    if miss > 0.0:
        return f"{obj} was put down {miss * 100:.0f} cm beside the target"
    low = max(min_z - float(pos[2]), 0.0) * 100
    return f"{obj} is {low:.0f} cm below the target's top: it sank in or is not standing on it"


class BehaviorError(ValueError):
    """User-facing: says what to do instead."""


# ---- the commands a robot is given, told to whoever records them (a demonstration, sim/demonstration.py) ----


def _commanded(op: str, tell: Callable[[BehaviorManager, dict[str, Any]], tuple[str, dict[str, Any]]]):
    """Make a public command report itself to `command_hooks` once it has worked: (robot, {"op", ...its numbers}). Only
    the outermost call reports: a trip that stops the robot first, or a controller (an errand) asking for a move
    while it runs, is part of that command and not another one."""

    def wrap(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        def command(self: BehaviorManager, *args: Any, **kwargs: Any):
            if not self.command_hooks:
                return method(self, *args, **kwargs)
            outer = not getattr(self._issuing, "depth", 0) and not self.session.in_controller
            self._issuing.depth = getattr(self._issuing, "depth", 0) + 1
            try:
                result = method(self, *args, **kwargs)
            finally:
                self._issuing.depth -= 1
            if outer:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                robot, event = tell(self, dict(bound.arguments))
                for hook in tuple(self.command_hooks):
                    hook(robot, {"op": op, **event})
            return result

        return command

    return wrap


def _floats(values: Any) -> list[float]:
    return [round(float(v), 5) for v in values]


def _tell_move(m: BehaviorManager, a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    event: dict[str, Any] = {"speed": float(a["speed"]), "turn": float(a["turn"])}
    for key in ("side", "climb"):
        if a[key]:
            event[key] = float(a[key])
    return m.resolve(a["robot"]), event


def _tell_go(m: BehaviorManager, a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    to = a["to"] if isinstance(a["to"], str) else _floats(a["to"])
    return m.resolve(a["robot"]), {"to": to, **({"speed": a["speed"]} if a["speed"] else {})}


def _tell_arm_target(m: BehaviorManager, a: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return m._arm_name(a["robot"]), {"position": _floats(a["position"]), "gripper": a["gripper"]}


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


GRASP_OPENING = 0.07  # m the jaws open when the arm's gripper does not tell
DOWN_BAND_STEP = 0.02  # m between the distances from an arm's base tried for where its gripper points down
CLEARANCE = 0.01  # m kept between a thing the advice moves and the arm's base or the other thing of the pick
TURN_STEP = 15.0  # degrees between the directions around the base tried when nothing fits on the thing's own line
FLAT = 0.02  # m: a thing thinner than this lies too flat for fingers to get round it (a tile, a plate)
PLACE_TOL = 0.015  # m sideways that still counts as on the target (see placed)
CLUTTER_MARGIN = 0.05  # m more over what the load is carried over: a small arm sags 4 to 9 cm under it on the way
CORRIDOR = 0.12  # m either side of the way between the pick and the place that what is tall counts
POUR_FORCE = 40.0  # N a cup is squeezed with while it is tipped
LINK_CLEAR = 0.02  # m a pour wants between the top of the finger links and the rim it lets balls run over
HEAVY = 0.08  # kg: a generated thing this heavy is held firmly (hints.FirmHold), as is one whose facts say `firm`
OPEN_ENOUGH = 0.05  # m a drawer is pulled out before pick_and_place puts things in it


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
        # What a demonstration listens to: (robot, {"op": "move", ...}) after each command given from outside
        self.command_hooks: list[Callable[[str, dict[str, Any]], None]] = []
        self._issuing = threading.local()
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
                        self._mobile_controller(name, kind, world, create=True, stand=True)
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
            if isinstance(self.session.controllers.get(name), whole_body.WholeBodyBalancer):
                info["balance"] = "whole-body balance: stands on its feet, holds a load, takes a push"
            walk = self._kept_walk_or_none(name)
            if walk is None and "balance" in info and "max_speed" not in info:
                info["walk_hint"] = "no walk has been shown to carry it, so move cannot walk it; it stands on its own"
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

    @_commanded("move", _tell_move)
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
        return self._cannot_learn_walk(robot) is None

    def _cannot_learn_walk(self, robot: str) -> str | None:
        """Why `robot` cannot learn a walk of its own in RL II, None if it can."""
        from robotforge.control.cpg import Body, CPGError

        try:
            Body.find(self.session.model, robot)
        except CPGError as exc:
            return str(exc)
        return None

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
            why = self._cannot_learn_walk(robot)
            how = LEARN_A_WALK if why is None else f"and it cannot learn one in the RL II lesson: {why}"
            raise BehaviorError(f"{robot} has not learned a walk of its own; {how}")
        if isinstance(existing, RhythmWalker) and existing.walk == walk:
            return existing
        try:
            ctrl = RhythmWalker(self.session.model, robot, walk, self._walk_home(robot, walk))
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def _walk_home(self, robot: str, walk: SavedWalk) -> np.ndarray:
        """The pose a kept walk swings around when its file does not say (one kept before files said). An MJCF
        robot's home was then the file's zero pose, its keyframe dropped as the world was built: played around the
        crouched keyframe instead, a critter's walk went 0.88 m in 4 s instead of 1.25."""
        world = self._world()
        entry = next((r for r in world.robots if r.name == robot), None) if world else None
        if walk.home is None and entry is not None and entry.mjcf_file:
            return self.session.model.qpos0.copy()
        return self.home_qpos()

    def _mobile_controller(
        self, robot: str, kind: Kind, world: World | None, create: bool, walker: str | None = None, stand: bool = False
    ):
        """The controller that walks or drives a robot: the one it has, or a new one (see MOBILE). `world` is the
        open world as the caller read it. `stand`: it is only to keep the robot up (a humanoid nobody tuned a gait
        to stands by whole-body balance, which holds a load and takes a push; the first move gives it a gait if one
        fits its legs)."""
        existing = self.session.controllers.get(robot)
        if walker is None and _is_mover(existing):  # keep whatever walks it now, e.g. for WASD driving
            return existing
        if walker == RHYTHM or kind not in MOBILE:
            return self._rhythm_walker(robot, existing, create)
        entry = _entry(robot, world)
        if stand and kind == Kind.HUMANOID and _stands_by_balance(entry):
            return self._stander(robot, existing, create)
        if walker not in (None, "gait") and kind in LEGGED:
            return self._policy_walker(robot, walker, existing, create, entry)
        if isinstance(existing, WALKERS):
            return existing
        if kind == Kind.HUMANOID and not _tuned_humanoid(entry):
            # A gait found from the legs of a file, catalogue or built humanoid is not known to walk it: every one
            # tried fell within seconds (docs/HUMANOID.md), so it stays on its feet by balance and move says so.
            if walker is None and self._kept_walk_or_none(robot) is not None:
                return self._rhythm_walker(robot, existing, create)
            raise BehaviorError(
                f"{robot} stands on its feet by whole-body balance, which holds a load and takes a push, but no "
                "walk has been shown to carry it: move, pose and go_to cannot walk it"
            )
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
            elif kind in (Kind.BOAT, Kind.ROV):  # thrusters; the pilot measures its own drag in the water now
                ctrl = marine.pilot_for(kind.value, m, robot, self.session.water, self.session.data)
            elif kind == Kind.AIRCRAFT:
                ctrl = flight.FlightController(m, robot, self.home_qpos())
            else:
                ctrl = DiffDrive(m, robot)
        except (ValueError, KeyError) as exc:
            if walker is None and self._kept_walk_or_none(robot) is not None:  # a body the gait code cannot walk
                return self._rhythm_walker(robot, existing, create)
            if isinstance(exc, KeyError):  # a leg joint no motor drives: a linkage, as in Oli or Cassie
                raise BehaviorError(
                    f"no gait fits the body of {robot}: a joint of its legs has no motor of its own (a linkage?), "
                    f"which the walkers need ({exc})"
                ) from exc
            raise BehaviorError(str(exc)) from exc
        top = library.get(entry.library).max_speed if entry is not None and entry.library else None
        loaded = self._loaded_speed(robot, entry)
        if loaded is not None:  # it carries more than it walks with at full speed (see library.Payload)
            top = loaded if top is None else min(top, loaded)
        if top is not None and isinstance(ctrl, (quadruped.QuadrupedGait, humanoid.HumanoidController)):
            ctrl.limit_speed(top)
        slope = library.get(entry.library).max_slope if entry is not None and entry.library else None
        if slope is not None and isinstance(ctrl, quadruped.QuadrupedGait):
            ctrl.limit_slope(slope)
        if create:
            self.session.set_controller(robot, ctrl)
        return ctrl

    def _stander(self, robot: str, existing: Any, create: bool) -> whole_body.WholeBodyBalancer:
        """Whole-body balance for a humanoid with no gait of its own: the one it has, or a new one."""
        if isinstance(existing, whole_body.WholeBodyBalancer):
            return existing
        try:
            ctrl = whole_body.WholeBodyBalancer(self.session.model, robot, self.home_qpos())
        except whole_body.BalanceError as exc:
            raise BehaviorError(str(exc)) from exc
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

    @_commanded("stop", lambda m, a: (m.resolve(a["robot"]), {}))
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

    @_commanded("go", _tell_go)
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
        own_map: Any = None,
        unknown_blocks: bool = True,
    ) -> navigation.Navigator:
        """Send a robot that moves to `to`: a point [x, y] (m) or the name of an object or another robot. It plans
        a path around what is in the way and follows it with its own walker (`walker` as for move) at up to
        `speed` m/s, for at most `seconds` of sim time (None: until it gets there). Raises navigation.NoWay, saying
        why, when there is no way; the Navigator returned says how the trip goes (`done`, `result()`).
        A trip to a point can end turned to `face` a point; `carrying` names a loose object the robot holds in its
        gripper, which goes with it: not in its way, and part of its footprint (see fetch.Errand).
        `own_map` (a perception Mapper) plans on the robot's own occupancy grid and follows its own pose estimate
        instead of the scene's truth (map_nav.py); unknown cells block unless `unknown_blocks` is False."""
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
            folding = self._fold_for_trip(robot, kind, ctrl, data) if carrying is None else None
        if folding is not None:
            data = folding[0]  # planned for as it will be, its arm folded
        if isinstance(ctrl, RhythmWalker) and not {"left", "right"} <= set(ctrl.rhythms):
            raise BehaviorError(
                f"{robot} walks only straight ahead (its learned walk has no turns yet), so it cannot follow a path; "
                "teach it turns in the RL II lesson, or drive it with move and turn"
            )
        if isinstance(ctrl, marine.RovPilot):  # under water it goes straight to a point in 3D
            return self._dive(robot, to, ctrl, speed, seconds, model, data)
        destination = self._destination(robot, to, model, data)
        if face is not None and destination.name is None:
            destination = dataclasses.replace(destination, face=(float(face[0]), float(face[1])))
        if isinstance(ctrl, marine.BoatPilot):  # on the water: a map of where it floats clear of the shore
            try:
                planner = water_nav.planner(model, data, robot, ctrl, destination, self._layers(version))
            except ValueError as exc:
                raise BehaviorError(str(exc)) from exc
        else:
            try:
                body = nav_map.measure_body(model, data, robot, kind, ctrl)
            except ValueError as exc:
                raise BehaviorError(str(exc)) from exc
            load = -1
            if carrying is not None:
                load = self._body(carrying)
                body = nav_map.with_load(model, data, robot, body, load)
            arm = self._carried_arm(robot, data)
            if own_map is not None:
                from robotforge.behaviors import map_nav  # here: only a trip on its own map needs the perception code

                planner = map_nav.MapPlanner(
                    model, robot, body, destination, own_map, arm, carried=load, unknown_blocks=unknown_blocks
                )
            else:
                planner = navigation.Planner(model, robot, body, destination, arm, self._layers(version), carried=load)
        try:
            plan = planner.plan(data)
        except navigation.NoWay as exc:
            exc.destination = destination.describe()
            raise
        top = float(ctrl.max_speed)
        own = getattr(ctrl, "cruise", None)  # a boat's own pace: it carries way and slows for the corners
        cruise = (own or navigation.default_cruise(top)) if speed is None else min(abs(speed), top)
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
            turns_walking=not isinstance(ctrl, RhythmWalker),
            start_time=float(data.time),
            deadline=float(data.time) + seconds if seconds is not None else None,
            still_mine=lambda: self.session.controllers.get(robot) is ctrl,
            ready=folding[1] if folding is not None else (lambda: True),
            on_finish=finished,
            others=[
                navigation.Other(name, nav_map.root_body(model, name), nav_map.footprint_radius(model, data, name))
                for name in self._names(self._world())
                if name != robot and "/" not in name
            ],
            crossings=lambda: self._other_movers(robot),
            pose=own_map.pose if own_map is not None else None,
            watch=planner.blocked_ahead if own_map is not None else None,
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

    def _other_movers(self, robot: str) -> list[Any]:
        """The other robots on trips across the floor, for the one named to give way to (traffic.py)."""
        found = []
        for name, trip in self._live_trips.items():
            if name != robot and isinstance(trip, navigation.Navigator):
                mover = trip.mover()
                if mover is not None:
                    found.append(mover)
        return found

    def _dive(self, robot: str, to, pilot: marine.RovPilot, speed, seconds, model, data) -> marine.DiveTrip:
        """move(to=...) for an ROV: straight to a point under water, [x, y, z] ([x, y] keeps the depth it holds), or
        next to an object or another robot at its height (marine.dive_target)."""
        named = None if isinstance(to, (list, tuple)) or NUMBER_LIST.fullmatch(str(to)) else to
        destination = self._destination(robot, to, model, data) if named is not None else None
        try:
            target, note = marine.dive_target(pilot, to, destination, model, data)
            via, way = marine.dive_route(pilot, model, data, target) if pilot.water is not None else ([], None)
        except navigation.NoWay:
            raise  # no water to dive in there: an answer (move says why), not a mistake in the call
        except ValueError as exc:
            raise BehaviorError(str(exc)) from exc
        key = TRIP + robot

        def finished(trip: marine.DiveTrip) -> None:  # runs inside a sim step, with the session lock held
            if self.session.controllers.get(key) is trip:
                self.session.set_controller(key, None)
                self._last_trips[robot] = trip.result()
            if self._live_trips.get(robot) is trip:
                del self._live_trips[robot]

        with self.session.lock:
            if self.session.controllers.get(robot) is not pilot:
                raise BehaviorError(f"{robot} was changed while its dive was planned; send it again")
            if speed is not None:
                pilot.cruise = min(abs(float(speed)), pilot.max_speed)
            now = float(self.session.data.time)
            trip = marine.DiveTrip(
                robot,
                pilot,
                target,
                self.session.data.subtree_com[pilot.root].copy(),
                start_time=now,
                deadline=now + seconds if seconds is not None else None,
                still_mine=lambda: self.session.controllers.get(robot) is pilot,
                on_finish=finished,
                name=destination.name if destination is not None else None,
                note="; ".join(n for n in (note, way) if n) or None,
                via=via,
            )
            self.session.set_controller(key, trip)
            self._live_trips[robot] = trip
        return trip

    @_commanded("heave", lambda m, a: (m.resolve(a["robot"]), {"rate": float(a["rate"])}))
    def heave(self, robot: str, rate: float) -> None:
        """Up (+) or down (-) in m/s for an ROV driven from the keyboard; it holds the depth it reaches."""
        with self.session.lock:
            pilot = self.session.controllers.get(self.resolve(robot))
            if isinstance(pilot, marine.RovPilot):
                pilot.command(pilot.speed, pilot.turn, heave=rate)

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

    def trip(self, robot: str) -> navigation.Navigator | marine.DiveTrip | airways.FlightTrip | None:
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
        elif isinstance(trip, (navigation.Navigator, marine.DiveTrip)):
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

    def _fold_for_trip(
        self, robot: str, kind: Kind, walker: Any, data: mujoco.MjData
    ) -> tuple[mujoco.MjData, Callable[[], bool]] | None:
        """For a trip of a robot carrying an idle arm that sticks out (FOLD_GAIN wider than folded): fold the arm over
        the robot first (ArmMotion.hold_pose, as an errand does), and (the world as it will be with the arm folded, when
        the trip may set off). None to go as it is: no arm, a busy one, one holding something, or one already in.
        ANYmal with a UR5e as it was built was 1.25 m wide and went through no door; folded, 0.67 m. Session lock
        held."""
        if self.errand(robot) is not None:
            return None  # an errand folds the arm itself, and holds what it carries
        arms = [a for a in self.arms(robot) if a != robot]
        if not arms:
            return None
        name = arms[0]
        busy = self.session.controllers.get(name)
        if busy is not None and not getattr(busy, "done", True):
            return None
        model = self.session.model
        tcp = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}/tcp")]
        for obj in self._movable():
            if float(np.linalg.norm(data.xpos[self._body(obj)] - tcp)) < HOLDING_NEAR:
                return None  # it holds something, or is right at it: left as it is
        try:
            arm = self._arm_motion(name)
        except BehaviorError:
            return None
        arm.sync(data)
        tuck = arm.hold_pose(0.0, 0.0)
        if tuck is None:
            return None
        folded = fetch._posed(model, data, arm, tuck[1])
        try:
            now = nav_map.measure_body(model, data, robot, kind, walker)
            then = nav_map.measure_body(model, folded, robot, kind, walker)
        except ValueError:
            return None
        # Narrower, or less far round as it turns: a mobile Panda's arm, held out ahead of it as built, caught a
        # swing door's leaf as the robot turned in the doorway (the map leaves the leaf out) and it stuck there.
        if then.radius > now.radius - FOLD_GAIN and then.reach > now.reach - FOLD_GAIN:
            return None
        moves = Hand(arm)._joint_way(arm.q, tuck[1], None, "stow", LOADED_TILT, -1)
        if moves is not None:
            arm.queue.extend(moves)
        else:
            arm.move_to(tuck[0])
        return folded, lambda: arm.done

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

    @_commanded("pose", lambda m, a: (m.resolve(a["robot"]), {"name": a["name"]}))
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
        found = {
            "robot": robot,
            "position": [round(float(v), 3) for v in d.xpos[root]],
            "heading_deg": round(math.degrees(math.atan2(rot[1, 0], rot[0, 0])), 1),
            "upright": bool(upright),
        }
        if getattr(ctrl, "blocked", False):  # a walker standing before ground too steep for it (quadruped.py)
            found["blocked"] = ctrl.status()["blocked"]
        return found

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
            if robot is None:  # an open catalogue arm: its file's servos know nothing of the weight they carry
                entry = _entry(name.split("/")[0], self._world())
                gravity = entry is not None and entry.catalogue is not None
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

    @_commanded("move_arm", _tell_arm_target)
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

    @_commanded("gripper", lambda m, a: (m._arm_name(a["robot"]), {"state": a["state"]}))
    def set_gripper(self, robot: str, state: str) -> ArmMotion:
        with self.session.lock:
            arm = self._arm(robot)
            if arm.gripper < 0:
                raise BehaviorError(f"{robot} has no gripper")
            arm.set_gripper(state)
        return arm

    @_commanded(
        "jog",
        lambda m, a: (m._arm_name(a["robot"]), {"velocity": _floats(a["velocity"]), "gripper": a["gripper"]}),
    )
    def jog_arm(self, robot: str, velocity: list[float], gripper: str | None = None) -> ArmMotion:
        """Teleoperation of an arm's tool: move it at `velocity` [x, y, z] m/s in the world, pointing down, until told
        otherwise (a zero velocity holds it; a gamepad's stream of these stops it too if it goes quiet). gripper
        "open" or "close" sets the hand at once."""
        v = np.asarray(velocity, float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise BehaviorError(f"velocity must be [x, y, z] in m/s, got {list(velocity)}")
        if gripper not in (None, "open", "close"):
            raise BehaviorError(f'gripper is "open" or "close", got {gripper!r}')
        with self.session.lock:
            name = self._arm_name(robot)
            self._check_carrier(name)
            arm = self._arm_motion(name)
            if gripper is not None and arm.gripper < 0:
                raise BehaviorError(f"{robot} has no gripper")
            arm.jog(v, self.session.data, gripper)
        return arm

    @_commanded(
        "pick_and_place",
        lambda m, a: (
            m._arm_name(a["robot"]),
            {"object": a["obj"], "target": a["target"] if isinstance(a["target"], str) else _floats(a["target"])},
        ),
    )
    def pick_and_place(
        self,
        robot: str,
        obj: str,
        target: str | list[float],
        go: bool = True,
        grasp: str | None = None,
        pour: bool = False,
    ) -> tuple[ArmMotion | fetch.Errand, dict[str, Any]]:
        """Pick `obj` up and put it on or in `target`. An arm that rides on a robot that drives or walks, told to fetch
        something out of its reach from where the robot stands, has the robot go over (fetch.Errand): to the object,
        then with it to the place. `go=False` keeps the robot where it is and refuses instead, as before. Returns what
        does it (an ArmMotion or an Errand: wait until `done`) and {object, target, region}. A generated object (mug,
        bottle, hammer, ...) is taken by one of the grasps its generator gives (`grasp` asks for a kind: "handle",
        "body", "shaft", "rim"), the first the arm can reach and carry; the answer says which. With `pour` the
        thing is a cup or a bowl with something in it: the arm carries it over the target container, tilts it about
        its lip until what is in it has run out, tips it back and puts it down where it stood."""
        with self.session.lock:
            name = self._arm_name(robot)
            if mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}/gripper") < 0:
                raise BehaviorError(f"{name} has no gripper to pick with; {self._gripper_hint(name)}")
            self._refuse_if_touching_another_robot(name)
            model, tool = self._library_tool(name)
            if tool is not None and tool.cannot_pick:
                raise BehaviorError(
                    f"{name} ({model.title}) cannot pick and place yet: {tool.cannot_pick}; move_arm and gripper work"
                )
            self._check_carrier(name)  # before its motion takes hold of the arm
            arm = self._arm_motion(name)
            obj = self._object_name(obj)
            pick = self._object_geometry(obj, arm, grasp)
            if pick.get("irregular") and arm.tool is None:  # fingers: where and which way to take hold, from its shape
                pick = self._planned_pick(arm, obj, pick)
            # A suction cup or a magnet picks from where it stands: an errand's grasp and carry are for fingers.
            goes = go and self._goes(arm.robot) and arm.tool is None
            riding = catching.velocity(self.session.model, self.session.data, self._body(obj))
            if riding is not None and not fetch.holding(self.session.data, arm, self._body(obj)):
                caught = self._catch(arm, obj, target, pick["half_height"], riding, goes)
                return caught, {"object": obj, "target": target, "region": caught.region}
            if arm.tool == "magnet":
                self._check_steel(arm.robot, obj)
            if pour:
                return arm, self._pour(arm, obj, target, pick)
            if isinstance(target, str) and target.strip().lower() == "hold" and not self._named("hold"):
                return arm, self._hold(arm, obj, pick)
            try:
                place, place_top, region = self._place_target(target, obj)
            except BehaviorError:
                if not goes or self._place_kind(target) != "terrain":
                    raise
                place = None  # a terrain patch: only a robot that goes over puts things on one
            if place is not None:  # (a plain sphere too: an apple carried into the cup beside the bowl was knocked out)
                place_top = max(place_top, self._tallest_between(obj, pick["center"], place))
            if place is not None and not (goes and fetch.holding(self.session.data, arm, self._body(obj))):
                try:
                    used = self._take(arm, pick, place, place_top)
                    if pick.get("built") is not None and _firm(pick["built"]) and used is not None:
                        hints.FirmHold(self.session, arm, self._body(obj), used.force, used.short).start()
                    elif pick.get("built") is None and self._is_ball(
                        obj
                    ):  # round all over: slips out of a plain squeeze
                        hints.FirmHold(self.session, arm, self._body(obj), hints.FORCE_MIN).start()
                except OutOfReach as exc:
                    # Out of reach only for a part of its own robot in the way of the carry (exc.over): that part
                    # goes with it wherever it drives, and the refusal names it.
                    if not goes or exc.over is not None:
                        text = self._pick_out_of_reach(arm, exc, pick["center"], place, target, obj)
                        if "object" in exc.bad and arm.tool is None and 2 * pick["half_height"] < FLAT:
                            text += (
                                f". {obj} is only {2000 * pick['half_height']:.0f} mm thick: fingers cannot close "
                                "round something that flat, a suction cup can (update_robot with "
                                "end_effector='suction')"
                            )
                        raise BehaviorError(text) from exc
                except ValueError as exc:
                    raise BehaviorError(str(exc)) from exc
                else:
                    info = {"object": obj, "target": target, "region": region}
                    if used is not None:
                        return arm, {**info, "grasp": used.describe()}
                    return arm, {**info, **({"grasp": pick["grasp"]} if "grasp" in pick else {})}
            errand = self._fetch(arm, obj, target)
        return errand, {"object": obj, "target": target, "region": errand.place.region}

    def _is_ball(self, obj: str) -> bool:
        entry = self._object_entry(obj)
        return entry is not None and entry.shape == "sphere"

    def _take(self, arm: ArmMotion, pick: dict[str, Any], place: np.ndarray, place_top: float) -> hints.Choice | None:
        """Queue the pick and place with the first grasp the arm can reach and carry (a generated object has several
        to try, best first); raises what the best one raised when none works, saying what the others did."""
        tries = pick.get("choices") or [None]
        first: OutOfReach | ValueError | None = None
        others: list[str] = []
        for choice in tries:
            point, half, faces = (
                (pick["center"], pick["half_height"], pick["faces"])
                if choice is None
                else (choice.point, choice.half_height, choice.faces)
            )
            try:
                reach = pick.get("reach_above", 0.0) if choice is None else 0.0
                arm.pick_and_place(point, half, place, place_top, faces, reach)
                return choice
            except (OutOfReach, ValueError) as exc:
                if first is None:
                    first = exc
                    pick["center"], pick["half_height"] = point, half  # the advice below is for the best grasp
                elif choice is not None:
                    others.append(f"{choice.hint.kind if choice.hint else 'body'}: {exc}")
        assert first is not None
        if others and isinstance(first, ValueError) and not isinstance(first, OutOfReach):
            first.args = (f"{first.args[0]} (the other grasps: {'; '.join(others)})",)
        raise first

    def hand_over(
        self, giver: str, receiver: str, obj: str, target: str | list[float]
    ) -> tuple[relay.Relay, dict[str, Any]]:
        """`giver` picks `obj` up and sets it down at a spot both arms reach and goes home; then `receiver` picks it
        up from there and puts it on or in `target`. Returns the relay (wait until `done`) and {object, target,
        region, spot}. Refuses before moving when either arm rides on a robot that drives or walks, either has no
        gripper, no spot suits both, or the receiver cannot put the object on the target."""
        with self.session.lock:
            names = []
            for who in (giver, receiver):
                name = self._arm_name(who)
                if mujoco.mj_name2id(self.session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}/gripper") < 0:
                    raise BehaviorError(f"{name} has no gripper to pick with; {self._gripper_hint(name)}")
                if self._goes(name):
                    raise BehaviorError(
                        f"{name} rides on a robot that moves: hand_over is for arms that stay where they are"
                    )
                self._check_carrier(name)
                names.append(name)
            if names[0] == names[1]:
                raise BehaviorError("hand_over needs two different arms")
            give, take = (self._arm_motion(n) for n in names)
            for name, motion in zip(names, (give, take), strict=True):
                if not motion.done or any(
                    isinstance(c, relay.Relay) and not c.done and name in (c.giver.robot, c.receiver)
                    for c in self.session.controllers.values()
                ):
                    raise BehaviorError(
                        f"{name} is still busy with a motion or a hand-over: wait for it to finish, or stop it first"
                    )
            obj = self._object_name(obj)
            pick = self._object_geometry(obj, give)  # a modelled object is taken by its hints
            half = pick["half_height"]
            if fetch.holding(self.session.data, give, self._body(obj)):
                raise BehaviorError(f"{names[0]} is already holding {obj}: put it down first")
            place, place_top, region = self._place_target(target)
            bottom = float(pick["center"][2]) - half
            avoid = [self.session.data.xpos[self._body(o)] for o in self._movable() if o != obj]
            if isinstance(target, list):
                avoid.append(place)
            spots = relay.hand_off_spots(self.session.data, give, take, avoid)
            if not spots:
                raise BehaviorError(
                    f"no spot on the floor is in reach of both {names[0]} and {names[1]}: stand them closer"
                )
            why = ""
            for spot in spots:
                down = np.array([spot[0], spot[1], bottom])
                try:
                    give.pick_and_place(pick["center"], half, down, None, pick["faces"])
                except (OutOfReach, ValueError) as exc:
                    give.queue.clear()
                    why = str(exc)
                    if isinstance(exc, OutOfReach) and "object" in exc.bad:
                        raise BehaviorError(
                            self._pick_out_of_reach(give, exc, pick["center"], down, target, obj)
                        ) from exc
                    continue
                saved = list(take.queue)
                try:
                    take.pick_and_place(
                        np.array([spot[0], spot[1], bottom + half]), half, place, place_top, pick["faces"]
                    )
                except (OutOfReach, ValueError) as exc:
                    give.queue.clear()
                    why = str(exc)
                    continue
                finally:
                    take.queue.clear()
                    take.queue.extend(saved)
                give.move_to(relay.parking(give.base_and_reach(self.session.data)[0], np.asarray(spot, float)))
                break
            else:
                raise BehaviorError(
                    f"{names[0]} and {names[1]} cannot hand {obj} over at any of the {len(spots)} spots in reach of "
                    f"both: {why}"
                )
            body = self._body(obj)

            def arrived() -> bool:
                return bool(np.linalg.norm(self.session.data.xpos[body][:2] - spot) < relay.ARRIVED)

            def start() -> Any:
                try:
                    motion, _ = self.pick_and_place(names[1], obj, target, go=False)
                except BehaviorError as exc:
                    return f"{names[0]} set {obj} down, but {names[1]} cannot take it to {target}: {exc}"
                return motion

            key = relay.KEY + names[0]
            handed = relay.Relay(
                give, names[1], obj, np.asarray(spot, float), arrived, start, on_done=lambda: self._drop_relay(key)
            )
            self.session.set_controller(names[0], give)
            self.session.set_controller(key, handed)
        return handed, {"object": obj, "target": target, "region": region, "spot": [round(float(v), 3) for v in spot]}

    def _drop_relay(self, key: str) -> None:
        with self.session.lock:
            self.session.set_controller(key, None)

    def _catch(
        self,
        arm: ArmMotion,
        obj: str,
        target: str | list[float],
        half: float,
        riding: np.ndarray,
        goes: bool,
        tries: int = 0,
    ) -> catching.Catch | catch_stand.Approach:
        """Catch `obj` as it rides by on a conveyor (catching.Catch), from where the robot stands, then put it on
        `target`: from there if the arm reaches it, else (an arm on a robot that goes) the robot carries it there
        (fetch.Errand). A robot that goes, whose arm does not reach the box's way from where it stands, first goes to
        a place beside the belt from which it does (catch_stand.Approach, which plans the catch again from where it
        stops, `tries` times at most). Raises BehaviorError before anything moves when the arm cannot meet it or put
        it down. Session lock held."""
        model, data = self.session.model, self.session.data
        body = self._body(obj)
        try:
            meet = catching.meeting(model, data, arm, body, half, riding)
        except catching.CatchError as exc:
            hint = "; or move the robot next to the conveyor first (move with to=...)" if goes else ""
            if goes and tries < catch_stand.TRIES:
                try:
                    return self._approach(arm, obj, target, body, half, riding, tries)
                except catch_stand.NoStand as why:
                    hint = f"; {why}"
                    if why.late:  # it fetches the box where it lies at the end of the belt
                        return self._approach(arm, obj, target, body, half, riding, tries, wait=True)
            raise BehaviorError(f"{obj} is riding on at {np.linalg.norm(riding):.2f} m/s: {exc}{hint}") from exc
        below = 2 * half if arm.tool is not None else half  # what the load hangs below the tool point
        try:
            place, place_top, region = self._place_target(target)
        except BehaviorError:
            if not goes:
                raise
            place, place_top, region = None, None, None
        puts = place is not None and self._puts_from(arm, meet, place, below, place_top)
        if not puts and not goes:
            raise BehaviorError(
                f"{arm.robot} can catch {obj} but not put it on {target} from where it stands; move the arm or the "
                f"{target} nearer each other"
            )
        name = arm.robot
        caught: catching.Catch

        def then(data: mujoco.MjData, ok: bool) -> None:  # inside a sim step, the session lock held
            arm.sync(data)
            arm.q = data.qpos[arm.chain.qpos_adr].copy()
            arm.queue.clear()
            self.session.set_controller(name, arm)
            if not ok:
                return
            if puts:
                try:
                    arm.put_down(place, below, place_top)
                except (OutOfReach, ValueError) as exc:
                    caught.note = f"it caught {obj} but could not put it down from where it lifted it: {exc}"
                    arm.queue.clear()
                caught.after = arm
                return
            # Folded over the robot first, then off to the place: an errand that starts holding its load goes at
            # once, and from the arm stretched out over the belt the fold was not found (hold_pose seeds its search
            # from the pose the arm is in), so it is looked for from the arm's home pose.
            now, arm.q = arm.q, arm.home.copy()
            tuck = arm.hold_pose(half, catching.load_radius(model, body))
            arm.q = now
            if tuck is None:
                caught.note = f"it caught {obj} but cannot fold it over {arm.robot.split('/')[0]} to carry it"
                return
            moves = Hand(arm, body)._joint_way(arm.q, tuck[1], (half, half), "stow", LOADED_TILT, body)
            if moves is not None:
                arm.queue.extend(moves)
            else:
                arm.move_to(tuck[0])

            def carry() -> fetch.Errand | str:
                try:
                    errand = self._fetch(arm, obj, target)
                except BehaviorError as exc:
                    return f"it caught {obj} but cannot carry it to {target}: {exc}"
                caught.region = errand.place.region
                return errand

            caught.after = catching.Then(arm, carry)
            self.session.set_controller(catching.KEY + arm.robot, caught.after)

        caught = catching.Catch(arm, body, half, meet, then=then, carries=not puts)
        caught.region = region
        self.session.set_controller(name, caught)
        return caught

    def _approach(
        self,
        arm: ArmMotion,
        obj: str,
        target: str | list[float],
        body: int,
        half: float,
        riding: np.ndarray,
        tries: int,
        wait: bool = False,
    ) -> catch_stand.Approach:
        """Send the robot to a place beside the belt from which `arm` catches `obj`, to catch it from there; with
        `wait`, to wait where it is for the box to lie still at the end of the belt, and fetch it there."""
        stance = None if wait else catch_stand.stand_for(self, arm, body, half, riding, self.session.data)
        robot = arm.robot.split("/", 1)[0]
        self.ready_to_go(robot)
        try:
            region = self._place_target(target)[2]
        except BehaviorError:
            region = None  # a terrain patch, or a place the robot goes to: known once it has the load
        approach = catch_stand.Approach(
            self,
            robot,
            arm,
            obj,
            body,
            half,
            target,
            None if stance is None else stance.spot,
            region,
            tries=tries,
            drive_s=0.0 if stance is None else stance.drive_s,
        )
        self._end_trip(robot)
        self.session.set_controller(catch_stand.KEY + robot, approach)
        return approach

    def catch_from_here(
        self, arm: ArmMotion, obj: str, target: str | list[float], half: float, riding: np.ndarray, tries: int
    ) -> catching.Catch | catch_stand.Approach:
        """Plan the catch again from where the robot stopped (catch_stand.Approach), inside a sim step."""
        return self._catch(arm, obj, target, half, riding, True, tries)

    def end_approach(self, approach: catch_stand.Approach) -> None:
        """Called by an approach as it ends or hands over (inside a sim step, the session lock held)."""
        if self.session.controllers.get(catch_stand.KEY + approach.robot) is approach:
            self.session.set_controller(catch_stand.KEY + approach.robot, None)

    def _puts_from(
        self, arm: ArmMotion, meet: catching.Meeting, place: np.ndarray, below: float, top: float | None
    ) -> bool:
        """Whether the arm, having caught the load and lifted it off the belt, puts it down on `place` from there."""
        now = arm.q
        arm.q = meet.q_lifted
        try:
            arm.put_down(place, below, top)
            return True
        except (OutOfReach, ValueError):
            return False
        finally:
            arm.queue.clear()
            arm.q = now

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
        self.ready_to_go(robot)
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
        self.start_errand(robot, errand)
        return errand

    def ready_to_go(self, robot: str) -> None:
        """Before planning an errand for `robot` (session lock held): its walker made, and refused if it has fallen."""
        _check_standing(self._mobile_controller(robot, self.kind(robot), self._world(), create=True))

    def start_errand(self, robot: str, errand: fetch.Errand | visit.Visit) -> None:
        """Set an errand going for `robot` (session lock held): a fetch, or a visit to work a mechanism."""
        self._end_trip(robot)
        self._last_trips.pop(robot, None)
        self.session.set_controller(fetch.KEY + robot, errand)
        self._errands[robot] = errand

    def errand(self, robot: str) -> fetch.Errand | visit.Visit | None:
        """The robot's errand (pick_and_place or operate that goes over), while it is on one."""
        found = self.session.controllers.get(fetch.KEY + robot)
        return found if isinstance(found, (fetch.Errand, visit.Visit)) else None

    def errand_ended(self, errand: fetch.Errand | visit.Visit) -> None:
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

    def scene_map(self, robot: str) -> nav_map.NavMap:
        """The map of where the robot's center may go, computed from the scene as it stands (what `move(to=...)`
        plans on without `map="own"`): for judging a robot's own map against it."""
        robot = self.resolve(robot)
        with self.session.lock:
            data = copy.copy(self.session.data)
        return self._fetch_planner(robot, data, None, None).map(data)

    def own_map_planner(self, robot: str, mapper: Any, unknown_blocks: bool = False) -> tuple[Any, mujoco.MjData]:
        """A planner on `mapper`'s grid for the robot where it stands (for exploring: what it can reach on its map,
        unknown space counting as open), and the data it is planned on."""
        from robotforge.behaviors import map_nav

        robot = self.resolve(robot)
        with self.session.lock:
            data = copy.copy(self.session.data)
        base = self._fetch_planner(robot, data, None, None)
        planner = map_nav.MapPlanner(
            self.session.model, robot, base.body, base.destination, mapper, unknown_blocks=unknown_blocks
        )
        return planner, data

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

    def _refuse_if_touching_another_robot(self, arm: str) -> None:
        """A robot whose parts press on another robot's cannot do a pick: it would push against it and fail without a
        word (two Pandas a metre apart whose hands meet at home)."""
        m, d = self.session.model, self.session.data
        mine = arm.split("/", 1)[0]

        def owner(geom: int) -> tuple[str, str]:
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[geom])) or ""
            return (body.split("/", 1)[0] if "/" in body else ""), body

        for c in range(d.ncon):
            (a, a_body), (b, b_body) = owner(d.contact[c].geom1), owner(d.contact[c].geom2)
            if a and b and a != b and mine in (a, b):
                other = b_body if a == mine else a_body
                here = a_body if a == mine else b_body
                raise BehaviorError(
                    f"{arm} touches another robot already ({here} against {other}): they start in each other's way "
                    f"and the arm would only push against it; move one of them with update_robot(pos=...) or turn it"
                )

    def position_of(self, obj: str) -> np.ndarray:
        """Where `obj` is now (m), to tell afterwards whether it moved."""
        return self.session.data.xpos[self._body(obj)].copy()

    def placed(self, obj: str, region: dict[str, Any], start: np.ndarray | None = None) -> dict[str, Any]:
        """Check whether `obj` ended up inside the target region (PLACE_TOL m of slack sideways: the task check judges
        by the target's whole top, and a mug 2 mm off a shelf slot stands on the shelf). When it did not, "why" says
        how it went wrong; `start`, where it was before the arm began, tells a pick that never lifted it."""
        body = self._body(obj)
        pos = self.session.data.xpos[body]
        center = np.asarray(region["center"])
        half = np.asarray(region["half_xy"])
        inside = bool(np.all(np.abs(pos[:2] - center[:2]) <= half + PLACE_TOL) and pos[2] >= region["min_z"])
        out = {"success": inside, "object_position": [round(float(v), 3) for v in pos]}
        if not inside:
            out["why"] = _why_not_placed(obj, pos, center, half, float(region["min_z"]), start)
        entry = self._object_entry(obj)
        built = hints.built_for(entry)
        if built is not None:  # a generated thing is also judged by how it stands
            out["tilt_deg"] = round(hints.tilt_deg(self.session.data, body, built), 1)
        return out

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

    def _check_steel(self, robot: str, obj: str) -> None:
        """An electromagnet holds steel only: say so before the arm goes for something it cannot hold (it went all
        the way there and back for a plastic plate, and came back with nothing)."""
        m = self.session.model
        body = self._body(obj)
        if end_effectors.magnetic(m, [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body]):
            return
        entry = self._object_entry(obj)
        made = f"is {entry.material or 'plastic'}" if entry is not None else "is not steel"
        raise BehaviorError(
            f"{robot}'s electromagnet holds steel only, and {obj} {made}: make it steel (update_object with "
            f"material='steel'), or change the tool (update_robot with end_effector='suction')"
        )

    def _object_geometry(self, name: str, arm: ArmMotion | None = None, grasp: str | None = None) -> dict[str, Any]:
        entry = self._object_entry(name)
        built = hints.built_for(entry)
        if built is not None and arm is not None:
            return self._hinted_geometry(name, entry, built, arm, grasp)
        if entry is not None and entry.shape in mechanisms.MECHANISMS:
            actions = " or ".join(mechanisms.MECHANISMS[entry.shape].actions) or "set its speed"
            raise BehaviorError(f"{name} is a {entry.shape}, built in place: operate can {actions} it")
        if entry is not None and entry.shape in mechanisms.SOFT:
            raise BehaviorError(
                f"{name} is a {entry.shape}, soft: pick_and_place carries solid things; move_arm and gripper can take "
                "hold of it by hand"
            )
        body = self._body(name)
        if entry is not None and entry.fixed:
            raise BehaviorError(f"{name} is fixed in place and cannot be picked up")
        if entry is not None and entry.shape in ("table", "bin"):
            raise BehaviorError(f"{name} is a {entry.shape}; pick up something small like a box or ball")
        center = self.session.data.xpos[body].copy()
        irregular = False
        if entry is not None and entry.shape in ("box", "sphere", "cylinder"):
            half = {"box": entry.size[-1] / 2, "sphere": entry.size[0] / 2, "cylinder": entry.size[-1] / 2}[entry.shape]
        else:
            model = self.session.model
            geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body and collides(model, i)]
            half = float(model.geom_rbound[geoms[0] if geoms else 0])
            # A thing made of several shapes, or of a shape that is no box, can or ball (a mesh, a T, a bottle): its
            # grasp is worked out from its shape (grasping.py).
            kinds = {int(model.geom_type[i]) for i in geoms}
            primitive = {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_SPHERE)}
            primitive |= {int(mujoco.mjtGeom.mjGEOM_CYLINDER)}
            irregular = len(geoms) > 1 or not kinds <= primitive
        faces = None  # a box's sides, for jaws closing square to them (see arm.SQUARE_TOL)
        if entry is not None and entry.shape == "box":
            sides = self.session.data.xmat[body].reshape(3, 3)[:, :2].T.copy()
            sides[:, 2] = 0.0
            if np.all(np.linalg.norm(sides, axis=1) > 0.5):  # standing upright, not on an edge
                faces = sides / np.linalg.norm(sides, axis=1, keepdims=True)
        return {
            "center": center,
            "half_height": half,
            "faces": faces,
            "irregular": irregular,
            "named": entry is not None,
        }

    def _planned_pick(self, arm: ArmMotion, obj: str, pick: dict[str, Any]) -> dict[str, Any]:
        """`pick` for a thing of no simple shape: where the tool goes, how far the load reaches below it and which
        way the jaws close, from the best grasp of its shape. A thing the world made (add_object) that no grasp holds
        is refused with why; an imported thing that no grasp holds is picked as it always was."""
        body = self._body(obj)
        model, data = self.session.model, self.session.data
        opening = fetch.gripper_opening(model, arm) or GRASP_OPENING
        parts = grasping.body_parts(model, data, body)
        try:
            grasps = grasping.plan_cached(parts, opening, data.xipos[body].copy()) if parts else []
        except ValueError:
            grasps = []
        if not grasps:
            if not pick["named"]:
                return pick
            points = np.vstack(parts) if parts else np.zeros((1, 3))
            sizes = ", ".join(f"{100 * float(v):.0f}" for v in points.max(axis=0) - points.min(axis=0))
            raise BehaviorError(
                f"{arm.robot} cannot take hold of {obj}: no way to close its {1000 * opening:.0f} mm jaws on its "
                f"{sizes} cm (x, y, z) that holds it, too wide across or too flat for fingers to get under; "
                "a suction cup may hold it (update_robot with end_effector='suction')"
            )
        best = grasps[0]
        faces = np.array([[best.closing[0], best.closing[1], 0.0]])
        return {
            **pick,
            "center": best.center,
            "half_height": best.below,
            "faces": faces,
            "reach_above": best.above,
            "grasp": best.as_dict(),
        }

    def plan_grasps(self, robot: str, obj: str) -> dict[str, Any]:
        """The ways `robot`'s gripper can take hold of `obj`, best first, worked out from the thing's shape: where the
        tool goes, which way the jaws close, how wide the thing is there and why it holds. The search runs outside the
        session lock (a dense mesh is cut down to its extremes first, a repeat is remembered), so the sim goes on."""
        with self.session.lock:
            name = self._arm_name(robot)
            arm = self._arm_motion(name)
            obj = self._object_name(obj)
            body = self._body(obj)
            model, data = self.session.model, self.session.data
            opening = fetch.gripper_opening(model, arm) or GRASP_OPENING
            parts = grasping.body_parts(model, data, body)
            com = data.xipos[body].copy()
        try:
            grasps = grasping.plan_cached(parts, opening, com) if parts else []
        except ValueError:
            grasps = []
        return {
            "object": obj,
            "robot": name,
            "jaws_open_m": round(float(opening), 3),
            "grasps": [g.as_dict() for g in grasps],
            "best": grasps[0].as_dict() if grasps else None,
            "why_none": None if grasps else "too wide across for the jaws, or too flat for fingers to get under",
        }

    def _named(self, name: str) -> bool:
        """Whether the world has an object called `name` (a target called "hold" is that object, not the keyword)."""
        world = self._world()
        return world is not None and any(o.name == name for o in world.objects)

    def _hold(self, arm: ArmMotion, obj: str, pick: dict[str, Any]) -> dict[str, Any]:
        """Queue taking `obj` and lifting it to stay held (target "hold": the receiving half of a hand-over)."""
        body = self._body(obj)
        began = float(self.session.data.xpos[body][2])
        first: Exception | None = None
        used = None
        for choice in pick.get("choices") or [None]:
            point, half, faces = (
                (pick["center"], pick["half_height"], pick["faces"])
                if choice is None
                else (choice.point, choice.half_height, choice.faces)
            )
            try:
                arm.pick_and_hold(point, half, faces)
                used = choice
                break
            except (OutOfReach, ValueError) as exc:
                first = first or exc
        else:
            if isinstance(first, OutOfReach):
                raise BehaviorError(
                    self._pick_out_of_reach(arm, first, pick["center"], pick["center"], "hold", obj)
                ) from first
            raise BehaviorError(str(first)) from first
        built = pick.get("built")
        if built is not None and _firm(built) and used is not None:
            hints.FirmHold(self.session, arm, body, used.force, used.short).start(keep=True)
        return {
            "object": obj,
            "target": "hold",
            "region": None,
            "hold": {"robot": arm.robot, "began_z": began},
            **({"grasp": used.describe()} if used is not None else {}),
        }

    def held(self, info: dict[str, Any]) -> dict[str, Any]:
        """Whether the arm that was told to hold a thing has it lifted in its fingers."""
        hold = info["hold"]
        arm = self._arm_motion(hold["robot"])
        body = self._body(info["object"])
        up = float(self.session.data.xpos[body][2]) - hold["began_z"]
        holding = fetch.holding(self.session.data, arm, body)
        return {"success": bool(holding and up >= HOLD_LIFT), "held": bool(holding), "lifted_cm": round(up * 100, 1)}

    def _pour(self, arm: ArmMotion, obj: str, target: Any, pick: dict[str, Any]) -> dict[str, Any]:
        """Queue taking `obj` (a cup, a mug, a bowl), pouring it into `target` and putting it back (pour.py)."""
        from robotforge.behaviors import pour as pouring

        built = pick.get("built")
        if built is None or "mouth_r" not in built.facts or built.facts.get("closed"):
            raise BehaviorError(f"{obj} has no open mouth to pour from: pour takes a cup, a mug or a bowl")
        if not isinstance(target, str):
            raise BehaviorError("pour into an object: give its name (a bowl, a cup)")
        world = self._world()
        names = [o.name for o in world.objects] if world else []
        found = wording.match(target, names)
        entry = self._object_entry(found) if found else None
        into = hints.built_for(entry)
        if entry is None or into is None or "mouth_r" not in into.facts:
            raise BehaviorError(f"{target} cannot be poured into: it needs an open mouth (a cup, a mug, a bowl)")
        d = self.session.data
        mouth = d.xpos[self._body(found)] + np.array([0.0, 0.0, float(into.facts["rim_z"])])
        cup = d.xpos[pick["body"]]
        balls = [
            o.name
            for o in world.objects
            if o.shape == "sphere" and not o.fixed and self._inside(d.xpos[self._body(o.name)], cup, built)
        ]
        spec = pouring.Pour(
            float(built.facts["rim_z"]), float(built.facts["mouth_r"]), 0.0, mouth, float(into.facts["mouth_r"])
        )
        base = d.xpos[pick["body"]].copy()
        first: Exception | None = None
        # A hand whose finger links reach above the cup's rim (a Robotiq's) holds it by the rim instead: balls caught
        # between a link and the mouth, upside down, and stayed in the cup.
        m = self.session.model
        palm = hints.palm_of(m, arm)
        rim = float(built.facts["rim_z"])
        high, _ = hints.choices(d, pick["body"], built, fetch.gripper_opening(m, arm), palm, None, m, True)
        by_rim = {id(c.hint): c for c in high}
        chosen = [
            by_rim.get(id(c.hint), c) if palm.links > rim - c.half_height + LINK_CLEAR else c
            for c in pick.get("choices") or []
        ]
        on_axis = [c for c in chosen if float(np.hypot(*(np.asarray(c.point[:2]) - base[:2]))) < 0.015]
        if not on_axis:
            raise BehaviorError(
                f"{obj} has no grip on its own axis (a body, a rim) to pour from: a handle would swing it"
            )
        for choice in on_axis:
            spec.held_at = choice.half_height
            try:
                toward = pouring.pick_and_pour(arm, choice.point, choice.half_height, choice.faces, spec, base)
                # tipped over, a cup turns in the fingers: held as firmly as a heavy thing is
                hints.FirmHold(self.session, arm, pick["body"], max(choice.force, POUR_FORCE), choice.short).start()
                break
            except (OutOfReach, ValueError) as exc:
                first = first or exc
        else:
            if isinstance(first, OutOfReach):
                raise BehaviorError(self._pick_out_of_reach(arm, first, pick["center"], base, target, obj)) from first
            raise BehaviorError(str(first)) from first
        return {
            "object": obj,
            "target": found,
            "pour": {
                "into": found,
                "toward_deg": round(math.degrees(math.atan2(toward[1], toward[0])), 1),
                "balls": balls,
                "mouth": [round(float(v), 3) for v in mouth],
            },
            "region": None,
        }

    @staticmethod
    def _inside(ball: np.ndarray, cup: np.ndarray, built: Any) -> bool:
        """Whether a ball lies in a cup's mouth, as the world has them."""
        reach = float(built.facts["mouth_r"]) + 0.01
        return bool(
            np.hypot(*(ball[:2] - cup[:2])) <= reach and -0.01 <= ball[2] - cup[2] <= float(built.facts["rim_z"]) + 0.02
        )

    def poured(self, info: dict[str, Any]) -> dict[str, Any]:
        """How a pour went: how many of the balls that were in the cup lie in the target now."""
        pour = info["pour"]
        d = self.session.data
        into = self._body(pour["into"])
        entry = self._object_entry(pour["into"])
        built = hints.built_for(entry)
        inside = sum(
            1 for n in pour["balls"] if built is not None and self._inside(d.xpos[self._body(n)], d.xpos[into], built)
        )
        total = len(pour["balls"])
        return {
            "success": total > 0 and inside >= math.ceil(0.75 * total),
            "balls_in_target": inside,
            "balls": total,
            "poured_toward_deg": pour["toward_deg"],
        }

    def _tallest_between(self, obj: str, start: np.ndarray, end: np.ndarray) -> float:
        """The height of the tallest thing in the corridor between two points (the things it carries over: a tray with
        cans in it, a shelf), beside what stands under the load's path, so that it is carried over them: the arm's
        planner knows only the place's own rim. Things no higher than the work surface do not count."""
        m, d = self.session.model, self.session.data
        me = self._body(obj)
        lo = np.minimum(start[:2], end[:2]) - CORRIDOR
        hi = np.maximum(start[:2], end[:2]) + CORRIDOR
        floor = float(min(start[2], end[2])) + 0.02
        top = -math.inf
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if b == 0 or b == me or "/" in name or not m.geom_contype[g]:
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            centre = d.geom_xpos[g] + rot @ m.geom_aabb[g][:3]
            half = np.abs(rot) @ m.geom_aabb[g][3:]
            if np.any(centre[:2] + half[:2] < lo) or np.any(centre[:2] - half[:2] > hi):
                continue
            top = max(top, float(centre[2] + half[2]))
        return top + CLUTTER_MARGIN if top > floor else -math.inf

    def _hinted_geometry(
        self, name: str, entry: ObjectEntry, built: Any, arm: ArmMotion, want: str | None
    ) -> dict[str, Any]:
        """Where a generated object is taken hold of: its generator's grasp hints at the pose it has now."""
        if entry.fixed:
            raise BehaviorError(f"{name} is fixed in place and cannot be picked up")
        body = self._body(name)
        m, d = self.session.model, self.session.data
        if arm.tool is not None:  # a suction cup or a magnet takes the middle of a flat top
            if built.facts.get("top_z") is None:
                raise BehaviorError(
                    f"{name} is a {entry.shape}: a {arm.tool} needs a flat top to hold by (a package, a tin, a book); "
                    "change the end effector to fingers (update_robot with end_effector='gripper')"
                )
            half = float(built.facts["top_z"]) / 2
            center = d.xpos[body] + d.xmat[body].reshape(3, 3) @ np.array([0.0, 0.0, half])
            return {"center": center, "half_height": half, "faces": None, "choices": None}
        opening = fetch.gripper_opening(m, arm)
        found, left = hints.choices(d, body, built, opening, hints.palm_of(m, arm), want, m)
        if not found:
            asked = f" (asked for a {want} grasp; it has {', '.join(hints.kinds_of(built)) or 'none'})" if want else ""
            why = "; ".join(left) or "it has no grasp a top-down gripper can use"
            raise BehaviorError(f"{arm.robot} cannot take hold of {name}, a {entry.shape}{asked}: {why}")
        first = found[0]
        return {
            "center": first.point,
            "half_height": first.half_height,
            "faces": first.faces,
            "choices": found,
            "built": built,
            "body": body,
        }

    def _place_target(
        self, target: str | list[float], obj: str | None = None
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
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
        base_name, slot_name = hints.parse_target(target, names) if "/" in target else (target, None)
        found = wording.match(base_name, names)
        entry = self._object_entry(found) if found else None
        if entry is None:
            raise BehaviorError(wording.unknown("object", target, names) + " (or give a position [x, y])")
        target = found
        m, d = self.session.model, self.session.data
        built = hints.built_for(entry)
        if built is not None:  # a generated fixture's slot, or the flat top of a generated thing
            return self._generated_place(target, built, slot_name, obj)
        if entry.shape == "shelf":
            return self._on_shelf(target, entry, slot_name)
        if entry.shape in mechanisms.CONTAINERS:
            return self._into(target)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
        if b >= 0:
            base = d.xpos[b].copy()
        else:  # fixed objects are plain geoms
            base = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, target)].copy()
        length, width = entry.size[0], entry.size[1] if len(entry.size) > 1 else entry.size[0]
        if entry.shape in ("cylinder", "sphere"):  # [diameter, height]: round, as wide each way
            width = length
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

    def _generated_place(
        self, name: str, built: Any, slot_name: str | None, moving: str | None
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Put something in a slot of the generated object `name` (a rack's socket, a tray, a shelf board) or on its
        flat top: the first free slot unless one is named ("rack/2")."""
        m, d = self.session.model, self.session.data
        taken = []
        for other in self._movable():
            if other != moving and other != name:
                taken.append(d.xpos[self._body(other)].copy())
        try:
            size = self._spread(moving)
            place = hints.place_for(m, d, name, built, slot_name, taken, size)
        except KeyError as exc:
            raise BehaviorError(exc.args[0]) from exc
        except LookupError as exc:
            raise BehaviorError(str(exc)) from exc
        if place is None:
            raise BehaviorError(
                f"{name} is a {built.shape}: nothing can be put in or on it (a rack, a tray and a box can)"
            )
        return place.point, place.top, place.region

    def _spread(self, name: str | None) -> float:
        """How much room a generated thing wants round it in a tray: its widest half extent from above, at most the
        width of a finger's reach (a long tool is not spread like a long box: it is laid, not stood)."""
        built = hints.built_for(self._object_entry(name)) if name else None
        if built is None:
            return 0.0
        return min(float(max(built.half[0], built.half[1])), SPREAD_MAX)

    def _on_shelf(self, name: str, entry: ObjectEntry, slot: str | None) -> tuple[np.ndarray, float, dict[str, Any]]:
        """The top board of a shelf: the only one with nothing over it, so the only one a gripper coming from above
        reaches (`name/2` asks for the second board counting from the bottom, refused unless it is the top)."""
        from robotforge.design import props

        m, d = self.session.model, self.session.data
        levels = props.shelf_levels(entry.size)
        want = len(levels) if slot is None else int(slot) if slot.isdigit() else 0
        if not 1 <= want <= len(levels):
            raise BehaviorError(f"{name} has boards 1 to {len(levels)}, counted from the bottom")
        if want != len(levels):
            raise BehaviorError(
                f"board {want} of {name} has another board over it: a gripper coming from above cannot get a thing in; "
                f"the top board ({len(levels)}) can take it"
            )
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        base = d.xpos[b].copy() if b >= 0 else d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)].copy()
        rot = d.xmat[b].reshape(3, 3) if b >= 0 else np.eye(3)
        top = float(base[2] + levels[-1])
        half = np.abs(rot[:2, :2]) @ np.array([entry.size[0] / 2 - 0.04, entry.size[1] / 2 - 0.04])
        spot = np.array([base[0], base[1], top])
        region = {"center": spot.tolist(), "half_xy": [float(half[0]), float(half[1])], "min_z": top - 0.012}
        return spot, top, region

    def _into(self, name: str) -> tuple[np.ndarray, float, dict[str, Any]]:
        """Place point, rim and success region for putting something in drawer `name`, as far open as it is now:
        only the part pulled out of the chest can take it."""
        m, d = self.session.model, self.session.data
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}/inside")
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/drawer")
        if site < 0 or joint < 0:
            raise BehaviorError(f"{name} has nothing to put things in")
        opened = float(d.qpos[m.jnt_qposadr[joint]])
        if opened < OPEN_ENOUGH:
            raise BehaviorError(
                f"{name} is closed ({opened:.2f} m open): open it first with operate on {name}, then put things in it"
            )
        # In the drawer's own frame its front panel is at y = 0 and the chest's front at y = opened: things go on
        # the floor between the two, the part pulled out, coming down over the drawer's front, its highest edge.
        # Clearing the chest's top instead, the kitchen's arm could not carry the plate that high.
        drawer = int(m.site_bodyid[site])
        rot = d.xmat[drawer].reshape(3, 3)
        center = d.xpos[drawer] + rot @ np.array([0.0, opened / 2, float(m.site_pos[site][2])])
        half_out = max(opened - mechanisms.PANEL, 0.0) / 2
        across = np.abs(rot[:2, :2]) @ np.array([float(m.site_size[site][0]), half_out])
        front = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{name}/drawer_front")
        top = float(d.geom_xpos[front][2] + m.geom_size[front][2]) if front >= 0 else float(center[2])
        region = {"center": center.tolist(), "half_xy": across.tolist(), "min_z": float(center[2]) - 0.01}
        return center, top, region

    # ---- status --------------------------------------------------------------------------

    def status(self) -> list[dict[str, Any]]:
        done = [
            {"behavior": "navigate", "navigating": robot, "state": "done", "last_trip": result}
            for robot, result in self._last_trips.items()
        ]
        shown = [c for c in self.session.controllers.values() if not getattr(c, "hidden", False)]
        return [c.status() for c in shown] + done


def _told_kind(entry: Any) -> str | None:
    """The kind a world entry declares: its template, or for a library robot what the catalog says
    (G1's joints are not named like our template's, so its shape alone does not say humanoid)."""
    return entry.template or (library.get(entry.library).kind if entry.library else None)


def _entry(robot: str, world: World | None) -> RobotEntry | None:
    return next((r for r in world.robots if r.name == robot), None) if world else None


def _tuned_humanoid(entry: RobotEntry | None) -> bool:
    """A humanoid whose walking gait was tuned for it: our template's and the library's (G1 learned, T1 and OP3 on
    the balance gait). Files, catalogue robots and robots built from parts are not: they stand by whole-body balance."""
    return entry is None or not (entry.file or entry.catalogue or entry.build)


def _stands_by_balance(entry: RobotEntry | None) -> bool:
    """A humanoid that whole-body balance keeps up while nothing moves it: every one without a gait tuned for it,
    and the library's learned walkers (G1), whose policy takes a push but falls under a load of a fifth of its own
    mass. Not one with something mounted on it: the mounted arm's controller and the balance would both drive its
    joints. The first move hands a tuned one over to its walk; for the others move refuses (see
    _mobile_controller)."""
    if not _tuned_humanoid(entry):
        return True
    return _learned_humanoid(entry) and not entry.mounts


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
