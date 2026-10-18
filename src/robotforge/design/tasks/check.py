"""Is a task done? The success check, the distance still to go and the reward, from where things are.

`judge(task, world, state)` reads only positions and mechanism readings (`State`), so the same rules judge the live
simulation (`live_state`), every step of a training environment, and the positions an agent's tool calls reported
(scripts/agent_eval.py). Where a place is comes from world.yaml (a table's top, a bin's walls, a room's floor
marker); a thing that moves (a bowl, a tote) is taken where the state has it.

The checks, by kind:

- go_to: the robot's middle within `radius` of the target's footprint (inside the room, for a room).
- bring, put, push: the thing's middle over the target's footprint and at its height: inside a bin or a drawer
  (between its floor and just over its rim), on a top (from the top to 35 cm over it), anywhere in a room, or on a
  floor mark.
- pick_up: the thing 5 cm or more higher than it started.
- open, close: a drawer out 60% of its travel or more (in to 1 cm), a door open 45 degrees or more (to 5).
- press: the button pressed at least once since the attempt began. switch_on: the switch on.

The distance for the reward adds the gap still to close (the thing to its place, the robot to its target, a drawer to
open) and half the robot's (or its hand's) distance to the thing it has to take. reward = -distance, plus
SUCCESS_REWARD while the task is done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from robotforge.design import mechanisms as mechanism_design
from robotforge.design import props
from robotforge.design.generate.space import Footprint
from robotforge.design.tasks.spec import TaskEntry

SUCCESS_REWARD = 2.0
LIFTED = 0.05  # m above its start that counts as picked up
RIM = 0.03  # m over a container's rim that still counts as in it
ON_TOP = 0.35  # m over a top a thing's middle may be and count as on it
NEAR = 0.25  # m: next to a small thing
OPEN_SHARE = 0.6  # of a drawer's travel
DOOR_OPEN, DOOR_CLOSED, DRAWER_CLOSED = 45.0, 5.0, 0.01
HAND_WEIGHT = 0.5
FALLEN = 0.5  # m added to the distance of a robot that has fallen over
PATCH = 1.0  # m: a terrain patch counts as reached within this of its middle
TOPS = ("table", "counter", "bench", "pallet", "box", "cabinet", "drawer", "conveyor")


@dataclass
class State:
    """Where things are: the robot's middle (and whether it is upright), its hand, each thing's middle, and each
    mechanism's reading as observe gives it ({"open_m": ...}, {"angle_deg": ...}, {"presses": ...}, {"on": ...})."""

    robot: tuple[float, float, float] | None = None
    upright: bool | None = None
    hand: tuple[float, float, float] | None = None
    things: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    mechanisms: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Verdict:
    done: bool
    distance: float  # m (or its share, for a mechanism), still to go
    reward: float
    why: str  # one line: what is still missing, or what made it done
    known: bool = True  # False: the state lacked what the check needs (the thing was never seen)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "done": self.done,
            "distance_m": round(self.distance, 3),
            "reward": round(self.reward, 3),
            "why": self.why,
        }
        if not self.known:
            out["known"] = False
        return out


@dataclass(frozen=True)
class Region:
    """Where a thing has to be: a footprint seen from above and a height range for its middle."""

    area: Footprint
    low: float
    high: float
    what: str  # container, top, room, mark, near, shelf


def _entry(world: Any, name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None
    for o in _objects(world):
        if o["name"] == name:
            return o
    return None


def _objects(world: Any) -> list[dict[str, Any]]:
    objects = world["objects"] if isinstance(world, dict) else world.objects
    return [o if isinstance(o, dict) else o.model_dump(mode="json") for o in objects]


def _patch(world: Any, name: str) -> dict[str, Any] | None:
    patches = world.get("terrain", []) if isinstance(world, dict) else world.terrain
    for t in patches:
        t = t if isinstance(t, dict) else t.model_dump(mode="json")
        if t.get("name") == name:
            return t
    return None


def _patch_area(patch: dict[str, Any]) -> Footprint:
    """What a terrain patch covers seen from above (a ramp with its landing, a flight of stairs, hills)."""
    from robotforge.design import terrain

    try:
        x0, x1, y0, y1 = terrain._world_bounds(terrain.parse(dict(patch)))
    except (ValueError, KeyError):
        return Footprint(patch["pos"][0], patch["pos"][1], PATCH, PATCH)
    return Footprint((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2)


def _base(o: dict[str, Any]) -> float:
    """The height of what an object stands on: its z when given (props and tables are placed by their base), else
    the floor."""
    return float(o["pos"][2]) if len(o["pos"]) == 3 else 0.0


def region(world: Any, name: str, state: State | None = None) -> Region | None:
    """Where something must be to be in or on `name`. None when the world has nothing called that."""
    o = _entry(world, name)
    if o is None:
        patch = _patch(world, name)
        if patch is None:
            return None
        # a terrain patch (a loading dock, a ramp): over it
        return Region(_patch_area(patch), -math.inf, math.inf, "mark")
    shape, size = o["shape"], o["size"]
    pos = list(o["pos"])
    if state is not None and name in state.things:  # it moves (a bowl, a tote): where it is now
        pos = list(state.things[name])
    hx, hy = props.footprint(shape, size)
    area = Footprint(pos[0], pos[1], hx, hy, float(o.get("yaw") or 0.0))
    if shape == "marker":
        return Region(area, -math.inf, math.inf, "mark")
    base = _base(o)
    if shape == "bin":
        if state is not None and name in state.things:
            base = pos[2]  # a loose bin's body is at its base
        wall = 0.005
        inner = Footprint(area.x, area.y, max(hx - wall, 0.005), max(hy - wall, 0.005), area.yaw)
        return Region(inner, base, base + size[2] + RIM, "container")
    if shape == "drawer":
        travel = 0.9 * (size[1] - 3 * mechanism_design.PANEL)
        lx, ly = area.world(0.0, -travel / 2)
        return Region(Footprint(lx, ly, hx, hy + travel / 2, area.yaw), base, base + size[2], "container")
    if shape == "shelf":
        return Region(area, base, base + size[2], "shelf")
    if shape == "box":  # a fixed box (a bed, a crate): its middle is at its z
        top = (pos[2] if len(pos) == 3 else size[2] / 2) + size[2] / 2
        return Region(area, top - 0.02, top + ON_TOP, "top")
    if shape == "pallet":
        return Region(area, base + size[2] - 0.02, base + size[2] + ON_TOP, "top")
    if shape in TOPS:
        top = base + size[2]
        return Region(area, top - 0.02, top + ON_TOP, "top")
    # a small thing: next to it
    here = state.things.get(name) if state is not None else None
    x, y = (here[0], here[1]) if here else (pos[0], pos[1])
    return Region(Footprint(x, y, NEAR, NEAR), -math.inf, math.inf, "near")


def _miss(region: Region, p: tuple[float, float, float]) -> float:
    """How far a point is from being in the region (0 inside)."""
    flat = region.area.gap_to(p[0], p[1])
    up = max(region.low - p[2], 0.0, p[2] - region.high)
    return math.hypot(flat, up)


def _from(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.dist(a[:2], b[:2])


def judge(task: TaskEntry, world: Any, state: State, start: State | None = None) -> Verdict:
    """The verdict on `task` in `state`. `start` is the state the attempt began in (pick_up and press compare with
    it; without it, the task's own start)."""
    kind = task.kind
    if kind == "go_to":
        return _go_to(task, world, state)
    if kind in ("bring", "put", "push"):
        return _put(task, world, state)
    if kind == "pick_up":
        return _pick_up(task, state, start)
    return _mechanism(task, world, state, start)


def _verdict(done: bool, distance: float, why: str, known: bool = True) -> Verdict:
    reward = -distance + (SUCCESS_REWARD if done else 0.0)
    return Verdict(done, distance, reward, why, known)


def _go_to(task: TaskEntry, world: Any, state: State) -> Verdict:
    if state.robot is None:
        return _verdict(False, 0.0, f"{task.robot} was not seen", known=False)
    target = _entry(world, task.target)
    reg = region(world, task.target, state)
    if reg is None:
        return _verdict(False, 0.0, f"there is no {task.target} in the world", known=False)
    if target is not None and target["shape"] == "marker" and len(target["size"]) == 2 and _is_room(target):
        inside = Footprint(reg.area.x, reg.area.y, max(reg.area.hx - 0.15, 0.05), max(reg.area.hy - 0.15, 0.05))
        gap = inside.gap_to(state.robot[0], state.robot[1])
        done = gap == 0.0
    else:
        gap = reg.area.gap_to(state.robot[0], state.robot[1])
        done = gap <= task.radius
    if done and state.upright is False:
        return _verdict(False, gap + FALLEN, f"{task.robot} got to {task.target} but is not upright")
    why = f"{task.robot} is at {task.target}" if done else f"{task.robot} is {gap:.2f} m from {task.target}"
    return _verdict(done, 0.0 if done else gap, why)


def _is_room(o: dict[str, Any]) -> bool:
    """A room's floor marker covers a room (a generated home names them after their rooms); a goal mark is small."""
    return min(o["size"]) > 1.0


def _put(task: TaskEntry, world: Any, state: State) -> Verdict:
    thing = state.things.get(task.object)
    if thing is None:
        return _verdict(False, 0.0, f"{task.object} was not seen", known=False)
    reg = region(world, task.target, state)
    if reg is None:
        return _verdict(False, 0.0, f"there is no {task.target} in the world", known=False)
    target = _entry(world, task.target)
    if reg.what == "mark" and target is not None and _is_room(target):
        reg = Region(reg.area, -math.inf, math.inf, "room")
    miss = _miss(reg, thing)
    done = miss == 0.0
    taker = state.hand or state.robot
    distance = miss + (0.0 if done else HAND_WEIGHT * _from(taker, thing))
    if done:
        why = f"{task.object} is {'in' if reg.what in ('container', 'room', 'shelf') else 'on'} {task.target}"
    else:
        why = f"{task.object} is {miss:.2f} m from {task.target}"
    return _verdict(done, distance, why)


def _pick_up(task: TaskEntry, state: State, start: State | None) -> Verdict:
    thing = state.things.get(task.object)
    began = start.things.get(task.object) if start is not None else None
    if began is None:
        saved = task.start.objects.get(task.object)
        began = tuple(saved) if saved is not None and len(saved) == 3 else None
    if thing is None or began is None:
        return _verdict(False, 0.0, f"{task.object} was not seen", known=False)
    lift = thing[2] - began[2]
    done = lift >= LIFTED
    taker = state.hand or state.robot
    distance = max(LIFTED - lift, 0.0) + (0.0 if done else HAND_WEIGHT * _from(taker, thing))
    why = f"{task.object} is up {lift * 100:.0f} cm" if done else f"{task.object} is {lift * 100:.0f} cm up of 5"
    return _verdict(done, distance, why)


def _mechanism(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    reading = state.mechanisms.get(task.object)
    o = _entry(world, task.object)
    if reading is None or o is None:
        return _verdict(False, 0.0, f"{task.object} was not seen", known=False)
    taker = state.hand or state.robot
    near = HAND_WEIGHT * _from(taker, tuple(o["pos"][:2]) + (0.0,)) if taker is not None else 0.0
    shape = o["shape"]
    if task.kind in ("open", "close") and shape == "drawer":
        travel = float(reading.get("of_m") or 0.9 * (o["size"][1] - 3 * mechanism_design.PANEL))
        out = float(reading.get("open_m") or 0.0)
        if task.kind == "open":
            done, left = out >= OPEN_SHARE * travel, max(OPEN_SHARE * travel - out, 0.0)
        else:
            done, left = out <= DRAWER_CLOSED, max(out - DRAWER_CLOSED, 0.0)
        why = f"{task.object} is out {out * 100:.0f} cm of {travel * 100:.0f}"
        return _verdict(done, left + (0.0 if done else near), why)
    if task.kind in ("open", "close"):
        angle = float(reading.get("angle_deg") or 0.0)
        if task.kind == "open":
            done, left = angle >= DOOR_OPEN, max(DOOR_OPEN - angle, 0.0) / 90.0
        else:
            done, left = angle <= DOOR_CLOSED, max(angle - DOOR_CLOSED, 0.0) / 90.0
        return _verdict(done, left + (0.0 if done else near), f"{task.object} is open {angle:.0f} degrees")
    if task.kind == "press":
        presses = int(reading.get("presses") or 0)
        before = int((start.mechanisms.get(task.object) or {}).get("presses") or 0) if start is not None else 0
        done = presses > before or bool(reading.get("pressed"))
        return _verdict(done, 0.0 if done else 1.0 + near, f"{task.object} pressed {presses - before} times")
    if task.kind == "switch_on":
        done = bool(reading.get("on"))
        return _verdict(done, 0.0 if done else 1.0 + near, f"{task.object} is {'on' if done else 'off'}")
    return _verdict(False, 0.0, f"cannot judge {task.kind} on a {shape}", known=False)
