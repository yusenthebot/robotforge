"""The harder tasks' success checks: things that have to end up a certain way, judged from where things are.

- upright: the thing stands on a surface of the target (a shelf's board, a table, a tray's floor), within `tilt`
  degrees (15 by default) of upright, and is at rest.
- stack: `items` are stacked bottom to top: each one stands on the top of the one under it, centred on it, level.
- insert: the thing stands in a socket of the target (a rack's, "rack/2" for the second): down to its floor, upright.
- sort: every thing in `items` is inside the tray `into` names for it.
- pour: `items` are the balls that were in the cup `object`; enough of them (three quarters) lie in the cup `target`.
- hand_over: the thing is held by the robot `target` (touched by it, lifted), and no longer by the robot that gave it.

All of it reads only positions, tilts, speeds and contacts (`State`), like the older kinds, so the same rules judge a
live simulation, a training environment and a planner's prediction. The geometry of a fixture (a rack's sockets, a
tray's inside) comes from its generator (`robotforge.objects`), turned and placed as world.yaml has it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robotforge import objects
from robotforge.design import props
from robotforge.design.generate.space import Footprint
from robotforge.design.tasks.check import SUCCESS_REWARD, State, Verdict, _entry, _from
from robotforge.design.tasks.spec import TaskEntry

REST_SPEED = 0.08  # m/s: slower than this a thing is at rest (a freshly put thing still settles)
UPRIGHT = 15.0  # degrees from upright a thing may stand at, unless the task says
LEVEL = 12.0  # degrees a thing in a stack may lean
LANDED = 0.02  # m a thing's base may be off the top it stands on
POURED = 0.75  # share of the balls that must end up in the target cup
HELD_LIFT = 0.04  # m a handed-over thing is above where it began
TUCK = 0.01  # m a socket's floor may be above the base of what stands in it


def _verdict(task: TaskEntry, done: bool, distance: float, why: str, known: bool = True) -> Verdict:
    return Verdict(
        done, 0.0 if done else distance, -(0.0 if done else distance) + (SUCCESS_REWARD if done else 0.0), why, known
    )


# ---- what is where ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Spot:
    """A place a thing stands: a rectangle seen from above, the height of its floor and (for a socket) its radius."""

    area: Footprint
    z: float
    name: str
    radius: float | None = None
    rim: float | None = None


def _base(world: Any, name: str, state: State) -> tuple[float, float, float] | None:
    """Where a thing's base is: the state's for a loose one, the world's for a fixed one."""
    o = _entry(world, name)
    if o is None:
        return None
    if name in state.things:
        x, y, z = state.things[name]
        return (x, y, z - _centre_height(o))
    pos = o["pos"]
    return (float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) == 3 else 0.0)


def _centre_height(o: dict[str, Any]) -> float:
    """How far above its base the body's origin of a loose thing is: nothing for a generated one (its origin is the
    middle of its base), half its height for a plain box, cylinder or ball."""
    shape, size = o["shape"], o["size"]
    if shape in objects.GENERATORS:
        return 0.0
    if shape == "box":
        return size[2] / 2
    if shape == "cylinder":
        return size[1] / 2
    if shape == "sphere":
        return size[0] / 2
    return 0.0


def _built(o: dict[str, Any]) -> Any:
    return objects.built_of(o) if o["shape"] in objects.GENERATORS else None


def top_of(o: dict[str, Any]) -> float:
    """The height of a thing's flat top over its base."""
    built = _built(o)
    if built is not None:
        return float(built.facts.get("top_z") or built.height)
    shape, size = o["shape"], o["size"]
    return {"box": size[2], "cylinder": size[1], "sphere": size[0]}.get(shape, size[-1])


def half_of(o: dict[str, Any]) -> tuple[float, float]:
    built = _built(o)
    if built is not None:
        return built.half
    return props.footprint(o["shape"], o["size"])


def surfaces(world: Any, name: str) -> list[Spot]:
    """The places a thing can stand on or in the object `name` (`name/slot` names one): a shelf's boards, a table's
    top, a generated fixture's slots."""
    base_name, _, slot = name.partition("/")
    o = _entry(world, base_name)
    if o is None:
        return []
    pos = o["pos"]
    x, y, z0 = float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) == 3 else 0.0
    yaw = float(o.get("yaw") or 0.0)
    built = _built(o)
    out: list[Spot] = []
    if built is not None and built.slots:
        fp = Footprint(x, y, 0, 0, yaw)
        for s in built.slots:
            if slot and s.name != slot:
                continue
            wx, wy = fp.world(s.pos[0], s.pos[1])
            half = (s.radius, s.radius) if s.radius is not None else s.half
            out.append(Spot(Footprint(wx, wy, half[0], half[1], yaw), z0 + s.pos[2], s.name, s.radius, z0 + s.rim))
        return out
    hx, hy = half_of(o)
    area = Footprint(x, y, hx, hy, yaw)
    shape, size = o["shape"], o["size"]
    if shape == "shelf":
        inner = Footprint(x, y, max(hx - props.PANEL, 0.01), max(hy - props.PANEL, 0.01), yaw)
        out = [Spot(inner, z0 + top, str(i + 1)) for i, top in enumerate(props.shelf_levels(size))]
        return [s for s in out if not slot or s.name == slot]
    if shape in ("table", "counter", "bench", "pallet", "drawer", "cabinet", "conveyor"):
        return [Spot(area, z0 + size[2], "top")]
    if built is not None:
        return [Spot(Footprint(x, y, hx * 0.85, hy * 0.85, yaw), z0 + top_of(o), "top")]
    if shape == "box":
        return [Spot(area, z0 + size[2], "top")]
    return []


def _tilt(state: State, name: str) -> float:
    return float(state.tilts.get(name, 0.0))


def _rest(state: State, name: str) -> float:
    return float(state.speeds.get(name, 0.0))


def _stands(state: State, name: str, limit: float) -> tuple[bool, str]:
    tilt, speed = _tilt(state, name), _rest(state, name)
    if tilt > limit:
        return False, f"{name} is tipped {tilt:.0f} degrees (at most {limit:.0f})"
    if speed > REST_SPEED:
        return False, f"{name} is still moving ({speed:.2f} m/s)"
    return True, ""


# ---- the kinds -------------------------------------------------------------------------------------------------


def judge(task: TaskEntry, world: Any, state: State, start: State | None = None) -> Verdict:
    return {
        "upright": _upright,
        "stack": _stack,
        "insert": _insert,
        "sort": _sort,
        "pour": _pour,
        "hand_over": _hand_over,
    }[task.kind](task, world, state, start)


def _lacks(task: TaskEntry, what: str) -> Verdict:
    return _verdict(task, False, 0.0, what, known=False)


def _upright(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    o = _entry(world, task.object)
    base = _base(world, task.object, state) if o is not None else None
    if base is None or task.object not in state.things:
        return _lacks(task, f"{task.object} was not seen")
    spots = surfaces(world, task.target or "")
    if not spots:
        return _lacks(task, f"there is nothing to stand on in {task.target}")
    best = min(spots, key=lambda s: s.area.gap_to(base[0], base[1]) + max(abs(base[2] - s.z) - LANDED, 0.0))
    gap = best.area.gap_to(base[0], base[1])
    rise = max(abs(base[2] - best.z) - LANDED, 0.0)
    limit = task.tilt or UPRIGHT
    ok, why = _stands(state, task.object, limit)
    done = gap == 0.0 and rise == 0.0 and ok
    taker = state.hand or state.robot
    miss = gap + rise + max(_tilt(state, task.object) - limit, 0.0) / 90.0 * 0.3
    distance = miss + (0.0 if done else 0.5 * _from(taker, state.things[task.object]))
    if done:
        text = f"{task.object} stands on {task.target}, {_tilt(state, task.object):.0f} degrees off upright"
    elif gap > 0 or rise > 0:
        text = f"{task.object} is {gap + rise:.2f} m from standing on {task.target}"
    else:
        text = why
    return _verdict(task, done, distance, text)


def _stack(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    items = task.items or []
    if len(items) < 2:
        return _lacks(task, "a stack needs two or more things")
    seen = [n for n in items if n in state.things or _entry(world, n) is not None]
    if len(seen) < len(items):
        return _lacks(task, f"{', '.join(n for n in items if n not in seen)} was not seen")
    misses, whys = 0.0, []
    for lower, upper in zip(items[:-1], items[1:], strict=True):
        lo, up = _entry(world, lower), _entry(world, upper)
        bl, bu = _base(world, lower, state), _base(world, upper, state)
        if lo is None or up is None or bl is None or bu is None:
            return _lacks(task, f"{upper} or {lower} is not in the world")
        top = bl[2] + top_of(lo)
        hx, hy = half_of(lo)
        slack = 0.85
        yaw = state.yaws.get(lower, float(lo.get("yaw") or 0.0))
        area = Footprint(bl[0], bl[1], hx * slack, hy * slack, yaw)
        gap = area.gap_to(bu[0], bu[1])
        rise = max(abs(bu[2] - top) - LANDED, 0.0)
        miss = gap + rise
        if miss == 0.0:
            ok, why = _stands(state, upper, LEVEL)
            if not ok:
                miss, why_ = 0.1, why
                whys.append(why_)
        else:
            whys.append(f"{upper} is {miss:.2f} m from standing on {lower}")
        misses += miss
    ok, why = _stands(state, items[0], LEVEL)
    done = misses == 0.0 and ok
    if not ok and misses == 0.0:
        misses += 0.1
        whys.append(why)
    text = f"{' on '.join(reversed(items))} is stacked" if done else (whys[0] if whys else "not stacked yet")
    return _verdict(task, done, misses, text)


def _insert(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    o = _entry(world, task.object)
    base = _base(world, task.object, state) if o is not None else None
    if base is None or task.object not in state.things:
        return _lacks(task, f"{task.object} was not seen")
    sockets = [s for s in surfaces(world, task.target or "") if s.radius is not None]
    if not sockets:
        return _lacks(task, f"{task.target} has no sockets")
    best = min(sockets, key=lambda s: math.hypot(base[0] - s.area.x, base[1] - s.area.y))
    off = math.hypot(base[0] - best.area.x, base[1] - best.area.y)
    inside = off <= (best.radius or 0.0) * 0.85
    depth = base[2] - best.z  # how far its base is over the socket's floor
    down = depth <= TUCK
    limit = task.tilt or 10.0
    ok, why = _stands(state, task.object, limit)
    done = inside and down and ok
    taker = state.hand or state.robot
    distance = off + max(depth - TUCK, 0.0) + max(_tilt(state, task.object) - limit, 0.0) / 90.0 * 0.3
    distance += 0.0 if done else 0.5 * _from(taker, state.things[task.object])
    if done:
        text = f"{task.object} stands in socket {best.name} of {task.target.split('/')[0]}"
    elif not inside:
        text = f"{task.object} is {off * 100:.0f} cm from the middle of the nearest socket"
    elif not down:
        text = f"{task.object} sits {depth * 100:.1f} cm over the bottom of the socket"
    else:
        text = why
    return _verdict(task, done, distance, text)


def _sort(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    into = task.into or {}
    if not into:
        return _lacks(task, "nothing says where each thing goes")
    wrong, miss, first = 0, 0.0, ""
    for item, tray in into.items():
        o = _entry(world, item)
        base = _base(world, item, state) if o is not None else None
        spots = surfaces(world, tray)
        if base is None or not spots or item not in state.things:
            return _lacks(task, f"{item} or {tray} was not seen")
        spot = spots[0]
        gap = spot.area.gap_to(base[0], base[1])
        low = spot.z - 0.01
        high = (spot.rim or spot.z + 0.05) + 0.04
        away = max(low - base[2], base[2] - high, 0.0)
        if gap > 0 or away > 0:
            wrong += 1
            miss += gap + away
            first = first or f"{item} is not in {tray}"
    done = wrong == 0
    text = (
        f"all {len(into)} things are in their trays" if done else f"{wrong} of {len(into)} not in their tray: {first}"
    )
    return _verdict(task, done, miss + 0.05 * wrong, text)


def _pour(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    balls = task.items or []
    spots = surfaces(world, task.target or "")
    target = _entry(world, task.target)
    if not balls or target is None:
        return _lacks(task, "there is nothing to pour into")
    built = _built(target)
    base = _base(world, task.target, state)
    if built is None or base is None or "mouth_r" not in built.facts:
        return _lacks(task, f"{task.target} is not something to pour into")
    cx, cy = base[0], base[1]
    reach = float(built.facts["mouth_r"])
    floor = base[2] + float(built.facts.get("floor_z", 0.0))
    rim = base[2] + float(built.facts["rim_z"])
    inside = 0
    for name in balls:
        p = state.things.get(name)
        if p is not None and math.hypot(p[0] - cx, p[1] - cy) <= reach and floor - 0.01 <= p[2] <= rim + 0.01:
            inside += 1
    need = max(1, math.ceil(POURED * len(balls)))
    done = inside >= need
    left = max(need - inside, 0)
    taker = state.hand or state.robot
    distance = 0.05 * left + (0.0 if done else 0.3 * _from(taker, (cx, cy, rim)))
    text = f"{inside} of {len(balls)} balls are in {task.target}" + ("" if done else f" ({need} needed)")
    del spots
    return _verdict(task, done, distance, text)


def _hand_over(task: TaskEntry, world: Any, state: State, start: State | None) -> Verdict:
    thing = state.things.get(task.object)
    if thing is None:
        return _lacks(task, f"{task.object} was not seen")
    began = (start.things.get(task.object) if start is not None else None) or tuple(
        task.start.objects.get(task.object) or ()
    )
    touching = state.touching.get(task.object, set())
    receiver, giver = task.target, task.robot
    lifted = (thing[2] - began[2]) if began else 0.0
    held = receiver in touching and giver not in touching
    done = held and lifted >= HELD_LIFT
    if done:
        text = f"{receiver} holds {task.object}, {giver} has let go"
    elif giver in touching and receiver in touching:
        text = f"{giver} and {receiver} both hold {task.object}: the hand-over is under way"
    elif receiver in touching:
        text = f"{receiver} touches {task.object} but it is not lifted ({lifted * 100:.0f} cm)"
    elif giver in touching:
        text = f"{giver} still holds {task.object}"
    else:
        text = f"nobody holds {task.object}"
    distance = 0.0 if done else 0.4 + (0.0 if giver in touching or receiver in touching else 0.3)
    return _verdict(task, done, distance, text)
