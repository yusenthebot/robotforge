"""From a task sentence to a task in a world: find the thing and the place it names, choose the robot, and add what is
missing (a thing to fetch, a bin, a robot that can do it) by the same rules generated worlds are furnished by.

A thing is found by its name ("red_cup"), else by what it is, its color and the room it is in ("the red cup in the
kitchen"): the first that fits, in the world's order, and the answer says which when several fit. A place is an
object (a table, a bin, a drawer), a room (its floor marker) or a mark ("the goal"). The robot is the one named, else
the first in the world that can do the task: one that walks or drives for go_to and push, one with an arm on a base
that goes for bring (a lift picker is added when there is none), an arm that reaches for put and pick_up, an arm next
to the mechanism for open, close, press and switch_on (a template arm is put on the mechanism's top when there is
none).
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from robotforge import wording
from robotforge.design import props
from robotforge.design.generate import things as making
from robotforge.design.generate import words
from robotforge.design.generate.draft import Draft, Item
from robotforge.design.generate.space import Footprint, Surface
from robotforge.design.scenes import Content
from robotforge.design.tasks import read as reading
from robotforge.design.tasks.spec import DEFAULT_SECONDS, Randomize, Ref, Start, TaskEntry

# what an object's name says it is, by concept (a generated world names things this way; a hand-made one may too)
NAME_KEYS: dict[str, tuple[str, ...]] = {
    "table": ("table",),
    "desk": ("desk",),
    "shelf": ("shelf", "rack"),
    "rack": ("rack",),
    "counter": ("counter", "sideboard", "vanity"),
    "bin": ("bin", "tote", "basket", "tray", "bowl"),
    "bowl": ("bowl",),
    "drawer": ("drawer", "nightstand"),
    "box": ("box", "parcel", "crate"),
    "cube": ("cube", "block"),
    "can": ("can", "bottle", "jar"),
    "pillar": ("pillar", "column"),
}
# what shape a concept is, for things a hand-made world named otherwise ("block_2" is a box)
SHAPES: dict[str, tuple[str, ...]] = {
    "cube": ("box",), "box": ("box",), "ball": ("sphere",), "apple": ("sphere",), "cup": ("cylinder",),
    "can": ("cylinder",), "plate": ("cylinder",), "bin": ("bin",), "bowl": ("bin",), "table": ("table",),
    "desk": ("table",), "shelf": ("shelf",), "rack": ("shelf",), "counter": ("counter",), "drawer": ("drawer",),
    "cabinet": ("cabinet",), "button": ("button",), "switch": ("switch",), "lever": ("lever",), "valve": ("valve",),
    "bench": ("bench",), "tree": ("tree",), "rock": ("rock",), "pallet": ("pallet",), "conveyor": ("conveyor",),
    "pillar": ("pillar",),
}  # fmt: skip
PARTS = ("_back", "_mattress", "_pillow", "_arm_1", "_arm_2")  # pieces of furniture that are not the piece
MOBILE = ("quadruped", "hexapod", "humanoid", "rover", "balancer", "snake")
NEEDS_ARM = ("bring", "put", "pick_up", "open", "close", "press", "switch_on")
MECHANISM_KINDS = ("open", "close", "press", "switch_on")
ARM_REACH = 0.55  # m from a fixed arm's base: what it reaches on its own top (the template arm's comfortable reach)
ARM_NEAR = (0.28, 0.42)  # m from a mechanism, where an arm is put to work it


@dataclass
class RobotInfo:
    name: str
    kind: str  # behaviors.Kind value
    arms: list[str]
    pos: tuple[float, float]


@dataclass
class Plan:
    task: TaskEntry
    add_objects: list[dict[str, Any]] = field(default_factory=list)
    add_robot: dict[str, Any] | None = None  # a RobotEntry's fields
    arm_spots: list[dict[str, Any]] = field(default_factory=list)  # places to try for an added arm, best first
    notes: list[str] = field(default_factory=list)


class TaskProblem(ValueError):
    """The sentence names something the world does not have, or nothing can do it: the message says what to do."""


def plan(
    world: Any,
    sentence: str,
    robots: list[RobotInfo],
    robot: str | None = None,
    name: str | None = None,
    seed: int = 0,
    spawn_height: Any = None,
) -> Plan:
    """The task `sentence` asks for in `world`, and what has to be added for it. `spawn_height(entry)` is how high a
    robot's base sits over what it stands on (to put an arm on a top)."""
    asked = reading.read(sentence)
    rng = random.Random(f"{seed}:{sentence.strip().lower()}")
    objects = [o.model_dump(mode="json") for o in world.objects]
    notes: list[str] = []
    out = Plan(task=None, notes=notes)  # type: ignore[arg-type]
    kind = asked.kind
    thing = place = None
    if kind in ("go_to", "bring", "put", "push"):
        if asked.place is None:
            raise TaskProblem(f"say where: {kind.replace('_', ' ')} to what? ({sentence!r})")
        place = _find_place(objects, world, asked.place, notes)
        if place is None and asked.place.concept in ("bin", "bowl"):
            place = _add_bin(world, objects, asked.place, rng, out, robots)
        if place is None:
            raise TaskProblem(_missing(asked.place, objects, world))
    if kind != "go_to":
        if asked.thing is None or asked.thing.concept in (None, "room", "goal"):
            raise TaskProblem(f"say what to {kind.replace('_', ' ')}: the sentence names no thing ({sentence!r})")
        # of the things that fit, one not where it has to go already (a can generated on the shelf it is to go to)
        there = _already(objects, place) if place is not None else None
        thing = _find(objects, world, asked.thing, notes, there)
        if thing is None:
            thing = _add_thing(world, objects, asked.thing, robots, rng, out)
        elif there is not None and there(next(o for o in objects if o["name"] == thing)):
            notes.append(f"{thing} is in or on {place} already: the task is done before it starts")
    chosen = _robot(kind, asked.robot, robot, robots, objects, thing, place, world, out, spawn_height)
    entry = _robot_entry(world, chosen)
    start = Start(
        robot=[*map(float, entry.pos[:2]), float(entry.yaw)] if entry is not None else None,
        objects={thing: _pos_of(objects, out.add_objects, thing)} if thing and kind not in MECHANISM_KINDS else {},
    )
    if start.robot is None and out.add_robot is not None:
        start.robot = [*map(float, out.add_robot["pos"][:2]), float(out.add_robot.get("yaw", 0.0))]
    fixed = _fixed(chosen, robots, out)
    task = TaskEntry(
        name=name or _task_name(kind, thing, place, world),
        sentence=sentence.strip(),
        kind=kind,
        robot=chosen,
        object=thing,
        target=place,
        seconds=DEFAULT_SECONDS[kind],
        start=start,
        randomize=Randomize(
            robot_xy=0.0 if fixed else 0.3,
            robot_yaw=0.0 if fixed else 30.0,
            object_xy=0.0 if kind in MECHANISM_KINDS or kind == "go_to" else 0.04,
        ),
    )
    out.task = task
    return out


# ---- finding things -------------------------------------------------------------------------------------------


def _rooms(objects: list[dict[str, Any]]) -> dict[str, Footprint]:
    """Room floors: markers over a metre each way (a generated home names them after the rooms)."""
    out = {}
    for o in objects:
        if o["shape"] == "marker" and len(o["size"]) == 2 and min(o["size"]) > 1.0:
            hx, hy = o["size"][0] / 2, o["size"][1] / 2
            out[o["name"]] = Footprint(o["pos"][0], o["pos"][1], hx, hy, float(o.get("yaw") or 0.0))
    return out


def _room_of(objects: list[dict[str, Any]], room_kind: str) -> list[str]:
    """Room markers of a kind: "bedroom" is bedroom, bedroom_2, ..."""
    return [n for n in _rooms(objects) if re.sub(r"_\d+$", "", n) == room_kind]


def _in_room(o: dict[str, Any], objects: list[dict[str, Any]], room_kind: str | None) -> bool:
    if room_kind is None:
        return True
    rooms = _rooms(objects)
    return any(rooms[r].contains(o["pos"][0], o["pos"][1]) for r in _room_of(objects, room_kind))


def _is(o: dict[str, Any], concept: str, by_shape: bool = False) -> bool:
    """Is `o` a `concept`: by its name ("red_cup", "bedroom_desk"), or with `by_shape` by its shape (a loose ball
    for "ball", in a world that named it "toy")."""
    name = re.sub(r"_\d+$", "", o["name"])
    if o["name"].endswith(PARTS) or o["shape"] == "marker":
        return False
    if by_shape:  # only a thing whose name says nothing it is ("purple_book" is a book, not a cube)
        if any(_named(name, c) for c in words.THINGS):
            return False
        return o["shape"] in SHAPES.get(concept, ()) and concept not in ("table", "box") and not o.get("fixed", False)
    return _named(name, concept)


def _named(name: str, concept: str) -> bool:
    keys = NAME_KEYS.get(concept, (concept,))
    return any(k == name or name.endswith("_" + k) or name.startswith(k + "_") or f"_{k}_" in name for k in keys)


def _colored(o: dict[str, Any], color: str | None) -> bool:
    if color is None:
        return True
    if color in o["name"].split("_"):
        return True
    return isinstance(o.get("color"), str) and o["color"] == color


def _already(objects: list[dict[str, Any]], place: str) -> Any:
    """Whether a thing (an object's entry) is in or on `place` as the world stands."""
    from robotforge.design.tasks.check import _miss, region

    reg = region({"objects": objects, "terrain": []}, place)
    if reg is None:
        return lambda o: False
    return lambda o: _miss(reg, (o["pos"][0], o["pos"][1], o["pos"][2] if len(o["pos"]) == 3 else 0.0)) == 0.0


def _find(objects: list[dict[str, Any]], world: Any, ref: Ref, notes: list[str], done: Any = None) -> str | None:
    if ref.name and any(o["name"] == ref.name for o in objects):
        return ref.name
    exact = wording.match(ref.text.replace("the ", "").strip(), [o["name"] for o in objects])
    if exact:
        return exact
    found: list[dict[str, Any]] = []
    for by_shape in (False, True):  # what its name says first; its shape only when no name says it
        found = [o for o in objects if _is(o, ref.concept or "", by_shape) and _colored(o, ref.color)]
        found = [o for o in found if _in_room(o, objects, ref.room)]
        if found:
            break
    if not found:
        return None
    if done is not None:  # those not there yet first
        found.sort(key=lambda o: bool(done(o)))
    if len(found) > 1:
        notes.append(
            f"{len(found)} things fit {ref.text!r}: took {found[0]['name']} (others: "
            f"{', '.join(o['name'] for o in found[1:4])}); name one to choose"
        )
    return found[0]["name"]


def _find_place(objects: list[dict[str, Any]], world: Any, ref: Ref, notes: list[str]) -> str | None:
    if ref.concept == "room" and ref.room:
        rooms = _room_of(objects, ref.room)
        if rooms:
            if len(rooms) > 1:
                notes.append(f"{len(rooms)} rooms are a {ref.room.replace('_', ' ')}: took {rooms[0]}")
            return rooms[0]
        return None
    if ref.concept == "goal":
        marks = [o["name"] for o in objects if o["shape"] == "marker" and o["name"] in ("goal", "finish", "target")]
        marks = marks or [o["name"] for o in objects if o["shape"] == "marker" and o["name"] not in _rooms(objects)]
        return marks[0] if marks else None
    for t in world.terrain:  # a terrain patch said by its name ("the loading dock") before anything it is near
        if re.search(rf"\b{re.escape(t.name)}\b", ref.text):
            return t.name
    if ref.concept is not None:
        # a kind of thing said and none found: no place (not a room whose name the words happen to contain)
        return _find(objects, world, ref, notes)
    for n in [o["name"] for o in objects]:  # a name said as it is ("the dock")
        if re.search(rf"\b{re.escape(n)}\b", ref.text):
            return n
    return None


def _missing(ref: Ref, objects: list[dict[str, Any]], world: Any) -> str:
    what = ref.text.strip() or "that place"
    rooms = list(_rooms(objects))
    hint = f" Rooms: {', '.join(rooms)}." if rooms else ""
    places = [o["name"] for o in objects if o["shape"] in ("table", "counter", "bin", "shelf", "drawer", "marker")
              and o["name"] not in rooms][:8]  # fmt: skip
    listing = f" Places: {', '.join(places)}." if places else ""
    return f"no {what!r} in this world.{hint}{listing} Name one of them, or add it with add_object first"


def _pos_of(objects: list[dict[str, Any]], added: list[dict[str, Any]], name: str) -> list[float]:
    for o in objects + added:
        if o["name"] == name:
            return [float(v) for v in o["pos"]]
    return []


# ---- adding what is missing -------------------------------------------------------------------------------------


def surfaces_of(objects: list[dict[str, Any]]) -> list[Surface]:
    """The tops of a world's furniture as surfaces to put things on, with what stands on them already."""
    tops: list[Surface] = []
    rooms = _rooms(objects)
    for o in objects:
        shape, size = o["shape"], o["size"]
        if shape not in ("table", "counter", "drawer", "cabinet", "shelf") or o["name"].endswith(PARTS):
            continue
        hx, hy = props.footprint(shape, size)
        fp = Footprint(o["pos"][0], o["pos"][1], hx, hy, float(o.get("yaw") or 0.0))
        base = float(o["pos"][2]) if len(o["pos"]) == 3 else 0.0
        room = next((r for r, area in rooms.items() if area.contains(fp.x, fp.y)), None)
        if shape == "shelf":
            levels = props.shelf_levels(size)
            for lo, hi in zip(levels, levels[1:], strict=False):
                area = Footprint(fp.x, fp.y, hx - 0.05, hy - 0.05, fp.yaw)
                tops.append(Surface(o["name"], area, base + lo, hi - lo - props.PANEL, room, kind="shelf"))
            continue
        area = Footprint(fp.x, fp.y, max(hx - 0.04, 0.01), max(hy - 0.04, 0.01), fp.yaw)
        tops.append(Surface(o["name"], area, base + size[2], 2.0, room, kind="table" if shape != "drawer" else "chest"))
    for o in objects:  # what already stands on each top
        if len(o["pos"]) != 3 or o["shape"] == "marker":
            continue
        hx, hy = props.footprint(o["shape"], o["size"])
        for s in tops:
            if s.area.grown(0.05).contains(o["pos"][0], o["pos"][1]) and abs(o["pos"][2] - s.z) < 0.3:
                s.items.append(Footprint(o["pos"][0], o["pos"][1], hx, hy, float(o.get("yaw") or 0.0)))
    return tops


def _draft(world: Any, objects: list[dict[str, Any]], rng: random.Random, start: tuple[float, float]) -> Draft:
    xs = [o["pos"][0] for o in objects] + [start[0]]
    ys = [o["pos"][1] for o in objects] + [start[1]]
    bounds = (min(xs) - 3.0, max(xs) + 3.0, min(ys) - 3.0, max(ys) + 3.0)
    draft = Draft(rng, Content(floor={}, objects=[dict(o) for o in objects]), bounds)
    for o in objects:
        if o["shape"] == "marker" or len(o["pos"]) == 3 and o["shape"] in ("box", "sphere", "cylinder", "bin"):
            continue
        fixed = o.get("fixed", props.SHAPES[o["shape"]].fixed)
        if not fixed:
            continue
        hx, hy = props.footprint(o["shape"], o["size"])
        draft.obstacles[o["name"]] = Footprint(o["pos"][0], o["pos"][1], hx, hy, float(o.get("yaw") or 0.0))
    draft.surfaces = surfaces_of(objects)
    draft.map_floor(start)
    return draft


def _mover(robots: list[RobotInfo]) -> tuple[float, float]:
    for r in robots:
        if r.kind != "arm":
            return r.pos
    return robots[0].pos if robots else (0.0, 0.0)


def _add_thing(world: Any, objects: list[dict[str, Any]], ref: Ref, robots: list[RobotInfo], rng: random.Random,
               out: Plan) -> str:  # fmt: skip
    if ref.concept not in words.ITEMS:
        raise TaskProblem(_missing(ref, objects, world))
    item = making.make(ref.concept, rng, ref.color)
    draft = _draft(world, objects, rng, _mover(robots))
    tops = [s for s in draft.surfaces if ref.room is None or (s.room and re.sub(r"_\d+$", "", s.room) == ref.room)]
    if not tops and not draft.surfaces:  # nothing to put it on: on the floor in front of the robot
        x, y = _mover(robots)
        tops = [Surface("floor", Footprint(x + 0.8, y, 0.3, 0.3), 0.0, kind="ground")]
    placed = draft.place(item, tops or draft.surfaces)
    if placed is None:
        raise TaskProblem(f"no room for a {ref.concept} on any top; add one with add_object and name it")
    new = draft.content.objects[-1]
    out.add_objects.append(new)
    objects.append(new)
    out.notes.append(f"added {placed}: the world had no {ref.text.strip()!r}")
    return placed


def _add_bin(world: Any, objects: list[dict[str, Any]], ref: Ref, rng: random.Random, out: Plan,
             robots: list[RobotInfo]) -> str | None:  # fmt: skip
    draft = _draft(world, objects, rng, _mover(robots))
    item = Item("bin", "bin", (0.2, 0.15, 0.08), ref.color or "gray", 0.15, named="bin")
    tops = [s for s in draft.surfaces if ref.room is None or (s.room and re.sub(r"_\d+$", "", s.room) == ref.room)]
    placed = draft.place(item, tops or draft.surfaces)
    if placed is None:
        return None
    new = draft.content.objects[-1]
    out.add_objects.append(new)
    objects.append(new)
    out.notes.append(f"added {placed}: the world had no {ref.text.strip()!r}")
    return placed


# ---- the robot ----------------------------------------------------------------------------------------------


def _robot_entry(world: Any, name: str) -> Any:
    return next((r for r in world.robots if r.name == name), None)


def _fixed(name: str, robots: list[RobotInfo], out: Plan) -> bool:
    info = next((r for r in robots if r.name == name), None)
    if info is not None:
        return info.kind == "arm"
    return out.add_robot is not None and out.add_robot.get("template") == "arm"


WANTED = {"dog": ("quadruped",), "go2": ("quadruped",), "rover": ("rover",), "humanoid": ("humanoid",),
          "arm": ("arm",), "picker": MOBILE}  # fmt: skip


def _robot(kind, said, given, robots, objects, thing, place, world, out: Plan, spawn_height) -> str:
    names = [r.name for r in robots]
    if given is not None:
        found = wording.match(given, names)
        if found is None:
            raise TaskProblem(wording.unknown("robot", given, names, many="robots"))
        return found
    able = [r for r in robots if _can(kind, r, objects, thing, place, world)]
    if said is not None:
        able = [r for r in able if r.kind in WANTED.get(said, ()) and (said != "picker" or r.arms)]
    if able:
        return able[0].name
    if kind in MECHANISM_KINDS or (kind in ("put", "pick_up") and said == "arm"):
        return _add_arm(objects, thing, place, world, out, spawn_height)
    taken = set(names) | {o["name"] for o in objects}
    starter = {"dog": {"template": "quadruped"}, "go2": {"library": "go2"}, "rover": {"template": "rover"},
               "humanoid": {"template": "humanoid"}}.get(said or "")  # fmt: skip
    if kind in NEEDS_ARM or starter is None:
        if kind in ("go_to", "push") and starter is None:
            starter = {"template": "rover"} if _indoors(objects) else {"template": "quadruped"}
        else:
            from robotforge.design import gallery

            starter = {"build": gallery.build("lift_picker")}
    base = {"quadruped": "dog", "rover": "rover", "humanoid": "humanoid"}.get(starter.get("template", ""), "picker")
    if "library" in starter:
        base = starter["library"]
    name = base
    k = 1
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    out.add_robot = {"name": name, **starter}
    out.notes.append(f"added {name} to do it: no robot in the world could")
    return name


def _indoors(objects: list[dict[str, Any]]) -> bool:
    return bool(_rooms(objects)) or any(o["shape"] == "wall" for o in objects)


def _can(kind: str, r: RobotInfo, objects, thing, place, world) -> bool:
    if kind in ("go_to", "push"):
        return r.kind in MOBILE
    if kind == "bring":
        return r.kind in MOBILE and bool(r.arms)
    if kind in ("put", "pick_up"):
        if r.kind in MOBILE and r.arms:
            return True
        if r.kind != "arm":
            return False
        spots = [p for p in (_xy(objects, thing), _xy(objects, place)) if p is not None]
        return all(math.dist(r.pos, p) <= ARM_REACH + 0.15 for p in spots)
    # a mechanism: an arm within reach of it
    spot = _xy(objects, thing)
    return r.kind == "arm" and spot is not None and math.dist(r.pos, spot) <= ARM_REACH


def _xy(objects: list[dict[str, Any]], name: str | None) -> tuple[float, float] | None:
    for o in objects:
        if o["name"] == name:
            return (o["pos"][0], o["pos"][1])
    return None


def _add_arm(objects, thing, place, world, out: Plan, spawn_height) -> str:
    """A template arm on the top the mechanism stands on (or on the floor beside it), facing it from 28 to 42 cm."""
    from robotforge.design.world import RobotEntry

    o = next(o for o in objects if o["name"] == thing)
    base = float(o["pos"][2]) if len(o["pos"]) == 3 else 0.0
    taken = {r.name for r in world.robots} | {x["name"] for x in objects}
    name = "arm" if "arm" not in taken else next(f"arm_{k}" for k in range(2, 99) if f"arm_{k}" not in taken)
    under = [
        s
        for s in surfaces_of(objects)
        if s.owner != thing
        and s.kind in ("table", "chest")
        and s.z <= base + 0.01
        and s.area.grown(0.05).contains(o["pos"][0], o["pos"][1])
    ]
    top = max(under, key=lambda s: s.z) if under and base - max(s.z for s in under) < 0.2 else None  # fmt: skip
    hx, hy = props.footprint(o["shape"], o["size"])
    mech = Footprint(o["pos"][0], o["pos"][1], hx, hy, float(o.get("yaw") or 0.0))
    entry = RobotEntry(name=name, template="arm")
    lift = spawn_height(entry) if spawn_height else 0.0
    spots = []
    # in front of it first (where its drawer or door comes out, as on the kitchen scene's island), then beside it
    ahead = [(0.0, -d) for d in (0.5, 0.46, 0.54, 0.42)]
    beside = [(side * d, ly) for d in (0.26, 0.3, 0.22, 0.34) for side in (1, -1) for ly in (0.0, 0.06, -0.06)]
    for lx, ly in ahead + beside:
        x, y = mech.world(lx, ly)
        base_fp = Footprint(x, y, 0.08, 0.08)
        if base_fp.overlaps(mech, 0.03):
            continue
        if top is not None:
            if not top.area.contains(x, y, 0.06) or any(base_fp.overlaps(i, 0.03) for i in top.items):
                continue
            z = top.z
        else:
            z = 0.0
        pos = [round(x, 3), round(y, 3)] + ([round(z + lift, 4)] if z > 0 else [])
        # facing out over the top's front edge (its arm folded clear of the mechanism), else toward it
        toward = math.degrees(math.atan2(mech.y - y, mech.x - x))
        for yaw in (toward + 90.0, toward, mech.yaw - 90.0):
            spots.append({"name": name, "template": "arm", "pos": pos, "yaw": round((yaw + 180) % 360 - 180, 1)})
    if not spots:
        raise TaskProblem(f"no room for an arm beside {thing}: add one with add_robot(template='arm', pos=...) near it")
    out.add_robot = spots[0]
    out.arm_spots = spots
    out.notes.append(f"added {name} beside {thing} to work it")
    return name


def _task_name(kind: str, thing: str | None, place: str | None, world: Any) -> str:
    base = f"{kind}_{thing or place}"[:28]
    base = re.sub(r"[^a-z0-9_]", "_", base.lower())
    taken = {t.name for t in getattr(world, "tasks", [])}
    if base not in taken:
        return base
    k = 2
    while f"{base}_{k}" in taken:
        k += 1
    return f"{base}_{k}"


__all__ = ["Plan", "RobotInfo", "TaskProblem", "plan", "surfaces_of"]
