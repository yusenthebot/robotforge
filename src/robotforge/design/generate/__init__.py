"""One sentence to a world: rooms with furniture, a warehouse, an outdoor place with terrain, placed by rules.

    load_scene(description="a small flat with a kitchen and a bedroom, three apples", seed=2)

`generate(description, seed)` reads the sentence (words.py), lays the place out (house.py, warehouse.py,
outdoor.py) and fills it by rules (draft.py): furniture on the floor without overlapping, against walls or with room
to walk round it, doorways kept clear; small things on tops 0.3 to 1.0 m high, inside the edges, apart, and within an
arm's reach of floor the robot can get to from its start. Then the physics checks it (`settle`): the world is
compiled and run for a second and a half with no robots, and anything that started inside something or moved (a ball
rolled off, a cup tipped) is put somewhere else, or left out with a note. The result is ordinary scene content
(scenes.Content), so it loads like a preset: robots go to its places, everything can be edited afterwards, and the
same sentence and seed give the same world.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotforge.design import gallery
from robotforge.design.generate import house, outdoor, warehouse, words
from robotforge.design.generate.draft import Draft
from robotforge.design.generate.space import Footprint
from robotforge.design.scenes import Content, Slot

SETTLE_SECONDS = 1.5
MOVED = 0.03  # m a thing may move while settling and still count as standing where it was put
TIPPED = math.radians(15)
STILL = 0.05  # m/s: faster than this at the end, it is still going
INSIDE = 0.004  # m of overlap at the start that counts as starting inside something
ROUNDS = 3  # times things that did not settle are put elsewhere
# The robot a world starts with, by what the sentence asked for; else by kind of place
STARTERS = {
    "picker": {"name": "picker", "design": "lift_picker"},
    "dog": {"name": "dog", "template": "quadruped"},
    "go2": {"name": "go2", "library": "go2"},
    "rover": {"name": "rover", "template": "rover"},
    "humanoid": {"name": "humanoid", "template": "humanoid"},
    "arm": {"name": "arm", "template": "arm"},
}
DEFAULT_STARTER = {"house": "picker", "warehouse": "picker", "outdoor": "dog"}


@dataclass
class Generated:
    content: Content
    wish: words.Wish
    rooms: dict[str, list[float]] = field(default_factory=dict)  # room name -> [x0, y0, x1, y1]
    notes: list[str] = field(default_factory=list)
    moved: int = 0  # things the settling check put elsewhere
    loose: list[str] = field(default_factory=list)  # the small things put on surfaces, to pick up

    def summary(self) -> dict:
        objects = self.content.objects
        loose = self.loose
        return {
            "understood": self.wish.summary(),
            **({"rooms": self.rooms} if self.rooms else {}),
            "objects": len(objects),
            "things_to_pick_up": loose[:20] + (["..."] if len(loose) > 20 else []),
            **({"terrain": [t["name"] for t in self.content.terrain]} if self.content.terrain else {}),
            "notes": self.notes,
        }


def generate(description: str, seed: int = 0, check: bool = True) -> Generated:
    """The world `description` asks for. check=False skips the physics check (tests of the layout alone)."""
    if not description.strip():
        raise ValueError("describe the place in a sentence, e.g. 'a kitchen with a table and two cups'")
    wish = words.read(description)
    rng = random.Random(f"{seed}:{description.strip().lower()}")
    draft = Draft(rng, Content(floor={"style": "checker"}), (-10.0, 10.0, -10.0, 10.0))
    {"house": house.build, "warehouse": warehouse.build, "outdoor": outdoor.build}[wish.kind](draft, wish)
    moved = settle(draft) if check else 0
    _arm_slots(draft)
    content = draft.content
    starter = dict(STARTERS[wish.robot or DEFAULT_STARTER[wish.kind]])
    design = starter.pop("design", None)
    if design is not None:
        starter["build"] = gallery.build(design)
    content.starter = [starter]
    content.tips = _tips(draft, wish)
    rooms = {
        n: [round(v, 2) for v in (r.bounds()[0], r.bounds()[2], r.bounds()[1], r.bounds()[3])]
        for n, r in draft.rooms.items()
    }
    return Generated(content, wish, rooms, list(dict.fromkeys(draft.notes)), moved, list(draft.items))


WORLD_NAMES = {"house": "home", "warehouse": "warehouse", "outdoor": "outdoors"}


def world_name(description: str) -> str:
    """The name of the world a description opens when none is open: what kind of place it is."""
    return WORLD_NAMES[words.read(description).kind]


def settle(draft: Draft) -> int:
    """Compile the world without robots, run it, and put elsewhere what started inside something or did not stay
    put; returns how many were moved. What still does not settle after a few rounds is left out, with a note."""
    moved = 0
    for round_ in range(ROUNDS + 1):
        bad = unsettled(draft.content, list(draft.items))
        if not bad:
            return moved
        for name in bad:
            item, surface = draft.items[name]
            draft.take_back(name)
            if round_ == ROUNDS:
                draft.notes.append(f"left out a {item.concept} that would not stay put")
                continue
            others = [s for s in draft.surfaces if s is not surface and s.room == (surface.room if surface else None)]
            if draft.place(item, others or draft.surfaces, name=name) is not None:
                moved += 1
            else:
                draft.notes.append(f"no steady place was left for a {item.concept}")
    return moved


def unsettled(content: Content, loose: list[str], seconds: float = SETTLE_SECONDS) -> list[str]:
    """The loose things of `content` that start inside something, or move, tip or keep going in `seconds` of
    physics with nothing touching them."""
    from robotforge.design.world import World, compile_world
    from robotforge.sim.placement import owner

    world = World.model_validate(
        {"name": "settling", "floor": content.floor, "terrain": content.terrain, "objects": content.objects}
    )
    model = compile_world(world).model
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    bodies = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in loose}
    bodies = {n: b for n, b in bodies.items() if b >= 0}
    bad: set[str] = set()
    for c in data.contact[: data.ncon]:
        if c.dist < -INSIDE:
            for g in (c.geom1, c.geom2):
                name = owner(model, int(g))
                if name in bodies:
                    bad.add(name)
    start = {n: (data.xpos[b].copy(), data.xquat[b].copy()) for n, b in bodies.items()}
    steps = int(round(seconds / model.opt.timestep))
    mujoco.mj_step(model, data, nstep=steps)
    for n, b in bodies.items():
        pos, quat = start[n]
        dq = abs(float(np.dot(quat, data.xquat[b])))
        turned = 2 * math.acos(min(1.0, dq))
        speed = float(np.linalg.norm(data.cvel[b][3:]))
        if np.linalg.norm(data.xpos[b] - pos) > MOVED or turned > TIPPED or speed > STILL:
            bad.add(n)
    return sorted(bad)


def _arm_slots(draft: Draft) -> None:
    """Places for an arm on the tops that have things on them: at the edge of the top facing the room, turned to
    face in, so an arm put there reaches what is on it."""
    slots = []
    for s in draft.surfaces:
        if s.kind not in ("table", "counter") or not (0.6 <= s.z <= 1.0) or not s.items:
            continue
        a = s.area
        # the long edge at the top's own -y is its front (a counter's, a table's toward the room's middle too)
        lx, ly = 0.0, -a.hy + 0.06
        x, y = a.world(lx, ly)
        yaw = (a.yaw + 90.0) % 360.0
        slots.append(Slot((round(x, 3), round(y, 3)), yaw if yaw <= 180 else yaw - 360, s.z))
    draft.content.arm_slots = slots[:4]


def _tips(draft: Draft, wish: words.Wish) -> list[str]:
    loose = [o["name"] for o in draft.content.objects if o["name"] in draft.items]
    tips = []
    if draft.rooms:
        tips.append("each room's floor is a marker named after it: move(robot, to='" + next(iter(draft.rooms)) + "')")
    if loose:
        tips.append(f"pick_and_place(robot, '{loose[0]}', <a table or a bin>) fetches it with a robot that has an arm")
    return tips


__all__ = ["Footprint", "Generated", "generate", "settle", "unsettled", "world_name"]
