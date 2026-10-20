"""Many tasks at once: a suite of generated worlds, each with a task, for agent evaluations and for learning.

    from robotforge.design.tasks.suite import make_suite
    cases = make_suite(20, seed=0, out="suite")   # suite/worlds/<case>/world.yaml and suite/suite.json

Each case is a world generated from a sentence (homes with different rooms, warehouses, parks with terrain) and a task
sentence made from what is in it ("bring the red cup from the kitchen to the desk in the bedroom", "open the drawer
in the kitchen", "walk to the goal across the stairs"), added with add_task as a user's sentence would be, so the
world on disk has the task, its robot, its start and its randomization. The same count and seed give the same suite.
scripts/agent_eval.py --suite runs an agent through each one and judges it with the task's own check;
control/tasks/world_task.world_env opens any of them for reinforcement learning.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOM_SETS = (
    ("kitchen", "living room"),
    ("kitchen", "bedroom"),
    ("kitchen", "living room", "bedroom"),
    ("living room", "bedroom", "office"),
    ("kitchen", "dining room"),
    ("office",),
    ("kitchen",),
    ("kitchen", "living room", "bedroom", "office"),
)
SIZES = ("", "small ", "large ")
HOMES = ("a {size}flat with {rooms}", "a {size}house with {rooms}", "an apartment with {rooms}")
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")
ITEMS = ("cup", "ball", "cube", "book", "can")
OUTDOOR = (
    "a park with stairs, a ramp and a few trees",
    "a rocky hillside with stepping stones and boulders",
    "a forest trail with a ramp and stairs",
    "a garden with hills, benches and a picnic table",
    "a beach with a platform and stepping stones",
    "a meadow with stairs and a ramp for my robot dog",
)
WAREHOUSES = (
    "a warehouse with {n} rows of racks",
    "a small warehouse with {n} rows of shelves and pallets",
    "a large warehouse with {n} rows of racks, pallets and a conveyor",
)
ZH_ROOMS = {"kitchen": "厨房", "living room": "客厅", "bedroom": "卧室", "office": "书房", "dining room": "餐厅"}
ZH_ITEMS = {"cup": "杯子", "ball": "球", "cube": "方块", "book": "书", "can": "罐子", "apple": "苹果", "box": "盒子"}
ZH_COLORS = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "purple": "紫色", "orange": "橙色"}


@dataclass
class Case:
    name: str
    world: str  # the world's sentence
    world_seed: int
    task: str  # the task's sentence
    kind: str
    robot: str
    object: str | None
    target: str | None
    lang: str

    def prompt(self) -> str:
        """What a user would type for this case, with the world already on disk."""
        if self.lang == "zh":
            return f"打开我的 {self.name} 世界。{self.task}"
        return f"Open my world {self.name}. {self.task}"


def _rooms_text(rooms: tuple[str, ...]) -> str:
    named = [f"{'an' if r[0] in 'aeiou' else 'a'} {r}" for r in rooms]
    return named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]


def world_sentence(rng: random.Random, kind: str) -> str:
    if kind == "warehouse":
        return rng.choice(WAREHOUSES).format(n=rng.choice(("two", "three", "four")))
    if kind == "outdoor":
        return rng.choice(OUTDOOR)
    rooms = rng.choice(ROOM_SETS)
    text = rng.choice(HOMES).format(size=rng.choice(SIZES), rooms=_rooms_text(rooms))
    if rng.random() < 0.6:
        color, item = rng.choice(COLORS), rng.choice(ITEMS)
        text += f", {rng.choice(('two', 'three'))} {item}s and a {color} {item}"
    return text


def _said(name: str) -> str:
    """How a sentence says an object called `name`: red_cup -> "the red cup", bedroom_desk -> "the desk in the
    bedroom", living_room_coffee_table -> "the coffee table in the living room"."""
    for room in ("living_room", "dining_room", "kitchen", "bedroom", "office", "bathroom"):
        if name.startswith(room + "_"):
            rest = re.sub(r"_\d+$", "", name[len(room) + 1 :]).replace("_", " ")
            return f"the {rest} in the {room.replace('_', ' ')}"
    return "the " + re.sub(r"_\d+$", "", name).replace("_", " ")


def _said_zh(name: str) -> str:
    for room, zh in (("living_room", "客厅"), ("dining_room", "餐厅"), ("kitchen", "厨房"), ("bedroom", "卧室"),
                     ("office", "书房")):  # fmt: skip
        if name.startswith(room + "_"):
            rest = name[len(room) + 1 :]
            word = {"table": "桌子", "desk": "书桌", "coffee_table": "茶几", "counter": "台面", "shelf": "书架",
                    "nightstand": "床头柜", "drawer": "抽屉", "sideboard": "餐边柜", "tv_stand": "电视柜"}  # fmt: skip
            return f"{zh}的{word.get(re.sub(r'_[0-9]+$', '', rest), rest)}"
    parts = re.sub(r"_\d+$", "", name).split("_")
    if len(parts) == 2 and parts[0] in ZH_COLORS and parts[1] in ZH_ITEMS:
        return f"{ZH_COLORS[parts[0]]}的{ZH_ITEMS[parts[1]]}"
    if parts[-1] in ZH_ITEMS:
        return ZH_ITEMS[parts[-1]]
    return name


def task_sentence(rng: random.Random, kind: str, world: Any, loose: list[str], lang: str) -> str | None:
    """A task to ask in `world`, as a user would say it; None when the world has nothing for it."""
    objects = {o.name: o for o in world.objects}
    rooms = [n for n, o in objects.items() if o.shape == "marker" and len(o.size) == 2 and min(o.size) > 1.0]
    tops = [n for n, o in objects.items() if o.shape in ("table", "counter") and not n.endswith(("tv_stand",))]
    drawers = [n for n, o in objects.items() if o.shape == "drawer" and len(o.pos) == 3]
    room_of = {}
    for n in loose:
        o = objects[n]
        for r in rooms:
            m = objects[r]
            if abs(o.pos[0] - m.pos[0]) <= m.size[0] / 2 and abs(o.pos[1] - m.pos[1]) <= m.size[1] / 2:
                room_of[n] = r
    zh = lang == "zh"
    if kind == "bring" and loose and tops:
        thing = rng.choice(loose)
        far = [t for t in tops if room_of.get(thing) is None or not t.startswith(room_of[thing])] or tops
        place = rng.choice(far)
        if zh:
            return f"把{_said_zh(thing)}拿到{_said_zh(place)}上"
        return f"Bring {_said(thing)} to {_said(place)}."
    if kind == "put" and loose:
        bins = [n for n, o in objects.items() if o.shape == "bin" and n not in ("office_bin", "bathroom_bin")]
        if not bins:
            return None
        thing, place = rng.choice(loose), rng.choice(bins)
        return f"把{_said_zh(thing)}放进{_said_zh(place)}里" if zh else f"Put {_said(thing)} in {_said(place)}."
    if kind == "pick_up" and loose:
        thing = rng.choice(loose)
        return f"拿起{_said_zh(thing)}" if zh else f"Pick up {_said(thing)}."
    if kind == "open" and drawers:
        thing = rng.choice(drawers)
        return f"打开{_said_zh(thing)}" if zh else f"Open {_said(thing)}."
    if kind == "go_to":
        if rooms and len(rooms) > 1:
            room = rng.choice(rooms[1:])
            return f"去{ZH_ROOMS.get(room.replace('_', ' '), room)}" if zh else f"Go to the {room.replace('_', ' ')}."
        if "goal" in objects:
            return "走到终点的红色标记那里" if zh else "Walk along the path to the goal at its end."
        if "dock" in [t.name for t in world.terrain]:
            return "开到装货平台（dock）上" if zh else "Drive up onto the loading dock."
        benches = [n for n in objects if n.startswith("bench")]
        if benches:
            return "走到长椅旁边" if zh else "Go to the bench."
    return None


PLAN = {  # kinds of task tried for each kind of place, in turn
    "house": ("bring", "open", "go_to", "put", "pick_up", "bring"),
    "warehouse": ("bring", "go_to", "pick_up", "bring"),
    "outdoor": ("go_to",),
}


@contextlib.contextmanager
def _home(path: Path) -> Iterator[None]:
    before = os.environ.get("ROBOTFORGE_HOME")
    os.environ["ROBOTFORGE_HOME"] = str(path)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("ROBOTFORGE_HOME", None)
        else:
            os.environ["ROBOTFORGE_HOME"] = before


def make_suite(
    count: int, seed: int = 0, out: str | Path = "suite", zh_share: float = 0.3, verify: bool = True
) -> list[Case]:
    """`count` cases into `out` (worlds under out/worlds, the list in out/suite.json). With `verify` a bring or put
    case goes in only if pick_and_place gets it done from the start (a minute or two for a suite of eight); without,
    only the cheap checks (not done at the start, planned without refusal) are made."""
    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    out = Path(out)
    rng = random.Random(seed)
    places = ("house", "house", "warehouse", "outdoor", "house")
    cases: list[Case] = []
    with _home(out):
        tries = 0
        while len(cases) < count and tries < count * 4:
            tries += 1
            i = len(cases)
            place = places[i % len(places)]
            lang = "zh" if rng.random() < zh_share else "en"
            text, wseed = world_sentence(rng, place), rng.randrange(1000)
            kind = PLAN[place][(i // len(places)) % len(PLAN[place])]
            pg = Playground(session=SimSession.from_xml("<mujoco/>"))
            name = f"case_{i + 1:02d}"
            pg.new_world(name)
            made = pg.load_scene(description=text, seed=wseed)["scene"]["generated"]
            held = {o.name: o for o in pg.current_world().objects}
            loose = [n for n in made["things_to_pick_up"] if n != "..." and not _container(held.get(n))]
            sentence = task_sentence(rng, kind, pg.current_world(), loose, lang)
            if sentence is None:
                continue
            try:
                task = pg.add_task(sentence, name="task")["task"]
            except ValueError:
                continue
            if _unfit(pg, task, verify) is not None:  # a suite of tasks nobody can do measures nothing
                continue
            cases.append(Case(name, text, wseed, sentence, task["kind"], task["robot"], task["object"],
                              task["target"], lang))  # fmt: skip
    (out / "suite.json").write_text(
        json.dumps({"count": count, "seed": seed, "cases": [asdict(c) for c in cases]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return cases


def _container(o: Any) -> bool:
    """A bin or a bowl: something to put things in, not a thing to bring."""
    return o is not None and (o.shape == "bin" or re.sub(r"_\d+$", "", o.name).endswith(("bowl", "bin", "basket")))


ATTEMPT_S = 150.0  # simulated seconds the check gives pick_and_place to do a case (an errand takes 15 to 60)


def _unfit(pg: Any, task: dict[str, Any], verify: bool = True) -> str | None:
    """Why a task just added should not go in a suite, or None: it is done before it starts, or what an agent's
    pick_and_place would do (the arm's own planner, then the sim) does not get it done. Run on the open world, which
    is already saved: nothing here reaches the file."""
    status = task.get("status") or {}
    if status.get("done") and task["kind"] != "go_to":
        return "done before it starts"
    if task["kind"] in ("bring", "put") and task.get("object") and task.get("target"):
        try:
            motion, _ = pg.behaviors.pick_and_place(task["robot"], task["object"], task["target"])
        except Exception as exc:  # the planner's refusal is the answer: the arm cannot reach, cannot hold, ...
            return str(exc)
        if not verify:
            return None
        pg.run_fast(lambda: motion.done, ATTEMPT_S)
        pg.run_fast(lambda: False, 1.0)
        if not pg.task_status().get(task["name"], {}).get("done"):
            return "pick_and_place does not get it done"
    return None


def make_hard_suite(
    count: int, seed: int = 0, out: str | Path = "suite", arms: tuple[str, ...] = ("panda", "ur5e")
) -> list[Case]:
    """`count` cases of the hard tasks (a mug upright on a shelf, a stack, a bottle into a rack, a bin sorted into
    trays, a pour, a tool handed to the other arm), each a bench world with the task added from its sentence, in the
    same layout as make_suite (scripts/agent_eval.py --suite runs either). Kinds and arms take turns; the seed of a
    case moves its objects."""
    from robotforge.design.tasks import hard_bench
    from robotforge.design.tasks.spec import HARD_KINDS

    out = Path(out)
    cases: list[Case] = []
    with _home(out):
        for i in range(count):
            kind, arm = HARD_KINDS[i % len(HARD_KINDS)], arms[(i // len(HARD_KINDS)) % len(arms)]
            case_seed = seed * 1000 + i
            name = f"case_{i + 1:02d}"
            pg = hard_bench.fresh(str(out), name)
            made = hard_bench.build(pg, kind, case_seed, arm)
            task = made["task"]
            two = " and a second one beside it" if kind == "hand_over" else ""
            text = f"a table with a {arm} arm{two}"
            cases.append(Case(name, text, case_seed, made["sentence"], task["kind"], task["robot"], task["object"],
                              task["target"], "en"))  # fmt: skip
    (out / "suite.json").write_text(
        json.dumps({"count": count, "seed": seed, "cases": [asdict(c) for c in cases]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return cases


def load_suite(folder: str | Path) -> list[Case]:
    data = json.loads((Path(folder) / "suite.json").read_text(encoding="utf-8"))
    return [Case(**c) for c in data["cases"]]
