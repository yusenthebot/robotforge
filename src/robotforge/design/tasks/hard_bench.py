"""A bench for each hard task and a scripted expert that does it, for success rates, demos and suites.

    from robotforge.design.tasks import hard_bench
    report = hard_bench.run("stack", seed=3, arm="ur5e")     # {"done": True, "why": ..., "steps": [...], ...}

`build(pg, kind, seed, arm)` furnishes the open world of a Playground: an arm on a table (a Panda, or a UR5e with a
Robotiq 2F-85), the fixtures and things the task needs, placed from the seed, and adds the task with `add_task`, so the
sentence and its reading are the ones an agent's would be. `solve(pg)` then makes the pick_and_place calls an agent
would make, one thing at a time, and reports each. Every failure is given a cause (refused, never lifted, slipped,
tipped over, landed outside, knocked something over), so a table of rates says why the rest went wrong.
"""

from __future__ import annotations

import math
import random
import tempfile
import time
from typing import Any

import numpy as np

from robotforge import objects
from robotforge.design.tasks.hard import ARMS, SORT_CLASS, TABLE_Z

SENTENCES = {
    "upright": "put the mug on the shelf upright",
    "stack": "stack the {top} on the {mid} on the book",
    "insert": "insert the bottle into the rack",
    "sort": "sort the things in the bin into the trays by kind",
    "pour": "pour the balls from the cup into the bowl",
    "hand_over": "hand the hammer to the other arm",
}


def bench(pg: Any, arm: str = "panda") -> None:
    """The arm on a table, ready for a task's things."""
    spec = ARMS[arm]
    pg.add_robot("arm", library=spec["library"], mounts=spec["mounts"], pos=[0, 0, TABLE_Z])
    pg.add_object("table", shape="table", size=[1.4, 1.2, TABLE_Z], pos=[0.4, 0], fixed=True, color="#d9d2c5")


def fresh() -> Any:
    """A Playground with an empty world in a folder of its own (the real Menagerie models are used)."""
    import os

    from robotforge.playground import Playground
    from robotforge.sim.session import SimSession

    os.environ["ROBOTFORGE_HOME"] = tempfile.mkdtemp()
    pg = Playground(session=SimSession.from_xml("<mujoco/>"))
    pg.new_world("bench")
    return pg


def build(pg: Any, kind: str, seed: int = 0, arm: str = "panda") -> dict[str, Any]:
    """Furnish the open world for `kind` and add the task; returns what was made."""
    rng = random.Random(f"{kind}:{seed}")
    bench(pg, arm)
    sentence = {
        "upright": _upright,
        "stack": _stack,
        "insert": _insert,
        "sort": _sort,
        "pour": _pour,
    }[kind](pg, rng, seed)
    pg.run_fast(lambda: False, 0.6)
    added = pg.add_task(sentence, robot="arm", name=kind, seed=seed)
    return {
        "kind": kind,
        "seed": seed,
        "arm": arm,
        "sentence": sentence,
        "task": added["task"],
        "notes": added.get("notes", []),
    }


def _upright(pg: Any, rng: random.Random, seed: int) -> str:
    pg.add_object("shelf", shape="shelf", size=[0.40, 0.20, 0.20], pos=[0.52, 0.31, TABLE_Z], color="#b08a5a")
    pg.add_object("mug", shape="mug", seed=seed, pos=[0.45, -0.20, TABLE_Z], yaw=rng.uniform(-70, 70))
    return SENTENCES["upright"]


def _stack(pg: Any, rng: random.Random, seed: int) -> str:
    mid, top = rng.sample(["tin", "brick"], 2)
    pg.add_object("book", shape="book", seed=seed, pos=[0.52, 0.02, TABLE_Z], yaw=rng.uniform(-15, 15))
    pg.add_object(mid, shape=mid, seed=seed, pos=[0.33, -0.28, TABLE_Z], yaw=rng.uniform(-30, 30))
    pg.add_object(top, shape=top, seed=seed, pos=[0.56, -0.30, TABLE_Z], yaw=rng.uniform(-30, 30))
    return SENTENCES["stack"].format(top=top, mid=mid)


def _insert(pg: Any, rng: random.Random, seed: int) -> str:
    bottle = objects.object_for("bottle", seed=seed)
    pg.add_object(
        "rack", shape="rack", pos=[0.52, 0.30, TABLE_Z], yaw=rng.uniform(-12, 12), params={"bottle": bottle.size[0]}
    )
    pg.add_object("bottle", shape="bottle", seed=seed, pos=[0.40, -0.24, TABLE_Z])
    return SENTENCES["insert"]


def _pour(pg: Any, rng: random.Random, seed: int) -> str:
    """A tumbler with a handful of balls in it, and a bowl across the table to pour them into."""
    pg.add_object(
        "bowl", shape="bowl", seed=seed, pos=[0.50 + rng.uniform(-0.03, 0.03), 0.25 + rng.uniform(-0.03, 0.03), TABLE_Z]
    )
    x, y = 0.45 + rng.uniform(-0.03, 0.03), -0.22 + rng.uniform(-0.03, 0.03)
    pg.add_object("cup", shape="cup", seed=seed, params={"style": "tumbler", "lid": False}, pos=[x, y, TABLE_Z])
    count = rng.randint(6, 9)
    ring = 0.55 * float(objects.object_for("cup", seed=seed, style="tumbler", lid=False).facts["mouth_r"])
    for i in range(count):  # layers of four, each turned an eighth from the one below: they rest, not wedge
        layer, k = divmod(i, 4)
        a = math.pi / 2 * k + math.pi / 4 * layer
        pg.add_object(
            f"ball{i}", shape="sphere", size=[0.012], mass=0.003, color="#d9453d",
            pos=[x + ring * math.cos(a), y + ring * math.sin(a), TABLE_Z + 0.02 + 0.021 * layer],
        )  # fmt: skip
    return SENTENCES["pour"]


SORT_KINDS = (("fruit", ("apple", "orange")), ("cans", ("can", "can")), ("boxes", ("brick", "tin")))
TRAY_SPOTS = ((0.28, 0.42, 0.0), (0.54, 0.44, 0.0), (0.76, 0.26, 70.0))
BIN = (0.46, -0.30, 0.34, 0.26)  # x, y, length, width of the bin the things start in


def _sort(pg: Any, rng: random.Random, seed: int) -> str:
    colors = {"fruit": "#7cb342", "cans": "#3b82c4", "boxes": "#c0852b"}
    for (tag, _), (x, y, yaw) in zip(SORT_KINDS, TRAY_SPOTS, strict=True):
        pg.add_object(f"{tag}_tray", shape="tray", size=[0.20, 0.16, 0.05], pos=[x, y, TABLE_Z], yaw=yaw,
                      color=colors[tag], params={"tag": tag})  # fmt: skip
    bx, by, bl, bw = BIN
    pg.add_object(
        "bin", shape="tray", size=[bl, bw, 0.06], pos=[bx, by, TABLE_Z], color="#9aa3ad", params={"tag": "none"}
    )
    placed: list[tuple[float, float, float]] = []
    names = []
    kinds = [s for _, shapes in SORT_KINDS for s in shapes]
    rng.shuffle(kinds)
    counts: dict[str, int] = {}
    for shape in kinds:
        counts[shape] = counts.get(shape, 0) + 1
        item_seed = seed * 7 + counts[shape]
        built = objects.object_for(shape, seed=item_seed)
        radius = max(built.half) + 0.012
        spot = _free(rng, placed, radius, bx, by, bl / 2 - 0.045, bw / 2 - 0.045)
        placed.append((*spot, radius))
        name = f"{shape}_{counts[shape]}"
        pg.add_object(
            name, shape=shape, seed=item_seed, pos=[spot[0], spot[1], TABLE_Z + 0.006], yaw=rng.uniform(0, 360)
        )
        names.append(name)
    return SENTENCES["sort"]


def _free(rng: random.Random, placed: list[tuple[float, float, float]], radius: float, cx: float, cy: float,
          hx: float, hy: float) -> tuple[float, float]:  # fmt: skip
    """A spot in the bin, a finger's width from everything already there: a crowded bin, not a heap."""
    best, gap = (cx, cy), -1.0
    for _ in range(300):
        x, y = cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy)
        room = min((math.hypot(x - px, y - py) - radius - pr for px, py, pr in placed), default=1.0)
        if room >= 0.03:
            return x, y
        if room > gap:
            best, gap = (x, y), room
    return best


# ---- the expert ------------------------------------------------------------------------------------------------------


def act(pg: Any, thing: str, target: str, robot: str = "arm", grasp: str | None = None) -> dict[str, Any]:
    """One pick_and_place as an agent makes it: refused (with its reason) or run to its end and judged."""
    arm = pg.behaviors.resolve(robot, arm=True)
    m, d = pg.session.model, pg.session.data
    body = m.body(thing).id
    start = d.xpos[body].copy()
    row: dict[str, Any] = {"thing": thing, "target": target}
    try:
        motion, info = pg.behaviors.pick_and_place(arm, thing, target, True, grasp)
    except Exception as exc:  # the planner's refusal is the answer
        row.update(refused=str(exc)[:300])
        return row
    best = [float(start[2])]

    def watch() -> bool:
        best.append(float(d.xpos[body][2]))
        return bool(motion.done)

    t0 = float(d.time)
    pg.run_fast(watch, 60)
    pg.run_fast(lambda: _still(d, m, body), 2.5)
    result = pg.behaviors.placed(thing, info["region"])
    row.update(
        success=bool(result["success"]),
        tilt_deg=result.get("tilt_deg"),
        grasp=(info.get("grasp") or {}).get("kind"),
        lifted_cm=round((max(best) - float(start[2])) * 100, 1),
        sim_s=round(float(d.time) - t0, 1),
        moved_cm=round(float(np.hypot(*(d.xpos[body][:2] - start[:2]))) * 100, 1),
    )
    return row


def _still(d: Any, m: Any, body: int) -> bool:
    adr = int(m.body_dofadr[body])
    return bool(np.linalg.norm(d.qvel[adr : adr + 3]) < 0.01)


def solve(pg: Any, name: str | None = None, robot: str = "arm") -> dict[str, Any]:
    """Do the task `name` (the only one by default) the way an agent would, and report. The verdict is the task's own
    check; each step says what happened, and a failure has a cause."""
    world = pg.current_world()
    task = next(t for t in world.tasks if name in (None, t.name))
    t0, wall = float(pg.session.data.time), time.time()
    steps: list[dict[str, Any]] = []
    if task.kind == "pour":
        return _finish(pg, task, [_pour_step(pg, task.object, task.target, robot)], t0, wall)
    order = _order(pg, task)
    placed: list[str] = []
    for thing, target in order:
        before = _positions(pg, task)
        step = act(pg, thing, target, robot)
        after = _positions(pg, task)
        # only a thing already put away counts as knocked over: one waiting in a crowded bin is nudged by the fingers
        step["knocked"] = sorted(n for n in placed if _moved(before[n], after[n]))
        if step.get("success"):
            placed.append(thing)
        steps.append(step)
        if "refused" in step:
            break
    return _finish(pg, task, steps, t0, wall)


def _pour_step(pg: Any, thing: str, target: str, robot: str) -> dict[str, Any]:
    """One pour as an agent makes it: pick_and_place(pour=true), run to its end, then what came of it."""
    arm = pg.behaviors.resolve(robot, arm=True)
    d = pg.session.data
    row: dict[str, Any] = {"thing": thing, "target": target}
    try:
        motion, info = pg.behaviors.pick_and_place(arm, thing, target, True, None, True)
    except Exception as exc:  # the planner's refusal is the answer
        row.update(refused=str(exc)[:300])
        return row
    t0 = float(d.time)
    pg.run_fast(lambda: bool(motion.done), 120)
    pg.run_fast(lambda: False, 1.5)
    result = pg.behaviors.poured(info)
    row.update(
        success=bool(result["success"]),
        balls_in_target=result["balls_in_target"],
        balls=result["balls"],
        toward_deg=result["poured_toward_deg"],
        sim_s=round(float(d.time) - t0, 1),
    )
    return row


def _finish(pg: Any, task: Any, steps: list[dict[str, Any]], t0: float, wall: float) -> dict[str, Any]:
    pg.run_fast(lambda: False, 1.2)
    verdict = pg.task_status()[task.name]
    report = {
        "task": task.name,
        "kind": task.kind,
        "done": bool(verdict["done"]),
        "why": verdict["why"],
        "steps": steps,
        "sim_s": round(float(pg.session.data.time) - t0, 1),
        "wall_s": round(time.time() - wall, 1),
    }
    if not report["done"]:
        report["cause"] = cause(steps, verdict["why"])
    return report


def _order(pg: Any, task: Any) -> list[tuple[str, str]]:
    """The pick_and_place calls the task takes: what, onto what."""
    if task.kind == "upright":
        return [(task.object, task.target)]
    if task.kind == "insert":
        return [(task.object, task.target)]
    if task.kind == "stack":
        items = task.items
        return [(upper, lower) for lower, upper in zip(items[:-1], items[1:], strict=True)]
    if task.kind == "sort":
        d, m = pg.session.data, pg.session.model
        base = d.xpos[pg.behaviors._root_body("arm")][:2]
        near = sorted(task.items, key=lambda n: float(np.hypot(*(d.xpos[m.body(n).id][:2] - base))))
        return [(n, task.into[n]) for n in near]
    raise ValueError(f"no expert for {task.kind}")


def _positions(pg: Any, task: Any) -> dict[str, np.ndarray]:
    m, d = pg.session.model, pg.session.data
    names = [n for n in [task.object, *(task.items or [])] if n]
    return {n: d.xpos[m.body(n).id].copy() for n in dict.fromkeys(names)}


def _moved(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.linalg.norm(a - b) > 0.02)


def cause(steps: list[dict[str, Any]], why: str) -> str:
    """Why a task failed, from its steps: the first step that went wrong, and how."""
    for s in steps:
        if "refused" in s:
            return f"refused: {s['refused'][:140]}"
        if "balls_in_target" in s and not s.get("success"):
            return f"only {s['balls_in_target']} of {s['balls']} balls reached {s['target']} (the rest spilled)"
        if s.get("knocked"):
            return f"knocked over {', '.join(s['knocked'])} while handling {s['thing']}"
        if not s.get("success"):
            if s["lifted_cm"] < 2.0:
                return f"{s['thing']} was never lifted (the grip did not take hold)"
            if s["moved_cm"] < 15.0:
                return f"{s['thing']} slipped out of the fingers on the way"
            return f"{s['thing']} landed outside {s['target']}"
        if (s.get("tilt_deg") or 0.0) > 25.0 and SORT_CLASS.get(s["thing"].rsplit("_", 1)[0], "") != "tools":
            return f"{s['thing']} ended tipped over ({s['tilt_deg']:.0f} degrees)"
    return f"the check says: {why}"


def run(kind: str, seed: int = 0, arm: str = "panda") -> dict[str, Any]:
    """A whole trial: a fresh world, the bench, the expert."""
    pg = fresh()
    made = build(pg, kind, seed, arm)
    report = solve(pg)
    return {**report, "seed": seed, "arm": arm, "sentence": made["sentence"]}
