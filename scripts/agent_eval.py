"""A standing evaluation of robotforge with a real agent: Codex does a fixed set of beginner tasks through the
plugin, and a deterministic check reads what really happened.

    uv run python scripts/agent_eval.py [--only NAME,...] [--ref REF | --source SOURCE] [--label TEXT]
    uv run python scripts/agent_eval.py --rescore RUN_DIR          # check a finished run again, no agent
    uv run python scripts/agent_eval.py --transcript RUN_DIR/tasks/NAME   # read one session
    uv run python scripts/agent_eval.py --compare BEFORE.json [MORE.json] [--after AFTER.json ...]
    uv run python scripts/agent_eval.py --list

Every task is one `codex exec --json` session with a fresh ROBOTFORGE_HOME, so the worlds it leaves on disk are
its own; at most MAX_JOBS sessions run at once, each for at most MAX_SESSION_SECONDS. The plugin is installed
into a scratch CODEX_HOME from a local marketplace: a `git archive` of --ref (default HEAD), so a run evaluates
exactly one commit; --source takes a plugin folder or a GitHub marketplace (owner/repo or owner/repo@ref)
instead. The installed plugin's environment is built before the first session: the eval measures how an agent
does the tasks, not the cold install (docs/PLUGINS.md measures that). Codex gets no browser, computer use or
ChatGPT apps, which would add tokens and variance that have nothing to do with robotforge.

A task's check reads only what happened: the MCP tool calls and their results in the event stream, and the
worlds saved on disk (world.yaml). It never reads what the agent says it did. Each check is a short list of
criteria; a task passes when all of them do. Alongside, every session records its robotforge tool calls, the
calls that failed, shell commands, input, cached and output tokens and wall time.

A run writes a JSON summary (no absolute paths) to --results, by default docs/evals/agent/<date>-<label>.json,
for later rounds to compare against; the raw event streams and worlds stay in the run folder under --work.
This is not a test: it needs Codex, a login and the network. tests/test_agent_eval.py checks the criteria on a
recorded stream.
"""

from __future__ import annotations

import argparse
import colorsys
import datetime as dt
import functools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "docs" / "evals" / "agent"
SERVER = "robotforge"  # the MCP server's name in .codex-plugin/mcp.json
PLUGIN = "robotforge@robotforge"  # plugin@marketplace, as .claude-plugin/marketplace.json names them
MODEL = "gpt-6.1-sol"
EFFORT = "medium"
MAX_SESSION_SECONDS = 600  # a beginner would not wait longer for one request
MAX_JOBS = 2  # sessions run on the user's ChatGPT plan: never more than two at once
# Codex features that are not robotforge's: a browser, computer use, image generation and ChatGPT apps would add
# tokens and variance to every session (round 31's session listed every ChatGPT app before starting).
OFF_FEATURES = ("apps", "browser_use", "browser_use_external", "in_app_browser", "computer_use", "image_generation")
SHORT = 300  # characters of an error or a message kept in the summary

# ---- what a session did -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One MCP tool call as Codex reports it: the tool, its arguments and the text of its result or error."""

    server: str
    tool: str
    args: dict[str, Any]
    ok: bool
    text: str

    @functools.cached_property
    def data(self) -> Any:
        """The result as JSON (robotforge answers in JSON), or None for an error or plain text."""
        if not self.ok:
            return None
        try:
            return json.loads(self.text)
        except ValueError:
            return None


@dataclass(frozen=True)
class Session:
    calls: tuple[Call, ...]
    commands: tuple[str, ...]  # shell commands the agent ran
    messages: tuple[str, ...]  # what it said, in order
    usage: dict[str, int]  # tokens summed over turns: input_tokens, cached_input_tokens, output_tokens, ...
    problems: tuple[str, ...] = ()  # errors Codex itself reported (a failed turn, a dropped stream)

    @property
    def ours(self) -> list[Call]:
        return [c for c in self.calls if c.server == SERVER]

    def results(self, *tools: str) -> list[dict[str, Any]]:
        """The JSON results of robotforge's successful calls to these tools, in order."""
        return [c.data for c in self.ours if c.ok and c.tool in tools and isinstance(c.data, dict)]


ERROR_PREFIX = "Error executing tool "


def _call(item: dict[str, Any]) -> Call:
    result = item.get("result") or {}
    parts = result.get("content") or [] if isinstance(result, dict) else []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
    error = item.get("error")
    failed = item.get("status") == "failed" or bool(error) or bool(isinstance(result, dict) and result.get("isError"))
    if error and not text:
        text = error.get("message", "") if isinstance(error, dict) else str(error)
    if failed and text.startswith(ERROR_PREFIX):  # "Error executing tool move: ..." is the SDK's frame, not ours
        text = text.split(": ", 1)[-1]
    args = item.get("arguments")
    return Call(item.get("server", ""), item.get("tool", ""), args if isinstance(args, dict) else {}, not failed, text)


def parse_events(lines: Iterable[str]) -> Session:
    """Read `codex exec --json` output: one JSON event per line; anything else (a log line) is skipped."""
    calls, commands, messages, problems = [], [], [], []
    usage: dict[str, int] = {}
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "turn.completed":
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        elif kind in ("turn.failed", "error"):
            problems.append(json.dumps(event.get("error") or event.get("message") or event, ensure_ascii=False))
        if kind != "item.completed":
            continue
        item = event.get("item") or {}
        match item.get("type"):
            case "mcp_tool_call":
                calls.append(_call(item))
            case "command_execution":
                commands.append(str(item.get("command", "")))
            case "agent_message":
                messages.append(str(item.get("text", "")))
            case "error":
                problems.append(str(item.get("message", "")))
    return Session(tuple(calls), tuple(commands), tuple(messages), usage, tuple(problems))


# ---- what is on disk --------------------------------------------------------------------------------------


@dataclass
class World:
    name: str
    dir: Path
    data: dict[str, Any]  # world.yaml as plain data; {} when it does not parse

    @property
    def robots(self) -> list[dict[str, Any]]:
        return [r for r in self.data.get("robots") or [] if isinstance(r, dict)]

    @property
    def objects(self) -> list[dict[str, Any]]:
        return [o for o in self.data.get("objects") or [] if isinstance(o, dict)]

    def object(self, name: str) -> dict[str, Any] | None:
        return next((o for o in self.objects if o.get("name") == name), None)

    @functools.cached_property
    def kinds(self) -> dict[str, str]:
        """Each robot's kind as robotforge's `robots` tool reports it, from the world compiled as saved."""
        from robotforge.playground import Playground

        try:
            playground = Playground.with_demo()
            playground.open_world(str(self.dir))
            return {r["name"]: r["kind"] for r in playground.behaviors.robots()}
        except Exception as exc:  # a world the agent left broken has no kinds; the check says so
            return {"": f"does not compile: {type(exc).__name__}: {exc}"[:SHORT]}


def load_worlds(home: Path) -> list[World]:
    worlds = []
    for path in sorted((home / "worlds").glob("*/world.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}
        worlds.append(World(path.parent.name, path.parent, data if isinstance(data, dict) else {}))
    return worlds


@dataclass
class Evidence:
    """Everything a check may read: the session and the worlds it left."""

    session: Session
    worlds: list[World]
    work: Path  # the agent's working folder, where a task's own files are

    def kind(self, world: World, robot: dict[str, Any]) -> str:
        """A template or library robot is the kind it was declared as (no need to compile it); any other is
        recognized by its shape, as robotforge does."""
        from robotforge.design import library, templates

        name = robot.get("name", "")
        if robot.get("template"):
            template = templates.canonical(str(robot["template"]))
            if template in ("quadruped", "arm", "humanoid", "rover"):
                return template
        if robot.get("library"):
            try:
                return library.get(str(robot["library"])).kind
            except (KeyError, ValueError):
                pass
        return world.kinds.get(name, "unknown")

    def robots(self, *kinds: str) -> list[tuple[World, dict[str, Any]]]:
        return [(w, r) for w in self.worlds for r in w.robots if not kinds or self.kind(w, r) in kinds]


# ---- where things went --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading_deg: float | None
    upright: bool | None


def _pose(entry: dict[str, Any]) -> Pose | None:
    pos = entry.get("position")
    if not (isinstance(pos, list) and len(pos) >= 2):
        return None
    return Pose(float(pos[0]), float(pos[1]), entry.get("heading_deg"), entry.get("upright"))


def poses(session: Session, robot: str) -> list[Pose]:
    """Where robotforge reported `robot`, in order: before and after every move, in every `robots` list, at the
    end of a run_controller, and as its free body ("<robot>/<body>") in observe and simulate. Only move and robots
    say whether it is upright."""
    out = []
    for call in session.ours:
        data = call.data
        if not isinstance(data, dict):
            continue
        if call.tool == "move":
            out += [p for end in ("from", "to") if (e := data.get(end)) and e.get("robot") == robot and (p := _pose(e))]
        elif call.tool == "robots":
            out += [p for e in data.get("robots") or [] if e.get("name") == robot and (p := _pose(e))]
        elif call.tool == "run_controller" and data.get("robot") == robot:
            out += [p for e in [{"position": (data.get("result") or {}).get("position")}] if (p := _pose(e))]
        elif call.tool in ("observe", "simulate"):
            moved = {name: m.get("to") for name, m in (data.get("moved") or {}).items() if isinstance(m, dict)}
            bodies = data.get("free_bodies") or moved
            out += [
                p for name, pos in bodies.items() if name.startswith(f"{robot}/") and (p := _pose({"position": pos}))
            ]
    return out


def moves(session: Session, robot: str) -> list[dict[str, Any]]:
    """Results of `move` calls that moved `robot` for a set time (the result names it as the world spells it)."""
    return [m for m in session.results("move") if (m.get("from") or {}).get("robot") == robot and "distance_m" in m]


def travelled(session: Session, robot: str) -> float:
    """How far `robot` went: the distances its moves measured, or how far apart its first and last reported
    positions are (a script or `simulate` can move it too), whichever is more. Meters."""
    measured = sum(float(m.get("distance_m") or 0) for m in moves(session, robot))
    seen = poses(session, robot)
    net = math.dist((seen[0].x, seen[0].y), (seen[-1].x, seen[-1].y)) if seen else 0.0
    return max(measured, net)


def turned_left(session: Session, robot: str) -> float:
    """Degrees turned to the left over all moves (a right turn counts against it)."""
    return sum(float(m.get("turned_deg") or 0) for m in moves(session, robot))


def upright_at_end(session: Session, robot: str) -> bool | None:
    seen = [p.upright for p in poses(session, robot) if p.upright is not None]
    return seen[-1] if seen else None


def object_positions(session: Session, name: str) -> list[list[float]]:
    """Where robotforge reported object `name`, in order: pick_and_place, observe and simulate results."""
    out = []
    for call in session.ours:
        data = call.data
        if not isinstance(data, dict):
            continue
        if call.tool == "pick_and_place" and data.get("object") == name and data.get("object_position"):
            out.append(data["object_position"])
        elif call.tool == "observe" and name in (data.get("free_bodies") or {}):
            out.append(data["free_bodies"][name])
        elif call.tool == "simulate" and name in (data.get("moved") or {}):
            out.append(data["moved"][name].get("to"))
    return [p for p in out if isinstance(p, list) and len(p) >= 3]


CONTAINERS = ("bin", "box")
RIM_SLACK = 0.03  # m above a container's rim that still counts as in it (the object rests on what is inside)


def inside(point: list[float], container: dict[str, Any]) -> bool:
    """Is `point` within the container's footprint (as saved in world.yaml) and, for one standing on the ground,
    no higher than just above its rim? (One given a height, on a table say, is checked by its footprint only.)"""
    pos, size = container.get("pos") or [0, 0], container.get("size") or [0.05, 0.05, 0.05]
    if len(size) < 3 or len(pos) < 2:
        return False
    yaw = math.radians(float(container.get("yaw") or 0))
    dx, dy = point[0] - pos[0], point[1] - pos[1]
    along = dx * math.cos(yaw) + dy * math.sin(yaw)
    across = -dx * math.sin(yaw) + dy * math.cos(yaw)
    low = len(pos) > 2 or point[2] <= size[2] + RIM_SLACK
    return abs(along) <= size[0] / 2 and abs(across) <= size[1] / 2 and low


def looks(color: Any, hue: str) -> bool:
    """Does a color as world.yaml saves it (a name, a hex code or [r, g, b]) look red, green or blue?"""
    from robotforge.design import components

    try:
        r, g, b = components.color(color)[:3]
    except (ValueError, TypeError):
        return False
    h, _, s = colorsys.rgb_to_hls(r, g, b)
    degrees = h * 360
    span = {"red": (-25, 25), "green": (75, 165), "blue": (190, 260)}[hue]
    centered = degrees - 360 if degrees > 180 and hue == "red" else degrees
    return s >= 0.3 and span[0] <= centered <= span[1]


# ---- the tasks --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    name: str
    ok: bool
    observed: str


Check = Callable[[Evidence], list[Criterion]]


@dataclass(frozen=True)
class Task:
    name: str
    lang: str
    covers: tuple[str, ...]
    prompt: str  # "{work}" is the session's working folder
    check: Check
    setup: Callable[[Path, Path], None] | None = None  # (work, robotforge home) before the session

    def text(self, work: Path) -> str:
        return self.prompt.replace("{work}", str(work))


def _best(rows: list[tuple[Any, list[Criterion]]], empty: list[Criterion]) -> list[Criterion]:
    """The candidate (a robot, an object) that meets the most criteria, or `empty` when there is none."""
    if not rows:
        return empty
    return max(rows, key=lambda row: sum(c.ok for c in row[1]))[1]


def _named(world: World, robot: dict[str, Any]) -> str:
    return f"{world.name}/{robot.get('name')}"


def _walk_criteria(ev: Evidence, world: World, robot: dict[str, Any], kind: str, at_least: float) -> list[Criterion]:
    name = robot["name"]
    distance = travelled(ev.session, name)
    upright = upright_at_end(ev.session, name)
    return [
        Criterion(f"a {kind}", True, _named(world, robot)),
        Criterion(f"moved >= {at_least} m", distance >= at_least, f"{distance:.2f} m"),
        Criterion("upright at the end", upright is True, str(upright)),
    ]


def check_dog_grass(ev: Evidence) -> list[Criterion]:
    rows = []
    for world, robot in ev.robots("quadruped"):
        floor = (world.data.get("floor") or {}).get("style")
        rows.append((robot, [Criterion("on grass", floor == "grass", f"floor {floor}")]))
        rows[-1][1].extend(_walk_criteria(ev, world, robot, "quadruped", 1.0))
    return _best(rows, [Criterion("a quadruped", False, "none in any world")])


def check_arm_cube(ev: Evidence) -> list[Criterion]:
    arms = ev.robots("arm")
    rows = []
    for world, _ in arms:
        reds = [o for o in world.objects if looks(o.get("color"), "red") and o.get("shape") not in ("bin", "table")]
        for red in reds:
            seen = object_positions(ev.session, red["name"])
            last = seen[-1] if seen else None
            holders = [
                o
                for o in world.objects
                if o.get("shape") in CONTAINERS and o is not red and _bigger(o, red) and last and inside(last, o)
            ]
            where = f"{red['name']} last at {last}" if last else f"{red['name']} never reported moved"
            rows.append(
                (
                    red,
                    [
                        Criterion("an arm", True, _named(world, arms[0][1])),
                        Criterion("a red object", True, red["name"]),
                        Criterion("it ended in a box or bin", bool(holders), where),
                    ],
                )
            )
    empty = [Criterion("an arm", bool(arms), "found" if arms else "none"), Criterion("a red object", False, "none")]
    return _best(rows, empty)


def _bigger(container: dict[str, Any], thing: dict[str, Any]) -> bool:
    """A box only holds something whose footprint is clearly smaller (a red cube on a red box is not "in" it)."""
    a, b = container.get("size") or [0], thing.get("size") or [0]
    return min(a[:2]) >= 1.5 * max(b[:2]) if len(a) >= 2 else False


def check_go2_turn(ev: Evidence) -> list[Criterion]:
    rows = []
    for world, robot in ev.robots():
        if str(robot.get("library") or "").lower() != "go2":
            continue
        crit = _walk_criteria(ev, world, robot, "Go2 from the library", 0.3)
        left = turned_left(ev.session, robot["name"])
        crit.insert(2, Criterion("turned left >= 30 deg", left >= 30, f"{left:.0f} deg"))
        rows.append((robot, crit))
    return _best(rows, [Criterion("a Go2 from the library", False, "none in any world")])


ROVER_URDF = "my_rover/urdf/my_rover.urdf"


def setup_urdf_rover(work: Path, _home: Path) -> None:
    """A small URDF a beginner might download: a box on two wheels and a caster, its body drawn by a mesh found
    through a package:// path (the import doctor resolves it)."""
    sys.path.insert(0, str(REPO / "tests"))
    from importer_fixtures import box_stl, rover_urdf

    text = rover_urdf().replace(
        '<link name="chassis">',
        '<link name="chassis"><visual><geometry><mesh filename="package://my_rover/meshes/chassis.stl"/>'
        "</geometry></visual>",
        1,
    )
    (work / ROVER_URDF).parent.mkdir(parents=True, exist_ok=True)
    (work / ROVER_URDF).write_text(text, encoding="utf-8")
    box_stl(work / "my_rover" / "meshes" / "chassis.stl", (0.3, 0.2, 0.08))


def check_urdf_rover(ev: Evidence) -> list[Criterion]:
    rows = []
    for world, robot in ev.robots():
        file = str(robot.get("file") or "")
        if not file.endswith("my_rover.urdf"):
            continue
        crit = _walk_criteria(ev, world, robot, "robot imported from my_rover.urdf", 0.3)
        rows.append((robot, crit))
    return _best(rows, [Criterion("a robot imported from my_rover.urdf", False, "none in any world")])


def check_hexapod_build(ev: Evidence) -> list[Criterion]:
    rows = []
    for world, robot in ev.robots():
        if not robot.get("build"):
            continue
        kind = ev.kind(world, robot)
        crit = _walk_criteria(ev, world, robot, "robot built from parts", 0.2)
        crit.insert(1, Criterion("it is a hexapod", kind == "hexapod", kind))
        rows.append((robot, crit))
    return _best(rows, [Criterion("a robot built from parts", False, "none in any world")])


NEAR_ROCK = 0.6  # m from the rock's side: "next to it"
FAR_START = 1.0  # m: a rock closer than this at the start leaves nothing to drive


def _gap(x: float, y: float, rock: dict[str, Any]) -> float:
    """Meters from (x, y) to the side of a rock (world.yaml saves a rock's size as [diameter, height])."""
    return math.dist((x, y), rock["pos"][:2]) - float((rock.get("size") or [0.5])[0]) / 2


def check_rover_rock(ev: Evidence) -> list[Criterion]:
    rows = []
    for world in ev.worlds:
        rocks = [o for o in world.objects if o.get("shape") == "rock" and len(o.get("pos") or []) >= 2]
        for robot in world.robots if rocks else []:
            seen = poses(ev.session, robot.get("name", ""))
            if not seen:
                continue
            start, end = robot.get("pos") or [seen[0].x, seen[0].y], seen[-1]
            rock = min(rocks, key=lambda r: _gap(end.x, end.y, r))
            at_end, at_start = _gap(end.x, end.y, rock), _gap(start[0], start[1], rock)
            rows.append(
                (
                    robot,
                    [
                        Criterion("a robot and a rock", True, f"{_named(world, robot)}, {rock['name']}"),
                        Criterion(
                            f"rock >= {FAR_START} m away at the start", at_start >= FAR_START, f"{at_start:.2f} m"
                        ),
                        Criterion(f"ended within {NEAR_ROCK} m of it", at_end <= NEAR_ROCK, f"{at_end:.2f} m"),
                        Criterion("upright at the end", end.upright is not False, str(end.upright)),
                    ],
                )
            )
    return _best(rows, [Criterion("a robot and a rock", False, "no robot reported in a world with a rock")])


def check_learn_control(ev: Evidence) -> list[Criterion]:
    plans = [r.get("lesson") for r in ev.session.results("lessons")]
    got = sorted({p.get("number") for p in plans if isinstance(p, dict) and p.get("number") is not None})
    ran = [
        c
        for c in ev.session.ours
        if c.tool in ("run_controller", "compare_controllers") and c.ok and "balance" in str(c.args.get("task", ""))
    ]
    return [
        Criterion("opened the control lesson (lessons 6)", 6 in got, f"lessons opened: {got or 'none'}"),
        Criterion("ran a balance controller", bool(ran), f"{len(ran)} successful run_controller/compare_controllers"),
    ]


WORKSHOP = """\
name: workshop
floor:
  style: wood
robots:
- name: arm
  template: arm
  pos: [0.0, 0.0]
objects:
- name: green_cube
  shape: box
  size: [0.04, 0.04, 0.04]
  pos: [0.25, 0.1]
  color: green
- name: bin
  shape: bin
  pos: [1.2, -0.4]
- name: ball
  shape: sphere
  size: [0.06]
  pos: [-0.3, 0.3]
  color: red
"""


def setup_workshop(_work: Path, home: Path) -> None:
    """A world the user made earlier, with the bin left out of the arm's reach (1.27 m away; it reaches 0.5)."""
    (home / "worlds" / "workshop").mkdir(parents=True, exist_ok=True)
    (home / "worlds" / "workshop" / "world.yaml").write_text(WORKSHOP, encoding="utf-8")


def check_fix_workshop(ev: Evidence) -> list[Criterion]:
    world = next((w for w in ev.worlds if w.name == "workshop"), None)
    if world is None:
        return [Criterion("the workshop world is still there", False, "gone")]
    ball, cube = world.object("ball"), world.object("green_cube")
    kept = [n for n in ("arm", "green_cube", "ball") if n in {e.get("name") for e in world.robots + world.objects}]
    seen = object_positions(ev.session, "green_cube")
    last = seen[-1] if seen else None
    holders = [o["name"] for o in world.objects if o.get("shape") in CONTAINERS and cube and _bigger(o, cube)]
    holding = [n for n in holders if last and inside(last, world.object(n) or {})]
    return [
        Criterion("the ball is blue", bool(ball) and looks(ball.get("color"), "blue"), str(ball and ball.get("color"))),
        Criterion("the green cube ended in the bin", bool(holding), f"last at {last}, containers {holders}"),
        Criterion("arm, cube and ball still there", len(kept) == 3, ", ".join(kept)),
    ]


TASKS: dict[str, Task] = {
    t.name: t
    for t in (
        Task(
            "dog_grass",
            "zh",
            ("template", "walk", "floor"),
            "给我一只四足机器狗，放在草地上，让它往前走 10 秒。",
            check_dog_grass,
        ),
        Task(
            "arm_cube",
            "en",
            ("template", "objects", "pick and place"),
            "Build me a robot arm and have it put a red cube into a box.",
            check_arm_cube,
        ),
        Task(
            "go2_turn",
            "zh",
            ("real robot", "walk", "turn"),
            "我想看看真正的 Unitree Go2 机器狗。帮我放一台，让它往前走几步，再向左转个弯。",
            check_go2_turn,
        ),
        Task(
            "urdf_rover",
            "en",
            ("import", "drive"),
            # A distance the check can hold the session to: asked for "a little", a session drove 0.29 m.
            "I downloaded a robot description to {work}/" + ROVER_URDF + ". Can you load it and drive it forward "
            "about half a meter?",
            check_urdf_rover,
            setup_urdf_rover,
        ),
        Task(
            "hexapod_build",
            "zh",
            ("parts", "walk"),
            "用零件帮我拼一个六条腿的机器人，像蜘蛛那样，然后让它走几步。",
            check_hexapod_build,
        ),
        Task(
            "rover_rock",
            "zh",
            ("template or parts", "objects", "drive to a place"),
            "造一台火星车，再放一块大石头，把车开到石头旁边停下。",
            check_rover_rock,
        ),
        Task(
            "learn_control",
            "zh",
            ("lessons", "controllers"),
            "我是新手，想学控制算法，比如 PID 是怎么让机器人保持平衡的。能一步步教我吗？",
            check_learn_control,
        ),
        Task(
            "fix_workshop",
            "en",
            ("edit a world", "recover from a mistake", "pick and place"),
            "Open my workshop world. Paint the ball blue, and have the arm drop the green cube into the bin.",
            check_fix_workshop,
            setup_workshop,
        ),
    )
}

# ---- one task's outcome -----------------------------------------------------------------------------------


@dataclass
class Outcome:
    task: str
    lang: str
    covers: list[str]
    passed: bool
    criteria: list[dict[str, Any]]
    tool_calls: int  # robotforge's tools
    failed_calls: int
    other_mcp_calls: int  # Codex's own MCP tools (resources and the like)
    shell_commands: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    wall_seconds: float
    exit_code: int | None
    timed_out: bool
    tools: list[str] = field(default_factory=list)  # robotforge tools in call order, failures marked with !
    failures: list[dict[str, str]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    final_message: str = ""


def judge(task: Task, session: Session, home: Path, work: Path, meta: dict[str, Any]) -> Outcome:
    evidence = Evidence(session, load_worlds(home), work)
    try:
        criteria = task.check(evidence)
    except Exception as exc:  # a check that crashes on an odd world is a bug in the check: show it, do not pass
        criteria = [Criterion("check ran", False, f"{type(exc).__name__}: {exc}"[:SHORT])]
    ours = session.ours
    failed = [c for c in ours if not c.ok]
    usage = session.usage
    run_dir = work.parent.parent.parent  # <run>/tasks/<task>/work

    def tidy(text: str) -> str:  # the summary is committed: no paths of this machine
        return _short(text.replace(str(run_dir), "RUN").replace(str(Path.home()), "~"))

    return Outcome(
        task=task.name,
        lang=task.lang,
        covers=list(task.covers),
        passed=bool(criteria) and all(c.ok for c in criteria) and not meta.get("timed_out"),
        criteria=[asdict(c) for c in criteria],
        tool_calls=len(ours),
        failed_calls=len(failed),
        other_mcp_calls=len(session.calls) - len(ours),
        shell_commands=len(session.commands),
        input_tokens=usage.get("input_tokens", 0),
        cached_input_tokens=usage.get("cached_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        reasoning_tokens=usage.get("reasoning_output_tokens", 0),
        wall_seconds=round(float(meta.get("wall_seconds", 0.0)), 1),
        exit_code=meta.get("exit_code"),
        timed_out=bool(meta.get("timed_out")),
        tools=[c.tool + ("" if c.ok else "!") for c in ours],
        failures=[
            {"tool": c.tool, "args": tidy(json.dumps(c.args, ensure_ascii=False)), "error": tidy(c.text)}
            for c in failed
        ],
        problems=[tidy(p) for p in session.problems],
        final_message=tidy(session.messages[-1] if session.messages else ""),
    )


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SHORT else text[: SHORT - 3] + "..."


# ---- running sessions -------------------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs: Any) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", **kwargs)
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(cmd[:4])} failed ({done.returncode}): {done.stderr.strip()[-800:]}")
    return done.stdout


def export_ref(ref: str, dest: Path) -> str:
    """The plugin as committed at `ref`: `git archive` leaves out the venv, caches and anything uncommitted."""
    commit = _run(["git", "-C", str(REPO), "rev-parse", "--short", f"{ref}^{{commit}}"]).strip()
    archive = dest.parent / "source.tar"
    _run(["git", "-C", str(REPO), "archive", "--format=tar", "-o", str(archive), commit])
    dest.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
    return commit


def codex_home(run_dir: Path, source: str, model: str, effort: str, work_dirs: list[Path]) -> Path:
    """A scratch CODEX_HOME: the login (a link to ~/.codex/auth.json), a minimal config and the plugin installed
    from `source`, with its environment built so the first session does not wait for it."""
    home = run_dir / "codex_home"
    home.mkdir(parents=True)
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.exists():
        raise SystemExit(f"no Codex login at {auth}: run `codex login` first")
    (home / "auth.json").symlink_to(auth)
    features = "\n".join(f"{name} = false" for name in OFF_FEATURES)
    trusted = "\n".join(f'[projects."{w}"]\ntrust_level = "trusted"\n' for w in work_dirs)
    (home / "config.toml").write_text(
        f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\napproval_policy = "never"\n'
        f'sandbox_mode = "danger-full-access"\nsuppress_unstable_features_warning = true\n\n'
        f"[features]\n{features}\n\n{trusted}",
        encoding="utf-8",
    )
    env = {**os.environ, "CODEX_HOME": str(home)}
    add = ["codex", "plugin", "marketplace", "add"]
    if "/" in source and not Path(source).exists():  # owner/repo or owner/repo@ref on GitHub
        repo, _, ref = source.partition("@")
        add += [repo, *(["--ref", ref] if ref else [])]
    else:
        add.append(str(Path(source).resolve()))
    _run(add, env=env)
    installed = _run(["codex", "plugin", "add", PLUGIN], env=env)
    root = next(
        (Path(line.split(":", 1)[1].strip()) for line in installed.splitlines() if "plugin root:" in line), None
    )
    if root is None or not (root / "pyproject.toml").exists():
        raise SystemExit(f"could not find the installed plugin: {installed.strip()}")
    clean = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # ours, not the plugin's
    _run(["uv", "sync", "--frozen", "--no-dev"], cwd=root, env=clean)
    # The first import after an install takes about 20 s on macOS (native libraries scanned once); do it now.
    _run(["uv", "run", "--frozen", "--no-dev", "python", "-c", "import robotforge.mcp_server"], cwd=root, env=clean)
    return home


def run_session(task: Task, task_dir: Path, home: Path, timeout: float) -> dict[str, Any]:
    """One `codex exec` session; the event stream goes to events.jsonl, its stderr to stderr.txt."""
    work, rf_home = task_dir / "work", task_dir / "rf_home"
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(CODEX_HOME=str(home), ROBOTFORGE_HOME=str(rf_home), ROBOTFORGE_NO_BROWSER="1")
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(work), task.text(work)]
    start = time.monotonic()
    timed_out = False
    with (task_dir / "events.jsonl").open("w", encoding="utf-8") as out, (task_dir / "stderr.txt").open("w") as err:
        # stdin from /dev/null: with a terminal or pipe attached, `codex exec` waits for more prompt forever.
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)  # the whole group: Codex and the robotforge server it started
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    meta = {"wall_seconds": round(time.monotonic() - start, 1), "exit_code": proc.returncode, "timed_out": timed_out}
    (task_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def prepare(task: Task, task_dir: Path) -> None:
    (task_dir / "work").mkdir(parents=True)
    (task_dir / "rf_home").mkdir()
    if task.setup:
        task.setup(task_dir / "work", task_dir / "rf_home")


def score(task: Task, task_dir: Path) -> Outcome:
    lines = (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    return judge(task, parse_events(lines), task_dir / "rf_home", task_dir / "work", meta)


# ---- reports ----------------------------------------------------------------------------------------------


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    n = max(len(outcomes), 1)
    return {
        "passed": sum(o.passed for o in outcomes),
        "tasks": len(outcomes),
        "tool_calls": sum(o.tool_calls for o in outcomes),
        "failed_calls": sum(o.failed_calls for o in outcomes),
        "shell_commands": sum(o.shell_commands for o in outcomes),
        "input_tokens": sum(o.input_tokens for o in outcomes),
        "uncached_input_tokens": sum(o.input_tokens - o.cached_input_tokens for o in outcomes),
        "output_tokens": sum(o.output_tokens for o in outcomes),
        "mean_wall_seconds": round(sum(o.wall_seconds for o in outcomes) / n, 1),
    }


def _k(tokens: int) -> str:
    return f"{tokens / 1000:.0f}k"


def table(outcomes: list[dict[str, Any]]) -> str:
    head = f"{'task':15s}{'lang':5s}{'result':7s}{'checks':7s}{'calls':>6s}{'failed':>7s}{'shell':>6s}"
    head += f"{'input':>8s}{'uncached':>9s}{'output':>7s}{'wall s':>7s}"
    rows = [head]
    for o in outcomes:
        ok = sum(c["ok"] for c in o["criteria"])
        result = "PASS" if o["passed"] else ("TIME" if o["timed_out"] else "fail")
        rows.append(
            f"{o['task']:15s}{o['lang']:5s}{result:7s}{f'{ok}/{len(o["criteria"])}':7s}{o['tool_calls']:6d}"
            f"{o['failed_calls']:7d}{o['shell_commands']:6d}{_k(o['input_tokens']):>8s}"
            f"{_k(o['input_tokens'] - o['cached_input_tokens']):>9s}{_k(o['output_tokens']):>7s}"
            f"{o['wall_seconds']:7.0f}"
        )
    return "\n".join(rows)


def details(outcomes: list[dict[str, Any]]) -> str:
    out = []
    for o in outcomes:
        out.append(f"\n{o['task']} ({'pass' if o['passed'] else 'FAIL'}): {' '.join(o['tools'])}")
        out += [f"  [{'x' if c['ok'] else ' '}] {c['name']}: {c['observed']}" for c in o["criteria"]]
        out += [f"  failed {f['tool']} {f['args']}\n    -> {f['error']}" for f in o["failures"]]
        out += [f"  codex: {p}" for p in o["problems"]]
    return "\n".join(out)


def merge(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Several results files as one run: a task in a later file replaces the same task in an earlier one."""
    tasks: dict[str, dict[str, Any]] = {}
    for run in runs:
        tasks.update({o["task"]: o for o in run["tasks"]})

    def joined(key: str) -> str:
        return "+".join(dict.fromkeys(str(r[key]) for r in runs))

    return {**runs[0], "label": joined("label"), "commit": joined("commit"), "tasks": list(tasks.values())}


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Before and after, task by task: result, calls, failed calls, input tokens, wall time."""
    old = {o["task"]: o for o in before["tasks"]}
    rows = [
        f"{before['label']} ({before['commit']}) -> {after['label']} ({after['commit']})",
        f"{'task':15s}{'result':>14s}{'calls':>10s}{'failed':>9s}{'input':>14s}{'uncached':>12s}{'wall s':>11s}",
    ]

    def pair(a: Any, b: Any) -> str:
        return f"{a} -> {b}"

    for new in after["tasks"]:
        o = old.get(new["task"])
        if o is None:
            continue
        verdict = pair("pass" if o["passed"] else "fail", "pass" if new["passed"] else "fail")
        uncached = pair(
            _k(o["input_tokens"] - o["cached_input_tokens"]), _k(new["input_tokens"] - new["cached_input_tokens"])
        )
        rows.append(
            f"{new['task']:15s}{verdict:>14s}{pair(o['tool_calls'], new['tool_calls']):>10s}"
            f"{pair(o['failed_calls'], new['failed_calls']):>9s}"
            f"{pair(_k(o['input_tokens']), _k(new['input_tokens'])):>14s}{uncached:>12s}"
            f"{pair(round(o['wall_seconds']), round(new['wall_seconds'])):>11s}"
        )
    return "\n".join(rows)


def transcript(task_dir: Path) -> str:
    """One session as a reader wants it: what the agent said, every tool call and shell command, in order."""
    out = []
    for line in (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        item = event.get("item") or {}
        if event.get("type") == "turn.completed":
            out.append(f"USAGE {event.get('usage')}")
        if event.get("type") != "item.completed":
            continue
        match item.get("type"):
            case "agent_message":
                out.append("SAY " + textwrap.shorten(item.get("text", ""), 600))
            case "command_execution":
                out.append(f"SHELL {textwrap.shorten(item.get('command', ''), 300)} (exit {item.get('exit_code')})")
            case "mcp_tool_call":
                call = _call(item)
                mark = "TOOL" if call.ok else "FAILED"
                args = textwrap.shorten(json.dumps(call.args, ensure_ascii=False), 400)
                out.append(f"{mark} {call.server}.{call.tool} {args}\n    -> {textwrap.shorten(call.text, 500)}")
    return "\n".join(out)


def report(run: dict[str, Any]) -> str:
    s = run["summary"]
    lines = [
        f"robotforge agent eval {run['label']} ({run['commit']}), {run['model']} {run['effort']}, codex {run['codex']}",
        table(run["tasks"]),
        f"passed {s['passed']}/{s['tasks']}; {s['tool_calls']} robotforge calls, {s['failed_calls']} failed; "
        f"input {_k(s['input_tokens'])} ({_k(s['uncached_input_tokens'])} uncached), output {_k(s['output_tokens'])}",
        details(run["tasks"]),
    ]
    return "\n".join(lines)


# ---- main -------------------------------------------------------------------------------------------------


def evaluate(options: argparse.Namespace) -> dict[str, Any]:
    names = options.only.split(",") if options.only else list(TASKS)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise SystemExit(f"unknown task {', '.join(unknown)}; tasks: {', '.join(TASKS)}")
    jobs = max(1, min(options.jobs, MAX_JOBS))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (options.work or Path(tempfile.gettempdir()) / "robotforge-agent-eval") / stamp
    run_dir.mkdir(parents=True)
    if options.source:
        source, commit = options.source, options.source.rpartition("@")[2] if "@" in options.source else "source"
    else:
        commit = export_ref(options.ref, run_dir / "source")
        source = str(run_dir / "source")
    label = options.label or commit
    task_dirs = {n: run_dir / "tasks" / n for n in names}
    for name, task_dir in task_dirs.items():
        prepare(TASKS[name], task_dir)
    print(f"run folder: {run_dir}", flush=True)
    home = codex_home(run_dir, source, options.model, options.effort, [d / "work" for d in task_dirs.values()])

    def one(name: str) -> Outcome:
        meta = run_session(TASKS[name], task_dirs[name], home, options.timeout)
        outcome = score(TASKS[name], task_dirs[name])
        print(f"  {name}: {'pass' if outcome.passed else 'FAIL'} in {meta['wall_seconds']:.0f} s", flush=True)
        return outcome

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        outcomes = list(pool.map(one, names))
    run = {
        "label": label,
        "commit": commit,
        "started": stamp,
        "codex": _run(["codex", "--version"]).strip().split()[-1],
        "model": options.model,
        "effort": options.effort,
        "max_session_seconds": options.timeout,
        "summary": summarize(outcomes),
        "tasks": [asdict(o) for o in outcomes],
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return run


def rescore(run_dir: Path) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    outcomes = [score(TASKS[o["task"]], run_dir / "tasks" / o["task"]) for o in run["tasks"]]
    run.update(summary=summarize(outcomes), tasks=[asdict(o) for o in outcomes])
    (run_dir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return run


def save(run: dict[str, Any], results: Path) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    path = results / f"{run['started'][:8]}-{run['label']}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated task names (default: all)")
    parser.add_argument("--ref", default="HEAD", help="commit of this repository to evaluate (default HEAD)")
    parser.add_argument("--source", help="a plugin folder or GitHub marketplace (owner/repo[@ref]) instead of --ref")
    parser.add_argument("--label", help="name of the run in the results file (default: the commit)")
    parser.add_argument("--jobs", type=int, default=MAX_JOBS, help=f"sessions at once (at most {MAX_JOBS})")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--effort", default=EFFORT, help="Codex reasoning effort")
    parser.add_argument("--timeout", type=float, default=MAX_SESSION_SECONDS, help="seconds per session")
    parser.add_argument("--work", type=Path, help="where run folders go (default: the system temp folder)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="where the JSON summary goes")
    parser.add_argument("--rescore", type=Path, metavar="RUN_DIR", help="check a finished run again")
    parser.add_argument("--transcript", type=Path, metavar="TASK_DIR", help="print one session")
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="RESULTS",
        help="results files to compare: before, then after (the last one) unless --after names them; several "
        "files are merged, a later file's task replacing an earlier one's (a few tasks run again)",
    )
    parser.add_argument("--after", nargs="+", type=Path, metavar="RESULTS", help="with --compare: the after files")
    parser.add_argument("--list", action="store_true", help="print the tasks")
    options = parser.parse_args(argv)
    if options.list:
        for t in TASKS.values():
            print(f"{t.name:15s}{t.lang:4s}{t.prompt}\n{'':19s}covers: {', '.join(t.covers)}")
        return 0
    if options.transcript:
        print(transcript(options.transcript))
        return 0
    if options.compare:
        before = [json.loads(p.read_text(encoding="utf-8")) for p in options.compare]
        after = [json.loads(p.read_text(encoding="utf-8")) for p in options.after or []]
        if not after and len(before) > 1:
            after = [before.pop()]  # --compare BEFORE AFTER
        if not after:
            raise SystemExit("--compare needs a results file to compare with: --compare BEFORE AFTER")
        print(compare(merge(before), merge(after)))
        return 0
    if shutil.which("codex") is None and not options.rescore:
        raise SystemExit("the codex CLI is not on PATH")
    run = rescore(options.rescore) if options.rescore else evaluate(options)
    print(report(run))
    print(f"\nsaved {save(run, options.results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
