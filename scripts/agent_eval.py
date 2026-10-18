"""A standing evaluation of robotforge with a real agent: Codex does a fixed set of beginner tasks through the
plugin, and a deterministic check reads what really happened.

    uv run python scripts/agent_eval.py [--only NAME,...] [--ref REF | --source SOURCE] [--label TEXT]
    uv run python scripts/agent_eval.py --rescore RUN_DIR          # check a finished run again, no agent
    uv run python scripts/agent_eval.py --transcript RUN_DIR/tasks/NAME   # read one session
    uv run python scripts/agent_eval.py --compare BEFORE.json [MORE.json] [--after AFTER.json ...]
    uv run python scripts/agent_eval.py --fixture RUN_DIR/tasks/NAME tests/fixtures/agent_eval/NAME
    uv run python scripts/agent_eval.py --list

Every task is one `codex exec --json` session with a fresh ROBOTFORGE_HOME, so the worlds it leaves on disk are
its own; at most MAX_JOBS sessions run at once, each for at most MAX_SESSION_SECONDS. The plugin is installed
into a scratch CODEX_HOME from a local marketplace: a `git archive` of --ref (default HEAD), so a run evaluates
exactly one commit; --source takes a plugin folder or a GitHub marketplace (owner/repo or owner/repo@ref)
instead. The installed plugin's environment is built before the first session: the eval measures how an agent
does the tasks, not the cold install (docs/PLUGINS.md measures that). Codex gets no browser, computer use or
ChatGPT apps, which would add tokens and variance that have nothing to do with robotforge.

A task's check reads only what happened: the MCP tool calls and their results in the event stream, and the
worlds saved on disk (world.yaml). It never reads what the agent says it did. Where a robot or an object went is
read from the calls made while its world was open (a dog walked in another world is not the garden's dog). Each
check is a short list of criteria; a task passes when all of them do. Alongside, every session records its
robotforge tool calls, the calls that failed, shell commands, input, cached and output tokens, wall time and the
errors Codex itself logged.

A run writes a JSON summary (no absolute paths, each task's prompt kept) to --results, by default
docs/evals/agent/<date>-<label>.json (with the task names when it did not run them all, never over another file),
for later rounds to compare against; the raw event streams and worlds stay in the run folder under --work.
This is not a test: it needs Codex, a login and the network. tests/test_agent_eval.py checks the criteria on a
recorded stream.
"""

from __future__ import annotations

import argparse
import colorsys
import datetime as dt
import functools
import itertools
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
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
# s between two sessions starting: Codex installs its own skills into CODEX_HOME as a session starts, and two
# starting together broke that install in 3 of about 30 sessions ("failed to install system skills").
START_GAP = 10.0
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

    @functools.cached_property
    def worlds(self) -> tuple[str | None, ...]:
        """For each call, the folder of the world open when it ran, or None (before the first world, or a model
        opened from a file). robotforge's world results name it: new_world, open_world, undo and every edit
        return "world" and "world_dir"; a call that fails leaves the open world as it was."""
        current, out = None, []
        for call in self.calls:
            data = call.data if call.server == SERVER else None
            if isinstance(data, dict):
                if call.tool == "open_model":
                    current = None
                elif isinstance(data.get("world_dir"), str):
                    current = re.split(r"[\\/]", data["world_dir"].rstrip("\\/"))[-1]
                elif isinstance(data.get("world"), str):
                    current = data["world"]
            out.append(current)
        return tuple(out)

    def in_world(self, name: str) -> Session:
        """The session with only the calls made while world `name` (its folder) was open: a dog walked in another
        world is not the garden's dog, though it has the same name."""
        calls = tuple(c for c, w in zip(self.calls, self.worlds, strict=True) if w == name)
        return replace(self, calls=calls)


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
    def compiled(self) -> Any:
        """The world compiled as saved, in a robotforge Playground, or why it does not compile (a string)."""
        from robotforge.playground import Playground

        try:
            playground = Playground.with_demo()
            playground.open_world(str(self.dir))
            return playground
        except Exception as exc:  # a world the agent left broken compiles to nothing; the check says so
            return f"does not compile: {type(exc).__name__}: {exc}"[:SHORT]

    @property
    def kinds(self) -> dict[str, str]:
        """Each robot's kind as robotforge's `robots` tool reports it, from the world compiled as saved."""
        if isinstance(self.compiled, str):
            return {"": self.compiled}
        return {r["name"]: r["kind"] for r in self.compiled.behaviors.robots()}

    @functools.cached_property
    def arms(self) -> dict[str, list[str]]:
        """The arms each robot carries as robotforge's `robots` tool lists them (a centaur's two hands, the arm on
        a Go2's back), from the world compiled as saved; {} when it does not compile."""
        if isinstance(self.compiled, str):
            return {}
        return {r["name"]: list(r.get("arms") or []) for r in self.compiled.behaviors.robots()}

    def outline(self, robot: str) -> tuple[float, float, float, float] | None:
        """The robot seen from above at its start, in its own frame (front +x, left +y): (back, front, right, left)
        in meters from its root body, the box around all its geoms (wheels included). None when it does not
        compile."""
        import mujoco
        import numpy as np

        if isinstance(self.compiled, str):
            return None
        m, d = self.compiled.session.model, self.compiled.session.data
        mujoco.mj_forward(m, d)
        bodies = [
            b
            for b in range(m.nbody)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(f"{robot}/")
        ]
        if not bodies:
            return None
        frame, origin = d.xmat[bodies[0]].reshape(3, 3), d.xpos[bodies[0]]
        corners = []
        for g in (g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies):
            center, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            for sign in itertools.product((-1, 1), repeat=3):
                world = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ (center + np.multiply(sign, half))
                corners.append(frame.T @ (world - origin))
        if not corners:
            return None
        x, y = np.array(corners)[:, 0], np.array(corners)[:, 1]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def load_worlds(home: Path) -> list[World]:
    """The worlds a session left, as robotforge reads them: with the defaults a file leaves out filled in (a bin
    written without a size is 0.2 x 0.15 x 0.08 m), or as plain YAML when robotforge would refuse the file."""
    from robotforge.design.world import World as Saved

    worlds = []
    for path in sorted((home / "worlds").glob("*/world.yaml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        try:
            data = Saved.from_yaml(text).model_dump(mode="json")
        except ValueError:
            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError:
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

    def seen(self, world: World) -> Session:
        """What the session did while `world` was open: where its robots and objects went is read from these calls
        only."""
        return self.session.in_world(world.name)


# ---- where things went --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading_deg: float | None
    upright: bool | None


UPRIGHT_COS = 0.7  # robotforge's pose: upright while the body's up axis is within about 45 degrees of vertical


def _pose(entry: dict[str, Any]) -> Pose | None:
    pos = entry.get("position")
    if not (isinstance(pos, list) and len(pos) >= 2):
        return None
    return Pose(float(pos[0]), float(pos[1]), entry.get("heading_deg"), entry.get("upright"))


def _sensed(position: Any, quat: Any) -> Pose | None:
    """A pose from observe: the free body's position and, when the robot has an IMU, its orientation sensor
    ([w, x, y, z], the base's frame) for the heading and whether it is upright."""
    pose = _pose({"position": position})
    if pose is None or not (isinstance(quat, list) and len(quat) == 4):
        return pose
    w, x, y, z = (float(v) for v in quat)
    heading = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return Pose(pose.x, pose.y, round(heading, 1), 1 - 2 * (x * x + y * y) > UPRIGHT_COS)


def poses(session: Session, robot: str) -> list[Pose]:
    """Where robotforge reported `robot`, in order: before and after every move, in every `robots` list, at the
    end of a run_controller, and as its free body ("<robot>/<body>") in observe and simulate. Move, robots and an
    observe of a robot with an IMU (its orientation sensor) say which way it faces and whether it is upright."""
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
        elif call.tool == "observe":
            quat = (data.get("sensors") or {}).get(f"{robot}/orientation")
            bodies = data.get("free_bodies") or {}
            out += [p for name, pos in bodies.items() if name.startswith(f"{robot}/") and (p := _sensed(pos, quat))]
        elif call.tool == "simulate":
            moved = {name: m.get("to") for name, m in (data.get("moved") or {}).items() if isinstance(m, dict)}
            out += [
                p for name, pos in moved.items() if name.startswith(f"{robot}/") and (p := _pose({"position": pos}))
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


MOVE_DEFAULTS = {"speed": 0.2, "turn": 0.0, "seconds": 3.0}  # robotforge's move tool, for arguments left out


def _sent(speed: float, turn: float, seconds: float, heading_deg: float) -> tuple[float, float] | None:
    """The way a move sends a robot facing heading_deg, as a unit vector in the world: straight ahead (or back, for
    a negative speed), or along the chord of its arc when it turns; None for a turn on the spot or a full circle."""
    angle = turn * seconds
    if abs(angle) < 1e-6:
        dx, dy = speed * seconds, 0.0
    else:
        dx, dy = speed / turn * math.sin(angle), speed / turn * (1.0 - math.cos(angle))
    size, h = math.hypot(dx, dy), math.radians(heading_deg)
    if size < 1e-6:
        return None
    return (dx * math.cos(h) - dy * math.sin(h)) / size, (dx * math.sin(h) + dy * math.cos(h)) / size


def _toward(start: Pose, goal: Any) -> tuple[float, float] | None:
    """The way from `start` to a trip's destination ([x, y], as move(to=...) reports it), as a unit vector."""
    if not (isinstance(goal, list) and len(goal) >= 2):
        return None
    dx, dy = float(goal[0]) - start.x, float(goal[1]) - start.y
    size = math.hypot(dx, dy)
    return (dx / size, dy / size) if size > 1e-6 else None


def progress(session: Session, robot: str, yaw_deg: float = 0.0) -> float:
    """How far `robot` went the way it was sent (meters; negative when it went the other way): each move's
    displacement along the way its speed and turn sent it (a trip, move(to=...), toward its destination), or its
    first to last reported position along the way it faced first (a script or `simulate` can move it too, and
    yaw_deg is how it was saved), whichever is more. A hexapod on weak servos sent forwards went 0.21 m
    backwards; counted as distance, it passed "walk a few steps"."""
    along = 0.0
    for c in session.ours:
        m = c.data
        if c.tool != "move" or not isinstance(m, dict) or (m.get("from") or {}).get("robot") != robot:
            continue
        start, end = _pose(m.get("from") or {}), _pose(m.get("to") or {})
        args = {**MOVE_DEFAULTS, **{k: v for k, v in c.args.items() if k in MOVE_DEFAULTS and v is not None}}
        if start is None or end is None or "distance_m" not in m:
            continue
        if c.args.get("to") is not None:  # it plans its own way there, turns and all: sent toward where it goes
            way = _toward(start, (m.get("destination") or {}).get("at") or (m.get("destination") or {}).get("point"))
        else:
            way = _sent(float(args["speed"]), float(args["turn"]), float(args["seconds"]), start.heading_deg or 0.0)
        along += (end.x - start.x) * way[0] + (end.y - start.y) * way[1] if way else 0.0
    seen = poses(session, robot)
    if not seen:
        return along
    h = math.radians(seen[0].heading_deg if seen[0].heading_deg is not None else yaw_deg)
    return max(along, (seen[-1].x - seen[0].x) * math.cos(h) + (seen[-1].y - seen[0].y) * math.sin(h))


def turned_left(session: Session, robot: str) -> float:
    """Degrees turned to the left over all moves (a right turn counts against it)."""
    return sum(float(m.get("turned_deg") or 0) for m in moves(session, robot))


def upright_at_end(session: Session, robot: str) -> bool | None:
    """The last upright value robotforge reported for `robot` (simulate and run_controller give none)."""
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


# Only robotforge's open container: a "box" shape is a solid cuboid, and a cube left on top of one is not in it.
CONTAINERS = ("bin",)
RIM_SLACK = 0.03  # m above a container's rim that still counts as in it (the object rests on what is inside)


def over(point: list[float], container: dict[str, Any]) -> bool:
    """Is `point` (x, y) within the container's footprint, as world.yaml saves it?"""
    pos, size = container.get("pos") or [0, 0], container.get("size") or [0.05, 0.05, 0.05]
    if len(size) < 2 or len(pos) < 2 or len(point) < 2:
        return False
    yaw = math.radians(float(container.get("yaw") or 0))
    dx, dy = point[0] - pos[0], point[1] - pos[1]
    along = dx * math.cos(yaw) + dy * math.sin(yaw)
    across = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return abs(along) <= size[0] / 2 and abs(across) <= size[1] / 2


def inside(point: list[float], container: dict[str, Any]) -> bool:
    """Is `point` within the container's footprint and, for one standing on the ground, no higher than just above
    its rim? (One given a height, on a table say, is checked by its footprint only.)"""
    size = container.get("size") or [0.05, 0.05, 0.05]
    if len(size) < 3 or len(point) < 3:
        return False
    low = len(container.get("pos") or []) > 2 or point[2] <= size[2] + RIM_SLACK
    return over(point, container) and low


def arm_calls(session: Session, name: str) -> list[tuple[int, str]]:
    """The successful arm calls that could have carried object `name`, with their place among robotforge's calls:
    pick_and_place naming it (as the agent or the world spells it), or closing a gripper by hand (move_arm or
    gripper)."""
    found = []
    for i, c in enumerate(session.ours):
        if not c.ok:
            continue
        named = name in (c.args.get("object"), (c.data or {}).get("object") if isinstance(c.data, dict) else None)
        if c.tool == "pick_and_place" and named:
            found.append((i, "pick_and_place"))
        elif (c.tool == "move_arm" and c.args.get("gripper") == "close") or (
            c.tool == "gripper" and c.args.get("state") == "close"
        ):
            found.append((i, f"{c.tool} close"))
    return found


def arm_moved(session: Session, name: str) -> list[str]:
    """The successful arm calls that could have carried object `name` (arm_calls), by tool."""
    return [what for _, what in arm_calls(session, name)]


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
    name, there = robot["name"], ev.seen(world)
    went = progress(there, name, float(robot.get("yaw") or 0))
    upright = upright_at_end(there, name)
    return [
        Criterion(f"a {kind}", True, _named(world, robot)),
        Criterion(
            f"went >= {at_least} m the way it was sent",
            went >= at_least,
            f"{went:.2f} m ({travelled(there, name):.2f} m in all)",
        ),
        Criterion("upright at the end", upright is True, str(upright)),
    ]


def check_dog_grass(ev: Evidence) -> list[Criterion]:
    rows = []
    for world, robot in ev.robots("quadruped"):
        floor = (world.data.get("floor") or {}).get("style")
        rows.append((robot, [Criterion("on grass", floor == "grass", f"floor {floor}")]))
        rows[-1][1].extend(_walk_criteria(ev, world, robot, "quadruped", 1.0))
    return _best(rows, [Criterion("a quadruped", False, "none in any world")])


def _carried(ev: Evidence, world: World, thing: dict[str, Any], start: list[float] | None = None) -> list[Criterion]:
    """Did an arm carry `thing` into a container? It starts outside every container (at `start` when the task set
    it there), an arm call handled it, and its last reported position is in one."""
    name = thing["name"]
    containers = [o for o in world.objects if o.get("shape") in CONTAINERS and o is not thing and _bigger(o, thing)]
    pos = thing.get("pos") or []
    starts = [o["name"] for o in containers if over(pos, o)]
    if start is not None:
        outside = Criterion(f"{name} starts where it was", pos[:2] == start, f"saved at {pos}")
    else:
        outside = Criterion(f"{name} starts outside the bin", bool(pos) and not starts, f"saved at {pos}")
    handled = arm_moved(ev.seen(world), name)
    seen = object_positions(ev.seen(world), name)
    last = seen[-1] if seen else None
    holding = [o["name"] for o in containers if last and inside(last, o)]
    where = f"last at {last}, bins {[o['name'] for o in containers]}" if last else f"{name} never reported moved"
    return [
        outside,
        Criterion("an arm call handled it", bool(handled), ", ".join(handled) or "no pick_and_place or gripper"),
        Criterion(f"{name} ended in a bin", bool(holding), where),
    ]


def check_arm_cube(ev: Evidence) -> list[Criterion]:
    arms = ev.robots("arm")
    rows = []
    for world, arm in arms:
        reds = [o for o in world.objects if looks(o.get("color"), "red") and o.get("shape") not in ("bin", "table")]
        for red in reds:
            found = [Criterion("an arm", True, _named(world, arm)), Criterion("a red object", True, red["name"])]
            rows.append((red, found + _carried(ev, world, red)))
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
        left = turned_left(ev.seen(world), robot["name"])
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


# "Next to it": at most this far from the rock's side to the nearest point of the rover's outline (its box seen
# from above, wheels included), about one rover length. Measured from the outline, not the center, so a long rover
# that stops with its nose 0.38 m from the rock is as close as a short one would be.
NEAR_ROCK = 0.5
FAR_START = 1.0  # m from the rover's outline at the start: a rock closer than this leaves nothing to drive


def _gap(x: float, y: float, heading_deg: float, outline: tuple[float, ...], rock: dict[str, Any]) -> float:
    """Meters from a robot's outline, placed at (x, y) facing heading_deg, to the side of a rock (world.yaml saves a
    rock's size as [diameter, height]); 0 or less when they touch."""
    back, front, right, left = outline
    dx, dy = rock["pos"][0] - x, rock["pos"][1] - y
    h = math.radians(heading_deg)
    along, across = dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)
    nearest = math.hypot(along - min(max(along, back), front), across - min(max(across, right), left))
    return nearest - float((rock.get("size") or [0.5])[0]) / 2


def _heading_at_end(seen: list[Pose], start: list[float], yaw: float) -> float:
    """Which way the robot faced at its last reported pose: as reported there, else the way it went from its
    start (it drove to the rock), else its saved yaw."""
    end = seen[-1]
    if end.heading_deg is not None:
        return float(end.heading_deg)
    if math.dist((end.x, end.y), start[:2]) > 0.1:
        return math.degrees(math.atan2(end.y - start[1], end.x - start[0]))
    return yaw


def check_rover_rock(ev: Evidence) -> list[Criterion]:
    rows = []
    for world in ev.worlds:
        rocks = [o for o in world.objects if o.get("shape") == "rock" and len(o.get("pos") or []) >= 2]
        there = ev.seen(world)
        for robot in world.robots if rocks else []:
            name = robot.get("name", "")
            seen = poses(there, name)
            if not seen:
                continue
            kind = ev.kind(world, robot)
            outline = world.outline(name) or (0.0, 0.0, 0.0, 0.0)
            start, yaw = robot.get("pos") or [seen[0].x, seen[0].y], float(robot.get("yaw") or 0)
            heading, end = _heading_at_end(seen, start, yaw), seen[-1]
            rock = min(rocks, key=lambda r: _gap(end.x, end.y, heading, outline, r))
            at_end = _gap(end.x, end.y, heading, outline, rock)
            at_start = _gap(start[0], start[1], yaw, outline, rock)
            upright = upright_at_end(there, name)
            center = math.dist((end.x, end.y), rock["pos"][:2]) - float((rock.get("size") or [0.5])[0]) / 2
            rows.append(
                (
                    robot,
                    [
                        Criterion(
                            "a rover and a rock", kind == "rover", f"{_named(world, robot)} ({kind}), {rock['name']}"
                        ),
                        Criterion(
                            f"rock >= {FAR_START} m away at the start", at_start >= FAR_START, f"{at_start:.2f} m"
                        ),
                        Criterion(
                            f"ended within {NEAR_ROCK} m of it",
                            at_end <= NEAR_ROCK,
                            f"{at_end:.2f} m from its outline ({center:.2f} m from its center)",
                        ),
                        Criterion("upright at the end", upright is True, str(upright)),
                    ],
                )
            )
    return _best(rows, [Criterion("a rover and a rock", False, "no robot reported moving in a world with a rock")])


def check_learn_control(ev: Evidence) -> list[Criterion]:
    """Lesson 6 opened, and a balance controller ran live, in the world the learner watches. compare_controllers
    runs off-screen and leaves the live world as it was (an observe after it showed time 0 and the pole untouched),
    so it is counted beside, not as, the run."""
    plans = [r.get("lesson") for r in ev.session.results("lessons")]
    got = sorted({p.get("number") for p in plans if isinstance(p, dict) and p.get("number") is not None})

    def balance(tool: str) -> list[Call]:
        return [c for c in ev.session.ours if c.tool == tool and c.ok and "balance" in str(c.args.get("task", ""))]

    live, table = balance("run_controller"), balance("compare_controllers")
    held = [c.data.get("success") for c in live if isinstance(c.data, dict)]
    return [
        Criterion("opened the control lesson (lessons 6)", 6 in got, f"lessons opened: {got or 'none'}"),
        Criterion(
            "ran a balance controller live",
            bool(live),
            f"{len(live)} run_controller (success {held}), {len(table)} compare_controllers (off-screen)",
        ),
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


WORKSHOP_CUBE = [0.25, 0.1]  # where setup_workshop puts the green cube: the arm has to carry it from there


def check_fix_workshop(ev: Evidence) -> list[Criterion]:
    world = next((w for w in ev.worlds if w.name == "workshop"), None)
    if world is None:
        return [Criterion("the workshop world is still there", False, "gone")]
    ball, cube = world.object("ball"), world.object("green_cube")
    kept = [n for n in ("arm", "green_cube", "ball") if n in {e.get("name") for e in world.robots + world.objects}]
    carried = _carried(ev, world, cube, WORKSHOP_CUBE) if cube else [Criterion("the green cube", False, "gone")]
    return [
        Criterion("the ball is blue", bool(ball) and looks(ball.get("color"), "blue"), str(ball and ball.get("color"))),
        *carried,
        Criterion("arm, cube and ball still there", len(kept) == 3, ", ".join(kept)),
    ]


# A world the user saved earlier and then edited by hand, with the dog's template misspelled: opening it fails the
# same way every time, so the session has to meet the mistake and recover from it.
GARDEN = """\
name: garden
floor:
  style: grass
robots:
- name: dog
  template: quadrupd
  pos: [0.0, 0.0]
objects:
- name: ball
  shape: sphere
  size: [0.12]
  pos: [1.5, 0.8]
  color: orange
"""
GARDEN_WALK = 0.5  # m: "about a meter", at least half of it


def setup_garden(_work: Path, home: Path) -> None:
    (home / "worlds" / "garden").mkdir(parents=True, exist_ok=True)
    (home / "worlds" / "garden" / "world.yaml").write_text(GARDEN, encoding="utf-8")


def check_fix_garden(ev: Evidence) -> list[Criterion]:
    """The garden opens (its dog's template fixed, nothing else lost) and its dog walked while it was open. Meeting
    the mistake is not required: a session that reads the file and fixes it first is the best one, and the failed
    calls are counted beside the verdict anyway."""
    world = next((w for w in ev.worlds if w.name == "garden"), None)
    if world is None:
        return [Criterion("the garden world is still there", False, "gone")]
    there = ev.seen(world)
    dog = next((r for r in world.robots if r.get("name") == "dog"), {})
    walked, upright = progress(there, "dog", float(dog.get("yaw") or 0)), upright_at_end(there, "dog")
    kinds = world.kinds
    return [
        Criterion(
            "garden opens with its dog a quadruped",
            kinds.get("dog") == "quadruped",
            ", ".join(f"{k or 'world'}: {v}" for k, v in kinds.items()) or "no robots",
        ),
        Criterion(
            "the ball is still there",
            world.object("ball") is not None,
            ", ".join(o.get("name", "") for o in world.objects),
        ),
        Criterion(f"the dog walked >= {GARDEN_WALK} m forward in the garden", walked >= GARDEN_WALK, f"{walked:.2f} m"),
        Criterion("upright at the end", upright is True, str(upright)),
    ]


# ---- going somewhere: footprints seen from above ------------------------------------------------------------

ROUND = ("sphere", "cylinder", "pillar", "rock", "tree")  # their footprint is a circle of size[0] across
Corners = list[tuple[float, float]]


def _box_corners(x: float, y: float, heading_deg: float, *sides: float) -> Corners:
    """The corners, in order, of a box whose sides are (back, front, right, left) in its own frame (front +x, left
    +y), placed at (x, y) facing heading_deg."""
    back, front, right, left = sides
    h = math.radians(heading_deg)
    c, s = math.cos(h), math.sin(h)
    ends = ((back, right), (front, right), (front, left), (back, left))
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in ends]


def _footprint(thing: dict[str, Any]) -> Corners | None:
    """The corners of a box-like object (a counter, a bench, a table) seen from above, as world.yaml saves it; None
    for a round one."""
    size, pos = thing.get("size") or [], thing.get("pos") or []
    if thing.get("shape") in ROUND or len(size) < 2 or len(pos) < 2:
        return None
    a, b = float(size[0]) / 2, float(size[1]) / 2
    return _box_corners(float(pos[0]), float(pos[1]), float(thing.get("yaw") or 0), -a, a, -b, b)


def _edges(poly: Corners) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return list(zip(poly, poly[1:] + poly[:1], strict=True))


def _separated(a: Corners, b: Corners) -> bool:
    """Do two convex polygons not overlap? Some edge's normal separates them (a point is a polygon of one)."""
    for (x0, y0), (x1, y1) in _edges(a) + _edges(b):
        nx, ny = y1 - y0, x0 - x1
        pa, pb = [nx * x + ny * y for x, y in a], [nx * x + ny * y for x, y in b]
        if max(pa) < min(pb) - 1e-12 or max(pb) < min(pa) - 1e-12:
            return True
    return False


def _to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ab, ap = (b[0] - a[0], b[1] - a[1]), (p[0] - a[0], p[1] - a[1])
    size = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if size < 1e-12 else min(max((ap[0] * ab[0] + ap[1] * ab[1]) / size, 0.0), 1.0)
    return math.dist(p, (a[0] + t * ab[0], a[1] + t * ab[1]))


def polygon_gap(a: Corners, b: Corners) -> float:
    """Meters between two convex polygons seen from above, 0 when they overlap: apart, the nearest points are a
    corner of one and an edge of the other."""
    if not _separated(a, b):
        return 0.0
    return min(_to_segment(p, *edge) for one, other in ((a, b), (b, a)) for p in one for edge in _edges(other))


def _gap_to(x: float, y: float, heading_deg: float, outline: tuple[float, ...], thing: dict[str, Any]) -> float:
    """Meters from a robot's outline, placed at (x, y) facing heading_deg, to an object's side: its footprint for a
    box-like one (a counter, a bench), its circle for a round one; 0 or less when they touch."""
    corners = _footprint(thing)
    if corners is None:
        return _gap(x, y, heading_deg, outline, thing)
    return polygon_gap(_box_corners(x, y, heading_deg, *outline), corners)


def landing(patch: Any) -> Corners | None:
    """The corners of a ramp's landing, its flat top, as world.yaml saves the ramp: the patch's middle at pos,
    the way up along +x before its yaw, the landing after the slope (between the two of up_down, first for
    down). None for other terrain."""
    if not isinstance(patch, dict) or patch.get("type") != "ramp" or len(patch.get("pos") or []) < 2:
        return None
    length, top, width = (float(patch.get(k) or d) for k, d in (("length", 1.5), ("landing", 0.8), ("width", 1.2)))
    direction = patch.get("direction") or "up_down"
    first = -((2 if direction == "up_down" else 1) * length + top) / 2 + (length if direction != "down" else 0.0)
    x, y = (float(v) for v in patch["pos"][:2])
    return _box_corners(x, y, float(patch.get("yaw") or 0), first, first + top, -width / 2, width / 2)


def _start(robot: dict[str, Any], seen: list[Pose]) -> tuple[float, float, float]:
    """Where a robot started and which way it faced: as saved in world.yaml (an edit of its pos moves its start
    there, so a robot put at the goal by hand made no trip), else its first reported pose."""
    pos = robot.get("pos") or []
    if len(pos) >= 2:
        return float(pos[0]), float(pos[1]), float(robot.get("yaw") or 0)
    return seen[0].x, seen[0].y, float(seen[0].heading_deg or 0)


# Next to something: at most this far from the robot's outline (wheels and feet included) to its side. robotforge's
# move(to=...) stops 0.15 m short of an object and accepts up to 0.3 m more where furniture keeps it off.
NEXT_TO = 0.5
WALKED = 0.3  # m: "walk over to it", from a start not next to it already


def _went_next_to(ev: Evidence, world: World, robot: dict[str, Any], things: list[dict], what: str) -> list[Criterion]:
    """Did `robot` go from a start not next to one of `things` to next to it, and stay up? Its start is where it was
    saved; where it went, the poses robotforge reported while its world was open."""
    name, there = robot.get("name", ""), ev.seen(world)
    if not things:
        return [Criterion(f"{what} in its world", False, f"none in {world.name}")]
    seen = poses(there, name)
    if not seen:
        return [Criterion(f"{name} reported moving", False, f"no pose of {name} reported in {world.name}")]
    outline = world.outline(name) or (0.0, 0.0, 0.0, 0.0)
    x, y, yaw = _start(robot, seen)
    end = seen[-1]
    heading = _heading_at_end(seen, [x, y], yaw)
    thing = min(things, key=lambda t: _gap_to(end.x, end.y, heading, outline, t))
    at_start, at_end = _gap_to(x, y, yaw, outline, thing), _gap_to(end.x, end.y, heading, outline, thing)
    went, upright = travelled(there, name), upright_at_end(there, name)
    return [
        Criterion(
            f"not next to {what} at the start",
            at_start > NEXT_TO,
            f"{at_start:.2f} m from {thing.get('name')}, saved at [{x:.2f}, {y:.2f}]",
        ),
        Criterion(f"went >= {WALKED} m", went >= WALKED, f"{went:.2f} m"),
        Criterion(
            f"ended within {NEXT_TO} m of {what}",
            at_end <= NEXT_TO,
            f"{at_end:.2f} m from {thing.get('name')} (from its outline), at [{end.x:.2f}, {end.y:.2f}]",
        ),
        Criterion("upright at the end", upright is True, str(upright)),
    ]


# "Across the warehouse": at least this far from the dock's top at the start, not from the foot of its ramp. The
# warehouse puts its robot at [-5.5, 2.6], 12 m from it.
DOCK_FAR = 3.0


def check_warehouse_dock(ev: Evidence) -> list[Criterion]:
    """A robot that started across the warehouse ends with its middle on the dock's top (a ramp's landing, as the
    warehouse scene's dock is), upright."""
    rows = []
    for world in ev.worlds:
        tops = [(str(p.get("name")), c) for p in world.data.get("terrain") or [] if (c := landing(p))]
        there = ev.seen(world)
        for robot in world.robots if tops else []:
            name = robot.get("name", "")
            seen = poses(there, name)
            if not seen:
                continue
            kind = ev.kind(world, robot)
            x, y, _ = _start(robot, seen)
            end, upright = seen[-1], upright_at_end(there, name)
            dock, top = min(tops, key=lambda t: polygon_gap([(end.x, end.y)], t[1]))
            far, off = polygon_gap([(x, y)], top), polygon_gap([(end.x, end.y)], top)
            crit = [
                Criterion("a robot that drives or walks, and a dock", kind != "arm", f"{name} ({kind}), {dock}"),
                Criterion(f"started >= {DOCK_FAR} m from the dock's top", far >= DOCK_FAR, f"{far:.2f} m"),
                Criterion("ended on the dock's top", off == 0.0, f"at [{end.x:.2f}, {end.y:.2f}], {off:.2f} m off"),
                Criterion("upright at the end", upright is True, str(upright)),
            ]
            rows.append((robot, crit))
    none = "no robot reported moving in a world with a ramp"
    return _best(rows, [Criterion("a robot that drives or walks, and a dock", False, none)])


def check_centaur_bench(ev: Evidence) -> list[Criterion]:
    """A centaur (built from parts, as a gallery design is saved: four legs on a walking gait and two arms) walked
    in the park from a start away from a bench to next to one."""
    rows = []
    for world in ev.worlds:
        benches = [o for o in world.objects if o.get("shape") == "bench"]
        floor = (world.data.get("floor") or {}).get("style")
        for robot in world.robots:
            if not robot.get("build"):
                continue
            name = robot["name"]
            kind, hands = ev.kind(world, robot), world.arms.get(name, [])
            crit = [
                Criterion(
                    "a centaur: four legs that walk and two arms",
                    kind == "quadruped" and len(hands) >= 2,
                    f"{_named(world, robot)} ({kind}, arms {hands})",
                ),
                Criterion("in a park (grass)", floor == "grass", f"floor {floor}"),
            ]
            rows.append((robot, crit + _went_next_to(ev, world, robot, benches, "a bench")))
    return _best(rows, [Criterion("a centaur: four legs that walk and two arms", False, "no robot built from parts")])


def check_go2_arm_kitchen(ev: Evidence) -> list[Criterion]:
    """A Go2 carrying an arm (a build that mounts the library's Go2 and has an arm, as the go2_arm design is) walked
    in the kitchen from a start away from the island to next to it."""
    from robotforge.design.builder import edit

    rows = []
    for world in ev.worlds:
        islands = [o for o in world.objects if "island" in str(o.get("name"))]
        for robot in world.robots:
            name = robot.get("name", "")
            mounts = [str(m).lower() for m in edit.libraries_in(robot.get("build"))]
            hands = world.arms.get(name, [])
            crit = [
                Criterion(
                    "a Go2 carrying an arm",
                    "go2" in mounts and bool(hands),
                    f"{_named(world, robot)} (mounts {mounts or 'nothing'}, arms {hands})",
                )
            ]
            rows.append((robot, crit + _went_next_to(ev, world, robot, islands, "the island")))
    return _best(rows, [Criterion("a Go2 carrying an arm", False, "no robot in any world")])


# A world the user made: three cubes the same size, named by number, one red; the camera has to say which.
DESK = """\
name: desk
floor:
  style: wood
robots:
- name: arm
  template: arm
  pos: [0.0, 0.0]
objects:
- name: block_1
  shape: box
  size: [0.04, 0.04, 0.04]
  pos: [0.28, 0.16]
  color: blue
- name: block_2
  shape: box
  size: [0.04, 0.04, 0.04]
  pos: [0.3, -0.06]
  color: red
- name: block_3
  shape: box
  size: [0.04, 0.04, 0.04]
  pos: [0.18, -0.24]
  color: green
- name: bin
  shape: bin
  size: [0.16, 0.14, 0.06]
  pos: [-0.05, 0.3]
  color: gray
"""
DESK_RED = "block_2"
DESK_BLOCKS = {"block_1": [0.28, 0.16], "block_2": [0.3, -0.06], "block_3": [0.18, -0.24]}


def setup_desk(_work: Path, home: Path) -> None:
    (home / "worlds" / "desk").mkdir(parents=True, exist_ok=True)
    (home / "worlds" / "desk" / "world.yaml").write_text(DESK, encoding="utf-8")


def found_objects(call: Call) -> list[str]:
    """The objects an observe(find=...) saw in its picture, by name (a find of a robot's part or of something
    fixed names no object)."""
    found = call.data.get("found") if isinstance(call.data, dict) else None
    objects = found.get("objects") if isinstance(found, dict) else None
    return [str(o["object"]) for o in objects or [] if isinstance(o, dict) and o.get("object")]


def check_find_red(ev: Evidence) -> list[Criterion]:
    """The camera found the red cube (observe with find named it) before an arm handled it, and the arm carried it
    into the bin, the other cubes left out of it."""
    world = next((w for w in ev.worlds if w.name == "desk"), None)
    if world is None:
        return [Criterion("the desk world is still there", False, "gone")]
    there = ev.seen(world)
    finds = [(i, c) for i, c in enumerate(there.ours) if c.tool == "observe" and c.ok and c.args.get("find")]
    saw = [i for i, c in finds if DESK_RED in found_objects(c)]
    handled = [i for i, _ in arm_calls(there, DESK_RED)]
    seen = sorted({n for _, c in finds for n in found_objects(c)})
    red = world.object(DESK_RED)
    carried = _carried(ev, world, red, DESK_BLOCKS[DESK_RED]) if red else [Criterion(DESK_RED, False, "gone")]
    bins = [o for o in world.objects if o.get("shape") in CONTAINERS]
    strays = []  # the other cubes that ended in a bin (their last reported place) or are gone
    for name in sorted(DESK_BLOCKS.keys() - {DESK_RED}):
        block = world.object(name)
        last = (object_positions(there, name) or [block.get("pos") or [] if block else []])[-1]
        if block is None or any(inside(last, b) for b in bins):
            strays.append(name if block else f"{name} gone")
    return [
        Criterion(
            f"the camera found {DESK_RED}, the red cube",
            bool(saw),
            f"{len(finds)} observe(find=...): {', '.join(str(c.args.get('find')) for _, c in finds) or 'none'}; "
            f"saw {seen or 'no object'}",
        ),
        Criterion(
            "found before an arm handled it",
            bool(saw) and bool(handled) and saw[0] < handled[0],
            f"first find at call {saw[0] if saw else None}, first arm call at {handled[0] if handled else None}",
        ),
        *carried,
        Criterion("the other cubes are still out of the bin", not strays, ", ".join(strays) or "yes"),
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
            ("edit a world", "pick and place"),
            "Open my workshop world. Paint the ball blue, and have the arm drop the green cube into the bin.",
            check_fix_workshop,
            setup_workshop,
        ),
        Task(
            "fix_garden",
            "en",
            ("open a world", "recover from a mistake", "walk"),
            "Open my garden world and have the dog walk forward about a meter.",
            check_fix_garden,
            setup_garden,
        ),
        Task(
            "warehouse_dock",
            "en",
            ("scene", "go to a place", "ramp"),
            "Load the warehouse scene and send a robot all the way across the warehouse to the loading dock.",
            check_warehouse_dock,
        ),
        Task(
            "centaur_bench",
            "zh",
            ("gallery design", "scene", "go to an object"),
            "我想要一个半人马机器人：四条腿，上身像人，有两只手。把它放到公园里，让它自己走到长椅旁边。",
            check_centaur_bench,
        ),
        Task(
            "find_red",
            "zh",
            ("camera", "find by color", "pick and place"),
            "打开我的 desk 世界，里面有几个颜色不同的方块。用摄像头找出红色的那个，然后让机械臂把它放进盒子里。",
            check_find_red,
            setup_desk,
        ),
        Task(
            "go2_arm_kitchen",
            "en",
            ("gallery design", "real robot", "scene", "go to an object"),
            "Put a Go2 robot dog with an arm on its back in the kitchen, and have it walk over to the island.",
            check_go2_arm_kitchen,
        ),
    )
}

# ---- one task's outcome -----------------------------------------------------------------------------------


@dataclass
class Outcome:
    task: str
    lang: str
    covers: list[str]
    prompt: str | None  # as the session got it ("{work}" for its folder); None for a run that did not record it
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
        for path, short in ((run_dir, "RUN"), (Path.home(), "~")):
            # as written, and as JSON writes it inside arguments (Windows backslashes doubled)
            text = text.replace(str(path), short).replace(json.dumps(str(path))[1:-1], short)
        return _short(_forward_slashes(text))

    return Outcome(
        task=task.name,
        lang=task.lang,
        covers=list(task.covers),
        prompt=meta.get("prompt"),
        passed=bool(criteria) and all(c.ok for c in criteria) and not meta.get("timed_out"),
        criteria=[{**asdict(c), "observed": tidy(c.observed)} for c in criteria],
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
    # A literal TOML key: a Windows path's backslashes would be escapes in a quoted one.
    trusted = "\n".join(f"[projects.'{w}']\ntrust_level = \"trusted\"\n" for w in work_dirs)
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


_starting = threading.Lock()
_last_start = [-math.inf]  # time.monotonic() of the last session's start


def _wait_to_start() -> None:
    """At least START_GAP seconds after the last session started (see START_GAP)."""
    with _starting:
        time.sleep(max(0.0, _last_start[0] + START_GAP - time.monotonic()))
        _last_start[0] = time.monotonic()


def run_session(task: Task, task_dir: Path, home: Path, timeout: float) -> dict[str, Any]:
    """One `codex exec` session; the event stream goes to events.jsonl, its stderr to stderr.txt."""
    _wait_to_start()
    work, rf_home = task_dir / "work", task_dir / "rf_home"
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(CODEX_HOME=str(home), ROBOTFORGE_HOME=str(rf_home), ROBOTFORGE_NO_BROWSER="1")
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(work), task.text(work)]
    start = time.monotonic()
    timed_out = False
    with (task_dir / "events.jsonl").open("wb") as out, (task_dir / "stderr.txt").open("wb") as err:  # raw bytes
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
    meta = {
        "wall_seconds": round(time.monotonic() - start, 1),
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "prompt": task.prompt,
    }
    (task_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def prepare(task: Task, task_dir: Path) -> None:
    (task_dir / "work").mkdir(parents=True)
    (task_dir / "rf_home").mkdir()
    if task.setup:
        task.setup(task_dir / "work", task_dir / "rf_home")


def _events(task_dir: Path) -> list[str]:
    # Written as raw bytes; a stray invalid byte in a tool's output must not lose the session.
    return (task_dir / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()


# Errors Codex logs that leave the session as it would be: its model list not refreshed (it keeps the configured
# model; 4 of the 8 baseline sessions logged it and ran normally). The summary counts the sessions with any other.
HARMLESS_CODEX_ERRORS = ("failed to refresh available models",)


def codex_errors(task_dir: Path) -> list[str]:
    """The errors Codex logged on stderr ("... ERROR codex_...: failed to install system skills ..."): a session
    that started broken shows in the summary instead of passing as one more agent's mistake."""
    try:
        lines = (task_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line.split(" ERROR ", 1)[1] for line in lines if " ERROR " in line]


def score(task: Task, task_dir: Path) -> Outcome:
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    session = parse_events(_events(task_dir))
    session = replace(session, problems=(*session.problems, *codex_errors(task_dir)))
    return judge(task, session, task_dir / "rf_home", task_dir / "work", meta)


def score_safely(task: Task, task_dir: Path) -> Outcome:
    """score, or an outcome that failed with the reason when scoring itself broke: one task's broken files must
    not lose the outcomes of the others (run.json is written once all of them are in)."""
    try:
        return score(task, task_dir)
    except Exception as exc:
        crashed = Criterion("check ran", False, f"scoring failed: {type(exc).__name__}: {exc}"[:SHORT])
        try:
            meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        failed = replace(task, check=lambda _: [crashed])
        return judge(failed, Session((), (), (), {}), task_dir / "rf_home", task_dir / "work", meta)


# ---- reports ----------------------------------------------------------------------------------------------


def summarize(outcomes: list[Outcome]) -> dict[str, Any]:
    n = max(len(outcomes), 1)
    return {
        "passed": sum(o.passed for o in outcomes),
        "tasks": len(outcomes),
        "tool_calls": sum(o.tool_calls for o in outcomes),
        "failed_calls": sum(o.failed_calls for o in outcomes),
        "shell_commands": sum(o.shell_commands for o in outcomes),
        "sessions_with_problems": sum(
            any(not any(h in p for h in HARMLESS_CODEX_ERRORS) for p in o.problems) for o in outcomes
        ),
        "input_tokens": sum(o.input_tokens for o in outcomes),
        "uncached_input_tokens": sum(o.input_tokens - o.cached_input_tokens for o in outcomes),
        "output_tokens": sum(o.output_tokens for o in outcomes),
        "mean_wall_seconds": round(sum(o.wall_seconds for o in outcomes) / n, 1),
    }


def _k(tokens: int) -> str:
    return f"{tokens / 1000:.0f}k"


def table(outcomes: list[dict[str, Any]]) -> str:
    head = f"{'task':16s}{'lang':5s}{'result':7s}{'checks':7s}{'calls':>6s}{'failed':>7s}{'shell':>6s}"
    head += f"{'input':>8s}{'uncached':>9s}{'output':>7s}{'wall s':>7s}"
    rows = [head]
    for o in outcomes:
        ok = sum(c["ok"] for c in o["criteria"])
        result = "PASS" if o["passed"] else ("TIME" if o["timed_out"] else "fail")
        rows.append(
            f"{o['task']:16s}{o['lang']:5s}{result:7s}{f'{ok}/{len(o["criteria"])}':7s}{o['tool_calls']:6d}"
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
    """Before and after, task by task: result, calls, failed calls, input and output tokens, wall time."""
    old = {o["task"]: o for o in before["tasks"]}
    rows = [
        f"{before['label']} ({before['commit']}) -> {after['label']} ({after['commit']})",
        f"{'task':16s}{'result':>14s}{'calls':>10s}{'failed':>9s}{'input':>14s}{'uncached':>12s}{'output':>12s}"
        f"{'wall s':>11s}",
    ]

    def pair(a: Any, b: Any) -> str:
        return f"{a} -> {b}"

    notes = []
    for new in after["tasks"]:
        o = old.get(new["task"])
        if o is None:
            continue
        if o.get("prompt") != new.get("prompt"):  # not the same task any more, or a side did not record it
            said = [p if p is not None else "(not recorded)" for p in (o.get("prompt"), new.get("prompt"))]
            notes.append(f"  {new['task']}: the prompts differ:\n    before: {said[0]}\n    after:  {said[1]}")
        verdict = pair("pass" if o["passed"] else "fail", "pass" if new["passed"] else "fail")
        uncached = pair(
            _k(o["input_tokens"] - o["cached_input_tokens"]), _k(new["input_tokens"] - new["cached_input_tokens"])
        )
        output = pair(_k(o.get("output_tokens", 0)), _k(new.get("output_tokens", 0)))
        rows.append(
            f"{new['task']:16s}{verdict:>14s}{pair(o['tool_calls'], new['tool_calls']):>10s}"
            f"{pair(o['failed_calls'], new['failed_calls']):>9s}"
            f"{pair(_k(o['input_tokens']), _k(new['input_tokens'])):>14s}{uncached:>12s}{output:>12s}"
            f"{pair(round(o['wall_seconds']), round(new['wall_seconds'])):>11s}"
        )
    return "\n".join(rows + (["warning: compared across different prompts", *notes] if notes else []))


# Tools whose whole result a check may read; any other result is cut to SHORT characters in a fixture.
CHECKED_RESULTS = ("move", "robots", "pick_and_place", "simulate", "run_controller", "compare_controllers")


WORLD_KEYS = ("world", "world_dir", "robots", "objects", "model")  # which world is open (Session.worlds)
FOUND_KEYS = ("n", "color", "object", "robot", "fixed", "position", "held_by")  # of each find observe reports


def _trim_result(tool: str, text: str) -> str:
    if tool == "lessons" and text.startswith("{"):
        data = json.loads(text)
        if isinstance(data.get("lesson"), dict):  # the plan is long; the check reads only which lesson it is
            data = {"lesson": {k: data["lesson"].get(k) for k in ("number", "title")}}
        return json.dumps(data, ensure_ascii=False)
    if tool == "observe" and text.startswith("{"):
        data = json.loads(text)  # the image and joint lists go; where things are and which way robots face stay
        kept = {k: data[k] for k in ("time", "free_bodies") if k in data}
        orientation = {k: v for k, v in (data.get("sensors") or {}).items() if k.endswith("/orientation")}
        found = data.get("found")
        if isinstance(found, dict):  # what the camera saw, without the long how-to-read-it text
            seen = [{k: o[k] for k in FOUND_KEYS if k in o} for o in found.get("objects") or [] if isinstance(o, dict)]
            kept["found"] = {**{k: found[k] for k in ("view", "color") if k in found}, "objects": seen}
        return json.dumps({**kept, **({"sensors": orientation} if orientation else {})}, ensure_ascii=False)
    if tool not in CHECKED_RESULTS and text.startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and ("world" in data or "world_dir" in data or tool == "open_model"):
            return json.dumps({k: data[k] for k in WORLD_KEYS if k in data}, ensure_ascii=False)
    if tool not in CHECKED_RESULTS and len(text) > SHORT:
        return text[:SHORT] + "..."
    return text


def _trim_args(args: Any) -> Any:
    """A build's parts lists, which no check reads, become a count; every other argument stays as it was sent."""
    if isinstance(args, dict):
        return {
            k: f"({len(v)} parts trimmed)" if k == "parts" and isinstance(v, list) else _trim_args(v)
            for k, v in args.items()
        }
    return args


def _forward_slashes(text: str) -> str:
    """The rest of a path after RUN or ~ with forward slashes, so a summary reads the same from Windows."""
    return re.sub(r"(RUN|~)((?:\\{1,2}[\w.\-]+)+)", lambda m: m[1] + re.sub(r"\\+", "/", m[2]), text)


def trim(task_dir: Path, dest: Path) -> int:
    """One session as a test fixture (tests/fixtures/agent_eval): its events without images, shell output, long
    messages or results no check reads, the worlds it left, and the run folder's path written as RUN, the home
    folder's as ~. Returns the number of events kept."""
    run_dir = task_dir.parent.parent

    def clean(text: str) -> str:
        for path, short in ((run_dir, "RUN"), (Path.home(), "~")):
            text = text.replace(str(path), short).replace(json.dumps(str(path))[1:-1], short)
        return text

    lines = []
    for line in _events(task_dir):
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        if event.get("type") in ("item.started", "item.updated"):
            continue
        item = event.get("item") or {}
        if event.get("type") == "item.completed":
            match item.get("type"):
                case "command_execution":
                    item["aggregated_output"] = ""
                case "agent_message":
                    item["text"] = str(item.get("text", ""))[:SHORT]
                case "mcp_tool_call":
                    result = item.get("result") if isinstance(item.get("result"), dict) else {}
                    texts = [c for c in result.get("content") or [] if isinstance(c, dict) and c.get("type") == "text"]
                    for c in texts:
                        c["text"] = _trim_result(str(item.get("tool")), str(c.get("text", "")))
                    if result:
                        result["content"] = texts
                    item["arguments"] = _trim_args(item.get("arguments"))
        lines.append(clean(json.dumps(event, ensure_ascii=False)))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = json.loads((task_dir / "meta.json").read_text(encoding="utf-8"))
    (dest / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    for world in (task_dir / "rf_home" / "worlds").glob("*/world.yaml"):
        target = dest / "rf_home" / "worlds" / world.parent.name / "world.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean(world.read_text(encoding="utf-8")), encoding="utf-8")
    return len(lines)


def transcript(task_dir: Path) -> str:
    """One session as a reader wants it: what the agent said, every tool call and shell command, in order."""
    out = []
    for line in _events(task_dir):
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
        f"input {_k(s['input_tokens'])} ({_k(s['uncached_input_tokens'])} uncached), output {_k(s['output_tokens'])}; "
        f"{s.get('sessions_with_problems', 0)} sessions with Codex errors",
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
        try:
            run_session(TASKS[name], task_dirs[name], home, options.timeout)
        except Exception as exc:  # the other sessions go on; this one is scored as far as it got
            print(f"  {name}: the session broke: {type(exc).__name__}: {exc}", flush=True)
        outcome = score_safely(TASKS[name], task_dirs[name])
        print(f"  {name}: {'pass' if outcome.passed else 'FAIL'} in {outcome.wall_seconds:.0f} s", flush=True)
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
    outcomes = [score_safely(TASKS[o["task"]], run_dir / "tasks" / o["task"]) for o in run["tasks"]]
    run.update(summary=summarize(outcomes), tasks=[asdict(o) for o in outcomes])
    (run_dir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return run


def save(run: dict[str, Any], results: Path) -> Path:
    """The results file: --results itself when it names a .json file (written over on purpose, say a rescore),
    else <date>-<label>[-<tasks>].json in that folder, the tasks named when the run did not do them all. A file
    already there is never written over: the start time goes into the name. (A rerun of two tasks on the same
    commit and day once replaced the full run's file.)"""
    if results.suffix == ".json":
        path = results
    else:
        names = [o["task"] for o in run["tasks"]]
        some = "" if set(names) >= set(TASKS) else "-" + "+".join(names)
        path = results / f"{run['started'][:8]}-{run['label']}{some}.json"
        stem, n = f"{path.stem}-{run['started'][9:]}", 1
        while path.exists():
            path, n = path.with_name(f"{stem}{'' if n == 1 else f'-{n}'}.json"), n + 1
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--results", type=Path, default=RESULTS, help="folder for the JSON summary, or a .json file to write over"
    )
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
    parser.add_argument(
        "--fixture", nargs=2, type=Path, metavar=("TASK_DIR", "DEST"), help="trim one session into a test fixture"
    )
    parser.add_argument("--list", action="store_true", help="print the tasks")
    options = parser.parse_args(argv)
    if options.list:
        for t in TASKS.values():
            print(f"{t.name:16s}{t.lang:4s}{t.prompt}\n{'':20s}covers: {', '.join(t.covers)}")
        return 0
    if options.transcript:
        print(transcript(options.transcript))
        return 0
    if options.fixture:
        print(f"{trim(*options.fixture)} events kept in {options.fixture[1]}")
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
