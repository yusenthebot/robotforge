"""Record short demo videos of the real viewer: robots doing things, with the whole page (panels, plots) in view.

    uv run python scripts/demos.py [--only NAME ...] [--out <folder>]

Each demo builds a world, serves it in this process (so the demo can call the robots' behaviors directly: a pick
and place, a trip across a scene), opens it in headless Google Chrome and records the page with the DevTools
screencast while a timeline of actions plays: Python calls on the robots, and page scripts for what a learner does
with the mouse (a drag, a lesson button). ffmpeg turns the frames into an MP4 at 30 frames a second.

Needs Google Chrome (or CHROME=...) and ffmpeg on PATH. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screenshots as S  # noqa: E402  (the world helpers and Chrome finder the screenshots use)

HOST = "127.0.0.1"
FPS = 30
WIDTH, HEIGHT, SCALE = 1920, 1080, 1.0  # CSS px and device pixel ratio (the screencast sends CSS pixels)
READY_TIMEOUT = 90.0  # s for a world to compile, stream in and draw


# ---- a DevTools client that also takes events (the screencast's frames) ------------------------------------


class Cdp:
    def __init__(self, ws: Any) -> None:
        self.ws, self.session, self._id = ws, None, 0
        self._waiting: dict[int, asyncio.Future] = {}
        self.frames: list[tuple[float, bytes]] = []
        self.recording = False

    async def read(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("id") in self._waiting:
                self._waiting.pop(msg["id"]).set_result(msg)
            elif msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                if self.recording:
                    self.frames.append((float(p["metadata"]["timestamp"]), base64.b64decode(p["data"])))
                asyncio.create_task(self.call("Page.screencastFrameAck", sessionId=p["sessionId"]))

    async def call(self, method: str, browser: bool = False, **params: Any) -> dict[str, Any]:
        self._id += 1
        future = asyncio.get_running_loop().create_future()
        self._waiting[self._id] = future
        msg: dict[str, Any] = {"id": self._id, "method": method, "params": params}
        if self.session and not browser:
            msg["sessionId"] = self.session
        await self.ws.send(json.dumps(msg))
        reply = await future
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}")
        return reply.get("result", {})

    async def eval(self, expression: str, wait: bool = True) -> Any:
        out = await self.call("Runtime.evaluate", expression=expression, awaitPromise=wait, returnByValue=True)
        if "exceptionDetails" in out:
            raise RuntimeError(f"page script failed: {out['exceptionDetails']}")
        return out.get("result", {}).get("value")


# ---- demos ---------------------------------------------------------------------------------------------------

Action = Callable[[Any], None] | str  # a call on the Playground, or a page script


@dataclass
class Demo:
    name: str
    world: Callable[[Any], None]
    seconds: float
    timeline: list[tuple[float, Action]] = field(default_factory=list)  # (s after recording starts, action)
    page: str = ""  # a page script run before recording (the camera, a lesson opened)
    settle: float = 1.5  # s between the page drawing and recording
    # Ends the recording `tail` s after this says the story is over (a flight landed), within `seconds`: on a busy
    # machine the simulation runs slower than the wall clock, and a fixed length cut the landing off.
    until: Callable[[Any], bool] | None = None
    tail: float = 1.5


def _follow(robot: str, offset: tuple[float, float, float], ease: float = 0.06) -> str:
    """Page script: keep the camera on `robot` from `offset` metres away, easing after it (runs until replaced)."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge;
        const id = rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic).id;
        const off = {json.dumps(list(offset))};
        (async () => {{
            let at = rf.bodyPosition(id);
            while (window.__cam === me) {{
                const p = rf.bodyPosition(id);
                at = at.map((v, i) => v + {ease} * (p[i] - v));
                rf.lookAt(at.map((v, i) => v + off[i]), [at[0], at[1], at[2] * 0.6]);
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


def _orbit(
    center: tuple[float, float, float], radius: float, height: float, start_deg: float, turn_deg: float, seconds: float
) -> str:
    """Page script: circle the camera around `center`, `turn_deg` in `seconds`."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge, t0 = performance.now(), c = {json.dumps(list(center))};
        (async () => {{
            while (window.__cam === me) {{
                const s = Math.min(1, (performance.now() - t0) / {seconds * 1000});
                const a = ({start_deg} + {turn_deg} * s) * Math.PI / 180;
                rf.lookAt([c[0] + {radius} * Math.cos(a), c[1] + {radius} * Math.sin(a), c[2] + {height}], c);
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


def _drag(robot_body: str, by: tuple[float, float, float], seconds: float, lift: float = 0.04) -> str:
    """Page script: grab a body with the mouse (on its back, `lift` m above its middle) and pull it by `by` m over
    `seconds`, as a learner would, then let go."""
    return f"""
        const rf = window.__robotforge, view = document.getElementById("view");
        const id = rf.scene.bodies.findIndex((b) => b.name === {json.dumps(robot_body)});
        const p0 = rf.bodyPosition(id), by = {json.dumps(list(by))};
        const at = (x, y) => ({{ clientX: x, clientY: y, button: 0, pointerId: 9, bubbles: true }});
        const grab = [p0[0], p0[1], p0[2] + {lift}];
        let [x, y] = rf.screenOfPoint(grab);
        view.dispatchEvent(new PointerEvent("pointerdown", at(x, y)));
        const n = Math.round({seconds} * 30);
        for (let k = 1; k <= n; k++) {{
            [x, y] = rf.screenOfPoint(grab.map((v, i) => v + (by[i] * k) / n));
            view.dispatchEvent(new PointerEvent("pointermove", at(x, y)));
            await new Promise((r) => setTimeout(r, 1000 / 30));
        }}
        view.dispatchEvent(new PointerEvent("pointerup", at(x, y)));
    """


def _chase(robot: str, heading: tuple[float, float], back: float, side: float, up: float) -> str:
    """Page script: a camera behind an aircraft and `side` metres to its left, `up` above, swinging round as it
    turns (its heading is read from how it moves, eased so a corner is a smooth turn of the camera)."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge;
        const id = rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic).id;
        let at = rf.bodyPosition(id), last = at.slice(), dir = {json.dumps(list(heading))}, eye = null;
        (async () => {{
            while (window.__cam === me) {{
                const p = rf.bodyPosition(id);
                const vx = p[0] - last[0], vy = p[1] - last[1];
                last = p;
                const step = Math.hypot(vx, vy);
                if (step > 0.004) {{
                    dir = [dir[0] * 0.97 + (0.03 * vx) / step, dir[1] * 0.97 + (0.03 * vy) / step];
                    const n = Math.hypot(dir[0], dir[1]);
                    dir = [dir[0] / n, dir[1] / n];
                }}
                at = at.map((v, i) => v + 0.12 * (p[i] - v));
                const want = [
                    at[0] - {back} * dir[0] - {side} * dir[1],
                    at[1] - {back} * dir[1] + {side} * dir[0],
                    at[2] + {up},
                ];
                eye = eye ? eye.map((v, i) => v + 0.08 * (want[i] - v)) : want;
                rf.lookAt(eye, [at[0], at[1], at[2] - 0.25]);
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


# Round the park's small tree (tree_6 at 2.43, -4.02 with seed 0) at 2.2 m, from the start spot on the path and back.
SQUARE = [[0.63, -2.22, 2.2], [4.23, -2.22, 2.2], [4.23, -5.82, 2.2], [0.63, -5.82, 2.2], [0.63, -2.22, 2.2]]


# Behind the park's tree_4 (at -3.90, -4.12, its crown 2.1 m up over the spot) and in under its crown, round its
# trunk, to land beside it: one move with four points; the way round the trunk is the planner's.
UNDER_TREE = [[-5.6, -3.0, 1.3], [-5.6, -4.12, 1.3], [-3.3, -4.12, 1.3], [-3.3, -4.12, 0.0]]


def _fly(robot: str, points: list[list[float]], speed: float) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        trip = pg.behaviors.navigate(robot, points, speed)
        print(f"{robot}: flight planned, {trip.plan.length:.1f} m", flush=True)

    return act


def _gallery_moving(pg: Any) -> None:
    S._gallery_line_up(pg)


def _go2_arm_table(pg: Any) -> None:
    """Where its own refusal said the PiPER on its back reaches pointing down: 0.18 to 0.40 m out on either side."""
    pg.add_robot("go2_arm", design="go2_arm")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.05, 0.36], color="red")
    pg.add_object("bin", shape="bin", size=[0.16, 0.16, 0.06], pos=[0.0, -0.36], color="blue")


def _lift_tables(pg: Any) -> None:
    pg.add_robot("picker", design="lift_picker")
    pg.add_object("low", shape="table", size=[0.3, 0.3, 0.45], pos=[0.45, 0.25], fixed=True, color="#9c6b3f")
    pg.add_object("high", shape="table", size=[0.3, 0.3, 0.62], pos=[0.4, -0.27], fixed=True, color="#7a8691")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.42, 0.25, 0.47], color="orange")


def _panda_floor(pg: Any) -> None:
    pg.add_robot("mobile_panda", design="mobile_panda")
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], pos=[0.75, 0.3], color="red")
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[0.7, -0.35], color="blue")


def _rover_rocks(pg: Any) -> None:
    pg.add_robot("mars_rover", design="mars_rover")
    S._rocks(pg)


def _centaur(pg: Any) -> None:
    pg.add_robot("centaur", design="centaur")


def _dogs(pg: Any) -> None:
    pg.add_robot("dog", template="quadruped", pos=[0.0, 0.0])


# Three robots generated from a sentence each, side by side: one drives, two walk (S.GENERATED 1, 0 and 4).
GENERATED_THREE = [("courier", 1, [0.0, 1.15]), ("mantis", 0, [0.0, 0.0]), ("dog", 4, [0.0, -1.1])]


def _generated_three(pg: Any) -> None:
    for name, sentence, pos in GENERATED_THREE:
        pg.add_robot(name, describe=S.GENERATED[sentence], pos=pos)


def _follow_group(robots: list[str], offset: tuple[float, float, float], ease: float = 0.05) -> str:
    """Page script: keep the camera on the middle of these robots from `offset` metres away, easing after it."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge;
        const ids = {json.dumps(robots)}.map(
            (r) => rf.scene.bodies.find((b) => b.name.startsWith(r + "/") && b.dynamic).id);
        const off = {json.dumps(list(offset))};
        const mid = () => {{
            const ps = ids.map((i) => rf.bodyPosition(i));
            return [0, 1, 2].map((k) => ps.reduce((s, p) => s + p[k], 0) / ps.length);
        }};
        (async () => {{
            let at = mid();
            while (window.__cam === me) {{
                const p = mid();
                at = at.map((v, i) => v + {ease} * (p[i] - v));
                rf.lookAt(at.map((v, i) => v + off[i]), [at[0], at[1], 0.15]);
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


def _spliced_walkers(pg: Any) -> None:
    """Three real robots each carrying another: a Go2 with a PiPER, an ANYmal C with a UR5e and its Robotiq gripper, and
    a Spot with a Unitree Z1, in a row."""
    pg.add_robot("go2", library="go2", pos=[-1.6, 0.0], mounts=[{"library": "piper"}])
    pg.add_robot(
        "anymal", library="anymal_c", pos=[0.0, 0.0], mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85"}]
    )
    pg.add_robot("spot", library="spot", pos=[1.6, 0.0], mounts=[{"library": "z1"}])


def _go2_piper_errand(pg: Any) -> None:
    """A Go2 carrying a PiPER, a cube and a bin a few metres ahead, and the errand planned; paused until the recording
    plays it (as for the kitchen's apple)."""
    pg.add_robot("go2", library="go2", pos=[0.0, 0.0], mounts=[{"library": "piper"}])
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.6, 0.8], color="red")
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[1.6, 0.15], color="blue")
    pg.run_fast(lambda: False, 1.0)
    pg.control("pause")
    pg.behaviors.pick_and_place("go2", "cube", "bin")


def _kitchen_fetch(pg: Any) -> None:
    """A mobile Panda in the kitchen, planning to bring the apple from the island to the dining table; the world is
    paused until the recording plays it (the plan takes seconds, and the errand would start while the page loads)."""
    S._kitchen_panda(pg)
    pg.run_fast(lambda: False, 1.0)  # settled on its wheels
    pg.control("pause")
    pg.behaviors.pick_and_place(pg.behaviors.resolve("mm", arm=True), "apple", "dining_table")


def _generated_bring(pg: Any) -> None:
    """A home generated from a sentence and its task: the lift picker brings the red cup from the kitchen to the desk
    in the bedroom; paused until the recording plays it (the errand plans while the page loads)."""
    S._generated_world(S.GENERATED_WORLDS[0])(pg)
    pg.run_fast(lambda: False, 1.0)
    pg.control("pause")
    pg.behaviors.pick_and_place(pg.behaviors.resolve("picker", arm=True), "red_cup", "bedroom_desk")


def _task_done(pg: Any) -> bool:
    return bool(pg.task_status().get("task", {}).get("done"))


def _play(pg: Any) -> None:
    pg.control("play")


def _pick(robot: str, obj: str, target: str) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        pg.behaviors.pick_and_place(robot, obj, target)

    return act


def _move(robot: str, speed: float, turn: float = 0.0) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        pg.behaviors.move(robot, speed, turn)

    return act


def _go(robot: str, to: Any) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        pg.behaviors.navigate(robot, to)

    return act


def _after_trip(robot: str, hold: float, longest: float = 40.0) -> Callable[[Any], None]:
    """An action that waits (the timeline with it) until `robot` is at the end of its trip, then `hold` s more: the
    viewer's clock runs behind the wall clock on a busy machine, and a cut timed in seconds came before the boat
    was at the dock."""

    def act(pg: Any) -> None:
        end = time.monotonic() + longest
        while pg.behaviors.trip(robot) is not None and time.monotonic() < end:
            time.sleep(0.1)
        time.sleep(hold)

    return act


def _dive_then_go(robot: str, z: float, to: list[float]) -> Callable[[Any], None]:
    """An action: the ROV dives straight down where it floats to `z`, then goes on to `to` (the camera crosses the
    surface quickly instead of lingering at it while the ROV glides down a long slope)."""

    def act(pg: Any) -> None:
        x, y = pg.behaviors.pose(robot)["position"][:2]
        pg.behaviors.navigate(robot, [x, y, z])
        _after_trip(robot, 0.3, 12.0)(pg)
        pg.behaviors.navigate(robot, to)

    return act


def _trip_over(robot: str, hold: float) -> Callable[[Any], bool]:
    """For Demo.until: `robot` ended its trip `hold` s ago."""
    since: list[float] = []

    def over(pg: Any) -> bool:
        if pg.behaviors.trip(robot) is not None:
            since.clear()
            return False
        if not since:
            since.append(time.monotonic())
        return time.monotonic() - since[0] >= hold

    return over


def _water_lake(pg: Any) -> None:
    """The lake with its boat at the east end, and an ROV floating near it."""
    S._in_scene("lake", {"boat": {"template": "boat"}, "rov": {"template": "rov"}})(pg)


def _dive_camera(
    robot: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    seconds: float,
    orbit_deg: float,
    orbit_seconds: float,
) -> str:
    """Page script: follow `robot` from `start` metres away, easing to `end` over `seconds` (above the water down
    through the surface with a diving ROV), then circle it by `orbit_deg` over `orbit_seconds`."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge, t0 = performance.now();
        rf.select(null);
        const id = rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic).id;
        const a = {json.dumps(list(start))}, b = {json.dumps(list(end))};
        const smooth = (s) => s * s * (3 - 2 * s);
        (async () => {{
            let at = rf.bodyPosition(id);
            while (window.__cam === me) {{
                const t = (performance.now() - t0) / 1000;
                const p = rf.bodyPosition(id);
                at = at.map((v, i) => v + 0.15 * (p[i] - v));
                const s = smooth(Math.min(1, t / {seconds}));
                let off = a.map((v, i) => v + s * (b[i] - v));
                const turn = Math.max(0, Math.min(1, (t - {seconds}) / {orbit_seconds})) * {orbit_deg} * Math.PI / 180;
                const c = Math.cos(turn), n = Math.sin(turn);
                off = [off[0] * c - off[1] * n, off[0] * n + off[1] * c, off[2]];
                rf.lookAt(at.map((v, i) => v + off[i]), p);  // the camera eases after it; the aim stays on it
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


def _follow_level(robot: str, offset: tuple[float, float, float], ease: float = 0.05) -> str:
    """Page script: keep the camera on `robot` from `offset` metres away, aimed at the robot's own height (a boat on
    the water, an ROV under it: _follow aims lower, at the ground under a walker)."""
    return f"""
        window.__cam = Symbol();
        const me = window.__cam, rf = window.__robotforge;
        rf.select(null);
        const id = rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic).id;
        const off = {json.dumps(list(offset))};
        (async () => {{
            let at = rf.bodyPosition(id);
            while (window.__cam === me) {{
                const p = rf.bodyPosition(id);
                at = at.map((v, i) => v + {ease} * (p[i] - v));
                rf.lookAt(at.map((v, i) => v + off[i]), at);
                await new Promise((r) => requestAnimationFrame(r));
            }}
        }})();
    """


# ---- G10: a quick change of end effectors -----------------------------------------------------------------


def _swap_bench(pg: Any) -> None:
    """A UR5e on the floor with a Robotiq 2F-85, three things to put in a bin: one for each end effector."""
    pg.add_robot("ur5e", library="ur5e")
    pg.mount("robotiq_2f85", "ur5e")
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], pos=[0.5, 0.25], color="orange")
    pg.add_object("ball", shape="sphere", size=[0.05], pos=[0.55, 0.05], color="red")
    pg.add_object("can", shape="cylinder", size=[0.05, 0.08], pos=[0.45, -0.1], color="green")
    pg.add_object("bin", shape="bin", size=[0.2, 0.2, 0.08], pos=[0.3, -0.4], fixed=True, color="blue")


UNSELECT = "window.__robotforge.select(null);"
# The world reloaded after a swap selects its robot again, a moment later: keep it unselected for a while.
KEEP_UNSELECTED = (
    "for (let k = 0; k < 60; k++) { window.__robotforge.select(null); await new Promise((r) => setTimeout(r, 50)); }"
)
_PICKED: list[str] = []  # what the hand swap has put in the bin: the last one ends the film


def _picked(robot: str, obj: str, target: str, longest: float = 40.0) -> Callable[[Any], None]:
    """An action: pick `obj` into `target` and wait (the timeline with it) until the arm is done, in the viewer's real
    time."""

    def act(pg: Any) -> None:
        motion, _ = pg.behaviors.pick_and_place(robot, obj, target)
        end = time.monotonic() + longest
        while not motion.done and time.monotonic() < end:
            time.sleep(0.1)
        time.sleep(0.6)
        _PICKED.append(obj)

    return act


def _swap(robot: str, tool: str) -> Callable[[Any], None]:
    """An action: a quick change of the arm's end effector (mount swaps the one on its flange), then a moment for the
    viewer to show the new one."""

    def act(pg: Any) -> None:
        out = pg.mount(tool, robot)
        print(f"{robot}: {out.get('swapped')} -> {out['mounted']}", flush=True)
        time.sleep(1.2)

    return act


G10_DEMOS = [
    Demo(
        "g10_hand_swap",
        _swap_bench,
        75.0,
        timeline=[
            (1.0, _picked("ur5e", "cube", "bin")),
            (1.05, KEEP_UNSELECTED),  # from just before the swap: no ring round its base
            (1.1, _swap("ur5e", "leap_hand")),
            (1.2, _picked("ur5e", "ball", "bin")),
            (1.25, KEEP_UNSELECTED),
            (1.3, _swap("ur5e", "allegro_hand")),
            (1.4, _picked("ur5e", "can", "bin")),
        ],
        page=UNSELECT + _orbit((0.38, -0.05, 0.32), 1.3, 0.6, -100.0, 60.0, 55.0),
        until=lambda _pg: "can" in _PICKED,
        tail=1.0,
    ),
]


# Legged robots on rough ground (G8): the quadrupeds in their lanes (screenshots.py) and the humanoids walking.


def _humanoids(pg: Any) -> None:
    pg.add_robot("g1", library="g1", pos=[0.0, 0.9])
    pg.add_robot("t1", library="t1", pos=[0.0, -0.4])
    pg.add_robot("op3", library="op3", pos=[0.0, -1.3])


def _humanoids_walk(pg: Any) -> None:
    pg.behaviors.move("g1", 0.45, 0.5)  # a circle of about 0.9 m radius
    pg.behaviors.move("t1", 0.1, 0.0)
    pg.behaviors.move("op3", 0.09, 0.0)


QUADRUPEDS = ("go2", "go1", "a1", "anymal_c", "spot", "barkour")


def _quadrupeds(pg: Any) -> None:
    for i, robot in enumerate(QUADRUPEDS):
        pg.add_robot(robot, library=robot, pos=[0.0, (i - 2.5) * 0.95])


def _quadrupeds_walk(pg: Any) -> None:
    for robot in QUADRUPEDS:
        pg.behaviors.move(robot, 0.3, 0.0)


DEMOS = [
    Demo(  # every library quadruped on the same command (move 0.3 m/s), abreast
        "legged_quadrupeds",
        _quadrupeds,
        14.0,
        timeline=[(0.5, _quadrupeds_walk)],
        page="window.__robotforge.select(null);" + _follow_group(list(QUADRUPEDS), (4.4, 0.4, 1.5), ease=0.05),
        settle=1.0,
    ),
    Demo(
        "gallery_tour",
        _gallery_moving,
        20.0,
        timeline=[
            (1.0, _move("mars_rover", 0.25)),
            (1.0, _move("centaur", 0.2)),
            (1.0, _move("wheel_biped", 0.3, 0.4)),
            (1.0, _move("scorpion", 0.12)),
        ],
        page=_orbit((-0.55, 0.15, 0.3), 3.3, 1.35, -70.0, 75.0, 20.0),
    ),
    Demo(
        "go2_arm_pick",
        _go2_arm_table,
        22.0,
        timeline=[(1.0, _pick("go2_arm", "cube", "bin"))],
        page=_orbit((0.0, 0.0, 0.25), 1.9, 1.05, -35.0, 50.0, 22.0),
    ),
    Demo(
        "lift_picker_tables",
        _lift_tables,
        20.0,
        timeline=[(1.0, _pick("picker", "cube", "high"))],
        page=_orbit((0.3, 0.0, 0.45), 2.0, 0.9, -55.0, 45.0, 20.0),
    ),
    Demo(
        "mobile_panda_pick",
        _panda_floor,
        20.0,
        timeline=[(1.0, _pick("mobile_panda", "cube", "bin"))],
        page=_orbit((0.4, 0.0, 0.35), 2.3, 1.2, -60.0, 50.0, 20.0),
    ),
    Demo(
        "fetch_kitchen_apple",
        _kitchen_fetch,
        58.0,
        timeline=[(0.3, _play)],
        page=_follow("mm", (0.6, -1.0, 3.0), ease=0.04),  # steep: over the island into the aisle behind it
        settle=1.0,
    ),
    Demo(  # a world and a task from one sentence, done start to end: the check says done as the cup lands
        "generated_home_bring",
        _generated_bring,
        90.0,
        timeline=[(0.3, _play)],
        page=_follow("picker", (0.4, -2.2, 3.4), ease=0.04),
        settle=1.0,
        until=_task_done,
        tail=2.0,
    ),
    Demo(
        "nav_warehouse",
        S._in_scene("warehouse", {"go2": {"library": "go2"}}),
        30.0,
        timeline=[(1.0, _go("go2", [6.0, -1.4]))],
        page=_follow("go2", (-2.0, -2.4, 3.0)),
    ),
    Demo(
        "dog_takes_a_pull",
        _dogs,
        16.0,
        timeline=[
            (2.0, _drag("dog/trunk", (0.0, 0.45, 0.0), 2.0)),
            (7.0, _drag("dog/trunk", (0.5, 0.0, 0.0), 2.5)),
            (12.0, _move("dog", 0.3)),
        ],
        page=_follow("dog", (0.75, -1.05, 0.5), ease=0.03),
    ),
    Demo(
        "mars_rover_rocks",
        _rover_rocks,
        18.0,
        timeline=[(1.0, _move("mars_rover", 0.25, 0.15))],
        page=_follow("mars_rover", (1.1, -1.5, 0.8), ease=0.04),
    ),
    Demo(
        "centaur_walk",
        _centaur,
        16.0,
        timeline=[(1.0, _move("centaur", 0.22)), (9.0, _move("centaur", 0.15, 0.6))],
        page=_follow("centaur", (1.2, -1.4, 0.75), ease=0.04),
    ),
    Demo(  # across the lake, between the buoys, to the dock while the crates bob; then the ROV dives
        "water_lake",
        _water_lake,
        45.0,
        timeline=[
            (0.3, _go("boat", "dock")),
            (12.0, _after_trip("boat", 1.2)),  # then the cut, however long the crossing took
            # From the side it goes, south-east: it heads south-west, and the buoys are north of its way.
            (12.01, _dive_camera("rov", (1.2, -1.6, 0.9), (0.9, -0.9, -0.02), 2.5, -25.0, 7.0)),
            (12.02, _dive_then_go("rov", 0.45, [3.0, -0.2, 0.4])),  # clear of the buoys on its way
        ],
        page=_follow_level("boat", (3.0, -3.0, 1.9), ease=0.12),
        settle=1.0,
        until=_trip_over("rov", 1.0),  # and the tail: 2.5 s holding at the goal
        tail=1.5,
    ),
    Demo(  # the lidar on its back and the depth camera on its nose ride along without a wobble or a gap
        "sensors_go2_walk",
        S._sensing_go2((-4.6, 0.0)),  # the open floor between the racks and the pallets
        13.0,
        timeline=[(0.8, _move("go2", 0.45))],
        page=_follow("go2", (0.84, 0.72, 0.5), ease=0.05),
    ),
    Demo(  # what the sensors sense: the 3D lidar's points stream down the aisle, the depth picture rides beside it
        "sensors_go2_aisle",
        S._sensing_go2((-4.8, 2.6)),
        16.0,
        timeline=[(0.8, _go("go2", [1.6, 2.6]))],
        page=_follow("go2", (-1.55, -0.3, 0.95), ease=0.05),
    ),
    Demo(  # take off from the path, fly a square round a tree, come back and land where it started
        "drone_park_square",
        S._in_scene("park", {"drone": {"template": "drone"}}),
        40.0,
        timeline=[(0.3, _fly("drone", [*SQUARE, [0.0, -1.0, 0.0]], 2.0))],
        page=_chase("drone", (0.45, -0.89), 1.5, 0.8, 0.5),
        settle=0.8,
        until=lambda pg: pg.behaviors.trip("drone") is None,  # landed (or stopped)
        tail=1.2,
    ),
    Demo(  # from the path, behind a tree, round its trunk and in under its crown to land beside it
        "drone_round_trunk",
        S._in_scene("park", {"drone": {"template": "drone"}}),
        40.0,
        timeline=[(0.3, _fly("drone", UNDER_TREE, 1.5))],
        page=_chase("drone", (-0.94, -0.34), 1.8, 0.7, 0.1),  # nearly level: the trees and the crown in view
        settle=0.8,
        until=lambda pg: pg.behaviors.trip("drone") is None,
        tail=1.2,
    ),
    Demo(  # the splices walk: three real robots, each wearing another, side by side
        "splice_walkers",
        _spliced_walkers,
        12.0,
        timeline=[(1.0, _move("go2", 0.25)), (1.0, _move("anymal", 0.25)), (1.0, _move("spot", 0.25))],
        page="window.__robotforge.select(null);" + _follow_group(["go2", "anymal", "spot"], (0.3, -3.9, 1.1)),
        settle=1.0,
    ),
    Demo(  # a Go2 with a PiPER on its back walks to a cube, picks it and drops it in the bin
        "splice_go2_pick",
        _go2_piper_errand,
        44.0,
        timeline=[(0.3, _play)],
        page="window.__robotforge.select(null);" + _follow("go2", (1.3, -2.0, 1.1), ease=0.05),
    ),
    Demo(  # the kitchen's arm pulls a drawer open
        "kitchen_drawer",
        lambda pg: pg.load_scene("kitchen"),
        13.0,
        timeline=[(0.5, S._operate("arm", "drawer", "open", 0.2))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.12, -0.82, 1.33], [0.26, -0.02, 0.95]);",
        settle=0.5,
    ),
    Demo(
        "lesson_learn_to_walk",
        lambda pg: None,
        28.0,
        timeline=[(0.5, S._press("train"))],
        page=S._lesson("learn_to_walk"),
        settle=0.5,
    ),
    Demo(  # four quadrupeds up and down 8 cm stairs they see, each in its own lane
        "legged_stairs",
        S._lanes("stairs", S.LANE_STAIRS, 2.0),
        17.0,
        timeline=[(0.5, S._move_all(0.3))],
        page="window.__robotforge.select(null);" + _follow("a1", (-2.4, -3.3, 1.9), ease=0.04),
        settle=1.0,
    ),
    Demo(  # and across 3.5 cm gravel
        "legged_gravel",
        S._lanes("rough", S.LANE_GRAVEL, 3.0, style="rock"),
        15.0,
        timeline=[(0.5, S._move_all(0.35))],
        page="window.__robotforge.select(null);" + _follow("a1", (-2.4, -3.3, 1.9), ease=0.04),
        settle=1.0,
    ),
    Demo(  # the humanoids: G1 on its learned walk round a circle, T1 and OP3 on the balance controller
        "legged_humanoids",
        _humanoids,
        16.0,
        timeline=[(0.5, _humanoids_walk)],
        page="window.__robotforge.select(null);" + _orbit((0.3, 0.7, 0.5), 5.0, 1.4, 15.0, -30.0, 16.0),
        settle=1.0,
    ),
    # The factory: the belt brings boxes to the stop, the camera says their colors, the suction cup sorts them (the
    # sorter of scripts/factory_bench.py, in real time). Close on the arm, so the cup is seen taking hold.
    Demo(
        "factory_sorting",
        lambda pg: pg.load_scene("factory", options={"boxes": 12}),
        42.0,
        timeline=[(0.2, S._sorting)],
        page="window.__robotforge.select(null);" + _orbit((0.3, -0.26, 0.6), 0.78, 0.52, -66.0, 30.0, 42.0),
        settle=0.5,
    ),
    Demo(
        "generated_three",
        _generated_three,
        15.0,
        timeline=[(0.5, _move(name, 0.25)) for name, *_ in GENERATED_THREE],
        page="window.__robotforge.select(null);" + _follow_group([n for n, *_ in GENERATED_THREE], (1.35, -2.4, 1.0)),
        settle=1.0,
    ),
    *G10_DEMOS,
]


# ---- recording -----------------------------------------------------------------------------------------------


def _serve(pg: Any, port: int) -> None:
    from robotforge import cli

    threading.Thread(target=lambda: asyncio.run(cli.serve(pg, HOST, port, False)), daemon=True).start()
    S._wait_http(f"http://{HOST}:{port}/", READY_TIMEOUT)


async def _record(demo: Demo, pg: Any, port: int, chrome_port: int) -> list[tuple[float, bytes]]:
    import websockets

    version = json.loads(S._wait_http(f"http://{HOST}:{chrome_port}/json/version", 20))
    async with websockets.connect(version["webSocketDebuggerUrl"], max_size=None) as ws:
        cdp = Cdp(ws)
        reader = asyncio.create_task(cdp.read())
        target = (await cdp.call("Target.createTarget", url="about:blank"))["targetId"]
        cdp.session = (await cdp.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"]
        await cdp.call(
            "Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=SCALE, mobile=False
        )
        await cdp.call("Page.enable")
        await cdp.call("Page.bringToFront")
        await cdp.call("Emulation.setFocusEmulationEnabled", enabled=True)
        stored = {"robotforge.coached": "1", "robotforge.world": "0"}
        settings = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in stored.items())
        await cdp.call("Page.addScriptToEvaluateOnNewDocument", source=f"try {{ {settings} }} catch (e) {{}}")
        await cdp.call("Page.navigate", url=f"http://{HOST}:{port}/?lang=en")  # the promo is in English
        end = time.monotonic() + READY_TIMEOUT
        while not await cdp.eval(
            "!!(window.__robotforge && window.__robotforge.scene && window.__robotforge.time > 0)"
        ):
            if time.monotonic() > end:
                raise TimeoutError(f"{demo.name}: the viewer never drew its scene")
            await asyncio.sleep(0.25)
        if demo.page:
            await cdp.eval(f"(async () => {{ {demo.page} }})()")
        await asyncio.sleep(demo.settle)
        await cdp.call(
            "Page.startScreencast",
            format="jpeg",
            quality=88,
            maxWidth=int(WIDTH * SCALE),
            maxHeight=int(HEIGHT * SCALE),
            everyNthFrame=1,
        )
        cdp.recording = True
        t0 = time.monotonic()
        pending = sorted(demo.timeline, key=lambda e: e[0])
        end = t0 + demo.seconds
        while time.monotonic() < end:
            if demo.until is not None and not pending and end - time.monotonic() > demo.tail and demo.until(pg):
                end = time.monotonic() + demo.tail
            while pending and time.monotonic() - t0 >= pending[0][0]:
                _, action = pending.pop(0)
                if isinstance(action, str):
                    asyncio.create_task(cdp.eval(f"(async () => {{ {action} }})()"))  # runs on while we record
                else:
                    try:
                        await asyncio.to_thread(action, pg)
                    except Exception as exc:  # a refused action is part of the demo's story: say it, record on
                        print(f"{demo.name}: {getattr(action, '__name__', 'action')} refused: {exc}", flush=True)
            await asyncio.sleep(0.02)
        cdp.recording = False
        await cdp.call("Page.stopScreencast")
        await cdp.call("Target.closeTarget", targetId=target, browser=True)
        reader.cancel()
        return cdp.frames


def _encode(frames: list[tuple[float, bytes]], out: Path) -> None:
    """The frames at their own times, resampled to FPS, as an H.264 MP4 that plays everywhere."""
    if len(frames) < 2:
        raise RuntimeError(f"{out.name}: only {len(frames)} frames were recorded")
    with tempfile.TemporaryDirectory(prefix="rf-demo-") as tmp:
        lines = ["ffconcat version 1.0"]
        for k, (t, jpeg) in enumerate(frames):
            name = f"f{k:05d}.jpg"
            (Path(tmp) / name).write_bytes(jpeg)
            gap = (frames[k + 1][0] - t) if k + 1 < len(frames) else 1 / FPS
            lines += [f"file '{name}'", f"duration {max(gap, 0.001):.4f}"]
        lines.append(f"file 'f{len(frames) - 1:05d}.jpg'")  # the concat demuxer drops the last duration otherwise
        (Path(tmp) / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(Path(tmp) / "list.txt"),
                "-fps_mode",
                "cfr",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags",
                "+faststart",
                str(out),
            ],
            check=True,
        )


def take(demo: Demo, out_dir: Path) -> Path:
    home = Path(tempfile.mkdtemp(prefix="rf-demo-home-"))
    os.environ["ROBOTFORGE_HOME"] = str(home)  # the real Menagerie models (downloaded once), not the test stand-ins
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world(demo.name)
    demo.world(pg)
    port, chrome_port = S._free_port(), S._free_port()
    _serve(pg, port)
    chrome = subprocess.Popen(
        [
            S._chrome(),
            "--headless=new",
            f"--remote-debugging-port={chrome_port}",
            f"--user-data-dir={home / 'chrome'}",
            "--lang=zh-CN",
            "--hide-scrollbars",
            "--no-first-run",
            "--use-angle=metal" if sys.platform == "darwin" else "--use-gl=angle",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = out_dir / f"{demo.name}.mp4"
    try:
        frames = asyncio.run(_record(demo, pg, port, chrome_port))
        _encode(frames, out)
    finally:
        chrome.terminate()
        shutil.rmtree(home, ignore_errors=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="names of the demos to record")
    parser.add_argument("--out", type=Path, default=Path("demos"))
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is not on PATH")
    args.out.mkdir(parents=True, exist_ok=True)
    for demo in DEMOS:
        if args.only and demo.name not in args.only:
            continue
        t0 = time.monotonic()
        path = take(demo, args.out)
        print(f"{demo.name}: {path} ({path.stat().st_size / 1e6:.1f} MB, {time.monotonic() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
