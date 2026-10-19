"""Take the README's screenshots of the real viewer, the same way every time.

    uv run python scripts/screenshots.py [--only NAME ...] [--out docs/images] [--lang zh-CN]

Each shot builds a world in a fresh ROBOTFORGE_HOME, serves it with `robotforge open`, puts it in the
state the picture needs (a lesson started, knobs set, buttons pressed, a robot walking) over the viewer's
own WebSocket, then drives a headless Google Chrome over the DevTools protocol: it waits until the scene
has streamed in and the page is idle, optionally runs a little JavaScript (open a panel, place the camera)
and captures the page. Virtual time cannot be used for this: it runs ahead of the WebSocket.

Needs Google Chrome (or CHROME=/path/to/chrome) and the `websockets` package, which comes with the
viewer's server. Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/combos.py
CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "/usr/bin/google-chrome",
)
WIDTH, HEIGHT, SCALE = 1280, 800, 2  # CSS px and device pixel ratio
READY_TIMEOUT = 60.0  # s for a world to compile, stream in and draw


@dataclass(frozen=True)
class Shot:
    name: str
    world: Callable[[Any], None]  # builds the world on a Playground
    viewer: list[dict[str, Any]] = field(default_factory=list)  # WebSocket messages, or {"wait": s}
    page: str = ""  # JavaScript run in the page before the capture
    # The page's remembered settings: a returning visitor, so no first-visit hint and no scene panel unless asked
    storage: dict[str, str] = field(default_factory=lambda: {"robotforge.coached": "1", "robotforge.world": "0"})
    settle: float = 2.0  # s after the page is ready
    size: tuple[int, int] = (WIDTH, HEIGHT)  # CSS px
    # Python calls on the robots, (s after the page has drawn, call on the Playground), for a picture mid-way through
    # a behavior: a shot with actions serves its world in this process (as scripts/demos.py does), so the calls act on
    # the very simulation the page shows, and is captured `settle` s after the last one.
    actions: list[tuple[float, Callable[[Any], None]]] = field(default_factory=list)
    # Python calls on the robots while the page watches (a lift, so a plot has it in its history), run before the
    # page script; a shot with one serves its world in this process instead of with `robotforge open`.
    act: Callable[[Any], None] | None = None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _chrome() -> str:
    for path in (os.environ.get("CHROME"), *CHROME_PATHS, shutil.which("google-chrome"), shutil.which("chromium")):
        if path and Path(path).exists():
            return path
    raise SystemExit("Google Chrome not found; set CHROME=/path/to/chrome")


def _wait_http(url: str, timeout: float) -> bytes:
    """The body of `url` once it answers."""
    end = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.read()
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(0.3)


class Page:
    """Chrome over the DevTools protocol: browser-level calls without a session, a tab's with its flat session."""

    def __init__(self, ws: Any, session: str | None = None) -> None:
        self.ws, self.session, self._id = ws, session, 0

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        self._id += 1
        msg: dict[str, Any] = {"id": self._id, "method": method, "params": params}
        if self.session:
            msg["sessionId"] = self.session
        await self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(await self.ws.recv())
            if reply.get("id") == self._id and reply.get("sessionId") == self.session:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})

    async def eval(self, expression: str) -> Any:
        out = await self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in out:
            raise RuntimeError(f"page script failed: {out['exceptionDetails']}")
        return out.get("result", {}).get("value")


async def _drive_viewer(port: int, messages: list[dict[str, Any]]) -> None:
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
        actuators: dict[str, int] = {}
        if any("actuator" in msg for msg in messages):  # by name (see _pose): the scene says each one's id
            while not actuators:
                raw = await ws.recv()
                if isinstance(raw, str) and json.loads(raw).get("type") == "scene":
                    actuators = {a["name"]: a["id"] for a in json.loads(raw)["actuators"]}
        for k, msg in enumerate(messages):
            if "wait" in msg:
                await asyncio.sleep(msg["wait"])
                continue
            if "actuator" in msg:
                msg = {"type": "ctrl", "id": actuators[msg["actuator"]], "value": msg["value"]}
            await ws.send(json.dumps({"id": f"shot{k}", **msg}))
            if msg.get("type") in ("lesson_open", "go_to", "pick_and_place"):  # a slow request: wait for its reply
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):  # body poses stream as binary frames
                        continue
                    reply = json.loads(raw)
                    if reply.get("type") == "reply" and reply.get("id") == f"shot{k}":
                        if not reply.get("ok"):
                            raise RuntimeError(f"viewer refused {msg}: {reply.get('message')}")
                        break


async def _act(shot: Shot, pg: Any) -> None:
    """Run a shot's Python calls on the served playground at their times."""
    t0 = time.monotonic()
    for at, action in sorted(shot.actions, key=lambda a: a[0]):
        await asyncio.sleep(max(0.0, at - (time.monotonic() - t0)))
        await asyncio.to_thread(action, pg)


async def _capture(chrome_port: int, port: int, shot: Shot, out: Path, lang: str, pg: Any = None) -> None:
    """Open the viewer in a tab, wait until it has drawn, put the world in the shot's state while the page
    watches (so plots have their history), and capture it."""
    import websockets

    version = json.loads(_wait_http(f"http://127.0.0.1:{chrome_port}/json/version", 20))
    async with websockets.connect(version["webSocketDebuggerUrl"], max_size=None) as ws:
        browser = Page(ws)
        target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
        session = (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"]
        page = Page(ws, session)
        width, height = shot.size
        await page.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=SCALE,
            mobile=width < 600,
        )
        await page.call("Page.enable")  # scripts for new documents need the Page domain on
        # A background tab gets its animation frames throttled, and the lesson plots draw on them
        await page.call("Page.bringToFront")
        await page.call("Emulation.setFocusEmulationEnabled", enabled=True)
        settings = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in shot.storage.items())
        await page.call("Page.addScriptToEvaluateOnNewDocument", source=f"try {{ {settings} }} catch (e) {{}}")
        # The page's language from the URL: headless Chrome on a Mac keeps the system's, whatever --lang says.
        await page.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang={lang}")
        end = time.monotonic() + READY_TIMEOUT
        drawn = "!!(window.__robotforge && window.__robotforge.scene && window.__robotforge.time > 0)"
        while not await page.eval(drawn):
            if time.monotonic() > end:
                raise TimeoutError(f"{shot.name}: the viewer never drew its scene")
            await asyncio.sleep(0.25)
        if shot.viewer:
            await _drive_viewer(port, shot.viewer)
        if shot.act is not None:
            await asyncio.to_thread(shot.act, pg)
        if shot.actions:
            if shot.page:  # the camera first, so the page watches the whole behavior
                await page.eval(f"(async () => {{ {shot.page} }})()")
            await _act(shot, pg)
        await asyncio.sleep(shot.settle)
        if shot.page and not shot.actions:
            await page.eval(f"(async () => {{ {shot.page} }})()")
            await asyncio.sleep(1.0)
        data = (await page.call("Page.captureScreenshot", format="png", captureBeyondViewport=False))["data"]
        _save(base64.b64decode(data), out)
        await browser.call("Target.closeTarget", targetId=target)


SAVED_WIDTH = 1600  # px: sharp on a high-density screen, small enough for the repository
JPEG_QUALITY = 84


def _save(png: bytes, out: Path) -> None:
    """Keep the capture as a JPEG SAVED_WIDTH wide (a 2560 px PNG of the viewer is 3 to 5 MB)."""
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    height = round(image.height * SAVED_WIDTH / image.width)
    image.resize((SAVED_WIDTH, height), Image.Resampling.LANCZOS).save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)


def _serve_here(pg: Any, port: int) -> None:
    """Serve the playground's world from this process, for a shot that acts on its robots."""
    import threading

    from robotforge import cli

    threading.Thread(target=lambda: asyncio.run(cli.serve(pg, "127.0.0.1", port, False)), daemon=True).start()
    _wait_http(f"http://127.0.0.1:{port}/", READY_TIMEOUT)


def take(shot: Shot, out_dir: Path, lang: str) -> Path:
    home = Path(tempfile.mkdtemp(prefix="rf-shot-"))
    os.environ["ROBOTFORGE_HOME"] = str(home)  # the real Menagerie models (downloaded once), not the test stand-ins
    env = dict(os.environ)
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world(shot.name)
    shot.world(pg)
    world_dir = home / "worlds" / shot.name
    port, chrome_port = _free_port(), _free_port()
    server = None
    if shot.act is not None or shot.actions:  # served here, so the shot's calls act on what the page shows
        _serve_here(pg, port)
    else:
        server = subprocess.Popen(
            [sys.executable, "-m", "robotforge", "open", str(world_dir), "--no-browser", "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    chrome = subprocess.Popen(
        [
            _chrome(),
            "--headless=new",
            f"--remote-debugging-port={chrome_port}",
            f"--user-data-dir={home / 'chrome'}",
            f"--lang={lang}",
            "--hide-scrollbars",
            "--no-first-run",
            "--use-angle=metal" if sys.platform == "darwin" else "--use-gl=angle",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out = out_dir / f"{shot.name}.jpg"
    try:
        _wait_http(f"http://127.0.0.1:{port}/", READY_TIMEOUT)
        asyncio.run(_capture(chrome_port, port, shot, out, lang, pg))
    finally:
        for proc in (chrome, server):
            if proc is None:
                continue
            proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)
        shutil.rmtree(home, ignore_errors=True)
    return out


# ---- the shots ---------------------------------------------------------------------------------------------


def _park(pg: Any) -> None:
    pg.add_robot("go2", library="go2", pos=[0.0, 0.35])
    pg.add_robot("dog", template="quadruped", pos=[0.2, -0.35])
    pg.load_scene("park")  # the scene puts kept robots on its free spots


# Lessons are played in the page, the way a learner does: the page that starts a lesson is the one that shows it.
_SLEEP = "await new Promise((r) => setTimeout(r, {ms}));"


def _lesson(lesson: str, *steps: str) -> str:
    """Page script: open Learn, start `lesson`, then run the steps (_press, _wait)."""
    start = f"""
        document.getElementById("learn-toggle").click();
        {_SLEEP.format(ms=500)}
        document.querySelector('[data-lesson="{lesson}"]').click();
        for (let k = 0; k < 100 && !document.getElementById("lesson-panel").checkVisibility(); k++) {{
            {_SLEEP.format(ms=100)}
        }}
        {_SLEEP.format(ms=1500)}
    """
    return start + "\n".join(steps)


def _press(control: str) -> str:
    return f"document.querySelector('button[data-control=\"{control}\"]').click();"


def _wait(seconds: float) -> str:
    return _SLEEP.format(ms=round(seconds * 1000))


def _walk(robot: str, speed: float) -> dict[str, Any]:
    return {"type": "teleop", "robot": robot, "speed": speed, "turn": 0.0}


# A rover built from parts that carries an arm (tests/test_builder.py builds the same one).
MOBILE_ARM = {
    "color": "yellow",
    "motor": "jga25",
    "base": {"shape": "box", "size": [0.34, 0.22, 0.06], "mass": 3.0, "sensors": ["imu"]},
    "parts": [
        {"name": "left_wheel", "at": [0, 0.13, -0.01], "shape": "wheel", "size": [0.1, 0.03], "mirror": True},
        {"name": "front_caster", "at": [0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "rear_caster", "at": [-0.14, 0, -0.04], "shape": "caster", "size": [0.04]},
        {"name": "arm", "at": "top", "shift": [0.06, 0, 0], "robot": {"template": "arm", "params": {"reach": 0.45}}},
    ],
}


def _hexacopter() -> dict[str, Any]:
    from robotforge.design.builder.guide import EXAMPLE_HEXACOPTER  # the builder guide's own example

    return EXAMPLE_HEXACOPTER


HEXACOPTER = _hexacopter()


DRONE_ROW = [("quad", -1.6), ("hexa_t", -0.7), ("octo", 0.3), ("hexa", 1.35), ("x2", 2.4), ("cf", 3.15)]


def _drone_family(pg: Any) -> None:
    sources = {
        "quad": {"template": "drone"},
        "hexa_t": {"template": "drone", "params": {"rotors": 6, "color": "orange", "arm_length": 0.24}},
        "octo": {"template": "drone", "params": {"rotors": 8, "color": "black", "arm_length": 0.3, "size": 0.2}},
        "hexa": {"build": HEXACOPTER},
        "x2": {"library": "skydio_x2"},
        "cf": {"library": "crazyflie"},
    }
    for name, x in DRONE_ROW:
        pg.add_robot(name, pos=[x, 0.0], **sources[name])
    pg.set_floor("grass")


def _in_scene(scene: str, robots: dict[str, dict[str, Any]], seed: int = 0) -> Callable[[Any], None]:
    """A world builder: these robots, then the scene, which puts them on its free spots."""

    def build(pg: Any) -> None:
        for name, source in robots.items():
            pg.add_robot(name, **source)
        pg.load_scene(scene, seed=seed)

    return build


def _go(robot: str, to: list[float] | str) -> dict[str, Any]:
    """Send a robot somewhere along a planned path (move with to=...)."""
    return {"type": "go_to", "robot": robot, "to": to}


def _fetch(robot: str, obj: str, target: list[float] | str) -> dict[str, Any]:
    """Have a robot with an arm fetch something: it goes over, picks it up, carries it and puts it down."""
    return {"type": "pick_and_place", "robot": robot, "object": obj, "target": target}


def _kitchen_panda(pg: Any) -> None:
    """The kitchen with a mobile Panda where its starter arm would stand on the floor (the first ground place)."""
    pg.load_scene("kitchen", starter=False)
    pg.add_robot("mm", design="mobile_panda", pos=[1.3, -1.0], yaw=90)


def _warehouse_picker(pg: Any) -> None:
    """The warehouse (seed 0) with a lift picker at its third ground place, south of the racks."""
    pg.load_scene("warehouse", starter=False)
    pg.add_robot("picker", design="lift_picker", pos=[2.0, -0.4], yaw=0)


def _when(body: str, condition: str, timeout: float = 150.0) -> str:
    """Page script: wait until `condition` (JavaScript on p, the body's position) holds, as the world goes on."""
    return f"""
        {{
            const rf = window.__robotforge, id = rf.scene.bodies.findIndex((b) => b.name === {json.dumps(body)});
            const end = performance.now() + {timeout * 1000};
            while (performance.now() < end) {{
                const p = rf.bodyPosition(id);
                if ({condition}) break;
                {_SLEEP.format(ms=100)}
            }}
        }}
    """


def _look_at(robots: list[str], offset: tuple[float, float, float]) -> str:
    """Page script: aim the camera at the middle of these robots, from `offset` metres away."""
    return f"""
        const rf = window.__robotforge;
        const at = {json.dumps(robots)}.map((r) => rf.bodyPosition(
            rf.scene.bodies.find((b) => b.name.startsWith(r + "/") && b.dynamic).id));
        const mid = [0, 1, 2].map((i) => at.reduce((s, p) => s + p[i], 0) / at.length);
        rf.lookAt(mid.map((v, i) => v + {json.dumps(list(offset))}[i]), mid);
    """


def _look_between(robot: str, body: str, offset: tuple[float, float, float]) -> str:
    """Page script: aim the camera at the middle of a robot's base and a body (at the body's height), from `offset`
    metres away."""
    return f"""
        {{
            const rf = window.__robotforge, mine = (b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic;
            const r = rf.bodyPosition(rf.scene.bodies.find(mine).id);
            const o = rf.bodyPosition(rf.scene.bodies.findIndex((b) => b.name === {json.dumps(body)}));
            const mid = [(r[0] + o[0]) / 2, (r[1] + o[1]) / 2, o[2] * 0.85];
            rf.lookAt(mid.map((v, i) => v + {json.dumps(list(offset))}[i]), mid);
        }}
    """


def _hover_look(robot: str, offset: tuple[float, float, float], drop: float) -> str:
    """Page script: look at a hovering aircraft from `offset` metres away, aiming `drop` metres below it, so its
    shadow on the ground shows how high it flies. Nothing is selected: the selection ring lies on the ground right
    under the aircraft, far below it in the picture."""
    return f"""
        const rf = window.__robotforge;
        rf.select(null);
        const p = rf.bodyPosition(rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")})).id);
        const off = {json.dumps(list(offset))};
        rf.lookAt(p.map((v, i) => v + off[i]), [p[0], p[1], p[2] - {drop}]);
    """


A1_URDF = Path(
    os.environ.get(
        "A1_URDF", "~/worktrees/_artifacts/import-doctor/repos/unitree_ros/robots/a1_description/urdf/a1.urdf"
    )
).expanduser()  # a checkout of unitree_ros; the shots that import it are skipped without one


def _a1(pg: Any) -> None:
    pg.add_robot("a1", file=str(A1_URDF))


def _nothing(pg: Any) -> None:
    """Lessons build their own world."""


def _slide(control: str, value: float) -> str:
    """Page script: move a lesson slider as a learner's drag ends."""
    return (
        f"{{ const i = document.querySelector('input[data-control=\"{control}\"]'); i.value = {value}; "
        "i.dispatchEvent(new Event('input')); }"
    )


def _tick(control: str) -> str:
    return f"document.querySelector('input[data-control=\"{control}\"]').click();"


def _choose(control: str, option: int) -> str:
    return f"document.querySelector('[data-choice=\"{control}\"] button:nth-child({option + 1})').click();"


def _answer(option: int) -> str:
    return f"document.querySelector('#lesson-quiz-options button:nth-child({option + 1})').click();"


_NEXT = "document.getElementById('lesson-next').click();"


def _drag(body: str, to: tuple[float, float, float]) -> str:
    """Page script: grab a body with the mouse and pull it to `to` (world m), in small moves, and let go."""
    return f"""
        {{
            const rf = window.__robotforge;
            const view = document.getElementById("view");
            const id = rf.scene.bodies.findIndex((b) => b.name === {json.dumps(body)});
            const [x0, y0] = rf.screenOf(id);
            const [x1, y1] = rf.screenOfPoint({json.dumps(list(to))});
            const at = (x, y) => ({{ clientX: x, clientY: y, button: 0, pointerId: 7, bubbles: true }});
            view.dispatchEvent(new PointerEvent("pointerdown", at(x0, y0)));
            for (let k = 1; k <= 12; k++) {{
                const [x, y] = [x0 + ((x1 - x0) * k) / 12, y0 + ((y1 - y0) * k) / 12];
                view.dispatchEvent(new PointerEvent("pointermove", at(x, y)));
                {_SLEEP.format(ms=60)}
            }}
            {_SLEEP.format(ms=400)}
            view.dispatchEvent(new PointerEvent("pointerup", at(x1, y1)));
        }}
    """


# The seeing lesson, played up to each picture: the red cube dragged, the mask narrowed, the height guessed.
_SEEING_MASK = (_drag("red_cube", (0.2, -0.12, 0.03)), _wait(1.5), _NEXT, _wait(1.5))
_SEEING_WHERE = (*_SEEING_MASK, _slide("width", 15), _wait(1.0), _NEXT, _wait(0.5), _answer(0), _wait(1.0))
_SEEING_LIGHT = (
    *_SEEING_WHERE,
    _slide("height", 2),
    _wait(1.0),
    _NEXT,
    _wait(0.5),
    _press("pick"),
    _wait(12.0),
    _NEXT,
    _wait(0.5),
    _press("test"),
    _wait(1.0),
    _tick("depth"),
    _wait(0.5),
    _press("test"),
    _wait(1.0),
    _NEXT,
    _wait(0.5),
    _choose("light", 1),
    _wait(1.5),
)

# The imitation lesson, played in the page as a learner plays it: the sliders moved a centimetre at a time (as a
# drag moves them), the grip toggle clicked, the guesses made, the buttons pressed, Next after every step.
_TEACH = """
    const rf = window.__robotforge;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const live = () => rf.lesson().live;
    const passed = (id) => rf.lesson().passed.includes(id);
    const until = async (ok, seconds) => {
        for (let k = 0; k < seconds * 20 && !ok(); k++) await sleep(50);
    };
    const control = (id) => document.querySelector(`#lesson-controls [data-control="${id}"]`);
    const press = (id) => document.querySelector(`button[data-control="${id}"]`).click();
    const choose = (id, k) => document.querySelector(`[data-choice="${id}"]`).children[k].click();
    const answer = (k) => document.querySelectorAll(".quiz-option")[k].click();
    const next = async () => { document.getElementById("lesson-next").click(); await sleep(400); };
    const slide = async (goal) => {
        const ids = Object.keys(goal);
        const start = Object.fromEntries(ids.map((id) => [id, Number(control(id).value)]));
        const n = Math.max(1, Math.ceil(Math.hypot(...ids.map((id) => goal[id] - start[id])) / 0.01));
        for (let i = 1; i <= n; i++) {
            for (const id of ids) {
                control(id).value = start[id] + ((goal[id] - start[id]) * i) / n;
                control(id).dispatchEvent(new Event("input"));
            }
            await sleep(50);
        }
        await until(() => Math.hypot(...live().hand.map((v, j) => v - live().target[j])) < 0.012, 3);
    };
    const grip = async () => { control("grip").click(); await sleep(600); };
    const demo = async () => {
        const before = rf.lesson().results.demos ?? 0;
        const [x, y] = live().cube;
        await slide({ tx: x, ty: y, tz: 0.1 });
        await slide({ tz: 0.025 });
        await sleep(400);
        await grip();
        await slide({ tz: 0.1 });
        await slide({ tx: 0.05, ty: 0.28 });  // over the bin (learn/cloning.py BIN_XY)
        await sleep(400);
        await grip();
        await until(() => (rf.lesson().results.demos ?? 0) > before, 4);
    };
"""


def _imitation(until: str, then: str = "") -> str:
    """Page script: play the imitation lesson up to the step `until`, its experiment done, then `then`."""
    plays = {
        "show": "await demo(); await until(() => passed('show'), 3);",
        "copy": "await next(); answer(0); press('try'); await until(() => passed('copy'), 20);",
        "far": "await next(); answer(1); choose('where', 1); press('try'); await until(() => passed('far'), 20);",
        "more": """await next();
            while ((rf.lesson().results.demos ?? 0) < 5) { press('new_cube'); await sleep(400); await demo(); }
            await until(() => passed('more'), 3);""",
        "test": "await next(); answer(2); press('test'); await until(() => passed('test'), 60);",
        "network": """await next(); answer(1); choose('method', 1); await sleep(300); press('test');
            await until(() => passed('network'), 90);""",
        "how_many": """await next(); answer(1); await sleep(300);
            document.querySelector('.sweep-buttons button').click(); await until(() => passed('how_many'), 180);""",
        "correct": """await next(); answer(1); choose('method', 0); choose('where', 1); await sleep(300);
            press('try'); await sleep(2000); press('take_over'); await sleep(300);
            if (control('grip').checked) await grip();
            await demo(); press('try'); await until(() => passed('correct'), 30);""",
        "ik": """await next(); answer(1); choose('method', 2); await sleep(300); press('test');
            await until(() => passed('ik'), 90);""",
    }
    steps = list(plays)
    return _lesson("imitation", _TEACH, *(plays[s] for s in steps[: steps.index(until) + 1]), then)


# The first demonstration caught mid-way: the cube in the gripper on its way to the bin, the panel scrolled to
# the pair being recorded and the sliders.
_CARRYING = """
    const [x, y] = live().cube;
    await slide({ tx: x, ty: y, tz: 0.1 });
    await slide({ tz: 0.025 });
    await sleep(400);
    await grip();
    await slide({ tz: 0.1 });
    await slide({ tx: (x + 0.05) / 2, ty: (y + 0.28) / 2 });
    document.getElementById("lesson-panel").scrollTop = 1e6;
"""


# A try caught mid-way, the panel on what it sees and does right now (its header said "the controller right now"
# over the learner's recorded pairs): fail the shot if the header does not say whose numbers they are.
_TRYING = """
    await demo();
    await until(() => passed('show'), 3);
    await next();
    answer(0);
    press('try');
    await sleep(700);  // on its way down to the cube (copying beside a demonstration has it in the bin by 2 s)
    document.getElementById("lesson-formula").scrollIntoView({ block: "center" });
    const head = document.querySelector("#lesson-formula .section-title").textContent;
    if (!/sees and does now|此刻看到/.test(head)) throw new Error(`the formula's header reads ${head}`);
"""


# W A S D in the imitation lesson go the way they point on screen (in the world's axes A moved the ball right from
# this camera): hold A, then W, and fail the shot if the ball went another way.
_KEYS = """
    await sleep(800);
    const hold = async (key) => {
        const at = rf.screenOfPoint(live().target);
        window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
        await sleep(600);
        window.dispatchEvent(new KeyboardEvent("keyup", { key, bubbles: true }));
        await sleep(400);
        const to = rf.screenOfPoint(live().target);
        return [to[0] - at[0], to[1] - at[1]];
    };
    const [ax, ay] = await hold("a");
    const [wx, wy] = await hold("w");
    if (!(ax < -20 && Math.abs(ay) < -ax / 2)) throw new Error(`A moved the ball ${ax}, ${ay} px: not left`);
    if (!(wy < -10 && Math.abs(wx) < -wy / 2)) throw new Error(`W moved the ball ${wx}, ${wy} px: not up`);
"""

# Dragging the ball in the imitation lesson: it slides level, so a drag to the point over the cube (at the ball's
# height) lands right over it, and a Shift-drag lowers it there (on the plane facing the camera it ended 4 cm off).
_DRAG = """
    await sleep(800);
    const canvas = document.querySelector("canvas");
    const ev = (type, x, y, el, shiftKey) => el.dispatchEvent(new PointerEvent(type, { clientX: x, clientY: y,
        button: 0, buttons: type === "pointerup" ? 0 : 1, pointerId: 1, bubbles: true, isPrimary: true, shiftKey }));
    const drag = async (to, shift) => {
        const at = rf.screenOfPoint(live().target);
        ev("pointerdown", at[0], at[1], canvas, shift);
        for (let k = 1; k <= 30; k++) {
            ev("pointermove", at[0] + ((to[0] - at[0]) * k) / 30, at[1] + ((to[1] - at[1]) * k) / 30, window, shift);
            await sleep(40);
        }
        ev("pointerup", to[0], to[1], window, shift);
        await sleep(2500);
        return [...live().target];
    };
    const [x, y, z] = live().cube;
    const over = await drag(rf.screenOfPoint([x, y, live().target[2]]), false);
    const down = await drag(rf.screenOfPoint([x, y, z + 0.005]), true);
    const off = (p) => Math.hypot(p[0] - x, p[1] - y);
    if (off(over) > 0.005) throw new Error(`the drag ended ${off(over)} m off the cube across the desk`);
    if (off(down) > 0.005 || down[2] - z > 0.015) throw new Error(`the Shift-drag ended at ${down}, cube ${[x, y, z]}`);
"""


def _mars_rover(pg: Any) -> None:
    """The rover an agent built in scripts/agent_eval.py's rover_rock task: 0.5 m of solid aluminum on 9 cm
    wheels and 12 V gearmotors, with the rock it had to reach."""
    params = {"color": "white", "length": 0.5, "width": 0.36, "wheel_radius": 0.09, "motor": "jga25"}
    pg.add_robot("mars_rover", template="rover", params={**params, "material": "aluminum"})
    pg.add_object("big_rock", shape="rock", size=[1.2, 0.85], pos=[3.0, 0.0], color="#875c46", fixed=True)


def _built_robots() -> Any:
    """The robots built from parts that the look tests use (tests/built_robots.py, plain data)."""
    sys.path.insert(0, str(ROOT / "tests"))
    import built_robots

    return built_robots


def _built(names: tuple[str, ...], spacing: float = 0.62) -> Callable[[Any], None]:
    """A world with some of the look tests' robots built from parts, in a row along y."""

    def world(pg: Any) -> None:
        gallery = _built_robots().GALLERY
        for i, name in enumerate(names):
            pg.add_robot(name, build=gallery[name], pos=[0.0, spacing * (i - (len(names) - 1) / 2)])

    return world


def _design(name: str, extras: Callable[[Any], None] | None = None) -> Callable[[Any], None]:
    """A world with one gallery design in it (named after the design), and what it works with."""

    def world(pg: Any) -> None:
        pg.add_robot(name, design=name)
        if extras is not None:
            extras(pg)

    return world


def _materials(pg: Any) -> None:
    """One small robot four times: printed PLA, anodised aluminium, carbon fibre and rubber."""
    probe = _built_robots().MATERIAL_PROBE
    for i, material in enumerate(("pla", "aluminum", "carbon", "rubber")):
        pg.add_robot(material, build={**probe, "material": material}, pos=[0.0, 0.3 * i - 0.45])


def _twin_table(pg: Any) -> None:
    pg.add_object("table", shape="table", size=[0.3, 0.7, 0.35], pos=[0.42, 0.0], fixed=True, color="#9c6b3f")
    pg.add_object("red", shape="box", size=[0.04, 0.04, 0.04], pos=[0.35, -0.15, 0.37])
    pg.add_object("blue", shape="box", size=[0.04, 0.04, 0.04], pos=[0.35, 0.15, 0.37], color="blue")


def _rocks(pg: Any) -> None:
    pg.set_terrain("rough", params={"size": [6, 6], "height": 0.04})


def _sensing_go2(pos: tuple[float, float] = (-3.2, 2.6)) -> Callable[[Any], None]:
    """A Go2 in the warehouse with a 3D lidar on its back, a depth camera on its nose and touch pads under its
    feet, each added in one call as an agent would (add_sensor)."""

    def build(pg: Any) -> None:
        pg.load_scene("warehouse", keep_robots=False, starter=False)
        pg.add_robot("go2", library="go2", pos=list(pos))
        pg.add_sensor("go2", {"type": "lidar3d", "rate_hz": 10})
        pg.add_sensor("go2", {"type": "camera", "mode": "depth"})
        for foot in ("FL", "FR", "RL", "RR"):
            pg.add_sensor("go2", {"type": "touch", "on": foot, "rows": 2, "cols": 2})

    return build


def _extend_plugins(pg: Any) -> None:
    """A Go2 among signposts and a red drum, with a Geiger sensor on its back: the signpost and the sensor are
    plugins, the very files docs/EXTENDING.md shows, trusted in the world's plugins folder as write_script does."""
    import re

    from robotforge.extend import REGISTRY

    page = (ROOT / "docs" / "EXTENDING.md").read_text(encoding="utf-8")
    folder = pg.store.dir / "plugins"
    folder.mkdir(exist_ok=True)
    for name in ("signpost", "geiger"):
        code = re.search(rf'```python title="plugins/{name}.py"\n(.*?)```', page, re.S)[1]
        (folder / f"{name}.py").write_text(code, encoding="utf-8", newline="\n")
    REGISTRY.trust(pg.store.dir)
    pg.set_floor("concrete")
    pg.add_robot("go2", library="go2", pos=[0.0, 0.0])
    pg.add_object("hot", shape="cylinder", size=[0.4, 0.7], color="red", fixed=True, pos=[1.6, 0.35])
    for k, (x, y, yaw, height) in enumerate([(0.7, 1.1, 15, 1.0), (2.6, -0.5, -30, 1.4), (1.9, 1.6, 200, 1.2)]):
        pg.add_object(f"sign{k + 1}", shape="signpost", size=[height], pos=[x, y], yaw=yaw)
    pg.add_sensor("go2", {"type": "geiger", "source": "hot", "radius": 3.0})


EXTEND_VIEW = (
    'document.getElementById("drawer-toggle").click();'
    "await new Promise((r) => setTimeout(r, 400));" + "window.__robotforge.lookAt([-0.3, -3.0, 1.2], [1.3, 0.4, 0.5]);"
)


def _panda_wrist(pg: Any) -> None:
    """A Panda with a force-torque sensor at its wrist and a 100 g cube on the floor in front of it."""
    pg.add_robot("panda", library="panda")
    pg.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], mass=0.1, pos=[0.5, 0.0], color="red")
    pg.add_sensor("panda", {"type": "force_torque", "on": "hand", "name": "wrist"})


def _lift_cube(pg: Any) -> None:
    """The Panda takes the cube off the floor and holds it up, in real time while the page watches: the wrist's
    force plot has the 100 g coming on in its history."""
    b = pg.behaviors

    def finish(motion: Any, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not motion.done and time.monotonic() < end:
            time.sleep(0.05)

    finish(b.move_arm("panda", [0.5, 0.0, 0.25], "open"), 15)
    finish(b.move_arm("panda", [0.5, 0.0, 0.03], None), 15)
    finish(b.set_gripper("panda", "close"), 5)
    finish(b.move_arm("panda", [0.5, 0.0, 0.3], None), 15)
    time.sleep(1.5)


LIDAR_PUCK = {"name": "lidar", "at": "top", "shape": "cylinder", "size": [0.06, 0.04], "joint": "fixed"}


def _on_plates(pg: Any) -> None:
    """A template dog, a template rover and a Go2, each given a lidar puck with add_part, which carries a robot that
    is not built from parts on a plate and puts the part there (design/builder/plate.py)."""
    for name, source, y in (("dog", {"template": "quadruped"}, 0.55), ("rover", {"template": "rover"}, 0.0)):
        pg.add_robot(name, **source, pos=[0.0, y])
    pg.add_robot("go2", library="go2", pos=[0.0, -0.75])
    for robot in pg.current_world().robots:
        pg.add_part(robot.name, {**LIDAR_PUCK, "sensors": [{"type": "lidar", "at": "top"}]})


def _kitchen_starts(pg: Any) -> None:
    """Robots added without pos after load_scene: each starts on the first free one of the kitchen's robot_places,
    clear of the island and of each other (a rover added after the Go2 used to go under its belly)."""
    pg.load_scene("kitchen")
    pg.add_robot("dog", design="go2_arm")
    pg.add_robot("rover", template="rover")
    pg.add_robot("pup", template="quadruped")


def _gallery_line_up(pg: Any) -> None:
    """Several designs side by side, for an overview picture."""
    for name, pos in (
        ("mars_rover", [0.15, 1.4]),
        ("centaur", [0.0, 0.4]),
        ("wheel_biped", [0.0, -0.3]),
        ("scorpion", [0.0, -0.95]),
        ("lift_picker", [-0.9, 0.9]),
        ("mobile_panda", [-2.0, -1.0]),
        ("twin_arm", [-0.9, -0.2]),
        ("tentacle", [-0.9, -1.0]),
    ):
        pg.add_robot(name, design=name, pos=pos)


SENSOR_WRIST_VIEW = _look_at(["panda"], (1.05, -0.95, 0.35))
SENSOR_CLOUD_VIEW = "window.__robotforge.lookAt([-5.0, 2.5, 1.5], [-2.5, 2.6, 0.25]);"  # down the aisle, from behind
SENSOR_DRAWER_VIEW = (
    'document.getElementById("drawer-toggle").click();'
    "await new Promise((r) => setTimeout(r, 400));" + _look_at(["go2"], (1.25, -0.95, 0.7))
)

# One picture per gallery design: (design, extra world, camera offset from the robot in metres).
GALLERY_SHOTS: list[tuple[str, Callable[[Any], None] | None, tuple[float, float, float]]] = [
    ("wheel_biped", None, (0.75, -0.55, 0.3)),
    ("lift_picker", None, (1.6, -1.25, 1.05)),
    ("mars_rover", _rocks, (0.85, -0.7, 0.45)),
    ("wheel_dog", None, (0.8, -0.6, 0.35)),
    ("centaur", None, (0.9, -0.7, 0.45)),
    ("scorpion", None, (0.75, -0.6, 0.45)),
    ("go2_arm", None, (1.1, -0.85, 0.55)),
    ("mobile_panda", None, (1.9, -1.45, 1.1)),
    ("twin_arm", _twin_table, (1.25, -0.9, 0.55)),
    ("tentacle", None, (0.95, -0.75, 0.45)),
]


def _look(eye: tuple[float, float, float], target: tuple[float, float, float]) -> str:
    """Page script: a fixed camera, eye and target in world meters (for robots bolted to the floor too)."""
    return f"window.__robotforge.lookAt({json.dumps(list(eye))}, {json.dumps(list(target))});"


def _new_walkers(pg: Any) -> None:
    """The library's newer dogs and humanoids side by side."""
    for i, name in enumerate(["anymal_c", "barkour", "op3", "t1"]):
        pg.add_robot(name, library=name, pos=[0.0, 1.95 - 1.3 * i])


def _new_arms(pg: Any) -> None:
    """The library's newer arms, the UR5e with a Robotiq gripper on its flange."""
    pg.add_robot("z1", library="z1", pos=[0.0, 1.6])
    pg.add_robot("ur5e", library="ur5e", pos=[0.0, 0.55], mounts=[{"library": "robotiq_2f85"}])
    pg.add_robot("xarm7", library="xarm7", pos=[0.0, -0.45])
    pg.add_robot("aloha", library="aloha", pos=[0.0, -1.6])


def _dogs_with_arms(pg: Any) -> None:
    """A Go2 carrying an AgileX PiPER and a Spot carrying a Unitree Z1, by a cube and a bin."""
    pg.add_robot("go2", library="go2", pos=[0.0, 0.5], mounts=[{"library": "piper"}])
    pg.add_robot("spot", library="spot", pos=[0.0, -0.7], mounts=[{"library": "z1"}])
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[1.6, 0.8], color="red")
    pg.add_object("bin", shape="bin", size=[0.18, 0.18, 0.08], pos=[1.6, 0.15], color="blue")


def _more_carriers(pg: Any) -> None:
    """Robots carrying robots beyond the Go2: an ANYmal C and a Barkour with PiPERs, a Booster T1 with an SO-100 on its
    head, and a UR5e wearing a Robotiq gripper."""
    pg.add_robot("anymal", library="anymal_c", pos=[0.0, 1.5], mounts=[{"library": "piper"}])
    pg.add_robot("barkour", library="barkour", pos=[0.0, 0.35], mounts=[{"library": "piper"}])
    pg.add_robot("t1", library="t1", pos=[0.0, -0.6], mounts=[{"library": "so_arm100"}])
    pg.add_robot("ur5e", library="ur5e", pos=[0.0, -1.6], mounts=[{"library": "robotiq_2f85"}])


def _barkour_so100(pg: Any) -> None:
    """A Barkour carrying an SO-100: with no flat spot on its back, the arm's base rests on the handle."""
    pg.add_robot("barkour", library="barkour", mounts=[{"library": "so_arm100"}])


def _go2_two_arms(pg: Any) -> None:
    """A Go2 carrying an AgileX PiPER and an SO-100 side by side (the second moved back clear of the first)."""
    pg.add_robot("go2", library="go2", mounts=[{"library": "piper"}, {"name": "arm2", "library": "so_arm100"}])


def _spot_z1(pg: Any) -> None:
    """A Spot carrying a Unitree Z1 beside the bump on its back."""
    pg.add_robot("spot", library="spot", mounts=[{"library": "z1"}])


def _anymal_ur5e(pg: Any) -> None:
    """An ANYmal C carrying a UR5e that wears a Robotiq 2F-85 on its flange: three robots in one."""
    pg.add_robot("anymal", library="anymal_c", mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85"}])


# The library's newer robots for a line-up, standing: (name, library, mounts, [x, y]). The tall ones stand behind,
# in the gaps between the short ones.
LINE_UP = [
    ("barkour", "barkour", [], [0.55, 1.75]),
    ("op3", "op3", [], [0.55, 0.6]),
    ("z1", "z1", [], [0.55, -0.4]),
    ("anymal", "anymal_c", [], [0.55, -1.75]),
    ("t1", "t1", [], [-0.75, 2.85]),
    ("ur5e", "ur5e", [{"library": "robotiq_2f85"}], [-0.75, 1.15]),
    ("xarm7", "xarm7", [], [-0.75, 0.1]),
    ("aloha", "aloha", [], [-0.75, -1.05]),
]


def _line_up(pg: Any) -> None:
    """The library's newer robots standing in two rows: walkers in front, arms behind."""
    for name, model, mounts, pos in LINE_UP:
        pg.add_robot(name, library=model, pos=pos, mounts=mounts)


def _framed(robot: str, offset: tuple[float, float, float], lift: float) -> str:
    """Page script: nothing selected, the camera `offset` metres from a point `lift` metres above `robot`'s body."""
    return f"""
        const rf = window.__robotforge;
        rf.select(null);
        const body = rf.scene.bodies.find((b) => b.name.startsWith({json.dumps(robot + "/")}) && b.dynamic);
        const at = rf.bodyPosition(body.id);
        const target = [at[0], at[1], at[2] + {lift}];
        rf.lookAt(target.map((v, i) => v + {json.dumps(list(offset))}[i]), target);
    """


def _complex_splices(pg: Any) -> None:
    """Three robots spliced from three or more: an ANYmal C carrying a UR5e that wears a Robotiq gripper, a Go2 with a
    PiPER and an SO-100 side by side, and a Spot with a Z1 and a PiPER."""
    pg.add_robot(
        "anymal", library="anymal_c", pos=[0.0, 1.6], mounts=[{"library": "ur5e"}, {"library": "robotiq_2f85"}]
    )
    pg.add_robot(
        "go2", library="go2", pos=[0.0, 0.2], mounts=[{"library": "piper"}, {"name": "arm2", "library": "so_arm100"}]
    )
    pg.add_robot(
        "spot", library="spot", pos=[0.0, -1.2], mounts=[{"library": "z1"}, {"name": "arm2", "library": "piper"}]
    )


def _tile_and_tray(pg: Any) -> None:
    pg.add_robot("arm", template="arm", end_effector="suction", params={"color": "orange"})
    pg.add_object("tile", shape="box", size=[0.07, 0.07, 0.008], pos=[0.25, -0.12], color="teal")
    pg.add_object("tray", shape="bin", size=[0.14, 0.14, 0.05], pos=[0.05, 0.28], color="gray")


def _pick(robot: str, obj: str, target: str) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        pg.behaviors.pick_and_place(robot, obj, target)

    return act


def _operate(robot: str, target: str, action: str, amount: float | None = None) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        from robotforge.behaviors import operate

        operate.start(pg.behaviors, robot, target, action, amount)

    return act


def _sorting(pg: Any) -> None:
    """Start the factory's sorter in the background, in real time (scripts/factory_bench.py)."""
    import threading

    from factory_bench import Sorter

    def wait(until: Callable[[], bool], seconds: float) -> None:
        start = float(pg.session.data.time)
        while not until() and float(pg.session.data.time) - start < seconds:
            time.sleep(0.02)

    sorter = Sorter(pg, wait, 0.1)
    threading.Thread(target=lambda: [sorter.cycle() for _ in range(30)], daemon=True).start()


def _mechanism_row(pg: Any) -> None:
    things = [
        ("drawer", {"shape": "drawer", "pos": [-0.75, 0.0]}),
        ("cabinet", {"shape": "cabinet", "pos": [-0.35, 0.0]}),
        ("button", {"shape": "button", "pos": [-0.05, 0.0]}),
        ("switch", {"shape": "switch", "pos": [0.12, 0.0]}),
        ("lever", {"shape": "lever", "pos": [0.32, 0.0], "yaw": 90}),
        ("valve", {"shape": "valve", "pos": [0.56, 0.0]}),
    ]
    for name, fields in things:
        pg.add_object(name, **fields)


def _drawer_for(robot: str, x: float) -> Callable[[Any], None]:
    """A library arm with a drawer chest in front of it, its bar toward the arm."""

    def world(pg: Any) -> None:
        pg.add_robot(robot, library=robot)
        pg.add_object("chest", shape="drawer", pos=[x, 0.0], yaw=-90, color="#6b8fb3")

    return world


def _soft(pg: Any) -> None:
    pg.add_object("hook_post", shape="pillar", size=[0.04, 1.0], pos=[-0.45, 0.0])
    pg.add_object("rope", shape="rope", size=[0.6, 0.014], pos=[-0.45, -0.02, 0.98])
    pg.add_object("table", shape="table", size=[0.5, 0.4, 0.35], pos=[0.25, 0.0], color="brown")
    pg.add_object("cloth", shape="cloth", size=[0.6, 0.5], pos=[0.25, 0.0, 0.5], color="teal")


# Legged robots on rough ground (G8): each quadruped in its own lane, across a flight of 8 cm stairs or a patch of
# 3.5 cm gravel, side by side (scripts/demos.py films the same lanes).
LANES = (("go2", -1.95), ("a1", -0.65), ("anymal_c", 0.65), ("barkour", 1.95))
LANE_STAIRS = {"steps": 3, "step_height": 0.08, "step_depth": 0.3, "landing": 0.8, "width": 1.2}
LANE_GRAVEL = {"size": [3.0, 1.2], "height": 0.035, "scale": 0.06, "seed": 3}


def _lanes(kind: str, params: dict[str, Any], length: float, style: str | None = None) -> Callable[[Any], None]:
    """A world with the LANES robots 0.4 m before a patch of terrain `length` m long, one patch per lane."""

    def build(pg: Any) -> None:
        for robot, y in LANES:
            pg.add_robot(robot, library=robot, pos=[-0.4, y])
            pg.add_terrain(kind, pos=[0.6 + length / 2, y], style=style, params=dict(params))

    return build


def _move_all(speed: float) -> Callable[[Any], None]:
    def act(pg: Any) -> None:
        for robot, _ in LANES:
            pg.behaviors.move(robot, speed, 0.0)

    return act


G8_SHOTS = [
    Shot(  # the four of them on the way up, each foot on a tread, none on a nosing
        "g8_stairs",
        _lanes("stairs", LANE_STAIRS, 2.0),
        actions=[(0.5, _move_all(0.3))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([5.0, -0.9, 1.7], [0.8, 0.0, 0.15]);",
        settle=5.0,
    ),
    Shot(  # close on the Go2 stepping up
        "g8_stairs_go2",
        _lanes("stairs", LANE_STAIRS, 2.0),
        actions=[(0.5, _move_all(0.3))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.5, -3.25, 0.55], [0.85, -1.95, 0.2]);",
        settle=3.6,
    ),
    Shot(
        "g8_gravel",
        _lanes("rough", LANE_GRAVEL, 3.0, style="rock"),
        actions=[(0.5, _move_all(0.35))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([2.6, -3.9, 1.3], [1.3, -0.2, 0.1]);",
        settle=4.0,
    ),
]


# Pictures of the mechanisms lane (G3): the lane's own folder, not the README's.
# ---- G10: mount anything on anything -----------------------------------------------------------------------------
# The ten combinations live in scripts/combos.py (each made in one sentence, moving and doing a small job); these are
# pictures of them standing, at 1920x1080.


def _combo(name: str) -> Callable[[Any], None]:
    def world(pg: Any) -> None:
        import combos

        combos.COMBOS[name].build(pg)

    return world


def _g10_shots() -> list[Shot]:
    import combos

    return [
        Shot(f"g10_{c.name}", _combo(c.name), page=_framed(c.robot, c.view, c.lift), size=(1920, 1080), settle=2.5)
        for c in combos.COMBOS.values()
    ]


# ---- G12: the open catalogue ---------------------------------------------------------------------------------------
# The 20 robots scripts/catalogue_bench.py draws (seed 12), each added by name for real and doing what its kind does:
# an arm reaching, a dog or humanoid walking, a base driving, a hand closing. scripts/catalogue_grid.py puts the
# pictures in one grid with their names.

G12_SEED = 12
G12_FRAME = """
    const rf = window.__robotforge;
    rf.select(null);
    const mine = rf.scene.bodies.filter((b) => b.name.startsWith("r/"));
    const points = mine.map((b) => rf.bodyPosition(b.id));
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    points.forEach((p) => {
        for (let i = 0; i < 3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
    });
    const c = [0, 1, 2].map((i) => (lo[i] + hi[i]) / 2);
    const radius = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.04) / 2;
    const dist = radius * 3.0 + 0.12;
    rf.lookAt([c[0] + dist * 0.8, c[1] - dist * 0.95, c[2] + dist * 0.45], c);
"""


def _g12_world(name: str) -> Callable[[Any], None]:
    def world(pg: Any) -> None:
        pg.add_robot("r", library=name)
        _g12_pedestal(pg)

    return world


def _g12_pedestal(pg: Any) -> None:
    """A hand or finger bolted down by its file hangs where the file has it, its drawn parts above the floor: give it a
    stand, so the picture does not show something floating."""
    import mujoco
    import numpy as np

    m, d = pg.session.model, pg.session.data
    root = pg.behaviors._root_body("r")
    if m.body_jntnum[root] and m.jnt_type[m.body_jntadr[root]] == mujoco.mjtJoint.mjJNT_FREE:
        return  # it stands on its own
    mujoco.mj_forward(m, d)
    bodies = {
        b for b in range(1, m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("r/")
    }
    boxes = []  # (low, high) of each drawn part along x, y and z, from its bounding box in the world
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in bodies or m.geom_rgba[g][3] == 0:
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        center = d.geom_xpos[g] + rot @ m.geom_aabb[g][:3]
        half = np.abs(rot) @ m.geom_aabb[g][3:]
        boxes.append((center - half, center + half))
    if not boxes:
        return
    low = min(float(b[0][2]) for b in boxes)
    if low > 0.03:
        lo = [min(float(b[0][i]) for b in boxes) for i in (0, 1)]
        hi = [max(float(b[1][i]) for b in boxes) for i in (0, 1)]
        size = [hi[0] - lo[0] + 0.06, hi[1] - lo[1] + 0.06, low]
        pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
        pg.add_object("stand", shape="box", size=size, pos=pos, color="gray", fixed=True)


def _g12_do(pg: Any) -> None:
    """Start what the robot's kind does, as the bench does: reach, walk, drive, close."""
    import mujoco
    import numpy as np

    from robotforge.behaviors.manager import BehaviorError

    kind = pg._about_catalogue_robot("r", pg.current_world().robots[0].catalogue)["kind"]
    d, m = pg.session.data, pg.session.model
    try:
        if kind == "arm":
            arm = pg.behaviors.arms("r")[0]
            site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{arm}/tcp")
            home = d.site_xpos[site].copy()
            for step in ((0.2, 0, 0), (0.15, 0, -0.15), (0.25, 0, 0.1), (0.1, 0.1, 0), (0.1, -0.1, -0.1)):
                try:
                    pg.behaviors.move_arm(arm, [float(home[i] + 0.4 * f) for i, f in enumerate(step)])
                    return
                except BehaviorError:
                    continue
        elif kind in ("quadruped", "humanoid"):
            pg.behaviors.move("r", 0.25, 0.0)
        elif kind == "wheeled":
            pg.behaviors.move("r", 0.3, 0.0)
        elif kind in ("hand", "other"):
            for a in range(m.nu):
                low, high = m.actuator_ctrlrange[a] if m.actuator_ctrllimited[a] else (-1.0, 1.0)
                d.ctrl[a] = high if abs(high) >= abs(low) else low
        elif kind == "aircraft":
            pg.behaviors.heave("r", 0.5)
    except (BehaviorError, ValueError, np.linalg.LinAlgError):
        pass  # a robot that cannot do its kind's job stands in the picture, as it does in the bench


def _g12_shots() -> list[Shot]:
    import catalogue_bench

    return [
        Shot(f"g12_{name}", _g12_world(name), act=_g12_do, page=G12_FRAME, settle=4.0)
        for name in catalogue_bench.draw(G12_SEED)
    ]


G3_SHOTS = [
    Shot(
        "g3_suction_tile",
        _tile_and_tray,
        actions=[(0.5, _pick("arm", "tile", "tray"))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.5, -0.45, 0.3], [0.13, 0.05, 0.08]);",
        settle=4.6,
    ),
    Shot(
        "g3_kitchen_drawer",
        lambda pg: pg.load_scene("kitchen"),
        actions=[(0.5, _operate("arm", "drawer", "open", 0.2))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.12, -0.82, 1.33], [0.26, -0.02, 0.95]);",
        settle=7.8,
    ),
    Shot(
        "g3_factory",
        lambda pg: pg.load_scene("factory"),
        actions=[(0.5, _sorting)],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.95, -0.95, 1.08], [0.2, -0.12, 0.62]);",
        settle=6.5,
    ),
    Shot(
        "g3_mechanisms",
        _mechanism_row,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.1, -1.25, 0.65], [-0.05, 0.0, 0.1]);",
    ),
    # A PiPER points down only low, so it comes in to the bar from in front of the drawer, not over its top.
    Shot(
        "g3_piper_drawer",
        _drawer_for("piper", 0.45),
        actions=[(0.5, _operate("piper", "chest", "open", 0.2))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.85, -0.8, 0.62], [0.2, 0.0, 0.2]);",
        settle=9.0,
    ),
    Shot(
        "g3_panda_drawer",
        _drawer_for("panda", 0.55),
        actions=[(0.5, _operate("panda", "chest", "open", 0.2))],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.35, -1.25, 1.0], [0.25, 0.0, 0.35]);",
        settle=10.0,
    ),
    Shot(
        "g3_soft",
        _soft,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.3, -1.6, 0.9], [-0.05, 0.0, 0.45]);",
        settle=4.0,
    ),
]

# The fifteen sentences the robot generator was proven with (tests/test_generator.py), each one robot.
GENERATED = [
    "a six-legged robot with two arms",
    "a tall wheeled courier with a lidar",
    "a snake with a gripper head",
    "a small orange robot dog",
    "a big carbon fiber quadruped with long legs and a camera head",
    "a spider robot in black and red",
    "a six-wheeled mars rover with an arm",
    "a self-balancing robot on two wheels",
    "a robot arm with six joints and a suction cup",
    "a long green snake with 12 segments",
    "a centaur in white and gold",
    "a tall humanoid",
    "a wheeled robot with two arms",
    "一只带激光雷达的蓝色四足机器狗",
    "a tentacle arm with a gripper",
]


def _generated_line_up(pg: Any) -> None:
    """All fifteen generated robots facing the camera: tall ones at the back, the two snakes lying across in front.
    Places are metres toward the camera (from the south-east) and to its left."""
    view = math.radians(-42)
    toward, left = (math.cos(view), math.sin(view)), (math.sin(view), -math.cos(view))
    places = [(i, -1.2, c - 2.0) for c, i in enumerate([11, 1, 8, 6, 12])]
    places += [(i, 0.0, c - 2.0) for c, i in enumerate([0, 10, 14, 4, 13])]
    places += [(i, 1.1, c - 1.0) for c, i in enumerate([3, 7, 5])]
    places += [(2, 2.0, 0.85), (9, 2.0, -0.85)]
    for i, a, b in places:
        pos = [round(a * toward[0] + b * left[0], 3), round(a * toward[1] + b * left[1], 3)]
        snake = "snake" in GENERATED[i]
        pg.add_robot(f"g{i + 1:02d}", describe=GENERATED[i], pos=pos, yaw=48 if snake else -42)


def _generated_four(pg: Any) -> None:
    """Four robots each built from one sentence, for the README: a humanoid, a courier, a six-legged one with arms, a
    big dog (GENERATED 11, 1, 0 and 4), in a row facing the camera."""
    for name, sentence, y in (("humanoid", 11, 1.55), ("courier", 1, 0.55), ("mantis", 0, -0.5), ("dog", 4, -1.5)):
        pg.add_robot(name, describe=GENERATED[sentence], pos=[0.0, y])


def _frame_robot(robot: str, direction=(1.0, -0.85, 0.55), scale: float = 1.9, least: float = 0.6) -> str:
    """Page script: nothing selected, the camera along `direction` from the middle of all of `robot`'s bodies (half
    way up from the ground to its highest one), far enough to see the whole robot whatever its size."""
    return f"""
        const rf = window.__robotforge;
        rf.select(null);
        const ours = (b) => b.name.startsWith({json.dumps(robot + "/")});
        const mine = rf.scene.bodies.map((b, i) => [b, i]).filter(([b]) => ours(b));
        const ps = mine.map(([, i]) => rf.bodyPosition(i));
        const lo = [0, 1, 2].map((k) => Math.min(...ps.map((p) => p[k])));
        const hi = [0, 1, 2].map((k) => Math.max(...ps.map((p) => p[k])));
        const mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, Math.max(hi[2], 0.06) / 2];
        const span = Math.max(0.8 * Math.max(hi[0] - lo[0], hi[1] - lo[1]), 1.5 * hi[2], 0.25);
        const d = {json.dumps(list(direction))}, n = Math.hypot(...d), far = Math.max({least}, {scale} * span);
        rf.lookAt(mid.map((v, k) => v + (d[k] / n) * far), mid);
    """


def _generated(sentence: str) -> Callable[[Any], None]:
    def world(pg: Any) -> None:
        pg.add_robot("bot", describe=sentence)

    return world


# One picture per generated robot (gen_01 ... gen_15), for the gallery grid; and all fifteen together.
GENERATED_SHOTS = [
    Shot(
        f"gen_{i + 1:02d}",
        _generated(sentence),
        # A snake is long and thin: seen from the side and close, it fills the picture instead of a thin diagonal
        page=(
            _frame_robot("bot", direction=(0.25, -1.0, 0.45), scale=1.7) if "snake" in sentence else _frame_robot("bot")
        ),
        settle=2.5,
    )
    for i, sentence in enumerate(GENERATED)
] + [
    Shot(
        "readme_generated",
        _generated_four,
        page="window.__robotforge.select(null);" + _look((2.9, -0.7, 1.05), (0.0, 0.05, 0.5)),
        settle=3.0,
    ),
    Shot(
        "generated_robots",
        _generated_line_up,
        page="window.__robotforge.select(null);" + _look((3.1, -2.75, 1.75), (0.4, -0.35, 0.1)),
        settle=3.0,
    ),
]


# ---- worlds and tasks generated from one sentence (G11a) ----------------------------------------------------------

# Twenty sentences, each a world and a task: load_scene(description=...) lays the world out, add_task turns the same
# sentence into its task. Their pictures are world_01 ... world_20.
GENERATED_WORLDS = [
    "In a small flat with a kitchen and a bedroom, bring the red cup from the kitchen to the bedroom desk.",
    "把厨房里的苹果拿到客厅的茶几上",
    "Open the drawer in the kitchen of a house with a kitchen and a dining room.",
    "In a warehouse with three rows of racks, bring the blue parcel to the packing table.",
    "Drive the robot onto the loading dock of a large warehouse with pallets.",
    "Walk the robot dog across the park to the goal past the stairs and the ramp.",
    "去公园里的长椅旁边，路上有台阶和坡道",
    "Put the green apple in the bowl in the kitchen.",
    "Pick up the red cube with the robot arm in the kitchen.",
    "Press the button on the kitchen counter.",
    "打开厨房台面上的柜子",
    "Go to the dining room of a house with a kitchen, a dining room and a living room.",
    "Bring the blue ball from the living room to the kitchen table.",
    "In a rocky hillside with stepping stones and boulders, walk to the goal.",
    "把蓝色的方块放进厨房的碗里",
    "Switch on the switch in the kitchen.",
    "Bring the purple can to the desk in the bedroom of a flat with a bedroom and a living room.",
    "Take the orange toy from the bedroom to the living room.",
    "In a large warehouse with pallets and a conveyor, bring the green parcel to the packing table.",
    "Go to the kitchen in a three-room apartment.",
]


def _generated_world(sentence: str) -> Callable[[Any], None]:
    """A world builder: the world the sentence describes, and its task."""

    def build(pg: Any) -> None:
        pg.load_scene(description=sentence, seed=0)
        pg.add_task(sentence, name="task")

    return build


def _world_overview(sentence: str) -> str:
    """Page script: the whole generated place from above and to the south, a little to the east."""
    from robotforge.design.generate import generate

    objects = generate(sentence, seed=0, check=False).content.objects
    xs = [o["pos"][0] for o in objects if o["shape"] != "marker"] or [0.0]
    ys = [o["pos"][1] for o in objects if o["shape"] != "marker"] or [0.0]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), (max(ys) - min(ys)) * 1.4, 5.5)
    eye = (round(cx + 0.18 * span, 2), round(cy - 0.78 * span, 2), round(0.72 * span, 2))
    return "window.__robotforge.select(null); " + _look(eye, (round(cx, 2), round(cy + 0.04 * span, 2), 0.0))


def _world_shots() -> list[Shot]:
    return [
        Shot(f"world_{k + 1:02d}", _generated_world(s), page=_world_overview(s), settle=3.0)
        for k, s in enumerate(GENERATED_WORLDS)
    ]


# ---- G12b: sensors and perception ---------------------------------------------------------------------------------
# A Go2 enters the warehouse it has never seen, draws its map and finds itself in it (scripts/perception_bench.py has
# the measured run); the pictures show the map growing over the floor.


def _go2_warehouse_start(pg: Any) -> None:
    pg.load_scene("warehouse", keep_robots=False, starter=False)
    pg.add_robot("go2", library="go2", pos=[-6.2, 2.6], yaw=0)


def _map_for(seconds: float, area: list[float] | None = None) -> Callable[[Any], None]:
    """Start mapping and explore for `seconds` of simulated time, as fast as the machine runs it."""

    def act(pg: Any) -> None:
        pg.perception.start("go2", {})
        trips = pg.perception.explore_trips("go2", seconds, area=area)
        try:
            while True:
                trip, limit = next(trips)
                pg.run_fast(lambda t=trip: t.done, limit + 1.0)
        except StopIteration:
            pass

    return act


G12_AREA = [-7.5, -3.4, 7.5, 5.4]  # the warehouse's floor: the open sides beyond it are left unexplored
G12_OVERVIEW = "const rf = window.__robotforge; rf.select(null); rf.lookAt([0.5, -5.5, 13.5], [0.5, 1.2, 0]);"

G12_SHOTS = [
    Shot(
        "g12_map_overview",
        _go2_warehouse_start,
        act=_map_for(60.0, G12_AREA),
        page=G12_OVERVIEW,
        size=(1920, 1080),
        settle=2.5,
    ),
]

SHOTS = [
    Shot(
        "hero",
        _park,
        viewer=[_walk("go2", 0.35), _walk("dog", 0.35), {"wait": 3.0}],
        page=_look_at(["go2", "dog"], (3.2, -1.9, 1.5)),
        settle=1.0,
    ),
    Shot("scene_kitchen", lambda pg: pg.load_scene("kitchen"), page=_look_at(["arm"], (1.1, -1.3, 0.9))),
    # The two rooms from the open south side, the door between them, the dog with its arm in the living room.
    Shot(
        "scene_home",
        lambda pg: pg.load_scene("home"),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.0, -6.2, 5.6], [0.0, 0.35, 0.2]);",
    ),
    Shot(
        "scene_home_door",
        lambda pg: pg.load_scene("home"),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([2.2, -3.4, 1.7], [0.0, -0.9, 0.8]);",
    ),
    Shot(
        "kitchen_starts",
        _kitchen_starts,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([3.0, -3.4, 2.6], [0.45, -0.35, 0.3]);",
        settle=2.0,
    ),
    # Navigation: the path still ahead shows as an amber line on the ground while the robot follows it.
    Shot(
        "nav_warehouse",
        _in_scene("warehouse", {"go2": {"library": "go2"}}),
        viewer=[_go("go2", [6.0, -1.4]), {"wait": 6.0}],
        page="window.__robotforge.lookAt([0.5, -7.5, 7.5], [0.5, 0.6, 0]);",
        settle=0.5,
    ),
    Shot(
        "nav_maze",
        _in_scene("maze", {"rover": {"template": "rover"}}),
        viewer=[_go("rover", [-2.0, 2.0]), {"wait": 0.5}],  # the far corner cell, through the entrance
        page="window.__robotforge.lookAt([0.0, -3.4, 6.6], [0.0, 0.1, 0]);",
        settle=0.5,
    ),
    Shot(
        "nav_kitchen",
        _in_scene("kitchen", {"mm": {"build": MOBILE_ARM}, "dog": {"template": "quadruped"}}),
        viewer=[_go("mm", "fruit"), _go("dog", [1.4, 1.0]), {"wait": 0.2}],
        page="window.__robotforge.lookAt([0.6, -4.2, 3.4], [0.0, 0.3, 0.2]);",
        settle=0.2,
    ),
    # Fetching: a mobile Panda brings the apple from the kitchen island to the dining table (behaviors/fetch.py):
    # reaching in tilted from the aisle behind the island, then carrying it folded over its deck on the way.
    Shot(
        "fetch_reach_in",
        _kitchen_panda,
        viewer=[_fetch("mm", "apple", "dining_table"), {"wait": 1.0}],
        page=_when("apple", "p[2] > 0.95") + _look_at(["mm"], (0.5, -1.0, 2.4)),  # just lifted off the island
        settle=0.2,
    ),
    Shot(
        "fetch_carry",
        _kitchen_panda,
        viewer=[_fetch("mm", "apple", "dining_table"), {"wait": 1.0}],
        page=_when("apple", "p[2] < 0.8 && p[1] < 0.3") + _look_at(["mm"], (-1.1, -1.5, 1.9)),  # out of the aisle
        settle=0.2,
    ),
    Shot(
        "fetch_put_down",
        _kitchen_panda,
        viewer=[_fetch("mm", "apple", "dining_table"), {"wait": 1.0}],
        # over the table first: on the way up the apple passes the table's height before it gets there
        page=_when("apple", "p[0] < -1.25 && p[2] > 0.84")
        + _when("apple", "p[0] < -1.25 && p[2] < 0.79")
        + _wait(1.2)
        + _look_at(["mm"], (1.6, -2.0, 1.8)),
        settle=0.2,
    ),
    # A lift picker fetches a parcel from a rack's third board and takes it up the ramp onto the loading dock.
    Shot(
        "fetch_warehouse_rack",
        _warehouse_picker,
        viewer=[_fetch("picker", "parcel_1", "dock"), {"wait": 1.0}],
        # just lifted off the board, seen from along the aisle, between the robot and the rack
        page=_when("parcel_1", "p[2] > 0.9") + _look_between("picker", "parcel_1", (2.3, 1.5, 0.75)),
        settle=0.2,
    ),
    Shot(
        "fetch_warehouse_dock",
        _warehouse_picker,
        viewer=[_fetch("picker", "parcel_1", "dock"), {"wait": 1.0}],
        page=_when("parcel_1", "p[0] > 5.0 && p[2] < 0.36") + _wait(1.5) + _look_at(["picker"], (1.7, 1.5, 1.4)),
        settle=0.2,
    ),
    Shot(  # the stairs, the curbs and the stepping stones are more than the dog climbs: its path goes round them
        "nav_obstacle_course",
        _in_scene("obstacle_course", {"dog": {"template": "quadruped"}}),
        viewer=[_go("dog", "finish"), {"wait": 2.0}],
        page="window.__robotforge.lookAt([-2.4, -5.0, 3.8], [6.8, 0.5, 0]);",
        settle=0.5,
    ),
    Shot("import_a1", _a1, page=_look_at(["a1"], (0.9, -1.1, 0.55))),
    Shot(
        "lesson_your_robot",
        _a1,
        page=_lesson(
            "learn_to_walk",
            "document.querySelector('[data-choice=\"robot\"] button:last-child').click();",
            _wait(1.0),
            _press("train"),
            _wait(22.0),
        ),
    ),
    Shot(  # after Keep the A1 learns its steering and turns, and checks its switches: a few minutes on a laptop
        "lesson_your_robot_kept",
        _a1,
        page=_lesson(
            "learn_to_walk",
            "document.querySelector('[data-choice=\"robot\"] button:last-child').click();",
            _wait(1.0),
            _press("train"),
            # until the training's result shows (12 s on an idle laptop, 28 s with other benches running)
            """for (let k = 0; k < 120 && !/学了 \\d+ 轮|\\d+ rounds \\(/.test(
                document.getElementById("lesson-result").textContent); k++) {
                await new Promise((r) => setTimeout(r, 500));
            }""",
            _press("keep"),
            # until the panel says the turns are learned too (the second sentence of the result), then show it
            """for (let k = 0; k < 600 && !/原地转弯：|turn on the spot:/.test(
                document.getElementById("lesson-result").textContent); k++) {
                await new Promise((r) => setTimeout(r, 1000));
            }
            document.getElementById("lesson-result").scrollIntoView({block: "end"});""",
            _wait(1.0),
        ),
    ),
    Shot("lesson_learn_to_walk", _nothing, page=_lesson("learn_to_walk", _press("train"), _wait(25.0))),
    Shot(  # the plots show 6 s: capture while the step response is still in them
        "bench_pid_twins",
        _nothing,
        page=_lesson("pid_twins", _wait(0.5), _press("test"), _wait(3.0)),
        settle=0.2,
    ),
    Shot("challenge_obstacle_course", _nothing, page=_lesson("obstacle_course", _wait(2.0))),
    Shot("lesson_seeing_mask", _nothing, page=_lesson("seeing", *_SEEING_MASK)),
    Shot("lesson_seeing_where", _nothing, page=_lesson("seeing", *_SEEING_WHERE)),
    Shot("lesson_seeing_depth", _nothing, page=_lesson("seeing", *_SEEING_LIGHT[:-4])),
    Shot("lesson_seeing_evening", _nothing, page=_lesson("seeing", *_SEEING_LIGHT)),
    Shot(
        "lesson_seeing_fixed",
        _nothing,
        page=_lesson("seeing", *_SEEING_LIGHT, _tick("balance"), _tick("clean"), _wait(1.5)),
    ),
    Shot("lesson_imitation_show", _nothing, page=_lesson("imitation", _TEACH, _CARRYING), settle=0.3),
    Shot("lesson_imitation_trying", _nothing, page=_lesson("imitation", _TEACH, _TRYING), settle=0.2),
    Shot("lesson_imitation_network", _nothing, page=_imitation("network"), settle=1.0),
    Shot(
        "lesson_imitation_sweep",
        _nothing,
        page=_imitation("how_many", 'document.getElementById("lesson-plots").scrollTop = 1e6;'),
        settle=1.0,
    ),
    Shot("lesson_imitation_correct", _nothing, page=_imitation("correct"), settle=1.0),
    Shot("lesson_imitation_compare", _nothing, page=_imitation("ik"), settle=1.0),
    Shot("lesson_imitation_keys", _nothing, page=_lesson("imitation", _TEACH, _KEYS), settle=0.3),
    Shot("lesson_imitation_drag", _nothing, page=_lesson("imitation", _TEACH, _DRAG), settle=0.3),
    Shot(  # driving at 0.3 m/s from the left: it rolls to the rock (before its caster was fixed it stood still)
        "eval_mars_rover",
        _mars_rover,
        viewer=[{"type": "teleop", "robot": "mars_rover", "speed": 0.3, "turn": 0.0}, {"wait": 5.0}],
        page="window.__robotforge.lookAt([1.6, -3.4, 1.9], [1.6, 0.0, 0.2]);",
        settle=0.5,
    ),
    # Robots built from parts, dressed: a dog, a sensor rover, a carbon spider and a dog carrying an arm.
    Shot(
        "built_robots",
        _built(("dog", "scout", "spider", "centaur")),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.8, -0.28, 0.66], [0.0, 0.0, 0.1]);",
    ),
    # Robots that are not built from parts, kept as they are under a plate that carries a lidar.
    Shot(
        "carried_on_plates",
        _on_plates,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.9, -1.4, 1.0], [0.0, -0.1, 0.15]);",
        settle=3.0,
    ),
    Shot(
        "built_materials",
        _materials,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.8, -0.25, 0.55], [0.0, 0.0, 0.12]);",
    ),
    *(
        Shot(f"gallery_{name}", _design(name, extras), page=_look_at([name], offset), settle=3.0)
        for name, extras, offset in GALLERY_SHOTS
    ),
    Shot("sensors_go2", _sensing_go2(), page=_look_at(["go2"], (0.85, 0.7, 0.42)), settle=1.0),
    # What the sensors sense, drawn by the viewer: the 3D lidar's points down the aisle and what the depth camera
    # sees beside the view; the robots drawer's sensor list; a Panda's wrist force while it lifts a cube.
    Shot("sensors_go2_cloud", _sensing_go2(), page=SENSOR_CLOUD_VIEW, settle=2.5),
    Shot("sensors_go2_drawer", _sensing_go2(), page=SENSOR_DRAWER_VIEW, settle=2.5),
    Shot("sensors_panda_wrist", _panda_wrist, act=_lift_cube, page=SENSOR_WRIST_VIEW, settle=0.5),
    # Code a user wrote: a signpost part and a Geiger sensor from plugin files (docs/EXTENDING.md), in the viewer.
    Shot(
        "extend_plugins",
        _extend_plugins,
        viewer=[_walk("go2", 0.25), {"wait": 2.0}, _walk("go2", 0.0)],
        page=EXTEND_VIEW,
        size=(1920, 1080),
        settle=2.0,
    ),
    Shot(
        "gallery",
        _gallery_line_up,
        page=_look_at(["centaur", "twin_arm"], (2.4, -2.2, 1.5)),
        settle=3.0,
    ),
    Shot("library_new_walkers", _new_walkers, page=_look((4.2, -1.3, 1.5), (0.0, 0.0, 0.4)), settle=3.0),
    Shot("library_new_arms", _new_arms, page=_look((3.4, -1.0, 1.6), (0.0, 0.0, 0.25)), settle=3.0),
    Shot(
        "splice_dogs_with_arms",
        _dogs_with_arms,
        viewer=[_walk("go2", 0.3), _walk("spot", 0.3), {"wait": 3.0}, _walk("go2", 0.0), _walk("spot", 0.0)],
        page=_look((2.6, -2.3, 1.3), (1.0, 0.0, 0.35)),
        settle=2.0,
    ),
    Shot("splice_more_carriers", _more_carriers, page=_look((3.6, -1.2, 1.5), (0.0, 0.0, 0.45)), settle=3.0),
    Shot("splice_complex", _complex_splices, page=_look((3.4, -1.6, 1.7), (0.0, 0.2, 0.5)), settle=3.0),
    Shot("splice_barkour_so100", _barkour_so100, page=_framed("barkour", (1.1, -0.9, 0.5), 0.02), settle=3.0),
    Shot("splice_go2_piper_so100", _go2_two_arms, page=_framed("go2", (-0.55, -1.3, 0.55), 0.08), settle=3.0),
    Shot("splice_spot_z1", _spot_z1, page=_framed("spot", (1.8, -1.45, 0.6), 0.12), settle=3.0),
    Shot("splice_anymal_ur5e_robotiq", _anymal_ur5e, page=_framed("anymal", (2.0, -1.55, 0.85), 0.0), settle=3.0),
    Shot(
        "library_line_up",
        _line_up,
        viewer=[{"wait": 2.0}],
        page="window.__robotforge.select(null); " + _look((4.9, 0.3, 1.75), (-0.1, 0.3, 0.42)),
        settle=3.0,
    ),
    # Water: the harbor with its boat, crates afloat and the dock; an ROV under water with the surface above it.
    Shot(
        "water_harbor",
        _in_scene("harbor", {"boat": {"template": "boat"}}),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([-5.5, -18.0, 6.6], [2.5, -1.5, 1.0]);",
        settle=3.0,
    ),
    Shot(  # low over the harbor mouth, out to sea: the far water and the sky over it meet at the horizon
        "water_harbor_horizon",
        _in_scene("harbor", {"boat": {"template": "boat"}}),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.0, 3.0, 2.6], [2.5, -12.0, 0.9]);",
        settle=3.0,
    ),
    Shot(
        "water_boat",
        _in_scene("harbor", {"boat": {"template": "boat"}}),
        viewer=[_walk("boat", 0.5), {"wait": 3.0}],
        page=_look_at(["boat"], (-2.2, -2.6, 1.2)),
        settle=0.5,
    ),
    Shot(
        "water_rov",
        _in_scene("pool", {"rov": {"template": "rov"}}),
        viewer=[_go("rov", [0.0, 0.0, 0.75]), {"wait": 2.0}],
        page=(  # from a little below and to the side of where it holds, the surface a ceiling over it
            "const rf = window.__robotforge; rf.select(null);"
            "const p = rf.bodyPosition(rf.scene.bodies.find((b) => b.name.startsWith('rov/') && b.dynamic).id);"
            "rf.lookAt([p[0] + 1.1, p[1] + 0.35, p[2] - 0.22], [p[0], p[1], p[2] + 0.06]);"
        ),
        settle=1.0,
    ),
    Shot(
        "water_lake",
        _in_scene("lake", {"boat": {"template": "boat"}}),
        page="window.__robotforge.select(null); window.__robotforge.lookAt([8.0, -13.0, 15.5], [-0.6, -0.9, 0.8]);",
        settle=3.0,
    ),
    # Aircraft hovering a metre over the park's path, their shadows on the grass below: the drone template, a
    # hexacopter built from parts (the builder guide's example), the Skydio X2.
    Shot(
        "drone_park",
        _in_scene("park", {"drone": {"template": "drone"}}),
        viewer=[_go("drone", [0.0, -1.0, 0.8]), {"wait": 5.0}],
        page=_hover_look("drone", (0.8, -1.0, 0.26), 0.16),
        settle=0.5,
    ),
    Shot(
        "drone_hexacopter",
        _in_scene("park", {"hexa": {"build": HEXACOPTER}}),
        viewer=[_go("hexa", [0.0, -1.0, 0.9]), {"wait": 5.0}],
        page=_hover_look("hexa", (0.95, -1.2, 0.32), 0.2),
        settle=0.5,
    ),
    Shot(
        "drone_x2",
        _in_scene("park", {"x2": {"library": "skydio_x2"}}),
        viewer=[_go("x2", [0.0, -1.0, 0.8]), {"wait": 5.0}],
        page=_hover_look("x2", (0.82, -1.02, 0.28), 0.12),
        settle=0.5,
    ),
    # Hovering under a tree's crown beside its trunk (tree_4, its crown 2.1 m up), where it once landed on the crown.
    Shot(
        "drone_under_crown",
        _in_scene("park", {"drone": {"template": "drone"}}),
        viewer=[_go("drone", [-3.3, -4.12, 1.2]), {"wait": 14.0}],  # a 6 s flight, on a busy machine
        page=_hover_look("drone", (1.3, -1.38, -0.12), -0.42),  # from a little below: the crown over it in view
        settle=0.5,
    ),
    # Landed on the grass beside the park's loose red ball: sent to z 0 at the ball it once landed on top of it and
    # tipped over; now that is refused, and this is the spot the refusal offers ([2.0, -0.6] has room).
    Shot(
        "drone_beside_ball",
        _in_scene("park", {"drone": {"template": "drone"}}),
        viewer=[_go("drone", [2.0, -0.6, 0.0]), {"wait": 8.0}],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([1.72, -1.95, 0.52], [1.72, -0.6, 0.1]);",
        settle=0.5,
    ),
    # Every aircraft side by side, each hovering a metre up: the template with 4, 6 and 8 rotors, the hexacopter
    # built from parts, the Skydio X2 and the palm-sized Crazyflie.
    Shot(
        "drone_family",
        _drone_family,
        viewer=[_go(name, [x, 0.0, 1.0]) for name, x in DRONE_ROW] + [{"wait": 6.0}],
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.8, -4.3, 1.85], [0.8, 0.0, 0.85]);",
        settle=0.5,
    ),
    *G3_SHOTS,
    *G8_SHOTS,
    *G12_SHOTS,
    *GENERATED_SHOTS,
    *_g10_shots(),
    *_g12_shots(),
    *_world_shots(),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="names of the shots to take")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--lang", default="zh-CN")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for shot in SHOTS:
        if args.only and shot.name not in args.only:
            continue
        if shot.world is _a1 and not A1_URDF.is_file():
            print(f"{shot.name}: skipped, no A1 URDF at {A1_URDF} (set A1_URDF)")
            continue
        t0 = time.monotonic()
        path = take(shot, args.out, args.lang)
        print(f"{shot.name}: {path} ({time.monotonic() - t0:.0f} s)")
    singles = [args.out / f"gen_{i + 1:02d}.jpg" for i in range(len(GENERATED))]
    if all(p.is_file() for p in singles) and (not args.only or any(n.startswith("gen_") for n in args.only)):
        print(f"generated grid: {generated_grid(singles, args.out / 'generated_grid.jpg')}")


def generated_grid(singles: list[Path], out: Path, cols: int = 5) -> Path:
    """The gen_ pictures in a grid, each cropped to its robot (no viewer bars) with its sentence under it."""
    from PIL import Image, ImageDraw, ImageFont

    tile_w, tile_h, caption, gap, margin, title_h = 560, 380, 56, 14, 28, 76
    crop = (220, 110, 1380, 900)  # of a SAVED_WIDTH (1600 x 1000) picture

    def font(size: int) -> Any:
        for name in ("/System/Library/Fonts/Hiragino Sans GB.ttc", "C:/Windows/Fonts/msyh.ttc"):
            if Path(name).is_file():  # one with Chinese, for the Chinese sentence
                return ImageFont.truetype(name, size)
        return ImageFont.load_default(size)

    rows = -(-len(singles) // cols)
    sheet = Image.new(
        "RGB",
        (2 * margin + cols * tile_w + (cols - 1) * gap, 2 * margin + title_h + rows * (tile_h + caption + gap) - gap),
        (236, 239, 243),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), "robotforge: robots generated from one sentence each", font=font(34), fill=(29, 32, 38))
    draw.text(
        (margin, margin + 44),
        'add_robot(name, describe="...")  -  built from parts, checked by the builder; each one walks, drives, '
        "slithers or reaches (real viewer, real simulation)",
        font=font(19),
        fill=(90, 96, 106),
    )
    for i, (path, sentence) in enumerate(zip(singles, GENERATED, strict=True)):
        tile = Image.open(path).convert("RGB").crop(crop).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        r, c = divmod(i, cols)
        x, y = margin + c * (tile_w + gap), margin + title_h + r * (tile_h + caption + gap)
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y + tile_h, x + tile_w - 1, y + tile_h + caption - 1], fill=(255, 255, 255))
        draw.text((x + 12, y + tile_h + 8), f"{i + 1:02d}", font=font(17), fill=(200, 120, 30))
        text = f'"{sentence}"'
        if draw.textlength(text, font=font(19)) < tile_w - 56:
            draw.text((x + 44, y + tile_h + 6), text, font=font(19), fill=(29, 32, 38))
        else:  # a long sentence goes on two lines at the same size, split near the middle
            words = text.split(" ") if " " in text else list(text)  # a sentence without spaces (Chinese) by character
            join = " ".join if " " in text else "".join
            cut = min(range(1, max(len(words), 2)), key=lambda k: abs(len(join(words[:k])) - len(join(words[k:]))))
            for line, part in enumerate((join(words[:cut]), join(words[cut:]))):
                draw.text((x + 44, y + tile_h + 3 + 24 * line), part, font=font(19), fill=(29, 32, 38))
    sheet.save(out, "JPEG", quality=88, optimize=True)
    return out


if __name__ == "__main__":
    main()
