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
        for k, msg in enumerate(messages):
            if "wait" in msg:
                await asyncio.sleep(msg["wait"])
                continue
            await ws.send(json.dumps({"id": f"shot{k}", **msg}))
            if msg.get("type") in ("lesson_open", "go_to"):  # a slow request: wait for its reply
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


def _look_at(robots: list[str], offset: tuple[float, float, float]) -> str:
    """Page script: aim the camera at the middle of these robots, from `offset` metres away."""
    return f"""
        const rf = window.__robotforge;
        const at = {json.dumps(robots)}.map((r) => rf.bodyPosition(
            rf.scene.bodies.find((b) => b.name.startsWith(r + "/") && b.dynamic).id));
        const mid = [0, 1, 2].map((i) => at.reduce((s, p) => s + p[i], 0) / at.length);
        rf.lookAt(mid.map((v, i) => v + {json.dumps(list(offset))}[i]), mid);
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
    ("barkour", "barkour", [], [0.55, 1.4]),
    ("op3", "op3", [], [0.55, 0.0]),
    ("anymal", "anymal_c", [], [0.55, -1.4]),
    ("z1", "z1", [], [0.55, -2.8]),
    ("t1", "t1", [], [-0.7, 2.1]),
    ("ur5e", "ur5e", [{"library": "robotiq_2f85"}], [-0.7, 0.7]),
    ("xarm7", "xarm7", [], [-0.7, -0.7]),
    ("aloha", "aloha", [], [-0.7, -2.1]),
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


# Pictures of the mechanisms lane (G3): the lane's own folder, not the README's.
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


SHOTS = [
    Shot(
        "hero",
        _park,
        viewer=[_walk("go2", 0.35), _walk("dog", 0.35), {"wait": 3.0}],
        page=_look_at(["go2", "dog"], (3.2, -1.9, 1.5)),
        settle=1.0,
    ),
    Shot("scene_kitchen", lambda pg: pg.load_scene("kitchen"), page=_look_at(["arm"], (1.1, -1.3, 0.9))),
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
    Shot("splice_spot_z1", _spot_z1, page=_framed("spot", (1.6, -1.3, 0.75), -0.02), settle=3.0),
    Shot("splice_anymal_ur5e_robotiq", _anymal_ur5e, page=_framed("anymal", (2.0, -1.55, 0.85), 0.0), settle=3.0),
    Shot(
        "library_line_up",
        _line_up,
        page="window.__robotforge.select(null); " + _look((4.8, -0.35, 2.7), (-0.1, -0.35, 0.25)),
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


if __name__ == "__main__":
    main()
