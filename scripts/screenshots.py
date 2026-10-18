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


async def _capture(chrome_port: int, port: int, shot: Shot, out: Path, lang: str) -> None:
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
        await asyncio.sleep(shot.settle)
        if shot.page:
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
        asyncio.run(_capture(chrome_port, port, shot, out, lang))
    finally:
        for proc in (chrome, server):
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


SHOTS = [
    Shot(
        "hero",
        _park,
        viewer=[_walk("go2", 0.35), _walk("dog", 0.35), {"wait": 3.0}],
        page=_look_at(["go2", "dog"], (3.2, -1.9, 1.5)),
        settle=1.0,
    ),
    Shot("scene_kitchen", lambda pg: pg.load_scene("kitchen"), page=_look_at(["arm"], (1.1, -1.3, 0.9))),
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
    Shot(
        "built_materials",
        _materials,
        page="window.__robotforge.select(null); window.__robotforge.lookAt([0.8, -0.25, 0.55], [0.0, 0.0, 0.12]);",
    ),
    *(
        Shot(f"gallery_{name}", _design(name, extras), page=_look_at([name], offset), settle=3.0)
        for name, extras, offset in GALLERY_SHOTS
    ),
    Shot(
        "gallery",
        _gallery_line_up,
        page=_look_at(["centaur", "twin_arm"], (2.4, -2.2, 1.5)),
        settle=3.0,
    ),
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
