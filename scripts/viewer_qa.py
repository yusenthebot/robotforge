"""A pixel pass over the viewer: every panel and the dock, plain worlds (robots in each scene, their paths, the
gallery's designs and their joint sliders) and every lesson's first and last step, in Chinese and English, on a
phone, a laptop and a wide screen.

    uv run python scripts/viewer_qa.py --out DIR [--lessons pd_control,gait] [--no-lessons] [--no-panels]
                                       [--no-worlds] [--sizes phone,laptop,wide] [--langs zh-CN,en]

One viewer server runs in this process on a fresh ROBOTFORGE_HOME (the real Menagerie models), and one headless
Chrome per language and size watches it at once, each sized from the start, so every page frames its own camera
the way a learner's would. A lesson is started by a click in the first page; the others open it from the scene
the server sends, as a second window would. It is played with the moves of robotforge.learn.playthrough, but in
real time: the viewer's own loop steps the physics and ticks the lesson (so the plots get every sample), and the
player only waits for simulated time to pass. Each page is captured on the lesson's first step, after its last
step passed, and after Finish.

Every capture comes with a layout report (DIR/report.json, a summary printed at the end): panels on top of each
other or off the screen, text cut off, words broken across two lines (Intl.Segmenter decides what a word is, so
Chinese counts too), panels that scroll sideways, the robots on screen that a panel covers, script errors.

Needs Google Chrome (or CHROME=/path/to/chrome). Maintainer tool, not part of robotforge.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
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
import screenshots  # noqa: E402  (the same Chrome driver and world helpers)

SIZES: dict[str, tuple[int, int, int]] = {  # CSS px and device pixel ratio
    "phone": (390, 844, 3),
    "laptop": (1280, 800, 2),
    "wide": (1920, 1080, 1),  # a desktop monitor
}
LANGS = ("zh-CN", "en")
SETTLE = 2.5  # s for a page to draw, frame its camera and ease its view into the free space
EXIT_TIMEOUT = 240.0  # s for a lesson to end and the world to come back
OPEN_TIMEOUT = 60.0  # s for a Chrome started on a busy machine to take the DevTools connection
RETURNING = {"robotforge.coached": "1", "robotforge.world": "0"}  # a returning visitor: no hint, no open panel
SMALL_ROBOT = 30  # CSS px: a robot drawn less tall than this is reported as too small to make out

# ---- the layout report, run in the page ----------------------------------------------------------------------

LINT = r"""
(() => {
  const W = innerWidth, H = innerHeight;
  const lang = document.documentElement.lang || "en";
  const out = { size: [W, H], errors: (window.__qaErrors || []).slice(0, 8), rects: {}, overlaps: [], offscreen: [],
    clipped: [], broken: [], sideways: [], robots: {}, tiny: [] };
  const shown = (el) => {
    if (!el || !el.isConnected) return false;
    for (let e = el; e; e = e.parentElement) {
      if (e.hidden) return false;
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden" || Number(cs.opacity) < 0.05) return false;
    }
    return el.getClientRects().length > 0;
  };
  const tag = (el) => el.id ? "#" + el.id
    : (typeof el.className === "string" && el.className.trim() ? "." + el.className.trim().split(/\s+/)[0]
    : el.tagName.toLowerCase());
  const where = (el) => {
    const parts = [];
    for (let e = el; e && e !== document.body && parts.length < 3; e = e.parentElement) parts.unshift(tag(e));
    return parts.join(" ");
  };
  const SURFACES = ["#learn-panel", "#lesson-panel", "#lesson-plots", "#world-panel", "#drawer", "#help", ".dock",
    "#drive", "#coach", "#toast", "#banner", "#empty", ".top .title", "#learn-toggle", "#world-toggle",
    "#drawer-toggle", "#help-toggle"];
  const surfaces = [];
  for (const s of SURFACES) {
    const el = document.querySelector(s);
    if (!el || !shown(el) || (s === "#toast" && !el.classList.contains("show"))) continue;
    const r = el.getBoundingClientRect();
    surfaces.push([s, el, r]);
    out.rects[s] = [r.left, r.top, r.right, r.bottom].map((v) => Math.round(v * 10) / 10);
  }
  // A toggle sits on the top bar by design; the bar's own pieces never overlap each other.
  for (let i = 0; i < surfaces.length; i++) for (let j = i + 1; j < surfaces.length; j++) {
    const [a, , ra] = surfaces[i], [b, , rb] = surfaces[j];
    const w = Math.min(ra.right, rb.right) - Math.max(ra.left, rb.left);
    const h = Math.min(ra.bottom, rb.bottom) - Math.max(ra.top, rb.top);
    if (w > 0.5 && h > 0.5) out.overlaps.push(`${a} / ${b}: ${w.toFixed(0)} x ${h.toFixed(0)} px`);
  }
  for (const [s, , r] of surfaces) {
    if (r.left < -0.5 || r.top < -0.5 || r.right > W + 0.5 || r.bottom > H + 0.5)
      out.offscreen.push(`${s}: ${[r.left, r.top, r.right, r.bottom].map((v) => v.toFixed(0)).join(", ")}`);
  }
  // Text cut off: an ellipsis or a hidden overflow with more inside than shows, or text running past the panel
  // edge that clips it. A panel's vertical scroll is not cut-off text.
  const clipBox = (el) => {
    for (let e = el.parentElement; e; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.overflowX !== "visible") return e;
    }
    return null;
  };
  const range = document.createRange();
  const segmenter = new Intl.Segmenter(lang, { granularity: "word" });
  const seen = new Set();
  for (const [s, root] of surfaces) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node.data;
      const el = node.parentElement;
      if (!text.trim() || !el || !shown(el) || seen.has(node)) continue;
      seen.add(node);
      const cs = getComputedStyle(el);
      const size = parseFloat(cs.fontSize);
      if (size < 10.5) out.tiny.push(`${where(el)} ${size}px "${text.trim().slice(0, 24)}"`);
      if ((cs.overflowX !== "visible" || cs.textOverflow === "ellipsis") && el.scrollWidth > el.clientWidth + 1)
        out.clipped.push(`${where(el)}: "${text.trim().slice(0, 40)}" needs ${el.scrollWidth} px, `
          + `has ${el.clientWidth}`);
      range.selectNodeContents(node);
      const box = range.getBoundingClientRect();
      const clip = clipBox(el);
      if (clip) {
        const c = clip.getBoundingClientRect();
        if (box.right > c.right + 1 || box.left < c.left - 1)
          out.clipped.push(`${where(el)}: "${text.trim().slice(0, 40)}" runs past ${tag(clip)} by `
            + `${Math.max(box.right - c.right, c.left - box.left).toFixed(0)} px`);
      }
      // Line breaks inside a word: the top of each character, and the words around every new line.
      const tops = [];
      for (let i = 0; i < text.length; i++) {
        range.setStart(node, i);
        range.setEnd(node, i + 1);
        const rs = range.getClientRects();
        tops.push(rs.length ? rs[0].top : null);
      }
      const lineH = parseFloat(cs.lineHeight) || size * 1.3;
      let prev = null;
      const breaks = [];
      for (let i = 0; i < text.length; i++) {
        if (tops[i] === null) continue;
        if (prev !== null && tops[i] > tops[prev] + lineH / 2) breaks.push(i);
        prev = i;
      }
      if (!breaks.length || cs.hyphens === "auto") continue; // a hyphenated word is broken on purpose
      const words = [...segmenter.segment(text)];
      for (const i of breaks) {
        const w = words.find((g) => g.index < i && i < g.index + g.segment.length);
        if (!w || !w.isWordLike) continue;
        const kind = text.trim().length <= 24 ? "label" : "prose";
        out.broken.push(`${kind} ${where(el)}: "${w.segment.slice(0, i - w.index)}|${w.segment.slice(i - w.index)}"`);
      }
    }
    if (root.scrollWidth > root.clientWidth + 1 && getComputedStyle(root).overflowX !== "visible")
      out.sideways.push(`${s}: ${root.scrollWidth} px of content in ${root.clientWidth}`);
  }
  // The robots on screen and how much of each a panel covers: mesh vertices projected with the viewer's camera.
  const rf = window.__robotforge;
  if (rf && rf.scene && rf.three) {
    const { scene, camera } = rf.three();
    const robotOf = new Map();
    // A world's robots are named "go2/trunk"; a lesson's rig has plain body names, and counts as one. Outside a
    // lesson a plain name is a loose object (the park's ball), not a robot.
    const lesson = !!(rf.lesson && rf.lesson());
    for (const b of rf.scene.bodies) {
      const robot = b.name.includes("/") ? b.name.split("/")[0] : lesson ? "rig" : null;
      if (b.dynamic && robot) robotOf.set(b.id, robot);
    }
    const points = {};
    const V = camera.position.constructor;
    const v = new V();
    camera.updateMatrixWorld();
    scene.updateMatrixWorld(true);
    scene.traverse((o) => {
      if (!o.isMesh || !o.visible || !robotOf.has(o.userData.body)) return;
      let visible = true;
      for (let p = o.parent; p; p = p.parent) if (!p.visible) visible = false;
      if (!visible) return;
      const pos = o.geometry.attributes.position;
      const robot = robotOf.get(o.userData.body);
      const list = (points[robot] ??= []);
      const every = Math.max(1, Math.floor(pos.count / 150));
      for (let k = 0; k < pos.count; k += every) {
        v.fromBufferAttribute(pos, k).applyMatrix4(o.matrixWorld).project(camera);
        if (v.z > 1) continue;
        list.push([((v.x + 1) / 2) * W, ((1 - v.y) / 2) * H]);
      }
    });
    const panels = surfaces.map(([, , r]) => r);
    for (const [robot, list] of Object.entries(points)) {
      if (!list.length) continue;
      let covered = 0, off = 0;
      for (const [x, y] of list) {
        if (x < 0 || y < 0 || x > W || y > H) off++;
        else if (panels.some((r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom)) covered++;
      }
      const share = (n) => Math.round((100 * n) / list.length);
      // How tall it is on the screen, in CSS px: on a phone a whole warehouse fitted leaves its robots a few px high.
      const ys = list.map(([, y]) => y);
      const tall = Math.round(Math.max(...ys) - Math.min(...ys));
      out.robots[robot] = { covered: share(covered), off: share(off), tall };
    }
  }
  out.tiny = [...new Set(out.tiny)].slice(0, 12);
  return out;
})()
"""

# Close what is open, as a learner would with the buttons: the picker, the drawers, help.
CLOSE_ALL = """
    for (const id of ["learn-toggle", "world-toggle", "drawer-toggle", "help-toggle"]) {
        const b = document.getElementById(id);
        if (b && b.getAttribute("aria-expanded") === "true") b.click();
    }
"""

ERRORS = """
    window.__qaErrors = [];
    window.addEventListener("error", (e) => window.__qaErrors.push(String(e.message)));
    window.addEventListener("unhandledrejection", (e) => window.__qaErrors.push(String(e.reason)));
    const consoleError = console.error.bind(console);
    console.error = (...args) => { window.__qaErrors.push(args.map(String).join(" ")); consoleError(...args); };
"""


# ---- the server, in this process --------------------------------------------------------------------------------


class Server:
    """The viewer for one world, as `robotforge open` serves it, on a thread of this process."""

    def __init__(self, world_dir: Path, port: int) -> None:
        import uvicorn

        from robotforge.cli import load_playground
        from robotforge.viewer.server import ViewerHub, create_app

        self.pg = load_playground(world_dir)
        self.port = port
        hub = ViewerHub(self.pg.session)
        self.pg.attach_viewer(hub)
        stream = hub.lesson_stream
        self.latest: dict[str, Any] = {}

        def recorded() -> dict[str, Any] | None:  # the player reads the update the pages got
            update = stream() if stream is not None else None
            if update and "live" in update:
                self.latest = update
            return update

        hub.lesson_stream = recorded
        config = uvicorn.Config(create_app(hub), host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> Server:
        self.thread.start()
        screenshots._wait_http(f"http://127.0.0.1:{self.port}/", screenshots.READY_TIMEOUT)
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def live_player(server: Server) -> Any:
    """A playthrough Player whose `play` waits for the viewer's loop to play the physics in real time."""
    from robotforge.learn.playthrough import TICK, Player, TimeUp

    class LivePlayer(Player):
        def play(self, seconds: float, realtime: bool = False) -> dict[str, Any]:
            want = max(TICK, seconds)
            done = 0.0
            last = self.session.data.time
            while done < want - 1e-9:
                if self.sim >= self._sim_end:
                    raise TimeUp(f"out of time: the step's {self.limits[0]:g} s of physics ran out")
                if time.perf_counter() >= self._wall_end:
                    raise TimeUp(f"out of time: the step's {self.limits[1]:g} s on the clock ran out")
                time.sleep(TICK / 2)
                now = self.session.data.time
                step = now - last if now >= last else now  # time went back (a reset): count from the new start
                last = now
                done += step
                self.sim += step
            self.update = server.latest or self.update
            return self.update

    return LivePlayer(server.pg)


def play_lesson(server: Server, lesson_id: str) -> list[Any]:
    """Make every step's moves in the lesson a page started; the reports of the steps played."""
    from robotforge.learn.playthrough import RECIPES, _play_step

    runner = server.pg.lessons
    if runner.lesson is None or runner.lesson.id != lesson_id:
        raise RuntimeError(f"{lesson_id} is not running")
    player = live_player(server)
    reports = []
    for step in runner.lesson.steps:
        report = _play_step(player, step, RECIPES[lesson_id])
        reports.append(report)
        if not report.passed:
            break
    return reports


# ---- the pages --------------------------------------------------------------------------------------------------


@dataclass
class Tab:
    """One headless Chrome showing the viewer at one size, in one language."""

    lang: str
    size: str
    port: int
    home: Path
    chrome: subprocess.Popen[bytes] | None = None
    ws: Any = None
    page: Any = None
    errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.size}_{self.lang.split('-')[0]}"

    async def open(self) -> None:
        import websockets

        chrome_port = screenshots._free_port()
        self.chrome = subprocess.Popen(
            [
                screenshots._chrome(),
                "--headless=new",
                f"--remote-debugging-port={chrome_port}",
                f"--user-data-dir={self.home / f'chrome-{self.label}'}",
                f"--lang={self.lang}",
                "--hide-scrollbars",
                "--no-first-run",
                "--use-angle=metal" if sys.platform == "darwin" else "--use-gl=angle",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        version = json.loads(screenshots._wait_http(f"http://127.0.0.1:{chrome_port}/json/version", 20))
        self.ws = await websockets.connect(
            version["webSocketDebuggerUrl"], max_size=None, ping_interval=None, open_timeout=OPEN_TIMEOUT
        )
        browser = screenshots.Page(self.ws)
        target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
        session = (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"]
        self.page = screenshots.Page(self.ws, session)
        width, height, scale = SIZES[self.size]
        phone = self.size == "phone"
        await self.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=scale, mobile=phone
        )
        if phone:  # a phone has a finger, not a mouse: the page hides its keyboard hints
            await self.page.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
        await self.page.call("Page.enable")
        await self.page.call("Page.bringToFront")
        await self.page.call("Emulation.setFocusEmulationEnabled", enabled=True)
        await self.page.call("Page.addScriptToEvaluateOnNewDocument", source=ERRORS)
        await self.navigate()

    async def navigate(self) -> None:
        await self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?lang={self.lang}")
        await self.until("!!(window.__robotforge && window.__robotforge.scene && window.__robotforge.time > 0)")

    async def eval(self, script: str) -> Any:
        return await self.page.eval(script)

    async def run(self, body: str) -> Any:
        return await self.page.eval(f"(async () => {{ {body} }})()")

    async def until(self, condition: str, timeout: float = screenshots.READY_TIMEOUT) -> None:
        end = time.monotonic() + timeout
        while not await self.eval(condition):
            if time.monotonic() > end:
                raise TimeoutError(f"{self.label}: never {condition}")
            await asyncio.sleep(0.2)

    async def capture(self, path: Path) -> dict[str, Any]:
        report = await self.eval(LINT)
        data = (await self.page.call("Page.captureScreenshot", format="png"))["data"]
        save(base64.b64decode(data), path)
        return report

    async def close(self) -> None:
        if self.ws is not None:
            with contextlib.suppress(Exception):
                await self.ws.close()
        if self.chrome is not None:
            self.chrome.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.chrome.wait(timeout=10)


def save(png: bytes, path: Path) -> None:
    """Keep the capture at its full device resolution, as a JPEG good enough to judge single pixels."""
    from PIL import Image

    Image.open(io.BytesIO(png)).convert("RGB").save(path, "JPEG", quality=92, optimize=True)


# ---- the pass ---------------------------------------------------------------------------------------------------


@dataclass
class Pass:
    tabs: list[Tab]
    out: Path
    reports: dict[str, Any] = field(default_factory=dict)

    async def everywhere(self, body: str) -> list[Any]:
        return await asyncio.gather(*(tab.run(body) for tab in self.tabs))

    async def capture(self, name: str) -> None:
        paths = [self.out / f"{name}_{tab.label}.jpg" for tab in self.tabs]
        found = await asyncio.gather(*(tab.capture(path) for tab, path in zip(self.tabs, paths, strict=True)))
        for path, report in zip(paths, found, strict=True):
            self.reports[path.name] = report
        print(f"  {name}: {', '.join(p.name for p in paths)}", flush=True)


PLAY = "document.getElementById('btn-play').click();"
PanelStep = tuple[str, str, float]  # capture name, page script, seconds to settle

PANELS: list[PanelStep] = [
    ("home", CLOSE_ALL, 1.0),
    ("learn", CLOSE_ALL + "document.getElementById('learn-toggle').click();", 1.0),
    ("learn_end", "document.getElementById('learn-panel').scrollTop = 1e6;", 0.5),
    ("world_maps", CLOSE_ALL + "document.getElementById('world-toggle').click();", 1.0),
    ("world_terrain", "document.querySelector('[data-tab=\"terrain\"]').click();", 0.6),
    ("world_modes", "document.querySelector('[data-tab=\"modes\"]').click();", 0.6),
    (
        "robots",
        CLOSE_ALL + "window.__robotforge.select('go2'); document.getElementById('drawer-toggle').click();",
        2.0,
    ),
    ("robots_end", "document.getElementById('drawer').scrollTop = 1e6;", 0.5),
    ("help", CLOSE_ALL + "document.getElementById('help-toggle').click();", 1.0),
    ("recording", CLOSE_ALL + "document.getElementById('btn-record').click();", 2.3),
    ("paused", "document.getElementById('btn-record').click(); document.getElementById('btn-play').click();", 1.0),
    ("keyboard_focus", PLAY + "document.getElementById('btn-reset').focus();", 0.6),
]


async def panels(run: Pass) -> None:
    # The very first visit: the hint over the view (it leaves after 9 s), and on a big screen the scene panel open.
    await run.everywhere("localStorage.clear();")
    await asyncio.gather(*(tab.navigate() for tab in run.tabs))
    await asyncio.sleep(1.5)
    await run.capture("first_visit")
    settings = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in RETURNING.items())
    await run.everywhere(settings + "localStorage.removeItem('robotforge.drawer');")
    await asyncio.gather(*(tab.navigate() for tab in run.tabs))
    # A recording ends in a download; hand it to nobody instead.
    await run.everywhere("window.__robotforge.onRecording = () => {};")
    await asyncio.sleep(SETTLE)
    for name, script, settle in PANELS:
        await run.everywhere(script)
        await asyncio.sleep(settle)
        await run.capture(name)
    await run.everywhere(CLOSE_ALL + "document.activeElement?.blur();")


# ---- plain worlds: robots in the scenes, their paths, the gallery's designs ------------------------------------

FIT = "document.getElementById('btn-frame').click();"  # what a learner presses after a map changes under them
MARK_SCENE = "window.__qaScene = window.__robotforge.scene;"
NEW_SCENE = "window.__robotforge.scene !== window.__qaScene && document.getElementById('loading').hidden"
Edit = Callable[[Any], Any]


def _scene(preset: str) -> Edit:
    return lambda pg: pg.load_scene(preset)


def _go(robot: str, to: list[float]) -> Edit:
    return lambda pg: pg.behaviors.navigate(robot, to)


def _gallery(pg: Any) -> None:
    """The park's robots out, an empty floor, and the gallery's line-up of designs in."""
    for robot in [r.name for r in pg.current_world().robots]:
        pg.remove(robot)
    pg.clear_scene()
    screenshots._gallery_line_up(pg)


def _select(robot: str, drawer: bool = True) -> str:
    opened = "document.getElementById('drawer-toggle').click();" if drawer else ""
    return CLOSE_ALL + f"window.__robotforge.select({robot!r});" + opened


@dataclass(frozen=True)
class WorldView:
    """One capture of a plain world: a change to the world in Python, as an agent's tool call makes it (None for
    none), whether that change rebuilds the scene the pages show, then the page script of a learner's clicks."""

    name: str
    edit: Edit | None
    script: str
    settle: float  # s
    rebuilds: bool = False


# Every page sees a change to the world; only a map picked in the page frames the camera on its own, so after an
# agent's change the pages press Fit view, as a learner would.
WORLDS: list[WorldView] = [
    WorldView("park_path", _go("go2", [-3.0, 2.6]), CLOSE_ALL, 3.0),
    WorldView("park_selected", None, _select("dog", drawer=False), 1.5),
    WorldView("kitchen", _scene("kitchen"), CLOSE_ALL + FIT, 2.5, rebuilds=True),
    WorldView("kitchen_robots", None, _select("go2"), 2.0),
    WorldView("warehouse", _scene("warehouse"), CLOSE_ALL + FIT, 2.5, rebuilds=True),
    WorldView("warehouse_path", _go("go2", [6.0, -1.4]), CLOSE_ALL, 4.0),
    WorldView("maze", _scene("maze"), CLOSE_ALL + FIT, 2.5, rebuilds=True),
    WorldView("obstacle_course", _scene("obstacle_course"), CLOSE_ALL + FIT, 2.5, rebuilds=True),
    WorldView("desk_lab", _scene("desk_lab"), CLOSE_ALL + FIT, 2.5, rebuilds=True),
    WorldView("gallery", _gallery, CLOSE_ALL + FIT, 3.5, rebuilds=True),
    WorldView("gallery_centaur", None, _select("centaur"), 2.0),
    WorldView("gallery_centaur_end", None, "document.getElementById('drawer').scrollTop = 1e6;", 0.5),
    WorldView(
        "gallery_modes",
        None,
        _select("centaur", drawer=False)
        + "document.getElementById('world-toggle').click();"
        + "document.querySelector('[data-tab=\"modes\"]').click();",
        1.0,
    ),
]


async def worlds(run: Pass, server: Server) -> None:
    for view in WORLDS:
        if view.edit is not None:
            await run.everywhere(MARK_SCENE)
            await asyncio.to_thread(view.edit, server.pg)
            if view.rebuilds:  # the last of the scenes the change sent: the gallery is eight edits in a row
                last = f"{NEW_SCENE} && window.__robotforge.scene.nbody === {server.pg.session.model.nbody}"
                await asyncio.gather(*(tab.until(last) for tab in run.tabs))
                await asyncio.sleep(1.0)  # the first poses come in and the lights are fitted
        await run.everywhere(view.script)
        await asyncio.gather(*(tab.until("!window.__robotforge.tweening()") for tab in run.tabs))
        await asyncio.sleep(view.settle)
        await run.capture(f"world_{view.name}")
    await run.everywhere(CLOSE_ALL + "document.activeElement?.blur();")


START = """
    if (document.getElementById("learn-toggle").getAttribute("aria-expanded") !== "true")
        document.getElementById("learn-toggle").click();
    await new Promise((r) => setTimeout(r, 400));
    document.querySelector('[data-lesson="%s"]').click();
"""


async def lesson(run: Pass, server: Server, lesson_id: str) -> list[Any]:
    lead = run.tabs[0]
    await lead.run(START % lesson_id)
    shows = f"window.__robotforge.lesson()?.id === {lesson_id!r} && !document.getElementById('lesson-panel').hidden"
    await asyncio.gather(*(tab.until(shows) for tab in run.tabs))
    await asyncio.gather(*(tab.until("!window.__robotforge.tweening()") for tab in run.tabs))
    await asyncio.sleep(SETTLE)
    await run.capture(f"{lesson_id}_first")
    steps = await asyncio.to_thread(play_lesson, server, lesson_id)
    await asyncio.sleep(1.5)  # the last update reaches every page
    await run.capture(f"{lesson_id}_last")
    if all(s.passed for s in steps):
        await run.everywhere("const b = document.getElementById('lesson-next'); if (!b.hidden) b.click();")
        await asyncio.sleep(1.0)
        await run.capture(f"{lesson_id}_done")
    await lead.run("document.getElementById('lesson-exit').click();")
    # Leaving waits for a training still running on its thread and rebuilds the world: give it time.
    await asyncio.gather(*(tab.until("window.__robotforge.lesson() === null", EXIT_TIMEOUT) for tab in run.tabs))
    await asyncio.sleep(1.0)
    return steps


async def recover(run: Pass, server: Server) -> None:
    """After a lesson that went wrong: leave it on the server and load every page afresh."""
    if server.pg.lessons.lesson is not None:
        await asyncio.to_thread(server.pg.lessons.exit)
    await asyncio.gather(*(tab.navigate() for tab in run.tabs))
    await asyncio.sleep(SETTLE)


def build_world(home: Path, build: Callable[[Any], None]) -> Path:
    from robotforge.playground import Playground

    pg = Playground.with_demo()
    pg.new_world("qa")
    build(pg)
    return home / "worlds" / "qa"


async def main_async(args: argparse.Namespace) -> int:
    from robotforge.learn import LESSONS

    lessons = [] if args.no_lessons else (args.lessons.split(",") if args.lessons else list(LESSONS))
    unknown = [i for i in lessons if i not in LESSONS]
    if unknown:
        raise SystemExit(f"unknown lessons: {', '.join(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    home = Path(tempfile.mkdtemp(prefix="rf-qa-"))
    os.environ["ROBOTFORGE_HOME"] = str(home)
    world_dir = build_world(home, screenshots._park)
    port = screenshots._free_port()
    report = args.out / "report.json"  # a pass may be run in parts: each adds to what the others found
    found = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"captures": {}, "lessons": {}}
    played: dict[str, Any] = found["lessons"]
    try:
        with Server(world_dir, port) as server:
            tabs = [Tab(lang, size, port, home) for lang in args.langs.split(",") for size in args.sizes.split(",")]
            run = Pass(tabs, args.out, found["captures"])

            def save() -> None:
                report.write_text(
                    json.dumps({"captures": run.reports, "lessons": played}, indent=1, ensure_ascii=False),
                    encoding="utf-8",
                )

            try:
                await asyncio.gather(*(tab.open() for tab in tabs))
                if not args.no_panels:
                    await panels(run)
                else:
                    settings = "".join(f"localStorage.setItem({k!r}, {v!r});" for k, v in RETURNING.items())
                    await run.everywhere(settings)
                    await asyncio.gather(*(tab.navigate() for tab in tabs))
                for lesson_id in lessons:
                    t0 = time.monotonic()
                    print(f"{lesson_id}:", flush=True)
                    try:
                        steps = await lesson(run, server, lesson_id)
                    except (TimeoutError, RuntimeError) as exc:  # note it, and go on with the next lesson
                        played[lesson_id] = [{"id": "", "passed": False, "failed": f"the pass broke off: {exc}"}]
                        print(f"  broke off: {exc}", flush=True)
                        save()
                        await recover(run, server)
                        continue
                    played[lesson_id] = [{"id": s.id, "passed": s.passed, "failed": s.failed} for s in steps]
                    save()
                    stuck = next((s for s in steps if not s.passed), None)
                    state = f"stuck at {stuck.id}: {stuck.failed}" if stuck else "played to the end"
                    print(f"  {state} ({time.monotonic() - t0:.0f} s)", flush=True)
                if not args.no_worlds:  # last: it changes the world the lessons come back to
                    print("worlds:", flush=True)
                    await worlds(run, server)
            finally:
                save()
                await asyncio.gather(*(tab.close() for tab in tabs))
    finally:
        shutil.rmtree(home, ignore_errors=True)
    summarize(run.reports)
    return 0


def summarize(reports: dict[str, Any]) -> None:
    """The findings, one line per capture that has any."""
    keys = ("errors", "overlaps", "offscreen", "clipped", "sideways", "broken")
    for name, r in reports.items():
        # Chinese prose may break between any two characters; a label, a heading or English may not.
        chinese = "_zh." in name
        r = {**r, "broken": [b for b in r.get("broken", []) if not (chinese and b.startswith("prose"))]}
        lines = [f"{k}: {'; '.join(r[k])}" for k in keys if r.get(k)]
        hidden = {robot: v for robot, v in r.get("robots", {}).items() if v["covered"] + v["off"] >= 20}
        if hidden:
            under = (f"{k} {v['covered']}% under a panel, {v['off']}% off screen" for k, v in hidden.items())
            lines.append("robots hidden: " + ", ".join(under))
        robots = r.get("robots", {}).items()
        small = {robot: v["tall"] for robot, v in robots if v.get("tall", SMALL_ROBOT) < SMALL_ROBOT}
        if small:
            lines.append("robots small: " + ", ".join(f"{k} {tall} px tall" for k, tall in small.items()))
        if lines:
            print(f"{name}\n    " + "\n    ".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lessons", help="comma-separated lesson ids (default: all)")
    parser.add_argument("--no-lessons", action="store_true")
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--no-worlds", action="store_true", help="skip the scenes, paths and gallery views")
    parser.add_argument("--sizes", default=",".join(SIZES))
    parser.add_argument("--langs", default=",".join(LANGS))
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
