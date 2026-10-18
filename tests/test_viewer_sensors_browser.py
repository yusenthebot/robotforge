"""The real viewer in headless Chrome: it draws what a robot's cameras see, for every camera picked, and a viewer's
show/hide switch holds for that sensor in that world only.

Picking another camera in the picture's head once left the panel blank (the new canvas was drawn with the old
one's context); only a page that renders can show that (tests/test_viewer_sensors.py checks the canvas choice in
Node too). Marked slow, and run where Chrome renders WebGL for sure (macOS, or anywhere with
ROBOTFORGE_BROWSER_TESTS=1), since a CI runner's software GL is not this test's subject.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn

from robotforge.playground import Playground
from robotforge.viewer.server import ViewerHub, create_app

CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
)
CANDIDATES = (os.environ.get("CHROME"), *CHROME_PATHS, shutil.which("google-chrome"))
CHROME = next((p for p in CANDIDATES if p and Path(p).exists()), None)
WANTED = sys.platform == "darwin" or os.environ.get("ROBOTFORGE_BROWSER_TESTS") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(CHROME is None, reason="Google Chrome is not installed"),
    pytest.mark.skipif(not WANTED, reason="set ROBOTFORGE_BROWSER_TESTS=1 to run browser tests here"),
]


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 30.0) -> bytes:
    end = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.read()
        except OSError:
            if time.monotonic() > end:
                raise
            time.sleep(0.2)


class Tab:
    """A page over the DevTools protocol (flat session), enough to run scripts in it."""

    def __init__(self, ws, session: str | None = None) -> None:
        self.ws, self.session, self.n = ws, session, 0

    async def call(self, method: str, **params):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params}
        if self.session:
            msg["sessionId"] = self.session
        await self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(await self.ws.recv())
            if reply.get("id") == self.n and reply.get("sessionId") == self.session:
                assert "error" not in reply, reply["error"]
                return reply.get("result", {})

    async def eval(self, expression: str):
        out = await self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
        assert "exceptionDetails" not in out, out["exceptionDetails"]
        return out.get("result", {}).get("value")


POST = {"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}}
READY = "!!(window.__robotforge?.scene && window.__robotforge.sensors().sensors.length)"


def _serve(pg: Playground):
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    port = _port()
    server = uvicorn.Server(uvicorn.Config(create_app(hub), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    _wait(f"http://127.0.0.1:{port}/")
    return server, thread, port


@pytest.fixture
def playground(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground.with_demo()


@pytest.fixture
def served(playground):
    """A post looking at a wall 2 m away with a depth, a segmentation and an rgb camera, served on a free port."""
    pg = playground
    pg.new_world("cameras")
    pg.add_robot("post", build=POST)
    pg.add_object("wall", shape="wall", pos=[2.05, 0], yaw=90, size=[3, 0.1, 1.0], color="red")
    for mode in ("depth", "segmentation", "rgb"):
        pg.add_sensor("post", {"type": "camera", "mode": mode, "name": mode})
    server, thread, port = _serve(pg)
    yield pg, port
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def chrome(tmp_path):
    """Headless Chrome's DevTools port."""
    chrome_port = _port()
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            f"--remote-debugging-port={chrome_port}",
            f"--user-data-dir={tmp_path / 'chrome'}",
            "--no-first-run",
            "--use-angle=metal" if sys.platform == "darwin" else "--use-gl=angle",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield chrome_port
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


@contextlib.asynccontextmanager
async def _page(chrome_port: int, url: str):
    import websockets

    version = json.loads(_wait(f"http://127.0.0.1:{chrome_port}/json/version"))
    async with websockets.connect(version["webSocketDebuggerUrl"], max_size=None) as ws:
        browser = Tab(ws)
        target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
        tab = Tab(ws, (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"])
        size = {"width": 1280, "height": 800, "deviceScaleFactor": 1, "mobile": False}
        await tab.call("Emulation.setDeviceMetricsOverride", **size)
        await tab.call("Page.bringToFront")
        await tab.call("Page.navigate", url=url)
        await _until(tab, READY)
        yield tab


async def _until(tab: Tab, expression: str, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while not await tab.eval(expression):
        assert time.monotonic() < end, f"the viewer never showed {expression}"
        await asyncio.sleep(0.25)


def test_every_camera_picked_draws_its_picture(served, chrome):
    _, port = served

    async def run() -> dict:
        async with _page(chrome, f"http://127.0.0.1:{port}/?lang=en") as tab:
            # Each camera picked in turn: the picture's middle (the wall, 1.65 m from the camera) and how much of it
            # is drawn at all (a blank canvas reads all zeros).
            return await tab.eval(
                """(async () => {
                  const out = {};
                  for (const mode of ["depth", "segmentation", "rgb"]) {
                    const buttons = [...document.querySelectorAll(".sensor-pip .seg button")];
                    buttons.find((b) => b.textContent === mode).click();
                    await new Promise((r) => setTimeout(r, 1500));
                    const c = document.querySelector(".sensor-pip canvas");
                    const px = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
                    let drawn = 0;
                    for (let i = 0; i < px.length; i += 4) if (px[i + 3] && (px[i] || px[i + 1] || px[i + 2])) drawn++;
                    const mid = 4 * (Math.floor(c.height / 2) * c.width + Math.floor(c.width / 2));
                    const middle = Array.from(px.slice(mid, mid + 3));
                    const picture = window.__robotforge.sensors().picture;
                    out[mode] = { drawn: drawn / (px.length / 4), middle, picture };
                  }
                  return out;
                })()"""
            )

    seen = asyncio.run(run())
    for mode, got in seen.items():
        assert got["picture"] == f"post/{mode}"
        assert got["drawn"] > 0.4, f"the {mode} picture is blank: {got}"
    assert seen["segmentation"]["middle"] != seen["depth"]["middle"]  # each canvas shows its own camera
    red, green, _ = seen["rgb"]["middle"]
    assert red > green + 40  # the red wall


def test_a_switch_holds_for_its_sensor_in_its_world_only(playground, chrome):
    """A lidar switched off in one world stays shown in the next world's lidar of the same name (it was hidden in
    every later world when the switch was kept by name alone), and a show request is applied once: a reload
    after the user switched the sensor off again keeps it off."""
    pg = playground
    pg.new_world("first")
    pg.add_robot("dog", build=POST)
    pg.add_sensor("dog", {"type": "lidar3d"})
    server, thread, port = _serve(pg)
    state = "window.__robotforge.sensors()"

    async def run() -> dict:
        out = {}
        async with _page(chrome, f"http://127.0.0.1:{port}/?lang=en") as tab:
            await tab.eval("window.__robotforge.showSensor('dog/lidar', false)")
            out["hidden"] = (await tab.eval(state))["shown"]
            pg.new_world("second")
            pg.add_robot("dog", build=POST)
            pg.add_sensor("dog", {"type": "lidar3d"})
            await _until(tab, f"{state}.points > 0")
            out["next_world"] = (await tab.eval(state))["shown"]
            pg.add_sensor("dog", {"type": "imu"})
            pg.show_sensors(["dog/lidar"])
            await _until(tab, f"{state}.shown.length === 2")
            await tab.eval("window.__robotforge.showSensor('dog/lidar', false)")
            await tab.call("Page.reload")
            await asyncio.sleep(0.5)
            await _until(tab, READY)
            await asyncio.sleep(1.0)  # a few sensors messages, each with the show request
            out["reloaded"] = (await tab.eval(state))["shown"]
        return out

    try:
        seen = asyncio.run(run())
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert seen["hidden"] == []
    assert seen["next_world"] == ["dog/lidar"]
    assert seen["reloaded"] == ["dog/imu"]
