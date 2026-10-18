"""The real viewer in headless Chrome draws what a robot's cameras see, for every camera picked.

Picking another camera in the picture's head once left the panel blank (the new canvas was drawn with the old
one's context); only a page that renders can show that. Marked slow, and run where Chrome renders WebGL for sure
(macOS, or anywhere with ROBOTFORGE_BROWSER_TESTS=1), since a CI runner's software GL is not this test's subject.
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


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A post looking at a wall 2 m away with a depth, a segmentation and an rgb camera, served on a free port."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("cameras")
    pg.add_robot("post", build={"base": {"shape": "box", "size": [0.2, 0.2, 0.2], "fixed": True}})
    pg.add_object("wall", shape="wall", pos=[2.05, 0], yaw=90, size=[3, 0.1, 1.0], color="red")
    for mode in ("depth", "segmentation", "rgb"):
        pg.add_sensor("post", {"type": "camera", "mode": mode, "name": mode})
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    port = _port()
    server = uvicorn.Server(uvicorn.Config(create_app(hub), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    _wait(f"http://127.0.0.1:{port}/")
    yield pg, port
    server.should_exit = True
    thread.join(timeout=10)


def test_every_camera_picked_draws_its_picture(served, tmp_path):
    import websockets

    _, port = served
    chrome_port = _port()
    chrome = subprocess.Popen(
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

    async def run() -> dict:
        version = json.loads(_wait(f"http://127.0.0.1:{chrome_port}/json/version"))
        async with websockets.connect(version["webSocketDebuggerUrl"], max_size=None) as ws:
            browser = Tab(ws)
            target = (await browser.call("Target.createTarget", url="about:blank"))["targetId"]
            tab = Tab(ws, (await browser.call("Target.attachToTarget", targetId=target, flatten=True))["sessionId"])
            size = {"width": 1280, "height": 800, "deviceScaleFactor": 1, "mobile": False}
            await tab.call("Emulation.setDeviceMetricsOverride", **size)
            await tab.call("Page.bringToFront")
            await tab.call("Page.navigate", url=f"http://127.0.0.1:{port}/?lang=en")
            end = time.monotonic() + 60
            while not await tab.eval("!!(window.__robotforge?.scene && window.__robotforge.sensors().sensors.length)"):
                assert time.monotonic() < end, "the viewer never drew the cameras"
                await asyncio.sleep(0.25)
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

    try:
        seen = asyncio.run(run())
    finally:
        chrome.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            chrome.wait(timeout=10)
    for mode, got in seen.items():
        assert got["picture"] == f"post/{mode}"
        assert got["drawn"] > 0.4, f"the {mode} picture is blank: {got}"
    assert seen["segmentation"]["middle"] != seen["depth"]["middle"]  # each canvas shows its own camera
    red, green, _ = seen["rgb"]["middle"]
    assert red > green + 40  # the red wall
