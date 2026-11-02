"""Native screenshots and pointer gestures share one gate at desktop/phone DPR1/2."""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import time

import numpy as np
import pytest
from PIL import Image
from test_viewer_layout import pytestmark, qa, viewer  # noqa: F401
from test_viewer_startup_browser import READY
from viewer_camera import settle_camera
from viewer_capture import NativeCapture


class TracedSocket:
    def __init__(self, ws):
        self.ws = ws
        self.sent = {}
        self.calls = []

    async def send(self, raw):
        message = json.loads(raw)
        row = {**message, "start": time.monotonic_ns()}
        self.calls.append(row)
        self.sent[message["id"]] = row
        await self.ws.send(raw)
        row["sent"] = time.monotonic_ns()

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self.ws.recv()
        message = json.loads(raw)
        if row := self.sent.get(message.get("id")):
            row["reply"] = time.monotonic_ns()
        return raw


@pytest.mark.parametrize("size", ["laptop", "phone"])
@pytest.mark.parametrize("dpr", [1, 2])
def test_concurrent_capture_and_native_gesture_do_not_share_screenshot_scaling(viewer, tmp_path, size, dpr):  # noqa: F811
    async def scenario(tab, server):
        server.pg.session.paused = True
        server.pg.session.reset()
        width, height, _ = qa.SIZES[size]
        await tab.page.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=dpr,
            mobile=size == "phone",
        )
        await tab.page.call("Page.navigate", url=f"http://127.0.0.1:{tab.port}/?lang=en")
        await tab.until(READY + " && window.__robotforge.time===0")
        await settle_camera(tab)
        assert await tab.eval("document.title.toLowerCase().includes('robotforge')")
        assert await tab.eval("location.href") == f"http://127.0.0.1:{tab.port}/?lang=en"
        assert await tab.eval("devicePixelRatio") == dpr

        def runtime():
            session = server.pg.session
            with session.lock:
                return {
                    "identity": (id(session.model), id(session.data), session.version, session.generation),
                    "time": session.data.time,
                    "model": {
                        k: getattr(session.model, k).tobytes()
                        for k in dir(session.model)
                        if isinstance(getattr(session.model, k), np.ndarray)
                    },
                    "data": {k: getattr(session.data, k).tobytes() for k in ("qpos", "qvel", "ctrl")},
                    "world": server.pg.store.file.read_bytes(),
                    "paused": session.paused,
                }

        before = runtime()
        socket = TracedSocket(tab.ws)
        cdp = importlib.import_module("demos").Cdp(socket)
        cdp.session = tab.page.session
        reader = asyncio.create_task(cdp.read())
        tab.page = cdp
        gate = NativeCapture(tab)
        screenshots = []
        try:
            await tab.run("""window.__captureEvents=[];
              document.addEventListener('click',e=>window.__captureEvents.push({trusted:e.isTrusted,
                target:e.target.closest('#btn-frame')?.id,client:[e.clientX,e.clientY]}),true);
            """)
            for index, screenshot_first in enumerate((True, False)):
                point = await tab.run("""const r=document.getElementById('btn-frame').getBoundingClientRect();
                  return [r.x+r.width/2,r.y+r.height/2];""")
                click = gate.click("#btn-frame") if index == 0 else gate.click(point=point)
                shot = gate.screenshot(format="jpeg", quality=90, fromSurface=True, captureBeyondViewport=False)
                results = await asyncio.wait_for(
                    asyncio.gather(*((shot, click) if screenshot_first else (click, shot))), 15
                )
                result = results[0 if screenshot_first else 1]
                raw = base64.b64decode(result["data"], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    assert image.format == "JPEG" and image.size == (width * dpr, height * dpr)
                (tmp_path / f"{size}-{dpr}-{index}.jpg").write_bytes(raw)
                screenshots.append(point)
                await settle_camera(tab)
            events = await tab.eval("window.__captureEvents")
            assert len(events) == 2
            for event, point in zip(events, screenshots, strict=True):
                assert event["trusted"] and event["target"] == "btn-frame", event
                assert event["client"] == pytest.approx(point, rel=0, abs=1), (event, point)
            assert not await tab.eval("window.__qaErrors")
            assert await tab.eval("window.__robotforge.three().renderer.getContext().getError()") == 0
            assert runtime() == before
            calls = socket.calls
            shots = [c for c in calls if c["method"] == "Page.captureScreenshot"]
            gestures = [c for c in calls if c["method"] in ("Input.dispatchMouseEvent", "Input.dispatchTouchEvent")]
            assert len(shots) == 2 and len(gestures) == (4 if size == "phone" else 6)
            assert all("reply" in c and c["start"] <= c["sent"] <= c["reply"] for c in shots + gestures)
            assert all(g["reply"] <= s["start"] or s["reply"] <= g["start"] for s in shots for g in gestures)
            assert shots[0]["reply"] <= gestures[0]["start"]
            assert gestures[-1]["reply"] <= shots[1]["start"]
            (tmp_path / f"{size}-{dpr}-trace.json").write_text(
                json.dumps({"events": events, "calls": calls}, indent=2), encoding="utf-8"
            )
        finally:
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
            assert reader.done()

    viewer(size, "en", scenario, world=lambda pg: pg.add_robot("rover", template="rover"))
