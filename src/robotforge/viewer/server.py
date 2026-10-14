"""Local viewer server: static three.js client + WebSocket stream of body poses.

One asyncio task steps the shared SimSession in real time and broadcasts poses
to every connected browser. Browsers send back control messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from robotforge.sim.session import SimSession
from robotforge.viewer import scene

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
FRAME_HZ = 60.0
STATE_HZ = 4.0


# Revalidate on every load (cheap 304s) so an upgraded robotforge never shows a stale UI.
NO_CACHE = {"Cache-Control": "no-cache"}


class RevalidatingStaticFiles(StaticFiles):
    def file_response(self, *args, **kwargs) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


class ViewerHub:
    """Connects one SimSession to any number of browser clients."""

    def __init__(self, session: SimSession) -> None:
        self.session = session
        self.clients: set[WebSocket] = set()
        self._payload_version = -1
        self._mesh_payload: scene.MeshPayload | None = None
        self._scene_msg: dict | None = None
        self._announced_version = session.version
        # Hooks the playground installs: extra message types (e.g. teleop) and extra scene info.
        self.handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.scene_extras: Callable[[], dict[str, Any]] | None = None

    # ---- scene description (cached per model version) -------------------------------

    def _ensure_scene(self) -> None:
        s = self.session
        if self._payload_version == s.version:
            return
        with s.lock:
            payload = scene.build_mesh_payload(s.model)
            self._scene_msg = scene.describe(s.model, s.version, payload.entries)
            if self.scene_extras is not None:
                self._scene_msg.update(self.scene_extras())
            self._mesh_payload = payload
            self._payload_version = s.version

    def scene_message(self) -> dict:
        self._ensure_scene()
        assert self._scene_msg is not None
        return self._scene_msg

    def mesh_blob(self, version: int) -> bytes | None:
        self._ensure_scene()
        if version != self._payload_version or self._mesh_payload is None:
            return None
        return self._mesh_payload.blob

    def state_message(self) -> dict:
        s = self.session
        return {
            "type": "state",
            "paused": s.paused,
            "speed": s.speed,
            "time": float(s.data.time),
            "rtf": round(s.realtime_factor, 3),
            "ctrl": [float(c) for c in s.data.ctrl],
            "source": s.source,
            "clients": len(self.clients),
            "behaviors": [c.status() for c in list(s.controllers.values())],
        }

    # ---- client messages ------------------------------------------------------------

    def handle(self, msg: dict[str, Any]) -> None:
        s = self.session
        kind = msg.get("type")
        if kind == "cmd":
            cmd = msg.get("cmd")
            if cmd == "play":
                s.paused = False
            elif cmd == "pause":
                s.paused = True
            elif cmd == "reset":
                s.reset()
            elif cmd == "step":
                s.paused = True
                s.step(1)
            else:
                raise ValueError(f"unknown cmd {cmd!r}")
        elif kind == "speed":
            s.speed = float(np.clip(float(msg["value"]), 0.05, 4.0))
        elif kind == "ctrl":
            s.set_ctrl(int(msg["id"]), float(msg["value"]))
        elif kind == "drag_start":
            s.start_drag(int(msg["body"]), np.asarray(msg["point"], float))
        elif kind == "drag_move":
            s.move_drag(np.asarray(msg["target"], float))
        elif kind == "drag_end":
            s.end_drag()
        elif kind in self.handlers:
            self.handlers[kind](msg)
        else:
            raise ValueError(f"unknown message type {kind!r}")

    # ---- broadcasting ---------------------------------------------------------------

    async def _send_all(self, *, text: str | None = None, data: bytes | None = None) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                if text is not None:
                    await ws.send_text(text)
                if data is not None:
                    await ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def run(self) -> None:
        """Step physics in real time and stream poses until cancelled."""
        last = time.perf_counter()
        last_state = 0.0
        while True:
            tick_start = time.perf_counter()
            # The real-time clock only runs while someone is watching, so an agent's
            # world does not drift between tool calls.
            if self.clients and self.session.clock_owner is None:
                self.session.advance(tick_start - last)
            last = tick_start
            if self.clients:
                if self.session.version != self._announced_version:
                    self._announced_version = self.session.version
                    await self._send_all(text=json.dumps(self.scene_message()))
                await self._send_all(data=self.session.body_poses().tobytes())
                if tick_start - last_state > 1.0 / STATE_HZ:
                    last_state = tick_start
                    await self._send_all(text=json.dumps(self.state_message()))
            elapsed = time.perf_counter() - tick_start
            await asyncio.sleep(max(0.0, 1.0 / FRAME_HZ - elapsed))


def create_app(hub: ViewerHub) -> Starlette:
    async def index(_request):
        return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)

    async def mesh_blob(request):
        blob = hub.mesh_blob(int(request.path_params["version"]))
        if blob is None:
            return Response("stale scene version", status_code=409)
        return Response(blob, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    async def scene_json(_request):
        return JSONResponse(hub.scene_message())

    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_text(json.dumps(hub.scene_message()))
            await ws.send_text(json.dumps(hub.state_message()))
            while True:
                raw = await ws.receive_text()
                try:
                    hub.handle(json.loads(raw))
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(hub.run())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return Starlette(
        routes=[
            Route("/", index),
            Route("/scene.json", scene_json),
            Route("/mesh/{version:int}.bin", mesh_blob),
            WebSocketRoute("/ws", ws_endpoint),
            Mount("/static", RevalidatingStaticFiles(directory=STATIC_DIR), name="static"),
        ],
        lifespan=lifespan,
    )
