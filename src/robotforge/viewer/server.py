"""Local viewer server: static three.js client + WebSocket stream of body poses.

One asyncio task steps the shared SimSession in real time and broadcasts poses
to every connected browser. Browsers send back control messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from robotforge.sim.session import SimSession
from robotforge.viewer import scene

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
FRAME_HZ = 60.0
# Pose frames a viewer that says which it has taken in ("seen") may have unread before it is sent no more. A page
# drawing a heavy scene on a busy machine took in one pose frame per drawn frame (25 a second of the 60 sent), and
# its picture fell 15 s behind the simulation; viewers that never say (older pages) get every frame as before.
MAX_POSES_AHEAD = 3
STATE_HZ = 4.0
LESSON_HZ = 20.0
SENSOR_HZ = 10.0
MAP_HZ = 4.0
REPORT_EVERY = 5.0  # s between log reports of a failing tick


# Revalidate on every load (cheap 304s) so an upgraded robotforge never shows a stale UI.
NO_CACHE = {"Cache-Control": "no-cache"}


def safe_host(host_header: str) -> bool:
    """The Host a request was sent to is this machine by name or an IP address (the viewer on a LAN). A domain
    name means DNS rebinding: a web page whose name was pointed at 127.0.0.1 to reach the viewer."""
    host = urlsplit(f"//{host_header}").hostname or ""
    if host == "localhost":
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def same_origin(origin: str | None, host_header: str) -> bool:
    """A browser says which page opened a WebSocket; only the viewer's own pages may. Without an Origin it is
    not a browser (a script or an agent on this machine, which can reach the port anyway)."""
    return origin is None or urlsplit(origin).netloc == host_header


class LocalOnly:
    """Browsers let any web page open a WebSocket to 127.0.0.1, so without this a site visited while the viewer
    runs could load scenes into the user's world or drive their robots."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
            host = headers.get("host", "")
            allowed = safe_host(host) and (scope["type"] == "http" or same_origin(headers.get("origin"), host))
            if not allowed:
                log.warning(
                    "refused a %s request for host %r from origin %r", scope["type"], host, headers.get("origin")
                )
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await PlainTextResponse("forbidden", status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


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
        # Hooks the playground installs: extra message types (e.g. teleop) and extra scene and state info.
        self.handlers: dict[str, Callable[[dict[str, Any]], None]] = {}
        # Messages whose work takes a while (loading a scene, setting up MPC): run in a worker thread so
        # the stream keeps flowing, then answered with {"type": "reply", "id", "ok", "data" | "message"}.
        self.slow_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.scene_extras: Callable[[], dict[str, Any]] | None = None
        self.state_extras: Callable[[], dict[str, Any]] | None = None
        # While a lesson runs: its live update (step, plots, overlay numbers), sent LESSON_HZ times a second.
        self.lesson_stream: Callable[[], dict[str, Any] | None] | None = None
        # What the robots' sensors measure, sent SENSOR_HZ times a second while the world has any (None when it
        # has none): {"type": "sensors", "version" (the model's), "time", "sensors": [...]}, one entry per sensor
        # with its "name", "robot", "type", "added" (put on with add_sensor rather than one the robot came with),
        # "uid" (the same while it stays on its robot, new when added again: the page keeps its switch under it),
        # world "pos" and "quat" (w, x, y, z), the "body" it rides on with its "local_pos" and "local_quat" there,
        # and what drawing it needs. lidar and lidar3d: "origin", "points" (hit points, flat x, y, z list, at most
        # 1200 a lidar and 2400 a message), "rays", "hits", "range_m", "channels"; camera: "mode", "fov_deg",
        # "size", "range_m", "forward", "up"; force_torque: "force_n", "torque_nm" (world vectors at the pos), and
        # "payload_kg" (what it holds) or "supported_kg" (held up from below); imu: "accel_m_s2",
        # "gyro_rad_s", "orientation_deg" (roll, pitch, yaw); touch: "cells_m" (world positions), "force_n" per
        # cell, "total_n", and the pad on its body: "cells_local", "cell_half", "cell_shape" (box or sphere),
        # "pad_quat"; rangefinder: "direction", "distance_m", "range_m"; gps and odometry: "position_m" (odometry
        # also "heading_deg"); encoders: "rad" (and "m") by joint. "t" is when the value was sampled; "status":
        # "waiting" while a delayed sensor has none yet. "show" ({"id", "names"}, open_viewer(show=...)): sensors to
        # switch on, applied once a tab. The viewer's sensors.js draws them (sensorkit.js lists the fields it reads,
        # and a test checks they are sent). See docs/SENSORS.md.
        self.sensor_stream: Callable[[], dict[str, Any] | None] | None = None
        # The maps robots draw while they move: {"type": "map", "version", "robots": [...]}, MAP_HZ times a second
        # while any robot maps (None otherwise). Each robot entry has "robot", "mapping" (false: the map is kept as
        # it is), "scans", its estimated "pose" [x, y, heading] and the "odometry" pose alone, its "particles" (flat
        # x, y list) with their "sigma_m", its "error_m" and "odometry_error_m" against the true pose (for the
        # page's readout), "mapped_m2", and now and then its "grid": {"x0", "y0", "res", "rows", "cols", "cells"}
        # with "cells" a base64 of a zlib-deflated byte per cell (0 unknown, 1 free, 2 occupied), row by row from
        # row 0 at y0. The viewer's mapview.js draws them. See docs/PERCEPTION.md.
        self.map_stream: Callable[[], dict[str, Any] | None] | None = None
        self._slow_tasks: set[asyncio.Task] = set()
        # Pose frames sent to each viewer, and how many it said it took in (only viewers that say so are paced).
        self._sent: dict[WebSocket, int] = {}
        self._seen: dict[WebSocket, int] = {}

    # ---- scene description (cached per model version) -------------------------------

    def _ensure_scene(self) -> None:
        s = self.session
        if self._payload_version == s.version:
            return
        with s.lock:
            version = s.version
            payload = scene.build_mesh_payload(s.model)
            message = scene.describe(s.model, version, payload.entries, payload.hfields)
        self._scene_msg, self._mesh_payload, self._payload_version = message, payload, version

    def scene_message(self) -> dict:
        """The model's description (cached per version) plus the extras, asked for afresh every time: the
        robots, the running lesson and its experiment record change without a new model version, and a viewer
        that reconnects mid-lesson must get them as they are now."""
        self._ensure_scene()
        assert self._scene_msg is not None
        if self.scene_extras is None:
            return self._scene_msg
        # The extras (world, lesson) take their own locks, so they are asked for with the session lock released.
        # Asking while holding it inverted the order of a lesson starting on another thread (lesson lock, then
        # session lock): a big model's long mesh build left both waiting forever.
        return {**self._scene_msg, **self.scene_extras()}

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
            # Not the hidden ones (a world's mechanism watcher): they are no robot's behavior.
            "behaviors": [
                {"robot": name, **c.status()}
                for name, c in list(s.controllers.items())
                if not getattr(c, "hidden", False)
            ],
            **(self.state_extras() if self.state_extras is not None else {}),
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

    def start_slow(self, ws: WebSocket, msg: dict[str, Any]) -> bool:
        """Run a slow message's handler off the event loop; False if `msg` is not a slow one."""
        if not isinstance(msg, dict):
            raise ValueError("viewer messages are JSON objects")
        handler = self.slow_handlers.get(msg.get("type"))
        if handler is None:
            return False
        task = asyncio.create_task(self._run_slow(ws, msg, handler))
        self._slow_tasks.add(task)
        task.add_done_callback(self._slow_tasks.discard)
        return True

    async def _run_slow(self, ws: WebSocket, msg: dict[str, Any], handler: Callable[[dict[str, Any]], Any]) -> None:
        reply: dict[str, Any] = {"type": "reply", "id": msg.get("id")}
        try:
            reply.update(ok=True, data=await asyncio.to_thread(handler, msg))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            reply.update(ok=False, message=str(exc))
        except Exception as exc:  # a bug: log it in full, and still tell the page so it does not wait forever
            log.exception("viewer request %r failed", msg.get("type"))
            reply.update(ok=False, message=f"internal error: {exc}")
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await ws.send_text(json.dumps(reply))

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

    def seen(self, ws: WebSocket) -> None:
        """The viewer took in one more pose frame."""
        self._seen[ws] = self._seen.get(ws, 0) + 1

    def forget(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        self._sent.pop(ws, None)
        self._seen.pop(ws, None)

    async def _send_poses(self, data: bytes) -> None:
        dead = []
        for ws in list(self.clients):
            sent, seen = self._sent.get(ws, 0), self._seen.get(ws)
            if seen is not None and sent - seen >= MAX_POSES_AHEAD:
                continue  # it has not taken in what it has: a newer frame goes once it has
            try:
                await ws.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
                continue
            self._sent[ws] = sent + 1
        for ws in dead:
            self.forget(ws)

    async def run(self) -> None:
        """Step physics in real time and stream poses until cancelled.

        One failing tick must not end the stream: it used to, silently, and the page then showed the old
        world for good while the server had moved on to a lesson. It is logged (with its traceback, at most
        every few seconds) and the next tick tries again."""
        last = time.perf_counter()
        last_state = 0.0
        last_lesson = 0.0
        last_sensors = 0.0
        last_map = 0.0
        last_report = -math.inf
        while True:
            tick_start = time.perf_counter()
            try:
                # The real-time clock only runs while someone is watching, so an agent's
                # world does not drift between tool calls.
                if self.clients and self.session.clock_owner is None:
                    self.session.advance(tick_start - last)
                last = tick_start
                if self.clients:
                    version = self.session.version
                    if version != self._announced_version:
                        message = self.scene_message()  # built first: announced only once it was
                        self._announced_version = version
                        await self._send_all(text=json.dumps(message))
                    await self._send_poses(self.session.body_poses().tobytes())
                    if tick_start - last_state > 1.0 / STATE_HZ:
                        last_state = tick_start
                        await self._send_all(text=json.dumps(self.state_message()))
                    if self.lesson_stream is not None and tick_start - last_lesson > 1.0 / LESSON_HZ:
                        last_lesson = tick_start
                        update = self.lesson_stream()
                        if update is not None:
                            await self._send_all(text=json.dumps(update))
                    if self.sensor_stream is not None and tick_start - last_sensors > 1.0 / SENSOR_HZ:
                        last_sensors = tick_start
                        update = self.sensor_stream()
                        if update is not None:
                            await self._send_all(text=json.dumps(update))
                    if self.map_stream is not None and tick_start - last_map > 1.0 / MAP_HZ:
                        last_map = tick_start
                        update = self.map_stream()
                        if update is not None:
                            await self._send_all(text=json.dumps(update))
            except Exception:
                last = tick_start
                if tick_start - last_report > REPORT_EVERY:
                    last_report = tick_start
                    log.exception("a viewer tick failed; the stream carries on")
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
                    msg = json.loads(raw)
                    if isinstance(msg, dict) and msg.get("type") == "seen":
                        hub.seen(ws)
                    elif not hub.start_slow(ws, msg):
                        hub.handle(msg)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except WebSocketDisconnect:
            pass
        finally:
            hub.forget(ws)

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
        middleware=[Middleware(LocalOnly)],
        lifespan=lifespan,
    )
