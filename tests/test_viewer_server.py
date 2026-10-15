import json
from pathlib import Path

import numpy as np
from starlette.testclient import TestClient

from robotforge.sim.session import SimSession
from robotforge.viewer.server import ViewerHub, create_app

DEMO = Path(__file__).parents[1] / "src" / "robotforge" / "assets" / "demo.xml"


def _client():
    session = SimSession.from_path(DEMO)
    hub = ViewerHub(session)
    return session, TestClient(create_app(hub), base_url="http://127.0.0.1:8737")


def _next_text(ws, kind):
    for _ in range(200):
        msg = ws.receive()
        if msg.get("text"):
            data = json.loads(msg["text"])
            if data["type"] == kind:
                return data
    raise AssertionError(f"no {kind} message")


def _next_frame(ws):
    for _ in range(200):
        msg = ws.receive()
        if msg.get("bytes"):
            return np.frombuffer(msg["bytes"], dtype=np.float32)
    raise AssertionError("no binary frame")


def test_index_and_static_assets_are_served():
    _, client = _client()
    with client:
        assert "robotforge" in client.get("/").text
        app_js = client.get("/static/app.js")
        assert app_js.status_code == 200
        assert app_js.headers["cache-control"] == "no-cache"
        assert client.get("/").headers["cache-control"] == "no-cache"
        assert client.get("/static/vendor/three/three.module.min.js").status_code == 200


def test_websocket_streams_scene_then_frames():
    session, client = _client()
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        desc = _next_text(ws, "scene")
        assert desc["nbody"] == session.model.nbody
        assert [a["name"] for a in desc["actuators"]] == ["yaw", "shoulder", "elbow"]
        frame = _next_frame(ws)
        assert frame.shape == (1 + 7 * desc["nbody"],)


def test_control_messages_reach_the_simulation():
    session, client = _client()
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        _next_text(ws, "scene")
        ws.send_text(json.dumps({"type": "ctrl", "id": 1, "value": 0.5}))
        ws.send_text(json.dumps({"type": "cmd", "cmd": "pause"}))
        state = _next_text(ws, "state")
        for _ in range(20):
            if state["paused"]:
                break
            state = _next_text(ws, "state")
        assert state["paused"] is True
        assert state["ctrl"][1] == 0.5


def test_bad_messages_return_errors_without_dropping_the_socket():
    _, client = _client()
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        _next_text(ws, "scene")
        ws.send_text(json.dumps({"type": "nope"}))
        assert "unknown message type" in _next_text(ws, "error")["message"]
        ws.send_text(json.dumps({"type": "ctrl", "id": 99, "value": 1}))
        assert "out of range" in _next_text(ws, "error")["message"]


def test_model_reload_pushes_a_new_scene():
    session, client = _client()
    with client, client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws:
        first = _next_text(ws, "scene")
        session.load_path(DEMO)
        second = _next_text(ws, "scene")
        assert second["version"] == first["version"] + 1
        assert client.get(f"/mesh/{first['version']}.bin").status_code == 409


def test_scene_extras_are_asked_for_without_the_session_lock():
    """Regression: extras take the lesson lock; asked for under the session lock, they deadlocked with a
    lesson starting on another thread (lesson lock first, then the session lock)."""
    import threading

    session = SimSession.from_path(DEMO)
    hub = ViewerHub(session)
    seen = {}

    def try_lock():
        seen["free"] = session.lock.acquire(timeout=1.0)
        if seen["free"]:
            session.lock.release()

    def extras():
        other = threading.Thread(target=try_lock)
        other.start()
        other.join()
        return {}

    hub.scene_extras = extras
    hub.scene_message()
    assert seen["free"]  # another thread could take the session lock while the extras were built


def test_a_reconnecting_viewer_gets_the_extras_as_they_are_now():
    """Regression: the extras (robots, the running lesson, its camera and record) were cached with the model
    version, so a page reloaded mid-lesson got them as they were when the lesson started."""
    session = SimSession.from_path(DEMO)
    hub = ViewerHub(session)
    state = {"runs": 0}
    hub.scene_extras = lambda: {"lesson": {"runs": state["runs"]}}
    assert hub.scene_message()["lesson"]["runs"] == 0
    state["runs"] = 2  # a training finished: no new model version
    assert hub.scene_message()["lesson"]["runs"] == 2
    assert hub.scene_message()["version"] == session.version


def test_other_web_pages_cannot_drive_the_viewer():
    """Browsers let any site open a WebSocket to 127.0.0.1: without a check, a page visited while the viewer
    runs could load scenes into the user's world. A name pointed at 127.0.0.1 (DNS rebinding) is refused too."""
    import pytest
    from starlette.websockets import WebSocketDisconnect

    _, client = _client()
    with client:
        host = {"host": "127.0.0.1:8737"}
        with client.websocket_connect("/ws", headers={**host, "origin": "http://127.0.0.1:8737"}) as ws:
            assert _next_text(ws, "scene")  # the viewer's own page
        with client.websocket_connect("/ws", headers=host) as ws:
            assert _next_text(ws, "scene")  # no Origin: a script or agent on this machine
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/ws", headers={**host, "origin": "https://evil.example"}),
        ):
            pass
        rebound = {"host": "evil.example:8737", "origin": "http://evil.example:8737"}
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws", headers=rebound):
            pass
        assert client.get("/scene.json", headers={"host": "evil.example:8737"}).status_code == 403
        assert client.get("/scene.json", headers={"host": "localhost:8737"}).status_code == 200
        assert client.get("/scene.json", headers={"host": "192.168.1.20:8737"}).status_code == 200  # --host on a LAN
