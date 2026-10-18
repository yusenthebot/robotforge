import io
import json
import threading
import warnings
from pathlib import Path

import mujoco
import pytest
from mcp import Client
from PIL import Image

from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import render
from robotforge.sim.render import RenderUnavailable, SnapshotRenderer, render_unavailable_reason
from robotforge.sim.session import SimSession

DEMO = Path(__file__).parents[1] / "src" / "robotforge" / "assets" / "demo.xml"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def session() -> SimSession:
    return SimSession.from_path(DEMO)


@pytest.fixture
def no_gl(monkeypatch):
    """Pretend this machine has no OpenGL, whatever it really has."""
    monkeypatch.setattr(render, "_probe_gl", lambda: "RuntimeError: no display")


def test_snapshot_is_a_real_picture(needs_gl, session):
    png = SnapshotRenderer().png(session, "front", 160, 120)
    image = Image.open(io.BytesIO(png))
    assert image.size == (160, 120)
    colors = image.convert("RGB").getcolors(maxcolors=160 * 120)
    assert len(colors) > 10  # a blank frame would be one flat color


def test_one_renderer_serves_threads_that_are_alive_at_the_same_time(needs_gl, session):
    # MCP runs sync tools on pooled worker threads. A GL context stays current in the thread that last used
    # it, and EGL refuses to hand it to another live thread, so the second thread must not touch it directly.
    renderer, shapes, errors = SnapshotRenderer(), [], []
    first_rendered, second_rendered = threading.Event(), threading.Event()

    def shoot() -> None:
        try:
            shapes.append(renderer.rgb(session, "top", 64, 48).shape)
        except Exception as exc:  # collected so the main thread can fail with the cause
            errors.append(exc)

    def first() -> None:
        shoot()
        first_rendered.set()
        second_rendered.wait(30)  # stay alive, still holding whatever context it made current

    def second() -> None:
        first_rendered.wait(30)
        shoot()
        second_rendered.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    renderer.close()
    assert not errors
    assert shapes == [(48, 64, 3)] * 2


@pytest.fixture
def broken_renderer(monkeypatch):
    """Make mujoco.Renderer fail like a driverless machine does, and forget cached probe results around it."""
    attempts = []

    def install(error: Exception, warning: str | None = None) -> list:
        def broken(*args, **kwargs):
            attempts.append(1)
            if warning:
                warnings.warn(warning, stacklevel=2)  # GLFW reports the real cause this way
            raise error

        render._probe_gl.cache_clear()
        monkeypatch.setattr(mujoco, "Renderer", broken)
        return attempts

    yield install
    monkeypatch.undo()
    render._probe_gl.cache_clear()


def test_failed_probe_is_reported_once_as_one_line(broken_renderer):
    attempts = broken_renderer(RuntimeError("could not create\nwindow"))
    assert render_unavailable_reason() == "RuntimeError: could not create window"
    assert render_unavailable_reason() == "RuntimeError: could not create window"
    assert len(attempts) == 1


def test_probe_reason_includes_what_the_driver_warned_about(broken_renderer):
    broken_renderer(RuntimeError("gladLoadGL error"), warning="WGL: the driver does not support OpenGL")
    assert render_unavailable_reason() == "RuntimeError: gladLoadGL error (WGL: the driver does not support OpenGL)"


def test_snapshot_without_gl_raises_a_clear_error(no_gl, session):
    with pytest.raises(RenderUnavailable, match="no display"):
        SnapshotRenderer().png(session)


def test_bad_arguments_are_reported_the_same_without_gl(no_gl, session):
    renderer = SnapshotRenderer()
    with pytest.raises(ValueError, match="unknown view"):
        renderer.png(session, "sideways")
    with pytest.raises(ValueError, match="out of range"):
        renderer.png(session, "front", 8, 8)


@pytest.mark.anyio
async def test_observe_without_gl_returns_the_state_and_a_note(no_gl):
    server = build_server(Playground.with_demo(), 0)
    async with Client(server) as client:
        obs = await client.call_tool("observe", {"view": "front"})
        assert not obs.is_error
        assert len(obs.content) == 1
        state = json.loads(obs.content[0].text)
        assert "joints" in state and "shoulder" in state["joints"]
        assert "No image" in state["note"] and "no display" in state["note"]

        bad = await client.call_tool("observe", {"view": "sideways"})
        assert bad.is_error and "unknown view" in bad.content[0].text

        plain = await client.call_tool("observe", {"include_image": False})
        assert "note" not in json.loads(plain.content[0].text)
