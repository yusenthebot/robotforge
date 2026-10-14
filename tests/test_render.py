import io
import json
import threading
from pathlib import Path

import mujoco
import pytest
from mcp import Client
from PIL import Image

from robotforge.cli import find_free_port
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


def test_one_renderer_serves_calls_from_different_threads(needs_gl, session):
    renderer, sizes, errors = SnapshotRenderer(), [], []

    def shoot() -> None:
        try:
            sizes.append(renderer.rgb(session, "top", 64, 48).shape)
        except Exception as exc:  # collected so the main thread can fail with the cause
            errors.append(exc)

    for _ in range(3):
        thread = threading.Thread(target=shoot)
        thread.start()
        thread.join()
    renderer.close()
    assert not errors
    assert sizes == [(48, 64, 3)] * 3


def test_failed_probe_is_reported_once_as_one_line(monkeypatch):
    attempts = []

    def broken(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("could not create\nwindow")

    render._probe_gl.cache_clear()
    monkeypatch.setattr(mujoco, "Renderer", broken)
    try:
        assert render_unavailable_reason() == "RuntimeError: could not create window"
        assert render_unavailable_reason() == "RuntimeError: could not create window"
        assert len(attempts) == 1
    finally:
        monkeypatch.undo()
        render._probe_gl.cache_clear()


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
    server = build_server(Playground.with_demo(), find_free_port(8790))
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
