"""observe(find=...): an agent asks a camera for the red one and gets where it is and what to call it."""

from __future__ import annotations

import json

import numpy as np
import pytest
from mcp import Client

from robotforge.cli import find_free_port
from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import render

pytestmark = pytest.mark.anyio
CUBES = {"cube_a": ("red", [0.22, -0.08]), "cube_b": ("blue", [0.2, 0.1]), "cube_c": ("green", [0.3, 0.02])}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def pg(tmp_path, monkeypatch):
    """An arm (blue, the template's colour) with three cubes in front of it and a bin, settled."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    playground = Playground.with_demo()
    playground.new_world("desk")
    playground.add_robot("arm", template="arm")
    for name, (color, pos) in CUBES.items():
        playground.add_object(name, shape="box", size=[0.04, 0.04, 0.04], pos=pos, color=color)
    playground.add_object("bin", shape="bin", pos=[0.0, 0.28], color="gray")
    playground.simulate(0.5)
    return playground


def _state(result) -> dict:
    return json.loads(result.content[0].text)


async def test_the_red_one_is_found_located_and_named(needs_gl, pg):
    truth = {name: pg.session.data.body(name).xpos.copy() for name in CUBES}
    async with Client(build_server(pg, find_free_port(8790))) as client:
        for view in ("top", "arm/wrist_cam", "default"):
            result = await client.call_tool("observe", {"view": view, "find": "red"})
            found = _state(result)["found"]
            assert found["view"] == view and found["color"] == "red"
            first = found["objects"][0]
            assert first["object"] == "cube_a", (view, found["objects"])
            assert np.linalg.norm(np.array(first["position"]) - truth["cube_a"]) < 0.01, view
            assert result.content[1].type == "image"  # the picture, the find boxed and numbered

        every = _state(await client.call_tool("observe", {"view": "top", "find": "all"}))["found"]["objects"]
        named = {o["object"]: o for o in every if "object" in o}
        assert {name: named[name]["color"] for name in named} == {n: c for n, (c, _) in CUBES.items()}
        for name, o in named.items():
            assert np.linalg.norm(np.array(o["position"]) - truth[name]) < 0.01, name
        assert {o.get("robot") for o in every if "object" not in o} == {"arm"}  # the blue arm is a robot's part

        plain = await client.call_tool("observe", {"view": "top", "find": "red", "include_image": False})
        assert len(plain.content) == 1 and _state(plain)["found"]["objects"][0]["object"] == "cube_a"


async def test_an_agent_picks_the_red_one_without_knowing_its_name(needs_gl, pg):
    async with Client(build_server(pg, find_free_port(8790))) as client:
        seen = _state(await client.call_tool("observe", {"view": "top", "find": "red", "include_image": False}))
        name = seen["found"]["objects"][0]["object"]
        picked = await client.call_tool("pick_and_place", {"robot": "arm", "object": name, "target": "bin"})
        assert not picked.is_error, picked.content[0].text
        await client.call_tool("simulate", {"seconds": 12})
    at = pg.session.data.body("cube_a").xpos
    assert abs(at[0] - 0.0) < 0.07 and abs(at[1] - 0.28) < 0.07  # in the bin


async def test_nothing_of_that_color_comes_with_a_hint(needs_gl, pg):
    async with Client(build_server(pg, find_free_port(8790))) as client:
        found = _state(await client.call_tool("observe", {"view": "top", "find": "purple"}))["found"]
        assert found["objects"] == [] and "another view" in found["hint"]


async def test_colors_it_cannot_find_are_refused_before_rendering(pg, monkeypatch):
    def no_rendering(*_args, **_kwargs):
        raise AssertionError("rendered for a color it cannot look for")

    monkeypatch.setattr(pg.renderer, "layers", no_rendering)
    async with Client(build_server(pg, find_free_port(8790))) as client:
        bad = await client.call_tool("observe", {"find": "chartreuse"})
        assert bad.is_error and "unknown color 'chartreuse'" in bad.content[0].text
        grey = await client.call_tool("observe", {"find": "grey"})
        assert grey.is_error and "too grey" in grey.content[0].text


async def test_without_opengl_find_returns_the_state_and_says_why(pg, monkeypatch):
    monkeypatch.setattr(render, "_probe_gl", lambda: "RuntimeError: no display")
    async with Client(build_server(pg, find_free_port(8790))) as client:
        result = await client.call_tool("observe", {"view": "top", "find": "red"})
        assert not result.is_error and len(result.content) == 1
        state = _state(result)
        assert "Cannot look for red" in state["note"] and "no display" in state["note"]
        assert "cube_a" in state["free_bodies"]


async def test_finds_are_named_from_the_world_that_was_seen(needs_gl, pg):
    """The picture and the names must come from one model: a world swapped in between has other geoms."""
    from robotforge.sim import finder, vision

    layers = pg.renderer.layers(pg.session, "top", 320, 240, geoms=True, exposure=vision.EXPOSURE)
    pg.new_world("empty")  # another agent call, between the picture and the naming
    found = finder.find(layers, "red")
    assert found[0]["object"] == "cube_a"
