"""observe(find=...): an agent asks a camera for the red one and gets where it is and what to call it."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from mcp import Client
from PIL import Image

from robotforge.mcp_server import build_server
from robotforge.playground import Playground
from robotforge.sim import finder, render, vision

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
    async with Client(build_server(pg, 0)) as client:
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
    async with Client(build_server(pg, 0)) as client:
        seen = _state(await client.call_tool("observe", {"view": "top", "find": "red", "include_image": False}))
        name = seen["found"]["objects"][0]["object"]
        picked = await client.call_tool("pick_and_place", {"robot": "arm", "object": name, "target": "bin"})
        assert not picked.is_error, picked.content[0].text
        await client.call_tool("simulate", {"seconds": 12})
    at = pg.session.data.body("cube_a").xpos
    assert abs(at[0] - 0.0) < 0.07 and abs(at[1] - 0.28) < 0.07  # in the bin


async def test_nothing_of_that_color_comes_with_a_hint(needs_gl, pg):
    async with Client(build_server(pg, 0)) as client:
        found = _state(await client.call_tool("observe", {"view": "top", "find": "purple"}))["found"]
        assert found["objects"] == [] and "another view" in found["hint"]


async def test_colors_it_cannot_find_are_refused_before_rendering(pg, monkeypatch):
    def no_rendering(*_args, **_kwargs):
        raise AssertionError("rendered for a color it cannot look for")

    monkeypatch.setattr(pg.renderer, "layers", no_rendering)
    async with Client(build_server(pg, 0)) as client:
        bad = await client.call_tool("observe", {"find": "chartreuse"})
        assert bad.is_error and "unknown color 'chartreuse'" in bad.content[0].text
        grey = await client.call_tool("observe", {"find": "grey"})
        assert grey.is_error and "too grey" in grey.content[0].text


async def test_without_opengl_find_returns_the_state_and_says_why(pg, monkeypatch):
    monkeypatch.setattr(render, "_probe_gl", lambda: "RuntimeError: no display")
    async with Client(build_server(pg, 0)) as client:
        result = await client.call_tool("observe", {"view": "top", "find": "red"})
        assert not result.is_error and len(result.content) == 1
        state = _state(result)
        assert "Cannot look for red" in state["note"] and "no display" in state["note"]
        assert "cube_a" in state["free_bodies"]


async def test_a_failure_while_looking_still_returns_the_state(pg, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("depth buffer lost")

    monkeypatch.setattr(pg, "find", broken)
    async with Client(build_server(pg, 0)) as client:
        result = await client.call_tool("observe", {"view": "top", "find": "red"})
        assert not result.is_error and len(result.content) == 1
        state = _state(result)
        assert "finding failed (RuntimeError: depth buffer lost)" in state["note"] and "cube_a" in state["free_bodies"]


def test_a_mechanism_with_motors_is_a_robot_even_without_a_robot_prefix(needs_gl, tmp_path, monkeypatch):
    """The demo model is a plain MJCF: its blue arm's bodies have no "arm/" prefix, and it came back as
    {"fixed": "base"}, as if it were built into the world."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    found, _ = Playground.with_demo().find("default", "all")
    kinds = {o[kind]: kind for o in found["objects"] for kind in finder.KINDS if kind in o}
    assert kinds == {"green_cube": "object", "red_cube": "object", "ball": "object", "base": "robot"}


async def test_finds_are_named_from_the_world_that_was_seen(needs_gl, pg):
    """The picture and the names must come from one model: a world swapped in between has other geoms."""
    layers = pg.renderer.layers(pg.session, "top", 320, 240, geoms=True, exposure=vision.EXPOSURE)
    pg.new_world("empty")  # another agent call, between the picture and the naming
    found = finder.find(layers, "red")
    assert found[0].entry["object"] == "cube_a"


def test_a_box_partly_behind_the_arm_is_located_and_said_to_be_hidden(needs_gl, tmp_path, monkeypatch):
    """The arm's hand lent the box's blob a few edge pixels 10 cm above the box: it came out 7.7 cm off."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("desk")
    pg.add_robot("arm", template="arm")
    pg.add_object("box", shape="box", size=[0.05, 0.05, 0.05], pos=[0.15, 0.2], color="yellow")
    pg.simulate(0.5)
    found, _ = pg.find("default", "yellow")
    box = found["objects"][0]
    assert box["object"] == "box" and box.get("hidden")
    assert np.linalg.norm(np.array(box["position"]) - pg.session.data.body("box").xpos) < 0.01


def test_the_wrist_camera_says_when_the_cube_is_cut_off(needs_gl, pg):
    """Close over the cube, the wrist camera's picture cuts it, and the estimate is off by a cm or two."""
    truth = pg.session.data.body("cube_a").xpos.copy()
    for aim, cut in (([0.22, -0.02, 0.09], True), ([0.22, -0.08, 0.13], False)):
        pg.behaviors.move_arm("arm", aim)
        pg.simulate(2.0)
        cube = pg.find("arm/wrist_cam", "red")[0]["objects"][0]
        assert cube["object"] == "cube_a" and bool(cube.get("cut_off")) == cut, (aim, cube)
        if not cut:
            assert np.linalg.norm(np.array(cube["position"]) - truth) < cube["about_cm"] / 100 + 0.002


def test_in_a_kitchen_every_loose_object_seen_is_listed_before_the_scenery(needs_gl, tmp_path, monkeypatch):
    """The wooden floor, cabinets and shelf are found as orange; they filled the list and hid the cups and the
    apple. And the cereal box on the island's edge sat 15 cm low: its support was taken from the island's side."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.load_scene("kitchen")
    pg.simulate(0.5)
    layers = pg.renderer.layers(pg.session, "default", 640, 480, geoms=True, exposure=vision.EXPOSURE)
    every = finder.find(layers, "all")
    listed = finder.summary("default", "all", every)["objects"]
    loose = [e.entry["object"] for e in every if "object" in e.entry]
    assert {"cereal", "red_cup", "apple"} <= set(loose)
    assert [o["object"] for o in listed if "object" in o] == loose  # all of them, first
    ranks = [next((i for i, kind in enumerate(finder.KINDS) if kind in o), len(finder.KINDS)) for o in listed]
    assert ranks == sorted(ranks)  # objects, then robots, then fixed things, then blobs of no one thing
    names = [(k, o[k]) for o in listed for k in finder.KINDS if k in o]
    assert len(names) == len(set(names))  # each thing once
    cereal = next(o for o in listed if o.get("object") == "cereal")
    assert np.linalg.norm(np.array(cereal["position"]) - pg.session.data.body("cereal").xpos) < 0.01

    marked = np.array(Image.open(io.BytesIO(finder.marked_png(layers.rgb, every))))
    for seen in every[: finder.MAX_FOUND]:  # each box hugs its blob, a stroke's width out
        x0, y0, x1, y1 = seen.box
        x, y = max(x0 - 2, 0), (y0 + y1) // 2
        assert tuple(marked[y, x]) == finder.MARK, seen.entry


def test_a_ball_in_front_of_a_bench_is_not_pulled_back_to_it(needs_gl, tmp_path, monkeypatch):
    """From the park dog's camera the ball's edge blends in the bench 1.6 m behind it: it came out 77 cm off."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.load_scene("park")
    pg.simulate(0.5)
    ball = pg.find("dog/front_cam", "red")[0]["objects"][0]
    assert ball["object"] == "ball"
    # The camera sees the ball's near half from low down: its middle comes out a few cm short.
    assert np.linalg.norm(np.array(ball["position"]) - pg.session.data.body("ball").xpos) < 0.05
