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
        # Sent with every find, so it explains only the flags the finds carry (it was 697 characters every time).
        found = _state(plain)["found"]
        carried = {mark for o in found["objects"] for mark in finder.DOUBTS if mark in o}
        assert len(finder.HOW) < 300 and found["how"].startswith(finder.HOW)
        assert all((text in found["how"]) == (mark in carried) for mark, text in finder.DOUBTS.items())


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
        # It offered white, black and gray (which it then refused as too grey) and never "all".
        for refused in (bad, grey):
            text = refused.content[0].text
            assert all(name in text for name in vision.NAMED) and '"all"' in text
            assert not any(f" {name}," in text for name in ("white", "black", "gray"))


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


def test_numbers_on_the_picture_cover_no_other_find():
    """On a cube standing on another, the lower one's number sat on its box's top edge, under the upper box."""
    rgb = np.full((240, 320, 3), 200, np.uint8)
    lower, upper = (120, 120, 160, 160), (120, 84, 160, 119)  # x0, y0, x1, y1: the upper one rests on the lower
    found = [finder.Seen({"n": n}, box) for n, box in ((1, lower), (2, upper))]
    marked = np.array(Image.open(io.BytesIO(finder.marked_png(rgb, found))))
    tags = vision.blobs((marked == finder.TAG).all(axis=-1), min_area=20)
    assert len(tags) == 2  # each number on its own dark tag
    for tag in tags:
        tx0, ty0, tx1, ty1 = tag.box
        covered = [box for box in (lower, upper) if tx0 <= box[2] and box[0] <= tx1 and ty0 <= box[3] and box[1] <= ty1]
        assert not covered, (tag.box, covered)


async def test_the_marked_picture_is_as_bright_as_observe_and_skipped_without_an_image(needs_gl, pg, monkeypatch):
    """The boxes were drawn on the finder's darker picture (its exposure is 0.6), and it was drawn and encoded
    even when the agent asked for no image."""
    plain = np.array(Image.open(io.BytesIO(pg.snapshot("default"))).convert("RGB"), dtype=float)
    _, png = pg.find("default", "red")
    marked = np.array(Image.open(io.BytesIO(png)).convert("RGB"), dtype=float)
    assert abs(marked.mean() - plain.mean()) < 0.02 * plain.mean()

    def no_drawing(*_args, **_kwargs):
        raise AssertionError("drew the picture for a find without an image")

    monkeypatch.setattr(finder, "marked_png", no_drawing)
    async with Client(build_server(pg, 0)) as client:
        result = await client.call_tool("observe", {"view": "top", "find": "red", "include_image": False})
        assert not result.is_error and len(result.content) == 1
        assert _state(result)["found"]["objects"][0]["object"] == "cube_a"


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
    for aim, cut in (([0.22, -0.02, 0.09], True), ([0.22, -0.08, 0.13], False), ([0.22, -0.02, 0.13], False)):
        pg.behaviors.move_arm("arm", aim)
        pg.simulate(2.0)
        cube = pg.find("arm/wrist_cam", "red")[0]["objects"][0]
        assert cube["object"] == "cube_a" and bool(cube.get("cut_off")) == cut, (aim, cube)
        # Close up, the cube's own shaded side fills the ring round it; it is no surface the cube may stand on.
        assert "raised" not in cube, (aim, cube)
        if not cut:
            assert np.linalg.norm(np.array(cube["position"]) - truth) < cube["about_cm"] / 100 + 0.002


def test_in_a_kitchen_every_loose_object_seen_is_listed_before_the_scenery(needs_gl, tmp_path, monkeypatch):
    """The wooden floor, cabinets and shelf were found as orange; they filled the list and hid the cups and the
    apple, and swallowed the orange on the table. And the cereal box on the island's edge sat 15 cm low: its
    support was taken from the island's side."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.load_scene("kitchen")
    pg.simulate(0.5)
    layers = pg.renderer.layers(pg.session, "default", 640, 480, geoms=True, exposure=vision.EXPOSURE)
    every = finder.find(layers, "all")
    listed = finder.summary("default", "all", every)["objects"]
    loose = [e.entry["object"] for e in every if "object" in e.entry]
    assert {"cereal", "red_cup", "apple", "fruit"} <= set(loose)  # the orange on the wooden table too
    assert next(e.entry for e in every if e.entry.get("object") == "fruit")["color"] == "orange"
    assert any("blue_cup" in e.entry.get("merged", []) for e in every)  # touching the blue arm: one blob
    assert [o["object"] for o in listed if "object" in o] == loose  # all of them, first
    ranks = [
        0 if "merged" in o else next((i for i, kind in enumerate(finder.KINDS) if kind in o), len(finder.KINDS))
        for o in listed
    ]
    assert ranks == sorted(ranks)  # with loose objects in them, then robots, fixed things, blobs of no one thing
    names = [(k, o[k]) for o in listed for k in finder.KINDS if k in o]
    assert len(names) == len(set(names))  # each thing once
    cereal = next(o for o in listed if o.get("object") == "cereal")
    assert np.linalg.norm(np.array(cereal["position"]) - pg.session.data.body("cereal").xpos) < 0.01

    marked = np.array(Image.open(io.BytesIO(finder.marked_png(layers.rgb, every))))
    for seen in every[: finder.MAX_FOUND]:  # each box hugs its blob, a stroke's width out
        x0, y0, x1, y1 = seen.box
        x, y = max(x0 - 2, 0), (y0 + y1) // 2
        assert tuple(marked[y, x]) == finder.MARK, seen.entry


def test_touching_things_of_one_color_are_one_find_that_names_them_all(needs_gl, tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("desk")
    for name, y in (("left_cube", -0.02), ("right_cube", 0.02)):  # side by side, faces touching
        pg.add_object(name, shape="box", size=[0.04, 0.04, 0.04], pos=[0.3, y], color="red")
    pg.simulate(0.3)
    found, _ = pg.find("top", "red")
    assert len(found["objects"]) == 1
    both = found["objects"][0]
    assert {both.get("object"), *both["merged"]} >= {"left_cube", "right_cube"}
    assert np.linalg.norm(np.array(both["position"][:2]) - [0.3, 0.0]) < 0.01  # the middle of the two


def _cube_from(pg: Playground, view: str) -> dict:
    found, _ = pg.find(view, "red")
    return next(o for o in found["objects"] if o.get("object") == "cube")


FLAGGED = (*finder.FLAGS, "held_by")


def test_a_cube_on_a_cube_is_located_on_it_or_flagged(needs_gl, tmp_path, monkeypatch):
    """Every find took the desk as the surface under it: a red cube on a blue one came out 2.5 cm low from every
    view, 9 cm tall and with no flag. A ball seen from above ends at its widest and must stay on the desk."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("desk")
    pg.add_robot("arm", template="arm")
    pg.add_object("under", shape="box", size=[0.05, 0.05, 0.05], pos=[0.25, 0.1], color="blue")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, 0.1, 0.075], color="red")
    pg.add_object("ball", shape="sphere", size=[0.06], pos=[0.3, -0.18], color="orange")
    pg.simulate(1.0)
    truth = pg.session.data.body("cube").xpos
    for view in ("default", "front", "right", "top"):
        cube = _cube_from(pg, view)
        error = np.linalg.norm(np.array(cube["position"]) - truth)
        assert error < 0.01 or any(flag in cube for flag in FLAGGED), (view, error, cube)
    default = _cube_from(pg, "default")
    assert not any(flag in default for flag in FLAGGED) and abs(default["size"][2] - 0.04) < 0.005
    assert _cube_from(pg, "top").get("raised")  # from straight above its sides cannot be seen
    ball = pg.find("top", "orange")[0]["objects"][0]
    assert ball["object"] == "ball" and "raised" not in ball
    assert np.linalg.norm(np.array(ball["position"]) - pg.session.data.body("ball").xpos) < 0.01


def test_a_cube_in_the_gripper_is_located_at_its_height_and_said_to_be_held(needs_gl, pg):
    """Mid-carry the cube came out 4 cm low (the desk taken as what it stands on) with no flag, and the gripper
    hides part of it, which moves its middle a cm or two across."""
    pg.behaviors.pick_and_place("arm", "cube_a", "bin")
    for _ in range(200):
        pg.simulate(0.05)
        if pg.session.data.body("cube_a").xpos[2] > 0.08:
            break
    pg.session.paused = True
    truth = pg.session.data.body("cube_a").xpos
    assert truth[2] > 0.08
    for view in ("default", "front", "right"):
        found, _ = pg.find(view, "red")
        cube = next(o for o in found["objects"] if o.get("object") == "cube_a")
        assert cube["held_by"] == "arm", (view, cube)
        assert abs(cube["position"][2] - truth[2]) < 0.01, (view, cube)
        assert "held_by: that robot holds it" in found["how"]


def test_a_cube_on_the_one_the_gripper_closed_on_is_held_too(needs_gl, tmp_path, monkeypatch):
    """An agent closed the gripper at the found middle of a red cube on a blue one: the fingers took the wider blue
    cube and lifted both. The red one touches only the blue one, yet the hand covers part of it (1.5 cm off)."""
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    pg = Playground.with_demo()
    pg.new_world("desk")
    pg.add_robot("arm", template="arm")
    pg.add_object("under", shape="box", size=[0.05, 0.05, 0.05], pos=[0.25, 0.1], color="blue")
    pg.add_object("cube", shape="box", size=[0.04, 0.04, 0.04], pos=[0.25, 0.1, 0.075], color="red")
    pg.simulate(1.0)
    x, y, z = _cube_from(pg, "default")["position"]
    for aim, grip in (([x, y, z + 0.08], "open"), ([x, y, z], None)):
        pg.behaviors.move_arm("arm", aim, grip)
        pg.simulate(1.0)
    pg.behaviors.set_gripper("arm", "close")
    pg.simulate(1.0)
    pg.behaviors.move_arm("arm", [x, y, z + 0.08])
    pg.simulate(1.0)
    assert pg.session.data.body("cube").xpos[2] > 0.12  # lifted with the blue one under it
    cube = _cube_from(pg, "default")
    assert cube["held_by"] == "arm", cube
    assert abs(cube["position"][2] - pg.session.data.body("cube").xpos[2]) < 0.01  # its height is still right


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
