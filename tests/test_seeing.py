"""The seeing lesson (Sensing III): its camera, its estimates, its pick, and what it says without OpenGL.

The whole lesson is played step by step in tests/test_playthrough.py.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from robotforge.learn import seeing
from robotforge.learn.engine import LessonContext
from robotforge.learn.playthrough import Drag, Player, Set
from robotforge.playground import Playground
from robotforge.sim import render
from robotforge.sim.render import SnapshotRenderer

TUNED = {"hue": 0, "width": 15, "sat": 0.3, "val": 0.2, "balance": 0, "clean": 0}


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return Playground.with_demo()


def test_back_projection_and_depth_locate_the_cube_within_millimetres(needs_gl):
    """The red cube put at known spots across the mat, seen by the lesson's overhead camera."""
    lesson = seeing.SeeingLesson()
    model = lesson.build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    adr = model.jnt_qposadr[model.body(seeing.TARGET).jntadr[0]]
    renderer = SnapshotRenderer()
    plane, table, depth = [], [], []
    for x, y in ((0.2, -0.1), (0.3, -0.05), (0.15, 0.03), (0.32, 0.1), (0.25, 0.0)):
        data.qpos[adr : adr + 7] = [x, y, seeing.CUBE / 2, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        layers = renderer.layers_of(model, data, seeing.CAMERA, seeing.WIDTH, seeing.HEIGHT, 0, geoms=True)
        truth = np.array([x, y, seeing.CUBE / 2])
        on_plane = seeing.look(layers, layers.rgb, TUNED | {"height": 2.0, "depth": 0}, lesson.owner)
        on_table = seeing.look(layers, layers.rgb, TUNED | {"height": 0.0, "depth": 0}, lesson.owner)
        by_depth = seeing.look(layers, layers.rgb, TUNED | {"height": 0.0, "depth": 1}, lesson.owner)
        assert on_plane.on == [seeing.TARGET]
        plane.append(np.linalg.norm(on_plane.estimate - truth))
        table.append(on_table.estimate[1] - truth[1])
        depth.append(np.linalg.norm(by_depth.estimate - truth))
        assert by_depth.size == pytest.approx([0.04, 0.04, 0.04], abs=0.01)  # it measures the cube too
    renderer.close()
    assert max(plane) < 0.004 and max(depth) < 0.005  # scripts/vision_bench.py: 0.11 and 0.26 cm on average
    assert min(table) > 0.01  # the table's plane puts every estimate behind the cube, as the lesson says


def test_the_default_range_takes_in_orange_and_pink_and_a_narrow_one_only_red(needs_gl, pg):
    pg.lessons.start("seeing")
    player = Player(pg)
    player.play(0.5)
    assert sorted(pg.lessons.ctx.results["seen"]["on"]) == ["orange_cube", "pink_cube", "red_cube"]
    Drag("red_cube", (0.21, -0.11, 0.03)).apply(player)
    pg.lessons.handle({"action": "next"})
    Set(width=15).apply(player)
    player.play(0.3)
    assert pg.lessons.ctx.results["seen"]["on"] == ["red_cube"]
    assert "mask" in pg.lessons.passed
    pg.lessons.exit()


def test_the_arm_reaches_for_where_vision_says(needs_gl, pg):
    """With the plane on the table the estimate lies 1.5 to 2 cm behind the cube, and the arm aims there."""
    pg.lessons.start("seeing")
    player = Player(pg)
    lesson = pg.lessons.lesson
    pg.lessons.passed.update({"look", "mask", "where"})  # straight to the pick; its buttons unlock with the step
    pg.lessons.step = 3
    Set(width=15, height=0).apply(player)
    player.play(0.5)
    pg.lessons.handle({"action": "press", "control": "pick"})
    assert lesson.picking is not None
    off = lesson.picking["off_cm"]
    assert 1.2 < off < 2.5
    estimate = pg.lessons.ctx.results["seen"]
    assert estimate["err_cm"] > 2.0  # 3D: the height is wrong too
    pg.lessons.exit()


def test_a_light_switched_on_is_not_judged_on_a_frame_taken_before(pg):
    """The evening step passed the moment the lamp was chosen: its check read the last daylight frame."""
    ctx = LessonContext(values={"light": 1.0, "depth": 0.0})
    ctx.results["seen"] = {"on": [seeing.TARGET], "light": 0, "depth": 0, "err_cm": 0.1}
    assert not seeing._fixed_for_the_evening(ctx)
    ctx.results["seen"]["light"] = 1
    assert seeing._fixed_for_the_evening(ctx)
    ctx.results["seen"]["depth"] = 1  # the plane step likewise waits for a frame seen on the plane
    assert not seeing._close_on_the_plane(ctx)


def test_test_spots_and_shuffles_keep_cubes_apart_on_the_mat(pg):
    pg.lessons.start("seeing")
    lesson = pg.lessons.lesson
    model, data = pg.session.model, pg.session.data
    (cx, cy), (sx, sy) = seeing.MAT
    spots = lesson.test_spots(data)
    assert len(spots) == seeing.SPOTS
    others = [data.body(f"{c}_cube").xpos[:2] for c, _ in seeing.CUBES if f"{c}_cube" != seeing.TARGET]
    for x, y in spots:
        assert abs(x - cx) <= sx / 2 - seeing.INSET + 1e-9 and abs(y - cy) <= sy / 2 - seeing.INSET + 1e-9
        assert all(math.dist((x, y), o) >= seeing.SPACING for o in others)
    for _ in range(5):
        lesson._shuffle(model, data, pg.lessons.ctx)
        at = {c: data.body(f"{c}_cube").xpos[:2].copy() for c, _ in seeing.CUBES}
        x, y = at["red"]
        assert x >= 0.12 and math.hypot(x, y) <= seeing.REACH  # the arm can always pick the red one
        names = list(at)
        assert all(math.dist(at[a], at[b]) >= seeing.SPACING for i, a in enumerate(names) for b in names[i + 1 :])
    pg.lessons.exit()


def test_without_opengl_the_lesson_says_so(pg, monkeypatch):
    monkeypatch.setattr(render, "_probe_gl", lambda: "RuntimeError: no display")
    monkeypatch.setattr(seeing, "render_unavailable_reason", lambda: "RuntimeError: no display")
    pg.lessons.start("seeing")
    update = Player(pg).play(0.5)
    note = update["live"]["vision"]["note"]
    assert "OpenGL" in note["en"] and "no display" in note["en"] and "OpenGL" in note["zh"]
    assert update["live"]["readout"] == note
    assert "frames" not in pg.lessons.ctx.results  # nothing pretends to have seen
    pg.lessons.exit()
