"""Local route pieces protect saved/live occupants and publish with reversible state."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401

from robotforge.design import layouts
from robotforge.playground import PlaygroundError, viewer_catalog
from robotforge.sim.session import fresh_data
from robotforge.viewer import terrain_create


def place(pg, preset="ramp", pos=None, **patch):  # noqa: F811
    return pg.add_terrain_for_play(
        {
            "preset": preset,
            "pos": [3.0, 0.0, 0.0] if pos is None else pos,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **patch,
        }
    )


def snapshot(pg):  # noqa: F811
    return (
        pg.store.file.read_bytes(),
        pg.session.model,
        pg.session.data,
        pg.session.version,
        pg.session.generation,
        pg.session.paused,
        [a.copy() for a in (pg.session.data.qpos, pg.session.data.qvel, pg.session.data.ctrl)],
        float(pg.session.data.time),
    )


def unchanged(pg, old):  # noqa: F811
    assert pg.store.file.read_bytes() == old[0]
    assert pg.session.model is old[1] and pg.session.data is old[2]
    assert (pg.session.version, pg.session.generation, pg.session.paused) == old[3:6]
    for actual, expected in zip(
        (pg.session.data.qpos, pg.session.data.qvel, pg.session.data.ctrl), old[6], strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert pg.session.data.time == old[7]


@pytest.mark.parametrize(
    "patch",
    [
        {"preset": "hills"},
        {"params": {}},
        {"size": "large"},
        {"pos": [0]},
        {"pos": [0, 0, 0, 0]},
        {"pos": [float("nan"), 0]},
        {"pos": [101, 0]},
        {"pos": [True, 0]},
        {"pos": ["0", 0]},
        {"yaw": True},
        {"yaw": 90.0},
        {"yaw": 45},
        {"generation": True},
        {"generation": -1},
        {"world_sha256": "bad"},
        {"version": "2"},
    ],
)
def test_invalid_request_cannot_change_world_or_runtime(pg, patch):  # noqa: F811
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="invalid terrain placement"):
        place(pg, **patch)
    unchanged(pg, before)


def test_above_ground_click_is_not_silently_flattened(pg):  # noqa: F811
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="on the ground"):
        place(pg, pos=[3, 0, 0.75])
    unchanged(pg, before)


@pytest.mark.parametrize("preset,count", [("ramp", 3), ("stairs", 7)])
@pytest.mark.parametrize("yaw", [0, 90, 180, 270])
def test_catalog_preview_boxes_match_actual_compiled_terrain(pg, preset, count, yaw):  # noqa: F811
    pg.set_floor("wood")
    boxes = next(e["boxes"] for e in viewer_catalog()["terrain_presets"] if e["name"] == preset)
    result = place(pg, preset, yaw=yaw)
    patch = pg.current_world().terrain[0]
    assert patch.style == "wood" and patch.yaw == yaw and len(boxes) == count
    angle = math.radians(yaw)
    rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    for i, box in enumerate(boxes):
        g = pg.session.model.geom(result["terrain"] if i == 0 else f"{result['terrain']}/{i}").id
        np.testing.assert_allclose(pg.session.data.geom_xpos[g], rot @ box["center"] + [3, 0, 0], atol=1e-12)
        np.testing.assert_array_equal(pg.session.model.geom_size[g], box["half"])
        pitch = box["pitch"]
        local_z = np.array([math.sin(pitch), 0, math.cos(pitch)])
        np.testing.assert_allclose(pg.session.data.geom_xmat[g].reshape(3, 3)[:, 2], rot @ local_z, atol=1e-12)
    assert pg.session.paused and pg.session.data.time == 0


def test_appending_and_exact_latest_undo_preserve_prior_route_and_objects(pg):  # noqa: F811
    pg.add_object("box", shape="box", pos=[-5, 0], size=[0.2, 0.2, 0.2])
    first = place(pg)
    before = pg.current_world()
    second = place(pg, "stairs", [-3, 3], yaw=90)
    assert len(pg.current_world().terrain) == 2
    pg.object_for_play({"action": "undo_add", **{k: second["undo"][k] for k in ("token", "version")}})
    assert pg.current_world() == before and pg.current_world().terrain[0].name == first["terrain"]
    assert pg.session.paused
    with pytest.raises(PlaygroundError):
        pg.object_for_play({"action": "undo_add", **{k: second["undo"][k] for k in ("token", "version")}})


def test_terrain_edit_returns_to_saved_starts_not_a_carried_pose(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-5, 0])
    home = fresh_data(pg.session.model)
    pg.session.data.qpos[0] = -4.0
    pg.session.data.qvel[:] = 0.2
    pg.session.data.ctrl[:] = 0.5
    pg.session.data.time = 0.4
    pg.session.paused = False
    place(pg)
    np.testing.assert_array_equal(pg.session.data.qpos, home.qpos)
    np.testing.assert_array_equal(pg.session.data.qvel, home.qvel)
    np.testing.assert_array_equal(pg.session.data.ctrl, home.ctrl)
    assert pg.session.paused and pg.session.data.time == 0


@pytest.mark.parametrize("x", [0, 6])
def test_saved_and_live_robot_footprints_are_both_protected(pg, x):  # noqa: F811
    pg.add_robot("rover", template="rover")
    pg.session.data.qpos[0] = 6
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        place(pg, pos=[x, 0])
    unchanged(pg, before)


def test_fixed_overhead_table_is_not_mistaken_for_clear_ground(pg):  # noqa: F811
    pg.add_object("table", shape="table", size=[1, 1, 0.8], pos=[3, 0])
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by table"):
        place(pg)
    unchanged(pg, before)


def test_whole_existing_heightfield_domain_blocks_a_crossing_edge(pg):  # noqa: F811
    pg.add_terrain("hills", name="lawn", params={"size": [6, 6]})
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by lawn"):
        place(pg, pos=[4.8, 0])
    unchanged(pg, before)


def test_rotated_extent_not_just_center_is_checked(pg):  # noqa: F811
    pg.add_terrain("stairs", name="old")
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by old"):
        place(pg, pos=[0, 2.3], yaw=90)
    unchanged(pg, before)
    place(pg, pos=[0, 2.3], yaw=0)
    assert len(pg.current_world().terrain) == 2


def test_water_area_checks_full_feature_extent_but_allows_dry_ground(pg):  # noqa: F811
    pg.set_floor(water={"level": 0.1, "area": [0, 0, 2, 2]})
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="dry ground"):
        place(pg, pos=[3.8, 1])
    unchanged(pg, before)
    place(pg, pos=[6, 1])


def test_live_movement_during_compilation_is_rechecked(pg, monkeypatch):  # noqa: F811
    pg.add_robot("rover", template="rover")
    original = terrain_create.compile_world
    saved = pg.store.file.read_bytes()

    def compile_and_move(*args):
        result = original(*args)
        pg.session.data.qpos[0] = 3
        mujoco.mj_forward(pg.session.model, pg.session.data)
        return result

    monkeypatch.setattr(terrain_create, "compile_world", compile_and_move)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        place(pg)
    assert pg.store.file.read_bytes() == saved and pg.session.data.qpos[0] == 3


@pytest.mark.parametrize("change", ["reset", "disk"])
def test_state_or_disk_change_during_compile_prevents_publication(pg, monkeypatch, change):  # noqa: F811
    original = terrain_create.compile_world
    saved = pg.store.file.read_bytes()

    def compile_and_change(*args):
        result = original(*args)
        if change == "reset":
            pg.session.reset()
        else:
            pg.store.file.write_bytes(saved + b"\n# external editor\n")
        return result

    monkeypatch.setattr(terrain_create, "compile_world", compile_and_change)
    with pytest.raises(PlaygroundError, match="changed"):
        place(pg)
    assert not pg.current_world().terrain
    assert pg.store.file.read_bytes() == (saved if change == "reset" else saved + b"\n# external editor\n")


def test_partial_runtime_install_failure_restores_disk_and_exact_runtime(pg, monkeypatch):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-4, 0])
    pg.session.step(30)
    before = snapshot(pg)
    files = {p: p.read_bytes() for p in [*pg.store.history.glob("*.yaml"), pg.store.build_dir / "world.xml"]}
    original = pg._load_world

    def broken(*args):
        original(*args)
        raise OSError("injected install failure")

    monkeypatch.setattr(pg, "_load_world", broken)
    with pytest.raises(OSError, match="injected install failure"):
        place(pg)
    unchanged(pg, before)
    assert all(p.read_bytes() == b for p, b in files.items())
    assert set(pg.store.history.glob("*.yaml")) == {p for p in files if p.parent == pg.store.history}


def test_undo_refuses_a_live_occupant_without_changing_it(pg):  # noqa: F811
    pg.add_robot("rover", template="rover")
    result = place(pg)
    pg.session.data.qpos[:3] = [3, 0, 0.5]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        pg.object_for_play({"action": "undo_add", **{k: result["undo"][k] for k in ("token", "version")}})
    unchanged(pg, before)


def test_reset_invalidates_terrain_undo(pg):  # noqa: F811
    result = place(pg)
    pg.session.reset()
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="changed"):
        pg.object_for_play({"action": "undo_add", **{k: result["undo"][k] for k in ("token", "version")}})
    unchanged(pg, before)


def test_custom_hooks_refuse_before_model_install(pg):  # noqa: F811
    pg.session.reset_hooks.append(lambda: None)
    before = snapshot(pg)
    with pytest.raises(layouts.LayoutError, match="custom reset hooks"):
        place(pg)
    unchanged(pg, before)


def test_compile_cannot_start_an_implicit_model_download(pg, monkeypatch):  # noqa: F811
    from robotforge.design import menagerie

    original = terrain_create.compile_world
    before = snapshot(pg)

    def needing_download(*args):
        menagerie.require_downloads_allowed("uncached robot")
        return original(*args)

    monkeypatch.setattr(terrain_create, "compile_world", needing_download)
    with pytest.raises(menagerie.MenagerieError, match="not cached"):
        place(pg)
    unchanged(pg, before)
    menagerie.require_downloads_allowed("an explicit later download")  # Scope must restore the previous policy.


def test_reset_after_commit_cannot_rebind_the_undo_receipt_to_a_new_generation(pg, monkeypatch):  # noqa: F811
    original = terrain_create._commit

    def committed_then_reset(*args, **kwargs):
        result = original(*args, **kwargs)
        pg.session.reset()
        return result

    monkeypatch.setattr(terrain_create, "_commit", committed_then_reset)
    result = place(pg)
    assert pg._play_addition.generation < pg.session.generation
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="changed"):
        pg.object_for_play({"action": "undo_add", **{k: result["undo"][k] for k in ("token", "version")}})
    unchanged(pg, before)


def test_failed_disk_rollback_restores_old_runtime_and_refuses_more_edits(pg, monkeypatch):  # noqa: F811
    before = snapshot(pg)
    original_load, original_replace = pg._load_world, layouts.os.replace

    def failing_load(*args):
        original_load(*args)
        raise OSError("install failure")

    def failed_rollback(source, target):
        if target == pg.store.file and source.read_bytes() == before[0]:
            raise OSError("rollback write failure")
        return original_replace(source, target)

    monkeypatch.setattr(pg, "_load_world", failing_load)
    monkeypatch.setattr(layouts.os, "replace", failed_rollback)
    with pytest.raises(PlaygroundError, match="rollback failed; reopen"):
        place(pg)
    assert pg.session.model is before[1] and pg.session.data is before[2]
    assert pg.session.paused and pg._play_world_sha256 is None
    assert pg.store.file.read_bytes() != before[0]
    with pytest.raises(PlaygroundError, match="changed"):
        place(pg, world_sha256=layouts.sha(before[0]))


def test_legacy_python_terrain_replacement_still_replaces_all_patches(pg):  # noqa: F811
    place(pg)
    pg.add_terrain("stairs", name="old_script", pos=[0, 4])
    assert len(pg.current_world().terrain) == 2
    pg.set_terrain("ramp")
    assert len(pg.current_world().terrain) == 1
    pg.terrain_for_play("flat")
    assert not pg.current_world().terrain
