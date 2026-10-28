"""Moving authored route blocks protects both footprints and keeps exact inverse state."""

from __future__ import annotations

from contextlib import contextmanager

import mujoco
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_terrain_create import place, snapshot, unchanged

from robotforge.design import layouts, menagerie
from robotforge.playground import PlaygroundError
from robotforge.viewer import terrain_create, terrain_edit


def move(pg, name="dock", pos=None, **extra):  # noqa: F811
    return pg.edit_terrain_for_play(
        {
            "action": "edit_terrain",
            "name": name,
            "pos": [6, 0] if pos is None else pos,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **extra,
        }
    )


def undo(pg, result):  # noqa: F811
    return pg.edit_terrain_for_play({"action": "undo_terrain_edit", **result["undo_terrain_edit"]})


def dock(pg):  # noqa: F811
    pg.add_terrain("ramp", name="dock", pos=[3, 0], params={"length": 0.5, "landing": 0.3, "width": 0.6})


def test_custom_fields_order_and_relative_heading_survive_move_and_undo(pg):  # noqa: F811
    pg.add_terrain(
        "ramp",
        name="dock",
        yaw=13.5,
        params={
            "length": 2.1,
            "width": 2.4,
            "angle": 8,
            "landing": 2,
            "direction": "up",
            "style": "wood",
            "friction": 0.7,
        },
    )
    pg.add_terrain("stairs", name="other", pos=[-5, 5])
    pg.add_object("prop", pos=[-5, -5])
    before = pg.current_world()
    result = move(pg, pos=[4, -4, 0], turns=1)
    after = pg.current_world()
    assert after.terrain[0].model_dump() == {**before.terrain[0].model_dump(), "pos": [4, -4], "yaw": 103.5}
    assert after.terrain[1:] == before.terrain[1:] and after.objects == before.objects
    assert pg.session.paused and pg.session.data.time == 0
    assert pg._play_terrain_edit.generation == result["generation"]
    undo(pg, result)
    assert pg.current_world() == before and pg._play_terrain_edit is None
    with pytest.raises(PlaygroundError, match="no longer available"):
        undo(pg, result)


@pytest.mark.parametrize(
    "patch",
    [
        {"turns": True},
        {"turns": 1.0},
        {"turns": 4},
        {"turns": -1},
        {"yaw_delta": True},
        {"yaw_delta": "15"},
        {"yaw_delta": None},
        {"yaw_delta": float("nan")},
        {"yaw_delta": float("inf")},
        {"yaw_delta": -float("inf")},
        {"yaw_delta": 361},
        {"yaw_delta": -361},
        {"turns": 0, "yaw_delta": 15},
        {"turns": 1, "yaw_delta": 0},
        {"turns": 0, "yaw_delta": 0},
        {"yaw": 90},
        {"pos": [True, 0]},
        {"pos": ["3", 0]},
        {"pos": [float("nan"), 0]},
        {"pos": [101, 0]},
        {"pos": [1]},
        {"version": True},
        {"generation": -1},
        {"world_sha256": "bad"},
        {"params": {}},
    ],
)
def test_strict_request_refuses_without_mutation(pg, patch):  # noqa: F811
    dock(pg)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="invalid terrain edit"):
        move(pg, **patch)
    unchanged(pg, before)


def test_metadata_uses_exact_owner_and_instance_geometry(pg):  # noqa: F811
    dock(pg)
    pg.add_terrain("stairs", name="dock_extra", pos=[-5, 0], params={"steps": 2, "step_height": 0.08})
    info = terrain_edit.metadata(pg)
    assert info["version"] == pg.session.version and info["world_sha256"] == pg._play_world_sha256
    for entry in info["terrains"]:
        names = [pg.session.model.geom(g).name for g in entry["geoms"]]
        assert names == [entry["name"], *[f"{entry['name']}/{i}" for i in range(1, len(entry["boxes"]))]]
    pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# outside editor\n")
    assert terrain_edit.metadata(pg) == {"terrains": []}


@pytest.mark.parametrize("yaw", [13.5, 7213.5, -7213.5])
@pytest.mark.parametrize("delta", [15, -15, 14.5])
def test_relative_fine_heading_preserves_authored_turns_and_exact_undo(pg, yaw, delta):  # noqa: F811
    pg.add_terrain("ramp", name="dock", pos=[3, 0], yaw=yaw, params={"length": 0.5, "landing": 0.3, "width": 0.6})
    before = pg.current_world()
    result = move(pg, yaw_delta=delta)
    after = pg.current_world()
    assert after.terrain[0].yaw == yaw + delta
    assert after.terrain[0].model_dump(exclude={"pos", "yaw"}) == before.terrain[0].model_dump(exclude={"pos", "yaw"})
    assert pg.session.paused and pg.session.data.time == 0
    undo(pg, result)
    assert pg.current_world() == before


@pytest.mark.parametrize("delta", [0, -360, 360])
def test_zero_net_rotation_keeps_model_world_and_previous_inverse(pg, delta):  # noqa: F811
    dock(pg)
    move(pg, yaw_delta=15)
    receipt = pg._play_terrain_edit
    before = snapshot(pg)
    response = move(pg, pos=pg.current_world().terrain[0].pos, yaw_delta=delta)
    assert response["unchanged"] and pg._play_terrain_edit is receipt
    unchanged(pg, before)


def test_relative_fine_rotation_cannot_bypass_target_clearance(pg):  # noqa: F811
    dock(pg)
    pg.add_object("blocker", shape="box", size=[0.1, 0.1, 0.2], pos=[6, 0.6], fixed=True)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied"):
        move(pg, yaw_delta=45)
    unchanged(pg, before)
    move(pg, yaw_delta=0)
    assert pg.current_world().terrain[0].yaw == 0


def test_heightfields_are_not_offered_as_movable_blocks(pg):  # noqa: F811
    pg.add_terrain("hills", name="dock")
    assert not terrain_edit.metadata(pg)["terrains"]
    with pytest.raises(PlaygroundError, match="editable ramp or stair"):
        move(pg)


def test_overlapping_own_old_footprint_is_allowed(pg):  # noqa: F811
    dock(pg)
    move(pg, pos=[3.1, 0])
    assert pg.current_world().terrain[0].pos == [3.1, 0]


def test_no_op_preserves_existing_undo_and_history(pg):  # noqa: F811
    addition = place(pg, pos=[3, 0])
    receipt = pg._play_addition
    before = snapshot(pg)
    history = list(pg.store.history.glob("*.yaml"))
    result = move(pg, name=addition["terrain"], pos=[3, 0])
    assert result["unchanged"] and pg._play_addition is receipt
    unchanged(pg, before)
    assert list(pg.store.history.glob("*.yaml")) == history


@pytest.mark.parametrize("where", ["source", "target"])
@pytest.mark.parametrize("pose", ["saved", "live"])
def test_saved_and_live_robot_at_either_footprint_refuses(pg, where, pose):  # noqa: F811
    dock(pg)
    x = 3 if where == "source" else 6
    pg.add_robot("rover", template="rover", pos=[x if pose == "saved" else -4, 0])
    pg.session.data.qpos[0] = -4 if pose == "saved" else x
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        move(pg)
    unchanged(pg, before)


@pytest.mark.parametrize("x", [3, 6])
def test_fixed_overhead_object_also_protects_either_position(pg, x):  # noqa: F811
    dock(pg)
    pg.add_object("table", shape="table", size=[0.5, 0.5, 0.7], pos=[x, 0])
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by table"):
        move(pg)
    unchanged(pg, before)


def test_target_terrain_and_water_full_extents_are_checked(pg):  # noqa: F811
    dock(pg)
    pg.add_terrain("stairs", name="other", pos=[6.6, 0])
    with pytest.raises(PlaygroundError, match="occupied by other"):
        move(pg)
    pg.set_floor(water={"level": 0.1, "area": [5.5, 1, 7, 2]})
    with pytest.raises(PlaygroundError, match="dry ground"):
        move(pg, pos=[6, 2.3], turns=1)


def test_tabletop_click_is_refused(pg):  # noqa: F811
    dock(pg)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="on the ground"):
        move(pg, pos=[6, 0, 0.75])
    unchanged(pg, before)


@pytest.mark.parametrize("where", [3, 6])
def test_live_occupant_moving_in_during_compile_blocks_either_area(pg, monkeypatch, where):  # noqa: F811
    dock(pg)
    pg.add_robot("rover", template="rover", pos=[-4, 0])
    saved = pg.store.file.read_bytes()
    original = terrain_create.compile_world

    def moved(*args):
        compiled = original(*args)
        pg.session.data.qpos[0] = where
        return compiled

    monkeypatch.setattr(terrain_create, "compile_world", moved)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        move(pg)
    assert pg.store.file.read_bytes() == saved and pg.session.data.qpos[0] == where


@pytest.mark.parametrize("change", ["reset", "disk"])
def test_compile_race_refuses_and_preserves_external_change(pg, monkeypatch, change):  # noqa: F811
    dock(pg)
    original, saved = terrain_create.compile_world, pg.store.file.read_bytes()

    def changed(*args):
        compiled = original(*args)
        if change == "reset":
            pg.session.reset()
        else:
            pg.store.file.write_bytes(saved + b"\n# external\n")
        return compiled

    monkeypatch.setattr(terrain_create, "compile_world", changed)
    with pytest.raises(PlaygroundError, match="changed"):
        move(pg)
    assert pg.store.file.read_bytes() == (saved if change == "reset" else saved + b"\n# external\n")


@pytest.mark.parametrize("where", [3, 6])
def test_undo_checks_current_and_original_spot_and_can_retry(pg, where):  # noqa: F811
    dock(pg)
    pg.add_robot("rover", template="rover", pos=[-4, 0])
    result = move(pg)
    pg.session.data.qpos[0] = where
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        undo(pg, result)
    unchanged(pg, before)
    pg.session.data.qpos[0] = -4
    undo(pg, result)
    assert pg.current_world().terrain[0].pos == [3, 0]


@pytest.mark.parametrize("fail", ["load", "publish"])
def test_partial_install_restores_exact_runtime_files_and_receipt_identities(pg, monkeypatch, fail):  # noqa: F811
    place(pg, pos=[-4, 0])
    result = move(pg, name="ramp", pos=[-3.8, 0])
    for field in ("_play_addition", "_play_edit", "_play_robot_edit"):
        setattr(pg, field, object())
    pg._places = {"original": "slots"}
    identities = {
        field: getattr(pg, field)
        for field in ("_play_addition", "_play_edit", "_play_robot_edit", "_play_terrain_edit", "_places")
    }
    before = snapshot(pg)
    receipt = pg._play_terrain_edit
    files = {p: p.read_bytes() for p in [*pg.store.history.glob("*.yaml"), pg.store.build_dir / "world.xml"]}
    if fail == "load":
        original = pg._load_world

        def broken(*args):
            original(*args)
            raise OSError("injected load failure")

        monkeypatch.setattr(pg, "_load_world", broken)
    else:
        original = terrain_edit.TerrainEdit

        def broken(*args):
            pg._play_terrain_edit = original(*args)
            raise OSError("injected publish failure")

        monkeypatch.setattr(terrain_edit, "TerrainEdit", broken)
    with pytest.raises(OSError, match="injected"):
        move(pg, name="ramp", pos=[4, 0])
    unchanged(pg, before)
    assert all(getattr(pg, field) is value for field, value in identities.items())
    assert pg._play_terrain_edit is receipt
    assert all(p.read_bytes() == b for p, b in files.items())
    assert set(pg.store.history.glob("*.yaml")) == {p for p in files if p.parent == pg.store.history}
    assert result["undo_terrain_edit"]["token"] == receipt.token


def test_cached_only_compile_and_generation_bound_receipt(pg, monkeypatch):  # noqa: F811
    dock(pg)
    original = terrain_create.compile_world

    def download(*args):
        menagerie.require_downloads_allowed("uncached")
        return original(*args)

    with monkeypatch.context() as m:
        m.setattr(terrain_create, "compile_world", download)
        with pytest.raises(menagerie.MenagerieError, match="not cached"):
            move(pg)
    original_commit = terrain_edit._commit

    def reset_after(*args, **kwargs):
        result = original_commit(*args, **kwargs)
        pg.session.reset()
        return result

    monkeypatch.setattr(terrain_edit, "_commit", reset_after)
    result = move(pg)
    assert pg._play_terrain_edit.generation < pg.session.generation
    with pytest.raises(PlaygroundError, match="changed"):
        undo(pg, result)


def test_successful_other_edit_clears_old_terrain_receipt(pg):  # noqa: F811
    dock(pg)
    result = move(pg)
    pg.add_object("new", pos=[-4, 0])
    assert pg._play_terrain_edit is None
    with pytest.raises(PlaygroundError, match="no longer available"):
        undo(pg, result)


def test_custom_reset_hooks_refuse_and_hide_metadata(pg):  # noqa: F811
    dock(pg)
    pg.session.reset_hooks.append(lambda: None)
    assert not terrain_edit.metadata(pg)["terrains"]
    with pytest.raises(layouts.LayoutError, match="custom reset hooks"):
        move(pg)


@pytest.mark.parametrize("operation", ["add", "move", "undo"])
def test_post_commit_cleanup_warning_is_returned(pg, monkeypatch, operation):  # noqa: F811
    dock(pg)
    result = move(pg) if operation == "undo" else None
    original = layouts._install_world

    @contextmanager
    def warning(*args, **kwargs):
        with original(*args, **kwargs) as notes:
            yield notes
        notes.append("injected post-commit cleanup warning")

    monkeypatch.setattr(layouts, "_install_world", warning)
    response = place(pg, pos=[-4, 0]) if operation == "add" else undo(pg, result) if operation == "undo" else move(pg)
    assert "injected post-commit cleanup warning" in response["notes"]
