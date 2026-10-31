"""Same-name rigid layout moves preserve complete arrangements and a guarded inverse."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_copy import state, unchanged
from test_viewer_object_group import duplicate, prepare, stack, table, undo

from robotforge.playground import PlaygroundError


def moving(pg, names=("base", "top")):  # noqa: F811
    return prepare(pg, names, mode="move")


def move(pg, prepared, **patch):  # noqa: F811
    return pg.edit_object_for_play(
        {
            "action": "move_object_group",
            "token": prepared["token"],
            "pos": [3.0, 0.0, 0.0],
            "yaw": 0.0,
            **{key: prepared[key] for key in ("version", "generation", "world_sha256")},
            **patch,
        }
    )


def pose(pg, name, xyz, yaw=0.0):  # noqa: F811
    adr = pg.session.model.jnt_qposadr[pg.session.model.joint(name).id]
    angle = math.radians(yaw) / 2
    pg.session.data.qpos[adr : adr + 7] = [*xyz, math.cos(angle), 0, 0, math.sin(angle)]
    mujoco.mj_forward(pg.session.model, pg.session.data)


@pytest.mark.parametrize("reverse", [False, True])
def test_move_keeps_names_world_order_authored_settings_and_one_compile_inverse(pg, monkeypatch, reverse):  # noqa: F811
    from robotforge.design import store as store_module

    table(pg)
    originals = pg.store.load().objects
    before = state(pg)
    calls, loads = [], []
    compile_world, load_world = store_module.compile_world, pg._load_world

    def compile_count(*args, **kwargs):
        calls.append(1)
        return compile_world(*args, **kwargs)

    def load_count(*args, **kwargs):
        loads.append(1)
        return load_world(*args, **kwargs)

    monkeypatch.setattr(store_module, "compile_world", compile_count)
    monkeypatch.setattr(pg, "_load_world", load_count)
    names = [obj.name for obj in (reversed(originals) if reverse else originals)]
    prepared = moving(pg, names)
    assert not calls and prepared["mode"] == "move"
    result = move(pg, prepared, yaw=37.0)
    moved = pg.store.load().objects
    assert result["objects"] == [obj.name for obj in originals] and result["moved"]
    assert len(calls) == len(loads) == 1 and pg.session.version == before[5] + 1
    assert len(list(pg.store.history.glob("*.yaml"))) == len(before[2]) + 1
    for old, new in zip(originals, moved, strict=True):
        assert old.model_dump(exclude={"pos", "yaw"}) == new.model_dump(exclude={"pos", "yaw"})
        assert new.yaw == pytest.approx(37)
    restored = undo(pg, result)
    assert restored["objects"] == result["objects"] and restored["moved"]
    assert "removed" not in restored and restored["undo_edit"] is None
    assert pg.store.load().objects == originals and len(calls) == len(loads) == 2
    assert pg.session.version == before[5] + 2
    np.testing.assert_allclose(pg.session.data.qpos, before[7], atol=1e-12, rtol=0)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


def test_live_layout_height_shift_and_undo_restore_author_entries_and_original_live_pose(pg):  # noqa: F811
    pg.add_object("base", shape="box", size=[0.4, 0.4, 0.2], pos=[-2, 0], material="wood")
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0, 0.3], material="steel")
    pg.add_object("target", shape="table", size=[1.4, 1.4, 0.5], pos=[3, 0])
    pg.add_object("other", shape="sphere", size=[0.1], pos=[7, 0])
    pg.session.paused = True
    originals = pg.store.load().objects
    pose(pg, "base", [-1.8, 0.1, 0.1], 20)
    pose(pg, "top", [-1.8, 0.1, 0.3], 20)
    pose(pg, "other", [7.3, 0.2, 0.05])
    before = pg.session.data.qpos.copy()
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 2.0
    result = move(pg, moving(pg), pos=[3, 0, 0.5], yaw=47.0)
    entries = pg.store.load().objects
    assert [obj.pos[2] for obj in entries[:2]] == pytest.approx([0.6, 0.8])
    assert [obj.yaw for obj in entries[:2]] == pytest.approx([67, 67])
    assert list(pg.session.data.body("other").xpos) == pytest.approx([7.3, 0.2, 0.05])
    undo(pg, result)
    assert pg.store.load().objects == originals
    np.testing.assert_allclose(pg.session.data.qpos, before, atol=1e-12, rtol=0)
    assert not np.any(pg.session.data.qvel) and pg.session.data.time == 0 and pg.session.paused


@pytest.mark.parametrize("delta,yaw", [([0.1, 0, 0], 0.0), ([0, 0, 0], 90.0)])
def test_old_selected_footprint_is_vacated_atomically(pg, delta, yaw):  # noqa: F811
    table(pg)
    prepared = moving(pg, ["base", "top", "bin"])
    target = [a + b for a, b in zip(prepared["anchor"], delta, strict=True)]
    result = move(pg, prepared, pos=target, yaw=yaw)
    assert result["objects"] == ["base", "top", "bin"]
    assert len(pg.store.load().objects) == 3


def test_unchanged_pivot_entry_uses_validated_target_pose_instead_of_old_live_pose(pg):  # noqa: F811
    for name, y in (("left", -1), ("pivot", 0), ("right", 1)):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[-2, y, 0.1], yaw=180 if name == "pivot" else 0)
    pg.session.paused = True
    original = pg.store.load().objects
    pose(pg, "pivot", [-2, 0, 0.1])
    before = pg.session.data.qpos.copy()
    prepared = moving(pg, ["left", "pivot", "right"])
    result = move(pg, prepared, pos=prepared["anchor"], yaw=180.0)
    assert pg.store.load().objects[1] == original[1]
    rotation = pg.session.data.body("pivot").xmat.reshape(3, 3)
    assert math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])) == pytest.approx(180)
    undo(pg, result)
    assert pg.store.load().objects == original
    np.testing.assert_allclose(pg.session.data.qpos, before, atol=1e-12, rtol=0)


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_prepared_token_cannot_cross_operation_modes(pg, mode):  # noqa: F811
    stack(pg)
    prepared = prepare(pg, mode=mode)
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError, match="preparation|mode"):
        (move if mode == "copy" else duplicate)(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is token


@pytest.mark.parametrize("phase", ["saved", "live", "after_prepare"])
def test_cannot_leave_unselected_load_behind(pg, phase):  # noqa: F811
    table(pg)
    # The unselected fixed bin rests on the table in both the authored and live world.
    if phase != "saved":
        pg._edit(lambda world: world.objects.pop())
        pg.add_object("payload", shape="box", size=[0.1, 0.1, 0.1], pos=[7, 0])
    pg.session.paused = True
    prepared = moving(pg) if phase == "after_prepare" else None
    if phase != "saved":
        pose(pg, "payload", [-1.7, 0, 0.55])
    before = state(pg)
    with pytest.raises(PlaygroundError, match="supports"):
        move(pg, prepared) if prepared else moving(pg)
    unchanged(pg, before)


def test_task_bound_group_is_rejected_but_existing_copy_stays_available(pg):  # noqa: F811
    from robotforge.design.tasks.spec import TaskEntry

    stack(pg)
    pg.add_robot("arm", template="arm", pos=[7, 0])
    pg._edit(lambda world: world.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="top")))
    pg.session.paused = True
    before = state(pg)
    with pytest.raises(PlaygroundError, match="task"):
        moving(pg)
    unchanged(pg, before)
    assert prepare(pg)["token"]


@pytest.mark.parametrize("case", ["saved", "live", "edge", "floating"])
def test_invalid_target_keeps_preparation_for_retry(pg, case):  # noqa: F811
    stack(pg)
    if case in {"saved", "live"}:
        pg.add_object("obstacle", shape="box", size=[0.2, 0.2, 0.2], pos=[3, 0])
        pose(pg, "obstacle", [5, 0, 0.1])
    elif case == "edge":
        pg.add_object("target", shape="table", size=[0.2, 0.2, 0.5], pos=[3, 0])
    pg.session.paused = True
    prepared = moving(pg)
    before, token = state(pg), pg._play_group
    position = [5 if case == "live" else 3, 0, 0.5 if case in {"edge", "floating"} else 0]
    with pytest.raises(PlaygroundError):
        move(pg, prepared, pos=position)
    unchanged(pg, before)
    assert pg._play_group is token
    assert move(pg, prepared, pos=[7, 0, 0])["moved"]


@pytest.mark.parametrize("obstacle", ["load", "old_spot"])
def test_undo_checks_current_load_and_original_destination_then_allows_retry(pg, obstacle):  # noqa: F811
    stack(pg)
    pg.add_object("payload", shape="box", size=[0.1, 0.1, 0.1], pos=[7, 0])
    pg.session.paused = True
    original = pg.store.load().objects
    result = move(pg, moving(pg))
    receipt = pg._play_edit
    pose(pg, "payload", [3, 0, 0.45] if obstacle == "load" else [-2, 0, 0.15])
    before = state(pg)
    with pytest.raises(PlaygroundError):
        undo(pg, result)
    unchanged(pg, before)
    assert pg._play_edit is receipt
    pose(pg, "payload", [7, 0, 0.05])
    assert undo(pg, result)["moved"]
    assert pg.store.load().objects == original


def test_zero_transform_preserves_full_turns_history_previous_undo_and_live_state(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    result = move(pg, moving(pg))
    prepared = moving(pg)
    before, receipt = state(pg), pg._play_edit

    def forbidden(*args, **kwargs):
        raise AssertionError("a zero transform must not compile")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    noop = move(pg, prepared, pos=prepared["anchor"], yaw=360.0)
    assert noop["unchanged"] and noop["version"] == before[5]
    unchanged(pg, before)
    assert pg._play_edit is receipt and receipt.token == result["undo_edit"]["token"]


def test_nonzero_move_retains_authored_full_turns(pg):  # noqa: F811
    stack(pg)

    def full_turns(world):
        for obj in world.objects:
            obj.yaw = 720.0

    pg._edit(full_turns)
    pg.session.paused = True
    originals = pg.store.load().objects
    result = move(pg, moving(pg), yaw=30.0)
    assert [obj.yaw for obj in pg.store.load().objects] == pytest.approx([750, 750])
    undo(pg, result)
    assert pg.store.load().objects == originals


def test_late_undo_selected_pose_failure_preserves_moved_world_and_its_receipt(pg, monkeypatch):  # noqa: F811
    from robotforge.viewer import object_group

    stack(pg)
    result = move(pg, moving(pg))
    before, receipt = state(pg), pg._play_edit
    original = object_group.carry._apply

    def fail_selected(model, data, snapshot, chosen):
        if pg.session.version > before[5] and chosen("top"):
            raise ValueError("injected inverse pose failure")
        return original(model, data, snapshot, chosen)

    monkeypatch.setattr(object_group.carry, "_apply", fail_selected)
    with pytest.raises(ValueError, match="inverse pose failure"):
        undo(pg, result)
    unchanged(pg, before)
    assert pg._play_edit is receipt


@pytest.mark.parametrize("failure", ["compile", "validate", "load_after", "selected_pose"])
def test_move_failure_rolls_back_runtime_disk_history_and_receipts(pg, monkeypatch, failure):  # noqa: F811
    from robotforge.design import store as store_module
    from robotforge.viewer import object_group

    stack(pg)
    previous = duplicate(pg, prepare(pg))
    prepared = moving(pg)
    before, receipt, token = state(pg), pg._play_edit, pg._play_group

    def fail(*args, **kwargs):
        raise ValueError("injected move failure")

    if failure == "compile":
        monkeypatch.setattr(store_module, "compile_world", fail)
    elif failure == "validate":
        monkeypatch.setattr(object_group, "_support_graph", fail)
    elif failure == "selected_pose":
        original = object_group.carry._apply

        def selected(model, data, snapshot, chosen):
            if pg.session.version > before[5] and chosen("top"):
                fail()
            return original(model, data, snapshot, chosen)

        monkeypatch.setattr(object_group.carry, "_apply", selected)
    else:
        original = pg._load_world

        def load(*args, **kwargs):
            original(*args, **kwargs)
            fail()

        monkeypatch.setattr(pg, "_load_world", load)
    with pytest.raises(ValueError, match="injected move failure"):
        move(pg, prepared, pos=[7, 0, 0])
    unchanged(pg, before)
    assert pg._play_edit is receipt and pg._play_group is token
    assert receipt.token == previous["undo_edit"]["token"]
