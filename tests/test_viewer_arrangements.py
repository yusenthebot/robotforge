"""Persistent arrangements reuse group validation and its single guarded transaction."""

from __future__ import annotations

import math

import mujoco
import pytest
from test_arrangements import rename_or_skip
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_copy import state, unchanged
from test_viewer_object_group import guard, stack, table, undo

from robotforge.design import arrangements


def action(pg, operation, **fields):  # noqa: F811
    return pg.arrangements_for_play({"action": "arrangement_" + operation, **fields})


def save(pg, names=("base", "top"), label="Stack"):  # noqa: F811
    return action(pg, "save", names=list(names), name=label, **guard(pg))["saved"]


def prepare(pg, saved):  # noqa: F811
    return action(pg, "prepare", arrangement_id=saved["id"], sha256=saved["sha256"], **guard(pg))


def insert(pg, prepared, **patch):  # noqa: F811
    return action(
        pg,
        "insert",
        **{
            "token": prepared["token"],
            "pos": [3.0, 0.0, 0.0],
            "yaw": 0.0,
            **{key: prepared[key] for key in ("version", "generation", "world_sha256")},
            **patch,
        },
    )


def test_save_list_remove_do_not_compile_change_runtime_or_consume_undo_preparation(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module
    from robotforge.viewer.object_group import Prepare, _prepare

    stack(pg)
    preparation = _prepare(pg, Prepare(names=["base", "top"], **guard(pg)))
    receipt = pg._play_edit
    prepared = pg._play_group
    before = state(pg)

    def forbidden(*args, **kwargs):
        pytest.fail("catalogue operation compiled a world")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    saved = save(pg)
    assert action(pg, "list")["entries"] == [saved]
    action(pg, "remove", arrangement_id=saved["id"], sha256=saved["sha256"], **guard(pg))
    unchanged(pg, before)
    assert pg._play_group is prepared and pg._play_group.token == preparation["token"]
    assert pg._play_edit is receipt


@pytest.mark.parametrize("layout", ["stack", "table"])
def test_insert_one_compile_version_history_unique_names_and_normal_undo(pg, monkeypatch, layout):  # noqa: F811
    from robotforge.design import store as store_module

    (stack if layout == "stack" else table)(pg)
    originals = pg.store.load().objects
    saved = save(pg, [obj.name for obj in originals])
    before = state(pg)
    calls = []
    real_compile = store_module.compile_world

    def counted(*args, **kwargs):
        calls.append(1)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(store_module, "compile_world", counted)
    prepared = prepare(pg, saved)
    assert not calls
    assert set(prepared["members"][0]) == {"name", "shape", "size", "offset", "yaw"}
    result = insert(pg, prepared, yaw=37.0)
    assert len(calls) == 1 and pg.session.version == before[5] + 1
    assert len(list(pg.store.history.glob("*.yaml"))) == len(before[2]) + 1
    world = pg.store.load()
    for source, clone, preview in zip(originals, world.objects[len(originals) :], prepared["members"], strict=True):
        assert source.model_dump(exclude={"name", "pos", "yaw"}) == clone.model_dump(exclude={"name", "pos", "yaw"})
        x, y, z = preview["offset"]
        angle = math.radians(37)
        assert clone.pos == pytest.approx(
            [3 + x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle), z]
        )
        assert clone.yaw == pytest.approx(preview["yaw"] + 37)
    assert undo(pg, result)["removed"] == result["objects"]
    assert pg.store.load().objects == originals
    assert action(pg, "list")["entries"] == [saved]


@pytest.mark.parametrize(
    "shape,size",
    [
        ("box", [0.2, 0.3, 0.2]),
        ("sphere", [0.2]),
        ("cylinder", [0.2, 0.3]),
        ("table", [0.8, 0.6, 0.5]),
        ("bin", [0.4, 0.4, 0.2]),
    ],
)
@pytest.mark.parametrize("fixed", [True, False])
def test_every_supported_shape_preserves_full_fields_and_unwrapped_live_yaw(pg, shape, size, fixed):  # noqa: F811
    fields = {"surface_material": "steel"}
    if shape in {"box", "sphere", "cylinder"}:
        fields["material"] = "wood"
    for index in range(2):
        pg.add_object(
            f"prop{index}",
            shape=shape,
            size=size,
            fixed=fixed,
            pos=[-3 + index * 1.5, 0],
            yaw=725,
            mass=None,
            color=[0.2, 0.4, 0.6, 0.8],
            **fields,
        )
    pg.session.paused = True
    if not fixed and shape != "table":
        for index in range(2):
            joint = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, f"prop{index}")
            address = pg.session.model.jnt_qposadr[joint]
            pg.session.data.qpos[address] += 0.2
            angle = math.radians(35) / 2
            pg.session.data.qpos[address + 3 : address + 7] = [math.cos(angle), 0, 0, math.sin(angle)]
        mujoco.mj_forward(pg.session.model, pg.session.data)
    originals = pg.store.load().objects
    saved = save(pg, [obj.name for obj in originals])
    prepared = prepare(pg, saved)
    expected_yaw = 755 if not fixed and shape != "table" else 725
    assert all(member["yaw"] == pytest.approx(expected_yaw) for member in prepared["members"])
    result = insert(pg, prepared, yaw=20.0)
    for original, cloned in zip(originals, pg.store.load().objects[2:], strict=True):
        assert cloned.model_dump(exclude={"name", "pos", "yaw"}) == original.model_dump(exclude={"name", "pos", "yaw"})
        assert cloned.yaw == pytest.approx(expected_yaw + 20)
    assert result["objects"] == ["prop0_2", "prop1_2"]


def test_global_cross_world_record_needs_no_live_sources(pg, monkeypatch):  # noqa: F811
    from robotforge.viewer import object_group

    stack(pg)
    saved = save(pg)
    pg.new_world("destination")
    pg.session.paused = True
    assert action(pg, "list")["entries"] == [saved]

    def forbidden(*args, **kwargs):
        pytest.fail("stored arrangement consulted a source geometry pose")

    monkeypatch.setattr(object_group, "_pose_digest", forbidden)
    prepared = prepare(pg, saved)
    assert insert(pg, prepared)["objects"] == ["base_2", "top_2"]


def test_furniture_finishes_survive_saved_library_and_cross_world_insert_inverse(pg):  # noqa: F811
    table(pg)

    def finishes(world):
        world.objects[0].surface_material = "wood"
        world.objects[2].surface_material = "steel"

    pg._edit(finishes)
    pg.session.paused = True
    originals = pg.store.load().objects
    saved = save(pg, [entry.name for entry in originals], label="Furniture")
    pg.new_world("furniture_destination")
    pg.session.paused = True
    assert action(pg, "list")["entries"] == [saved]
    result = insert(pg, prepare(pg, saved), yaw=37.0)
    for original, cloned in zip(originals, pg.store.load().objects, strict=True):
        assert cloned.model_dump(exclude={"name", "pos", "yaw"}) == original.model_dump(exclude={"name", "pos", "yaw"})
    assert [entry.surface_material for entry in pg.store.load().objects] == ["wood", None, "steel"]
    undo(pg, result)
    assert pg.store.load().objects == [] and action(pg, "list")["entries"] == [saved]


def test_occupied_target_preserves_prepared_for_retry(pg):  # noqa: F811
    stack(pg)
    prepared = prepare(pg, save(pg))
    before = state(pg)
    token = pg._play_group
    with pytest.raises(arrangements.ArrangementError, match="occupied") as error:
        insert(pg, prepared, pos=[-2.0, 0.0, 0.0])
    assert error.value.code == "arrangement_occupied"
    unchanged(pg, before)
    assert pg._play_group is token
    assert insert(pg, prepared)["objects"] == ["base_2", "top_2"]


@pytest.mark.parametrize("mutation", ["tamper", "delete", "root"])
def test_record_changes_before_insert_are_rejected_without_compile(pg, monkeypatch, mutation):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    saved = save(pg)
    prepared = prepare(pg, saved)
    path = arrangements.home() / f"{saved['id']}.json"
    if mutation == "tamper":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "delete":
        path.unlink()
    else:
        root = arrangements.home()
        rename_or_skip(root, root.with_name("old_arrangements"))
        root.mkdir()
    before = state(pg)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid arrangement compiled")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    with pytest.raises(arrangements.ArrangementError):
        insert(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is None


@pytest.mark.parametrize("mutation", ["tamper", "delete", "root"])
def test_postcompile_record_changes_fail_before_publication(pg, monkeypatch, mutation):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    saved = save(pg)
    prepared = prepare(pg, saved)
    before = state(pg)
    real_compile = store_module.compile_world

    def changed(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        root = arrangements.home()
        path = root / f"{saved['id']}.json"
        if mutation == "tamper":
            path.write_bytes(path.read_bytes() + b" ")
        elif mutation == "delete":
            path.unlink()
        else:
            rename_or_skip(root, root.with_name("old_arrangements"))
            root.mkdir()
        return compiled

    monkeypatch.setattr(store_module, "compile_world", changed)
    with pytest.raises(arrangements.ArrangementError):
        insert(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is None


def test_reset_other_preparation_and_stale_guard_invalidate_token(pg):  # noqa: F811
    stack(pg)
    saved = save(pg)
    prepared = prepare(pg, saved)
    pg.session.reset()
    with pytest.raises(arrangements.ArrangementError):
        insert(pg, prepared)
    pg.session.paused = True
    prepared = prepare(pg, saved)
    pg.edit_object_for_play({"action": "prepare_object_group", "names": ["base", "top"], **guard(pg)})
    with pytest.raises(arrangements.ArrangementError, match="expired"):
        insert(pg, prepared)


def test_runtime_mutation_during_compile_rejected_and_original_source_pose_not_required(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    prepared = prepare(pg, save(pg))
    original_world = pg.store.file.read_bytes()
    real_compile = store_module.compile_world

    def changed(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        pg.session.data.time += 1.0
        return compiled

    monkeypatch.setattr(store_module, "compile_world", changed)
    with pytest.raises(arrangements.ArrangementError, match="pose changed"):
        insert(pg, prepared)
    assert pg.store.file.read_bytes() == original_world


@pytest.mark.parametrize(
    "patch",
    [{"mode": "copy"}, {"names": ["base"]}, {"name": "bad\nname"}, {"names": ["base", "base"]}, {"file": "world.yaml"}],
)
def test_invalid_save_input_cannot_change_any_world_state(pg, patch):  # noqa: F811
    stack(pg)
    before = state(pg)
    message = {"names": ["base", "top"], "name": "Example", **guard(pg), **patch}
    with pytest.raises(arrangements.ArrangementError):
        action(pg, "save", **message)
    unchanged(pg, before)


def test_sixteen_members_preserve_actual_live_relative_positions(pg):  # noqa: F811
    from robotforge.design.world import ObjectEntry

    def objects(world):
        world.objects = [
            ObjectEntry(name=f"block{index}", size=[0.1, 0.1, 0.1], pos=[-8 + index * 0.3, 0]) for index in range(16)
        ]

    pg._edit(objects)
    pg.session.paused = True
    joint = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, "block0")
    address = pg.session.model.jnt_qposadr[joint]
    pg.session.data.qpos[address + 1] = 0.4
    mujoco.mj_forward(pg.session.model, pg.session.data)
    saved = save(pg, [f"block{index}" for index in range(16)])
    prepared = prepare(pg, saved)
    assert len(prepared["members"]) == 16
    assert prepared["members"][0]["offset"][1] == pytest.approx(0.4 - 0.4 / 16)
    assert prepared["members"][1]["offset"][1] == pytest.approx(-0.4 / 16)
    assert len(insert(pg, prepared, pos=[5.0, 0.0, 0.0])["objects"]) == 16


def test_support_mismatch_and_tilt_reject_without_publishing(pg):  # noqa: F811
    stack(pg)
    prepared = prepare(pg, save(pg))
    before = state(pg)
    token = pg._play_group
    with pytest.raises(arrangements.ArrangementError, match="support") as error:
        insert(pg, prepared, pos=[3.0, 0.0, 0.3])
    assert error.value.code == "arrangement_target"
    unchanged(pg, before)
    assert pg._play_group is token
    joint = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, "top")
    address = pg.session.model.jnt_qposadr[joint]
    angle = math.radians(20) / 2
    pg.session.data.qpos[address + 3 : address + 7] = [math.cos(angle), math.sin(angle), 0, 0]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    with pytest.raises(arrangements.ArrangementError, match="tilted"):
        save(pg, label="Tilted")
    assert len(action(pg, "list")["entries"]) == 1


def test_target_checks_both_authored_and_current_obstacles(pg):  # noqa: F811
    stack(pg)
    pg.add_object("obstacle", shape="box", size=[0.2, 0.2, 0.2], pos=[3.0, 0.0])
    pg.session.paused = True
    saved = save(pg)
    joint = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_JOINT, "obstacle")
    address = pg.session.model.jnt_qposadr[joint]
    pg.session.data.qpos[address] = 5.0
    mujoco.mj_forward(pg.session.model, pg.session.data)
    prepared = prepare(pg, saved)
    before = state(pg)
    for target in (3.0, 5.0):
        with pytest.raises(arrangements.ArrangementError, match="occupied"):
            insert(pg, prepared, pos=[target, 0.0, 0.0])
        unchanged(pg, before)
    assert insert(pg, prepared, pos=[7.0, 0.0, 0.0])["objects"] == ["base_2", "top_2"]


def test_failed_runtime_install_rolls_back_world_and_preserves_preparation(pg, monkeypatch):  # noqa: F811
    stack(pg)
    prepared = prepare(pg, save(pg))
    before = state(pg)
    token = pg._play_group

    def failed(*args, **kwargs):
        raise RuntimeError("install failed")

    monkeypatch.setattr(pg, "_load_world", failed)
    with pytest.raises(RuntimeError, match="install failed"):
        insert(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is token


def test_prepare_rejects_replaced_ancestor_even_when_arrangement_root_inode_is_preserved(pg, monkeypatch, tmp_path):  # noqa: F811
    from robotforge.design import store as store_module

    parent = tmp_path / "library_parent"
    root = parent / "arrangements"
    monkeypatch.setattr(arrangements, "home", lambda: root)
    stack(pg)
    saved = save(pg)
    prepared = prepare(pg, saved)
    root_identity = root.stat().st_dev, root.stat().st_ino
    moved_parent = tmp_path / "moved_parent"
    rename_or_skip(parent, moved_parent)
    parent.mkdir()
    rename_or_skip(moved_parent / "arrangements", root)
    assert (root.stat().st_dev, root.stat().st_ino) == root_identity
    before = state(pg)
    record_path = root / f"{saved['id']}.json"
    record_bytes = record_path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("arrangement with a replaced ancestor reached compilation")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    with pytest.raises(arrangements.ArrangementError, match="changed"):
        insert(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is None
    assert record_path.read_bytes() == record_bytes


def test_remove_while_running_preserves_world_runtime_undo_and_preparation(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    saved = save(pg)
    inserted = insert(pg, prepare(pg, saved))
    prepare(pg, saved)
    receipt, preparation = pg._play_edit, pg._play_group
    assert receipt.token == inserted["undo_edit"]["token"]
    pg.session.paused = False
    before = state(pg)

    def forbidden(*args, **kwargs):
        pytest.fail("removing an arrangement compiled a world")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    result = action(pg, "remove", arrangement_id=saved["id"], sha256=saved["sha256"], **guard(pg))
    assert result["entries"] == []
    unchanged(pg, before)
    assert not pg.session.paused
    assert pg._play_edit is receipt and pg._play_group is preparation


def test_observed_destination_bytes_change_permanently_consumes_token_even_after_restore(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    prepared = prepare(pg, save(pg))
    token = pg._play_group
    original = pg.store.file.read_bytes()
    changed = original + b"\n"
    pg.store.file.write_bytes(changed)
    external = state(pg)

    def forbidden(*args, **kwargs):
        pytest.fail("stale destination reached compilation")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    with pytest.raises(arrangements.ArrangementError) as first:
        insert(pg, prepared)
    assert first.value.code == "arrangement_stale"
    unchanged(pg, external)
    assert token.invalidated and pg._play_group is None
    pg.store.file.write_bytes(original)
    restored = state(pg)
    with pytest.raises(arrangements.ArrangementError) as retry:
        insert(pg, prepared)
    assert retry.value.code == "arrangement_stale"
    unchanged(pg, restored)


def test_postcompile_destination_bytes_change_consumes_token_and_preserves_external_bytes(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    prepared = prepare(pg, save(pg))
    token = pg._play_group
    before = state(pg)
    original = pg.store.file.read_bytes()
    changed = original + b"\n"
    real_compile = store_module.compile_world
    calls = []

    def change_destination(*args, **kwargs):
        compiled = real_compile(*args, **kwargs)
        calls.append(1)
        pg.store.file.write_bytes(changed)
        return compiled

    monkeypatch.setattr(store_module, "compile_world", change_destination)
    with pytest.raises(arrangements.ArrangementError) as first:
        insert(pg, prepared)
    assert first.value.code == "arrangement_stale"
    assert calls == [1]
    assert token.invalidated and pg._play_group is None
    assert pg.store.file.read_bytes() == changed
    pg.store.file.write_bytes(original)
    unchanged(pg, before)
    with pytest.raises(arrangements.ArrangementError) as retry:
        insert(pg, prepared)
    assert retry.value.code == "arrangement_stale"
    assert calls == [1]
    unchanged(pg, before)


def test_wrong_request_token_does_not_consume_valid_preparation(pg):  # noqa: F811
    stack(pg)
    prepared = prepare(pg, save(pg))
    token = pg._play_group
    before = state(pg)
    wrong = "0" * 32 if prepared["token"] != "0" * 32 else "1" * 32
    with pytest.raises(arrangements.ArrangementError) as error:
        insert(pg, prepared, token=wrong)
    assert error.value.code == "arrangement_stale"
    unchanged(pg, before)
    assert pg._play_group is token and not token.invalidated
    assert insert(pg, prepared)["objects"] == ["base_2", "top_2"]
