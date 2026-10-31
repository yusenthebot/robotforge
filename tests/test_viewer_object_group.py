"""Rigid layout copying checks the actual support graph and commits one guarded transaction."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_copy import state, unchanged

from robotforge.playground import PlaygroundError
from robotforge.viewer.object_edit import metadata


def guard(pg):  # noqa: F811
    return {
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": metadata(pg)["world_sha256"],
    }


def prepare(pg, names=("base", "top"), **patch):  # noqa: F811
    return pg.edit_object_for_play({"action": "prepare_object_group", "names": list(names), **guard(pg), **patch})


def duplicate(pg, prepared, **patch):  # noqa: F811
    return pg.edit_object_for_play(
        {
            "action": "duplicate_object_group",
            "token": prepared["token"],
            "pos": [3.0, 0.0, 0.0],
            "yaw": 0.0,
            **{k: prepared[k] for k in ("version", "generation", "world_sha256")},
            **patch,
        }
    )


def undo(pg, result):  # noqa: F811
    return pg.edit_object_for_play({"action": "undo_object_edit", **result["undo_edit"]})


def stack(pg):  # noqa: F811
    pg.add_object("base", shape="box", size=[0.4, 0.4, 0.2], fixed=True, pos=[-2.0, 0.0], material="wood")
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2.0, 0.0, 0.3], material="steel")
    pg.session.paused = True


def table(pg):  # noqa: F811
    pg.add_object("base", shape="table", size=[1.2, 0.8, 0.5], pos=[-2.0, 0.0], fixed=True)
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2.25, 0.0, 0.6], material="steel")
    pg.add_object("bin", shape="bin", size=[0.3, 0.3, 0.2], pos=[-1.75, 0.0, 0.5], fixed=True)
    pg.session.paused = True


@pytest.mark.parametrize("layout", ["stack", "table"])
def test_supported_layout_has_one_compile_version_history_and_undo(pg, monkeypatch, layout):  # noqa: F811
    from robotforge.design import store as store_module

    (stack if layout == "stack" else table)(pg)
    originals = pg.store.load().objects
    before = state(pg)
    calls = []
    real_compile = store_module.compile_world

    def counted(*args, **kwargs):
        calls.append(1)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(store_module, "compile_world", counted)
    prepared = prepare(pg, [obj.name for obj in originals])
    assert not calls and len(prepared["members"]) == len(originals)
    assert prepared["anchor"][2] == pytest.approx(0)
    result = duplicate(pg, prepared, yaw=37.0)
    assert len(calls) == 1 and pg.session.version == before[5] + 1
    assert len(list(pg.store.history.glob("*.yaml"))) == len(before[2]) + 1
    world = pg.store.load()
    assert world.objects[: len(originals)] == originals
    assert len(result["objects"]) == len(originals) and len(set(result["objects"])) == len(originals)
    assert pg.session.paused and pg._play_group is None
    for original, clone in zip(originals, world.objects[len(originals) :], strict=True):
        assert original.model_dump(exclude={"name", "pos", "yaw"}) == clone.model_dump(exclude={"name", "pos", "yaw"})
        assert clone.yaw == pytest.approx(37)
    assert undo(pg, result)["removed"] == result["objects"]
    assert len(calls) == 2 and pg.store.load().objects == originals
    assert pg.session.version == before[5] + 2
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


@pytest.mark.parametrize("shape,size", [("sphere", [0.2]), ("cylinder", [0.2, 0.3]), ("bin", [0.3, 0.3, 0.2])])
@pytest.mark.parametrize("fixed", [False, True])
def test_all_rigid_shapes_on_a_selected_table(pg, shape, size, fixed):  # noqa: F811
    pg.add_object("base", shape="table", size=[1.0, 0.8, 0.5], pos=[-2, 0])
    z = 0.5 + (0 if shape == "bin" else size[-1] / 2)
    pg.add_object("top", shape=shape, size=size, fixed=fixed, pos=[-2, 0, z], color=[0.1, 0.2, 0.3, 0.7], mass=1.3)
    pg.session.paused = True
    result = duplicate(pg, prepare(pg), yaw=-73.0)
    clone = pg.store.load().objects[-1]
    assert clone.name == result["objects"][-1] and clone.pos[2] == pytest.approx(z)
    assert clone.fixed is fixed and clone.mass == 1.3 and clone.color == [0.1, 0.2, 0.3, 0.7]


def test_current_upright_pose_common_transform_and_height_shift(pg):  # noqa: F811
    pg.add_object("base", shape="box", size=[0.4, 0.4, 0.2], pos=[-2.0, 0])
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-2.0, 0.0, 0.3])
    pg.add_object("support", shape="table", size=[1.4, 1.4, 0.5], pos=[3, 0])
    pg.session.paused = True
    angle = math.radians(20)
    for name, z in (("base", 0.1), ("top", 0.3)):
        adr = pg.session.model.jnt_qposadr[pg.session.model.joint(name).id]
        pg.session.data.qpos[adr : adr + 7] = [-1.8, 0.1, z, math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    source_pose = pg.session.data.qpos.copy()
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 2
    prepared = prepare(pg)
    assert prepared["anchor"] == pytest.approx([-1.8, 0.1, 0])
    duplicate(pg, prepared, pos=[3, 0, 0.5], yaw=47.0)
    clones = pg.store.load().objects[-2:]
    assert [obj.pos[2] for obj in clones] == pytest.approx([0.6, 0.8])
    assert [obj.yaw for obj in clones] == pytest.approx([67, 67])
    np.testing.assert_array_equal(pg.session.data.qpos[: len(source_pose)], source_pose)
    assert not np.any(pg.session.data.qvel) and pg.session.data.time == 0 and pg.session.paused


@pytest.mark.parametrize(
    "patch",
    [
        {"names": ["base"]},
        {"names": ["base", "base"]},
        {"names": ["base", "missing"]},
        {"names": ["base"] * 17},
        {"names": ["ba", "top"]},
        {"version": True},
        {"generation": -1},
        {"world_sha256": "0" * 64},
        {"mass": 3},
    ],
)
def test_invalid_selection_does_not_change_state(pg, patch):  # noqa: F811
    stack(pg)
    before = state(pg)
    with pytest.raises(PlaygroundError):
        prepare(pg, **patch)
    unchanged(pg, before)


@pytest.mark.parametrize(
    "patch",
    [
        {"pos": [1, 2]},
        {"pos": [True, 2, 0]},
        {"pos": [101, 0, 0]},
        {"yaw": float("nan")},
        {"yaw": True},
        {"material": "wood"},
        {"token": "0" * 32},
    ],
)
def test_invalid_target_does_not_consume_preparation(pg, patch):  # noqa: F811
    stack(pg)
    prepared = prepare(pg)
    receipt, before = pg._play_group, state(pg)
    with pytest.raises(PlaygroundError):
        duplicate(pg, prepared, **patch)
    unchanged(pg, before)
    assert pg._play_group is receipt


@pytest.mark.parametrize("reason", ["tilt", "float", "overlap", "playing", "mechanism"])
def test_invalid_source_layout_is_rejected(pg, reason):  # noqa: F811
    stack(pg)
    if reason == "mechanism":
        pg.add_object("drawer", shape="drawer", pos=[8, 0])
    elif reason == "playing":
        pg.session.paused = False
    else:
        adr = pg.session.model.jnt_qposadr[pg.session.model.joint("top").id]
        if reason == "tilt":
            pg.session.data.qpos[adr + 3 : adr + 7] = [math.cos(0.1), math.sin(0.1), 0, 0]
        else:
            pg.session.data.qpos[adr + 2] = 0.8 if reason == "float" else 0.1
        mujoco.mj_forward(pg.session.model, pg.session.data)
    before = state(pg)
    with pytest.raises(PlaygroundError):
        prepare(pg, ["base", "drawer"] if reason == "mechanism" else ["base", "top"])
    unchanged(pg, before)


@pytest.mark.parametrize("change", ["reset", "pose", "playing", "new_preparation", "file", "readonly"])
def test_stale_or_uneditable_world_rejects_prepared_copy(pg, change):  # noqa: F811
    stack(pg)
    prepared = prepare(pg)
    if change == "reset":
        pg.session.reset()
    elif change == "pose":
        pg.session.data.qpos[0] += 0.02
    elif change == "playing":
        pg.session.paused = False
    elif change == "new_preparation":
        prepare(pg)
    elif change == "file":
        pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# another writer\n")
    else:
        pg.store = None
    with pytest.raises(PlaygroundError):
        duplicate(pg, prepared)


@pytest.mark.parametrize("case", ["originals", "saved", "live", "table_edge", "missing_support", "uneven"])
def test_target_collision_and_full_support_rejections(pg, case):  # noqa: F811
    table(pg)
    if case in {"saved", "live"}:
        pg.add_object("obstacle", shape="box", size=[0.2, 0.2, 0.2], pos=[3, 0, 0.5])
        if case == "live":
            pg.session.data.qpos[pg.session.model.jnt_qposadr[pg.session.model.joint("obstacle").id]] = 5
    elif case == "table_edge":
        pg.add_object("support", shape="table", size=[0.8, 0.8, 0.5], pos=[3, 0])
    elif case == "uneven":
        pg.add_terrain("stairs", name="steps", pos=[3, 0], params={"steps": 3, "step_height": 0.1, "step_depth": 0.3})
    pg.session.paused = True
    prepared = prepare(pg, ["base", "top", "bin"])
    pos = [-2, 0, 0] if case == "originals" else [5, 0, 0] if case == "live" else [3, 0, 0]
    if case in {"table_edge", "missing_support"}:
        pos[2] = 0.5
    before = state(pg)
    with pytest.raises(PlaygroundError):
        duplicate(pg, prepared, pos=pos)
    unchanged(pg, before)


def test_bin_wall_collision_is_not_lost_to_worldbody_ownership(pg):  # noqa: F811
    pg.add_object("base", shape="bin", size=[0.6, 0.4, 0.2], pos=[-2, 0], fixed=True)
    pg.add_object("top", shape="sphere", size=[0.1], pos=[-2, 0, 0.055])
    pg.add_object("obstacle", shape="box", size=[0.02, 0.02, 0.02], pos=[3.2975, 0, 0.1], fixed=True)
    pg.session.paused = True
    before = state(pg)
    with pytest.raises(PlaygroundError, match="occupied"):
        duplicate(pg, prepare(pg))
    unchanged(pg, before)


def test_undo_blocks_external_live_load_and_allows_retry(pg):  # noqa: F811
    stack(pg)
    pg.add_object("payload", shape="box", size=[0.1, 0.1, 0.1], pos=[7, 0])
    pg.session.paused = True
    result = duplicate(pg, prepare(pg))
    adr = pg.session.model.jnt_qposadr[pg.session.model.joint("payload").id]
    pg.session.data.qpos[adr : adr + 3] = [3, 0, 0.45]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = state(pg)
    with pytest.raises(PlaygroundError, match="supports"):
        undo(pg, result)
    unchanged(pg, before)
    pg.session.data.qpos[adr : adr + 3] = [7, 0, 0.05]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    assert undo(pg, result)["removed"] == result["objects"]


@pytest.mark.parametrize("failure", ["compile", "validate", "load_before", "load_after"])
def test_failures_restore_disk_runtime_history_and_previous_receipts(pg, monkeypatch, failure):  # noqa: F811
    from robotforge.design import store as store_module
    from robotforge.viewer import object_group

    stack(pg)
    previous = duplicate(pg, prepare(pg))
    prepared = prepare(pg)
    receipt, preparation, before = pg._play_edit, pg._play_group, state(pg)

    def fail(*args, **kwargs):
        raise ValueError("injected failure")

    if failure == "compile":
        monkeypatch.setattr(store_module, "compile_world", fail)
    elif failure == "validate":
        monkeypatch.setattr(object_group, "_support_graph", fail)
    else:
        original = pg._load_world

        def load(*args, **kwargs):
            if failure == "load_after":
                original(*args, **kwargs)
            fail()

        monkeypatch.setattr(pg, "_load_world", load)
    with pytest.raises(ValueError, match="injected"):
        duplicate(pg, prepared, pos=[6, 0, 0])
    unchanged(pg, before)
    assert pg._play_edit is receipt and pg._play_group is preparation
    assert receipt.token == previous["undo_edit"]["token"]


@pytest.mark.parametrize("phase", ["during_load", "after_load"])
@pytest.mark.parametrize("external_kind", ["valid", "malformed"])
def test_load_failure_preserves_external_world_and_invalidates_preparation(pg, monkeypatch, phase, external_kind):  # noqa: F811
    stack(pg)
    duplicate(pg, prepare(pg))
    prepared = prepare(pg)
    receipt, before = pg._play_edit, state(pg)
    external_world = pg.store.load()
    external_world.name = "external_writer"
    external = external_world.to_yaml().encode("utf-8") if external_kind == "valid" else b"objects: [broken yaml\n"
    original_load = pg._load_world

    def failing_load(*args, **kwargs):
        if phase == "after_load":
            original_load(*args, **kwargs)
        pg.store.file.write_bytes(external)
        raise RuntimeError("unrelated model installation failure")

    monkeypatch.setattr(pg, "_load_world", failing_load)
    with pytest.raises(RuntimeError, match="unrelated model installation failure"):
        duplicate(pg, prepared, pos=[6, 0, 0])
    after = state(pg)
    assert after[0] == external and after[1:3] == before[1:3]
    assert after[3] is before[3] and after[4] is before[4] and after[5:7] == before[5:7]
    np.testing.assert_array_equal(after[7], before[7])
    np.testing.assert_array_equal(after[8], before[8])
    assert after[9] == before[9] and pg._play_edit is receipt
    assert pg.session.paused and pg._play_group is None
    assert pg._play_world_sha256 is None and pg._play_world_descriptor is None
    assert metadata(pg) == {"objects": []}
    with pytest.raises(PlaygroundError, match="no longer available"):
        duplicate(pg, prepared, pos=[6, 0, 0])
    assert pg.store.file.read_bytes() == external


def test_load_failure_does_not_overwrite_external_build_or_history(pg, monkeypatch):  # noqa: F811
    stack(pg)
    prepared = prepare(pg)
    before = state(pg)
    original_load = pg._load_world
    external_world = before[0] + b"\n# outside writer\n"
    external_xml = b"<mujoco model='outside writer'/>\n"
    external_history = b"name: external_history\n"
    new_record = pg.store.history / "external.yaml"
    old_record = next(iter(pg.store.history.glob("*.yaml")))

    def failing_load(*args, **kwargs):
        original_load(*args, **kwargs)
        pg.store.file.write_bytes(external_world)
        (pg.store.build_dir / "world.xml").write_bytes(external_xml)
        new_record.write_bytes(external_history)
        old_record.write_bytes(external_history)
        raise ValueError("an unrelated load failure")

    monkeypatch.setattr(pg, "_load_world", failing_load)
    with pytest.raises(ValueError, match="unrelated load failure"):
        duplicate(pg, prepared)
    assert pg.store.file.read_bytes() == external_world
    assert (pg.store.build_dir / "world.xml").read_bytes() == external_xml
    expected_history = {**before[2], new_record.name: external_history, old_record.name: external_history}
    assert {path.name: path.read_bytes() for path in pg.store.history.glob("*.yaml")} == expected_history
    assert pg.session.model is before[3] and pg.session.data is before[4]
    assert pg._play_world_sha256 is None and pg._play_group is None


@pytest.mark.parametrize("directory", ["build", "history"])
@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_rollback_never_follows_replaced_artifact_directory(pg, monkeypatch, tmp_path, directory, replacement):  # noqa: F811
    stack(pg)
    prepared = prepare(pg)
    before = state(pg)
    original_load = pg._load_world
    owned_dir = pg.store.build_dir if directory == "build" else pg.store.history
    displaced = tmp_path / "displaced-owned-directory"
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"outside-owned content")
    expected = {}

    def failing_load(*args, **kwargs):
        original_load(*args, **kwargs)
        # Use the exact candidate artifact bytes so content-only checks would wrongly
        # claim and mutate these externally supplied files after the directory swap.
        if directory == "build":
            name = "world.xml"
            contents = (owned_dir / name).read_bytes()
        else:
            name = next(path.name for path in owned_dir.glob("*.yaml") if path.name not in before[2])
            contents = (owned_dir / name).read_bytes()
        (outside / name).write_bytes(contents)
        expected.update({path.name: path.read_bytes() for path in outside.iterdir()})
        owned_dir.rename(displaced)
        if replacement == "symlink":
            owned_dir.symlink_to(outside, target_is_directory=True)
        else:
            outside.rename(owned_dir)
        raise RuntimeError("failed after outside directory replacement")

    monkeypatch.setattr(pg, "_load_world", failing_load)
    with pytest.raises(RuntimeError, match="outside directory replacement"):
        duplicate(pg, prepared)
    external_directory = outside if replacement == "symlink" else owned_dir
    assert {path.name: path.read_bytes() for path in external_directory.iterdir()} == expected
    assert pg.store.file.read_bytes() == before[0]
    assert pg.session.model is before[3] and pg.session.data is before[4]
    assert pg._play_world_sha256 is None and pg._play_group is None


def test_rollback_preserves_external_history_deletion(pg, monkeypatch):  # noqa: F811
    stack(pg)
    prepared = prepare(pg)
    before = state(pg)
    deleted = next(iter(pg.store.history.glob("*.yaml")))
    original_load = pg._load_world

    def failing_load(*args, **kwargs):
        original_load(*args, **kwargs)
        deleted.unlink()
        raise RuntimeError("failed after external history deletion")

    monkeypatch.setattr(pg, "_load_world", failing_load)
    with pytest.raises(RuntimeError, match="external history deletion"):
        duplicate(pg, prepared)
    expected = {name: raw for name, raw in before[2].items() if name != deleted.name}
    assert {path.name: path.read_bytes() for path in pg.store.history.glob("*.yaml")} == expected
    assert not deleted.exists() and pg.store.file.read_bytes() == before[0]
    assert (pg.store.build_dir / "world.xml").read_bytes() == before[1]
    assert pg.session.model is before[3] and pg.session.data is before[4]


@pytest.mark.parametrize("external_delete", [False, True])
def test_rollback_restores_only_actual_store_prunes_at_history_limit(pg, monkeypatch, external_delete):  # noqa: F811
    from robotforge.design.store import HISTORY_LIMIT

    stack(pg)
    prepared = prepare(pg)
    history = pg.store.history
    for index in range(HISTORY_LIMIT - len(list(history.glob("*.yaml")))):
        (history / f"{index:020d}.yaml").write_bytes(pg.store.file.read_bytes())
    before = state(pg)
    oldest = sorted(history.glob("*.yaml"))[0]
    latest = sorted(history.glob("*.yaml"))[-1]
    original_load = pg._load_world

    def failing_load(*args, **kwargs):
        original_load(*args, **kwargs)
        assert not oldest.exists() and len(list(history.glob("*.yaml"))) == HISTORY_LIMIT
        if external_delete:
            latest.unlink()
        raise RuntimeError("failed after actual history prune")

    monkeypatch.setattr(pg, "_load_world", failing_load)
    with pytest.raises(RuntimeError, match="actual history prune"):
        duplicate(pg, prepared)
    expected = {name: raw for name, raw in before[2].items() if not external_delete or name != latest.name}
    assert {path.name: path.read_bytes() for path in history.glob("*.yaml")} == expected
    assert oldest.read_bytes() == before[2][oldest.name]
    assert pg.store.file.read_bytes() == before[0] and (pg.store.build_dir / "world.xml").read_bytes() == before[1]
    assert pg.session.model is before[3] and pg.session.data is before[4]


@pytest.mark.parametrize("change", ["reset", "pose", "velocity", "play", "file"])
def test_postcompile_rechecks_before_any_history_or_publish(pg, monkeypatch, change):  # noqa: F811
    from robotforge.design import store as store_module

    stack(pg)
    prepared = prepare(pg)
    before = state(pg)
    original = store_module.compile_world
    external = before[0] + b"\n# concurrent writer\n"

    def compile_and_change(*args, **kwargs):
        result = original(*args, **kwargs)
        if change == "reset":
            pg.session.reset()
        elif change == "pose":
            pg.session.data.qpos[0] += 0.01
        elif change == "velocity":
            pg.session.data.qvel[0] += 0.01
        elif change == "play":
            pg.session.paused = False
        else:
            pg.store.file.write_bytes(external)
        return result

    monkeypatch.setattr(store_module, "compile_world", compile_and_change)
    with pytest.raises(ValueError):
        duplicate(pg, prepared)
    assert pg.session.model is before[3]
    assert pg.store.file.read_bytes() == (external if change == "file" else before[0])
    assert (pg.store.build_dir / "world.xml").read_bytes() == before[1]
    assert {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")} == before[2]


def test_store_callback_uses_sole_compilation_and_cannot_write_after_failure(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    before = state(pg)
    calls = []
    original = store_module.compile_world

    def counted(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    def refuse(candidate, compiled):
        assert compiled is calls[0]
        assert pg.store.file.read_bytes() == before[0]
        raise ValueError("validation refused")

    monkeypatch.setattr(store_module, "compile_world", counted)
    with pytest.raises(ValueError, match="validation refused"):
        pg.store.apply(lambda world: None, post_compile=refuse)
    assert len(calls) == 1
    unchanged(pg, before)


def test_store_callback_external_write_is_rechecked(pg):  # noqa: F811
    before = state(pg)
    external = before[0] + b"\n# validator concurrent writer\n"

    def interfere(candidate, compiled):
        pg.store.file.write_bytes(external)

    with pytest.raises(ValueError, match="changed during validation"):
        pg.store.apply(lambda world: None, post_compile=interfere)
    assert pg.store.file.read_bytes() == external
    assert (pg.store.build_dir / "world.xml").read_bytes() == before[1]
    assert {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")} == before[2]


def test_sixteen_exact_members_name_collisions_and_independent_editing(pg):  # noqa: F811
    from robotforge.design.world import ObjectEntry

    entries = [
        ObjectEntry(name=f"block{i}", size=[0.1, 0.1, 0.1], pos=[-4 + i * 0.2, 0], fixed=True) for i in range(16)
    ]
    entries.append(ObjectEntry(name="block0_2", pos=[8, 8], fixed=True))
    pg._edit(lambda world: world.objects.extend(entries))
    pg.session.paused = True
    prepared = prepare(pg, [entry.name for entry in entries[:16]])
    result = duplicate(pg, prepared, pos=[3, 0, 0], yaw=90.0)
    assert len(result["objects"]) == 16 and result["objects"][0] == "block0_3"
    chosen = result["objects"][0]
    moved = pg.edit_object_for_play(
        {"action": "edit_object", "name": chosen, "pos": [6, 6, 0], "yaw": 27.0, **guard(pg)}
    )
    assert moved["object"] == chosen
    assert len([obj for obj in pg.store.load().objects if obj.name in result["objects"]]) == 16
    with pytest.raises(PlaygroundError):
        undo(pg, result)


def test_source_task_bindings_are_preserved_and_not_copied(pg):  # noqa: F811
    from robotforge.design.tasks.spec import TaskEntry

    stack(pg)
    pg.add_robot("arm", template="arm", pos=[-6, 0])
    pg._edit(lambda world: world.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="top")))
    pg.session.paused = True
    original_tasks = pg.store.load().tasks
    result = duplicate(pg, prepare(pg))
    assert pg.store.load().tasks == original_tasks
    assert len(pg.store.load().objects) == 4 and "top_2" in result["objects"]


def test_source_roots_on_different_heights_are_rejected(pg):  # noqa: F811
    pg.add_object("platform", shape="box", size=[1, 1, 0.2], pos=[-2, 0], fixed=True)
    pg.add_object("base", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0, 0.3])
    pg.add_object("top", shape="box", size=[0.2, 0.2, 0.2], pos=[-3, 0])
    pg.session.paused = True
    with pytest.raises(PlaygroundError, match="support height"):
        prepare(pg)
