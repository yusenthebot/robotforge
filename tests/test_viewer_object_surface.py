"""Guarded finish edits carry current poses, reset motion explicitly, and roll back atomically."""

from contextlib import contextmanager

import mujoco
import numpy as np
import pytest
from test_prop_materials import materials
from test_viewer_creation import pg  # noqa: F401
from test_viewer_terrain_create import snapshot, unchanged
from test_visuals import assert_same_physics

from robotforge.design import layouts, menagerie
from robotforge.design.tasks.spec import TaskEntry
from robotforge.playground import PlaygroundError
from robotforge.sim.execution import ExecutionError, RunLimits
from robotforge.viewer import object_surface
from robotforge.viewer.object_edit import metadata


def apply(pg, finish="wood", name="block", **extra):  # noqa: F811
    return pg.edit_object_for_play(
        {
            "action": "set_object_surface",
            "name": name,
            "surface_material": finish,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **extra,
        }
    )


def undo(pg, result):  # noqa: F811
    return pg.edit_object_for_play({"action": "undo_object_edit", **result["undo_edit"]})


def test_apply_and_undo_keep_latest_live_poses_but_reset_motion(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("block", material="steel", size=[0.2, 0.2, 0.2])
    saved = pg.store.load()
    j = pg.session.model.joint("block")
    adr = int(j.qposadr[0])
    pg.session.data.qpos[adr : adr + 3] = [0.9, 0.6, 0.4]
    pg.session.data.qvel[:] = 0.2
    pg.session.data.ctrl[:] = 0.3
    pg.session.data.time = 1.2
    mujoco.mj_forward(pg.session.model, pg.session.data)
    pose, controls = pg.session.data.qpos.copy(), pg.session.data.ctrl.copy()
    result = apply(pg)
    assert result["paused"] and pg.session.paused and pg.session.data.time == 0
    np.testing.assert_allclose(pg.session.data.qpos, pose, atol=1e-12)
    np.testing.assert_array_equal(pg.session.data.ctrl, controls)
    assert not pg.session.data.qvel.any()
    entry = pg.store.load().objects[0]
    assert entry.model_dump(exclude={"surface_material"}) == saved.objects[0].model_dump()
    assert entry.material == "steel" and entry.surface_material == "wood"
    pg.session.data.qpos[adr] += 0.5
    latest = pg.session.data.qpos.copy()
    undo(pg, result)
    np.testing.assert_allclose(pg.session.data.qpos, latest, atol=1e-12)
    assert pg.store.load() == saved and pg._play_edit is None
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


def test_original_reopen_and_duplicate_preserve_authored_override(pg):  # noqa: F811
    pg.add_object("block", material="steel")
    apply(pg)
    entry = metadata(pg)["objects"][0]
    assert entry["surface_material"] == "wood"
    pg.edit_object_for_play(
        {
            "action": "duplicate_object",
            "name": "block",
            "pos": [2, 0],
            "yaw": 0,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
        }
    )
    assert [o.surface_material for o in pg.store.load().objects] == ["wood", "wood"]
    directory = str(pg.store.dir)
    pg.open_world(directory)
    assert pg.store.load().objects[0].surface_material == "wood"
    result = apply(pg, None)
    assert pg.store.load().objects[0].surface_material is None
    assert pg.store.load().objects[0].material == "steel"
    undo(pg, result)
    assert pg.store.load().objects[0].surface_material == "wood"


def test_python_update_can_explicitly_clear_finish_without_changing_physical_material(pg):  # noqa: F811
    pg.add_object("block", material="steel")
    pg.update_object("block", surface_material="wood")
    assert pg.store.load().objects[0].surface_material == "wood"
    pg.update_object("block", surface_material=None)
    assert pg.store.load().objects[0].surface_material is None
    assert pg.store.load().objects[0].material == "steel"
    with pytest.raises(PlaygroundError, match="nothing to change"):
        pg.update_object("block", mass=None)


def test_supporting_and_task_bound_primitive_can_change_surface(pg):  # noqa: F811
    pg.add_object("block", fixed=True, size=[1, 1, 0.2], pos=[0, 0])
    pg.add_object("ball", shape="sphere", size=[0.1], pos=[0, 0, 0.25])
    pg.add_robot("arm", template="arm", pos=[-2, 0])
    pg._edit(lambda w: w.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="ball")))
    apply(pg, "steel")
    apply(pg, "wood", name="ball")
    assert [o.surface_material for o in pg.store.load().objects] == ["steel", "wood"]
    assert pg.store.load().tasks[0].object == "ball"


def test_noop_keeps_history_identity_runtime_and_prior_undo(pg):  # noqa: F811
    pg.add_object("block")
    result = apply(pg)
    receipt = pg._play_edit
    before = snapshot(pg)
    history = list(pg.store.history.glob("*.yaml"))
    assert apply(pg)["unchanged"]
    unchanged(pg, before)
    assert pg._play_edit is receipt and list(pg.store.history.glob("*.yaml")) == history
    undo(pg, result)


@pytest.mark.parametrize(
    "shape,fixed,yaw",
    [("table", True, 0), ("table", False, 37), ("bin", False, 37), ("bin", True, 0), ("bin", True, 37)],
)
def test_furniture_finish_noop_original_inverse_and_reopen_preserve_authored_physics(
    pg,  # noqa: F811
    monkeypatch,
    shape,
    fixed,
    yaw,
):
    pg.add_object("block", shape=shape, fixed=fixed, yaw=yaw, color="#5689ab", mass=0.731)
    original, original_model = pg.store.load(), pg.session.model
    original_materials = materials(original_model)
    assert metadata(pg)["objects"][0]["surface_material"] is None
    if pg.session.model.nq:
        adr = int(pg.session.model.joint("block").qposadr[0])
        pg.session.data.qpos[adr] += 0.2
        mujoco.mj_forward(pg.session.model, pg.session.data)
    pose = pg.session.data.qpos.copy()
    pg.session.data.qvel[:] = 0.3
    pg.session.data.time = 2.0
    applied = apply(pg, "steel")
    np.testing.assert_allclose(pg.session.data.qpos, pose, atol=1e-12, rtol=0)
    assert not pg.session.data.qvel.any() and pg.session.data.time == 0 and pg.session.paused
    assert_same_physics(original_model, pg.session.model, steps=0)
    assert set(materials(pg.session.model)) == ({"rf_metal", "rf_rubber"} if shape == "table" else {"rf_metal"})
    entry = pg.store.load().objects[0]
    assert entry.material is None and entry.surface_material == "steel"
    assert entry.model_dump(exclude={"surface_material"}) == original.objects[0].model_dump()
    assert metadata(pg)["objects"][0]["surface_material"] == "steel"
    receipt, before = pg._play_edit, snapshot(pg)

    def forbidden(*args, **kwargs):
        pytest.fail("an unchanged furniture finish must not compile or load")

    with monkeypatch.context() as context:
        context.setattr(object_surface, "compile_world", forbidden)
        context.setattr(pg, "_load_world", forbidden)
        assert apply(pg, "steel")["unchanged"]
    unchanged(pg, before)
    assert pg._play_edit is receipt
    undo(pg, applied)
    assert pg.store.load() == original and materials(pg.session.model) == original_materials
    apply(pg, "wood")
    cleared = apply(pg, None)
    assert pg.store.load() == original and materials(pg.session.model) == original_materials
    undo(pg, cleared)
    directory = str(pg.store.dir)
    pg.open_world(directory)
    assert pg.store.load().objects[0].surface_material == "wood"
    assert metadata(pg)["objects"][0]["surface_material"] == "wood"


@pytest.mark.parametrize(
    "patch",
    [
        {"surface_material": "gold"},
        {"surface_material": True},
        {"surface_material": 1},
        {"mass": 3},
        {"material": "steel"},
        {"pos": [2, 0]},
        {"version": True},
        {"generation": -1},
        {"world_sha256": "bad"},
        {"name": "bloc"},
    ],
)
def test_bad_requests_leave_disk_and_runtime_unchanged(pg, patch):  # noqa: F811
    pg.add_object("block")
    before = snapshot(pg)
    with pytest.raises(PlaygroundError):
        apply(pg, **patch)
    unchanged(pg, before)


def test_finish_is_required_and_unsupported_shape_is_not_offered(pg):  # noqa: F811
    pg.add_object("desk", shape="drawer")
    assert metadata(pg)["objects"] == []
    with pytest.raises(PlaygroundError, match="existing box"):
        apply(pg, name="desk")
    with pytest.raises(PlaygroundError, match="invalid prop surface"):
        pg.edit_object_for_play(
            {
                "action": "set_object_surface",
                "name": "desk",
                "version": pg.session.version,
                "generation": pg.session.generation,
                "world_sha256": pg._play_world_sha256,
            }
        )


@pytest.mark.parametrize("changed", ["reset", "disk", "lesson", "hooks", "execution"])
def test_runtime_and_stale_guards_refuse_before_publication(pg, changed):  # noqa: F811
    pg.add_object("block")
    msg = {
        "action": "set_object_surface",
        "name": "block",
        "surface_material": "wood",
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": pg._play_world_sha256,
    }
    if changed == "reset":
        pg.session.reset()
    if changed == "disk":
        pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# external\n")
    if changed == "lesson":
        pg.lessons.lesson = "test"
    if changed == "hooks":
        pg.session.reset_hooks.append(lambda: None)
    before = snapshot(pg)
    if changed == "execution":
        with pg.session.execution.run("test", RunLimits()), pytest.raises(ExecutionError):
            pg.edit_object_for_play(msg)
    else:
        with pytest.raises((PlaygroundError, layouts.LayoutError)):
            pg.edit_object_for_play(msg)
    unchanged(pg, before)
    if changed == "lesson":
        pg.lessons.lesson = None


@pytest.mark.parametrize("fail", ["compile", "load", "publish"])
def test_failures_restore_files_exact_runtime_and_receipts(pg, monkeypatch, fail):  # noqa: F811
    pg.add_object("block")
    apply(pg)
    receipt = pg._play_edit
    before = snapshot(pg)
    files = {p: p.read_bytes() for p in [*pg.store.history.glob("*.yaml"), pg.store.build_dir / "world.xml"]}
    if fail == "compile":

        def broken(*args):
            menagerie.require_downloads_allowed("uncached")

        monkeypatch.setattr(object_surface, "compile_world", broken)
        error = menagerie.MenagerieError
    elif fail == "load":
        original = pg._load_world

        def broken(*args):
            original(*args)
            raise OSError("injected load failure")

        monkeypatch.setattr(pg, "_load_world", broken)
        error = OSError
    else:

        def broken(*args):
            raise OSError("injected publish failure")

        monkeypatch.setattr(object_surface, "SurfaceEdit", broken)
        error = OSError
    with pytest.raises(error):
        apply(pg, "steel")
    unchanged(pg, before)
    assert pg._play_edit is receipt and all(p.read_bytes() == b for p, b in files.items())
    assert set(pg.store.history.glob("*.yaml")) == {p for p in files if p.parent == pg.store.history}


def test_after_prepare_guard_uses_latest_pose_and_rejects_external_change(pg, monkeypatch):  # noqa: F811
    pg.add_object("block")
    original = object_surface.compile_world

    def moved(*args):
        result = original(*args)
        pg.session.data.qpos[0] = 0.75
        return result

    monkeypatch.setattr(object_surface, "compile_world", moved)
    apply(pg)
    assert pg.session.data.qpos[0] == 0.75
    external = pg.store.file.read_bytes() + b"\n# external\n"

    def changed(*args):
        result = original(*args)
        pg.store.file.write_bytes(external)
        return result

    monkeypatch.setattr(object_surface, "compile_world", changed)
    model = pg.session.model
    with pytest.raises(PlaygroundError):
        apply(pg, "steel")
    assert pg.store.file.read_bytes() == external and pg.session.model is model


def test_other_edit_invalidates_surface_undo_and_cleanup_warning_is_returned(pg, monkeypatch):  # noqa: F811
    pg.add_object("block")
    original = layouts._install_world

    @contextmanager
    def warning(*args, **kwargs):
        with original(*args, **kwargs) as notes:
            yield notes
        notes.append("cleanup warning")

    monkeypatch.setattr(layouts, "_install_world", warning)
    result = apply(pg)
    assert "cleanup warning" in result["notes"]
    pg.add_object("other", pos=[2, 0])
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


@pytest.mark.parametrize("changed", ["reset", "hooks", "lesson"])
def test_second_guard_refuses_runtime_change_during_preparation(pg, monkeypatch, changed):  # noqa: F811
    pg.add_object("block")
    original = object_surface.compile_world
    before = pg.store.file.read_bytes()
    model = pg.session.model

    def altered(*args):
        result = original(*args)
        if changed == "reset":
            pg.session.reset()
        if changed == "hooks":
            pg.session.reset_hooks.append(lambda: None)
        if changed == "lesson":
            pg.lessons.lesson = "test"
        return result

    monkeypatch.setattr(object_surface, "compile_world", altered)
    with pytest.raises((PlaygroundError, layouts.LayoutError)):
        apply(pg)
    assert pg.store.file.read_bytes() == before and pg.session.model is model
    assert pg.store.load().objects[0].surface_material is None
    if changed == "lesson":
        pg.lessons.lesson = None


def test_failed_undo_keeps_receipt_retryable(pg, monkeypatch):  # noqa: F811
    pg.add_object("block")
    result = apply(pg)
    receipt, before = pg._play_edit, snapshot(pg)
    original = pg._load_world

    def broken(*args):
        original(*args)
        raise OSError("injected undo failure")

    with monkeypatch.context() as m:
        m.setattr(pg, "_load_world", broken)
        with pytest.raises(OSError, match="undo failure"):
            undo(pg, result)
    unchanged(pg, before)
    assert pg._play_edit is receipt
    undo(pg, result)
    assert pg.store.load().objects[0].surface_material is None


def test_unrecoverable_install_pauses_and_invalidates_authoring(pg, monkeypatch):  # noqa: F811
    pg.add_object("block")
    original = layouts._install_world

    @contextmanager
    def damaged(*args, **kwargs):
        with original(*args, **kwargs) as notes:
            yield notes
        raise layouts.LayoutError("recovery", "injected recovery failure")

    monkeypatch.setattr(layouts, "_install_world", damaged)
    with pytest.raises(PlaygroundError, match="rollback failed"):
        apply(pg)
    assert pg.session.paused and pg._play_world_sha256 is None
    assert metadata(pg) == {"objects": []}
