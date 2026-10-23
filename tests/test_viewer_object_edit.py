"""Saved/live transactions for named rigid-prop editing and its one-use inverse."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401

from robotforge.playground import PlaygroundError
from robotforge.viewer.object_edit import metadata


def request(pg, name="block", **patch):  # noqa: F811
    info = metadata(pg)
    return {
        "action": "edit_object",
        "name": name,
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": info["world_sha256"],
        "pos": [2.0, 0.0, 0.0],
        "yaw": 90,
        **patch,
    }


def remove(pg, name="block", **patch):  # noqa: F811
    msg = request(pg, name, **patch)
    msg.pop("pos")
    msg.pop("yaw")
    msg["action"] = "remove_object"
    return pg.edit_object_for_play(msg)


def undo(pg, result):  # noqa: F811
    return pg.edit_object_for_play({"action": "undo_object_edit", **result["undo_edit"]})


@pytest.mark.parametrize(
    "shape,size",
    [
        ("sphere", [0.18]),
        ("box", [0.2, 0.3, 0.4]),
        ("cylinder", [0.1, 0.24]),
        ("table", [0.8, 0.5, 0.5]),
        ("bin", [0.3, 0.2, 0.2]),
    ],
)
def test_move_preserves_identity_properties_and_reset_location(pg, shape, size):  # noqa: F811
    pg.add_object(
        "block",
        shape=shape,
        size=size,
        pos=[0, 0],
        **({"material": "wood"} if shape in {"box", "sphere", "cylinder"} else {}),
    )
    old = pg.store.load().objects[0]
    result = pg.edit_object_for_play(request(pg))
    new = pg.store.load().objects[0]
    assert len(pg.store.load().objects) == 1 and new.name == old.name
    assert new.model_dump(exclude={"pos", "yaw"}) == old.model_dump(exclude={"pos", "yaw"})
    assert new.pos == [2, 0] and new.yaw == 90 and result["object"] == "block"
    pg.session.reset()
    ids = metadata(pg)["objects"][0]["geoms"]
    assert all(pg.session.data.geom_xpos[g][0] == pytest.approx(2, abs=0.5) for g in ids)


@pytest.mark.parametrize(
    "patch",
    [
        {"generation": None},
        {"generation": True},
        {"version": "2"},
        {"yaw": True},
        {"yaw": float("nan")},
        {"pos": [float("inf"), 0]},
        {"pos": [101, 0]},
        {"pos": [0, True]},
        {"world_sha256": "x"},
        {"mass": 40},
        {"file": "x"},
        {"shape": "mesh"},
        {"name": "blo"},
    ],
)
def test_invalid_or_inexact_requests_are_atomic(pg, patch):  # noqa: F811
    pg.add_object("block", shape="box")
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError):
        pg.edit_object_for_play(request(pg, **patch))
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_undo_move_restores_saved_descriptor_and_actual_old_free_pose(pg):  # noqa: F811
    pg.add_object("block", shape="box", pos=[0, 0])
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    with pg.session.lock:
        joint = pg.session.model.joint("block")
        adr = int(joint.qposadr[0])
        pg.session.data.qpos[adr : adr + 3] = [0.7, 0.5, 0.3]
        pg.session.data.qpos[adr + 3 : adr + 7] = [0.9238795325, 0.3826834324, 0, 0]
        mujoco.mj_forward(pg.session.model, pg.session.data)
        before = pg.session.data.qpos.copy()
        saved = pg.store.load().model_dump()
    result = pg.edit_object_for_play(request(pg))
    assert not np.allclose(pg.session.data.qpos[adr : adr + 3], before[adr : adr + 3])
    undo(pg, result)
    assert pg.store.load().model_dump() == saved
    np.testing.assert_allclose(pg.session.data.qpos, before, atol=1e-12)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


def test_remove_exact_object_and_undo_keep_other_pose_and_original_order(pg):  # noqa: F811
    pg.add_object("block", shape="box")
    pg.add_object("block2", shape="sphere", pos=[1, 0])
    pg.session.data.qpos[7] = 1.4
    mujoco.mj_forward(pg.session.model, pg.session.data)
    result = remove(pg)
    assert [o.name for o in pg.store.load().objects] == ["block2"]
    assert pg.session.data.qpos[0] == 1.4
    undo(pg, result)
    assert [o.name for o in pg.store.load().objects] == ["block", "block2"]
    assert pg.session.data.qpos[7] == 1.4


@pytest.mark.parametrize("operation", ["move", "remove"])
@pytest.mark.parametrize("support", ["block", "table", "arm"])
def test_loaded_support_is_refused_saved_and_live(pg, operation, support):  # noqa: F811
    pg.add_object("block", shape="box", size=[0.3, 0.3, 0.2], fixed=True, pos=[0, 0])
    if support == "arm":
        pg.add_robot("arm", template="arm", pos=[0, 0, 0.2])
    else:
        pg.add_object("top", shape="box", size=[0.15, 0.15, 0.15], fixed=support == "table", pos=[0, 0, 0.275])
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="supports"):
        remove(pg) if operation == "remove" else pg.edit_object_for_play(request(pg))
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_saved_live_occupancy_and_retry_do_not_restore_target_old_pose(pg):  # noqa: F811
    pg.add_object("block", shape="box", pos=[0, 0])
    pg.add_object("other", shape="box", pos=[2, 0])
    with pytest.raises(PlaygroundError, match="occupied"):
        pg.edit_object_for_play(request(pg))
    pg.session.data.qpos[7] = 3
    mujoco.mj_forward(pg.session.model, pg.session.data)
    with pytest.raises(PlaygroundError, match="occupied"):
        pg.edit_object_for_play(request(pg, pos=[3, 0]))
    result = pg.edit_object_for_play(request(pg, pos=[4, 0]))
    assert result["object"] == "block" and pg.session.data.qpos[0] == 4 and pg.session.data.qpos[7] == 3


def test_stale_zero_time_reset_and_disk_edits_are_refused(pg):  # noqa: F811
    pg.add_object("block", shape="box")
    msg = request(pg)
    pg.session.reset()
    with pytest.raises(PlaygroundError, match="changed"):
        pg.edit_object_for_play(msg)
    msg = request(pg)
    pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# external\n")
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="changed"):
        pg.edit_object_for_play(msg)
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_metadata_maps_static_and_dynamic_geoms_without_guessing_presets(pg):  # noqa: F811
    pg.add_object("first_wall", shape="box", fixed=True)
    pg.add_object("ball", shape="sphere", pos=[2, 0])
    pg.add_object("door", shape="door", pos=[4, 0])
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    items = metadata(pg)["objects"]
    assert [e["name"] for e in items] == ["first_wall", "ball"]
    assert items[0]["geoms"] and items[1]["geoms"]
    assert all(pg.session.model.geom_bodyid[g] == 0 for g in items[0]["geoms"])


def test_fixed_bin_uses_all_walls_for_selection_collision_and_inverse(pg):  # noqa: F811
    pg.add_object("bin", shape="bin", size=[0.3, 0.2, 0.2], fixed=True, pos=[-2, 0])
    ids = metadata(pg)["objects"][0]["geoms"]
    assert {pg.session.model.geom(g).name for g in ids} >= {"bin", "bin_wall1", "bin_wall2", "bin_wall3", "bin_wall4"}
    result = pg.edit_object_for_play(request(pg, "bin", pos=[0, 0], yaw=0))
    undo(pg, result)
    pg.add_object("edge", shape="box", size=[0.04, 0.04, 0.04], pos=[0.1475, 0, 0.08])
    with pytest.raises(PlaygroundError, match="occupied"):
        pg.edit_object_for_play(request(pg, "bin", pos=[0, 0], yaw=0))
    remove(pg, "bin")


def test_fixed_bin_rim_support_is_not_treated_as_a_different_object(pg):  # noqa: F811
    pg.add_object("bin", shape="bin", size=[0.3, 0.2, 0.2], fixed=True, pos=[0, 0])
    pg.add_object("rim", shape="box", size=[0.04, 0.04, 0.04], fixed=True, pos=[0.1475, 0, 0.22])
    with pytest.raises(PlaygroundError, match="supports rim"):
        remove(pg, "bin")


def test_tilted_live_support_contact_below_aabb_top_is_protected(pg):  # noqa: F811
    pg.add_object("block", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0])
    pg.add_object("top", shape="box", size=[0.03, 0.03, 0.03], pos=[2, 0])
    d = pg.session.data
    d.qpos[:7] = [0, 0, 0.2, np.cos(np.pi / 12), 0, np.sin(np.pi / 12), 0]
    d.qpos[7:10] = [0.10196, 0, 0.2803]
    mujoco.mj_forward(pg.session.model, d)
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="supports top"):
        remove(pg)
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_external_disk_change_hides_editor_until_world_is_reloaded(pg):  # noqa: F811
    pg.add_object("block", shape="box")
    pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# external\n")
    assert metadata(pg) == {"objects": []}
    pg.open_world(str(pg.store.dir))
    assert metadata(pg)["objects"][0]["name"] == "block"


@pytest.mark.parametrize("phase", ["preflight", "commit"])
def test_external_write_during_compile_never_gets_overwritten(pg, monkeypatch, phase):  # noqa: F811
    from robotforge.design import store as store_module
    from robotforge.design import world as world_module

    pg.add_object("block", shape="box", pos=[0, 0])
    msg = request(pg)
    module = world_module if phase == "preflight" else store_module
    original = module.compile_world
    external = pg.store.file.read_bytes() + b"\n# competing write\n"

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        pg.store.file.write_bytes(external)
        return result

    monkeypatch.setattr(module, "compile_world", changed)
    model = pg.session.model
    with pytest.raises((PlaygroundError, ValueError)):
        pg.edit_object_for_play(msg)
    assert pg.store.file.read_bytes() == external and pg.session.model is model


def test_task_dependency_and_execution_admission_refuse_edit(pg):  # noqa: F811
    from robotforge.design.tasks.spec import TaskEntry
    from robotforge.sim.execution import ExecutionError, RunLimits

    pg.add_object("block", shape="box", pos=[0, 0])
    msg = request(pg)
    with pg.session.execution.run("test", RunLimits()), pytest.raises(ExecutionError):
        pg.edit_object_for_play(msg)
    pg.add_robot("arm", template="arm", pos=[-2, 0])
    pg._edit(lambda w: w.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="block")))
    with pytest.raises(PlaygroundError, match="used by task"):
        remove(pg)


@pytest.mark.parametrize("entrypoint", ["cli", "open_world"])
def test_world_compiled_from_old_descriptor_cannot_issue_edit_tokens(pg, monkeypatch, entrypoint):  # noqa: F811
    from robotforge.cli import load_playground
    from robotforge.design.store import WorldStore
    from robotforge.design.world import World

    pg.add_object("block", shape="box", pos=[0, 0])
    store = pg.store
    original = WorldStore.compile

    def changed(self):
        compiled = original(self)
        w = self.load()
        w.objects[0].pos = [4, 0]
        self.file.write_text(w.to_yaml(), encoding="utf-8")
        return compiled

    monkeypatch.setattr(WorldStore, "compile", changed)
    if entrypoint == "cli":
        opened = load_playground(store.dir)
    else:
        pg.open_world(str(store.dir))
        opened = pg
    assert metadata(opened) == {"objects": []}
    import hashlib

    msg = {
        "action": "edit_object",
        "name": "block",
        "pos": [2, 0],
        "yaw": 0,
        "version": opened.session.version,
        "generation": opened.session.generation,
        "world_sha256": hashlib.sha256(store.file.read_bytes()).hexdigest(),
    }
    before, model = store.file.read_bytes(), opened.session.model
    with pytest.raises(PlaygroundError, match="changed"):
        opened.edit_object_for_play(msg)
    assert store.file.read_bytes() == before and opened.session.model is model
    assert World.from_yaml(before.decode("utf-8")).objects[0].pos == [4, 0]


def test_undo_blocked_by_new_live_occupant_is_retryable(pg):  # noqa: F811
    pg.add_object("block", shape="box", pos=[0, 0])
    pg.add_object("other", shape="box", pos=[2, 0])
    result = pg.edit_object_for_play(request(pg, pos=[4, 0]))
    pg.session.data.qpos[7] = 0
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="occupied"):
        undo(pg, result)
    assert pg.store.file.read_bytes() == before and pg.session.model is model
    pg.session.data.qpos[7] = 2
    mujoco.mj_forward(pg.session.model, pg.session.data)
    undo(pg, result)
    assert pg.store.load().objects[0].pos == [0, 0]


@pytest.mark.parametrize("reference", ["sort_key", "start_object"])
def test_secondary_saved_task_references_prevent_dangling_objects(pg, reference):  # noqa: F811
    from robotforge.design.tasks.spec import Start, TaskEntry

    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_object("block", shape="box", pos=[0, 0])
    pg.add_object("bin", shape="bin", pos=[2, 0])
    if reference == "sort_key":
        task = TaskEntry(name="task", kind="sort", robot="rover", into={"block": "bin"})
    else:
        task = TaskEntry(
            name="task", kind="go_to", robot="rover", target="bin", start=Start(objects={"block": [0, 0, 0.025]})
        )
    pg._edit(lambda w: w.tasks.append(task))
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="used by task"):
        remove(pg)
    assert pg.store.file.read_bytes() == before and pg.session.model is model
