"""Server-owned prop copying, guarded inverse edits and saved/live collision boundaries."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401

from robotforge.playground import PlaygroundError
from robotforge.viewer.object_edit import _owners, metadata


def request(pg, name="source", **patch):  # noqa: F811
    info = metadata(pg)
    return {
        "action": "duplicate_object",
        "name": name,
        "pos": [2.0, 0.0, 0.0],
        "yaw": 107.0,
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": info["world_sha256"],
        **patch,
    }


def duplicate(pg, name="source", **patch):  # noqa: F811
    return pg.edit_object_for_play(request(pg, name, **patch))


def undo(pg, result):  # noqa: F811
    return pg.edit_object_for_play({"action": "undo_object_edit", **result["undo_edit"]})


def state(pg):  # noqa: F811
    return (
        pg.store.file.read_bytes(),
        (pg.store.build_dir / "world.xml").read_bytes(),
        {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")},
        pg.session.model,
        pg.session.data,
        pg.session.version,
        pg.session.generation,
        pg.session.data.qpos.copy(),
        pg.session.data.qvel.copy(),
        pg.session.data.time,
    )


def unchanged(pg, before):  # noqa: F811
    after = state(pg)
    assert before[:3] == after[:3]
    assert before[3] is after[3] and before[4] is after[4] and before[5:7] == after[5:7]
    np.testing.assert_array_equal(before[7], after[7])
    np.testing.assert_array_equal(before[8], after[8])
    assert before[9] == after[9]


@pytest.mark.parametrize(
    "shape,size",
    [
        ("sphere", [0.22]),
        ("box", [0.42, 0.23, 0.31]),
        ("cylinder", [0.22, 0.31]),
        ("table", [0.9, 0.65, 0.55]),
        ("bin", [0.45, 0.32, 0.26]),
    ],
)
@pytest.mark.parametrize("fixed", [False, True])
def test_complete_authored_settings_and_other_live_poses_are_preserved(pg, shape, size, fixed):  # noqa: F811
    fields = {"material": "steel"} if shape in {"sphere", "box", "cylinder"} else {}
    pg.add_object(
        "source", shape=shape, size=size, color=[0.2, 0.4, 0.6, 1], mass=1.7, fixed=fixed, pos=[-2, 0], yaw=17, **fields
    )
    source = pg.store.load().objects[0]
    before = pg.session.data.qpos.copy()
    if len(before):
        pg.session.data.qpos[0] -= 0.2
        pg.session.data.qvel[0] = 0.4
        mujoco.mj_forward(pg.session.model, pg.session.data)
        before = pg.session.data.qpos.copy()
    pg.session.data.time = 3.0
    result = duplicate(pg)
    original, clone = pg.store.load().objects
    assert original == source and clone.name == result["object"] == "source_2"
    assert clone.model_dump(exclude={"name", "pos", "yaw"}) == source.model_dump(exclude={"name", "pos", "yaw"})
    assert clone.pos == [2, 0] and clone.yaw == 107
    np.testing.assert_array_equal(pg.session.data.qpos[: len(before)], before)
    assert pg.session.data.time == 0 and not np.any(pg.session.data.qvel)
    owners = _owners(pg.session.model, pg.store.load())
    friction = {
        name: sorted(tuple(pg.session.model.geom_friction[g]) for g, owner in owners.items() if owner == name)
        for name in [source.name, clone.name]
    }
    assert friction[source.name] == friction[clone.name]
    pg.session.reset()
    undo_token = result["undo_edit"]
    with pytest.raises(PlaygroundError, match="changed|no longer"):
        pg.edit_object_for_play({"action": "undo_object_edit", **undo_token})
    assert pg.store.load().objects[0] == source and pg.store.load().objects[1] == clone


def test_default_mass_copy_of_copy_and_one_use_undo_remove_only_latest(pg):  # noqa: F811
    pg.add_object("source", shape="box", material="wood", pos=[-2, 0], yaw=12)
    source = pg.store.load().objects[0]
    duplicate(pg)
    result = duplicate(pg, "source_2", pos=[4, 0, 0])
    assert result["object"] == "source_2_2"
    out = undo(pg, result)
    assert out["removed"] == "source_2_2" and [o.name for o in pg.store.load().objects] == ["source", "source_2"]
    assert pg.store.load().objects[0] == source
    assert all(o.mass is None and o.material == "wood" for o in pg.store.load().objects)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)


def test_copy_is_upright_authored_shape_not_a_snapshot_of_live_tilt_velocity_or_time(pg):  # noqa: F811
    pg.add_object("source", shape="box", size=[0.2, 0.3, 0.2], color=[0.2, 0.4, 0.6, 0.4], pos=[-2, 0], yaw=37)
    pg.session.data.qpos[:7] = [-2.3, 0.2, 0.4, np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 4
    mujoco.mj_forward(pg.session.model, pg.session.data)
    old_pose = pg.session.data.qpos.copy()
    duplicate(pg, pos=[2, 0, 0.5], yaw=37)
    np.testing.assert_allclose(pg.session.data.qpos[:7], old_pose, atol=1e-12)
    expected = [2, 0, 0.6, np.cos(np.deg2rad(37) / 2), 0, 0, np.sin(np.deg2rad(37) / 2)]
    np.testing.assert_allclose(pg.session.data.qpos[7:], expected, atol=1e-12)
    owners = _owners(pg.session.model, pg.store.load())
    alphas = {
        name: sorted(float(pg.session.model.geom_rgba[g, 3]) for g, owner in owners.items() if owner == name)
        for name in ["source", "source_2"]
    }
    assert alphas["source"] == alphas["source_2"] and any(v == pytest.approx(0.4) for v in alphas["source"])
    assert not np.any(pg.session.data.qvel) and pg.session.data.time == 0


def test_maximum_length_names_check_all_entity_names_and_remain_valid(pg):  # noqa: F811
    name = "a" * 32
    pg.add_object(name, shape="box", fixed=True)
    pg.add_robot("a" * 30 + "_2", template="rover", pos=[8, 0])
    pg.add_terrain("ramp", name="a" * 30 + "_3", pos=[10, 10])
    result = duplicate(pg, name)
    assert result["object"] == "a" * 30 + "_4"
    copy = pg.store.load().objects[-1]
    assert len(copy.name) == 32
    result = duplicate(pg, copy.name, pos=[4, 0])
    assert result["object"] == "a" * 30 + "_5"


def test_loaded_support_and_task_bound_source_can_be_copied_without_moving_dependents(pg):  # noqa: F811
    from robotforge.design.tasks.spec import TaskEntry

    pg.add_object("source", shape="box", size=[0.4, 0.4, 0.2], fixed=True, pos=[0, 0])
    pg.add_object("payload", shape="box", size=[0.1, 0.1, 0.1], pos=[0, 0, 0.25])
    pg.add_robot("arm", template="arm", pos=[-2, 0])
    pg._edit(lambda w: w.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="source")))
    before = pg.store.load()
    poses = pg.session.data.qpos.copy()
    duplicate(pg)
    world = pg.store.load()
    assert world.objects[:2] == before.objects and world.tasks == before.tasks
    np.testing.assert_array_equal(pg.session.data.qpos, poses)


@pytest.mark.parametrize(
    "shape,size,height",
    [
        ("box", [0.2, 0.2, 0.2], 0.6),
        ("table", [0.5, 0.4, 0.3], 0.5),
        ("bin", [0.3, 0.2, 0.2], 0.5),
    ],
)
def test_surface_placement_uses_primitive_center_or_table_bin_base(pg, shape, size, height):  # noqa: F811
    pg.add_object("source", shape=shape, size=size, fixed=True, pos=[-2, 0])
    pg.add_object("support", shape="table", size=[0.9, 0.8, 0.5], fixed=True, pos=[2, 0])
    duplicate(pg, pos=[2, 0, 0.5], yaw=0)
    assert pg.store.load().objects[-1].pos == [2, 0, height]


def test_source_remains_an_available_stacking_surface(pg):  # noqa: F811
    pg.add_object("source", shape="box", size=[0.3, 0.3, 0.2], fixed=True, pos=[0, 0])
    duplicate(pg, pos=[0, 0, 0.2], yaw=0)
    assert pg.store.load().objects[-1].pos == pytest.approx([0, 0, 0.3], abs=1e-12)


@pytest.mark.parametrize("case", ["source", "saved", "live", "bin_wall"])
def test_occupied_saved_and_live_spots_fail_without_changing_any_active_state(pg, case):  # noqa: F811
    pg.add_object(
        "source",
        shape="bin" if case == "bin_wall" else "box",
        pos=[-2, 0],
        fixed=True,
        size=[0.3, 0.2, 0.2] if case == "bin_wall" else [0.2, 0.2, 0.2],
    )
    pg.add_object(
        "other", shape="box", size=[0.04, 0.04, 0.04], pos=[0.1475, 0, 0.08] if case == "bin_wall" else [2, 0]
    )
    pos = [-2, 0] if case == "source" else [2, 0]
    if case == "bin_wall":
        pos = [0, 0]
    if case == "live":
        pg.session.data.qpos[:3] = [3, 0, 0.1]
        mujoco.mj_forward(pg.session.model, pg.session.data)
        pos = [3, 0]
    before = state(pg)
    with pytest.raises(PlaygroundError, match="occupied"):
        duplicate(pg, pos=pos, yaw=0)
    unchanged(pg, before)


def test_undo_copy_refuses_live_load_and_can_retry_after_load_moves(pg):  # noqa: F811
    pg.add_object("source", shape="box", size=[0.3, 0.3, 0.2], fixed=True, pos=[-2, 0])
    pg.add_object("payload", shape="box", size=[0.1, 0.1, 0.1], pos=[4, 0])
    result = duplicate(pg, yaw=0)
    pg.session.data.qpos[:3] = [2, 0, 0.25]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = state(pg)
    with pytest.raises(PlaygroundError, match="supports"):
        undo(pg, result)
    unchanged(pg, before)
    pg.session.data.qpos[:3] = [4, 0, 0.05]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    assert undo(pg, result)["removed"] == "source_2"


@pytest.mark.parametrize(
    "patch",
    [
        {"mass": 4},
        {"material": "steel"},
        {"size": [2, 2, 2]},
        {"color": "red"},
        {"file": "x"},
        {"name": "missing"},
        {"name": "sour"},
        {"generation": True},
        {"yaw": True},
        {"yaw": float("nan")},
        {"pos": [True, 0]},
        {"world_sha256": "bad"},
    ],
)
def test_forged_fields_invalid_tokens_and_inexact_names_are_refused(pg, patch):  # noqa: F811
    pg.add_object("source", shape="box")
    before = state(pg)
    with pytest.raises(PlaygroundError):
        duplicate(pg, **patch)
    unchanged(pg, before)


def test_unsupported_mechanisms_and_active_execution_are_refused(pg):  # noqa: F811
    from robotforge.sim.execution import ExecutionError, RunLimits

    pg.add_object("source", shape="drawer")
    before = state(pg)
    with pytest.raises(PlaygroundError, match="editable"):
        duplicate(pg)
    unchanged(pg, before)
    with pg.session.execution.run("test", RunLimits()), pytest.raises(ExecutionError):
        duplicate(pg)


@pytest.mark.parametrize("change", ["reset", "file", "compile_failure"])
def test_compile_time_changes_and_failures_never_publish_a_copy(pg, monkeypatch, change):  # noqa: F811
    from robotforge.design import world as domain

    pg.add_object("source", shape="box")
    before = state(pg)
    compile_world = domain.compile_world
    external = before[0] + b"\n# external writer\n"

    def interfere(*args, **kwargs):
        if change == "compile_failure":
            raise ValueError("injected compile failure")
        result = compile_world(*args, **kwargs)
        if change == "reset":
            pg.session.reset()
        else:
            pg.store.file.write_bytes(external)
        return result

    monkeypatch.setattr(domain, "compile_world", interfere)
    with pytest.raises((PlaygroundError, ValueError)):
        duplicate(pg)
    assert pg.session.model is before[3]
    assert pg.store.file.read_bytes() == (external if change == "file" else before[0])
    assert (pg.store.build_dir / "world.xml").read_bytes() == before[1]
    assert [o.name for o in pg.store.load().objects] == ["source"]


def test_stale_zero_time_reset_and_descriptor_mismatch_refuse(pg):  # noqa: F811
    pg.add_object("source", shape="box")
    msg = request(pg)
    pg.session.reset()
    before = state(pg)
    with pytest.raises(PlaygroundError, match="changed"):
        pg.edit_object_for_play(msg)
    unchanged(pg, before)
    msg = request(pg)
    pg._play_world_descriptor = "0" * 64
    with pytest.raises(PlaygroundError, match="saved world differs"):
        pg.edit_object_for_play(msg)
    unchanged(pg, before)
