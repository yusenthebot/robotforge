"""Row arrangement uses authoritative extents and the existing guarded move transaction."""

from __future__ import annotations

# ruff: noqa: F811
import math

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_copy import state, unchanged
from test_viewer_object_group import duplicate, prepare, stack, undo
from test_viewer_object_group_move import move, pose

from robotforge.playground import PlaygroundError


def pair(pg):
    pg.add_object("wide", shape="box", size=[0.4, 0.2, 0.2], pos=[-2, -0.6], material="wood")
    pg.add_object("small", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0.6], material="steel")
    pg.session.paused = True


def row(pg, names=("wide", "small"), **patch):
    return prepare(pg, names, mode="move", layout="row", **{"gap": 0.25, **patch})


@pytest.mark.parametrize("reverse", [False, True])
def test_source_response_and_selection_order_use_unequal_complete_rotated_extents(pg, reverse):
    pg.add_object("table", shape="table", size=[1.2, 0.8, 0.5], fixed=True, pos=[-2, -1], yaw=90)
    pg.add_object("bin", shape="bin", size=[0.3, 0.5, 0.2], fixed=True, pos=[-2, 1], yaw=90)
    pg.session.paused = True
    names = ["bin", "table"] if reverse else ["table", "bin"]
    original = prepare(pg, names, mode="move")
    before = state(pg)
    arranged = row(pg, names)
    unchanged(pg, before)
    assert arranged["layout"] == "row" and arranged["gap"] == 0.25
    assert arranged["anchor"] == original["anchor"]
    assert arranged["pose_sha256"] == original["pose_sha256"]
    for source, member in zip(original["members"], arranged["members"], strict=True):
        assert {k: v for k, v in member.items() if k != "target_offset"} == source
        solid = [g for g in member["geoms"] if pg.session.model.geom_contype[g] or pg.session.model.geom_conaffinity[g]]
        # The table also owns four cloth skirts and four 1 cm radius capsule hems.
        assert len(solid) == (13 if member["name"] == "table" else 5)
    expected = [-0.535, 0.375] if reverse else [-0.375, 0.535]
    assert [member["target_offset"][0] for member in arranged["members"]] == pytest.approx(expected)
    assert all(member["target_offset"][1] == 0 for member in arranged["members"])
    result = move(pg, arranged, pos=[3, 0, 0], yaw=90.0)
    assert result["objects"] == ["table", "bin"]
    actual = {obj.name: obj for obj in pg.store.load().objects}
    for name, offset in zip(names, expected, strict=True):
        assert actual[name].pos[:2] == pytest.approx([3, offset])
        assert actual[name].yaw == pytest.approx(180)


@pytest.mark.parametrize("count", [2, 16])
def test_selection_bounds_generate_a_centered_row_without_reordering_world(pg, count):
    names = [f"prop{i}" for i in range(count)]
    for index, name in enumerate(names):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[-2, index * 0.5])
    pg.session.paused = True
    prepared = row(pg, list(reversed(names)), gap=0.05)
    assert [m["name"] for m in prepared["members"]] == list(reversed(names))
    expected = [(i - (count - 1) / 2) * 0.25 for i in range(count)]
    assert [m["target_offset"][0] for m in prepared["members"]] == pytest.approx(expected)
    assert move(pg, prepared)["objects"] == names


@pytest.mark.parametrize(
    "patch",
    [
        {"gap": None},
        {"gap": True},
        {"gap": "0.25"},
        {"gap": float("nan")},
        {"gap": float("inf")},
        {"gap": float("-inf")},
        {"gap": 0.049},
        {"gap": 2.001},
        {"layout": "circle"},
        {"layout": "row", "mode": "copy"},
        {"layout": "preserve", "gap": 0.25},
        {"layout": "preserve", "gap": None},
        {"names": ["wide"]},
        {"names": ["wide", "wide"]},
        {"names": ["wide"] * 17},
        {"target_offsets": [[0, 0, 0], [1, 0, 0]]},
    ],
)
def test_invalid_prepare_is_atomic_and_retains_previous_preparation(pg, patch):
    pair(pg)
    prepared = row(pg)
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError):
        prepare(pg, **{"names": ["wide", "small"], "mode": "move", "layout": "row", "gap": 0.25, **patch})
    unchanged(pg, before)
    assert pg._play_group is token and token.token == prepared["token"]


def test_row_requires_explicit_gap_and_preserve_keeps_exact_response_shape(pg):
    pair(pg)
    with pytest.raises(PlaygroundError):
        prepare(pg, ["wide", "small"], mode="move", layout="row")
    copied = prepare(pg, ["wide", "small"])
    preserved = prepare(pg, ["wide", "small"], mode="move", layout="preserve")
    assert (
        set(copied)
        == set(preserved)
        == {"mode", "token", "pose_sha256", "anchor", "version", "generation", "world_sha256", "members"}
    )
    assert set(preserved["members"][0]) == {"name", "shape", "size", "pos", "yaw", "geoms", "offset"}


@pytest.mark.parametrize(
    "patch", [{"gap": 1.0}, {"layout": "row"}, {"target_offset": [0, 0, 0]}, {"names": ["wide", "small"]}]
)
def test_commit_never_accepts_client_layout_overrides(pg, patch):
    pair(pg)
    prepared = row(pg)
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError):
        move(pg, prepared, **patch)
    unchanged(pg, before)
    assert pg._play_group is token


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_selected_stack_rejected_only_for_row_preserve_remains_valid(pg, mode):
    stack(pg)
    before = state(pg)
    with pytest.raises(PlaygroundError, match="independent|unstack"):
        row(pg, ["base", "top"])
    unchanged(pg, before)
    prepared = prepare(pg, mode=mode)
    assert (move if mode == "move" else duplicate)(pg, prepared)["objects"]


@pytest.mark.parametrize("source", ["saved", "live", "both"])
def test_source_props_must_share_same_external_support_in_saved_and_live_worlds(pg, source):
    for name, x in (("left", -2), ("right", 2)):
        pg.add_object(name, shape="table", size=[1.5, 1.5, 0.5], pos=[x, 0])
    for name, y in (("wide", -0.3), ("small", 0.3)):
        x = 2 if source in {"saved", "both"} and name == "small" else -2
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[x, y, 0.6])
    if source == "saved":
        pose(pg, "small", [-2, 0.3, 0.6])
    elif source == "live":
        pose(pg, "small", [2, 0.3, 0.6])
    pg.session.paused = True
    before = state(pg)
    with pytest.raises(PlaygroundError, match="shared external"):
        row(pg)
    unchanged(pg, before)


def test_saved_and_live_selection_cannot_use_two_different_shared_supports(pg):
    for name, x in (("left", -2), ("right", 2)):
        pg.add_object(name, shape="table", size=[1.5, 1.5, 0.5], pos=[x, 0])
    for name, y in (("wide", -0.3), ("small", 0.3)):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[-2, y, 0.6])
    for name, y in (("wide", -0.3), ("small", 0.3)):
        pose(pg, name, [2, y, 0.6])
    pg.session.paused = True
    with pytest.raises(PlaygroundError, match="shared external"):
        row(pg)


@pytest.mark.parametrize("phase", ["saved", "live", "after_prepare"])
def test_external_load_is_rejected_before_and_after_prepare(pg, phase):
    pair(pg)
    pg.add_object("load", shape="box", size=[0.1, 0.1, 0.1], pos=[-2, -0.6, 0.25] if phase == "saved" else [7, 0])
    pg.session.paused = True
    prepared = row(pg) if phase == "after_prepare" else None
    if phase != "saved":
        pose(pg, "load", [-2, -0.6, 0.25])
    before = state(pg)
    with pytest.raises(PlaygroundError, match="supports"):
        move(pg, prepared) if prepared else row(pg)
    unchanged(pg, before)


@pytest.mark.parametrize("phase", ["prepare", "after_prepare"])
def test_tilted_or_changed_selected_source_never_moves(pg, phase):
    pair(pg)
    prepared = row(pg) if phase == "after_prepare" else None
    adr = pg.session.model.jnt_qposadr[pg.session.model.joint("wide").id]
    pg.session.data.qpos[adr + 3 : adr + 7] = [math.cos(0.1), math.sin(0.1), 0, 0]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = state(pg)
    with pytest.raises(PlaygroundError, match="tilted|pose changed"):
        move(pg, prepared) if prepared else row(pg)
    unchanged(pg, before)


def test_task_bound_row_is_rejected(pg):
    from robotforge.design.tasks.spec import TaskEntry

    pair(pg)
    pg.add_robot("arm", template="arm", pos=[7, 0])
    pg._edit(lambda world: world.tasks.append(TaskEntry(name="pick", kind="pick_up", robot="arm", object="small")))
    pg.session.paused = True
    before = state(pg)
    with pytest.raises(PlaygroundError, match="task"):
        row(pg)
    unchanged(pg, before)


@pytest.mark.parametrize("source", ["saved", "live"])
def test_target_occupancy_rejects_whole_transaction_and_preserves_retry(pg, source):
    pair(pg)
    pg.add_object("obstacle", shape="box", size=[0.4, 0.4, 0.2], pos=[3, 0])
    pose(pg, "obstacle", [5, 0, 0.1])
    pg.session.paused = True
    prepared = row(pg)
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError, match="occupied"):
        move(pg, prepared, pos=[3 if source == "saved" else 5, 0, 0])
    unchanged(pg, before)
    assert pg._play_group is token
    assert move(pg, prepared, pos=[7, 0, 0])["moved"]


def test_same_anchor_changes_internal_positions_once_then_exact_undo(pg, monkeypatch):
    from robotforge.design import store as store_module

    pair(pg)
    originals = pg.store.load().objects
    pose(pg, "wide", [-2.1, -0.6, 0.1], 30)
    pose(pg, "small", [-1.9, 0.6, 0.1], -20)
    before = state(pg)
    calls = []
    original = store_module.compile_world

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(store_module, "compile_world", counted)
    prepared = row(pg)
    assert not calls
    result = move(pg, prepared, pos=prepared["anchor"])
    assert result.get("unchanged") is None and result["moved"]
    assert len(calls) == 1 and pg.session.version == before[5] + 1
    assert len(list(pg.store.history.glob("*.yaml"))) == len(before[2]) + 1
    moved = pg.store.load().objects
    assert [obj.pos[1] for obj in moved] == pytest.approx([0, 0])
    assert [obj.yaw for obj in moved] == pytest.approx([30, -20])
    for source, target in zip(originals, moved, strict=True):
        assert source.model_dump(exclude={"pos", "yaw"}) == target.model_dump(exclude={"pos", "yaw"})
    undo(pg, result)
    assert len(calls) == 2 and pg.store.load().objects == originals
    np.testing.assert_allclose(pg.session.data.qpos, before[7], rtol=0, atol=1e-12)


def test_genuine_noop_keeps_previous_receipt_history_velocity_time_and_full_turns(pg, monkeypatch):
    from robotforge.design import store as store_module

    pair(pg)
    pg._edit(lambda world: [setattr(obj, "yaw", 720.0) for obj in world.objects])
    pg.session.paused = True
    result = move(pg, row(pg))
    assert [obj.yaw for obj in pg.store.load().objects] == [720, 720]
    prepared = row(pg)
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 3.0
    before, receipt, token = state(pg), pg._play_edit, pg._play_group

    def forbidden(*args, **kwargs):
        raise AssertionError("a genuine row no-op must not compile")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    result_noop = move(pg, prepared, pos=[3, 0, 0], yaw=360.0)
    assert result_noop["unchanged"]
    unchanged(pg, before)
    assert pg._play_edit is receipt and pg._play_group is token
    assert receipt.token == result["undo_edit"]["token"]


def test_live_row_different_from_authored_still_commits_and_undo_restores_both(pg):
    pair(pg)
    original = pg.store.load().objects
    # Widths .4/.2 plus .25 gap span .85: centers are -.225 and .325.
    pose(pg, "wide", [-0.225, 0, 0.1])
    pose(pg, "small", [0.325, 0, 0.1])
    before = state(pg)
    prepared = row(pg)
    result = move(pg, prepared, pos=[0, 0, 0])
    assert result.get("unchanged") is None and pg.session.version == before[5] + 1
    np.testing.assert_allclose(pg.session.data.qpos, before[7], rtol=0, atol=1e-12)
    undo(pg, result)
    assert pg.store.load().objects == original
    np.testing.assert_allclose(pg.session.data.qpos, before[7], rtol=0, atol=1e-12)


@pytest.mark.parametrize("failure", ["compile", "validate", "load_after", "selected_pose"])
def test_row_failure_restores_disk_runtime_history_preparation_and_previous_undo(pg, monkeypatch, failure):
    from robotforge.design import store as store_module
    from robotforge.viewer import object_group

    pair(pg)
    previous = duplicate(pg, prepare(pg, ["wide", "small"]))
    prepared = row(pg)
    before, receipt, token = state(pg), pg._play_edit, pg._play_group

    def fail(*args, **kwargs):
        raise ValueError("injected row failure")

    if failure == "compile":
        monkeypatch.setattr(store_module, "compile_world", fail)
    elif failure == "validate":
        monkeypatch.setattr(object_group, "_validate_targets", fail)
    elif failure == "selected_pose":
        original = object_group.carry._apply

        def selected(model, data, snapshot, chosen):
            if pg.session.version > before[5] and chosen("wide"):
                fail()
            return original(model, data, snapshot, chosen)

        monkeypatch.setattr(object_group.carry, "_apply", selected)
    else:
        original = pg._load_world

        def load(*args, **kwargs):
            original(*args, **kwargs)
            fail()

        monkeypatch.setattr(pg, "_load_world", load)
    with pytest.raises(ValueError, match="injected row failure"):
        move(pg, prepared, pos=[7, 0, 0])
    unchanged(pg, before)
    assert pg._play_edit is receipt and pg._play_group is token
    assert receipt.token == previous["undo_edit"]["token"]


@pytest.mark.parametrize("change", ["new_preparation", "reset", "running", "disk"])
def test_stale_row_tokens_never_mutate_world(pg, change):
    pair(pg)
    prepared = row(pg)
    if change == "new_preparation":
        row(pg, ["small", "wide"])
    elif change == "reset":
        pg.session.reset()
    elif change == "running":
        pg.session.paused = False
    else:
        pg.store.file.write_text(pg.store.file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    before = state(pg)
    with pytest.raises(PlaygroundError):
        move(pg, prepared)
    unchanged(pg, before)


def test_support_disappearing_after_prepare_is_requalified(pg):
    pg.add_object("support", shape="box", size=[1.5, 1.5, 0.2], pos=[-2, 0])
    for name, y in (("wide", -0.3), ("small", 0.3)):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[-2, y, 0.3])
    pg.session.paused = True
    prepared = row(pg)
    pose(pg, "support", [4, 0, 0.1])
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError, match="shared external"):
        move(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is token


def test_source_qualification_repeated_after_target_validation_before_publish(pg, monkeypatch):
    from robotforge.viewer import object_group

    pair(pg)
    pg.add_object("load", shape="box", size=[0.1, 0.1, 0.1], pos=[7, 0])
    pg.session.paused = True
    prepared = row(pg)
    before, token = state(pg), pg._play_group
    validate = object_group._validate_targets

    def arrive_after_validation(*args, **kwargs):
        validate(*args, **kwargs)
        pose(pg, "load", [-2, -0.6, 0.25])

    monkeypatch.setattr(object_group, "_validate_targets", arrive_after_validation)
    with pytest.raises(PlaygroundError, match="supports"):
        move(pg, prepared)
    after = state(pg)
    assert after[:3] == before[:3] and after[5:7] == before[5:7]
    assert after[3] is before[3] and after[4] is before[4]
    assert pg._play_group is token
    assert list(pg.session.data.body("load").xpos) == pytest.approx([-2, -0.6, 0.25])


def test_semantic_noop_preserves_implicit_height_and_authored_full_turns(pg, monkeypatch):
    from robotforge.design import store as store_module

    for name, x in (("wide", -0.225), ("small", 0.225)):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[x, 0], yaw=720)
    pg.session.paused = True
    original = pg.store.load().objects
    prepared = row(pg)
    before = state(pg)

    def forbidden(*args, **kwargs):
        raise AssertionError("equivalent implicit height must not normalize authored entries")

    monkeypatch.setattr(store_module, "compile_world", forbidden)
    assert move(pg, prepared, pos=[0, 0, 0])["unchanged"]
    unchanged(pg, before)
    assert pg.store.load().objects == original
    assert all(len(obj.pos) == 2 and obj.yaw == 720 for obj in original)


def test_tiny_live_tilt_not_hidden_by_upright_eligibility_noop_threshold(pg):
    for name, x in (("wide", -0.225), ("small", 0.225)):
        pg.add_object(name, shape="box", size=[0.2, 0.2, 0.2], pos=[x, 0])
    pg.session.paused = True
    adr = pg.session.model.jnt_qposadr[pg.session.model.joint("wide").id]
    angle = 1e-6
    pg.session.data.qpos[adr + 3 : adr + 7] = [math.cos(angle / 2), math.sin(angle / 2), 0, 0]
    mujoco.mj_forward(pg.session.model, pg.session.data)
    prepared = row(pg)
    result = move(pg, prepared, pos=[0, 0, 0])
    assert result.get("unchanged") is None
    np.testing.assert_allclose(pg.session.data.body("wide").xmat, np.eye(3).ravel(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("shape,size", [("sphere", [0.4]), ("cylinder", [0.4, 0.2])])
def test_round_solids_use_true_edge_extents_instead_of_rotated_bounding_square(pg, shape, size):
    pg.add_object("wide", shape=shape, size=size, pos=[-2, -0.6], yaw=45)
    pg.add_object("small", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0.6])
    pg.session.paused = True
    prepared = row(pg)
    assert [member["target_offset"][0] for member in prepared["members"]] == pytest.approx([-0.225, 0.325])
    result = move(pg, prepared)
    assert result["moved"]
    wide, small = pg.store.load().objects
    assert small.pos[0] - 0.1 - (wide.pos[0] + 0.2) == pytest.approx(0.25)
    assert wide.yaw == pytest.approx(45)


@pytest.mark.parametrize("shape", ["bin", "table"])
def test_target_collision_uses_compound_walls_and_legs_not_only_named_geom(pg, shape):
    pg.add_object("wide", shape=shape, size=[0.4, 0.4, 0.5], fixed=True, pos=[-2, -0.6])
    pg.add_object("small", shape="box", size=[0.2, 0.2, 0.2], pos=[-2, 0.6])
    # The wide pivot is x=2.775 for either shape; table hems affect the other pivot.
    obstacle = [2.9725, 0, 0.15] if shape == "bin" else [2.925, 0.15, 0.2]
    pg.add_object("obstacle", shape="box", size=[0.02, 0.02, 0.05], fixed=True, pos=obstacle)
    pg.session.paused = True
    prepared = row(pg)
    before, token = state(pg), pg._play_group
    with pytest.raises(PlaygroundError, match="occupied"):
        move(pg, prepared)
    unchanged(pg, before)
    assert pg._play_group is token


def test_flex_world_is_rejected_without_mutation(pg):
    pair(pg)
    pg.add_object("cloth", shape="cloth", size=[0.2, 0.2], pos=[7, 0, 0.8])
    pg.session.paused = True
    before = state(pg)
    with pytest.raises(PlaygroundError, match="rigid|soft|flex"):
        row(pg)
    unchanged(pg, before)


def test_maximum_gap_is_accepted_and_used_as_edge_gap(pg):
    pair(pg)
    prepared = row(pg, gap=2)
    assert prepared["gap"] == 2
    assert [member["target_offset"][0] for member in prepared["members"]] == pytest.approx([-1.1, 1.2])
    assert move(pg, prepared)["moved"]
