"""Batch primitive finishes are one strict, atomic edit with a heterogeneous inverse."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_object_surface import apply as single
from test_viewer_object_surface import undo
from test_viewer_terrain_create import snapshot, unchanged

from robotforge.design import layouts
from robotforge.playground import PlaygroundError
from robotforge.viewer import object_surface


def batch(pg, names=("box", "ball"), finish="wood", **patch):  # noqa: F811
    return pg.edit_object_for_play(
        {
            "action": "set_object_surfaces",
            "names": list(names),
            "surface_material": finish,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **patch,
        }
    )


def populate(playground):
    playground.add_object("box", material="steel", size=[0.2, 0.2, 0.2])
    playground.add_object("ball", shape="sphere", size=[0.1], pos=[1, 0], surface_material="plastic")
    playground.add_object("column", shape="cylinder", size=[0.1, 0.3], pos=[2, 0], surface_material="steel")
    playground.add_object("desk", shape="table", size=[1, 0.6, 0.7], pos=[4, 0])


def test_mixed_batch_one_compile_load_history_preserves_physics_and_latest_undo(pg, monkeypatch):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    populate(pg)
    initial = pg.store.load()
    old_model = pg.session.model
    pg.session.data.qpos[0] += 0.1
    pg.session.data.ctrl[:] = 0.2
    pg.session.data.qvel[:] = 0.3
    pg.session.data.time = 1
    mujoco.mj_forward(old_model, pg.session.data)
    pose, controls = pg.session.data.qpos.copy(), pg.session.data.ctrl.copy()
    calls = []
    compile0, load0 = object_surface.compile_world, pg._load_world

    def compile_once(*args):
        calls.append("compile")
        return compile0(*args)

    def load_once(*args):
        calls.append("load")
        return load0(*args)

    monkeypatch.setattr(object_surface, "compile_world", compile_once)
    monkeypatch.setattr(pg, "_load_world", load_once)
    version, generation = pg.session.version, pg.session.generation
    history = len(list(pg.store.history.glob("*.yaml")))
    names = ["column", "box", "ball"]
    result = batch(pg, names)
    assert result["objects"] == names and "object" not in result and "moved" not in result
    assert calls == ["compile", "load"]
    assert (pg.session.version, pg.session.generation) == (version + 1, generation + 1)
    assert len(list(pg.store.history.glob("*.yaml"))) == history + 1
    saved = pg.store.load()
    assert [o.surface_material for o in saved.objects[:3]] == ["wood"] * 3
    assert saved.objects[3] == initial.objects[3]
    for actual, original in zip(saved.objects, initial.objects, strict=True):
        assert actual.model_dump(exclude={"surface_material"}) == original.model_dump(exclude={"surface_material"})
    for field in [
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
        "geom_solref",
        "geom_solimp",
        "sensor_type",
        "sensor_objid",
        "sensor_objtype",
        "sensor_dim",
    ]:
        np.testing.assert_array_equal(getattr(old_model, field), getattr(pg.session.model, field))
    np.testing.assert_allclose(pg.session.data.qpos, pose, atol=1e-12)
    np.testing.assert_array_equal(pg.session.data.ctrl, controls)
    assert pg.session.paused and pg.session.data.time == 0 and not pg.session.data.qvel.any()
    receipt, before_noop = pg._play_edit, snapshot(pg)
    assert batch(pg, names)["unchanged"]
    unchanged(pg, before_noop)
    assert pg._play_edit is receipt and calls == ["compile", "load"]
    assert len(list(pg.store.history.glob("*.yaml"))) == history + 1
    pg.session.data.qpos[0] += 0.2
    latest = pg.session.data.qpos.copy()
    restored = undo(pg, result)
    assert restored["objects"] == names and restored["undo_edit"] is None
    assert pg.store.load() == initial and pg._play_edit is None
    assert calls == ["compile", "load", "compile", "load"]
    np.testing.assert_allclose(pg.session.data.qpos, latest, atol=1e-12)
    with pytest.raises(PlaygroundError):
        undo(pg, result)


def test_batch_matches_sequential_existing_surface_operations_original_and_reopen(pg):  # noqa: F811
    populate(pg)
    names = ["box", "ball", "column"]
    result = batch(pg, names, "steel")
    authored = pg.store.load()
    undo(pg, result)
    for name in names:
        single(pg, "steel", name=name)
    assert pg.store.load() == authored
    cleared = batch(pg, names, None)
    assert [o.surface_material for o in pg.store.load().objects[:3]] == [None] * 3
    undo(pg, cleared)
    assert pg.store.load() == authored
    pg.open_world(str(pg.store.dir))
    assert pg.store.load() == authored


@pytest.mark.parametrize("count", [2, 16])
def test_member_boundaries_and_mixed_changed_unchanged_targets(pg, monkeypatch, count):  # noqa: F811
    for index in range(count):
        pg.add_object(f"p{index}", pos=[index * 0.4, 0], surface_material="wood" if index else None)
    names = [f"p{i}" for i in reversed(range(count))]
    calls, compile0 = [], object_surface.compile_world

    def compile_once(*args):
        calls.append(1)
        return compile0(*args)

    monkeypatch.setattr(object_surface, "compile_world", compile_once)
    result = batch(pg, names)
    assert result["objects"] == names and len(calls) == 1
    undo(pg, result)
    assert [o.surface_material for o in pg.store.load().objects] == [None] + ["wood"] * (count - 1)


@pytest.mark.parametrize(
    "patch",
    [
        {"names": []},
        {"names": ["box"]},
        {"names": [f"p{i}" for i in range(17)]},
        {"names": ["box", "box"]},
        {"names": ["box", ""]},
        {"names": ["box", "x" * 33]},
        {"names": ["box", True]},
        {"names": ["box", 3]},
        {"names": "box"},
        {"names": ["box", "missing"]},
        {"names": ["box", "desk"]},
        {"surface_material": "gold"},
        {"surface_material": True},
        {"name": "box"},
        {"material": "wood"},
        {"version": True},
        {"generation": True},
        {"generation": -1},
        {"world_sha256": "bad"},
    ],
)
def test_invalid_member_or_guard_refuses_the_whole_edit(pg, patch):  # noqa: F811
    populate(pg)
    before = snapshot(pg)
    with pytest.raises(PlaygroundError):
        batch(pg, **patch)
    unchanged(pg, before)


@pytest.mark.parametrize("missing", ["surface_material", "version", "generation", "world_sha256"])
def test_required_fields_and_singular_protocol_remain_strict(pg, missing):  # noqa: F811
    populate(pg)
    message = dict(
        action="set_object_surfaces",
        names=["box", "ball"],
        surface_material=None,
        version=pg.session.version,
        generation=pg.session.generation,
        world_sha256=pg._play_world_sha256,
    )
    del message[missing]
    before = snapshot(pg)
    with pytest.raises(PlaygroundError):
        pg.edit_object_for_play(message)
    with pytest.raises(PlaygroundError):
        single(pg, name="box", names=["box", "ball"])
    unchanged(pg, before)


@pytest.mark.parametrize("failure", ["compile", "load", "publish", "carry"])
@pytest.mark.parametrize("undoing", [False, True])
def test_failure_restores_exact_files_runtime_and_retryable_receipt(pg, monkeypatch, failure, undoing):  # noqa: F811
    populate(pg)
    result = batch(pg)
    receipt, before = pg._play_edit, snapshot(pg)
    files = {p: p.read_bytes() for p in [*pg.store.history.glob("*.yaml"), pg.store.build_dir / "world.xml"]}
    with monkeypatch.context() as context:
        if failure == "publish":
            install0 = layouts._install_world

            @contextmanager
            def broken(*args):
                with install0(*args) as notes:
                    yield notes
                    raise OSError("injected batch failure")

            context.setattr(layouts, "_install_world", broken)
        else:

            def broken(*args, **kwargs):
                raise OSError("injected batch failure")

            target, field = {
                "compile": (object_surface, "compile_world"),
                "load": (pg, "_load_world"),
                "carry": (object_surface.carry, "_apply"),
            }[failure]
            context.setattr(target, field, broken)
        with pytest.raises(OSError, match="injected batch failure"):
            undo(pg, result) if undoing else batch(pg, finish="steel")
    unchanged(pg, before)
    assert pg._play_edit is receipt
    assert all(p.read_bytes() == data for p, data in files.items())
    assert set(pg.store.history.glob("*.yaml")) == {p for p in files if p.parent == pg.store.history}
    undo(pg, result)
    assert [o.surface_material for o in pg.store.load().objects[:2]] == [None, "plastic"]


@pytest.mark.parametrize("mutation", ["reset", "disk", "hooks", "lesson"])
def test_second_guard_rejects_mutation_during_preparation(pg, monkeypatch, mutation):  # noqa: F811
    populate(pg)
    compile0, model = object_surface.compile_world, pg.session.model
    original = pg.store.file.read_bytes()

    def mutate(*args):
        compiled = compile0(*args)
        if mutation == "reset":
            pg.session.reset()
        elif mutation == "disk":
            pg.store.file.write_bytes(original + b"\n# external\n")
        elif mutation == "hooks":
            pg.session.reset_hooks.append(lambda: None)
        else:
            pg.lessons.lesson = "test"
        return compiled

    monkeypatch.setattr(object_surface, "compile_world", mutate)
    with pytest.raises((PlaygroundError, layouts.LayoutError)):
        batch(pg)
    assert pg.session.model is model
    assert pg.store.load().objects[0].surface_material is None
    assert pg.store.file.read_bytes() == (original + b"\n# external\n" if mutation == "disk" else original)
    if mutation == "lesson":
        pg.lessons.lesson = None


def test_receipt_checks_each_precise_member_and_rejects_cross_type_tokens(pg):  # noqa: F811
    populate(pg)
    old = single(pg, name="box")
    result = batch(pg, finish="steel")
    with pytest.raises(PlaygroundError):
        undo(pg, old)
    receipt = pg._play_edit
    member = receipt.members[1]
    pg._play_edit = replace(receipt, members=(receipt.members[0], replace(member, index=0)))
    before = snapshot(pg)
    with pytest.raises(PlaygroundError, match="prop changed"):
        undo(pg, result)
    unchanged(pg, before)
    pg._play_edit = receipt
    undo(pg, result)
    newer = single(pg, "plastic", name="box")
    with pytest.raises(PlaygroundError):
        undo(pg, result)
    undo(pg, newer)
