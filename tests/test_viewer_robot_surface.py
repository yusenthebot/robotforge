"""Guarded robot body styling carries poses and uses the shared latest-only edit slot."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_terrain_create import snapshot, unchanged

from robotforge.playground import PlaygroundError
from robotforge.viewer import robot_surface


def apply(pg, finish="steel", body=None, **patch):  # noqa: F811
    parts = robot_surface.metadata(pg)["robots"][0]["parts"]
    return pg.edit_robot_surface_for_play(
        {
            "action": "set_robot_surface",
            "name": "bot",
            "body": body or parts[0]["body"],
            "surface_material": finish,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **patch,
        }
    )


def undo(pg, result):  # noqa: F811
    return pg.edit_robot_surface_for_play({"action": "undo_robot_surface", **result["undo_surface"]})


def test_apply_noop_original_undo_and_reopen(pg):  # noqa: F811
    pg.add_robot("bot", template="rover")
    pg.add_robot("other", template="arm", pos=[-2, 0])
    before = pg.store.load()
    original = pg.session.model
    original_alpha = original.mat_rgba[original.geom_matid].copy()
    pg.session.data.qpos[0] += 0.2
    pg.session.data.qvel[:] = 0.1
    pg.session.data.ctrl[:] = 0.2
    pg.session.data.time = 1.0
    mujoco.mj_forward(original, pg.session.data)
    pose, ctrl = pg.session.data.qpos.copy(), pg.session.data.ctrl.copy()
    result = apply(pg)
    np.testing.assert_allclose(pg.session.data.qpos, pose, atol=1e-12)
    np.testing.assert_array_equal(pg.session.data.ctrl, ctrl)
    assert pg.session.paused and pg.session.data.time == 0 and not pg.session.data.qvel.any()
    for field in [
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_rgba",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
    ]:
        np.testing.assert_array_equal(getattr(pg.session.model, field), getattr(original, field))
    np.testing.assert_array_equal(pg.session.model.mat_rgba[pg.session.model.geom_matid, 3], original_alpha[:, 3])
    assert pg.store.load().robots[1] == before.robots[1]
    receipt = pg._play_edit
    state = snapshot(pg)
    assert apply(pg)["unchanged"]
    unchanged(pg, state)
    assert pg._play_edit is receipt
    undo(pg, result)
    assert pg.store.load() == before and pg._play_edit is None
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, result)
    body = robot_surface.metadata(pg)["robots"][0]["parts"][0]["body"]
    apply(pg, "wood", body)
    directory = str(pg.store.dir)
    pg.open_world(directory)
    assert pg.store.load().robots[0].body_surfaces == {body: "wood"}
    original = apply(pg, None, body)
    assert not pg.store.load().robots[0].body_surfaces
    undo(pg, original)
    assert pg.store.load().robots[0].body_surfaces == {body: "wood"}


@pytest.mark.parametrize(
    "patch",
    [
        {"body": "absent"},
        {"body": True},
        {"surface_material": "gold"},
        {"version": True},
        {"generation": -1},
        {"world_sha256": "bad"},
        {"name": "other"},
        {"mass": 1},
    ],
)
def test_invalid_requests_do_not_touch_world_runtime_or_receipts(pg, patch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    first = apply(pg)
    receipt, state = pg._play_edit, snapshot(pg)
    with pytest.raises(PlaygroundError):
        apply(pg, **patch)
    unchanged(pg, state)
    assert pg._play_edit is receipt
    undo(pg, first)


def test_added_sensor_housings_and_definitions_are_not_restyled(pg):  # noqa: F811
    pg.add_robot("bot", template="rover")
    pg.add_sensor("bot", {"type": "camera", "name": "head"})
    old = pg.session.model
    before = {
        old.mesh(int(old.geom_dataid[g])).name: old.geom_matid[g]
        for g in range(old.ngeom)
        if old.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and "head_sensor_" in old.mesh(int(old.geom_dataid[g])).name
    }
    assert before
    part = robot_surface.metadata(pg)["robots"][0]["parts"][0]
    assert not any(
        g in part["geoms"]
        for g in range(old.ngeom)
        if old.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and "head_sensor_" in old.mesh(int(old.geom_dataid[g])).name
    )
    apply(pg)
    new = pg.session.model
    for g in range(new.ngeom):
        if new.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            name = new.mesh(int(new.geom_dataid[g])).name
            if name in before:
                assert new.material(int(new.geom_matid[g])).name == old.material(int(before[name])).name
    for field in ["cam_pos", "cam_quat", "cam_fovy", "sensor_type", "sensor_objid", "site_pos", "site_quat"]:
        np.testing.assert_array_equal(getattr(old, field), getattr(new, field))


def test_failed_publication_and_reset_invalidate_only_when_appropriate(pg, monkeypatch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    first = apply(pg)
    receipt, state = pg._play_edit, snapshot(pg)
    load = pg._load_world

    def fail(compiled):
        load(compiled)
        raise RuntimeError("injected post-publication failure")

    monkeypatch.setattr(pg, "_load_world", fail)
    with pytest.raises(RuntimeError, match="injected"):
        apply(pg, "wood")
    unchanged(pg, state)
    assert pg._play_edit is receipt
    monkeypatch.setattr(pg, "_load_world", load)
    pg.session.reset()
    with pytest.raises(PlaygroundError):
        undo(pg, first)


@pytest.mark.parametrize("source", ["template", "library", "urdf"])
def test_three_robot_sources_apply_and_restore_original(pg, source, tmp_path, monkeypatch):  # noqa: F811
    if source == "template":
        pg.add_robot("bot", template="rover")
    elif source == "library":
        monkeypatch.setenv("ROBOTFORGE_MENAGERIE", str(Path(__file__).parent / "fixtures/menagerie"))
        pg.add_robot("bot", library="go2")
    else:
        path = tmp_path / "robot.urdf"
        path.write_text("""<robot name="sample"><link name="base"><inertial><mass value="1"/>
          <inertia ixx=".1" ixy="0" ixz="0" iyy=".1" iyz="0" izz=".1"/></inertial>
          <visual><geometry><box size=".2 .3 .4"/></geometry><material name="coat">
          <color rgba=".2 .3 .4 1"/></material></visual>
          <collision><geometry><box size=".2 .3 .4"/></geometry></collision></link></robot>""")
        pg.add_robot("bot", file=str(path))
    original = pg.session.model
    result = apply(pg, "wood")
    assert pg.store.load().robots[0].body_surfaces
    undo(pg, result)
    restored = pg.session.model
    for field in ["geom_matid", "geom_rgba", "mat_rgba", "body_mass", "body_inertia", "geom_friction"]:
        np.testing.assert_array_equal(getattr(original, field), getattr(restored, field))


def test_robot_and_prop_undo_types_do_not_cross(pg):  # noqa: F811
    from test_viewer_object_surface import apply as prop_surface

    pg.add_robot("bot", template="rover")
    pg.add_object("block")
    robot = apply(pg)
    with pytest.raises(PlaygroundError):
        pg.edit_object_for_play({"action": "undo_object_edit", **robot["undo_surface"]})
    prop = prop_surface(pg)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, robot)
    pg.edit_object_for_play({"action": "undo_object_edit", **prop["undo_edit"]})


def test_failed_compilation_keeps_existing_edit_and_exact_world(pg, monkeypatch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    first = apply(pg)
    receipt, state = pg._play_edit, snapshot(pg)

    def fail(*args, **kwargs):
        raise ValueError("injected compiler failure")

    monkeypatch.setattr(robot_surface, "compile_world", fail)
    with pytest.raises(ValueError, match="compiler"):
        apply(pg, "wood")
    unchanged(pg, state)
    assert pg._play_edit is receipt
    assert first["undo_surface"]["token"] == receipt.token
