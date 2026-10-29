"""One explicit part selection is one guarded rebuild and one inverse."""

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401
from test_viewer_robot_surface import apply as single
from test_viewer_robot_surface import undo
from test_viewer_terrain_create import snapshot, unchanged

from robotforge.playground import PlaygroundError
from robotforge.viewer import robot_surface


def parts(pg):  # noqa: F811
    return [p["body"] for p in robot_surface.metadata(pg)["robots"][0]["parts"]]


def batch(pg, selected, surface="steel", **patch):  # noqa: F811
    return pg.edit_robot_surface_for_play(
        {
            "action": "set_robot_surface",
            "name": "bot",
            "bodies": selected,
            "surface_material": surface,
            "version": pg.session.version,
            "generation": pg.session.generation,
            "world_sha256": pg._play_world_sha256,
            **patch,
        }
    )


def test_batch_one_rebuild_carries_pose_preserves_physics_and_undoes_mixed_finishes(pg, monkeypatch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    pg.add_robot("other", template="arm", pos=[-2, 0])
    selected = parts(pg)[:3]
    assert len(selected) == 3
    single(pg, "wood", selected[0])
    single(pg, "plastic", selected[1])
    initial = pg.store.load()
    original = pg.session.model
    pg.session.data.qpos[0] += 0.1
    pg.session.data.ctrl[:] = 0.2
    pg.session.data.qvel[:] = 0.4
    pg.session.data.time = 1
    mujoco.mj_forward(original, pg.session.data)
    pose, ctrl = pg.session.data.qpos.copy(), pg.session.data.ctrl.copy()
    compile0, load0 = robot_surface.compile_world, pg._load_world
    calls = []

    def compile_once(*args):
        calls.append("compile")
        return compile0(*args)

    def load_once(*args):
        calls.append("load")
        return load0(*args)

    monkeypatch.setattr(robot_surface, "compile_world", compile_once)
    monkeypatch.setattr(pg, "_load_world", load_once)
    version = pg.session.version
    result = batch(pg, selected)
    assert calls == ["compile", "load"] and pg.session.version == version + 1
    assert result["bodies"] == selected and "body" not in result
    assert pg.store.load().robots[0].body_surfaces == dict.fromkeys(selected, "steel")
    assert pg.store.load().robots[1] == initial.robots[1]
    for field in [
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_rgba",
        "geom_friction",
        "geom_contype",
        "geom_conaffinity",
    ]:
        np.testing.assert_array_equal(getattr(original, field), getattr(pg.session.model, field))
    np.testing.assert_allclose(pg.session.data.qpos, pose, atol=1e-12)
    np.testing.assert_array_equal(pg.session.data.ctrl, ctrl)
    assert pg.session.paused and pg.session.data.time == 0 and not pg.session.data.qvel.any()
    undo(pg, result)
    assert pg.store.load() == initial
    cleared = batch(pg, selected[:2], None)
    assert not pg.store.load().robots[0].body_surfaces
    undo(pg, cleared)
    assert pg.store.load() == initial
    pg.open_world(str(pg.store.dir))
    assert pg.store.load() == initial


def test_all_noop_preserves_receipt_and_mixed_noop_changes_compile_once(pg, monkeypatch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    selected = parts(pg)[:3]
    first = batch(pg, selected[:2])
    receipt, state = pg._play_edit, snapshot(pg)
    compile0 = robot_surface.compile_world
    calls = []

    def compile_once(*args):
        calls.append(True)
        return compile0(*args)

    monkeypatch.setattr(robot_surface, "compile_world", compile_once)
    assert batch(pg, selected[:2])["unchanged"]
    unchanged(pg, state)
    assert pg._play_edit is receipt and not calls
    result = batch(pg, selected)
    assert len(calls) == 1
    undo(pg, result)
    assert pg.store.load().robots[0].body_surfaces == dict.fromkeys(selected[:2], "steel")
    with pytest.raises(PlaygroundError):
        undo(pg, first)


@pytest.mark.parametrize(
    "patch",
    [
        {"bodies": []},
        {"bodies": None},
        {"bodies": "chassis"},
        {"bodies": [True]},
        {"bodies": ["x" * 161]},
        {"bodies": ["chassis", "chassis"]},
        {"bodies": [f"part{i}" for i in range(65)]},
        {"body": "chassis"},
        {"body": None},
        {"bodies": ["chassis", "absent"]},
        {"bodies": ["chassis", "other/chassis"]},
        {"generation": -1},
        {"world_sha256": "0" * 64},
        {"surface_material": "gold"},
    ],
)
def test_entire_invalid_selection_is_rejected_atomically(pg, monkeypatch, patch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    selected = parts(pg)[:2]
    batch(pg, selected, "wood")
    receipt, state = pg._play_edit, snapshot(pg)

    def forbidden(*_args):
        pytest.fail("invalid batch must fail before compilation")

    monkeypatch.setattr(robot_surface, "compile_world", forbidden)
    with pytest.raises(PlaygroundError):
        batch(pg, selected, **patch)
    unchanged(pg, state)
    assert pg._play_edit is receipt


def test_resulting_override_limit_accounts_for_previously_styled_parts(pg, tmp_path, monkeypatch):  # noqa: F811
    path = tmp_path / "many.xml"
    children = "".join(f'<body name="part{i}" pos="{i * 0.02} 0 .3"><geom size=".01"/></body>' for i in range(66))
    path.write_text(
        f'<mujoco><worldbody><body name="base"><freejoint/><geom size=".1"/>{children}</body></worldbody></mujoco>',
        encoding="utf-8",
    )
    pg.add_robot("bot", file=str(path))
    selected = parts(pg)
    assert len(selected) >= 65
    batch(pg, selected[:64])
    receipt, state = pg._play_edit, snapshot(pg)

    def forbidden(*_args):
        pytest.fail("result mapping limit must fail before compilation")

    monkeypatch.setattr(robot_surface, "compile_world", forbidden)
    with pytest.raises(PlaygroundError, match="64"):
        batch(pg, selected[64:65])
    unchanged(pg, state)
    assert pg._play_edit is receipt


def test_failed_batch_publication_rolls_back_all_members_and_previous_receipt(pg, monkeypatch):  # noqa: F811
    pg.add_robot("bot", template="rover")
    selected = parts(pg)[:3]
    single(pg, "plastic", selected[0])
    receipt, state = pg._play_edit, snapshot(pg)
    load = pg._load_world

    def fail(compiled):
        load(compiled)
        raise RuntimeError("injected batch publication")

    monkeypatch.setattr(pg, "_load_world", fail)
    with pytest.raises(RuntimeError, match="injected batch"):
        batch(pg, selected)
    unchanged(pg, state)
    assert pg._play_edit is receipt
