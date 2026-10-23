"""Local object placement and one-use inverse edits against saved and live physics."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest
from starlette.testclient import TestClient
from test_viewer_creation import pg  # noqa: F401

from robotforge.playground import PlaygroundError, viewer_catalog
from robotforge.sim.execution import ExecutionError, RunLimits
from robotforge.viewer.creation import OBJECTS, SCALES
from robotforge.viewer.server import ViewerHub, create_app


def add(pg, preset="ball", pos=None, **extra):  # noqa: F811
    return pg.object_for_play(
        {"preset": preset, "pos": [0.0, 0.0, 0.0] if pos is None else pos, "version": pg.session.version, **extra}
    )


def undo(pg, reply, **extra):  # noqa: F811
    return pg.object_for_play({"action": "undo_add", **{k: reply["undo"][k] for k in ["token", "version"]}, **extra})


@pytest.mark.parametrize("preset", list(OBJECTS))
@pytest.mark.parametrize("size", list(SCALES))
def test_all_server_owned_presets_scale_and_support_heights(pg, preset, size):  # noqa: F811
    reply = add(pg, preset, [0.0, 0.0, 0.8], size=size, yaw=90)
    entry = pg.store.load().objects[0]
    expected = [s * SCALES[size] for s in OBJECTS[preset]["size"]]
    assert entry.size == expected and entry.yaw == 90
    expected_z = 0.8 + (0 if entry.shape == "table" else expected[-1] / 2)
    assert entry.pos == pytest.approx([0, 0, expected_z])
    assert reply["object"] == preset and len(reply["undo"]["token"]) == 32
    pg.session.step(10)
    assert np.all(np.isfinite(pg.session.data.qpos))


@pytest.mark.parametrize("yaw", [0, 90, 180, 270])
def test_quarter_turns_and_ground_following(pg, yaw):  # noqa: F811
    add(pg, "barrier", yaw=yaw)
    assert pg.store.load().objects[0].pos == [0, 0]
    geom = mujoco.mj_name2id(pg.session.model, mujoco.mjtObj.mjOBJ_GEOM, "barrier")
    np.testing.assert_allclose(
        pg.session.data.geom_xmat[geom][:2], [np.cos(np.deg2rad(yaw)), -np.sin(np.deg2rad(yaw))], atol=1e-12
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"preset": "../ball"},
        {"preset": "ramp"},
        {"size": "giant"},
        {"size": 2},
        {"yaw": 45},
        {"yaw": True},
        {"yaw": False},
        {"yaw": 90.0},
        {"mass": 100},
        {"fixed": False},
        {"shape": "mesh"},
        {"file": "x"},
        {"pos": [float("nan"), 0]},
        {"pos": [0, 101]},
        {"pos": [0, True]},
        {"version": True},
        {"version": "2"},
    ],
)
def test_invalid_request_cannot_change_saved_or_live_world(pg, patch):  # noqa: F811
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="invalid object placement"):
        add(pg, **patch)
    assert pg.store.file.read_bytes() == before and pg.session.model is model


@pytest.mark.parametrize(
    "first,second",
    [("barrier", "barrier"), ("table", "table"), ("block", "barrier"), ("table", "ball"), ("pin", "pin")],
)
def test_static_static_and_dynamic_overlap_are_refused(pg, first, second):  # noqa: F811
    add(pg, first)
    # A ball inside the tabletop, rather than on its support surface.
    pos = [0, 0, 0.4] if first == "table" and second == "ball" else [0, 0, 0]
    before, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="occupied"):
        add(pg, second, pos)
    assert pg.store.file.read_bytes() == before and pg.session.model is model


def test_tabletop_touch_and_block_stacking_are_allowed(pg):  # noqa: F811
    add(pg, "table")
    add(pg, "block", [0, 0, 0.5])
    add(pg, "block", [0, 0, 0.68])
    add(pg, "ball", [0.25, 0, 0.5])
    assert len(pg.store.load().objects) == 4


@pytest.mark.parametrize("preset", list(OBJECTS))
def test_nondefault_preset_additions_can_be_undone(pg, preset):  # noqa: F811
    reply = add(pg, preset, [0, 0, 0], size="large", yaw=270)
    flags = pg.session.model.opt.disableflags
    undo(pg, reply)
    assert not pg.store.load().objects
    assert pg.session.model.opt.disableflags == flags


def test_coincident_cylinder_box_is_refused_and_candidate_options_restore(pg):  # noqa: F811
    from robotforge.viewer.creation import _object_obstacles

    pg.add_object("pin", shape="cylinder", size=[0.2, 0.2], fixed=True, pos=[0, 0, 0.1])
    pg.add_object("block", shape="box", size=[0.2, 0.2, 0.2], fixed=True, pos=[0, 0, 0.1])
    flags = pg.session.model.opt.disableflags
    assert _object_obstacles(pg.session.model, pg.session.data, "block") == {"pin": pytest.approx(0.2)}
    assert pg.session.model.opt.disableflags == flags


def test_prop_add_and_undo_preserve_unrelated_poses_even_during_deep_contact(pg):  # noqa: F811
    for name, x in [("a", -2), ("b", 2)]:
        pg.add_object(name, shape="sphere", size=[0.2], pos=[x, 0])
    pg.session.data.qpos[0] = 0
    pg.session.data.qpos[7] = 0.15
    mujoco.mj_forward(pg.session.model, pg.session.data)
    before = pg.session.data.qpos.copy()
    reply = add(pg, "block", [4, 0])
    np.testing.assert_array_equal(pg.session.data.qpos[:14], before)
    undo(pg, reply)
    np.testing.assert_array_equal(pg.session.data.qpos, before)


@pytest.mark.parametrize("operation", ["add", "undo"])
def test_external_write_during_final_compile_is_not_overwritten(pg, monkeypatch, operation):  # noqa: F811
    from robotforge.design import store as store_module

    reply = add(pg)
    original = store_module.compile_world
    external = pg.store.file.read_bytes() + b"\n# concurrent editor\n"

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        pg.store.file.write_bytes(external)
        return result

    monkeypatch.setattr(store_module, "compile_world", changed)
    model = pg.session.model
    with pytest.raises(ValueError, match="changed during compilation"):
        add(pg, "block", [2, 0]) if operation == "add" else undo(pg, reply)
    assert pg.store.file.read_bytes() == external and pg.session.model is model


def test_saved_live_and_during_compile_movement_are_checked(pg, monkeypatch):  # noqa: F811
    from robotforge.design import world as world_module

    pg.add_robot("rover", template="rover")
    with pg.session.lock:
        pg.session.data.qpos[0] = 2
        mujoco.mj_forward(pg.session.model, pg.session.data)
    for x in [0, 2]:
        with pytest.raises(PlaygroundError, match="occupied by rover"):
            add(pg, "barrier", [x, 0])
    original = world_module.compile_world

    def moving(*args, **kwargs):
        result = original(*args, **kwargs)
        pg.session.data.qpos[0] = 4
        mujoco.mj_forward(pg.session.model, pg.session.data)
        return result

    monkeypatch.setattr(world_module, "compile_world", moving)
    before = pg.store.file.read_bytes()
    with pytest.raises(PlaygroundError, match="occupied by rover"):
        add(pg, "barrier", [4, 0])
    assert pg.store.file.read_bytes() == before and pg.session.data.qpos[0] == 4


def test_undo_removes_only_added_entity_and_preserves_others_live_pose(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    reply = add(pg)
    pg.session.data.qpos[0] = -3
    pg.session.step(10)
    before = pg.session.data.qpos[:7].copy()
    result = undo(pg, reply)
    assert result["removed"] == "ball" and not pg.store.load().objects
    np.testing.assert_array_equal(pg.session.data.qpos[:7], before)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, reply)


@pytest.mark.parametrize("change", ["reset", "edit", "store", "lesson", "disk"])
def test_undo_refuses_invalidated_receipts_without_discarding_other_edits(pg, change):  # noqa: F811
    reply = add(pg)
    store = pg.store
    if change == "reset":
        pg.session.reset()
    elif change == "edit":
        pg.add_object("other", pos=[2, 0])
    elif change == "store":
        pg.store = None
    elif change == "lesson":
        pg.lessons.start("pid_twins")
    else:
        store.file.write_text(store.file.read_text(encoding="utf-8") + "\n# external change\n", encoding="utf-8")
    before, model = store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="simulation changed"):
        undo(pg, reply)
    assert store.file.read_bytes() == before and pg.session.model is model


def test_wrong_receipt_cannot_consume_valid_receipt_and_compile_failure_is_retryable(pg, monkeypatch):  # noqa: F811
    from robotforge.design import store as store_module

    reply = add(pg)
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, reply, token="0" * 32)
    original = store_module.compile_world

    def broken(*args, **kwargs):
        raise ValueError("compile failed")

    monkeypatch.setattr(store_module, "compile_world", broken)
    before = pg.store.file.read_bytes()
    with pytest.raises(ValueError, match="compile failed"):
        undo(pg, reply)
    assert pg.store.file.read_bytes() == before
    monkeypatch.setattr(store_module, "compile_world", original)
    undo(pg, reply)


def test_later_client_add_invalidates_prior_receipt_and_admission_is_enforced(pg):  # noqa: F811
    first = add(pg)
    second = add(pg, "block", [2, 0])
    with pytest.raises(PlaygroundError, match="no longer"):
        undo(pg, first)
    with pg.session.execution.run("test", RunLimits()), pytest.raises(ExecutionError):
        undo(pg, second)
    undo(pg, second)
    assert [o.name for o in pg.store.load().objects] == ["ball"]


def test_addition_catalog_and_actual_websocket_undo(pg):  # noqa: F811
    assert [e["name"] for e in viewer_catalog()["object_presets"]] == list(OBJECTS)
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1:8737") as client,
        client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws,
    ):
        reply = None
        for ident, payload in [
            (1, {"action": "add_object", "preset": "pin", "pos": [2, 0, 0], "version": pg.session.version}),
            (2, None),
        ]:
            if ident == 2:
                payload = {"action": "undo_add", **{k: reply["undo"][k] for k in ["token", "version"]}}
            ws.send_text(json.dumps({"id": ident, "type": "world", **payload}))
            for _ in range(400):
                msg = ws.receive()
                data = json.loads(msg["text"]) if msg.get("text") else {}
                if data.get("type") == "reply" and data["id"] == ident:
                    assert data["ok"], data
                    reply = data["data"]
                    break
            else:
                pytest.fail("missing reply")
    assert not pg.store.load().objects
