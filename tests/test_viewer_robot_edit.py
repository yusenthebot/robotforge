"""Saved/live robot-start placement, exact identity, protected publication and one-use undo."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401

from robotforge.design import layouts
from robotforge.playground import PlaygroundError
from robotforge.sim import carry
from robotforge.viewer.robot_edit import metadata


def request(pg, name="rover", **changes):  # noqa: F811
    info = metadata(pg)
    return {
        "action": "edit_robot_start",
        "name": name,
        "version": pg.session.version,
        "generation": pg.session.generation,
        "world_sha256": info["world_sha256"],
        "pos": [2.0, 0.0, 0.0],
        "yaw": 90,
        **changes,
    }


def undo(pg, result):  # noqa: F811
    return pg.edit_robot_start_for_play({"action": "undo_robot_start", **result["undo_robot_start"]})


@pytest.mark.parametrize("template", ["rover", "arm", "quadruped", "humanoid"])
def test_saved_start_preserves_configuration_and_reset(pg, template):  # noqa: F811
    pg.add_robot("rover", template=template, pos=[-2, 0])
    before = pg.store.load().robots[0]
    result = pg.edit_robot_start_for_play(request(pg))
    after = pg.store.load().robots[0]
    assert before.model_dump(exclude={"pos", "yaw"}) == after.model_dump(exclude={"pos", "yaw"})
    assert after.pos == [2, 0] and after.yaw == 90
    assert result["robot"] == "rover" and pg.session.paused
    initial = pg.session.data.qpos.copy()
    pg.session.step(10)
    pg.session.reset()
    np.testing.assert_array_equal(pg.session.data.qpos, initial)
    assert len(list(pg.store.history.glob("*.yaml"))) >= 1


@pytest.mark.parametrize(
    "change",
    [
        {"name": "rove"},
        {"name": "rover/mounted"},
        {"name": "../../rover"},
        {"version": True},
        {"version": "2"},
        {"generation": None},
        {"generation": True},
        {"world_sha256": "x"},
        {"pos": [0]},
        {"pos": [float("inf"), 0]},
        {"pos": [101, 0]},
        {"pos": [0, True]},
        {"yaw": True},
        {"yaw": float("nan")},
        {"yaw": 3601},
        {"params": {}},
        {"file": "other.xml"},
    ],
)
def test_invalid_target_or_parameters_never_publish(pg, change):  # noqa: F811
    pg.add_robot("rover", template="rover")
    model, data, raw = pg.session.model, pg.session.data, pg.store.file.read_bytes()
    with pytest.raises((PlaygroundError, layouts.LayoutError)):
        pg.edit_robot_start_for_play(request(pg, **change))
    assert pg.session.model is model and pg.session.data is data and pg.store.file.read_bytes() == raw


def test_undo_restores_complete_target_pose_joints_and_controls(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    with pg.session.lock:
        pg.session.data.qpos[0] = -1.8
        for j in range(pg.session.model.njnt):
            if pg.session.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                pg.session.data.qpos[pg.session.model.jnt_qposadr[j]] = 0.3
        pg.session.data.ctrl[:] = 0.2
        mujoco.mj_forward(pg.session.model, pg.session.data)
        before = carry.take(pg.session.model, pg.session.data)
        world = pg.store.load().model_dump()
    result = pg.edit_robot_start_for_play(request(pg))
    undo(pg, result)
    restored = carry.take(pg.session.model, pg.session.data)
    for field in ("free", "joints", "controls"):
        left, right = getattr(before, field), getattr(restored, field)
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
    assert pg.store.load().model_dump() == world
    assert pg.session.paused and np.all(pg.session.data.qvel == 0)
    with pytest.raises(PlaygroundError, match="no longer available"):
        undo(pg, result)


def test_other_owners_keep_current_pose_and_control_with_edit_velocity_semantics(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_robot("other", template="arm", pos=[0, 2])
    pg.add_object("ball", shape="sphere", pos=[0, -2])
    with pg.session.lock:
        pg.session.data.qpos[pg.session.model.joint("ball").qposadr[0]] = 0.4
        joint = pg.session.model.joint("other/shoulder")
        pg.session.data.qpos[joint.qposadr[0]] = 0.25
        pg.session.data.ctrl[:] = 0.13
        pg.session.data.qvel[:] = 1
        mujoco.mj_forward(pg.session.model, pg.session.data)
        previous = carry.take(pg.session.model, pg.session.data)
    pg.edit_robot_start_for_play(request(pg))
    current = carry.take(pg.session.model, pg.session.data)
    for field in ("free", "joints", "controls"):
        for key, value in getattr(previous, field).items():
            if carry.owner(key) != "rover":
                np.testing.assert_array_equal(value, getattr(current, field)[key])
    assert np.all(pg.session.data.qvel == 0)


@pytest.mark.parametrize("x", [0.0, 2.0])
def test_both_saved_and_current_other_robot_starts_are_occupied(pg, x):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_robot("other", template="rover", pos=[0, 0])
    with pg.session.lock:
        free = next(
            j
            for j in range(pg.session.model.njnt)
            if pg.session.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and pg.session.model.body(int(pg.session.model.jnt_bodyid[j])).name.startswith("other/")
        )
        pg.session.data.qpos[pg.session.model.jnt_qposadr[free]] = 2
        mujoco.mj_forward(pg.session.model, pg.session.data)
    raw, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="occupied"):
        pg.edit_robot_start_for_play(request(pg, pos=[x, 0, 0]))
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


def test_fixed_arm_fits_real_tabletop_and_fake_height_rejects(pg):  # noqa: F811
    pg.add_robot("rover", template="arm", pos=[-2, 0])
    pg.add_object("table", shape="table", size=[2, 2, 0.8], pos=[1, 0], fixed=True)
    result = pg.edit_robot_start_for_play(request(pg, pos=[1, 0, 0.8]))
    assert pg.store.load().robots[0].pos[2] == pytest.approx(0.8)
    undo(pg, result)
    raw = pg.store.file.read_bytes()
    with pytest.raises(PlaygroundError, match="flat top"):
        pg.edit_robot_start_for_play(request(pg, pos=[1, 0, 1.5]))
    assert pg.store.file.read_bytes() == raw


@pytest.mark.parametrize(
    "shape,fixed,size,z", [("box", False, [2, 2, 0.8], 0.8), ("table", True, [0.1, 0.1, 0.8], 0.8)]
)
def test_dynamic_or_too_small_platform_rejects(pg, shape, fixed, size, z):  # noqa: F811
    pg.add_robot("rover", template="arm", pos=[-2, 0])
    pg.add_object("support", shape=shape, fixed=fixed, size=size, pos=[1, 0])
    raw, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError):
        pg.edit_robot_start_for_play(request(pg, pos=[1, 0, z]))
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


@pytest.mark.parametrize("event", ["reset", "disk", "world"])
def test_compile_races_cannot_publish(pg, monkeypatch, event):  # noqa: F811
    from robotforge.viewer import robot_edit

    pg.add_robot("rover", template="rover")
    original = robot_edit.compile_world
    request_before = request(pg)
    changed = {}

    def compile(*args, **kwargs):
        result = original(*args, **kwargs)
        if event == "reset":
            pg.session.reset()
        elif event == "disk":
            pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n")
        else:
            pg.new_world("another")
        changed.update(raw=pg.store.file.read_bytes(), model=pg.session.model)
        return result

    monkeypatch.setattr(robot_edit, "compile_world", compile)
    with pytest.raises(PlaygroundError, match="changed"):
        pg.edit_robot_start_for_play(request_before)
    assert pg.store.file.read_bytes() == changed["raw"] and pg.session.model is changed["model"]


@pytest.mark.parametrize("where", ["load", "sensors", "perception"])
def test_runtime_install_failure_restores_disk_history_and_runtime(pg, monkeypatch, where):  # noqa: F811
    pg.add_robot("rover", template="rover")
    raw, xml = pg.store.file.read_bytes(), (pg.store.build_dir / "world.xml").read_bytes()
    history = {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")}
    state = {key: getattr(pg.session, key) for key in ("model", "data", "version", "generation", "paused")}
    if where == "load":
        original = pg._load_world

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected")

        monkeypatch.setattr(pg, "_load_world", fail)
    else:
        owner = pg.sensors if where == "sensors" else pg.perception
        original = owner.rebuild

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("injected")

        monkeypatch.setattr(owner, "rebuild", fail)
    with pytest.raises(RuntimeError, match="injected"):
        pg.edit_robot_start_for_play(request(pg))
    assert pg.store.file.read_bytes() == raw and (pg.store.build_dir / "world.xml").read_bytes() == xml
    assert {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")} == history
    assert all(
        getattr(pg.session, key) is value if key in {"model", "data"} else getattr(pg.session, key) == value
        for key, value in state.items()
    )


def test_external_reset_and_later_edit_invalidate_undo(pg):  # noqa: F811
    pg.add_robot("rover", template="rover")
    result = pg.edit_robot_start_for_play(request(pg))
    pg.session.reset()
    with pytest.raises(PlaygroundError):
        undo(pg, result)
    result = pg.edit_robot_start_for_play(request(pg, pos=[-2, 0]))
    pg.add_object("later", shape="sphere", pos=[0, 2])
    with pytest.raises(PlaygroundError):
        undo(pg, result)


def test_active_custom_reset_hook_is_refused_before_publication(pg):  # noqa: F811
    pg.add_robot("rover", template="rover")
    raw = pg.store.file.read_bytes()
    pg.session.reset_hooks.append(lambda: None)
    with pytest.raises(PlaygroundError, match="custom reset hooks") as refused:
        pg.edit_robot_start_for_play(request(pg))
    assert refused.value.code == "robot_start_runtime"
    assert pg.store.file.read_bytes() == raw


def test_undo_preserves_manual_leg_control_after_an_actual_step(pg):  # noqa: F811
    pg.add_robot("rover", template="quadruped")
    pg.behaviors.let_go(["rover"])
    assert "rover" not in pg.session.controllers
    before = pg.session.data.ctrl.copy()
    result = pg.edit_robot_start_for_play(request(pg))
    assert "rover" in pg.session.controllers
    undo(pg, result)
    assert "rover" not in pg.session.controllers
    pg.session.step(1)
    assert "rover" not in pg.session.controllers
    np.testing.assert_array_equal(before, pg.session.data.ctrl)


def test_mounted_robot_and_added_sensor_move_as_one_unchanged_design(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", mounts=[{"template": "arm"}])
    pg.add_sensor("rover", {"type": "lidar", "name": "route_scan"})
    before = pg.store.load().robots[0]
    names = [entry["name"] for entry in metadata(pg)["robots"]]
    assert names == ["rover"]
    model = pg.session.model
    mounted_joints = [model.joint(j).name for j in range(model.njnt) if "/arm/" in model.joint(j).name]
    assert mounted_joints
    snapshot = carry.take(model, pg.session.data)
    result = pg.edit_robot_start_for_play(request(pg))
    after = pg.store.load().robots[0]
    assert after.model_dump(exclude={"pos", "yaw"}) == before.model_dump(exclude={"pos", "yaw"})
    assert all(pg.session.model.joint(name).id >= 0 for name in mounted_joints)
    undo(pg, result)
    restored = carry.take(pg.session.model, pg.session.data)
    for name in mounted_joints:
        np.testing.assert_array_equal(restored.joints[name], snapshot.joints[name])


def test_fixed_table_above_water_is_accepted_but_underwater_floor_is_not(pg):  # noqa: F811
    pg.add_robot("rover", template="arm", pos=[-2, 0])
    pg.add_object("table", shape="table", size=[2, 2, 0.8], pos=[1, 0], fixed=True)
    pg.set_floor(water={"level": 0.3, "area": [0, -1, 2, 1]})
    with pytest.raises(PlaygroundError, match="dry ground"):
        pg.edit_robot_start_for_play(request(pg, pos=[1, 0, 0]))
    result = pg.edit_robot_start_for_play(request(pg, pos=[1, 0, 0.8]))
    assert pg.store.load().robots[0].pos[2] == pytest.approx(0.8)
    undo(pg, result)


def test_movement_during_compilation_is_rechecked_against_current_poses(pg, monkeypatch):  # noqa: F811
    from robotforge.viewer import robot_edit

    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_robot("other", template="rover", pos=[-4, 0])
    original = robot_edit.compile_world
    raw, model = pg.store.file.read_bytes(), pg.session.model

    def compile(*args, **kwargs):
        result = original(*args, **kwargs)
        for j in range(model.njnt):
            body = model.body(int(model.jnt_bodyid[j])).name
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and body.startswith("other/"):
                pg.session.data.qpos[model.jnt_qposadr[j]] = 2
        mujoco.mj_forward(model, pg.session.data)
        return result

    monkeypatch.setattr(robot_edit, "compile_world", compile)
    with pytest.raises(PlaygroundError, match="occupied"):
        pg.edit_robot_start_for_play(request(pg))
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


@pytest.mark.parametrize("target", ["world.xml", "world.yaml"])
def test_disk_publish_failure_rolls_back_both_files_and_history(pg, monkeypatch, target):  # noqa: F811
    pg.add_robot("rover", template="rover")
    raw, xml = pg.store.file.read_bytes(), (pg.store.build_dir / "world.xml").read_bytes()
    history = {path.name: path.read_bytes() for path in pg.store.history.glob("*.yaml")}
    model, original, fired = pg.session.model, layouts.os.replace, False

    def fail(source, destination):
        nonlocal fired
        if destination.name == target and not fired:
            fired = True
            raise OSError("injected publish")
        return original(source, destination)

    monkeypatch.setattr(layouts.os, "replace", fail)
    with pytest.raises(OSError, match="injected publish"):
        pg.edit_robot_start_for_play(request(pg))
    assert fired and pg.session.model is model
    assert pg.store.file.read_bytes() == raw and (pg.store.build_dir / "world.xml").read_bytes() == xml
    assert {path.name: path.read_bytes() for path in pg.store.history.glob("*.yaml")} == history


def test_failed_xml_rollback_pauses_and_invalidates_even_after_yaml_recovers(pg, monkeypatch):  # noqa: F811
    pg.add_robot("rover", template="rover")
    raw, model = pg.store.file.read_bytes(), pg.session.model
    original, calls = layouts.os.replace, 0

    def fail(source, destination):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected XML rollback")
        return original(source, destination)

    def load(*args, **kwargs):
        raise RuntimeError("injected model install")

    monkeypatch.setattr(layouts.os, "replace", fail)
    monkeypatch.setattr(pg, "_load_world", load)
    with pytest.raises(PlaygroundError, match="rollback failed") as failed:
        pg.edit_robot_start_for_play(request(pg))
    assert failed.value.code == "robot_start_recovery"
    assert pg.session.model is model and pg.store.file.read_bytes() == raw
    assert pg.session.paused and pg._play_world_sha256 is None and metadata(pg)["robots"] == []


def test_failed_undo_does_not_consume_the_valid_receipt(pg, monkeypatch):  # noqa: F811
    pg.add_robot("rover", template="rover")
    result = pg.edit_robot_start_for_play(request(pg))
    receipt, raw = pg._play_robot_edit, pg.store.file.read_bytes()
    original = pg._load_world

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected undo")

    monkeypatch.setattr(pg, "_load_world", fail)
    with pytest.raises(RuntimeError, match="injected undo"):
        undo(pg, result)
    assert pg._play_robot_edit is receipt and pg.store.file.read_bytes() == raw
    monkeypatch.setattr(pg, "_load_world", original)
    undo(pg, result)
    assert pg.store.load().robots[0].pos == [0, 0]


def test_other_robot_odometry_estimate_survives_the_start_edit(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    pg.add_robot("other", template="rover", pos=[0, 3])
    pg.add_sensor("other", {"type": "odometry", "name": "route_odom"})
    sensor = pg.sensors.suite.sensors["other/route_odom"]
    sensor.estimate = np.array([0.02, 3.01, 0.15])
    sensor.traveled = 4.25
    estimate = sensor.estimate.copy()
    pg.edit_robot_start_for_play(request(pg))
    current = pg.sensors.suite.sensors["other/route_odom"]
    np.testing.assert_array_equal(current.estimate, estimate)
    assert current.traveled == 4.25


def test_robot_start_error_code_crosses_the_actual_websocket_boundary(pg):  # noqa: F811
    import json

    from starlette.testclient import TestClient

    from robotforge.viewer.server import ViewerHub, create_app

    pg.add_robot("rover", template="rover")
    message = {"type": "world", "id": 42, **request(pg)}
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    pg.session.reset_hooks.append(lambda: None)
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1:8737") as client,
        client.websocket_connect("/ws", headers={"host": "127.0.0.1:8737"}) as ws,
    ):
        ws.send_text(json.dumps(message))
        for _ in range(60):
            frame = ws.receive()
            reply = json.loads(frame["text"]) if frame.get("text") else {}
            if reply.get("type") == "reply" and reply.get("id") == 42:
                assert reply["ok"] is False and reply["code"] == "robot_start_runtime"
                assert "changing a robot start" in reply["message"]
                break
        else:
            pytest.fail("missing robot-start reply")
