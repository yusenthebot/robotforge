"""Durable authored layouts, guarded publication, file boundaries and failure recovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from starlette.testclient import TestClient

from robotforge.design import layouts
from robotforge.design.store import HISTORY_LIMIT
from robotforge.design.world import RobotEntry
from robotforge.playground import Playground
from robotforge.sim.execution import RunLimits
from robotforge.sim.session import SimSession, fresh_data
from robotforge.viewer.server import ViewerHub, create_app


@pytest.fixture
def pg(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    result = Playground(session=SimSession.from_xml("<mujoco/>"))
    result.new_world("layouts")
    result.add_robot("rover", template="rover", pos=[0, 0])
    result.add_object("barrier", shape="box", pos=[2, 0], size=[0.8, 0.12, 0.22], fixed=True)
    result.session.paused = True
    return result


def catalog(pg):
    return pg.layouts_for_play({"action": "layout_list"})


def guard(pg):
    data = catalog(pg)
    return {k: data[k] for k in ("version", "generation", "world_sha256")}


def save(pg, name="My course"):
    return pg.layouts_for_play({"action": "layout_save", **guard(pg), "name": name})["entries"][0]


def target(pg, entry, action="restore", token=None):
    return pg.layouts_for_play(
        {
            "action": "layout_" + action,
            **(guard(pg) if token is None else token),
            "layout_id": entry["id"],
            "sha256": entry["sha256"],
        }
    )


def disk(pg):
    return {
        "world": pg.store.file.read_bytes(),
        "xml": (pg.store.build_dir / "world.xml").read_bytes(),
        "history": {p.name: p.read_bytes() for p in pg.store.history.glob("*.yaml")},
    }


def test_save_does_not_reset_live_motion_or_change_current_files(pg):
    with pg.session.lock:
        pg.session.data.qpos[0] = 3
        pg.session.data.qvel[0] = 0.4
        pg.session.data.time = 1.2
        mujoco.mj_forward(pg.session.model, pg.session.data)
    before = disk(pg)
    model, version, generation = pg.session.model, pg.session.version, pg.session.generation
    entry = save(pg, "  我的路线  ")
    assert entry["name"] == "我的路线" and len(entry["id"]) == 32
    assert disk(pg) == before and pg.session.model is model
    assert (pg.session.version, pg.session.generation) == (version, generation)
    assert pg.session.data.time == 1.2 and pg.session.data.qpos[0] == 3 and pg.session.data.qvel[0] == 0.4


def test_restore_all_design_fields_at_authored_start_and_ordinary_undo(pg):
    pg.add_terrain("ramp", pos=[4, 0])
    original = pg.current_world().model_dump(mode="json")
    entry = save(pg)
    pg.load_scene("warehouse", seed=3)
    before = pg.current_world().model_dump(mode="json")
    with pg.session.lock:
        pg.session.data.qpos[0] = 4
        pg.session.data.qvel[0] = 1
        pg.session.data.time = 4
    version, generation = pg.session.version, pg.session.generation
    out = target(pg, entry)
    assert pg.current_world().model_dump(mode="json") == original
    assert out["version"] == version + 1 and out["generation"] == generation + 1
    assert pg.session.paused and pg.session.data.time == 0
    expected = fresh_data(pg.session.model)
    np.testing.assert_array_equal(pg.session.data.qpos, expected.qpos)
    np.testing.assert_array_equal(pg.session.data.qvel, expected.qvel)
    pg.undo()
    assert pg.current_world().model_dump(mode="json") == before
    assert catalog(pg)["entries"][0] == entry


def test_named_layout_survives_history_pruning_and_a_fresh_playground(pg):
    entry = save(pg)
    original = pg.current_world().model_dump(mode="json")
    path = pg.store.dir / ".layouts" / (entry["id"] + ".json")
    frozen = path.read_bytes()
    for i in range(HISTORY_LIMIT + 2):
        pg.store.apply(lambda w, i=i: setattr(w.floor, "friction", 0.8 if i % 2 else 1.2))
    assert len(list(pg.store.history.glob("*.yaml"))) == HISTORY_LIMIT
    other = Playground(session=SimSession.from_xml("<mujoco/>"))
    other.open_world(str(pg.store.dir))
    assert catalog(other)["entries"] == [entry]
    target(other, entry)
    assert other.current_world().model_dump(mode="json") == original and path.read_bytes() == frozen


def test_relative_model_keeps_the_original_world_asset_base(pg):
    path = pg.store.dir / "models"
    path.mkdir()
    xml = path / "local.xml"
    xml.write_text(
        '<mujoco><worldbody><body name="base"><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    pg._edit(lambda w: w.robots.append(RobotEntry(name="local", file="models/local.xml", pos=[-2, 0])))
    entry = save(pg)
    pg.load_scene("park", keep_robots=False)
    target(pg, entry)
    assert pg.current_world().robots[-1].file == "models/local.xml"
    assert "local/base" in [pg.session.model.body(i).name for i in range(pg.session.model.nbody)]
    assert pg.store.dir == path.parent


def test_missing_asset_fails_without_changing_current_design_or_model(pg):
    xml = pg.store.dir / "local.xml"
    xml.write_text('<mujoco><worldbody><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>', encoding="utf-8")
    pg._edit(lambda w: w.robots.append(RobotEntry(name="local", file="local.xml", pos=[-2, 0])))
    entry = save(pg)
    pg.load_scene("park", keep_robots=False)
    before, model = disk(pg), pg.session.model
    xml.unlink()
    with pytest.raises(layouts.LayoutError) as error:
        target(pg, entry)
    assert error.value.code == "layout_assets" and disk(pg) == before and pg.session.model is model


@pytest.mark.parametrize("value", ["", " " * 5, "a" * 65, "bad\nname", "bad\x00name", "bad\u202ename", True, 42])
def test_invalid_names_never_publish_a_record(pg, value):
    before, model = disk(pg), pg.session.model
    with pytest.raises(layouts.LayoutError):
        save(pg, value)
    assert disk(pg) == before and pg.session.model is model and not catalog(pg)["entries"]


def test_normalized_names_capacity_remove_and_immutable_saves(pg, monkeypatch):
    monkeypatch.setattr(layouts, "MAX_LAYOUTS", 2)
    entry = save(pg, "Cafe\u0301")
    path = pg.store.dir / ".layouts" / (entry["id"] + ".json")
    original = path.read_bytes()
    for name in [" Café ", "CAFÉ"]:
        with pytest.raises(layouts.LayoutError, match="already saved"):
            save(pg, name)
    other = save(pg, "Other")
    with pytest.raises(layouts.LayoutError, match="remove one"):
        save(pg, "Third")
    assert path.read_bytes() == original
    model, before = pg.session.model, disk(pg)
    target(pg, other, "remove")
    assert pg.session.model is model and disk(pg) == before and catalog(pg)["entries"] == [entry]
    save(pg, "Third")
    assert len(catalog(pg)["entries"]) == 2


@pytest.mark.parametrize(
    "patch",
    [
        {"version": True},
        {"generation": -1},
        {"world_sha256": "bad"},
        {"path": "../somewhere"},
        {"file": "any.xml"},
        {"name": None},
    ],
)
def test_request_schema_forbids_arbitrary_paths_and_bad_tokens(pg, patch):
    request = {"action": "layout_save", **guard(pg), "name": "Test", **patch}
    with pytest.raises(layouts.LayoutError) as exc:
        pg.layouts_for_play(request)
    assert exc.value.code == "layout_invalid_request" and not catalog(pg)["entries"]


def test_stale_generation_version_world_and_record_are_refused(pg):
    entry = save(pg)
    token = guard(pg)
    pg.session.reset()
    with pytest.raises(layouts.LayoutError, match="changed"):
        target(pg, entry, token=token)
    token = guard(pg)
    pg.add_object("ball", shape="sphere", pos=[3, 1], size=[0.15])
    with pytest.raises(layouts.LayoutError, match="changed"):
        target(pg, entry, token=token)
    with pytest.raises(layouts.LayoutError, match="saved layout changed"):
        target(pg, {**entry, "sha256": "0" * 64})
    token = guard(pg)
    pg.store.file.write_bytes(pg.store.file.read_bytes() + b"\n# external edit\n")
    with pytest.raises(layouts.LayoutError, match="changed"):
        target(pg, entry, token=token)


@pytest.mark.parametrize("change", ["reset", "world", "checkpoint"])
def test_compile_time_changes_do_not_commit(pg, monkeypatch, change):
    entry = save(pg)
    pg.load_scene("warehouse")
    before, model = disk(pg), pg.session.model
    compile_world = layouts.compile_world

    def interfere(*args):
        result = compile_world(*args)
        if change == "reset":
            pg.session.reset()
        elif change == "world":
            pg.store.file.write_bytes(before["world"] + b"\n# concurrent\n")
        else:
            p = pg.store.dir / ".layouts" / (entry["id"] + ".json")
            p.write_bytes(p.read_bytes() + b" ")
        return result

    monkeypatch.setattr(layouts, "compile_world", interfere)
    with pytest.raises(layouts.LayoutError, match="changed"):
        target(pg, entry)
    assert pg.session.model is model
    assert (pg.store.build_dir / "world.xml").read_bytes() == before["xml"]
    assert pg.store.file.read_bytes() == before["world"] + (b"\n# concurrent\n" if change == "world" else b"")


@pytest.mark.parametrize(
    "failure", ["stage", "history_link", "xml_replace", "world_replace", "load_before", "load_after"]
)
def test_failed_restore_preserves_design_build_history_and_live_data(pg, monkeypatch, failure):
    entry = save(pg)
    pg.load_scene("warehouse")
    before, model, data = disk(pg), pg.session.model, pg.session.data
    initial = data.qpos.copy()
    version, generation = pg.session.version, pg.session.generation
    if failure == "stage":
        monkeypatch.setattr(layouts, "_stage", lambda *_: (_ for _ in ()).throw(OSError("injected stage")))
    elif failure == "history_link":
        monkeypatch.setattr(layouts.os, "link", lambda *_: (_ for _ in ()).throw(OSError("injected history")))
    elif failure.endswith("replace"):
        replace, failed = layouts.os.replace, False

        def replace_once(source, target):
            nonlocal failed
            wanted = pg.store.file if failure == "world_replace" else pg.store.build_dir / "world.xml"
            if Path(target) == wanted and not failed:
                failed = True
                raise OSError("injected replace")
            return replace(source, target)

        monkeypatch.setattr(layouts.os, "replace", replace_once)
    else:
        load = pg._load_world

        def broken(compiled):
            if failure == "load_after":
                load(compiled)
            raise ValueError("injected install")

        monkeypatch.setattr(pg, "_load_world", broken)
    with pytest.raises((layouts.LayoutError, ValueError), match="injected"):
        target(pg, entry)
    assert disk(pg) == before
    assert pg.session.model is model and pg.session.data is data
    assert (pg.session.version, pg.session.generation) == (version, generation)
    np.testing.assert_array_equal(data.qpos, initial)
    assert not list(pg.store.dir.rglob(".layout-*.tmp"))
    assert catalog(pg)["available"]


def test_damaged_oversized_and_symlink_records_fail_closed(pg, tmp_path, monkeypatch):
    entry = save(pg)
    path = pg.store.dir / ".layouts" / (entry["id"] + ".json")
    raw = path.read_bytes()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(layouts.LayoutError, match="damaged"):
        catalog(pg)
    path.write_bytes(raw)
    monkeypatch.setattr(layouts, "MAX_RECORD_BYTES", len(raw) - 1)
    with pytest.raises(layouts.LayoutError, match="size"):
        catalog(pg)
    monkeypatch.setattr(layouts, "MAX_RECORD_BYTES", len(raw) + 1)
    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(raw)
    path.symlink_to(outside)
    with pytest.raises(layouts.LayoutError):
        catalog(pg)
    assert outside.read_bytes() == raw


def test_active_execution_lessons_and_plain_models(pg):
    with pg.session.execution.run("test", RunLimits()), pytest.raises(layouts.LayoutError) as error:
        save(pg)
    assert error.value.code == "layout_busy"
    pg.lessons.start("pid_twins")
    assert catalog(pg)["reason"] == "layout_lesson"
    pg.lessons.exit()
    pg.store = None
    assert catalog(pg)["reason"] == "layout_no_world"


def test_no_periodic_layout_reads_and_structured_compatible_websocket_error(pg, monkeypatch):
    hub = ViewerHub(pg.session)
    pg.attach_viewer(hub)
    monkeypatch.setattr(layouts, "list_layouts", lambda _: (_ for _ in ()).throw(AssertionError("unexpected listing")))
    hub.scene_message()
    hub.robot_stream()
    hub.robot_stream()
    with (
        TestClient(create_app(hub), base_url="http://127.0.0.1") as client,
        client.websocket_connect("ws://127.0.0.1/ws") as ws,
    ):
        ws.send_json({"type": "world", "action": "layout_save", "id": 121, "name": "Test", "version": False})
        while True:
            frame = ws.receive()
            if frame.get("text"):
                message = json.loads(frame["text"])
                if message.get("type") == "reply":
                    break
        assert message["id"] == 121 and not message["ok"] and message["code"] == "layout_invalid_request"
        assert "message" in message


def test_failed_model_install_restores_sensors_mapping_flow_and_bound_task_container(pg, monkeypatch):
    entry = save(pg)
    pg.add_sensor("rover", {"type": "lidar", "name": "scan"})
    pg.add_sensor("rover", {"type": "odometry", "name": "odom"})
    pg.perception.start("rover", {"lidar": "scan", "odometry": "odom"})
    pg.flow.update("Course", [{"id": 0, "parent": -1, "name": "drive"}], "r", {"0": "moving"})
    pg.behaviors.move("rover", 0.2, 0)
    pg.session.step(20)
    pg._task_starts["marker"] = marker = object()
    tasks, suite, mapper = pg._task_starts, pg.sensors.suite, pg.perception.mapper("rover")
    controller = pg.session.controllers["rover"]
    flow, before = pg.flow.message(), disk(pg)
    data, hooks = pg.session.data, tuple(pg.session.step_hooks)
    load = pg._load_world

    def fail_after_load(compiled):
        load(compiled)
        raise ValueError("after real model installation")

    monkeypatch.setattr(pg, "_load_world", fail_after_load)
    with pytest.raises(ValueError, match="real model"):
        target(pg, entry)
    assert disk(pg) == before and pg.session.data is data
    assert pg.sensors.suite is suite and pg.perception.mapper("rover") is mapper
    assert pg._task_starts is tasks and tasks["marker"] is marker
    assert pg.flow.message() == flow and tuple(pg.session.step_hooks) == hooks
    assert pg.session.controllers["rover"] is controller
    pg.session.step(20)
    assert pg.read_sensors("rover")[0]["sensors"]
    assert not pg.perception.failed
    pg.session.reset()
    assert pg._task_starts is tasks and not tasks
    pg.behaviors.move("rover", 0.2, 0)
    pg.session.step(20)
    assert pg.session.data.time > 0


@pytest.mark.parametrize(
    "kind", ["trip", "errand", "reset_hook", "controller_hook", "experiment", "recording", "replay"]
)
def test_irreversible_runtime_effects_are_rejected_before_disk_changes(pg, kind):
    from robotforge.experiments.live import LiveExperiments

    entry = save(pg)
    before, model = disk(pg), pg.session.model
    if kind == "trip":
        pg.behaviors._live_trips["rover"] = SimpleNamespace(done=False)
    elif kind == "errand":
        pg.behaviors._errands["rover"] = object()
    elif kind == "reset_hook":
        pg.session.reset_hooks.append(lambda: pytest.fail("custom hook ran"))
    elif kind == "controller_hook":
        pg.session.controller_hooks.append(lambda *_: pytest.fail("custom observer ran"))
    elif kind == "experiment":
        LiveExperiments(pg.session).active[1] = object()
    elif kind == "recording":
        pg.demonstrations.recordings["rover"] = SimpleNamespace(active=True)
    else:
        pg.demonstrations.replays["rover"] = SimpleNamespace(done=False)
    with pytest.raises(layouts.LayoutError) as error:
        target(pg, entry)
    assert error.value.code == "layout_runtime" and disk(pg) == before and pg.session.model is model


def test_idle_experiment_observer_and_normal_drive_can_restore(pg):
    from robotforge.experiments.live import LiveExperiments

    observer = LiveExperiments(pg.session)
    entry = save(pg)
    pg.behaviors.move("rover", 0.2, 0)
    target(pg, entry)
    assert pg.session.data.time == 0 and not observer.active and not observer.errors


def test_cleanup_failure_after_commit_is_a_warning_and_keeps_source_and_live_aligned(pg, monkeypatch):
    entry = save(pg)
    authored = pg.store.file.read_bytes()
    pg.load_scene("warehouse")
    unlink = Path.unlink

    def fail_temp(path, *args, **kwargs):
        if path.name.startswith(".layout-") and path.exists():
            raise OSError("injected temporary cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp)
    result = target(pg, entry)
    assert any("temporary" in note for note in result["notes"])
    assert pg.store.file.read_bytes() == authored
    assert pg.session.paused and pg.session.data.time == 0 and catalog(pg)["available"]


@pytest.mark.parametrize("artifact", ["world", "xml", "history"])
def test_disk_rollback_failure_pauses_and_invalidates_design_token(pg, monkeypatch, artifact):
    entry = save(pg)
    pg.load_scene("warehouse")
    replace = layouts.os.replace
    writes = 0

    def fail_second_world_replace(source, destination):
        nonlocal writes
        wanted = pg.store.file if artifact == "world" else pg.store.build_dir / "world.xml"
        if Path(destination) == wanted:
            writes += 1
            if writes == 2:
                raise OSError("injected rollback failure")
        return replace(source, destination)

    monkeypatch.setattr(layouts.os, "replace", fail_second_world_replace)
    if artifact == "history":
        unlink = Path.unlink

        def fail_history(path, *args, **kwargs):
            if path.parent == pg.store.history and path.suffix == ".yaml":
                raise OSError("injected history rollback failure")
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_history)
    monkeypatch.setattr(pg, "_load_world", lambda *_: (_ for _ in ()).throw(ValueError("load failed")))
    with pytest.raises(layouts.LayoutError) as error:
        target(pg, entry)
    assert error.value.code == "layout_recovery" and pg.session.paused and pg._play_world_sha256 is None
    with pytest.raises(layouts.LayoutError, match="changed"):
        catalog(pg)


def test_save_publication_failure_never_creates_a_visible_checkpoint(pg, monkeypatch):
    before = disk(pg)
    monkeypatch.setattr(layouts.os, "link", lambda *_: (_ for _ in ()).throw(OSError("publication failed")))
    with pytest.raises(layouts.LayoutError, match="publication failed"):
        save(pg)
    assert not catalog(pg)["entries"] and disk(pg) == before
    assert not list(pg.store.dir.rglob(".layout-*.tmp"))


def test_source_limit_and_nonblocking_lock_refuse_without_publication(pg, monkeypatch):
    raw = pg.store.file.read_bytes()
    monkeypatch.setattr(layouts, "MAX_WORLD_BYTES", len(raw) - 1)
    with pytest.raises(layouts.LayoutError, match="size"):
        save(pg)
    monkeypatch.setattr(layouts, "MAX_WORLD_BYTES", len(raw))
    with layouts._locked(pg.store), pytest.raises(layouts.LayoutError) as error:
        save(pg)
    assert error.value.code == "layout_busy" and not catalog(pg)["entries"]
