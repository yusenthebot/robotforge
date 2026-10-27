"""Real-model source IDs use guarded publication and preserve existing API/state."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_viewer_creation import pg  # noqa: F401

from robotforge.design import combine, layouts
from robotforge.playground import PlaygroundError, viewer_catalog
from robotforge.sim import carry
from robotforge.viewer import model_create
from robotforge.viewer.progress import report, reporting


def saved(pg, name="local_arm", **fields):  # noqa: F811
    from robotforge.design.world import RobotEntry

    entry = RobotEntry(name="source", template="arm", **fields)
    combine.save(entry, name, pg.store.dir)
    return entry


def place(pg, name="local_arm", **fields):  # noqa: F811
    return pg.add_model_for_play(
        {
            "source": "design",
            "model": name,
            "pos": [2.0, 0.0],
            "version": pg.session.version,
            "generation": pg.session.generation,
            **fields,
        }
    )


def test_catalog_is_metadata_only_and_keeps_old_templates(pg, monkeypatch):  # noqa: F811
    saved(pg)
    monkeypatch.setattr(model_create, "compile_world", lambda *_: pytest.fail("catalog compiled a model"))
    monkeypatch.setattr(model_create.library, "model_dir", lambda *_: pytest.fail("catalog downloaded a model"))
    c = viewer_catalog()
    assert [x["name"] for x in c["robot_templates"]] == ["rover", "quadruped", "arm", "humanoid"]
    assert [(x["name"], x["source"]) for x in c["robot_models"]] == [
        ("local_arm", "design"),
        ("go2", "library"),
        ("panda", "library"),
    ]
    assert "file" not in json.dumps(c["robot_models"])


def test_saved_model_retains_configuration_live_state_and_existing_undo_api(pg):  # noqa: F811
    pg.add_robot("rover", template="rover", pos=[-2, 0])
    original = saved(pg, params={"color": "red"}, sensors=[{"type": "lidar", "name": "scan", "rays": 8}])
    pg.session.data.qpos[0] += 0.1
    before = carry.take(pg.session.model, pg.session.data)
    events = []
    with reporting(events.append):
        result = place(pg)
    entry = pg.store.load().robots[-1]
    assert entry.model_dump(exclude={"name", "pos"}) == original.model_dump(exclude={"name", "pos"})
    after = carry.take(pg.session.model, pg.session.data)
    for group in ["free", "joints", "controls"]:
        for n, v in getattr(before, group).items():
            np.testing.assert_array_equal(v, getattr(after, group)[n])
    assert pg.session.paused and np.all(pg.session.data.qvel == 0)
    assert [x["stage"] for x in events] == ["prepare", "compile", "publish"]
    pg.object_for_play({"action": "undo_add", "token": result["undo"]["token"], "version": result["version"]})
    assert [r.name for r in pg.store.load().robots] == ["rover"]


@pytest.mark.parametrize(
    "patch",
    [
        {"file": "/arbitrary.urdf"},
        {"source": "catalogue"},
        {"model": "../escape"},
        {"source": "library", "model": "not_allowed"},
        {"version": True},
        {"generation": False},
        {"pos": [float("nan"), 0.0]},
        {"params": {}},
        {"yaw": 90},
    ],
)
def test_untrusted_source_inputs_never_mutate_the_world(pg, patch):  # noqa: F811
    saved(pg)
    raw = pg.store.file.read_bytes()
    model = pg.session.model
    with pytest.raises(ValueError):
        place(pg, **patch)
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


@pytest.mark.parametrize("phase", ["compile", "publish", "runtime"])
def test_failed_model_publication_keeps_files_runtime_and_receipt(pg, monkeypatch, phase):  # noqa: F811
    saved(pg)
    place(pg)
    receipt = pg._play_addition
    raw = pg.store.file.read_bytes()
    model, data = pg.session.model, pg.session.data
    if phase == "compile":
        monkeypatch.setattr(model_create, "compile_world", lambda *_: (_ for _ in ()).throw(ValueError("compile fail")))
    elif phase == "publish":
        original = layouts.os.replace

        def replace(source, target):
            if target == pg.store.file:
                raise OSError("publish fail")
            return original(source, target)

        monkeypatch.setattr(layouts.os, "replace", replace)
    else:
        original = pg._load_world

        def load(compiled):
            original(compiled)
            raise ValueError("runtime fail")

        monkeypatch.setattr(pg, "_load_world", load)
    with pytest.raises((ValueError, OSError, layouts.LayoutError)):
        place(pg, pos=[4.0, 0.0])
    assert pg.store.file.read_bytes() == raw and pg.session.model is model and pg.session.data is data
    assert pg._play_addition is receipt


def test_stale_download_completion_never_publishes(pg, monkeypatch):  # noqa: F811
    raw = pg.store.file.read_bytes()
    model = pg.session.model

    def download(*_):
        pg.session.reset()

    monkeypatch.setattr(model_create.library, "model_dir", download)
    with pytest.raises(PlaygroundError, match="world changed"):
        place(pg, source="library", name="go2")
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


def test_model_must_fit_fixed_table_and_live_space(pg):  # noqa: F811
    saved(pg)
    pg.add_object("table", shape="table", size=[2, 2, 0.8], pos=[2, 0], fixed=True)
    with pytest.raises(PlaygroundError, match="fit"):
        place(pg, pos=[2.95, 0.0, 0.8])
    place(pg, pos=[2.0, 0.0, 0.8])
    assert pg.store.load().robots[-1].pos == [2, 0, 0.8]
    with pytest.raises(PlaygroundError, match="occupied"):
        place(pg, pos=[2.0, 0.0, 0.8])


def test_reset_invalidates_undo_and_long_names_remain_valid(pg):  # noqa: F811
    name = "a" * 32
    saved(pg, name)
    place(pg, name)
    result = place(pg, name, pos=[4.0, 0.0])
    assert result["robot"] == "a" * 30 + "_2"
    pg.session.reset()
    with pytest.raises(PlaygroundError, match="world changed"):
        pg.object_for_play({"action": "undo_add", "token": result["undo"]["token"], "version": result["version"]})


def test_progress_callback_is_nested_and_operation_scoped():
    outer, inner = [], []
    report("outside")
    with reporting(outer.append):
        report("first")
        with reporting(inner.append):
            report("inner")
        report("last")
    report("outside")
    assert outer == [{"stage": "first"}, {"stage": "last"}] and inner == [{"stage": "inner"}]


def test_unpreservable_candidate_never_gets_an_undo_receipt(pg):  # noqa: F811
    from robotforge.design.world import RobotEntry

    src = pg.store.dir / "mocap.xml"
    src.write_text(
        '<mujoco><worldbody><body name="tracked" mocap="true"><geom size=".1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    combine.save(RobotEntry(name="source", file=str(src)), "tracked", pg.store.dir)
    raw, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(PlaygroundError, match="runtime state"):
        place(pg, "tracked")
    assert pg.store.file.read_bytes() == raw and pg.session.model is model and pg._play_addition is None


def test_saved_design_never_implicitly_downloads_dependencies(pg, monkeypatch):  # noqa: F811
    from robotforge.design import menagerie
    from robotforge.design.world import RobotEntry

    combine.save(RobotEntry(name="source", library="go2"), "my_go2", pg.store.dir)
    monkeypatch.setattr(menagerie, "local_folder", lambda _: None)
    raw, model = pg.store.file.read_bytes(), pg.session.model
    with pytest.raises(ValueError, match="not cached"):
        place(pg, "my_go2")
    assert pg.store.file.read_bytes() == raw and pg.session.model is model


def test_bound_download_feedback_survives_worker_threads_without_leaking():
    from concurrent.futures import ThreadPoolExecutor

    from robotforge.viewer.progress import download_callback

    events = []
    with reporting(events.append):
        callback = download_callback("download")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(callback, 0.5, "file").result()
    report("outside")
    assert events == [{"stage": "download", "fraction": 0.5}]
