"""Camera shelves use source identity, independently of titles or scene contents."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

from robotforge.sim.session import SimSession
from robotforge.viewer import server

XML = "<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>"


def test_camera_scope_resolves_relative_and_symlink_sources_but_separates_same_names(tmp_path, monkeypatch):
    first = tmp_path / "a" / "world.xml"
    second = tmp_path / "b" / "world.xml"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text(XML, encoding="utf-8")
    alias = tmp_path / "alias.xml"
    alias.symlink_to(first)
    monkeypatch.chdir(tmp_path)
    scope = server._camera_scope(str(first))
    assert len(scope) == 64 and set(scope) <= set("0123456789abcdef")
    assert server._camera_scope("a/world.xml") == server._camera_scope(str(alias)) == scope
    assert server._camera_scope(str(second)) != scope
    first.write_text(XML + "\n", encoding="utf-8")
    assert server._camera_scope(str(first)) == scope


@pytest.mark.parametrize("source", ["", "<xml>", "<memory>", "lesson:test", "missing/world.xml", "bad\x00path"])
def test_camera_scope_does_not_invent_persistent_identity_for_transient_or_invalid_sources(source):
    assert server._camera_scope(source) is None


def test_camera_scope_unifies_verified_case_aliases_on_case_insensitive_volumes(tmp_path):
    path = tmp_path / "SavedScene.xml"
    path.write_text(XML, encoding="utf-8")
    alias = path.with_name("savedscene.xml")
    if not alias.exists() or not alias.samefile(path):
        pytest.skip("this filesystem is case-sensitive")
    assert server._camera_scope(str(alias)) == server._camera_scope(str(path))


def test_camera_scope_never_folds_an_unverified_case_alias(tmp_path, monkeypatch):
    path = tmp_path / "SavedScene.xml"
    path.write_text(XML, encoding="utf-8")
    expected = hashlib.sha256(
        b"robotforge.camera-view-scope.v1\0" + str(path.resolve()).encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    monkeypatch.setattr(Path, "samefile", Mock(return_value=False))
    assert server._camera_scope(str(path)) == expected


def test_scope_does_not_accept_directories_or_escape_missing_source_errors(tmp_path, monkeypatch):
    assert server._camera_scope(str(tmp_path)) is None
    monkeypatch.setattr(Path, "resolve", Mock(side_effect=PermissionError("unreadable")))
    assert server._camera_scope(str(tmp_path / "world.xml")) is None


def test_scene_scope_is_version_cached_and_captures_source_with_model_snapshot(tmp_path, monkeypatch):
    first = tmp_path / "one.xml"
    second = tmp_path / "two.xml"
    for path in (first, second):
        path.write_text(XML, encoding="utf-8")
    session = SimSession.from_path(first)
    hub = server.ViewerHub(session)
    expected = server._camera_scope(str(first))
    scoped = Mock(wraps=server._camera_scope)
    monkeypatch.setattr(server, "_camera_scope", scoped)
    assert hub.scene_message()["view_scope"] == expected
    assert hub.scene_message()["view_scope"] == expected
    assert scoped.call_count == 1
    # A normal reload gets a new model version and therefore a freshly bound source.
    session.load_path(second)
    message = hub.scene_message()
    assert message["view_scope"] == server._camera_scope(str(second)) != expected
    assert message["version"] == session.version
    assert hub.state_message()["source"] == str(second)


def test_model_description_source_is_captured_before_building_payload(tmp_path, monkeypatch):
    first = tmp_path / "one.xml"
    second = tmp_path / "two.xml"
    for path in (first, second):
        path.write_text(XML, encoding="utf-8")
    session = SimSession.from_path(first)
    hub = server.ViewerHub(session)
    describe = server.scene.describe

    def replace_source(*args, **kwargs):
        message = describe(*args, **kwargs)
        session.source = str(second)
        return message

    monkeypatch.setattr(server.scene, "describe", replace_source)
    assert hub.scene_message()["view_scope"] == server._camera_scope(str(first))


def test_memory_model_has_optional_null_scope_without_changing_other_scene_fields():
    session = SimSession.from_xml(XML)
    hub = server.ViewerHub(session)
    message = hub.scene_message()
    assert message["view_scope"] is None
    assert message["nbody"] == session.model.nbody
