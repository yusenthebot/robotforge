"""Persistent records have a closed primitive boundary and bounded safe disk access."""

from __future__ import annotations

import os
from datetime import UTC
from uuid import uuid4

import pytest

from robotforge.design import arrangements as arrangements


@pytest.fixture
def library_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTFORGE_HOME", str(tmp_path))
    return tmp_path / "arrangements"


def symlink_or_skip(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as exc:
        if os.name == "nt":
            pytest.skip(f"Windows symlink permission unavailable: {exc}")
        raise


def rename_or_skip(source, destination):
    try:
        source.rename(destination)
    except PermissionError as exc:
        if os.name == "nt":
            pytest.skip(f"Windows denied this directory-replacement race setup while a lock is open: {exc}")
        raise


def members(count=2):
    return [
        {
            "name": f"block{i}",
            "shape": "box",
            "size": [0.2, 0.2, 0.2],
            "pos": [i * 0.4, 0.0, 0.1],
            "yaw": 720.0,
            "color": [0.1, 0.2, 0.3, 1.0],
            "mass": None,
            "fixed": False,
            "material": "wood",
            "surface_material": "steel",
        }
        for i in range(count)
    ]


def test_immutable_global_records_unicode_names_restart_and_removal(library_home):
    with arrangements.locked() as saved:
        first = saved.save("  Cafe\u0301  ", members())
        raw = saved.read(first["id"])
        assert first["name"] == "Café"
        assert first["sha256"] == arrangements.sha(raw)
        assert saved.record(first["id"], first["sha256"])["members"] == members()
        with pytest.raises(arrangements.ArrangementError, match="already in use"):
            saved.save("CAFÉ", members())
        assert saved.read(first["id"]) == raw
    with arrangements.locked() as restarted:
        assert restarted.entries() == [first]
        restarted.remove(first["id"], first["sha256"])
        assert restarted.entries() == []


@pytest.mark.parametrize("shape,size", [("table", [1, 0.6, 0.7]), ("bin", [0.4, 0.3, 0.2])])
@pytest.mark.parametrize("finish", [None, "plastic", "wood", "steel"])
def test_furniture_visual_finishes_and_legacy_members_persist_without_physical_material(
    library_home, shape, size, finish
):
    selected = members()
    for member in selected:
        member.update(shape=shape, size=size)
        member.pop("material")
        member.pop("surface_material")
        if finish is not None:
            member["surface_material"] = finish
    with arrangements.locked() as saved:
        record = saved.save("Furniture", selected)
    with arrangements.locked() as reopened:
        assert reopened.record(record["id"], record["sha256"])["members"] == selected
        assert reopened.entries() == [record]


@pytest.mark.parametrize("shape", ["table", "bin"])
@pytest.mark.parametrize("material", ["plastic", "wood", "steel"])
def test_furniture_record_physical_material_stays_forbidden(shape, material):
    selected = {**members()[0], "shape": shape, "material": material, "surface_material": "wood"}
    with pytest.raises(arrangements.ArrangementError, match="unsupported material"):
        arrangements.validate_member(selected)


def test_two_to_sixteen_members_and_sixteen_record_limit(library_home):
    with arrangements.locked() as saved:
        for count in (2, 16):
            entry = saved.save(f"Count {count}", members(count))
            assert entry["count"] == count
        for count in (0, 1, 17):
            with pytest.raises(arrangements.ArrangementError):
                saved.save("Invalid count", members(count))
        for number in range(14):
            saved.save(f"Arrangement {number}", members())
        assert len(saved.entries()) == 16
        with pytest.raises(arrangements.ArrangementError, match="16"):
            saved.save("Overflow", members())


@pytest.mark.parametrize(
    "patch",
    [
        {"shape": "plugin_shape"},
        {"shape": "../box"},
        {"file": "robot.xml"},
        {"params": {}},
        {"seed": 1},
        {"speed": 0},
        {"strength": 1},
        {"resist": 1},
        {"geoms": [1]},
        {"size": [0.2, 0.2, float("inf")]},
        {"pos": [float("nan"), 0, 0]},
        {"yaw": 3601},
        {"mass": True},
        {"fixed": "yes"},
        {"color": [0.1, 0.2, 999]},
        {"material": "plugin"},
        {"surface_material": "chrome"},
        {"shape": "table", "material": "wood"},
    ],
)
def test_closed_schema_rejects_unknown_shapes_fields_and_nonfinite_values_before_objectentry(library_home, patch):
    from robotforge.design.world import ObjectEntry

    entries = members()
    entries[0].update(patch)
    with pytest.raises(arrangements.ArrangementError):
        arrangements.validate_member(entries[0])
    # No plugin/file parameter survives the closed boundary even if ObjectEntry accepts it elsewhere.
    assert "file" not in ObjectEntry.model_fields


@pytest.mark.parametrize("label", ["", " ", "a" * 65, "bad\nname", "bad\x00name"])
def test_invalid_names(library_home, label):
    with arrangements.locked() as saved, pytest.raises(arrangements.ArrangementError):
        saved.save(label, members())


def test_damaged_regular_record_stays_visible_and_is_explicitly_removable(library_home):
    with arrangements.locked() as saved:
        healthy = saved.save("Healthy", members())
        damaged_id = uuid4().hex
        raw = b'{"schema_version":1,"members":'
        (library_home / f"{damaged_id}.json").write_bytes(raw)
        rows = saved.entries()
        assert healthy in rows
        bad = next(row for row in rows if row["id"] == damaged_id)
        assert bad["damaged"] and bad["sha256"] == arrangements.sha(raw)
        assert (library_home / f"{damaged_id}.json").read_bytes() == raw
        with pytest.raises(arrangements.ArrangementError):
            saved.record(damaged_id, bad["sha256"])
        saved.remove(damaged_id, bad["sha256"])
        assert saved.entries() == [healthy]


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "oversized"])
def test_unsafe_record_never_followed_or_blocked_and_cannot_be_removed(library_home, tmp_path, kind):
    with arrangements.locked() as saved:
        victim = tmp_path / "victim"
        victim.write_bytes(b"untouched")
        record_id = uuid4().hex
        path = library_home / f"{record_id}.json"
        if kind == "symlink":
            symlink_or_skip(path, victim)
        elif kind == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("This platform does not provide FIFOs")
            os.mkfifo(path)
        elif kind == "directory":
            path.mkdir()
        else:
            path.write_bytes(b"x" * (arrangements.MAX_BYTES + 1))
        row = saved.entries()[0]
        assert row["damaged"] and row["sha256"] is None
        with pytest.raises((arrangements.ArrangementError, OSError)):
            saved.remove(record_id, "0" * 64)
        assert victim.read_bytes() == b"untouched" and path.exists()


def test_root_symlink_and_replacement_fail_closed(library_home, tmp_path):
    with arrangements.locked() as saved:
        saved.save("Sample", members())
        moved = tmp_path / "old_arrangements"
        rename_or_skip(library_home, moved)
        library_home.mkdir()
        with pytest.raises(arrangements.ArrangementError, match="changed"):
            saved.save("Fail", members())
    library_home.rmdir()
    symlink_or_skip(library_home, moved, directory=True)
    with pytest.raises(arrangements.ArrangementError), arrangements.locked():
        pass


def test_file_digest_cannot_remove_replacement_and_duplicate_json_fields_rejected(library_home):
    with arrangements.locked() as saved:
        entry = saved.save("Original", members())
        path = library_home / f"{entry['id']}.json"
        raw = path.read_bytes()
        path.write_bytes(raw + b" ")
        with pytest.raises(arrangements.ArrangementError, match="changed"):
            saved.remove(entry["id"], entry["sha256"])
        assert path.read_bytes() == raw + b" "
        duplicate = raw[:-1] + b',"schema_version":1}'
        with pytest.raises(arrangements.ArrangementError):
            arrangements.decode(duplicate, entry["id"])


def test_bounded_directory_scan_and_stage_fsync_cleanup(library_home, monkeypatch):
    with arrangements.locked() as saved:
        real_fsync = os.fsync
        calls = []

        def fail(descriptor):
            calls.append(descriptor)
            if len(calls) == 1:
                raise OSError("disk full")
            return real_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fail)
        with pytest.raises(OSError):
            saved.save("Failure", members())
        assert not list(library_home.glob("*.json"))
        assert not list(library_home.glob(".stage-*"))
        monkeypatch.setattr(os, "fsync", real_fsync)
        for number in range(arrangements.MAX_SCAN):
            (library_home / f".stage-{number}").touch()
        with pytest.raises(arrangements.ArrangementError, match="Too many"):
            saved.entries()


def test_externally_injected_seventeenth_record_fails_without_truncating_or_deleting(library_home):
    import json
    from datetime import datetime

    expected = {}
    with arrangements.locked() as saved:
        for index in range(17):
            record_id = uuid4().hex
            record = {
                "schema_version": 1,
                "id": record_id,
                "name": f"External {index}",
                "created_at": datetime.now(UTC).isoformat(),
                "members": members(),
            }
            raw = json.dumps(record, allow_nan=False).encode()
            path = library_home / f"{record_id}.json"
            path.write_bytes(raw)
            expected[path] = raw
        with pytest.raises(arrangements.ArrangementError, match="more than 16"):
            saved.entries()
        with pytest.raises(arrangements.ArrangementError, match="more than 16"):
            saved.save("Additional", members())
        assert {path: path.read_bytes() for path in library_home.glob("*.json")} == expected


def test_conflicting_os_lock_returns_busy_without_waiting(library_home):
    import time

    with arrangements.locked():
        started = time.monotonic()
        with pytest.raises(arrangements.ArrangementError) as error, arrangements.locked():
            pytest.fail("a second library operation acquired an occupied OS lock")
        assert error.value.code == "arrangement_busy"
        assert time.monotonic() - started < 1.0


def test_checked_path_adapter_simulation_without_posix_directory_apis(library_home, monkeypatch):
    """Exercise the fallback on the host; this does not certify execution on Windows."""
    monkeypatch.setattr(arrangements, "_DESCRIPTOR_IO", False)
    for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"):
        monkeypatch.delattr(os, flag, raising=False)
    real_open, real_link, real_unlink, real_scandir = os.open, os.link, os.unlink, os.scandir

    def path_open(path, flags, mode=0o777, **keywords):
        assert "dir_fd" not in keywords
        return real_open(path, flags, mode, **keywords)

    def path_link(source, destination, **keywords):
        assert not keywords
        return real_link(source, destination)

    def path_unlink(path, **keywords):
        assert not keywords
        return real_unlink(path)

    def path_scandir(path):
        assert not isinstance(path, int)
        return real_scandir(path)

    monkeypatch.setattr(os, "open", path_open)
    monkeypatch.setattr(os, "link", path_link)
    monkeypatch.setattr(os, "unlink", path_unlink)
    monkeypatch.setattr(os, "scandir", path_scandir)
    with arrangements.locked() as saved:
        assert saved.fd is None
        first = saved.save("Checked path", members())
        assert saved.entries() == [first]
        assert saved.record(first["id"], first["sha256"])["members"] == members()
        saved.remove(first["id"], first["sha256"])
        assert saved.entries() == []


def test_checked_path_adapter_rejects_root_replacement_and_unsafe_leaf(library_home, monkeypatch, tmp_path):
    monkeypatch.setattr(arrangements, "_DESCRIPTOR_IO", False)
    with arrangements.locked() as saved:
        entry = saved.save("Original", members())
        path = library_home / f"{entry['id']}.json"
        path.unlink()
        path.mkdir()
        with pytest.raises(arrangements.ArrangementError):
            saved.read(entry["id"])
        moved = tmp_path / "old_checked_library"
        rename_or_skip(library_home, moved)
        library_home.mkdir()
        with pytest.raises(arrangements.ArrangementError, match="changed"):
            saved.entries()


def test_windows_lock_adapter_simulation_uses_nonblocking_one_byte(library_home, monkeypatch, tmp_path):
    """Mock msvcrt at its boundary without claiming a Windows execution result."""
    import sys
    from types import SimpleNamespace

    path = tmp_path / "mock_windows_lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    calls = []

    def locking(fd, mode, count):
        calls.append((fd, mode, count, os.lseek(fd, 0, os.SEEK_CUR)))
        raise OSError("lock occupied")

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, locking=locking))
    try:
        with pytest.raises(arrangements.ArrangementError) as error:
            arrangements._lock_file(descriptor, windows=True)
        assert error.value.code == "arrangement_busy"
        assert calls == [(descriptor, 2, 1, 0)]
        assert os.fstat(descriptor).st_size == 1
    finally:
        os.close(descriptor)
