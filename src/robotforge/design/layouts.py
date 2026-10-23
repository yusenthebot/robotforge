"""Durable named designs in the original world's asset directory, separate from transient undo."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import stat
import tempfile
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from robotforge.design.store import HISTORY_LIMIT, WorldStore
from robotforge.design.world import CompiledWorld, World, compile_world
from robotforge.extend import load_world_plugins

MAX_LAYOUTS = 16
MAX_WORLD_BYTES = 512 * 1024
MAX_RECORD_BYTES = 2 * MAX_WORLD_BYTES + 4096
ID = re.compile(r"^[0-9a-f]{32}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


class LayoutError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = "layout_" + code


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def layout_name(value: str) -> str:
    value = unicodedata.normalize("NFC", value.strip())
    if not 1 <= len(value) <= 64 or any(unicodedata.category(c).startswith("C") for c in value):
        raise LayoutError("invalid_name", "Use a layout name with 1 to 64 characters and no control characters")
    return value


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    id: str
    name: str
    created_at: str
    world_yaml: str
    world_sha256: str

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return layout_name(value)


def _root(store: WorldStore, create: bool = False) -> Path:
    root = store.dir / ".layouts"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise LayoutError("damaged", "The saved-layout directory is not a regular directory")
    if create:
        root.mkdir(exist_ok=True)
    return root


def _read(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise LayoutError("missing", "The saved design is unavailable; refresh the layouts")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise LayoutError("missing", "The saved design is not a regular file")
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise LayoutError("damaged", "The saved design exceeds the supported size")
    return raw


def _record(store: WorldStore, identity: str) -> tuple[Record, bytes]:
    if not ID.fullmatch(identity):
        raise LayoutError("invalid_request", "Invalid saved-layout identifier")
    raw = _read(_root(store) / (identity + ".json"), MAX_RECORD_BYTES)
    try:
        record = Record.model_validate_json(raw)
        if record.id != identity or not DIGEST.fullmatch(record.world_sha256):
            raise ValueError("invalid identity or digest")
        world = record.world_yaml.encode("utf-8")
        if len(world) > MAX_WORLD_BYTES or sha(world) != record.world_sha256:
            raise ValueError("layout content changed")
    except (ValidationError, ValueError, UnicodeError) as exc:
        raise LayoutError("damaged", f"Saved layout {identity} is damaged: {exc}") from exc
    return record, raw


def list_layouts(store: WorldStore) -> list[dict]:
    root = _root(store)
    records = []
    for p in sorted(root.glob("*.json")):
        if not ID.fullmatch(p.stem):
            continue
        if len(records) >= MAX_LAYOUTS:
            raise LayoutError("damaged", "Too many saved-layout records")
        record, raw = _record(store, p.stem)
        records.append({"id": record.id, "name": record.name, "sha256": sha(raw), "created_at": record.created_at})
    return sorted(records, key=lambda entry: (entry["created_at"], entry["id"]), reverse=True)


@contextlib.contextmanager
def _locked(store: WorldStore) -> Iterator[None]:
    """OS releases this lock on exit/crash; no stale lock-directory leases or extra dependency."""
    path = _root(store, create=True) / ".lock"
    if path.is_symlink():
        raise LayoutError("damaged", "The saved-layout lock is not a regular file")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise LayoutError("damaged", "The saved-layout lock is not a regular file")
        if os.name == "nt":
            import msvcrt

            if not os.fstat(fd).st_size:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise LayoutError("busy", "Another saved-layout operation is running; try again") from exc
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LayoutError("busy", "Another saved-layout operation is running; try again") from exc
        yield
    finally:
        os.close(fd)


def _stage(path: Path, data: bytes) -> Path:
    fd, name = tempfile.mkstemp(prefix=".layout-", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def save_layout(store: WorldStore, name: str, expected: str) -> dict:
    name = layout_name(name)
    with _locked(store):
        entries = list_layouts(store)
        if any(e["name"].casefold() == name.casefold() for e in entries):
            raise LayoutError("duplicate", "That layout name is already saved; choose another name")
        if len(entries) >= MAX_LAYOUTS:
            raise LayoutError("full", f"This world has {MAX_LAYOUTS} saved layouts; remove one before saving another")
        raw = _read(store.file, MAX_WORLD_BYTES)
        if sha(raw) != expected:
            raise LayoutError("changed", "The world changed; refresh and try again")
        World.from_yaml(raw.decode("utf-8"))
        identity = uuid4().hex
        record = Record(
            schema_version=1,
            id=identity,
            name=name,
            created_at=datetime.now(UTC).isoformat(),
            world_yaml=raw.decode("utf-8"),
            world_sha256=expected,
        )
        data = record.model_dump_json().encode("utf-8")
        if len(data) > MAX_RECORD_BYTES:
            raise LayoutError("damaged", "The saved design exceeds the supported size")
        path = _root(store) / (identity + ".json")
        temp = _stage(path, data)
        try:
            if sha(_read(store.file, MAX_WORLD_BYTES)) != expected:
                raise LayoutError("changed", "The world changed; refresh and try again")
            os.link(temp, path)  # immutable publication; never overwrite a saved creation
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove saved-layout temporary file %s", temp)
        return {"id": identity, "name": name, "sha256": sha(data), "created_at": record.created_at}


def remove_layout(store: WorldStore, identity: str, expected: str) -> None:
    with _locked(store):
        _, raw = _record(store, identity)
        if sha(raw) != expected:
            raise LayoutError("changed", "The saved layout changed; refresh before removing it")
        (_root(store) / (identity + ".json")).unlink()


@dataclass(frozen=True)
class PreparedLayout:
    id: str
    digest: str
    name: str
    before: bytes
    after: bytes
    compiled: CompiledWorld
    xml: bytes


def prepare_restore(store: WorldStore, identity: str, expected: str) -> PreparedLayout:
    record, raw = _record(store, identity)
    if sha(raw) != expected:
        raise LayoutError("changed", "The saved layout changed; refresh before restoring it")
    before = _read(store.file, MAX_WORLD_BYTES)
    try:
        load_world_plugins(store.dir)
        world = World.from_yaml(record.world_yaml)
        if world.name != World.from_yaml(before.decode("utf-8")).name:
            raise LayoutError("damaged", "This layout belongs to another world")
        compiled = compile_world(world, store.dir)
        xml = compiled.to_xml().encode("utf-8")
    except LayoutError:
        raise
    except (ValueError, OSError) as exc:
        raise LayoutError("assets", f"Could not prepare the saved layout: {exc}") from exc
    return PreparedLayout(identity, expected, record.name, before, record.world_yaml.encode("utf-8"), compiled, xml)


@contextlib.contextmanager
def install(store: WorldStore, prepared: PreparedLayout) -> Iterator[list[str]]:
    """Stage disk writes, publish authoritative YAML last, roll back on a failed model install.

    Individual replaces are atomic, not a crash-atomic multi-file transaction. XML is derived.
    A later undo-pruning failure is a reported warning, not a false failed-restore result.
    """
    with _locked(store):
        _, raw = _record(store, prepared.id)
        if sha(raw) != prepared.digest or _read(store.file, MAX_WORLD_BYTES) != prepared.before:
            raise LayoutError("changed", "The world or saved layout changed while preparing restoration")
        xml = store.build_dir / "world.xml"
        if xml.is_symlink() or store.history.is_symlink() or store.build_dir.is_symlink():
            raise LayoutError("damaged", "The world build/history directory is not a regular directory")
        store.build_dir.mkdir(exist_ok=True)
        store.history.mkdir(exist_ok=True)
        old_xml = xml.read_bytes() if xml.exists() else None
        staged: list[Path] = []
        taken = [int(p.stem) for p in store.history.glob("*.yaml") if p.stem.isdigit()]
        history = store.history / f"{max(time.time_ns(), max(taken, default=0) + 1)}.yaml"
        notes = []
        changed_xml = changed_world = saved_history = False
        try:

            def stage(path, data):
                temp = _stage(path, data)
                staged.append(temp)
                return temp

            new_xml = stage(xml, prepared.xml)
            new_world = stage(store.file, prepared.after)
            undo = stage(history, prepared.before)
            rollback_world = stage(store.file, prepared.before)
            rollback_xml = stage(xml, old_xml) if old_xml is not None else None
            if (
                _read(store.file, MAX_WORLD_BYTES) != prepared.before
                or sha(_record(store, prepared.id)[1]) != prepared.digest
            ):
                raise LayoutError("changed", "The world or saved layout changed while staging restoration")
            os.link(undo, history)
            saved_history = True
            os.replace(new_xml, xml)
            changed_xml = True
            os.replace(new_world, store.file)
            changed_world = True
            yield notes
        except BaseException as exc:
            failures = []
            if changed_world:
                try:
                    os.replace(rollback_world, store.file)
                except OSError as error:
                    failures.append(str(error))
            if changed_xml:
                try:
                    if rollback_xml is None:
                        xml.unlink(missing_ok=True)
                    else:
                        os.replace(rollback_xml, xml)
                except OSError as error:
                    failures.append(str(error))
            if saved_history:
                try:
                    history.unlink(missing_ok=True)
                except OSError as error:
                    failures.append(str(error))
            if failures:
                raise LayoutError("recovery", "Layout rollback needs repair: " + "; ".join(failures)) from exc
            raise
        finally:
            for temp in staged:
                try:
                    temp.unlink(missing_ok=True)
                except OSError as exc:
                    notes.append(f"Could not remove layout temporary file: {exc}")
        for old in sorted(store.history.glob("*.yaml"))[:-HISTORY_LIMIT]:
            try:
                old.unlink()
            except OSError as exc:
                notes.append(f"Layout restored, but old undo history could not be pruned: {exc}")
