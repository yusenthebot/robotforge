"""Bounded, immutable prop arrangements shared by every local world."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from robotforge.design.object_surfaces import SURFACE_SHAPES

MAX_RECORDS = 16
MAX_BYTES = 64 * 1024
MAX_SCAN = 64
SHAPES = {"box": 3, "sphere": 1, "cylinder": 2, "table": 3, "bin": 3}
FIELDS = {"name", "shape", "size", "pos", "yaw", "color", "mass", "fixed", "material", "surface_material"}
ID = re.compile(r"[0-9a-f]{32}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_DESCRIPTOR_IO = os.name != "nt"
Identity = tuple[int, int, int]
Identities = dict[Path, Identity]


class ArrangementError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = "arrangement_" + code
        super().__init__(message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def name(value: str) -> str:
    if not isinstance(value, str):
        raise ArrangementError("invalid_name", "Give the arrangement a name")
    value = unicodedata.normalize("NFC", value.strip())
    if not 1 <= len(value) <= 64 or any(unicodedata.category(c).startswith("C") for c in value):
        raise ArrangementError("invalid_name", "Use 1 to 64 visible characters for the arrangement name")
    return value


def _number(value, low, high) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _vector(value, length, low, high) -> bool:
    return isinstance(value, list) and len(value) == length and all(_number(v, low, high) for v in value)


def validate_member(value: dict) -> dict:
    """Check the entire safe primitive boundary before ObjectEntry can consult registries."""
    if (
        not isinstance(value, dict)
        or set(value) - FIELDS
        or not FIELDS - {"material", "surface_material"} <= set(value)
    ):
        raise ArrangementError("damaged", "Arrangement contains unsupported object fields")
    shape = value["shape"]
    if not isinstance(shape, str) or shape not in SHAPES:
        raise ArrangementError("damaged", "Arrangement contains an unsupported prop shape")
    if not isinstance(value["name"], str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,31}", value["name"]):
        raise ArrangementError("damaged", "Arrangement has an invalid prop name")
    if not _vector(value["size"], SHAPES[shape], 0.000001, 50):
        raise ArrangementError("damaged", "Arrangement has invalid prop dimensions")
    if not _vector(value["pos"], 3, -100, 100) or not _number(value["yaw"], -3600, 3600):
        raise ArrangementError("damaged", "Arrangement has an invalid relative pose")
    if type(value["fixed"]) is not bool or (value["mass"] is not None and not _number(value["mass"], 0.000001, 1e6)):
        raise ArrangementError("damaged", "Arrangement has invalid physical parameters")
    for key, shapes in (("material", {"box", "sphere", "cylinder"}), ("surface_material", SURFACE_SHAPES)):
        if key in value and (shape not in shapes or value[key] not in {"plastic", "wood", "steel"}):
            raise ArrangementError("damaged", "Arrangement has an unsupported material")
    color = value["color"]
    if isinstance(color, str):
        if len(color) > 32:
            raise ArrangementError("damaged", "Arrangement color is too long")
    elif not isinstance(color, list) or len(color) not in (3, 4) or not all(_number(c, 0, 255) for c in color):
        raise ArrangementError("damaged", "Arrangement has invalid color channels")
    # These imports happen only after the closed field and shape allowlist above.
    from robotforge.design.components import color as check_color
    from robotforge.design.world import ObjectEntry

    try:
        check_color(color)
        ObjectEntry.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise ArrangementError("damaged", "Arrangement contains invalid prop parameters") from exc
    return value


def decode(raw: bytes, record_id: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        if not isinstance(value, dict) or set(value) != {"schema_version", "id", "name", "created_at", "members"}:
            raise ValueError("record fields")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["id"] != record_id:
            raise ValueError("record identity")
        if not ID.fullmatch(record_id) or UUID(hex=record_id).version != 4:
            raise ValueError("record id")
        if name(value["name"]) != value["name"]:
            raise ValueError("record name")
        timestamp = value["created_at"]
        if not isinstance(timestamp, str) or len(timestamp) > 40 or datetime.fromisoformat(timestamp).tzinfo is None:
            raise ValueError("record time")
        members = value["members"]
        if not isinstance(members, list) or not 2 <= len(members) <= 16:
            raise ValueError("member count")
        for member in members:
            validate_member(member)
        if len({member["name"] for member in members}) != len(members):
            raise ValueError("duplicate members")
        return value
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        raise ArrangementError("damaged", "This saved arrangement is damaged or unsupported") from exc


def home() -> Path:
    from robotforge.design.store import worlds_home

    return worlds_home().parent / "arrangements"


def _identity(info):
    return info.st_dev, info.st_ino, info.st_mode


def _regular(info) -> bool:
    return stat.S_ISREG(info.st_mode) and not _reparse(info)


def _reparse(info) -> bool:
    # Reject Windows junctions and other reparse points as well as ordinary symlinks.
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


class LockedLibrary:
    """Descriptor-relative POSIX IO, or identity-checked Windows path IO."""

    def __init__(self, root: Path, fd: int | None, identities: Identities) -> None:
        self.root, self.fd, self.identities = root, fd, identities
        self.identity = identities[root]
        self.lock_identity = None

    def _stat(self, leaf):
        if self.fd is not None:
            return os.stat(leaf, dir_fd=self.fd, follow_symlinks=False)
        return (self.root / leaf).lstat()

    def check(self, expected: Identities | None = None) -> None:
        try:
            if any(_identity(path.lstat()) != identity for path, identity in self.identities.items()):
                raise OSError("library path changed")
            if self.lock_identity is not None and _identity(self._stat(".lock")) != self.lock_identity:
                raise OSError("library lock changed")
            if self.fd is not None and _identity(os.fstat(self.fd)) != self.identity:
                raise OSError("library root changed")
            if expected is not None and expected != self.identities:
                raise OSError("library ancestor changed")
        except OSError as exc:
            raise ArrangementError("changed", "The arrangement library changed; refresh it") from exc

    def _open(self, leaf, flags, mode=0o600):
        self.check()
        before = None
        try:
            before = self._stat(leaf)
        except FileNotFoundError:
            if not flags & os.O_CREAT:
                raise
        if before is not None and not _regular(before):
            raise ArrangementError("damaged", "Arrangement files must be regular files")
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        if self.fd is not None:
            descriptor = os.open(leaf, flags, mode, dir_fd=self.fd)
        else:
            descriptor = os.open(self.root / leaf, flags, mode)
        try:
            info = os.fstat(descriptor)
            if not _regular(info) or _identity(self._stat(leaf)) != _identity(info):
                raise ArrangementError("changed", "Arrangement file changed while opening it")
            if before is not None and _identity(before) != _identity(info):
                raise ArrangementError("changed", "Arrangement file was replaced while opening it")
            self.check()
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _sync_directory(self):
        if self.fd is not None:
            os.fsync(self.fd)

    def _unlink(self, leaf, expected):
        self.check()
        info = self._stat(leaf)
        if not _regular(info) or _identity(info) != expected:
            raise ArrangementError("changed", "Arrangement file changed before removal")
        if self.fd is not None:
            os.unlink(leaf, dir_fd=self.fd)
        else:
            os.unlink(self.root / leaf)
        self.check()

    def _publish(self, stage, leaf, expected):
        self.check()
        info = self._stat(stage)
        if not _regular(info) or _identity(info) != expected:
            raise ArrangementError("changed", "Arrangement stage changed before publication")
        if self.fd is not None:
            os.link(stage, leaf, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
        else:
            # A hard link publishes create-only on Windows too; never replace an existing record.
            os.link(self.root / stage, self.root / leaf)
        self.check()
        if _identity(self._stat(leaf)) != expected:
            raise ArrangementError("changed", "Arrangement file changed during publication")
        self._sync_directory()

    def _read_file(self, record_id: str):
        if not isinstance(record_id, str) or not ID.fullmatch(record_id):
            raise ArrangementError("invalid_request", "Invalid arrangement ID")
        descriptor = self._open(record_id + ".json", os.O_RDONLY)
        try:
            info = os.fstat(descriptor)
            if info.st_size > MAX_BYTES:
                raise ArrangementError("damaged", "Arrangement must be a regular file of at most 64 KiB")
            raw = os.read(descriptor, MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ArrangementError("damaged", "Arrangement exceeds 64 KiB")
            if _identity(self._stat(record_id + ".json")) != _identity(info):
                raise ArrangementError("changed", "Arrangement changed while reading it")
            self.check()
            return raw, _identity(info)
        finally:
            os.close(descriptor)

    def read(self, record_id: str) -> bytes:
        return self._read_file(record_id)[0]

    def record(self, record_id: str, digest: str) -> dict:
        try:
            raw = self.read(record_id)
        except FileNotFoundError as exc:
            raise ArrangementError("changed", "This arrangement was removed; refresh the library") from exc
        if not isinstance(digest, str) or not DIGEST.fullmatch(digest) or sha(raw) != digest:
            raise ArrangementError("changed", "This arrangement changed; refresh the library")
        return decode(raw, record_id)

    def entries(self) -> list[dict]:
        rows = []
        self.check()
        with os.scandir(self.fd if self.fd is not None else self.root) as scan:
            for index, item in enumerate(scan):
                if index >= MAX_SCAN:
                    raise ArrangementError("storage", "Too many files in the arrangement library")
                if item.name == ".lock" or item.name.startswith(".stage-"):
                    continue
                if not re.fullmatch(r"[0-9a-f]{32}\.json", item.name):
                    raise ArrangementError("storage", "Unexpected file in the arrangement library")
                if len(rows) >= MAX_RECORDS:
                    raise ArrangementError("storage", "The arrangement library contains more than 16 records")
                record_id, digest = item.name[:-5], None
                try:
                    raw = self.read(record_id)
                    digest = sha(raw)
                    value = decode(raw, record_id)
                    rows.append(
                        {
                            "id": record_id,
                            "name": value["name"],
                            "sha256": digest,
                            "created_at": value["created_at"],
                            "count": len(value["members"]),
                        }
                    )
                except (ArrangementError, OSError):
                    rows.append(
                        {
                            "id": record_id,
                            "name": "Damaged arrangement",
                            "sha256": digest,
                            "created_at": None,
                            "count": 0,
                            "damaged": True,
                            "reason": "This arrangement is damaged or cannot be read safely",
                        }
                    )
        self.check()
        return sorted(rows, key=lambda row: (row["created_at"] or "", row["id"]), reverse=True)

    def save(self, label: str, members: list[dict]) -> dict:
        label = name(label)
        rows = self.entries()
        if len(rows) >= MAX_RECORDS:
            raise ArrangementError("limit", "The library already holds 16 arrangements; remove one first")
        if any(not row.get("damaged") and row["name"].casefold() == label.casefold() for row in rows):
            raise ArrangementError("duplicate_name", "That arrangement name is already in use")
        record_id = uuid4().hex
        record = {
            "schema_version": 1,
            "id": record_id,
            "name": label,
            "created_at": datetime.now(UTC).isoformat(),
            "members": members,
        }
        raw = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(raw) > MAX_BYTES:
            raise ArrangementError("limit", "This arrangement exceeds 64 KiB")
        decode(raw, record_id)
        stage = ".stage-" + uuid4().hex
        fd = self._open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        stage_identity = _identity(os.fstat(fd))
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self._publish(stage, record_id + ".json", stage_identity)
        finally:
            self._unlink(stage, stage_identity)
        self.check()
        return {
            "id": record_id,
            "name": label,
            "sha256": sha(raw),
            "created_at": record["created_at"],
            "count": len(members),
        }

    def remove(self, record_id: str, digest: str) -> None:
        raw, identity = self._read_file(record_id)
        if not isinstance(digest, str) or not DIGEST.fullmatch(digest) or sha(raw) != digest:
            raise ArrangementError("changed", "This arrangement changed; refresh the library")
        self._unlink(record_id + ".json", identity)
        self._sync_directory()
        self.check()


def _lock_file(descriptor, *, windows=None):
    """Fail promptly rather than waiting while a caller holds the simulation lock."""
    if windows is None:
        windows = os.name == "nt"
    if windows:
        import msvcrt

        if not os.fstat(descriptor).st_size:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ArrangementError("busy", "Another arrangement operation is running; try again") from exc
    else:
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArrangementError("busy", "Another arrangement operation is running; try again") from exc


def _path_root(root):
    """Windows fallback: reject reparse points and pin every visible ancestor identity."""
    identities = {}
    for path in reversed((root, *root.parents)):
        if any(_identity(parent.lstat()) != identity for parent, identity in identities.items()):
            raise ArrangementError("changed", "The arrangement library path changed")
        try:
            info = path.lstat()
        except FileNotFoundError:
            path.mkdir(mode=0o700)
            info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or _reparse(info):
            raise ArrangementError("storage", "Arrangement library folders must be regular directories")
        identities[path] = _identity(info)
    return identities


@contextmanager
def locked() -> Iterator[LockedLibrary]:
    root = home().absolute()
    descriptors = []
    lock_fd = None
    try:
        if _DESCRIPTOR_IO:
            fd = os.open(root.anchor, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            descriptors.append(fd)
            identities = {Path(root.anchor): _identity(os.fstat(fd))}
            path = Path(root.anchor)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            for part in root.parts[1:]:
                path = path / part
                try:
                    child = os.open(part, flags, dir_fd=fd)
                except FileNotFoundError:
                    os.mkdir(part, 0o700, dir_fd=fd)
                    os.fsync(fd)
                    child = os.open(part, flags, dir_fd=fd)
                descriptors.append(child)
                fd = child
                identities[path] = _identity(os.fstat(fd))
        else:
            fd = None
            identities = _path_root(root)
        library = LockedLibrary(root, fd, identities)
        lock_fd = library._open(".lock", os.O_RDWR | os.O_CREAT)
        lock_info = os.fstat(lock_fd)
        _lock_file(lock_fd)
        library.lock_identity = _identity(lock_info)
        library.check()
        yield library
    except OSError as exc:
        raise ArrangementError("storage", "Cannot safely access the local arrangement library") from exc
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
