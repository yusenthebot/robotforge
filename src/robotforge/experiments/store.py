"""Atomic experiment records with immutable content-addressed artifacts."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .provenance import provenance
from .schema import ArtifactRef, ExperimentRecord, canonical, check_id, relative_artifact

RECORD_FILE = "experiment.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def safe_path(folder: Path, relative: str) -> Path:
    relative_artifact(relative)
    base = folder.resolve()
    target = base / relative
    if not target.resolve().is_relative_to(base):
        raise ValueError("artifact path escapes experiment folder through a symlink")
    return target


def _atomic(path: Path, content: bytes) -> None:
    """Unique same-directory temporary file; readers see only complete JSON."""
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def record_path(path: str | Path) -> Path:
    path = Path(path)
    return path / RECORD_FILE if path.is_dir() else path


def load_record(path: str | Path) -> ExperimentRecord:
    path = record_path(path)
    record = ExperimentRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
    if record.fingerprint != record.specification_fingerprint():
        raise ValueError("experiment specification fingerprint does not match its contents")
    for artifact in record.artifacts:
        safe_path(path.parent, artifact.path)
    return record


def artifact_path(path: str | Path, ref: ArtifactRef | dict[str, Any]) -> Path:
    """Resolve a reference relative to a record and verify its size and SHA-256."""
    return verified_artifact(record_path(path).parent, ArtifactRef.model_validate(ref))


def verified_artifact(folder: Path, ref: ArtifactRef) -> Path:
    path = safe_path(folder, ref.path)
    content = path.read_bytes()
    if len(content) != ref.size or hashlib.sha256(content).hexdigest() != ref.sha256:
        raise ValueError(f"experiment artifact integrity failed: {ref.role}")
    return path


class ExperimentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def begin(
        self,
        kind: str,
        *,
        environment: dict | None = None,
        task: dict | None = None,
        algorithm: dict | None = None,
        seeds: dict | None = None,
        budget: dict | None = None,
        parent: dict | None = None,
        origins: list[dict] | None = None,
    ) -> ExperimentHandle:
        now = _now()
        record = ExperimentRecord(
            id=str(uuid4()),
            kind=kind,
            created_at=now,
            updated_at=now,
            environment=environment or {},
            task=task or {},
            algorithm=algorithm or {},
            seeds=seeds or {},
            budget=budget or {},
            parent=parent,
            origins=origins or [],
            version=provenance(),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        folder = safe_path(self.root, check_id(record.id))
        folder.mkdir(exist_ok=False)
        handle = ExperimentHandle(folder, record)
        handle._save(record)
        return handle

    def resume(self, reference: dict[str, str] | str | Path) -> ExperimentHandle:
        """Resume from a saved reference; caller must maintain one writer per record."""
        path = record_path(reference["path"] if isinstance(reference, dict) else reference).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("experiment reference is outside this store")
        record = load_record(path)
        if path.parent.name != record.id:
            raise ValueError("experiment directory does not match its id")
        return ExperimentHandle(path.parent, record)

    def open(self, experiment_id: str) -> ExperimentHandle:
        folder = safe_path(self.root, check_id(experiment_id))
        return ExperimentHandle(folder, load_record(folder))


class ExperimentHandle:
    def __init__(self, folder: Path, record: ExperimentRecord):
        self.folder = folder
        self.record = record
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self.folder / RECORD_FILE

    def _save(self, record: ExperimentRecord) -> None:
        record = ExperimentRecord.model_validate(record.model_dump())
        record.updated_at = _now()
        record.fingerprint = record.specification_fingerprint()
        _atomic(safe_path(self.folder, RECORD_FILE), canonical(record))
        self.record = record

    def update(self, **changes: Any) -> None:
        """Merge metadata sections; reject unknown fields and changes after termination."""
        allowed = {
            "reproducibility",
            "fingerprints",
            "metrics",
            "budget_used",
            "environment",
            "task",
            "algorithm",
            "seeds",
            "budget",
            "origins",
        }
        if changes.keys() - allowed:
            raise ValueError(f"unknown experiment fields: {sorted(changes.keys() - allowed)}")
        with self._lock:
            self._running()
            data = self.record.model_dump()
            for key, value in changes.items():
                if value is not None:
                    data[key] = {**data[key], **value} if isinstance(data[key], dict) else value
            self._save(ExperimentRecord.model_validate(data))

    def _running(self) -> None:
        if self.record.status != "running":
            raise ValueError("experiment already finished")

    def artifact(self, path: str | Path, role: str, copy: bool = True) -> ArtifactRef:
        """Always retain immutable bytes; copy=False requires a file already inside this record."""
        source = Path(path).resolve()
        if not copy and not source.is_relative_to(self.folder.resolve()):
            raise ValueError("copy=False artifact must already be inside the experiment folder")
        return self._artifact_bytes(source.read_bytes(), role, source.suffix.lstrip(".").lower() or "bin")

    def _artifact_bytes(self, content: bytes, role: str, format: str) -> ArtifactRef:
        sha = hashlib.sha256(content).hexdigest()
        # Never use source names in a destination path.
        extension = format if format.isalnum() and len(format) <= 12 else "bin"
        rel = f"artifacts/{sha}.{extension}"
        ref = ArtifactRef(path=rel, role=role, sha256=sha, size=len(content), format=extension)
        with self._lock:
            self._running()
            destination = safe_path(self.folder, rel)
            destination.parent.mkdir(exist_ok=True)
            # A store-wide object has hard links in each record, so records remain self-contained
            # when copied and repeated models/policies occupy only one set of bytes on disk.
            shared = safe_path(self.folder.parent, f".objects/{sha}.{extension}")
            shared.parent.mkdir(exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".object.", dir=shared.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(tmp_name, shared)
                except FileExistsError:
                    existing = shared.read_bytes()
                    if len(existing) != ref.size or hashlib.sha256(existing).hexdigest() != sha:
                        raise ValueError("existing content-addressed object is corrupt") from None
                with contextlib.suppress(FileExistsError):
                    os.link(shared, destination)
                verified_artifact(self.folder, ref)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
            data = self.record.model_dump()
            if ref not in self.record.artifacts:
                data["artifacts"].append(ref.model_dump())
            self._save(ExperimentRecord.model_validate(data))
        return ref

    def snapshot_environment(
        self,
        model: Any,
        data: Any,
        world: Any = None,
        scope: str | None = None,
        robot: str | None = None,
    ) -> dict[str, Any]:
        from .snapshot import snapshot_environment

        return snapshot_environment(self, model, data, world=world, scope=scope, robot=robot)

    def finish(self, status: str, reason: str | None = None, error: str | None = None) -> None:
        if status not in ("completed", "stopped", "failed", "cancelled"):
            raise ValueError("terminal status must be completed, stopped, failed or cancelled")
        with self._lock:
            self._running()
            data = self.record.model_dump()
            data.update(status=status, finished_at=_now(), termination={"reason": reason, "error": error})
            self._save(ExperimentRecord.model_validate(data))

    def reference(self) -> dict[str, str]:
        return {"id": self.record.id, "path": str(self.path), "kind": self.record.kind}
