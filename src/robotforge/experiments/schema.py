"""Finite, versioned experiment records and deterministic specification identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Status = Literal["running", "completed", "stopped", "failed", "cancelled"]


def plain(value: Any) -> Any:
    """Accept JSON and NumPy values, rejecting nonfinite numbers and arbitrary objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment numbers must be finite")
        return value
    if isinstance(value, BaseModel):
        return plain(value.model_dump(mode="python"))
    if type(value).__module__.split(".")[0] == "numpy":
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("experiment JSON object keys must be strings")
        return {k: plain(v) for k, v in value.items()}
    raise ValueError(f"cannot store {type(value).__name__} in an experiment")


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def relative_artifact(value: str) -> str:
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or posix.is_absolute()
        or windows.drive
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("artifact path must be a safe relative POSIX path")
    return value


def check_id(value: str) -> str:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("experiment id must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("experiment id must be a canonical UUID")
    return value


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    format: str

    _safe_path = field_validator("path")(relative_artifact)


def semantic(value: Any) -> Any:
    """Keep arbitrary input names: path/output/elapsed/id can all be controller parameters."""
    return plain(value)


def _reference_identity(value: Any) -> Any:
    """Strip location only from recognized reference schemas at caller-selected metadata sites."""
    if not isinstance(value, dict):
        return value
    try:
        artifact = ArtifactRef.model_validate(value)
    except ValueError:
        pass
    else:
        return artifact.model_dump(exclude={"path"})
    if set(value) == {"id", "path", "kind"} and isinstance(value["path"], str):
        try:
            check_id(value["id"])
        except (ValueError, TypeError):
            pass
        else:
            return {"kind": value["kind"]}
    return value


def _environment_identity(environment: dict[str, Any]) -> dict[str, Any]:
    # These fields are adapter metadata, never task parameters or nested robot/world data.
    result = {key: value for key, value in environment.items() if key not in {"path", "folder", "world_dir"}}
    for name in ("env_spec", "train_spec"):
        spec = result.get(name)
        if isinstance(spec, dict):
            result[name] = {key: value for key, value in spec.items() if key != "world_dir"}
    snapshot = result.get("snapshot")
    if isinstance(snapshot, dict) and snapshot.get("format") == "mujoco_mjb_integration_v1":
        result["snapshot"] = {
            key: _reference_identity(value) if key in {"model", "state", "world"} else value
            for key, value in snapshot.items()
        }
    return result


def _algorithm_identity(algorithm: dict[str, Any]) -> dict[str, Any]:
    result = dict(algorithm)
    # Comparison adapters retain named policy artifacts here. Never descend into params,
    # method_params or supplied_params, even if user data happens to look like a reference.
    policies = result.get("policies")
    if isinstance(policies, dict):
        result["policies"] = {
            name: {key: _reference_identity(value) for key, value in policy.items()}
            if isinstance(policy, dict)
            else policy
            for name, policy in policies.items()
        }
    return result


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    id: str
    kind: str = Field(min_length=1, max_length=80)
    created_at: str
    updated_at: str
    finished_at: str | None = None
    status: Status = "running"
    termination: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    algorithm: dict[str, Any] = Field(default_factory=dict)
    seeds: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    budget_used: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    fingerprints: dict[str, Any] = Field(default_factory=dict)
    version: dict[str, Any] = Field(default_factory=dict)
    origins: list[dict[str, Any]] = Field(default_factory=list)
    parent: dict[str, Any] | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    fingerprint: str = ""

    _uuid = field_validator("id")(check_id)

    @model_validator(mode="before")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        return plain(value)

    @model_validator(mode="after")
    def _identity(self) -> ExperimentRecord:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", self.kind):
            raise ValueError("experiment kind uses lowercase letters, digits, underscores and hyphens")
        return self

    def specification_fingerprint(self) -> str:
        specification = {
            key: semantic(getattr(self, key))
            for key in ("schema_version", "kind", "task", "seeds", "version", "budget")
        }
        specification["environment"] = _environment_identity(self.environment)
        specification["algorithm"] = _algorithm_identity(self.algorithm)
        specification["origins"] = [_reference_identity(origin) for origin in self.origins]
        return digest(specification)
