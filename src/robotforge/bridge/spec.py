"""The policy card: everything robotforge must reproduce for a learned policy to work.

A policy is a function from an observation vector to joint targets, and it only works
if the observation is built exactly as during training: the same terms, in the same
order, in the same joint order, with the same scales. The card (policy.json, next to
policy.onnx) writes that contract down so it can be checked instead of guessed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

FORMAT = "robotforge-policy/1"
NAME_RE = re.compile(r"\A[a-z][a-z0-9_]{0,47}\Z")

# Observation terms robotforge can rebuild from a MuJoCo state. Base terms are measured at
# `base_site` (an IMU site) when the card names one, else at the robot's root body.
TERM_HELP: dict[str, str] = {
    "base_ang_vel": "angular velocity of the base in its own frame (a gyro), rad/s",
    "base_lin_vel": "linear velocity of the base in its own frame, m/s",
    "projected_gravity": "gravity direction in the base frame (unit vector)",
    "neg_up_axis_world": "the base's up axis in world coordinates, negated (UniLab's projected gravity "
    "from a framezaxis sensor); equals projected_gravity only while the robot faces +x",
    "joint_pos_rel": "joint angles minus the default pose, in the card's joint order, rad",
    "joint_vel": "joint velocities in the card's joint order, rad/s",
    "last_action": "the policy's previous raw output",
    "velocity_command": "commanded [forward m/s, left m/s, turn rad/s]",
    "gait_phase": "per-foot gait clock: (phase + offset) mod 1, advanced by frequency every step; in cycles [0, 1) "
    "or radians [0, 2 pi)",
}
TermName = Literal[
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "neg_up_axis_world",
    "joint_pos_rel",
    "joint_vel",
    "last_action",
    "velocity_command",
    "gait_phase",
]


class SpecError(ValueError):
    """User-facing: the policy card is missing, malformed or does not fit."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObsTerm(_Strict):
    term: TermName
    scale: float | list[float] = 1.0
    clip: float | None = Field(None, gt=0)
    history: int = Field(1, ge=1, le=50)  # stacked oldest first, per term
    frequency: float | None = Field(None, gt=0)  # gait_phase: cycles per second
    offsets: list[float] | None = None  # gait_phase: one per foot
    always_advance: bool = True  # gait_phase: False = the clock stops while the command is zero
    unit: Literal["cycle", "radian"] = "cycle"  # gait_phase: [0, 1) or [0, 2 pi)


class PDGains(_Strict):
    """The joint PD loop the policy was trained with, per joint in the card's order."""

    kp: list[float]
    kd: list[float]
    joint_damping: list[float] | None = None  # passive joint damping during training
    torque_limit: list[float] | None = None


class CommandRanges(_Strict):
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    # Training zeroed planar commands slower than this (m/s), so the policy never saw them.
    planar_dead_zone: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> CommandRanges:
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise ValueError(f"commands.{name}: lower bound {lo} is above upper bound {hi}")
        return self


class PolicySpec(_Strict):
    format: Literal["robotforge-policy/1"]
    name: str
    title: str
    robot: str | None = None  # library robot it was trained for (robotforge's name, e.g. "go2")
    summary: str = ""
    onnx: str = "policy.onnx"
    input_name: str = "obs"
    output_name: str | None = None  # None = the model's first output
    policy_dt: float = Field(gt=0, le=0.2)  # seconds between policy steps
    joints: list[str] = Field(min_length=1)  # action[i] moves joints[i]; joint terms use this order
    default_joint_pos: list[float]
    action_scale: float | list[float]
    action_clip: float | None = Field(None, gt=0)
    pd: PDGains
    base_site: str | None = None
    observations: list[ObsTerm] = Field(min_length=1)
    commands: CommandRanges
    trained_with: dict[str, Any] = Field(default_factory=dict)  # provenance, free form

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME_RE.match(v):
            raise ValueError("policy names use lowercase letters, digits and _, start with a letter, max 48 chars")
        return v

    @field_validator("onnx")
    @classmethod
    def _local_file(cls, v: str) -> str:
        if Path(v).name != v:  # the model sits next to its card; no paths that point elsewhere
            raise ValueError(f"onnx must be a file name next to policy.json, got {v!r}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> PolicySpec:
        n = len(self.joints)
        if len(set(self.joints)) != n:
            raise ValueError("joints are listed more than once")
        per_joint = {
            "default_joint_pos": self.default_joint_pos,
            "pd.kp": self.pd.kp,
            "pd.kd": self.pd.kd,
            "pd.joint_damping": self.pd.joint_damping,
            "pd.torque_limit": self.pd.torque_limit,
            "action_scale": self.action_scale if isinstance(self.action_scale, list) else None,
        }
        for field, values in per_joint.items():
            if values is not None and len(values) != n:
                raise ValueError(f"{field} has {len(values)} values for {n} joints")
        for term in self.observations:
            if term.term == "gait_phase" and (term.frequency is None or not term.offsets):
                raise ValueError("gait_phase needs frequency and offsets")
            if isinstance(term.scale, list) and len(term.scale) != self.term_dim(term):
                raise ValueError(f"{term.term}: {len(term.scale)} scales for {self.term_dim(term)} values")
        return self

    def term_dim(self, term: ObsTerm) -> int:
        """Values one step of this term contributes (before history)."""
        if term.term in ("joint_pos_rel", "joint_vel", "last_action"):
            return len(self.joints)
        if term.term == "gait_phase":
            return len(term.offsets or [])
        return 3

    @property
    def obs_dim(self) -> int:
        return sum(self.term_dim(t) * t.history for t in self.observations)

    def layout(self) -> list[dict[str, Any]]:
        """Where each term sits in the observation vector, for explanations and debugging."""
        out, start = [], 0
        for t in self.observations:
            width = self.term_dim(t) * t.history
            out.append({"term": t.term, "start": start, "size": width, "meaning": TERM_HELP[t.term]})
            start += width
        return out


def load_spec(directory: Path) -> PolicySpec:
    """Read and validate `directory`/policy.json."""
    path = directory / "policy.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"{directory} has no policy.json (the card that describes the policy)") from None
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path} is not valid JSON: {exc}") from exc
    try:
        spec = PolicySpec.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'card'}: {e['msg']}" for e in exc.errors())
        raise SpecError(f"{path}: {problems}") from None
    if not (directory / spec.onnx).is_file():
        raise SpecError(f"{directory} has policy.json but not its model file {spec.onnx}")
    return spec


def save_spec(spec: PolicySpec, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "policy.json"
    path.write_text(json.dumps(spec.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
