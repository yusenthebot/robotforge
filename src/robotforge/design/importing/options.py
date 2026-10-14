"""What a caller can ask of the import doctor, and what it hands back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import mujoco
from pydantic import BaseModel, ConfigDict, ValidationError

from robotforge.design.importing.report import DoctorReport


class ImportOptions(BaseModel):
    """Optional knobs, passed as `params` of a robot entry or as keyword arguments."""

    model_config = ConfigDict(extra="forbid")

    base: Literal["auto", "fixed", "floating"] = "auto"  # URDF: arm-like or free-standing; mesh: fixed or dropped
    decompose: bool = False  # split concave collision meshes into convex pieces (slow, cached)
    respect_effort_limits: bool = False  # URDF: keep the file's <limit effort> even where it cannot hold the pose
    scale: float | None = None  # plain meshes: multiply the size (default: guess meters vs millimeters)
    mass: float | None = None  # plain meshes that float: kg (default: from volume at 500 kg/m^3)
    xacro_args: dict[str, str] | None = None  # values for <xacro:arg> names

    @classmethod
    def parse(cls, params: dict[str, Any] | ImportOptions | None) -> ImportOptions:
        if isinstance(params, ImportOptions):
            return params
        try:
            return cls.model_validate(params or {})
        except ValidationError as exc:
            parts = [f"{'.'.join(str(x) for x in e['loc']) or 'value'}: {e['msg']}" for e in exc.errors()]
            allowed = ", ".join(cls.model_fields)
            raise ValueError(f"bad import option ({'; '.join(parts)}). Import options are: {allowed}") from exc


@dataclass
class ImportResult:
    """A spec ready to attach to a world, how to place it, and the doctor's notes."""

    spec: mujoco.MjSpec
    report: DoctorReport
    kind: Literal["robot", "object"]
    floating: bool
    spawn_height: float
    home_qpos: dict[str, float]
    home_ctrl: dict[str, float]
