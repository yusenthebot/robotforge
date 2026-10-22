"""Shared robot capability descriptions, derived from the assembled model.

Kinds describe a body; capabilities describe the interfaces that can operate it.
The frozen structural part is cached per compiled model. Runtime movement providers
and source metadata are supplied afresh by the behavior manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco

from robotforge.behaviors import Kind
from robotforge.behaviors.arm import arm_joints

# Preserve the viewer's tried task/body combinations. This is presentation policy,
# not a replacement for task validation when a controller is actually started.
TASK_KINDS: dict[str, frozenset[Kind]] = {
    "stand": frozenset({Kind.QUADRUPED}),
    "walk": frozenset({Kind.QUADRUPED}),
    "balance": frozenset({Kind.BALANCER, Kind.OTHER}),
    "reach": frozenset({Kind.ARM}),
    "drive_to": frozenset({Kind.ROVER}),
    "hover": frozenset({Kind.AIRCRAFT}),
}


@dataclass(frozen=True)
class Support:
    supported: bool
    reason: str | None = None
    endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            **({"reason": self.reason} if self.reason else {}),
            **({"endpoint": self.endpoint} if self.endpoint else {}),
        }


@dataclass(frozen=True)
class ArmCapability:
    name: str
    tcp: str
    move: Support
    gripper: Support

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tcp": self.tcp, "move": self.move.to_dict(), "gripper": self.gripper.to_dict()}


@dataclass(frozen=True)
class RobotStructure:
    arms: tuple[ArmCapability, ...]
    actuators: tuple[str, ...]
    sensors: tuple[str, ...]
    tasks: tuple[tuple[str, Support], ...]


def inspect_robot(model: mujoco.MjModel, name: str, kind: str) -> RobotStructure:
    """Inspect the final tree, including mounts. Never attach or create a controller."""
    from robotforge.control.tasks import TaskError, make_task

    arms = []
    prefix = name + "/"
    for site in range(model.nsite):
        tcp = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site) or ""
        if not (tcp.startswith(prefix) and tcp.endswith("/tcp")):
            continue
        endpoint = tcp.removesuffix("/tcp")
        reason = None
        if kind == "gripper":
            reason = "a standalone gripper has no arm to move it; use its raw actuator controls or mount it on an arm"
        elif not arm_joints(model, site):
            reason = "the tool point has no powered joints that move it"
        move = Support(reason is None, reason, endpoint)
        grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, endpoint + "/gripper")
        grip_reason = reason or ("the arm has no gripper actuator" if grip < 0 else None)
        arms.append(ArmCapability(endpoint, tcp, move, Support(grip_reason is None, grip_reason, endpoint)))

    movable = [a for a in arms if a.move.supported]
    tasks = []
    for task, kinds in TASK_KINDS.items():
        endpoint = name
        if task == "reach" and len(movable) == 1:
            endpoint = movable[0].name
        allowed = kind in kinds and endpoint == name
        reason = None
        if task == "reach" and len(movable) != 1:
            reason = "choose an arm endpoint explicitly" if movable else "no movable arm with a tool point"
        elif not allowed:
            reason = "this mode is not offered for this robot's body"
        else:
            try:
                make_task(task, model, endpoint)
            except TaskError as exc:
                reason = str(exc)
        tasks.append((task, Support(reason is None, reason, endpoint if reason is None else None)))

    def names(obj: mujoco.mjtObj, count: int) -> tuple[str, ...]:
        found = (mujoco.mj_id2name(model, obj, i) or "" for i in range(count))
        return tuple(n for n in found if n.startswith(prefix))

    return RobotStructure(
        tuple(arms),
        names(mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        names(mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        tuple(tasks),
    )


def describe(
    structure: RobotStructure,
    info: dict[str, Any],
    source: dict[str, Any],
    move_reason: str | None = None,
    learned: bool = False,
    *,
    move_supported: bool | None = None,
    kept_walk_supported: bool = True,
) -> dict[str, Any]:
    """A fresh JSON value for all consumers; callers cannot mutate cached structure."""
    top = info.get("max_speed")
    goes = top is not None and top > 0 and move_supported is not False
    movement = Support(
        goes, None if goes else move_reason or info.get("walk_hint") or "no movement provider fits this body"
    )
    axes = ["forward", "turn"] if goes else []
    if goes and info["kind"] == "aircraft":
        axes += ["slide", "climb"]
    elif goes and info["kind"] == "rov":
        axes += ["heave"]
    movable = [a.name for a in structure.arms if a.move.supported]
    grippers = [a.name for a in structure.arms if a.gripper.supported]
    modes = ["drive"] if goes else ["manual"]
    if learned:
        modes.append("learned")
    if (
        kept_walk_supported
        and "learned_walk" in info
        and info["kind"] in {k.value for k in Kind if k not in (Kind.ARM, Kind.OTHER)}
    ):
        modes.append("rhythm")
    return {
        "schema_version": 1,
        "source": source,
        "move": {**movement.to_dict(), "axes": axes, **({"max_speed": top} if goes else {})},
        "move_arm": {
            **Support(bool(movable), None if movable else "no movable arm with a tool point").to_dict(),
            "endpoints": movable,
        },
        "gripper": {
            **Support(bool(grippers), None if grippers else "no movable arm with a gripper actuator").to_dict(),
            "endpoints": grippers,
        },
        "arms": [a.to_dict() for a in structure.arms],
        "manual": {
            **Support(
                bool(structure.actuators), None if structure.actuators else "no actuators belong to this robot"
            ).to_dict(),
            "actuators": list(structure.actuators),
        },
        "sensors": list(structure.sensors),
        "tasks": {key: support.to_dict() for key, support in structure.tasks},
        "drive_modes": modes,
    }


def source_of(entry: Any | None) -> dict[str, Any]:
    """Keep provenance separate from composition; no new world-file fields are needed."""
    if entry is None:
        return {"kind": "model", "composed": False, "components": []}
    kind = next(
        (key for key in ("template", "library", "catalogue", "file", "build") if getattr(entry, key, None)), "model"
    )
    out: dict[str, Any] = {"kind": kind, "composed": bool(entry.mounts or entry.build), "components": []}
    if kind not in ("model", "build"):
        out["id"] = getattr(entry, kind)
    if entry.generated:
        out["generated"] = True
    for mount in entry.mounts:
        part = {
            key: getattr(mount, key) for key in ("name", "template", "library", "design") if getattr(mount, key, None)
        }
        out["components"].append(part)
    return out
