"""Check that a planned robot behaves: it holds its pose, follows commands, and does not fall over."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from robotforge.design.importing.plan import ActuatorPlan, RigPlan, build_spec

GEOM = mujoco.mjtGeom
HOLD_SECONDS = 3.0
TRACK_SECONDS = 0.6
TRACK_SETTLE_SECONDS = 0.4
TRACK_STEP_REV = 0.25  # rad: the small move used to check that sliders follow
TRACK_STEP_SLIDE = 0.03  # m
MAX_HOLD_DRIFT_REV = 0.03  # rad
MAX_HOLD_DRIFT_SLIDE = 0.01  # m


@dataclass
class HoldResult:
    finite: bool
    drift: float  # worst joint error relative to what counts as "holding" (1.0 = at the limit)
    drift_raw: str  # human readable worst error
    drop: float  # how far a floating base sank, as a fraction of the robot's size
    upright: bool
    tracking: float = 0.0  # worst share of a small commanded move that was not achieved (fixed bases only)
    tracking_raw: str = ""
    blockers: tuple[tuple[str, str], ...] = ()  # link pairs touching while a joint failed to reach its target

    @property
    def holds(self) -> bool:
        return self.finite and self.drift <= 1.0 and self.drop < 0.25 and self.upright and self.tracking <= 0.2


def tipped(result: HoldResult) -> bool:
    return result.drop >= 0.25 or not result.upright


def score(result: HoldResult) -> float:
    return result.drift + 10 * result.drop + (0 if result.upright else 100) + 5 * result.tracking


def _test_world(urdf: Path, plan: RigPlan) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    """The robot at its home pose with every motor commanded to hold it; floating robots get a floor.

    A fixed robot is tested without one: a floor can only add contacts that have nothing to do with its motors.
    """
    world = mujoco.MjSpec()
    world.option.timestep = 0.002
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if plan.base == "floating":
        world.worldbody.add_geom(type=GEOM.mjGEOM_PLANE, size=[0, 0, 0.05])
    frame = world.worldbody.add_frame(pos=[0, 0, plan.spawn_height])
    world.attach(build_spec(urdf, plan), prefix="t/", frame=frame)
    model = world.compile()
    data = mujoco.MjData(model)
    for joint, value in plan.home_qpos.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{joint}")]] = value
    for a in plan.actuators:
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"t/{a.joint}")] = a.home
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"t/{plan.root_body}")
    return model, data, root


def _run(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> bool:
    """Step the simulation; False when it blows up."""
    for step in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step % 50 == 0 and not (np.isfinite(data.qpos).all() and np.abs(data.qpos).max() < 1e6):
            return False
    return True


def hold_test(urdf: Path, plan: RigPlan, seconds: float = HOLD_SECONDS) -> HoldResult:
    """Put the robot on a floor with every motor at its home pose, then nudge each joint; check it behaves."""
    model, data, root = _test_world(urdf, plan)
    start_height = float(data.subtree_com[root][2])  # center of mass: robust when the base origin sits on the floor
    if not _run(model, data, seconds):
        return HoldResult(False, math.inf, "the simulation blew up", 1.0, False)

    worst, worst_text = 0.0, "none"
    for a in plan.actuators:
        if a.kind != "position":
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{a.joint}")
        err = abs(float(data.qpos[model.jnt_qposadr[jid]]) - a.home)
        slide = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        ratio = err / (MAX_HOLD_DRIFT_SLIDE if slide else MAX_HOLD_DRIFT_REV)
        if ratio >= worst:
            worst, worst_text = ratio, f"{a.joint} off by {err:.3g} {'m' if slide else 'rad'}"
    drop, upright = 0.0, True
    if plan.base == "floating":
        # Measured against the robot's size, not its height: a wheeled base can have its center of mass 2 cm up.
        drop = max(0.0, (start_height - float(data.subtree_com[root][2])) / max(float(model.stat.extent), 0.05))
        upright = float(data.xmat[root].reshape(3, 3)[2, 2]) > 0.7
    result = HoldResult(True, worst, worst_text, drop, upright)
    if plan.base == "fixed" and result.finite and result.drift <= 1.0:
        result.tracking, result.tracking_raw, result.blockers = _tracking(model, data, plan)
    return result


def _tracking(
    model: mujoco.MjModel, data: mujoco.MjData, plan: RigPlan
) -> tuple[float, str, tuple[tuple[str, str], ...]]:
    """Nudge each joint on its own and return the worst share of the move that was not achieved.

    Joints are tested one at a time: moving them all at once measures how they shake each other, not their motors.
    """
    worst, text = 0.0, ""
    blockers: set[tuple[str, str]] = set()
    for a in plan.actuators:
        if a.kind != "position":
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"t/{a.joint}")
        slide = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        up, down = a.ctrl_hi - a.home, a.home - a.ctrl_lo
        delta = min(TRACK_STEP_SLIDE if slide else TRACK_STEP_REV, 0.8 * max(up, down))
        if delta <= 1e-6:
            continue
        delta = delta if up >= down else -delta
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"t/{a.joint}")
        data.ctrl[aid] = a.home + delta
        if not _run(model, data, TRACK_SECONDS):
            return math.inf, "the simulation blew up", ()
        achieved = float(data.qpos[model.jnt_qposadr[jid]])
        progress = (achieved - a.home) / delta  # 1.0 means it got exactly where it was told to go
        missing = max(1.0 - progress, progress - 1.5, 0.0)  # falling short, or flinging past by half the move
        if missing > 0.2:
            blockers |= _touching_links(model, data)
        if missing >= worst:
            worst, text = missing, f"{a.joint} reached {achieved:.3g} of {a.home + delta:.3g}"
        data.ctrl[aid] = a.home
        if not _run(model, data, TRACK_SETTLE_SECONDS):
            return math.inf, "the simulation blew up", ()
    return worst, text, tuple(sorted(blockers))


def _touching_links(model: mujoco.MjModel, data: mujoco.MjData) -> set[tuple[str, str]]:
    """Pairs of the robot's own links that are in contact right now (names without the test prefix)."""
    pairs = set()
    for i in range(data.ncon):
        c = data.contact[i]
        a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if a == b or a == 0 or b == 0 or c.dist >= 0:
            continue
        names = sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, x).removeprefix("t/") for x in (a, b))
        pairs.add((names[0], names[1]))
    return pairs


def tune_gains(urdf: Path, plan: RigPlan) -> HoldResult | None:
    """Raise motor stiffness until the robot holds its pose and follows commands; keeps the best in `plan`.

    Returns None when the plan has no motors or the simulation blows up at every setting.
    """
    if not plan.actuators:
        return None
    base = [ActuatorPlan(**asdict(a)) for a in plan.actuators]
    best: tuple[float, float, HoldResult] | None = None
    for scale in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        _scale_gains(plan, base, scale)
        result = hold_test(urdf, plan)
        if result.finite:
            value = score(result)
            stalled = best is not None and value > 0.97 * best[0]  # more stiffness no longer helps
            if best is None or value < best[0]:
                best = (value, scale, result)
            if result.holds or stalled:
                break
    if best is None:
        _scale_gains(plan, base, 1.0)
        return None
    _, scale, result = best
    _scale_gains(plan, base, scale)
    return result


def _scale_gains(plan: RigPlan, base: list[ActuatorPlan], scale: float) -> None:
    plan.gain_scale = scale
    for target, source in zip(plan.actuators, base, strict=True):
        if source.kind != "position":
            continue
        target.kp = source.kp * scale
        target.kv = source.kv * math.sqrt(scale)
        if not source.force_from_file and source.force is not None:
            target.force = source.force * scale
