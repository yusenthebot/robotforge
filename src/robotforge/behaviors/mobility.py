"""Read-only mobility limits from the bindings also used by runtime controllers."""

from __future__ import annotations

import mujoco
import numpy as np

from robotforge.behaviors import Kind
from robotforge.sim.water import Hydro


def describe_mobility(
    model: mujoco.MjModel,
    robot: str,
    kind: Kind | str,
    home_qpos: np.ndarray,
    water: Hydro | None,
    data: mujoco.MjData | None,
) -> float:
    """Return the default mover's speed cap without creating a controller or changing live state.

    Policy cards, kept walks, active controllers and library caps belong to the manager. Binding failures are
    ValueError, as for runtime mover construction; all kinematic discovery uses private MjData.
    """
    try:
        return _describe(model, robot, kind, home_qpos, water, data)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"robot {robot!r} cannot bind its {kind} mover: {exc}") from exc


def _describe(model, robot, kind, home_qpos, water, data) -> float:
    if kind in (Kind.QUADRUPED, Kind.HEXAPOD):
        from robotforge.behaviors import quadruped

        root, specs, period = quadruped.robot_binding(model, robot, home_qpos)
        _, legs, _ = quadruped.bind_legs(model, robot, specs, root, home_qpos)
        length = float(np.mean([np.linalg.norm(leg.home_foot - leg.hip) for leg in legs]))
        return quadruped.stride_speed(0.55 * length, period if period is not None else quadruped.gait_period(length))
    if kind == Kind.HUMANOID:
        from robotforge.behaviors import humanoid

        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{robot}/left_foot") >= 0:
            legs, feet, _upper, root, _imu = humanoid.template_binding(model, robot)
        else:
            legs, feet, _upper, root, _imu, _roles, _flats = humanoid.robot_binding(model, robot, home_qpos)
        humanoid.leg_actuators(model, legs)
        posed = humanoid._posed_at_home(model, home_qpos, root)
        soles = [posed.xpos[body] + posed.xmat[body].reshape(3, 3) @ point for body, point in feet]
        _, length, width = humanoid.standing_geometry(model, posed, legs, soles)
        gait = humanoid.default_gait(length, width)
        return gait.max_stride / gait.step_time
    if kind == Kind.ROVER:
        from robotforge.behaviors.drive import drive_binding, drive_speed
        from robotforge.behaviors.steering import steering_binding, steering_speed

        modules = steering_binding(model, robot, home_qpos)
        return steering_speed(modules) if modules is not None else drive_speed(model, drive_binding(model, robot))
    if kind == Kind.BALANCER:
        from robotforge.behaviors.balance import balance_speed, find_balancer

        binding = find_balancer(model, robot, home_qpos)
        if binding is None:
            raise ValueError(f"robot {robot!r} is not a two-wheeled balancing robot")
        return balance_speed(binding)
    if kind == Kind.SNAKE:
        from robotforge.behaviors.snake import MAX_HZ, spine_binding, spine_speed_per_hz

        return spine_speed_per_hz(model, spine_binding(model, robot, home_qpos), home_qpos) * MAX_HZ
    if kind in (Kind.BOAT, Kind.ROV):
        from robotforge.behaviors.marine import marine_binding, marine_speed, root_body, surge_drag

        found = marine_binding(model, robot, kind)
        drag = surge_drag(kind, water, data, root_body(model, robot))
        return marine_speed(kind, sum(t.push * abs(t.direction[0]) for t in found), drag)
    if kind == Kind.AIRCRAFT:
        from robotforge.behaviors.airframe import find_airframe
        from robotforge.behaviors.flight import MAX_SPEED

        find_airframe(model, robot, home_qpos)
        return MAX_SPEED
    raise ValueError(f"robot {robot!r} has no default mover for kind {kind!r}")
