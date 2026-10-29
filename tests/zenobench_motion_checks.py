"""Checks of the cuRobo stand-in (robotforge.zenobench.motion) on a real ZenoBench scene.

Run by tests/test_zenobench_motion.py as `python -m robotforge.zenobench.python <this file>` from the
ZenoBench checkout (upstream's own make_rig builds the rig); prints one JSON line of measurements.
"""

from __future__ import annotations

import json
import time

import mujoco
import numpy as np
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from zeno_skills.kinematics import CARRY_TILT, TCP_DEPTH, tilt  # noqa: E402
from zeno_skills.runtime import make_rig  # noqa: E402

rig = make_rig(
    app, "tasks/kitchen_skills/scene.usd", "tasks/kitchen_skills/annotation.json", video=False, log=lambda s: None
)
rig.sync_world()
mo, kin = rig.kin.motion, rig.kin
out = {"backend": type(mo).__name__}
m, d = mo.model, mo.data
rng = np.random.default_rng(0)
ctx = mo._context(kin, None, {}, (), True)

# 1. the planning model's gripper (MuJoCo, from USD) against ZenoBench's own FK (numpy, from the URDF)
grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "/World/ZenoMalo/Asset/right_gripper_link")
errs = []
for q in ctx.lo + rng.random((25, len(ctx.lo))) * (ctx.hi - ctx.lo):
    mo._set_robot(d, ctx, q)
    mujoco.mj_kinematics(m, d)
    R = d.xmat[grip].reshape(3, 3)
    p_mj = d.xpos[grip] - TCP_DEPTH * R[:, 2]
    p_k, R_k = kin.tcp(q)
    errs.append((float(np.linalg.norm(p_mj - p_k)), float(np.linalg.norm(R - R_k))))
out["fk_pos_err"] = max(e[0] for e in errs)
out["fk_rot_err"] = max(e[1] for e in errs)

# 2. the travel posture is free; a check's cost
q0 = np.clip(kin.posture, ctx.lo, ctx.hi)
out["posture_free"] = bool(mo.free(kin, q0))
t = time.perf_counter()
for _ in range(300):
    mo.free(kin, q0)
out["free_check_us"] = (time.perf_counter() - t) / 300 * 1e6

# 3. IK round trip near the posture
p_t, R_t = kin.tcp(q0)
p_t = p_t + np.array([0.08, -0.05, 0.06])
q_ik, ok = mo.ik(kin, p_t, R_t, [q0], True)
out["ik_ok"] = bool(ok)
out["ik_err"] = float(np.linalg.norm(kin.tcp(q_ik)[0] - p_t))
out["ik_free"] = bool(mo.free(kin, q_ik))

# 4. self collision: the reduced (moving) sphere pairs agree with every non-ignored pair
links = mo.sph_link
i_all, j_all = np.triu_indices(len(links), 1)
keep = np.array(
    [a != b and frozenset((a, b)) not in mo.rs.ignore for a, b in zip(links[i_all], links[j_all], strict=True)]
)
i_all, j_all = i_all[keep], j_all[keep]
mismatch = collisions = 0
for q in ctx.lo + rng.random((5000, len(ctx.lo))) * (ctx.hi - ctx.lo):
    mo._set_robot(d, ctx, q)
    mujoco.mj_kinematics(m, d)
    X = d.geom_xpos[mo.sph]
    full = (np.linalg.norm(X[i_all] - X[j_all], axis=1) - mo.sph_r[i_all] - mo.sph_r[j_all]).min() >= 0
    mismatch += int(full != mo._row(d, ctx, q, "self"))
    collisions += int(not full)
out["self_mismatch"] = mismatch
out["self_collisions_seen"] = collisions

# 5. a 5 cm block put in the way of a straight joint move: the segment is blocked, a plan goes around
q1 = q0.copy()
q1[2] += 1.3  # shoulder swing
q1 = np.clip(q1, ctx.lo, ctx.hi)
mid = kin.tcp(q0 + 0.5 * (q1 - q0))[0]
body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, mo.ann.objects["wooden_block_a"]["body"])
adr = m.jnt_qposadr[m.body_jntadr[body]]
saved = mo.qpos_world.copy()
mo.qpos_world[adr : adr + 3] = mid
mo._boxes, mo._mask_key = None, None
out["goal_free"] = bool(mo.free(kin, q1))
out["straight_blocked"] = not mo.segment_free(kin, q0, q1)
t = time.perf_counter()
qs, status = mo.plan(kin, q0, q1)
out["plan_s"] = time.perf_counter() - t
out["plan_ok"] = qs is not None
if qs is not None:
    ctx = mo._context(kin, None, {}, (), True)
    out["plan_dense_free"] = mo._segments_free(ctx, [q0] + list(qs))
    out["plan_end_err"] = float(np.max(np.abs(qs[-1] - q1)))
    out["plan_waypoints"] = len(qs)
else:
    out["plan_status"] = status
# the arm holding that object: it is no obstacle to the holding arm
q_at = q0 + 0.5 * (q1 - q0)
out["at_object_free"] = bool(mo.free(kin, q_at))
mo.held["right"] = {"object:wooden_block_a"}
mo._mask_key = None
out["held_object_free"] = bool(mo.free(kin, q_at))
mo.held["right"] = set()
out["clearance_at_object"] = float(mo.clearance(kin, q_at[None])[0])
out["clearance_posture"] = float(mo.clearance(kin, q0[None])[0])
mo.qpos_world[:] = saved
mo._boxes, mo._mask_key = None, None

# 6. a tilt-held plan keeps the TCP's tilt at the goal's all the way
R1 = kin.tcp(q0)[1]
q_goal, ok = mo.ik(kin, kin.tcp(q0)[0] + np.array([0.12, 0.18, 0.08]), R1, [q0], True)
ctx = mo._context(kin, None, {}, (), True)
t = time.perf_counter()
qt, status = mo._plan_tilt(ctx, q0, q_goal, R1) if ok else (None, "no goal IK")
out["tilt_plan_s"] = time.perf_counter() - t
out["tilt_plan_ok"] = qt is not None
if qt is not None:
    dense = [qa + (qb - qa) * u for qa, qb in zip([q0] + qt[:-1], qt, strict=True) for u in np.linspace(0, 1, 5)]
    out["tilt_max"] = max(tilt(R1, kin.tcp(q)[1]) for q in dense)
    out["tilt_limit"] = CARRY_TILT
# 6b. a tilt-only levelled path still ends exactly at its goal (the hand must not stop short)
if ok:
    free_path = [q0 + (q_goal - q0) * u for u in np.linspace(0, 1, 12)[1:]]
    lv = mo._level_path(ctx, q0, free_path, R1, hold_position=False)
    out["level_end_err"] = None if lv is None else float(np.max(np.abs(lv[-1] - q_goal)))

# 6c. the task-space carry search: tilt fixed all the way, ends at the goal's TCP pose
if ok:
    t = time.perf_counter()
    ts = mo._plan_level_ts(ctx, q0, [q_goal], R1)
    out["ts_plan_s"] = time.perf_counter() - t
    out["ts_ok"] = ts is not None
    if ts is not None:
        out["ts_tilt_max"] = max(tilt(R1, kin.tcp(q)[1]) for q in ts)
        out["ts_end_err"] = float(np.linalg.norm(kin.tcp(ts[-1])[0] - kin.tcp(q_goal)[0]))
        out["ts_dense_free"] = mo._segments_free(ctx, [q0] + ts)
        out["ts_max_step"] = float(max(np.max(np.abs(b - a)) for a, b in zip([q0] + ts[:-1], ts, strict=True)))

# 7. a held object's box rides on the hand: its spheres sit inside the object's box, then switch off
ctx = mo._context(kin, None, {}, (), True)
name = mo._attach_held("right", kin, q0, {"name": "wooden_block_a"})
mo._set_robot(d, ctx, q0)
mujoco.mj_kinematics(m, d)
lo, hi = mo._owner_box("object:wooden_block_a")
on = mo.held_geoms["right"][: mo._held_n["right"]]  # the spheres the hand uses (switched on per query)
centres = d.geom_xpos[on]
out["held_attached"] = name
out["held_spheres"] = int(len(on))
out["held_inside_box"] = bool(np.all(centres > lo - 1e-6) and np.all(centres < hi + 1e-6))
# a left-arm query in between must not switch the right hand's held spheres off
mo.free(rig.left_kin, np.r_[rig.q_cmd[:2], rig.left_q_cmd])
mo.free(kin, q0)
out["held_after_left_query"] = int(np.count_nonzero(m.geom_contype[mo.held_geoms["right"]]))
mo._attach_held("right", kin, q0, None)
out["held_off"] = int(np.count_nonzero(m.geom_contype[mo.held_geoms["right"]]))

# 8. part_at moves an articulated part for the queries inside, and back
part = "part:kitchen_microwave"
box0 = mo._owner_box(part)
with mo.part_at("kitchen_microwave", -1.4):
    box1 = mo._owner_box(part)
box2 = mo._owner_box(part)
out["part_moves"] = float(np.abs(np.r_[box1] - np.r_[box0]).max())
out["part_restored"] = float(np.abs(np.r_[box2] - np.r_[box0]).max())
# 9. the point Jacobian collision repair uses matches finite differences of a sphere centre
ctx = mo._context(kin, None, {}, (), True)
s_geom = int(mo.sph[np.nonzero(mo.sph_link == "right_arm_link_6")[0][0]])
jacp = np.zeros((3, m.nv))
mo._set_robot(d, ctx, q0)
mujoco.mj_kinematics(m, d)
mujoco.mj_comPos(m, d)
mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[s_geom].copy(), m.geom_bodyid[s_geom])
J = jacp[:, mo.side_dof["right"]]
x0 = d.geom_xpos[s_geom].copy()
fd = np.zeros_like(J)
for k in range(J.shape[1]):
    dq = np.zeros(len(q0))
    dq[k] = 1e-6
    mo._set_robot(d, ctx, q0 + dq)
    mujoco.mj_kinematics(m, d)
    fd[:, k] = (d.geom_xpos[s_geom] - x0) / 1e-6
out["repair_jac_err"] = float(np.abs(J - fd).max())
out["repair_jac_norm"] = float(np.abs(J).max())
print("MOTION_CHECKS " + json.dumps(out), flush=True)
