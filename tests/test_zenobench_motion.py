"""The cuRobo stand-in for ZenoBench (robotforge.zenobench.motion): batched kinematics and IK, OMPL plans,
and, when a ZenoBench checkout from after its switch to cuRobo is available, the stand-in on a real scene.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation  # noqa: E402

from robotforge.zenobench import motion_kin as K  # noqa: E402

CHECKS = Path(__file__).with_name("zenobench_motion_checks.py")


def _toy_kin():
    """A 5-joint arm on a lift (prismatic z, then revolute z, y, y, x), like ArmKin's chain layout."""
    chain = [
        ("lift", "prismatic", (0.0, 0.0, 0.4), (0.0, 0.0, 1.0)),
        ("yaw", "revolute", (0.05, 0.0, 0.2), (0.0, 0.0, 1.0)),
        ("shoulder", "revolute", (0.0, 0.1, 0.0), (0.0, 1.0, 0.0)),
        ("mount", "fixed", (0.0, 0.0, 0.05), None),
        ("elbow", "revolute", (0.3, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("wrist", "revolute", (0.25, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ]
    names = [c[0] for c in chain if c[1] != "fixed"]
    return SimpleNamespace(
        chain=[(n, t, np.array(x), None if a is None else np.array(a)) for n, t, x, a in chain], names=names
    )


def _base():
    return np.array([1.0, -2.0, 0.1]), Rotation.from_euler("z", 0.7).as_matrix()


def test_batched_jacobian_matches_finite_differences():
    chain = K.Chain(_toy_kin(), tcp_depth=0.12)
    bp, bR = _base()
    rng = np.random.default_rng(1)
    Q = rng.uniform(-1.0, 1.0, (6, chain.n))
    tcp, R, J = chain.tcp_jac(Q, bp, bR)
    h = 1e-6
    for k in range(chain.n):
        dq = np.zeros(chain.n)
        dq[k] = h
        tcp2, R2 = chain.tcp(Q + dq, bp, bR)
        assert np.allclose((tcp2 - tcp) / h, J[:, :3, k], atol=1e-5)
        w = K.rotvec(R2 @ R.transpose(0, 2, 1)) / h
        assert np.allclose(w, J[:, 3:, k], atol=1e-5)


def test_rotvec_matches_scipy_including_half_turns():
    rv = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, np.pi - 1e-6, 0.0], [1e-8, 0.0, 0.0]])
    Rm = Rotation.from_rotvec(rv).as_matrix()
    assert np.allclose(K.rotvec(Rm), rv, atol=1e-6)


def test_ik_from_many_seeds_reaches_a_pose_inside_the_limits():
    kin = _toy_kin()
    chain = K.Chain(kin, tcp_depth=0.12)
    bp, bR = _base()
    lo, hi = np.r_[0.0, -2.5, -1.5, -2.0, -2.5], np.r_[0.3, 2.5, 1.5, 2.0, 2.5]
    q_true = np.r_[0.12, 0.4, -0.3, 0.8, 0.5]
    p_t, R_t = chain.tcp(q_true[None], bp, bR)
    seeds = lo + K.halton(16, chain.n) * (hi - lo)
    Q, ep, er = K.solve(chain, bp, bR, p_t[0], R_t[0], seeds, lo, hi, retract=seeds[0])
    ok = (ep < 0.004) & (er < 0.04)
    assert ok.any()
    assert np.all((lo - 1e-12 <= Q) & (hi + 1e-12 >= Q))
    assert ep[ok].max() < K.TOL_P * 2


def test_local_pick_keeps_continuity_over_the_limit_margin():
    kin = _toy_kin()
    lo, hi = np.full(5, -2.0), np.full(5, 2.0)
    seed = np.r_[0.0, 0.0, 1.8, 0.0, 0.0]
    near_limit = np.r_[0.0, 0.0, 1.95, 0.0, 0.0]  # 0.15 rad from the seed, 0.05 rad from a limit
    clear = np.r_[0.0, 0.0, 1.0, 0.0, 0.0]  # clear of the limits, 0.8 rad away
    args = dict(limit_clear=0.15, manip_w=0.0, max_step=0.25)
    q, ok = K.pick(kin, np.array([clear, near_limit]), seed, lo, hi, local=True, **args)
    assert ok and np.allclose(q, near_limit)


def test_halton_points_are_deterministic_and_in_the_unit_cube():
    a, b = K.halton(64, 9), K.halton(64, 9)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() < 1.0
    assert len({tuple(np.round(r, 9)) for r in a}) == 64


def test_distinct_drops_near_duplicates_and_keeps_order():
    Q = np.array([[0.0, 0.0], [1e-5, 0.0], [0.5, 0.5], [0.5, 0.5 + 1e-4]])
    assert np.array_equal(K.distinct(Q), Q[[0, 2]])


def test_ompl_plan_goes_around_a_wall_and_every_edge_is_valid():
    plan = pytest.importorskip("robotforge.zenobench.motion_plan")
    pytest.importorskip("ompl")

    def valid(q):
        return not (0.45 < q[0] < 0.55 and q[1] < 0.8)

    q0, q1 = np.array([0.1, 0.1]), np.array([0.9, 0.1])
    qs, status = plan.plan(q0, q1, np.zeros(2), np.ones(2), valid, 0.01, timeout=2.0, attempts=2)
    assert status is None and np.allclose(qs[-1], q1)
    path = [q0] + qs
    for a, b in zip(path[:-1], path[1:], strict=True):
        for u in np.linspace(0, 1, 50):
            assert valid(a + (b - a) * u)


def test_ompl_constrained_plan_stays_on_the_manifold():
    plan = pytest.importorskip("robotforge.zenobench.motion_plan")
    pytest.importorskip("ompl")
    cons = plan.Constraint(lambda x: np.array([np.linalg.norm(x) - 1.0]), lambda x: (x / np.linalg.norm(x))[None], 1)
    lo, hi = np.full(3, -2.0), np.full(3, 2.0)
    q0, q1 = cons.project(np.array([0.1, 0.0, -1.3]), lo, hi), cons.project(np.array([0.0, 0.1, 1.2]), lo, hi)

    def valid(q):
        return abs(q[2]) > 0.2 or q[0] > 0.3  # a band around the equator with a gap at +x

    qs, status = plan.plan(q0, q1, lo, hi, valid, 0.05, timeout=5.0, attempts=2, constraint=cons)
    assert status is None
    assert max(abs(np.linalg.norm(q) - 1.0) for q in qs) < 1e-3
    assert all(valid(q) for q in qs)


def test_robot_spheres_load_radii_buffer_and_ignored_pairs(tmp_path):
    from robotforge.zenobench.motion import RobotSpheres

    cfg = tmp_path / "robot.yml"
    cfg.write_text(
        "robot_cfg:\n  kinematics:\n    collision_sphere_buffer: 0.01\n"
        "    collision_spheres:\n"
        "      a: [{center: [0, 0, 0.1], radius: 0.05}, {center: [0, 0, 0], radius: -1}]\n"
        "      b: [{center: [0.2, 0, 0], radius: 0.03}]\n"
        "    self_collision_ignore: {a: [b]}\n"
    )
    rs = RobotSpheres.load(cfg)
    assert rs.links == ("a", "b")  # the negative radius (switched off) is dropped
    assert np.allclose(rs.radii, [0.06, 0.04])
    assert frozenset(("a", "b")) in rs.ignore


def _checkout():
    from robotforge.zenobench import source

    try:
        root = source.check()
    except source.SourceError:
        return None
    return root if (root / "zeno_skills" / "motion.py").exists() else None


@pytest.mark.skipif(sys.platform == "win32", reason="the ZenoBench scenes are checked on macOS and Linux")
def test_stand_in_on_a_real_kitchen_scene():
    pytest.importorskip("pxr")
    pytest.importorskip("ompl")
    root = _checkout()
    if root is None:
        pytest.skip("no ZenoBench checkout from after its switch to cuRobo (set ZENOBENCH_ROOT)")
    env = {**os.environ, "ZENOBENCH_ROOT": str(root)}
    run = subprocess.run(
        [sys.executable, "-m", "robotforge.zenobench.python", str(CHECKS)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    line = next((x for x in run.stdout.splitlines() if x.startswith("MOTION_CHECKS ")), None)
    assert line is not None, run.stdout[-2000:] + run.stderr[-2000:]
    r = json.loads(line.split(" ", 1)[1])
    assert r["backend"] == "MujocoMotion"
    assert r["fk_pos_err"] < 1e-6 and r["fk_rot_err"] < 1e-6  # MuJoCo's robot = ZenoBench's URDF chain
    assert r["posture_free"] and r["ik_ok"] and r["ik_err"] < 0.004 and r["ik_free"]
    assert r["self_mismatch"] == 0  # link culling misses no sphere pair
    assert r["goal_free"] and r["straight_blocked"]
    assert r["plan_ok"] and r["plan_dense_free"] and r["plan_end_err"] < 1e-9
    assert not r["at_object_free"] and r["held_object_free"]  # a held object is no obstacle to its hand
    assert r["clearance_at_object"] == 0.0 and r["clearance_posture"] > 0.0
    assert r["tilt_plan_ok"] and r["tilt_max"] < 0.01
    assert r.get("level_end_err") is None or r["level_end_err"] < 1e-9  # ends at the goal or is refused
    assert r["ts_ok"] and r["ts_tilt_max"] < 0.05 and r["ts_end_err"] < 0.004 and r["ts_dense_free"]
    assert r["ts_max_step"] <= 0.25  # continuous: no joint jumps more than MAX_STEP between waypoints
    assert r["held_attached"] == "wooden_block_a" and r["held_spheres"] > 0 and r["held_inside_box"]
    assert r["held_after_left_query"] == r["held_spheres"]  # the other arm's queries leave them on
    assert r["held_off"] == 0
    assert r["self_collisions_seen"] > 50
    assert r["repair_jac_norm"] > 0.05 and r["repair_jac_err"] < 1e-4
    assert r["part_moves"] > 0.05 and r["part_restored"] < 1e-9
